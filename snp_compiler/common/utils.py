from common.hw_config import MAX_GRID_HEIGHT
import math
import numpy as np
import tensorflow as tf

int32min = np.iinfo(np.int32).min
int32max = np.iinfo(np.int32).max
int16min = np.iinfo(np.int16).min
int16max = np.iinfo(np.int16).max
int8min = np.iinfo(np.int8).min
int8max = np.iinfo(np.int8).max
int48min = -2**47
int48max = 2**47-1
int30min = -2**29
int30max = 2**29-1

def getbits_from_int(input,start_bit,num_bits): 
   binary_input = bin(input)  
   binary_input = binary_input[2:]
   end_pos = len(binary_input) - start_bit +1
   start_pos = end_pos - num_bits
   binary_output = binary_input[start_pos : end_pos]
   output = int(binary_output,2)
   return output

# This applies the python bin() function on a number but applies 2s complement so it works with negatives
def bin2c(num, bits) -> str:
    assert bits > 0
    # E.g., for 8 bits, assert num is in the range [-128, 127]
    limit = 1 << (bits-1)
    assert (-limit <= num) and (num < limit)
    if num >= 0:
        return bin(num)[2:].zfill(bits)  # Positive numbers or zero
    else:
        return bin((1 << bits) + num)[2:]  # Negative numbers in two's complement

# Note: might be better to move this function to frontend_hardware_dependent.py and call it
# before adding to node['frontend'] so no need to check channel == None
def get_tflite_rq_params(node, channel):
    # Scale is 16 bits
    if (channel is None) or (type(node['frontend']['output_multiplier'])==int):
        scale = node['frontend']['output_multiplier']
    else:
        scale = node['frontend']['output_multiplier'][channel]

    # Bias is 48 bits but lowest 6 are 0
    if (channel is None) or (type(node['frontend']['cInputH'])==int):
        cInputH = node['frontend']['cInputH']   # 32 bits
        cInputL = node['frontend']['cInputL']   # 16 bits
    else:
        cInputH = node['frontend']['cInputH'][channel]   # 32 bits
        cInputL = node['frontend']['cInputL'][channel]   # 16 bits
    cInputL_shifted = (cInputL >> 6) & 0x3FF  # Mask with 0x3FF to keep only the top 10 bits
    bias = (cInputH << 10) | cInputL_shifted

    # Convert o_shift to 2 bit
    if (channel is None) or (type(node['frontend']['o_shift'])==int):
        o_shift = node['frontend']['o_shift']
    else:
        o_shift = node['frontend']['o_shift'][channel]
    assert o_shift in [8, 16, 24, 32]  # b00 -> 8, b01 -> 16, b10 -> 24, b11 -> 32
    rough_shift = (o_shift // 8) - 1

    return scale, bias, rough_shift

def get_rq_params(node, channel, folded=False):
    return get_tflite_rq_params(node, channel)

def list_of_lists_split_middle(list_of_list,idx):
    splited_list_of_lists = []
    for inner_list in list_of_list:
        middle = len(inner_list) // 2
        if idx == 0:
            split=inner_list[:middle]
        else:
            split = inner_list[middle:]
        splited_list_of_lists.append(split)
    return splited_list_of_lists

def list_of_lists_split_at_pos(list_of_list,idx,per_index_start_pos=None):
    if per_index_start_pos is None:
        per_index_start_pos = []
    splited_list_of_lists = []
    for inner_list in list_of_list:
        if idx == len(per_index_start_pos)-1:
            split=inner_list[per_index_start_pos[idx]:]
        else:
            split=inner_list[per_index_start_pos[idx]:per_index_start_pos[idx+1]]
        splited_list_of_lists.append(split)
    return splited_list_of_lists

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def allocated_blocks_to_string(allocated_blocks):
    out_str='['
    out_str=out_str+str(allocated_blocks[0])+':'
    out_str=out_str+str(allocated_blocks[-1])+']'
    return out_str

# Returns the number of lines that will be written by tile 0 of a 2-tile blob
# that is the first of a sequence of 2-tile blobs
def _get_tile0_WR_lines_of_first_2tile_blob(lines_written_in_prev_tile, is_first_2T_folding, num_k3, amm_height):

    # If is_first_2T_folding is true, double the amount of lines we need to read
    # This is the same as halving the amount written in the previous tile
    if is_first_2T_folding:
        lines_written_in_prev_tile /= 2

    # Normally, the first 2 tile in the chain will read the full 14 (grid size)
    # If that isn't possible, reduce the write size to match what has been written
    first_2T_blob_T0_read_lines = min(amm_height, int(lines_written_in_prev_tile))

    # Then we can only write that minus the number of k3
    first_2T_blob_T0_write_lines = first_2T_blob_T0_read_lines - num_k3

    return first_2T_blob_T0_write_lines

# Special case with 2 tiles
# consecutive_2_tile_blobs includes the current blob
def get_y_tile_sizes_2_tile_case(folded_height, k3_nodes_in_blob, consecutive_2_tile_blobs, prev_blob_lines_in_ddr, is_first_node_folding, amm_height):

    # Only call for 2 tile blobs
    assert consecutive_2_tile_blobs >= 1

    # First tile size

    # If the previous blob is a 2 tile blob, lines_in_ddr is the number of lines written in its tile 0.
    # So further decrease the number of lines.
    if consecutive_2_tile_blobs > 1:
        current_2T_blob_t0_write_lines = prev_blob_lines_in_ddr - k3_nodes_in_blob
    # This is the first 2-tile blob, but if it's the first in the entire graph, the input is ready
    elif prev_blob_lines_in_ddr == -1:
        current_2T_blob_t0_write_lines = amm_height - k3_nodes_in_blob
    # Otherwise, need to make sure this number of writes is possible given what is already in ddr
    else:
        current_2T_blob_t0_write_lines = _get_tile0_WR_lines_of_first_2tile_blob(prev_blob_lines_in_ddr, is_first_node_folding, k3_nodes_in_blob, amm_height)

    tile_sizes=[current_2T_blob_t0_write_lines]
    per_tile_read_start_line = [0]
    per_tile_write_start_line = [0]

    # Second tile size
    tile_size = amm_height-k3_nodes_in_blob
    tile_sizes.append(tile_size)
    per_tile_read_start_line.append(folded_height-amm_height)
    per_tile_write_start_line.append(folded_height-tile_size)

    # Normally these are the final tile sizes. However, if the #lines of the first tile
    # was reduced to where the sum of the two tile sizes does not reach the folded height,
    # it's now necessary to increase the first tile size. This will require waiting for
    # the previous blob's write to complete before reading.
    if tile_sizes[0] + tile_sizes[1] < folded_height:
        tile_sizes[0] = amm_height-k3_nodes_in_blob

    assert tile_sizes[0] + tile_sizes[1] >= folded_height
    return tile_sizes, per_tile_read_start_line, per_tile_write_start_line

def get_y_tile_sizes(folded_height,k3_nodes_in_blob=0,amm_height=MAX_GRID_HEIGHT,extra_lines=0):
    # extra_lines (asymmetric-padding MAXPOOL trigger, 2 lines): for a two-tile blob, moved from tile 1 to tile 0.
    # The total lines are unchanged.
    # Yaron - Changed how this logic works
    # All tiles use the same amount of lines. The last tile starts at the correct line so that the full grid covers its last line.
    # There is no need for line padding
    
    if folded_height <= amm_height: #single tile, no need to worry about K3 nodes
        tile_sizes=[folded_height]
        per_tile_read_start_line = [0]
        per_tile_write_start_line = [0]
    else:
        tile_sizes=[amm_height-k3_nodes_in_blob]
        per_tile_read_start_line = [0]
        per_tile_write_start_line = [0]
        handled_lines = amm_height-k3_nodes_in_blob
        tile_size = amm_height-2*k3_nodes_in_blob
        if tile_size <= 0:
            # k3_nodes_in_blob >= amm_height/2 makes the middle-tile step non-positive, so
            # handled_lines never advances toward folded_height below - previously an infinite
            # loop that appended to these lists until MemoryError. Fail fast with the values
            # instead, since a blob this k3-heavy relative to amm_height needs to be split/
            # re-blobbed by the caller, not tiled here.
            raise ValueError(
                'get_y_tile_sizes: non-positive tile_size=%d (folded_height=%d, '
                'k3_nodes_in_blob=%d, amm_height=%d) - blob has too many k3 nodes for '
                'amm_height, needs re-blobbing/splitting upstream' % (
                    tile_size, folded_height, k3_nodes_in_blob, amm_height))
        while (handled_lines+amm_height-k3_nodes_in_blob) < folded_height:
            #middle tiles (in cases of 3 or more)
            tile_sizes.append(tile_size)
            per_tile_read_start_line.append(handled_lines-k3_nodes_in_blob)
            per_tile_write_start_line.append(handled_lines)
            handled_lines += tile_size
        remaining_lines = folded_height-handled_lines
        tile_size = amm_height-k3_nodes_in_blob
        overlap_lines = tile_size-remaining_lines # we reduce the size of the tile before last, by overlap lines
        tile_sizes[-1] -= overlap_lines
        tile_sizes.append(tile_size)
        per_tile_read_start_line.append(folded_height-amm_height)
        per_tile_write_start_line.append(folded_height-(amm_height-k3_nodes_in_blob))

    if extra_lines and len(tile_sizes) == 2 and tile_sizes[0] <= 14:
        # Two-tile blob with the asymmetric-padding trigger: when the first tile has at most 14 lines,
        # add extra_lines to it and remove the same from the second tile.
        tile_sizes[0] += extra_lines
        tile_sizes[1] -= extra_lines
        per_tile_read_start_line[1] = folded_height - tile_sizes[1] - k3_nodes_in_blob
        per_tile_write_start_line[1] = folded_height - tile_sizes[1]
    return tile_sizes,per_tile_read_start_line,per_tile_write_start_line

def sigmoid(x):
  if x > 0:   
    z = np.exp(-x)
    return 1/(1+z)
  else:
    z = np.exp(x)
    return z/(1+z)

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def TfLiteRound(n, decimals=0):
    rounded_abs = round_half_up(abs(n), decimals)
    return math.copysign(rounded_abs, n)
   
def LUTTransform(transform, dequantized):
    return transform(dequantized)

# Use the same LUT generation code for both uint8_t and int8_t. Int8_t indexes
# will be directly casted to uint8_t, the int8 LUT will thus be ordered as [0,
# 1, ..., 127, -128, ..., -2, -1] instead of [-128, -127, ..., -1, 0, 1, ...,
# 126, 127].
def LUTPopulateInt8(input_scale, input_zero_point, output_scale, output_zero_point, itype=np.int8, otype=np.int8):
    lut_int8 = np.zeros((256,), dtype=np.int8)
    inverse_scale = 1.0 / output_scale
    
    transform = lambda s: s*sigmoid(s)
    
    max_idx, min_idx = 127, -128
    if itype==np.uint8:
        max_idx, min_idx = 255, 0
    max_out, min_out = 127, -128
    if otype==np.uint8:
        max_out, min_out = 255, 0    
    for idx in range(min_idx, max_idx+1):
        dequantized = input_scale * (idx - input_zero_point)
        transformed = LUTTransform(transform, dequantized)
        quantized = -128
        if otype==np.uint8:
            quantized = 0
        if not tf.math.is_nan(transformed) and not tf.math.is_inf(transformed):
            rescaled = TfLiteRound(transformed * inverse_scale)
            quantized = rescaled + output_zero_point
        lut_int8[np.uint8(idx)] = np.int8(max(min(max_out, quantized), min_out))
    return lut_int8

def saturate(x,minRange,maxRange):
    sat = x<minRange or x>maxRange
    x = max(min(x,maxRange),minRange)
    return x,sat

def quantize2MathBlock(acc,bias_data,scale,output_offset,output_activation_min,output_activation_max):
    OM_BITS = 16     # compressed output multiplier number of bits (unsigned)
    O_SHIFT_RANGE = [8,16,24,32]     # possible output shift

    # COMPILER ##################
    if acc != None:
        assert(acc <= int30max and acc >= int30min)
    max_val = (int30max+bias_data)*scale+output_offset
    min_val = (int30min+bias_data)*scale+output_offset
    expected_saturation = (max_val<output_activation_min and min_val<output_activation_min) or (max_val>output_activation_max and min_val>output_activation_max)
    if expected_saturation:
        scale = 1   # handles cases with large bias that will always cause overflow
    if scale==0:
        o_shift = O_SHIFT_RANGE[0]
        m_shift = -o_shift
        m_shift_min = -o_shift
    else:
        om = scale
        m_shift_min = 0
        while round(2.0*om) <= 2**OM_BITS-1:
            om *= 2.0
            m_shift_min -= 1
        while round(om) > 2**OM_BITS-1:
            om /= 2.0
            m_shift_min += 1
    while 1:
        indices = np.where(np.array(O_SHIFT_RANGE) <= -m_shift_min)[0]  # o_shift = -m_shift
        ind = 0
        if len(indices)>0:
            ind = indices[-1]
        o_shift = O_SHIFT_RANGE[ind]
        m_shift = -o_shift

        output_multiplier, om_sat = saturate(int(round(scale * 2.0**-m_shift)),0,2**(OM_BITS)-1)
        if om_sat: assert(m_shift<m_shift_min)
        else: assert(m_shift>=m_shift_min)

        bias_premult = bias_data*output_multiplier # using the floating-point scale here actually has a less accurate result in many cases
        offset = output_offset<<o_shift
        round_bit = 1<<(o_shift-1)
        _,m_sat_high = saturate(bias_premult+offset+round_bit+int30max*output_multiplier, int48min, int48max)
        _,m_sat_low = saturate(bias_premult+offset+round_bit+int30min*output_multiplier, int48min, int48max)
        cInput,c_sat = saturate(bias_premult+offset+round_bit, int48min, int48max)
        cInput = 64*math.floor(cInput/64.0) # round to zero lowest 6 bits, which has a negligible effect on precision
        cInputL = cInput%65536              # 16 bits
        cInputH = int(cInput-cInputL)>>16   # 32 bits    # Made int() because bias_data is float
        if m_sat_high or m_sat_low or c_sat:
            m_shift_min+=1
            if not expected_saturation:
                raise ValueError ('Something went wrong @quantize2MathBlock, expected saturation')
        else:
            break

    return output_multiplier, cInputH, cInputL, o_shift

def get_xslices_per_htile(num_xslices, h_tiles, kernel_size, is_folding_x=False, no_k3_nodes_in_blob=False, preserve_odd_split=False):
    # Folded-x kernel-3 node with 10 input x-slices split over 3 h-tiles: use [5,4,3] (output [3,2,2]) rather
    # than the even [4,4,4], which leaves a single output x-slice to overlap between h-tiles.
    if is_folding_x and kernel_size == 3 and num_xslices == 10 and h_tiles == 3:
        return [5, 4, 3]
    xslices_per_htile = []
    needed_xslices = (num_xslices + h_tiles - 1) if kernel_size != 1 else num_xslices
    # Parity of the TOTAL actually being split (includes the k3 halo, so it can differ from num_xslices).
    # The even-total redistribution below keys on this, not num_xslices: an EVEN split total tends to
    # [equal, equal] (both h-tiles same parity - if odd, e.g. 10 -> [5,5], the fold overshoots and must be
    # redistributed to [4,6]); an ODD split total is [ceil, floor] = [odd, even] (e.g. 21 -> [11,10]) whose
    # per-h-tile fold is already consistent and MUST be preserved (rounding it would move the odd to the
    # last h-tile, e.g. [10,11], and desync the fold - see program_compiler force_folding_x out handling).
    total_split = needed_xslices
    for h_tile_idx in range(h_tiles):
        if (h_tile_idx == h_tiles-1):
            xslices_per_tile = int(needed_xslices)
        else:
            # Divide the REMAINING slices by the REMAINING tiles, not the full h_tiles. Dividing
            # the (already decremented) needed_xslices by the constant h_tiles starves the middle
            # tiles and dumps the drift into the last tile (e.g. 8 over 4 -> [2,2,1,3]); dividing by
            # the tiles left to fill keeps the split balanced (-> [2,2,2,2]).
            xslices_per_tile = math.ceil(needed_xslices / (h_tiles - h_tile_idx))
            # An x-folding NON-LAST h-tile must hold complete fold-pairs (an even x-slice count) so the
            # 2:1 fold closes within the h-tile; the leftover accumulates in the LAST h-tile (which may
            # carry an odd count via the sequencer's multi_htile_first path). A plain balanced split can
            # put an odd count in a non-last h-tile even when the TOTAL is even (e.g. 10 over 2 -> [5,5]);
            # round the non-last count DOWN to the next even so the extra slices fall into the last h-tile
            # (-> [4,6], whose per-h-tile fold [2,3] then matches the output split). Restricted to even
            # totals so the existing odd-total reverse/pad handling (below, get_padded_input_xslices) is
            # left untouched; guarded to keep a non-last h-tile non-empty.
            # Skip this reduction if preserve_odd_split is True (for special folding_x nodes with odd input/output).
            if is_folding_x and (total_split % 2 == 0) and (xslices_per_tile % 2 != 0) and (xslices_per_tile > 1) and not preserve_odd_split:
                xslices_per_tile -= 1
            needed_xslices -= xslices_per_tile
        xslices_per_htile.append(xslices_per_tile)

    # NOTE: previously an odd-total x-folding split was REVERSED (leftover odd slice -> first h-tile). That
    # desynced the fold: for e.g. 21 over 2 the natural ceil split [11,10] became [10,11] while the output
    # split stayed [6,5] (per-h-tile fold of [11,10]), so the sequencer's fold engine command over-indexed
    # the last h-tile. The fold is computed PER h-tile (the sequencer's multi_htile_first path handles an odd
    # count within a tile), so the natural [ceil, ..., floor] split must be PRESERVED and the output derived
    # as its per-h-tile fold (see the force_folding_x out_xslices handling in program_compiler). No reverse.
    #
    # That desync only exists when kernel_size!=1 (the k3 halo makes needed_xslices != num_xslices, and the
    # per-h-tile fold is computed from the halo'd split). When the blob has no k3 nodes at all,
    # kernel_size==1 here, so needed_xslices==num_xslices and there's no halo to desync - the natural
    # [ceil, ..., floor] order is then just an artifact of iterating h-tiles low-to-high and should be
    # [floor, ..., ceil] instead (leftover slice in the LAST h-tile, not the first).
    #
    # Gated on is_folding_x so this never fires under program_compiler's blob_reverses_xslices strategy
    # (x_slices < 2*h_tiles case): that caller deliberately passes is_folding_x=False here so it can
    # apply ONE reversal itself, uniformly, at the blob level (leftover -> FIRST h-tile, matching the
    # sequencer's multi_htile_first path). Reversing here too would double-reverse and silently
    # reintroduce the natural [ceil, ..., floor] order that blob_reverses_xslices exists to avoid.
    if is_folding_x and no_k3_nodes_in_blob and (total_split % 2 != 0) and (xslices_per_htile[0] % 2 == 0):
        xslices_per_htile.reverse()

    return xslices_per_htile

def pad_k3_folding_output_xslices_per_htile(output_xslices_per_htile, output_x_slices, h_tiles):
    """For a k3 (kernel_size!=1) node with force_folding_x set, the output xslices_per_htile must
    sum to output_x_slices + h_tiles - 1 (one extra halo slice per h-tile boundary). If the plain
    get_xslices_per_htile split falls short of that total, increment individual h-tile entries by 1,
    starting at h-tile index 0, until the sum matches."""
    needed_total = output_x_slices + h_tiles - 1
    current_total = sum(output_xslices_per_htile)
    idx = 0
    while current_total < needed_total and len(output_xslices_per_htile) > 0:
        output_xslices_per_htile[idx % len(output_xslices_per_htile)] += 1
        current_total += 1
        idx += 1
    return output_xslices_per_htile

def amm_folding_mode(xslices_per_htile):
    if (xslices_per_htile % 8 == 0):
        num_packed_xslices = 8
    elif (xslices_per_htile % 4 == 0):
        num_packed_xslices = 4
    elif (xslices_per_htile % 2 == 0):
        num_packed_xslices = 2
    else:
        num_packed_xslices = 1

    return num_packed_xslices


def greedy_packed_xslice_groups(num_xslices):
    """Decompose num_xslices into a greedy largest-first combination of {8,4,2,1}.

    Returns a list of per-lane pack sizes (each a power of 2, cap 8) summing to num_xslices.
    Replaces the single 'highest uniform divisor' packing (amm_folding_mode) for the input layer:
    e.g. 12 -> [8,4], 6 -> [4,2], 18 -> [8,8,2], 3 -> [2,1], 4 -> [4], 1 -> [1].
    Each lane is one DMA transaction / one AMM block packing 'pack' x-slices contiguously.
    """
    groups = []
    remaining = int(num_xslices)
    while remaining > 0:
        lane = 8
        while lane > remaining:
            lane //= 2
        groups.append(lane)
        remaining -= lane
    return groups


def is_packed_xslices(num_packed_xslices):
    """True when num_packed_xslices is the per-lane combination marker (input-layer packing active).

    It is stored as a list (the greedy decomposition of the first h-tile's slice count) which doubles as
    the 'packing active' flag. Per-h-tile lane layouts are re-derived from each h-tile's own slice count
    (see packed_groups_for_htile) because h-tiles can have different slice counts.
    """
    return isinstance(num_packed_xslices, (list, tuple))


def packed_groups_for_htile(num_packed_xslices, xslices_per_htile_h):
    """Per-lane pack sizes for a given h-tile, or None when packing is not active.

    Re-derives the greedy decomposition from this h-tile's own logical slice count so that h_tiles>1
    (where each h-tile has a different slice count) is handled correctly.
    """
    if is_packed_xslices(num_packed_xslices):
        return greedy_packed_xslice_groups(xslices_per_htile_h)
    return None


def packed_num_lanes(num_packed_xslices, xslices_per_htile_h):
    """Number of physical AMM lanes / DMA transactions for a h-tile.

    Packed: one lane per greedy group of this h-tile. Scalar: legacy uniform factor (xslices // factor);
    the default factor 1 yields one lane per logical x-slice (unpacked).
    """
    groups = packed_groups_for_htile(num_packed_xslices, xslices_per_htile_h)
    if groups is not None:
        return len(groups)
    return xslices_per_htile_h // num_packed_xslices


def packed_lane_info(num_packed_xslices, xslices_per_htile_h, lane_idx):
    """For a lane index in a h-tile, return (lane_pack, lane_start_slice).

    lane_start_slice is the logical index of the lane's first x-slice (used for the lane's DDR x-offset).
    """
    groups = packed_groups_for_htile(num_packed_xslices, xslices_per_htile_h)
    if groups is not None:
        return groups[lane_idx], sum(groups[:lane_idx])
    return num_packed_xslices, lane_idx * num_packed_xslices


def packed_lane_of_logical(num_packed_xslices, xslices_per_htile_h, logical_xslice_num):
    """Map a logical x-slice index (within a h-tile) to its lane.

    Returns (lane_idx, offset_in_lane, lane_pack, lane_start_slice). For the scalar (uniform) form this
    reproduces the old //, % math.
    """
    groups = packed_groups_for_htile(num_packed_xslices, xslices_per_htile_h)
    if groups is not None:
        lane_start = 0
        for lane_idx, lane_pack in enumerate(groups):
            if logical_xslice_num < lane_start + lane_pack:
                return lane_idx, logical_xslice_num - lane_start, lane_pack, lane_start
            lane_start += lane_pack
        raise ValueError('logical x-slice %d out of range for packing %s (htile slices=%d)' % (
            logical_xslice_num, groups, xslices_per_htile_h))
    lane_pack = num_packed_xslices
    lane_idx = logical_xslice_num // lane_pack
    return lane_idx, logical_xslice_num % lane_pack, lane_pack, lane_idx * lane_pack