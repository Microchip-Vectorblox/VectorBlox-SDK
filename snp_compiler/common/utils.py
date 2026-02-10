from common.hw_config import MAX_GRID_HEIGHT, TFLITE_REQUANT
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
    if (channel == None) or (type(node['frontend']['output_multiplier'])==int):
        scale = node['frontend']['output_multiplier']
    else:
        scale = node['frontend']['output_multiplier'][channel]

    # Bias is 48 bits but lowest 6 are 0
    if (channel == None) or (type(node['frontend']['cInputH'])==int):
        cInputH = node['frontend']['cInputH']   # 32 bits
        cInputL = node['frontend']['cInputL']   # 16 bits
    else:
        cInputH = node['frontend']['cInputH'][channel]   # 32 bits
        cInputL = node['frontend']['cInputL'][channel]   # 16 bits
    cInputL_shifted = (cInputL >> 6) & 0x3FF  # Mask with 0x3FF to keep only the top 10 bits
    bias = (cInputH << 10) | cInputL_shifted

    # Convert o_shift to 2 bit
    if (channel == None) or (type(node['frontend']['o_shift'])==int):
        o_shift = node['frontend']['o_shift']
    else:
        o_shift = node['frontend']['o_shift'][channel]
    assert o_shift in [8, 16, 24, 32]  # b00 -> 8, b01 -> 16, b10 -> 24, b11 -> 32
    rough_shift = (o_shift // 8) - 1

    return scale, bias, rough_shift

def get_non_tflite_rq_params(node, channel, folded):
    prefix = ''
    if folded:
        prefix = 'folded_'
    if ((channel == None) or type(node['frontend'][prefix + 'requant_scale_uint14'])==int):
        return (
            node['frontend'][prefix + 'requant_scale_uint14'],
            node['frontend'][prefix + 'requant_bias_int12'],
            node['frontend'][prefix + 'mac_rough_shift_mux']
        )
    else:
        return (
            node['frontend'][prefix + 'requant_scale_uint14'][channel],
            node['frontend'][prefix + 'requant_bias_int12'][channel],
            node['frontend'][prefix + 'mac_rough_shift_mux'][channel]
        )
    

def get_rq_params(node, channel, folded=False):
    if TFLITE_REQUANT:
        return get_tflite_rq_params(node, channel)
    else:
        return get_non_tflite_rq_params(node, channel, folded=folded)

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

def list_of_lists_split_at_pos(list_of_list,idx,per_index_start_pos=[]):
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
def _get_tile0_WR_lines_of_first_2tile_blob(lines_written_in_prev_tile, is_first_2T_folding, num_k3):

    # If is_first_2T_folding is true, double the amount of lines we need to read
    # This is the same as halving the amount written in the previous tile
    if is_first_2T_folding:
        lines_written_in_prev_tile /= 2

    # Normally, the first 2 tile in the chain will read the full 14 (grid size)
    # If that isn't possible, reduce the write size to match what has been written
    first_2T_blob_T0_read_lines = min(MAX_GRID_HEIGHT, int(lines_written_in_prev_tile))

    # Then we can only write that minus the number of k3
    first_2T_blob_T0_write_lines = first_2T_blob_T0_read_lines - num_k3

    return first_2T_blob_T0_write_lines

# Special case with 2 tiles
# consecutive_2_tile_blobs includes the current blob
def get_y_tile_sizes_2_tile_case(folded_height, k3_nodes_in_blob, consecutive_2_tile_blobs, prev_blob_lines_in_ddr, is_first_node_folding):

    # Only call for 2 tile blobs
    assert consecutive_2_tile_blobs >= 1

    # First tile size

    # If the previous blob is a 2 tile blob, lines_in_ddr is the number of lines written in its tile 0.
    # So further decrease the number of lines.
    if consecutive_2_tile_blobs > 1:
        current_2T_blob_t0_write_lines = prev_blob_lines_in_ddr - k3_nodes_in_blob
    # This is the first 2-tile blob, but if it's the first in the entire graph, the input is ready
    elif prev_blob_lines_in_ddr == -1:
        current_2T_blob_t0_write_lines = MAX_GRID_HEIGHT - k3_nodes_in_blob
    # Otherwise, need to make sure this number of writes is possible given what is already in ddr
    else:
        current_2T_blob_t0_write_lines = _get_tile0_WR_lines_of_first_2tile_blob(prev_blob_lines_in_ddr, is_first_node_folding, k3_nodes_in_blob)

    tile_sizes=[current_2T_blob_t0_write_lines]
    per_tile_read_start_line = [0]
    per_tile_write_start_line = [0]

    # Second tile size
    tile_size = MAX_GRID_HEIGHT-k3_nodes_in_blob
    tile_sizes.append(tile_size)
    per_tile_read_start_line.append(folded_height-MAX_GRID_HEIGHT)
    per_tile_write_start_line.append(folded_height-tile_size)

    # Normally these are the final tile sizes. However, if the #lines of the first tile
    # was reduced to where the sum of the two tile sizes does not reach the folded height,
    # it's now necessary to increase the first tile size. This will require waiting for
    # the previous blob's write to complete before reading.
    if tile_sizes[0] + tile_sizes[1] < folded_height:
        tile_sizes[0] = MAX_GRID_HEIGHT-k3_nodes_in_blob

    assert tile_sizes[0] + tile_sizes[1] >= folded_height
    return tile_sizes, per_tile_read_start_line, per_tile_write_start_line

def get_y_tile_sizes(folded_height,k3_nodes_in_blob=0,add_padding_line = False):
    # Yaron - Changed how this logic works
    # All tiles use the same amount of lines. The last tile starts at the correct line so that the full grid covers its last line.
    # There is no need for line padding
    
    if folded_height <= MAX_GRID_HEIGHT: #single tile, no need to worry about K3 nodes
        tile_sizes=[folded_height]
        per_tile_read_start_line = [0]
        per_tile_write_start_line = [0]
    else:
        tile_sizes=[MAX_GRID_HEIGHT-k3_nodes_in_blob]
        per_tile_read_start_line = [0]
        per_tile_write_start_line = [0]
        handled_lines = MAX_GRID_HEIGHT-k3_nodes_in_blob
        tile_size = MAX_GRID_HEIGHT-2*k3_nodes_in_blob
        while (handled_lines+MAX_GRID_HEIGHT-k3_nodes_in_blob) < folded_height:
            #middle tiles (in cases of 3 or more)
            tile_sizes.append(tile_size)
            per_tile_read_start_line.append(handled_lines-k3_nodes_in_blob)
            per_tile_write_start_line.append(handled_lines)
            handled_lines += tile_size
        #Last tile - select start so that it ends exactly at the last line
#        tile_size = MAX_GRID_HEIGHT-k3_nodes_in_blob
#        tile_sizes.append(tile_size)
#        per_tile_read_start_line.append(folded_height-MAX_GRID_HEIGHT)
#        per_tile_write_start_line.append(folded_height-(MAX_GRID_HEIGHT-k3_nodes_in_blob))
# This is a better allocation that reduces the size of teh tile before last
        remaining_lines = folded_height-handled_lines
        tile_size = MAX_GRID_HEIGHT-k3_nodes_in_blob
        overlap_lines = tile_size-remaining_lines # we reduce the size of the tile before last, by overlap lines
        tile_sizes[-1] -= overlap_lines
#        per_tile_read_start_line[-1] -= overlap_lines
#        per_tile_write_start_line[-1] -= overlap_lines
        tile_sizes.append(tile_size)
        per_tile_read_start_line.append(folded_height-MAX_GRID_HEIGHT)
        per_tile_write_start_line.append(folded_height-(MAX_GRID_HEIGHT-k3_nodes_in_blob))

# This is a more efficient last tile, but currently causes a bug in sequencer code creation
#        tile_size = folded_height-handled_lines
#        tile_sizes.append(tile_size)
#        per_tile_read_start_line.append(folded_height-MAX_GRID_HEIGHT)
#        per_tile_write_start_line.append(folded_height-tile_size)
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
