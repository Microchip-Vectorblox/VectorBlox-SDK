from common.sequencer_ir import SequencerDDRReadEntry, SequencerDDRWriteEntry, SequencerEngineOperationEntry, DDRReadDstInfoAMM,\
                                DDRReadDstInfoWLOC, DDRReadDstInfoTables, SequencerCustomOperationEntry, CustomeOpCode,\
                                DDRReadDstInfoProgram
from common.enums import DDRCommandDstSrcType, EngineOpCode, DDRReadOffsetType, DDRWriteOffsetType,  ScalingMode, DDREntryType
from common.hw_config import DDR_MIN_READ_SIZE, DMR_TABLE_LENGTH_BYTES, TSNP_DDR_BYTES_PER_CHANNEL, TSNP_DDR_BOX_DEPTH, AMM_HEIGHT,\
                            MULTIPLE_INPUT_OPS, MAX_TABLE_LENGTH, MAX_GRID_WIDTH, \
                            PROGRAM_MEMORY_NUM_COMMANDS, BYTES_IN_SEQUENCER_COMMAND, MAX_X_WRAPPING, NO_NEIGHBOR_SLICE, OpType
import math
from common.debug_flags import DEBUG_OPTIMIZE_DMA_TRANSACTIONS, DEBUG_REMOVE_ENGINE_COMMAND,  DEBUG_AVOID_DDR_WRITE_WHILE_CONV,\
      DEBUG_AVOID_DDR_READ_WHILE_CONV,  DEBUG_2_TILE_RW_LINES
import common.debug_flags as debug_flags
import common.internal_representation as internal_representation
from common.utils import get_y_tile_sizes, get_y_tile_sizes_2_tile_case, is_packed_xslices, packed_num_lanes, packed_lane_info, packed_lane_of_logical
from common.tensor_ir import InputTensorInfo
import re
import numpy as np


# ==================== Helper Functions ====================

def node_avoids_ddr_write(node):
    # True if DDR writes for this node must NOT overlap conv: the global debug flag, OR the per-node
    # marker set by the single-node-blob pre-split fallback in program_compiler. Defaults to the global
    # behavior when the per-node flag is absent, so non-flagged nodes are unchanged.
    return DEBUG_AVOID_DDR_WRITE_WHILE_CONV or node['frontend'].get('avoid_ddr_write_while_conv', False)

def accumulate_backend_bytes(node, field_name, key, value):
    """Accumulate byte counts in node['backend'][field_name][key]."""
    backend = node['backend']
    if field_name not in backend:
        backend[field_name] = {}
    if key in backend[field_name]:
        backend[field_name][key] += value
    else:
        backend[field_name][key] = value


def resolve_to_unsplit_root_tensor_name(ir, tensor_name):
    """Walk split_tensor_to_original_tensor_map up to the pre-split root tensor name.

    Splitting a tensor (e.g. split_conv_in_channels_in_m on an oversized input-channel count)
    produces slice tensors like '<name>_split_ic0' that are absent from membership sets keyed on
    the original tensor name (ir.inputs, ir.tensors_from_mxp) - even though the slice still is,
    at read time, the same underlying data (an external graph input, an MXP-sourced tensor, etc).
    Any membership check against such a set must resolve to this root first, or it will silently
    misclassify every split descendant.
    """
    cur_name = tensor_name
    while cur_name in ir.split_tensor_to_original_tensor_map:
        cur_name = ir.split_tensor_to_original_tensor_map[cur_name]
    return cur_name

def is_graph_input_derived_tensor(ir, tensor_name):
    """True if tensor_name is (or was split from) a real graph-input tensor in ir.inputs."""
    return resolve_to_unsplit_root_tensor_name(ir, tensor_name) in ir.inputs


def align_table_length(length_bytes, table_name=None):
    """Align table length to DMR requirements and optionally validate against MAX_TABLE_LENGTH.

    Args:
        length_bytes: Raw table length in bytes
        table_name: If provided, validates against MAX_TABLE_LENGTH and raises ValueError if exceeded

    Returns:
        Aligned length in bytes
    """
    if length_bytes < DDR_MIN_READ_SIZE:
        length_bytes = DDR_MIN_READ_SIZE
    align_size = math.ceil(length_bytes / DMR_TABLE_LENGTH_BYTES)
    if (align_size % 2) == 1:
        align_size += 1
    aligned_length = align_size * DMR_TABLE_LENGTH_BYTES
    if table_name and (aligned_length / DMR_TABLE_LENGTH_BYTES) > MAX_TABLE_LENGTH:
        raise ValueError(f'{table_name} Table Length exceeds MAX_TABLE_LENGTH')
    return aligned_length


# ==================== Core Functions ====================

def get_grid_offset(grid_num,tensor_channels=0,tensor_lines=0):
    rounded_length = int(math.ceil(tensor_channels/TSNP_DDR_BOX_DEPTH)*TSNP_DDR_BOX_DEPTH)
    grid_offset = int(grid_num*rounded_length*TSNP_DDR_BYTES_PER_CHANNEL*tensor_lines) # We need the int since tensor lines can be float if its a y folding conv and original y is odd number of lines
    return grid_offset


def _first_mxp_consumer(ir, tensor_name):
    # Earliest consumer (in lexicographical topological order) of an MXP-sourced tensor.
    tensor_consumers = set(ir.tensors[tensor_name].consumers)
    consumers = [c for c in ir.lexicographical_topological_sorted_graph if c in tensor_consumers]
    return consumers[0] if consumers else None


def add_input_load_from_ddr_commands(ir,tensor_info:InputTensorInfo,current_node_sequence,node,set_flags_mask=0,wait_flags_mask=0,target_grid=0,ddr_read_offset_type=DDRReadOffsetType.INPUT_TENSOR):
    frontend = node['frontend']
    backend = node['backend']
    input_index = tensor_info.input_index
    tile_to_load_idx = tensor_info.tensor_tile_num
    h_tile_to_load_idx = tensor_info.tensor_h_tile_num
    xslice_to_load_idx = tensor_info.tensor_xslice_num

    if node['op_type'] in MULTIPLE_INPUT_OPS:
        input_tensor = frontend['input_tensors'][input_index]
    else:
        input_tensor = frontend['input_tensor']

    even_grid = not (target_grid % 2)
    current_blob = ir.tiling_blobs[frontend['tiling_blob_idx']]
    k3_nodes_in_blob = current_blob.k3_nodes
    amm_height = ir.amm_height

    blocks_key = 'allocated_amm_blocks_for_input_even_grid' if even_grid else 'allocated_amm_blocks_for_input_odd_grid'
    amm_allocated_blocks_for_input = backend[blocks_key][tile_to_load_idx][h_tile_to_load_idx][xslice_to_load_idx][input_index]

    # We need to make sure that allocated blocks for input are same for all amms. Otherwise we need to find a way to write to different addresses per each amm (which is currently not supported)
    allocated_blocks = amm_allocated_blocks_for_input[0]
    for current_amm_allocated_blocks in amm_allocated_blocks_for_input:
        if current_amm_allocated_blocks != allocated_blocks:
            raise ValueError('At read inputs from DDR to AMM. AMM allocation is not same for all AMMs. This is currently un-supported')
    amm_start_address = allocated_blocks[0] * ir.amm_block_size
    input_tensor_shape = input_tensor.shape
    input_tensor_length = input_tensor_shape[1]
    post_folding_input_tensor_length = input_tensor_length
    post_folding_tensor_lines = input_tensor.get_folded_shape()[2]
    y_folding_factor = input_tensor.folding_factor_y
    if 'force_folding_y' in frontend:
        post_folding_input_tensor_length *= 2
        y_folding_factor += 1
        post_folding_tensor_lines = int(math.ceil(post_folding_tensor_lines / 2))
    if 'force_unfolding_y' in frontend:
        if (input_tensor_length % 2) != 0:
            raise ValueError('Y Unfolding of odd channel tensor is not supported yet')
        post_folding_input_tensor_length = int(post_folding_input_tensor_length / 2)
        post_folding_tensor_lines *= 2
        if y_folding_factor > 0:
            y_folding_factor -= 1

    grid_offset = get_grid_offset(target_grid, tensor_channels=input_tensor_length, tensor_lines=input_tensor_shape[2]) # We calc grid offset with original (before folding) params (input_tensor_length,actual_tensor_lines) since in case of odd numbers the gird offset changes and we must have same grid offset as it was written to ddr
    input_amm_info = DDRReadDstInfoAMM(amm_start_address=amm_start_address)

    _, read_start_line, _ = get_num_of_lines_per_tile(post_folding_tensor_lines, k3_nodes_in_blob=k3_nodes_in_blob,
                                                    num_y_tiles=frontend['y_tiles'], current_tile_num=tile_to_load_idx, add_padding_line=False, amm_height=amm_height,
                                                    extra_lines=frontend.get('asym_extra_lines_read', 0) if frontend.get('asym_extra_lines_read', 0) > 0 else 0) # add_padding_line=False since in read we always read MAX_GRID_HEIGHT lines
    read_size_bytes = int(math.ceil(post_folding_input_tensor_length/32)*32 * 14 * (256 // 32)) # 256 bytes per 32 channels of each grid , 14 lines per read
    accumulate_backend_bytes(node, 'total_tensor_read_bytes', (tile_to_load_idx, h_tile_to_load_idx, xslice_to_load_idx), read_size_bytes)

    need_extra_line_padding = False
    for producer_name in ir.graph.predecessors(node['name']):
        producer = ir.graph.nodes()[producer_name]
        if producer['frontend'].get('concat_all_inputs_extra_line'):
            need_extra_line_padding = True
            break
    
    input_y = input_tensor_shape[2]
    if ('input_tensor' in frontend) and frontend['input_tensor'].is_avgPool_output:
        input_y += 1
    elif frontend.get('odd_height_y_fold_read', False):
        input_y += 1
    elif frontend.get('k3_1T_boundary_extra_line_read', False):
        input_y += 1
    # If consumer is an ADD layer with k3_1T_boundary_extra_line_write and has input from different blob, increment input_y
    elif node['op_type'] == OpType.ADD:
        consumer_blob_idx = frontend.get('tiling_blob_idx')
        if input_tensor.producer and input_tensor.producer in ir.graph.nodes:
            producer_node = ir.graph.nodes[input_tensor.producer]
            producer_blob_idx = producer_node['frontend'].get('tiling_blob_idx')
            if producer_blob_idx != consumer_blob_idx:
                if producer_node['frontend'].get('k3_1T_boundary_extra_line_write', False):
                    input_y += 1
    elif need_extra_line_padding:
        input_y += 1
    input_y += frontend.get('asym_extra_lines_read', 0)
    
    folding_factor_x = input_tensor.folding_factor_x
    input_z = input_tensor_shape[1] * (2**folding_factor_x)
    y_folding = pow(2, y_folding_factor)

    # Check if this tensor is from MXP
    tensor_from_mxp = False
    unsplit_root_tensor_name = resolve_to_unsplit_root_tensor_name(ir, input_tensor.name)
    if ir.sync_with_MXP and unsplit_root_tensor_name in ir.tensors_from_mxp:
        if unsplit_root_tensor_name in ir.io_tensor_names:
            ddr_read_offset_type = DDRReadOffsetType.INPUT_TENSOR
        else:
            ddr_read_offset_type = DDRReadOffsetType.MXP_BASE
        tensor_from_mxp = True

    if tensor_from_mxp:
        input_x = math.ceil(input_tensor_shape[3] / (2**folding_factor_x))
    else:
        input_x = math.ceil(input_tensor_shape[3] / (MAX_GRID_WIDTH * (2**folding_factor_x))) * MAX_GRID_WIDTH
    amm_write_mode = 0
    x_wrapping = 1

    # lane_start_slice is only meaningful for packed x-slices, but it's read
    # unconditionally below (based on is_packed_xslices alone), so it must be
    # computed here regardless of which branch below sets x_wrapping.
    lane_start_slice = None
    if is_packed_xslices(input_tensor.num_packed_xslices):
        # Per-lane packing combination (input layer): xslice_to_load_idx is the LANE index within this
        # h-tile. The lane's pack count and starting logical slice come from this h-tile's own greedy
        # decomposition (h-tiles can have different slice counts).
        lane_pack, lane_start_slice = packed_lane_info(
            input_tensor.num_packed_xslices, frontend['xslices_per_htile'][h_tile_to_load_idx], xslice_to_load_idx)

    #Yaron - DMA parameter change
    # Skipped for input-channel-split tensors: the merge trick would bake x_wrapping into input_x/z,
    # which the per-(x_fold_block, y_fold_block) DMR loop below can't account for.
    # The DMW that wrote this tensor keeps the plain layout when it feeds a scatter concat that in turn
    # feeds an input-channel-split layer, so the read must not merge it either.
    tensor_feeds_scatter_concat = any(_feeds_input_split_via_scatter_concat(ir, c) or
                                      _is_scatter_concat_with_mixed_input_channels(ir, c)
                                      for c in input_tensor.consumers)
    # The DMR also reads a tensor produced by a mixed-channel scatter concat, so keep that read plain too.
    tensor_produced_by_mixed_scatter_concat = _is_scatter_concat_with_mixed_input_channels(ir, input_tensor.producer)
    # The consuming node's DMR must stay plain when it reads a channel-split ADD whose pieces are unequal in channels.
    fed_by_channel_split_add = any(_is_unequal_channel_split_add(ir, p) for p in ir.graph.predecessors(node['name']))
    # A y-folded consumer reading an output-channel-split producer keeps the plain layout (per-piece y-block DMRs).
    _producer_for_dma = ir.graph.nodes.get(input_tensor.producer) if input_tensor.producer else None
    _is_output_split_for_dma = (_producer_for_dma is not None
                                and _producer_for_dma['frontend'].get('was_channel_split', False)
                                and not getattr(input_tensor, 'is_read_channel_slice', False))
    reads_output_split_with_y_fold = (_is_output_split_for_dma and folding_factor_x > 0 and y_folding_factor > 0)
    if DEBUG_OPTIMIZE_DMA_TRANSACTIONS and (x_wrapping == 1) and (ddr_read_offset_type == DDRReadOffsetType.MODEL_MEM) \
            and 'is_input_channel_split' not in frontend and not tensor_feeds_scatter_concat \
            and not tensor_produced_by_mixed_scatter_concat and not fed_by_channel_split_add \
            and not reads_output_split_with_y_fold:
        while (input_z > 1) and (input_z % 2 != 1) and (x_wrapping <= (MAX_X_WRAPPING//2)) and (input_x < 1024):
            input_z /= 2
            x_wrapping *= 2
            input_x *= 2
        input_z = int(math.ceil(input_z))
        if x_wrapping == 32:
            x_wrapping = 0
    elif is_packed_xslices(input_tensor.num_packed_xslices):
        x_wrapping = lane_pack
    else:
        x_wrapping = input_tensor.num_packed_xslices
    # Asymmetric-padding trigger: a node anywhere in this blob with asym_extra_lines_write > 0 adds its padded
    # lines to the read start of the blob's last y-tile.
    blob_asym_lines = max((ir.graph.nodes[n]['frontend'].get('asym_extra_lines_write', 0) for n in current_blob.nodes_in_blob), default=0)
    if (blob_asym_lines > 0) and tile_to_load_idx == frontend['y_tiles']-1:
        read_start_line += blob_asym_lines
    h_tiles = frontend['h_tiles']
    y_tile_ddr_offset = read_start_line * input_x * y_folding
    h_tile_ddr_offset = 0
    if h_tile_to_load_idx != h_tiles - 1:
        if is_packed_xslices(input_tensor.num_packed_xslices):
            # Packing only groups how slices are READ; the DDR layout is still one MAX_GRID_WIDTH-wide
            # column per logical slice, so later h-tiles are skipped by their logical slice counts (no
            # pack multiply). The k3 halo overlap is one logical slice per h-tile boundary.
            h_tile_ddr_offset = sum(frontend['xslices_per_htile'][h_tile_to_load_idx+1:]) * MAX_GRID_WIDTH
            if k3_nodes_in_blob > 0:
                h_tile_ddr_offset -= ((h_tiles - 1) - h_tile_to_load_idx) * MAX_GRID_WIDTH
        else:
            h_tile_ddr_offset = sum(frontend['xslices_per_htile'][h_tile_to_load_idx+1:]) * MAX_GRID_WIDTH * x_wrapping
            if k3_nodes_in_blob > 0:
                h_tile_ddr_offset -= ((h_tiles - 1) - h_tile_to_load_idx) * MAX_GRID_WIDTH * x_wrapping
    if is_packed_xslices(input_tensor.num_packed_xslices):
        x_slice_ddr_offset = lane_start_slice * MAX_GRID_WIDTH
    else:
        x_slice_ddr_offset = xslice_to_load_idx * MAX_GRID_WIDTH * x_wrapping
    if input_tensor.ddr_entry is None:
        # Walk the FULL reverse-map ancestor chain (mirrors _alias_ddr_entries_to_split_descendants'
        # multi-level recursion) so a >1-level nesting gap (e.g. an output-split's alias input that is
        # itself channel-split again) is visible instead of only checking the immediate parent.
        chain = []
        cur_name = input_tensor.name
        seen = set()
        while cur_name is not None and cur_name not in seen:
            seen.add(cur_name)
            cur_tensor = ir.tensors.get(cur_name)
            listed_as_child = None
            next_name = ir.split_tensor_to_original_tensor_map.get(cur_name)
            if next_name is not None:
                listed_as_child = cur_name in ir.original_tensor_to_split_tensor_map.get(next_name, [])
            chain.append('%s(has_ddr_entry=%s,children=%s,listed_in_parent=%s,parent=%s)' % (
                cur_name,
                (cur_tensor.ddr_entry is not None) if cur_tensor is not None else None,
                ir.original_tensor_to_split_tensor_map.get(cur_name),
                listed_as_child,
                next_name))
            cur_name = next_name
        producer_node = ir.graph.nodes.get(input_tensor.producer) if input_tensor.producer else None
        producer_out = producer_node['frontend'].get('output_tensor') if producer_node is not None else None
        # Self-heal: a split/re-split cycle can leave a node's cached input_tensor pointing at a
        # stale object that never got the alias applied to the canonical ir.tensors[name] object.
        # Re-resolve through ir.tensors, the source of truth the aliasing BFS actually mutates.
        canonical_tensor = ir.tensors.get(input_tensor.name)
        if canonical_tensor is not None and canonical_tensor is not input_tensor and canonical_tensor.ddr_entry is not None:
            input_tensor = canonical_tensor
            if node['op_type'] in MULTIPLE_INPUT_OPS:
                frontend['input_tensors'][input_index] = input_tensor
            else:
                frontend['input_tensor'] = input_tensor
    ddr_src_address = input_tensor.ddr_entry.address + y_tile_ddr_offset + grid_offset + h_tile_ddr_offset + x_slice_ddr_offset

    if x_wrapping == 32: # x-wrapping of 32 is 0 in the DMR tables
        x_wrapping = 0

    nlines_to_read = max(2, input_y) if input_y < amm_height else amm_height
    if node['op_type'] == OpType.MAXPOOL:
        f_h = math.ceil(input_y/y_folding)
        nlines_to_read = max(2, f_h) if f_h < amm_height else amm_height
    elif (blob_asym_lines > 0) and tile_to_load_idx == frontend['y_tiles']-1:
        nlines_to_read -= blob_asym_lines

    amm_folding_mode = 0
    last_valid_bytes = 0
    if tensor_from_mxp:
        if (tile_to_load_idx == 0) and (h_tile_to_load_idx == 0) and (xslice_to_load_idx == 0) and ('_split' not in node['name']):
            # An MXP tensor with several consumers waits on F5 only at its first consumer (execution order);
            # later consumers clear it.
            if node['name'] == _first_mxp_consumer(ir, input_tensor.name):
                wait_flags_mask |= 32 # TSNP wait for MXP on F5
        if unsplit_root_tensor_name not in ir.mxp_tensor_to_offset:
            raise ValueError('MXP-sourced tensor %s missing from ir.mxp_tensor_to_offset (node %s)' % (input_tensor.name, node['name']))
        # ir.mxp_tensor_to_offset is keyed by the pre-split root tensor; an input-channel-split slice's
        # own channel offset within the root's MXP buffer is added back at split_channel_lo*H*W
        # (channel-major layout, same convention as the DDR-alias case elsewhere).
        mxp_base_address = ir.mxp_tensor_to_offset[unsplit_root_tensor_name]
        slice_channel_lo = getattr(input_tensor, 'split_channel_lo', None)
        if slice_channel_lo is not None:
            bytes_per_channel = int(input_tensor_shape[2]) * int(input_tensor_shape[3])
            mxp_base_address += slice_channel_lo * bytes_per_channel
        # Offset type still stays at 0, but address needs to change
        ddr_src_address = mxp_base_address + y_tile_ddr_offset + grid_offset + h_tile_ddr_offset + x_slice_ddr_offset
        ir.input_ddr_offset.add(ddr_src_address)
        if x_wrapping > 1:
            amm_folding_mode = 1
        num_xslices = packed_num_lanes(input_tensor.num_packed_xslices, input_tensor.xslices_per_htile[h_tile_to_load_idx])
        if xslice_to_load_idx == num_xslices - 1:
            if (input_tensor_shape[3] % MAX_GRID_WIDTH) != 0:
                last_valid_bytes = input_tensor_shape[3] % MAX_GRID_WIDTH

    # Halo padding must use the zero-point of the data as it lives in DDR. If the producer fuses a
    # SiLU (or other) activation, input_tensor.zero_point can be stale (pre-activation), so prefer
    # the producer's activation output_zp - wrong here is harmless for k1 but wrong for k3 reads.
    input_pad_val = input_tensor.zero_point
    if input_tensor.producer is not None and ir.graph.has_node(input_tensor.producer):
        activation_silu = ir.graph.nodes[input_tensor.producer].get('attributes', {}).get('activation_silu')
        if activation_silu is not None and 'output_zp' in activation_silu:
            input_pad_val = activation_silu['output_zp'][0]

    # input_pad_val must be a single scalar (zero_point is per-tensor by convention; only weights are
    # legitimately per-channel). A non-uniform array means something upstream mis-assigned a per-channel
    # value here - fail loudly rather than silently truncating to element 0.
    if isinstance(input_pad_val, np.ndarray):
        if input_pad_val.size != 1 and not np.all(input_pad_val == input_pad_val.flat[0]):
            raise ValueError(
                'DDR read for tensor %r (node %r) has a non-scalar, non-uniform zero_point %s - '
                'input_pad_val must be a single per-tensor halo padding value' %
                (input_tensor.name, node['name'], input_pad_val))
        input_pad_val = int(input_pad_val.flat[0])

    producer_node_name = input_tensor.producer
    producer_node = ir.graph.nodes[producer_node_name] if producer_node_name and ir.graph.has_node(producer_node_name) else None
    is_output_channel_split_tensor = (
        producer_node is not None
        and producer_node['frontend'].get('was_channel_split', False)
        and not getattr(input_tensor, 'is_read_channel_slice', False)
    )

    if 'is_input_channel_split' in node['frontend'] and folding_factor_x > 0:
        # Channel-range slice of a conv's ORIGINAL (pre-split) input, X-folded. Each x-fold block in DDR holds the
        # original channel range; when the producer was output-channel-split, its pieces sit one after another inside
        # each x-fold block, so a slice can overlap several pieces. One DMR per (x-fold block, y-fold block, piece
        # overlapping the slice) reads the slice at that piece's folded base.
        num_x_fold_blocks = 2 ** folding_factor_x
        input_z = input_z // num_x_fold_blocks
        original_input_tensor_name = resolve_to_unsplit_root_tensor_name(ir, input_tensor.name)
        original_input_tensor_shape = ir.tensors[original_input_tensor_name].shape
        depth = original_input_tensor_shape[1]
        height = input_y
        width = original_input_tensor_shape[3] // num_x_fold_blocks
        width = math.ceil(width / MAX_GRID_WIDTH) * MAX_GRID_WIDTH
        base_ddr_src_address = ddr_src_address

        # Producer output pieces: (channel start, channel count) for each output-channel-split piece of the root,
        # in piece order. Empty when the producer is not output-channel-split.
        producer_pieces = []
        if producer_node is not None and producer_node['frontend'].get('was_channel_split', False):
            root_pieces = ir.original_tensor_to_split_tensor_map.get(original_input_tensor_name, [])
            for piece_name in root_pieces:
                if '_split_ic' in piece_name or piece_name not in ir.tensors:
                    continue
                piece = ir.tensors[piece_name]
                lo = getattr(piece, 'split_channel_lo', None)
                if lo is not None:
                    producer_pieces.append((lo, piece.shape[1]))
            producer_pieces.sort()

        # This slice's channel range within the original input.
        slice_lo = getattr(input_tensor, 'split_channel_lo', None) or 0
        slice_hi = slice_lo + input_tensor.shape[1]

        num_y_fold_blocks = y_folding if y_folding_factor > 0 else 1
        base_amm_start_address = amm_start_address
        block_input_y = input_y // num_y_fold_blocks

        if not producer_pieces:
            # Producer not output-split: one contiguous channel range per x-fold block (checked-in layout).
            block_stride = depth * height * width
            total_blocks = num_x_fold_blocks * num_y_fold_blocks
            block_num = 0
            for block_idx in range(num_x_fold_blocks):
                for y_block_idx in range(num_y_fold_blocks):
                    block_wait_flags = wait_flags_mask if block_num == 0 else 0
                    block_set_flags = set_flags_mask if block_num == total_blocks - 1 else 0
                    block_src_address = base_ddr_src_address + block_idx * block_stride + y_block_idx * width
                    block_amm_start_address = base_amm_start_address + (y_block_idx * num_x_fold_blocks + block_idx) * input_z * x_wrapping
                    sequencer_entry = SequencerDDRReadEntry(src_address=block_src_address, ddr_read_offset_type=ddr_read_offset_type, input_pad_val=input_pad_val,
                                                            amm_write_mode=amm_write_mode, amm_start_address=block_amm_start_address, nlines_to_read=nlines_to_read, input_x=width * num_y_fold_blocks,
                                                            input_y=block_input_y, input_z=input_z, y_folding=1, x_wrapping=x_wrapping, amm_folding_mode=amm_folding_mode,
                                                            dst_type=DDRCommandDstSrcType.AMM, dst_info=input_amm_info, set_flags=block_set_flags, wait_flags=block_wait_flags)
                    current_node_sequence.append(sequencer_entry)
                    block_num += 1
            return current_node_sequence

        # Producer output-split: the folded layout puts piece p at folded channel 2*lo_p, so the channels of a piece
        # overlapping [slice_lo, slice_hi) start at (2*lo_p + max(slice_lo, lo_p) - lo_p) within its x-phase block.
        overlaps = []
        for lo_p, cs_p in producer_pieces:
            ov_lo, ov_hi = max(slice_lo, lo_p), min(slice_hi, lo_p + cs_p)
            if ov_hi > ov_lo:
                overlaps.append((lo_p, cs_p, ov_lo, ov_hi))
        total_blocks = num_x_fold_blocks * num_y_fold_blocks * len(overlaps)
        block_num = 0
        for block_idx in range(num_x_fold_blocks):
            for y_block_idx in range(num_y_fold_blocks):
                for lo_p, cs_p, ov_lo, ov_hi in overlaps:
                    block_wait_flags = wait_flags_mask if block_num == 0 else 0
                    block_set_flags = set_flags_mask if block_num == total_blocks - 1 else 0
                    # Folded channel of this overlap in x-phase block_idx: 2*lo_p is the piece's base (both phases),
                    # block_idx*cs_p selects the phase, and (ov_lo - lo_p) is the offset inside the piece.
                    folded_channel = 2 * lo_p + block_idx * cs_p + (ov_lo - lo_p)
                    block_src_address = base_ddr_src_address + folded_channel * height * width + y_block_idx * width
                    block_count = ov_hi - ov_lo
                    block_amm_start_address = base_amm_start_address + (y_block_idx * num_x_fold_blocks + block_idx) * input_z * x_wrapping
                    sequencer_entry = SequencerDDRReadEntry(src_address=block_src_address, ddr_read_offset_type=ddr_read_offset_type, input_pad_val=input_pad_val,
                                                            amm_write_mode=amm_write_mode, amm_start_address=block_amm_start_address, nlines_to_read=nlines_to_read, input_x=width * num_y_fold_blocks,
                                                            input_y=block_input_y, input_z=block_count, y_folding=1, x_wrapping=x_wrapping, amm_folding_mode=amm_folding_mode,
                                                            dst_type=DDRCommandDstSrcType.AMM, dst_info=input_amm_info, set_flags=block_set_flags, wait_flags=block_wait_flags)
                    current_node_sequence.append(sequencer_entry)
                    block_num += 1
        return current_node_sequence

    elif is_output_channel_split_tensor and folding_factor_x > 0 and y_folding_factor > 0 and not ('force_unfolding_y' in frontend):
        # Plain consumer of an x-folded, output-channel-split producer: the producer's pieces are separate folded
        # regions (piece p at folded channel 2*lo_p, each x-phase block holds its piece's channels). Read every
        # piece in full, one DMR per (x-fold block, y-fold block, piece), using the same folded addressing as the
        # input-channel-split branch.
        num_x_fold_blocks = 2 ** folding_factor_x
        root_name = resolve_to_unsplit_root_tensor_name(ir, input_tensor.name)
        height = input_tensor_shape[2]
        width_folded = math.ceil(input_tensor_shape[3] / MAX_GRID_WIDTH) * MAX_GRID_WIDTH
        base_ddr_src_address = ddr_src_address
        pieces = []
        for piece_name in ir.original_tensor_to_split_tensor_map.get(root_name, []):
            if '_split_ic' in piece_name or piece_name not in ir.tensors:
                continue
            piece = ir.tensors[piece_name]
            lo = getattr(piece, 'split_channel_lo', None)
            if lo is not None:
                pieces.append((lo, piece.shape[1]))
        pieces.sort()
        # Y-fold: one DMR per (x-fold block, y-fold block, piece). Each y-block starts one row further in the
        # piece's folded plane, so the even lines of every channel are read first, then the odd lines.
        num_y_fold_blocks = y_folding
        total_blocks = num_x_fold_blocks * num_y_fold_blocks * len(pieces)
        block_num = 0
        block_input_y = input_y // num_y_fold_blocks
        for block_idx in range(num_x_fold_blocks):
            for y_block_idx in range(num_y_fold_blocks):
                for lo_p, cs_p in pieces:
                    block_wait_flags = wait_flags_mask if block_num == 0 else 0
                    block_set_flags = set_flags_mask if block_num == total_blocks - 1 else 0
                    # Remap into the producer's folded region: piece p starts at folded channel 2*lo_p, and x-phase
                    # block_idx selects its half (block_idx*cs_p).
                    folded_channel = 2 * lo_p + block_idx * cs_p
                    block_src_address = base_ddr_src_address + folded_channel * height * width_folded + y_block_idx * width_folded
                    block_amm_start_address = amm_start_address + (y_block_idx * num_x_fold_blocks + block_idx) * input_z * x_wrapping
                    sequencer_entry = SequencerDDRReadEntry(src_address=block_src_address, ddr_read_offset_type=ddr_read_offset_type, input_pad_val=input_pad_val,
                                                            amm_write_mode=amm_write_mode, amm_start_address=block_amm_start_address, nlines_to_read=nlines_to_read,
                                                            input_x=width_folded * num_y_fold_blocks, input_y=block_input_y, input_z=cs_p, y_folding=1,
                                                            x_wrapping=x_wrapping, amm_folding_mode=amm_folding_mode,
                                                            dst_type=DDRCommandDstSrcType.AMM, dst_info=input_amm_info, set_flags=block_set_flags, wait_flags=block_wait_flags)
                    current_node_sequence.append(sequencer_entry)
                    block_num += 1
        return current_node_sequence

    elif is_output_channel_split_tensor and folding_factor_x > 0 and 'force_unfolding_y' in frontend:
        # Output-channel-split tensor (own dedicated DDR region, own shape) that is also folded on both
        # x and y: read it as 4 DMR commands (2 x-fold blocks x 2 y-fold blocks), each covering 1/4 of
        # this tensor's own size, laid out y-outer/x-inner to match the AMM channel-interleave convention
        # used by the input-channel-split branch above (ddr_ir.py's tensor_xslice_yfold: y outer, x inner).
        num_x_fold_blocks = 2 ** folding_factor_x
        num_pieces = len(ir.original_tensor_to_split_tensor_map.get(
            resolve_to_unsplit_root_tensor_name(ir, input_tensor.name), [])) or num_x_fold_blocks
        total_blocks = num_x_fold_blocks * num_pieces
        tensor_total_size = input_tensor_shape[1] * input_tensor_shape[2] * input_tensor_shape[3]
        block_stride = tensor_total_size // total_blocks
        base_ddr_src_address = ddr_src_address
        base_amm_start_address = amm_start_address
        block_input_z = input_z // total_blocks
        block_num = 0
        for s_p_idx in range(num_pieces):
            for x_idx in range(num_x_fold_blocks):
                block_wait_flags = wait_flags_mask if block_num == 0 else 0
                block_set_flags = set_flags_mask if block_num == total_blocks - 1 else 0
                block_src_address = base_ddr_src_address + (s_p_idx * num_x_fold_blocks + x_idx) * block_stride
                block_amm_start_address = base_amm_start_address + (s_p_idx + x_idx * num_pieces) * block_input_z * x_wrapping
                sequencer_entry = SequencerDDRReadEntry(src_address=block_src_address, ddr_read_offset_type=ddr_read_offset_type, input_pad_val=input_pad_val,
                                                        amm_write_mode=amm_write_mode, amm_start_address=block_amm_start_address, nlines_to_read=nlines_to_read, input_x=input_x,
                                                        input_y=input_y, input_z=block_input_z, y_folding=1, x_wrapping=x_wrapping, amm_folding_mode=amm_folding_mode,
                                                        dst_type=DDRCommandDstSrcType.AMM, dst_info=input_amm_info, set_flags=block_set_flags, wait_flags=block_wait_flags, last_valid_bytes=last_valid_bytes)
                current_node_sequence.append(sequencer_entry)
                block_num += 1
        return current_node_sequence

    sequencer_entry = SequencerDDRReadEntry(src_address=ddr_src_address, ddr_read_offset_type=ddr_read_offset_type, input_pad_val=input_pad_val,
                                            amm_write_mode=amm_write_mode, amm_start_address=amm_start_address, nlines_to_read=nlines_to_read, input_x=input_x,
                                            input_y=input_y, input_z=input_z, y_folding=y_folding, x_wrapping=x_wrapping, amm_folding_mode=amm_folding_mode,
                                            dst_type=DDRCommandDstSrcType.AMM, dst_info=input_amm_info, set_flags=set_flags_mask, wait_flags=wait_flags_mask, last_valid_bytes=last_valid_bytes)
    current_node_sequence.append(sequencer_entry)
    return current_node_sequence
        
def _is_unequal_channel_split_add(ir, node_name):
    # True when node_name is an ADD op that was channel split and whose output pieces differ in channel count.
    if not ir.graph.has_node(node_name):
        return False
    node = ir.graph.nodes[node_name]
    if node['op_type'] != OpType.ADD or not node['frontend'].get('was_channel_split', False):
        return False
    out_tensor = node['frontend'].get('output_tensor')
    if out_tensor is None:
        return False
    # Pieces of the same ADD output share one original tensor via split_tensor_to_original_tensor_map.
    original_out_name = ir.split_tensor_to_original_tensor_map.get(out_tensor.name, out_tensor.name)
    piece_channels = {int(t.shape[1]) for name, t in ir.tensors.items()
                      if ir.split_tensor_to_original_tensor_map.get(name) == original_out_name
                      and getattr(t, 'split_channel_lo', None) is not None}
    return len(piece_channels) > 1

def _is_scatter_concat_with_mixed_input_channels(ir, consumer_name):
    # True when consumer_name is a scatter concat whose inputs do not all have the same channel count.
    if not ir.graph.has_node(consumer_name):
        return False
    frontend = ir.graph.nodes[consumer_name]['frontend']
    if not frontend.get('is_scatter_concat', False):
        return False
    channel_counts = {it.shape[1] for it in frontend.get('input_tensors', [])}
    return len(channel_counts) > 1

def _feeds_input_split_via_scatter_concat(ir, consumer_name):
    # True when consumer_name is a scatter concat whose output feeds an input-channel-split layer.
    if not ir.graph.has_node(consumer_name):
        return False
    if not ir.graph.nodes[consumer_name]['frontend'].get('is_scatter_concat', False):
        return False
    return any(ir.graph.has_node(out_consumer) and
               ir.graph.nodes[out_consumer]['frontend'].get('is_input_channel_split', False)
               for out_consumer in ir.graph.successors(consumer_name))

def get_num_of_lines_per_tile(
        folded_height,
        k3_nodes_in_blob=0,
        num_y_tiles=1,
        current_tile_num=0,
        add_padding_line=False,
        consecutive_2_tile_blobs=1,
        prev_blob_lines_in_ddr=-1,
        is_current_blob_folding=False,
        change_read_write_length=False,
        amm_height=AMM_HEIGHT,
        extra_lines=0):
    
    folded_height_padded = folded_height + (1 if add_padding_line else 0)

    if DEBUG_2_TILE_RW_LINES and num_y_tiles == 2 and change_read_write_length:
        per_tile_num_of_lines,per_tile_read_start_line,per_tile_write_start_line = get_y_tile_sizes_2_tile_case(
            folded_height_padded,
            k3_nodes_in_blob=k3_nodes_in_blob,
            consecutive_2_tile_blobs=consecutive_2_tile_blobs,
            prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
            is_first_node_folding=is_current_blob_folding,
            amm_height=amm_height,
        )
    else:
        per_tile_num_of_lines,per_tile_read_start_line,per_tile_write_start_line = get_y_tile_sizes(folded_height_padded,k3_nodes_in_blob=k3_nodes_in_blob,amm_height=amm_height,extra_lines=extra_lines)
    num_of_lines = per_tile_num_of_lines[current_tile_num]
    read_start_line = per_tile_read_start_line[current_tile_num]
    write_start_line = per_tile_write_start_line[current_tile_num]
    return num_of_lines,read_start_line,write_start_line


def add_tensor_write_to_ddr_commands(ir,tensor_to_write,current_node_sequence,node,set_flags_mask=0,wait_flags_mask=0,
                                     consecutive_2_tile_blobs=1,prev_blob_lines_in_ddr=-1,source_grid=0,
                                     current_tile_num=0,current_htile_num=0,current_xslice_num=0,override_output_start_line=None):
    frontend = node['frontend']
    backend = node['backend']
    op_type = node['op_type']
    h_tiles = frontend['h_tiles']

    ddr_entry_type = tensor_to_write.ddr_entry.type
    if ddr_entry_type == DDREntryType.INTERMEDIATE_TENSOR:
        ddr_write_offset_type = DDRWriteOffsetType.MODEL_MEM
    elif ddr_entry_type == DDREntryType.OUTPUT_TENSOR:
        ddr_write_offset_type = DDRWriteOffsetType.OUTPUT_TENSOR
    else:
        raise ValueError('Unsupported tensor type for write. Please check...')

    even_grid = not (source_grid % 2)
    blocks_key = 'allocated_amm_blocks_for_output_even_grid' if even_grid else 'allocated_amm_blocks_for_output_odd_grid'
    amm_allocated_blocks_for_output = backend[blocks_key][current_tile_num][current_htile_num][current_xslice_num]

    # We need to make sure that allocated blocks for input are same for all amms. Otherwise we need to find a way to write to different addresses per each amm (which is currently not supported)
    allocated_blocks = amm_allocated_blocks_for_output[0]
    for current_amm_allocated_blocks in amm_allocated_blocks_for_output:
        if current_amm_allocated_blocks != allocated_blocks:
            raise ValueError('At write outputs from AMM to DDR. AMM allocation is not same for all AMMs. This is currently un-supported')

    amm_start_address = allocated_blocks[0] * ir.amm_block_size
    actual_tensor_shape = tensor_to_write.shape
    output_tensor_length = actual_tensor_shape[1]
    current_blob = ir.tiling_blobs[frontend['tiling_blob_idx']]
    k3_nodes_in_blob = current_blob.k3_nodes
    is_current_blob_folding = 'force_folding_y' in ir.graph.nodes()[current_blob.nodes_in_blob[0]]['frontend']
    grid_offset = get_grid_offset(source_grid, tensor_channels=output_tensor_length, tensor_lines=actual_tensor_shape[2])

    line_padding = frontend.get('odd_height_y_fold_write',False)
    num_of_lines, read_start_line, write_start_line = get_num_of_lines_per_tile(
            tensor_to_write.get_folded_shape()[2], k3_nodes_in_blob=k3_nodes_in_blob,
            num_y_tiles=frontend['y_tiles'], current_tile_num=current_tile_num,
            add_padding_line=line_padding, consecutive_2_tile_blobs=consecutive_2_tile_blobs,
            prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
            is_current_blob_folding=is_current_blob_folding,
            change_read_write_length=True, amm_height=ir.amm_height)

    write_size_bytes = int(math.ceil(output_tensor_length/32)*32 * num_of_lines * (256 // 32)) # 256 bytes per 32 channels
    accumulate_backend_bytes(node, 'total_tensor_write_bytes', (current_tile_num, current_htile_num, current_xslice_num), write_size_bytes)

    output_y = actual_tensor_shape[2]
    if frontend.get('is_avgPool'):
        output_y += 1
    elif frontend.get('odd_height_y_fold_write', False):
        output_y += 1
        if current_tile_num == frontend['y_tiles']-1:
            num_of_lines -= 1
    elif frontend.get('k3_1T_boundary_extra_line_write'):
        output_y += 1
        num_of_lines += 1
    output_y += frontend.get('asym_extra_lines_write', 0)
    folding_factor_x = tensor_to_write.folding_factor_x
    output_z = actual_tensor_shape[1] * (2**folding_factor_x)
    y_folding = pow(2, tensor_to_write.folding_factor_y)

    # Check if this tensor is to MXP
    tensor_to_mxp = False
    if op_type == OpType.RESIZE:
        tensor_to_write.name = tensor_to_write.name.replace('_x', '')
    unsplit_root_tensor_to_write_name = resolve_to_unsplit_root_tensor_name(ir, tensor_to_write.name)
    if ir.sync_with_MXP and unsplit_root_tensor_to_write_name in ir.tensors_to_mxp:
        if unsplit_root_tensor_to_write_name in ir.io_tensor_names:
            ddr_write_offset_type = DDRWriteOffsetType.OUTPUT_TENSOR
        else:
            ddr_write_offset_type = DDRWriteOffsetType.MXP_BASE
        tensor_to_mxp = True

    if tensor_to_mxp:
        output_x = math.ceil(actual_tensor_shape[3] / (2**folding_factor_x))
    else:
        output_x = math.ceil(actual_tensor_shape[3] / (MAX_GRID_WIDTH * (2**folding_factor_x))) * MAX_GRID_WIDTH    
    amm_write_mode = 0
    x_wrapping = 1

    #Yaron - DMA parameter change
    # Skipped when the consumer reads this as an input-channel-split slice: its own manual DMR loop
    # can't account for the channel-merge trick's effect on the DDR layout.
    # A resize is split into an _x pass and a _y pass, and the _x output's only consumer is the _y pass. For the
    # DMA decision, a resize _x output is judged by the consumers of its paired _y node's output instead.
    consumers_for_dma = list(tensor_to_write.consumers)
    producer_name_for_dma = tensor_to_write.producer
    if producer_name_for_dma in ir.graph.nodes and ir.graph.nodes[producer_name_for_dma]['op_type'] == OpType.RESIZE \
            and producer_name_for_dma.endswith('_x'):
        resize_y_name = producer_name_for_dma[:-2] + '_y'
        if resize_y_name in ir.graph.nodes:
            consumers_for_dma = list(ir.graph.successors(resize_y_name))
    consumer_is_input_channel_split = any(
        'is_input_channel_split' in ir.graph.nodes()[consumer_node_name]['frontend']
        for consumer_node_name in consumers_for_dma)
    # Same rule as the read side: a tensor that feeds a scatter concat whose output feeds an
    # input-channel-split layer keeps the plain DDR layout.
    consumer_feeds_input_split_via_scatter_concat = any(
        _feeds_input_split_via_scatter_concat(ir, consumer_node_name) or
        _is_scatter_concat_with_mixed_input_channels(ir, consumer_node_name)
        for consumer_node_name in consumers_for_dma)
    # An ADD that was channel split into pieces of unequal size writes each piece in a layout the merge trick can't follow.
    producer_is_unequal_split_add = _is_unequal_channel_split_add(ir, tensor_to_write.producer)
    # An output-channel-split, x-folded producer writes its pieces as separate folded regions, so keep its DMW plain.
    producer_output_split_with_fold_x = frontend.get('was_channel_split', False) and folding_factor_x > 0 and (tensor_to_write.folding_factor_y == 0)
    # A non-ADD producer whose tensor feeds an unequally split ADD, where that ADD also takes an input from another
    # unequally split ADD, keeps its write plain: the chained unequal pieces can't follow the merge trick's layout.
    consumed_by_unequal_split_add = (
        ir.graph.nodes[tensor_to_write.producer]['op_type'] != OpType.ADD
        and any(_is_unequal_channel_split_add(ir, c) and
                any(_is_unequal_channel_split_add(ir, p) for p in ir.graph.predecessors(c))
                for c in consumers_for_dma if ir.graph.has_node(c)))
    dma_opt_disabled = not (DEBUG_OPTIMIZE_DMA_TRANSACTIONS and (x_wrapping == 1) and (ddr_write_offset_type == DDRWriteOffsetType.MODEL_MEM)
                            and not consumer_is_input_channel_split and not consumer_feeds_input_split_via_scatter_concat
                            and not producer_is_unequal_split_add and not consumed_by_unequal_split_add
                            and not producer_output_split_with_fold_x)
    # Recorded on the producer so the numeric simulator (neuronix_ops.export_tensor_to_nxo) writes the same
    # plain layout the sequencer used for this tensor; saved with numsim.nxir.
    if tensor_to_write.producer in ir.graph.nodes:
        ir.graph.nodes[tensor_to_write.producer]['frontend']['dma_opt_disabled'] = dma_opt_disabled
    if not dma_opt_disabled:
        while (output_z > 1) and (output_z % 2 != 1) and (x_wrapping <= (MAX_X_WRAPPING//2)) and (output_x < 1024):
            output_z /= 2
            x_wrapping *= 2
            output_x *= 2
        output_z = int(math.ceil(output_z))
        if x_wrapping == 32:
            x_wrapping = 0

    write_start_line_padded = write_start_line
    if (frontend.get('asym_extra_lines_write', 0) > 0) and current_tile_num == frontend['y_tiles']-1:
        write_start_line_padded += frontend.get('asym_extra_lines_write', 0)
    y_tile_ddr_offset = write_start_line_padded * output_x * y_folding
    x_slice_ddr_offset = current_xslice_num * MAX_GRID_WIDTH * x_wrapping
    h_tile_ddr_offset = 0
    if (current_htile_num != h_tiles - 1):
        remaining_out_xslices = sum(frontend['out_xslices_per_htile'][current_htile_num+1:])
        if k3_nodes_in_blob > 0 :
            remaining_out_xslices -= ((h_tiles - 1) - current_htile_num)
        h_tile_ddr_offset = remaining_out_xslices * MAX_GRID_WIDTH * x_wrapping

    if override_output_start_line is not None:
        write_start_line = override_output_start_line
    in_grid_write_start_line = write_start_line - read_start_line
    ddr_dst_address = tensor_to_write.ddr_entry.address + y_tile_ddr_offset + grid_offset + h_tile_ddr_offset + x_slice_ddr_offset

    amm_wstrb_last = 0
    amm_wstrb_mode = 0
    last_xslice_in_htile = (current_xslice_num == frontend['out_xslices_per_htile'][current_htile_num] - 1)
    is_resize = (op_type == OpType.RESIZE)
    actual_xslices = math.ceil(actual_tensor_shape[3]/MAX_GRID_WIDTH)
    total_xslices = sum(frontend['out_xslices_per_htile'])
    skip_extra_xslice_resize = (current_htile_num == 0) and (total_xslices > actual_xslices)
    if is_resize and skip_extra_xslice_resize:
        last_xslice_in_htile = (current_xslice_num == frontend['out_xslices_per_htile'][current_htile_num] - 2)
    if tensor_to_mxp:
        if last_xslice_in_htile and (current_htile_num == frontend['h_tiles']-1):
            if current_tile_num == frontend['y_tiles'] - 1:
                send_sync_to_MXP = True

                # Every split chunk writes to MXP but MXP expects one F4 per sync group, and which
                # chunk is actually last depends on execution order, not split-name suffix - so set
                # F4 on every to-MXP write here and let program_compiler.py's post-pass strip it from
                # all but the actual last DMW command.
                if send_sync_to_MXP:
                    set_flags_mask |= 16

        if last_xslice_in_htile and (current_htile_num == 0) and (actual_tensor_shape[3] % MAX_GRID_WIDTH != 0):
            amm_wstrb_last = actual_tensor_shape[3] % MAX_GRID_WIDTH
            amm_wstrb_mode = 0

        if unsplit_root_tensor_to_write_name not in ir.mxp_tensor_to_offset:
            raise ValueError('MXP-bound tensor %s missing from ir.mxp_tensor_to_offset (node %s)' % (tensor_to_write.name, node['name']))
        # ir.mxp_tensor_to_offset is keyed by the pre-split root tensor; a scattered concat input's
        # own channel offset within the root's MXP buffer is added back at split_channel_lo*H*W
        # (channel-major layout, same convention as the read side above).
        mxp_base_address = ir.mxp_tensor_to_offset[unsplit_root_tensor_to_write_name]
        slice_channel_lo = getattr(tensor_to_write, 'split_channel_lo', None)
        if slice_channel_lo is not None:
            bytes_per_channel = int(actual_tensor_shape[2]) * int(actual_tensor_shape[3])
            mxp_base_address += slice_channel_lo * bytes_per_channel
        # Offset type still stays at 0, but address needs to change
        ddr_dst_address = mxp_base_address + y_tile_ddr_offset + grid_offset + h_tile_ddr_offset + x_slice_ddr_offset
        ir.output_ddr_offset.add(ddr_dst_address)

    if (k3_nodes_in_blob > 0) and (current_htile_num != 0):
        if last_xslice_in_htile:
            if folding_factor_x > 0 and (frontend['xslices_per_htile'][current_htile_num] % 2 == 1):
                amm_wstrb_last = int((15 - k3_nodes_in_blob) // (2 ** folding_factor_x))
            else:
                amm_wstrb_last = 15 - k3_nodes_in_blob

    if (amm_wstrb_last != 0) and (x_wrapping != 1):
        amm_wstrb_mode = 1

    tmp_set_flags_mask = set_flags_mask
    if is_resize:
        if skip_extra_xslice_resize and current_htile_num == 0 and (current_xslice_num == frontend['out_xslices_per_htile'][current_htile_num] - 1):
            return current_node_sequence
        elif skip_extra_xslice_resize and current_htile_num == 0 and last_xslice_in_htile:
            set_flags_mask |= 8
        ddr_dst_address += write_start_line * output_x
        output_x *= 2
        tmp_set_flags_mask = 0

    sequencer_entry = SequencerDDRWriteEntry(dst_address=ddr_dst_address, ddr_write_offset_type=ddr_write_offset_type, nlines_to_write=num_of_lines, output_x=output_x, output_y=output_y, output_z=output_z, amm_start_address=amm_start_address,
                                            grid_start_line=in_grid_write_start_line, src_type=DDRCommandDstSrcType.AMM, y_folding=y_folding, x_wrapping=x_wrapping,
                                            amm_wstrb_last=amm_wstrb_last, amm_wstrb_mode=amm_wstrb_mode, amm_write_mode=amm_write_mode, set_flags=tmp_set_flags_mask, wait_flags=wait_flags_mask)
    if tensor_to_mxp and (tmp_set_flags_mask & 16):
        sequencer_entry.mxp_sync_root = unsplit_root_tensor_to_write_name
    current_node_sequence.append(sequencer_entry)

    if is_resize:
        ddr_dst_address += 2*x_wrapping*MAX_GRID_WIDTH*(tensor_to_write.x_slices//2)
        if (total_xslices > actual_xslices):
            ddr_dst_address -= x_wrapping*MAX_GRID_WIDTH*(total_xslices-actual_xslices)
            if tensor_to_mxp:
                ddr_dst_address -= (actual_tensor_shape[3] % MAX_GRID_WIDTH)
        sequencer_entry = SequencerDDRWriteEntry(dst_address=ddr_dst_address, ddr_write_offset_type=ddr_write_offset_type, nlines_to_write=num_of_lines, output_x=output_x, output_y=output_y, output_z=output_z, amm_start_address=amm_start_address,
                        grid_start_line=in_grid_write_start_line, src_type=DDRCommandDstSrcType.AMM, y_folding=y_folding, x_wrapping=x_wrapping,
                        amm_wstrb_last=amm_wstrb_last, amm_wstrb_mode=amm_wstrb_mode, amm_write_mode=amm_write_mode, set_flags=set_flags_mask, wait_flags=0)
        if tensor_to_mxp and (set_flags_mask & 16):
            sequencer_entry.mxp_sync_root = unsplit_root_tensor_to_write_name
        current_node_sequence.append(sequencer_entry)

    return current_node_sequence

def get_wloc_mask_from_group(wloc_group):
    mask=0
    for grid_idx in wloc_group:
        mask+=1<<grid_idx
    return mask

def add_wloc_load_from_ddr_commands(current_node_sequence,node,tables_buffer_id,split_idx,current_tile_num=0):
    grids_wloc_commands = node['backend']['grids_cbc'].alex_wlocs[0][split_idx]
    wloc_groups_axi0 = [[0],[1]] # Single grid, 2 WLOC tables

    for wloc_group_idx,wloc_group in enumerate(wloc_groups_axi0):
        wloc_mask= get_wloc_mask_from_group(wloc_group)
        current_wloc = grids_wloc_commands[wloc_group_idx]

        current_wloc_ddr_address = current_wloc.cmd_mem_address
        wloc_length_bytes = align_table_length(len(current_wloc.cmd_mem), table_name='WLOC0')
        accumulate_backend_bytes(node, 'total_wloc_tables_bytes', current_tile_num, wloc_length_bytes)

        target_wloctable_info = DDRReadDstInfoWLOC(wloc_mask=wloc_mask,wloc_buffer_id=tables_buffer_id)
        sequencer_entry = SequencerDDRReadEntry(src_address=current_wloc_ddr_address,length = wloc_length_bytes, buffer_id=tables_buffer_id,
                        dst_type = DDRCommandDstSrcType.WLOC, dst_info = target_wloctable_info, set_flags = 0, wait_flags = 0)
        current_node_sequence.append(sequencer_entry)
    return current_node_sequence


def add_table_load_from_ddr_command_alex(table_list, tableType, write_mask, current_node_sequence,node,tables_buffer_id,split_idx =0,set_flags_mask = 0,current_tile_num=0):
    table_list_a_ddr_address = table_list.cmd_mem_address
    table_list_length_bytes = align_table_length(len(table_list.cmd_mem))

    #TODO: Dans, need to add axi
    target_read_dst_info_tables = DDRReadDstInfoTables(table_buffer_id=tables_buffer_id,axi_id=0)
    sequencer_entry = SequencerDDRReadEntry(src_address=table_list_a_ddr_address, length = table_list_length_bytes,
                                             buffer_id=tables_buffer_id,               dst_type = tableType,
                                             dst_info = target_read_dst_info_tables,   set_flags = set_flags_mask,
                                             wait_flags = 0, write_mask_alex = write_mask)
    current_node_sequence.append(sequencer_entry)
    return current_node_sequence

def add_engine_command(ir,current_node_sequence,node, set_flags_mask = 0, wait_flags_mask = 0,single_split_tables_buffer_id=0, splitted_tables_buffer_id=0,
                       reset_amm_address=1,description='',current_tile_num=0, current_htile_num=0, current_xslice_num=0, dummy_folding_command=False):
    frontend = node['frontend']
    backend = node['backend']
    op_type = node['op_type']
    is_resize_y = (op_type == OpType.RESIZE) and ('_y' in node['name'])
    amm_block_size = ir.amm_block_size
    input_blocks = backend['allocated_amm_blocks_for_input_even_grid'][current_tile_num][current_htile_num]
    output_blocks = backend['allocated_amm_blocks_for_output_even_grid'][current_tile_num][current_htile_num]
    xslices_in_htile = frontend['xslices_per_htile'][current_htile_num]
    current_blob_idx = frontend['tiling_blob_idx']
    current_blob = ir.tiling_blobs[current_blob_idx]
    k3_nodes_in_blob = current_blob.k3_nodes
        
    if op_type == OpType.ADD or ('AVERAGEPOOL2D' in node['name']):
        input_pad_value = 0
    else:
        input_pad_value = frontend['input_tensor_zp']
    
    activation_silu = node['attributes'].get('activation_silu')
    if activation_silu is not None:
        output_tensor_zp = activation_silu['output_zp'][0]
    else:
        output_tensor_zp = frontend['output_tensor_zp']
    output_pad_value = output_tensor_zp.item() if isinstance(output_tensor_zp, np.ndarray) and output_tensor_zp.size == 1 else (frontend['output_tensor'].zero_point if isinstance(output_tensor_zp, np.ndarray) else output_tensor_zp)
    
    ### Output Padding X ####################
    current_tiling_blob = ir.tiling_blobs[current_blob_idx]
    last_node_in_blob = (node['name'] == current_tiling_blob.nodes_in_blob[-1])
    # If a dummy folding command follows this (real) last x-slice of h-tile 0, that dummy command
    # becomes the true last x-slice at the blob's output, so this one must not get blob-output padding.
    input_xslices_not_double_output = (current_htile_num == 0 and frontend.get('force_folding_x') and
        frontend['input_tensor'].x_slices != 2 * frontend['output_tensor'].x_slices)
    dummy_command_follows = (not dummy_folding_command) and frontend.get('force_folding_x') and \
        ((xslices_in_htile % 2 == 1 and ((frontend['h_tiles'] == 1) or (k3_nodes_in_blob == 0 and current_htile_num == 0))) or input_xslices_not_double_output)
    if (current_htile_num == 0) and (current_xslice_num == frontend['xslices_per_htile'][current_htile_num] - 1) and not dummy_command_follows:
        output_padding_start_x = backend['output_padding_start_x']
    else:
        output_padding_start_x = 15

    ## Output Padding Y  ####################
    num_y_tiles = frontend['y_tiles']
    if dummy_folding_command: # The dummy slice need for the next
        output_padding_start_y = 0
    elif current_tile_num==(num_y_tiles-1):
        # Asymmetric-padding trigger's padded lines occupy the top of the last tile's output padding.
        output_padding_start_y = backend['output_padding_start_y'] - frontend.get('asym_extra_lines_write', 0)
    else:
        output_padding_start_y = ir.amm_height

    pad_mode = 1

    if 'input_tensors' in frontend:
        input_tensor = frontend['input_tensors'][0]
    else:
        input_tensor = frontend['input_tensor']

    input_index = 0
    current_op_channels = math.ceil(frontend['input_channels'] / MAX_GRID_WIDTH) * MAX_GRID_WIDTH
    num_packed_xslices = input_tensor.num_packed_xslices
    packing_active = is_packed_xslices(num_packed_xslices) or num_packed_xslices > 1
    if packing_active:
        # Map the logical x-slice to its lane. Lanes can hold different counts (e.g. [8,4]), so
        # lane-boundary neighbor offsets use the relevant lane's own pack size, not a uniform factor.
        xslice_idx, xslice_idx_offset, lane_pack, _ = packed_lane_of_logical(num_packed_xslices, xslices_in_htile, current_xslice_num)
        if is_packed_xslices(num_packed_xslices):
            prev_lane_pack = packed_lane_info(num_packed_xslices, xslices_in_htile, xslice_idx-1)[0] if xslice_idx > 0 else lane_pack
        else:
            prev_lane_pack = num_packed_xslices
        amm_start_address_for_input_even_grid = input_blocks[xslice_idx][input_index][0][0]*amm_block_size + (xslice_idx_offset*current_op_channels)
        if current_xslice_num == 0 and xslices_in_htile > 1:
            slice_backward_offset = NO_NEIGHBOR_SLICE
            slice_forward_offset = amm_start_address_for_input_even_grid + current_op_channels
        elif current_xslice_num == 0 and xslices_in_htile == 1:
            slice_backward_offset = NO_NEIGHBOR_SLICE
            slice_forward_offset = NO_NEIGHBOR_SLICE
        elif current_xslice_num == xslices_in_htile-1:
            # The last logical x-slice can still be the sole occupant of its own packed lane (e.g.
            # [8,8,8,8,1]: slice 32 is lane 4's only slot) - its backward neighbor is then the
            # previous lane's last slot, not this lane's base minus one op's channels (underflows).
            if xslice_idx_offset == 0:
                slice_backward_offset = input_blocks[xslice_idx-1][input_index][0][0]*amm_block_size + (prev_lane_pack-1)*current_op_channels
            else:
                slice_backward_offset = amm_start_address_for_input_even_grid - current_op_channels
            slice_forward_offset = NO_NEIGHBOR_SLICE
        elif dummy_folding_command:
            slice_backward_offset = NO_NEIGHBOR_SLICE
            slice_forward_offset = NO_NEIGHBOR_SLICE
        else:
            if xslice_idx_offset == 0:
                slice_backward_offset = input_blocks[xslice_idx-1][input_index][0][0]*amm_block_size + (prev_lane_pack-1)*current_op_channels
            else:
                slice_backward_offset = amm_start_address_for_input_even_grid - current_op_channels
            if xslice_idx_offset == lane_pack - 1:
                slice_forward_offset = input_blocks[xslice_idx+1][input_index][0][0]*amm_block_size
            else:
                slice_forward_offset = amm_start_address_for_input_even_grid + current_op_channels
    elif is_resize_y:
        pass
    else:
        amm_start_address_for_input_even_grid = input_blocks[current_xslice_num][input_index][0][0]*amm_block_size
        if current_xslice_num == 0 and xslices_in_htile > 1:
            slice_backward_offset = NO_NEIGHBOR_SLICE
            slice_forward_offset = input_blocks[current_xslice_num+1][input_index][0][0]*amm_block_size
        elif current_xslice_num == 0 and xslices_in_htile == 1:
            slice_backward_offset = NO_NEIGHBOR_SLICE
            slice_forward_offset = NO_NEIGHBOR_SLICE
        elif current_xslice_num == xslices_in_htile-1:
            slice_backward_offset = input_blocks[current_xslice_num-1][input_index][0][0]*amm_block_size
            slice_forward_offset = NO_NEIGHBOR_SLICE
        elif dummy_folding_command:
            slice_backward_offset = NO_NEIGHBOR_SLICE
            slice_forward_offset = NO_NEIGHBOR_SLICE
        else:
            slice_backward_offset = input_blocks[current_xslice_num-1][input_index][0][0]*amm_block_size
            slice_forward_offset = input_blocks[current_xslice_num+1][input_index][0][0]*amm_block_size

    if op_type == OpType.MAXPOOL:
        operation = EngineOpCode.MAXPOOL
    else:
        operation = EngineOpCode.CONV
    scaling_mode = ScalingMode.RQDIRECT

    fold_left_right = []
    amm_start_address_for_output_even_grid = []
    use_h_rule = False
    if 'force_unfolding_x' in frontend:
        scaling_mode = ScalingMode.UNFOLDING
        slices_ratio = math.ceil(frontend['out_xslices_per_htile'][current_htile_num] / xslices_in_htile)
        out_split = frontend['out_xslices_per_htile'][current_htile_num]
        in_split = frontend['xslices_per_htile'][current_htile_num]
        use_h_rule = out_split < 2 * in_split and frontend['h_tiles'] > 1 and ((out_split % 2) != 0) and (current_htile_num != frontend['h_tiles'] - 1)
        if use_h_rule:
            # Multi-h-tile unfold below 2x: each output x-slice k of this h-tile comes from slices_ratio outputs
            # per input slice. Outputs past the h-tile's real count are dropped (no dummy slot). Fold is 1 for k=0,
            # else (k+1)%2; the name offset is the first output index of this input slice.
            for i in range(slices_ratio):
                k = slices_ratio*current_xslice_num + i
                if k >= out_split:
                    break
                amm_start_address_for_output_even_grid.append(output_blocks[k][0][0]*amm_block_size)
                fold_left_right.append(1 if k == 0 else (k + 1) % 2)
            current_xslice_offset = slices_ratio*current_xslice_num
        else:
            # An odd out_xslices_per_htile means the last input x-slice unfolds into only one real output
            # slice, not slices_ratio - the second half is a dummy slot with no output_blocks entry.
            # Detect this by checking whether the full slices_ratio pair would overflow output_blocks,
            # rather than the node-level is_dummy_unfolding flag (which doesn't know which h-tile it's in).
            is_last_xslice = current_xslice_num == frontend['xslices_per_htile'][current_htile_num] - 1
            if is_last_xslice and \
               slices_ratio * (current_xslice_num + 1) > frontend['out_xslices_per_htile'][current_htile_num]:
                amm_start_address_for_output_even_grid.append(output_blocks[slices_ratio*current_xslice_num][0][0]*amm_block_size)
                fold_left_right.append(0)
                current_xslice_offset = slices_ratio*current_xslice_num
            else:
                for i in range(slices_ratio):
                    idx = slices_ratio*current_xslice_num+i
                    amm_start_address_for_output_even_grid.append(output_blocks[idx][0][0]*amm_block_size)
                    fold_left_right.append(int(i % 2))
                    current_xslice_offset = slices_ratio*current_xslice_num
    elif 'force_folding_x' in frontend:
        slices_ratio = 1
        scaling_mode = ScalingMode.FOLDING2_1
        # First htile in a multi-htile fold offsets the slice index by +1 to skip the dummy slice (and the very
        # first xslice anchors at index 0 with fold_left_right=1).
        multi_htile_first = frontend['h_tiles'] > 1 and ((frontend['xslices_per_htile'][current_htile_num] % 2 != 0) or ((dummy_command_follows or dummy_folding_command) and (frontend['xslices_per_htile'][current_htile_num] % 2 == 0)))
        if multi_htile_first and (k3_nodes_in_blob !=0) and current_htile_num == 0:
            if (current_xslice_num == 0):
                current_xslice_offset = 0
                fold_left_right.append(1)
            else:
                current_xslice_offset = int((current_xslice_num+1)/2)
                if dummy_folding_command:
                    fold_left_right.append(1)
                else:
                    fold_left_right.append(int((current_xslice_num+1) % 2))
        else:
            current_xslice_offset = int(current_xslice_num/2)
            fold_left_right.append(int((current_xslice_num+dummy_folding_command) % 2))
        amm_start_address_for_output_even_grid.append(output_blocks[current_xslice_offset][0][0]*amm_block_size)
    elif 'is_hw_x_resize' in frontend:
        if scaling_mode != ScalingMode.RQDIRECT:
            raise ValueError('Op cant be both folding and resizing. Please check...')
        scaling_mode = ScalingMode.RESIZE1_2
        slices_ratio = int(frontend['output_tensor'].xslices_per_htile[current_htile_num] / xslices_in_htile)
        for i in range(slices_ratio):
            amm_start_address_for_output_even_grid.append(output_blocks[slices_ratio*current_xslice_num+i][0][0]*amm_block_size)
            fold_left_right.append(int(i % 2))
        current_xslice_offset = slices_ratio*current_xslice_num
    else:
        slices_ratio = 1
        amm_start_address_for_output_even_grid.append(output_blocks[current_xslice_num][0][0]*amm_block_size)
        fold_left_right.append(0)
        current_xslice_offset = current_xslice_num

    if is_resize_y:
        # The whole engine-command emit loop is a no-op for RESIZE _y; the only side effect
        # is the metadata-shape lookup, which isn't observable since no entry is appended.
        return current_node_sequence

    # Hoisted invariants for the metadata applied to every emitted entry below.
    output_tensor = frontend['output_tensor']
    folded_tensor_shape = output_tensor.get_folded_shape()
    line_padding = frontend.get('odd_height_y_fold_read',False)
    num_of_lines, read_start_line, write_start_line = get_num_of_lines_per_tile(
            folded_tensor_shape[2], k3_nodes_in_blob=k3_nodes_in_blob,
            num_y_tiles=num_y_tiles, current_tile_num=current_tile_num,
            add_padding_line=line_padding, amm_height=ir.amm_height,
            extra_lines=frontend.get('asym_extra_lines_write', 0) if frontend.get('asym_extra_lines_write', 0) > 0 else 0)
    if op_type in MULTIPLE_INPUT_OPS:
        folded_input_channels = frontend['input_tensors'][0].get_folded_shape()[1]
    else:
        folded_input_channels = frontend['input_tensor'].get_folded_shape()[1]
    folded_output_channels = folded_tensor_shape[1]

    for idx in range(len(amm_start_address_for_output_even_grid)):
        if slices_ratio != 1:
            slice_num = description.split('_')[-1]
            slice_num = re.findall(r'\d+', slice_num)
            index = description.rfind(slice_num[0])
            description = description[:index]
            description += str(int(current_xslice_offset+idx))

            if use_h_rule:
                # Odd output split under multi-h-tile unfold: output 0 reads input 0; outputs 1 and 2 read input 1.
                input_slice = int((slices_ratio*current_xslice_num+idx+1)/2)
                amm_start_address_for_input_even_grid = input_blocks[input_slice][input_index][0][0]*amm_block_size
                if input_slice == 0:
                    slice_backward_offset = NO_NEIGHBOR_SLICE
                    slice_forward_offset = input_blocks[input_slice+1][input_index][0][0]*amm_block_size
                elif input_slice == xslices_in_htile-1:
                    slice_backward_offset = input_blocks[input_slice-1][input_index][0][0]*amm_block_size
                    slice_forward_offset = NO_NEIGHBOR_SLICE
                else:
                    slice_backward_offset = input_blocks[input_slice-1][input_index][0][0]*amm_block_size
                    slice_forward_offset = input_blocks[input_slice+1][input_index][0][0]*amm_block_size
                        
        if (int(slices_ratio*current_xslice_num+idx) != 0) or dummy_folding_command:
            wait_flags_mask = wait_flags_mask & ~(1 << 0)
            # The offload-complete (global DDR / F3) wait is a one-shot: the previous tile's blob-output
            # offload sets it once for the whole x-slice group, so only the first engine command must wait
            # on it. Strip it from the remaining x-slices (like the axi0 tables flag above) so they don't
            # each re-wait a never-re-set flag and deadlock.
            wait_flags_mask = wait_flags_mask & ~(1 << ir.global_ddr_transaction_flag)

        sequencer_entry = SequencerEngineOperationEntry(operation=operation, AMM_URAM_EVEN_GRID_src_addr=amm_start_address_for_input_even_grid,
                AMM_URAM_EVEN_GRID_dst_addr=amm_start_address_for_output_even_grid[idx], output_padding_start_x=output_padding_start_x,
                output_padding_start_y=output_padding_start_y, output_pad_value=output_pad_value, input_pad_value=input_pad_value, pad_mode=pad_mode,
                slice_backward_offset=slice_backward_offset, slice_forward_offset=slice_forward_offset, fold_left_right=fold_left_right[idx],
                wloc_buffer_id=splitted_tables_buffer_id, rt_buffer_id=splitted_tables_buffer_id, rqparams_buffer_id=splitted_tables_buffer_id, nlt_buffer_id=single_split_tables_buffer_id,
                scaling_mode=scaling_mode, reset_amm_address=reset_amm_address, set_flags=set_flags_mask, wait_flags=wait_flags_mask, description=description)

        # We add the below to sequencer engine command metadata so that it can be used by FPGA simulator to gather intermediate tensors tiles and compare to numeric simulator nxo files
        sequencer_entry.num_of_lines = num_of_lines
        sequencer_entry.write_start_line = write_start_line
        sequencer_entry.read_start_line = read_start_line
        sequencer_entry.conv_input_channels = folded_input_channels
        sequencer_entry.conv_output_channels = folded_output_channels
        current_node_sequence.append(sequencer_entry)
    return current_node_sequence

def read_needed_tensors_from_ddr(ir,tensors_info,node,wait_flags_allocator,current_node_sequence,current_tile_num=0,current_htile_num=0,read_ddr_wait_flag = 0,
                                read_ddr_set_flag=0,is_immediate_read = False,conv_in_process=False,tile_write_in_process=False):                                 
    # Check if any input tensors need to be loaded from ddr
    read_ddr_to_amm_set_flag_mask = 0
    read_in_process = False
    tensors_read_during_this_split=0
    number_of_tensors_read_from_ddr=0
    is_first_read_in_tile = True
    for tensor_idx,tensor_to_load_from_ddr in enumerate(tensors_info):
        if tensor_to_load_from_ddr.load_at_execution_of_tile_idx == current_tile_num and tensor_to_load_from_ddr.load_at_execution_of_h_tile_idx == current_htile_num :
            number_of_tensors_read_from_ddr+=1
            tensors_read_during_this_split+=1
            read_in_process = True
            tensor_to_load_from_ddr_name = tensor_to_load_from_ddr.name
            if is_graph_input_derived_tensor(ir, tensor_to_load_from_ddr_name):
                ddr_read_offset_type=DDRReadOffsetType.INPUT_TENSOR
            else:
                ddr_read_offset_type=DDRReadOffsetType.MODEL_MEM
            # No need to condition intermediate tensor read by its write flag since we alreaedy made sure it was written to ddr at the offloading node
            # Since DDR read command is blocking we dont need wmt wait flag as the ddr read to amm will not start until wmt read ends anyway
            is_first_op_first_tile = (ir.lexicographical_topological_sorted_graph[0] == node['name']) and (current_tile_num==0) and (current_htile_num==0) and (tensor_to_load_from_ddr.tensor_xslice_num==0)
            ddr_read_wait_flags = []
            if not is_first_op_first_tile: # If its first op first tile we dont need to wait since nothing happend before
                if is_immediate_read:
                    ddr_read_wait_flags.append(read_ddr_set_flag) # We need to wait for previous op to finish since mem is full (this is the reason we offloaded this tensor)
                    if tile_write_in_process and number_of_tensors_read_from_ddr==1: # If its an immediate read and tile write is in process we need to wait for it to finish before starting read. This is relevant only for the 1st tensor we read
                        ddr_read_wait_flags.append(ir.global_ddr_transaction_flag)
                        # We clean the tile_write_in_process at end of all tensors read since we want each read to wait for last read
                else:
                    if tile_write_in_process:
                        ddr_read_wait_flags.append(read_ddr_set_flag) #If its a tile read and write is in process we need to wait for it to finish
                        # We clean the tile_write_in_process at end of all tensors read since we want each read to wait for last read
            avoid_ddr_read_while_conv = debug_flags.DEBUG_AVOID_DDR_READ_WHILE_CONV or node['frontend'].get('avoid_ddr_read_while_conv', False)
            if avoid_ddr_read_while_conv and conv_in_process:
                ddr_read_wait_flags = [read_ddr_wait_flag]
                if tensors_read_during_this_split!=1:
                    ddr_read_wait_flags.append(read_ddr_set_flag) # If DDR reads are blocking and its not the 1st tensor we read we wait for last tensor read to finishe

            read_ddr_to_amm_set_flag_mask = wait_flags_allocator.get_mask_from_flags([read_ddr_set_flag])
            read_ddr_to_amm_wait_flag_mask = wait_flags_allocator.get_mask_from_flags(ddr_read_wait_flags)
            node['backend']['read_ddr_to_amm_set_flag_mask'] = read_ddr_to_amm_set_flag_mask

            # We always wait for last op to complete or current op write to ddr to complete
            # We dont allow read during last op conv since we assume that tensors were offloaded to DDR because of lack of mem.
            # Only after last op is complete, its input tensors can be deallocated and allow space for current op input read from DDR
            consumer_node = tensor_to_load_from_ddr.consumer_node # Since load from ddr can be executed by node X(node) but actually load input of node Y(consumer_node) we need to get wmt and all tensor data from consumer node
            
            # #Yaron - Only first grid in first tensor within each tile needs to have a wait mask
            if is_first_read_in_tile:
                wait_mask = read_ddr_to_amm_wait_flag_mask
            else:
                wait_mask = 0
            #Yaron - Last ddr read is when last grid and last tensor in this specific tile
            last_read_ddr_to_amm = True
            for future_tensor in tensors_info[tensor_idx+1:]:
                if (future_tensor.load_at_execution_of_tile_idx == current_tile_num
                        and future_tensor.load_at_execution_of_h_tile_idx == current_htile_num):
                    last_read_ddr_to_amm = False
                    break
            if last_read_ddr_to_amm:
                set_flags = read_ddr_to_amm_set_flag_mask
            else:
                set_flags = 0

            wmt_read_ready_flag_mask = wait_mask

            current_node_sequence = add_input_load_from_ddr_commands(ir,tensor_to_load_from_ddr,current_node_sequence,consumer_node,set_flags_mask=set_flags,
                    wait_flags_mask=wmt_read_ready_flag_mask,target_grid=0,ddr_read_offset_type=ddr_read_offset_type) # If input to layer is in ddr add load ddr commands
            
            is_first_read_in_tile = False
    # Only emit the read-vs-conv serializing NOP for PRE-reads: immediate reads are already
    # serialized via the local-DDR (F2) flag chain, and emitting the NOP there double-consumes
    # the read's F2 set and can leave an orphaned global-DDR (F3) wait downstream.
    if (debug_flags.DEBUG_AVOID_DDR_READ_WHILE_CONV or node['frontend'].get('avoid_ddr_read_while_conv', False)) and read_in_process and not is_immediate_read:
        read_in_process = False
        write_ddr_from_amm_set_flag_mask = read_ddr_to_amm_set_flag_mask
        node['backend']['write_ddr_from_amm_set_flag_mask'] = write_ddr_from_amm_set_flag_mask
        if read_ddr_set_flag==2:
            sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags = read_ddr_to_amm_set_flag_mask, wait_flags = read_ddr_to_amm_set_flag_mask)
        else:
            sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags = 0, wait_flags = read_ddr_to_amm_set_flag_mask)
        current_node_sequence.extend([sequencer_entry])
    if tile_write_in_process and read_in_process: # We clean the tile_write_in_process at end of all tensors read since we want each read to wait for last read.
                                                  # We also check that read_in_process, to make sure that indeed at least one tensor was read (tensor_to_load_from_ddr.load_at_execution_of_tile_idx == current_tile_num)
        tile_write_in_process = False
                
    return read_in_process, tile_write_in_process          

# Get the command to load the next instructions from DDR
def get_command_to_load_next_instructions(set_mask, start_address, num_instructions, sequencer_program_ddr_address):
    return SequencerDDRReadEntry(
        src_address = start_address * BYTES_IN_SEQUENCER_COMMAND + sequencer_program_ddr_address,
        length = num_instructions * BYTES_IN_SEQUENCER_COMMAND,
        dst_type = DDRCommandDstSrcType.PROGRAM,
        dst_info = DDRReadDstInfoProgram(program_mem_start_address=start_address * BYTES_IN_SEQUENCER_COMMAND), # Seems to be unused
        set_flags = set_mask,
        wait_flags = 0)

# Add special operation / no op that waits on flag set above to complete
def get_nop_command_to_wait_for_flag(wait_mask):
    return SequencerCustomOperationEntry(
        operation=CustomeOpCode.NOP,
        set_flags = 0,
        wait_flags = wait_mask)

# Split sequencer program commands list into groups of instructions
def split_sequencer_program(ir, sequencer_program_ddr_address):
    instruction_limit = PROGRAM_MEMORY_NUM_COMMANDS
    original_sequencer_program = ir.sequencer_program.commands_list
    new_sequencer_program = []

    instruction_count = 0
    for original_instruction_index, original_instruction in enumerate(original_sequencer_program):
        new_sequencer_program.append(original_instruction)
        instruction_count += 1
        if instruction_count == instruction_limit-2:

            # Load the next (instruction_limit) instructions, or fewer if there are not that many
            num_instructions_to_load = instruction_limit
            num_original_instructions_remaining = len(original_sequencer_program) - original_instruction_index - 1
            if num_original_instructions_remaining < instruction_limit:
                num_instructions_to_load = num_original_instructions_remaining
                # Use multiple of 2 lines (required by hardware)
                if num_instructions_to_load % 2 == 1:
                    num_instructions_to_load += 1

            # Add DMA request to read the new instructions. Set F1.
            load_new_instructions_command = get_command_to_load_next_instructions(
                set_mask=2, start_address=len(new_sequencer_program)+2, num_instructions=num_instructions_to_load,
                sequencer_program_ddr_address=sequencer_program_ddr_address)
            new_sequencer_program.append(load_new_instructions_command)

            # Add special operation / no op that waits on flag set above (F1) to complete
            nop_command = get_nop_command_to_wait_for_flag(wait_mask=2)
            new_sequencer_program.append(nop_command)

            # Reset instruction count
            instruction_count = 0

    ir.sequencer_program.commands_list = new_sequencer_program

# Get the command to load zeros from DDR into the final block of each AMM grid
def get_load_zeros_from_ddr_command(ir):
    # Get the previously-created tensor with zeros
    zero_tensor = ir.tensors['SYSTEM_zero_tensor']

    # Write only the last block
    amm_start_address = (ir.amm_num_blocks - 1) * ir.amm_block_size
    input_amm_info = DDRReadDstInfoAMM(amm_start_address=amm_start_address)

    # Create the DDR read command
    sequencer_entry = SequencerDDRReadEntry(
        src_address = zero_tensor.ddr_entry.address,
        ddr_read_offset_type = DDRReadOffsetType.MODEL_MEM,
        amm_write_mode = 0,
        amm_start_address = amm_start_address,
        nlines_to_read = ir.amm_height,
        input_x = MAX_GRID_WIDTH,
        input_y = ir.amm_height,
        input_z = ir.amm_block_size,
        y_folding = 1,
        x_wrapping = 1,
        dst_type = DDRCommandDstSrcType.AMM,
        dst_info = input_amm_info,
        set_flags = 0,
        wait_flags = 0)

    return sequencer_entry

# Return a sequence of commands at the start of the sequencer program
def generate_initial_command_sequence(ir: internal_representation.IR):
    initial_command_sequence = []

    # Add command to load zeros from DDR into last (64th) block of each AMM
    load_zeros_command = get_load_zeros_from_ddr_command(ir)
    initial_command_sequence.append(load_zeros_command)

    return initial_command_sequence

def add_blob_outputs_write_to_ddr_commands(ir, current_tiling_blob, node, current_node_sequence,
                                           wait_flags_allocator, axi0_tables_load_flag, axi1_tables_load_flag,
                                           local_ddr_transaction_flag, global_ddr_transaction_flag,
                                           last_tile_of_last_node_in_last_blob,
                                           tile_read_in_process, tile_write_in_process,
                                           consecutive_2_tile_blobs, prev_blob_lines_in_ddr,
                                           current_tile_num, current_htile_num):
    """Write all of a tiling blob's outputs to DDR, emitting the surrounding NOP/sync commands."""
    write_ddr_from_amm_set_flags = None
    num_blob_outputs = len(current_tiling_blob.outputs)
    for output_tensor_index, blob_output_tensor in enumerate(current_tiling_blob.outputs):
        output_tensor_name = blob_output_tensor.name
        if not DEBUG_REMOVE_ENGINE_COMMAND:
            if last_tile_of_last_node_in_last_blob:
                write_ddr_from_amm_set_flag_mask = 0
            else:
                write_ddr_from_amm_set_flag_mask = wait_flags_allocator.get_mask_from_flags([local_ddr_transaction_flag])
            write_ddr_from_amm_wait_flag_mask = wait_flags_allocator.get_mask_from_flags([local_ddr_transaction_flag])
            if output_tensor_index == 0:
                sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags=0, wait_flags=write_ddr_from_amm_wait_flag_mask)
                current_node_sequence.extend([sequencer_entry])
                if write_ddr_from_amm_set_flag_mask != 0:
                    sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags=write_ddr_from_amm_set_flag_mask, wait_flags=0)
                    current_node_sequence.extend([sequencer_entry])
            ddr_write_wait_flags = []
        else:
            ddr_write_wait_flags = [axi0_tables_load_flag, axi1_tables_load_flag]
        # A 2-tile blob followed by another 2-tile blob can produce 2 writes with no read in
        # between: tile1 is read mid-blob, then both tiles' outputs write before the next blob reads.
        if tile_read_in_process or tile_write_in_process:
            ddr_write_wait_flags.append(global_ddr_transaction_flag)
            tile_read_in_process = False
        if output_tensor_name in ir.outputs and last_tile_of_last_node_in_last_blob:
            write_ddr_from_amm_set_flags = [local_ddr_transaction_flag]
        else:
            write_ddr_from_amm_set_flags = [global_ddr_transaction_flag]
            tile_write_in_process = True

        ddr_write_wait_flag_mask = wait_flags_allocator.get_mask_from_flags(ddr_write_wait_flags)

        grids_to_write = 1
        for current_grid in range(grids_to_write):
            if current_grid == 0 and output_tensor_index == 0:
                wait_flags = ddr_write_wait_flag_mask
            else:
                wait_flags = 0
            last_write_ddr_from_amm = (current_grid == (grids_to_write - 1)) and (output_tensor_index == num_blob_outputs - 1)
            set_flags = []
            if last_write_ddr_from_amm:
                set_flags = write_ddr_from_amm_set_flags
            set_flags_mask = wait_flags_allocator.get_mask_from_flags(set_flags)
            output_tensor_producer_node_name = blob_output_tensor.producer
            output_tensor_producer_node = ir.graph.nodes()[output_tensor_producer_node_name]
            producer_xslices = output_tensor_producer_node['frontend']['out_xslices_per_htile'][current_htile_num]
            
            for current_xslice_num in range(producer_xslices):
                set_mask = set_flags_mask if (current_xslice_num == producer_xslices - 1) else 0
                wait_mask = wait_flags if (current_xslice_num == 0) else 0
                current_node_sequence = add_tensor_write_to_ddr_commands(ir, blob_output_tensor, current_node_sequence, output_tensor_producer_node,
                        set_flags_mask=set_mask, wait_flags_mask=wait_mask, source_grid=current_grid,
                        consecutive_2_tile_blobs=consecutive_2_tile_blobs, prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
                        current_tile_num=current_tile_num, current_htile_num=current_htile_num, current_xslice_num=current_xslice_num)

    if node_avoids_ddr_write(node) or last_tile_of_last_node_in_last_blob:
        tile_write_in_process = False
        write_ddr_from_amm_set_flag_mask = wait_flags_allocator.get_mask_from_flags(write_ddr_from_amm_set_flags)
        node['backend']['write_ddr_from_amm_set_flag_mask'] = write_ddr_from_amm_set_flag_mask
        sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags=0, wait_flags=write_ddr_from_amm_set_flag_mask)
        current_node_sequence.extend([sequencer_entry])

    return current_node_sequence, tile_read_in_process, tile_write_in_process


def generate_layer_command_sequence(ir: internal_representation.IR,node,wait_flags_allocator,single_split_tables_buffer_id,
                                    splitted_tables_buffer_allocator,current_tile_num=0,current_htile_num=0,tile_read_in_process=False,
                                    tile_write_in_process=False,consecutive_2_tile_blobs=1,prev_blob_lines_in_ddr=-1,conv_in_process=False):
    frontend = node['frontend']
    backend = node['backend']
    op_type = node['op_type']
    get_mask = wait_flags_allocator.get_mask_from_flags

    axi0_tables_load_flag = ir.axi0_tables_load_flag
    axi1_tables_load_flag = ir.axi1_tables_load_flag
    local_ddr_transaction_flag = ir.local_ddr_transaction_flag
    global_ddr_transaction_flag = ir.global_ddr_transaction_flag

    current_node_sequence = []
    # Need to add code for knowing which blob we are and if we are 1st node in blob, add read comman, if we are last add write command
    current_blob_idx = frontend['tiling_blob_idx']
    current_tiling_blob = ir.tiling_blobs[current_blob_idx]
    last_node_in_blob = (node['name'] == current_tiling_blob.nodes_in_blob[-1])
    last_blob = current_blob_idx == (len(ir.tiling_blobs) - 1)
    last_tile = (current_tile_num == (current_tiling_blob.y_tiles - 1)) and (current_htile_num == (current_tiling_blob.h_tiles - 1))
    last_tile_of_last_node_in_last_blob = (last_node_in_blob and last_tile and last_blob)
    if op_type == OpType.CONCAT:
        current_op_wait_flags = []
        read_ddr_to_amm_set_flag = None
        if 'tensors_to_load_immediately_from_ddr' in backend:
            tensors_info = backend['tensors_to_load_immediately_from_ddr']
            # A scattered concat input (aliased directly into the concat's output DDR region via
            # split_tensor_to_original_tensor_map) is never re-read here - its producer already
            # wrote its data straight into the concat's output tensor, so no DMR is needed.
            # Skip only the inputs that this concat itself scatters: those alias into THIS concat's output region.
            # A tensor that is aliased into some other (scattered) concat must still be read by any other consumer.
            this_output = node['outputs'][0]
            tensors_info = [t for t in tensors_info
                            if ir.split_tensor_to_original_tensor_map.get(getattr(t, 'name', None)) != this_output]
            if len(tensors_info) > 0:
                read_ddr_to_amm_set_flag = local_ddr_transaction_flag
                read_in_process, tile_write_in_process = read_needed_tensors_from_ddr(ir, tensors_info, node, wait_flags_allocator, current_node_sequence,
                                                current_tile_num=current_tile_num, current_htile_num=current_htile_num, read_ddr_wait_flag=read_ddr_to_amm_set_flag,
                                                read_ddr_set_flag=read_ddr_to_amm_set_flag, is_immediate_read=True, conv_in_process=conv_in_process,
                                                tile_write_in_process=tile_write_in_process)
                # In concat op there is no actual op so we insert a wait op to make sure needed tensors are in mem
                # We then set back the flag since next op will be waiting on it
                if read_in_process:
                    read_ddr_to_amm_set_flag_mask = get_mask([read_ddr_to_amm_set_flag])
                    sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags=0, wait_flags=read_ddr_to_amm_set_flag_mask)
                    current_node_sequence.extend([sequencer_entry])
                    current_op_wait_flags.append(read_ddr_to_amm_set_flag)
                    sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags=read_ddr_to_amm_set_flag_mask, wait_flags=0)
                    current_node_sequence.extend([sequencer_entry])
                    current_op_wait_flags.append(read_ddr_to_amm_set_flag)

    elif (op_type == OpType.RESIZE) and ('_y' in node['name']):
        frontend['input_tensor'].ddr_entry = frontend['output_tensor'].ddr_entry
        for index in range(len(current_tiling_blob.outputs)):
            if current_tiling_blob.outputs[index].name == frontend['output_tensor'].name:
                break
        current_tiling_blob.outputs[index] = frontend['input_tensor']

        current_node_sequence, tile_read_in_process, tile_write_in_process = add_blob_outputs_write_to_ddr_commands(
                ir, current_tiling_blob, node, current_node_sequence,
                wait_flags_allocator, axi0_tables_load_flag, axi1_tables_load_flag,
                local_ddr_transaction_flag, global_ddr_transaction_flag,
                last_tile_of_last_node_in_last_blob,
                tile_read_in_process, tile_write_in_process,
                consecutive_2_tile_blobs, prev_blob_lines_in_ddr,
                current_tile_num, current_htile_num)
    else:
        axi0_tables_read_set_flag_mask = get_mask([axi0_tables_load_flag])
        current_op_wait_flags = [axi0_tables_load_flag]

        intermediate_tensor_read_wait_flag = None
        read_ddr_to_amm_set_flag = None
        grids_cbc = backend['grids_cbc']
        num_wloc_splits = len(grids_cbc.alex_wlocs[0])

        # Check if any tensors need to be offloaded to DDR
        if 'optimal_offload_point_for_tensor' in backend:
            offloaded_tensor_name = backend['optimal_offload_point_for_tensor']

            # We use the flag allocated for ddr write (making sure its producer op finished) and deallocate it
            ddr_write_intermediate_tensor_wait_flag_mask = get_mask([local_ddr_transaction_flag])

            offloaded_tensor = ir.tensors[offloaded_tensor_name]
            offloaded_tensor_producer_node = ir.graph.nodes()[offloaded_tensor.producer]
            producer_xslices = offloaded_tensor_producer_node['frontend']['out_xslices_per_htile'][current_htile_num]
            
            write_intermediate_from_amm_to_ddr_set_flag_mask = get_mask([local_ddr_transaction_flag])
            backend['write_intermediate_from_amm_to_ddr_set_flag_mask'] = write_intermediate_from_amm_to_ddr_set_flag_mask

            grids_to_write = 1
            for current_grid in range(grids_to_write):
                wait_flags = ddr_write_intermediate_tensor_wait_flag_mask if current_grid == 0 else 0
                last_write_ddr_from_amm = (current_grid == grids_to_write - 1)
                set_flags = write_intermediate_from_amm_to_ddr_set_flag_mask if last_write_ddr_from_amm else 0

                for current_xslice_num in range(producer_xslices):
                    set_mask = set_flags if (current_xslice_num == producer_xslices - 1) else 0
                    wait_mask = wait_flags if (current_xslice_num == 0) else 0
                    current_node_sequence = add_tensor_write_to_ddr_commands(ir, offloaded_tensor, current_node_sequence, offloaded_tensor_producer_node, set_flags_mask=set_mask,
                            consecutive_2_tile_blobs=consecutive_2_tile_blobs, prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
                            current_htile_num=current_htile_num, wait_flags_mask=wait_mask, source_grid=current_grid, current_xslice_num=current_xslice_num) # If output from layer should be written to ddr (e.g. workload output) add write ddr commands

        splitted_tables_buffer_id = None
        is_first_op = ir.lexicographical_topological_sorted_graph[0] == node['name']
        op_set_flag = local_ddr_transaction_flag

        for split_idx in range(num_wloc_splits):
            allocated_buffers_of_splitted_tables = splitted_tables_buffer_allocator.get_allocated_buffers_indexes()
            splitted_tables_buffer_id = splitted_tables_buffer_allocator.allocate_buffer()
            if len(allocated_buffers_of_splitted_tables) > 1:
                raise ValueError('Didnt expect all buffers to be allocated!')
            elif len(allocated_buffers_of_splitted_tables) == 1:
                splitted_tables_buffer_allocator.deallocate_buffer(allocated_buffers_of_splitted_tables[0])

            #1xNLF (Itamar asks to load the NLF before all others tablesd)
            if split_idx == 0:
                current_node_sequence = add_table_load_from_ddr_command_alex(grids_cbc.nlf[0], DDRCommandDstSrcType.NONLINEARFUNCTION, 0,
                        current_node_sequence, node, tables_buffer_id=single_split_tables_buffer_id, current_tile_num=current_tile_num) # Add the commands to load RQPARAMS from DDR

            # We first add all the table read commands: wloc, rqloc, rqparams
            current_node_sequence = add_wloc_load_from_ddr_commands(current_node_sequence, node, splitted_tables_buffer_id, split_idx, current_tile_num=current_tile_num) # Add the commands to load WLOC split

            #2xRQPARAM
            current_node_sequence = add_table_load_from_ddr_command_alex(grids_cbc.alex_rqParam[split_idx][0], DDRCommandDstSrcType.RQPARAMS, 1,
                    current_node_sequence, node, tables_buffer_id=splitted_tables_buffer_id, current_tile_num=current_tile_num) # Add the commands to load RQPARAMS from DDR

            #1xRT (Alex: have to be last because of set_flags_mask settings - they are important for last table)
            current_node_sequence = add_table_load_from_ddr_command_alex(grids_cbc.RTable[split_idx], DDRCommandDstSrcType.RESULTTABLE, 0,
                    current_node_sequence, node, tables_buffer_id=splitted_tables_buffer_id, set_flags_mask=axi0_tables_read_set_flag_mask, current_tile_num=current_tile_num) # Add the commands to load RQPARAMS from DDR

            if split_idx == 0:
                # Check if any input tensors need to be loaded from ddr
                read_ddr_to_amm_set_flag = None
                if 'tensors_to_load_immediately_from_ddr' in backend:
                    tensors_info = backend['tensors_to_load_immediately_from_ddr']
                    if len(tensors_info) > 0:
                        read_ddr_to_amm_set_flag = local_ddr_transaction_flag
                        read_in_process, tile_write_in_process = read_needed_tensors_from_ddr(ir, tensors_info, node, wait_flags_allocator, current_node_sequence,
                                                                                current_tile_num=current_tile_num, current_htile_num=current_htile_num, read_ddr_wait_flag=read_ddr_to_amm_set_flag,
                                                                                read_ddr_set_flag=read_ddr_to_amm_set_flag, is_immediate_read=True,
                                                                                conv_in_process=conv_in_process, tile_write_in_process=tile_write_in_process)
                        if read_in_process or not (is_first_op and current_tile_num == 0 and current_htile_num == 0):
                            current_op_wait_flags.append(read_ddr_to_amm_set_flag)

                if 'tensors_to_offload_to_ddr' in backend: # If this node needs to offload tensors, make sure its execution starts only after offloaded tensors were written to DDR
                    tensors_to_offload_to_ddr = backend['tensors_to_offload_to_ddr']
                    if len(tensors_to_offload_to_ddr) > 1:
                        raise ValueError('Currently only single tensor offload is supported')
                    if not node_avoids_ddr_write(node): # if avoiding DDR-write-while-conv then DDR write was blocking (not in parallel to a conv) so it must have finished before reaching actual offloading point (The place where we actually need its AMM)
                        intermediate_tensor_read_wait_flag = local_ddr_transaction_flag
                        current_op_wait_flags.append(intermediate_tensor_read_wait_flag)

            # This node reuses the AMM lane the previous tile's blob output was offloaded from (see
            # offload_wait_node in program_compiler) - wait for that offload's F3 before overwriting the
            # lane (cross-tile WAR), then clear tile_write_in_process since F3 is one-shot and consumed.
            if split_idx == 0 and backend.get('offload_wait_node') and tile_write_in_process:
                current_op_wait_flags.append(global_ddr_transaction_flag)
                tile_write_in_process = False

            # And finally we can perform the command
            current_op_wait_flags.extend([op_set_flag]) # We wait for first read or last op to finish
            engine_wait_flags_mask = get_mask(current_op_wait_flags)

            if local_ddr_transaction_flag in current_op_wait_flags: # This gets wait flag back to having only tables axi0 and axi1 read wait, ready for nexe split
                current_op_wait_flags.remove(local_ddr_transaction_flag)
            if global_ddr_transaction_flag in current_op_wait_flags: # One-shot offload wait: drop it so later splits don't re-wait the consumed flag
                current_op_wait_flags.remove(global_ddr_transaction_flag)

            if not DEBUG_REMOVE_ENGINE_COMMAND:
                if split_idx == (num_wloc_splits - 1):
                    following_nodes_params = frontend['following_nodes_params']
                    engine_command_set_flags = [local_ddr_transaction_flag]

                    if len(following_nodes_params) != 0: # TODO, when we implement write to DDR in middle of workload we will have to do this even when there are following nodes and move the per following node set flag to the DDR read
                        if 'tensor_to_offload_name' in backend: #This node produces an intermediate tensor that will be written to ddr before next op
                            intermediate_tensor_name = backend['tensor_to_offload_name']
                            engine_command_ddr_write_intermediate_tensor_set_flag = local_ddr_transaction_flag
                            engine_command_set_flags.append(engine_command_ddr_write_intermediate_tensor_set_flag)
                            ir.intermediate_ddr_tensors[intermediate_tensor_name].write_wait_flag = engine_command_ddr_write_intermediate_tensor_set_flag

                    engine_command_set_flag_mask = get_mask(engine_command_set_flags)
                else: # If we are not in last split
                    engine_command_set_flag_mask = get_mask([local_ddr_transaction_flag])

                reset_amm_address = 1 if split_idx == 0 else 0
                xslices_in_htile = frontend['xslices_per_htile'][current_htile_num]
                node_name = node['name']
                tile_htile_prefix = node_name + '_tile' + str(current_tile_num) + '_htile' + str(current_htile_num) + '_xslice'

                for current_xslice_num in range(xslices_in_htile):
                    engine_command_description = tile_htile_prefix + str(current_xslice_num)
                    current_node_sequence = add_engine_command(ir, current_node_sequence, node, set_flags_mask=engine_command_set_flag_mask,
                                        wait_flags_mask=engine_wait_flags_mask, single_split_tables_buffer_id=single_split_tables_buffer_id,
                                        splitted_tables_buffer_id=splitted_tables_buffer_id, reset_amm_address=reset_amm_address,
                                        description=engine_command_description, current_tile_num=current_tile_num, current_htile_num=current_htile_num,
                                        current_xslice_num=current_xslice_num) # Add the command to initiate engine operation (e.g. Conv)
                # if there is a folding, and the additional dummy slice is needed, we add it here
                # h-tile 0 also needs the dummy slice whenever this tensor's total input xslices
                # aren't exactly double the output xslices (force_folding_x means output should be
                # half of input; a mismatch here means h-tile 0's real xslices alone won't cover it).
                input_xslices_not_double_output = (current_htile_num == 0 and frontend.get('force_folding_x') and
                    frontend['input_tensor'].x_slices != 2 * frontend['output_tensor'].x_slices)
                if (xslices_in_htile % 2 == 1 and frontend.get('force_folding_x') and
                    ((frontend['h_tiles'] == 1) or (current_tiling_blob.k3_nodes == 0 and current_htile_num == 0))) or input_xslices_not_double_output:
                    # We add a dummy command for the last slice
                    engine_command_description = tile_htile_prefix + str(current_xslice_num + 1)
                    current_node_sequence = add_engine_command(ir, current_node_sequence, node, set_flags_mask=engine_command_set_flag_mask,
                                        wait_flags_mask=engine_wait_flags_mask, single_split_tables_buffer_id=single_split_tables_buffer_id,
                                        splitted_tables_buffer_id=splitted_tables_buffer_id, reset_amm_address=reset_amm_address,
                                        description=engine_command_description, current_tile_num=current_tile_num, current_htile_num=current_htile_num,
                                        current_xslice_num=current_xslice_num, dummy_folding_command=True)

                conv_in_process = True
            else: # if DEBUG_REMOVE_ENGINE_COMMAND
                engine_command_set_flag_mask = 0
                for following_node_params in frontend['following_nodes_params']:
                    following_node = ir.graph.nodes[following_node_params[0]]
                    following_node['backend']['previous_op_set_flag'] = engine_wait_flags_mask
            backend['engine_command_set_flag_mask'] = engine_command_set_flag_mask

        # We then add the read from DDR to AMM of next tile/h-tile.
        # next_tile_tensors_* fires on the LAST h-tile (pre-reads next y-tile, htile=0).
        # next_htile_tensors_* fires on non-last h-tiles (pre-reads next h-tile within same y-tile).
        read_ddr_to_amm_set_flag = None
        is_last_htile = current_htile_num == frontend['h_tiles'] - 1
        pre_read_marker = 'next_tile_tensors_to_load_from_ddr' if is_last_htile else 'next_htile_tensors_to_load_from_ddr'
        tensors_info = backend.get(pre_read_marker, [])
        if tensors_info:
            read_ddr_to_amm_set_flag = global_ddr_transaction_flag
            current_op_wait_flags.append(read_ddr_to_amm_set_flag)
            read_in_process, tile_write_in_process = read_needed_tensors_from_ddr(ir, tensors_info, node, wait_flags_allocator, current_node_sequence,
                                                            current_tile_num=current_tile_num, current_htile_num=current_htile_num, read_ddr_wait_flag=local_ddr_transaction_flag,
                                                            read_ddr_set_flag=read_ddr_to_amm_set_flag, is_immediate_read=False,
                                                            conv_in_process=conv_in_process, tile_write_in_process=tile_write_in_process)
            if read_in_process:
                tile_read_in_process = True
        if ('wait_for_last_tile_write_end' in backend) and tile_write_in_process:
            tile_num_for_wait = backend['wait_for_last_tile_write_end']
            if current_tile_num == tile_num_for_wait and not node_avoids_ddr_write(node):
                tile_write_in_process = False
                write_ddr_from_amm_wait_flag_mask = get_mask([global_ddr_transaction_flag])
                sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags=0, wait_flags=write_ddr_from_amm_wait_flag_mask)
                current_node_sequence.extend([sequencer_entry])

        if ('next_tile_read_node' in backend) and tile_write_in_process: #If we got to the point of next tile read and write is still in progress we need to wait for its end since in this point we deallocated the written tensor AMM
            tile_write_in_process = False
            write_ddr_from_amm_wait_flag_mask = get_mask([global_ddr_transaction_flag])
            sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags=0, wait_flags=write_ddr_from_amm_wait_flag_mask)
            current_node_sequence.extend([sequencer_entry])

        # We then add the write from AMM to DDR if needed
        if last_node_in_blob: # Output tensor is blob/workload output and needs to be written to DDR
            current_node_sequence, tile_read_in_process, tile_write_in_process = add_blob_outputs_write_to_ddr_commands(
                    ir, current_tiling_blob, node, current_node_sequence,
                    wait_flags_allocator, axi0_tables_load_flag, axi1_tables_load_flag,
                    local_ddr_transaction_flag, global_ddr_transaction_flag,
                    last_tile_of_last_node_in_last_blob,
                    tile_read_in_process, tile_write_in_process,
                    consecutive_2_tile_blobs, prev_blob_lines_in_ddr,
                    current_tile_num, current_htile_num)

    return current_node_sequence,tile_read_in_process,tile_write_in_process

