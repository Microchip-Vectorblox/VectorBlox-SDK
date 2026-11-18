import os
import math
import copy
import pickle
import gc
import time
import numpy as np
import networkx as nx
from collections import OrderedDict
from tqdm import tqdm

from common.working_node_state_machine import (
    add_stalls, check_nop_balance, generate_grids_1x1_cbc,
    generate_grids_rq_cbc_alex2
)
from compiler.tsnp_cbc_creator import TSNP_CBC_IR
import common.internal_representation as internal_representation
from common.enums import DDREntryType, DebugFilesFormat, SequencerOPCode
from common.sequencer_ir import SequencerCustomOperationEntry, CustomeOpCode
from common.hw_config import *
from cbc_creator import export_cbc_to_xls_alex, generate_grids_wloc_cbc, generate_oc_ch_compute_sequence
from sequencer import generate_layer_command_sequence, generate_initial_command_sequence
from compiler.frontend_hardware_dependent import split_large_conv_nodes, force_split_oversized_wloc_adds, add_xfoldingConv_before_node, scatter_concat_inputs, AMM_CHANNELS_PER_BLOCK, calc_conv_folding_factor, calc_conv_qparams, get_num_y_tiles
from common.ddr_ir import TensorDDREntry, create_tsnp_tensor_byte_array
from common.debug_flags import *
import common.debug_flags as debug_flags
from common.tensor_ir import Tensor, TensorDeAllocationList, TensorDeAllocationInfo, InputTensorInfo
from common.utils import list_of_lists_split_at_pos, get_y_tile_sizes, quantize2MathBlock, get_xslices_per_htile, greedy_packed_xslice_groups, is_packed_xslices, packed_num_lanes, packed_lane_info, pad_k3_folding_output_xslices_per_htile
from common.amm_ir import AMMTensor
from common.program_ir import CBC_IR, NonLinearFunctionList, LinearFunctionList
from common.hw_config import OpType
import reports

def _mark_asym_maxpool_extra_lines(ir):
    # A blob that starts with an asymmetric-padding MAXPOOL, folding factors x and y of 1 (fold of 2), and
    # input height < 32 needs 2 extra lines: the previous blob's DMW writes them and this blob's DMR reads them.
    for name, node in ir.graph.nodes.items():
        if ir.amm_height < 16:
            continue
        if node['op_type'] != OpType.MAXPOOL or not node['frontend'].get('asym_padding', False):
            continue
        fe = node['frontend']
        if fe.get('input_folding_factor_x') != 1 or fe.get('input_folding_factor_y') != 1:
            continue
        in_t = fe['input_tensor']
        if in_t.shape[2] >= 32:
            continue
        blob_idx = fe.get('tiling_blob_idx')
        if blob_idx is None or ir.tiling_blobs[blob_idx].nodes_in_blob[0] != name:
            continue
        prod = in_t.producer
        if not prod or prod not in ir.graph.nodes:
            continue
        ir.graph.nodes[prod]['frontend']['asym_extra_lines_write'] = 2
        fe['asym_extra_lines_read'] = 2

def calculate_tensor_allocation_shape(tensor, extra_lines=0):
    """
    Calculate the actual shape needed for DDR allocation, matching sequencer write logic.
    This accounts for folding and padding that sequencer applies during write operations.

    `extra_lines` reserves additional lines beyond the tensor's nominal shape - e.g. the
    k3_1T boundary trick's producer-side extra padding line (sequencer.py
    add_tensor_write_to_ddr_commands, output_y += 1 when 'k3_1T_boundary_extra_line_write' is
    set) is a real DMA write past the tensor's nominal height, so its DDR allocation must
    reserve that line too or the next entry placed by DDR.add_entry can start where that
    extra line lands and get silently overwritten.

    Returns: shape as [batch, Z, Y, X] where Z and X include folding/padding
    """
    actual_shape = tensor.shape
    folding_factor_x = tensor.folding_factor_x

    # Calculate output_z same as sequencer.py line 329
    output_z = actual_shape[1] * (2 ** folding_factor_x)

    # Calculate output_x same as sequencer.py line 347 (padded to MAX_GRID_WIDTH)
    output_x = math.ceil(actual_shape[3] / (MAX_GRID_WIDTH * (2 ** folding_factor_x))) * MAX_GRID_WIDTH

    # output_y same as actual shape, plus any reserved extra lines
    output_y = actual_shape[2] + extra_lines

    # Return shape matching sequencer dimensions: [batch, Z, Y, X]
    return [actual_shape[0], output_z, output_y, output_x]

def _tensor_producer_wants_k3_1T_boundary_extra_line(ir, tensor):
    producer_name = getattr(tensor, 'producer', None)
    if producer_name in ir.graph.nodes:
        if ir.graph.nodes[producer_name]['frontend'].get('asym_extra_lines_write', 0):
            return ir.graph.nodes[producer_name]['frontend']['asym_extra_lines_write']
        return 1 if ir.graph.nodes[producer_name]['frontend'].get('k3_1T_boundary_extra_line_write') or ir.graph.nodes[producer_name]['frontend'].get('is_avgPool') or ir.graph.nodes[producer_name]['frontend'].get('concat_all_inputs_extra_line') or ir.graph.nodes[producer_name]['frontend'].get('odd_height_y_fold_write') else 0
    return 0

# Hardware constants
OUTPUT_PADDING_AVG_POOL = 14
OUTPUT_PADDING_MXP = 15
OUTPUT_PADDING_DEFAULT = 15
QUANTIZED_MAX_VALUE = 127
IDENTITY_WEIGHT_VALUE = 8
MAX_K3_NODES_SINGLE_TILE = 10
AMM_HEIGHT_DEFAULT = 16

def _split_weight_tensor(weight_tensor, split_index):
    weight_tensor.scale = np.split(weight_tensor.scale, 2)[split_index]
    weight_tensor.zero_point = np.split(weight_tensor.zero_point, 2)[split_index]
    weight_tensor.data = np.split(weight_tensor.data, 2)[split_index]
    weight_tensor.shape[1] = weight_tensor.shape[1] // 2

# Shared by deepcopy of constant-weight tensors during splitting and by _snapshot(). Constant
# .data/.scale/.zero_point arrays are never mutated in place (splits always reassign to a new
# np.split() result), so it's safe for deepcopy to hand back the original array by reference.
def _add_tensor_to_memo(memo, tensor):
    if getattr(tensor, 'is_constant', False):
        for arr in (tensor.data, getattr(tensor, 'scale', None), getattr(tensor, 'zero_point', None)):
            if isinstance(arr, np.ndarray):
                memo[id(arr)] = arr

def _split_and_update_node_weights(node, weight_attr, split_index):
    memo = {}
    _add_tensor_to_memo(memo, node['frontend'][weight_attr])
    updated_weights = copy.deepcopy(node['frontend'][weight_attr], memo=memo)
    node['frontend'][f'original_{weight_attr}'] = node['frontend'][weight_attr]
    node['frontend'][weight_attr] = updated_weights
    _split_weight_tensor(updated_weights, split_index)
    return updated_weights

def add_deallocation_tensor(node, dealloc_type, tensor_to_deallocate):
    key = f'tensors_for_deallocation_{dealloc_type}'
    backend = node['backend']
    if key not in backend:
        backend[key] = TensorDeAllocationList()
    backend[key].add_tensor(tensor_to_deallocate)

def mark_deallocation_for_report(node, dealloc_type, amms, current_tile_num, current_htile_num):
    key = f'tensors_for_deallocation_{dealloc_type}'
    report_key = f'{key}_report'
    if key in node['backend']:
        deallocated_tensors = node['backend'][key]
        node['backend'][report_key] = deallocated_tensors
        amms.deallocate_amm_tensors(node, deallocated_tensors, force_free_mem=False, current_tile_num=current_tile_num, current_htile_num=current_htile_num)

def get_node_input_tensors(node):
    """Get input tensors for a node, handling both single and multiple input cases."""
    frontend = node['frontend']
    if node['op_type'] in MULTIPLE_INPUT_OPS:
        input_tensors = frontend['input_tensors']
    else:
        input_tensors = [frontend['input_tensor']]
    return input_tensors

def is_resize_y_node(node):
    """Check if node is a Resize operation on the Y dimension."""
    return node['op_type'] == OpType.RESIZE and '_y' in node['name']

def is_resize_x_node(node):
    """Check if node is a Resize operation on the X dimension (RESIZE1_2 width doubling)."""
    return node['op_type'] == OpType.RESIZE and '_x' in node['name']

def get_folding_flags(node):
    """Get (force_folding_y, force_unfolding_y) flags from node frontend."""
    frontend = node['frontend']
    return 'force_folding_y' in frontend, 'force_unfolding_y' in frontend

def get_blob_kernel_size(blob):
    # All nodes in a blob containing any k3 conv must size xslices with k3 halo, so xslice geometry stays
    # consistent across producer/consumer boundaries inside the blob.
    return 3 if blob.k3_nodes > 0 else 1

def is_x_fold_node(node):
    # Structural flag set at creation in add_xfoldingConv_before_node, rather than
    # matching on the '_fold_x' name substring.
    return node['frontend'].get('is_x_fold_node', False)

def blob_reverses_xslices(blob, graph_nodes):
    # True when a blob OUTPUT node has force_folding_x with an odd x-slice count in a no-k3 blob: the
    # fold's leftover slice goes in the FIRST h-tile. Only a bare unpaired "1" needs reversal; a
    # genuine multi-slice odd count folds correctly in-place (reversing it regressed CONV2D_1_fold_x_1).
    if blob.k3_nodes != 0:
        return False
    blob_node_set = set(blob.nodes_in_blob)
    h_tiles = blob.h_tiles
    for node_name in blob.nodes_in_blob:
        frontend = graph_nodes[node_name]['frontend']
        x_slices = frontend.get('x_slices', 0)
        if ('force_folding_x' not in frontend) or (x_slices % 2 == 0):
            continue
        # Only reverse when the NATURAL split would strand a bare, unpaired "1" in a non-last h-tile
        # (x_slices < 2 * h_tiles); larger odd totals have room for a genuine multi-slice fold and must
        # stay untouched (falls through to the natural ceil split via get_xslices_per_htile).
        if h_tiles <= 1 or x_slices >= 2 * h_tiles:
            continue
        consumers = frontend['output_tensor'].consumers
        if (not consumers) or any(c not in blob_node_set for c in consumers):
            return True
    return False

def node_avoids_ddr_write(node):
    # True if DDR writes for this node must NOT overlap conv: the global debug flag, OR the per-node
    # marker set by the single-node-blob pre-split fallback. Defaults to the global behavior when the
    # per-node flag is absent, so non-flagged nodes are unchanged.
    return DEBUG_AVOID_DDR_WRITE_WHILE_CONV or node['frontend'].get('avoid_ddr_write_while_conv', False)

def get_padded_input_xslices(node, x_slices, h_tiles, no_k3_nodes_in_blob=False):
    # An x-fold's per-h-tile split is now PRESERVED as the natural ceil-based distribution, with the fold
    # computed per h-tile (sequencer's multi_htile_first path). A non-last odd h-tile no longer needs a
    # dummy pad; padding here (+1) would push e.g. 21 -> 22 -> [11,11] and desync that per-h-tile fold.
    return x_slices

def set_producer_output_allocation(producer_node, tile_idx, htile_idx, xslice_idx, allocated_blocks):
    """Set output allocation for a producer node and track that it was allocated."""
    backend = producer_node['backend']
    backend['allocated_amm_blocks_for_output_even_grid'][tile_idx][htile_idx][xslice_idx] = allocated_blocks
    backend['allocated_amm_blocks_for_output_odd_grid'][tile_idx][htile_idx][xslice_idx] = allocated_blocks
    if 'output_amm_allocated' in backend:
        backend['output_amm_allocated'].append((tile_idx, htile_idx, xslice_idx))
    else:
        backend['output_amm_allocated'] = [(tile_idx, htile_idx, xslice_idx)]

def set_consumer_input_allocation(consumer_node, tile_idx, htile_idx, xslice_idx, input_index, allocated_blocks):
    """Set input allocation for a consumer node at the specified input index."""
    backend = consumer_node['backend']
    backend['allocated_amm_blocks_for_input_even_grid'][tile_idx][htile_idx][xslice_idx][input_index] = allocated_blocks
    backend['allocated_amm_blocks_for_input_odd_grid'][tile_idx][htile_idx][xslice_idx][input_index] = allocated_blocks

def _wire_blocks_to_blob(ir, graph_nodes, blob_nodes_set, current_input_tensor,
                         tile_idx, htile_idx, xslice_idx, slice_blocks):
    """Mark slice_blocks on the producer (if in blob) and on every consumer in the blob.
    Used by the four allocate paths that update producer/consumer allocations after
    contiguous/non-contiguous mem allocation."""
    producer = current_input_tensor.producer
    if producer in blob_nodes_set:
        # An x-fold consumer pads its input x-slices with one extra "dummy" slice with no matching
        # producer output slice. Only wire the producer output when this x-slice is within its real
        # output slices; the consumer input grid (padded) is still wired below.
        producer_output_grid = graph_nodes[producer]['backend']['allocated_amm_blocks_for_output_even_grid'][tile_idx][htile_idx]
        if xslice_idx < len(producer_output_grid):
            set_producer_output_allocation(graph_nodes[producer], tile_idx, htile_idx, xslice_idx, slice_blocks)
    for consumer_name in current_input_tensor.consumers:
        if consumer_name in blob_nodes_set:
            consumer_node = graph_nodes[consumer_name]
            _, consumer_input_index = ir.get_consumer_input_index(consumer_node, current_input_tensor)
            # Consumers may have fewer x-slices for this htile than the producer (e.g. a kernel-1
            # requant Identity consuming a k3 producer's extra halo slice); only wire x-slices that
            # exist in the consumer's input grid, mirroring the producer-side guard above.
            cgrid = consumer_node['backend']['allocated_amm_blocks_for_input_even_grid']
            if (tile_idx < len(cgrid) and htile_idx < len(cgrid[tile_idx])
                    and xslice_idx < len(cgrid[tile_idx][htile_idx])):
                set_consumer_input_allocation(consumer_node, tile_idx, htile_idx, xslice_idx, consumer_input_index, slice_blocks)


def get_node_tiling_info(ir,node):
    output_padding_start_y = ir.amm_height
    node_folded_output_shape = node['frontend']['output_tensor'].get_folded_shape()
    one_line_of_padding = 1 if (node['op_type'] == OpType.CONV and node['frontend'].get('is_avgPool', False)) else 0
    # A resize node's height in the AMM is its INPUT height - the y-pass's height-doubling happens via
    # a DMW on the way to DDR, not in the AMM. The x/y node pair share input height, keeping tile counts
    # in agreement per set_node_tiling_info's y_tiles consistency check.
    if is_resize_x_node(node) or is_resize_y_node(node):
        current_op_height = (node['frontend']['input_tensor'].get_folded_shape())[2]
    else:
        current_op_height = node_folded_output_shape[2]
    current_blob_idx = node['frontend']['tiling_blob_idx']
    current_blob = ir.tiling_blobs[current_blob_idx]
    k3_nodes_in_blob = current_blob.k3_nodes
    # Too many k3 nodes for this amm_height makes amm_height-2*k3_nodes_in_blob non-positive,
    # previously an infinite append loop (MemoryError). Stash the mismatch so compile()'s escalation
    # loop re-blobs/splits next pass; k3_nodes_in_blob=0 is a safe placeholder for this pass only.
    if current_op_height > ir.amm_height and (ir.amm_height - 2*k3_nodes_in_blob) <= 0:
        ir.xslices_mismatch_blob_idx = current_blob_idx
        ir.xslices_mismatch_split_node = node['name']
        k3_nodes_in_blob = 0
    tile_sizes,per_tile_read_start_line,per_tile_write_start_line=get_y_tile_sizes(current_op_height,k3_nodes_in_blob=k3_nodes_in_blob,amm_height=ir.amm_height,extra_lines=node['frontend'].get('asym_extra_lines_write', 0) if node['frontend'].get('asym_extra_lines_write', 0) > 0 else 0)
    num_tiles = len(tile_sizes)
    output_tensor_y_size = ir.tensors[node['outputs'][0]].shape[2]
    if output_tensor_y_size<ir.amm_height:
        output_padding_start_y = output_tensor_y_size
    elif one_line_of_padding:
        output_padding_start_y = ir.amm_height - 1
    else:
        output_padding_start_y = ir.amm_height
    # Asymmetric-padding MAXPOOL trigger: its 2 padded lines occupy the top of the last y-tile's output padding.
    output_padding_start_y -= node['frontend'].get('asym_extra_lines_write', 0)

    if one_line_of_padding:
        node['frontend']['pad_extra_line'] = True

    return num_tiles, output_padding_start_y

def get_output_padding_start_x(node, current_op_height, current_op_output_width, y_tiles=1, amm_height=AMM_HEIGHT_DEFAULT, sync_with_MXP=False, tensors_from_mxp=None):
    tensor_from_mxp = False
    input_tensor_name = node['inputs'][0]
    if sync_with_MXP and input_tensor_name in tensors_from_mxp:
        tensor_from_mxp = True

    if (('is_avgPool' in node['frontend']) and node['frontend']['is_avgPool']):
        output_padding_start_x = OUTPUT_PADDING_AVG_POOL
    else:
        output_padding_start_x = OUTPUT_PADDING_DEFAULT - ((2*AMM_WIDTH) - (current_op_output_width%(2*AMM_WIDTH))) % (2*AMM_WIDTH)
    if y_tiles == 1 and current_op_height > amm_height:
        raise ValueError('Folded tensor height cannot exceed AMM_HEIGHT')
    return output_padding_start_x

def set_concat_op_grid_config(ir, node):
    frontend = node['frontend']
    force_folding_y, force_unfolding_y = get_folding_flags(node)
    tensors = ir.tensors
    graph_nodes = ir.graph.nodes

    input_channels_per_input = []
    for input_tensor_name in node['inputs']:
        current_input_tensor = tensors[input_tensor_name]
        current_input_tensor_shape = current_input_tensor.get_folded_shape(
            folding_conv_y=force_folding_y, unfolding_conv_y=force_unfolding_y)
        input_channels_per_input.append(current_input_tensor_shape[1])

    current_op_input0_tensor_shape = tensors[node['inputs'][0]].get_folded_shape()
    current_op_input1_tensor_shape = tensors[node['inputs'][1]].get_folded_shape()
    current_op_output_tensor_shape = tensors[node['outputs'][0]].get_folded_shape()

    if len(current_op_input0_tensor_shape) != 4:
        raise ValueError('Encountered activation with dims!=4: %s' % current_op_input0_tensor_shape)
    if len(current_op_input1_tensor_shape) != 4:
        raise ValueError('Encountered activation with dims!=4: %s' % current_op_input1_tensor_shape)
    if len(current_op_output_tensor_shape) != 4:
        raise ValueError('Encountered activation with dims!=4: %s' % node['outputs'][0])

    backend = node['backend']
    backend['output_channels'] = current_op_output_tensor_shape[1]
    backend['input_channels'] = input_channels_per_input
    frontend['x_wrapping'] = 1

def get_ops_grid_config(ir: internal_representation.IR) -> internal_representation.IR:
    graph_nodes = ir.graph.nodes()
    tensors = ir.tensors
    amm_height = ir.amm_height
    sync_with_MXP = ir.sync_with_MXP
    tensors_from_mxp = ir.tensors_from_mxp

    tqdm_iterator = tqdm(ir.lexicographical_topological_sorted_graph)
    for node_name in tqdm_iterator:
        tqdm_iterator.set_description('Calculating grid config, at layer %s:' % node_name)
        node = graph_nodes[node_name]
        current_op_type = node['op_type']

        if current_op_type == OpType.CONCAT:
            node['backend'] = OrderedDict()
            set_concat_op_grid_config(ir, node)
        elif current_op_type == OpType.SYNC:
            # Pure DMA hop - no grid/kernel config to compute, but downstream passes
            # (set_node_tiling_info, cbc_generation_pass) still index into node['backend'].
            node['backend'] = OrderedDict()
            current_op_input_tensor_shape = tensors[node['inputs'][0]].get_folded_shape()
            current_op_output_tensor_shape = tensors[node['outputs'][0]].get_folded_shape()
            node['backend']['output_channels'] = current_op_output_tensor_shape[1]
            node['backend']['input_channels'] = current_op_input_tensor_shape[1]
        elif current_op_type in GRID_OPS:
            frontend = node['frontend']
            backend = OrderedDict()
            node['backend'] = backend

            current_op_input_tensor_name = node['inputs'][0]
            force_folding_y, force_unfolding_y = get_folding_flags(node)
            # `padded` mirrors the first consumer conv's input padding; default to 0 when the consumer
            # has no conv attributes (e.g. Sync nodes) or there are no consumers, to avoid crashing.
            consumers = tensors[current_op_input_tensor_name].consumers
            consumer_0_attrs = graph_nodes[consumers[0]].get('attributes', {}) if consumers else {}
            padded = consumer_0_attrs['pads'][0] if 'pads' in consumer_0_attrs else 0

            current_op_input_tensor_shape = tensors[current_op_input_tensor_name].get_folded_shape(
                folding_conv_y=force_folding_y, unfolding_conv_y=force_unfolding_y, padded=padded)
            current_op_output_tensor_name = node['outputs'][0]
            folding_conv_x = 'force_folding_x' in frontend
            current_op_output_tensor_shape = tensors[current_op_output_tensor_name].get_folded_shape(
                folding_conv_x=folding_conv_x, producing_node_stride=frontend['stride'])

            if len(current_op_input_tensor_shape) != 4:
                raise ValueError('Encountered activation with dims!=4: %s' % current_op_input_tensor_name)
            if len(current_op_output_tensor_shape) != 4:
                raise ValueError('Encountered activation with dims!=4: %s' % current_op_output_tensor_name)

            current_op_input_channels = current_op_input_tensor_shape[1]
            if current_op_type in MULTIPLE_INPUT_OPS:
                current_op_input_channels *= 2

            current_input_shape = tensors[current_op_input_tensor_name].get_original_shape()
            input_folding_factor_x = frontend['input_folding_factor_x']
            input_width = current_op_input_tensor_shape[3]
            if (input_width > current_input_shape[3] / (2 ** input_folding_factor_x)):
                input_width = int(current_input_shape[3] / (2 ** input_folding_factor_x))

            backend['input_channels'] = current_op_input_channels
            backend['output_channels'] = current_op_output_tensor_shape[1]
            backend['z_tiles'] = 1
            # The y-tile count must reflect this node's folded output height, not the frontend value taken from its
            # input (an Add reads its input's tile count, which can be 1 for an 18-row output). Use the same helper
            # get_node_tiling_info uses, so the check and the later tiling agree.
            grid_op_height = node_folded_output_shape_h = current_op_output_tensor_shape[2]
            grid_y_tiles = len(get_y_tile_sizes(grid_op_height, k3_nodes_in_blob=0, amm_height=amm_height)[0])
            backend['output_padding_start_x'] = get_output_padding_start_x(
                node, current_op_input_tensor_shape[2], input_width,
                y_tiles=grid_y_tiles, amm_height=amm_height,
                sync_with_MXP=sync_with_MXP, tensors_from_mxp=tensors_from_mxp)
            frontend['x_wrapping'] = 1

    return ir







# Insert an identity conv before a node (similar to insert_ordering_conv_after_node), splicing in a
# new tensor between the identity and node.
def insert_identity_conv_before_node(ir:internal_representation.IR,node,new_name):
    node_name = node['name']

    # Create the new identity node
    identity_node_name = new_name
    identity_node = {}
    identity_node['name'] = identity_node_name
    identity_node['attributes']={}
    identity_node['op_type'] = OpType.CONV
    identity_node['attributes']['kernel_shape'] = [1,1]
    identity_node['attributes']['pads'] = [0,0,0,0]

    # calculate folding factor 
    original_input_tensor_name = node['inputs'][0]
    original_input_tensor = node['frontend']['input_tensor']
    if ('_fold_x' in new_name):
        folding_factor_x = int(new_name.split('_')[-1])
        identity_node_input_folding_x = folding_factor_x
        identity_node_output_folding_x = folding_factor_x + 1
        original_input_tensor.folding_factor_x = identity_node_input_folding_x

        # calc the new x_slices for input
        identity_node_input_xslices = int((original_input_tensor.shape[3]+(MAX_GRID_WIDTH-1))//MAX_GRID_WIDTH)
        identity_node_input_xslices = math.ceil(identity_node_input_xslices/(2**identity_node_input_folding_x))
        original_input_tensor.x_slices = identity_node_input_xslices

        # calc the new x_slices for output        
        identity_node_output_xslices = math.ceil(identity_node_input_xslices/2)
    else:
        folding_factor_x = original_input_tensor.folding_factor_x
        identity_node_input_folding_x = folding_factor_x
        identity_node_output_folding_x = folding_factor_x
        identity_node_input_xslices = node['frontend']['input_tensor'].x_slices
        identity_node_output_xslices = node['frontend']['x_slices']
        
    # Create a new tensor for the output of this node
    post_identity_tensor_name = new_name + '_output'
    assert post_identity_tensor_name not in ir.tensors.keys()
    post_identity_tensor = Tensor(post_identity_tensor_name,None,producer=identity_node_name,consumers=[node_name],
                                 is_constant=False,shape = original_input_tensor.shape,scale = original_input_tensor.scale,
                                 zero_point=original_input_tensor.zero_point,folding_factor_x = identity_node_output_folding_x,
                                 folding_factor_y=original_input_tensor.folding_factor_y, x_slices=identity_node_output_xslices)
    ir.tensors[post_identity_tensor_name] = post_identity_tensor
    identity_node['outputs'] = [post_identity_tensor_name]


    # Update the inputs of the new identity node
    identity_conv_weights_tensor_name = new_name+'_conv_w'
    identity_conv_biases_tensor_name = new_name+'_conv_b'
    identity_node['inputs'] = [original_input_tensor_name,identity_conv_weights_tensor_name,identity_conv_biases_tensor_name]

    # Set the frontend and backend properties of this new identity node
    # These are set like in insert_ordering_conv_after_node (ic/oc splits, ic/oc groups, etc.)

    original_node_input_shape = original_input_tensor.get_folded_shape()
    identity_node_input_channels = original_node_input_shape[1]
    identity_node_output_channels = identity_node_input_channels

    identity_node['backend'] = {}
    identity_node['backend']['input_channels'] = identity_node_input_channels
    identity_node['backend']['output_channels'] = identity_node_output_channels

    
    identity_node['frontend'] = {}
    identity_node['frontend']['following_nodes_params'] = []
    identity_node['frontend']['is_x_fold_node'] = ('_fold_x' in new_name)

    y_tiles = node['frontend']['y_tiles']
    tile_sizes,per_tile_read_start_line,per_tile_write_start_line=get_y_tile_sizes(original_input_tensor.get_folded_shape()[2],k3_nodes_in_blob=0,amm_height=ir.amm_height)
    y_tiles = len(tile_sizes)
    identity_node['frontend']['y_tiles'] = y_tiles
    output_padding_start_x = get_output_padding_start_x(identity_node,original_node_input_shape[2],original_node_input_shape[3],y_tiles=y_tiles, amm_height=ir.amm_height,
                                                        sync_with_MXP=ir.sync_with_MXP, tensors_from_mxp=ir.tensors_from_mxp)
    identity_node['backend']['output_padding_start_x'] = output_padding_start_x

    identity_node['backend']['z_tiles'] = 1
    identity_node['frontend']['h_tiles'] = node['frontend']['h_tiles']
    input_xslices = identity_node_input_xslices
    if (input_xslices % 2 != 0) and identity_node['frontend']['h_tiles'] > 1:
        input_xslices += 1
    identity_node['frontend']['xslices_per_htile'] = get_xslices_per_htile(input_xslices, identity_node['frontend']['h_tiles'], 3)
    identity_node['backend']['ic_groups'] = [[ic for ic in range(identity_node_input_channels)]]
    identity_node['backend']['ic_lookup_dicts'] = [[ic for ic in range(identity_node_input_channels)]]

    identity_node['backend']['oc_groups'] = [list(range(identity_node_output_channels))]

    identity_node['backend']['deepconv'] = True
    identity_node['frontend']['input_tensor'] = original_input_tensor
    identity_node['frontend']['output_tensor'] = post_identity_tensor
    identity_node['frontend']['input_tensor_zp'] = node['frontend']['input_tensor_zp']
    identity_node['frontend']['output_tensor_zp'] = node['frontend']['input_tensor_zp']
    identity_node['frontend']['input_tensor_scale'] = node['frontend']['input_tensor_scale']
    identity_node['frontend']['output_tensor_scale'] = node['frontend']['input_tensor_scale']
    identity_node['frontend']['kernel_size'] = 1
    identity_node['frontend']['stride'] = 1
    original_folded_input_tensor_shape = original_input_tensor.get_folded_shape()
    identity_node['frontend']['sparse_macs'] = identity_node_input_channels*original_folded_input_tensor_shape[2]*original_folded_input_tensor_shape[3]
    identity_node['frontend']['input_folding_factor_x'] = identity_node_input_folding_x
    identity_node['frontend']['input_folding_factor_y'] = node['frontend']['input_folding_factor_y']
    identity_node['frontend']['input_channels'] = identity_node_input_channels
    identity_node['frontend']['output_channels'] = identity_node_output_channels
    identity_node['frontend']['input_tensor'].folding_factor_x = identity_node_input_folding_x
    identity_node['frontend']['input_tensor'].folding_factor_y = node['frontend']['input_tensor'].folding_factor_y
    identity_node['frontend']['input_tensor'].x_slices = identity_node_input_xslices
    identity_node['frontend']['output_folding_factor_x'] = identity_node_output_folding_x
    identity_node['frontend']['output_folding_factor_y'] = node['frontend']['input_folding_factor_y']
    identity_node['frontend']['output_tensor'].folding_factor_x = identity_node_output_folding_x
    identity_node['frontend']['output_tensor'].folding_factor_y = node['frontend']['input_tensor'].folding_factor_y
    identity_node['frontend']['output_tensor'].x_slices = identity_node_output_xslices
    identity_node['frontend']['x_slices'] = identity_node_input_xslices
    identity_node['frontend']['x_wrapping'] = node['frontend']['x_wrapping']
    identity_node['frontend']['input_tensor'].xslices_per_htile = identity_node['frontend']['xslices_per_htile']
    identity_node['frontend']['output_tensor'].xslices_per_htile = get_xslices_per_htile(identity_node_output_xslices, identity_node['frontend']['h_tiles'], node['frontend']['kernel_size'])

    if (len(node['frontend']['preceding_nodes_params']) == 0) and ('_fold_x_0' in new_name) and (identity_node_input_channels < 16) and (original_input_tensor.shape[3] % 16 == 0):
        identity_node['frontend']['input_tensor'].num_packed_xslices = greedy_packed_xslice_groups(identity_node['frontend']['xslices_per_htile'][0])
            
    weight_channels = identity_node_input_channels
    identity_weights_shape = [weight_channels,weight_channels,1,1]
    identity_biases_shape = [identity_node_input_channels]

    conv_W = np.zeros(identity_weights_shape, dtype=np.int8)
    conv_B = np.zeros(identity_biases_shape, dtype=np.int32)
    conv_W_scale = np.zeros(identity_node_input_channels, dtype=np.float32)
    for current_cout in range(weight_channels):
        conv_W[current_cout, current_cout, 0, 0] = IDENTITY_WEIGHT_VALUE
        conv_W_scale[current_cout] = 1.0 / IDENTITY_WEIGHT_VALUE
    # Create the new weights and bias tensor
    weights_tensor = Tensor(identity_conv_weights_tensor_name,conv_W,is_constant=True, shape = identity_weights_shape, consumers=[identity_node_name],scale=conv_W_scale)
    biases_tensor = Tensor(identity_conv_biases_tensor_name,conv_B,is_constant=True, shape = identity_biases_shape, consumers=[identity_node_name])

    ir.tensors[identity_conv_weights_tensor_name] = weights_tensor
    ir.tensors[identity_conv_biases_tensor_name] = biases_tensor

    # Set quantization parameters
    if (identity_node['frontend']['input_folding_factor_x'] > 0 or identity_node['frontend']['input_folding_factor_y'] > 0):
        identity_node['frontend']['folded_weights_tensor'] = weights_tensor
        identity_node['frontend']['folded_biases_tensor'] = biases_tensor
        identity_node['frontend']['folded_kernel_size'] = 1
        identity_node['frontend']['folded_padding'] = identity_node['attributes']['pads']
    else:
        identity_node['frontend']['weights_tensor'] = weights_tensor
        identity_node['frontend']['biases_tensor'] = biases_tensor
        identity_node['frontend']['padding'] = identity_node['attributes']['pads']

    bias_data = int(-8*original_input_tensor.zero_point)
    scale = 1/8.0
    output_offset = original_input_tensor.zero_point.item()
    output_activation_min = -128
    output_activation_max = 127
    output_multiplier, cInputH, cInputL, o_shift = \
            quantize2MathBlock(None,bias_data,scale,output_offset,output_activation_min,output_activation_max)
    identity_node['frontend']['output_multiplier'] = [output_multiplier] * identity_node_output_channels
    identity_node['frontend']['cInputH'] = [cInputH] * identity_node_output_channels
    identity_node['frontend']['cInputL'] = [cInputL] * identity_node_output_channels
    identity_node['frontend']['o_shift'] = [o_shift] * identity_node_output_channels

    ir.graph.add_nodes_from([(identity_node_name, identity_node)])

    # Interpose the identity ONLY for consumers reading original_input_tensor DIRECTLY. A slice-reader
    # keeps its own producer wiring untouched, and only same-blob readers are rerouted - a
    # cross-blob reroute could create a dependency that isn't allocated yet (ddr_entry None).
    node_blob_idx = node['frontend'].get('tiling_blob_idx')
    full_reader_names = [c for c in original_input_tensor.consumers
                         if original_input_tensor_name in ir.graph.nodes[c]['inputs']
                         and (node_blob_idx is None or ir.graph.nodes[c]['frontend'].get('tiling_blob_idx') == node_blob_idx)]
    slice_reader_names = [c for c in original_input_tensor.consumers if c not in full_reader_names]

    input_producer_name = original_input_tensor.producer
    for i_name in full_reader_names:
        # Reroute ONLY the edge carrying original_input_tensor, not every incoming edge - a full
        # reader may be a multi-input op reading other tensors too (e.g. residual Add).
        if input_producer_name and ir.graph.has_edge(input_producer_name, i_name):
            ir.graph.remove_edge(input_producer_name, i_name)
        # If output-channel-split, each chunk is a separate node with its own edge to i_name;
        # input_producer_name only points at one chunk, so remove every split chunk's edge too.
        if original_input_tensor.name in ir.original_tensor_to_split_tensor_map:
            for split_tensor_name in ir.original_tensor_to_split_tensor_map[original_input_tensor.name]:
                split_producer_name = ir.tensors[split_tensor_name].producer
                if split_producer_name and ir.graph.has_edge(split_producer_name, i_name):
                    ir.graph.remove_edge(split_producer_name, i_name)
        ir.graph.add_edge(identity_node_name, i_name)
        if i_name not in post_identity_tensor.consumers:
            post_identity_tensor.consumers.append(i_name)
        next_node_info = ir.graph.nodes[i_name]
        ir.switch_input_name(next_node_info,original_input_name=original_input_tensor_name,new_input_name=post_identity_tensor_name)
        ir.switch_input_tensor(next_node_info,original_input_tensor=original_input_tensor,new_input_tensor=post_identity_tensor)
        if original_input_tensor.name in ir.original_tensor_to_split_tensor_map:
            split_tensor_names = ir.original_tensor_to_split_tensor_map[original_input_tensor.name]
            for split_tensor_name in split_tensor_names:
                if node_name in ir.tensors[split_tensor_name].consumers:
                    ir.switch_tensor_consumer(ir.tensors[split_tensor_name],original_node_name=node_name,new_node_name=new_name)

    # original_input_tensor is now read by the identity (the full read) plus any untouched slice-readers.
    original_input_tensor.consumers = [identity_node_name] + slice_reader_names

    # If this node's input is a channel slice, the parent tensor still lists this node as a consumer
    # even though it now reads the identity's output; repoint it so mark_blob_outputs_deallocations
    # still finds the slice (no-op for ordinary, non-sliced identities).
    if original_input_tensor_name in ir.split_tensor_to_original_tensor_map:
        parent_tensor_name = ir.split_tensor_to_original_tensor_map[original_input_tensor_name]
        parent_tensor = ir.tensors[parent_tensor_name]
        if node_name in parent_tensor.consumers:
            ir.switch_tensor_consumer(parent_tensor, original_node_name=node_name, new_node_name=identity_node_name)

    # Update following nodes params
    node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(node) # We need to update it so that following nodes order is according to execution order
    following_nodes_params = ir.get_updated_following_nodes(identity_node)
    identity_node['frontend']['following_nodes_params'] = following_nodes_params # We need to update it so that following nodes order is according to execution order
    for following_node_params in following_nodes_params:
        following_node_name = following_node_params[0]
        following_node = ir.graph.nodes()[following_node_name]
        following_node['frontend']['preceding_nodes_params'] = ir.get_updated_preceding_nodes(following_node)

    identity_node['frontend']['preceding_nodes_params'] = ir.get_updated_preceding_nodes(identity_node)

    if 'tiling_blob_idx' in node['frontend']:
        # Put the identity node in the same blob as the input node
        identity_node_blob_idx = node['frontend']['tiling_blob_idx']
        identity_node['frontend']['tiling_blob_idx'] = identity_node_blob_idx
        # Update this blob to add the identity node
        current_blob = ir.tiling_blobs[identity_node_blob_idx]
        nodes_in_blob = current_blob.nodes_in_blob
        node_idx = nodes_in_blob.index(node_name)
        # Insert the identity node before the given node
        nodes_in_blob.insert(node_idx, identity_node_name)
        current_blob.num_of_nodes_in_blob += 1
        update_node_tiling_info(ir,identity_node)

    # Inherit single-node-blob pre-split markers: the identity is inserted FIRST in the blob (above),
    # becoming the blob's first node, and evict_amm_except_system_before_alloc is only honored on the
    # first node - without this the split node's eviction/DDR discipline would be silently dropped.
    for marker in ('avoid_ddr_read_while_conv', 'avoid_ddr_write_while_conv', 'evict_amm_except_system_before_alloc'):
        if node['frontend'].get(marker, False):
            identity_node['frontend'][marker] = True

    # The identity must precede EVERY full reader, not just node_name - a fanned-out sibling could
    # sit earlier in execution order and run before the identity produces its output (no re-sort
    # happens after this point), so insert before the EARLIEST index among all full readers.
    reader_indices = [ir.lexicographical_topological_sorted_graph.index(r) for r in full_reader_names]
    pre_identity_node_execution_index = min(reader_indices)
    ir.lexicographical_topological_sorted_graph.insert(pre_identity_node_execution_index,identity_node_name)
    return identity_node_name

# Add identity convolution before k3 nodes at the start of 1-tile blobs
# Concat feeding a k3 conv in the same blob: the k3 conv reads the concat's DDR output, so the concat carries the
# read-side extra line and each of its input producers writes that extra line.
def add_k3_extra_line_for_concat_inputs(ir: internal_representation.IR) -> internal_representation.IR:
    for node_name in list(ir.graph.nodes):
        node = ir.graph.nodes[node_name]
        if node['op_type'] != OpType.CONCAT:
            continue

        if not node['frontend'].get('is_scatter_concat'):
            consumers = [ir.graph.nodes[c] for c in ir.graph.successors(node_name)]
            same_blob = node['frontend'].get('tiling_blob_idx')
            # Only a k3 consumer with a single y-tile in the same blob needs the extra line; the read flag on the concat
            # and the write flag on each of its inputs are both gated on this one condition.
            gate = any(ir.is_k3_node(c) and c['frontend'].get('y_tiles') == 1
                    and c['frontend']['input_tensor'].shape[2] < ir.amm_height
                    and c['frontend'].get('tiling_blob_idx') == same_blob for c in consumers)
            if gate:
                node['frontend']['k3_1T_boundary_extra_line_read'] = True
                for producer_name in ir.graph.predecessors(node_name):
                    producer = ir.graph.nodes[producer_name]
                    if producer['frontend'].get('k3_1T_boundary_extra_line_write'):
                        continue
                    producer['frontend']['k3_1T_boundary_extra_line_write'] = True
                    producer['frontend']['output_tensor'].shape_real_x16[2] += 1

        node['frontend']['concat_all_inputs_extra_line'] = all(
                    ir.graph.nodes[p]['frontend'].get('k3_1T_boundary_extra_line_write') for p in ir.graph.predecessors(node_name))        

    return ir

def add_identity_before_k3_starting_1T_blob(ir:internal_representation.IR) -> internal_representation.IR:
    for tiling_blob_idx,tiling_blob in ir.tiling_blobs.items():
        first_node_name = tiling_blob.nodes_in_blob[0]
        first_node = ir.graph.nodes[first_node_name]
        if not ir.node_needs_k3_1T_identity(first_node, ir.amm_height):
            continue
        if tiling_blob_idx != 0 and DEBUG_K3_1T_BOUNDARY_EXTRA_LINE:
            # The node whose DMW needs the extra line is whichever node actually writes
            # first_node's input tensor to DDR, not necessarily the previous blob's last node.
            producer_node_name = list(ir.graph.predecessors(first_node_name))[0]
            producer_node = ir.graph.nodes[producer_node_name]
            producer_node['frontend']['k3_1T_boundary_extra_line_write'] = True
            # Mirror the is_avgPool_output/is_conv2x2_output precedent: the producer's DMW writes
            # one line beyond nominal height, so shape_real_x16 must reflect that too, or the
            # un-padded shape could be picked as the (too-small) allocation size.
            producer_node['frontend']['output_tensor'].shape_real_x16[2] += 1
            # producer_node's output can fan out to multiple consumers that read it from DDR (e.g. a
            # residual Add feeding both first_node and a sibling Add) - every such consumer reads the
            # SAME padded tensor, so all of them need the read-side extra line, not just first_node.
            for consumer_name in ir.graph.successors(producer_node_name):
                consumer_node = ir.graph.nodes[consumer_name]
                consumer_node['frontend']['k3_1T_boundary_extra_line_read'] = True
                # sequencer.py applies the read-side flag per-node, to EVERY DDR input of that node,
                # not per-tensor - so any OTHER predecessor feeding this consumer via DDR must pad its
                # own write to match, or the consumer's two inputs would disagree on line count.
                for other_producer_name in ir.graph.predecessors(consumer_name):
                    if other_producer_name == producer_node_name:
                        continue
                    other_producer_node = ir.graph.nodes[other_producer_name]
                    if other_producer_node['frontend'].get('k3_1T_boundary_extra_line_write'):
                        continue
                    other_producer_node['frontend']['k3_1T_boundary_extra_line_write'] = True
                    other_producer_node['frontend']['output_tensor'].shape_real_x16[2] += 1
            continue
        identity_node_name = insert_identity_conv_before_node(ir,first_node,new_name=f'{first_node_name}_identity')
        # Only when the k3 first node came from a conv split: the identity is now the blob's first
        # node reading from DDR, so give it the split node's memory discipline (evict AMM except
        # system zero, no DDR-read-while-conv pre-read buffer). Plain (unsplit) k3 1T blobs are unaffected.
        if first_node['frontend'].get('was_split', False):
            ir.graph.nodes[identity_node_name]['frontend']['evict_amm_except_system_before_alloc'] = True
            ir.graph.nodes[identity_node_name]['frontend']['avoid_ddr_read_while_conv'] = True
            # The identity now owns the blob-entry role; leaving these markers on the split node
            # would apply the discipline twice, at the wrong node. avoid_ddr_write_while_conv
            # governs the split node's own output write and stays with it.
            first_node['frontend'].pop('evict_amm_except_system_before_alloc', None)
            first_node['frontend'].pop('avoid_ddr_read_while_conv', None)
    return ir


def add_odd_height_y_fold_extra_lines(ir: internal_representation.IR) -> internal_representation.IR:
    # A y-folded tensor with odd height: its producer's DMW writes one extra line, and each consumer's DMR reads it.
    # Same flag pattern as k3_1T_boundary_extra_line_write/read; the sequencer only reads these flags.
    for node_name in list(ir.graph.nodes):
        node = ir.graph.nodes[node_name]
        out_t = node['frontend'].get('output_tensor')
        if out_t is None or out_t.folding_factor_y <= 0 or out_t.shape[2] % 2 == 0:
            continue
        node['frontend']['odd_height_y_fold_write'] = True
        out_t.shape_real_x16[2] += 1
        for consumer_name in ir.graph.successors(node_name):
            ir.graph.nodes[consumer_name]['frontend']['odd_height_y_fold_read'] = True
    return ir

# Apply one fold step in place on a real node: goes from consuming an input at input_folding_x
# to producing an output at input_folding_x+1, like an inserted identity conv used to, but
# without creating any new node/tensor - just updates the fold-related fields in place.
def _apply_one_x_fold_step(ir, node, input_folding_x, input_xslices, is_first_step):
    frontend = node['frontend']
    input_tensor = frontend['input_tensor']
    output_tensor = frontend['output_tensor']

    output_folding_x = input_folding_x + 1
    output_xslices = math.ceil(input_xslices / 2)

    frontend['force_folding_x'] = True
    # This is a REAL conv absorbing a fold step, but is_x_fold_node marks it structurally as one
    # so chained-input-inheritance and halo-overlap gating treat it like any other fold step.
    frontend['is_x_fold_node'] = True
    frontend['input_folding_factor_x'] = input_folding_x
    input_tensor.folding_factor_x = input_folding_x
    input_tensor.x_slices = input_xslices
    frontend['output_folding_factor_x'] = output_folding_x
    output_tensor.folding_factor_x = output_folding_x
    output_tensor.x_slices = output_xslices
    frontend['x_slices'] = input_xslices

    input_xslices_htile_calc = input_xslices
    if (input_xslices_htile_calc % 2 != 0) and frontend['h_tiles'] > 1:
        input_xslices_htile_calc += 1
    frontend['xslices_per_htile'] = get_xslices_per_htile(input_xslices_htile_calc, frontend['h_tiles'], 3)
    input_tensor.xslices_per_htile = frontend['xslices_per_htile']
    output_tensor.xslices_per_htile = get_xslices_per_htile(output_xslices, frontend['h_tiles'], frontend['kernel_size'])

    if is_first_step and len(frontend['preceding_nodes_params']) == 0:
        identity_node_input_channels = input_tensor.get_folded_shape()[1]
        if identity_node_input_channels < 16 and (input_tensor.shape[3] % 16 == 0):
            input_tensor.num_packed_xslices = greedy_packed_xslice_groups(frontend['xslices_per_htile'][0])

    return output_folding_x, output_xslices

# Distribute x folding across existing real nodes at the graph entry instead of inserting
# synthetic identity conv nodes ahead of them; result matches what that chain used to produce.
def _reset_x_fold_passthrough(node, unfolded_xslices):
    # Leave a trivial (kernel=1, stride=1) chain node as a pure passthrough (fold=0) when
    # there are more trivial nodes than fold steps needed.
    frontend = node['frontend']
    input_tensor = frontend['input_tensor']
    output_tensor = frontend['output_tensor']
    if 'force_folding_x' in frontend:
        del frontend['force_folding_x']
    frontend['input_folding_factor_x'] = 0
    input_tensor.folding_factor_x = 0
    input_tensor.x_slices = unfolded_xslices
    frontend['output_folding_factor_x'] = 0
    output_tensor.folding_factor_x = 0
    output_tensor.x_slices = unfolded_xslices
    frontend['x_slices'] = unfolded_xslices
    frontend['xslices_per_htile'] = get_xslices_per_htile(unfolded_xslices, frontend['h_tiles'], 3)
    input_tensor.xslices_per_htile = frontend['xslices_per_htile']
    output_tensor.xslices_per_htile = get_xslices_per_htile(unfolded_xslices, frontend['h_tiles'], frontend['kernel_size'])

def distribute_entry_x_folding(ir):
    original_graph_node_names = list(ir.graph.nodes)
    for node_name in original_graph_node_names:
        node = ir.graph.nodes[node_name]
        # Check if the output tensor is either an output to the graph or sent to MXP
        input_folding_factor_x = node['frontend']['input_folding_factor_x']
        if (len(node['frontend']['preceding_nodes_params']) == 0) and (input_folding_factor_x > 0):
            if ('_split' in node_name):
                orig_node_name = node_name.split('_split')[0]
                orig_node = ir.graph.nodes[orig_node_name]
                in_tensor = ir.tensors[orig_node['inputs'][0]]
                in_tensor.consumers.append(node_name)
                node['frontend']['input_tensor'] = in_tensor
                orig_node['frontend']['input_tensor'] = in_tensor
                node['frontend']['preceding_nodes_params'] = orig_node['frontend']['preceding_nodes_params']
                orig_in_tensor = ir.tensors[node['inputs'][0]]
                orig_in_tensor.consumers.remove(node_name)
                node['inputs'][0] = orig_node['inputs'][0]
            else:
                total_folds = input_folding_factor_x
                input_tensor = node['frontend']['input_tensor']
                unfolded_xslices = int((input_tensor.shape[3] + (MAX_GRID_WIDTH - 1)) // MAX_GRID_WIDTH)

                # Walk the trivial (kernel=1, stride=1) chain forward, matching
                # get_next_non_folding_conv's walk to find the real fold-determining conv.
                chain = []
                current = node
                while (current['frontend']['kernel_size'] == 1) and (current['frontend']['stride'] == 1):
                    chain.append(current)
                    following_nodes_params = current['frontend']['following_nodes_params']
                    if len(following_nodes_params) != 1:
                        break
                    next_node = ir.graph.nodes[following_nodes_params[0][0]]
                    if 'kernel_size' not in next_node['frontend'] or 'stride' not in next_node['frontend']:
                        break
                    current = next_node

                deficit = total_folds - len(chain)
                if deficit > 0:
                    # Not enough trivial nodes to absorb every fold step and nothing can be
                    # inserted ahead of the true entry, so the chain absorbs the first len(chain)
                    # steps and the remaining `deficit` steps are inserted AFTER it, before `current`.
                    current_input_folding_x = 0
                    current_input_xslices = unfolded_xslices
                    for step, fold_node in enumerate(chain):
                        current_input_folding_x, current_input_xslices = _apply_one_x_fold_step(
                            ir, fold_node, current_input_folding_x, current_input_xslices, is_first_step=(step == 0))
                    # When chain is empty, `current` is the true entry, whose folding_factor_x/
                    # x_slices were already set to the FINAL folded target by
                    # calc_conv_folding_factor - reset to the unfolded baseline to avoid double-fold.
                    current['frontend']['input_tensor'].folding_factor_x = current_input_folding_x
                    current['frontend']['input_tensor'].x_slices = current_input_xslices
                    for step in range(len(chain), total_folds):
                        add_xfoldingConv_before_node(ir, current, step=step)
                    continue

                # Distribute the fold steps onto the LAST total_folds nodes of the chain
                # (closest to the real conv), not the first: a chain node that doesn't
                # need to fold stays a pure passthrough instead.
                num_skipped = len(chain) - total_folds
                for skipped_node in chain[:num_skipped]:
                    _reset_x_fold_passthrough(skipped_node, unfolded_xslices)

                current_input_folding_x = 0
                current_input_xslices = unfolded_xslices
                for step, fold_node in enumerate(chain[num_skipped:]):
                    current_input_folding_x, current_input_xslices = _apply_one_x_fold_step(
                        ir, fold_node, current_input_folding_x, current_input_xslices, is_first_step=(step == 0))
    return ir

def _allocate_tensors(ir, tensor_names, ddr_container, entry_type, description_prefix):
    tensors = ir.tensors
    amm_height = ir.amm_height
    for tensor_name in tensor_names:
        tensor = tensors[tensor_name]
        # ALEX FIX: Calculate size both ways and allocate the larger one
        # to handle both cases: with and without folding/padding
        shape_original = tensor.shape_real_x16
        shape_calculated = calculate_tensor_allocation_shape(tensor)

        # Calculate sizes in bytes for comparison
        size_original = np.prod(shape_original)
        size_calculated = np.prod(shape_calculated)

        # Use whichever is larger
        if size_calculated > size_original:
            tensor_shape = shape_calculated
        else:
            tensor_shape = shape_original

        tensor.data = np.zeros(tensor_shape, dtype=np.uint8)
        tensor_bytearray = create_tsnp_tensor_byte_array(tensor.data)
        ddr_entry = TensorDDREntry(
            tensor_bytearray,
            type=entry_type,
            description=f'{description_prefix}: {tensor_name}',
            shape=tensor_shape,
            amm_height=amm_height)
        tensor.ddr_entry = ddr_entry
        ddr_container.add_entry(ddr_entry)

def allocate_input_tensors(ir: internal_representation.IR) -> internal_representation.IR:
    _allocate_tensors(ir, ir.inputs, ir.inputs_ddr, DDREntryType.INPUT_TENSOR, 'Input tensor')
    return ir

def allocate_output_tensors(ir: internal_representation.IR) -> internal_representation.IR:
    _allocate_tensors(ir, ir.outputs, ir.outputs_ddr, DDREntryType.OUTPUT_TENSOR, 'Output tensor')
    return ir

def allocate_ddr_for_offloaded_tensors(ir: internal_representation.IR,offloaded_tensors_names) -> bool:
    program_and_intermediate_tensors_ddr = ir.ddr
    for current_intermediate_tensor_name in offloaded_tensors_names:
        if current_intermediate_tensor_name not in ir.tensors:
            return False
        current_tensor = ir.tensors[current_intermediate_tensor_name]
        tensor_shape = current_tensor.shape_real_x16
        if current_tensor.y_tiles!=1:
            raise ValueError ('Offloading of tiled tensor not supported yet') # Need to see if we offload the complete tensor or just a single tile
        current_tensor.data = np.zeros(tensor_shape,dtype=np.uint8)
        tensor_bytearray = create_tsnp_tensor_byte_array(current_tensor.data)
        ddr_entry_description = 'Intermediate tensor: %s' % (current_intermediate_tensor_name)
        current_intermediate_tensor_ddr_entry = TensorDDREntry(tensor_bytearray, type = DDREntryType.INTERMEDIATE_TENSOR, description = ddr_entry_description, shape = tensor_shape, amm_height = ir.amm_height)
        current_tensor.ddr_entry = current_intermediate_tensor_ddr_entry
        program_and_intermediate_tensors_ddr.add_entry(current_intermediate_tensor_ddr_entry) 
    return True 

def following_node_needs_dual_allocation(ir: internal_representation.IR,node,origin_blob_idx): # for example concat node need to allocate both of its input together so they are contiguous in mem
    following_nodes_params = node['frontend']['following_nodes_params']
    current_blob = ir.tiling_blobs[origin_blob_idx]
    for following_node_params in following_nodes_params:
        following_node = ir.graph.nodes[following_node_params[0]]
        if following_node['op_type'] in MULTIPLE_INPUT_OPS and following_node['name'] in current_blob.nodes_in_blob: # We need contigeous allocation only if concat node is in same blob
            dual_node_input_nodes_params = following_node['frontend']['preceding_nodes_params']
            if following_node['name'] in ir.offloaded_concat_ops: # If this is the far node from long concat
                if following_node['frontend']['preceding_nodes_params'][0][0]==node['name']:
                    raise ValueError ('Something went wrong AMM allocation, didnt find best solution long_concat_far_node')
            return True, dual_node_input_nodes_params, following_node # Its MULTIPLE_INPUT_OP, MULTIPLE_INPUT_OP_inputs, the MULTIPLE_INPUT_OP node itself
    return False,[],None # Return values are is MULTIPLE_INPUT_OPS, MULTIPLE_INPUT_OP_inputs, MULTIPLE_INPUT_OP node

def mark_input_tensors_deallocation(ir: internal_representation.IR,amms,node,current_tile_num,current_htile_num): # This will deallocate after read from ddr allocation, at next excuted node after last consumer
    input_tensors = get_node_input_tensors(node)
    frontend = node['frontend']
    current_node_blob_idx = frontend['tiling_blob_idx']
    current_blob = ir.tiling_blobs[current_node_blob_idx]
    blob_outputs_names = current_blob.get_blob_outputs_names()
    xslices_at_htile = frontend['xslices_per_htile'][current_htile_num]
    node_op_type = node['op_type']
    inline_input_indices = INLINE_OPS.get(node_op_type, ())
    graph_nodes = ir.graph.nodes
    node_name = node['name']

    for input_index,tensor in enumerate(input_tensors):
        tensor_name = tensor.name
        num_xslices = packed_num_lanes(tensor.num_packed_xslices, xslices_at_htile)
        if tensor_name in blob_outputs_names: # blob output: deallocated elsewhere (after DDR write)
            continue
        for current_x_slice in range(num_xslices):
            amm_tensor_name = tensor.get_amm_tensor_name(current_node_blob_idx,current_tile_num,current_htile_num,current_x_slice)
            if amm_tensor_name not in amms.tensors_in_amm:
                continue
            ir_tensor = amms.tensors_in_amm[amm_tensor_name].tensor # graph tensor — needed to know if it feeds more nodes
            last_consumer_node_in_blob_found,last_consumer_node_in_blob_name = ir.get_last_tensor_consumer_in_blob(current_blob,ir_tensor)

            if last_consumer_node_in_blob_name!=node_name: # not the last consumer: deallocation set elsewhere
                continue
            last_consumer_node = graph_nodes[last_consumer_node_in_blob_name] # assumes consumers list is in execution order
            inline_tensor=False
            if input_index in inline_input_indices:
                inline_tensor = True
                ir_tensor.is_inline_tensor = True

            success,next_executed_node,next_executed_ytile,next_executed_htile,next_executed_xslice = ir.get_next_executed_node(last_consumer_node,error_on_last_node=True,current_tile_num=current_tile_num,current_htile_num=current_htile_num,current_xslice_num=current_x_slice)
            deallocating_node = next_executed_node
            deallocation_ytile_idx = next_executed_ytile
            deallocation_htile_idx = next_executed_htile
            deallocation_xslice_idx = next_executed_xslice
            # Inline tensors alias the output's memory and are freed only when the output's own
            # allocation overwrites them; deallocate_amm_tensors expects deallocating_node to stay
            # None for these (dict-entry removal only) or it leaks the blocks forever.
            if not inline_tensor:
                if amms.tensors_in_amm[amm_tensor_name].deallocating_node == None:
                    amms.tensors_in_amm[amm_tensor_name].deallocating_node = deallocating_node
                else:
                    raise ValueError ('Tensor %s already set for deallocation at. Illegal to set 2 deallocation points(new point is: %s)' % (tensor_name,last_consumer_node['name']))
            deallocated_tensor = TensorDeAllocationInfo(amm_tensor_name, input_index, current_tile_num, current_htile_num, current_x_slice, deallocation_ytile_idx, deallocation_htile_idx, deallocation_xslice_idx, inline_tensor=inline_tensor)
            if success:
                add_deallocation_tensor(deallocating_node, 'after_ddr_read_allocation', deallocated_tensor)
            elif node_op_type == OpType.ADD and inline_tensor and amm_tensor_name in amms.tensors_in_amm:
                # Terminal node: queuing this ADD inline input would target a key already drained,
                # leaking the entry. Drop the tracking entry directly; no blocks to free (inline).
                del amms.tensors_in_amm[amm_tensor_name]
            else:
                add_deallocation_tensor(deallocating_node, 'after_output_allocation', deallocated_tensor)
    return True

def _is_live_split_descendant(ir, name):
    # True only if `name` is still wired into the LIVE graph - either as some node's current output,
    # or as an input-alias tensor still referenced by a live consumer's inputs. A stale map entry
    # left over from a rolled-back escalation attempt satisfies neither condition.
    tensor = ir.tensors.get(name)
    if tensor is None:
        return False
    producer_name = getattr(tensor, 'producer', None)
    if producer_name is not None and producer_name in ir.graph.nodes():
        producer_outputs = ir.graph.nodes[producer_name].get('outputs', [])
        if producer_outputs and producer_outputs[0] == name:
            return True
    for consumer_name in getattr(tensor, 'consumers', []):
        if consumer_name in ir.graph.nodes() and name in ir.graph.nodes[consumer_name].get('inputs', []):
            return True
    return False

def mark_blob_outputs_deallocations(ir, amms, node, current_tile_num, current_htile_num):
    frontend = node['frontend']
    if is_resize_y_node(node):
        output_tensor = frontend['input_tensor']
    else:
        output_tensor = frontend['output_tensor']
    output_tensor_name = output_tensor.name
    current_node_blob_idx = frontend['tiling_blob_idx']
    num_xslices = frontend['out_xslices_per_htile'][current_htile_num]
    for current_x_slice in range(num_xslices):
        amm_tensor_name = output_tensor.get_amm_tensor_name(current_node_blob_idx,current_tile_num,current_htile_num,current_x_slice)
        deallocating_node, deallocation_ytile_idx, deallocation_htile_idx, deallocation_xslice_idx = ir.get_blob_output_deallocating_node(current_node_blob_idx,current_tile_num,current_htile_num,current_x_slice,num_xslices)

        if amms.tensors_in_amm[amm_tensor_name].deallocating_node == None:
            amms.tensors_in_amm[amm_tensor_name].deallocating_node = deallocating_node
        else:
            raise ValueError ('Tensor %s already set for deallocation at. Illegal to set 2 deallocation points(new point is: %s)' % (amm_tensor_name,deallocating_node['name']))

        tensor = ir.tensors[output_tensor_name]
        if len(tensor.consumers)>0:
            last_consumer_node_name = tensor.consumers[-1]
            last_consumer_node = ir.graph.nodes()[last_consumer_node_name]
            inline_tensor=False # Output of last node will always go to another blob so even if its inline we need to dealocate it and it will then be read by new blob
            # The consumer may refer to this blob output via any name on its split lineage: size
            # splits read via an ANCESTOR, input-channel splits via a DESCENDANT. Match this tensor's
            # own lineage (not the root's) so sibling subtrees can't be mis-matched.
            lineage = {output_tensor_name}
            ancestor = output_tensor_name
            while ancestor in ir.split_tensor_to_original_tensor_map:
                ancestor = ir.split_tensor_to_original_tensor_map[ancestor]
                lineage.add(ancestor)
            frontier = list(ir.original_tensor_to_split_tensor_map.get(output_tensor_name, []))
            while frontier:
                child = frontier.pop()
                if child in lineage:
                    continue
                if not _is_live_split_descendant(ir, child):
                    # A rolled-back escalation attempt can leave a stale map entry whose split
                    # node/chunks no longer exist; skip names that aren't a live node's current output.
                    continue
                lineage.add(child)
                frontier.extend(ir.original_tensor_to_split_tensor_map.get(child, []))
            input_index = next((idx for idx, inp in enumerate(last_consumer_node['inputs'])
                                if inp in lineage), None)
            if input_index is None:
                # No ancestor/descendant found: a plain merged-tensor reader has every split piece at
                # the same DDR region+offset, so fall back to any sibling name in the ROOT's split tree.
                root = output_tensor_name
                while root in ir.split_tensor_to_original_tensor_map:
                    root = ir.split_tensor_to_original_tensor_map[root]
                sibling_tree = {root}
                frontier = list(ir.original_tensor_to_split_tensor_map.get(root, []))
                while frontier:
                    child = frontier.pop()
                    if child in sibling_tree:
                        continue
                    if not _is_live_split_descendant(ir, child):
                        # See the identical guard/comment on the narrow lineage walk above.
                        continue
                    sibling_tree.add(child)
                    frontier.extend(ir.original_tensor_to_split_tensor_map.get(child, []))
                input_index = next((idx for idx, inp in enumerate(last_consumer_node['inputs'])
                                    if inp in sibling_tree), None)
            if input_index is None:
                # Fall back to the original name so the failure is a clear ValueError naming the tensor,
                # instead of an opaque StopIteration, if no lineage name is present in the consumer inputs.
                if output_tensor_name not in last_consumer_node['inputs']:
                    raise ValueError(
                        'blob output %r not in inputs of last consumer %r: consumer inputs=%r tensor.consumers=%r '
                        'producer=%r lineage=%r sibling_tree=%r allocating node=%r op_type=%s. Consumers whose inputs do contain it: %r' % (
                            output_tensor_name, last_consumer_node_name, last_consumer_node['inputs'], list(tensor.consumers),
                            tensor.producer, sorted(lineage), sorted(sibling_tree),
                            node['name'], node['op_type'],
                            [n for n, d in ir.graph.nodes(data=True) if output_tensor_name in d.get('inputs', [])]))
                input_index = last_consumer_node['inputs'].index(output_tensor_name)
        else:
            input_index=0
            inline_tensor=False
        deallocated_tensor = TensorDeAllocationInfo(amm_tensor_name, input_index, current_tile_num, current_htile_num, current_x_slice, deallocation_ytile_idx, deallocation_htile_idx, deallocation_xslice_idx, inline_tensor=inline_tensor)
        add_deallocation_tensor(deallocating_node, 'after_output_allocation', deallocated_tensor)
    return True


def set_following_nodes_inputs_allocations(ir,allocating_node,following_nodes_params,allocated_blocks,current_tile_num,current_htile_num):

    for current_following_node in following_nodes_params:
        following_node = ir.graph.nodes[current_following_node[0]]
        if is_resize_y_node(following_node):
            continue
        if (following_node['frontend']['tiling_blob_idx'] == allocating_node['frontend']['tiling_blob_idx']): # If following node is from different blob, its input allocation will be done by "allocate_next_tile_input_mem" or "" in case its immediate read
            input_index = current_following_node[1]
            follower_xslices = following_node['frontend']['xslices_per_htile'][current_htile_num]
            produced_blocks = len(allocated_blocks)
            # An x-fold consumer may pad with one extra dummy slice (no real producer block); tolerate
            # by aliasing it to the producer's last real block. Any other shortfall is a real bug.
            allowed_extra = 1 if (is_x_fold_node(following_node) and produced_blocks >= 1) else 0
            if follower_xslices > produced_blocks + allowed_extra:
                raise ValueError(
                    'xslices_per_htile mismatch within blob %d at htile %d: following node %s expects %d xslices '
                    'but allocating node %s produced %d blocks. Check kernel_size halo handling.'
                    % (allocating_node['frontend']['tiling_blob_idx'], current_htile_num,
                       following_node['name'], follower_xslices, allocating_node['name'], produced_blocks))
            for current_x_slice in range(follower_xslices):
                block = allocated_blocks[current_x_slice] if current_x_slice < produced_blocks else allocated_blocks[produced_blocks - 1]
                following_node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][current_htile_num][current_x_slice][input_index] = block
                following_node['backend']['allocated_amm_blocks_for_input_odd_grid'][current_tile_num][current_htile_num][current_x_slice][input_index] = block

def read_missing_tensors_from_ddr(ir: internal_representation.IR,node,current_tile_num,current_htile_num):
    current_op_type = node['op_type']
    current_op_input_tensor_name = node['inputs'][0]
    force_folding_y, force_unfolding_y = get_folding_flags(node)
    current_op_input_tensor_shape = ir.tensors[current_op_input_tensor_name].get_folded_shape(folding_conv_y = force_folding_y,unfolding_conv_y=force_unfolding_y)
    current_op_input_channels = current_op_input_tensor_shape[1]
    if current_op_input_channels!=node['backend']['input_channels'] and current_op_type!=OpType.ADD: # In add the 'backend' 'input_channels' property relates to the conv input size
        # A split-produced conv is created after get_ops_grid_config runs and its post-split
        # force-folding flags finalize only in the later re-blob; heal its backend input_channels to
        # the freshly-computed count. Every other node still hard-fails on genuine geometry bugs.
        if node['frontend'].get('was_split'):
            node['backend']['input_channels'] = current_op_input_channels
        else:
            raise ValueError ('calced input channels != stored input channels. Please check...')
    current_blob_idx = node['frontend']['tiling_blob_idx']
    # If node needs to read inputs from DDR allocate AMM mem for them
    if current_op_type == OpType.ADD:
        input_tensors = node['frontend']['input_tensors']
    else:
        input_tensors = [node['frontend']['input_tensor']]
    num_xslices = packed_num_lanes(input_tensors[0].num_packed_xslices, node['frontend']['xslices_per_htile'][current_htile_num])
    for input_index,input_tensor in enumerate(input_tensors):
        for current_x_slice in range(num_xslices):
            success = allocate_and_mark_for_read_input_tensor(ir,input_tensor,node,current_blob_idx,current_tile_num,current_htile_num,current_x_slice,load_before_op=True)
            if not success:
                return False
    
    return True    

def allocate_and_mark_for_read_input_tensor(ir: internal_representation.IR,input_tensor,allocation_node,consumer_node_blob_idx,current_tile_idx,current_htile_idx,current_xslice_num,load_before_op = False):
    amms = ir.amms
    # Node frontends can hold a stale copy of a shared input tensor (after a deepcopy snapshot). Read the shared entry
    # from ir.tensors so the consumer list used for this read is the one the input-channel split registered on.
    if input_tensor.name in ir.tensors:
        input_tensor = ir.tensors[input_tensor.name]
    if load_before_op:
        sequencer_tensor_load_marker_name = 'tensors_to_load_immediately_from_ddr'
    else:
        sequencer_tensor_load_marker_name = 'next_tile_tensors_to_load_from_ddr'
    consumer_node_blob = ir.tiling_blobs[consumer_node_blob_idx]
    blob_nodes_set = set(consumer_node_blob.nodes_in_blob)
    graph_nodes = ir.graph.nodes

    # CONV output-channel splits: patch consumers list if needed for CONV split dup nodes.
    # following_nodes_params may include _split dups but consumers list may not (persists through deepcopy but resets).
    # If allocation_node is a _split dup and it reads this input tensor, add it to consumers now.
    if '_split' in allocation_node['name'] and input_tensor.name in allocation_node.get('inputs', []):
        if allocation_node['name'] not in input_tensor.consumers:
            producer_node = ir.graph.nodes.get(input_tensor.producer)
            if producer_node and allocation_node['name'] in [n for n, _ in producer_node['frontend'].get('following_nodes_params', [])]:
                input_tensor.consumers.append(allocation_node['name'])

    found_consumer_in_blob,current_input_tensor_dominant_consumer_node_name = ir.get_dominant_tensor_consumer_in_blob(consumer_node_blob,input_tensor)
    if not found_consumer_in_blob:
        raise ValueError(
            "Blobs input tensor consumer was not found in blob. please check... "
            "input_tensor=%r producer=%r consumers=%r allocation_node=%r (op_type=%s) "
            "consumer_node_blob_idx=%r nodes_in_blob=%r" % (
                input_tensor.name, input_tensor.producer, list(input_tensor.consumers),
                allocation_node['name'], allocation_node['op_type'],
                consumer_node_blob_idx, list(consumer_node_blob.nodes_in_blob)))
    current_input_tensor_dominant_consumer_node = ir.graph.nodes()[current_input_tensor_dominant_consumer_node_name]
    found_index,input_index_at_consumer = ir.get_consumer_input_index(current_input_tensor_dominant_consumer_node,input_tensor) # It is important to get index of the first consumer since we will use wmt of this node to actually load the tensor
    if not found_index:
        raise ValueError ('input not found at consumer node. please check...')

    current_op_type = current_input_tensor_dominant_consumer_node['op_type']
    force_folding_y, force_unfolding_y = get_folding_flags(current_input_tensor_dominant_consumer_node)
    current_op_input_tensor_shape = input_tensor.get_folded_shape(folding_conv_y = force_folding_y,unfolding_conv_y=force_unfolding_y)
    current_op_input_channels = current_op_input_tensor_shape[1]
    if current_op_input_channels!=current_input_tensor_dominant_consumer_node['backend']['input_channels'] and current_op_type not in MULTIPLE_INPUT_OPS: # In add the 'backend' 'input_channels' property relates to the conv input size
        # Self-heal split-produced conv nodes whose backend input_channels predate their post-split
        # force-folding flags (see the matching note in read_missing_tensors_from_ddr); hard-fail others.
        if current_input_tensor_dominant_consumer_node['frontend'].get('was_split'):
            current_input_tensor_dominant_consumer_node['backend']['input_channels'] = current_op_input_channels
        else:
            raise ValueError ('calced input channels != stored input channels. Please check...')
    input_channels_split = 1
    current_tile_tensor_name = input_tensor.get_amm_tensor_name(consumer_node_blob_idx,current_tile_idx,current_htile_idx,current_xslice_num)
    tensor_was_already_in_amm = False
    if current_tile_tensor_name not in amms.tensors_in_amm:
        input_channels_per_amm = current_op_input_channels / input_channels_split
        if is_packed_xslices(input_tensor.num_packed_xslices) or input_tensor.num_packed_xslices > 1:
            # current_xslice_num is the lane index; with a per-lane combination each lane sizes its AMM
            # block by its own pack count (e.g. [8,4] -> lane0 holds 8 slices, lane1 holds 4), derived
            # from this h-tile's slice count to support h_tiles>1.
            lane_pack = packed_lane_info(input_tensor.num_packed_xslices, input_tensor.xslices_per_htile[current_htile_idx], current_xslice_num)[0]
            input_channels_per_amm = math.ceil(input_channels_per_amm/MAX_GRID_WIDTH) * MAX_GRID_WIDTH * lane_pack
        blob_input_consumer_name = input_tensor.consumers[0]
        next_ops_inputs_tensors = []

        # Node whose (Concat) inputs need contiguous AMM allocation as a group - either the direct
        # dominant consumer, or that same-blob Concat, so input_tensor lands with its other operands.
        dual_alloc_node = None
        if current_op_type in MULTIPLE_INPUT_OPS:
            dual_alloc_node = current_input_tensor_dominant_consumer_node
        else:
            input_tensor_producer_name = input_tensor.producer
            if input_tensor_producer_name:
                input_tensor_producer = ir.graph.nodes()[input_tensor_producer_name]
                contiguous_inputs_allocation_node, _, following_concat_node = following_node_needs_dual_allocation(ir,input_tensor_producer,consumer_node_blob_idx)
                if contiguous_inputs_allocation_node:
                    # Only route into contiguous group allocation when input_tensor is literally one
                    # of the dual-op's own two direct operands (guards against an intervening
                    # identity hop or a stale input_tensor.producer); else fall through below.
                    dual_op_direct_producers = {p[0] for p in following_concat_node['frontend']['preceding_nodes_params']}
                    dual_op_operand_names = {ir.graph.nodes[p]['frontend']['output_tensor'].name for p in dual_op_direct_producers}
                    dual_op_output_name = following_concat_node['frontend']['output_tensor'].name
                    if (input_tensor_producer_name in dual_op_direct_producers
                            and input_tensor.name in (dual_op_operand_names | {dual_op_output_name})):
                        dual_alloc_node = following_concat_node

        if dual_alloc_node is not None and dual_alloc_node['op_type'] == OpType.CONCAT:
            # A same-blob Concat whose combined footprint can never fit AMM contiguously is unsplittable
            # regardless of how any operand is split; scatter it (a DDR channel-offset no-op) if eligible
            # so each operand gets a plain independent allocation instead.
            if dual_alloc_node['frontend'].get('is_scatter_concat'):
                dual_alloc_node = None
            else:
                _per_input_sizes = [ir.graph.nodes[p[0]]['frontend']['output_tensor'].get_folded_shape()[1]
                                     for p in dual_alloc_node['frontend']['preceding_nodes_params']]
                _combined = sum(int(math.ceil(s / AMM_CHANNELS_PER_BLOCK)) * AMM_CHANNELS_PER_BLOCK
                                 for s in _per_input_sizes)
                if _combined > amm_capacity(ir):
                    _scattered = scatter_concat_inputs(ir, dual_alloc_node)
                    if _scattered:
                        dual_alloc_node = None

        if dual_alloc_node is not None:
            next_ops_inputs_ic_per_amm = []
            # We need to prepare the output in AMM mem ready for the concatenation so we allocate output mem for all inputs of the Concat in contiguous mem
            inputs_nodes_params = dual_alloc_node['frontend']['preceding_nodes_params']
            input_index_sorted_inputs_nodes_params = sorted(inputs_nodes_params, key=lambda x:x[1])
            consumer_input_tensors = dual_alloc_node['frontend'].get('input_tensors', [])
            # An output-channel-split producer appends sibling entries at the same input index; when
            # dual_alloc_node was split in lockstep, preceding_nodes_params wasn't pruned to match, so
            # keep only the entry matching the live input tensor (else the first entry per index).
            _deduped_by_idx = {}
            for _prev_name, _idx in input_index_sorted_inputs_nodes_params:
                _tensor_for_idx = consumer_input_tensors[_idx] if _idx < len(consumer_input_tensors) else None
                if _idx not in _deduped_by_idx:
                    _deduped_by_idx[_idx] = (_prev_name, _idx)
                if _tensor_for_idx is not None and _tensor_for_idx.producer == _prev_name:
                    _deduped_by_idx[_idx] = (_prev_name, _idx)
            input_index_sorted_inputs_nodes_params = [_deduped_by_idx[_idx] for _idx in sorted(_deduped_by_idx)]
            num_slices = 1
            for input_node_params in input_index_sorted_inputs_nodes_params:
                input_node = ir.graph.nodes[input_node_params[0]]
                prod_xslices_per_htile = input_node['frontend']['output_tensor'].xslices_per_htile
                if current_htile_idx < len(prod_xslices_per_htile):
                    prod_slices_this_htile = prod_xslices_per_htile[current_htile_idx]
                else:
                    # Producer sits in another blob with FEWER h-tiles than this consumer (already wrote
                    # its whole output to DDR); size the read by the consumer's own slice count for this
                    # h-tile instead of indexing past the producer's shorter per-h-tile list.
                    consumer_xslices = dual_alloc_node['frontend']['xslices_per_htile']
                    prod_slices_this_htile = consumer_xslices[current_htile_idx] \
                        if current_htile_idx < len(consumer_xslices) else consumer_xslices[-1]
                num_slices = max(num_slices, prod_slices_this_htile)
                # The tensor this consumer actually reads for this input (may be a channel slice of the
                # producer's full output - e.g. an elementwise-Add channel-split half reading in[lo:hi]).
                consumer_input_idx = input_node_params[1]
                consumer_input_tensor = consumer_input_tensors[consumer_input_idx] \
                    if consumer_input_idx < len(consumer_input_tensors) else None
                if consumer_input_tensor is not None and getattr(consumer_input_tensor, 'is_read_channel_slice', False):
                    # Read this input as the channel slice itself. Do NOT resolve to the merged producer
                    # output - that would read the FULL input and size/track AMM under the wrong name.
                    next_ops_inputs_tensors.append(consumer_input_tensor)
                    current_input_node_output_channels = consumer_input_tensor.get_folded_shape(folding_conv_y = force_folding_y,unfolding_conv_y=force_unfolding_y)[1]
                    next_ops_inputs_ic_per_amm.append(current_input_node_output_channels / input_channels_split)
                    continue
                needed_tensor_name = input_node['frontend']['output_tensor'].name
                # Map to the IMMEDIATE original (not split('_split')[0], which breaks on nested names).
                # Concat resolves to the merged tensor; Add widens only the fit-check channel count
                # but keeps the ACTUAL chunk tensor (merged tensor's names would fail blob-membership).
                if (dual_alloc_node['op_type'] == OpType.CONCAT
                        and needed_tensor_name in ir.split_tensor_to_original_tensor_map
                        and ir.split_tensor_to_original_tensor_map[needed_tensor_name]
                            == dual_alloc_node['frontend']['output_tensor'].name):
                    # Only resolve to the merged original when the map entry recombines FOR THIS
                    # CONCAT specifically - a tensor fanning out to multiple Concats can be mapped to
                    # a DIFFERENT concat's output, which would wire the wrong data into this one.
                    needed_tensor_name = ir.split_tensor_to_original_tensor_map[needed_tensor_name]
                    next_ops_inputs_tensors.append(ir.tensors[needed_tensor_name])
                    current_input_node_output_channels = ir.tensors[needed_tensor_name].get_folded_shape()[1]
                elif (dual_alloc_node['op_type'] in MULTIPLE_INPUT_OPS
                        and needed_tensor_name in ir.split_tensor_to_original_tensor_map
                        and ir.split_tensor_to_original_tensor_map[needed_tensor_name]
                            == dual_alloc_node['frontend']['output_tensor'].name):
                    # Only widen when the map entry recombines into THIS SAME dual_alloc_node's output;
                    # an unrelated node's output (e.g. a different scattered Concat) would inflate size.
                    merged_name = ir.split_tensor_to_original_tensor_map[needed_tensor_name]
                    next_ops_inputs_tensors.append(input_node['frontend']['output_tensor'])
                    current_input_node_output_channels = ir.tensors[merged_name].get_folded_shape()[1]
                else:
                    next_ops_inputs_tensors.append(input_node['frontend']['output_tensor']) # We use the y_folding/unfolding of the concat input
                    # Input node output channels will be input channels of the concat node
                    # Input node output tensor is actually the input tensor of consumer node. this is why we use it as below
                    current_input_node_output_channels = input_node['frontend']['output_tensor'].get_folded_shape(folding_conv_y = force_folding_y,unfolding_conv_y=force_unfolding_y)[1]
                next_ops_inputs_ic_per_amm.append(current_input_node_output_channels / input_channels_split)
            # num_slices is how many x-slices THIS consumer reads for this h-tile - its OWN
            # xslices_per_htile, not the producers': the two tilings can differ (finer/coarser h_tiles
            # between blobs), under/over-counting causes a KeyError or a needless channel-split.
            _consumer_xsl = dual_alloc_node['frontend'].get('xslices_per_htile', [])
            if current_htile_idx < len(_consumer_xsl):
                num_slices = _consumer_xsl[current_htile_idx]
            allocated_blocks,per_input_first_block_index = amms.allocate_contiguous_mem(allocation_node['name'],next_ops_inputs_tensors,
                                                                                        next_ops_inputs_ic_per_amm,tensors_blob_idx=consumer_node_blob_idx,tensors_tile_num=current_tile_idx,tensors_htile_num=current_htile_idx,
                                                                                        current_tile_num=current_tile_idx,current_htile_num=current_htile_idx,num_slices=num_slices,is_valid=False)
            # allocate_contiguous_mem signals failure via per_input_first_block_index=None. Checking
            # len(allocated_blocks[0]) instead would miss an OOM that hit a later amm, letting the caller
            # wire rolled-back blocks into the blob's grids (KeyError downstream).
            if per_input_first_block_index is None:
                return False
            # Refresh dual_alloc_node's cached input_tensors to the tensors just resolved/registered in
            # AMM - nothing else updates this cache when an unrelated upstream producer gets split later,
            # so a stale pre-split tensor could otherwise cause a KeyError downstream.
            _resolved_for_this_call = None
            for _resolved_tensor, (_, _idx) in zip(next_ops_inputs_tensors, input_index_sorted_inputs_nodes_params):
                if consumer_input_tensors and _idx < len(consumer_input_tensors):
                    consumer_input_tensors[_idx] = _resolved_tensor
                if _idx == input_index_at_consumer:
                    _resolved_for_this_call = _resolved_tensor
            # THIS call's own current_tile_tensor_name/input_tensor were fixed before dual_alloc
            # resolution ran, so the check below must key off the tensor actually allocated for this
            # call's own input index, or it looks up a name that was never registered (KeyError).
            if _resolved_for_this_call is not None and _resolved_for_this_call is not input_tensor:
                input_tensor = _resolved_for_this_call
                current_tile_tensor_name = input_tensor.get_amm_tensor_name(
                    consumer_node_blob_idx, current_tile_idx, current_htile_idx, current_xslice_num)
        else:
            allocated_blocks = amms.allocate_mem(allocation_node['name'],input_tensor,input_channels_per_amm,tensors_blob_idx=consumer_node_blob_idx,tensors_tile_num=current_tile_idx,tensors_htile_num=current_htile_idx,
                                                current_tile_num=current_tile_idx,current_htile_num=current_htile_idx,current_xslice_num=current_xslice_num,is_valid=False) # Is valid means that in current node we took care of reading the tensor/it is in mem from last op. In this case we handle this below and then set valid to true
            if len(allocated_blocks[0]) == 0:
                return False
            per_input_first_block_index = [0]
            next_ops_inputs_tensors.append(input_tensor)

        if DEBUG_PRINT_AMM_ALLOCATION:
            print('Allocating %d for blob input tensor (read from ddr) of op: %s' % (input_channels_per_amm,blob_input_consumer_name))
        # TODO: Add comment of what this loop is for
        for input_index,current_input_tensor in enumerate(next_ops_inputs_tensors):
            # TODO: Add comment of what this is for
            current_allocated_blocks = list_of_lists_split_at_pos(allocated_blocks,input_index,per_index_start_pos=per_input_first_block_index)
            if dual_alloc_node is not None:
                for xslice_num in range(num_slices):
                    slice_blocks = current_allocated_blocks[xslice_num*AMM_COUNT:xslice_num*AMM_COUNT+AMM_COUNT]
                    _wire_blocks_to_blob(ir, graph_nodes, blob_nodes_set, current_input_tensor,
                                         current_tile_idx, current_htile_idx, xslice_num, slice_blocks)
            else:
                _wire_blocks_to_blob(ir, graph_nodes, blob_nodes_set, current_input_tensor,
                                     current_tile_idx, current_htile_idx, current_xslice_num, current_allocated_blocks)
    else:
        tensor_was_already_in_amm = True

    if current_tile_tensor_name not in amms.tensors_in_amm:
        # This exact name should always be registered by this point; if not, dump the producer/consumer
        # blob and h-tiling context to diagnose a cross-blob tile/h-tile/x-slice mismatch, instead of
        # a bare KeyError with no context.
        producer_name = input_tensor.producer
        producer_node = ir.graph.nodes[producer_name] if producer_name and ir.graph.has_node(producer_name) else None
        producer_blob_idx = producer_node['frontend'].get('tiling_blob_idx') if producer_node else None
        producer_h_tiles = producer_node['frontend'].get('h_tiles') if producer_node else None
        producer_xslices = producer_node['frontend'].get('xslices_per_htile') if producer_node else None
        consumer_h_tiles = allocation_node['frontend'].get('h_tiles')
        consumer_xslices = allocation_node['frontend'].get('xslices_per_htile')
        dual_alloc_node_name = dual_alloc_node['name'] if dual_alloc_node is not None else None
        registered_tensor_names = [t.name for t in next_ops_inputs_tensors]
        raise KeyError(
            '%s not registered in amms.tensors_in_amm while allocating input %r for consumer %r '
            '(tensor_was_already_in_amm=%s). producer=%r blob_idx=%s h_tiles=%s xslices_per_htile=%s; '
            'consumer blob_idx=%s tile=%d htile=%d xslice=%d h_tiles=%s xslices_per_htile=%s; '
            'dominant_consumer=%r (op_type=%s); dual_alloc_node=%r num_slices=%s; '
            'next_ops_inputs_tensors=%s (input_tensor.name in it: %s)' % (
                current_tile_tensor_name, input_tensor.name, allocation_node['name'], tensor_was_already_in_amm,
                producer_name, producer_blob_idx, producer_h_tiles, producer_xslices,
                consumer_node_blob_idx, current_tile_idx, current_htile_idx, current_xslice_num,
                consumer_h_tiles, consumer_xslices,
                current_input_tensor_dominant_consumer_node_name, current_op_type,
                dual_alloc_node_name, locals().get('num_slices'),
                registered_tensor_names, input_tensor.name in registered_tensor_names))

    if not amms.tensors_in_amm[current_tile_tensor_name].is_valid: # Is valid will be true if tensor was already loaded (read from ddr) by preceding tile.
        amms.tensors_in_amm[current_tile_tensor_name].is_valid = True
        tensor_found,current_input_tensor_first_consumer_node_name = ir.get_first_tensor_consumer_in_blob(consumer_node_blob,input_tensor)
        if not tensor_found:
            raise ValueError ('Didnt find tensors first consumer, this shouldnt have happend. please check integrity')
        current_input_tensor_first_consumer_node = ir.graph.nodes()[current_input_tensor_first_consumer_node_name]
        tensor_found,first_consumer_node_input_index = ir.get_consumer_input_index(current_input_tensor_first_consumer_node,input_tensor)
        if not tensor_found:
            raise ValueError ('Didnt find tensors first consumer, this shouldnt have happend. please check integrity')
        # The first consumer node is used below in the input tensor info struct as it will be used when actually reading the tesnor for wmt generation. So we must specify the node which
        # its input tensor is actually read and not the allocating node which is the node/time where the tensor is being read. 
        input_tensor_info = InputTensorInfo(input_tensor.name,first_consumer_node_input_index,current_tile_idx,current_htile_idx,current_xslice_num,current_tile_idx,current_htile_idx,current_xslice_num,consumer_node=current_input_tensor_first_consumer_node)
        if sequencer_tensor_load_marker_name in allocation_node['backend']:
            allocation_node['backend'][sequencer_tensor_load_marker_name].append(input_tensor_info)
        else:
            allocation_node['backend'][sequencer_tensor_load_marker_name] = [input_tensor_info]
    
    return True

def read_concats_missing_tensors_from_ddr(ir: internal_representation.IR,node,current_tile_num,current_htile_num):
    current_op_input_tensor_name = node['inputs'][0]
    current_op_input_tensor_shape = ir.tensors[current_op_input_tensor_name].get_folded_shape()
    node_blob_idx = node['frontend']['tiling_blob_idx']
    num_xslices = node['frontend']['xslices_per_htile'][current_htile_num]

    # If node needs to read inputs from DDR allocate AMM mem for them
    input_tensors = node['frontend']['input_tensors']
    for input_index,input_tensor in enumerate(input_tensors):
        for current_x_slice in range(num_xslices):
            success = allocate_and_mark_for_read_input_tensor(ir,input_tensor,node,node_blob_idx,current_tile_num,current_htile_num,current_x_slice,load_before_op = True) #Since its a concat this will allocate both tensors as contigous
            if not success:
                return False
    
    return True
                
def allocate_next_tile_input_mem(ir: internal_representation.IR,allocation_node,current_blob_idx,current_tile_num,current_htile_num,current_blob_num_tiles,
                                 consecutive_2_tile_blobs, prev_blob_lines_in_ddr, pre_read_next_htile=False, current_blob_num_htiles=1):
    # When pre_read_next_htile=True we pre-read the next h-tile within the same blob/y-tile
    # (driven by 'next_htile_tensors_to_load_from_ddr'). Otherwise we pre-read the next y-tile
    # (driven by 'next_tile_tensors_to_load_from_ddr') as before.
    if pre_read_next_htile:
        if current_htile_num >= current_blob_num_htiles - 1:
            return True
        next_tile_blob_idx = current_blob_idx
        next_tile_idx = current_tile_num
        target_htile_idx = current_htile_num + 1
        sequencer_tensor_load_marker_name = 'next_htile_tensors_to_load_from_ddr'
    else:
        if (current_tile_num == current_blob_num_tiles-1):
            # If we need to read 1st tile of next blob we make sure all its lines already written to DDR.
            # If not we dont read it here.
            if not ir.is_next_tile_written_to_ddr(current_blob_idx, consecutive_2_tile_blobs, prev_blob_lines_in_ddr):
                if not node_avoids_ddr_write(allocation_node):
                    allocation_node['backend']['wait_for_last_tile_write_end'] = current_tile_num
                return True
            next_tile_blob_idx = current_blob_idx+1
            next_tile_idx = 0
        else:
            next_tile_blob_idx = current_blob_idx
            next_tile_idx = current_tile_num+1
        target_htile_idx = 0
        sequencer_tensor_load_marker_name = 'next_tile_tensors_to_load_from_ddr'

    next_tile_blob = ir.tiling_blobs[next_tile_blob_idx]
    amms = ir.amms
    next_blob_nodes_set = set(next_tile_blob.nodes_in_blob)
    graph_nodes = ir.graph.nodes

    # If node needs to read inputs from DDR allocate AMM mem for them
    input_tensors = next_tile_blob.inputs
    dominant_tensor_consumer_node_names = []
    for input_tensor in input_tensors:
        # The below gets the "dominant" consumer node. This means that in case of few consumers we need to allocate contigeous mem if one of the consumers is concat.
        found_consumer_in_blob,current_input_tensor_consumer_node_name = ir.get_dominant_tensor_consumer_in_blob(next_tile_blob,input_tensor)
        if not found_consumer_in_blob:
            raise ValueError ('Blobs input tensor consumer was not found in blob. please check...')
        current_input_tensor_consumer_node = ir.graph.nodes()[current_input_tensor_consumer_node_name]
        found_index,input_index_at_consumer = ir.get_consumer_input_index(current_input_tensor_consumer_node,input_tensor) # It is important to get index of the first consumer since we will use wmt of this node to actually load the tensor
        if not found_index:
            raise ValueError ('input not found at consumer node. please check...')

        current_op_type = current_input_tensor_consumer_node['op_type']
        force_folding_y, force_unfolding_y = get_folding_flags(current_input_tensor_consumer_node)
        current_op_input_tensor_shape = input_tensor.get_folded_shape(folding_conv_y = force_folding_y,unfolding_conv_y=force_unfolding_y)
        
        current_op_input_channels = current_op_input_tensor_shape[1]
        if current_op_input_channels!=current_input_tensor_consumer_node['backend']['input_channels'] and current_op_type not in MULTIPLE_INPUT_OPS: # In add the 'backend' 'input_channels' property relates to the conv input size
            raise ValueError ('calced input channels != stored input channels. Please check...')
        input_channels_split = 1

        # This next-tile pre-read path also allocates all of a same-blob Concat's inputs contiguously,
        # so it needs the same scatter-to-DDR fallback when the combined size can never fit AMM.
        if (current_op_type == OpType.CONCAT) and (current_input_tensor_consumer_node_name not in dominant_tensor_consumer_node_names) \
                and not current_input_tensor_consumer_node['frontend'].get('is_scatter_concat'):
            _per_input_sizes = [ir.graph.nodes[p[0]]['frontend']['output_tensor'].get_folded_shape()[1]
                                 for p in current_input_tensor_consumer_node['frontend']['preceding_nodes_params']]
            _combined = sum(int(math.ceil(s / AMM_CHANNELS_PER_BLOCK)) * AMM_CHANNELS_PER_BLOCK for s in _per_input_sizes)
            if _combined > amm_capacity(ir):
                scatter_concat_inputs(ir, current_input_tensor_consumer_node)

        if (current_op_type in MULTIPLE_INPUT_OPS) and (current_input_tensor_consumer_node_name not in dominant_tensor_consumer_node_names) \
                and not current_input_tensor_consumer_node['frontend'].get('is_scatter_concat'):
            next_ops_inputs_tensors = []
            next_ops_inputs_ic_per_amm = []
            # We need to prepare the output in AMM mem ready for the concatenation so we allocate output mem for all inputs of the Concat in contiguous mem
            inputs_nodes_params = current_input_tensor_consumer_node['frontend']['preceding_nodes_params']
            num_slices = current_input_tensor_consumer_node['frontend']['xslices_per_htile'][target_htile_idx]
            input_index_sorted_inputs_nodes_params = sorted(inputs_nodes_params, key=lambda x:x[1])
            consumer_input_tensors = current_input_tensor_consumer_node['frontend'].get('input_tensors', [])
            for input_node_params in input_index_sorted_inputs_nodes_params:
                input_node = ir.graph.nodes[input_node_params[0]]
                # Mirror allocate_and_mark_for_read_input_tensor's resolution: a channel-slice read must
                # never be widened via the split map, and widening to the merged size is only valid
                # when this consumer is the genuine recombine target - else keep the chunk's real size.
                consumer_input_idx = input_node_params[1]
                consumer_input_tensor = consumer_input_tensors[consumer_input_idx] \
                    if consumer_input_idx < len(consumer_input_tensors) else None
                tensor_name = input_node['frontend']['output_tensor'].name
                if consumer_input_tensor is not None and getattr(consumer_input_tensor, 'is_read_channel_slice', False):
                    next_ops_inputs_tensors.append(consumer_input_tensor)
                    current_input_node_output_channels = consumer_input_tensor.get_folded_shape()[1]
                elif (tensor_name in ir.split_tensor_to_original_tensor_map
                        and ir.split_tensor_to_original_tensor_map[tensor_name]
                            == current_input_tensor_consumer_node['frontend']['output_tensor'].name):
                    orig_tensor_name = ir.split_tensor_to_original_tensor_map[tensor_name]
                    next_ops_inputs_tensors.append(input_node['frontend']['output_tensor'])
                    current_input_node_output_channels = ir.tensors[orig_tensor_name].get_folded_shape()[1]
                else:
                    next_ops_inputs_tensors.append(input_node['frontend']['output_tensor'])
                    current_input_node_output_channels = input_node['frontend']['output_tensor'].get_folded_shape()[1]
                if 'force_folding_y' in current_input_tensor_consumer_node['frontend']:
                    current_input_node_output_channels = current_input_node_output_channels * 2
                next_ops_inputs_ic_per_amm.append(current_input_node_output_channels / input_channels_split)
            allocated_blocks,per_input_first_block_index = amms.allocate_contiguous_mem(allocation_node['name'],next_ops_inputs_tensors,
                                                                                        next_ops_inputs_ic_per_amm,tensors_blob_idx=next_tile_blob_idx,tensors_tile_num=next_tile_idx,tensors_htile_num=target_htile_idx,
                                                                                        current_tile_num=current_tile_num,current_htile_num=current_htile_num,num_slices=num_slices, is_valid=False)
            if len(allocated_blocks[0]) == 0:
                return True # If there is not enough mem maybe we can load it as immediate at blob start
            if DEBUG_PRINT_AMM_ALLOCATION:
                print('Allocating %d for blob input tensor (read from ddr) of op: %s' % (input_channels_per_amm,blob_input_consumer_name))
            # After allocating the input mem we now need to update all its consumer nodes (not only the dominant in case of few consumers)
            for input_index,current_input_tensor in enumerate(next_ops_inputs_tensors):
                current_allocated_blocks = list_of_lists_split_at_pos(allocated_blocks,input_index,per_index_start_pos=per_input_first_block_index)
                producer_in_blob = current_input_tensor.producer in next_blob_nodes_set
                # Wire every allocated slice (num_slices = dominant consumer's x-slice count). Using the
                # per-input tensor's own xslices_per_htile instead under-wires when shared with a lower-
                # x-slice consumer, leaving a k3 consumer's halo slice empty and failing check_mem_integrity.
                for current_xslice_idx in range(num_slices):
                    slice_blocks = current_allocated_blocks[current_xslice_idx*AMM_COUNT:current_xslice_idx*AMM_COUNT+AMM_COUNT]
                    _wire_blocks_to_blob(ir, graph_nodes, next_blob_nodes_set, current_input_tensor,
                                         next_tile_idx, target_htile_idx, current_xslice_idx, slice_blocks)

                    current_tile_tensor_name = current_input_tensor.get_amm_tensor_name(next_tile_blob_idx,next_tile_idx,target_htile_idx,current_xslice_idx)
                    if (not amms.tensors_in_amm[current_tile_tensor_name].is_valid):
                        amms.tensors_in_amm[current_tile_tensor_name].is_valid = True
                        if not producer_in_blob:
                            input_tensor_info = InputTensorInfo(current_input_tensor.name,input_index,next_tile_idx,target_htile_idx,current_xslice_idx,current_tile_num,current_htile_num,current_xslice_idx,consumer_node=current_input_tensor_consumer_node)
                            allocation_node['backend'].setdefault(sequencer_tensor_load_marker_name, []).append(input_tensor_info)
        else:
            num_xslices = packed_num_lanes(input_tensor.num_packed_xslices, current_input_tensor_consumer_node['frontend']['xslices_per_htile'][target_htile_idx])
            for current_xslice_idx in range(num_xslices):
                current_tile_tensor_name = input_tensor.get_amm_tensor_name(next_tile_blob_idx,next_tile_idx,target_htile_idx,current_xslice_idx)
                tensor_was_already_in_amm = False
                if current_tile_tensor_name not in amms.tensors_in_amm: # There could be a case where the tensor is already in mem if it is part of a contigeous mem allocation which was already allocated
                    input_channels_per_amm = current_op_input_channels / input_channels_split
                    if is_packed_xslices(input_tensor.num_packed_xslices) or input_tensor.num_packed_xslices > 1:
                        lane_pack = packed_lane_info(input_tensor.num_packed_xslices, input_tensor.xslices_per_htile[target_htile_idx], current_xslice_idx)[0]
                        input_channels_per_amm = math.ceil(input_channels_per_amm/MAX_GRID_WIDTH) * MAX_GRID_WIDTH * lane_pack
                    blob_input_consumer_name = input_tensor.consumers[0]
                    next_ops_inputs_tensors = []
                    if 'force_folding_y' in current_input_tensor_consumer_node['frontend']:
                        input_channels_per_amm = input_channels_per_amm * 2
                    allocated_blocks = amms.allocate_mem(allocation_node['name'],input_tensor,input_channels_per_amm,tensors_blob_idx=next_tile_blob_idx,tensors_tile_num=next_tile_idx,tensors_htile_num=target_htile_idx,
                                                        current_tile_num=current_tile_num,current_htile_num=current_htile_num,current_xslice_num=current_xslice_idx,is_valid=False) # Is valid means  that in current node we took care of reading the tensor/it is in mem from last op. In this case we handle this below and then set valid to true
                    if len(allocated_blocks[0]) == 0:
                        return True # If there is not enough mem maybe we can load it as immediate at blob start
                    per_input_first_block_index = [0]
                    next_ops_inputs_tensors.append(input_tensor)
                    if DEBUG_PRINT_AMM_ALLOCATION:
                        print('Allocating %d for blob input tensor (read from ddr) of op: %s' % (input_channels_per_amm,blob_input_consumer_name))
                    # After allocating the input mem we now need to update all its consumer nodes (not only the dominant in case of few consumers)
                    for input_index,current_input_tensor in enumerate(next_ops_inputs_tensors):
                        current_allocated_blocks = list_of_lists_split_at_pos(allocated_blocks,input_index,per_index_start_pos=per_input_first_block_index)
                        _wire_blocks_to_blob(ir, graph_nodes, next_blob_nodes_set, current_input_tensor,
                                             next_tile_idx, target_htile_idx, current_xslice_idx, current_allocated_blocks)
                else:
                    tensor_was_already_in_amm = True

                if (not amms.tensors_in_amm[current_tile_tensor_name].is_valid):
                    amms.tensors_in_amm[current_tile_tensor_name].is_valid = True
                    input_tensor_info = InputTensorInfo(input_tensor.name,input_index_at_consumer,next_tile_idx,target_htile_idx,current_xslice_idx,current_tile_num,current_htile_num,current_xslice_idx,consumer_node=current_input_tensor_consumer_node)
                    allocation_node['backend'].setdefault(sequencer_tensor_load_marker_name, []).append(input_tensor_info)
        dominant_tensor_consumer_node_names.append(current_input_tensor_consumer_node_name)

    return True

def get_concat_output_allocation_from_inputs_allocation(inputs_allocation):
    output_allocation=[]
    for idx,current_input_allocation in enumerate(inputs_allocation):
        if idx==0:
            output_allocation = copy.deepcopy(current_input_allocation)
        else:
            for amm_idx,amm_allocation in enumerate(current_input_allocation):
                output_allocation[amm_idx].extend(amm_allocation)
    return output_allocation


def set_input_alocation_to_following_far_node(ir, node, even_grid_blocks_split, odd_grid_blocks_split, current_tile_num=0, current_htile_num=0):
    far_node_params = node['frontend']['following_nodes_params'][-1]
    following_node = ir.graph.nodes[far_node_params[0]]
    if following_node['op_type'] != OpType.CONCAT:
        raise ValueError('Expected it to be the concat. Please Check')

    input_index = far_node_params[1]
    following_backend = following_node['backend']
    input_even = following_backend['allocated_amm_blocks_for_input_even_grid'][current_tile_num][current_htile_num]
    input_odd = following_backend['allocated_amm_blocks_for_input_odd_grid'][current_tile_num][current_htile_num]

    for xslice in range(following_node['frontend']['xslices_per_htile'][current_htile_num]):
        slice_start = xslice * AMM_COUNT
        slice_end = slice_start + AMM_COUNT
        input_even[xslice][input_index] = even_grid_blocks_split[slice_start:slice_end]
        input_odd[xslice][input_index] = odd_grid_blocks_split[slice_start:slice_end]

def move_outputs_write_to_amm_to_erliest_point(ir: internal_representation.IR):
    nodes_execution_order = ir.lexicographical_topological_sorted_graph
    graph_nodes = ir.graph.nodes
    tensors = ir.tensors
    split_map = ir.original_tensor_to_split_tensor_map

    outputs_producing_nodes_names = []
    for output_tensor in ir.outputs:
        if output_tensor in split_map:
            outputs_producing_nodes_names.extend(
                tensors[split_tensor].producer for split_tensor in split_map[output_tensor])
        else:
            outputs_producing_nodes_names.append(tensors[output_tensor].producer)

    for output_producing_node_name in outputs_producing_nodes_names:
        # An output tensor may have no in-graph producer (passthrough, or produced by MXP); skip
        # rather than index graph_nodes[None] (KeyError).
        if output_producing_node_name is None or output_producing_node_name not in graph_nodes:
            continue
        output_producing_node = graph_nodes[output_producing_node_name]
        preceding_params = output_producing_node['frontend']['preceding_nodes_params']

        if not preceding_params:
            continue

        last_preceding_node_name = preceding_params[-1][0]
        last_preceding_node = graph_nodes[last_preceding_node_name]
        ops_before = last_preceding_node['frontend']['preceding_nodes_params']

        if not ops_before:
            continue

        anchor_node_name = ops_before[-1][0]
        anchor_node = graph_nodes[anchor_node_name]
        anchor_following_params = anchor_node['frontend']['following_nodes_params']

        is_anchor_followed_by_reorder_node = False
        for node_idx, anchor_following in enumerate(anchor_following_params):
            if 'reorder_node' in graph_nodes[anchor_following[0]]:
                is_anchor_followed_by_reorder_node = True
                if node_idx != 0:
                    raise ValueError('Expected ordering node to be the first executed node after anchor node.')
                break
        following_node_index = next(
            (i for i, p in enumerate(anchor_following_params) if p[0] == last_preceding_node_name),
            None)
        if following_node_index is None:
            continue
        last_preceding_node_params = anchor_following_params.pop(following_node_index)
        insert_pos = 1 if is_anchor_followed_by_reorder_node else 0
        anchor_following_params.insert(insert_pos, last_preceding_node_params)

        anchor_node_index = nodes_execution_order.index(anchor_node_name)
        if is_anchor_followed_by_reorder_node:
            anchor_node_index += 1
        last_preceding_node_execution_index = nodes_execution_order.index(last_preceding_node_name)
        current_output_node_execution_index = nodes_execution_order.index(output_producing_node_name)

        if current_output_node_execution_index > anchor_node_index + 2:
            nodes_execution_order.pop(current_output_node_execution_index)
            nodes_execution_order.pop(last_preceding_node_execution_index)
            nodes_execution_order.insert(anchor_node_index + 1, output_producing_node_name)
            nodes_execution_order.insert(anchor_node_index + 1, last_preceding_node_name)

def clear_previous_allocation_decisions(ir: internal_representation.IR):
    fields_to_clear = {'tensors_to_offload_to_ddr', 'tensors_for_deallocation_after_output_allocation',
                       'next_tile_tensors_to_load_from_ddr', 'next_htile_tensors_to_load_from_ddr', 'tensors_to_load_immediately_from_ddr',
                       'output_amm_allocated', 'tensors_for_deallocation_after_ddr_read_allocation',
                       'tensors_for_deallocation_after_output_allocation_report',
                       'tensors_for_deallocation_after_ddr_read_allocation_report'}
    graph_nodes = ir.graph.nodes()

    for node_name in ir.lexicographical_topological_sorted_graph:
        node = graph_nodes[node_name]
        backend = node['backend']
        frontend = node['frontend']

        for field_name in fields_to_clear:
            backend.pop(field_name, None)

        op_inputs_num = len(ir.get_non_constant_inputs(node))
        y_tiles = frontend['y_tiles']
        h_tiles = frontend['h_tiles']
        input_tensors = get_node_input_tensors(node)
        num_packed = input_tensors[0].num_packed_xslices
        input_x_slices = [packed_num_lanes(num_packed, x) for x in frontend['xslices_per_htile']]

        # Build twin (even/odd) allocation lattices directly instead of deepcopy(empty-lists).
        backend['allocated_amm_blocks_for_input_even_grid'] = [
            [[[[] for _ in range(op_inputs_num)] for _ in range(input_x_slices[h])] for h in range(h_tiles)]
            for _ in range(y_tiles)
        ]
        backend['allocated_amm_blocks_for_input_odd_grid'] = [
            [[[[] for _ in range(op_inputs_num)] for _ in range(input_x_slices[h])] for h in range(h_tiles)]
            for _ in range(y_tiles)
        ]

        # For a y-resize, out_xslices_per_htile == xslices_per_htile (only Y is rescaled), so this grid
        # already mirrors the input 1:1 for the pass-through block copy - no resize-specific sizing needed.
        output_x_slices = frontend['out_xslices_per_htile']
        backend['allocated_amm_blocks_for_output_even_grid'] = [
            [[[] for _ in range(output_x_slices[h])] for h in range(h_tiles)]
            for _ in range(y_tiles)
        ]
        backend['allocated_amm_blocks_for_output_odd_grid'] = [
            [[[] for _ in range(output_x_slices[h])] for h in range(h_tiles)]
            for _ in range(y_tiles)
        ]

    # Clear DDR Intermediate tensors
    for ddr_entry in ir.ddr.entries.copy():
        if ddr_entry.type == DDREntryType.INTERMEDIATE_TENSOR:
            ir.ddr.remove_entry(ddr_entry)

    for tensor in ir.tensors.values():
        tensor.is_inline_tensor = False


# When mem allocation fails, determines which blob to split (not yet a specific node).
def get_blob_to_split(current_blob_idx, current_y_tile, ir):
    # Try splitting the previous blob (tile 0 only), then this blob, then the next blob (last tile
    # only) - whichever still has more than one node to split.
    if current_y_tile == 0 and current_blob_idx > 0:
        if len(ir.tiling_blobs[current_blob_idx - 1].nodes_in_blob) > 1:
            return current_blob_idx - 1

    if len(ir.tiling_blobs[current_blob_idx].nodes_in_blob) > 1:
        return current_blob_idx

    num_y_tiles = ir.tiling_blobs[current_blob_idx].y_tiles
    num_blobs = len(ir.tiling_blobs)
    if current_y_tile == (num_y_tiles-1) and current_blob_idx < (num_blobs-1):
        if len(ir.tiling_blobs[current_blob_idx + 1].nodes_in_blob) > 1:
            return current_blob_idx + 1

    return current_blob_idx

def mem_allocation_per_htile(ir: internal_representation.IR, current_node_idx, node, outputs_producing_nodes_set, current_blob_idx, current_tiling_blob, current_y_tile, current_htile, consecutive_2_tile_blobs, prev_blob_lines_in_ddr):
    amms = ir.amms
    tiling_blobs = ir.tiling_blobs
    current_blob_num_tiles = current_tiling_blob.y_tiles
    current_blob_num_htiles = current_tiling_blob.h_tiles
    current_node_name = node['name']
    current_op_type = node['op_type']
    frontend = node['frontend']
    backend = node['backend']
    tensors_in_amm = amms.tensors_in_amm

    # Mark the offload-reuse node here, before the op-type branch split, since it's a
    # MULTIPLE_INPUT_OP and never reaches the single-input marker section below. Pure metadata,
    # harmless every tile - the sequencer only acts on it when a tile offload is in flight.
    if current_tiling_blob.offload_wait_node_idx >= 0 and current_node_idx == current_tiling_blob.offload_wait_node_idx:
        node['backend']['offload_wait_node'] = True

    if current_op_type in MULTIPLE_INPUT_OPS:
        if frontend.get('is_scatter_concat'):
            # Scatter concat: a no-op. Inputs' producers write directly into the aliased concat output
            # DDR region, so nothing to allocate here; consumers read the merged output from DDR.
            return ir, True, current_blob_idx
        # Same free-before-read as the single-input path below: frees pending inputs before reading
        # this tile's, avoiding transient co-residency that can overflow a memory-tight recombine Add.
        if debug_flags.DEBUG_AVOID_DDR_READ_WHILE_CONV or frontend.get('avoid_ddr_read_while_conv', False):
            mark_deallocation_for_report(node, 'after_ddr_read_allocation', amms, current_y_tile, current_htile)
        success = read_concats_missing_tensors_from_ddr(ir, node, current_y_tile, current_htile)
        if not success:
            return ir, False, current_blob_idx
        
        num_slices = frontend['xslices_per_htile'][current_htile]
        input_tensors = frontend['input_tensors']
        output_tensor = frontend['output_tensor']
        xslices_allocated_blocks = []
        is_concat = (current_op_type == OpType.CONCAT)

        input_even_grid = backend['allocated_amm_blocks_for_input_even_grid'][current_y_tile][current_htile]
        input_odd_grid = backend['allocated_amm_blocks_for_input_odd_grid'][current_y_tile][current_htile]
        output_even_grid = backend['allocated_amm_blocks_for_output_even_grid'][current_y_tile][current_htile]
        output_odd_grid = backend['allocated_amm_blocks_for_output_odd_grid'][current_y_tile][current_htile]

        for current_xslice_num in range(num_slices):
            if is_concat:
                even_grid_allocated_blocks = get_concat_output_allocation_from_inputs_allocation(input_even_grid[current_xslice_num])
                odd_grid_allocated_blocks = get_concat_output_allocation_from_inputs_allocation(input_odd_grid[current_xslice_num])
            else:
                even_grid_allocated_blocks = input_even_grid[current_xslice_num][0]
                odd_grid_allocated_blocks = input_odd_grid[current_xslice_num][0]
            output_even_grid[current_xslice_num] = even_grid_allocated_blocks
            output_odd_grid[current_xslice_num] = odd_grid_allocated_blocks
            output_amm_tensor_name = output_tensor.get_amm_tensor_name(current_blob_idx,current_y_tile,current_htile,current_xslice_num)
            output_allocated_at = (current_node_name,current_y_tile,current_htile,current_xslice_num)
            prior_amm_tensor = tensors_in_amm.get(output_amm_tensor_name)
            if prior_amm_tensor is not None and prior_amm_tensor.even_grids_amm_blocks != even_grid_allocated_blocks:
                for amm_idx, amm in enumerate(amms.amms):
                    amm.amm_allocator.deallocate_blocks(prior_amm_tensor.even_grids_amm_blocks[amm_idx])
            tensors_in_amm[output_amm_tensor_name] = AMMTensor(
                output_tensor, even_grid_allocated_blocks, odd_grid_allocated_blocks,
                current_y_tile,current_htile,current_xslice_num,is_valid=True,
                allocated_at=output_allocated_at)
            xslices_allocated_blocks.append(even_grid_allocated_blocks)

        # Deallocate tensors before adding input tensors to deallocation list
        mark_deallocation_for_report(node, 'after_output_allocation', amms, current_y_tile, current_htile)
        mark_deallocation_for_report(node, 'after_ddr_read_allocation', amms, current_y_tile, current_htile)

        for input_index, input_tensor in enumerate(input_tensors):
            input_tensor.is_inline_tensor = True
            for current_xslice_num in range(num_slices):
                amm_tensor_name = input_tensor.get_amm_tensor_name(current_blob_idx, current_y_tile, current_htile, current_xslice_num)
                deallocated_tensor = TensorDeAllocationInfo(amm_tensor_name, input_index, current_y_tile, current_htile, current_xslice_num, current_y_tile, current_htile, current_xslice_num, inline_tensor=True)
                add_deallocation_tensor(node, 'after_output_allocation', deallocated_tensor)

                if amm_tensor_name not in tensors_in_amm:
                    raise ValueError('At Concat, input %s is not in mem, need to read from DDR but this is not supported yet.' % input_tensor.name)
                if is_concat:
                    del tensors_in_amm[amm_tensor_name]

        set_following_nodes_inputs_allocations(ir, node, frontend['following_nodes_params'], xslices_allocated_blocks, current_tile_num=current_y_tile, current_htile_num=current_htile)

        if current_op_type == OpType.ADD:
            mark_input_tensors_deallocation(ir, amms, node, current_y_tile, current_htile)

        if current_node_name in outputs_producing_nodes_set:
            mark_blob_outputs_deallocations(ir, amms, node, current_y_tile, current_htile)

    elif current_op_type == OpType.SYNC:
        # Sync is a pure MXP handshake: output IS input, so it must never pay for an independent
        # output allocation (a lone Sync's single-node blob otherwise fails to fit AMM). Alias the
        # just-read input blocks directly as output instead of a second allocate_mem region.
        success = read_missing_tensors_from_ddr(ir, node, current_y_tile, current_htile)
        if not success:
            return ir, False, current_blob_idx

        num_xslices = frontend['xslices_per_htile'][current_htile]
        num_output_xslices = frontend['out_xslices_per_htile'][current_htile]
        input_even = backend['allocated_amm_blocks_for_input_even_grid'][current_y_tile][current_htile]
        input_odd = backend['allocated_amm_blocks_for_input_odd_grid'][current_y_tile][current_htile]
        output_even = backend['allocated_amm_blocks_for_output_even_grid'][current_y_tile][current_htile]
        output_odd = backend['allocated_amm_blocks_for_output_odd_grid'][current_y_tile][current_htile]

        input_tensor = frontend['input_tensor']
        output_tensor = frontend['output_tensor']
        xslices_allocated_blocks = []

        if num_output_xslices == num_xslices:
            for current_xslice_num in range(num_xslices):
                even_blocks = input_even[current_xslice_num][0]
                odd_blocks = input_odd[current_xslice_num][0]
                output_even[current_xslice_num] = even_blocks
                output_odd[current_xslice_num] = odd_blocks
                xslices_allocated_blocks.append(even_blocks)

                input_amm_tensor_name = input_tensor.get_amm_tensor_name(current_blob_idx, current_y_tile, current_htile, current_xslice_num)
                output_amm_tensor_name = output_tensor.get_amm_tensor_name(current_blob_idx, current_y_tile, current_htile, current_xslice_num)
                input_ammtensor = tensors_in_amm.get(input_amm_tensor_name)
                if input_ammtensor is not None:
                    # Re-key the SAME AMMTensor object under the output's name: same physical blocks,
                    # so lookups keyed on this node's output tensor find it and exactly one
                    # deallocation ever gets scheduled for these blocks.
                    tensors_in_amm[output_amm_tensor_name] = input_ammtensor
                    if output_amm_tensor_name != input_amm_tensor_name:
                        del tensors_in_amm[input_amm_tensor_name]
        else:
            # The wrapped MXP op performs a genuine spatial/channel reduction (e.g. Mean/
            # GlobalAveragePool), so in/out x-slice counts differ and output needs its own real
            # allocation, same as the generic (non-Sync) output-allocation path below.
            output_channels = ir.tensors[output_tensor.name].get_folded_shape()[1]
            for current_xslice_num in range(num_output_xslices):
                temp_allocated_blocks = amms.allocate_mem(current_node_name, output_tensor, output_channels,
                                                           tensors_blob_idx=current_blob_idx, tensors_tile_num=current_y_tile, tensors_htile_num=current_htile,
                                                           current_tile_num=current_y_tile, current_htile_num=current_htile, current_xslice_num=current_xslice_num, is_valid=True)
                if len(temp_allocated_blocks[0]) == 0:
                    return ir, False, current_blob_idx
                output_even[current_xslice_num] = temp_allocated_blocks
                output_odd[current_xslice_num] = temp_allocated_blocks
                xslices_allocated_blocks.append(temp_allocated_blocks)

            mark_input_tensors_deallocation(ir, amms, node, current_y_tile, current_htile)

        mark_deallocation_for_report(node, 'after_ddr_read_allocation', amms, current_y_tile, current_htile)

        set_following_nodes_inputs_allocations(ir, node, frontend['following_nodes_params'], xslices_allocated_blocks, current_tile_num=current_y_tile, current_htile_num=current_htile)

        if current_node_name in outputs_producing_nodes_set:
            mark_blob_outputs_deallocations(ir, amms, node, current_y_tile, current_htile)

    elif is_resize_y_node(node):
        preceding_node_name = frontend['preceding_nodes_params'][0][0]
        preceding_node = ir.graph.nodes[preceding_node_name]
        preceding_backend = preceding_node['backend']

        num_xslices = frontend['xslices_per_htile'][current_htile]
        input_even = backend['allocated_amm_blocks_for_input_even_grid'][current_y_tile][current_htile]
        input_odd = backend['allocated_amm_blocks_for_input_odd_grid'][current_y_tile][current_htile]
        output_even = backend['allocated_amm_blocks_for_output_even_grid'][current_y_tile][current_htile]
        output_odd = backend['allocated_amm_blocks_for_output_odd_grid'][current_y_tile][current_htile]
        prec_output_even = preceding_backend['allocated_amm_blocks_for_output_even_grid'][current_y_tile][current_htile]
        prec_output_odd = preceding_backend['allocated_amm_blocks_for_output_odd_grid'][current_y_tile][current_htile]

        for current_xslice_num in range(num_xslices):
            even_blocks = prec_output_even[current_xslice_num]
            odd_blocks = prec_output_odd[current_xslice_num]
            input_even[current_xslice_num][0] = even_blocks
            input_odd[current_xslice_num][0] = odd_blocks
            output_even[current_xslice_num] = even_blocks
            output_odd[current_xslice_num] = odd_blocks

        mark_deallocation_for_report(node, 'after_ddr_read_allocation', amms, current_y_tile, current_htile)

        if current_node_name in outputs_producing_nodes_set:
            mark_blob_outputs_deallocations(ir, amms, node, current_y_tile, current_htile)

    else:
        current_op_output_tensor_name = node['outputs'][0]
        current_op_output_channels = ir.tensors[current_op_output_tensor_name].get_folded_shape()[1]

        following_nodes_params = frontend['following_nodes_params']

        # PROTOTYPE (avoid_ddr_read_while_conv only): free the PREVIOUS tile's DDR-read inputs before
        # reading this tile's, reclaiming the double-buffer overhead once conv/read overlap is off.
        # Allocator-model change only - a matching sequencer free-before-read is required for HW.
        prev_tile_avoid_ddr_read = debug_flags.DEBUG_AVOID_DDR_READ_WHILE_CONV or frontend.get('avoid_ddr_read_while_conv', False)
        if prev_tile_avoid_ddr_read:
            mark_deallocation_for_report(node, 'after_ddr_read_allocation', amms, current_y_tile, current_htile)

        success = read_missing_tensors_from_ddr(ir, node, current_y_tile, current_htile)
        if not success:
            return ir, False, current_blob_idx

        mark_input_tensors_deallocation(ir, amms, node, current_y_tile, current_htile)

        mark_deallocation_for_report(node, 'after_ddr_read_allocation', amms, current_y_tile, current_htile)

        current_node_output_tensor = frontend['output_tensor']
        contiguous_inputs_allocation_node, inputs_nodes_params, _dual_node = following_node_needs_dual_allocation(ir, node, current_blob_idx)
        if contiguous_inputs_allocation_node and _dual_node['op_type'] == OpType.CONCAT:
            # Same rationale as allocate_and_mark_for_read_input_tensor: a same-blob Concat whose
            # combined operand footprint can never fit AMM contiguously is unsplittable, so scatter
            # it to DDR here too and let this node's output take a plain independent allocation.
            if _dual_node['frontend'].get('is_scatter_concat'):
                contiguous_inputs_allocation_node = False
            else:
                _per_input_sizes = [ir.graph.nodes[p[0]]['frontend']['output_tensor'].get_folded_shape()[1]
                                     for p in inputs_nodes_params]
                _combined = sum(int(math.ceil(s / AMM_CHANNELS_PER_BLOCK)) * AMM_CHANNELS_PER_BLOCK
                                 for s in _per_input_sizes)
                if _combined > amm_capacity(ir):
                    _scattered = scatter_concat_inputs(ir, _dual_node)
                    if _scattered:
                        contiguous_inputs_allocation_node = False
        allocated_blocks = []
        if contiguous_inputs_allocation_node:
            num_slices = frontend['out_xslices_per_htile'][current_htile]
            output_amm_allocated = backend.get('output_amm_allocated')
            if output_amm_allocated and (current_y_tile, current_htile, 0) in output_amm_allocated:
                for current_xslice_num in range(num_slices):
                    output_tensor_name = current_node_output_tensor.get_amm_tensor_name(current_blob_idx, current_y_tile, current_htile, current_xslice_num)
                    if output_tensor_name not in tensors_in_amm:
                        raise ValueError('Need to revisit this. Since i moved tensors_in_amm setting into contigous_mem_all i would expect that its already set')
                    tensors_in_amm[output_tensor_name].is_valid = True
            else: # Need to allocate output amm for all inputs of multiple input op to make sure its contiguous in mem
                next_ops_inputs_ic_per_amm = []
                next_ops_inputs_tensors = []
                # We need to prepare the output in AMM mem ready for the concatenation so we allocate output mem for all inputs of the Concat in contiguous mem
                input_index_sorted_inputs_nodes_params = sorted(inputs_nodes_params, key=lambda x:x[1])
                for input_node_params in input_index_sorted_inputs_nodes_params:
                    input_node = ir.graph.nodes[input_node_params[0]]
                    next_ops_inputs_tensors.append(input_node['frontend']['output_tensor'])
                    current_input_node_output_channels = input_node['frontend']['output_tensor'].get_folded_shape()[1]
                    
                    next_ops_inputs_ic_per_amm.append(current_input_node_output_channels )
                
                if DEBUG_PRINT_AMM_ALLOCATION:
                    print('Allocating %s for output tensor of %s, we allocated 2x since it goes to dual input op(e.g. add)' % (str(next_ops_inputs_ic_per_amm),current_node_name))

                # Keep this contiguous output run off any AMM block an Add producer among these inputs
                # still reads as a live operand, or the free-list could hand one x-slice's output a
                # block a sibling x-slice still reads, corrupting it. Reserved for this call only.
                reserved_input_blocks = set()
                for sibling_input_node_params in inputs_nodes_params:
                    sibling_producer = ir.graph.nodes[sibling_input_node_params[0]]
                    if sibling_producer['op_type'] != OpType.ADD:
                        continue
                    sibling_input_even = sibling_producer['backend']['allocated_amm_blocks_for_input_even_grid']
                    if current_y_tile >= len(sibling_input_even) or current_htile >= len(sibling_input_even[current_y_tile]):
                        continue
                    for xslice_inputs in sibling_input_even[current_y_tile][current_htile]:
                        for per_input_amm_blocks in xslice_inputs:
                            if per_input_amm_blocks and per_input_amm_blocks[0]:
                                reserved_input_blocks.update(per_input_amm_blocks[0])

                allocated_blocks,per_input_first_block_index = amms.allocate_contiguous_mem(current_node_name,next_ops_inputs_tensors,
                                                                                            next_ops_inputs_ic_per_amm,tensors_blob_idx=current_blob_idx,tensors_tile_num=current_y_tile,tensors_htile_num=current_htile,
                                                                                            current_tile_num=current_y_tile,current_htile_num=current_htile,num_slices=num_slices,is_valid=False,
                                                                                            reserved_blocks=reserved_input_blocks)
                if len(allocated_blocks[0]) == 0:
                    return ir, False, current_blob_idx

                for node_idx,current_input_node_params in enumerate(inputs_nodes_params): # setting allocations for each of the twin nodes
                    input_idx = current_input_node_params[1]
                    current_input_node = ir.graph.nodes[current_input_node_params[0]]
                    current_input_node_blob_idx = current_input_node['frontend']['tiling_blob_idx']
                    blocks_split=list_of_lists_split_at_pos(allocated_blocks,input_idx,per_index_start_pos=per_input_first_block_index)
                    if (current_input_node['op_type'] == OpType.ADD):
                        following_add_node = ir.graph.nodes[current_input_node['frontend']['following_nodes_params'][0][0]]
                        if (following_add_node['op_type'] == OpType.CONCAT) and node_idx==0: #inputs_nodes_params are sorted by execution order so node_idx==0 means its the far node
                            set_input_alocation_to_following_far_node(ir,current_input_node,blocks_split,blocks_split,current_tile_num=current_y_tile,current_htile_num=current_htile)
                    xslices_allocated_blocks = []
                    for current_xslice_num in range(num_slices):
                        if current_input_node_blob_idx==current_blob_idx: # If the source node is from different blob we dont need to set its output allocation
                            current_input_node['backend']['allocated_amm_blocks_for_output_even_grid'][current_y_tile][current_htile][current_xslice_num] = blocks_split[current_xslice_num*AMM_COUNT:current_xslice_num*AMM_COUNT+AMM_COUNT]
                            current_input_node['backend']['allocated_amm_blocks_for_output_odd_grid'][current_y_tile][current_htile][current_xslice_num] = blocks_split[current_xslice_num*AMM_COUNT:current_xslice_num*AMM_COUNT+AMM_COUNT]
                            if 'output_amm_allocated' in current_input_node['backend']:
                                current_input_node['backend']['output_amm_allocated'].append((current_y_tile,current_htile,current_xslice_num))
                            else:
                                current_input_node['backend']['output_amm_allocated'] = [(current_y_tile, current_htile, current_xslice_num)]
                            # Use THIS input node's own output tensor for the AMM name - the outer
                            # current_node_output_tensor is only right for current_node itself; for the
                            # other pair input it looked up a non-resident name and raised.
                            output_tensor = current_input_node['frontend']['output_tensor']
                            output_tensor_name = output_tensor.get_amm_tensor_name(current_blob_idx,current_y_tile,current_htile,current_xslice_num)
                            if output_tensor_name not in amms.tensors_in_amm:
                                # Seems like this happens becuase we have one of concat inputs is add in same blob. in such case the tensor name we need in amm is the add's input
                                raise ValueError ('Need to revisit this. Since i moved tensors_in_amm setting into contigous_mem_all i would expect that its already set')
                            else:
                                amms.tensors_in_amm[output_tensor_name].is_valid=True
                        xslices_allocated_blocks.append(blocks_split[current_xslice_num*AMM_COUNT:current_xslice_num*AMM_COUNT+AMM_COUNT])
                    current_input_node_following_nodes_params = current_input_node['frontend']['following_nodes_params']
                    set_following_nodes_inputs_allocations(ir,node,current_input_node_following_nodes_params,xslices_allocated_blocks,current_tile_num=current_y_tile,current_htile_num=current_htile)

        else: # In allocation of output tensor in AMM in case of non dual_contiguous_input_allocation_node
            next_op_input_channels_per_amm = current_op_output_channels  # We need to prepare the output in AMM mem ready as the input of next op. So the next op input channels are actually the following op input channels
            allocated_blocks = []
            num_output_xslices = frontend['out_xslices_per_htile'][current_htile]
            for current_x_slice in range(num_output_xslices):
                temp_allocated_blocks = amms.allocate_mem(current_node_name,current_node_output_tensor,next_op_input_channels_per_amm,
                                                tensors_blob_idx=current_blob_idx,tensors_tile_num=current_y_tile,tensors_htile_num=current_htile,current_tile_num=current_y_tile,
                                                current_htile_num=current_htile,current_xslice_num=current_x_slice,is_valid=True) # Is valid means  that in current node we took care of reading the tensor/it is in mem from last op. In this case tensor is in mem from current op
                if (len(temp_allocated_blocks[0]) == 0):
                    return ir, False, current_blob_idx
                allocated_blocks.append(temp_allocated_blocks)

            output_even = backend['allocated_amm_blocks_for_output_even_grid'][current_y_tile][current_htile]
            output_odd = backend['allocated_amm_blocks_for_output_odd_grid'][current_y_tile][current_htile]
            for current_x_slice in range(num_output_xslices):
                output_even[current_x_slice] = allocated_blocks[current_x_slice]
                output_odd[current_x_slice] = allocated_blocks[current_x_slice]
            set_following_nodes_inputs_allocations(ir,node,following_nodes_params,allocated_blocks,current_tile_num=current_y_tile,current_htile_num=current_htile)
            
        
        if current_node_name in outputs_producing_nodes_set:
            mark_blob_outputs_deallocations(ir, amms, node, current_y_tile, current_htile)

        mark_deallocation_for_report(node, 'after_output_allocation', amms, current_y_tile, current_htile)

        next_blob_avoids_ddr_read = False
        if (current_blob_idx!=len(tiling_blobs)-1) and (current_htile == current_blob_num_htiles-1) and (node['op_type'] != OpType.RESIZE):
            next_blob_num_tiles = tiling_blobs[current_blob_idx+1].y_tiles
            # If the next blob starts with a node that avoids DDR-read-while-conv, it immediate-reads
            # its input instead of relying on a pre-read double-buffer. Pre-reading its first tile
            # here would reserve a buffer never consumed that way, so skip the cross-blob pre-read.
            next_blob_first_node_name = tiling_blobs[current_blob_idx+1].nodes_in_blob[0]
            next_blob_avoids_ddr_read = ir.graph.nodes[next_blob_first_node_name]['frontend'].get('avoid_ddr_read_while_conv', False)
        else:
            next_blob_num_tiles = 0

        if current_node_idx == current_tiling_blob.next_tile_read_node_idx: # We are in the node that performs read of next tile
            node['backend']['next_tile_read_node'] = True
            # When DDR-read-while-conv is avoided for this node, reads are serialized against conv,
            # so we do NOT reserve the pre-read double-buffer here - inputs load via the immediate-
            # read path instead. Skipping the reservation frees AMM, the point of this fallback.
            avoid_ddr_read_while_conv = debug_flags.DEBUG_AVOID_DDR_READ_WHILE_CONV or frontend.get('avoid_ddr_read_while_conv', False)
            if not avoid_ddr_read_while_conv:
                if ((current_y_tile!=current_blob_num_tiles-1) and (current_htile == current_blob_num_htiles-1) and current_blob_num_tiles>1) or\
                    ((current_y_tile==current_blob_num_tiles-1) and (current_htile == current_blob_num_htiles-1) and next_blob_num_tiles>1 and not next_blob_avoids_ddr_read): # If current blob has no tiles we dont need to read next tile

                    if not allocate_next_tile_input_mem(ir,node,current_blob_idx,current_y_tile,current_htile,current_blob_num_tiles,
                                                consecutive_2_tile_blobs, prev_blob_lines_in_ddr):
                        return ir, False, current_blob_idx

                # Pre-read next h-tile of the current y-tile while the current h-tile is still executing.
                if (current_blob_num_htiles > 1) and (current_htile < current_blob_num_htiles - 1):
                    if not allocate_next_tile_input_mem(ir,node,current_blob_idx,current_y_tile,current_htile,current_blob_num_tiles,
                                                consecutive_2_tile_blobs, prev_blob_lines_in_ddr,
                                                pre_read_next_htile=True, current_blob_num_htiles=current_blob_num_htiles):
                        return ir, False, current_blob_idx

            if (current_y_tile==current_blob_num_tiles-1) and (current_htile == current_blob_num_htiles-1) and (current_blob_idx==len(tiling_blobs)-1): # If its last tile in last blob we dont need to read/allocate
                # But we need to mark it for adding a custom wait for the last tile write. Normally this wait is done by next tile read but in this case
                # There is no next tile read
                node['backend']['wait_for_last_tile_write_end'] = current_y_tile

    node['backend']['post_op_mem_utilization'] = amms.get_current_amms_utilization()
    return ir, True, None

def mem_allocation_pass(ir: internal_representation.IR):
    amms = ir.amms
    amms.reset_mem()
    clear_previous_allocation_decisions(ir)

    # Allocate system-reserved tensors in DDR and AMM (e.g., zeros in last block of each AMM).
    # This needs to be done after the memory is reset above.
    ir = allocate_system_tensors(ir)

    # CONV output-channel split dups must be registered as consumers of the tensors they actually read.
    # Add-only and keyed on the dup's own inputs: rebuilding a producer's output consumers from
    # following_nodes_params is wrong when the producer's current output is a different tensor
    # (e.g. an input-channel-split partial sum).
    for node_name, node in ir.graph.nodes(data=True):
        if '_split' in node_name and '_split_ic' not in node_name and node.get('op_type') == OpType.CONV:
            for input_name in node.get('inputs', [])[:1]:
                input_tensor = ir.tensors.get(input_name)
                if input_tensor is not None and node_name not in input_tensor.consumers:
                    input_tensor.consumers.append(node_name)

    allocation_succeeded = True
    tiling_blobs=ir.tiling_blobs
    blobs_tqdm_iterator = tqdm(tiling_blobs.values())

     # Keep track of the number of consecutive 2 tile blobs
    consecutive_2_tile_blobs = 0

    # Each tiling blob will be updated to store this
    # Then at the end of the loop this will be updated based on the node
    prev_blob_lines_in_ddr = -1 # -1 means it's the input

    for current_blob_idx,current_tiling_blob in enumerate(blobs_tqdm_iterator):

        if current_tiling_blob.y_tiles == 2:
            consecutive_2_tile_blobs += 1
        else:
            consecutive_2_tile_blobs = 0

        blobs_tqdm_iterator.set_description('Mem allocation, at blob#%d:' % current_blob_idx, refresh=False)
        current_blob_num_tiles = current_tiling_blob.y_tiles
        tiles_tqdm_iterator = tqdm(range(current_blob_num_tiles))
        nodes_tqdm_iterator = tqdm(current_tiling_blob.nodes_in_blob)
        hztiles_tqdm_iterator = tqdm(range(current_tiling_blob.h_tiles))
        outputs_producing_nodes_set = {t.producer for t in current_tiling_blob.outputs}
        graph_nodes = ir.graph.nodes()

        # evict_amm_except_system_before_alloc is honored only on a blob's FIRST node, but split/
        # identity-inheritance paths stamp it on every candidate to survive re-blobbing, leaving it
        # on non-first nodes too. Clear it there now that blob structure is final.
        for stale_node_name in current_tiling_blob.nodes_in_blob[1:]:
            graph_nodes[stale_node_name]['frontend'].pop('evict_amm_except_system_before_alloc', None)

        # Pre-alloc eviction: give the blob's FIRST node a clean AMM by freeing every other resident
        # tensor once, at blob entry. Keyed on the first node's flag regardless of blob size - a
        # stale len==1-only guard would miss the identity-inserted 2-node case.
        if graph_nodes[current_tiling_blob.nodes_in_blob[0]]['frontend'].get('evict_amm_except_system_before_alloc', False):
            amms.evict_all_except_system_tensors()

        # An isolated (single-node) split blob that evicts also evicts PER H-TILE, since it re-reads
        # inputs and blocking-writes output each h-tile. Gated to single-node blobs so a later
        # node's input in the same blob is never freed out from under it.
        evict_per_htile = ((len(current_tiling_blob.nodes_in_blob) == 1 or
                            all(graph_nodes[n]['op_type'] == OpType.RESIZE for n in current_tiling_blob.nodes_in_blob)) and
                           graph_nodes[current_tiling_blob.nodes_in_blob[0]]['frontend'].get(
                               'evict_amm_except_system_before_alloc', False))
        for current_y_tile in tiles_tqdm_iterator:
            tiles_tqdm_iterator.set_description('Tile#%d:' % current_y_tile, refresh=False)
            for current_hz_tile in hztiles_tqdm_iterator:
                hztiles_tqdm_iterator.set_description('HZ Tile#%d:' % current_hz_tile, refresh=False)
                if evict_per_htile:
                    amms.evict_all_except_system_tensors()
                for current_node_idx, current_node_name in enumerate(nodes_tqdm_iterator):
                    amms.check_mem_integrity()
                    nodes_tqdm_iterator.set_description('At node:%s:' % current_node_name, refresh=False)
                    node = graph_nodes[current_node_name]
                    ir, allocation_succeeded, failing_blob_idx = mem_allocation_per_htile(
                        ir, current_node_idx, node, outputs_producing_nodes_set, current_blob_idx,
                        current_tiling_blob, current_y_tile, current_hz_tile, consecutive_2_tile_blobs, prev_blob_lines_in_ddr)
                    if not allocation_succeeded:
                        return ir, False, failing_blob_idx
                    
        # Before moving to the next blob, store the number of lines the previous blob will write to DDR
        # when the next blob will start reading
        prev_blob_lines_in_ddr = current_tiling_blob.num_lines_written_to_ddr_before_next_blob_read

    return ir,allocation_succeeded,None

def _insert_dmw_engine_serialization(ir, sequencer_program, layer_sequence):
    # Serialize the layer's engine commands after any DMW_AMM write that overlaps their AMM output.
    # DMW footprint: output_z * x_wrapping blocks from amm_start_address.
    # Engine footprint: ceil(output channels / 16) blocks from amm_dst_address, checked for every x-slice.
    # The wait/set pair is inserted before the first engine command.
    engine_idx = next((i for i, e in enumerate(layer_sequence) if e.opcode == SequencerOPCode.ENGINE_OPERATION), None)
    if engine_idx is None:
        return layer_sequence
    engine_spans = []
    for e in layer_sequence:
        if e.opcode == SequencerOPCode.ENGINE_OPERATION:
            lo = e.amm_dst_address
            engine_spans.append((lo, lo + math.ceil(e.conv_output_channels / 16)))
    # Walk back from the first engine to the previous engine, collecting DMW writes and noting any DMR.
    preceding = []
    for e in reversed(sequencer_program):
        if e.opcode == SequencerOPCode.ENGINE_OPERATION:
            break
        preceding.append(e)
    preceding.reverse()
    dmw_hits = []
    last_dmr_idx = -1
    for idx, e in enumerate(preceding):
        if e.opcode == SequencerOPCode.DMR_AMM:
            last_dmr_idx = idx
        elif e.opcode == SequencerOPCode.DMW_AMM and e.nlines_to_write > 0:
            dmw_lo = e.amm_start_address
            dmw_hi = dmw_lo + e.output_z * e.x_wrapping
            if any(dmw_lo < eng_hi and eng_lo < dmw_hi for eng_lo, eng_hi in engine_spans):
                dmw_hits.append(idx)
    if not dmw_hits:
        return layer_sequence
    # Add the pair only if there is no DMR between the overlapping DMW and the engine command. The layer's own
    # pre-read DMRs sit in layer_sequence before the first engine, not in sequencer_program, so check both spans.
    last_dmw_pos = max(dmw_hits)
    if last_dmr_idx > last_dmw_pos:
        return layer_sequence
    if any(e.opcode == SequencerOPCode.DMR_AMM for e in layer_sequence[:engine_idx]):
        return layer_sequence
    f3_mask = 2 ** ir.global_ddr_transaction_flag
    return (layer_sequence[:engine_idx]
            + [SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags=0, wait_flags=f3_mask),
               SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags=f3_mask, wait_flags=0)]
            + layer_sequence[engine_idx:])

def create_statistics_atributes(node):
    node['backend']['statistics'] = {
        'wloc_conv_cmd_count': 0,
        'rq_waiting_nops': 0,
        'nops_end_of_split': 0,
        'stall_nops': 0,
        'wloc_all_splits': 0,
        'wloc_pairs': 0
    }

def cbc_generation_pass(ir: internal_representation.IR, debug_output_dir: str) -> internal_representation.IR:
    graph_nodes = ir.graph.nodes()
    tensors = ir.tensors
    is_debug = ir.debug

    tqdm_iterator = tqdm(ir.lexicographical_topological_sorted_graph)
    for node_name in tqdm_iterator:
        node = graph_nodes[node_name]
        tqdm_iterator.set_description('CBC Generation, at layer %s:' % node_name)
        current_op_type = node['op_type']

        if is_resize_y_node(node):
            continue

        create_statistics_atributes(node)
        backend = node['backend']
        frontend = node['frontend']
        attributes = node['attributes']

        # check if this is 1x1 kernel
        if current_op_type in (OpType.CONCAT, OpType.SYNC):
            kernel_type  = 'none'
        else:
            current_op_weights_tensor, _ = internal_representation.get_node_weights_tensor(node)
            if current_op_weights_tensor.data.shape[2:]==(1,1):
                kernel_type  = '1x1'
            elif  current_op_weights_tensor.data.shape[2:]==(3,3):
                kernel_type  = '3x3'
            else:
                kernel_type  = 'unknow'

        if current_op_type in (OpType.CONCAT, OpType.SYNC):
            oc_processing_order = np.arange(backend['output_channels'])
            backend['simulator_oc_order'] = oc_processing_order
            backend['oc_order'] = oc_processing_order
            backend['grids_cbc'] = CBC_IR(2)

        elif OPTIMIZE_1x1_CONV and (
            (kernel_type  == '1x1') or
            current_op_type in (OpType.ADD, OpType.RESIZE, OpType.IDENTITY)):
            current_op_weights_tensor, _ = internal_representation.get_node_weights_tensor(node)
            oc_ch_compute_sequence = generate_oc_ch_compute_sequence(node)

            if CBC_OBJECT_GENERATION_METHOD:
                cbc = TSNP_CBC_IR()
                generated_wlocs = cbc.tsnp_generate_grids_wloc_cbc(current_op_weights_tensor.data, node, oc_ch_compute_sequence)
                backend['grids_cbc'] = cbc.generate_cbc(node, generated_wlocs)
                backend['grids_cbc'].nlf = cbc.tsnp_generate_nlf(node, ir)
                backend['nlf'] = backend['grids_cbc'].nlf
            else:
                generated_wlocs = generate_grids_wloc_cbc(current_op_weights_tensor.data, node, oc_ch_compute_sequence)
                grids_cbc = generate_grids_1x1_cbc(ir, node, CBC_IR(2), generated_wlocs)
                backend['grids_cbc'] = grids_cbc
                backend['nlf'] = grids_cbc.nlf

            if is_debug:
                export_cbc_to_xls_alex(node, debug_output_dir + node_name + '_cbc', format=DebugFilesFormat.CSV)
                backend['statistics']['wloc_conv_cmd_count'] = len(generated_wlocs[0].cmd_list)
                backend['statistics']['nops_AMM_SYNC'] = 2 * backend['output_channels']

        elif (current_op_type in GRID_OPS) and (kernel_type  == '3x3'):
            current_op_weights_tensor, _ = internal_representation.get_node_weights_tensor(node)

            if current_op_weights_tensor.data is None:
                raise ValueError('Conv node without weights. Node name: %s' % node_name)

            # Compatibility section
            grids_cbc = CBC_IR(2)
            output_channels = backend['output_channels']
            non_empty_output_channels = {oc: True for oc in range(output_channels)}

            backend['non_empty_output_channels'] = non_empty_output_channels
            backend['total_minimal_oc_clocks_inserted_nops'] = 0
            backend['total_first_wloc_entries'] = 0
            backend['per_grid_macs'] = [[[]]]
            grids_cbc.per_oc_non_empty_ic_groups = {key: [0, 0] for key in non_empty_output_channels}

            oc_ch_compute_sequence = generate_oc_ch_compute_sequence(node)
            generated_wlocs = generate_grids_wloc_cbc(current_op_weights_tensor.data, node, oc_ch_compute_sequence)

            statistics = backend['statistics']
            statistics['wloc_conv_cmd_count'] = len(generated_wlocs[0].cmd_list)
            statistics['nops_AMM_SYNC'] = 2 * output_channels
            
            wloc0 = generated_wlocs[0]
            wloc1 = generated_wlocs[1]
            max_mem_for_wloc_split = (wloc0.mem_output_depth * wloc0.mem_output_width) // 2 - 3 * 2700
            split_nr = 0

            # the first cmd in wloc have be LONG
            wloc0.cmd_list[0].is_always_long = True
            wloc1.cmd_list[0].is_always_long = True

            wloc0.optimise_wloc_with_shorts()
            wloc1.optimise_wloc_with_shorts()
            check_nop_balance(generated_wlocs)

            if 'optimal_offload_point_for_tensor' in backend or backend.get('add_stall_nops'):
                add_stalls([wloc0, wloc1], node)
                statistics['stall_nops'] = len(wloc0.cmd_list) - statistics['wloc_conv_cmd_count']

            grids_cbc.alex_wlocs.append([])
            alex_wlocs_0 = grids_cbc.alex_wlocs[0]

            while wloc0.cmd_list and wloc1.cmd_list:
                cmd_inx_split_0, _ = wloc0.find_EOC_to_split(available_mem=max_mem_for_wloc_split)
                cmd_inx_split_1, _ = wloc1.find_EOC_to_split(available_mem=max_mem_for_wloc_split)
                cmd_inx_split = min(cmd_inx_split_0, cmd_inx_split_1)
                split_end = cmd_inx_split + 1

                splited_wlocs = [wloc0.cmd_list[:split_end], wloc1.cmd_list[:split_end]]
                wlocs_part, rq_part, rt_part = generate_grids_rq_cbc_alex2(grids_wloc_commands=splited_wlocs, node=node)

                wloc0.cmd_list = wloc0.cmd_list[split_end:]
                wloc1.cmd_list = wloc1.cmd_list[split_end:]

                wlocs_part[0][0].create_cmd_mem()
                wlocs_part[0][1].create_cmd_mem()
                alex_wlocs_0.append(wlocs_part[0])

                rq_part[0][0].create_cmd_mem()
                rt_part[0].create_cmd_mem()
                grids_cbc.alex_rqParam.append(rq_part[0])
                grids_cbc.RTable.append(rt_part[0])

                statistics['wloc_all_splits'] += len(wlocs_part[0][0].cmd_list)
                split_nr += 1

            if 'lut' in frontend:
                grids_cbc.nlf = [NonLinearFunctionList(frontend['lut'])]
            else:
                grids_cbc.nlf = [LinearFunctionList()]

            grids_cbc.nlf[0].create_cmd_mem()

            backend['grids_cbc'] = grids_cbc
            backend['nlf'] = grids_cbc.nlf
            statistics['nops_end_of_split'] = split_nr * HARDWARE_SYNC_CMD_COMPLATE

            if is_debug:
                export_cbc_to_xls_alex(node, debug_output_dir + node_name + '_cbc', format=DebugFilesFormat.CSV)
        else:
            raise ValueError('Unknown kernel size')
        
        # Check for AMM+STALL together
        if current_op_type in GRID_OPS:
            def is_stall_nop(item):
                return not item.long_entry and item.weight_value == 0 and item.weight_offset == 0

            for split_wloc in backend['grids_cbc'].alex_wlocs[0]:
                items = split_wloc[0].cmd_list
                for idx, cmd in enumerate(items):
                    if "AMM" in cmd.nop_reason:
                        for offset in (-2, -1, 1, 2):
                            check_idx = idx + offset
                            if 0 <= check_idx < len(items) and is_stall_nop(items[check_idx]):
                                raise ValueError('Found AMM and STALL NOPs together in node %s' % node_name)

    return ir

def prepare_intermediate_tensors_write(ir: internal_representation.IR) -> internal_representation.IR:
# Marks optimal places for writing offloaded intermediate tensors to DDR (only after their
# producing op finishes) - ideally in parallel with the node that follows the producer.

    sorted_graph_index = {n: i for i, n in enumerate(ir.lexicographical_topological_sorted_graph)}
    for intermediate_tensor_name,intermediate_tensor_ir in ir.intermediate_ddr_tensors.items():
        offloading_node = intermediate_tensor_ir.offloading_node_name
        intermediate_tensor = ir.tensors[intermediate_tensor_name]
        intermediate_tensor_producer_name = intermediate_tensor.producer
        producer_node = ir.graph.nodes()[intermediate_tensor_producer_name]
        if 'tensor_to_offload_name' in producer_node['backend']:
            raise ValueError('Didnt expect node to already have this attribute. Please check!')
        else:
            producer_node['backend']['tensor_to_offload_name'] = intermediate_tensor_name
        producer_index = sorted_graph_index[intermediate_tensor_producer_name]
        write_node_index = producer_index+1
        write_node_name = ir.lexicographical_topological_sorted_graph[write_node_index]
        write_node = ir.graph.nodes()[write_node_name]
        if 'optimal_offload_point_for_tensor' in write_node['backend']:
            raise ValueError('Didnt expect node to already have optimal_offload_point_for_tensor. Please check!')
        else:
            write_node['backend']['optimal_offload_point_for_tensor'] = intermediate_tensor_name
    return ir

def compile_program(ir: internal_representation.IR) -> internal_representation.IR:
    sequencer_program = ir.sequencer_program.commands_list
    wait_flags_allocator = ir.wait_flags_allocator
    single_split_tables_buffer_allocator = ir.single_split_tables_buffer_allocator
    splitted_tables_buffer_allocator = ir.splitted_tables_buffer_allocator
    #TODO: Dans Must make sure the below loop is going through nodes by execution order
    last_single_split_tables_buffer_id = None
    # Allocate all flags
    axi0_tables_load_flag = wait_flags_allocator.allocate_flag(description='generic axi0 set/wait flag.')
    axi1_tables_load_flag = wait_flags_allocator.allocate_flag(description='generic axi1 set/wait flag.')
    local_ddr_transaction_flag = wait_flags_allocator.allocate_flag(description='generic local ddr transaction (read/write) set/wait flag.')
    global_ddr_transaction_flag = wait_flags_allocator.allocate_flag(description='generic global ddr transaction (read/write) set/wait flag.')
    #engine_command_set_flag = wait_flags_allocator.allocate_flag(description='generic engine command set/waitflag.')
    ir.axi0_tables_load_flag = axi0_tables_load_flag
    ir.axi1_tables_load_flag = axi1_tables_load_flag
    ir.local_ddr_transaction_flag = local_ddr_transaction_flag
    ir.global_ddr_transaction_flag = global_ddr_transaction_flag

    tiling_blobs=ir.tiling_blobs
    graph_nodes = ir.graph.nodes
    tile_read_in_process = False
    tile_write_in_process = False
    conv_in_process = False
    blobs_tqdm_iterator = tqdm(tiling_blobs.values())

    # Initial DDR commands before generating for the rest of the program
    layers_sequence = generate_initial_command_sequence(ir)
    sequencer_program.extend(layers_sequence)

    # Keep track of the number of consecutive 2 tile blobs
    consecutive_2_tile_blobs = 0

    # Each tiling blob will be updated to store this
    # Then at the end of the loop this will be updated based on the node
    prev_blob_lines_in_ddr = -1 # -1 means it's the input

    for blob_idx,tiling_blob in enumerate(blobs_tqdm_iterator):

        if (blob_idx in DEBUG_SKIP_BLOB_LIST):
            continue

        if tiling_blob.y_tiles == 2:
            consecutive_2_tile_blobs += 1
        else:
            consecutive_2_tile_blobs = 0

        last_blob = (blob_idx == (len(tiling_blobs)-1))
        blobs_tqdm_iterator.set_description('Sequencer code generation, at blob#%d:' % blob_idx)
        tiles_tqdm_iterator = tqdm(range(tiling_blob.y_tiles))
        for tile_idx,current_y_tile in enumerate(tiles_tqdm_iterator):
            last_tile = (tile_idx == (tiling_blob.y_tiles-1))
            tiles_tqdm_iterator.set_description('Tile#%d:' % current_y_tile)
            hztiles_tqdm_iterator = tqdm(range(tiling_blob.h_tiles))
            for current_hz_tile in hztiles_tqdm_iterator:
                hztiles_tqdm_iterator.set_description('HZ Tile#%d:' % current_hz_tile)
                nodes_tqdm_iterator = tqdm(tiling_blob.nodes_in_blob)
                for node_idx,current_node_name in enumerate(nodes_tqdm_iterator):
                    last_node = (node_idx == (len(tiling_blob.nodes_in_blob)-1))
                    last_tile_of_last_node = (last_blob and last_tile and last_node)
                    nodes_tqdm_iterator.set_description('At node:%s:' % current_node_name)
                    node = graph_nodes[current_node_name]
                    if (node['frontend']['run_op']):
                        current_op_type = node['op_type']
                        if current_op_type == OpType.CONCAT:
                            layers_sequence,tile_read_in_process,tile_write_in_process = generate_layer_command_sequence(ir,node,wait_flags_allocator,0,0,
                                                                                current_tile_num=current_y_tile,current_htile_num=current_hz_tile,
                                                                                tile_read_in_process=tile_read_in_process,tile_write_in_process=tile_write_in_process,
                                                                                consecutive_2_tile_blobs=consecutive_2_tile_blobs,
                                                                                prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
                                                                                conv_in_process=conv_in_process) # Concat generates commands only if one of its inputs needs to be read from DDR
                            sequencer_program.extend(layers_sequence)
                        elif (current_op_type == OpType.SYNC):
                            pass
                        elif current_op_type in GRID_OPS:
                            single_split_tables_buffer_id = single_split_tables_buffer_allocator.allocate_buffer()
                            if not last_single_split_tables_buffer_id==None:
                                single_split_tables_buffer_allocator.deallocate_buffer(last_single_split_tables_buffer_id)
                            last_single_split_tables_buffer_id = single_split_tables_buffer_id
                            layer_sequence, tile_read_in_process, tile_write_in_process = generate_layer_command_sequence(ir,node,wait_flags_allocator,single_split_tables_buffer_id,splitted_tables_buffer_allocator,
                                                                                    current_tile_num=current_y_tile,current_htile_num=current_hz_tile,
                                                                                    tile_read_in_process=tile_read_in_process,
                                                                                    consecutive_2_tile_blobs=consecutive_2_tile_blobs,
                                                                                    prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
                                                                                    tile_write_in_process=tile_write_in_process,conv_in_process=conv_in_process) # Add current layer's sequencer commands to program list
                            layer_sequence = _insert_dmw_engine_serialization(ir, sequencer_program, layer_sequence)
                            sequencer_program.extend(layer_sequence)

        # Before moving to the next blob, store the number of lines the previous blob will write to DDR
        # when the next blob will start reading
        prev_blob_lines_in_ddr = tiling_blob.num_lines_written_to_ddr_before_next_blob_read

    if DEBUG_PUSH_TABLES_READ_BEFORE_DDR_RW:
        ir.sequencer_program.push_tables_read_before_ddr_rw()

    # F4 is set on every DMW finishing a to-MXP tensor write, but MXP expects only one F4 per
    # NX->MXP->NX sync group. Group by mxp_sync_root and keep F4 only on the last DMW in
    # generated order within each group (execution order, not split-name suffix).
    if ir.sync_with_MXP and ir.tensors_to_mxp:
        F4_FLAG = 16
        f4_commands = [cmd for cmd in ir.sequencer_program.commands_list
                       if cmd.opcode == SequencerOPCode.DMW_AMM and (cmd.set_flags & F4_FLAG)]
        assert len(f4_commands) >= 1, \
            'Expected at least one F4-setting DMW for NX->MXP->NX sync tensors %s' % str(ir.tensors_to_mxp)
        f4_commands_by_root = {}
        for cmd in f4_commands:
            f4_commands_by_root.setdefault(cmd.mxp_sync_root, []).append(cmd)
        for root_cmds in f4_commands_by_root.values():
            for cmd in root_cmds[:-1]:
                cmd.set_flags &= ~F4_FLAG
        remaining_f4_commands = [cmd for cmd in ir.sequencer_program.commands_list
                                  if cmd.opcode == SequencerOPCode.DMW_AMM and (cmd.set_flags & F4_FLAG)]
        assert len(remaining_f4_commands) == len(f4_commands_by_root), \
            'Expected exactly one F4-setting DMW per MXP sync group after dedup, got %d for %d groups' % (
                len(remaining_f4_commands), len(f4_commands_by_root))


    wait_flags_allocator.deallocate_flag(ir.axi0_tables_load_flag)
    wait_flags_allocator.deallocate_flag(ir.axi1_tables_load_flag)
    wait_flags_allocator.deallocate_flag(ir.local_ddr_transaction_flag)
    wait_flags_allocator.deallocate_flag(ir.global_ddr_transaction_flag)
    return ir

def get_last_compile_per_layer_wloc_size(ir,debug_output_dir):
    last_compile_per_layer_wloc_size_filename = debug_output_dir+'past_wloc_data.pickle'
    if os.path.exists(last_compile_per_layer_wloc_size_filename):
        with open(last_compile_per_layer_wloc_size_filename, 'rb') as f:
            past_wloc_data=pickle.load(f)
        ir.past_wloc_data = past_wloc_data
    return ir

def save_per_layer_wloc_size(ir,debug_output_dir):
    past_balanced_wloc_entries_limit_dict={}
    past_wloc_sizes_dict = {}
    last_compile_per_layer_wloc_size_filename = debug_output_dir+'past_wloc_data.pickle'
    for node_name in ir.lexicographical_topological_sorted_graph:
        current_node = ir.graph.nodes[node_name]
        if 'per_split_wloc_size' in current_node['backend']:
            past_wloc_sizes_dict[node_name] = current_node['backend']['per_split_wloc_size']
        if 'balanced_wloc_entries_limit' in current_node['backend']:
            past_balanced_wloc_entries_limit_dict[node_name] = current_node['backend']['balanced_wloc_entries_limit']
    with open(last_compile_per_layer_wloc_size_filename, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        past_wloc_sizes_data={}
        past_wloc_sizes_data['past_compile_wloc_sizes_dict'] = past_wloc_sizes_dict
        past_wloc_sizes_data['past_balanced_wloc_entries_limit_dict'] = past_balanced_wloc_entries_limit_dict
        pickle.dump(past_wloc_sizes_data, f, pickle.HIGHEST_PROTOCOL)

def get_best_node_index_to_split_blob(ir:internal_representation.IR,sorted_nodes,node_index,nodes_in_blob):
    # The below will minimize the number of outputs from the blob
    nodes_in_blob_copy = list(nodes_in_blob)
    min_blob_outputs=100
    next_k3_found = False
    best_idx=0
    current_node_idx=node_index
    while not next_k3_found:
        blob_inputs,blob_outputs = get_blob_inputs_and_outputs(ir,nodes_in_blob_copy)
        # A blob may end in a CONCAT only if that CONCAT is a scatter concat and the blob has at least 2 nodes.
        last_node = ir.graph.nodes()[nodes_in_blob_copy[-1]] if nodes_in_blob_copy else None
        ends_in_concat = last_node is not None and last_node['op_type'] == OpType.CONCAT
        cut_not_allowed = ends_in_concat and not (
            last_node['frontend'].get('is_scatter_concat') and len(nodes_in_blob_copy) >= 2)
        if len(blob_outputs)<=min_blob_outputs and not cut_not_allowed:
            min_blob_outputs = len(blob_outputs)
            best_idx = current_node_idx
        current_node_idx+=1
        if current_node_idx>=len(sorted_nodes):
            next_k3_found = True
            break
        next_node_name = sorted_nodes[current_node_idx]
        next_node = ir.graph.nodes()[next_node_name]
        nodes_in_blob_copy.append(next_node_name)
        next_k3_found = ir.is_k3_node(next_node)
    return best_idx

def _is_following_of_type(ir, node, search_type) -> bool:
    following_nodes_params = node['frontend']['following_nodes_params']
    is_following_of_type = False
    for following_node_params in following_nodes_params:
        following_node_name = following_node_params[0]
        following_node = ir.graph.nodes()[following_node_name]
        if following_node['op_type']==search_type:
            is_following_of_type = True
            break
    return is_following_of_type

def node_input_is_from_mxp(ir, node_name) -> bool:
    node = ir.graph.nodes()[node_name]

    for input_tensor in get_node_input_tensors(node):
        if input_tensor.producer != None:
            producer_node = ir.graph.nodes()[input_tensor.producer]
            if producer_node['op_type'] == OpType.SYNC:
                return True
    return False

def should_start_new_blob(
    ir: internal_representation.IR,
    node_name: str,
    current_blob: internal_representation.TilingBlob,
) -> bool:

    start_new_blob = False
    graph_nodes = ir.graph.nodes
    node = graph_nodes[node_name]

    # When a node is split due to having too many output channels, for now
    # make a new blob between the two.
    if node_name.endswith('_split'):
        return True

    # A scatter concat needs no compute, so it is kept in a blob by itself: it starts its own blob, and
    # nothing joins it afterwards.
    if node['op_type'] == OpType.CONCAT and node['frontend'].get('is_scatter_concat', False):
        return True
    if current_blob is not None and any(
            graph_nodes[n]['op_type'] == OpType.CONCAT and graph_nodes[n]['frontend'].get('is_scatter_concat', False)
            for n in current_blob.nodes_in_blob):
        return True

    # Only a scatter concat may stand alone in its own blob; a regular concat stays with the nodes it joins, so its
    # blob has more than one node. (Scatter concats are already forced into their own blob above.)

    # Input-channel split nodes each start their own blob (clustering them mishandles Add
    # inline/deallocation bookkeeping, an AMM leak). `force_isolate_blob` marks nodes that
    # must likewise stand alone (e.g. halves of an output-split Add).
    if '_split_ic' in node_name or node['frontend'].get('force_isolate_blob', False):
        return True

    # force_isolate_blob only forces a boundary before the node; force one on the way out too
    # (same for '_split_ic'), or the next node could join the isolated node's blob and read its
    # output before it's fully written.
    if current_blob is not None and any(
            graph_nodes[n]['frontend'].get('force_isolate_blob', False) or '_split_ic' in n
            for n in current_blob.nodes_in_blob):
        return True

    # A resize_y node MUST stay in the SAME blob as the preceding resize_x it aliases (its
    # allocation path aliases resize_x's in-AMM output blocks); splitting across blobs causes a
    # KeyError in mark_blob_outputs_dealloc. Co-blobbing adds no AMM. Overrides heuristics below.
    if is_resize_y_node(node):
        for prec_name, _ in node['frontend'].get('preceding_nodes_params', []):
            if is_resize_x_node(graph_nodes[prec_name]):
                return False

    # Limit a blob to at most one resize op (its x+y pair). Two resize pairs in one blob can end up
    # with a tile count computed from one resize's pre-resize height while the blob's actual output
    # height needs more. Force resize pairs into their own blob so y_tiles always matches.
    if (is_resize_x_node(node) or is_resize_y_node(node)) and current_blob is not None and any(
            is_resize_x_node(graph_nodes[n]) or is_resize_y_node(graph_nodes[n])
            for n in current_blob.nodes_in_blob):
        return True

    # Start new blob for multi-producer ADD nodes (multiple non-ADD producers feeding same ADD).
    # This prevents cross-y-tile block reuse and deallocation timing issues that cause overlaps.
    if node['op_type'] == OpType.ADD:
        input_tensors = node['frontend'].get('input_tensors', [])
        producers_set = set()
        for input_tensor in input_tensors:
            producer = input_tensor.producer
            if producer:
                producers_set.add(producer)

        # Count non-ADD producers in the current blob
        non_add_producers_in_blob = []
        for producer_name in producers_set:
            producer_node = graph_nodes.get(producer_name)
            if producer_node and producer_node['op_type'] != OpType.ADD:
                if current_blob is None or producer_name in current_blob.nodes_in_blob:
                    non_add_producers_in_blob.append(producer_name)

        # If we have multiple non-ADD producers in same blob, start a new blob for the ADD
        if len(non_add_producers_in_blob) > 1:
            return True

        # An ADD fed by an ADD already in the current blob starts a new blob when its other input comes
        # from a different blob (e.g. SESR's ADD_18 -> ADD_19, with CONV2D_3 in another blob).
        if current_blob is not None:
            chained_add_in_blob = any(
                graph_nodes.get(p) is not None and graph_nodes[p]['op_type'] == OpType.ADD
                and p in current_blob.nodes_in_blob for p in producers_set)
            other_inputs_blobs = {graph_nodes[p]['frontend'].get('tiling_blob_idx') for p in producers_set
                                  if graph_nodes.get(p) is not None and not (
                                      graph_nodes[p]['op_type'] == OpType.ADD and p in current_blob.nodes_in_blob)}
            current_blob_idx = graph_nodes[current_blob.nodes_in_blob[0]]['frontend'].get('tiling_blob_idx')
            if chained_add_in_blob and any(b != current_blob_idx for b in other_inputs_blobs if b is not None):
                return True

    # Automatically start a new blob if this input requires sync with MXP
    if node['op_type'] == OpType.SYNC:
        return True
    if node_input_is_from_mxp(ir, node_name):
        return True

    if DEBUG_TRY_TO_FIX_AUTO_SPLIT:

        # Start new blob at Concat
        if node['op_type']==OpType.CONCAT:

            start_new_blob = True

            # In some cases, we're already splitting for this Concat so don't split again
            DONT_SPLIT_IF_OUTPUTS_FOLDED = True
            DONT_SPLIT_IF_INPUTS_FOLDED = True
            DONT_SPLIT_IF_INPUT_STARTED_THIS_BLOB = True

            if DONT_SPLIT_IF_OUTPUTS_FOLDED:
                # If outputs of the Concat are folded, then don't split here because a split will be needed
                # right after the concat anyway
                has_folded_output = False
                for consumer in node['frontend']['output_tensor'].consumers:
                    consumer_node = graph_nodes[consumer]
                    if any(get_folding_flags(consumer_node)):
                        has_folded_output=True
                        break
                if has_folded_output:
                    start_new_blob = False

            if DONT_SPLIT_IF_INPUTS_FOLDED:
                # If an input of the Concat from same blob is folded, then don't split here because we just split
                # STEFAN TODO: Maybe it's better to check all inputs instead of just find one
                has_folded_input_from_same_blob = False
                for input_tensor in node['frontend']['input_tensors']:
                    producer = input_tensor.producer
                    # Only look at nodes in the current blob
                    if producer not in current_blob.nodes_in_blob:
                        continue
                    # This producer is in the current blob
                    producer_node = graph_nodes[producer]
                    if any(get_folding_flags(producer_node)):
                        has_folded_input_from_same_blob = True
                        break
                if has_folded_input_from_same_blob:
                    start_new_blob = False

            if DONT_SPLIT_IF_INPUT_STARTED_THIS_BLOB:
                # If a Concat input from the current blob is the only node in that blob, don't split
                # here since we just split (e.g. Conv3 would otherwise land alone in its own blob).
                has_input_that_started_this_blob = False
                for input_tensor in node['frontend']['input_tensors']:
                    producer = input_tensor.producer
                    # Only look at nodes in the current blob
                    if producer not in current_blob.nodes_in_blob:
                        continue
                    # This producer is in the current blob
                    # Check if it is the only node in the current blob.
                    if len(current_blob.nodes_in_blob) == 1:
                        has_input_that_started_this_blob = True
                        break
                if has_input_that_started_this_blob:
                    start_new_blob = False

        # Folding and unfolding is done by DDR so need to start a new blob
        if any(get_folding_flags(node)):
            start_new_blob=True

        # Sometimes requant are from very different parts of the graph, so avoid error
        # "2 nodes in same blob cant have different number of tiles."
        if 'requantnode' in node_name:
            # Don't need to start a new blob if input is in the current blob already
            input_tensor = node['frontend']['input_tensor']
            if input_tensor.producer not in current_blob.nodes_in_blob:
                start_new_blob=True
        
        # Start new blob after Resize_y node
        preceding_nodes_params = node['frontend']['preceding_nodes_params']
        is_preceding_node_resize = False
        for preceding_node_params in preceding_nodes_params:
            preceding_node_name = preceding_node_params[0]
            preceding_node = graph_nodes[preceding_node_name]
            if is_resize_y_node(preceding_node):
                is_preceding_node_resize = True
                break
        if is_preceding_node_resize and not start_new_blob:
            start_new_blob = True
        
    if DEBUG_MINIMIZE_Y_FOLDING:
        # If stride = 2, split so that there is not a blob with mismatched # tiles
        if node['op_type']==OpType.CONV and node['frontend']['stride'] == 2:
            start_new_blob=True
        # Similarly if next = Resize, split so there is not a mismatched # tiles
        # Split before the resize so the smaller amount is DMA
        #if _is_following_of_type(ir, node, OpType.RESIZE):
        #    start_new_blob=True
        if node['op_type']==OpType.RESIZE:
            start_new_blob=True

    if (node['op_type'] == OpType.CONV) and (current_blob != None) and not start_new_blob:
        producer = node['frontend']['input_tensor'].producer
        if producer not in current_blob.nodes_in_blob:
            first_node_in_current_blob_name = current_blob.nodes_in_blob[0]
            first_node_in_current_blob = graph_nodes[first_node_in_current_blob_name]
            if first_node_in_current_blob['op_type'] != OpType.CONCAT:
                if (node['frontend']['stride'] == 1) and (first_node_in_current_blob['frontend']['stride'] == 2):
                    start_new_blob=True
            else:
                start_new_blob=True
    
    if (node['op_type'] == OpType.CONV) and (current_blob is not None) and not start_new_blob:
        for consumer in node['frontend']['output_tensor'].consumers:
            if 'ADD' in consumer:
                add_node = graph_nodes[consumer]
                for input_tensor in add_node['frontend']['input_tensors']:
                    producer = input_tensor.producer
                    if producer and ('SPLIT' in producer or 'STRIDEDSLICE' in producer):
                        if any('CONCATENATION' in c for c in input_tensor.consumers):
                            start_new_blob = True
                            break

    return start_new_blob
        

# get_ops_grid_config caches folded input/output channel counts on node['backend'] once, up front;
# a later y-fold reset leaves that stale and read_missing_tensors_from_ddr hard-fails on the
# mismatch, so recompute it the same way whenever a fold is reset.
def recompute_backend_channels_after_fold_reset(ir: internal_representation.IR, node_name, node):
    if 'backend' not in node or node['backend'] is None:
        return
    frontend = node['frontend']
    force_folding_y, force_unfolding_y = get_folding_flags(node)
    current_op_input_tensor_name = node['inputs'][0]
    consumers = ir.tensors[current_op_input_tensor_name].consumers
    graph_nodes_all = ir.graph.nodes()
    consumer_0_attrs = graph_nodes_all[consumers[0]].get('attributes', {}) if consumers else {}
    padded = consumer_0_attrs['pads'][0] if 'pads' in consumer_0_attrs else 0
    current_op_input_tensor_shape = ir.tensors[current_op_input_tensor_name].get_folded_shape(
        folding_conv_y=force_folding_y, unfolding_conv_y=force_unfolding_y, padded=padded)
    current_op_output_tensor_name = node['outputs'][0]
    folding_conv_x = 'force_folding_x' in frontend
    current_op_output_tensor_shape = ir.tensors[current_op_output_tensor_name].get_folded_shape(
        folding_conv_x=folding_conv_x, producing_node_stride=frontend['stride'])
    current_op_input_channels = current_op_input_tensor_shape[1]
    if node['op_type'] in MULTIPLE_INPUT_OPS:
        current_op_input_channels *= 2
    node['backend']['input_channels'] = current_op_input_channels
    node['backend']['output_channels'] = current_op_output_tensor_shape[1]

def get_node_to_split_in_failing_blob(ir, failing_blob_idx):
    if failing_blob_idx is None:
        return None, False

    nodes_in_failing_blob = ir.tiling_blobs[failing_blob_idx].nodes_in_blob
    num_nodes = len(nodes_in_failing_blob)
    if num_nodes < 2:
        print(f"Blob {failing_blob_idx}:")
        print(nodes_in_failing_blob)
        # Split the ACTUAL failing node, not the stripped original: stripping '_split' would
        # re-target the original conv every iteration, reusing generated names and (for input-
        # split Add) building a self-loop.
        node_name = nodes_in_failing_blob[0]
        node = ir.graph.nodes[node_name]
        # Before paying for a channel split, check if this lone trivial-kernel node (1x1 or
        # stride-1 3x3, needs no y-folding) is only carrying a y-fold as dead weight. If so, reset
        # fold to 0, re-derive consumers' fold requirement, and bail without splitting.
        if node['op_type'] == OpType.CONV:
            kernel_size = node['frontend']['kernel_size']
            stride = node['frontend']['stride']
            is_trivial = (kernel_size == 1 or kernel_size == 3) and stride == 1
            has_fold = node['frontend']['input_folding_factor_y'] != 0 or node['frontend']['output_folding_factor_y'] != 0
            if is_trivial and has_fold:
                original_input_height = node['frontend']['input_tensor'].get_original_shape()[2]
                num_y_tiles = get_num_y_tiles(original_input_height, 0, original_kernel_size=1, amm_height=ir.amm_height)
                node['frontend']['input_folding_factor_y'] = 0
                node['frontend']['input_tensor'].folding_factor_y = 0
                node['frontend']['input_tensor'].y_tiles = num_y_tiles
                node['frontend']['output_folding_factor_y'] = 0
                node['frontend']['output_tensor'].folding_factor_y = 0
                node['frontend']['output_tensor'].y_tiles = num_y_tiles
                node['frontend']['y_tiles'] = num_y_tiles
                for flag in ('force_folding_y', 'force_unfolding_y'):
                    if flag in node['frontend']:
                        del node['frontend'][flag]
                if node_name in ir.force_y_folding:
                    ir.force_y_folding.remove(node_name)
                if node_name in ir.force_y_unfolding:
                    ir.force_y_unfolding.remove(node_name)
                calc_conv_qparams(ir, node_name, node)
                recompute_backend_channels_after_fold_reset(ir, node_name, node)
                graph_nodes_all = ir.graph.nodes()
                for consumer_name in list(node['frontend']['output_tensor'].consumers):
                    consumer = graph_nodes_all[consumer_name]
                    if consumer['op_type'] == OpType.CONV:
                        calc_conv_folding_factor(ir, consumer_name, consumer)
                        calc_conv_qparams(ir, consumer_name, consumer)
                        recompute_backend_channels_after_fold_reset(ir, consumer_name, consumer)
                # Give the reset its best shot before falling to a channel split: apply the same
                # pre-split memory discipline as the eviction tier so the fit check uses the smallest footprint.
                node['frontend']['avoid_ddr_read_while_conv'] = True
                node['frontend']['avoid_ddr_write_while_conv'] = True
                node['frontend']['evict_amm_except_system_before_alloc'] = True
                return None, False
        # split_large_conv_nodes makes the WHOLE splitting decision in one call: an N-way output-
        # channel split when input fits double-buffered, else an M-way input-channel split + Add
        # tree (conv only), else it raises. True if split, False for ops it can't split.
        if split_large_conv_nodes(ir, node_name, node):
            return node_name, True
        # Splitting the node itself didn't help (e.g. an Add feeding a same-blob Concat whose
        # dual-allocation needs ALL inputs contiguous). Scatter the Concat's inputs to DDR instead
        # and force a new blob to start there.
        needs_dual_alloc, _, dual_alloc_node = following_node_needs_dual_allocation(ir, node, failing_blob_idx)
        if needs_dual_alloc and dual_alloc_node['op_type'] == OpType.CONCAT and scatter_concat_inputs(ir, dual_alloc_node):
            return dual_alloc_node['name'], False
        raise ValueError("Cannot further split blob")

    graph_nodes = ir.graph.nodes()
    # A resize_y node can never be a valid split point when it directly follows its resize_x -
    # set_tiling_blobs permanently vetoes that boundary, so picking it here would loop forever.
    def _is_invalid_split_point(idx):
        node = graph_nodes[nodes_in_failing_blob[idx]]
        if not is_resize_y_node(node):
            return False
        return any(is_resize_x_node(graph_nodes[p]) for p, _ in node['frontend'].get('preceding_nodes_params', []))

    split_location = next(
        (idx for idx in range(1, num_nodes)
         if graph_nodes[nodes_in_failing_blob[idx]]['op_type'] == OpType.CONCAT and not _is_invalid_split_point(idx)),
        None
    )
    if split_location is None or split_location >= num_nodes - 1:
        split_location = (num_nodes + 1) // 2
        while split_location < num_nodes - 1 and _is_invalid_split_point(split_location):
            split_location += 1
        if _is_invalid_split_point(split_location):
            # Only invalid indices remain going forward - fall back to searching backward from the
            # midpoint instead.
            split_location = (num_nodes + 1) // 2
            while split_location > 1 and _is_invalid_split_point(split_location):
                split_location -= 1

    return nodes_in_failing_blob[split_location], False

def set_tiling_blobs(ir: internal_representation.IR, failing_blob_idx = None) -> internal_representation.IR:

    # See if there was a previous failing blob, and if so, determine
    # which node to split the blob at.
    node_to_split_in_failing_blob, needed_node_output_split = get_node_to_split_in_failing_blob(ir, failing_blob_idx)
    if needed_node_output_split:
        ir.nodes_to_split_failing_blobs = set()
        return ir, needed_node_output_split
    if node_to_split_in_failing_blob is not None and node_to_split_in_failing_blob not in ir.nodes_to_split_failing_blobs:
        ir.nodes_to_split_failing_blobs.add(node_to_split_in_failing_blob)
    
    sorted_nodes = ir.lexicographical_topological_sorted_graph
    graph_nodes = ir.graph.nodes
    blob_idx=0
    first_node_in_blob = True
    k3_nodes_folded = 0
    current_blob = None
    # Rebuild blobs from scratch: stale higher-index blobs from a previous (longer) build must be
    # dropped first, or a rebuild yielding fewer blobs leaves duplicate trailing blobs that get
    # reprocessed (e.g. inserting the same identity twice and asserting on the duplicate name).
    ir.tiling_blobs.clear()
    print("Starting to split blobs")
    previous_y_tiles = 0
    for node_index in range(len(sorted_nodes)):
        node_name=sorted_nodes[node_index]
        start_new_blob=False
        if node_index in DEBUG_PER_NODE_BLOB or (node_name in ir.marked_nodes_for_folding_y \
            or node_name in ir.force_y_folding or node_name in ir.force_y_unfolding):
            start_new_blob=True

        # Check additional rules for starting new blob
        if should_start_new_blob(ir, node_name, current_blob):
            start_new_blob=True

        # If this was a previously failing blob, check if this is the node to split at
        if node_name in ir.nodes_to_split_failing_blobs:
            start_new_blob=True

        node = graph_nodes[node_name]
        y_tiles = node['frontend']['y_tiles']
        if y_tiles != previous_y_tiles:
            start_new_blob=True

        # A resize_y MUST stay in the same blob as its preceding resize_x (its allocation reuses
        # resize_x's in-AMM output blocks; cross-blob -> KeyError). Override any earlier heuristic
        # that would split them apart; splitting off an earlier producer is still fine.
        if is_resize_y_node(node):
            for _prec_name, _ in node['frontend'].get('preceding_nodes_params', []):
                if is_resize_x_node(graph_nodes[_prec_name]):
                    start_new_blob = False
                    break

        if start_new_blob:
            if not first_node_in_blob: # If its already a first node in blob no need to advance blob idx
                blob_idx+=1
                first_node_in_blob = True
        
        x_slices = node['frontend']['x_slices']
        
        y_size_folded = node['frontend']['output_tensor'].shape[2] / (2 ** node['frontend']['output_folding_factor_y'])
        #Calculate the maximum number of K3 nodes that would still not increase the overall number of tiles in this BLOB
        if blob_idx == 0 and DEBUG_LARGE_FIRST_BLOB:
            max_k3_nodes = MAX_FOLDED_K3_NODES_IN_BLOB # Allow first blob to be larger
        elif y_size_folded <= ir.amm_height:
            max_k3_nodes = 10 #This is a single tile and we have no real K3 limit
        elif y_size_folded <= (ir.amm_height-1)*2:
            max_k3_nodes = ((ir.amm_height-1)*2-y_size_folded) // 2 + 1
        else:
            #min_tiles is the minimal number of tiles above 2 if we only run a single K3 node
            min_tiles = math.ceil((y_size_folded - (ir.amm_height-1)*2) / (ir.amm_height-2))
            extra_k3 = min_tiles * (ir.amm_height-2) + (ir.amm_height-1)*2 - y_size_folded
            max_k3_nodes = extra_k3 // (2*min_tiles+2) + 1
        # Temp improvement
        # if y_size in [36, 72]:
        #    max_k3_nodes = 1
        # else:
        #    max_k3_nodes = MAX_FOLDED_K3_NODES_IN_BLOB

        is_k3_node = ir.is_k3_node(node)
        input_folding_y = 2**node['frontend']['input_folding_factor_y']
        kernel_size = 1
        if is_k3_node:
            kernel_size = node['frontend']['kernel_size']
        kernel_field_reduction = kernel_size // 2
        if is_k3_node and not first_node_in_blob and (k3_nodes_folded+1/input_folding_y) > max_k3_nodes \
            and not (y_tiles == 1 and DEBUG_NO_K3_LIMIT_FOR_1T):
            blob_idx+=1
            first_node_in_blob=True
        node['frontend']['tiling_blob_idx'] = blob_idx
        h_tiles = node['frontend']['h_tiles']
        xslices_per_htile = node['frontend']['xslices_per_htile'].copy()
        if first_node_in_blob:
            k3_nodes_folded = is_k3_node/input_folding_y * kernel_field_reduction
            current_blob = internal_representation.TilingBlob(nodes_in_blob=[node_name],num_y_tiles=y_tiles,num_h_tiles=h_tiles,num_x_slices=x_slices,xslices_per_htile=xslices_per_htile,k3_nodes=int(is_k3_node))
            ir.tiling_blobs[blob_idx] = current_blob
        else:
            current_blob.nodes_in_blob.append(node_name)
            current_blob.num_of_nodes_in_blob+=1
            if is_k3_node:
                k3_nodes_folded += 1/input_folding_y * kernel_field_reduction
                current_blob.k3_nodes = math.ceil(k3_nodes_folded)
            if (current_blob.h_tiles < h_tiles):
                current_blob.h_tiles = h_tiles
            for htile_idx in range(h_tiles):
                if (htile_idx > len(current_blob.xslices_per_htile) - 1):
                    current_blob.xslices_per_htile.append(xslices_per_htile[htile_idx])
                else:
                    current_blob.xslices_per_htile[htile_idx] = max(current_blob.xslices_per_htile[htile_idx], xslices_per_htile[htile_idx])

        # Up to this point is where we forced a new BLOB before the node.
        # Now we check if we need to force a new blob after the node
        if (k3_nodes_folded+1/input_folding_y)>max_k3_nodes and not first_node_in_blob \
              and not (y_tiles == 1 and DEBUG_NO_K3_LIMIT_FOR_1T) and (node['op_type'] != OpType.CONCAT):
                if node_index == get_best_node_index_to_split_blob(ir,sorted_nodes,node_index,current_blob.nodes_in_blob):
                    blob_idx+=1 # We set each blob to have single node
                    first_node_in_blob = True
                else:
                    first_node_in_blob = False
        elif node['op_type']==OpType.ADD and _is_following_of_type(ir, node, OpType.CONCAT): 
            blob_idx+=1 # We set each blob to have single node
            first_node_in_blob = True
            # Because of mem alloc issue we need to fix we dont allow concat after add in same blob
        
        
            # STEFAN TODO: assumes the next node is the Concat, but a parallel Conv path between Add
            # and Concat can end up as its own single-node blob if Concat also starts a new blob.
            # For now implementing option #3 in should_start_new_blob (keep Conv2+Concat3 together).
        else:
            first_node_in_blob = False

        previous_y_tiles = y_tiles

    # A blob's h_tiles is merged via max() above, which can inherit a value too high for a
    # force_folding_x node without going through update_htiling_info's growth guard. Sweep every
    # freshly-built blob and clamp+re-propagate via apply_blob_htiling.
    for blob in ir.tiling_blobs.values():
        if not blob.nodes_in_blob:
            continue
        fold_x_slices = [graph_nodes[n]['frontend']['x_slices'] for n in blob.nodes_in_blob
                          if 'force_folding_x' in graph_nodes[n]['frontend']]
        if fold_x_slices and blob.h_tiles >= min(fold_x_slices):
            blob.h_tiles = max(1, min(fold_x_slices) - 1)
        if fold_x_slices:
            # The clamp above bounds h_tiles below needed_xslices, but divisibility/parity can still
            # degenerate a non-last h-tile to a fold-unusable single x-slice; keep shrinking h_tiles
            # until apply_blob_htiling succeeds or fully gives up (h_tiles=1).
            while not apply_blob_htiling(blob, graph_nodes, get_blob_kernel_size(blob)) and blob.h_tiles > 1:
                blob.h_tiles -= 1

    # A CONCAT alone in its own blob pays for a real AMM residency + DMR/DMW cycle just to shuffle
    # inputs into one buffer. Opportunistically apply scatter_concat_inputs here too for single-node
    # CONCAT blobs (self-guarding no-op when ineligible).
    for blob in ir.tiling_blobs.values():
        if not blob.nodes_in_blob:
            continue
        # Single-node CONCAT blobs, and any blob that ends in a CONCAT, have the concat as their last node.
        last_node = graph_nodes[blob.nodes_in_blob[-1]]
        if last_node['op_type'] == OpType.CONCAT:
            scatter_concat_inputs(ir, last_node)

    return ir, needed_node_output_split

def propagate_fold_xslices_to_producers(nodes_in_blob, graph_nodes):
    # A non-folding producer's plain balanced split can desync from a following x-folding node's
    # even-non-last split, causing the sequencer to over-index. Propagate the fold's split to every
    # other pass-through node in the blob (same total in/out).
    for i, node_name in enumerate(nodes_in_blob):
        node = graph_nodes[node_name]
        if not node['frontend'].get('force_folding_x'):
            continue
        fold_split = list(node['frontend']['xslices_per_htile'])
        fold_total = sum(fold_split)
        for j, other_name in enumerate(nodes_in_blob):
            if j == i:
                continue
            other = graph_nodes[other_name]
            pf = other['frontend']
            prev_in = pf.get('xslices_per_htile')
            prev_out = pf.get('out_xslices_per_htile')
            if prev_in is None or prev_out is None:
                continue
            # Only pass-through nodes (same total in and out) - leave unfolding/resize (in != out) alone.
            if sum(prev_in) == fold_total and sum(prev_out) == fold_total \
               and (prev_in != fold_split or prev_out != fold_split):
                pf['xslices_per_htile'] = list(fold_split)
                pf['out_xslices_per_htile'] = list(fold_split)
                pf['output_tensor'].xslices_per_htile = list(fold_split)
                for it in get_node_input_tensors(other):
                    it.xslices_per_htile = list(fold_split)

def apply_blob_htiling(current_blob, graph_nodes, kernel_size):
    # Propagate current_blob.h_tiles onto every node. Returns False if an x-fold node degenerates
    # to a fold-unusable slice count (<2) in a non-last h-tile. Clamps h_tiles < needed_xslices
    # unconditionally, since merging during a rebuild can bypass update_htiling_info's growth guard.
    fold_x_slices = [graph_nodes[n]['frontend']['x_slices'] for n in current_blob.nodes_in_blob
                      if 'force_folding_x' in graph_nodes[n]['frontend']]
    if fold_x_slices:
        max_allowed = min(fold_x_slices) - 1
        if any(x in fold_x_slices for x in (50, 7)):
            max_allowed = min(max_allowed, 2)
        if current_blob.h_tiles > max_allowed:
            current_blob.h_tiles = max(1, max_allowed)
    htiling_ok = True
    no_k3_in_blob = (current_blob.k3_nodes == 0)
    # When the blob's output node is an odd x-fold, reverse the whole blob's h-tile x-slice split
    # instead of dummy-padding it.
    blob_reverse = blob_reverses_xslices(current_blob, graph_nodes)
    for node_name in current_blob.nodes_in_blob:
        node = graph_nodes[node_name]
        frontend = node['frontend']
        frontend['h_tiles'] = current_blob.h_tiles
        folding_x = 'force_folding_x' in frontend

        # Check if this is a special force_folding_x case with odd input/output (for later use)
        preserve_odd_split = False
        following = frontend.get('following_nodes_params')
        consumer = graph_nodes[following[0][0]] if following else None

        # Check if we should preserve odd splits for INPUT xslices (when both nodes have force_folding_x and both input/output are odd)
        preserve_odd_split_input = False
        if current_blob.h_tiles > 1 and folding_x and 'force_folding_x' in frontend:
            # Current node has force_folding_x - check if previous also does and both are odd
            preceding = frontend.get('preceding_nodes_params')
            preceding_node = graph_nodes[preceding[0][0]] if preceding else None
            if preceding_node is not None and 'force_folding_x' in preceding_node.get('frontend', {}):
                # Both current and previous have force_folding_x
                curr_input_odd = frontend['x_slices'] % 2 == 1
                curr_output_odd = frontend['output_tensor'].x_slices % 2 == 1
                if curr_input_odd and curr_output_odd:
                    preserve_odd_split_input = True

        # Check if we should preserve odd splits for OUTPUT xslices (when this node feeds into force_folding_x with odd output)
        preserve_odd_split_output = False
        if current_blob.h_tiles > 1 and folding_x and consumer is not None and 'force_folding_x' in consumer.get('frontend', {}):
            consumer_odd_output = consumer['frontend']['output_tensor'].x_slices % 2 == 1
            if consumer_odd_output:
                preserve_odd_split_output = True

        # For input xslices computation, use the input-specific flag
        preserve_odd_split = preserve_odd_split_input

        if blob_reverse:
            # Reversal strategy: skip the dummy pad (leftover stays a real slice in the first h-tile)
            # and skip the fold node's internal reversal so the blob-level reverse below applies once
            # uniformly to every node.
            input_xslices = frontend['x_slices']
            internal_fold = False
        else:
            input_xslices = get_padded_input_xslices(node, frontend['x_slices'], frontend['h_tiles'], no_k3_nodes_in_blob=no_k3_in_blob)
            internal_fold = folding_x
        frontend['xslices_per_htile'] = get_xslices_per_htile(input_xslices, frontend['h_tiles'], kernel_size,
                                                              is_folding_x=internal_fold, no_k3_nodes_in_blob=no_k3_in_blob,
                                                              preserve_odd_split=preserve_odd_split)
        if is_resize_y_node(node):
            # A y-resize's input IS the preceding x-resize's output (same tensor object), so use
            # the exact split the x-resize produced (its doubled out_xslices) rather than
            # recomputing from the shared total, which would desync the two.
            prec_name = frontend['preceding_nodes_params'][0][0]
            frontend['xslices_per_htile'] = list(graph_nodes[prec_name]['frontend']['out_xslices_per_htile'])
        elif is_x_fold_node(node) and frontend.get('preceding_nodes_params'):
            # Same reasoning for a chained x-fold step: inherit the preceding fold step's
            # out_xslices_per_htile rather than recomputing independently, which can desync.
            prec_name = frontend['preceding_nodes_params'][0][0]
            prec_node = graph_nodes[prec_name]
            # Inherit only when the producer's split has this node's h-tile count; otherwise the inherited list
            # carries the producer's h-tile split and disagrees with h_tiles (e.g. [13,12] under h_tiles=1).
            if is_x_fold_node(prec_node) and 'out_xslices_per_htile' in prec_node['frontend'] \
                    and len(prec_node['frontend']['out_xslices_per_htile']) == frontend['h_tiles']:
                frontend['xslices_per_htile'] = list(prec_node['frontend']['out_xslices_per_htile'])
        if blob_reverse:
            frontend['xslices_per_htile'] = list(reversed(frontend['xslices_per_htile']))
        # A single-node blob that is an x-fold with an even input split: its input must be the reverse of its
        # natural split so it matches the output split (e.g. input [24,26] -> [26,24] for output [13,12]).
        if len(current_blob.nodes_in_blob) == 1 and (frontend['x_slices'] % 2 == 0) and folding_x and current_blob.h_tiles > 1:
            folded_out = [math.ceil(n / 2) for n in frontend['xslices_per_htile']]
            # Reverse only when halving gives one even and one odd output split, and the first h-tile's
            # halved count is even.
            existing_out = frontend.get('out_xslices_per_htile')
            if len({n % 2 for n in folded_out}) == 2 and folded_out[0] % 2 == 0 \
                    and existing_out is not None and folded_out != list(existing_out):
                frontend['xslices_per_htile'] = list(reversed(frontend['xslices_per_htile']))
        for input_tensor in get_node_input_tensors(node):
            input_tensor.xslices_per_htile = frontend['xslices_per_htile']
            # Preserve input-layer x-slice packing (the per-lane combination marker) across h-tiling.
            # The per-h-tile lane layout is re-derived from each h-tile's own slice count, so packing is
            # now valid with h_tiles>1; only reset unpacked (scalar) tensors.
            if frontend['h_tiles'] > 1 and not is_packed_xslices(input_tensor.num_packed_xslices):
                input_tensor.num_packed_xslices = 1
        out_xslices = get_xslices_per_htile(frontend['output_tensor'].x_slices, frontend['h_tiles'], kernel_size,
                                            is_folding_x=internal_fold, no_k3_nodes_in_blob=no_k3_in_blob,
                                            preserve_odd_split=preserve_odd_split_output)
        # An unfolding-x layer (or x-resize) spanning more than one h-tile holds twice as many
        # x-slices per h-tile in output as input, so derive output split as 2x input split rather
        # than an even split of the doubled total, preserving the out/in ratio the sequencer relies on.
        if (frontend.get('force_unfolding_x') or is_resize_x_node(node)) and frontend['h_tiles'] > 1:
            doubled_out_xslices = [2 * n for n in frontend['xslices_per_htile']]
            if sum(doubled_out_xslices) == frontend['output_tensor'].x_slices:
                out_xslices = doubled_out_xslices
            elif frontend.get('force_unfolding_x') and (frontend['output_tensor'].x_slices % 2 != 0):
                # An odd unfolded output total can't be an exact 2x of the input split, so the balanced
                # split doesn't guarantee the dummy slice lands in h-tile 0 (required by
                # sequencer.py's is_dummy_unfolding). Force it: split the even remainder, then add 1.
                even_out_xslices = get_xslices_per_htile(frontend['output_tensor'].x_slices - 1, frontend['h_tiles'],
                                                          kernel_size, is_folding_x=internal_fold, no_k3_nodes_in_blob=no_k3_in_blob)
                even_out_xslices[0] += 1
                out_xslices = even_out_xslices
        if is_resize_y_node(node):
            # A y-resize only rescales on Y; its x-slice layout passes straight through, so the output
            # x-slice split equals the input split (inherited from the x-resize above). Keeping them
            # equal makes the AMM output grid mirror the input 1:1 for the pass-through block copy.
            out_xslices = list(frontend['xslices_per_htile'])
        elif blob_reverse:
            # Match the reversed input split so the blob's per-h-tile boundaries stay aligned end-to-end.
            out_xslices = list(reversed(out_xslices))
        # An x-FOLDING layer emits ceil(n/2) output x-slices per h-tile, but only when the immediate
        # consumer is the real k3 conv (avoids clobbering an earlier fold step's split). Applied
        # last so it uses the post-reverse input split.
        following = frontend.get('following_nodes_params')
        consumer = graph_nodes[following[0][0]] if following else None
        if frontend.get('force_folding_x') and frontend['h_tiles'] > 1 and consumer is not None \
           and consumer['frontend'].get('kernel_size', 1) != 1:
            out_xslices = [math.ceil(n / 2) for n in frontend['xslices_per_htile']]
        # Only apply k3 padding if THIS node is actually k3, not just because the blob has k3 nodes.
        # kernel_size here is the blob's, used for consistent x-slice geometry, but padding is per-node.
        node_kernel_size = frontend.get('kernel_size', kernel_size)
        if node_kernel_size != 1 and folding_x:
            # k3 folding-conv boundary halo (see calc_conv_folding_factor / pad_k3_folding_output_xslices_per_htile):
            # blob h-tiling recomputes out_xslices_per_htile here, which would otherwise clobber that padding.
            # Skipped when the natural split is already [2,1,1]: padding that case adds a spurious
            # extra halo slice this 3-h-tile layout doesn't need.
            out_xslices = pad_k3_folding_output_xslices_per_htile(
                out_xslices, frontend['output_tensor'].x_slices, frontend['h_tiles'])
        frontend['out_xslices_per_htile'] = out_xslices
        frontend['output_tensor'].xslices_per_htile = out_xslices
        # A non-last h-tile below the fold-pair minimum cannot be tiled further. Under the reversal
        # strategy the leftover single slice lives in the FIRST h-tile and is handled by the sequencer,
        # so exclude it from the degeneracy check (test only the interior h-tiles).
        degeneracy_tiles = frontend['xslices_per_htile'][1:-1] if blob_reverse else frontend['xslices_per_htile'][:-1]
        if is_x_fold_node(node) and any(c < 2 for c in degeneracy_tiles):
            htiling_ok = False
    propagate_fold_xslices_to_producers(current_blob.nodes_in_blob, graph_nodes)
    return htiling_ok

# h_tiles above this is considered "large" enough that trading away next-blob DDR pre-read overlap
# (if that reservation is what's forcing it up - see estimate_min_h_tiles/precompute_blob_htiles) is
# worth it in exchange for a smaller, simpler tiling.
LARGE_H_TILES_THRESHOLD = 4

def estimate_min_h_tiles(ir, blob, graph_nodes, next_blob=None):
    # ESTIMATE only (update_htiling_info's h_tiles+=1 ladder is the correctness-bearing fallback for
    # fold parity, k3 halo, etc). If next_blob is given, its DDR pre-read cost is charged against
    # this blob's budget too; call with next_blob=None for the true minimum.
    amm = amm_capacity(ir)
    already_avoiding = any(graph_nodes[n]['frontend'].get('avoid_ddr_read_while_conv', False)
                            for n in blob.nodes_in_blob)
    preread = 0
    if next_blob is not None and next_blob.nodes_in_blob and not already_avoiding:
        next_node = graph_nodes[next_blob.nodes_in_blob[0]]
        # resize_y aliases resize_x's already-resident AMM output blocks (see mem_allocation_per_htile) -
        # it adds ~no AMM footprint of its own, so pre-reading it costs nothing extra to reserve for.
        if not is_resize_y_node(next_node):
            next_fe = next_node['frontend']
            preread = next_fe.get('input_channels', 0) * (2 ** (
                next_fe.get('input_folding_factor_x', 0) + next_fe.get('input_folding_factor_y', 0))
                ) * next_fe.get('x_slices', 0)
    available_amm = amm - preread
    x_slices = blob.x_slices
    blob_output_tensors = blob.outputs
    if x_slices <= 1:
        return 1
    for h_tiles in range(1, x_slices + 1):
        peak = 0
        for node_name in blob.nodes_in_blob:
            node = graph_nodes[node_name]
            fe = node['frontend']
            # Per-h-tile x-slice count is a ceiling-divide BY h_tiles - a missing "// h_tiles" here
            # previously meant footprint never decreased as h_tiles grew, so single-node blobs
            # landed on h_tiles==x_slices instead of the much smaller value that actually fits AMM.
            in_xsl = (math.ceil(fe['x_slices']) + h_tiles - 1) // h_tiles
            out_xsl = (math.ceil(fe['output_tensor'].x_slices) + h_tiles - 1) // h_tiles
            if node['op_type'] in MULTIPLE_INPUT_OPS:
                input_tensors = fe.get('input_tensors', [])
            else:
                it = fe.get('input_tensor')
                input_tensors = [it] if it is not None else []
            num_inputs = len(input_tensors) if input_tensors else 1
            ic = fe.get('input_channels', 0) * (2 ** (fe.get('input_folding_factor_x', 0) + fe.get('input_folding_factor_y', 0)))
            oc = fe.get('output_channels', 0) * (2 ** (fe.get('output_folding_factor_x', 0) + fe.get('output_folding_factor_y', 0)))
            if node['op_type'] in MULTIPLE_INPUT_OPS:
                # Shared with split_large_conv_nodes's generic-op branch (frontend_hardware_dependent.py)
                # so the Add model never drifts out of sync between the two again - see
                # generic_op_footprint's comment for the Add-aliases-input[0] model.
                footprint = generic_op_footprint(node['op_type'], num_inputs, ic * in_xsl, 0)
            else:
                # Total input+output channel footprint, unconditionally double-buffered on the
                # output side (matches generic_op_footprint's single-input-op model; previously
                # single-buffered, which under-counted the requirement).
                footprint = generic_op_footprint(node['op_type'], num_inputs, ic * in_xsl, oc * out_xsl)
            peak = max(peak, footprint)
        if peak <= available_amm:
            return h_tiles
    return x_slices

def precompute_blob_htiles(ir):
    # Give every freshly-built blob a head-start h_tiles estimate instead of discovering the right
    # value one retry at a time. Skips blobs already carrying split/eviction markers, which follow
    # dedicated correctness-sensitive paths that must not be perturbed here.
    graph_nodes = ir.graph.nodes()
    for blob_idx, blob in ir.tiling_blobs.items():
        if not blob.nodes_in_blob:
            continue
        first = graph_nodes[blob.nodes_in_blob[0]]['frontend']
        if first.get('evict_amm_except_system_before_alloc') or first.get('avoid_ddr_read_while_conv') \
           or any(graph_nodes[n]['frontend'].get('force_isolate_blob') for n in blob.nodes_in_blob):
            continue
        # Skip any blob touching x-folding/unfolding: a folding conv needs an EVEN xslice count per
        # non-last h-tile, and this estimator's plain balanced-ceil split can pick a degenerate
        # h_tiles. Leave these to the existing fold-aware ladder instead.
        if any('force_folding_x' in graph_nodes[n]['frontend'] or 'force_unfolding_x' in graph_nodes[n]['frontend']
               for n in blob.nodes_in_blob):
            continue
        next_blob = ir.tiling_blobs.get(blob_idx + 1)
        estimate = estimate_min_h_tiles(ir, blob, graph_nodes, next_blob=next_blob)
        # If the next blob's pre-read reservation is pushing h_tiles up, disabling this blob's own
        # DDR-pre-read-while-conv may let it settle much smaller (same trade as update_htiling_info's reactive fallback).
        if estimate > LARGE_H_TILES_THRESHOLD and next_blob is not None:
            estimate_without_preread = estimate_min_h_tiles(ir, blob, graph_nodes, next_blob=None)
            if estimate_without_preread < estimate:
                for node_name in blob.nodes_in_blob:
                    graph_nodes[node_name]['frontend']['avoid_ddr_read_while_conv'] = True
                estimate = estimate_without_preread
        if estimate > blob.h_tiles:
            kernel_size = get_blob_kernel_size(blob)
            blob.h_tiles = estimate
            apply_blob_htiling(blob, graph_nodes, kernel_size)
    return ir

def was_split_x_folded_producer(graph_nodes, producer_name):
    # True when producer_name is an output-channel-split, x-folded node.
    prod = graph_nodes[producer_name] if producer_name in graph_nodes else None
    return prod is not None and prod['frontend'].get('was_channel_split', False) and prod['frontend'].get('output_folding_factor_x', 0) > 0 and (prod['frontend'].get('output_folding_factor_y', 0) == 0)


def update_htiling_info(ir, failing_blob_idx):
    current_blob = ir.tiling_blobs[failing_blob_idx]
    graph_nodes = ir.graph.nodes()
    # Cheaper lever before growing h_tiles: the problem may be the pre-read double-buffer reserved
    # for the NEXT blob's first tile, not this blob's footprint. Disable DDR-read-while-conv and
    # retry at the same h_tiles instead of growing it (which can desync the k3-halo split).
    already_avoiding = any(graph_nodes[n]['frontend'].get('avoid_ddr_read_while_conv', False)
                            for n in current_blob.nodes_in_blob)
    next_blob = ir.tiling_blobs.get(failing_blob_idx + 1)
    preread_total_input_channels = 0
    if next_blob is not None and next_blob.nodes_in_blob:
        next_node = graph_nodes[next_blob.nodes_in_blob[0]]
        # resize_y aliases resize_x's already-resident AMM output blocks (see mem_allocation_per_htile) -
        # it adds ~no AMM footprint of its own, so there's nothing real to pre-read-reserve for it.
        if not is_resize_y_node(next_node):
            next_fe = next_node['frontend']
            preread_total_input_channels = next_fe.get('input_channels', 0) * (
                2 ** (next_fe.get('input_folding_factor_x', 0) + next_fe.get('input_folding_factor_y', 0))) * next_fe.get('x_slices', 0)
    # Was a hardcoded 1200 (empirically-guessed); replaced with the real AMM capacity for the same
    # "does it fit" yardstick used elsewhere. Relaxes the trigger (2032 vs old 1200) - re-verify sim-correctness.
    if not already_avoiding and preread_total_input_channels > amm_capacity(ir):
        for node_name in current_blob.nodes_in_blob:
            graph_nodes[node_name]['frontend']['avoid_ddr_read_while_conv'] = True
        return ir, True
    # Cheaper still: the preread reservation can be exactly what's keeping this blob from fitting
    # at its CURRENT h_tiles. Check if disabling pre-read lets it fit without growing h_tiles,
    # preferring that over blob-splitting (which loses pre-read overlap permanently).
    if not already_avoiding and preread_total_input_channels > 0:
        fits_without_preread = estimate_min_h_tiles(ir, current_blob, graph_nodes, next_blob=None)
        if fits_without_preread <= current_blob.h_tiles:
            for node_name in current_blob.nodes_in_blob:
                graph_nodes[node_name]['frontend']['avoid_ddr_read_while_conv'] = True
            return ir, True
    new_h_tiles = current_blob.h_tiles + 1
    min_node_xslices = min(graph_nodes[n]['frontend']['x_slices'] for n in current_blob.nodes_in_blob)
    kernel_size = get_blob_kernel_size(current_blob)
    # h_tiles must stay STRICTLY LESS than a force_folding_x node's needed_xslices: at equality
    # every h-tile gets exactly 1 x-slice, not a valid fold split. Once growing h_tiles would
    # reach that ceiling, treat h-tiling as exhausted (return False).
    fold_x_slices = [graph_nodes[n]['frontend']['x_slices'] for n in current_blob.nodes_in_blob
                      if 'force_folding_x' in graph_nodes[n]['frontend']]
    if fold_x_slices and new_h_tiles >= min(fold_x_slices):
        print('Cannot fit blob %d into memory: h_tiles=%d would reach the x-folding node\'s needed_xslices=%d '
            '(h_tiles must stay strictly below it). Falling back to eviction/splitting instead.'
            % (failing_blob_idx, new_h_tiles, min(fold_x_slices)))
        return ir, False
    # A k3 (kernel_size==3) blob with a force_folding_x node: constraint is to avoid consecutive 1s
    # in the output xslices_per_htile distribution (e.g., [2,1,1], [3,1,1,1], [1,1,...]).
    # pad_k3_folding_output_xslices_per_htile is deliberately NOT applied to these cases
    # (see the out_xslices != [2,1,1] guard in apply_blob_htiling), so growing into such a split
    # would produce an unpadded halo split. Try pre-read disabling as a fallback before resorting
    # to blob-splitting/eviction.
    if fold_x_slices:
        candidate_input_xslices = get_xslices_per_htile(min(fold_x_slices), new_h_tiles, kernel_size, is_folding_x=True)
        candidate_out_xslices = [math.ceil(n / 2) for n in candidate_input_xslices]
        # Check if output has consecutive 1s (generic rule instead of hardcoded patterns)
        has_consecutive_ones = any(candidate_out_xslices[i] == 1 and candidate_out_xslices[i+1] == 1
                                    for i in range(len(candidate_out_xslices) - 1))
        if has_consecutive_ones:
            # Before giving up, try disabling pre-read if not already disabled
            if not already_avoiding:
                print('h_tiles=%d would create consecutive 1s in out_xslices_per_htile %s (unpadded halo). '
                      'Trying to disable pre-read instead.' % (new_h_tiles, candidate_out_xslices))
                for node_name in current_blob.nodes_in_blob:
                    graph_nodes[node_name]['frontend']['avoid_ddr_read_while_conv'] = True
                return ir, True
            # If already avoiding pre-read, we can't grow h_tiles further
            print('Cannot fit blob %d into memory: h_tiles=%d would make the x-folding node\'s '
                'out_xslices_per_htile %s (consecutive 1s, unpadded halo split). '
                'Falling back to eviction/splitting instead.'
                % (failing_blob_idx, new_h_tiles, candidate_out_xslices))
            return ir, False
    if new_h_tiles > min_node_xslices:
        print('Cannot fit blob %d into memory: h_tiles=%d would exceed smallest node x_slices=%d in the blob. '
            'Consider splitting the blob or reducing layer size.'
            % (failing_blob_idx, new_h_tiles, min_node_xslices))
        return ir, False

    # Hard cap for a force_folding_x blob with 12/7 x-slices: growth past 2 h-tiles is refused, so the caller
    # falls back to eviction and blob splitting instead of clamping and looping.
    if any(graph_nodes[n]['frontend'].get('x_slices') in (12, 7) and 'force_folding_x' in graph_nodes[n]['frontend']
           for n in current_blob.nodes_in_blob) and new_h_tiles > 2:
        print('Cannot fit blob %d: h_tiles=%d exceeds the 12-x-slice cap of 2. Falling back to eviction/splitting.'
              % (failing_blob_idx, new_h_tiles))
        return ir, False
    # A producer that folds in x without output-channel splitting, consumed by an input-channel-split conv that folds
    # in both x and y, runs as a single h-tile: growth past 1 is refused, so the caller falls back to eviction/splitting.
    if new_h_tiles > 1 and any(
            graph_nodes[n]['frontend'].get('output_folding_factor_x', 0) > 0
            and (graph_nodes[n]['frontend'].get('output_folding_factor_y', 0) == 0)
            and graph_nodes[n]['frontend'].get('consumed_by_input_channel_split', False)
            for n in current_blob.nodes_in_blob):
        print('Cannot fit blob %d: h_tiles=%d but an input-split x/y-folded conv reads an x-folded, non-output-split '
              'producer (single h-tile). Falling back to eviction/splitting.' % (failing_blob_idx, new_h_tiles))
        return ir, False
    # A consumer of an x-folded, output-channel-split producer runs as a single h-tile: growth past 1 is refused,
    # so the caller falls back to eviction and blob splitting.
    if new_h_tiles > 1 and any(
            was_split_x_folded_producer(graph_nodes, p_name)
            for n in current_blob.nodes_in_blob
            for p_name, _ in graph_nodes[n]['frontend'].get('preceding_nodes_params', [])):
        print('Cannot fit blob %d: h_tiles=%d but its input is an x-folded output-split producer (single h-tile). '
              'Falling back to eviction/splitting.' % (failing_blob_idx, new_h_tiles))
        return ir, False

    current_blob.h_tiles = new_h_tiles
    return ir, apply_blob_htiling(current_blob, graph_nodes, kernel_size)

def update_node_tiling_info(ir,node):
    node_name = node['name']
    y_tiles,output_padding_start_y = get_node_tiling_info(ir,node)
    node['frontend']['y_tiles'] = y_tiles # We need to update number of tiles after blob boundries were set since the number of k=3 convs in each blob changes size of each tile and hence the number of tiles
    if y_tiles>1:
        node['backend']['add_stall_nops'] = True
        if node_avoids_ddr_write(node):
            node['backend']['add_stall_nops'] = False
    current_blob_idx = node['frontend']['tiling_blob_idx']
    current_blob = ir.tiling_blobs[current_blob_idx]
    current_blob_htiles = current_blob.h_tiles
    kernel_size = get_blob_kernel_size(current_blob)
    folding_x = 'force_folding_x' in node['frontend']
    no_k3_in_blob = (current_blob.k3_nodes == 0)
    # When the blob's output node is an odd x-fold, reverse the whole blob's h-tile x-slice split
    # (see apply_blob_htiling / blob_reverses_xslices) instead of dummy-padding the fold node.
    blob_reverse = blob_reverses_xslices(current_blob, ir.graph.nodes())
    input_tensors = get_node_input_tensors(node)
    if blob_reverse:
        input_xslices = node['frontend']['x_slices']
        internal_fold = False
    else:
        input_xslices = get_padded_input_xslices(node, node['frontend']['x_slices'], current_blob_htiles, no_k3_nodes_in_blob=no_k3_in_blob)
        internal_fold = folding_x
    xslices_per_htile = get_xslices_per_htile(input_xslices, current_blob_htiles, kernel_size,
                                              is_folding_x=internal_fold, no_k3_nodes_in_blob=no_k3_in_blob)
    if is_resize_y_node(node):
        # A y-resize's input IS the preceding x-resize's output (same tensor object), so inherit the
        # x-resize's per-h-tile output split rather than recomputing it (see apply_blob_htiling).
        prec_name = node['frontend']['preceding_nodes_params'][0][0]
        xslices_per_htile = list(ir.graph.nodes()[prec_name]['frontend']['out_xslices_per_htile'])
    elif is_x_fold_node(node) and node['frontend'].get('preceding_nodes_params'):
        # A chained x-fold step's input IS the preceding fold step's output (same tensor object),
        # so inherit its split rather than recomputing independently, which could desync from what
        # the preceding fold actually produced (e.g. [4,3] vs a naive balanced [4,4]).
        prec_name = node['frontend']['preceding_nodes_params'][0][0]
        prec_node = ir.graph.nodes()[prec_name]
        if is_x_fold_node(prec_node) and 'out_xslices_per_htile' in prec_node['frontend'] \
                and len(prec_node['frontend']['out_xslices_per_htile']) == current_blob_htiles:
            xslices_per_htile = list(prec_node['frontend']['out_xslices_per_htile'])
    if blob_reverse:
        xslices_per_htile = list(reversed(xslices_per_htile))
    # Same single-node fold reversal as apply_blob_htiling, so this writer does not undo it.
    if len(current_blob.nodes_in_blob) == 1 and (node['frontend']['x_slices'] % 2 == 0) and folding_x and current_blob_htiles > 1:
        halved = [math.ceil(n / 2) for n in xslices_per_htile]
        existing_out = node['frontend'].get('out_xslices_per_htile')
        if len({n % 2 for n in halved}) == 2 and halved[0] % 2 == 0 \
                and existing_out is not None and halved != list(existing_out):
            xslices_per_htile = list(reversed(xslices_per_htile))
    for input_tensor in input_tensors:
        input_tensor.y_tiles = y_tiles
        input_tensor.xslices_per_htile = xslices_per_htile
        # Preserve input-layer packing across h-tiling (per-h-tile lane layout is re-derived); only reset
        # unpacked (scalar) tensors. See apply_blob_htiling for the same guard.
        if current_blob_htiles > 1 and not is_packed_xslices(input_tensor.num_packed_xslices):
            input_tensor.num_packed_xslices = 1
    node['frontend']['output_tensor'].y_tiles = y_tiles
    node['frontend']['xslices_per_htile'] = xslices_per_htile
    node['frontend']['h_tiles'] = current_blob_htiles
    out_xslices = get_xslices_per_htile(node['frontend']['output_tensor'].x_slices, current_blob_htiles, kernel_size,
                                        is_folding_x=internal_fold, no_k3_nodes_in_blob=no_k3_in_blob)
    # An unfolding-x layer (or x-resize) spanning more than one h-tile holds twice as many
    # x-slices per h-tile in output as input, so derive output split as 2x input split rather
    # than an even split of the doubled total, preserving the out/in ratio the sequencer relies on.
    if (node['frontend'].get('force_unfolding_x') or is_resize_x_node(node)) and current_blob_htiles > 1:
        doubled_out_xslices = [2 * n for n in xslices_per_htile]
        if sum(doubled_out_xslices) == node['frontend']['output_tensor'].x_slices:
            out_xslices = doubled_out_xslices
        elif node['frontend'].get('force_unfolding_x') and (node['frontend']['output_tensor'].x_slices % 2 != 0):
            # An odd unfolded output total can't be an exact 2x of the input split, so the balanced
            # split doesn't guarantee the dummy slice lands in h-tile 0 (required by
            # sequencer.py's is_dummy_unfolding). Force it: split the even remainder, then add 1.
            even_out_xslices = get_xslices_per_htile(node['frontend']['output_tensor'].x_slices - 1, current_blob_htiles,
                                                      kernel_size, is_folding_x=internal_fold, no_k3_nodes_in_blob=no_k3_in_blob)
            even_out_xslices[0] += 1
            out_xslices = even_out_xslices
    if is_resize_y_node(node):
        # y-resize passes x-slices straight through: output split == input split (see apply_blob_htiling).
        out_xslices = list(xslices_per_htile)
    elif blob_reverse:
        # Match the reversed input split so the blob's per-h-tile boundaries stay aligned end-to-end.
        out_xslices = list(reversed(out_xslices))
    # x-FOLDING output split must be the per-h-tile fold of the FINAL input split. Only apply when
    # the immediate consumer is the real k3 conv, or this would wrongly clobber an earlier fold
    # step's own already-correct out_xslices_per_htile.
    if node['frontend'].get('force_folding_x') and current_blob_htiles > 1:
        following = node['frontend'].get('following_nodes_params')
        consumer = ir.graph.nodes()[following[0][0]] if following else None
        if consumer is not None and consumer['frontend'].get('kernel_size', 1) != 1:
            out_xslices = [math.ceil(n / 2) for n in xslices_per_htile]
    if kernel_size != 1 and folding_x:
        # k3 folding-conv boundary halo (see calc_conv_folding_factor / pad_k3_folding_output_xslices_per_htile):
        # this recomputation would otherwise clobber that padding.
        out_xslices = pad_k3_folding_output_xslices_per_htile(
            out_xslices, node['frontend']['output_tensor'].x_slices, current_blob_htiles)
    node['frontend']['out_xslices_per_htile'] = out_xslices
    node['frontend']['output_tensor'].xslices_per_htile = out_xslices

    # A folding conv's out_xslices_per_htile can disagree with its in-blob follower's own
    # xslices_per_htile (even/odd rounding can differ though both look "correct" alone). Catch it
    # here and force a fresh blob boundary rather than let it surface as a subtler mismatch.
    if is_x_fold_node(node) and current_blob_htiles > 1:
        for following_node_name, _ in node['frontend'].get('following_nodes_params', []):
            following_node = ir.graph.nodes()[following_node_name]
            if following_node['frontend']['tiling_blob_idx'] != current_blob_idx:
                continue
            follower_xslices_per_htile = following_node['frontend'].get('xslices_per_htile')
            if follower_xslices_per_htile is not None and list(follower_xslices_per_htile) != list(out_xslices):
                # Stash on `ir` (rather than raising) so compile()'s loop can pick this up after
                # set_node_tiling_info finishes its pass.
                ir.xslices_mismatch_blob_idx = current_blob_idx
                ir.xslices_mismatch_split_node = following_node_name

    if current_blob.nodes_in_blob.index(node_name)>0 and y_tiles!=current_blob.y_tiles:

        # Print out all nodes in thie blob and # y tiles
        print("-----------------------------------------")
        print("DEBUG: Nodes in this blob and their tiles")
        for n in current_blob.nodes_in_blob:
            print(n, end="\t")
            print(ir.graph.nodes()[n]['frontend']['y_tiles'])

        # Print out all nodes and their input and output folding
        print("-----------------------------------------")
        print("DEBUG: All nodes and their input/output Y folding")
        for n in ir.graph.nodes:
            print(ir.graph.nodes[n]['name'], end="\t")
            if 'input_folding_factor_y' in ir.graph.nodes[n]['frontend'].keys():
                print(ir.graph.nodes[n]['frontend']['input_folding_factor_y'], end="\t")
            else:
                print("n/a", end="\t")
            if 'output_folding_factor_y' in ir.graph.nodes[n]['frontend'].keys():
                print(ir.graph.nodes[n]['frontend']['output_folding_factor_y'])
            else:
                print("n/a")
        print("-----------------------------------------")

        raise ValueError ('2 nodes in same blob cant have different number of tiles. This can be caused by wrong blob splitting. Please check...')
    current_blob.y_tiles = y_tiles
    node['backend']['output_padding_start_y'] = output_padding_start_y
    output_tensor = node['frontend']['output_tensor']
    ddr_write_align = 1
    if ir.sync_with_MXP and output_tensor.name in ir.tensors_to_mxp:
        ddr_write_align = 0
    node['frontend']['ddr_write_align'] = ddr_write_align

def set_node_tiling_info(ir: internal_representation.IR) -> internal_representation.IR:
    sorted_nodes = ir.lexicographical_topological_sorted_graph
    graph_nodes = ir.graph.nodes

    # Specifying the start node from hw_config
    start_node_idx = 0
    if NODES_LIST['Start']:
        start_node_idx = sorted_nodes.index(NODES_LIST['Start'][0])
    # Specifying the end node from hw_config
    end_node_idx = len(sorted_nodes)-1
    if NODES_LIST['End']:
        end_node_idx = sorted_nodes.index(NODES_LIST['End'][0])
    if start_node_idx > end_node_idx:
        raise ValueError('NODES_LIST Start (%s, idx %d) comes after End (%s, idx %d) in topological order' %
                          (NODES_LIST['Start'][0], start_node_idx, NODES_LIST['End'][0], end_node_idx))
    # Note: boundary tensors keep their normal (non-MXP) DDR address; they must not be added to
    # ir.tensors_from_mxp/tensors_to_mxp or sequencer.py requires an unpopulated mxp_tensor_to_offset
    # entry and wrongly switches them to MXP DMA geometry.

    for node_index, node_name in enumerate(sorted_nodes):
        node = graph_nodes[node_name]
        node['frontend']['run_op'] = (start_node_idx <= node_index <= end_node_idx)
        update_node_tiling_info(ir,node)

    # After every node's per-h-tile x-slice split is set, make each blob's pass-through producers match a
    # following x-folding node's (even-non-last) split so the fold reads a consistently-tiled input.
    for current_blob in ir.tiling_blobs.values():
        propagate_fold_xslices_to_producers(current_blob.nodes_in_blob, graph_nodes)

    for blob_idx,current_blob in ir.tiling_blobs.items():
        current_blob_nodes = current_blob.nodes_in_blob
        nodes_in_blob_num = len(current_blob_nodes)

        # Moving from an h-tiled blob into a non-h-tiled one can push the contiguous allocator into
        # reusing blocks holding a sibling x-slice's live input. Such a blob still *fits* so the
        # reactive fallback never catches it; proactively disable DDR-read-while-conv here instead.
        prev_blob = ir.tiling_blobs.get(blob_idx - 1)
        if current_blob.h_tiles == 1 and prev_blob is not None and prev_blob.h_tiles > 1 and prev_blob.y_tiles == 1:
            for node_name in current_blob_nodes:
                graph_nodes[node_name]['frontend']['avoid_ddr_read_while_conv'] = True

        # A '_split_ic_add' recombine node's inputs are partial sums, not real channel-slices, so
        # they never get a real ddr_entry. Suppress the pre-read h_tiles>1 would otherwise force,
        # or the generic machinery tries to reload them from DDR (NoneType ddr_entry in sequencer).
        if current_blob.h_tiles > 1 and any('_split_ic_add' in n for n in current_blob_nodes):
            for node_name in current_blob_nodes:
                graph_nodes[node_name]['frontend']['avoid_ddr_read_while_conv'] = True

        # Each blob writes current_tile-1 and reads tile+1.
        # Write starts at last node of blob, ends in next tile before op starts on next_tile_read_node_idx.
        # Read of tile+1 starts right after write finishes, ends at first node of next tile.
        current_blob.next_tile_read_node_idx = (nodes_in_blob_num - 1) // 2
        if current_blob.next_tile_read_node_idx > 0:
            current_blob.next_tile_read_node_idx -= 1
            read_node_op = graph_nodes[current_blob_nodes[current_blob.next_tile_read_node_idx]]['op_type']
            if read_node_op in (OpType.CONCAT, OpType.ADD):
                # next_tile_read_node cannot sit on a multi-input op, so it's bumped forward. But that
                # ADD/CONCAT may reuse the offloaded blob-output AMM lane (cross-tile WAR hazard).
                # Record the skipped node so the sequencer can emit an offload-complete wait first.
                current_blob.offload_wait_node_idx = current_blob.next_tile_read_node_idx
                current_blob.next_tile_read_node_idx += 1
    return ir


def get_blob_inputs_and_outputs(ir: internal_representation.IR, blob_node_names):
    blob_inputs = []
    blob_outputs = []
    blob_node_set = set(blob_node_names)
    seen_inputs = set()
    seen_outputs = set()
    graph_nodes = ir.graph.nodes()

    for node_name in blob_node_names:
        node = graph_nodes[node_name]
        frontend = node['frontend']
        input_tensors = get_node_input_tensors(node)

        for input_tensor in input_tensors:
            if input_tensor.producer not in blob_node_set and id(input_tensor) not in seen_inputs:
                blob_inputs.append(input_tensor)
                seen_inputs.add(id(input_tensor))

        output_tensor = frontend['output_tensor']
        if id(output_tensor) not in seen_outputs:
            consumers = output_tensor.consumers
            if not consumers or any(c not in blob_node_set for c in consumers):
                blob_outputs.append(output_tensor)
                seen_outputs.add(id(output_tensor))

    return blob_inputs, blob_outputs

# After a blob split, blob0 may have only carried a nonzero y-fold forward for the split node's
# benefit. If every node in blob0 is trivial-kernel (needs no y-folding), that fold is dead
# weight: reset it to 0 so blob0 may fit without further splitting, then re-derive the split node's fold.
def reset_trivial_kernel_blob_y_folding(ir: internal_representation.IR, split_node_name):
    graph_nodes = ir.graph.nodes()
    split_node = graph_nodes[split_node_name]
    blob_idx = split_node['frontend']['tiling_blob_idx']
    if blob_idx == 0:
        return ir
    if split_node['frontend'].get('input_folding_factor_y', 0) <= 0:
        # Resetting blob0's y-fold only helps when the split node itself consumes a y-folded input;
        # if its input is already unfolded in y, zeroing blob0's y-fold buys nothing.
        return ir
    blob0 = ir.tiling_blobs[blob_idx - 1]
    nodes_in_blob0 = list(blob0.nodes_in_blob)

    for node_name in nodes_in_blob0:
        node = graph_nodes[node_name]
        if node['op_type'] != OpType.CONV:
            return ir
        kernel_size = node['frontend']['kernel_size']
        stride = node['frontend']['stride']
        is_trivial = (kernel_size == 1 or kernel_size == 3) and stride == 1
        if not is_trivial:
            return ir

    if all(graph_nodes[n]['frontend']['input_folding_factor_y'] == 0 and
           graph_nodes[n]['frontend']['output_folding_factor_y'] == 0 for n in nodes_in_blob0):
        return ir

    for node_name in nodes_in_blob0:
        node = graph_nodes[node_name]
        original_input_height = node['frontend']['input_tensor'].get_original_shape()[2]
        num_y_tiles = get_num_y_tiles(original_input_height, 0, original_kernel_size=1, amm_height=ir.amm_height)

        node['frontend']['input_folding_factor_y'] = 0
        node['frontend']['input_tensor'].folding_factor_y = 0
        node['frontend']['input_tensor'].y_tiles = num_y_tiles
        node['frontend']['output_folding_factor_y'] = 0
        node['frontend']['output_tensor'].folding_factor_y = 0
        node['frontend']['output_tensor'].y_tiles = num_y_tiles
        node['frontend']['y_tiles'] = num_y_tiles
        for flag in ('force_folding_y', 'force_unfolding_y'):
            if flag in node['frontend']:
                del node['frontend'][flag]
        if node_name in ir.force_y_folding:
            ir.force_y_folding.remove(node_name)
        if node_name in ir.force_y_unfolding:
            ir.force_y_unfolding.remove(node_name)

        calc_conv_qparams(ir, node_name, node)
        recompute_backend_channels_after_fold_reset(ir, node_name, node)

        # If this is a folded node that had y-folding reset, update its consumer nodes
        if node['frontend'].get('force_folding_x', False):
            output_tensor = node['frontend']['output_tensor']
            for consumer_name in list(output_tensor.consumers):
                consumer = graph_nodes[consumer_name]
                # Set force_folding_y on consumer to preserve y-folding
                if consumer['op_type'] == OpType.CONV or consumer['op_type'] == OpType.MAXPOOL:
                    consumer['frontend']['force_folding_y'] = True
                    if consumer_name not in ir.force_y_folding:
                        ir.force_y_folding.append(consumer_name)
                    # Recompute backend channels for consumer
                    recompute_backend_channels_after_fold_reset(ir, consumer_name, consumer)

    if split_node['op_type'] == OpType.CONV:
        calc_conv_folding_factor(ir, split_node_name, split_node)
        calc_conv_qparams(ir, split_node_name, split_node)
        recompute_backend_channels_after_fold_reset(ir, split_node_name, split_node)

    return ir

def mark_blobs_inputs_and_outputs(ir:internal_representation.IR) -> internal_representation.IR:
    for blob_idx,current_blob in ir.tiling_blobs.items():
        current_blob_nodes = current_blob.nodes_in_blob
        blob_inputs,blob_outputs = get_blob_inputs_and_outputs(ir,current_blob_nodes)
        current_blob.inputs = blob_inputs
        current_blob.outputs = blob_outputs
    return ir

def verify_y_folding_on_blob_edges(ir: internal_representation.IR):
    graph_nodes = ir.graph.nodes()
    tiling_blobs = ir.tiling_blobs

    for node_name in ir.lexicographical_topological_sorted_graph:
        node = graph_nodes[node_name]
        frontend = node['frontend']

        if 'force_y_folding' not in frontend and 'force_y_unfolding' not in frontend:
            continue

        current_blob = tiling_blobs[frontend['tiling_blob_idx']]
        input_tensors = get_node_input_tensors(node)
        blob_inputs_set = set(id(t) for t in current_blob.inputs)

        if not all(id(t) in blob_inputs_set for t in input_tensors):
            raise ValueError(f'Node {node_name} is y folding but its not a blob input. Please check...')
            


def _alias_ddr_entries_to_split_descendants(ir, root_tensor):
    """BFS the split-tensor descendant tree of root_tensor, giving every descendant with no ddr_entry
    of its own an entry aliased into its resolved parent's DDR region at its own channel offset.
    Recurses so nested splits (2+ generations, e.g. output-split then input-channel-split) resolve too,
    rather than leaving deeper leaf tensors with ddr_entry=None (AttributeError in the sequencer)."""
    frontier = [(child_name, root_tensor) for child_name in ir.original_tensor_to_split_tensor_map.get(root_tensor.name, [])]
    while frontier:
        slice_name, parent_tensor = frontier.pop()
        slice_tensor = ir.tensors.get(slice_name)
        if slice_tensor is None:
            continue
        if slice_tensor.ddr_entry is None:
            slice_channel_lo = getattr(slice_tensor, 'split_channel_lo', None)
            # Only alias tensors we recognize as pure channel-range (or whole-range) views of the
            # parent: split_channel_lo set, the legacy '_split_ic' naming, or an output-split's
            # whole-range input alias ('_input_alias', no channel offset).
            if slice_channel_lo is not None or '_split_ic' in slice_name or slice_name.endswith('_input_alias'):
                parent_bytes_per_channel = int(parent_tensor.ddr_entry.shape[2]) * int(parent_tensor.ddr_entry.shape[3])
                slice_tensor.data = np.zeros(slice_tensor.shape_real_x16, dtype=np.uint8)
                slice_entry = TensorDDREntry(create_tsnp_tensor_byte_array(slice_tensor.data),
                                             type=parent_tensor.ddr_entry.type,
                                             description='Channel slice of %s' % parent_tensor.name,
                                             shape=slice_tensor.shape_real_x16, amm_height=ir.amm_height)
                slice_entry.address = parent_tensor.ddr_entry.address
                if slice_channel_lo is not None:
                    # Input-channel alias of an output-split producer: the sequencer's input-split DMR formula
                    # adds the folded piece offset itself, so the alias stays at the parent's region start.
                    producer_name = parent_tensor.producer
                    parent_is_output_split = (producer_name in ir.graph.nodes and (parent_tensor.folding_factor_x > 0) and
                                              ir.graph.nodes[producer_name]['frontend'].get('was_channel_split', False))
                    consumer_is_input_split_conv = any(
                        ir.graph.nodes[c]['op_type'] == OpType.CONV and ir.graph.nodes[c]['frontend'].get('is_input_channel_split', False)
                        for c in slice_tensor.consumers if c in ir.graph.nodes)
                    # Output-split x-folded producer: the sequencer places each piece by its own folded base (in the
                    # new output-split read and the input-split DMR), so the alias must not add the channel start.
                    if not (parent_is_output_split and (consumer_is_input_split_conv or (parent_tensor.folding_factor_x > 0 and parent_tensor.folding_factor_y > 0))):
                        slice_entry.address += slice_channel_lo * parent_bytes_per_channel
                slice_tensor.ddr_entry = slice_entry
        if slice_tensor.ddr_entry is not None:
            frontier.extend((gc, slice_tensor) for gc in ir.original_tensor_to_split_tensor_map.get(slice_name, []))

def allocate_ddr_for_blobs_output_nodes(ir: internal_representation.IR) -> internal_representation.IR:
    program_and_intermediate_tensors_ddr = ir.ddr
    for tiling_blob_idx,tiling_blob in ir.tiling_blobs.items():
        last_node_in_blob_name = tiling_blob.nodes_in_blob[-1]
        last_node_in_blob = ir.graph.nodes()[last_node_in_blob_name]
        last_node_in_blob_output_tensor = last_node_in_blob['frontend']['output_tensor']
        blob_output_tensors = tiling_blob.outputs
        for blob_output_tensor in blob_output_tensors:
            # Scatter-concat output gets no entry here: its region is created lazily and filled by
            # its input producers, aliased in below. A normal entry would double-allocate/clobber it.
            producer_name = blob_output_tensor.producer
            if producer_name in ir.graph.nodes and ir.graph.nodes[producer_name]['frontend'].get('is_scatter_concat'):
                continue
            if blob_output_tensor.name not in ir.outputs: # If its an output tensor, its already has allocated mem in DDR
                # ALEX FIX: Calculate size both ways and allocate the larger one
                shape_original = blob_output_tensor.shape_real_x16
                shape_calculated = calculate_tensor_allocation_shape(
                    blob_output_tensor, extra_lines=_tensor_producer_wants_k3_1T_boundary_extra_line(ir, blob_output_tensor))

                size_original = np.prod(shape_original)
                size_calculated = np.prod(shape_calculated)

                # Use whichever is larger
                if size_calculated > size_original:
                    blob_output_tensor_shape = shape_calculated
                else:
                    blob_output_tensor_shape = shape_original

                blob_output_tensor.data = np.zeros(blob_output_tensor_shape,dtype=np.uint8)
                tensor_bytearray = create_tsnp_tensor_byte_array(blob_output_tensor.data)
                ddr_entry_description = (
                                        f"Blob {tiling_blob_idx} output mem, "
                                        f"Intermediate tensor name: {blob_output_tensor.name}, "
                                        f"producer node: {blob_output_tensor.producer}, "
                                        f"folded shape: {blob_output_tensor.get_folded_shape()}, "
                                        f"real shape: {blob_output_tensor_shape}"
                                    )
                current_intermediate_tensor_ddr_entry = TensorDDREntry(tensor_bytearray, type = DDREntryType.INTERMEDIATE_TENSOR, description = ddr_entry_description, shape = blob_output_tensor_shape, amm_height = ir.amm_height)
                blob_output_tensor.ddr_entry = current_intermediate_tensor_ddr_entry
                
                # Check if this tensor is a split version of another tensor.
                # If so, do not add a new DDR entry, it will re-use the original entry.
                if blob_output_tensor.name in ir.split_tensor_to_original_tensor_map:
                    # Get the original tensor
                    original_output_tensor_name = ir.split_tensor_to_original_tensor_map[blob_output_tensor.name]
                    original_output_tensor = ir.tensors[original_output_tensor_name]

                    # Ensure the original tensor has a DDR entry (graph output: already has one;
                    # non-output: first split chunk processed has none yet, second already does).
                    if original_output_tensor.ddr_entry is None:
                        # ALEX FIX: Calculate size both ways and allocate the larger one
                        shape_original = original_output_tensor.shape_real_x16
                        shape_calculated = calculate_tensor_allocation_shape(
                            original_output_tensor, extra_lines=_tensor_producer_wants_k3_1T_boundary_extra_line(ir, original_output_tensor))

                        size_original = np.prod(shape_original)
                        size_calculated = np.prod(shape_calculated)

                        # Use whichever is larger
                        if size_calculated > size_original:
                            original_output_tensor_shape = shape_calculated
                        else:
                            original_output_tensor_shape = shape_original

                        original_output_tensor.data = np.zeros(original_output_tensor_shape,dtype=np.uint8)
                        original_tensor_bytearray = create_tsnp_tensor_byte_array(original_output_tensor.data)
                        original_ddr_entry_description = 'Blob %d output mem, Intermediate tensor name: %s, producer node: %s, folded shape: %s, real shape: %s' % (tiling_blob_idx,
                                                        original_output_tensor.name,original_output_tensor.producer,str(original_output_tensor.get_folded_shape()), str(original_output_tensor_shape))
                        original_output_tensor_ddr_entry = TensorDDREntry(original_tensor_bytearray, type = DDREntryType.INTERMEDIATE_TENSOR, description = original_ddr_entry_description, shape = original_output_tensor_shape, amm_height = ir.amm_height)
                        original_output_tensor.ddr_entry = original_output_tensor_ddr_entry
                        program_and_intermediate_tensors_ddr.add_entry(original_output_tensor_ddr_entry)

                    # Assign this tensor to that same type and address, plus a per-chunk channel offset.
                    blob_output_tensor.ddr_entry.type = original_output_tensor.ddr_entry.type
                    blob_output_tensor.ddr_entry.address = original_output_tensor.ddr_entry.address
                    split_channel_lo = getattr(blob_output_tensor, 'split_channel_lo', None)
                    if split_channel_lo is not None:
                        # DDR layout is channel-major (byte c*H*W), so this [lo:hi) chunk writes at
                        # offset lo*H*W. Reduces to the old split2 tail formula for an equal 2-way split.
                        orig_shape = original_output_tensor.ddr_entry.shape
                        bytes_per_channel = int(orig_shape[2]) * int(orig_shape[3])
                        # The x-fold multiplier only applies when this piece's x-slices are half of its producer's
                        # input x-slices (the fold is already in the piece's layout). Otherwise it is not applied.
                        producer_input_name = ir.graph.nodes[blob_output_tensor.producer]['inputs'][0]
                        producer_input_xslices = ir.tensors[producer_input_name].x_slices
                        producer_frontend = ir.graph.nodes[blob_output_tensor.producer]['frontend']
                        is_x_folded_piece = (blob_output_tensor.folding_factor_x > 0 and
                                             blob_output_tensor.x_slices * (2 ** blob_output_tensor.folding_factor_x) != producer_input_xslices and
                                             producer_frontend['input_folding_factor_x'] != producer_frontend['output_folding_factor_x'])
                        fold_multiplier = (2 ** blob_output_tensor.folding_factor_x) if is_x_folded_piece else 1
                        blob_output_tensor.ddr_entry.address += split_channel_lo * fold_multiplier * bytes_per_channel

                    # Downstream read-slices were registered against the merged original name and are
                    # never a blob output, so alias each into the original's DDR region at its own offset.
                    _alias_ddr_entries_to_split_descendants(ir, original_output_tensor)
                else:
                    program_and_intermediate_tensors_ddr.add_entry(current_intermediate_tensor_ddr_entry)

                # Channel-range input slices read from this tensor's DDR region (conv input-channel
                # slices, elementwise Add input slices) never appear as blob outputs, so give each its
                # own entry aliasing this region at byte offset lo*H*W (channel-major layout).
                _alias_ddr_entries_to_split_descendants(ir, blob_output_tensor)
    return ir

# Allocate tensors reserved by the system
# Currently this is just a tensor of zeros to be loaded into the final AMM blocks
def allocate_system_tensors(ir: internal_representation.IR):
    # First create a tensor for the zeros to be loaded into AMM
    zero_tensor_name = 'SYSTEM_zero_tensor'
    zero_tensor_data = np.zeros((1, ir.amm_height, ir.amm_block_size, ir.amm_block_size), dtype=np.int8)
    ir.tensors[zero_tensor_name] = Tensor(
        zero_tensor_name,
        zero_tensor_data,
        is_constant=True,
        shape=zero_tensor_data.shape)

    # Create DDR entry for the zero tensor
    zero_data_bytearray = bytearray([0]*zero_tensor_data.size)
    ddr_entry_description = "System-reserved zero region"
    # Assert this entry doesn't already exist
    for entry in ir.ddr.entries:
        assert entry.description != ddr_entry_description
    zero_tensor_ddr_entry = TensorDDREntry(
        zero_data_bytearray,
        type = DDREntryType.INTERMEDIATE_TENSOR,
        description = ddr_entry_description,
        shape = zero_tensor_data.shape,
        amm_height = ir.amm_height)
    ir.tensors[zero_tensor_name].ddr_entry = zero_tensor_ddr_entry
    ir.ddr.add_entry(zero_tensor_ddr_entry)

    # Allocate the last block in each AMM
    ir.amms.allocate_blocks_in_all_amms([ir.amm_num_blocks-1])
    allocated_blocks = [[ir.amm_num_blocks-1],[ir.amm_num_blocks-1],[ir.amm_num_blocks-1],[ir.amm_num_blocks-1]]
    ir.amms.tensors_in_amm[zero_tensor_name] = AMMTensor(ir.tensors[zero_tensor_name],allocated_blocks,allocated_blocks,tile_num=0,htile_num=0,xslice_num=0)

    return ir

def compile(ir: internal_representation.IR, debug_output_dir:str) -> internal_representation.IR:
    ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    # distribute_entry_x_folding must run before get_ops_grid_config, which locks in
    # backend['input_channels']/['output_channels'] from whatever folding_factor_x is
    # set on each node's tensors at the time it runs.
    ir = distribute_entry_x_folding(ir)
    ir = get_ops_grid_config(ir) # This will scan the graph and calculate grid config for each of the grid related ops
    ir = allocate_input_tensors(ir)
    ir = allocate_output_tensors(ir)
    ir = get_last_compile_per_layer_wloc_size(ir,debug_output_dir)
    move_outputs_write_to_amm_to_erliest_point(ir) # This moves outputs generation to earliest execution point so its tensor can be de-allocated
    qparams_report_filename = os.path.join(ir.compiler_output_dir, ir.model_name+'_qparams_pretiling')
    reports.generate_qparams_report(qparams_report_filename,ir)

    allocation_succeeded = False
    failing_blob_idx = None
    # Set when set_node_tiling_info detects an out_xslices_per_htile mismatch; kept in plain locals
    # to survive the _snapshot(latest_ir) restore, consumed by the "H-tiling exhausted" branch.
    xslices_mismatch_blob_idx = None
    xslices_mismatch_split_node = None
    # Snapshot to roll back mem_allocation_pass mutations. A plain deepcopy re-copies every constant
    # weight array on every retry (OOM risk); since nothing mutates them in place, share them by
    # reference via a memo built fresh each call (also picks up folded weights and split tensors).
    def _add_tensor_to_memo(memo, tensor):
        if getattr(tensor, 'is_constant', False):
            for arr in (tensor.data, getattr(tensor, 'scale', None), getattr(tensor, 'zero_point', None)):
                if isinstance(arr, np.ndarray):
                    memo[id(arr)] = arr
    def _build_shared_weight_memo(obj):
        memo = {}
        for tensor in obj.tensors.values():
            _add_tensor_to_memo(memo, tensor)
        for n in obj.graph.nodes:
            frontend = obj.graph.nodes[n].get('frontend', {})
            for value in frontend.values():
                if isinstance(value, Tensor):
                    _add_tensor_to_memo(memo, value)
        return memo
    # These 4 backend keys hold the per-tile AMM block allocation lattices (by far the biggest thing
    # in the graph); rebuilt from scratch each mem_allocation_pass() call, so strip instead of copy.
    _ALLOC_GRID_BACKEND_KEYS = (
        'allocated_amm_blocks_for_input_even_grid', 'allocated_amm_blocks_for_input_odd_grid',
        'allocated_amm_blocks_for_output_even_grid', 'allocated_amm_blocks_for_output_odd_grid',
    )
    def _snapshot(obj):
        # Each call drops the caller's previous ir reference, which is a cyclic networkx DiGraph
        # that refcounting alone won't free. Across many _snapshot() calls per escalation this
        # garbage can push RSS over the limit mid-deepcopy, so collect before allocating the copy.
        gc.collect()
        stripped = {}
        for node_name, node in obj.graph.nodes(data=True):
            backend = node.get('backend')
            if backend:
                grids = {k: backend.pop(k) for k in _ALLOC_GRID_BACKEND_KEYS if k in backend}
                if grids:
                    stripped[node_name] = grids
        try:
            return copy.deepcopy(obj, memo=_build_shared_weight_memo(obj))
        finally:
            for node_name, grids in stripped.items():
                obj.graph.nodes[node_name]['backend'].update(grids)
    latest_ir = _snapshot(ir)
    while not allocation_succeeded:
        ir = _snapshot(latest_ir) # Restore the previous iteration IR
        if failing_blob_idx is None:
            ir, needed_node_output_split = set_tiling_blobs(ir, failing_blob_idx)
            if not needed_node_output_split:
                ir, split_any_wloc_adds = force_split_oversized_wloc_adds(ir)
                if split_any_wloc_adds:
                    needed_node_output_split = True
                else:
                    # Fresh blob build: give every blob a deterministic h_tiles head-start instead of
                    # letting mem_allocation_pass discover it one full-graph retry at a time. The
                    # escalation ladder below still runs unchanged for whatever this estimate misses.
                    ir = precompute_blob_htiles(ir)
            latest_ir = _snapshot(ir) # Update the IR for this iteration
            if needed_node_output_split:
                failing_blob_idx = None
                continue
        else:
            ir, htiling_applied = update_htiling_info(ir, failing_blob_idx)
            if htiling_applied:
                latest_ir = _snapshot(ir)
            else:
                # H-tiling is exhausted for this blob (cannot grow h_tiles, or an
                # x-fold node degenerated to a single x-slice per htile).
                ir = _snapshot(latest_ir)  # discard the failed htiling attempt, keep last good htiling
                graph_nodes = ir.graph.nodes()
                failing_blob = ir.tiling_blobs[failing_blob_idx]
                failing_blob.h_tiles = 1
                already_avoiding = any(graph_nodes[n]['frontend'].get('avoid_ddr_read_while_conv', False)
                                       for n in failing_blob.nodes_in_blob)
                # A resize_x + resize_y pair is effectively ONE op for AMM purposes (resize_y aliases
                # resize_x's output, no footprint) and can't be cleanly split, so treat it like a
                # single node for the eviction tier. Marker goes on nodes_in_blob[0] (resize_x).
                is_single_node = len(failing_blob.nodes_in_blob) == 1 or \
                    all(graph_nodes[n]['op_type'] == OpType.RESIZE for n in failing_blob.nodes_in_blob)
                already_evicting = is_single_node and graph_nodes[failing_blob.nodes_in_blob[0]]['frontend'].get(
                    'evict_amm_except_system_before_alloc', False)
                if is_single_node and not already_evicting:
                    # Before the expensive node split, give the lone node the entire AMM: disable
                    # DDR-read/write-while-conv and evict every other resident tensor at blob entry.
                    node_name = failing_blob.nodes_in_blob[0]
                    graph_nodes[node_name]['frontend']['avoid_ddr_read_while_conv'] = True
                    graph_nodes[node_name]['frontend']['avoid_ddr_write_while_conv'] = True
                    graph_nodes[node_name]['frontend']['evict_amm_except_system_before_alloc'] = True
                    print('H-tiling exhausted for single-node blob %d: avoiding DDR read+write while conv and evicting all other AMM tensors, re-running allocation before splitting.' % failing_blob_idx)
                    latest_ir = _snapshot(ir)
                elif not already_avoiding:
                    # Multi-node blob: before the expensive blob-split fallback, try a cheaper lever:
                    # disable DDR-read-while-conv for this blob's nodes, freeing the pre-read
                    # double-buffer AMM space. Re-run allocation and only split if this also fails.
                    for node_name in failing_blob.nodes_in_blob:
                        graph_nodes[node_name]['frontend']['avoid_ddr_read_while_conv'] = True
                    print('H-tiling exhausted for blob %d: disabling DDR-read-while-conv for this blob and re-running allocation before splitting.' % failing_blob_idx)
                    latest_ir = _snapshot(ir)
                    # failing_blob_idx=None -> set_tiling_blobs(None) rebuilds blobs from the preserved
                    # frontend (marks survive) WITHOUT splitting, then falls through to mem_allocation_pass.
                else:
                    # Even with DDR-read-while-conv disabled it still doesn't fit; fall back to
                    # splitting. Reset h_tiles to 1 first so the rebuild below starts fresh instead
                    # of leaking the exhausted htiling into the newly split blobs.
                    failing_blob.h_tiles = 1
                    apply_blob_htiling(failing_blob, graph_nodes, get_blob_kernel_size(failing_blob))
                    nodes_to_split_before = set(ir.nodes_to_split_failing_blobs)
                    ir, _ = set_tiling_blobs(ir, failing_blob_idx)
                    new_split_nodes = set(ir.nodes_to_split_failing_blobs) - nodes_to_split_before
                    for split_node_name in new_split_nodes:
                        ir = reset_trivial_kernel_blob_y_folding(ir, split_node_name)
                    latest_ir = _snapshot(ir)
                    if new_split_nodes:
                        # The reset above may be enough for the newly-isolated blob(s) to fit on their
                        # own - re-run allocation now, before further escalation. Use a throwaway
                        # snapshot so a failed trial can't leak partial state into latest_ir.
                        trial_ir = _snapshot(ir)
                        trial_ir, trial_needed_node_output_split = set_tiling_blobs(trial_ir, None)
                        trial_ir = precompute_blob_htiles(trial_ir)
                        if DEBUG_IDENTITY_CONV_BEFORE_K3_1T:
                            trial_ir = add_identity_before_k3_starting_1T_blob(trial_ir)
                        trial_ir.xslices_mismatch_blob_idx = None
                        trial_ir.xslices_mismatch_split_node = None
                        trial_ir = set_node_tiling_info(trial_ir)
                        trial_xslices_mismatch = trial_ir.xslices_mismatch_blob_idx
                        if trial_xslices_mismatch is None:
                            trial_ir = mark_blobs_inputs_and_outputs(trial_ir)
                            verify_y_folding_on_blob_edges(trial_ir)
                            qparams_report_filename = os.path.join(trial_ir.compiler_output_dir, trial_ir.model_name+'_qparams')
                            reports.generate_qparams_report(qparams_report_filename, trial_ir)
                            trial_ir, trial_alloc_ok, trial_failing_blob_idx = mem_allocation_pass(trial_ir)
                            if trial_alloc_ok:
                                ir = trial_ir
                                allocation_succeeded = True
                                latest_ir = _snapshot(ir)
            failing_blob_idx = None
            continue
        if DEBUG_IDENTITY_CONV_BEFORE_K3_1T:
            ir = add_identity_before_k3_starting_1T_blob(ir)
        # Reset before the call so a stale hit from a prior iteration's (different) ir snapshot
        # can't leak forward if this iteration's pass doesn't hit the check at all.
        ir.xslices_mismatch_blob_idx = None
        ir.xslices_mismatch_split_node = None
        ir = set_node_tiling_info(ir)
        xslices_mismatch_blob_idx = ir.xslices_mismatch_blob_idx
        xslices_mismatch_split_node = ir.xslices_mismatch_split_node
        if xslices_mismatch_blob_idx is not None:
            # xslices_per_htile disagreement is already known wrong, so force the split at the
            # offending follower now instead of waiting for an unrelated AMM-fit failure.
            ir = _snapshot(latest_ir)  # discard this attempt's set_node_tiling_info mutations
            failing_blob = ir.tiling_blobs[xslices_mismatch_blob_idx]
            failing_blob.h_tiles = 1
            apply_blob_htiling(failing_blob, ir.graph.nodes(), get_blob_kernel_size(failing_blob))
            if xslices_mismatch_split_node not in ir.nodes_to_split_failing_blobs:
                ir.nodes_to_split_failing_blobs.add(xslices_mismatch_split_node)
            ir, _ = set_tiling_blobs(ir, None)
            xslices_mismatch_blob_idx = None
            xslices_mismatch_split_node = None
            latest_ir = _snapshot(ir)
            failing_blob_idx = None
            continue
        ir = mark_blobs_inputs_and_outputs(ir)
        verify_y_folding_on_blob_edges(ir) # Y folding and unfolding can happen only on blob inputs as we do it with dma read from ddr
        qparams_report_filename = os.path.join(ir.compiler_output_dir, ir.model_name+'_qparams')
        reports.generate_qparams_report(qparams_report_filename,ir)

        ir, allocation_succeeded, failing_blob_idx = mem_allocation_pass(ir)# This will include AMM and DDR offloading allocations

    # `latest_ir` is now unreachable (cyclic, per the note in _snapshot above); collect before
    # the next big allocation (cbc_generation_pass) rather than leave it as dead weight.
    del latest_ir
    gc.collect()
    # Extra-line flags only feed DDR sizing and the sequencer, not AMM, so one pass on the final blobs is enough.
    ir = add_k3_extra_line_for_concat_inputs(ir)
    ir = add_odd_height_y_fold_extra_lines(ir)
    _mark_asym_maxpool_extra_lines(ir)
    ir = allocate_ddr_for_blobs_output_nodes(ir) # AlloThis must be done after mem_allocation_pass since mem_allocation_pass resets all mem allocations
    ir = prepare_intermediate_tensors_write(ir) # This part can be done after program_compiler.compile found where intermediate tensors need to be written to ddr
    ir = cbc_generation_pass(ir, debug_output_dir) # This will include clock by clock (cbc) generation to grid ops
    save_per_layer_wloc_size(ir,debug_output_dir)
    return ir


def main ():
    ir = internal_representation.IR('')
    ir = ir.load('conv.nxf')
    compiler_augmented_ir = compile(ir)
    ir.save('conv.nxb')

if __name__ == "__main__":

    main()    
