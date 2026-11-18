import sys
sys.path.append('.')

from compiler.cbc_creator import calculate_pairing
import math
import shutil
import numpy as np
import torch
from numeric_simulator.defs import qTensor
from common.hw_config import BIAS_FRACTIONAL_BITS, MAX_GRID_HEIGHT, MAX_GRID_WIDTH, MAX_BIAS_BITS, MAX_MAC_BITS, MAX_REDUCE_BUS_WIDTH, FRACTIONAL_BITS
from common.debug_flags import CHECK_MAC_PAIRING_ON_FLY, DEBUG_MAC_MULTIPLICATIONS, DEBUG_MAX_OC_FOR_MULTIPLICATIONS_DEBUG, DEBUG_MAX_IC_FOR_MULTIPLICATIONS_DEBUG, DEBUG_PAIRING_USED, DEBUG_TIMING,\
                DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS, DEBUG_MAC_MULTIPLICATIONS_LAYERS, DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC,\
                DEBUG_SIMULATE_FOLDING, DEBUG_ALLOW_14BIT_BIAS, DEBUG_SIMULATOR_CLIP_ADD_BIAS_TO_INT12, DEBUG_SIMULATE_CONCAT_REQUANT, SIMULATE_IC_SPLIT, DEBUG_UNFOLD_INPUTS_BEFORE_CONCAT, \
                SIM_OUTPUT_MATRIX_CBC_FORMAT, DEBUG_TFLITE_REQUANT_SATURATE

import compiler.folding_algo as folding_algo
import datetime
from common.ddr_ir import create_tsnp_tensor_byte_array, create_tsnp_tensor_xwrap_array, create_nxd_byte_array
import os
import re 

int8min = np.iinfo(np.int8).min
int8max = np.iinfo(np.int8).max
_MAC_WRAP_MASK = (1 << 29) - 1
_MAC_WRAP_INV = ~_MAC_WRAP_MASK

# Set by tflite_simulator.simulate() so ops can inspect a consumer's op_type/shape (e.g. nx_resize
# trimming its '_y' output) without threading the graph through every op function's signature.
_CURRENT_GRAPH_NODES = None

def set_current_graph_nodes(graph_nodes):
    global _CURRENT_GRAPH_NODES
    _CURRENT_GRAPH_NODES = graph_nodes

def _find_producer_node_name(tensor_name):
    """Look up the name of the node whose outputs include tensor_name, for debug filenames."""
    if _CURRENT_GRAPH_NODES is None:
        return None
    for node in _CURRENT_GRAPH_NODES.values():
        if tensor_name in node['outputs']:
            return node['name']
    return None

def _find_original_producer_node_name(name, resolved_piece_names):
    """Producer node name for a pre-split tensor, with any '_splitN[_for_...]' suffix stripped.

    The pre-split tensor's own name usually has no producer node left in the graph after
    output-channel splitting (only the split pieces do, under '<node>_splitN' names), so look
    up a piece's producer instead and strip the split suffix to recover the original node name.
    """
    for piece_name in (resolved_piece_names or []):
        producer = _find_producer_node_name(piece_name)
        if producer is not None:
            return re.sub(r'_split\d+(_for_.*)?$', '', producer)
    return _find_producer_node_name(name) or name

def _find_original_name_for_piece(piece_name, original_tensor_to_split_tensor_map):
    """Reverse-lookup: which pre-split tensor name registered piece_name as one of its pieces."""
    if not original_tensor_to_split_tensor_map:
        return None
    for original_name, piece_names in original_tensor_to_split_tensor_map.items():
        if piece_name in piece_names:
            return original_name
    return None

def _clone_qtensor(t):
    """Shallow clone of a qTensor with a copied data array.

    real_tensor_data is shared by reference: callers only re-assign it,
    never mutate it in place, so a copy is unnecessary.
    """
    new = qTensor(t.data.copy(), scale=t.scale, zero_point=t.zero_point,
                  folding_factor_x=t.folding_factor_x, folding_factor_y=t.folding_factor_y,
                  x_slices=t.x_slices)
    new.real_tensor_data = t.real_tensor_data
    return new

def _wrap_mac_to_int30(mac_output):
    """Convert float MAC output to int64 wrapped to the HW's 30-bit signed accumulator.

    Faster equivalent of:
        mac_int = mac_output.astype(np.int64)
        mac_int = np.where(mac_int < 0, mac_int | ~mask, mac_int & mask)
    """
    mac_int = mac_output.astype(np.int64)
    neg = mac_int < 0
    np.bitwise_and(mac_int, _MAC_WRAP_MASK, out=mac_int)
    np.bitwise_or(mac_int, _MAC_WRAP_INV, out=mac_int, where=neg)
    return mac_int

def _merge_split_pieces_data(pieces):
    """Merge output-channel-split pieces' data into one real-channel-consistent block.

    Each piece was independently folded over its OWN real-channel width, not the merged
    total - concatenating each piece's already-folded .data directly would interleave
    unrelated pieces' data into the same folded-channel slot (piece 0's folded channel C
    and piece 1's folded channel C would both land at merged folded channel C instead of
    C and C+piece0_width), corrupting any downstream unfold. Unfold each piece to real
    channels first, concatenate in real-channel space (order-safe), then re-fold the
    combined real-channel data as one joint block.
    """
    x_factor = pieces[0].folding_factor_x
    y_factor = pieces[0].folding_factor_y
    if x_factor or y_factor:
        unfolded_pieces = [
            folding_algo.get_asym_unfolded_tensor(p.data, folding_factor_x=x_factor, folding_factor_y=y_factor)
            for p in pieces
        ]
        data = np.concatenate(unfolded_pieces, axis=1)
        data = folding_algo.get_asym_folded_input(data, input_folding_factor_x=x_factor, input_folding_factor_y=y_factor)
    else:
        data = np.concatenate([p.data for p in pieces], axis=1)
    return data


def _reassemble_split_tensor(name, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir=None, is_intermediate_node=True, extra_lines=0):
    """Reconstruct a pre-split tensor by concatenating its output-channel-split pieces.

    A consumer that was never itself split (e.g. a real op downstream of a synthetic
    x-fold node that got output-channel-split) still references the pre-split, merged
    tensor name - nothing computes under that name anymore, only under each piece's own
    '<name>_splitN' name (registered in original_tensor_to_split_tensor_map). Rebuild it
    on demand by concatenating the pieces along the channel axis, in registration order
    (already channel-ascending - see split_node_out_channels_in_n).

    The registered list can hold pieces from more than one split attempt on the same
    original name (e.g. a consumer-specific '<name>_splitN_for_<consumer>' set registered
    alongside an earlier plain '<name>_splitN' set that a later retry abandoned without
    cleaning up) - merge, don't overwrite, is deliberate elsewhere (mark_blob_outputs_
    deallocations' lineage walk depends on it), so stale entries accumulate. Only one such
    set is ever actually live (has real producer nodes writing into intermediate_tensors);
    group by the piece name with its split index stripped and use the first group that's
    entirely present, rather than demanding every registered name resolve at once.
    """
    piece_names = original_tensor_to_split_tensor_map.get(name)
    if not piece_names:
        return None
    if not any(re.search(r'_split\d+', p) for p in piece_names):
        # Scattered concat inputs register names already in channel order with no '_splitN'
        # suffix and only one registration per name, so just require every piece present.
        resolved = piece_names if all(p in intermediate_tensors for p in piece_names) else None
    else:
        groups = {}
        for p in piece_names:
            key = re.sub(r'_split\d+', '_split*', p)
            m = re.search(r'_split(\d+)', p)
            idx = int(m.group(1)) if m else 0
            groups.setdefault(key, []).append((idx, p))
        # Multiple '_splitN' groups can be registered (stale + live); picking the first fully-
        # present one silently reassembled a truncated tensor (resnet101 CONV2D_131 bug), so
        # pick the fully-present group with the largest total channel width instead.
        resolved = None
        best_width = -1
        for group in groups.values():
            group = [p for _, p in sorted(group)]
            if all(p in intermediate_tensors for p in group):
                width = sum(intermediate_tensors[p].data.shape[1] for p in group)
                if width > best_width:
                    resolved = group
                    best_width = width
    if resolved is None:
        return None
    pieces = [intermediate_tensors[p] for p in resolved]
    unordered_data = np.concatenate([p.data for p in pieces], axis=1)
    x_factor = pieces[0].folding_factor_x
    y_factor = pieces[0].folding_factor_y
    data = _merge_split_pieces_data(pieces)
    scale = pieces[0].scale
    if isinstance(scale, np.ndarray) and scale.size > 1:
        scale = np.concatenate([p.scale for p in pieces])
    zero_point = pieces[0].zero_point
    if isinstance(zero_point, np.ndarray) and zero_point.size > 1:
        zero_point = np.concatenate([p.zero_point for p in pieces])
    merged = qTensor(data, scale=scale, zero_point=zero_point,
                      folding_factor_x=x_factor, folding_factor_y=y_factor,
                      x_slices=pieces[0].x_slices)
    intermediate_tensors[name] = merged
    if debug_dir:
        original_node_name = _find_original_producer_node_name(name, resolved)
        debug_tensor_filename = os.path.join(debug_dir, (original_node_name + '_output_tensor.nxo'))
        export_tensor_to_nxo(debug_tensor_filename, unordered_data, list(range(unordered_data.shape[1])),
                              is_intermediate_node=is_intermediate_node, x_slices=merged.x_slices,
                              y_folding=y_factor, zero_point=zero_point,
                              extra_lines=extra_lines)
        if (x_factor > 0):
            debug_tensor_filename = os.path.join(debug_dir, (original_node_name + '_nosplit.nxo'))
            export_tensor_to_nxo(debug_tensor_filename, data, list(range(data.shape[1])),
                                is_intermediate_node=is_intermediate_node, x_slices=merged.x_slices,
                                y_folding=y_factor, zero_point=zero_point,
                                extra_lines=extra_lines)
    return merged

def _maybe_export_nosplit(node, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir, is_intermediate_node):
    """After an output-channel-split piece computes its own output, always combine all of that
    split's sibling pieces (once every one of them has been computed) into the pre-split tensor
    and store it in intermediate_tensors under its original name, so _load_input picks it up
    directly - covers both intermediate tensors and genuine model output tensors, unlike
    _reassemble_split_tensor's on-demand path (which only fires when some later non-split
    consumer happens to need the merged tensor). Also dumps '<original_node_name>_output_tensor.nxo'
    whenever debug_dir is available, regardless of DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS."""
    node_name = node['name']
    if '_split' not in node_name:
        return
    piece_name = node['outputs'][0]
    original_name = _find_original_name_for_piece(piece_name, original_tensor_to_split_tensor_map)
    if original_name is None:
        return
    _reassemble_split_tensor(original_name, intermediate_tensors, original_tensor_to_split_tensor_map,
                              debug_dir=debug_dir, is_intermediate_node=is_intermediate_node,
                              extra_lines=_export_extra_lines(node))

def _resolve_alias_name(input_name, model_inputs, intermediate_tensors,
                         split_tensor_to_original_tensor_map, original_tensor_to_split_tensor_map, debug_dir=None):
    """Chase split_tensor_to_original_tensor_map and reassemble split pieces on demand.

    Direct hit first - the alias chain must not be chased when the tensor was already
    computed standalone under this name (fan-out safety); see _load_input.
    """
    if input_name not in model_inputs and input_name not in intermediate_tensors:
        while input_name in split_tensor_to_original_tensor_map:
            input_name = split_tensor_to_original_tensor_map[input_name]
        if input_name not in model_inputs and input_name not in intermediate_tensors and original_tensor_to_split_tensor_map:
            _reassemble_split_tensor(input_name, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir=debug_dir)
    return input_name

def _load_input(node_name, input_name, model_inputs, intermediate_tensors,
                input_folding_factor_x=0, input_folding_factor_y=0, use_xwrap=False, split_tensor_to_original_tensor_map=None, original_tensor_to_split_tensor_map=None, debug_dir=None):
    """Load and optionally fold input tensor from model_inputs or intermediate_tensors.

    When folding is needed we clone the qTensor before mutating it (so the
    shared model_inputs entry isn't corrupted). Otherwise share the reference.
    """

    # Aliased tensors (scatter-concat, DDR-recombine) are still stored under their own name -
    # aliasing is a real-hardware DDR-address trick invisible here. Only chase the alias when
    # never computed standalone; a direct hit must win first or fan-out consumers get misdirected.
    input_name = _resolve_alias_name(input_name, model_inputs, intermediate_tensors,
                                      split_tensor_to_original_tensor_map, original_tensor_to_split_tensor_map, debug_dir=debug_dir)

    if input_name in model_inputs:
        will_fold = (input_folding_factor_x > 0 or input_folding_factor_y > 0) and DEBUG_SIMULATE_FOLDING
        if will_fold:
            input_tensor = _clone_qtensor(model_inputs[input_name])
            if use_xwrap:
                input_tensor.data = create_tsnp_tensor_xwrap_array(input_tensor.data, input_folding_factor_x=input_folding_factor_x)
                input_tensor.data = folding_algo.get_asym_folded_input(input_tensor.data, input_folding_factor_x=0, input_folding_factor_y=input_folding_factor_y)
            else:
                input_tensor.data = folding_algo.get_asym_folded_input(input_tensor.data, input_folding_factor_x=input_folding_factor_x, input_folding_factor_y=input_folding_factor_y)
            input_tensor.folding_factor_x = input_folding_factor_x
            input_tensor.folding_factor_y = input_folding_factor_y
        else:
            input_tensor = model_inputs[input_name]
    elif input_name in intermediate_tensors:
        input_tensor = intermediate_tensors[input_name]
    else:
        raise ValueError('Op %s input: %s not found!!' % (node_name, input_name))
    return input_tensor

def _get_tensor_shapes(node, folding_conv_x=False, folding_conv_y=False, unfolding_conv_y=False, stride=1):
    """Get folded or original tensor shapes based on DEBUG_SIMULATE_FOLDING flag."""
    if DEBUG_SIMULATE_FOLDING:
        folded_input_shape = node['frontend']['input_tensor'].get_folded_shape(folding_conv_y=folding_conv_y, unfolding_conv_y=unfolding_conv_y)
        folded_output_shape = node['frontend']['output_tensor'].get_folded_shape(folding_conv_x=folding_conv_x, producing_node_stride=stride)
    else:
        folded_input_shape = node['frontend']['input_tensor'].get_original_shape()
        folded_output_shape = node['frontend']['output_tensor'].get_original_shape()
    return folded_input_shape, folded_output_shape

def _apply_output_padding(output, node, output_pad_value, amm_height, tensors_to_mxp):
    """Apply width padding to MAX_GRID_WIDTH and output padding masks."""
    if (output.shape[3] != math.ceil(output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH) and (node['outputs'][0] not in tensors_to_mxp):
        padding_width = math.ceil(output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH - output.shape[3]
        padding_height = 0
        if ('pad_extra_line' in node['frontend']) and node['frontend']['pad_extra_line']:
            padding_height += 1
        output = np.pad(output, ((0, 0), (0, 0), (0, padding_height), (0, padding_width)), mode="constant", constant_values=output_pad_value)

    if (node['outputs'][0] not in tensors_to_mxp) and (node['backend']['output_padding_start_x'] < (MAX_GRID_WIDTH - 1)):
        output_pad_idxs = (MAX_GRID_WIDTH - 1) - node['backend']['output_padding_start_x']
        output[:,:,:,-output_pad_idxs:] = output_pad_value
    elif (node['outputs'][0] in tensors_to_mxp) and ('force_unfolding_x' not in node['frontend']):
        # An MXP-bound output can inherit grid-width padding from an already-aligned input even
        # though this node adds none; MXP wants the true width, so crop rather than mask.
        # Skipped for force_unfolding_x: output is still x-folded here (per-slice width), so
        # cropping now would discard all but slice 0 - nx_conv crops post-unfold instead.
        valid_width = int(output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH + node['backend']['output_padding_start_x'] + 1
        if output.shape[3] > valid_width:
            output = output[:, :, :, :valid_width]
    if (node['backend']['output_padding_start_y'] < amm_height) and (output.shape[2] > amm_height):
        output_pad_idxs = amm_height - node['backend']['output_padding_start_y']
        output[:,:,-output_pad_idxs:,:] = output_pad_value
    return output

def _get_oc_processing_order(node, output_channels):
    """Get output channel processing order from node backend or generate default."""
    if 'backend' in node and 'oc_order' in node['backend'] and DEBUG_SIMULATE_FOLDING:
        return node['backend']['oc_order']
    return list(range(output_channels))

def _create_output_tensor(output, node, output_folding_factor_x, output_folding_factor_y):
    """Create qTensor output with scale/zero_point/x_slices from node."""
    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output_x_slices = node['frontend']['output_tensor'].x_slices
    return qTensor(output, scale=output_scale, zero_point=output_zero_point,
                   folding_factor_x=output_folding_factor_x, folding_factor_y=output_folding_factor_y,
                   x_slices=output_x_slices)

def _debug_timing(node_name, start_time):
    """Print timing debug output if enabled."""
    if DEBUG_TIMING > 0:
        if DEBUG_MAC_MULTIPLICATIONS:
            print('Warning: DEBUG_MAC_MULTIPLICATIONS is on so timing includes debug file creation (which is significant)')
        print('At layer: %s, total time: %s' % (node_name, str(datetime.datetime.now() - start_time)))

def _apply_y_folding(input_data, folding_conv_y, unfolding_conv_y):
    """Apply y-axis folding or unfolding to input data."""
    if folding_conv_y and DEBUG_SIMULATE_FOLDING:
        return folding_algo.get_asym_folded_input(input_data, input_folding_factor_x=0, input_folding_factor_y=1)
    elif unfolding_conv_y and DEBUG_SIMULATE_FOLDING:
        return folding_algo.get_asym_unfolded_tensor(input_data, folding_factor_x=0, folding_factor_y=1)
    return input_data

def _shrink_x_folded_for_unfold(data, x_tensor):
    """When a tensor lives in a single x-slice (x_slices==1), its folded array is padded out to
    MAX_GRID_WIDTH before x-folding, so unfolding it verbatim doubles the padded width instead of
    the true original width, fabricating extra garbage columns beyond the real data. Truncate the
    folded data (last axis) down to the folded width that exactly covers the true original width
    (padded to even), before unfolding. No-op for multi-slice tensors."""
    if x_tensor.x_slices != 1:
        return data
    original_width = x_tensor.get_original_shape()[3]
    shrink_width = original_width + (original_width % 2)
    folded_shrink_width = shrink_width // 2
    if data.shape[3] > folded_shrink_width:
        data = data[:, :, :, :folded_shrink_width]
    return data

def _get_output_pad_value(node):
    """Get output padding value from activation or output tensor zero point."""
    output_tensor_zp = node['frontend']['output_tensor_zp']
    if ('activation_silu' in node['attributes']) and (node['attributes']['activation_silu'] is not None):
        output_tensor_zp = node['attributes']['activation_silu']['output_zp']
    output_zp_scalar = output_tensor_zp.item() if isinstance(output_tensor_zp, np.ndarray) and output_tensor_zp.size == 1 else (node['frontend']['output_tensor'].zero_point if isinstance(output_tensor_zp, np.ndarray) else output_tensor_zp)
    return output_zp_scalar

def _apply_lut(output, node):
    """Apply the node's non-linear LUT (SiLU / Requant) to the requantized integer output, mirroring
    the hardware CBC nlf stage: the uint8 output indexes a 256-entry int8 command-memory table.
    No-op when the node carries no LUT."""
    if 'lut' not in node['frontend']:
        return output
    output_uint8 = output.astype(np.uint8)
    nlf = node['backend']['nlf']
    cmd_mem = np.array(nlf[0].cmd_mem, dtype=np.int8)
    return cmd_mem[output_uint8]

def _clip_output_to_mxp(output, node, tensors_to_mxp, split_tensor_to_original_tensor_map):
    """Clip output width if tensor goes to MXP."""
    output_name = node['outputs'][0]
    output_name = output_name.replace('_x','')
    if output_name in split_tensor_to_original_tensor_map:
        output_name = split_tensor_to_original_tensor_map[output_name]
    if output_name in tensors_to_mxp:
        actual_width = node['frontend']['output_tensor'].shape[3]
        output = output[:,:,:,:actual_width]
    return output

def _debug_mac_calcs_prologue(node, q_w_int8_np, shape_src):
    if 'input_channels_reorder_dict' in node['backend']:
        input_channels_reorder_dict = node['backend']['input_channels_reorder_dict']
        reordered_channels = [item[0] for item in input_channels_reorder_dict.items()]
        q_w_int8_np = q_w_int8_np[:,reordered_channels,:,:]
    c = shape_src.shape[1]
    h = shape_src.shape[2]
    w = shape_src.shape[3]
    k_x=q_w_int8_np.shape[3]
    k_y=q_w_int8_np.shape[2]
    input_channels = q_w_int8_np.shape[1]
    output_channels = q_w_int8_np.shape[0]
    return q_w_int8_np, c, h, w, k_x, k_y, input_channels, output_channels

def export_mac_calcs_to_debug_file_tflite(filename,node,mac_int,q_input_padded,q_w_int8_np, multiplier, shift, biasH, biasL, output, oc_processing_order, amm_height):
    q_w_int8_np, c, h, w, k_x, k_y, input_channels, output_channels = _debug_mac_calcs_prologue(node, q_w_int8_np, mac_int)
    lut_output = _apply_lut(output, node) if 'lut' in node['frontend'] else None

    with open(filename,'w') as filehandler:        
        for current_oc_idx in range(min(DEBUG_MAX_OC_FOR_MULTIPLICATIONS_DEBUG,output_channels)):
            if current_oc_idx>=len(oc_processing_order):
                continue
            current_oc = oc_processing_order[current_oc_idx]
            if len(DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC)>0 and current_oc not in DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC:
                continue
            mtx_before_rq = np.zeros((w,h))
            for current_h in range(h):
                for current_w in range(w):
                    current_mac = mac_int[0,current_oc,current_h,current_w]
                    my_str = 'MAC Calcs for pixel (c=%d,h=%d,w=%d): %d\n' % (current_oc,current_h,current_w, current_mac)
                    filehandler.write(my_str)
                    mac=0
                    mac_pair = 0
                    for current_k_x in range(k_x):
                        for current_k_y in range(k_y):
                            one_by_one_conv_acc = 0
                            mblock_str = 'mblock (%d,%d)\n' % (((current_h+current_k_y-1+amm_height) % amm_height), ((current_w+current_k_x-1+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                            filehandler.write(mblock_str)
                            for current_ic in range(min(DEBUG_MAX_IC_FOR_MULTIPLICATIONS_DEBUG, input_channels)):
                                current_weight = q_w_int8_np[current_oc,current_ic,current_k_y,current_k_x]
                                current_input = q_input_padded[0,current_ic,current_h+current_k_y,current_w+current_k_x]
                                if current_weight != 0:
                                    mac                 += current_weight*current_input
                                    one_by_one_conv_acc += current_weight*current_input
                                    my_str = 'Input (%d) x Weight (%d) = %d, weight/input location [ic=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (current_input,current_weight,\
                                        current_weight*current_input,current_ic,current_ic,current_k_y,current_k_x, mac) 
                                    filehandler.write(my_str)

                            if CHECK_MAC_PAIRING_ON_FLY == True:
                                vct_current_weight = q_w_int8_np[current_oc,:,current_k_y,current_k_x]
                                vct_current_input  = q_input_padded[0,:,current_h+current_k_y,current_w+current_k_x]
                                if sum(abs(vct_current_weight)) == 0: # kernrl is 0- kernel
                                    continue
                                
                                one_by_one_conv_acc_pair = 0
                                if DEBUG_PAIRING_USED:
                                    pos_pairs, neg_pairs, non_pairs_inx = calculate_pairing(kernel_1x1=vct_current_weight)
                                    
                                else:
                                    non_pairs_inx  = np.array([i for i, num in enumerate(vct_current_weight) if num != 0])
                                    pos_pairs = []
                                    neg_pairs = []
                                    

                                my_str = 'Pairing version\n'

                                filehandler.write(my_str)
                                #pairs pos                                
                                for i_calc in pos_pairs:
                                    in_0 = vct_current_input[i_calc[0]]
                                    in_1 = vct_current_input[i_calc[1]]
                                    w_all= vct_current_weight[i_calc[0]]
                                    ic0  = i_calc[0]
                                    ic1  = i_calc[1]
                                    mac_pair+= (in_0+in_1)*w_all
                                    my_str = '[Input0 (%d)+ Input1(%d)]  x Weight (%d) = %d, weight/input location [ic0=%d(@amm=%d),ic1=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (in_0,in_1, w_all, (in_0+in_1)*w_all,ic0, ic0, ic1,ic1,current_k_y,current_k_x, mac_pair) 
                                    filehandler.write(my_str)
                                #pairs_neg
                                for i_calc in neg_pairs:
                                    in_0 = vct_current_input[i_calc[0]]
                                    in_1 = vct_current_input[i_calc[1]]
                                    w_all= vct_current_weight[i_calc[0]]
                                    ic0  = i_calc[0]
                                    ic1  = i_calc[1]
                                    mac_pair+= (in_0-in_1)*w_all
                                    my_str = '[Input0 (%d)- Input1(%d)]  x Weight (%d) = %d, weight/input location [ic0=%d(@amm=%d),ic1=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (in_0,in_1, w_all, (in_0-in_1)*w_all,ic0, ic0, ic1,ic1,current_k_y,current_k_x, mac_pair) 
                                    filehandler.write(my_str)
                                #no pairs
                                half_len = len(non_pairs_inx)//2
                                delta = len(non_pairs_inx)%2
                                for i_c in range(half_len):
                                    i_calc_w0 = non_pairs_inx[i_c]
                                    i_calc_w1 = non_pairs_inx[i_c+half_len+delta]

                                    in_0_w0 = vct_current_input[ i_calc_w0]
                                    in_0_w1 = vct_current_input[ i_calc_w1]

                                    w_w0= vct_current_weight[i_calc_w0]
                                    w_w1= vct_current_weight[i_calc_w1]

                                    ic0_w0  =  i_calc_w0
                                    ic0_w1  =  i_calc_w1

                                    mac_pair+= in_0_w0*w_w0+in_0_w1*w_w1
                                    my_str = '[Input_w0 (%d)]  x Weight_w0 (%d) + [Input_w1 (%d)]  x Weight_w1 (%d)= %d, weight/input location [ic0_w0=%d(@amm=%d),ic0_w1=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (in_0_w0, w_w0, in_0_w1, w_w1, (in_0_w0*w_w0 + in_0_w1*w_w1 ),ic0_w0, ic0_w0,ic0_w1,ic0_w1, current_k_y,current_k_x, mac_pair) 
                                    filehandler.write(my_str)
                                ##### add the odd wights if exixsts    
                                if len(non_pairs_inx)%2:
                                    i_calc_w0 = non_pairs_inx[half_len]
                                    i_calc_w1 = -1

                                    in_0_w0 = vct_current_input[i_calc_w0]
                                    in_0_w1 = 0

                                    w_w0= vct_current_weight[i_calc_w0]
                                    w_w1= 0

                                    ic0_w0  =  i_calc_w0
                                    ic0_w1  =  i_calc_w1
                                    mac_pair+= in_0_w0*w_w0+in_0_w1*w_w1
                                    my_str = '[Input_w0 (%d)]  x Weight_w0 (%d) + [Input_w1 (%d)]  x Weight_w1 (%d)= %d, weight/input location [ic0_w0=%d(@amm=%d),ic0_w1=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (in_0_w0, w_w0, in_0_w1, w_w1, (in_0_w0*w_w0 + in_0_w1*w_w1 ),ic0_w0, ic0_w0,ic0_w1,ic0_w1, current_k_y,current_k_x, mac_pair) 
                                    filehandler.write(my_str)

                                    
                                        

                                if len(neg_pairs)>0:
                                    one_by_one_conv_acc_pair += np.dot(vct_current_weight[neg_pairs[:,0]],(vct_current_input[neg_pairs[:,0]]-vct_current_input[neg_pairs[:,1]]))
                                if len(pos_pairs)>0:
                                    one_by_one_conv_acc_pair += np.dot(vct_current_weight[pos_pairs[:,0]],(vct_current_input[pos_pairs[:,0]]+vct_current_input[pos_pairs[:,1]]))
                                if len(non_pairs_inx)>0:        
                                    one_by_one_conv_acc_pair += np.dot(vct_current_weight[non_pairs_inx],vct_current_input[non_pairs_inx])
                                if (one_by_one_conv_acc!=one_by_one_conv_acc_pair):
                                    raise ValueError ('Pairing cacculation is not correct. Please check it') 


                    if (k_x==3):
                        file_str = 'mblock (%d,%d)\n' % (((current_h-1+amm_height) % amm_height), ((current_w+2+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                        filehandler.write(file_str)
                        file_str = 'mblock (%d,%d)\n' % (((current_h+amm_height) % amm_height), ((current_w+2+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                        filehandler.write(file_str)                    
                                        
                    my_str = 'Sum = %d\n' % mac
                    mtx_before_rq[current_w,current_h] = mac
                    filehandler.write(my_str)
                    if type(multiplier) == int: # Conv of Add op has scale and bias of type int and not list
                        rqparams_str = 'multiplier = %d, shift = %d, bias_H = %d, bias_L = %d\n' % (multiplier, shift, biasH, biasL)
                    else:
                        rqparams_str = 'multiplier = %d, shift = %d, bias_H = %d, bias_L = %d\n' % (multiplier[current_oc], shift[current_oc], biasH[current_oc], biasL[current_oc])
                    filehandler.write(rqparams_str)
                    in_hex = hex(output[0,current_oc,current_h,current_w] & 0xFF)[2:].upper().zfill(2)
                    rq_calcs_str = 'output (mac output after tflite requant)= %d, (%s)\n' % (output[0,current_oc,current_h,current_w], in_hex)
                    filehandler.write(rq_calcs_str)
                    if lut_output is not None:
                        lut_val = int(lut_output[0,current_oc,current_h,current_w])
                        lut_hex = hex(lut_val & 0xFF)[2:].upper().zfill(2)
                        filehandler.write('output (after LUT)= %d, (%s)\n' % (lut_val, lut_hex))

def export_mac_calcs_to_debug_file_maxpool(filename, node, q_input_padded, q_w_int8_np, output, oc_processing_order, amm_height):
    q_w_int8_np, c, h, w, k_x, k_y, input_channels, output_channels = _debug_mac_calcs_prologue(node, q_w_int8_np, output)

    # For maxpool, input_channels from weights is incorrect after x-folding
    # Use actual input channels from q_input_padded which is already unfolded
    actual_input_channels = q_input_padded.shape[1]

    with open(filename,'w') as filehandler:        
        for current_oc_idx in range(min(DEBUG_MAX_OC_FOR_MULTIPLICATIONS_DEBUG,output_channels)):
            if current_oc_idx>=len(oc_processing_order):
                continue
            current_oc = oc_processing_order[current_oc_idx]
            if len(DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC)>0 and current_oc not in DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC:
                continue
            mtx_before_rq = np.zeros((w,h))
            for current_h in range(h):
                for current_w in range(w):
                    my_str = 'MAC Calcs for pixel (c=%d,h=%d,w=%d): \n' % (current_oc,current_h,current_w)
                    filehandler.write(my_str)
                    mac=-128
                    for current_k_x in range(k_x):
                        for current_k_y in range(k_y):
                            mblock_str = 'mblock (%d,%d)\n' % (((current_h+current_k_y-1+amm_height) % amm_height), ((current_w+current_k_x-1+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                            filehandler.write(mblock_str)
                            for current_ic in range(min(DEBUG_MAX_IC_FOR_MULTIPLICATIONS_DEBUG, actual_input_channels)):
                                current_weight = q_w_int8_np[current_oc,current_ic,current_k_y,current_k_x]
                                current_input = q_input_padded[0,current_ic,current_h+current_k_y,current_w+current_k_x]
                                if current_weight != 0:
                                    mac=max(mac, current_input)
                                    my_str = 'Input (%d) , Weight (%d) , weight/input location [ic=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (current_input,current_weight,\
                                        current_ic,current_ic,current_k_y,current_k_x, mac) 
                                    filehandler.write(my_str)
                    if (k_x==3):
                        file_str = 'mblock (%d,%d)\n' % (((current_h-1+amm_height) % amm_height), ((current_w+2+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                        filehandler.write(file_str)
                        file_str = 'mblock (%d,%d)\n' % (((current_h+amm_height) % amm_height), ((current_w+2+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                        filehandler.write(file_str)                    
                                        
                    my_str = 'Max = %d\n' % mac
                    mtx_before_rq[current_w,current_h] = mac
                    filehandler.write(my_str)
                    in_hex = hex(output[0,current_oc,current_h,current_w] & 0xFF)[2:].upper().zfill(2)
                    rq_calcs_str = 'output (mac output after tflite requant)= %d, (%s)\n' % (output[0,current_oc,current_h,current_w], in_hex)
                    filehandler.write(rq_calcs_str)

def _export_extra_lines(node):
    # Total padded rows the DMW writes past the nominal height: the k3-1T boundary line (1) plus the
    # asymmetric-padding MAXPOOL trigger's lines.
    return int(bool(node['frontend'].get('k3_1T_boundary_extra_line_write', False))) + node['frontend'].get('asym_extra_lines_write', 0)

def _dma_opt_disabled(node):
    # Set by the sequencer on the producer when it wrote this tensor in the plain layout (no DMA merge).
    return bool(node['frontend'].get('dma_opt_disabled', False))

def export_tensor_to_nxo(filename,output,oc_processing_order, x_slices = 1, y_folding = 0, zero_point = 0, is_intermediate_node = False, extra_lines = 0, dma_opt_disabled = False):
    oc_processing_order = list(range(len(oc_processing_order))) ## alex changed the order of OC to be in right order
    if not DEBUG_SIMULATE_FOLDING:
        raise ValueError ('Currently, export to nxo only supported if simulate folding is enabled.')
    shuffled_tensor = output[0,oc_processing_order,:]
    shuffled_tensor = np.expand_dims(shuffled_tensor,axis=0)
    # Creating .nxd files for comparison with hardware
    # Unfolding for hardware output comparison, since AMM to DDR write happens
    if (y_folding > 0):
        shuffled_tensor = folding_algo.get_asym_unfolded_tensor(shuffled_tensor,folding_factor_x=0,folding_factor_y=y_folding)
        y_folding = 0
    if extra_lines:
        # Extra padded rows the hardware DMW writes past the nominal height: 1 for a k3-1T boundary
        # (sequencer.py k3_1T_boundary_extra_line_write), 2 for the asymmetric-padding MAXPOOL trigger.
        shuffled_tensor = np.pad(shuffled_tensor, ((0, 0), (0, 0), (0, extra_lines), (0, 0)), mode="constant", constant_values=zero_point)
    tensor_byte_array = bytearray(shuffled_tensor.astype(np.int8))
    with open(filename,'wb') as bin_file:
        bin_file.write(tensor_byte_array)
    if dma_opt_disabled:
        # The sequencer wrote this tensor in the plain layout (no DMA merge), so the .nxd is the same bytes as the .nxo.
        shutil.copyfile(filename, filename.replace('.nxo', '.nxd'))
    else:
        create_nxd_byte_array(shuffled_tensor, is_intermediate_node = is_intermediate_node, num_xslices = x_slices, y_folding = y_folding, zero_point = zero_point, filename = filename)

def compute_tflite_mac_shift(node_name, mac_requant, output_channels, output_multiplier, cInputH, cInputL, o_shift):
    acc = mac_requant
    accL = acc % 65536
    accH = (acc - accL) >> 16

    if isinstance(output_multiplier, (int, np.integer)):
        mult, cH, cL, sh = output_multiplier, cInputH, cInputL, o_shift
    else:
        mult = np.asarray(output_multiplier, dtype=np.int64).reshape(1, -1, 1, 1)
        cH   = np.asarray(cInputH,           dtype=np.int64).reshape(1, -1, 1, 1)
        cL   = np.asarray(cInputL,           dtype=np.int64).reshape(1, -1, 1, 1)
        sh   = np.asarray(o_shift,           dtype=np.int64).reshape(1, -1, 1, 1)

    p0 = accH * mult + cH
    c1 = cL + (p0 << 16)
    p1 = accL * mult + c1
    res = p1 >> sh

    if DEBUG_TFLITE_REQUANT_SATURATE:
        sat_mask = (res < int8min) | (res > int8max)
        sat11_mask = sat_mask & ((res < -(2**10)) | (res > (2**10 - 1)))
        if sat11_mask.any():
            _, och, r, c = np.argwhere(sat11_mask)[0]
            raise ValueError('11-bit Saturated @ Node: ' + node_name + ', Channel: ' + str(int(och)) + ', row: ' + str(int(r)) + ', col: ' + str(int(c)))

    return np.clip(res, int8min, int8max).astype(np.int64)

def _get_folding_flags_and_channels(node):
    folding_conv_x = 'force_folding_x' in node['frontend']
    folding_conv_y = 'force_folding_y' in node['frontend']
    unfolding_conv_y = 'force_unfolding_y' in node['frontend']
    folded_input_tensor_shape, folded_output_tensor_shape = _get_tensor_shapes(
        node, folding_conv_x, folding_conv_y, unfolding_conv_y, node['frontend']['stride'])
    input_channels = folded_input_tensor_shape[1]
    output_channels = folded_output_tensor_shape[1]
    return folding_conv_x, folding_conv_y, unfolding_conv_y, input_channels, output_channels

def _select_conv_params(node, input_folding_factor_x, input_folding_factor_y):
    if (input_folding_factor_x>0 or input_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING:
        kernel_size = node['frontend']['folded_kernel_size']
        stride = 1
        q_w_int8_np = node['frontend']['folded_weights_tensor'].data
        if node['frontend'].get('asym_padding', False) and 'asym_cbc_folded_weights_tensor' in node['frontend']:
            # Async padding Conv2D: simulate with the same fake (shifted) folded weights the CBC uses.
            # The original folded weights with symmetric folded padding give the symmetric result.
            q_w_int8_np = node['frontend']['asym_cbc_folded_weights_tensor'].data
        padding = node['frontend']['folded_padding']
    else:
        kernel_size = node['frontend']['kernel_size']
        stride = node['frontend']['stride']
        q_w_int8_np = node['frontend']['weights_tensor'].data
        padding = node['frontend']['padding']
    return kernel_size, stride, q_w_int8_np, padding

def _get_oc_processing_orders(node, output_channels, folding_conv_x, folding_conv_y, output_folding_factor_x, output_folding_factor_y):
    oc_processing_order = _get_oc_processing_order(node, output_channels)
    if ((folding_conv_x and output_folding_factor_x > 0) or (folding_conv_y and output_folding_factor_y > 0)) and DEBUG_SIMULATE_FOLDING:
        pre_folding_oc_processing_order = oc_processing_order
        if folding_conv_x:
            pre_folding_oc_processing_order = list((np.array(oc_processing_order)[::2]/2).astype(int))
    else:
        pre_folding_oc_processing_order = oc_processing_order
    return oc_processing_order, pre_folding_oc_processing_order

def nx_conv(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None,amm_height=MAX_GRID_HEIGHT,tensors_to_mxp=None,split_tensor_to_original_tensor_map=None, original_tensor_to_split_tensor_map=None):
    start_time = datetime.datetime.now()
    folding_conv_x, folding_conv_y, unfolding_conv_y, input_channels, output_channels = _get_folding_flags_and_channels(node)
    if 'force_unfolding_x' in node['frontend']:
        output_channels = output_channels * 2
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']
    kernel_size, stride, q_w_int8_np, padding = _select_conv_params(node, input_folding_factor_x, input_folding_factor_y)
    # padding is [pad_top, pad_bottom, pad_left, pad_right] (tflite_parser.py convention), but
    # torch.nn.functional.pad expects (left, right, top, bottom) for a 4-element pad on a 4D tensor.
    pad_top, pad_bottom, pad_left, pad_right = padding
    padding = [pad_left, pad_right, pad_top, pad_bottom]

    ic_splits = 1
    ic_groups = [list(range(input_channels))]
    # Default: weights and input use the same channel groups. An input-channel split overrides
    # these below (the two index sets may differ when the folded weights are still full-channel).
    weight_ic_groups = ic_groups
    input_ic_groups = ic_groups

    input_name = node['inputs'][0]
    input = _load_input(node_name, input_name, model_inputs, intermediate_tensors,
                        input_folding_factor_x, input_folding_factor_y, use_xwrap=True, split_tensor_to_original_tensor_map=split_tensor_to_original_tensor_map, original_tensor_to_split_tensor_map=original_tensor_to_split_tensor_map, debug_dir=debug_dir)
    if (node['frontend']['input_tensor'].x_slices) % 2 != 0 and node['frontend']['input_tensor'].x_slices != 1 and ('force_folding_x' in node['frontend']):
        # Clone before padding: _load_input returns the shared intermediate_tensors entry for
        # intermediate tensors, so in-place padding would corrupt it for other consumers (e.g. a
        # later Concat reading the un-padded width). See CONCATENATION_160 in yolov5n.
        input = _clone_qtensor(input)
        input.data = np.pad(input.data, ((0, 0), (0, 0), (0, 0), (0, MAX_GRID_WIDTH)), mode="constant", constant_values=node['frontend']['input_tensor_zp'])

    y_folded_input = _apply_y_folding(input.data, folding_conv_y, unfolding_conv_y)
    folded_hw = node['frontend']['input_tensor'].get_folded_shape()
    if input_folding_factor_y > 0 and folded_hw[2] < y_folded_input.shape[2]:
        y_folded_input = y_folded_input[:, :, :folded_hw[2], :]
    input_data_np = np.asarray(y_folded_input)
    pad_val = node['frontend']['input_tensor_zp']

    input_fp32_unpadded = input_data_np.astype(np.float32, copy=False)
    if '_fold_x' not in node_name:
        input_fp32_padded = torch.nn.functional.pad(
            torch.from_numpy(np.ascontiguousarray(input_fp32_unpadded)),
            padding, 'constant', float(pad_val)).numpy()
    else:
        input_fp32_padded = input_fp32_unpadded

    # int64 padded buffer is only needed by DEBUG_MAC_MULTIPLICATIONS' debug writer.
    need_int_padded = DEBUG_MAC_MULTIPLICATIONS and debug_dir
    q_input_padded = None
    if need_int_padded:
        npd_int = input_data_np.astype(np.int64, copy=False)
        if '_fold_x' not in node_name:
            q_input_padded = torch.nn.functional.pad(torch.from_numpy(npd_int), padding, 'constant', float(pad_val)).numpy()
        else:
            q_input_padded = npd_int

    # --- Input-channel split (split_conv_in_channels_in_m / _in_two) -------------------------
    # This slice-conv reads a channel slice of the producer's output, but the simulator loads the
    # FULL producer output (mapped back via split_tensor_to_original_tensor_map in _load_input).
    # The unfolded slice is NOT contiguous in the folded layout (folded_idx = block*C_full + ic,
    # block over fold positions), so selecting it means picking the SCATTERED folded indices
    # {block*C_full + offset + i} for every fold block. Applied to the folded input, and to the
    # weights too when folded_weights_tensor is still full-channel.
    input0_name = node['inputs'][0]
    # `input_channel_slice_offset` (stamped by the splitter) gives the unfolded offset;
    # `input_channels` gives this slice's unfolded count. Legacy two-way split (no marker) falls
    # back to inferring the offset from the half size.
    slice_offset = node['frontend'].get('input_channel_slice_offset', None)
    # Slices are named '<orig>_split_ic<j>' (0-indexed); slice 0 also gets offset 0 stamped, so
    # `0 is not None` still trips correctly. The endswith fallback only matters if slice_offset is
    # ever absent.
    ic_suffix_match = re.search(r'_split_ic(\d+)$', input0_name)
    # The producer sometimes already outputs only this slice's channel range (e.g. a k3-1T
    # identity conv) rather than aliasing the full original tensor - then the loaded input's
    # channel count already matches this node's declared input_channels, and no scatter
    # extraction should be applied (would double-slice). Only extract when loaded is wider.
    actual_loaded_ic = input_fp32_padded.shape[1]
    is_ic_slice = (('_split_ic' in input0_name) and (slice_offset is not None or ic_suffix_match is not None)
                   and actual_loaded_ic > input_channels)
    if is_ic_slice:
        this_unfolded_ic = node['frontend']['input_channels']          # this slice's UNFOLDED ch count
        nb = input_channels // this_unfolded_ic                         # number of fold blocks
        full_input_folded_ic = actual_loaded_ic                         # full folded input channels
        full_unfolded_ic = full_input_folded_ic // nb
        if slice_offset is not None:
            offset_unfolded = slice_offset
        elif ic_suffix_match.group(1) == '0':
            offset_unfolded = 0
        else:
            offset_unfolded = full_unfolded_ic - this_unfolded_ic
        scattered = [b * full_unfolded_ic + offset_unfolded + i
                     for b in range(nb) for i in range(this_unfolded_ic)]
        input_ic_groups = [scattered]
        # Weights: if folded weights are still full-channel use the same scattered indices;
        # if they were already sliced to this slice, take them contiguously.
        if q_w_int8_np.shape[1] == full_input_folded_ic:
            weight_ic_groups = [scattered]
        else:
            weight_ic_groups = [list(range(q_w_int8_np.shape[1]))]

    per_ic_split_output = []
    for current_ic_split in range(ic_splits):
        if SIMULATE_IC_SPLIT:
            weights = q_w_int8_np[:,weight_ic_groups[current_ic_split],:,:]
            input_fp32 = input_fp32_padded[:,input_ic_groups[current_ic_split],:,:]
            actual_input_channels = input_channels // ic_splits
        else:
            weights = q_w_int8_np
            input_fp32 = input_fp32_padded
            actual_input_channels = input_channels

        weights_t = torch.from_numpy(np.ascontiguousarray(weights.astype(np.float32)))
        input_t = torch.from_numpy(np.ascontiguousarray(input_fp32))
        mac_output = torch.nn.functional.conv2d(input_t, weights_t, bias=None, stride=stride, padding=0).numpy()
        mac_int = _wrap_mac_to_int30(mac_output)
        output_multiplier = node['frontend']['output_multiplier']
        cInputH = node['frontend']['cInputH']
        cInputL = node['frontend']['cInputL']
        o_shift = node['frontend']['o_shift']
        mac_shifted = compute_tflite_mac_shift(node_name, mac_int, output_channels, output_multiplier, cInputH, cInputL, o_shift)
        if SIMULATE_IC_SPLIT:
            per_ic_split_output.append(mac_shifted)

    if SIMULATE_IC_SPLIT:
        mac_shifted = np.zeros(per_ic_split_output[0].shape, dtype=np.int64)
        for current_split_output in per_ic_split_output:
            mac_shifted += current_split_output

    output = mac_shifted

    oc_processing_order, pre_folding_oc_processing_order = _get_oc_processing_orders(
        node, output_channels, folding_conv_x, folding_conv_y, output_folding_factor_x, output_folding_factor_y)

    if DEBUG_MAC_MULTIPLICATIONS and debug_dir:
        if (len(DEBUG_MAC_MULTIPLICATIONS_LAYERS) > 0 and node_name in DEBUG_MAC_MULTIPLICATIONS_LAYERS) or len(DEBUG_MAC_MULTIPLICATIONS_LAYERS) == 0:
            mac_calcs_filename = os.path.join(debug_dir, (node_name + '_MAC_CALCS_DEBUG.txt'))
            export_mac_calcs_to_debug_file_tflite(mac_calcs_filename, node, mac_int, input_fp32.astype(np.int64), weights,
                    output_multiplier, o_shift, cInputH, cInputL, output, pre_folding_oc_processing_order, amm_height)

    output = _apply_lut(output, node)

    output_pad_value = _get_output_pad_value(node)
    output = _apply_output_padding(output, node, output_pad_value, amm_height, tensors_to_mxp)

    if folding_conv_x and DEBUG_SIMULATE_FOLDING:
        if input_folding_factor_y > 0:
            output = folding_algo.get_asym_unfolded_tensor(output, folding_factor_x=0, folding_factor_y=input_folding_factor_y)
        output = folding_algo.get_asym_folded_input(output, input_folding_factor_x=1, input_folding_factor_y=input_folding_factor_y)
        oc_processing_order = list(range(output_channels * (2 ** (output_folding_factor_x - input_folding_factor_x))))
        if ((node['frontend']['input_tensor'].x_slices) % 2 != 0) and node['frontend']['input_tensor'].x_slices != 1:
            actual_width = -(-node['frontend']['output_tensor'].shape[3] // 2)
            if (actual_width != math.ceil(actual_width/MAX_GRID_WIDTH)*MAX_GRID_WIDTH) and (node['outputs'][0] not in tensors_to_mxp):
                output[:, :, :, actual_width:] = output_pad_value
        if (output.shape[3] != math.ceil(output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH) and (node['outputs'][0] not in tensors_to_mxp):
            padding_width = math.ceil(output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH - output.shape[3]
            output = np.pad(output, ((0, 0), (0, 0), (0, 0), (0, padding_width)), mode="constant", constant_values=output_pad_value)

    if 'force_unfolding_x' in node['frontend']:
        if node['frontend']['input_tensor'].x_slices == 1 and node['frontend']['output_tensor'].x_slices == 1:
            output = _shrink_x_folded_for_unfold(output, node['frontend']['output_tensor'])
        oc_processing_order = list((np.array(oc_processing_order)[::2]/2).astype(int))
        output = folding_algo.get_asym_unfolded_tensor(output, folding_factor_x=1, folding_factor_y=input_folding_factor_y)
        if node['frontend']['output_tensor'].x_slices != 2 * node['frontend']['x_slices']:
            output_width = MAX_GRID_WIDTH * node['frontend']['output_tensor'].x_slices
            output = output[:,:,:,:output_width]
        if input_folding_factor_y > 0:
            output = folding_algo.get_asym_folded_input(output, input_folding_factor_x=0, input_folding_factor_y=input_folding_factor_y)

    output = _clip_output_to_mxp(output, node, tensors_to_mxp, split_tensor_to_original_tensor_map)

    if DEBUG_SIMULATE_FOLDING and folding_conv_y and output_folding_factor_y == 0:
        if node['frontend']['stride'] != 2:
            raise ValueError('Expected this to be a strided conv. Please check...')

    if unfolding_conv_y:
        # Grid padding beyond the real height/width (y-tile rows, 16-wide grid columns) is dropped before MXP.
        _real_hw = node['frontend']['output_tensor'].get_original_shape()
        output = output[:, :, :_real_hw[2], :]
        
    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir, (node_name + '_output_tensor.nxo'))
        x_slices = node['frontend']['output_tensor'].x_slices
        # A conv produced by converting a transpose conv already emits its output in unfolded form,
        # so the nxo/nxd generation must not apply x/y folding (use 0), even though the node keeps
        # output_folding_factor_x/y set for downstream tiling. Only the export is overridden.
        export_tensor_to_nxo(debug_tensor_filename, output, oc_processing_order, is_intermediate_node=is_intermediate_node,
                            x_slices=x_slices, y_folding=output_folding_factor_y, zero_point=output_pad_value,
                            extra_lines=_export_extra_lines(node), dma_opt_disabled=_dma_opt_disabled(node))

    output_tensor = _create_output_tensor(output, node, output_folding_factor_x, output_folding_factor_y)
    intermediate_tensors[node['outputs'][0]] = output_tensor
    _maybe_export_nosplit(node, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir, is_intermediate_node)
    _debug_timing(node_name, start_time)

    return [output_tensor]

def nx_identity(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None,amm_height=MAX_GRID_HEIGHT,tensors_to_mxp=None,split_tensor_to_original_tensor_map=None, original_tensor_to_split_tensor_map=None):
    start_time = datetime.datetime.now()
    folding_conv_x, folding_conv_y, unfolding_conv_y, input_channels, output_channels = _get_folding_flags_and_channels(node)
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']
    kernel_size, stride, q_w_int8_np, padding = _select_conv_params(node, input_folding_factor_x, input_folding_factor_y)

    ic_splits = 1
    ic_groups = [list(range(input_channels))]

    input_name = node['inputs'][0]
    input = _load_input(node_name, input_name, model_inputs, intermediate_tensors,
                        input_folding_factor_x, input_folding_factor_y, use_xwrap=False, split_tensor_to_original_tensor_map=split_tensor_to_original_tensor_map, original_tensor_to_split_tensor_map=original_tensor_to_split_tensor_map, debug_dir=debug_dir)

    y_folded_input = _apply_y_folding(input.data, folding_conv_y, unfolding_conv_y)
    folded_hw = node['frontend']['input_tensor'].get_folded_shape()
    if input_folding_factor_y > 0 and folded_hw[2] < y_folded_input.shape[2]:
        y_folded_input = y_folded_input[:, :, :folded_hw[2], :]
    input_data_np = np.asarray(y_folded_input)
    pad_val = node['frontend']['input_tensor_zp']
    pad_spec = [(0,0),(0,0),(padding,padding),(padding,padding)]

    input_fp32_padded = np.pad(input_data_np.astype(np.float32, copy=False), pad_spec, mode='constant', constant_values=float(pad_val))

    if DEBUG_MAC_MULTIPLICATIONS:
        if input.real_tensor_data is not None:
            real_input_tensor = _apply_y_folding(input.real_tensor_data, folding_conv_y, unfolding_conv_y)
        else:
            real_input_tensor = input.data
        real_input_tensor_int = np.asarray(real_input_tensor, dtype=np.int64)
        real_input_tensor_padded = np.pad(real_input_tensor_int, pad_spec, mode='constant', constant_values=pad_val)

    per_ic_split_output = []
    for current_ic_split in range(ic_splits):
        if SIMULATE_IC_SPLIT:
            weights = q_w_int8_np[:,ic_groups[current_ic_split],:,:]
            input_fp32 = input_fp32_padded[:,ic_groups[current_ic_split],:,:]
            actual_input_channels = input_channels // ic_splits
        else:
            weights = q_w_int8_np
            input_fp32 = input_fp32_padded
            actual_input_channels = input_channels

        weights_t = torch.from_numpy(np.ascontiguousarray(weights.astype(np.float32)))
        input_t = torch.from_numpy(np.ascontiguousarray(input_fp32))
        mac_output = torch.nn.functional.conv2d(input_t, weights_t, bias=None, stride=stride, padding=0).numpy()
        mac_int = _wrap_mac_to_int30(mac_output)
        output_multiplier = node['frontend']['output_multiplier']
        cInputH = node['frontend']['cInputH']
        cInputL = node['frontend']['cInputL']
        o_shift = node['frontend']['o_shift']
        mac_shifted = compute_tflite_mac_shift(node_name, mac_int, output_channels, output_multiplier, cInputH, cInputL, o_shift)
        if SIMULATE_IC_SPLIT:
            per_ic_split_output.append(mac_shifted)

    if SIMULATE_IC_SPLIT:
        mac_shifted = np.zeros(per_ic_split_output[0].shape, dtype=np.int64)
        for current_split_output in per_ic_split_output:
            mac_shifted += current_split_output
    output = mac_shifted

    oc_processing_order, pre_folding_oc_processing_order = _get_oc_processing_orders(
        node, output_channels, folding_conv_x, folding_conv_y, output_folding_factor_x, output_folding_factor_y)

    if DEBUG_MAC_MULTIPLICATIONS and debug_dir:
        if (len(DEBUG_MAC_MULTIPLICATIONS_LAYERS) > 0 and node_name in DEBUG_MAC_MULTIPLICATIONS_LAYERS) or len(DEBUG_MAC_MULTIPLICATIONS_LAYERS) == 0:
            mac_calcs_filename = os.path.join(debug_dir, (node_name + '_MAC_CALCS_DEBUG.txt'))
            export_mac_calcs_to_debug_file_tflite(mac_calcs_filename, node, mac_int, real_input_tensor_padded, q_w_int8_np,
                    output_multiplier, o_shift, cInputH, cInputL, output, oc_processing_order, amm_height)

    output = _apply_lut(output, node)

    output_pad_value = _get_output_pad_value(node)
    output = _apply_output_padding(output, node, output_pad_value, amm_height, tensors_to_mxp)

    if folding_conv_x:
        output = folding_algo.get_asym_folded_input(output, input_folding_factor_x=1, input_folding_factor_y=0)
        oc_processing_order = list(range(output_channels * (2 ** (output_folding_factor_x - input_folding_factor_x))))
        if (output.shape[3] != math.ceil(output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH) and (node['outputs'][0] not in tensors_to_mxp):
            padding_width = math.ceil(output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH - output.shape[3]
            output = np.pad(output, ((0, 0), (0, 0), (0, 0), (0, padding_width)), mode="constant", constant_values=output_pad_value)

    output = _clip_output_to_mxp(output, node, tensors_to_mxp, split_tensor_to_original_tensor_map)

    if DEBUG_SIMULATE_FOLDING and folding_conv_y and output_folding_factor_y == 0:
        if node['frontend']['stride'] != 2:
            raise ValueError('Expected this to be a strided conv. Please check...')

    if unfolding_conv_y:
        # The unfold-y output carries grid padding beyond its real height and width (extra rows from y-tiling,
        # extra columns from the 16-wide grid). Crop to the logical output shape before it goes to MXP.
        _real_hw = node['frontend']['output_tensor'].get_original_shape()
        output = output[:, :, :_real_hw[2], :]

    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir, (node_name + '_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        export_tensor_to_nxo(debug_tensor_filename, output, oc_processing_order, is_intermediate_node=is_intermediate_node,
                            x_slices=x_slices, y_folding=output_folding_factor_y, zero_point=output_pad_value,
                            extra_lines=_export_extra_lines(node), dma_opt_disabled=_dma_opt_disabled(node))

    output_tensor = _create_output_tensor(output, node, output_folding_factor_x, output_folding_factor_y)
    intermediate_tensors[node['frontend']['output_tensor'].name] = output_tensor
    _maybe_export_nosplit(node, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir, is_intermediate_node)
    _debug_timing(node_name, start_time)

    return [output_tensor]

def nx_gemm(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None,amm_height=MAX_GRID_HEIGHT,tensors_to_mxp=None,split_tensor_to_original_tensor_map=None, original_tensor_to_split_tensor_map=None):
    start_time = datetime.datetime.now()
    folding_conv_x = 'force_folding_x' in node['frontend']
    folding_conv_y = 'force_folding_y' in node['frontend']

    if DEBUG_SIMULATE_FOLDING:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_folded_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape(folding_conv_x=folding_conv_x,
                                                                                        producing_node_stride=node['frontend']['stride'])
    else:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape()

    input_channels = folded_input_tensor_shape[1]
    output_channels = folded_output_tensor_shape[1]
    input_folding_factor = node['frontend']['input_folding_factor']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']
    q_w_int8_np = node['frontend']['weights_tensor'].data

    input_name = node['inputs'][0]
    input_name = _resolve_alias_name(input_name, model_inputs, intermediate_tensors,
                                      split_tensor_to_original_tensor_map, original_tensor_to_split_tensor_map, debug_dir=debug_dir)
    if input_name in model_inputs:
        input = _clone_qtensor(model_inputs[input_name])
        if input_folding_factor > 0 and DEBUG_SIMULATE_FOLDING:
            input.data = folding_algo.get_folded_input(input.data, input_folding_factor)
            input.folding_factor = input_folding_factor
    elif input_name in intermediate_tensors:
        input = intermediate_tensors[input_name]
    else:
        raise ValueError('Op %s input:%s not found!!' % (node_name, input_name))

    npd = np.expand_dims(np.asarray(input.data), (2, 3))
    converted_weights_tensor = np.expand_dims(q_w_int8_np.transpose(), (2, 3)).astype(np.float32)
    weights_t = torch.from_numpy(np.ascontiguousarray(converted_weights_tensor))
    input_t = torch.from_numpy(np.ascontiguousarray(npd.astype(np.float32, copy=False)))
    mac_output = torch.nn.functional.conv2d(input_t, weights_t, bias=None, stride=1, padding=0).numpy()
    mac_int = _wrap_mac_to_int30(mac_output)
    output = compute_tflite_mac_shift(node_name, mac_int, output_channels,
                                      node['frontend']['output_multiplier'], node['frontend']['cInputH'],
                                      node['frontend']['cInputL'], node['frontend']['o_shift'])

    output = _apply_lut(output, node)

    oc_processing_order = _get_oc_processing_order(node, output_channels)

    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir, (node_name + '_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        export_tensor_to_nxo(debug_tensor_filename, output, oc_processing_order, is_intermediate_node=is_intermediate_node,
                            x_slices=x_slices, y_folding=output_folding_factor_y, zero_point=_get_output_pad_value(node),
                            extra_lines=_export_extra_lines(node), dma_opt_disabled=_dma_opt_disabled(node))

    output = np.squeeze(output, (2, 3))
    output_tensor = _create_output_tensor(output, node, output_folding_factor_x, output_folding_factor_y)
    intermediate_tensors[node['outputs'][0]] = output_tensor
    _maybe_export_nosplit(node, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir, is_intermediate_node)
    _debug_timing(node_name, start_time)

    return [output_tensor]

def nx_add(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None,amm_height=MAX_GRID_HEIGHT,tensors_to_mxp=None,split_tensor_to_original_tensor_map=None, original_tensor_to_split_tensor_map=None):
    start_time = datetime.datetime.now()
    folding_conv_x = 'force_folding_x' in node['frontend']
    folding_conv_y = 'force_folding_y' in node['frontend']

    if DEBUG_SIMULATE_FOLDING:
        folded_input_tensor_shape = node['frontend']['input_tensors'][0].get_folded_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape(folding_conv_x=folding_conv_x,
                                            producing_node_stride=node['frontend']['stride'])
    else:
        folded_input_tensor_shape = node['frontend']['input_tensors'][0].get_original_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape()

    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']
    input_channels = folded_input_tensor_shape[1]
    output_channels = folded_output_tensor_shape[1]
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']

    if (input_folding_factor_x > 0 or input_folding_factor_y > 0) and DEBUG_SIMULATE_FOLDING:
        q_w_int8_np = node['frontend']['folded_weights_tensor'].data
    else:
        q_w_int8_np = node['frontend']['weights_tensor'].data

    add_inputs = []
    for input_name in node['inputs']:
        input_name = _resolve_alias_name(input_name, model_inputs, intermediate_tensors,
                                          split_tensor_to_original_tensor_map, original_tensor_to_split_tensor_map, debug_dir=debug_dir)
        if input_name in model_inputs:
            add_inputs.append(model_inputs[input_name])
        elif input_name in intermediate_tensors:
            add_inputs.append(intermediate_tensors[input_name])
        else:
            raise ValueError('Op %s input:%s not found!!' % (node_name, input_name))

    # A channel-split input (is_read_channel_slice) resolves above back to the FULL producer
    # output, not this piece's own range (mirrors nx_maxpool's identical re-slice). Without this,
    # concatenating add_inputs mismatches channel count against this piece's identity-sum weight.
    # Slice into a local array, not add_inputs[i].data - that may be a SHARED qTensor object with
    # sibling split pieces, and mutating it in place would corrupt those siblings later.
    add_input_data = [t.data for t in add_inputs]
    for i, input_name in enumerate(node['inputs']):
        input_tensor_meta = node['frontend']['input_tensors'][i]
        if getattr(input_tensor_meta, 'is_read_channel_slice', False):
            split_lo = input_tensor_meta.split_channel_lo
            split_cs = input_tensor_meta.shape[1]
            if add_input_data[i].shape[1] != split_cs:
                add_input_data[i] = add_input_data[i][:, split_lo:split_lo + split_cs, :, :]
        # One sibling may come from an x-folded conv (already cropped to true width by its own
        # unfold step) while another is a plain op left padded to a MAX_GRID_WIDTH multiple by
        # _apply_output_padding (nx_add has no crop step). Crop the padded side to match.
        real_shape = input_tensor_meta.get_folded_shape() if DEBUG_SIMULATE_FOLDING else input_tensor_meta.get_original_shape()
        real_height, real_width = real_shape[2], real_shape[3]
        if add_input_data[i].shape[2] > real_height:
            add_input_data[i] = add_input_data[i][:, :, :real_height, :]
        if add_input_data[i].shape[3] > real_width:
            add_input_data[i] = add_input_data[i][:, :, :, :real_width]

    input_data = np.concatenate([add_input_data[0], add_input_data[1]], axis=1)
    weights_t = torch.from_numpy(np.ascontiguousarray(q_w_int8_np.astype(np.float32)))
    input_t = torch.from_numpy(np.ascontiguousarray(input_data.astype(np.float32, copy=False)))
    mac_output = torch.nn.functional.conv2d(input_t, weights_t, bias=None, stride=1, padding=0).numpy()
    mac_int = _wrap_mac_to_int30(mac_output)
    output_multiplier = node['frontend']['output_multiplier']
    cInputH = node['frontend']['cInputH']
    cInputL = node['frontend']['cInputL']
    o_shift = node['frontend']['o_shift']
    output = compute_tflite_mac_shift(node_name, mac_int, output_channels, output_multiplier, cInputH, cInputL, o_shift)

    output = _apply_lut(output, node)

    oc_processing_order = _get_oc_processing_order(node, output_channels)
    if ((folding_conv_x and output_folding_factor_x > 0) or (folding_conv_y and output_folding_factor_y > 0)) and DEBUG_SIMULATE_FOLDING:
        channels_multiplier = 1
        if output_folding_factor_x > 0:
            channels_multiplier *= 2
        if output_folding_factor_y > 0:
            channels_multiplier *= 2
        pre_folding_oc_processing_order = list((np.array(oc_processing_order)[::channels_multiplier]/channels_multiplier).astype(int))
    else:
        pre_folding_oc_processing_order = oc_processing_order

    if DEBUG_MAC_MULTIPLICATIONS and debug_dir:
        if (len(DEBUG_MAC_MULTIPLICATIONS_LAYERS) > 0 and node_name in DEBUG_MAC_MULTIPLICATIONS_LAYERS) or len(DEBUG_MAC_MULTIPLICATIONS_LAYERS) == 0:
            mac_calcs_filename = os.path.join(debug_dir, (node_name + '_MAC_CALCS_DEBUG.txt'))
            npd_int = np.asarray(input_data, dtype=np.int64)
            export_mac_calcs_to_debug_file_tflite(mac_calcs_filename, node, mac_int, npd_int, q_w_int8_np,
                    output_multiplier, o_shift, cInputH, cInputL, output, pre_folding_oc_processing_order, amm_height)

    output_pad_value = _get_output_pad_value(node)
    output = _apply_output_padding(output, node, output_pad_value, amm_height, tensors_to_mxp)

    if ((folding_conv_x and output_folding_factor_x > 0) or (folding_conv_y and output_folding_factor_y > 0)) and DEBUG_SIMULATE_FOLDING:
        output = folding_algo.get_asym_folded_input(output, input_folding_factor_x=1, input_folding_factor_y=1)
    if DEBUG_SIMULATE_FOLDING and folding_conv_x and output_folding_factor_x == 0:
        if node['frontend']['stride'] != 2:
            raise ValueError('Expected this to be a strided conv. Please check...')
        output = folding_algo.get_asym_folded_input(output, input_folding_factor_x=1, input_folding_factor_y=0)
        output = output[:,::2,:,:]
    if DEBUG_SIMULATE_FOLDING and folding_conv_y and output_folding_factor_y == 0:
        if node['frontend']['stride'] != 2:
            raise ValueError('Expected this to be a strided conv. Please check...')
        output = folding_algo.get_asym_folded_input(output, input_folding_factor_x=0, input_folding_factor_y=1)
        output = output[:,::2,:,:]

    output = _clip_output_to_mxp(output, node, tensors_to_mxp, split_tensor_to_original_tensor_map)

    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir, (node_name + '_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        export_tensor_to_nxo(debug_tensor_filename, output, oc_processing_order, is_intermediate_node=is_intermediate_node,
                            x_slices=x_slices, y_folding=output_folding_factor_y, zero_point=output_pad_value,
                            extra_lines=_export_extra_lines(node), dma_opt_disabled=_dma_opt_disabled(node))

    output_tensor = _create_output_tensor(output, node, output_folding_factor_x, output_folding_factor_y)
    intermediate_tensors[node['outputs'][0]] = output_tensor
    _maybe_export_nosplit(node, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir, is_intermediate_node)
    _debug_timing(node_name, start_time)

    return [output_tensor]

def nx_maxpool(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None,amm_height=MAX_GRID_HEIGHT,tensors_to_mxp=None,split_tensor_to_original_tensor_map=None, original_tensor_to_split_tensor_map=None):
    start_time = datetime.datetime.now()
    folding_conv_x = 'force_folding_x' in node['frontend']
    folding_conv_y = 'force_folding_y' in node['frontend']
    unfolding_conv_y = 'force_unfolding_y' in node['frontend']

    folded_input_tensor_shape, folded_output_tensor_shape = _get_tensor_shapes(
        node, folding_conv_x, folding_conv_y, False, node['frontend']['stride'])

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']

    # A MaxPool converted 2x2->3x3 (calc_maxpool_qparams) stashes the original kernel_size/padding
    # here before overwriting to 3x3/[1,1,1,1] for HW bookkeeping. MaxPool has no weights to mask
    # extra window cells (unlike Conv), so the real reduction must use the native 2x2 window here.
    kernel_size = node['frontend'].get('maxpool_native_kernel_size', node['frontend']['kernel_size'])
    stride = node['frontend']['stride']
    padding = node['frontend'].get('maxpool_native_padding', node['frontend']['padding'])
    if not isinstance(padding, int):
        # padding is [pad_top, pad_bottom, pad_left, pad_right] (tflite_parser.py convention), but
        # torch.nn.functional.pad expects (left, right, top, bottom) for a 4-element pad on a 4D tensor.
        pad_top, pad_bottom, pad_left, pad_right = padding
        padding = [pad_left, pad_right, pad_top, pad_bottom]

    if (input_folding_factor_x > 0 or input_folding_factor_y > 0) and DEBUG_SIMULATE_FOLDING:
        q_w_int8_np = node['frontend']['folded_weights_tensor'].data
    else:
        q_w_int8_np = node['frontend']['weights_tensor'].data

    input_name = node['inputs'][0]
    input = _load_input(node_name, input_name, model_inputs, intermediate_tensors,
                        input_folding_factor_x, input_folding_factor_y, use_xwrap=False, split_tensor_to_original_tensor_map=split_tensor_to_original_tensor_map, original_tensor_to_split_tensor_map=original_tensor_to_split_tensor_map, debug_dir=debug_dir)

    y_folded_input = _apply_y_folding(input.data, folding_conv_y, unfolding_conv_y)

    if input_folding_factor_x > 0:
        y_folded_input = _shrink_x_folded_for_unfold(y_folded_input, node['frontend']['input_tensor'])

    if DEBUG_SIMULATE_FOLDING:
        unfolded_input = folding_algo.get_asym_unfolded_tensor(y_folded_input, folding_factor_x=input_folding_factor_x, folding_factor_y=input_folding_factor_y)
    else:
        unfolded_input = y_folded_input

    # A channel-split MaxPool chunk (is_read_channel_slice) has its input resolved back to the
    # FULL producer output by _load_input, so unfolded_input is still un-split. Re-apply this
    # chunk's [lo:lo+cs) slice here so it computes over its own channel range, not [0:cs).
    input_tensor_meta = node['frontend']['input_tensor']
    if getattr(input_tensor_meta, 'is_read_channel_slice', False):
        split_lo = input_tensor_meta.split_channel_lo
        split_cs = input_tensor_meta.shape[1]
        if unfolded_input.shape[1] != split_cs:
            unfolded_input = unfolded_input[:, split_lo:split_lo + split_cs, :, :]

    npd = np.asarray(unfolded_input, dtype=np.int64)
    pad_val = node['frontend']['input_tensor_zp']
    if isinstance(padding, int):
        q_input_padded = np.pad(npd, [(0,0),(0,0),(padding,padding),(padding,padding)], mode='constant', constant_values=pad_val)
    else:
        q_input_padded = torch.nn.functional.pad(torch.tensor(npd), padding, 'constant', float(pad_val)).numpy()

    m = torch.nn.MaxPool2d(kernel_size, stride=stride)
    output = m(torch.tensor(q_input_padded.astype(np.float32))).detach().numpy().astype(np.int64)
    output_channels = output.shape[1]

    output = _apply_lut(output, node)

    oc_processing_order = _get_oc_processing_order(node, output_channels)

    if DEBUG_MAC_MULTIPLICATIONS and debug_dir:
        if (len(DEBUG_MAC_MULTIPLICATIONS_LAYERS) > 0 and node_name in DEBUG_MAC_MULTIPLICATIONS_LAYERS) or len(DEBUG_MAC_MULTIPLICATIONS_LAYERS) == 0:
            mac_calcs_filename = os.path.join(debug_dir, (node_name + '_MAC_CALCS_DEBUG.txt'))
            export_mac_calcs_to_debug_file_maxpool(mac_calcs_filename, node, q_input_padded, q_w_int8_np, output, oc_processing_order, amm_height)

    output_pad_value = _get_output_pad_value(node)
    output = _apply_output_padding(output, node, output_pad_value, amm_height, tensors_to_mxp)

    if DEBUG_SIMULATE_FOLDING:
        output = folding_algo.get_asym_folded_input(output, input_folding_factor_x=output_folding_factor_x, input_folding_factor_y=output_folding_factor_y)

    output = _clip_output_to_mxp(output, node, tensors_to_mxp, split_tensor_to_original_tensor_map)

    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir, (node_name + '_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        export_tensor_to_nxo(debug_tensor_filename, output, oc_processing_order, is_intermediate_node=is_intermediate_node,
                            x_slices=x_slices, y_folding=output_folding_factor_y, zero_point=output_pad_value,
                            extra_lines=_export_extra_lines(node), dma_opt_disabled=_dma_opt_disabled(node))

    output_tensor = _create_output_tensor(output, node, output_folding_factor_x, output_folding_factor_y)
    intermediate_tensors[node['outputs'][0]] = output_tensor
    _maybe_export_nosplit(node, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir, is_intermediate_node)
    _debug_timing(node_name, start_time)

    return [output_tensor]

def nx_globalavgpool(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None,amm_height=MAX_GRID_HEIGHT,tensors_to_mxp=None,split_tensor_to_original_tensor_map=None, original_tensor_to_split_tensor_map=None):
    start_time = datetime.datetime.now()
    folded_input_tensor_shape, folded_output_tensor_shape = _get_tensor_shapes(node)

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']

    input_name = node['inputs'][0]
    input = _load_input(node_name, input_name, model_inputs, intermediate_tensors,
                        input_folding_factor_x, input_folding_factor_y, use_xwrap=False, split_tensor_to_original_tensor_map=split_tensor_to_original_tensor_map, original_tensor_to_split_tensor_map=original_tensor_to_split_tensor_map, debug_dir=debug_dir)

    if DEBUG_SIMULATE_FOLDING:
        unfolded_input = folding_algo.get_asym_unfolded_tensor(input.data, folding_factor_x=input_folding_factor_x, folding_factor_y=input_folding_factor_y)
    else:
        unfolded_input = input.data

    npd = np.asarray(unfolded_input, dtype=np.int64)
    original_height = node['frontend']['input_tensor'].get_original_shape()[2]
    original_width = node['frontend']['input_tensor'].get_original_shape()[3]
    npd = npd[:,:,0:original_height,0:original_width]
    input_scale = node['frontend']['input_tensor'].scale
    input_zero_point = node['frontend']['input_tensor'].zero_point

    dequantized_input = (npd - input_zero_point) * input_scale
    float_output = dequantized_input.mean(axis=(2, 3), keepdims=True)

    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output = (np.trunc(float_output / output_scale + 0.5) + output_zero_point).astype(np.int64)
    output_channels = folded_output_tensor_shape[1]

    output = _apply_lut(output, node)

    oc_processing_order = list(range(output_channels))

    output = folding_algo.get_asym_folded_input(output, input_folding_factor_x=output_folding_factor_x, input_folding_factor_y=output_folding_factor_y)

    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir, (node_name + '_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        if (output.shape[3] != math.ceil(output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH) and (node['outputs'][0] not in tensors_to_mxp):
            padding_width = math.ceil(output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH - output.shape[3]
            output_pad_value = _get_output_pad_value(node)
            output_padded = np.pad(output, ((0, 0), (0, 0), (0, 0), (0, padding_width)), mode="constant", constant_values=output_pad_value)
        else:
            output_padded = output

        output = _clip_output_to_mxp(output, node, tensors_to_mxp, split_tensor_to_original_tensor_map)
        export_tensor_to_nxo(debug_tensor_filename, output_padded, oc_processing_order, is_intermediate_node=is_intermediate_node,
                            x_slices=x_slices, y_folding=output_folding_factor_y, zero_point=output_pad_value,
                            extra_lines=_export_extra_lines(node), dma_opt_disabled=_dma_opt_disabled(node))

    output_tensor = _create_output_tensor(output, node, output_folding_factor_x, output_folding_factor_y)
    intermediate_tensors[node['outputs'][0]] = output_tensor
    _maybe_export_nosplit(node, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir, is_intermediate_node)
    _debug_timing(node_name, start_time)

    return [output_tensor]

def nx_resize(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None,amm_height=MAX_GRID_HEIGHT,tensors_to_mxp=None,split_tensor_to_original_tensor_map=None, original_tensor_to_split_tensor_map=None):
    start_time = datetime.datetime.now()

    if '_y' in node_name:
        output_name = node['outputs'][0]
        output_tensor = intermediate_tensors[output_name]

        # The '_x' stage pads height/width up to MAX_GRID alignment (see the padding block
        # below in the '_x' path), so this '_y' output can be wider/taller than what a
        # non-resize consumer actually expects. A following resize (chained resize_x) is fine
        # with the padded size - only trim when the next real consumer needs less.
        following_node_names = [n for n in node['outputs_consumers'][output_name]] if 'outputs_consumers' in node else \
            list(getattr(node['frontend'].get('output_tensor'), 'consumers', []) or [])
        if _CURRENT_GRAPH_NODES is not None and following_node_names:
            for following_name in following_node_names:
                following_node = _CURRENT_GRAPH_NODES.get(following_name)
                if following_node is None:
                    continue
                following_input_tensor = following_node.get('frontend', {}).get('input_tensor')
                if following_input_tensor is None:
                    continue
                expected_h, expected_w = following_input_tensor.get_original_shape()[2:4]
                cur_h, cur_w = output_tensor.data.shape[2:4]
                if expected_h < cur_h or expected_w < cur_w:
                    trimmed = _clone_qtensor(output_tensor)
                    trimmed.data = trimmed.data[:, :, :expected_h, :expected_w]
                    output_tensor = trimmed
                    intermediate_tensors[output_name] = output_tensor
                break
        return [output_tensor]
    
    if '_split' in node_name:
        output_folding_factor_x = node['frontend']['output_folding_factor_x']
        output_folding_factor_y = node['frontend']['output_folding_factor_y']
        output_scale = node['frontend']['output_tensor'].scale
        output_zero_point = node['frontend']['output_tensor'].zero_point
        output_x_slices = node['frontend']['output_tensor'].x_slices

        output_name = node['outputs'][0].split('_split')[0]
        output_tensor = intermediate_tensors[output_name]
        output = output_tensor.data
        channels_selected = node['frontend']['output_channels'] * (2 ** output_folding_factor_x) * (2 ** output_folding_factor_y)
        split_output = output[:, -channels_selected:, :, :]
        output_tensor = qTensor(split_output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
        intermediate_tensors[node['outputs'][0]] = output_tensor
        _maybe_export_nosplit(node, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir, is_intermediate_node)
        oc_processing_order = list(range(channels_selected))

        if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
            debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
            x_slices = node['frontend']['x_slices']
            if (split_output.shape[3] != math.ceil(split_output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH) and (node['outputs'][0] not in tensors_to_mxp):
                padding_width = math.ceil(split_output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH - split_output.shape[3]
                if ('activation_silu' in node['attributes']) and (node['attributes']['activation_silu'] != None):
                    output_pad_value = node['attributes']['activation_silu']['output_zp'][0]
                else:
                    output_pad_value = node['frontend']['output_tensor_zp']
                output_padded = np.pad(split_output, ((0, 0), (0, 0), (0, 0), (0, padding_width)), mode="constant", constant_values=output_pad_value)
            else:
                output_padded = split_output
            export_tensor_to_nxo(debug_tensor_filename,output_padded,oc_processing_order, is_intermediate_node = True, x_slices = x_slices, y_folding = output_folding_factor_y, zero_point=output_pad_value, extra_lines=_export_extra_lines(node), dma_opt_disabled=_dma_opt_disabled(node))
        
        return [output_tensor]

    if DEBUG_SIMULATE_FOLDING:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_folded_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape() # In case of folding conv we want to get the shape before output folding
    else:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape() # In case of folding conv we want to get the shape before output folding

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']

    #load quantized input with 0 padding
    input_name = node['inputs'][0]
    input_name = _resolve_alias_name(input_name, model_inputs, intermediate_tensors,
                                      split_tensor_to_original_tensor_map, original_tensor_to_split_tensor_map, debug_dir=debug_dir)
    if input_name in model_inputs:
        input = _clone_qtensor(model_inputs[input_name]) # We copy the input tensor sice we might fold it and dont want to change the inputs dict
        if (input_folding_factor_x>0 or input_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING:
            input.data = folding_algo.get_asym_folded_input(input.data,input_folding_factor_x=input_folding_factor_x,input_folding_factor_y=input_folding_factor_y)
            input.folding_factor_x = input_folding_factor_x
            input.folding_factor_y = input_folding_factor_y
    elif input_name in intermediate_tensors:
        input = intermediate_tensors[input_name]
    else:
        raise ValueError ('Op %s input:%s not found!!' % (node_name,input_name))

    folding_conv_y = 'force_folding_y' in node['frontend']
    unfolding_conv_y = 'force_unfolding_y' in node['frontend']
    if (folding_conv_y and output_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING: # y axis folding/unfolding is done on input read from DDR
        y_folded_input = folding_algo.get_asym_folded_input(input.data,input_folding_factor_x=0,input_folding_factor_y=1)
    elif unfolding_conv_y and DEBUG_SIMULATE_FOLDING: # y axis folding/unfolding is done on input read from DDR
        y_folded_input = folding_algo.get_asym_unfolded_tensor(input.data,folding_factor_x=0,folding_factor_y=1)
    else:
        y_folded_input = input.data

    # In order to resize using numpy repeat op we need to unfold the input tensor if needed
    if DEBUG_SIMULATE_FOLDING:
        unfolded_input = folding_algo.get_asym_unfolded_tensor(y_folded_input,folding_factor_x=input_folding_factor_x,folding_factor_y=input_folding_factor_y)
    else:
        unfolded_input = y_folded_input

    npd=np.asarray(unfolded_input,dtype=np.int64)
    original_height = node['frontend']['input_tensor'].get_original_shape()[2]
    original_width = node['frontend']['input_tensor'].get_original_shape()[3]
    original_channels = node['frontend']['input_tensor'].get_original_shape()[1]
    npd=npd[:,:,0:original_height,0:original_width]
    input_scale = node['frontend']['input_tensor'].scale
    input_zero_point = node['frontend']['input_tensor'].zero_point
    
    output = npd.repeat(2,axis=2).repeat(2,axis=3)

    output = _apply_lut(output, node)

    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output_channels = folded_output_tensor_shape[1]
    if 'backend' in node and 'oc_order' in node['backend'] and DEBUG_SIMULATE_FOLDING:
        oc_processing_order = node['backend']['oc_order']
    else:
        oc_processing_order = list(range(output_channels))
    pre_folding_oc_processing_order = oc_processing_order
        
    if ('_split' in node['outputs'][0]):
        channels_selected = node['frontend']['output_channels'] * (2 ** output_folding_factor_x) * (2 ** output_folding_factor_y)
        output = output[:, 0:channels_selected, :, :]
        oc_processing_order = list(range(channels_selected))

    output_pad_value = _get_output_pad_value(node)
    if (output.shape[3] != math.ceil(output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH) and (node['outputs'][0] not in tensors_to_mxp):
        padding_width = math.ceil(output.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH - output.shape[3]
        padding_height = 0
        if ('pad_extra_line' in node['frontend']) and node['frontend']['pad_extra_line']:
            padding_height += 1
        output = np.pad(output, ((0, 0), (0, 0), (0, padding_height), (0, padding_width)), mode="constant", constant_values=output_pad_value)
    output = _clip_output_to_mxp(output, node, tensors_to_mxp, split_tensor_to_original_tensor_map)
    
    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        output_name = node_name.replace('_x','')
        debug_tensor_filename = os.path.join(debug_dir,(output_name+'_output_tensor.nxo'))
        x_slices = math.ceil(output.shape[3]/MAX_GRID_WIDTH)
        export_tensor_to_nxo(debug_tensor_filename,output,oc_processing_order, is_intermediate_node = is_intermediate_node, x_slices = x_slices, y_folding = output_folding_factor_y, zero_point=output_pad_value, extra_lines=_export_extra_lines(node), dma_opt_disabled=_dma_opt_disabled(node))

    output_x_slices = node['frontend']['output_tensor'].x_slices
    size_y = output.shape[2] // 2
    output_x = output[:,:,:size_y,:]
    output_tensor_x = qTensor(output_x,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
    output_name = node['outputs'][0].split('_split')[0]
    intermediate_tensors[output_name] = output_tensor_x

    output_tensor = qTensor(output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
    following_node = _CURRENT_GRAPH_NODES.get(node_name.replace('_x','_y'))
    if following_node is None or ('_y' not in following_node['name']):
        raise ValueError ('Op %s should follow %s node' % (node_name.replace('_x','_y'),node_name))
    output_name = following_node['outputs'][0]
    intermediate_tensors[output_name] = output_tensor

    _debug_timing(node_name, start_time)

    if ('_split' in node['outputs'][0]):
        output_tensor = qTensor(split_output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
        intermediate_tensors[node['outputs'][0]] = output_tensor
        
    return [output_tensor]

def nx_concat(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None,amm_height=MAX_GRID_HEIGHT,tensors_to_mxp=None,split_tensor_to_original_tensor_map=None, original_tensor_to_split_tensor_map=None):
    start_time = datetime.datetime.now()
    if DEBUG_SIMULATE_FOLDING:
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape() # In case of folding conv we want to get the shape before output folding
    else:
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape() # In case of folding conv we want to get the shape before output folding

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']

    output_name = node['outputs'][0]
    if split_tensor_to_original_tensor_map and any(
            split_tensor_to_original_tensor_map.get(n) == output_name for n in node['inputs']):
        # Scatter-concat: each input's producer already wrote directly into this concat's own
        # output region, so there's nothing left to compute. Falling through would resolve every
        # input alias back to output_name and re-concat N copies onto itself.
        if output_name not in intermediate_tensors and original_tensor_to_split_tensor_map:
            _reassemble_split_tensor(output_name, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir=debug_dir)
        output_tensor = intermediate_tensors[output_name]
        if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
            output_channels = folded_output_tensor_shape[1]
            oc_processing_order = list(range(output_channels))
            debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
            x_slices = node['frontend']['x_slices']
            export_tensor_to_nxo(debug_tensor_filename,output_tensor.data,oc_processing_order, is_intermediate_node = is_intermediate_node, x_slices = x_slices, y_folding = output_folding_factor_y, zero_point=node['frontend']['output_tensor'].zero_point, extra_lines=_export_extra_lines(node), dma_opt_disabled=_dma_opt_disabled(node))
        return [output_tensor]

    #load quantized input with 0 padding
    concat_inputs=[]
    for input_idx,input_name in enumerate(node['inputs']):
        while input_name in split_tensor_to_original_tensor_map:
            input_name = split_tensor_to_original_tensor_map[input_name]
        if input_name not in model_inputs and input_name not in intermediate_tensors and original_tensor_to_split_tensor_map:
            _reassemble_split_tensor(input_name, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir=debug_dir)
        if input_name in model_inputs:
            input = model_inputs[input_name]
        elif input_name in intermediate_tensors:
            input = intermediate_tensors[input_name]
        else:
            raise ValueError ('Op %s input:%s not found!!' % (node_name,input_name))
        # A concat input that is a channel slice of a shared producer (e.g. CONV2D_77 is scattered into
        # CONCATENATION_84 and also feeds this concat) resolves to the full scattered tensor. Keep only
        # this input's [lo:lo+cs) channels so the concat doesn't re-add the other slices.
        _in_meta = node['frontend']['input_tensors'][input_idx]
        if getattr(_in_meta, 'is_read_channel_slice', False) or (input.data.shape[1] != _in_meta.shape[1] and getattr(_in_meta, 'split_channel_lo', None) is not None):
            _lo = _in_meta.split_channel_lo
            _cs = _in_meta.shape[1]
            if input.data.shape[1] != _cs:
                input = _clone_qtensor(input)
                input.data = input.data[:, _lo:_lo + _cs, :, :]
        if DEBUG_UNFOLD_INPUTS_BEFORE_CONCAT:
            input_folding_factor_x = node['frontend']['input_tensors'][input_idx].folding_factor_x
            input_folding_factor_y = node['frontend']['input_tensors'][input_idx].folding_factor_y
            unfolded_input = _clone_qtensor(input)
            unfolded_input.data = folding_algo.get_asym_unfolded_tensor(unfolded_input.data,folding_factor_x=input_folding_factor_x,folding_factor_y=input_folding_factor_y)
            unfolded_input.scale = node['frontend']['input_tensors_scale'][input_idx]
            unfolded_input.zero_point = node['frontend']['input_tensors_zp'][input_idx]
            concat_inputs.append(unfolded_input)
        else:
            folded_input = _clone_qtensor(input)
            folded_input.scale = node['frontend']['input_tensors_scale'][input_idx]
            folded_input.zero_point = node['frontend']['input_tensors_zp'][input_idx]
            concat_inputs.append(folded_input)

    output_tensor = node['frontend']['output_tensor']
    output_tensor_scale = output_tensor.scale
    output_tensor_zp = output_tensor.zero_point
    concat_inputs_data = []
    # "real" inputs mirror how FPGA actually sees them (needed so exported intermediate .nxo
    # matches FPGA): FPGA y-folds then concatenates, while the simulator concatenates then folds.
    # We reconcile this by re-arranging the concat output's oc_processing order so the following
    # Conv still "sees" a first-concat-then-fold tensor.
    real_concat_inputs_data = [] 
    for input_tensor_index,q_tensor in enumerate(concat_inputs):

        current_input_scale = q_tensor.scale
        current_input_zp = q_tensor.zero_point
        real_q_tensor_data = q_tensor.data
        # In real FPGA the concatenated tensors are folded first, violating normal fold channel
        # order (even0,odd0,even1,odd1 instead of all-even-then-all-odd). Fix by shuffling channels
        # back via the concat output's oc processing order, using output folding factor to account
        # for the DMA fold/unfold already applied to the input.
        if not DEBUG_UNFOLD_INPUTS_BEFORE_CONCAT: # If we dont simulate with unfold before concat we just need the additional fold/unfold of the input done already on the input
            force_y_folding = 'force_folding_y' in node['frontend']
            force_y_unfolding = 'force_unfolding_y' in node['frontend']
            if force_y_folding:
                real_q_tensor_data = folding_algo.get_asym_folded_input(q_tensor.data,input_folding_factor_x=0,input_folding_factor_y=1)
            if force_y_unfolding:
                real_q_tensor_data = folding_algo.get_asym_unfolded_tensor(q_tensor.data,folding_factor_x=0,folding_factor_y=1)

        else: 
            real_q_tensor_data = folding_algo.get_asym_folded_input(q_tensor.data,input_folding_factor_x=output_folding_factor_x,input_folding_factor_y=output_folding_factor_y)


        if DEBUG_SIMULATE_CONCAT_REQUANT:
            if current_input_scale!=output_tensor_scale or current_input_zp!=output_tensor_zp:
                tensor_data_float = ((q_tensor.data-current_input_zp)*current_input_scale)
                real_tensor_data_float = ((real_q_tensor_data-current_input_zp)*current_input_scale)
                rescaled_tensor_data = np.trunc(((tensor_data_float/output_tensor_scale)+output_tensor_zp)+0.5).astype(np.int64)
                real_rescaled_tensor_data = np.trunc(((real_tensor_data_float/output_tensor_scale)+output_tensor_zp)+0.5).astype(np.int64)
                concat_inputs_data.append(rescaled_tensor_data)
                real_concat_inputs_data.append(real_rescaled_tensor_data)
            else:
                concat_inputs_data.append(q_tensor.data)
                real_concat_inputs_data.append(real_q_tensor_data)
        else:
            if current_input_scale!=output_tensor_scale or current_input_zp!=output_tensor_zp:
                raise ValueError ('input q params are different from output qparams. Input name: %s' % (node['inputs'][input_tensor_index]))
            
            concat_inputs_data.append(q_tensor.data)
            real_concat_inputs_data.append(real_q_tensor_data) # In case of concat which is also y_folding the actual tensor is first folded and then concatenated while the next conv expects it to be first concatenated and then folded
    output = np.concatenate(concat_inputs_data,axis=1).astype(np.int64, copy=False)
    real_output = np.concatenate(real_concat_inputs_data,axis=1).astype(np.int64, copy=False)
    output_channels = folded_output_tensor_shape[1]
    # In concat ops we create 'simulator_oc_order' which holds the real processing order of concat inputs. this will be used to generate its nxo
    # then we use 'oc_order' to tell following layers to look on converted order.
    # The conversion is needed since if the inputs to concat are y folded the cocatenated tensor should keep the folding rules in which each y fold
    # contains data of all the inputs and only then comes the next y fold. while actually in fpga we have all y folds of input 1 then y folds of input 2 etc.
    if 'backend' in node and 'simulator_oc_order' in node['backend'] and DEBUG_SIMULATE_FOLDING:
        oc_processing_order = node['backend']['simulator_oc_order']
    elif 'backend' in node and 'oc_order' in node['backend'] and DEBUG_SIMULATE_FOLDING:
        oc_processing_order = node['backend']['oc_order']
    else:
        oc_processing_order = list(range(output_channels))
    if DEBUG_UNFOLD_INPUTS_BEFORE_CONCAT: #We fold back after concat. This will include input folding/unfolding conv
        output_folding_factor_x = node['frontend']['output_tensor'].folding_factor_x
        output_folding_factor_y = node['frontend']['output_tensor'].folding_factor_y
        output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=output_folding_factor_x,input_folding_factor_y=output_folding_factor_y)
    else:
        if DEBUG_SIMULATE_FOLDING and output_folding_factor_y==input_folding_factor_y+1:
            output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=0,input_folding_factor_y=1)
        elif DEBUG_SIMULATE_FOLDING and output_folding_factor_y!=input_folding_factor_y:
            raise ValueError ('This type of folding is not supported yet')
  
    if (node['outputs'][0] in tensors_to_mxp):
        actual_width = node['frontend']['output_tensor'].shape[3]
        output = output[:,:,:,:actual_width]

    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        export_tensor_to_nxo(debug_tensor_filename,output,oc_processing_order, is_intermediate_node = is_intermediate_node, x_slices = x_slices, y_folding = output_folding_factor_y, zero_point=output_tensor_zp, extra_lines=_export_extra_lines(node), dma_opt_disabled=_dma_opt_disabled(node))

    output_tensor = _create_output_tensor(output, node, output_folding_factor_x, output_folding_factor_y)
    output_tensor.real_tensor_data = real_output
    intermediate_tensors[node['outputs'][0]] = output_tensor
    _maybe_export_nosplit(node, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir, is_intermediate_node)
    _debug_timing(node_name, start_time)

    return [output_tensor]

def nx_flatten(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None,amm_height=MAX_GRID_HEIGHT,tensors_to_mxp=None,split_tensor_to_original_tensor_map=None, original_tensor_to_split_tensor_map=None):
    start_time = datetime.datetime.now()
    folded_input_tensor_shape, folded_output_tensor_shape = _get_tensor_shapes(node)

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']

    input_name = node['inputs'][0]
    input = _load_input(node_name, input_name, model_inputs, intermediate_tensors,
                        input_folding_factor_x, input_folding_factor_y, use_xwrap=False, split_tensor_to_original_tensor_map=split_tensor_to_original_tensor_map, original_tensor_to_split_tensor_map=original_tensor_to_split_tensor_map, debug_dir=debug_dir)

    if DEBUG_SIMULATE_FOLDING:
        unfolded_input = folding_algo.get_asym_unfolded_tensor(input.data, folding_factor_x=input_folding_factor_x, folding_factor_y=input_folding_factor_y)
    else:
        unfolded_input = input.data

    npd = np.asarray(unfolded_input, dtype=np.int64)
    output = npd.reshape(1, -1)
    output_channels = folded_output_tensor_shape[1]

    output = _apply_lut(output, node)

    oc_processing_order = list(range(output_channels))

    output = _clip_output_to_mxp(output, node, tensors_to_mxp, split_tensor_to_original_tensor_map)

    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir, (node_name + '_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        export_tensor_to_nxo(debug_tensor_filename, output, oc_processing_order, is_intermediate_node=is_intermediate_node,
                            x_slices=x_slices, y_folding=output_folding_factor_y, zero_point=_get_output_pad_value(node),
                            extra_lines=_export_extra_lines(node), dma_opt_disabled=_dma_opt_disabled(node))

    output_tensor = _create_output_tensor(output, node, output_folding_factor_x, output_folding_factor_y)
    intermediate_tensors[node['outputs'][0]] = output_tensor
    _maybe_export_nosplit(node, intermediate_tensors, original_tensor_to_split_tensor_map, debug_dir, is_intermediate_node)
    _debug_timing(node_name, start_time)

    return [output_tensor]

ops_directory = {
    'Conv': nx_conv,
    'Identity': nx_identity,
    'Add': nx_add,
    'MaxPool': nx_maxpool,
    'GlobalAveragePool': nx_globalavgpool,
    'Flatten': nx_flatten,
    'Gemm': nx_gemm,
    'Resize': nx_resize,
    'Concat': nx_concat,
}
