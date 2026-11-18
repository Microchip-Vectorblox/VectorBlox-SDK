import sys
sys.path.append('.')
import numpy as np
import networkx as nx
from common.hw_config import MULTIPLE_INPUT_OPS, OpType
from tqdm import tqdm

def update_input_scale_zp(ir, node, tensor_name):
    preceding_node = None
    preceding_requant_node = None    
    if (len(node['frontend']['preceding_nodes_params']) > 0):
        for i in range(len(node['frontend']['preceding_nodes_params'])):
            preceding_node_name = node['frontend']['preceding_nodes_params'][i][0]
            node_info = ir.graph.nodes[preceding_node_name]
            if (tensor_name in node_info['outputs']) and ('attributes' in node_info):
                if ('activation_silu' in node_info['attributes']):
                    preceding_node = node_info
                    break
                if ('requant_info' in node_info['attributes']):
                    preceding_requant_node = node_info
                    break
    if (preceding_node is not None) and (preceding_node['attributes']['activation_silu'] is not None):
        scale = preceding_node['attributes']['activation_silu']['output_scale'][0]
        zp = preceding_node['attributes']['activation_silu']['output_zp'][0]
    elif (preceding_requant_node is not None) and (preceding_requant_node['attributes']['requant_info'] is not None):
        scale = preceding_requant_node['attributes']['requant_info']['output_scale'][0]
        zp = preceding_requant_node['attributes']['requant_info']['output_zp'][0]
    else:
        scale = ir.tensors[tensor_name].scale
        zp = ir.tensors[tensor_name].zero_point 
    return scale, zp

# ============== Helper functions for common patterns ==============

def _set_input_tensor_params(ir, node):
    """Set input tensor parameters on node's frontend dict."""
    input_tensor_name = node['inputs'][0]
    input_tensor = ir.tensors[input_tensor_name]
    input_shape = input_tensor.get_original_shape()
    # inputs[0] is normally the activation, but some nodes - notably Sync nodes wrapping MXP ops
    # like reshape - list a small param tensor (e.g. a shape vector) first. When inputs[0] is not
    # activation-shaped (<2 dims), use the highest-rank input as the activation so input_channels
    # and the input scale/zp are taken from the real feature map (otherwise input_shape[1] crashes).
    if len(input_shape) < 2:
        best = None
        for cand_name in node['inputs']:
            if cand_name in ir.tensors:
                cand_shape = ir.tensors[cand_name].get_original_shape()
                if len(cand_shape) >= 2 and (best is None or len(cand_shape) > len(best[2])):
                    best = (cand_name, ir.tensors[cand_name], cand_shape)
        if best is not None:
            input_tensor_name, input_tensor, input_shape = best
    scale, zp = update_input_scale_zp(ir, node, input_tensor_name)

    node['frontend']['input_tensor'] = input_tensor
    node['frontend']['input_tensor_scale'] = scale
    node['frontend']['input_tensor_zp'] = zp
    node['frontend']['input_channels'] = input_shape[1]
    return input_tensor, input_shape

def _set_output_tensor_params(ir, node):
    """Set output tensor parameters on node's frontend dict."""
    output_tensor_name = node['outputs'][0]
    output_tensor = ir.tensors[output_tensor_name]
    output_shape = output_tensor.get_original_shape()

    node['frontend']['output_tensor'] = output_tensor
    node['frontend']['output_tensor_scale'] = output_tensor.scale
    node['frontend']['output_tensor_zp'] = output_tensor.zero_point
    node['frontend']['output_channels'] = output_shape[1]
    return output_tensor, output_shape

def _get_stride(node):
    """Extract stride from node attributes, defaulting to 1."""
    return node['attributes'].get('strides', [1])[0]

def _set_kernel_stride_padding(node):
    """Set kernel_size, stride, and padding on node's frontend dict."""
    node['frontend']['kernel_size'] = node['attributes']['kernel_shape'][0]  # assume symmetric
    node['frontend']['stride'] = _get_stride(node)
    node['frontend']['padding'] = node['attributes']['pads']

def _collect_multi_input_tensors(ir, node):
    """Collect input tensors, scales, and zero points for multi-input ops."""
    input_tensors = []
    input_tensors_scale = []
    input_tensors_zp = []
    input_shapes = []

    for tensor_name in node['inputs']:
        scale, zp = update_input_scale_zp(ir, node, tensor_name)
        input_tensors.append(ir.tensors[tensor_name])
        input_tensors_scale.append(scale)
        input_tensors_zp.append(zp)
        input_shapes.append(ir.tensors[tensor_name].get_original_shape())

    node['frontend']['input_tensors'] = input_tensors
    node['frontend']['input_tensors_scale'] = input_tensors_scale
    node['frontend']['input_tensors_zp'] = input_tensors_zp
    return input_shapes

def _set_weights_sparsity_params(ir, node, input_shape, stride, mac_multiplier=None):
    """Set weights tensor and sparsity parameters on node's frontend dict."""
    weights_tensor_name = node['inputs'][1]
    weights_tensor = ir.tensors[weights_tensor_name]

    node['frontend']['weights_tensor'] = weights_tensor
    dense_weights = weights_tensor.data.size
    sparse_weights = np.count_nonzero(weights_tensor.data)
    node['frontend']['dense_weights'] = dense_weights
    node['frontend']['sparse_weights'] = sparse_weights
    node['frontend']['weights_sparsity'] = 1 - (sparse_weights / dense_weights)

    if mac_multiplier is None:
        mac_multiplier = input_shape[2] * input_shape[3] / (stride * stride)
    dense_macs = dense_weights * mac_multiplier
    sparse_macs = sparse_weights * mac_multiplier
    node['frontend']['dense_macs'] = dense_macs
    node['frontend']['sparse_macs'] = sparse_macs
    node['frontend']['macs_sparsity'] = 1 - (sparse_macs / dense_macs)

    node['frontend']['weights_per_channel_scale'] = weights_tensor.scale
    node['frontend']['weights_per_channel_zp'] = weights_tensor.zero_point

    biases_tensor_name = node['inputs'][2]
    node['frontend']['biases_tensor'] = ir.tensors[biases_tensor_name]

# ============== Op-specific parameter functions ==============

def get_averpool_params(ir, node_name, node):
    get_conv_input_output_params(ir, node_name, node)

def get_conv_params(ir, node_name, node):
    get_conv_input_output_params(ir, node_name, node)
    get_conv_weight_params(ir, node_name, node)

def get_transpose_conv_params(ir, node_name, node):
    # TRANSPOSE_CONV is its own op type but shares CONV_2D tensor wiring: its inputs were reordered
    # into conv order [input, weights, bias] at node creation (see tflite_parser), so the input/
    # output tensors, kernel/stride/padding, and weights/bias params are set exactly like a Conv.
    # The transpose-specific hardware math (180-degree kernel flip, stride>1 input dilation, pad
    # recompute) is deferred; see vnnx_tflite.py:_process_transpose_conv_node for the reference.
    get_conv_input_output_params(ir, node_name, node)
    get_conv_weight_params(ir, node_name, node)

def get_conv_input_output_params(ir, node_name, node):
    _set_input_tensor_params(ir, node)
    _set_output_tensor_params(ir, node)
    _set_kernel_stride_padding(node)

def get_conv_weight_params(ir, node_name, node):
    input_tensor = node['frontend']['input_tensor']
    input_shape = input_tensor.get_original_shape()
    stride = _get_stride(node)
    _set_weights_sparsity_params(ir, node, input_shape, stride)

def get_gemm_params(ir, node_name, node):
    _, input_shape = _set_input_tensor_params(ir, node)
    _set_output_tensor_params(ir, node)
    _set_weights_sparsity_params(ir, node, input_shape, stride=1, mac_multiplier=1)

def get_maxpool_params(ir, node_name, node):
    get_conv_input_output_params(ir, node_name, node)

def get_general_params(ir, node_name, node):
    _set_input_tensor_params(ir, node)
    _set_output_tensor_params(ir, node)

def get_identity_params(ir, node_name, node):
    input_tensor, _ = _set_input_tensor_params(ir, node)
    output_tensor, _ = _set_output_tensor_params(ir, node)

    # A concat-requant identity deliberately changes scale/zp (input scale -> Concat scale), so do
    # not force its output qparams equal to its input. SPLIT/STRIDEDSLICE also legitimately differ.
    is_requant = node.get('attributes', {}).get('concat_requant', False)
    if 'SPLIT' not in node_name and 'STRIDEDSLICE' not in node_name and not is_requant:
        output_tensor.scale = input_tensor.scale
        output_tensor.zero_point = input_tensor.zero_point
        node['frontend']['output_tensor_scale'] = output_tensor.scale
        node['frontend']['output_tensor_zp'] = output_tensor.zero_point

def get_add_params(ir, node_name, node):
    input_shapes = _collect_multi_input_tensors(ir, node)

    if input_shapes[0] != input_shapes[1]:
        raise ValueError('At Add op (%s). Both inputs should have the same shape but got %s != %s' %
                        (node_name, str(input_shapes[0]), str(input_shapes[1])))

    node['frontend']['input_channels'] = input_shapes[0][1]  # assume same dimensions
    _set_output_tensor_params(ir, node)

def get_concat_params(ir, node_name, node):
    _collect_multi_input_tensors(ir, node)
    _set_output_tensor_params(ir, node)

def update_output_tensor_consumers_order(node):
    following_nodes_params = node['frontend']['following_nodes_params']
    output_tensor = node['frontend']['output_tensor']
    sorted_consumers_names = []
    for following_node in following_nodes_params:
        sorted_consumers_names.append(following_node[0])
    output_tensor.consumers = sorted_consumers_names

def parse_nodes_params(ir):
    sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    tqdm_iterator = tqdm(sorted_graph)
    for node_name in tqdm_iterator:
        node = ir.graph.nodes[node_name]
        tqdm_iterator.set_description('Building frontend IR, at layer %s:' % node_name)
        node['frontend']={}
        following_nodes_names = list(ir.graph.successors(node_name))
        if len(following_nodes_names)>1: # We want the following nodes struct to be lexicographical topological ordered (so that we can easily know which is last/first following node)
            sorted_following_nodes_names = sorted(following_nodes_names, key=lambda x: sorted_graph.index(x))
        else:
            sorted_following_nodes_names = following_nodes_names

        following_nodes_params = []
        for following_node_name in sorted_following_nodes_names:
            following_node = ir.graph.nodes[following_node_name]
            following_node_input_index = None
            if (len(node['outputs']) > 0):
                for output_idx in range(len(node['outputs'])):
                    if (node['outputs'][output_idx] in following_node['inputs']):
                        following_node_input_index = following_node['inputs'].index(node['outputs'][output_idx])
            following_nodes_params.append((following_node_name,following_node_input_index))

        node['frontend']['following_nodes_params'] = following_nodes_params
        preceding_nodes_names = list(ir.graph.predecessors(node_name))
        if len(preceding_nodes_names)>1: # We want the preceding nodes struct to be lexicographical topological ordered (so that we can easily know which is last/first following node)
            sorted_preceding_nodes_names = sorted(preceding_nodes_names, key=lambda x: sorted_graph.index(x))
        else:
            sorted_preceding_nodes_names = preceding_nodes_names
        preceding_nodes = []
        for preceding_node_name in sorted_preceding_nodes_names: # This will impose nodes ordering based on execution order
            preceding_node = ir.graph.nodes()[preceding_node_name]
            input_index = None
            if (len(preceding_node['outputs']) > 0):
                for output_idx in range(len(preceding_node['outputs'])):
                    if (preceding_node['outputs'][output_idx] in node['inputs']):
                        input_index = node['inputs'].index(preceding_node['outputs'][output_idx])
            preceding_nodes.append((preceding_node_name,input_index))
        node['frontend']['preceding_nodes_params'] = preceding_nodes
        if node['op_type'] == OpType.CONV:
            get_conv_params(ir,node_name,node)
        elif node['op_type'] == OpType.TRANSPOSE_CONV:
            get_transpose_conv_params(ir,node_name,node)
        elif node['op_type'] == OpType.AVERAGEPOOL:
            get_averpool_params(ir,node_name,node)
        elif node['op_type'] == OpType.MAXPOOL:
            get_maxpool_params(ir,node_name,node)
        elif node['op_type'] == OpType.ADD:
            get_add_params(ir,node_name,node)
        elif node['op_type'] == OpType.CONCAT:
            get_concat_params(ir,node_name,node)
        elif node['op_type'] == OpType.GEMM:
            get_gemm_params(ir,node_name,node)
        elif node['op_type'] == OpType.IDENTITY:
            get_identity_params(ir,node_name,node)
        else:
            get_general_params(ir,node_name,node)
        update_output_tensor_consumers_order(node)

    return ir

# Need to impose equal quant params to ops that have multiple inputs and are unable to requant (e.g. Concat).
# In such case we change output scale of previous op where needed, if previous op is also non requantize op (i.e. its input qparams are equal to output qparams)
# We insert identity node which is able to requant (This is actually a requant node)

def fuse_concat_requant_into_folding_producers(ir):
    """A Concat input whose producer performs x-folding/unfolding (force_folding_x /
    force_unfolding_x, only known once calc_nodes_folding_factor has run) can't have a standalone
    requant Identity conv appended after it - that breaks the folding chain. insert_concat_input_
    requant_nodes (tflite_parser.py) already inserted such an Identity as a fallback before folding
    factors were known; here, for exactly those producers, remove the standalone Identity and fuse
    its requant directly onto the producer's own output via requant_info (mirrors the Resize case
    handled at parse time, and consumed generically by apply_activation_silu_lut)."""
    graph = ir.graph
    for rq_node_name in list(graph.nodes):
        rq_node = graph.nodes[rq_node_name]
        if not rq_node.get('attributes', {}).get('concat_requant'):
            continue
        in_name = rq_node['inputs'][0]
        rq_tensor_name = rq_node['outputs'][0]
        in_tensor = ir.tensors[in_name]
        producer_name = in_tensor.producer
        if producer_name is None or not graph.has_node(producer_name):
            continue
        producer_node = graph.nodes[producer_name]
        if not (producer_node['frontend'].get('force_folding_x') or producer_node['frontend'].get('force_unfolding_x')):
            continue

        rq_tensor = ir.tensors[rq_tensor_name]
        out_scale, out_zp = rq_tensor.scale, rq_tensor.zero_point
        producer_node.setdefault('attributes', {})['requant_info'] = {
            'input_scale': [in_tensor.scale], 'input_zp': [in_tensor.zero_point],
            'output_scale': [out_scale], 'output_zp': [out_zp],
        }

        # Fuse producer's output scale/zp to the Concat's - update the same objects/caches the
        # generic qparams pipeline reads later (update_nodes_qparams already ran, so its cached
        # input_tensors_scale/zp on the Concat must be patched directly too).
        in_tensor.scale, in_tensor.zero_point = out_scale, out_zp
        producer_node['frontend']['output_tensor_scale'] = out_scale
        producer_node['frontend']['output_tensor_zp'] = out_zp

        for concat_name in list(rq_tensor.consumers):
            concat_node = graph.nodes[concat_name]
            for idx, concat_in in enumerate(concat_node['inputs']):
                if concat_in == rq_tensor_name:
                    concat_node['inputs'][idx] = in_name
                if 'input_tensors_scale' in concat_node.get('frontend', {}):
                    for i, t in enumerate(concat_node['frontend']['input_tensors']):
                        if t.name == rq_tensor_name or t is rq_tensor:
                            concat_node['frontend']['input_tensors'][i] = in_tensor
                            concat_node['frontend']['input_tensors_scale'][i] = out_scale
                            concat_node['frontend']['input_tensors_zp'][i] = out_zp
            graph.remove_edge(rq_node_name, concat_name)
            graph.add_edge(producer_name, concat_name)
            if rq_tensor_name in ir.tensors:
                in_tensor.consumers = [c if c != rq_node_name else concat_name for c in in_tensor.consumers]

        graph.remove_edge(producer_name, rq_node_name)
        graph.remove_node(rq_node_name)
        del ir.tensors[rq_tensor_name]

    return ir


def update_nodes_qparams(ir):
    sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    tqdm_iterator = tqdm(sorted_graph)
    for node_name in tqdm_iterator:
        node = ir.graph.nodes[node_name]
        tqdm_iterator.set_description('Getting quantization params, at layer %s:' % node_name)
        if node['op_type'] in MULTIPLE_INPUT_OPS:
            input_tensors_scales = []
            input_tensors_zp = []
            for input_tensor in node['frontend']['input_tensors']:
                scale, zp = update_input_scale_zp(ir, node, input_tensor.name)
                input_tensors_scales.append(scale)
                input_tensors_zp.append(zp)
            node['frontend']['input_tensors_scale'] = input_tensors_scales
            node['frontend']['input_tensors_zp'] = input_tensors_zp
        else:
            node['frontend']['input_tensors_scale'] = node['frontend']['input_tensor'].scale
            node['frontend']['input_tensors_zp'] = node['frontend']['input_tensor'].zero_point
        node['frontend']['output_tensor_scale'] = node['frontend']['output_tensor'].scale
        node['frontend']['output_tensor_zp'] = node['frontend']['output_tensor'].zero_point

    return ir
