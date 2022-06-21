import sys
import argparse
import json
import base64
import struct
import onnx
import numpy as np
from .utils import one_elem
from .onnx_helper import get_node_index, get_node_source, get_node_inputs
from .onnx_helper import get_tensor, get_attr, get_previous_nodes, get_model_input_shape
from .onnx_infer import onnx_activations, load_input
from .onnx_bias_correction import last_weighted_layer_output_ids

np.set_printoptions(suppress=True, precision=4, linewidth=120)


NETWORK_VERSION = 0.8
DO_STRIDES = True
CLIP_UNSIGNED = True
CVI_1x1 = True
INLINE_DEPTHWISE = True


multi_input_nodes = [
        "Sum",
        "Concat",
        ]

multi_output_nodes = [
        "Split",
        ]

subgraph_nodes = [ "Conv",
                   "Gemm",
                   "Sum",
                   "Concat",
                   "LRN",
                   "ArgMax"
]


implemented_relu = [
        "Relu",
        "LeakyRelu",
        "Linear",
        "PRelu",
        "Clip",
]

implemented_pooling = [
        "MaxPool",
        "AveragePool",
        "GlobalMaxPool",
        "GlobalAveragePool",
        "ReduceMean",
]

implemented_nodes = [
    "ImageScaler",
    "Identity",
    "Pad",
    "Add",
    "Abs",
    "Sub",
    "Mul",
    "Div",
    "BatchNormalization",
    "Dropout",
    "MatMul",
    "Softmax",
    "Resize",
    "Reshape",
    "Squeeze",
    "Flatten",
    "Unsqueeze",
    "Transpose",
    "Tile",
]

implemented_nodes += implemented_relu
implemented_nodes += implemented_pooling
implemented_nodes += subgraph_nodes


def mxp_number_buffers(network, aliased_ids):
    unique_buffers = []

    for key in ['output_id', 'input_id']:
        for layer in reversed(network['layers']):
            if key in layer:
                id = layer[key]
                if id in aliased_ids:
                    id = aliased_ids[id]
                if id not in unique_buffers:
                    unique_buffers.append(id)

    for key in ['output_id', 'input_id']:
        for layer in reversed(network['layers']):
            if key in layer:
                id = layer[key]
                if id in aliased_ids:
                    id = aliased_ids[id]
                assert(id in unique_buffers)
                layer[key] = len(unique_buffers) - 1 - unique_buffers.index(id)

    return network


def mxp_size_buffers(network, aliased_ids):
    unique_buffers = []
    for key in ['output_id', 'input_id']:
        for layer in reversed(network['layers']):
            if key in layer:
                id = layer[key]
                if id in aliased_ids:
                    id = aliased_ids[id]
                if id not in unique_buffers:
                    unique_buffers.append(id)


    buffer_sizes = [0 for _ in unique_buffers]
    for x in unique_buffers:
        for layer in reversed(network['layers']):
            id = layer['input_id']
            if id in aliased_ids:
                id = aliased_ids[id]
            if x == id:
                buffer_sizes[x] = max(buffer_sizes[x], layer['input_size'])

    for x in sorted(unique_buffers):
        if buffer_sizes[x] == 0:
            for layer in reversed(network['layers']):
                id = layer['input_id']
                if id in aliased_ids:
                    id = aliased_ids[id]
                if x == id:
                    buffer_sizes[x] += layer['output_size']

    return buffer_sizes


def mxp_number_layers(network):
    for l, layer in enumerate(network['layers']):
        network['layers'][l]['layer'] = l
        network['layers'][l]['num_sublayers'] = len(layer['sublayers'])

    return network


def mxp_set_replay(network, io):
    for l, layer in enumerate(network['layers']):
        use_replay = 0
        if layer['use_replay'] == 1 and all([s['use_replay'] == 1 for s in layer['sublayers']]):
            if io and layer['input_id'] not in io['input_ids'] and layer['output_id'] not in io['output_ids']:
                use_replay = 1
        network['layers'][l]['use_replay'] = use_replay
    return network


def mxp_set_cvi(network):
    last_output_ids = last_weighted_layer_output_ids(network['layers'])
    for l, layer in enumerate(network['layers']):
        if layer['op_type'] == 'Conv':
            network['layers'][l]['use_cvi'] = 1
            if layer['group'] == layer['kernels']:
                if layer['strides'] in [[1,1], [2,2]] and layer['dilations'] == [1,1]:
                    network['layers'][l]['use_depthwise'] = 1
            if not CVI_1x1 or layer['output_id'] in last_output_ids:
                if layer['m'] == 1 and layer['n'] == 1:
                    network['layers'][l]['use_cvi'] = 0

    return network


def mxp_gemm_to_conv(network):
    for l, layer in enumerate(network['layers']):
        if layer['op_type'] == 'Gemm':
            channels = network['layers'][l]['input_size']
            kernels = network['layers'][l]['output_shape'][0]

            network['layers'][l]['op_type'] = "Conv"
            network['layers'][l]['output_shape'] = (kernels, 1, 1)
            network['layers'][l]['channels'] = channels
            network['layers'][l]['kernels'] = kernels
            network['layers'][l]['kernel_shape'] = [1,1]
            network['layers'][l]['dilations'] = [1,1]
            network['layers'][l]['strides'] = [1,1]
            network['layers'][l]['group'] = 1
            network['layers'][l]['m'] = 1
            network['layers'][l]['n'] = 1
            network['layers'][l]['use_cvi'] = 0
            network['layers'][l]['use_depthwise'] = 0
            network['layers'][l]['use_strided'] = 0

    return network


def set_unsigned(network, io):
    prev_ids, next_ids = io
    for p in prev_ids:
        network['layers'][p]['output_unsigned'] = 1
        network['layers'][p]['sublayers'] = [s for s in network['layers'][p]['sublayers'] if s['op_type'] not in ['Relu', 'Clip']]
    for n in next_ids:
        network['layers'][n]['input_unsigned'] = 1


def mxp_set_unsigned(network, use_uint8_inputs=False):
    ''' currently sets unsigned inputs if initial uint8 layer set to True'''
    ''' conv cvi outputs unsigned if Relu + followed by conv cvi'''
    ''' following conv cvi nodes get unsigned inputs (still conservative)'''
    unsigned_sets = []
    for l, layer in enumerate(network['layers']):
        network['layers'][l]['input_unsigned'] = 0
        network['layers'][l]['output_unsigned'] = 0

    if use_uint8_inputs:
        first = network['layers'][0]
        first_sublayers = [_['op_type'] for _ in first['sublayers']]
        first['input_unsigned'] = 1
        # if first['op_type'] == 'Identity' and 'Add' not in first_sublayers and 'Mul' not in first_sublayers:
        if first['op_type'] == 'Identity' and 'Add' not in first_sublayers:
            next_layer_ids = [n for n, next in enumerate(network['layers']) if first['output_id'] == next['input_id']]
            if all([network['layers'][n]['op_type'] == 'Conv' for n in next_layer_ids]):
                if all([network['layers'][n]['use_cvi'] for n in next_layer_ids]):
                    set_unsigned(network, ([0], next_layer_ids))

    for l, layer in enumerate(network['layers']):
        if layer['op_type'] == 'Conv' and layer['use_cvi']:
            next_layer_ids = [n for n, next in enumerate(network['layers']) if layer['output_id'] == next['input_id']]
            sublayer_ops = [_['op_type'] for _ in layer['sublayers']]

            if len(sublayer_ops) == 0 and len(next_layer_ids) == 1 and network['layers'][next_layer_ids[0]]['op_type'] == 'Identity':
                # special case where Conv followed immediately by Identity
                id = next_layer_ids[0]
                id_layer = network['layers'][id]
                is_concat = len([n for n in network['layers'] if id_layer['input_id'] == n['output_id']]) > 1
                if not is_concat:
                    id_next_layer_ids = [n for n, next in enumerate(network['layers']) if id_layer['output_id'] == next['input_id']]
                    if valid_unsigned(id_layer):
                        if all([network['layers'][n]['op_type'] in ['Conv'] for n in id_next_layer_ids]):
                            if all([network['layers'][n]['use_cvi'] for n in id_next_layer_ids]):
                                set_unsigned(network, ([l], next_layer_ids))
                                set_unsigned(network, ([id], id_next_layer_ids))
            else:
                if valid_unsigned(layer):
                    if all([network['layers'][n]['op_type'] in ['Conv'] for n in next_layer_ids]):
                        if all([network['layers'][n]['use_cvi'] for n in next_layer_ids]):
                            set_unsigned(network, ([l], next_layer_ids))

    return network


def valid_unsigned(layer):
    sublayer_ops = [_['op_type'] for _ in layer['sublayers']]
    valid_clip = False
    if CLIP_UNSIGNED and 'Clip' in sublayer_ops:
        clip = [_ for _ in layer['sublayers'] if _['op_type'] == 'Clip'][0]
        if clip['min'] == 0.0 and clip['max'] >= 1.0:
            valid_clip = True
    if ('Relu' in sublayer_ops or valid_clip) and ('AveragePool' not in sublayer_ops) and ('Add' not in sublayer_ops):
        return True
    return False


def fuse_layers(network, fuse_pairs):
    for p, f in fuse_pairs:
        previous_layer = network['layers'][p]
        fuse_layer = network['layers'][f]

        previous_layer['output_id'] = fuse_layer['output_id']
        previous_layer['output_description'] = fuse_layer['output_description']
        sublayers = fuse_layer['sublayers']

        fuse_layer['sublayers'] = []
        previous_layer['sublayers'].append(fuse_layer)
        previous_layer['sublayers'] += sublayers
        previous_layer['output_shape'] = fuse_layer['output_shape']

    for _, f in fuse_pairs[::-1]:
        network['layers'] = network['layers'][:f] + network['layers'][f+1:]


def mxp_inline_depthwise(network):
    if INLINE_DEPTHWISE:
        fuse_pairs = []
        for l, layer in enumerate(network['layers']):
            if layer['op_type'] == 'Conv' and layer['use_cvi'] and not layer['use_depthwise'] and layer['kernel_shape'] == [1,1]:
                next_layers = [n for n,next in enumerate(network['layers']) if layer['output_id'] == next['input_id']]
                if len(next_layers) == 1:
                    next_layer = network['layers'][next_layers[0]]
                    if next_layer['op_type'] == 'Conv' and next_layer['use_cvi'] and next_layer['use_depthwise']:
                        if next_layer['strides'] == [1,1] and next_layer['kernel_shape'] == [3,3]:
                            fuse_pairs.append((l, next_layers[0]))

        fuse_layers(network, fuse_pairs)

    return network


def mxp_remove_nop_identity(network):
    nop_pairs = []
    for l, layer in enumerate(network['layers']):
        if layer['op_type'] == 'Identity' and not 'output_strides' in layer and len(layer['sublayers']) == 0:
            next_layers = [n for n, next in enumerate(network['layers']) if layer['output_id'] == next['input_id']]
            if len(next_layers) == 1:
                n = next_layers[0]
                next_layer = network['layers'][n]
                prev_layers = [p for p, prev in enumerate(network['layers']) if next_layer['input_id'] == prev['output_id']]
                if len(prev_layers) == 1 and next_layer['op_type'] not in ['Gemm']:
                    nop_pairs.append((l, n))

    for l, n in nop_pairs:
        nop_layer = network['layers'][l]
        next_layer = network['layers'][n]

        next_layer['input_id'] = nop_layer['input_id']
        next_layer['input_description'] = nop_layer['input_description']

    for nop, _ in nop_pairs[::-1]:
        network['layers'] = network['layers'][:nop] + network['layers'][nop+1:]

    return network


def pads6(arg):
    if type(arg) == list:
        pads = arg
    else:
        node = arg
        pads = np.asarray(get_attr(node, 'pads')).tolist()
    if pads is None:
        pads = [0, 0, 0, 0, 0, 0]
    elif len(pads) == 8:
        pads = pads[1:4] + pads[5:]
    elif len(pads) == 4:
        pads = [0] + pads[0:2] + [0] + pads[2:]
    assert(len(pads) == 6)

    return pads


def get_shapes(activations, stats, node):
    if node.op_type in multi_input_nodes:
        input_shapes = [activations[n].shape[1:] for n in node.input]
    else:
        input_shapes = [activations[n].shape[1:] for n in node.input[:1]]
    output_shapes = [activations[n].shape[1:] for n in node.output]

    return input_shapes, output_shapes


def shape3d(shape):
    if len(shape) == 1:
        return 1, 1, shape[0]
    elif len(shape) == 2:
        return 1, shape[0], shape[1]
    else:
        return shape


def generate_mxp_graph(model_name, activations, stats, first_node_name, last_node_name, io_info,
                       input_type, ignore_strides=False, inline_depthwise=False, remove_nops=False, verbose=False):
    """activations+pooling merged into subgraphs"""
    network = {}
    network['layers'] = []
    network['test_input'] = None
    network['test_output'] = None
    network['scale'] = 1.0

    model = onnx.load(model_name)
    nodes = model.graph.node
    inits = model.graph.initializer

    aliased_io = {}

    idx = get_node_index(nodes, first_node_name)
    if idx == None:
        if verbose:
            print('{} does not exist\nopen {} in Netron + check spelling'.format(first_node_name, mname))
    assert(idx != None)

    last_idx = get_node_index(nodes, last_node_name)
    if last_idx == None:
        if verbose:
            print('{} does not exist\nopen {} in Netron + check spelling'.format(last_node_name, mname))
    assert(last_idx != None)

    while True:
        node = nodes[idx]
        if verbose:
            print(node.name, node.op_type)
        src_node = get_node_source(nodes, node.input[0])
        
        if src_node == None:
            input_id = node.input[0]
        else:
            input_id = src_node.output[0]
        output_id = node.output[0]


        if len(network['layers']) == 0:
            previous = None
        else:
            previous = network['layers'][-1]
        for layer in network['layers']:
            if layer['output_id'] == input_id:
                previous = layer

        input_shapes, output_shapes = get_shapes(activations, stats, node)
        assert node.op_type in multi_output_nodes or len(output_shapes) == 1, "Multi-output nodes not supported for op-type {}".format(node.op_type)
        output_shape = output_shapes[0]


        input_buffer_offset = 0
        if src_node and src_node.op_type in multi_output_nodes:
            input_id = src_node.input[0]

            _, previous_output_shapes = get_shapes(activations, stats, src_node)
            for o, output in enumerate(src_node.output):
                if output in node.input:
                    break
                input_buffer_offset += int(np.prod(previous_output_shapes[o]))


        if node.op_type == "Conv":
            c, m, n = input_shapes[0]
            kernel_shape = np.asarray(get_attr(node, 'kernel_shape')).tolist()
            assert(get_attr(node, 'pads') == None or not any(get_attr(node, 'pads')))

            group = get_attr(node, 'group')
            strides = np.asarray(get_attr(node, 'strides')).tolist()
            dilations = np.asarray(get_attr(node, 'dilations')).tolist()
            if not group:
                group = 1
            if not strides:
                strides = [1, 1]
            if not dilations:
                dilations = [1, 1]

            use_strided = 0

            accomodated_strides = False
            if strides == [1, 1] or strides == [2, 2] or strides == [4, 4]:
                accomodated_strides = True

            if DO_STRIDES and not ignore_strides and accomodated_strides:
                if (strides[0] > 1 or strides[1] > 1) and group == 1: # TODO handle depthwise as well
                    assert(previous['output_size'] == int(np.prod(input_shapes[0])))
                    use_strided = 1
                    previous['output_strides'] = strides
                    if verbose:
                        print('adding output strides to previous node')

                    m = m + (m % strides[0])
                    n = n + (n % strides[1])
                    if int(np.prod(input_shapes[0])) != int(c*m*n):
                        if verbose:
                            print('adjusting size for strided maps')
                        previous['output_size'] = int(c*4*m//strides[0]*n//strides[1])
                        previous['output_shape'] = (c*4,m//strides[0],n//strides[1])

            w = get_tensor(inits, node.input[1])
            kernels, channels, _, _ = w.shape
            if len(node.input) == 3:
                b = get_tensor(inits, node.input[2])

            conv_layer = {
                    'op_type': node.op_type,
                    'name': node.name,
                    'use_replay': 1,
                    'input_size': int(c*m*n),
                    'output_size': int(np.prod(output_shape)),
                    'input_shape':input_shapes[0],
                    'output_shape':output_shape,
                    'input_id': input_id,
                    'output_id': output_id,
                    'input_description': input_id,
                    'output_description': output_id,
                    'channels': channels * group,
                    'kernels': kernels,
                    'kernel_shape': kernel_shape,
                    'dilations': dilations,
                    'strides': strides,
                    'group': group,
                    'm': m,
                    'n': n,
                    'dma_offset': 0,
                    'input_buffer_offset': input_buffer_offset,
                    'output_buffer_offset': 0,
                    'use_cvi': 0,
                    'use_depthwise': 0,
                    'use_strided': use_strided,
                    "biases": [],
                    "weights":  [],
                    "sublayers": [],
                    }

            w = w.flatten().tolist()
            conv_layer['weights'] = base64.b64encode(struct.pack("f"*len(w), *w)).decode()

            if len(node.input) == 3:
                b = b.flatten().tolist()
            else:
                b = [0 for _ in range(kernels)]
            conv_layer['biases'] = base64.b64encode(struct.pack("f"*len(b), *b)).decode()

            network['layers'].append(conv_layer)

        elif node.op_type == "Gemm":
            w = get_tensor(inits, node.input[1])
            output_size, input_size = w.shape

            if len(node.input) == 3:
                b = get_tensor(inits, node.input[2])

            gemm_layer = {
                    'op_type': node.op_type,
                    'name': node.name,
                    'use_replay': 1,
                    'input_size': int(np.prod(input_shapes[0])),
                    'output_size': int(np.prod(output_shape)),
                    'input_shape':input_shapes[0],
                    'output_shape': output_shape,
                    'gemm_input_size': input_size,
                    'gemm_output_size': output_size,
                    'input_id': input_id,
                    'output_id': output_id,
                    'input_description': input_id,
                    'output_description': output_id,
                    'dma_offset': 0,
                    'input_buffer_offset': input_buffer_offset,
                    'output_buffer_offset': 0,
                    'channels': 1,
                    'm': 1,
                    'n': output_size,
                    "biases": [],
                    "weights":  [],
                    "sublayers": [],
                    }

            w = w.flatten().tolist()
            gemm_layer['weights'] = base64.b64encode(struct.pack("f"*len(w), *w)).decode()

            if len(node.input) == 3:
                b = b.flatten().tolist()
            else:
                b = [0 for _ in range(output_size)]
            gemm_layer['biases'] = base64.b64encode(struct.pack("f"*len(b), *b)).decode()
            network['layers'].append(gemm_layer)

        elif node.op_type in multi_input_nodes:
            node_inputs = get_previous_nodes(nodes, node)
            shapes = input_shapes

            if node.op_type == "Sum":
                assert(all([x == shapes[0] for x in shapes[1:]]))
            elif node.op_type == "Concat":
                assert(all([x[1:] == shapes[0][1:] for x in shapes[1:]]))


            output_buffer_offset = 0
            input_descriptions = []
            for n, node_input in enumerate(node_inputs):
                assert(len(node_input.output) == 1)
                noutput = node_input.output[0]
                input_descriptions.append(noutput)
                aliased_io[noutput] = input_id
                for l, layer in enumerate(network['layers']):
                    if layer['output_id'] == noutput: # if layer pointing to this node
                        network['layers'][l]['output_buffer_offset'] = output_buffer_offset # and offset appropriately
                output_buffer_offset += int(np.prod(input_shapes[n]))

            if node.op_type == "Sum":
                channels, m, n = shape3d(output_shape)
                sum_layer = {
                        'op_type': "Sum",
                        'name': node.name,
                        'use_replay': 1,
                        'input_size': int(sum([np.prod(s) for s in input_shapes])),
                        'output_size': int(np.prod(output_shape)),
                        'input_shape':input_shapes[0],
                        'output_shape':output_shape,
                        'input_id': input_id,
                        'output_id': output_id,
                        'input_description': ','.join(input_descriptions),
                        'output_description': output_id,
                        'channels': channels,
                        'm': m,
                        'n': n,
                        'dma_offset': 0,
                        'input_buffer_offset': input_buffer_offset,
                        'output_buffer_offset': 0,
                        'num_inputs': len(node.input),
                        "sublayers": [],
                        }
                network['layers'].append(sum_layer)
            elif node.op_type == "Concat":
                channels, m, n = shape3d(output_shape)
                concat_layer = {
                        'op_type': "Identity",
                        'name': node.name,
                        'use_replay': 1,
                        'input_size': int(sum([np.prod(s) for s in input_shapes])) + input_buffer_offset,
                        'output_size': int(np.prod(output_shape)),
                        'input_shape':input_shapes[0],
                        'output_shape':output_shape,
                        'input_id': input_id,
                        'output_id': output_id,
                        'input_description': ','.join(input_descriptions),
                        'output_description': output_id,
                        'channels': channels,
                        'm': m,
                        'n': n,
                        'dma_offset': 0,
                        'input_buffer_offset': input_buffer_offset,
                        'output_buffer_offset': 0,
                        "sublayers": [],
                        }
                network['layers'].append(concat_layer)

        elif node.op_type == "Identity":
            shapes = input_shapes

            channels, m, n = shape3d(output_shape)
            identity_layer = {
                    'op_type': node.op_type,
                    'name': node.name,
                    'use_replay': 1,
                    'input_size': int(sum([np.prod(s) for s in input_shapes])) + input_buffer_offset,
                    'output_size': int(np.prod(output_shape)),
                    'input_shape':input_shapes[0],
                    'output_shape':output_shape,
                    'input_id': input_id,
                    'output_id': output_id,
                    'input_description': input_id,
                    'output_description': output_id,
                    'channels': channels,
                    'm': m,
                    'n': n,
                    'dma_offset': 0,
                    'input_buffer_offset': input_buffer_offset,
                    'output_buffer_offset': 0,
                    "sublayers": [],
                    }
            network['layers'].append(identity_layer)

        elif node.op_type == "LRN":
            shapes = input_shapes
            channels, m, n = shape3d(output_shape)
            lrn_layer = {
                    'op_type': node.op_type,
                    'name': node.name,
                    'use_replay': 0,
                    'input_size': int(sum([np.prod(s) for s in input_shapes])),
                    'output_size': int(np.prod(output_shape)),
                    'input_shape':input_shapes[0],
                    'output_shape':output_shape,
                    'input_id': input_id,
                    'output_id': output_id,
                    'input_description': input_id,
                    'output_description': output_id,
                    'channels': channels,
                    'm': m,
                    'n': n,
                    'alpha': get_attr(node, 'alpha'),
                    'beta': get_attr(node, 'beta'),
                    'bias': get_attr(node, 'bias'),
                    'size': get_attr(node, 'size'),
                    'scale': 1.0,
                    'dma_offset': 0,
                    'input_buffer_offset': input_buffer_offset,
                    'output_buffer_offset': 0,
                    "sublayers": [],
                    }
            network['layers'].append(lrn_layer)

        elif node.op_type == "Scale":
            scale_sublayer = {
                    'op_type': 'Scale',
                    'name': node.name,
                    "use_replay": 1,
                    'scale': get_attr(node, 'scale'),
                    }
            previous['sublayers'].append(scale_sublayer)
            previous['output_id'] = output_id
            previous['output_description'] = output_id

        elif node.op_type in ["GlobalAveragePool", "GlobalMaxPool"]:
            assert(previous['n'] == previous['m'])
            kernel_shape = np.asarray(get_attr(node, 'kernel_shape')).tolist()
            strides = np.asarray(get_attr(node, 'strides')).tolist()
            pads = pads6(node)
            pool_sublayer = {
                    'op_type': node.op_type.replace('Global', ''),
                    'name': node.name,
                    'use_replay': 0,
                    'kernel_shape': [previous['m'], previous['n']],
                    'strides': [previous['m'], previous['n']],
                    'pads': pads,
                    }
            previous['sublayers'].append(pool_sublayer)
            previous['output_id'] = output_id
            previous['output_description'] = output_id
            previous['output_size'] = int(np.prod(output_shape))
            previous['output_shape'] = (output_shape)

        elif node.op_type in ["MaxPool", "AveragePool"]:
            kernel_shape = np.asarray(get_attr(node, 'kernel_shape')).tolist()

            if node.op_type == "AveragePool": #TODO quick fix for tf average pool quirk
                if kernel_shape[0] * kernel_shape[1] == previous['m'] * previous['n']:
                    kernel_shape = [previous['m'], previous['n']]
            strides = np.asarray(get_attr(node, 'strides')).tolist()
            if strides is None:
                strides = [ 1 for _ in kernel_shape]
            pads = pads6(node)
            pool_sublayer = {
                    'op_type': node.op_type,
                    'name': node.name,
                    'use_replay': 1,
                    'kernel_shape': kernel_shape,
                    'strides': strides,
                    'pads': pads,
                    }
            previous['sublayers'].append(pool_sublayer)
            previous['output_id'] = output_id
            previous['output_description'] = output_id
            previous['output_size'] = int(np.prod(output_shape))
            previous['output_shape'] = (output_shape)

        elif node.op_type == "PRelu":
            slope = get_tensor(inits, node.input[1])
            slope = slope.flatten().tolist()
            prelu_sublayer = {
                    'op_type': node.op_type,
                    'name': node.name,
                    'use_replay': 1,
                    'slope': slope,
                    }
            previous['sublayers'].append(prelu_sublayer)
            previous['output_id'] = output_id
            previous['output_description'] = output_id

        elif node.op_type == "LeakyRelu":
            alpha = get_attr(node, 'alpha')
            if alpha is None:
                alpha = .01
            leaky_sublayer = {
                    'op_type': node.op_type,
                    'name': node.name,
                    'use_replay': 1,
                    'alpha': alpha
                    }
            previous['sublayers'].append(leaky_sublayer)
            previous['output_id'] = output_id
            previous['output_description'] = output_id

        elif node.op_type == "Relu":
            relu_sublayer = {
                    'op_type': node.op_type,
                    'name': node.name,
                    'use_replay': 1,
                    }
            previous['sublayers'].append(relu_sublayer)
            previous['output_id'] = output_id
            previous['output_description'] = output_id

        elif node.op_type == "Clip":
            clip_sublayer = {
                    'op_type': node.op_type,
                    'name': node.name,
                    'use_replay': 1,
                    'min': float(get_tensor(inits,node.input[1])),
                    'max': float(get_tensor(inits,node.input[2])),
                    }
            previous['sublayers'].append(clip_sublayer)
            previous['output_id'] = output_id
            previous['output_description'] = output_id

        elif node.op_type == "Pad":
            pads = pads6(get_tensor(inits,node.input[1]).tolist())
            value = int(get_tensor(inits,node.input[2]))
            if value < -1:
                value = -1
            if value > 1:
                value = 1
            pad_sublayer = {
                    'op_type': node.op_type,
                    'name': node.name,
                    'use_replay': 1,
                    'value': value,
                    'pads': pads,
                    }
            previous['sublayers'].append(pad_sublayer)
            previous['output_id'] = output_id
            previous['output_description'] = output_id
            previous['output_size'] = int(np.prod(output_shape))
            previous['output_shape'] = (output_shape)
        elif node.op_type in ["Add", "Mul", "Sub", "Div"]:

            skip = False
            if node.op_type == "Mul":
                next_nodes = get_node_inputs(nodes, node.output[0])
                if output_id in io_info['output_ids']:
                    if verbose:
                        print('removing final scale node')
                    skip = True

                elif previous['op_type'] in ["LRN"]:
                    if verbose:
                        print('skipping mul after lrn')
                    array = get_tensor(inits, node.input[1])
                    if array is None:
                        array = get_tensor(inits, node.input[0])
                    previous['scale'] = float(array[0])
                    print('skipping mul after lrn', previous['scale'], previous['input_id'], previous['output_id'])

                    skip = True

                elif next_nodes[0].op_type in ["Softmax", "Sigmoid"]:
                    if verbose:
                        print('skipping mul before softmax')
                    skip = True

            array = get_tensor(inits, node.input[1])
            if array is None:
                array = get_tensor(inits, node.input[0])
                c = activations[node.input[1]].shape[1]
            else:
                c = input_shapes[0][0]

            dims = array.shape
            if node.op_type in ["Add"]: # TODO for scalar Add
                if dims == 0:
                    array = np.ones((c, 1)) * array
                    dims = array.shape

            array = array.flatten().tolist()
            if not skip:
                arithmetic_sublayer = {
                        'op_type': node.op_type,
                        'name': node.name,
                        'use_replay': 1,
                        'dims': list(dims),
                        'array': array,
                        }
                previous['sublayers'].append(arithmetic_sublayer)
            previous['output_id'] = output_id
            previous['output_description'] = output_id

        elif node.op_type in ["Abs", "Max", "Mean", "Min", "Neg", "Not"]:
            unary_sublayer = {
                    'op_type': node.op_type,
                    'name': node.name,
                    'use_replay': 1,
                    }
            previous['sublayers'].append(unary_sublayer)
            previous['output_id'] = output_id
            previous['output_description'] = output_id
            previous['output_size'] = int(np.prod(output_shape))

        elif node.op_type == "Reshape":
            dims = get_tensor(inits, node.input[1])

            if len(dims) == 4 and dims[-1] == 2:
                idx += 6
                node = nodes[idx]
                output_id = node.output[0]
                _, output_shapes = get_shapes(activations, stats, node)
                output_shape = output_shapes[0]
                channels, m, n = shape3d(output_shape)
                reorg_layer = {
                        'op_type': "Reorg",
                        'name': node.name,
                        'use_replay': 0,
                        'input_size': int(sum([np.prod(s) for s in input_shapes])),
                        'output_size': int(np.prod(output_shape)),
                        'input_shape':input_shapes[0],
                        'output_shape': output_shape,
                        'input_id': input_id,
                        'output_id': output_id,
                        'input_description': input_id,
                        'output_description': output_id,
                        'channels': channels,
                        'm': m,
                        'n': n,
                        'dma_offset': 0,
                        'input_buffer_offset': input_buffer_offset,
                        'output_buffer_offset': 0,
                        "sublayers": [],
                        "stride": int(dims[-1]),
                        }
                network['layers'].append(reorg_layer)
            else:
                previous['output_id'] = output_id
                previous['output_description'] = output_id
                output_shape = output_shapes[0]
                channels, m, n = shape3d(output_shape)
                if previous['output_shape'] != output_shape:
                    if previous['n'] == 1 and m == 1 and previous['channels'] == channels:
                        previous['m'] = m
                        previous['n'] = n

        elif node.op_type in ["Flatten",'Cast']:
            previous['output_id'] = output_id
            previous['output_description'] = output_id
        elif node.op_type == "Resize":
            scales = get_tensor(inits, node.input[2])
            assert(scales[0] == 1 and scales[1] == 1)
            mode = get_attr(node, 'mode').decode()
            assert(mode == 'nearest' or mode == 'linear')
            shapes = input_shapes[:1]
            channels, m, n = shape3d(output_shape)
            in_size= [d for d in one_elem(input_shapes)[1:]]
            replay = 0 if in_size == [1,1] else 1
            resize_layer = {
                'op_type': node.op_type,
                'name': node.name,
                'use_replay': replay,
                'input_size': int(np.prod(one_elem(input_shapes))),
                'output_size': int(np.prod(output_shape)),
                'input_shape':input_shapes[0],
                'output_shape':output_shape,
                'input_id': input_id,
                'output_id': output_id,
                'input_description': input_id,
                'output_description': output_id,
                'channels': channels,
                'mode' :mode,
                'm': m,
                'n': n,
                'dma_offset': 0,
                'input_buffer_offset': input_buffer_offset,
                'output_buffer_offset': 0,
                "sublayers": [],
                'scale': [float(scales[2]),float(scales[3])],
            }
            network['layers'].append(resize_layer)
        elif node.op_type == "Tile":
            tile = get_tensor(inits, node.input[1])[-3:]
            channels, m, n = input_shapes[0]
            replay = 1
            tile_layer = {
                'op_type': node.op_type,
                'name': node.name,
                'use_replay': replay,
                'input_size': int(np.prod(one_elem(input_shapes))),
                'output_size': int(np.prod(output_shape)),
                'input_shape':input_shapes[0],
                'output_shape':output_shape,
                'input_id': input_id,
                'output_id': output_id,
                'input_description': input_id,
                'output_description': output_id,
                'channels': channels,
                'm': m,
                'n': n,
                'dma_offset': 0,
                'input_buffer_offset': input_buffer_offset,
                'output_buffer_offset': 0,
                "sublayers": [],
                'tile': [int(x) for x in tile],
            }
            network['layers'].append(tile_layer)
        elif node.op_type == "ArgMax":
            input_shape = one_elem(input_shapes)
            channels, m, n = shape3d(input_shape)
            argmax_layer = {
                'op_type': node.op_type,
                'name': node.name,
                'use_replay': 0,
                'input_size': int(sum([np.prod(s) for s in input_shapes])),
                'output_size': int(np.prod(output_shape)),
                'input_shape':input_shapes[0],
                'output_shape':output_shape,
                'input_id': input_id,
                'output_id': output_id,
                'input_description': input_id,
                'output_description': output_id,
                'channels': channels,
                'm': m,
                'n': n,
                'dma_offset': 0,
                'input_buffer_offset': input_buffer_offset,
                'output_buffer_offset': 0,
                "sublayers": [],
                'scale': [float(scales[2]),float(scales[3])],
            }
            network['layers'].append(argmax_layer)

        elif node.op_type in ["Softmax", "Sigmoid"]:
            prev = get_previous_nodes(nodes, node)[0]
            if prev.op_type == "Mul":
                scale  = get_tensor(inits, prev.input[1])
                scale = scale.flatten().tolist()
            else:
                scale = [1.0]
            if len(scale) > 1:
                raise NotImplementedError("Broadcast scale not implemented for softmax")

            shapes = input_shapes
            channels, m, n = shape3d(output_shape)
            softmax_layer = {
                'op_type': node.op_type,
                'name': node.name,
                'use_replay': 0,
                'input_size': int(sum([np.prod(s) for s in input_shapes])),
                'output_size': int(np.prod(output_shape)),
                'input_shape':input_shapes[0],
                'output_shape':output_shape,
                'input_id': input_id,
                'output_id': output_id,
                'input_description': input_id,
                'output_description': output_id,
                'channels': channels,
                'm': m,
                'n': n,
                'dma_offset': 0,
                'input_buffer_offset': input_buffer_offset,
                'output_buffer_offset': 0,
                "sublayers": [],
                'scale': scale,
                'size': len(scale),
            }
            network['layers'].append(softmax_layer)

        elif node.op_type == "Transpose":
            shapes = input_shapes

            channels, m, n = shape3d(output_shape)
            permutation =[p-1 for p in get_attr(node, 'perm')[1:]]
            transpose_layer = {
                'op_type': node.op_type,
                'name': node.name,
                'use_replay': 1,
                'input_size': int(sum([np.prod(s) for s in input_shapes])),
                'output_size': int(np.prod(output_shape)),
                'input_shape':input_shapes[0],
                'output_shape':output_shape,
                'input_id': input_id,
                'output_id': output_id,
                'input_description': input_id,
                'output_description': output_id,
                'channels': channels,
                'm': m,
                'n': n,
                'dma_offset': 0,
                'input_buffer_offset': input_buffer_offset,
                'output_buffer_offset': 0,
                'permutation':permutation,
                "sublayers": [],
            }
            network['layers'].append(transpose_layer)
        elif node.op_type == "ReduceMean":
            assert(get_attr(node, 'axes') == [1])
            shapes = input_shapes
            channels, m, n = shape3d(output_shape)

            reducemean_layer = {
                'op_type': node.op_type,
                'name': node.name,
                'use_replay': 1,
                'input_size': int(sum([np.prod(s) for s in input_shapes])),
                'output_size': int(np.prod(output_shape)),
                'input_shape':input_shapes[0],
                'output_shape':output_shape,
                'input_id': input_id,
                'output_id': output_id,
                'input_description': input_id,
                'output_description': output_id,
                'channels': channels,
                'm0': input_shapes[0][-2],
                'm': m,
                'n': n,
                'dma_offset': 0,
                'input_buffer_offset': input_buffer_offset,
                'output_buffer_offset': 0,
                "sublayers": [],
            }
            network['layers'].append(reducemean_layer)
        elif node.op_type in multi_output_nodes:
            pass
        else:
            # dims = get_tensor(inits, node.input[1])
            raise RuntimeError('Unknown node type:{} '.format(node.op_type))

        idx += 1
        if idx > last_idx:
            break

    unsigned_network_inputs = input_type == np.uint8

    network = mxp_set_replay(network, io_info)
    network = mxp_set_cvi(network)

    network = mxp_number_buffers(network, aliased_io)
    buffers = mxp_size_buffers(network, aliased_io)

    if remove_nops:
        network = mxp_remove_nop_identity(network)

    network = mxp_set_unsigned(network, unsigned_network_inputs)
    if inline_depthwise:
        network = mxp_inline_depthwise(network)

    network = mxp_number_layers(network)

    network['num_layers'] = len(network['layers'])
    network['buffers'] = buffers

    return network


def run_generate_graph(model_src, model_stats, io_info,  input_image,
                       input_scale=1./255.,input_type=np.uint8, ignore_strides=False, inline_depthwise=False, remove_nops=False):
    assert input_type in (np.int8,np.uint8)
    stats = None
    if model_stats:
        with open(model_stats) as f:
            stats = json.load(f)

    test_input = None
    input_shape = get_model_input_shape(model_src)
    if input_image:
        test_input = load_input(input_image, input_scale, input_shape)
    activations = onnx_activations(model_src, test_input)

    graph = onnx.load(model_src).graph
    nodes = graph.node

    # generate graph from onnx
    graph_mxp = generate_mxp_graph(model_src, activations, stats, nodes[0].name, nodes[-1].name, io_info, input_type, ignore_strides, inline_depthwise, remove_nops)
    graph_mxp['version'] = NETWORK_VERSION

    # set test inputs / outputs
    test_inputs = [activations[_].flatten().tolist() for _ in io_info['input_ids']]
    test_outputs = [activations[_].flatten().tolist() for _ in io_info['output_ids']]
    graph_mxp['test_input'] = [base64.b64encode(struct.pack("f"*len(_), *_)).decode() for _ in test_inputs]
    graph_mxp['test_output'] = [base64.b64encode(struct.pack("f"*len(_), *_)).decode() for _ in test_outputs]

    return json.dumps(graph_mxp, indent=1)


def generate_graph(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_src')
    parser.add_argument('model_dst')
    parser.add_argument('model_type', choices=['yolo', 'deep', 'mnist', 'imagenet', 'classifier','generic'])
    parser.add_argument('image')
    parser.add_argument('-s', '--scale', type=float, default=1./255.)
    parser.add_argument('-f', '--first', default=None)
    parser.add_argument('-l', '--last', default=None)
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument('--stats', default=None)

    args = parser.parse_args(argv)

    stats = None
    if args.stats:
        with open(args.stats) as f:
            stats = json.load(f)

    graph = onnx.load(args.model_src).graph
    nodes = graph.node

    first_node = nodes[0] # TODO make function match run_generate_graph
    last_node = nodes[-1]
    if args.first is not None:
        try:
            first_node = [n for n in nodes if n.name == args.first][0]
        except:
            if args.verbose:
                print('first node {} not found, defaulting to {}'.format(args.first, first_node.name))
    if args.last is not None:
        try:
            last_node = [n for n in nodes if n.name == args.last][0]
        except:
            if args.verbose:
                print('last node {} not found, defaulting to {}'.format(args.last, last_node.name))

    input_shape = get_model_input_shape(args.model_src)
    test_input = load_input(args.image, args.scale, input_shape)
    activations = onnx_activations(args.model_src, test_input)
    if first_node.op_type in multi_input_nodes:
        i0 = []
        for input in first_node.input:
            i0 += activations[input].flatten().tolist()
    else:
        i0 = activations[first_node.input[0]].flatten().tolist()
    o0 = activations[last_node.output[0]].flatten().tolist()

    graph_mxp = generate_mxp_graph(args.model_src, activations, stats, first_node.name, last_node.name, None, verbose=args.verbose)

    graph_mxp['test_input'] = base64.b64encode(struct.pack("f"*len(i0), *i0)).decode()
    graph_mxp['test_output'] = base64.b64encode(struct.pack("f"*len(o0), *o0)).decode()
    graph_mxp['model'] = args.model_type
    graph_mxp['version'] = NETWORK_VERSION

    with open(args.model_dst, 'w') as f:
        json.dump(graph_mxp, f, indent=1)


if __name__ == "__main__":

    import sys
    generate_graph(sys.argv[1:])
