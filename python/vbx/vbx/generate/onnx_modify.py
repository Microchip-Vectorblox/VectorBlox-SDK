import onnx
import numpy as np
import argparse
import math

from .onnx_helper import get_node_source_index, get_node_index, get_node_inputs
from .onnx_helper import get_node_source, get_previous_nodes
from .onnx_helper import has_attr, get_attr, set_attr, get_tensor, set_tensor, get_shape
from .onnx_helper import onnx_save_graph

from .onnx_infer import onnx_allclose


np.set_printoptions(suppress=True, precision=4, linewidth=120)


NETWORK_VERSION = 0.8

injected_nodes = 0

multipath_nodes = [
        "Sum",
        "Concat",
        ]

subgraph_nodes = [
        "Conv",
        "Gemm",
        "Sum",
        "Concat",
        "LRN",
]


def onnx_get_io_ids(model_name):
    model = onnx.load(model_name)
    graph = model.graph

    input_ids = [_.name for _ in graph.input]
    output_ids = [_.name for _ in graph.output]

    return input_ids, output_ids


def inject_identity_before(nodes, next):
    buf = '{}_i'.format(next.name)
    inject = onnx.helper.make_node('Identity', ['tmp'], [buf], name=buf)
    return inject_node_before(nodes, next, inject)


def inject_identity_between(nodes, prev, next):
    global injected_nodes
    buf = '{}_i{}'.format(next.name, injected_nodes)
    injected_nodes += 1

    idx = list(next.input).index(prev.output[0])
    next.input[idx] = buf
    inject = onnx.helper.make_node('Identity', [prev.output[0]], [buf], name=buf)

    idx = get_node_index(nodes, next.name)
    nodes = nodes[:idx] + [inject] + nodes[idx:]
    return nodes


def inject_node_before(nodes, next, inject):
    assert(len(inject.output) == 1)
    inject.input[0] = next.input[0]
    next.input[0] = inject.output[0]

    idx = get_node_index(nodes, next.name)
    nodes = nodes[:idx] + [inject] + nodes[idx:]
    return nodes


def inject_identity_after(nodes, prev):
    buf = '{}_pi'.format(prev.name)
    inject = onnx.helper.make_node('Identity', [buf], ['tmp'], name=buf)
    return inject_node_after(nodes, prev, inject)


def inject_softmax_after(nodes, prev):
    buf = '{}_soft'.format(prev.name)
    inject = onnx.helper.make_node('Softmax', [buf], ['tmp'], name=buf)
    return inject_node_after(nodes, prev, inject)


def inject_node_after(nodes, prev, inject):
    assert(len(inject.output) == 1)
    inject.output[0] = prev.output[0]
    prev.output[0] = inject.input[0]

    idx = get_node_index(nodes, prev.name)
    nodes = nodes[:idx+1] + [inject] + nodes[idx+1:]
    return nodes


def onnx_inject_scale_before_special(nodes, inits):
    global injected_nodes

    nodes_to_inject = []
    for n, node in enumerate(nodes):
        if node.op_type in multipath_nodes or node.op_type in ['LRN', 'Softmax', 'Sigmoid']:
            prev = get_node_source(nodes, node.input[0])
            if prev is None or prev.op_type not in ['Split']:
                inputs = list(node.input)
                for i, iname in enumerate(inputs):
                    buf = '{}_s{}'.format(iname, injected_nodes)
                    injected_nodes += 1
                    wname = buf + '_weights'
                    x = np.asarray([1.0,])
                    weights = onnx.helper.make_tensor(wname, onnx.TensorProto.FLOAT, x.shape, x.flatten())

                    scale = onnx.helper.make_node('Mul', [iname, wname], [buf], name=buf)

                    node.input.remove(iname)
                    node.input.extend([buf])
                    inits.extend([weights])

                    nodes_to_inject += [scale]

    for node in nodes_to_inject:
        idx = get_node_source_index(nodes, node.input[0]) + 1
        nodes = nodes[:idx] + [node] + nodes[idx:]

    return nodes, inits


def onnx_inject_scale_after_special(nodes, inits):
    global injected_nodes
    nodes_to_inject = []
    for n, node in enumerate(nodes):
        next_nodes = get_node_inputs(nodes, node.output[0])
        if len(next_nodes) and node.op_type in ['LRN']:
            output = node.output[0]
            buf = '{}_post_s{}'.format(output, injected_nodes)
            injected_nodes += 1
            wname = buf + '_weights'
            x = np.asarray([1.0,])
            weights = onnx.helper.make_tensor(wname, onnx.TensorProto.FLOAT, x.shape, x.flatten())
            inits.extend([weights])

            scale = onnx.helper.make_node('Mul', [output, wname], [buf], name=buf)

            for next in next_nodes:
                next.input[0] = buf

            nodes_to_inject += [scale]

    for node in nodes_to_inject:
        idx = get_node_source_index(nodes, node.input[0]) + 1
        nodes = nodes[:idx] + [node] + nodes[idx:]

    return nodes, inits

def onnx_inject_scale_after_sum(nodes, inits):
    global injected_nodes
    nodes_to_inject = []
    for n, node in enumerate(nodes):
        if node.op_type == 'Sum':
            output = node.output[0]
            buf = '{}_s{}'.format(output, injected_nodes)
            injected_nodes += 1
            wname = buf + '_weights'
            x = np.asarray([1.0,])
            weights = onnx.helper.make_tensor(wname, onnx.TensorProto.FLOAT, x.shape, x.flatten())
            inits.extend([weights])

            scale = onnx.helper.make_node('Mul', [output, wname], [buf], name=buf)

            next_nodes = get_node_inputs(nodes, node.output[0])
            for next_node in next_nodes:
                for i,inp in enumerate(next_node.input):
                    if inp == output:
                        next_node.input[i] = buf

            nodes_to_inject += [scale]

    for node in nodes_to_inject:
        idx = get_node_source_index(nodes, node.input[0]) + 1
        nodes = nodes[:idx] + [node] + nodes[idx:]

    return nodes, inits


def onnx_inject_final_scale(nodes, inits, outputs):
    global injected_nodes
    for output in outputs:
        node = get_node_source(nodes, output.name)
        oname = node.output[0]

        buf = oname + '_scale'
        wname = buf + '_W{}'.format(injected_nodes)
        injected_nodes += 1
        x = np.asarray([1.0,])
        weights = onnx.helper.make_tensor(wname, onnx.TensorProto.FLOAT, x.shape, x.flatten())

        scale = onnx.helper.make_node('Mul', [oname, wname], [buf], name=buf)

        nodes.extend([scale])
        inits.extend([weights])

        for output in outputs:
            if output.name == oname:
                output.name = buf

    return nodes, inits, outputs


def onnx_inject_initial_identity(nodes, inputs):
    global injected_nodes

    for input in inputs:
        nnodes = get_node_inputs(nodes, input.name)
        assert(len(nnodes)==1)
        node=nnodes[0]

        iname = input.name
        buf = 'begin{}_{}'.format(injected_nodes,iname)
 
        injected_nodes += 1

        id = onnx.helper.make_node('Identity', [buf], [iname], name=buf)

        for node in [id]:
            idx = get_node_source_index(nodes, node.input[0]) + 1
            nodes = nodes[:idx] + [node] + nodes[idx:]

        for input in inputs:
            if input.name == iname:
                input.name = buf


    return nodes, inputs


def onnx_inject_identity_before_scaled_special(nodes):
    nodes_to_inject = []
    prev = None

    for n, node in enumerate(nodes):
        # if node.op_type in ["LRN", "Softmax", "Sigmoid", "Transpose"]:
        if node.op_type in ["LRN", "Softmax", "Sigmoid"]:
            if prev.op_type == 'Mul':
                nodes_to_inject += [prev]
            else:
                nodes_to_inject += [node]
        prev = node

    for node in nodes_to_inject:
        nodes = inject_identity_before(nodes, node)

    return nodes


def onnx_inject_identity_after_split(nodes):
    nodes_to_inject = []
    prev = None

    for n, node in enumerate(nodes):
        if node.op_type in ["Split"]:
            for output in node.output:
                next_nodes = get_node_inputs(nodes, output)
                if not (next_nodes is None):
                    for next_node in next_nodes:
                        nodes_to_inject += [next_node]
    for node in nodes_to_inject:
        nodes = inject_identity_before(nodes, node)

    return nodes


def onnx_inject_identity_after_special(nodes):
    nodes_to_inject = []
    prev = None

    for n, node in enumerate(nodes):
        # if node.op_type in ["Concat", "LRN",'Resize','Transpose']:
        # if node.op_type in ["Concat", "LRN",'Resize']:
        if node.op_type in ["LRN",'Resize']:
            nodes_to_inject += [node]
    for node in nodes_to_inject:
        nodes = inject_identity_after(nodes, node)

    return nodes


def onnx_inject_identity_after_biased(nodes):
    nodes_to_inject = []
    prev = None

    for n, node in enumerate(nodes):
        if node.op_type in ["Conv", "Gemm", "Add"]:
            next_nodes = get_node_inputs(nodes, node.output[0])
            if len(next_nodes) == 1:
                next_node = next_nodes[0]
                if next_node.op_type not in subgraph_nodes:
                    nodes_to_inject += [(node, next_node)]
                #     if next_node.op_type == 'Mul':
                #         next_nodes = get_node_inputs(nodes, next_node.output[0])
                #         if next_nodes[0].op_type not in ['LRN', 'Softmax']:
                #             nodes_to_inject += [(node, next_node)]
                #     else:
                #         nodes_to_inject += [(node, next_node)]
                elif next_node.op_type in multipath_nodes:
                    nodes_to_inject += [(node, next_node)]

    for (prev, next) in nodes_to_inject:
        nodes = inject_identity_between(nodes, prev, next)

    return nodes


def onnx_inject_identity_before_specified_nodes(nodes, specified_nodes):
    nodes_to_inject = []
    prev = None

    for n, node in enumerate(nodes):
        if node.name in specified_nodes:
            nodes_to_inject += [node]

    for node in nodes_to_inject:
        nodes = inject_identity_before(nodes, node)

    return nodes


def onnx_inject_identity_before_multipath_subgraph(nodes):
    nodes_to_inject = []
    for n, node in enumerate(nodes):
        next_nodes = get_node_inputs(nodes, node.output[0])
        if len(next_nodes) > 1:
            for next_node in next_nodes:
                if next_node.op_type not in subgraph_nodes and next_node.op_type not in multipath_nodes:
                    nodes_to_inject += [next_node]

    for node in nodes_to_inject:
        nodes = inject_identity_before(nodes, node)

    return nodes


def onnx_inject_identity_before_concat(nodes):
    nodes_to_inject = []
    for n, node in enumerate(nodes):
        next_nodes = get_node_inputs(nodes, node.output[0])
        if len(next_nodes) > 1:
            for next_node in next_nodes:
                if next_node.op_type == "Concat":
                    nodes_to_inject += [(node, next_node)]

    for (prev, next) in nodes_to_inject:
        nodes = inject_identity_between(nodes, prev, next)

    return nodes


def onnx_inject_identity_before_multipath_strided_conv(nodes):
    nodes_to_inject = []
    for n, node in enumerate(nodes):
        next_nodes = get_node_inputs(nodes, node.output[0])
        if len(next_nodes) > 1: # multi_path
            for next_node in next_nodes:
                if next_node.op_type == 'Conv':
                    strides = get_attr(next_node, 'strides')
                    if strides[0] > 1 or strides[1] > 1:
                        nodes_to_inject += [next_node]

    for node in nodes_to_inject:
        nodes = inject_identity_before(nodes, node)

    return nodes


def onnx_breakout_pad(nodes,inits):
    nodes_to_inject = []
    for n, node in enumerate(nodes):
        if node.op_type == "MaxPool":
            if get_attr(node, 'auto_pad'):
                if get_attr(node, 'auto_pad') == b'SAME_UPPER' or get_attr(node, 'auto_pad') == b'SAME_LOWER':
                    kh_diff = get_attr(node, 'kernel_shape')[0] - get_attr(node, 'strides')[0]
                    kw_diff = get_attr(node, 'kernel_shape')[1] - get_attr(node, 'strides')[1]
                    if kh_diff > 0 or kw_diff > 0:
                        if get_attr(node, 'auto_pad') == b'SAME_UPPER':
                            pads = [0, 0, kh_diff, kw_diff]
                        elif get_attr(node, 'auto_pad') == b'SAME_LOWER':
                            pads = [kh_diff, kw_diff, 0, 0]
                        set_attr(node, 'auto_pad', b'VALID')
                        node.attribute.extend([onnx.helper.make_attribute("pads", pads)])

            elif get_attr(node, 'ceil_mode'):
                set_attr(node, 'ceil_mode', 0)
                kh_diff = get_attr(node, 'kernel_shape')[0] - get_attr(node, 'strides')[0]
                kw_diff = get_attr(node, 'kernel_shape')[1] - get_attr(node, 'strides')[1]
                pads = [0, 0, kh_diff, kw_diff]
                set_attr(node, 'pads', pads)

            pads = get_attr(node, 'pads')
            if pads and sum(pads) > 0:
                pads_next = [0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]]
                set_attr(node, 'pads', [0 for _ in pads])
                buf = '{}_p'.format(node.name)
                pads_tensor = onnx.helper.make_tensor('pads_{}'.format(buf),
                                                      onnx.TensorProto.INT64,
                                                      (len(pads_next),),
                                                      pads_next)
                value_tensor = onnx.helper.make_tensor('value_{}'.format(buf),
                                                       onnx.TensorProto.FLOAT,
                                                       (1,),
                                                       [-999])
                inits.append(pads_tensor)
                inits.append(value_tensor)

                pad = onnx.helper.make_node('Pad', [node.input[0],"pads_"+buf,"value_"+buf], [buf], mode="constant", name=buf)
                node.input[0] = buf
                nodes_to_inject += [pad]


        elif node.op_type == "Conv":
            if has_attr(node, "auto_pad") and not has_attr(node, "pads"):
                if get_attr(node, 'auto_pad') == b'SAME_UPPER' or get_attr(node, 'auto_pad') == b'SAME_LOWER':
                    attr = [_ for _ in node.attribute if _.name == "auto_pad"][0]
                    node.attribute.remove(attr)
                    kernel_shape = get_attr(node, 'kernel_shape')
                    assert kernel_shape[0] == kernel_shape[1]
                    k = math.floor(kernel_shape[0] / 2)
                    node.attribute.extend([onnx.helper.make_attribute("pads", [k, k, k, k])])
                    set_attr(node, 'auto_pad', b'VALID')

            pads = get_attr(node, 'pads')
            if pads and sum(pads) > 0:
                pads_next = [0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]]
                pads_value = np.float32(0)
                set_attr(node, 'pads', [0 for _ in pads])
                buf = '{}_p'.format(node.name)
                pads_tensor = onnx.helper.make_tensor('pads_{}'.format(buf),
                                                      onnx.TensorProto.INT64,
                                                      (len(pads_next),),
                                                      pads_next)
                value_tensor = onnx.helper.make_tensor('value_{}'.format(buf),
                                                       onnx.TensorProto.FLOAT,
                                                       (1,),
                                                       [0])
                inits.append(pads_tensor)
                inits.append(value_tensor)
                pad = onnx.helper.make_node('Pad', [node.input[0],"pads_"+buf,"value_"+buf], [buf], name=buf)
                node.input[0] = buf
                nodes_to_inject += [pad]
    for node in nodes_to_inject:
        idx = get_node_source_index(nodes, node.input[0]) + 1
        nodes = nodes[:idx] + [node] + nodes[idx:]

    return nodes,inits


def replace_gemm_with_conv(nodes, inits, gemm_node):
    node_id = gemm_node.name

    conv_inputs = ['pre_{}'.format(node_id)]
    conv_inputs.append('g2c_W{}'.format(node_id))
    conv_outputs = ['post_{}'.format(node_id)]
    if len(gemm_node.input) == 3:
        conv_inputs.append('g2c_b{}'.format(node_id))

    pre = onnx.helper.make_node(
            'Reshape',
            inputs=[gemm_node.input[0], 'shape_pre_{}'.format(node_id)],
            outputs=['pre_{}'.format(node_id)],
            name = str('pre_{}'.format(node_id))
            )

    post = onnx.helper.make_node(
            'Reshape',
            inputs=['post_{}'.format(node_id), 'shape_post_{}'.format(node_id)],
            outputs=gemm_node.output,
            name = str('post_{}'.format(node_id))
            )

    conv = onnx.helper.make_node(
        'Conv',
        inputs = conv_inputs,
        outputs = conv_outputs,
        strides = [1,1],
        dilations = [1,1],
        kernel_shape = [1,1],
        pads = [0,0,0,0],
        name = node_id,
    )
    idx = get_node_index(nodes, gemm_node.name)

    nodes = nodes[:idx] + [pre, conv, post] + nodes[idx:]
    weights = get_tensor(inits, gemm_node.input[1])
    tensor = onnx.helper.make_tensor('g2c_W{}'.format(node_id),
            onnx.TensorProto.FLOAT,
            [weights.shape[0],weights.shape[1],1,1],
            weights.flatten().tolist(),
            )
    inits.append(tensor)

    if len(gemm_node.input) == 3:
        biases = get_tensor(inits, gemm_node.input[2])
        tensor = onnx.helper.make_tensor('g2c_b{}'.format(node_id),
                onnx.TensorProto.FLOAT,
                [biases.shape[0],],
                biases.flatten().tolist(),
                )
        inits.append(tensor)

    reshape_pre = np.array([-1,weights.shape[1],1,1], dtype=np.int64)
    tensor = onnx.helper.make_tensor('shape_pre_{}'.format(node_id),
            onnx.TensorProto.INT64,
            reshape_pre.shape,
            reshape_pre.tolist(),
            )
    inits.append(tensor)



    next_nodes = get_node_inputs(nodes, gemm_node.output[0])
    if len(next_nodes) == 0 or weights.shape[0]==1:
        reshape_post = np.array([-1,weights.shape[0]], dtype=np.int64)
    else:
        reshape_post = np.array([-1,1,1,weights.shape[0]], dtype=np.int64)

    tensor = onnx.helper.make_tensor('shape_post_{}'.format(node_id),
            onnx.TensorProto.INT64,
            reshape_post.shape,
            reshape_post.tolist(),
            )
    inits.append(tensor)

    return nodes, inits


def clean_init(inits, name):
    init = [_ for _ in inits if _.name == name]
    if len(init) == 1:
        inits.remove(init[0])
    return inits


def onnx_optimize_replace_gemm_width_conv(nodes, inits):
    cut_nodes = []
    for node in nodes:
        if node.op_type == 'Gemm':
            w = get_tensor(inits, node.input[1])
            cut_nodes.append(node)
            nodes, inits = replace_gemm_with_conv(nodes, inits, node)
    for cut_node in cut_nodes:
        for input in cut_node.input:
            inits = clean_init(inits, input)
        nodes.remove(cut_node)
    return nodes, inits


def move_activations_up_reshape(nodes, reshape_node):
    node_id = reshape_node.name

    node = reshape_node
    last_node = reshape_node
    activation_node = None

    while True:
        next_nodes = get_node_inputs(nodes, node.output[0])
        if len(next_nodes) != 1:
            break
        else:
            if len(next_nodes) == 1 and next_nodes[0].op_type == 'Reshape':
                node = next_nodes[0]
                last_node = node
            else: 
                if next_nodes[0].op_type in ['Relu']:#TODO
                    activation_node = next_nodes[0]
                break

    if not activation_node == None:
        bump = 'bumped_{}'.format(node_id)
        act_input = activation_node.input
        act_input[0] = reshape_node.input[0]
        act = onnx.helper.make_node(
                activation_node.op_type,
                inputs=act_input,
                outputs=[bump],
                name = str(bump)
                )
        for i, o in enumerate(activation_node.output):
            last_node.output[i] = o
        reshape_node.input[0] = bump
        nodes.remove(activation_node)
        idx = get_node_index(nodes, reshape_node.name)
        nodes = nodes[:idx] + [act] + nodes[idx:]

    return nodes


def onnx_optimize_activations_up_reshape(nodes):
    reshape_nodes = []
    for node in nodes:
        if node.op_type == 'Reshape':
            reshape_nodes.append(node)
    for node in reshape_nodes:
        nodes = move_activations_up_reshape(nodes, node)
    return nodes


def replace_reshape(nodes, inits, value_info, outputs, reshape_node):
    node_id = reshape_node.name

    prev = get_node_source(nodes, reshape_node.input[0])
    if prev != None:
        start = get_shape(value_info, prev.output[0])

        reshape_nodes = []
        next_node = reshape_node
        while True:
            reshape_nodes.append(next_node)
            next_nodes = get_node_inputs(nodes, next_node.output[0])
            if len(next_nodes) == 1 and next_nodes[0].op_type == 'Reshape':
                next_node = next_nodes[0]
            else:
                break

        if len(reshape_nodes) > 1:
            stop = get_shape(value_info, reshape_nodes[-1].output[0])
            if stop is None:
                stop = get_shape(outputs, reshape_nodes[-1].output[0])
            reshape = None
            not_needed = start == stop
            if len(stop) == 1:
                if next_node.op_type == 'Flatten':
                    not_needed = True
            if not_needed:
                for i, o in enumerate(reshape_nodes[-1].output):
                    prev.output[i] = reshape_nodes[-1].output[i]
            else:
                old_reshape = reshape_node.input[1]
                if len(stop) == 1:
                    reshape = onnx.helper.make_node(
                            'Flatten',
                            inputs=[reshape_node.input[0]],
                            outputs=reshape_nodes[-1].output,
                            name = reshape_node.name
                            )
                else:
                    stop = np.asarray([-1]+stop, dtype=np.int64)
                    # stop[0] = -1
                    tensor = onnx.helper.make_tensor('simple_{}'.format(reshape_node.name),
                            onnx.TensorProto.INT64,
                            stop.shape,
                            stop.tolist(),
                            )
                    inits.append(tensor)

                    reshape_node.input[1] = 'simple_{}'.format(reshape_node.name)
                    reshape = onnx.helper.make_node(
                            'Reshape',
                            inputs=reshape_node.input,
                            outputs=reshape_nodes[-1].output,
                            name = reshape_node.name
                            )
                idx = get_node_index(nodes, reshape_node.name)
                inits = clean_init(inits, old_reshape)

            for n, node in enumerate(reshape_nodes):
                nodes.remove(node)
                if not_needed or n > 0:
                    inits = clean_init(inits, node.input[1])

            if reshape:
                nodes = nodes[:idx] + [reshape] + nodes[idx:]

    return nodes, inits


def onnx_optimize_replace_reshape(nodes, inits, value_info, outputs):
    reshape_nodes = []
    for node in nodes:
        if node.op_type == 'Reshape':
            reshape_nodes.append(node)
    for node in reshape_nodes:
        nodes, inits = replace_reshape(nodes, inits, value_info, outputs, node)
    return nodes, inits


def fuse_conv_nodes(nodes, inits, pair):

    prev, node = pair
    prev_w = get_tensor(inits, prev.input[1])
    w = get_tensor(inits, node.input[1])
    if len(node.input) == 3:
        b = get_tensor(inits, node.input[2])
    prev_w0 = prev_w*w
    inits = set_tensor(inits, prev.input[1], prev_w0)
    if len(prev.input) == 3:
        prev_b = get_tensor(inits, prev.input[2])
        prev_b0 = prev_b*w
        if not b is None:
            prev_b0 += b 
        inits = set_tensor(inits, prev.input[2], prev_b0)
    elif not b is None:
        node_id = prev.name
        prev.input.append(node.input[2])

    next_nodes = get_node_inputs(nodes, node.output[0])
    for next in next_nodes:
        for i, input in enumerate(next.input):
            if input == node.output[0]:
                next.input[i] = prev.output[0]
    nodes.remove(node)
    inits = clean_init(inits, node.input[1])

    return nodes, inits


def onnx_optimize_fuse_conv(nodes, inits):
    conv_nodes = []
    for node in nodes:
        if node.op_type == 'Conv':
            next_nodes = get_node_inputs(nodes, node.output[0])
            if len(next_nodes) == 1 and next_nodes[0].op_type in ['Conv']:#TODO
                wshape = get_tensor(inits, node.input[1]).shape
                next_wshape = get_tensor(inits, next_nodes[0].input[1]).shape
                if next_wshape[1:] == (1,1,1): #TODO currently catching missed norm layers
                    conv_nodes.append((node, next_nodes[0]))
    for fuse_pair in conv_nodes:
        nodes, inits = fuse_conv_nodes(nodes, inits, fuse_pair)
    return nodes, inits

def remove_RGB2BGR(nodes, inits, set):
    split_node = set[0]
    concat_node = set[1]
    weighted_nodes = set[2]
    assert(split_node.output[0] == concat_node.input[2])
    assert(split_node.output[1] == concat_node.input[1])
    assert(split_node.output[2] == concat_node.input[0])

    # change input for node following split-concat to be split input
    post_concat = get_node_inputs(nodes, concat_node.output[0])
    for node in post_concat:
        node.input[0] = split_node.input[0]

    # swap channels for weights
    for node in weighted_nodes:
        w = get_tensor(inits, node.input[1])
        if w.shape[-3] == 3:
            w_next = w.copy()
            if w.ndim == 4:
                w_next[:,0] = w[:,2]
                w_next[:,2] = w[:,0]
            else:
                w_next[0] = w[2]
                w_next[2] = w[0]
            inits = set_tensor(inits, node.input[1], w_next)

    # delete split + concat
    init = [_ for _ in inits if _.name == split_node.input[1]][0]
    inits.remove(init)
    nodes.remove(split_node)
    nodes.remove(concat_node)

    return nodes, inits


def onnx_optimize_remove_RGB2BGR(nodes, inits):
    sets = []
    for node in nodes:
        if node.op_type == 'Concat':
            previous_nodes = get_previous_nodes(nodes, node)
            if len(previous_nodes) == 1:
                prev = previous_nodes[0]
                if prev.op_type == 'Split':
                    weighted_nodes = []
                    next_nodes = get_node_inputs(nodes, node.output[0])
                    while len(next_nodes) == 1:
                        next = next_nodes[0]
                        if next.op_type in ['Add', 'Mul', 'Conv']:
                            weighted_nodes.append(next)
                        if next.op_type == 'Conv':
                            # if not channel-mul conv, then can invert channels
                            group = get_attr(next, 'group')
                            w = get_tensor(inits, next.input[1])
                            if group is None or group != w.shape[0]:
                                sets.append([prev, node, weighted_nodes])
                                break
                        next_nodes = get_node_inputs(nodes, next.output[0])
                        
    for set in sets:
        nodes, inits = remove_RGB2BGR(nodes, inits, set)

    return nodes, inits


def onnx_optimize_nhw1_input(nodes, inputs, inits): # nhwc no longer handled by openvino 2022.1
    for input in inputs:
        if input.type.tensor_type.shape.dim[3].dim_value == 1:
            inodes = get_node_inputs(nodes, input.name)
            assert(len(inodes) == 1)
            node = inodes[0]

            scale_shift_nodes = []
            while 1:
                next_nodes = get_node_inputs(nodes, node.output[0])
                if len(next_nodes) == 1:
                    node = next_nodes[0]
                    if node in ['Mul', 'Add']:
                        scale_shift_nodes.append(node)
                    else:
                        break
            if node.op_type == 'Reshape':
                shape = get_tensor(inits, node.input[1])
                if len(shape) == 2 and shape[0] == -1:
                    # change input shape to transposed shape
                    channels = input.type.tensor_type.shape.dim[3].dim_value
                    height = input.type.tensor_type.shape.dim[1].dim_value
                    width = input.type.tensor_type.shape.dim[2].dim_value

                    input.type.tensor_type.shape.dim[1].dim_value = channels
                    input.type.tensor_type.shape.dim[2].dim_value = height
                    input.type.tensor_type.shape.dim[3].dim_value = width

                    for snode in scale_shift_nodes:
                        w = get_tensor(inits, snode.input[1])
                        w_t = w.transpose((0,3,1,2))
                        inits = set_tensor(inits, snode.input[1], w_t)
    return nodes, inputs, inits


def onnx_optimize_remove_nhwc_input_transpose(nodes, inputs):

    for input in inputs:
        inodes = get_node_inputs(nodes, input.name)
        assert(len(inodes) == 1)
        node = inodes[0]
        if node.op_type == 'Transpose' and get_attr(node, 'perm') == [0,3,1,2]:
            # set node after tranpose to be the input
            next_nodes = get_node_inputs(nodes, node.output[0])
            assert(len(next_nodes) == 1)
            next_nodes[0].input[0] = node.input[0]
            nodes.remove(node)

            # change input shape to transposed shape
            channels = input.type.tensor_type.shape.dim[3].dim_value
            height = input.type.tensor_type.shape.dim[1].dim_value
            width = input.type.tensor_type.shape.dim[2].dim_value

            input.type.tensor_type.shape.dim[1].dim_value = channels
            input.type.tensor_type.shape.dim[2].dim_value = height
            input.type.tensor_type.shape.dim[3].dim_value = width


    return nodes, inputs


def onnx_optimize_remove_nhwc_output_transpose(nodes, outputs):
    for output in outputs:
        node = get_node_source(nodes, output.name)
        if node.op_type == 'Transpose' and get_attr(node, 'perm') == [0,2,3,1]:
            # set node before tranpose to be the output
            previous_nodes = get_previous_nodes(nodes, node)
            assert(len(previous_nodes) == 1)
            previous_nodes[0].output[0] = node.output[0]
            nodes.remove(node)

            # change output shape to transposed shape
            channels = output.type.tensor_type.shape.dim[3].dim_value
            height = output.type.tensor_type.shape.dim[1].dim_value
            width = output.type.tensor_type.shape.dim[2].dim_value

            output.type.tensor_type.shape.dim[1].dim_value = channels
            output.type.tensor_type.shape.dim[2].dim_value = height
            output.type.tensor_type.shape.dim[3].dim_value = width


    return nodes, outputs


def onnx_optimize_remove_input_identity(nodes, inputs):
    for input in inputs:
        inodes = get_node_inputs(nodes, input.name)
        assert(len(inodes) == 1)
        node = inodes[0]
        if node.op_type == 'Identity':
            # set node after tranpose to be the input
            next_nodes = get_node_inputs(nodes, node.output[0])
            assert(len(next_nodes) == 1)
            next_nodes[0].input[0] = node.input[0]
            nodes.remove(node)

    return nodes, inputs



def onnx_optimize_graph(model_src, model_dst, verbose=False):
    model = onnx.load(model_src)
    graph = model.graph
    inputs, outputs = graph.input, graph.output
    nodes, inits = graph.node, graph.initializer

    
    nodes, inputs = onnx_optimize_remove_input_identity(nodes, inputs)
    nodes, inputs, inits = onnx_optimize_nhw1_input(nodes, inputs, inits) # nhwc no longer handled by openvino 2022.1
    nodes, inputs = onnx_optimize_remove_nhwc_input_transpose(nodes, inputs)
    nodes, outputs = onnx_optimize_remove_nhwc_output_transpose(nodes, outputs)

    nodes, inits = onnx_optimize_remove_RGB2BGR(nodes, inits)
    nodes, inits = onnx_optimize_replace_gemm_width_conv(nodes, inits)
    nodes = onnx_optimize_activations_up_reshape(nodes)

    onnx_save_graph(nodes, inputs, outputs, inits, model_dst, model_dst, infer_shapes=True)
    model = onnx.load(model_dst)
    graph = model.graph
    inputs, outputs = graph.input, graph.output
    nodes, inits = graph.node, graph.initializer
    value_info = graph.value_info

    nodes, inits = onnx_optimize_replace_reshape(nodes, inits, value_info, outputs)
    nodes, inits = onnx_optimize_fuse_conv(nodes, inits)

    for output in outputs:
        output.type.tensor_type.shape.dim[0].dim_param = "N"
    onnx_save_graph(nodes, inputs, outputs, inits, model_dst, "opt")

    if verbose:
        print(onnx_allclose(model_src, model_dst))


def onnx_pre_graph(model_src, model_dst, verbose=False):
    model = onnx.load(model_src)
    graph = model.graph
    inputs, outputs = graph.input, graph.output
    nodes, inits = graph.node, graph.initializer

    nodes, inits = onnx_inject_scale_before_special(nodes, inits)
    nodes, inits = onnx_inject_scale_after_special(nodes, inits)

    nodes, inits, outputs = onnx_inject_final_scale(nodes, inits, outputs)

    onnx_save_graph(nodes, inputs, outputs, inits, model_dst, "pre")

    if verbose:
        print(onnx_allclose(model_src, model_dst))


def onnx_post_graph(model_src, model_dst, verbose=False):
    model = onnx.load(model_src)
    graph = model.graph
    nodes, inputs, outputs, inits = graph.node, graph.input, graph.output, graph.initializer

    #nodes, inits, outputs = onnx_remove_mul(nodes, inits, outputs)

    nodes, inits = onnx_breakout_pad(nodes, inits)

    # nodes = onnx_inject_identity_before_concat(nodes)

    nodes = onnx_inject_identity_before_multipath_subgraph(nodes)
    # nodes = onnx_inject_identity_before_scaled_special(nodes)

    nodes, inputs = onnx_inject_initial_identity(nodes, inputs)
    nodes = onnx_inject_identity_after_special(nodes)
    nodes = onnx_inject_identity_after_split(nodes)

    nodes, inits, outputs = onnx_remove_mul(nodes, inits, outputs)

    # nodes = inject_softmax_after(nodes, nodes[-1])
    nodes = onnx_inject_identity_before_multipath_strided_conv(nodes)

    onnx_save_graph(nodes, inputs, outputs, inits, model_dst, "post")

    if verbose:
        print(onnx_allclose(model_src, model_dst))


def onnx_isolate_biases_graph(model_src, model_dst, verbose=False):
    model = onnx.load(model_src)
    graph = model.graph
    nodes, inputs, outputs, inits = graph.node, graph.input, graph.output, graph.initializer

    nodes = onnx_inject_identity_after_biased(nodes)

    onnx_save_graph(nodes, inputs, outputs, inits, model_dst, "bias")

    if verbose:
        print(onnx_allclose(model_src, model_dst))


def onnx_remove_mul(nodes, inits, outputs):
    nodes_to_remove = []

    for n, node in enumerate(nodes):
        if node.op_type == "Mul":
            # get dst, array and src names
            dst = node.output[0]


            w = get_tensor(inits, node.input[1])
            arr = node.input[1]
            src = node.input[0]

            if w is None:
                w = get_tensor(inits, node.input[0])
                arr = node.input[0]
                src = node.input[1]

            # remove node and connect src to dst
            if np.array_equal(w, np.ones(w.shape)):
                next_nodes = get_node_inputs(nodes, dst)
                for next in next_nodes:
                    for i, input in enumerate(next.input):
                        if input == dst:
                            next.input[i] = src
                nodes_to_remove.append(node)
                init = [_ for _ in inits if _.name == arr][0]
                inits.remove(init)
                # print('removing node {}'.format(node.name))

                for output in outputs:
                    if output.name == node.output[0]:
                        output.name = node.input[0]
            else:
                next_nodes = get_node_inputs(nodes, dst)
                if len(next_nodes) == 1 and next_nodes[0].op_type not in ['Softmax', 'Sigmoid']:
                    prev = get_node_source(nodes, node.input[0])

                    prev_conv = get_previous_conv(nodes, prev)
                    if prev_conv is not None:
                        prev_w = get_tensor(inits, prev_conv.input[1])
                        prev_w0 = prev_w*w
                        inits = set_tensor(inits, prev_conv.input[1], prev_w0)
                        if len(prev_conv.input) == 3:
                            prev_b = get_tensor(inits, prev_conv.input[2])
                            prev_b0 = prev_b*w
                            inits = set_tensor(inits, prev_conv.input[2], prev_b0)

                        next_nodes = get_node_inputs(nodes, dst)
                        for next in next_nodes:
                            for i, input in enumerate(next.input):
                                if input == dst:
                                    next.input[i] = src
                        nodes_to_remove.append(node)
                        init = [_ for _ in inits if _.name == arr][0]
                        inits.remove(init)

    for node in nodes_to_remove:
        idx = get_node_index(nodes, node.name)
        del nodes[idx]

    return nodes, inits, outputs


def get_previous_conv(nodes, node):
    if node is None or len(get_node_inputs(nodes, node.output[0])) > 1:
        return None
    elif node.op_type in ['Conv']:
        return node
    elif node.op_type in ['Tile', 'Resize', 'MaxPool', 'Relu', 'LeakyRelu', 'PRelu', 'Reshape']:
        prev = get_node_source(nodes, node.input[0])
        return get_previous_conv(nodes, prev)
    
    return None


def onnx2vnnx(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['pre', 'post'])
    parser.add_argument('model_src')
    parser.add_argument('model_dst')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(argv)

    if args.mode == 'pre':
        onnx_pre_graph(args.model_src, args.model_dst, args.verbose)
    elif args.mode == 'post':
        onnx_post_graph(args.model_src, args.model_dst, args.verbose)
    elif args.mode == 'bias':
        onnx_isolate_biases_graph(args.model_src, args.model_dst, args.verbose)


if __name__ == "__main__":

    import sys
    onnx2vnnx(sys.argv[1:])
