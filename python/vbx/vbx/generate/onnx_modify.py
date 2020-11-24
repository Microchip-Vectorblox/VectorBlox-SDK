import onnx
import numpy as np
import argparse
import math

from .onnx_helper import get_node_source_index, get_node_index, get_node_inputs
from .onnx_helper import get_node_source, get_previous_nodes
from .onnx_helper import has_attr, get_attr, set_attr, get_tensor, set_tensor
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
        if node.op_type in multipath_nodes or node.op_type in ['LRN', 'Softmax']:
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
        if node.op_type in ['LRN']:
            output = node.output[0]
            buf = '{}_post_s{}'.format(output, injected_nodes)
            injected_nodes += 1
            wname = buf + '_weights'
            x = np.asarray([1.0,])
            weights = onnx.helper.make_tensor(wname, onnx.TensorProto.FLOAT, x.shape, x.flatten())
            inits.extend([weights])

            scale = onnx.helper.make_node('Mul', [output, wname], [buf], name=buf)

            next_nodes = get_node_inputs(nodes, node.output[0])
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
    node = nodes[-1]
    oname = node.output[0]

    buf = str(int(oname) + 1)
    wname = buf + '_W{}'.format(injected_nodes)
    injected_nodes += 1
    x = np.asarray([1.0,])
    weights = onnx.helper.make_tensor(wname, onnx.TensorProto.FLOAT, x.shape, x.flatten())

    scale = onnx.helper.make_node('Mul', [oname, wname], [buf], name=buf)

    # nodes = nodes + [scale]
    nodes.extend([scale]) # TODO this works
    inits.extend([weights])

    for output in outputs:
        if output.name == oname:
            output.name = buf

    return nodes, inits, outputs


def onnx_inject_initial_identity(nodes, inputs):
    global injected_nodes
    node = nodes[0]
    iname = node.input[0]
    buf = 'begin{}'.format(injected_nodes)
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
        if node.op_type in ["LRN", "Softmax", "Transpose"]:
            if prev.op_type == 'Mul':
                nodes_to_inject += [prev]
            else:
                nodes_to_inject += [node]
        prev = node

    for node in nodes_to_inject:
        nodes = inject_identity_before(nodes, node)

    return nodes


def onnx_inject_identity_after_special(nodes):
    nodes_to_inject = []
    prev = None

    for n, node in enumerate(nodes):
        if node.op_type in ["Concat", "LRN",'Resize']:
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
                    # print('injecting identity before', next_node.name)
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

    nodes = onnx_inject_identity_before_concat(nodes)

    nodes = onnx_inject_identity_before_multipath_subgraph(nodes)
    # nodes = onnx_inject_identity_before_scaled_special(nodes)

    nodes, inputs = onnx_inject_initial_identity(nodes, inputs)
    nodes = onnx_inject_identity_after_special(nodes)

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
                if len(next_nodes) == 1 and next_nodes[0].op_type not in ['Softmax']:
                    prev = get_node_source(nodes, node.input[0])
                    if prev is not None and len(get_node_inputs(nodes, prev.output[0])) == 1 and prev.op_type in ['Conv']:
                        # print('can absorb!')
                        prev_w = get_tensor(inits, prev.input[1])
                        prev_w0 = prev_w*w
                        inits = set_tensor(inits, prev.input[1], prev_w0)
                        if len(prev.input) == 3:
                            prev_b = get_tensor(inits, prev.input[2])
                            prev_b0 = prev_b*w
                            inits = set_tensor(inits, prev.input[2], prev_b0)

                        next_nodes = get_node_inputs(nodes, dst)
                        for next in next_nodes:
                            for i, input in enumerate(next.input):
                                if input == dst:
                                    next.input[i] = src
                        nodes_to_remove.append(node)
                        init = [_ for _ in inits if _.name == arr][0]
                        inits.remove(init)
                        # print('removing node {}'.format(node.name))


    for node in nodes_to_remove:
        idx = get_node_index(nodes, node.name)
        del nodes[idx]

    return nodes, inits, outputs


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
