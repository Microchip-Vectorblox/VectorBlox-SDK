import argparse
import os
import json
import onnx
import numpy as np

from .onnx_helper import get_node_inputs, get_node_source, get_previous_nodes
from .onnx_helper import get_tensor, set_tensor, get_attr, set_attr
from .onnx_helper import onnx_save_graph, load_statistics
from .onnx_infer import onnx_random_input, onnx_infer, onnx_activations
from .utils import one_elem

np.set_printoptions(suppress=True, precision=4, linewidth=120)


weighted_nodes = ["Conv", "Gemm", "Mul"]
memoized_weighted = {}


def get_previous_weighted(nodes, node):
    global memoized_weighted

    if node.name in memoized_weighted:
        return memoized_weighted[node.name]

    previous = get_previous_nodes(nodes, node)
    if len(previous) == 0:
        previous_weighted = None
    elif len(previous) == 1:
        if previous[0].op_type not in weighted_nodes:
            previous_weighted = get_previous_weighted(nodes, previous[0])
        else:
            previous_weighted = [previous[0]]
    else:
        multi = []
        if node.op_type == "Sum":
            multi += [node]
        for p in previous:
            multi += get_previous_weighted(nodes, p)
        previous_weighted = multi

    memoized_weighted[node.name] = previous_weighted

    return previous_weighted


def get_previous_max(nodes, inits, maximums, node):
    previous_weighted = get_previous_weighted(nodes, node)
    if previous_weighted is None:
        return maximums[node.input[0]]

    elif len(previous_weighted) == 1:
        if previous_weighted[0].name in maximums:
            return get_node_max(nodes,inits,maximums,previous_weighted[0])
        else:
            if previous_weighted[0].op_type == 'Mul':
                next_nodes = get_node_inputs(nodes, previous_weighted[0].output[0])
                if len(next_nodes) == 1:
                    if next_nodes[0].op_type in ['Softmax', 'Sigmoid']:
                        return np.asarray(1.)
                    if next_nodes[0].op_type in ['Argmax']:
                        return np.asarray(1.)
                    if next_nodes[0].op_type in ['LRN']:
                        return maximums[next_nodes[0].name]

            return get_previous_max(nodes, inits, maximums, previous_weighted[0])
    else:
        #a = max([maximums[pw.name] for pw in previous_weighted])
        b = max([get_node_max(nodes,inits,maximums,pw) for pw in previous_weighted])
        #print(a,b,np.abs(a-b))
        return b

def get_node_max(nodes,inits,maximums,node):
    relu_after = False
    clip_after = False
    next_node = node
    
    while True:
        next_nodes = get_node_inputs(nodes, next_node.output[0])
        if len(next_nodes) != 1:
            #if branch point or no next layer stop searching for relu
            break;
        next_node=next_nodes[0]
        if next_node.op_type in ('LeakyRelu', 'Relu'):
            relu_after = next_node
        elif next_node.op_type in ('Clip', ):
            clip_after = next_node
        elif next_node.op_type not in ('Pad','Pool'):
            #nodes that don't impact maximums
            break

    if relu_after:
        return maximums[relu_after.output[0]]
        # return min(maximums[node.output[0]], maximums[relu_after.output[0]])
    elif clip_after:
        return maximums[clip_after.output[0]]
        # return min(maximums[node.output[0]], maximums[clip_after.output[0]])
    else:
        return maximums[node.output[0]]


def onnx_normalize_convolve(inits, arr, group, previous_max, next_max):
    w = get_tensor(inits, arr)
    w0 = (w * previous_max / next_max).astype(w.dtype)
    return set_tensor(inits, arr, w0)


def onnx_normalize_scalar_add(inits, arr, next_max):
    b = get_tensor(inits, arr)
    b0 = (b / next_max).astype(b.dtype)
    return set_tensor(inits, arr, b0)


def onnx_normalize_scalar_mul(inits, arr, previous_max, next_max):
    w = get_tensor(inits, arr)
    w0 = (w * previous_max / next_max).astype(w.dtype)
    return set_tensor(inits, arr, w0)


def onnx_normalize_graph(nodes, inits, maximums, verbose=False):
    clip_nodes = [node for node in nodes if node.op_type == "Clip"]
    for node in clip_nodes:
        clip_max = get_tensor(inits, node.input[2])
        maximums[one_elem(node.output)] = clip_max

    for node in nodes:
        inputs = get_node_inputs(nodes, node.output[0])
        if len(inputs):
            next = inputs[0]
        else:
            next = None
        prev = get_previous_max(nodes, inits, maximums, node)
        if node.op_type == "Conv":
            current = get_node_max(nodes,inits,maximums,node)
            if verbose:
                print('next', node.name, node.op_type, prev, current)

            inits = onnx_normalize_convolve(inits, node.input[1], get_attr(node, 'group'), prev, current)
            if len(node.input) == 3:
                inits = onnx_normalize_scalar_add(inits, node.input[2], current)

        elif node.op_type == "Gemm":
            current = maximums[node.output[0]]

            if verbose:
                print('next', node.name, node.op_type, prev, current)

            inits = onnx_normalize_scalar_mul(inits, node.input[1], prev, current)
            if len(node.input) == 3:
                inits = onnx_normalize_scalar_add(inits, node.input[2], current)

        elif node.op_type == "Mul":  # constant mul
            current = None

            if node == nodes[-1]:
                if verbose:
                    print('last node')
                current = prev
            elif next and next.op_type in ["Softmax", "Sigmoid"]:
                current = np.ones(prev.shape) # denormalize

            elif next and next.op_type in ["LRN"]:
                current = np.ones(prev.shape) # denormalize

                alpha = get_attr(next, 'alpha')
                alpha_ = float(alpha * prev**2)
                set_attr(next, 'alpha', alpha_)

                current = prev

            w = get_tensor(inits, node.input[1])
            if w is None:
                src = node.input[0]
            else:
                src = node.input[1]

            if current is None:
                if node.output[0] in maximums:
                    current = maximums[node.output[0]]
                else:
                    current = prev * np.max(get_tensor(inits, src).flatten())

                if next and next.op_type == "Add":
                    # print('Optimize SS', current, get_tensor(inits, src).flatten(), get_tensor(inits, next.input[1]).flatten())
                    pass

            previous = get_previous_nodes(nodes, node)
            if len(previous) == 1 and previous[0].op_type == 'LRN':
                current = maximums[node.input[0]]

            if verbose:
                print('next', node.name, node.op_type, prev, current)

            inits = onnx_normalize_scalar_mul(inits, src, prev, current)
            maximums[node.name] = current

        elif node.op_type == "Add":  # constant add
            if verbose:
                print('next', node.name, node.op_type, prev)
            inits = onnx_normalize_scalar_add(inits, node.input[1], prev)
            maximums[node.name] = np.asarray(prev)
            # maximums[node.name] = np.asarray([prev, prev])

        elif node.op_type == "Clip":
            if verbose:
                print('clip', node.name, node.op_type, prev)
            clip_min = get_tensor(inits, node.input[1])
            clip_max = get_tensor(inits, node.input[2])
            set_tensor(inits, node.input[1], clip_min/prev)
            set_tensor(inits, node.input[2], clip_max/prev)

    return nodes, inits


def onnx_normalize_graph_multi(nodes, inits, maximums, verbose=False):

    for node in nodes:
        previous = get_previous_nodes(nodes, node)
        if len(previous) > 1:
            prev_max = get_previous_max(nodes, inits, maximums, node)
            if verbose:
                print(node.name, node.op_type, prev_max)
            for p in previous:
                value = get_previous_max(nodes, inits, maximums, p)
                if verbose:
                    print(p.op_type, value)

                arr = np.asarray([value / prev_max])
                inits = onnx_normalize_scalar_mul(inits, p.input[1], arr, np.ones(arr.shape))

    return nodes, inits


def run_normalize_graph(model_src, model_stats, model_dst):
    model = onnx.load(model_src)
    graph = model.graph
    nodes, inits, inputs, outputs = graph.node, graph.initializer, graph.input, graph.output

    # normalize across layers
    stats = load_statistics(model_stats)

    nodes, inits = onnx_normalize_graph(nodes, inits, stats, False)
    nodes, inits = onnx_normalize_graph_multi(nodes, inits, stats, False)
    onnx_save_graph(nodes, inputs, outputs, inits, model_dst, "normalized")

    # get output scale factors
    output_maximums = []
    for output in outputs:
        node = get_node_source(nodes, output.name)
        if node.op_type not in weighted_nodes:
            m = get_previous_max(nodes,inits,stats,node) 
        else:
            m = get_node_max(nodes,inits, stats, node)

        output_maximums.append(float(m))

    input_maximums = [float(stats[nodes[0].input[0]])]

    return input_maximums, output_maximums


def normalize_graph(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_stats')
    parser.add_argument('model_src')
    parser.add_argument('model_dst')
    parser.add_argument('-m', '--mode', choices=['layers', 'channels'], default='layers')
    parser.add_argument('-s', '--scale', type=float, default=255.)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args(argv)

    model = onnx.load(args.model_src)
    graph = model.graph
    nodes, inits, inputs, outputs = graph.node, graph.initializer, graph.input, graph.output

    # normalize across layers
    stats = load_statistics(args.model_stats)

    nodes, inits = onnx_normalize_graph(nodes, inits, stats, False)
    nodes, inits = onnx_normalize_graph_multi(nodes, inits, stats, False)
    onnx_save_graph(nodes, inputs, outputs, inits, args.model_dst, "normalized")

    # get output scale factors
    output_maximums = []
    for output in outputs:
        node = get_node_source(nodes, output.name)
        output_maximums.append(get_previous_max(nodes, stats, node))

    # output_maximums = run_normalize_graph(args.model_src, args.model_stats, args.model_dst)

    # check output is equivalent
    if args.verbose:
        # input_array = onnx_random_input(args.model_src)
        # output_a = onnx_infer(args.model_src, input_array)
        # output_b = onnx_infer(args.model_dst, input_array/args.scale)

        # for i,o in enumerate(output_b):
        #     output_b[i] *= output_maximums[i]
        #     print(i, output_maximums[i])

        # print(np.allclose(output_a, output_b, atol=1e-05))

        input_array = onnx_random_input(args.model_src)
        activations_a = onnx_activations(args.model_src, input_array)
        activations_b = onnx_activations(args.model_dst, input_array/args.scale)

        for o, output in enumerate(outputs):
            print(output.name, output_maximums[o])
            arr_a = activations_a[output.name]
            arr_b = activations_a[output.name]
            arr_b *= output_maximums[o]
            print(np.allclose(arr_a, arr_b, atol=1e-05))


if __name__ == "__main__":

    import sys
    normalize_graph(sys.argv[1:])
