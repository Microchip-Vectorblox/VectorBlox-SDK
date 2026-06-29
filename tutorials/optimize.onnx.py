import argparse
import onnx
from onnx import version_converter
from onnx.tools import update_model_dims
import subprocess, shlex


def get_parent_nodes(nodes, input_nodes):
    parent_nodes = []
    for j in range(len(nodes)):
        for output in nodes[j].output:
            for k in input_nodes:
                if output in nodes[k].input:
                    parent_nodes.append(j)
    return parent_nodes


# move transpose up
def transpose_up(model):
    nodes = model.graph.node

    i = 0
    while i < len(nodes):
        if nodes[i].op_type == "Transpose" and onnx.helper.get_node_attr_value(nodes[i], "perm") == [0,3,1,2]:
            p0 = get_parent_nodes(nodes, [i])
            if len(p0) == 1 and nodes[p0[0]].op_type == "Mul":
                p1 = get_parent_nodes(nodes, [p0[0]])
                if len(p1) == 2 and nodes[p1[0]].op_type == "Mul" and nodes[p1[1]].op_type == "Sigmoid":
                    idx = [p1[0], p1[1], p0[0], i]
                    n0,n1,n2,n3 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]], nodes[idx[3]]

                    perm = onnx.helper.get_node_attr_value(n3, "perm")
                    mod_n0 = onnx.helper.make_node("Transpose", [n0.input[0]], [n0.output[0]], perm=perm)
                    mod_n1 = onnx.helper.make_node("Mul", [n0.output[0], n0.input[1]], [n1.output[0]])
                    mod_n2 = onnx.helper.make_node("Sigmoid", [n1.output[0]], [n2.output[0]])
                    mod_n3 = onnx.helper.make_node("Mul", [n1.output[0], n2.output[0]], [n3.output[0]])

                    for idx,mod in zip(idx,[mod_n0, mod_n1, mod_n2, mod_n3]):
                        nodes = nodes[:idx] + [mod] + nodes[idx+1:]
        i += 1

    return nodes

# move sigmoid up
def sigmoid_up(model):
    nodes = model.graph.node

    i = 0
    while i < len(nodes):
        if nodes[i].op_type == "Sigmoid":
            p0 = get_parent_nodes(nodes, [i])
            if len(p0) == 1 and nodes[p0[0]].op_type == "Reshape":
                p1 = get_parent_nodes(nodes, [p0[0]])
                if len(p1) == 1 and nodes[p1[0]].op_type == "Transpose" and onnx.helper.get_node_attr_value(nodes[p1[0]], "perm") == [0,2,3,1]:
                    idx = [p1[0], p0[0], i]
                    n0,n1,n2 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]

                    perm = onnx.helper.get_node_attr_value(n0, "perm")
                    mod_n0 = onnx.helper.make_node("Sigmoid", [n0.input[0]], [n0.output[0]])
                    mod_n1 = onnx.helper.make_node("Transpose", [n0.output[0]], ['reshaped_{}'.format(i)], perm=perm)
                    mod_n2 = onnx.helper.make_node("Reshape", ['reshaped_{}'.format(i), n1.input[1]], [n2.output[0]])

                    for idx,mod in zip(idx,[mod_n0, mod_n1, mod_n2]):
                        nodes = nodes[:idx] + [mod] + nodes[idx+1:]
        i += 1

    return nodes


# remove squeeze unsqueeze
def remove_squeeze(model):
    nodes = model.graph.node
    i = 0
    while i < len(nodes) - 2:
        n0 = nodes[i+0]
        n1 = nodes[i+1]

        if n0.op_type == "Squeeze" and n1.op_type == "Reshape":
            ishape, oshape = None, None
            for v in model.graph.value_info:
                if v.name in n0.input:
                    ishape = [_.dim_value for _ in v.type.tensor_type.shape.dim]
            for v in model.graph.value_info:
                if v.name in n1.output:
                    oshape = [_.dim_value for _ in v.type.tensor_type.shape.dim]
            if ishape and oshape and tuple(ishape) == tuple(oshape):
                nodes[i+2].input[0] = nodes[i].input[0]
                nodes = nodes[:i] + nodes[i+2:]
                i -= 2
        i += 1
    return nodes


# replace reduce sum
def replace_reducesum(model):
    nodes = model.graph.node

    i = 0
    while i < len(nodes):
        if nodes[i].op_type == "Transpose" and onnx.helper.get_node_attr_value(nodes[i], "perm") == [0,3,1,2]:
            p0 = get_parent_nodes(nodes, [i])
            if len(p0) == 1 and nodes[p0[0]].op_type == "ReduceSum":
                p1 = get_parent_nodes(nodes, [p0[0]])
                if len(p1) == 1 and nodes[p1[0]].op_type == "Transpose":
                    p2 = get_parent_nodes(nodes, [p1[0]])
                    if len(p2) == 1 and nodes[p2[0]].op_type == "Concat":
                        idx = [p2[0], p1[0], p0[0], i]
                        n0,n1,n2,n3 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]], nodes[idx[3]]

                        input_nodes = []
                        for input in n0.input:
                            for j in range(len(nodes)):
                                if input in nodes[j].output:
                                    input_nodes.append(j)

                        if all([nodes[_].op_type == "Unsqueeze" for _ in input_nodes]):
                            parent_nodes = get_parent_nodes(nodes, input_nodes)

                            if len(input_nodes) == 2:
                                add_inputs = [nodes[_].output[0] for _ in parent_nodes]
                                mod_n0 = onnx.helper.make_node("Add", add_inputs, [n3.output[0]])

                                nodes = nodes[:idx[3]+1] + [mod_n0] + nodes[idx[3]+1:] #insert Add nodes
                                for k in sorted(idx + input_nodes)[::-1]: #remove rest
                                    nodes = nodes[:k] + nodes[k+1:]

                            elif len(input_nodes) == 3:
                                add_inputs0 = [nodes[_].output[0] for _ in parent_nodes[:2]]
                                mod_n0 = onnx.helper.make_node("Add", add_inputs0, ['additional_add_{}'.format(i)])
                                mod_n1 = onnx.helper.make_node("Add", [mod_n0.output[0], nodes[parent_nodes[-1]].output[0]], [n3.output[0]])
                                nodes = nodes[:idx[3]+1] + [mod_n0, mod_n1] + nodes[idx[3]+1:] #insert Add nodes
                                for k in sorted(idx + input_nodes)[::-1]: #remove rest
                                    nodes = nodes[:k] + nodes[k+1:]
        i += 1

    return nodes


def update_nodes(model, nodes):
    graph = model.graph
    graph = onnx.helper.make_graph(nodes, 'update', graph.input, graph.output, graph.initializer, 'update', graph.value_info)
    onnx.checker.check_graph(graph)
    model = onnx.helper.make_model(graph)

    inputs = {_.name: [d.dim_value for d in _.type.tensor_type.shape.dim] for _ in graph.input}
    outputs = {_.name: [d.dim_value for d in _.type.tensor_type.shape.dim] for _ in graph.output}
    model = update_model_dims.update_inputs_outputs_dims(model, inputs, outputs)

    for val in model.graph.value_info:
        for i in range(len(val.type.tensor_type.shape.dim)):
            val.type.tensor_type.shape.dim[i].Clear()

    for init in model.graph.initializer:
        used = False
        for node in model.graph.node:
            if init.name in node.input or init.name in node.output:
                used = True
                break
        if not used:
            model.graph.initializer.remove(init)

    model = onnx.shape_inference.infer_shapes(model)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('dst')
    args = parser.parse_args()

    model = onnx.load(args.src)

    nodes = remove_squeeze(model)
    model = update_nodes(model, nodes)

    nodes = transpose_up(model)
    model = update_nodes(model, nodes)

    nodes = replace_reducesum(model)
    model = update_nodes(model, nodes)

    onnx.save(model, args.dst)

