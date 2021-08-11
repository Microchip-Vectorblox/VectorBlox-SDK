import json
import numpy as np
import onnx
from onnx import AttributeProto
from onnx import numpy_helper, helper, checker


def trace_linear_path_backwards(nodes, nname):
    "get list of nodes before nname, until a node that splits"
    node = get_node_source(nodes, nname)
    if node is None:
        return []

    assert(len(node.output) == 1)
    inputs = [get_node_source(nodes, n) for n in node.input if get_node_source(nodes, n) != None]
    if len(inputs) == 1:
        return [node.name] + trace_linear_path_backwards(nodes, get_node(nodes, inputs[0].name).output[0])

    return [node.name]

def get_tensor(inits, name):
    value = None
    x = [_ for _ in inits if _.name == name]

    if len(x) == 1:
        value = onnx.numpy_helper.to_array(x[0])

    return value

def get_model_input_shape(model_file):
    model = onnx.load(model_file)

    # The model is represented as a protobuf structure and it can be accessed
    # using the standard python-for-protobuf methods  
    # iterate through inputs of the graph
    for input in model.graph.input:
        # get type of input tensor
        tensor_type = input.type.tensor_type
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            return [d.dim_value for d in tensor_type.shape.dim if d.HasField("dim_value") ]
            # iterate through dimensions of the shape:
def set_tensor(inits, name, arr):
    x = [i for (i, _) in enumerate(inits) if _.name == name]

    if len(x) == 1:
        idx = x[0]
        inits[idx].raw_data = arr.tobytes()
        del inits[idx].float_data[:]

    return inits


def get_previous_nodes(nodes, node):
    paths = [trace_linear_path_backwards(nodes, n) for n in node.input]
    return [get_node(nodes, path[0]) for path in paths if path != []]




def get_node_source(nodes, output_name):
    "Get node by output name"
    node = None
    x = [n for n in nodes if output_name in n.output]
    assert len(x)<=1,"More than 1 node with output named {}".format(output_name)
    if len(x) == 1:
        node = x[0]
    return node


def get_node(nodes, nname):
    "Get node by node name"
    node = None
    x = [n for n in nodes if nname == n.name]

    if len(x) == 1:
        node = x[0]
    assert len(x)<=1,"More than 1 node with name {}".format(nname)

    return node


def get_node_inputs(nodes, nname):
    "Get list of nodes that have input of name nname"
    x = [n for n in nodes if nname in n.input]
    return x


def change_input_dim(model, batch_dim=None):
    sym_batch_dim = "N"
    inputs = model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        if batch_dim:
            dim1.dim_value = batch_dim
        else:
            dim1.dim_param = sym_batch_dim


def onnx_save_model(graph, fname, producer=None):
    checker.check_graph(graph)
    if producer:
        model_proto = helper.make_model(graph, producer=producer)
    else:
        model_proto = helper.make_model(graph)
    change_input_dim(model_proto)
    checker.check_model(model_proto)
    model_string = model_proto.SerializeToString()

    with open(fname, 'wb') as f:
        f.write(model_string)



def onnx_save_graph(nodes, inputs, outputs, inits, fname, graph_name, producer=None):
    graph = onnx.helper.make_graph(nodes, graph_name, inputs, outputs, inits)
    onnx_save_model(graph, fname, producer)


def has_attr(node, prop):
    x = [_ for _ in node.attribute if _.name == prop]

    return len(x) == 1


def get_attr(node, prop):
    value = None
    x = [_ for _ in node.attribute if _.name == prop]

    if len(x) == 1:
        attribute = x[0]
        if attribute.type == AttributeProto.INT:
            value = attribute.i
        elif attribute.type == AttributeProto.INTS:
            value = attribute.ints
        elif attribute.type == AttributeProto.FLOAT:
            value = attribute.f
        elif attribute.type == AttributeProto.STRING:
            value = attribute.s

    return value


def set_attr(node, prop, value):
    x = [_ for _ in node.attribute if _.name == prop]

    if len(x) == 1:
        attribute = x[0]
        if attribute.type == AttributeProto.INT:
            attribute.i = value
        elif attribute.type == AttributeProto.INTS:
            attribute.ints[:] = value
        elif attribute.type == AttributeProto.FLOAT:
            attribute.f = value
        elif attribute.type == AttributeProto.STRING:
            attribute.s = value


def get_node_source_index(nodes, nname):
    # node = None
    node = -1
    x = [i for i, _ in enumerate(nodes) if nname in _.output]
    if len(x) == 1:
        node = x[0]
    return node


def get_node_index(nodes, nname):
    node = None
    x = [i for i, _ in enumerate(nodes) if nname == _.name]
    if len(x) == 1:
        node = x[0]
    return node


def load_statistics(fname, mode=None):
    stats = {}
    with open(fname) as f:
        j = json.load(f)
        for arr in j:
            channel_maximums = np.stack((np.asarray(arr['max'],dtype=np.float32),
                                         np.asarray(arr['min'],dtype=np.float32)), axis=-1)
            symetric_channel_maximums = np.max(np.abs(channel_maximums), axis=-1)
            positive_channel_maximums = np.max(channel_maximums, axis=-1)

            stats[str(arr['id'])] = np.max(symetric_channel_maximums)
            if 'threshold' in arr:
                if stats[str(arr['id'])] > arr['threshold']:
                    stats[str(arr['id'])] = np.asarray(arr['threshold'])
            if 'ScaleShift' in arr['name'] and arr['id'] > 1:
                stats[str(arr['id']) + '_ss'] = stats[str(arr['id'])]
    return stats
