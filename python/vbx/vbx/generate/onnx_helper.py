import json
import numpy as np
import onnx
from onnx import AttributeProto
from onnx import numpy_helper, helper, checker, shape_inference


def trace_linear_path_backwards(nodes, nname):
    "get list of nodes before nname, until a node that splits"
    node = get_node_source(nodes, nname)
    if node is None:
        return []

    #assert(len(node.output) == 1)
    inputs = [get_node_source(nodes, n) for n in node.input if get_node_source(nodes, n) != None]
    if len(inputs) == 1:
        return [node.name] + trace_linear_path_backwards(nodes, get_node(nodes, inputs[0].name).output[0])

    return [node.name]


def get_shape(inits, name):
    shape = None
    x = [_ for _ in inits if _.name == name]

    if len(x) == 1:
        tensor_type = x[0].type.tensor_type
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            shape = [d.dim_value for d in tensor_type.shape.dim if d.HasField("dim_value") ]

    return shape


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
    nodes = [get_node(nodes, path[0]) for path in paths if path != []] 
    unique = []
    for n in nodes:
        if n not in unique:
            unique.append(n)
    return unique


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


def onnx_save_model(graph, fname, infer_shapes=False, producer=None):
    checker.check_graph(graph)
    if producer:
        model_proto = helper.make_model(graph, producer=producer)
    else:
        model_proto = helper.make_model(graph)
    change_input_dim(model_proto)
    checker.check_model(model_proto)
    if infer_shapes:
        model_proto = shape_inference.infer_shapes(model_proto)
    model_string = model_proto.SerializeToString()

    with open(fname, 'wb') as f:
        f.write(model_string)



def onnx_save_graph(nodes, inputs, outputs, inits, fname, graph_name, infer_shapes=False, producer=None):
    graph = onnx.helper.make_graph(nodes, graph_name, inputs, outputs, inits)
    onnx_save_model(graph, fname, infer_shapes, producer)


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



def dump_onnx_io_names(json_file, graph):
    with open(json_file, 'r') as f:
        data = json.load(f)

    onnxx = onnx.load(graph)
    in_names = [k.name for k in onnxx.graph.input]
    out_names = [k.name for k in onnxx.graph.output]

    if '.opt.' in graph:
        key = 'opt'
    elif '.pre.' in graph:
        key = 'pre'
    elif '.norm.' in graph:
        key = 'norm'
    elif '.post.' in graph:
        key = 'post'
    for n,xml_name in enumerate(data['inputs'].keys()):
        data['inputs'][xml_name][key] = in_names[n]
    for n,xml_name in enumerate(data['outputs'].keys()):
        data['outputs'][xml_name][key] = out_names[n]

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def get_model_output_xml_names(json_file, graph):
    with open(json_file, 'r') as f:
        data = json.load(f)

    onnxx = onnx.load(graph)
    out_names = [k.name for k in onnxx.graph.output]
    if '.opt.' in graph:
        key = 'opt'
    elif '.pre.' in graph:
        key = 'pre'
    elif '.norm.' in graph:
        key = 'norm'
    elif '.post.' in graph:
        key = 'post'
    else:
        key = 'onnx'

    xml_names_in_order=[]
    for name in out_names:
        for xml_name,mapping in data['outputs'].items():
            if mapping[key] == name:
                xml_names_in_order.append(xml_name)
                break
        
    return xml_names_in_order

# TODO dump input_description, output_description names instead
def dump_vnnx_io_names(ionames_json, vnnx_json_string):
    with open(ionames_json, 'r') as f:
        data = json.load(f)
    vnnx_graph = json.loads(vnnx_json_string)

    js_layers = vnnx_graph['layers']
    inputs = [l['input_id'] for l in js_layers]
    outputs = [l['output_id'] for l in js_layers]
    output_indices = [n for n, l in enumerate(js_layers) if l['output_id'] not in inputs]
    input_indices = [n for n, l in enumerate(js_layers) if l['input_id'] not in outputs]
    graph_outputs = [(js_layers[i]['output_id'], js_layers[i]['output_description']) for i in output_indices]
    graph_inputs = [(js_layers[i]['input_id'], js_layers[i]['input_description']) for i in input_indices]

    # assumption: vnnx node input/output descriptions would be the same as names used in post.onnx
    for inp_id, inp_name in graph_inputs:
        for xml_name, mapping in data['inputs'].items():
            if inp_name == mapping['post']:
                data['inputs'][xml_name]['vnnx'] = inp_id
                break
    for out_id, out_name in graph_outputs:
        for xml_name, mapping in data['outputs'].items():
            if out_name == mapping['post']:
                data['outputs'][xml_name]['vnnx'] = out_id
                break

    with open(ionames_json, 'w') as f:
        json.dump(data, f, indent=4)

# TODO adjust for input_description, output_description names instead
def reorder_vnnx_input_arrays(sess_inputs, json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    vnnx_inputs = sorted(sess_inputs.items(), key = lambda x : data['inputs'][x[0]]['vnnx'])
    vnnx_inputs = [v[1] for v in vnnx_inputs]

    return vnnx_inputs


def get_vnnx_output_xml_names(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    vnnx_out_to_xml_name = dict()

    for xml_name,mapping in data['outputs'].items():
        vnnx_out_to_xml_name[mapping['vnnx']] = xml_name

    vnnx_out_to_xml_name = sorted(vnnx_out_to_xml_name.items(), key = lambda x : vnnx_out_to_xml_name[x[0]])
    out_names_in_order = [v[1] for v in vnnx_out_to_xml_name]
    
    return out_names_in_order