import sys
sys.path.append('.')
import onnx
import onnx.helper
from onnx import numpy_helper
import numpy as np
import networkx as nx
import common.internal_representation as internal_representation
from common.tensor_ir import Tensor
from collections import OrderedDict
from typing import List, Union


# Maps values from the AttributeType enum to their string representations, e.g., {1: "FLOAT"}
ATTR_TYPE_MAPPING = dict(zip(onnx.AttributeProto.AttributeType.values(), onnx.AttributeProto.AttributeType.keys()))

# Maps an ONNX attribute to the corresponding Python property
ONNX_PYTHON_ATTR_MAPPING = {
    "FLOAT": "f",
    "INT": "i",
    "STRING": "s",
    "TENSOR": "t",
    "GRAPH": "g",
    "FLOATS": "floats",
    "INTS": "ints",
    "STRINGS": "strings",
}

def get_onnx_tensor_shape(onnx_tensor: Union[onnx.ValueInfoProto, onnx.TensorProto]) -> List[int]:
    shape = None
    if isinstance(onnx_tensor, onnx.TensorProto):
        shape = onnx_tensor.dims
    else:
        if onnx_tensor.type.tensor_type.HasField("shape"):
            shape = []
            for dim in onnx_tensor.type.tensor_type.shape.dim:
                if dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                elif dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append(None)
    return shape


def get_onnx_tensor_dtype(onnx_tensor: Union[onnx.ValueInfoProto, onnx.TensorProto]) -> np.dtype:
    if isinstance(onnx_tensor, onnx.TensorProto):
        onnx_type = onnx_tensor.data_type
    else:
        onnx_type = onnx_tensor.type.tensor_type.elem_type
    if onnx_type in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE:
        return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type]
    return None

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor

def import_tensor(onnx_tensor: Union[onnx.ValueInfoProto, onnx.TensorProto]) -> Tensor:
    if isinstance(onnx_tensor, onnx.TensorProto):
        tensor = Tensor(onnx_tensor.name,onnx.numpy_helper.to_array(onnx_tensor),get_onnx_tensor_dtype(onnx_tensor))
        return tensor
    else:
        tensor = Tensor(onnx_tensor.name,np.zeros(get_onnx_tensor_shape(onnx_tensor)),get_onnx_tensor_dtype(onnx_tensor))
        return tensor

def onnx_attrs_to_dict(attrs): # dans see: https://github.com/NVIDIA/TensorRT/blob/master/tools/onnx-graphsurgeon/onnx_graphsurgeon/importers/onnx_importer.py
    attr_dict = {}
    for attr in attrs:

        def process_attr(attr_str: str):
            processed = getattr(attr, ONNX_PYTHON_ATTR_MAPPING[attr_str])
            if attr_str == "STRING":
                processed = processed.decode()
            elif attr_str == "TENSOR":
                #raise ValueError('onnx attribute of tensor type not supported currently')
                processed = import_tensor(processed)
            elif attr_str == "GRAPH":
                raise ValueError('onnx attribute of graph type not supported currently')
                #processed = OnnxImporter.import_graph(
                #    processed, misc.combine_dicts(tensor_map, subgraph_tensor_map)
                #)
            elif attr_str == "FLOATS" or attr_str == "INTS":
                processed = list(processed)
            elif attr_str == "STRINGS":
                processed = [p.decode() for p in processed]
            return processed

        if attr.type in ATTR_TYPE_MAPPING:
            attr_str = ATTR_TYPE_MAPPING[attr.type]
            if attr_str in ONNX_PYTHON_ATTR_MAPPING:
                attr_dict[attr.name] = process_attr(attr_str)
            else:
                raise ValueError(
                    "Attribute of type {:} is currently unsupported. Skipping attribute.".format(attr_str)
                )
        else:
            raise ValueError(
                "Attribute type: {:} was not recognized. Was the graph generated with a newer IR version than the installed `onnx` package? Skipping attribute.".format(
                    attr.type
                )
            )
    return attr_dict

def remove_dequant_node(ir,node_name):
    graph = ir.graph
    node = graph.nodes()[node_name]
    node_inputs = node['inputs']
    node_outputs = node['outputs']
    if len(node_inputs)!=3 or len(node_outputs)!=1:
        raise ValueError('dequantize node removal is only possible if node has 3 input and 1 output. Node name = %s' % (node_name))
    # get the const's scale and zero point and remove relevant tensors
    scale_tensor_name = node_inputs[1]
    zp_tensor_name = node_inputs[2] 
    scale = ir.tensors[scale_tensor_name].data
    zero_point = ir.tensors[zp_tensor_name].data
    del(ir.tensors[scale_tensor_name])
    del(ir.tensors[zp_tensor_name])
    # We remove the pre-dequant node tensor and keep the post dequant tensor
    post_dequant_tensor_name = node_outputs[0]
    pre_dequant_tensor_name = node_inputs[0]
    pre_dequant_node_name = ir.tensors[pre_dequant_tensor_name].producer
    pre_dequant_node_outputs = graph.nodes()[pre_dequant_node_name]['outputs']
    outputs_copy = pre_dequant_node_outputs.copy()
    for idx,output in enumerate(outputs_copy):
        if output ==pre_dequant_tensor_name:
           pre_dequant_node_outputs[idx] = post_dequant_tensor_name
    ir.tensors[post_dequant_tensor_name].scale = scale
    ir.tensors[post_dequant_tensor_name].zero_point = zero_point
    ir.tensors[post_dequant_tensor_name].data = ir.tensors[pre_dequant_tensor_name].data # Copy the data of tensor from the pre-dequant
    del(ir.tensors[pre_dequant_tensor_name]) # remove the tensor
    succesive_dequant_nodes_names = ir.graph.succ[node_name]
    graph.remove_node(node_name)
    for succesive_qdq_nodes_name in succesive_dequant_nodes_names:
        ir.graph.add_edge(pre_dequant_node_name,succesive_qdq_nodes_name)
    #next_node = ir.graph.succ[node_name]
    #prev_node = ir.graph.pred[node_name]

def remove_const_dequantizer(ir,node_name):
    graph = ir.graph
    node = graph.nodes()[node_name]
    node_inputs = node['inputs']
    node_outputs = node['outputs']
    if len(node_inputs)!=3 or len(node_outputs)!=1:
        raise ValueError('const dequantize node removal is only possible if node has 3 input and 1 output. Node name = %s' % (node_name))
    # get the const's scale and zero point and remove relevant tensors
    scale_tensor_name = node_inputs[1]
    zp_tensor_name = node_inputs[2] 
    scale = ir.tensors[scale_tensor_name].data
    zero_point = ir.tensors[zp_tensor_name].data
    del(ir.tensors[scale_tensor_name])
    del(ir.tensors[zp_tensor_name])
    post_dequant_tensor_name = node_outputs[0]
    pre_dequant_tensor_name = node_inputs[0]
    ir.tensors[post_dequant_tensor_name].scale = scale
    ir.tensors[post_dequant_tensor_name].zero_point = zero_point
    ir.tensors[post_dequant_tensor_name].data = ir.tensors[pre_dequant_tensor_name].data # Copy the data of tensor from the pre-dequant
    ir.tensors[post_dequant_tensor_name].is_constant = True
    del(ir.tensors[pre_dequant_tensor_name]) # remove the tensor
    graph.remove_node(node_name)
    #next_node = ir.graph.succ[node_name]
    #prev_node = ir.graph.pred[node_name]

def remove_qdq_nodes(ir,node_name):
    tensors_for_deletion = []
    # This is allowed only if the node has single input and single output
    graph = ir.graph
    quantize_node_name = list(graph.pred[node_name])[0]
    succesive_qdq_nodes_names = list(graph.succ[node_name]) # This returns a list of input nodes to the quant node. It has either 0 or 1 elements
    pred_qdq_nodes_names = list(graph.pred[quantize_node_name])
    dequant_node = graph.nodes()[node_name]
    quant_node = graph.nodes()[quantize_node_name]
    dequant_node_inputs = dequant_node['inputs']
    dequant_node_outputs = dequant_node['outputs']
    quant_node_inputs = quant_node['inputs']
    qdq_input_name = quant_node_inputs[0]
    post_dequant_tensor_name = dequant_node_outputs[0]
    quant_node_outputs = quant_node['outputs']
    
    if len(dequant_node_inputs)!=3 or len(dequant_node_outputs)!=1:
        raise ValueError('const dequantize node removal is only possible if node has 3 input and 1 output. Node name = %s' % (node_name))
    # get the dequant node const's scale and zero point and remove relevant tensors
    dequant_scale_tensor_name = dequant_node_inputs[1]
    dequant_zp_tensor_name = dequant_node_inputs[2] 
    quant_scale_tensor_name = dequant_node_inputs[1]
    quant_zp_tensor_name = dequant_node_inputs[2] 
    scale = ir.tensors[dequant_scale_tensor_name].data
    zero_point = ir.tensors[dequant_zp_tensor_name].data
    # Remove the scale and zp tensors for the quant and dequant nodes we are about to remove (both nodes use same tensors so we need to remove only once)
    tensors_for_deletion.append(dequant_scale_tensor_name)
    tensors_for_deletion.append(dequant_zp_tensor_name)
    # Remove the tensor between the quant and dequant nodes
    tensors_for_deletion.append(quant_node_outputs[0])
    # we keep name of the post qdq tensor and update the names of tensor preceeding the qdq
    # Set the scale and zp values to the tensor which goes out of dequant node
    # Remove the tensor before the quant node and replace it with the tensor after the dequant
    if quantize_node_name.replace('_QuantizeLinear','') in ir.outputs: # If its a workload's output we keep the post qdq tensor name else we keep the pre qdq tensor name
        tensors_for_deletion.append(qdq_input_name)
        ir.tensors[post_dequant_tensor_name].scale = scale
        ir.tensors[post_dequant_tensor_name].zero_point = zero_point
        # We need to update the output tensor name of the qdq predecessor node
        pred_node = graph.nodes()[pred_qdq_nodes_names[0]]
        pred_node_outputs = pred_node['outputs']
        for idx,pred_node_output in enumerate(pred_node_outputs.copy()): # We iterate over a copy of the list since we want to change the list
            if pred_node_output == qdq_input_name: # If this input is the one we are changing
                pred_node_outputs[idx] = post_dequant_tensor_name
        ir.tensors[post_dequant_tensor_name].producer = pred_qdq_nodes_names[0] # We update the procuder name of the output tensor as it is no longer the dequant node but rather the node before the qdq
    else:
        if post_dequant_tensor_name in ir.outputs: # If models came from cut of quantized graph the output tensor will have qdq output tensor name. we change it to pre qdq name
            ir.outputs[ir.outputs.index(post_dequant_tensor_name)] = qdq_input_name
            ir.tensors[qdq_input_name].data = ir.tensors[post_dequant_tensor_name].data
        tensors_for_deletion.append(post_dequant_tensor_name)
        ir.tensors[qdq_input_name].scale = scale
        ir.tensors[qdq_input_name].zero_point = zero_point
        # We need to update the input tensor name of the qdq successive nodes
        for succ_node_name in succesive_qdq_nodes_names:
            succ_node = graph.nodes()[succ_node_name]
            succ_node_inputs = succ_node['inputs']
            for idx,succ_node_input in enumerate(succ_node_inputs.copy()): # We iterate over a copy of the list since we want to change the list
                if succ_node_input == dequant_node_outputs[0]: # If this input is the one we are changing
                    succ_node_inputs[idx] = qdq_input_name
        ir.tensors[qdq_input_name].consumers = succesive_qdq_nodes_names # We also update input tensor to qdq that its consumers are no longer the quant node but rather the following nodes

    graph.remove_node(node_name) # remove the dequant node
    graph.remove_node(quantize_node_name) # remove the quant node
    for pred_qdq_nodes_name in pred_qdq_nodes_names:
        for succesive_qdq_nodes_name in succesive_qdq_nodes_names:
            ir.graph.add_edge(pred_qdq_nodes_name,succesive_qdq_nodes_name)
    # TODO: Dans if there was a preceeding op to the quant op we need to restore the edge between it and the op that follows the qdq
    #next_node = ir.graph.succ[node_name]
    #prev_node = ir.graph.pred[node_name]
    return tensors_for_deletion

def onnx_to_ir(ir,onnx_graph):
    #import ir constants
    for constant in onnx_graph.initializer:
        tensor = numpy_helper.to_array(constant)
        ir.tensors[constant.name] = Tensor(constant.name,tensor,is_constant=True,shape = list(tensor.shape))
    # import shapes of activations - model must have shape inference data
    for tensor_info in list(onnx_graph.value_info):
        if tensor_info.name not in ir.tensors:
            tensor_shape = []
            for d in tensor_info.type.tensor_type.shape.dim:
                tensor_shape.append(d.dim_value)
            ir.tensors[tensor_info.name] = Tensor(tensor_info.name,None,is_constant=False, shape = tensor_shape)
    #import graph inputs
    for graph_input in onnx_graph.input:
        shape = get_onnx_tensor_shape(graph_input)
        type = get_onnx_tensor_dtype(graph_input)
        ir.inputs.append(graph_input.name)
        ir.tensors[graph_input.name] = Tensor(graph_input.name,np.zeros(shape,dtype=type),is_constant=False, shape=shape)
    #import graph outputs
    for graph_output in onnx_graph.output:
        shape = get_onnx_tensor_shape(graph_output)
        type = get_onnx_tensor_dtype(graph_output)
        ir.outputs.append(graph_output.name)
        ir.tensors[graph_output.name] = Tensor(graph_output.name,np.zeros(shape,dtype=type),is_constant=False,shape=shape)
    # import ir graph
    for node in onnx_graph.node: # ONNX list of nodes is in execution order
        attributes_dict = {}
        attributes_dict['name'] = node.name
        attributes_dict['attributes'] = onnx_attrs_to_dict(node.attribute)
        attributes_dict['op_type'] = node.op_type
        ir.graph.add_nodes_from([(node.name, attributes_dict)])
        # Check if inputs are already in
        inputs = []
        for idx,input in enumerate(node.input): # Update the tensors dict. Each input to the node should have it as consumer
            if input not in ir.tensors:
                if node.op_type =='Resize' and idx==1:
                    continue # In Resize op the 2nd input is provided as null so we can set the 3rd input which is scales
                else:
                    raise ValueError('Input tensor not found, this is bug since is should have already been allocated')
            inputs.append(input)
            input_tensor = ir.tensors[input]
            input_tensor.consumers.append(node.name)
            # If the input comes from a node, add a graph edge (connection between 2 nodes)
            if input_tensor.producer: # Input which doesnt have a producer is an input tensor to the graph or constant
                ir.graph.add_edge(input_tensor.producer,node.name)
        ir.graph.nodes[node.name]['inputs'] = inputs
        outputs = []
        for output in node.output:
            if output not in ir.tensors:
                raise ValueError ('Found tensor without shape. Name: %s. Make sure the supplied graph includes shape inference.' % output)
                ir.tensors[output] = Tensor(output,None,is_constant=False)
            outputs.append(output)
            output_tensor = ir.tensors[output]
            output_tensor.producer = node.name # Update the tensors dict. the output tensor to the node should have it as consumer
        ir.graph.nodes[node.name]['outputs'] = outputs
        #current_node = ir.graph.
    #remove_const_dequantizer(ir,list(ir.graph.nodes)[0])
    return ir

def remove_quantization_nodes(ir):
    dequant_nodes = []
    tensors_for_deletion = []
    for (p, d) in ir.graph.nodes(data=True):
        if d['op_type'] == 'DequantizeLinear':
            dequant_nodes.append(p)
    for dequant_node_name in dequant_nodes:
        pred_nodes = ir.graph.pred[dequant_node_name]
        succ_nodes = ir.graph.succ[dequant_node_name]
        if len(pred_nodes)==0: # If the dequant node has no preceeding nodes than its a constant input dequantizing
            remove_const_dequantizer(ir,dequant_node_name)
        elif len(pred_nodes)==1:
            if ir.graph.nodes()[list(pred_nodes)[0]]['op_type']=='QuantizeLinear': # Its a qdq node
                tensors_to_remove = remove_qdq_nodes(ir,dequant_node_name)
                tensors_for_deletion.extend(tensors_to_remove)
            else:
                remove_dequant_node(ir,dequant_node_name)
    for tensor_to_remove in set(tensors_for_deletion): # We mark the tensors and only remove them at end because some of them (scale tensors etc.) are used by more than one nodes
        del(ir.tensors[tensor_to_remove])
    return ir

def create_sample_onnx_model():
# Create a dummy convolutional neural network.

    # IO tensors (ValueInfoProto).
    model_input_name = "X"
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           [1, 512, 7, 7])
    model_output_name = "Y"
    model_output_channels = 512
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           [1, 512, 7, 7])

    # Create a Conv node (NodeProto).
    # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#conv
    conv1_output_node_name = "Y"
    # Dummy weights for conv.
    conv1_in_channels = 512
    conv1_out_channels = 512
    conv1_kernel_shape = (1, 1)
    conv1_pads = (0, 0, 0, 0)
    conv1_W = np.ones(shape=(conv1_out_channels, conv1_in_channels,
                             *conv1_kernel_shape)).astype(np.float32)
    conv1_B = np.ones(shape=(conv1_out_channels)).astype(np.float32)
    # Create the initializer tensor for the weights.
    conv1_W_initializer_tensor_name = "Conv1_W"
    conv1_W_initializer_tensor = create_initializer_tensor(
        name=conv1_W_initializer_tensor_name,
        tensor_array=conv1_W,
        data_type=onnx.TensorProto.FLOAT)
    conv1_B_initializer_tensor_name = "Conv1_B"
    conv1_B_initializer_tensor = create_initializer_tensor(
        name=conv1_B_initializer_tensor_name,
        tensor_array=conv1_B,
        data_type=onnx.TensorProto.FLOAT)

    conv1_node = onnx.helper.make_node(
        name="Conv1",  # Name is optional.
        op_type="Conv",
        # Must follow the order of input and output definitions.
        # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
        inputs=[
            model_input_name, conv1_W_initializer_tensor_name,
            conv1_B_initializer_tensor_name
        ],
        outputs=[conv1_output_node_name],
        # The following arguments are attributes.
        kernel_shape=conv1_kernel_shape,
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=conv1_pads,
    )
    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=[conv1_node],
        name="ConvGraph",
        inputs=[X],  # Graph input
        outputs=[Y],  # Graph output
        initializer=[
            conv1_W_initializer_tensor, conv1_B_initializer_tensor,
        ],
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    onnx.save(model_def, "convnet.onnx")

    return model_def

def main ():
    model_def = create_sample_onnx_model()
    model_def_resnet50 = onnx.load('resnet50.onnx')
    model_def_mobilenetv2 = onnx.load('mobilenetv2.onnx')
    model_def_conv = onnx.load('conv_ort_pcq_quantized.onnx')
    model_name = 'Conv14x14_simple'
    ir = internal_representation.IR(model_name)
    ir = onnx_to_ir(ir,model_def_conv.graph)
    internal_representation.draw_graph_from_ir(ir,'Imported from onnx')
    ir = remove_quantization_nodes(ir)
    internal_representation.draw_graph_from_ir(ir,'After qdq removal')
    ir.save('conv.nxi')
    #loaded_ir = load_ir('conv.nxi')
    




if __name__ == "__main__":

    main()    