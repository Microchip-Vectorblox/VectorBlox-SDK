import sys
sys.path.append('.')
import onnx
import onnx.helper
from onnx import numpy_helper
import numpy as np
import networkx as nx
import common.internal_representation as internal_representation
from collections import OrderedDict
from typing import List, Union
from common.hw_config import REQUANTING_OPS, NON_REQUANTING_OPS, MULTIPLE_INPUT_OPS, MULTIPLE_INPUT_NON_REQUANTING_OPS, TFLITE_REQUANT
from common.tensor_ir import Tensor
from tqdm import tqdm


def get_shared_quant_params_nodes_names(ir,node_name,shared_quant_params_nodes_names):
    if node_name in shared_quant_params_nodes_names:
        return shared_quant_params_nodes_names
    node = ir.graph.nodes[node_name]
    following_nodes_params = node['frontend']['following_nodes_params']
    preceding_nodes_params = node['frontend']['preceding_nodes_params']
    current_op_type = node['op_type']
    shared_quant_params_nodes_names.update([node_name])
    if current_op_type in REQUANTING_OPS:
        for following_node_params in following_nodes_params:
            following_node = ir.graph.nodes[following_node_params[0]]
            following_node_op_type = following_node['op_type']
            if following_node_op_type in NON_REQUANTING_OPS:
                shared_quant_params_nodes_names = get_shared_quant_params_nodes_names(ir,following_node['name'],shared_quant_params_nodes_names)
    elif current_op_type in NON_REQUANTING_OPS:
        for following_node_params in following_nodes_params:
            following_node = ir.graph.nodes[following_node_params[0]]
            following_node_op_type = following_node['op_type']
            if following_node_op_type in NON_REQUANTING_OPS:
                shared_quant_params_nodes_names = get_shared_quant_params_nodes_names(ir,following_node['name'],shared_quant_params_nodes_names)
        for preceding_node_params in preceding_nodes_params:
            preceding_node = ir.graph.nodes[preceding_node_params[0]]
            shared_quant_params_nodes_names = get_shared_quant_params_nodes_names(ir,preceding_node['name'],shared_quant_params_nodes_names)
    else:
        raise ValueError ('Op type %s not supported in quant param setting. Please check...' % current_op_type)
    return shared_quant_params_nodes_names

def get_max_scale_and_zp(ir,shared_quant_params_nodes_names):
    selected_scale=0
    selected_zp=0
    sorted_graph = list(nx.lexicographical_topological_sort(ir.graph)) # We want to sort the nodes in execution order so that we always get same result in case of equal scales but different zp
    sorted_shared_quant_params_nodes_names = sorted(shared_quant_params_nodes_names, key=lambda x: sorted_graph.index(x))
    sorted_shared_quant_params_nodes = [ir.graph.nodes[sorted_shared_quant_params_node_name] for sorted_shared_quant_params_node_name in sorted_shared_quant_params_nodes_names]

    for current_node in sorted_shared_quant_params_nodes:
        nodes_output_tensor_name = current_node['outputs'][0]
        nodes_output_tensor = ir.tensors[nodes_output_tensor_name]
        output_scale=nodes_output_tensor.scale
        output_zp=nodes_output_tensor.zero_point
        if output_scale>selected_scale:
            selected_scale = output_scale
            selected_zp = output_zp
    return (selected_scale, selected_zp)

def insert_requant_node_between(ir:internal_representation.IR,target_node,source_node):
    original_tensor_name = source_node['outputs'][0]
    original_tensor = ir.tensors[original_tensor_name]
    requanted_tensor_name = original_tensor_name+'_requantized'
    if requanted_tensor_name in ir.tensors:
        raise ValueError ('Tensor name %s already exists, need to handle multiple requants case' % requanted_tensor_name)
    requant_node_name = original_tensor_name+'_requantnode'
    requant_node_name = ''.join(e for e in requant_node_name if (e.isalnum() or e=='_')) # Removing special characters from node name as it is used as filename for debug file name (.xlsx)
    requanted_tensor = Tensor(requanted_tensor_name,None,producer=requant_node_name,consumers=[target_node['name']],
                                 is_constant=False,shape = original_tensor.shape,scale = original_tensor.scale,
                                 zero_point=original_tensor.zero_point,folding_factor_x=original_tensor.folding_factor_x,
                                 folding_factor_y=original_tensor.folding_factor_y)
    ir.tensors[requanted_tensor_name] = requanted_tensor
    requant_node = {}
    requant_node['name'] = requant_node_name
    requant_node['attributes']={}
    requant_node['op_type'] = 'Conv' # We use Conv and not Identity as we want it to be able to be a foldable op
    requant_node['attributes']['kernel_shape'] = [1,1]
    requant_node['attributes']['pads'] = [0,0,0,0]
    requant_node['outputs'] = [requanted_tensor_name]
    requant_node['frontend'] = {}
    requant_node_input_tensor = source_node['frontend']['output_tensor']
    original_input_shape = requant_node_input_tensor.get_original_shape()
    original_input_channels = original_input_shape[1]
    requant_node['frontend']['input_channels'] = original_input_channels
    original_output_channels = original_input_shape[1] # input and output channels are the same
    requant_node['frontend']['output_channels'] = original_output_channels
    requant_node['frontend']['preceding_nodes_params'] = [(source_node['name'],0)]
    requant_node['frontend']['input_tensor'] = requant_node_input_tensor
    requant_node['frontend']['output_tensor'] = requanted_tensor
    requant_node['frontend']['input_tensor_scale'] = requant_node_input_tensor.scale
    requant_node['frontend']['input_tensor_zp'] = requant_node_input_tensor.zero_point   
    requant_node['frontend']['output_tensor_scale'] = requanted_tensor.scale
    requant_node['frontend']['output_tensor_zp'] = requanted_tensor.zero_point
    requant_node['frontend']['kernel_size'] = 1
    requant_node['frontend']['stride'] = 1
    if TFLITE_REQUANT:
        requant_node['frontend']['padding'] = requant_node['attributes']['pads']
    else:
        requant_node['frontend']['padding'] = requant_node['attributes']['pads'][0]
    
    
    original_weights_tensor_shape = [original_output_channels,original_input_channels,1,1]
    original_weights_tensor = np.zeros(original_weights_tensor_shape,dtype=np.int64)
    weight_value = 32 # Changed from 8 to 32 in MCHP numerics so that scale will be smaller and right shift will be bigger
    for oc in range(original_output_channels):
            original_weights_tensor[oc,oc,0,0] = weight_value
    w_int8_np = original_weights_tensor.astype(np.int8)
    original_weights_tensor_name = requant_node_name + '_original_weights_tensor'
    per_channel_scale = np.full((original_output_channels),1/weight_value)
    per_channel_zp = np.full((original_output_channels),0)
    tensor = Tensor(original_weights_tensor_name,w_int8_np,is_constant=True,shape = original_weights_tensor_shape,scale = per_channel_scale,zero_point=per_channel_zp)
    ir.tensors[original_weights_tensor_name] = tensor
    requant_node['frontend']['weights_tensor'] = tensor
    requant_node['frontend']['weights_per_channel_scale'] = per_channel_scale
    requant_node['frontend']['weights_per_channel_zp'] = per_channel_zp
    requant_node['frontend']['sparse_macs'] = original_input_shape[1]*original_input_shape[2]*original_input_shape[3]

    original_biases_tensor_shape = [original_output_channels]
    original_biases_tensor = np.zeros(original_biases_tensor_shape,dtype=np.int64)
    original_biases_tensor_name = requant_node_name + '_original_biases_tensor'
    per_channel_scale = np.full((original_output_channels),1.0)
    per_channel_zp = np.full((original_output_channels),0)
    biases_tensor = Tensor(original_biases_tensor_name,original_biases_tensor,is_constant=True,shape = original_biases_tensor_shape,scale=per_channel_scale,zero_point=per_channel_zp)
    ir.tensors[original_biases_tensor_name] = biases_tensor
    requant_node['frontend']['biases_tensor'] = biases_tensor
    requant_node['inputs'] = [original_tensor_name,original_weights_tensor_name,original_biases_tensor_name]

    ir.graph.add_node(requant_node_name,**requant_node) # When the node is created it copies the dictionary attributes and create a new dict
    created_requant_node = ir.graph.nodes[requant_node_name]
    ir.graph.add_edge(source_node['name'],requant_node_name)
    ir.graph.add_edge(requant_node_name,target_node['name'])
    ir.graph.remove_edge(source_node['name'],target_node['name'])

    ir.switch_tensor_consumer(original_tensor,original_node_name=target_node['name'],new_node_name=requant_node_name)
    ir.switch_input_name(target_node,original_input_name = original_tensor_name,new_input_name=requanted_tensor_name)# Update the 'inputs' field in target node
    ir.switch_input_tensor(target_node,original_input_tensor = original_tensor,new_input_tensor=requanted_tensor)# Update the 'input_tensors' field in target node
    source_node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(source_node) # Update the following_nodes_params field in source node - It is important to also update following nodes according to execution order
    target_node['frontend']['preceding_nodes_params'] = ir.get_updated_preceding_nodes(target_node) # Update the preceding_nodes_params field in target node
    created_requant_node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(requant_node)

def check_and_insert_requant_nodes(ir,node):
    if node['op_type'] in MULTIPLE_INPUT_NON_REQUANTING_OPS:
        output_qparams = ir.get_nodes_output_qparams(node)
        for preceding_node_params in node['frontend']['preceding_nodes_params']:
            preceding_node = ir.graph.nodes[preceding_node_params[0]]
            # We want to insert a requant node if the preceding node cant change its output qparams or if it is feeding multiple outputs (since we dont want the qparams change to effect accuracy of other graph parts)
            if preceding_node['op_type'] in NON_REQUANTING_OPS or len(preceding_node['frontend']['following_nodes_params'])>1:
                input_qparams = ir.get_nodes_output_qparams(preceding_node)
                if ('activation_silu' in preceding_node['attributes']) and (preceding_node['attributes']['activation_silu'] != None):
                    scale = preceding_node['attributes']['activation_silu']['output_scale'][0]
                    zp = preceding_node['attributes']['activation_silu']['output_zp'][0]
                    input_qparams = (scale, zp)
                if input_qparams!=output_qparams:
                    insert_requant_node_between(ir,node,preceding_node)

def update_concat_quant_params(ir,node):
    shared_quant_params_nodes_names = set()
    shared_quant_params_nodes_names = get_shared_quant_params_nodes_names(ir,node['name'],shared_quant_params_nodes_names)
    # Updated scale and zero_point of requant node to match that of concatenation node from the tflite graph
    for shared_qparams_node_name in shared_quant_params_nodes_names:
        if 'requantnode' in shared_qparams_node_name:
            shared_qparams_node = ir.graph.nodes[shared_qparams_node_name]
            shared_qparams_node['frontend']['output_tensor_scale'] = node['frontend']['output_tensor_scale']
            shared_qparams_node['frontend']['output_tensor_zp'] = node['frontend']['output_tensor_zp']
            original_requant_node_name = shared_qparams_node_name.split("_")[0]
            for i in range(len(node['frontend']['input_tensors'])):
                if (original_requant_node_name in node['frontend']['input_tensors'][i].name):
                    node['frontend']['input_tensors_scale'][i] = shared_qparams_node['frontend']['output_tensor_scale']
                    node['frontend']['input_tensors_zp'][i] = shared_qparams_node['frontend']['output_tensor_zp']
                    node['frontend']['input_tensors'][i].scale = shared_qparams_node['frontend']['output_tensor_scale']
                    node['frontend']['input_tensors'][i].zero_point = shared_qparams_node['frontend']['output_tensor_zp']

def old_get_output_quant_params(ir,node):
    following_nodes_params = node['frontend']['following_nodes_params']
    nodes_output_tensor_name = node['outputs'][0]
    nodes_output_tensor = ir.tensors[nodes_output_tensor_name]
    output_scale=nodes_output_tensor.scale
    output_zp=nodes_output_tensor.zero_point
    found_following_concat = False
    quant_params_updated = False
    selected_scale = 0
    selected_zp = 0
    for following_node_params in following_nodes_params: # if one of the following nodes is concat we need to overwrite our scale and zp with concat's output params
        following_node = ir.graph.nodes[following_node_params[0]]
        if following_node['op_type'] == 'Concat':
            if found_following_concat:
                print ('Output goes to 2 different concat ops, not supported yet since we need to choose which scale to take')
            found_following_concat = True
            concat_output_tensor_name = following_node['outputs'][0]
            concat_output_tensor = ir.tensors[concat_output_tensor_name]
            concat_output_scale = concat_output_tensor.scale
            concat_output_zp = concat_output_tensor.zero_point
            if concat_output_scale>selected_scale:
                selected_scale = concat_output_scale
                selected_zp = concat_output_zp
    if selected_scale!= output_scale:
        output_scale = selected_scale
        nodes_output_tensor.scale = selected_scale
        quant_params_updated = True
    if selected_zp != output_zp:
        output_zp = selected_zp
        nodes_output_tensor.zero_point = selected_zp
        quant_params_updated = True
    return output_scale, output_zp,quant_params_updated

def update_input_scale_zp(ir, node, tensor_name):
    preceding_node = None
    if (len(node['frontend']['preceding_nodes_params']) > 0):
        for i in range(len(node['frontend']['preceding_nodes_params'])):
            preceding_node_name = node['frontend']['preceding_nodes_params'][i][0]
            node_info = ir.graph.nodes[preceding_node_name]
            if (tensor_name in node_info['outputs']) and ('attributes' in node_info):
                if ('activation_silu' in node_info['attributes']):
                    preceding_node = node_info
                    break
    if (preceding_node != None) and (preceding_node['attributes']['activation_silu'] != None):
        scale = preceding_node['attributes']['activation_silu']['output_scale'][0]
        zp = preceding_node['attributes']['activation_silu']['output_zp'][0]
    else:
        scale = ir.tensors[tensor_name].scale
        zp = ir.tensors[tensor_name].zero_point
    return scale, zp
def get_averpool_params(ir,node_name,node):
    get_conv_input_output_params(ir,node_name,node)

def get_conv_params(ir,node_name,node):
    get_conv_input_output_params(ir,node_name,node)
    get_conv_weight_params(ir,node_name,node)

def get_conv_input_output_params(ir,node_name,node):

    current_op_input_tensor_name = node['inputs'][0]
    current_op_input_tensor_shape = ir.tensors[current_op_input_tensor_name].get_original_shape()
    scale, zp = update_input_scale_zp(ir, node, current_op_input_tensor_name)
    node['frontend']['input_tensor'] = ir.tensors[current_op_input_tensor_name]
    node['frontend']['input_tensor_scale'] = scale
    node['frontend']['input_tensor_zp'] = zp    
    current_op_input_channels = current_op_input_tensor_shape[1]
    node['frontend']['input_channels'] = current_op_input_channels

    current_op_output_tensor_name = node['outputs'][0]
    current_op_output_tensor_shape = ir.tensors[current_op_output_tensor_name].get_original_shape()
    current_op_output_channels = current_op_output_tensor_shape[1]
    node['frontend']['output_tensor'] = ir.tensors[current_op_output_tensor_name]
    current_op_output_tensor = ir.tensors[current_op_output_tensor_name]

    node['frontend']['output_tensor_scale'] = current_op_output_tensor.scale
    node['frontend']['output_tensor_zp'] = current_op_output_tensor.zero_point

    node['frontend']['output_channels'] = current_op_output_channels
    node['frontend']['kernel_size'] = node['attributes']['kernel_shape'][0] # we assume symetric kernel shape
    if 'strides' in node['attributes']:
        stride = node['attributes']['strides'][0]
    else:
        stride = 1
    node['frontend']['stride'] = stride
    if TFLITE_REQUANT:
        node['frontend']['padding'] = node['attributes']['pads']
    else:
        node['frontend']['padding'] = node['attributes']['pads'][0]


def get_conv_weight_params(ir,node_name,node): 
    current_op_input_tensor_name = node['inputs'][0]
    current_op_input_tensor_shape = ir.tensors[current_op_input_tensor_name].get_original_shape() 
    if 'strides' in node['attributes']:
        stride = node['attributes']['strides'][0]
    else:
        stride = 1  
    weights_tensor_name = node['inputs'][1]
    weights_tensor = ir.tensors[weights_tensor_name]
    node['frontend']['weights_tensor'] = weights_tensor
    weights_tensor_shape = weights_tensor.get_original_shape()
    dense_weights = weights_tensor.data.size
    sparse_weights = np.count_nonzero(weights_tensor.data)
    node['frontend']['dense_weights'] = dense_weights
    node['frontend']['sparse_weights'] = sparse_weights
    node['frontend']['weights_sparsity'] = 1- (sparse_weights/dense_weights)
    dense_macs = dense_weights*current_op_input_tensor_shape[2]*current_op_input_tensor_shape[3]/(stride*stride)
    sparse_macs = sparse_weights*current_op_input_tensor_shape[2]*current_op_input_tensor_shape[3]/(stride*stride)
    node['frontend']['dense_macs'] = dense_macs
    node['frontend']['sparse_macs'] = sparse_macs
    node['frontend']['macs_sparsity'] = 1- (sparse_macs/dense_macs)
    node['frontend']['weights_per_channel_scale'] = ir.tensors[weights_tensor_name].scale
    node['frontend']['weights_per_channel_zp'] = ir.tensors[weights_tensor_name].zero_point
    biases_tensor_name = node['inputs'][2]
    #node['frontend']['biases_data'] = ir.tensors[biases_tensor_name].data * ir.tensors[biases_tensor_name].scale
    node['frontend']['biases_tensor'] = ir.tensors[biases_tensor_name]

def get_gemm_params(ir,node_name,node):

    current_op_input_tensor_name = node['inputs'][0]
    current_op_input_tensor_shape = ir.tensors[current_op_input_tensor_name].get_original_shape()
    scale, zp = update_input_scale_zp(ir, node, current_op_input_tensor_name)
    node['frontend']['input_tensor'] = ir.tensors[current_op_input_tensor_name]
    node['frontend']['input_tensor_scale'] = scale
    node['frontend']['input_tensor_zp'] = zp
    current_op_input_channels = current_op_input_tensor_shape[1]
    node['frontend']['input_channels'] = current_op_input_channels

    current_op_output_tensor_name = node['outputs'][0]
    current_op_output_tensor_shape = ir.tensors[current_op_output_tensor_name].get_original_shape()
    current_op_output_channels = current_op_output_tensor_shape[1]
    node['frontend']['output_tensor'] = ir.tensors[current_op_output_tensor_name]
    current_op_output_tensor = ir.tensors[current_op_output_tensor_name]

    node['frontend']['output_tensor_scale'] = current_op_output_tensor.scale
    node['frontend']['output_tensor_zp'] = current_op_output_tensor.zero_point

    node['frontend']['output_channels'] = current_op_output_channels
    weights_tensor_name = node['inputs'][1]
    weights_tensor = ir.tensors[weights_tensor_name]
    node['frontend']['weights_tensor'] = weights_tensor
    dense_weights = weights_tensor.data.size
    sparse_weights = np.count_nonzero(weights_tensor.data)
    node['frontend']['dense_weights'] = dense_weights
    node['frontend']['sparse_weights'] = sparse_weights
    node['frontend']['weights_sparsity'] = 1- (sparse_weights/dense_weights)
    dense_macs = dense_weights
    sparse_macs = sparse_weights
    node['frontend']['dense_macs'] = dense_macs
    node['frontend']['sparse_macs'] = sparse_macs
    node['frontend']['macs_sparsity'] = 1- (sparse_macs/dense_macs)
    node['frontend']['weights_per_channel_scale'] = ir.tensors[weights_tensor_name].scale
    node['frontend']['weights_per_channel_zp'] = ir.tensors[weights_tensor_name].zero_point
    biases_tensor_name = node['inputs'][2]
    node['frontend']['biases_tensor'] = ir.tensors[biases_tensor_name]

def get_maxpool_params(ir,node_name,node):

    current_op_input_tensor_name = node['inputs'][0]
    current_op_input_tensor_shape = ir.tensors[current_op_input_tensor_name].get_original_shape()
    scale, zp = update_input_scale_zp(ir, node, current_op_input_tensor_name)
    node['frontend']['input_tensor'] = ir.tensors[current_op_input_tensor_name]
    node['frontend']['input_tensor_scale'] = scale
    node['frontend']['input_tensor_zp'] = zp
    current_op_input_channels = current_op_input_tensor_shape[1]
    node['frontend']['input_channels'] = current_op_input_channels

    current_op_output_tensor_name = node['outputs'][0]
    current_op_output_tensor_shape = ir.tensors[current_op_output_tensor_name].get_original_shape()
    current_op_output_channels = current_op_output_tensor_shape[1]
    node['frontend']['output_tensor'] = ir.tensors[current_op_output_tensor_name]

    current_op_output_tensor = ir.tensors[current_op_output_tensor_name]
    node['frontend']['output_tensor_scale'] = current_op_output_tensor.scale
    node['frontend']['output_tensor_zp'] = current_op_output_tensor.zero_point

    node['frontend']['output_channels'] = current_op_output_channels
    node['frontend']['kernel_size'] = node['attributes']['kernel_shape'][0] # we assume symetric kernel shape
    if 'strides' in node['attributes']:
        stride = node['attributes']['strides'][0]
    else:
        stride = 1
    node['frontend']['stride'] = stride
    if TFLITE_REQUANT:
        node['frontend']['padding'] = node['attributes']['pads']
    else:
        node['frontend']['padding'] = node['attributes']['pads'][0]
    
def get_general_params(ir,node_name,node):

    current_op_input_tensor_name = node['inputs'][0]
    current_op_input_tensor_shape = ir.tensors[current_op_input_tensor_name].get_original_shape()
    scale, zp = update_input_scale_zp(ir, node, current_op_input_tensor_name)
    node['frontend']['input_tensor'] = ir.tensors[current_op_input_tensor_name]
    node['frontend']['input_tensor_scale'] = scale
    node['frontend']['input_tensor_zp'] = zp
    current_op_input_channels = current_op_input_tensor_shape[1]
    node['frontend']['input_channels'] = current_op_input_channels

    current_op_output_tensor_name = node['outputs'][0]
    current_op_output_tensor_shape = ir.tensors[current_op_output_tensor_name].get_original_shape()
    current_op_output_channels = current_op_output_tensor_shape[1]
    node['frontend']['output_tensor'] = ir.tensors[current_op_output_tensor_name]

    current_op_output_tensor = ir.tensors[current_op_output_tensor_name]
    node['frontend']['output_tensor_scale'] = current_op_output_tensor.scale
    node['frontend']['output_tensor_zp'] = current_op_output_tensor.zero_point

    node['frontend']['output_channels'] = current_op_output_channels

def get_identity_params(ir,node_name,node):
    current_op_input_tensor_name = node['inputs'][0]
    current_op_input_tensor = ir.tensors[current_op_input_tensor_name]
    current_op_input_tensor_shape = current_op_input_tensor.get_original_shape()
    scale, zp = update_input_scale_zp(ir, node, current_op_input_tensor_name)
    node['frontend']['input_tensor'] = ir.tensors[current_op_input_tensor_name]
    node['frontend']['input_tensor_scale'] = scale
    node['frontend']['input_tensor_zp'] = zp
    current_op_input_channels = current_op_input_tensor_shape[1]
    node['frontend']['input_channels'] = current_op_input_channels

    current_op_output_tensor_name = node['outputs'][0]
    current_op_output_tensor = ir.tensors[current_op_output_tensor_name]
    current_op_output_tensor_shape = current_op_output_tensor.get_original_shape()
    current_op_output_channels = current_op_output_tensor_shape[1]
    node['frontend']['output_tensor'] = current_op_output_tensor

    if 'SPLIT' not in node_name and 'STRIDEDSLICE' not in node_name:
        current_op_output_tensor.scale = current_op_input_tensor.scale
        current_op_output_tensor.zero_point = current_op_input_tensor.zero_point

    node['frontend']['output_tensor_scale'] = current_op_output_tensor.scale
    node['frontend']['output_tensor_zp'] = current_op_output_tensor.zero_point

    node['frontend']['output_channels'] = current_op_output_channels

def get_add_params(ir,node_name,node):

    current_op_input_tensor_names = node['inputs']
    current_op_input_tensors_shape = []
    input_tensors = []
    input_tensors_scale = []
    input_tensors_zero_point = []
    for current_input_tensor_name in current_op_input_tensor_names:
        scale, zp = update_input_scale_zp(ir, node, current_input_tensor_name)
        current_op_input_tensors_shape.append(ir.tensors[current_input_tensor_name].get_original_shape())
        input_tensors.append(ir.tensors[current_input_tensor_name])
        input_tensors_scale.append(scale)
        input_tensors_zero_point.append(zp)
    if current_op_input_tensors_shape[0] != current_op_input_tensors_shape[1]:
        raise ValueError ('At Add op (%s). Both inputs should have the same shape but got %s != %s' % (node_name,str(current_op_input_tensors_shape[0]),str(current_op_input_tensors_shape[1])))
    node['frontend']['input_tensors'] = input_tensors
    node['frontend']['input_tensors_scale'] = input_tensors_scale
    node['frontend']['input_tensors_zp'] = input_tensors_zero_point
    current_op_input_channels = current_op_input_tensors_shape[0][1] #We assume both inputs have same dimentions. Broadcasting is not supported
    node['frontend']['input_channels'] = current_op_input_channels

    current_op_output_tensor_name = node['outputs'][0]
    current_op_output_tensor_shape = ir.tensors[current_op_output_tensor_name].get_original_shape()
    current_op_output_channels = current_op_output_tensor_shape[1]
    node['frontend']['output_tensor'] = ir.tensors[current_op_output_tensor_name]

    current_op_output_tensor = ir.tensors[current_op_output_tensor_name]
    node['frontend']['output_tensor_scale'] = current_op_output_tensor.scale
    node['frontend']['output_tensor_zp'] = current_op_output_tensor.zero_point

    node['frontend']['output_channels'] = current_op_output_channels

def get_concat_params(ir,node_name,node):

    current_op_input_tensor_names = node['inputs']
    current_op_input_tensors_shape = []
    input_tensors = []
    input_tensors_scale = []
    input_tensors_zero_point = []
    for current_input_tensor_name in current_op_input_tensor_names:
        scale, zp = update_input_scale_zp(ir, node, current_input_tensor_name)
        current_op_input_tensors_shape.append(ir.tensors[current_input_tensor_name].get_original_shape())
        input_tensors.append(ir.tensors[current_input_tensor_name])
        input_tensors_scale.append(scale)
        input_tensors_zero_point.append(zp)
    node['frontend']['input_tensors'] = input_tensors
    node['frontend']['input_tensors_scale'] = input_tensors_scale
    node['frontend']['input_tensors_zp'] = input_tensors_zero_point

    current_op_output_tensor_name = node['outputs'][0]
    current_op_output_tensor_shape = ir.tensors[current_op_output_tensor_name].get_original_shape()
    current_op_output_channels = current_op_output_tensor_shape[1]
    node['frontend']['output_tensor'] = ir.tensors[current_op_output_tensor_name]
    node['frontend']['output_tensor_scale'] = ir.tensors[current_op_output_tensor_name].scale
    node['frontend']['output_tensor_zp'] = ir.tensors[current_op_output_tensor_name].zero_point
    node['frontend']['output_channels'] = current_op_output_channels

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
            if (len(preceding_node['outputs']) > 0):
                for output_idx in range(len(preceding_node['outputs'])):
                    if (preceding_node['outputs'][output_idx] in node['inputs']):
                        input_index = node['inputs'].index(preceding_node['outputs'][output_idx])
            preceding_nodes.append((preceding_node_name,input_index))
        node['frontend']['preceding_nodes_params'] = preceding_nodes
        if node['op_type'] == 'Conv':
            get_conv_params(ir,node_name,node)
        elif node['op_type'] == 'AveragePool':
            get_averpool_params(ir,node_name,node)
        elif node['op_type'] == 'MaxPool':
            get_maxpool_params(ir,node_name,node)
        elif node['op_type'] == 'Add':
            get_add_params(ir,node_name,node)
        elif node['op_type'] == 'Concat':
            get_concat_params(ir,node_name,node)
        elif node['op_type'] == 'Gemm':
            get_gemm_params(ir,node_name,node)
        elif node['op_type'] == 'Identity':
            get_identity_params(ir,node_name,node)
        else:
            get_general_params(ir,node_name,node)
        update_output_tensor_consumers_order(node)

    return ir

# Need to impose equal quant params to ops that have multiple inputs and are unable to requant (e.g. Concat). 
# In such case we cahnge output scale of previous op where needed, if previous op is also non requantize op (i.e. its input qparams are equal to output qparams)
# We insert identity node which is able to requant (This is acually a requant node)

def update_nodes_qparams(ir):
    sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    tqdm_iterator = tqdm(sorted_graph)
    for node_name in tqdm_iterator:
        node = ir.graph.nodes[node_name]
        tqdm_iterator.set_description('Getting quantization params, at layer %s:' % node_name)
        if node['op_type'] in MULTIPLE_INPUT_NON_REQUANTING_OPS:
            check_and_insert_requant_nodes(ir,node)
            update_concat_quant_params(ir,node) # This will update qparams of concat's precedding nodes if needed so that all inputs will have same qparams
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
