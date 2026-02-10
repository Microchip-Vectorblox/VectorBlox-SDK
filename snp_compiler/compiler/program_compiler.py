import sys
import os

from common.working_node_state_mashine import add_stalls, check_nop_balance, generate_grids_1x1_cbc, generate_grids_identity_cbc,  generate_grids_rq_cbc_alex2
sys.path.append('.')
sys.path.append(os.path.dirname(__file__))
import common.internal_representation as internal_representation
from common.enums import DDREntryType, CustomeOpCode, GridConfig, DebugFilesFormat
from collections import OrderedDict
from common.hw_config import *
from channels_balancing_algo import split_noncontigues_input_channels,get_optimal_oc_split_and_order
from cbc_creator import export_cbc_to_xls_alex, generate_grids_wloc_cbc, generate_rqparams_ir,\
                        export_cbc_to_xls, get_nodes_real_ic_lookup_dicts, get_per_ic_group_sorted_weight_activation_pairs
from sequencer import generate_layer_command_sequence, generate_initial_command_sequence
from file_writer import prepare_wloc_ir,write_wloc_hex_files,prepare_rqloc_ir,write_rqloc_hex_file,prepare_rqparams_ir,\
                        write_rqparams_hex_file,write_oc_processing_order, write_wloc_hex_files_alex
import math
from common.ddr_ir import TensorDDREntry, create_tsnp_tensor_byte_array
from common.debug_flags import *
from frontend_hardware_dependant import split_conv_out_channels_in_two, calc_conv_qparams
import folding_algo
import copy
from common.tensor_ir import Tensor,TensorDeAllocationList,TensorDeAllocationInfo,InputTensorInfo
import numpy as np
from common.utils import list_of_lists_split_middle, list_of_lists_split_at_pos,get_y_tile_sizes
from tqdm import tqdm
from common.amm_ir import AMMTensor
import networkx as nx
import pickle
import os
from common.program_ir import CBC_IR, NonLinearFunctionList, LinearFunctionList
import reports
from common.utils import quantize2MathBlock

REORDER_OC_BEFORE_BORROW = True

def preceding_nodes_max_z_tiles(ir,node):
    max_z_tiles = 0
    for preceding_node_params in node['frontend']['preceding_nodes_params']:
        preceding_node = ir.graph.nodes[preceding_node_params[0]]
        preceding_node_z_tiles = preceding_node['backend']['z_tiles']
        if preceding_node_z_tiles>max_z_tiles:
            max_z_tiles = preceding_node_z_tiles
    return max_z_tiles

def get_node_tiling_info(ir,node):
    #old historical concept of Yaron
    # output_padding_start_y = AMM_HEIGHT
    # node_folded_output_shape = node['frontend']['output_tensor'].get_folded_shape()
    # current_op_height = node_folded_output_shape[2]
    # current_blob_idx = node['frontend']['tiling_blob_idx']
    # current_blob = ir.tiling_blobs[current_blob_idx]
    # k3_nodes_in_blob = current_blob.k3_nodes
    # tile_sizes,per_tile_read_start_line,per_tile_write_start_line=get_y_tile_sizes(current_op_height,k3_nodes_in_blob=k3_nodes_in_blob,add_padding_line=False)
    # num_tiles = len(tile_sizes)
    # output_padding_start_y = per_tile_write_start_line[-1]-per_tile_read_start_line[-1]+tile_sizes[-1]

    # new consept by Yaron and Alex
    output_padding_start_y = AMM_HEIGHT
    node_folded_output_shape = node['frontend']['output_tensor'].get_folded_shape()
    #one_line_of_padding = False # Alex: Have to think more about this
    one_line_of_padding = 1 if (node['op_type']=="Conv" and (('is_avgPool' in node['frontend']) and node['frontend']['is_avgPool'])) else 0
    #current_op_height = node_folded_output_shape[2] +  one_line_of_padding# Make sure its even
    current_op_height = node_folded_output_shape[2]
    current_blob_idx = node['frontend']['tiling_blob_idx']
    current_blob = ir.tiling_blobs[current_blob_idx]
    k3_nodes_in_blob = current_blob.k3_nodes
    tile_sizes,per_tile_read_start_line,per_tile_write_start_line=get_y_tile_sizes(current_op_height,k3_nodes_in_blob=k3_nodes_in_blob,add_padding_line=False)
    num_tiles = len(tile_sizes)
    output_tensor_y_size = ir.tensors[node['outputs'][0]].shape[2]
    if output_tensor_y_size<AMM_HEIGHT:
        output_padding_start_y = output_tensor_y_size
    elif one_line_of_padding:
        output_padding_start_y = AMM_HEIGHT - 1
    else:
        output_padding_start_y = AMM_HEIGHT

    '''if current_op_height<=AMM_HEIGHT:
        output_padding_start_y = current_op_height
    else:
        if kernel_size==1:
           output_padding_start_y =  current_op_height % AMM_HEIGHT
        else:
           output_padding_start_y =  ((current_op_height-1) % (AMM_HEIGHT-2))+1'''

    if one_line_of_padding:
        node['frontend']['pad_extra_line'] = True

    return num_tiles,output_padding_start_y


    
def get_op_grid_config(node,current_op_height,current_op_width,current_op_input_channels,current_op_output_width,kernel_size=1,y_tiles=1, x_slices = 1, k3_nodes_in_blob=0):

    
    # Set grid mode per each GRID OP
    deep_conv = False
    current_grid_mode = None

    for grid_config in GRID_CONFIGS.items():
        if (current_op_width/x_slices) <=grid_config[0]:
            current_grid_mode = grid_config[1]
            break
    
    if current_grid_mode==None:
        raise ValueError('At get_op_grid_config, unable to set grid mode')
    # Set input_channels_split and output_channels_split per each GRID OP
    current_op_num_grids = MAX_GRID_COUNT
    per_grid_max_allowed_input_channels = 2 ** LONG_ENTRY_BITS
    #Alex comment this, because we are not using DConv
    # if kernel_size==1:
    #     deep_conv = True        
    # else:
    #     deep_conv = False        
    z_tiles = 1
    if DEBUG_FORCE_IC_SPLIT:
        input_channels_split = DEBUG_FORCE_IC_SPLIT
    else:
        input_channels_split = math.ceil(current_op_input_channels / per_grid_max_allowed_input_channels)
    if node['op_type'] in LIMITED_GRIDS_OPS:
        current_op_max_grids = LIMITED_GRIDS_OPS[node['op_type']]
        output_channels_split = 1
        if current_op_max_grids == 1:
            if current_grid_mode != GridConfig.H14xW8:
                raise ValueError ('Single grid ops are supporting only %s grid mode', str(GridConfig.H14xW8))
        elif current_op_max_grids == 2:
            current_op_max_grids = 2 
            if current_grid_mode == GridConfig.H14xW16:
                output_channels_split = 1
            elif current_grid_mode == GridConfig.H14xW8:
                output_channels_split = 2
            else:
                raise ValueError ('2 grid ops are not supporting %s grid mode', str(current_grid_mode))

        elif current_op_max_grids == 4:
            current_op_max_grids = 8 # In order to allow single 14x16 grid we need to allow 8 grids and use only 4 of them (grid 0,2,4,6)
            if current_grid_mode == GridConfig.H14xW32:
                output_channels_split = 1
            elif current_grid_mode == GridConfig.H14xW16:
                output_channels_split = 2
            elif current_grid_mode == GridConfig.H14xW8:
                output_channels_split = 4
            else:
                raise ValueError ('4 grid ops are supporting only %s grid mode', str(GridConfig.H14xW32))
        else:
            raise ValueError ('%d max grids per op is not supported' % current_op_max_grids)
        current_op_num_grids = current_op_max_grids
        if input_channels_split!=1:
            raise ValueError ('Not supporting limited grid ops with input channel spliting')
        
    else:
        if current_op_num_grids % input_channels_split !=0:
            raise ValueError ('Not enough grids for op to support %d input channels splits' % input_channels_split)
        
        output_channels_split = current_op_num_grids // input_channels_split

        if current_grid_mode == GridConfig.H14xW16:
            output_channels_split = output_channels_split // 2
            if input_channels_split > 1:
                raise ValueError ('H14xW16 mode does not support input channels split')
        elif current_grid_mode == GridConfig.H14xW32:
            if input_channels_split > 1:
                raise ValueError ('H14xW32 mode does not support input channels split')
            output_channels_split = 2
        #elif current_grid_mode == GridConfig.H14xW8:
        #    if input_channels_split > 1:
        #        raise ValueError ('H14xW8 mode does not support input channels split')
        #    current_op_num_grids = 1
        #    output_channels_split = 1
    if (('is_avgPool' in node['frontend']) and node['frontend']['is_avgPool']):
        output_padding_start_x = 14
    else:
        output_padding_start_x = 15 - ((2*AMM_WIDTH) - (current_op_output_width%(2*AMM_WIDTH))) % (2*AMM_WIDTH)
    if  y_tiles==1 and current_op_height > AMM_HEIGHT:
        raise ValueError ('Error, folded tensor height cant exceed AMM_HEIGHT!')
    return current_grid_mode, input_channels_split, output_channels_split, current_op_num_grids,deep_conv,z_tiles,output_padding_start_x

def set_concat_op_grid_config(ir,node):
    input_tensors_names = node['inputs']
    input_tensors = []
    input_nodes_names = []
    input_nodes = []
    input_channels_per_input = []
    inputs_grid_mode = []
    inputs_is_hw_x_resize = []
    inputs_is_foldingconv = []
    force_folding_y = 'force_folding_y' in node['frontend']
    force_unfolding_y = 'force_unfolding_y' in node['frontend']
    for input_tensor_name in input_tensors_names:
        current_input_tensor = ir.tensors[input_tensor_name]
        input_tensors.append(current_input_tensor)
        input_node_name = ir.tensors[input_tensor_name].producer
        input_nodes_names.append(input_node_name)
        current_input_tensor_shape = current_input_tensor.get_folded_shape(folding_conv_y = force_folding_y,unfolding_conv_y=force_unfolding_y)
        input_channels_per_input.append(current_input_tensor_shape[1])
        current_input_node = ir.graph.nodes[input_node_name]
        input_nodes.append(current_input_node)
        current_input_grid_mode = current_input_node['backend']['gridmode']
        inputs_grid_mode.append(current_input_grid_mode)
        current_input_is_hw_x_resize = 1 if 'is_hw_x_resize' in current_input_node['frontend'] else 0
        inputs_is_hw_x_resize.append(current_input_is_hw_x_resize)
        current_input_is_foldingconv = 1 if 'force_folding_x' in current_input_node['frontend'] else 0
        inputs_is_foldingconv.append(current_input_is_foldingconv)

    current_op_input0_tensor_shape = input_tensors[0].get_folded_shape()
    current_op_input1_tensor_shape = input_tensors[0].get_folded_shape()
    current_op_output_tensor_name = node['outputs'][0]
    current_op_output_tensor_shape = ir.tensors[current_op_output_tensor_name].get_folded_shape()
    if len(current_op_input0_tensor_shape)!=4:
        raise ValueError ('Encountered activation with dims!=4: %s' % current_op_input0_tensor_shape)
    if len(current_op_input1_tensor_shape)!=4:
        raise ValueError ('Encountered activation with dims!=4: %s' % current_op_input1_tensor_shape)
    if len(current_op_output_tensor_shape)!=4:
        raise ValueError ('Encountered activation with dims!=4: %s' % current_op_output_tensor_name)
    current_op_output_channels = current_op_output_tensor_shape[1]
    node['backend']['output_channels'] = current_op_output_channels
    node['backend']['input_channels'] = input_channels_per_input

    grid_mode = None
    ic_splits = None
    for input_index,current_input_grid_mode in enumerate(inputs_grid_mode):
        if inputs_is_hw_x_resize[input_index]:
            actual_grid_mode = GridConfig.H14xW32
        elif inputs_is_foldingconv[input_index]:
            if current_input_grid_mode == GridConfig.H14xW32:
                actual_grid_mode = GridConfig.H14xW16
            elif current_input_grid_mode == GridConfig.H14xW16:
                actual_grid_mode = GridConfig.H14xW8
            else:
                raise ValueError ('Didnt expect folding at such conv. Please check')
        else:
            actual_grid_mode = current_input_grid_mode
        if grid_mode:
            if actual_grid_mode!=grid_mode:
                raise ValueError ('Concat inputs have different grid mode, this is illegal')
        else:
            grid_mode = actual_grid_mode
        #This check was originally removed by Yaron temporarily
        if ic_splits:
           if ic_splits!=input_nodes[input_index]['backend']['ic_splits']:
                raise ValueError ('Concat inputs have different ic_splits, this is illegal')
        else:
            ic_splits = input_nodes[input_index]['backend']['ic_splits']

    grid_count = get_num_virtual_grids(grid_mode)
    input_channels_split = 1
    node['backend']['deepconv'] = True
    node['backend']['gridmode'] = grid_mode
    node['backend']['ic_splits'] = input_channels_split
    node['backend']['grid_count'] = grid_count
    if grid_mode == GridConfig.H28xW28:
        node['backend']['oc_splits'] = 1
    else:
        node['backend']['oc_splits'] = grid_count // input_channels_split
    concats_number_of_inputs = len(node['frontend']['preceding_nodes_params'])
    y_tiles = node['frontend']['y_tiles']
    if grid_mode == GridConfig.H14xW8:
        node['frontend']['x_wrapping'] = 0
    else:
        node['frontend']['x_wrapping'] = 1

def fold_ic_groups(current_next_op_ic_groups,node,node_input_index):
    if node['op_type'] in MULTIPLE_INPUT_OPS:
        unfolded_input_channels = node['backend']['input_channels'][node_input_index]
    else:
        unfolded_input_channels = node['backend']['input_channels']
    updated_next_op_ic_groups=[]
    for ic_group in current_next_op_ic_groups:
        updated_ic_group=[0 for i in range(unfolded_input_channels*2)]
        for ic_idx,ic in enumerate(ic_group):
            updated_ic_group[ic_idx] = ic
            updated_ic_group[ic_idx+unfolded_input_channels] = ic+unfolded_input_channels
        updated_next_op_ic_groups.append(updated_ic_group)
    return updated_next_op_ic_groups

def unfold_ic_groups(current_next_op_ic_groups,node,node_input_index):
    if node['op_type'] in MULTIPLE_INPUT_OPS:
        folded_input_channels = node['backend']['input_channels'][node_input_index]
    else:
        folded_input_channels = node['backend']['input_channels']
    updated_next_op_ic_groups=[]
    for ic_group in current_next_op_ic_groups:
        updated_ic_group=[]
        for ic in ic_group:
            if ic<(folded_input_channels//2):
                updated_ic_group.append(ic)
        updated_next_op_ic_groups.append(updated_ic_group)
    return updated_next_op_ic_groups

def get_next_op_grid_config(ir,node,following_nodes_params):
    if len(following_nodes_params)==0:
        node['backend']['following_op_ic_split'] = 1
        node['backend']['following_op_oc_split'] = MAX_GRID_COUNT
        current_op_output_channels = node['backend']['output_channels']
        node['backend']['following_op_ic_groups'] = [[i for i in range(current_op_output_channels)]] # If its an op before output (no following ops) we want ouput of current op to be written to all amms
    else:
        next_op_ic_splits = None
        next_op_oc_splits = None
        next_op_ic_groups = None
        next_ops_ic_groups = []
        for current_following_node_params in following_nodes_params: # Verify that all following ops have same grid config
            current_following_node = ir.graph.nodes[current_following_node_params[0]]
            current_following_node_input_index = current_following_node_params[1]
            if next_op_ic_splits:
                if current_following_node['backend']['ic_splits'] != next_op_ic_splits:
                    raise ValueError ('Error: multiple successor ops have different grid config (ic_splits)')
            else:
                next_op_ic_splits = current_following_node['backend']['ic_splits']
            if next_op_oc_splits:
                if current_following_node['op_type'] in LIMITED_GRIDS_OPS:
                    if current_following_node['op_type']=='MaxPool': # We fake the current_following_node_oc_splits to match regular ops
                        current_following_node_oc_splits = 4
                    else:
                        raise ValueError ('Currently only MaxPool in H14xW16 is supported in limited grid option')
                else:
                    current_following_node_oc_splits = current_following_node['backend']['oc_splits']

                if current_following_node_oc_splits != next_op_oc_splits:
                    raise ValueError ('Error: multiple successor ops have different grid config (oc_splits)')
            else:
                if current_following_node['op_type'] in LIMITED_GRIDS_OPS:
                    if current_following_node['op_type']=='MaxPool': # We fake the current_following_node_oc_splits to match regular ops
                        current_following_node_oc_splits = 4
                    else:
                        raise ValueError ('Currently only MaxPool in H14xW16 is supported in limited grid option')
                else:
                    next_op_oc_splits = current_following_node['backend']['oc_splits']
            if current_following_node['op_type'] in MULTIPLE_INPUT_OPS: # TWO INPUT OPS have 2 ic groups we take the one that is relevant to the input we check
                current_next_op_ic_groups = current_following_node['backend']['ic_groups'][current_following_node_input_index]
            else:
                current_next_op_ic_groups = current_following_node['backend']['ic_groups']
            next_ops_ic_groups.append(current_next_op_ic_groups)
            next_op_is_y_folding = 'force_folding_y' in current_following_node['frontend']
            next_op_is_y_unfolding = 'force_unfolding_y' in current_following_node['frontend']
            if next_op_is_y_folding: # We want to get unfolded ic groups as this is what will be used for DDR write
                current_next_op_ic_groups = unfold_ic_groups(current_next_op_ic_groups,current_following_node,current_following_node_input_index)
            if next_op_is_y_unfolding: # We want to get folded ic groups as this is what will be used for DDR write
                current_next_op_ic_groups = fold_ic_groups(current_next_op_ic_groups,current_following_node,current_following_node_input_index)

            if next_op_ic_groups:
                if current_following_node['frontend']['tiling_blob_idx'] == node['frontend']['tiling_blob_idx']: # We dont check this if following node is not in same blob. If thats the case it might have a different input channels if it is force_folding/unfolding on read
                    if current_next_op_ic_groups != next_op_ic_groups:
                        raise ValueError ('Error: multiple successor ops have different grid config (ic_groups)')
            else:
                #if current_following_node['']
                next_op_ic_groups = current_next_op_ic_groups
        node['backend']['following_op_ic_split'] = next_op_ic_splits
        node['backend']['following_op_oc_split'] = next_op_oc_splits
        node['backend']['following_op_ic_groups'] = next_op_ic_groups

# The below loops over the ops and gets configuration of the grid per each op including input channel and output channel splitting
def get_ops_grid_config(ir: internal_representation.IR) -> internal_representation.IR:
    nodes_in_graph = ir.lexicographical_topological_sorted_graph
    tqdm_iterator = tqdm(nodes_in_graph)
    for node_name in tqdm_iterator:
        tqdm_iterator.set_description('Calculating grid config, at layer %s:' % node_name)
        node = ir.graph.nodes()[node_name]
        current_op_type = node['op_type']
        if current_op_type == 'Concat':
            node['backend'] = OrderedDict()
            set_concat_op_grid_config(ir,node)
        elif current_op_type in GRID_OPS:
            node['backend'] = OrderedDict() # This will include all compiler backend metadata for the op
            current_op_input_tensor_name = node['inputs'][0]
            force_folding_y = 'force_folding_y' in node['frontend']
            force_unfolding_y = 'force_unfolding_y' in node['frontend']
            consumer_0_name = ir.tensors[current_op_input_tensor_name].consumers[0]
            padded = ir.graph.nodes()[consumer_0_name]['attributes']['pads'][0] if 'pads' in ir.graph.nodes()[consumer_0_name]['attributes'] else 0
            current_op_input_tensor_shape = ir.tensors[current_op_input_tensor_name].get_folded_shape(folding_conv_y = force_folding_y,unfolding_conv_y=force_unfolding_y, padded =padded)
            current_op_output_tensor_name = node['outputs'][0]
            folding_conv_x = 'force_folding_x' in node['frontend']
            current_op_output_tensor_shape = ir.tensors[current_op_output_tensor_name].get_folded_shape(folding_conv_x=folding_conv_x,producing_node_stride=node['frontend']['stride']) # If its a folding conv, the conv itself needs to "see" folding_conv_x factor before hw folding
            if len(current_op_input_tensor_shape)!=4:
                raise ValueError ('Encountered activation with dims!=4: %s' % current_op_input_tensor_name)
            if len(current_op_output_tensor_shape)!=4:
                raise ValueError ('Encountered activation with dims!=4: %s' % current_op_output_tensor_name)
            current_op_output_channels = current_op_output_tensor_shape[1]
            current_op_input_channels = current_op_input_tensor_shape[1]
            if current_op_type in MULTIPLE_INPUT_OPS: # (e.g. ADD, CONCAT)
                current_op_input_channels*=2
            node['backend']['input_channels'] = current_op_input_channels
            node['backend']['output_channels'] = current_op_output_channels
            current_op_height = current_op_input_tensor_shape[2]
            current_op_width = current_op_input_tensor_shape[3]
            current_op_output_width = current_op_output_tensor_shape[3]
            _,kernel_size = internal_representation.get_node_weights_tensor(node)
            y_tiles = node['frontend']['y_tiles']
            x_slices = node['frontend']['x_slices']
            current_grid_mode, input_channels_split, output_channels_split,\
            current_op_num_grids,deep_conv,z_tiles,\
            output_padding_start_x = get_op_grid_config(node,current_op_height,current_op_width,current_op_input_channels,current_op_output_width,
                                                        kernel_size=kernel_size,y_tiles=y_tiles, x_slices=x_slices)
            
            node['backend']['deepconv'] = deep_conv
            node['backend']['gridmode'] = current_grid_mode
            node['backend']['ic_splits'] = input_channels_split
            node['backend']['grid_count'] = current_op_num_grids
            node['backend']['oc_splits'] = output_channels_split
            node['backend']['z_tiles'] = z_tiles
            node['backend']['output_padding_start_x'] = output_padding_start_x
            
            if current_grid_mode == GridConfig.H14xW8:
                node['frontend']['x_wrapping'] = 0
            else:
                node['frontend']['x_wrapping'] = 1
                
    return ir

def get_concat_channel_setup(ir: internal_representation.IR,node):
    current_op_input_channels = node['backend']['input_channels']
    #if 'force_folding_y' in node['frontend']:
    #    for input_index in range(len(current_op_input_channels)):
    #        current_op_input_channels[input_index]*=2

    ic_groups = [[[i for i in range(current_input_ic)]] for current_input_ic in current_op_input_channels]
    node['backend']['ic_groups'] = ic_groups

    all_inputs_direct_dict = []
    for current_input_ic_groups in ic_groups:
        current_input_direct_dicts = []
        for ic_group in current_input_ic_groups:
            current_input_direct_dict = {}
            for ic in ic_group:
                current_input_direct_dict[ic] = ic
            current_input_direct_dicts.append(current_input_direct_dict)
        all_inputs_direct_dict.append(copy.deepcopy(current_input_direct_dicts))
    node['backend']['ic_lookup_dicts'] = all_inputs_direct_dict

def get_ops_channel_balancing(ir: internal_representation.IR) -> internal_representation.IR:

    nodes_in_graph = ir.lexicographical_topological_sorted_graph
    tqdm_iterator = tqdm(nodes_in_graph)
    for node_name in tqdm_iterator:
        tqdm_iterator.set_description('Setting channels balancing, at layer %s:' % node_name)
        if node_name.endswith('ordering_conv'):
            continue
        node = ir.graph.nodes()[node_name]
        current_op_type = node['op_type']
        if current_op_type == 'Concat':
            get_concat_channel_setup(ir,node)
        if current_op_type in GRID_OPS:
            #if 'force_folding_y' in node['frontend']:
            #    raise ValueError ('Need to add support for input read from DDR folding')
                # This will involve having double number of input channels

            current_op_input_channels = node['backend']['input_channels']
            current_op_output_channels = node['backend']['output_channels']
            input_channels_split = node['backend']['ic_splits']
            current_op_weights_tensor,_ = internal_representation.get_node_weights_tensor(node)
            
            #create_balancing of input channels that not restricted in order but split size must be #input_channels/#splits
            per_oc_ic_group_macs, ic_groups = split_noncontigues_input_channels(current_op_input_channels,
                                                                                current_op_output_channels, current_op_weights_tensor.data,
                                                                                input_channels_split)
            # Since input channel processing order has no effect on efficiency we sort each group for easier debugging
            for current_group in range(len(ic_groups)):
                ic_groups[current_group].sort()
            
            if current_op_type in MULTIPLE_INPUT_OPS:
                if len(ic_groups)>2:
                    raise ValueError ('>2 input split not supported as input to dual input op')
                input_0_ic_groups = list_of_lists_split_middle(ic_groups,0)
                input_channels_per_input = current_op_input_channels // 2
                input_1_ic_groups = (np.array(list_of_lists_split_middle(ic_groups,1))-input_channels_per_input).tolist()
                node['backend']['ic_groups'] = [input_0_ic_groups,input_1_ic_groups]
                direct_dicts0 = []
                for ic_group in input_0_ic_groups:
                    direct_dict = {}
                    for ic in ic_group:
                        direct_dict[ic] = ic
                    direct_dicts0.append(direct_dict)
                direct_dicts1 = []
                for ic_group in input_1_ic_groups:
                    direct_dict = {}
                    for ic in ic_group:
                        direct_dict[ic] = ic
                    direct_dicts1.append(direct_dict)
                node['backend']['ic_lookup_dicts'] = [[direct_dicts0],[direct_dicts1]]
            else:
                direct_dicts = []
                for ic_group in ic_groups:
                    direct_dict = {}
                    for ic in ic_group:
                        direct_dict[ic] = ic
                    direct_dicts.append(direct_dict)
                node['backend']['ic_groups'] = ic_groups
                node['backend']['ic_lookup_dicts'] = direct_dicts
            output_channels_split = node['backend']['oc_splits']
            if REORDER_OC_BEFORE_BORROW:
                #find optimal output channel processing order. Find next channel which brings minimal diff between queues
                #Calculate output channels splits
                # optimize_oc_order is working only in cases of ic_splits
                oc_groups = get_optimal_oc_split_and_order(per_oc_ic_group_macs, output_channels_split,optimize_oc_order = True)
            else:
                oc_groups = get_optimal_oc_split_and_order(per_oc_ic_group_macs, output_channels_split,optimize_oc_order = False)

                
            # go over splits and see whats the maximum fifo size
            #total_macs, idle_macs, per_layer_efficiency = channels_balancing_algo.simulate_mac_processing_shiftprocessing(node_name,node,oc_groups,per_oc_ic_group_macs)
            #all_layers_total_macs += total_macs
            #if per_layer_efficiency<min_layer_efficiency:
            #    min_layer_efficiency = per_layer_efficiency
            node['backend']['oc_groups'] = oc_groups

    return ir

def create_concat_node(ir:internal_representation.IR,input0_node_name,input1_node_name):
    concat_node_name='concat_'+input0_node_name+'_'+input1_node_name
    ir.graph.add_node(concat_node_name)
    ir.graph.add_edge(input0_node_name,concat_node_name)
    ir.graph.add_edge(input1_node_name,concat_node_name)
    return concat_node_name
def is_external_input_to_blob(input_name,ir:internal_representation.IR,nodes_in_blob):
    nodes_in_graph = ir.lexicographical_topological_sorted_graph
    input_producer_node_name = ir.tensors[input_name].producer
    if input_producer_node_name in nodes_in_graph: # This makes sure its not a constant node
        if input_producer_node_name not in nodes_in_blob:
            return True
    return False
def duplicate_node_and_change_attributes(ir,source_node,nodes_in_blob,first_node_in_blob:bool):
    copy_node = copy.deepcopy(source_node)
    op_type = source_node['op_type']
    # Update source nodes attributes
    if op_type=='Conv':
        ir.tensors[source_node['outputs'][0]].shape[1] = ir.tensors[source_node['outputs'][0]].shape[1] // 2 # We will have half the output channels
        if source_node['frontend']['input_folding_factor']>0:
            updated_weights_tensor = copy.deepcopy(source_node['frontend']['folded_weights_tensor'])
            source_node['frontend']['original_folded_weights_tensor'] = source_node['frontend']['folded_weights_tensor']
            source_node['frontend']['folded_weights_tensor'] = updated_weights_tensor
            updated_weights_tensor.scale = np.split(updated_weights_tensor.scale,2)[0]
            updated_weights_tensor.zero_point = np.split(updated_weights_tensor.zero_point,2)[0]
            updated_weights_tensor.data = np.split(updated_weights_tensor.data,2)[0]
            updated_weights_tensor.shape[1] = updated_weights_tensor.shape[1] // 2
        else:
            updated_weights_tensor = copy.deepcopy(source_node['frontend']['weights_tensor'])
            source_node['frontend']['original_weights_tensor'] = source_node['frontend']['weights_tensor']
            source_node['frontend']['weights_tensor'] = updated_weights_tensor
            updated_weights_tensor.scale = np.split(updated_weights_tensor.scale,2)[0]
            updated_weights_tensor.zero_point = np.split(updated_weights_tensor.zero_point,2)[0]
            updated_weights_tensor.data = np.split(updated_weights_tensor.data,2)[0]
            updated_weights_tensor.shape[1] = updated_weights_tensor.shape[1] // 2

        source_node['frontend']['output_channels'] = source_node['frontend']['output_channels'] // 2
        source_node['backend']['output_channels'] = source_node['backend']['output_channels'] // 2

    source_node_name = source_node['name']
    dup_node_name = source_node_name+'_tile1'
    copy_node['name'] = dup_node_name
    # Update duplicated node 'inputs' filed
    for idx,input_name in enumerate(source_node['inputs']):
        if is_external_input_to_blob(input_name,ir,nodes_in_blob): # If the input producer is not part of blob we keep it as is
            continue
        copy_node['inputs'][idx] = input_name+'_tile1'
    # Update duplicated node 'outputs' filed
    duplicated_node_output_tensor_name = copy_node['outputs'][0]+'_tile1'
    copy_node['outputs'][0] = duplicated_node_output_tensor_name
    # Create new output tensor
    duplicated_node_output_tensor = copy.deepcopy(ir.tensors[source_node['outputs'][0]])
    duplicated_node_output_tensor.name = duplicated_node_output_tensor_name
    duplicated_node_output_tensor.shape[1] = duplicated_node_output_tensor.shape[1] # We already updated the source tensor size to half of output channels
    duplicated_node_output_tensor.producer += '_tile1'
    if source_node_name!=nodes_in_blob[-1]: # If its not the last node in blob
        duplicated_node_output_tensor.consumers[0]+='_tile1'
    ir.tensors[duplicated_node_output_tensor_name] = duplicated_node_output_tensor
    #update duplicated node weights
    if op_type=='Conv':
        if copy_node['frontend']['input_folding_factor']>0:
            updated_weights_tensor = copy.deepcopy(copy_node['frontend']['folded_weights_tensor'])
            copy_node['frontend']['original_folded_weights_tensor'] = copy_node['frontend']['folded_weights_tensor']
            copy_node['frontend']['folded_weights_tensor'] = updated_weights_tensor
            updated_weights_tensor.scale = np.split(updated_weights_tensor.scale,2)[1]
            updated_weights_tensor.zero_point = np.split(updated_weights_tensor.zero_point,2)[1]
            updated_weights_tensor.data = np.split(updated_weights_tensor.data,2)[1]
            updated_weights_tensor.shape[1] = updated_weights_tensor.shape[1] // 2
        else:
            updated_weights_tensor = copy.deepcopy(copy_node['frontend']['weights_tensor'])
            copy_node['frontend']['original_weights_tensor'] = copy_node['frontend']['weights_tensor']
            copy_node['frontend']['weights_tensor'] = updated_weights_tensor
            updated_weights_tensor.scale = np.split(updated_weights_tensor.scale,2)[1]
            updated_weights_tensor.zero_point = np.split(updated_weights_tensor.zero_point,2)[1]
            updated_weights_tensor.data = np.split(updated_weights_tensor.data,2)[1]
            updated_weights_tensor.shape[1] = updated_weights_tensor.shape[1] // 2
        copy_node['frontend']['output_channels'] = copy_node['frontend']['output_channels'] // 2
        copy_node['backend']['output_channels'] = copy_node['backend']['output_channels'] // 2

    return copy_node


def duplicate_blob(ir: internal_representation.IR,blob) -> internal_representation.IR:
    node_names = ir.lexicographical_topological_sorted_graph
    nodes_in_blob=[]
    for node_idx in blob:
        nodes_in_blob.append(node_names[node_idx])
    last_node_in_blob = ir.graph.nodes()[nodes_in_blob[-1]]
    duplication_blob = []
    for idx,source_node_idx in enumerate(blob):
        source_node_name = node_names[source_node_idx]
        source_node = ir.graph.nodes()[source_node_name]
        duplicated_node = duplicate_node_and_change_attributes(ir,source_node,nodes_in_blob,first_node_in_blob=(idx==0))
        ir.graph.add_node(duplicated_node['name'],**duplicated_node)
        duplication_blob.append(duplicated_node)
        if idx>0: # Starting from 2nd node in blob we add edge between current node and last node
            ir.graph.add_edge(duplication_blob[-2]['name'],duplicated_node['name'])
            if duplicated_node['op_type'] == 'Add': # If its an add node we need to add another input edge, the same as its source add node
                source_add_node_predecessors = list(ir.graph.predecessors(source_node['name'])) # We need to find the predecessor which is not part of the blob
                if source_add_node_predecessors[0] in nodes_in_blob:
                    external_add_input = source_add_node_predecessors[1]
                elif source_add_node_predecessors[1] in nodes_in_blob:
                    external_add_input = source_add_node_predecessors[0]
                else:
                    raise ValueError ('All add input nodes are in blob, something went wrong')
                ir.graph.add_edge(external_add_input,duplicated_node['name'])
    # Remove edges from last node in blob to its successors
    original_blob_successors_names = list(ir.graph.successors(last_node_in_blob['name']))
    ir.graph.remove_edges_from(last_node_in_blob)
    
    # We connect the start of duplicated blob to same input that goes to the original blob
    blob_predcesor = list(ir.graph.predecessors(node_names[blob[0]]))[0] 
    ir.graph.add_edge(blob_predcesor,duplication_blob[0]['name'])
    # Add virtual concat blob to connect the original blob and its duplication
    concat_node_name = create_concat_node(ir,nodes_in_blob[-1],duplication_blob[-1]['name'])
    # Connect edges from concat node to original blob successors
    for succesor in original_blob_successors_names:
        ir.graph.add_edge(concat_node_name,succesor)
    print('Thats it?')
    return ir

    


def generate_z_tiling_ops(ir: internal_representation.IR) -> internal_representation.IR: # This will duplicate blobs of z tiling
    node_names = ir.lexicographical_topological_sorted_graph
    blobs=[]
    current_blob=[]
    in_blob=False
    for idx,current_node_name in enumerate(node_names):
        current_node = ir.graph.nodes()[current_node_name]
        if current_node['backend']['z_tiles']>1:
            if in_blob==False:
                in_blob=True
                current_blob = [idx]
            else:
                current_blob.append(idx)
        else:
            if in_blob:
                blobs.append(current_blob)
                in_blob = False
    if in_blob:
            blobs.append(current_blob)
    for blob in blobs:
        ir=duplicate_blob(ir,blob)
    ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    return ir

def switch_input_tensor(node,original_input_tensor,new_input_tensor):
    original_input_tensor_name = original_input_tensor.name
    new_input_tensor_name = new_input_tensor.name
    found_input = False
    for idx,input_name in enumerate(node['inputs'].copy()):
        if input_name == original_input_tensor_name:
            node['inputs'][idx] = new_input_tensor_name
            found_input = True
    if not found_input:
        raise ValueError ('In switch_input_tensor, input tensor %s not found in nodes inputs' % original_input_tensor_name)
    if 'input_tensor' in node['frontend']:
        input_tensor = node['frontend']['input_tensor']
        if input_tensor!=original_input_tensor:
            raise ValueError ('In switch_input_tensor, input tensor %s not found in nodes backend input tensors' % original_input_tensor_name)
        else:
            node['frontend']['input_tensor'] = new_input_tensor
        
    elif 'input_tensors' in node['frontend']:
        found_input_tensor = False
        input_tensors = node['frontend']['input_tensors'].copy()
        for idx,input_tensor in enumerate(input_tensors):
            if input_tensor==original_input_tensor:
                node['frontend']['input_tensors'][idx] = new_input_tensor
                found_input_tensor = True
        if not found_input_tensor:
            raise ValueError ('In switch_input_tensor, input tensor %s not found in nodes backend input tensors' % original_input_tensor_name)
    else:
        raise ValueError ('backend, input tensor attribute not found in node')

def create_ordering_conv(ir: internal_representation.IR,node):
    node_name = node['name']
    ordering_node_name = node_name+'_ordering_conv'
    ordering_node = {}
    ordering_node['name'] = ordering_node_name
    ordering_node['attributes']={}
    ordering_node['op_type'] = 'Conv'
    ordering_node['attributes']['kernel_shape'] = [1,1]
    ordering_node['attributes']['pads'] = [0,0,0,0]
    ordering_node['outputs'] = node['outputs']
    if len(node['outputs'])!=1:
        raise ValueError ('We currently dont support output nodes with multiple output tensors')
    # Rename the node's output tensor name
    original_output_tensor_name = node['outputs'][0]
    original_output_tensor = node['frontend']['output_tensor']
    pre_ordering_tensor_name = 'pre_ordering_' + original_output_tensor_name
    updated_consumers = original_output_tensor.consumers+[ordering_node_name]
    pre_ordering_tensor = Tensor(pre_ordering_tensor_name,None,producer=node_name,consumers=updated_consumers,
                                 is_constant=False,shape = original_output_tensor.shape,scale = original_output_tensor.scale,
                                 zero_point=original_output_tensor.zero_point,folding_factor_x=original_output_tensor.folding_factor_x,
                                 folding_factor_y=original_output_tensor.folding_factor_y)
    ir.tensors[pre_ordering_tensor_name] = pre_ordering_tensor
    # If the workload's output producer node is also followed by other nodes we need to update their input to the new tensor we create between output node and ordering node
    original_node_following_nodes_names = original_output_tensor.consumers
    for following_node_name in original_node_following_nodes_names:
        following_node = ir.graph.nodes[following_node_name]
        switch_input_tensor(following_node,original_output_tensor,pre_ordering_tensor)

    node['frontend']['output_tensor'] = pre_ordering_tensor
    node['outputs'] = [pre_ordering_tensor_name]
    ordering_conv_weights_tensor_name = node_name+'_ordering_conv_w'
    ordering_conv_biases_tensor_name = node_name+'_ordering_conv_b'
    ordering_node['inputs'] = [pre_ordering_tensor_name,ordering_conv_weights_tensor_name,ordering_conv_biases_tensor_name]
    original_node_output_shape = node['frontend']['output_tensor'].get_folded_shape()
    ordering_node_input_channels = original_node_output_shape[1] # Backend input/output channels contains actual # of channels. If preceding conv is a folding one, its backend output channels will be before the folding so we need to take shape from the output tensor
    ordering_node_output_channels = ordering_node_input_channels # Ordering node output channels is equal to input channels

    ordering_node['backend'] = {}
    ordering_node['backend']['input_channels'] = ordering_node_input_channels
    ordering_node['backend']['output_channels'] = ordering_node_output_channels
    ordering_node['backend']['ic_splits'] = 1
    ordering_node['frontend'] = {}
    ordering_node['frontend']['following_nodes_params'] = []
    y_tiles = node['frontend']['y_tiles']
    x_slices = node['frontend']['x_slices']
    tile_sizes,per_tile_read_start_line,per_tile_write_start_line=get_y_tile_sizes(original_output_tensor.get_folded_shape()[2],k3_nodes_in_blob=0)
    ordering_node['frontend']['y_tiles'] = len(tile_sizes)
    current_grid_mode, _, _,current_op_num_grids,_,z_tiles,\
        output_padding_start_x = get_op_grid_config(ordering_node,original_node_output_shape[2],original_node_output_shape[3],original_node_output_shape[1],
                                                    original_node_output_shape[3],kernel_size=1,y_tiles=y_tiles,x_slices=x_slices,k3_nodes_in_blob=0)
    ordering_node['backend']['output_padding_start_x'] = output_padding_start_x

    ordering_node['backend']['gridmode'] = current_grid_mode
    ordering_node['backend']['grid_count'] = current_op_num_grids
    ordering_node['backend']['z_tiles'] = z_tiles
    output_channels_split = get_num_virtual_grids(current_grid_mode)
    ordering_node['backend']['oc_splits'] = output_channels_split
    ordering_node['backend']['ic_groups'] = [[ic for ic in range(ordering_node_input_channels)]]
    ordering_node['backend']['ic_lookup_dicts'] = [[ic for ic in range(ordering_node_input_channels)]]

    # Dans TODO: with the below, empty output channels will be written with "junk". This happens after ordering calc and rewrite of the ordering_conv weights which decide copy order of ic to oc
    ordering_node['backend']['oc_groups'] = [[oc for oc in range(0+i,ordering_node_output_channels,output_channels_split)] for i in range(output_channels_split)] 
    ordering_node['backend']['deepconv'] = True
    ordering_node['frontend']['input_tensor'] = pre_ordering_tensor
    ordering_node['frontend']['output_tensor'] = original_output_tensor
    ordering_node['frontend']['input_tensor_zp'] = node['frontend']['output_tensor_zp']
    ordering_node['frontend']['output_tensor_zp'] = node['frontend']['output_tensor_zp']
    ordering_node['frontend']['kernel_size'] = 1
    ordering_node['frontend']['stride'] = 1
    original_folded_output_tensor_shape = original_output_tensor.get_folded_shape()
    ordering_node['frontend']['sparse_macs'] = ordering_node_output_channels*original_folded_output_tensor_shape[2]*original_folded_output_tensor_shape[3]
    ordering_node['frontend']['input_folding_factor_x'] = node['frontend']['output_folding_factor_x']
    ordering_node['frontend']['input_folding_factor_y'] = node['frontend']['output_folding_factor_y']
    ordering_node['frontend']['input_tensor'].folding_factor_x = node['frontend']['output_tensor'].folding_factor_x
    ordering_node['frontend']['input_tensor'].folding_factor_y = node['frontend']['output_tensor'].folding_factor_y
    ordering_node['frontend']['output_folding_factor_x'] = node['frontend']['output_folding_factor_x']
    ordering_node['frontend']['output_folding_factor_y'] = node['frontend']['output_folding_factor_y']
    ordering_node['frontend']['output_tensor'].folding_factor_x = node['frontend']['output_tensor'].folding_factor_x
    ordering_node['frontend']['output_tensor'].folding_factor_y = node['frontend']['output_tensor'].folding_factor_y


    ordering_weights_shape = [ordering_node_output_channels,ordering_node_output_channels,1,1]
    ordering_biases_shape = [ordering_node_output_channels]
    # Add:
    # 1) new output tensor for node (the original output will be used by the ordering node)
    # 2) ordering node weights tensor
    # 3) ordering node biases tensor
    weights_tensor = Tensor(ordering_conv_weights_tensor_name,None,is_constant=True, shape = ordering_weights_shape, consumers=[ordering_node_name])
    biases_tensor = Tensor(ordering_conv_biases_tensor_name,None,is_constant=True, shape = ordering_biases_shape, consumers=[ordering_node_name])
    ir.tensors[ordering_conv_weights_tensor_name] = weights_tensor
    ir.tensors[ordering_conv_biases_tensor_name] = biases_tensor
    if (ordering_node['frontend']['input_folding_factor_x']>0 or ordering_node['frontend']['input_folding_factor_y']>0):
        ordering_node['frontend']['folded_weights_tensor'] = weights_tensor
        ordering_node['frontend']['folded_biases_tensor'] = biases_tensor
        ordering_node['frontend']['folded_kernel_size'] = 1
        ordering_node['frontend']['folded_requant_scale_shift'] = [(MAX_REDUCE_BUS_WIDTH - 3) for i in range(ordering_node_output_channels)] # This will cause zero shift in the mac stage
        requant_scale_uint14 = 2 ** (FRACTIONAL_BITS+1+4) # This will cause zero shift in the RQ stage
        ordering_node['frontend']['folded_requant_scale_uint14'] = [requant_scale_uint14 for i in range(ordering_node_output_channels)]
        mac_rough_shift_mux = 0
        ordering_node['frontend']['folded_mac_rough_shift_mux'] = [mac_rough_shift_mux for i in range(ordering_node_output_channels)]
        ordering_node['frontend']['folded_requant_bias_int12'] = [0 for i in range(ordering_node_output_channels)]
    else:
        ordering_node['frontend']['weights_tensor'] = weights_tensor
        ordering_node['frontend']['biases_tensor'] = biases_tensor
        ordering_node['frontend']['requant_scale_shift'] = [(MAX_REDUCE_BUS_WIDTH - 3) for i in range(ordering_node_output_channels)] # This will cause zero shift in the mac stage
        requant_scale_uint14 = 2 ** (FRACTIONAL_BITS+1+4) # This will cause zero shift in the RQ stage
        ordering_node['frontend']['requant_scale_uint14'] = [requant_scale_uint14 for i in range(ordering_node_output_channels)]
        mac_rough_shift_mux = 0
        ordering_node['frontend']['mac_rough_shift_mux'] = [mac_rough_shift_mux for i in range(ordering_node_output_channels)]
        ordering_node['frontend']['requant_bias_int12'] = [0 for i in range(ordering_node_output_channels)]

    if TFLITE_REQUANT:
        acc = None
        bias_data = int(-8*original_output_tensor.zero_point)
        scale = 1/8.0
        output_offset = original_output_tensor.zero_point.item()
        output_activation_min = -128
        output_activation_max = 127
        output_multiplier, cInputH, cInputL, o_shift = \
                quantize2MathBlock(acc,bias_data,scale,output_offset,output_activation_min,output_activation_max)
        ordering_node['frontend']['output_multiplier'] = [output_multiplier for i in range(ordering_node_output_channels)]
        ordering_node['frontend']['cInputH'] = [cInputH for i in range(ordering_node_output_channels)] 
        ordering_node['frontend']['cInputL'] = [cInputL for i in range(ordering_node_output_channels)] 
        ordering_node['frontend']['o_shift'] = [o_shift for i in range(ordering_node_output_channels)] 


    ordering_node['reorder_node'] = True
    ordering_node['output_reorder_node'] = True
    ir.graph.add_nodes_from([(ordering_node_name, ordering_node)])
    # Update producer of the original output tensor to be the conv reordering node
    original_output_tensor.producer = ordering_node_name
    ir.graph.add_edge(node_name,ordering_node_name)
    node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(node) # We need to update it so that following nodes order is according to execution order
    preceding_nodes_names = list(ir.graph.predecessors(ordering_node_name))
    preceding_nodes_params = [(preceding_nodes_names[0],0)]
    ordering_node['frontend']['preceding_nodes_params'] = preceding_nodes_params

def insert_ordering_conv_after_node(ir: internal_representation.IR,node):
    node_name = node['name']
    ordering_node_name = node_name+'_ordering_conv'
    ordering_node = {}
    ordering_node['name'] = ordering_node_name
    ordering_node['attributes']={}
    ordering_node['op_type'] = 'Conv'
    ordering_node['attributes']['kernel_shape'] = [1,1]
    ordering_node['attributes']['pads'] = [0,0,0,0]
    ordering_node['outputs'] = node['outputs']
    if len(node['outputs'])!=1:
        raise ValueError ('We currently dont support output nodes with multiple output tensors')
    # Rename the node's output tensor name
    original_output_tensor_name = node['outputs'][0]
    original_output_tensor = node['frontend']['output_tensor']
    pre_ordering_tensor_name = 'pre_ordering_' + original_output_tensor_name

    # If this is already an input, then need to pick a unique name
    #while pre_ordering_tensor_name in ir.tensors.keys():
    #    pre_ordering_tensor_name = 'redundant_' + pre_ordering_tensor_name
    # This fix should not be needed anymore though because 2 ordering convs are avoided
    assert pre_ordering_tensor_name not in ir.tensors.keys()

    updated_consumers = [ordering_node_name]
    pre_ordering_tensor = Tensor(pre_ordering_tensor_name,None,producer=node_name,consumers=updated_consumers,
                                 is_constant=False,shape = original_output_tensor.shape,scale = original_output_tensor.scale,
                                 zero_point=original_output_tensor.zero_point,folding_factor_x=original_output_tensor.folding_factor_x,
                                 folding_factor_y=original_output_tensor.folding_factor_y)
    ir.tensors[pre_ordering_tensor_name] = pre_ordering_tensor

    node['frontend']['output_tensor'] = pre_ordering_tensor
    node['outputs'] = [pre_ordering_tensor_name]
    ordering_conv_weights_tensor_name = node_name+'_ordering_conv_w'
    ordering_conv_biases_tensor_name = node_name+'_ordering_conv_b'
    ordering_node['inputs'] = [pre_ordering_tensor_name,ordering_conv_weights_tensor_name,ordering_conv_biases_tensor_name]
    original_node_output_shape = node['frontend']['output_tensor'].get_folded_shape()
    ordering_node_input_channels = original_node_output_shape[1] # Backend input/output channels contains actual # of channels. If preceding conv is a folding one, its backend output channels will be before the folding so we need to take shape from the output tensor
    ordering_node_output_channels = ordering_node_input_channels # Ordering node output channels is equal to input channels

    ordering_node['backend'] = {}
    ordering_node['backend']['input_channels'] = ordering_node_input_channels
    ordering_node['backend']['output_channels'] = ordering_node_output_channels
    ordering_node['backend']['ic_splits'] = 1
    ordering_node['frontend'] = {}
    ordering_node['frontend']['following_nodes_params'] = []
    y_tiles = node['frontend']['y_tiles']
    x_slices = node['frontend']['x_slices']
    tile_sizes,per_tile_read_start_line,per_tile_write_start_line=get_y_tile_sizes(original_output_tensor.get_folded_shape()[2],k3_nodes_in_blob=0)
    y_tiles = len(tile_sizes)
    ordering_node['frontend']['y_tiles'] = y_tiles
    current_grid_mode, _, _,current_op_num_grids,_,z_tiles,\
        output_padding_start_x = get_op_grid_config(ordering_node,original_node_output_shape[2],original_node_output_shape[3],original_node_output_shape[1],
                                                    original_node_output_shape[3],kernel_size=1,y_tiles=y_tiles,x_slices=x_slices,k3_nodes_in_blob=0)
    ordering_node['backend']['output_padding_start_x'] = output_padding_start_x

    ordering_node['backend']['gridmode'] = current_grid_mode
    ordering_node['backend']['grid_count'] = current_op_num_grids
    ordering_node['backend']['z_tiles'] = z_tiles
    output_channels_split = get_num_virtual_grids(current_grid_mode)
    ordering_node['backend']['oc_splits'] = output_channels_split
    ordering_node['backend']['ic_groups'] = [[ic for ic in range(ordering_node_input_channels)]]
    ordering_node['backend']['ic_lookup_dicts'] = [[ic for ic in range(ordering_node_input_channels)]]

    # Dans TODO: with the below, empty output channels will be written with "junk". This happens after ordering calc and rewrite of the ordering_conv weights which decide copy order of ic to oc
    ordering_node['backend']['oc_groups'] = [[oc for oc in range(0+i,ordering_node_output_channels,output_channels_split)] for i in range(output_channels_split)] 
    ordering_node['backend']['deepconv'] = True
    ordering_node['frontend']['input_tensor'] = pre_ordering_tensor
    ordering_node['frontend']['output_tensor'] = original_output_tensor
    ordering_node['frontend']['input_tensor_zp'] = node['frontend']['output_tensor_zp']
    ordering_node['frontend']['output_tensor_zp'] = node['frontend']['output_tensor_zp']
    ordering_node['frontend']['kernel_size'] = 1
    ordering_node['frontend']['stride'] = 1
    original_folded_output_tensor_shape = original_output_tensor.get_folded_shape()
    ordering_node['frontend']['sparse_macs'] = ordering_node_output_channels*original_folded_output_tensor_shape[2]*original_folded_output_tensor_shape[3]
    ordering_node['frontend']['input_folding_factor_x'] = node['frontend']['output_folding_factor_x']
    ordering_node['frontend']['input_folding_factor_y'] = node['frontend']['output_folding_factor_y']
    ordering_node['frontend']['input_tensor'].folding_factor_x = node['frontend']['output_tensor'].folding_factor_x
    ordering_node['frontend']['input_tensor'].folding_factor_y = node['frontend']['output_tensor'].folding_factor_y
    ordering_node['frontend']['output_folding_factor_x'] = node['frontend']['output_folding_factor_x']
    ordering_node['frontend']['output_folding_factor_y'] = node['frontend']['output_folding_factor_y']
    ordering_node['frontend']['output_tensor'].folding_factor_x = node['frontend']['output_tensor'].folding_factor_x
    ordering_node['frontend']['output_tensor'].folding_factor_y = node['frontend']['output_tensor'].folding_factor_y


    ordering_weights_shape = [ordering_node_output_channels,ordering_node_output_channels,1,1]
    ordering_biases_shape = [ordering_node_output_channels]
    # Add:
    # 1) new output tensor for node (the original output will be used by the ordering node)
    # 2) ordering node weights tensor
    # 3) ordering node biases tensor
    weights_tensor = Tensor(ordering_conv_weights_tensor_name,None,is_constant=True, shape = ordering_weights_shape, consumers=[ordering_node_name])
    biases_tensor = Tensor(ordering_conv_biases_tensor_name,None,is_constant=True, shape = ordering_biases_shape, consumers=[ordering_node_name])
    ir.tensors[ordering_conv_weights_tensor_name] = weights_tensor
    ir.tensors[ordering_conv_biases_tensor_name] = biases_tensor
    if (ordering_node['frontend']['input_folding_factor_x']>0 or ordering_node['frontend']['input_folding_factor_y']>0):
        ordering_node['frontend']['folded_weights_tensor'] = weights_tensor
        ordering_node['frontend']['folded_biases_tensor'] = biases_tensor
        ordering_node['frontend']['folded_kernel_size'] = 1
        ordering_node['frontend']['folded_requant_scale_shift'] = [(MAX_REDUCE_BUS_WIDTH - 3) for i in range(ordering_node_output_channels)] # This will cause zero shift in the mac stage
        ordering_node['frontend']['folded_requant_bias_int12'] = [0 for i in range(ordering_node_output_channels)]
        requant_scale_uint14 = 2 ** (FRACTIONAL_BITS+1+4) # This will cause zero shift in the RQ stage
        ordering_node['frontend']['folded_requant_scale_uint14'] = [requant_scale_uint14 for i in range(ordering_node_output_channels)]
        mac_rough_shift_mux = 0
        ordering_node['frontend']['folded_mac_rough_shift_mux'] = [mac_rough_shift_mux for i in range(ordering_node_output_channels)]

        ordering_node['frontend']['folded_requant_scale_float'] = 0.25
        if TFLITE_REQUANT:
            ordering_node['frontend']['folded_padding'] = ordering_node['attributes']['pads']
        else:
            ordering_node['frontend']['folded_padding'] = 0
    else:
        ordering_node['frontend']['weights_tensor'] = weights_tensor
        ordering_node['frontend']['biases_tensor'] = biases_tensor
        ordering_node['frontend']['requant_scale_shift'] = [(MAX_REDUCE_BUS_WIDTH - 3) for i in range(ordering_node_output_channels)] # This will cause zero shift in the mac stage
        requant_scale_uint14 = 2 ** (FRACTIONAL_BITS+1+4) # This will cause zero shift in the RQ stage
        ordering_node['frontend']['requant_scale_uint14'] = [requant_scale_uint14 for i in range(ordering_node_output_channels)]
        mac_rough_shift_mux = 0
        ordering_node['frontend']['mac_rough_shift_mux'] = [mac_rough_shift_mux for i in range(ordering_node_output_channels)]
        ordering_node['frontend']['requant_bias_int12'] = [0 for i in range(ordering_node_output_channels)]
        ordering_node['frontend']['requant_scale_float'] = 0.25
        if TFLITE_REQUANT:
            ordering_node['frontend']['padding'] = ordering_node['attributes']['pads']
        else:
            ordering_node['frontend']['padding'] = 0

    if TFLITE_REQUANT:
        acc = None
        bias_data = int(-8*original_output_tensor.zero_point)
        scale = 1/8.0
        output_offset = original_output_tensor.zero_point.item()
        output_activation_min = -128
        output_activation_max = 127
        output_multiplier, cInputH, cInputL, o_shift = \
                quantize2MathBlock(acc,bias_data,scale,output_offset,output_activation_min,output_activation_max)
        ordering_node['frontend']['output_multiplier'] = [output_multiplier for i in range(ordering_node_output_channels)]
        ordering_node['frontend']['cInputH'] = [cInputH for i in range(ordering_node_output_channels)]
        ordering_node['frontend']['cInputL'] = [cInputL for i in range(ordering_node_output_channels)]
        ordering_node['frontend']['o_shift'] = [o_shift for i in range(ordering_node_output_channels)]

    ordering_node['reorder_node'] = True
    ordering_node['y_unfolding_reorder_node'] = True
    ir.graph.add_nodes_from([(ordering_node_name, ordering_node)])
    # Update producer of the original output tensor to be the conv reordering node
    original_output_tensor.producer = ordering_node_name
    original_node_successors = list(ir.graph.successors(node['name']))
    for original_node_successor_name in original_node_successors:
        ir.graph.remove_edge(node['name'],original_node_successor_name)
        ir.graph.add_edge(ordering_node_name,original_node_successor_name)
    ir.graph.add_edge(node_name,ordering_node_name)
    node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(node) # We need to update it so that following nodes order is according to execution order
    following_nodes_params = ir.get_updated_following_nodes(ordering_node)
    ordering_node['frontend']['following_nodes_params'] = following_nodes_params # We need to update it so that following nodes order is according to execution order
    for following_node_params in following_nodes_params:
        following_node_name = following_node_params[0]
        following_node = ir.graph.nodes()[following_node_name]
        following_node['frontend']['preceding_nodes_params'] = ir.get_updated_preceding_nodes(following_node)

    ordering_node['frontend']['preceding_nodes_params'] = ir.get_updated_preceding_nodes(ordering_node)
    ordering_node_blob_idx = node['frontend']['tiling_blob_idx']
    ordering_node['frontend']['tiling_blob_idx'] = ordering_node_blob_idx
    # Update blob with ordering node
    current_blob = ir.tiling_blobs[ordering_node_blob_idx]
    nodes_in_blob = current_blob.nodes_in_blob
    node_idx = nodes_in_blob.index(node_name)
    nodes_in_blob.insert(node_idx+1,ordering_node_name)
    current_blob.num_of_nodes_in_blob+=1
    update_node_tiling_info(ir,ordering_node)
    # Update execution order
    pre_ordering_node_execution_index = ir.lexicographical_topological_sorted_graph.index(node_name)
    ir.lexicographical_topological_sorted_graph.insert(pre_ordering_node_execution_index+1,ordering_node_name)

def add_output_ordering_conv(ir: internal_representation.IR) -> internal_representation.IR:
    workload_output_tensors_names = ir.outputs
    output_nodes=[]
    for current_output_tensor_name in workload_output_tensors_names:
        if current_output_tensor_name in ir.tensors:
            current_output_node_name = ir.tensors[current_output_tensor_name].producer
        else:
            raise ValueError ('Workloads output %s not found in tensors db. Please check ...' % current_output_tensor_name)
        current_output_node = ir.graph.nodes[current_output_node_name]
        create_ordering_conv(ir,current_output_node)
    ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    return ir

def add_pre_folding_unfolding_ordering_conv_common(
    ir: internal_representation.IR,
    fold_type: str
) -> internal_representation.IR:

    assert fold_type in ['folding', 'unfolding']
    handled_nodes = []
    for tiling_blob_idx,tiling_blob in ir.tiling_blobs.items():
        blob_input_tensors = tiling_blob.inputs
        for blob_input_tensor in blob_input_tensors:
            for consumer_node_name in blob_input_tensor.consumers:
                if consumer_node_name in tiling_blob.nodes_in_blob:
                    consumer_node = ir.graph.nodes()[consumer_node_name]
                    if f'force_{fold_type}_y' in consumer_node['frontend']:
                        if consumer_node_name in handled_nodes:
                            continue
                        if consumer_node['op_type'] in MULTIPLE_INPUT_OPS:
                            raise ValueError (f'Multi input node: %s, currently dont support y_{fold_type}.' % consumer_node_name)
                        #We need to insert an ordering conv after the current tensor producer node
                        producer_node_name = blob_input_tensor.producer

                        # Example: if producer in blob A has 2 outputs in blobs B and C, and an ordering conv
                        # was already added to blob A for blob B, don't need to add another one in blob A for blob C
                        if 'ordering_conv' in producer_node_name:
                            continue

                        producer_node = ir.graph.nodes()[producer_node_name]
                        insert_ordering_conv_after_node(ir,producer_node)
                        handled_nodes.append(consumer_node_name)
    return ir

def add_pre_unfolding_ordering_conv(ir:internal_representation.IR) -> internal_representation.IR:
    return add_pre_folding_unfolding_ordering_conv_common(ir, "unfolding")

def add_pre_folding_ordering_conv(ir:internal_representation.IR) -> internal_representation.IR:
    return add_pre_folding_unfolding_ordering_conv_common(ir, "folding")

# Add ordering conv before Sync node
def add_pre_sync_ordering_conv(ir: internal_representation.IR) -> internal_representation.IR:
    sync_nodes = []
    for node_name in ir.graph.nodes():
        node = ir.graph.nodes()[node_name]
        if node['op_type'] == 'Sync':
            sync_nodes.append(node)

    for sync_node in sync_nodes:
        input_tensor_names = sync_node['inputs']
        for input_tensor_name in input_tensor_names:
            producer_node_name = ir.tensors[input_tensor_name].producer
            if 'ordering_conv' in producer_node_name:
                continue
            producer_node = ir.graph.nodes()[producer_node_name]
            insert_ordering_conv_after_node(ir, producer_node)

    return ir

def get_details_from_constants(var_factor):
    max_value = max(var_factor)
    min_value = min(var_factor)
    if max_value > 0 :
        zero_point = -128
        q_value = 127
        fp_value = max_value
    else:
        zero_point = 127
        q_value = -128
        fp_value = min_value
    scale = fp_value/(q_value-zero_point)
    q_value = [round((i/scale) + zero_point) for i in var_factor]
    return scale, zero_point, q_value

# Insert an identity conv before a node.
# This is similar to insert_ordering_conv_after_node.
# A new tensor is added in between, see example below:
#
#   Before:          After:
#
#                       │ Tensor1
#                    ┌──▼──┐
#    │ Tensor1       │ident│ Node1_identity
# ┌──▼──┐            └──┬──┘
# │node │ Node1         │ Tensor1_identity
# └──┬──┘       ──►  ┌──▼──┐
#    │ Tensor2       │node │ Node1
#    ▼               └──┬──┘
#                       │ Tensor2
#                       ▼
#
def insert_identity_conv_before_node(ir:internal_representation.IR,node,new_name):
    node_name = node['name']

    # Create the new identity node
    identity_node_name = new_name
    identity_node = {}
    identity_node['name'] = identity_node_name
    identity_node['attributes']={}
    identity_node['op_type'] = 'Conv'
    identity_node['op_sub_type'] = 'Identity'
    identity_node['attributes']['kernel_shape'] = [1,1]
    identity_node['attributes']['pads'] = [0,0,0,0]

    # calculate folding factor 
    original_input_tensor_name = node['inputs'][0]
    original_input_tensor = node['frontend']['input_tensor']
    if ('_fold_x' in new_name):
        folding_factor_x = int(new_name.split('_')[-1])
        identity_node_input_folding_x = folding_factor_x
        identity_node_output_folding_x = folding_factor_x + 1
        original_input_tensor.folding_factor_x = identity_node_input_folding_x

        # calc the new x_slices for input
        identity_node_input_xslices = int((original_input_tensor.shape[3]+15)//16)
        identity_node_input_xslices = math.ceil(identity_node_input_xslices/(2**identity_node_input_folding_x))
        original_input_tensor.x_slices = identity_node_input_xslices

        # calc the new x_slices for output        
        identity_node_output_xslices = math.ceil(identity_node_input_xslices/2)
    else:
        folding_factor_x = original_input_tensor.folding_factor_x
        identity_node_input_folding_x = folding_factor_x
        identity_node_output_folding_x = folding_factor_x
        identity_node_input_xslices = node['frontend']['input_tensor'].x_slices
        identity_node_output_xslices = node['frontend']['x_slices']
        
    # Create a new tensor for the output of this node
    post_identity_tensor_name = new_name + '_output'
    assert post_identity_tensor_name not in ir.tensors.keys()
    post_identity_tensor = Tensor(post_identity_tensor_name,None,producer=identity_node_name,consumers=[node_name],
                                 is_constant=False,shape = original_input_tensor.shape,scale = original_input_tensor.scale,
                                 zero_point=original_input_tensor.zero_point,folding_factor_x = identity_node_output_folding_x,
                                 folding_factor_y=original_input_tensor.folding_factor_y, x_slices=identity_node_output_xslices)
    ir.tensors[post_identity_tensor_name] = post_identity_tensor
    identity_node['outputs'] = [post_identity_tensor_name]


    # Update the inputs of the new identity node
    identity_conv_weights_tensor_name = new_name+'_conv_w'
    identity_conv_biases_tensor_name = new_name+'_conv_b'
    identity_node['inputs'] = [original_input_tensor_name,identity_conv_weights_tensor_name,identity_conv_biases_tensor_name]

    # Set the frontend and backend properties of this new identity node
    # These are set like in insert_ordering_conv_after_node (ic/oc splits, ic/oc groups, etc.)

    original_node_input_shape = original_input_tensor.get_folded_shape()
    identity_node_input_channels = original_node_input_shape[1]
    identity_node_output_channels = identity_node_input_channels

    identity_node['backend'] = {}
    identity_node['backend']['input_channels'] = identity_node_input_channels
    identity_node['backend']['output_channels'] = identity_node_output_channels

    identity_node['backend']['ic_splits'] = node['backend']['ic_splits']
    identity_node['frontend'] = {}
    identity_node['frontend']['following_nodes_params'] = []

    y_tiles = node['frontend']['y_tiles']
    tile_sizes,per_tile_read_start_line,per_tile_write_start_line=get_y_tile_sizes(original_input_tensor.get_folded_shape()[2],k3_nodes_in_blob=0)
    y_tiles = len(tile_sizes)
    identity_node['frontend']['y_tiles'] = y_tiles
    current_grid_mode, _, _,current_op_num_grids,_,z_tiles,\
        output_padding_start_x = get_op_grid_config(identity_node,original_node_input_shape[2],original_node_input_shape[3],original_node_input_shape[1],
                                                    original_node_input_shape[3],kernel_size=1,y_tiles=y_tiles,x_slices=identity_node_input_xslices,k3_nodes_in_blob=0)
    identity_node['backend']['output_padding_start_x'] = output_padding_start_x

    identity_node['backend']['gridmode'] = current_grid_mode
    identity_node['backend']['grid_count'] = current_op_num_grids
    identity_node['backend']['z_tiles'] = z_tiles
    output_channels_split = get_num_virtual_grids(current_grid_mode)
    identity_node['backend']['oc_splits'] = output_channels_split
    identity_node['backend']['ic_groups'] = [[ic for ic in range(identity_node_input_channels)]]
    identity_node['backend']['ic_lookup_dicts'] = [[ic for ic in range(identity_node_input_channels)]]

    identity_node['backend']['oc_groups'] = [[oc for oc in range(0+i,identity_node_output_channels,output_channels_split)] for i in range(output_channels_split)]
    identity_node['backend']['deepconv'] = True
    identity_node['frontend']['input_tensor'] = original_input_tensor
    identity_node['frontend']['output_tensor'] = post_identity_tensor
    identity_node['frontend']['input_tensor_zp'] = node['frontend']['input_tensor_zp']
    identity_node['frontend']['output_tensor_zp'] = node['frontend']['input_tensor_zp']
    identity_node['frontend']['input_tensor_scale'] = node['frontend']['input_tensor_scale']
    identity_node['frontend']['output_tensor_scale'] = node['frontend']['input_tensor_scale']
    identity_node['frontend']['kernel_size'] = 1
    identity_node['frontend']['stride'] = 1
    original_folded_input_tensor_shape = original_input_tensor.get_folded_shape()
    identity_node['frontend']['sparse_macs'] = identity_node_input_channels*original_folded_input_tensor_shape[2]*original_folded_input_tensor_shape[3]
    identity_node['frontend']['input_folding_factor_x'] = identity_node_input_folding_x
    identity_node['frontend']['input_folding_factor_y'] = node['frontend']['input_folding_factor_y']
    identity_node['frontend']['input_channels'] = identity_node_input_channels
    identity_node['frontend']['output_channels'] = identity_node_output_channels
    identity_node['frontend']['input_tensor'].folding_factor_x = identity_node_input_folding_x
    identity_node['frontend']['input_tensor'].folding_factor_y = node['frontend']['input_tensor'].folding_factor_y
    identity_node['frontend']['input_tensor'].x_slices = identity_node_input_xslices
    identity_node['frontend']['output_folding_factor_x'] = identity_node_output_folding_x
    identity_node['frontend']['output_folding_factor_y'] = node['frontend']['input_folding_factor_y']
    identity_node['frontend']['output_tensor'].folding_factor_x = identity_node_output_folding_x
    identity_node['frontend']['output_tensor'].folding_factor_y = node['frontend']['input_tensor'].folding_factor_y
    identity_node['frontend']['output_tensor'].x_slices = identity_node_output_xslices
    identity_node['frontend']['x_slices'] = identity_node_input_xslices
    identity_node['frontend']['x_wrapping'] = node['frontend']['x_wrapping']

    if DEBUG_OPTIMIZE_FIRST_LAYER_DDR_ACCESS:
        if (len(node['frontend']['preceding_nodes_params']) == 0) and ('_fold_x_0' in new_name) and (identity_node_input_xslices % 4 == 0):
            identity_node['frontend']['input_tensor'].num_packed_xslices = 4

    #weight_channels = int(identity_node_input_channels/(2**original_input_tensor.folding_factor_y))
    weight_channels = identity_node_input_channels
    identity_weights_shape = [weight_channels,weight_channels,1,1]
    identity_biases_shape = [identity_node_input_channels]

    if ('_fold_x_1' in new_name) and ir.uint8_int8_conversion and ((len(ir.mean) > 1) or (len(ir.scale) > 1)):
        assert len(ir.mean) == 3
        assert len(ir.scale) == 3
        
        scale_factor = [1 / j for j in ir.scale]
        scale, scale_zero_point, scale_q_value = get_details_from_constants(scale_factor)
        # mean_factor = [i / j for i, j in zip(ir.mean, ir.scale)]
        # mean_factor = [-1 * j for j in mean_factor]
        # mean_scale, mean_zero_point, mean_q_value = get_details_from_constants(mean_factor)

        w_ch = 3
        conv_W = np.zeros([w_ch,w_ch,1,1], dtype=np.int8)
        conv_B = np.zeros([w_ch], dtype=np.int32)
        conv_W_scale = np.zeros([w_ch], dtype=np.float)
        conv_B_scale = [1.0 for _ in range(identity_node_input_channels)]
        conv_W_zp = [0 for _ in range(identity_node_input_channels)]
        conv_B_zp = [0 for _ in range(identity_node_input_channels)]
        for k_ in range(w_ch):
            w_val = scale_q_value[k_]
            if w_val - scale_zero_point < 0:
                w_val_next = -127
            else:
                w_val_next = 127
            w_scale_next = scale / (w_val_next / (w_val - scale_zero_point))
            conv_W[k_,k_,0,0] += w_val_next
            conv_W_scale[k_] = w_scale_next
            conv_B[k_] = int(ir.mean[k_]*scale_zero_point) - original_input_tensor.zero_point.item()
        
        folded_weights = folding_algo.get_asym_folded_weights(conv_W,input_folding_factor_x=identity_node_input_folding_x, \
                                                            input_folding_factor_y=identity_node['frontend']['input_folding_factor_y'], \
                                                            stride_x=1,stride_y=1)
        conv_W_scale = folding_algo.get_asym_folded_per_oc_params(conv_W_scale,input_folding_factor_x=identity_node_input_folding_x, \
                                                                input_folding_factor_y=identity_node['frontend']['input_folding_factor_y'], \
                                                                stride_x=1,stride_y=1)
        weights_tensor = Tensor(identity_conv_weights_tensor_name,folded_weights,is_constant=True,shape=folded_weights.shape, \
                                scale=conv_W_scale, zero_point=conv_W_zp)
        
        folded_biases = folding_algo.get_asym_folded_per_oc_params(conv_B,input_folding_factor_x=identity_node_input_folding_x, \
                                                                input_folding_factor_y=identity_node['frontend']['input_folding_factor_y'], \
                                                                stride_x=1,stride_y=1)
        conv_B_scale = folding_algo.get_asym_folded_per_oc_params(conv_B_scale,input_folding_factor_x=identity_node_input_folding_x, \
                                                                input_folding_factor_y=identity_node['frontend']['input_folding_factor_y'], \
                                                                stride_x=1,stride_y=1)
        biases_tensor = Tensor(identity_conv_biases_tensor_name,folded_biases,is_constant=True,shape = folded_biases.shape,
                               scale=conv_B_scale, zero_point=conv_B_zp)
    else:
        conv_W = np.zeros(identity_weights_shape, dtype=np.int8)
        conv_B = np.zeros(identity_biases_shape, dtype=np.int32)
        conv_W_scale = np.zeros(identity_node_input_channels, dtype=np.float32)
        for current_cout in range(weight_channels):
            conv_W[current_cout,current_cout,0,0] = 8 # value in int8
            conv_W_scale[current_cout] = 1/8
        # Create the new weights and bias tensor
        weights_tensor = Tensor(identity_conv_weights_tensor_name,conv_W,is_constant=True, shape = identity_weights_shape, consumers=[identity_node_name],scale=conv_W_scale)
        biases_tensor = Tensor(identity_conv_biases_tensor_name,conv_B,is_constant=True, shape = identity_biases_shape, consumers=[identity_node_name])
    
    ir.tensors[identity_conv_weights_tensor_name] = weights_tensor
    ir.tensors[identity_conv_biases_tensor_name] = biases_tensor

    # Set quantization parameters
    if (identity_node['frontend']['input_folding_factor_x']>0 or identity_node['frontend']['input_folding_factor_y']>0):
        # if (identity_node['frontend']['input_folding_factor_y'] > 0):
        #     folded_weights = folding_algo.get_asym_folded_weights(weights_tensor.data,input_folding_factor_x=0,input_folding_factor_y=identity_node['frontend']['input_folding_factor_y'])
        #     kernel_shape = folded_weights.shape
        #     folded_weights_tensor_name = weights_tensor.name+'_folded'
        #     folded_weights_tensor = Tensor(folded_weights_tensor_name,folded_weights,is_constant=True,shape = folded_weights.shape,scale = weights_tensor.scale,
        #                              zero_point=weights_tensor.zero_point)
        #     identity_node['frontend']['folded_weights_tensor'] = folded_weights_tensor
        # else:
        identity_node['frontend']['folded_weights_tensor'] = weights_tensor
        identity_node['frontend']['folded_biases_tensor'] = biases_tensor
        identity_node['frontend']['folded_kernel_size'] = 1
        identity_node['frontend']['folded_requant_scale_shift'] = [(MAX_REDUCE_BUS_WIDTH - 3) for i in range(identity_node_output_channels)] # This will cause zero shift in the mac stage
        identity_node['frontend']['folded_requant_bias_int12'] = [0 for i in range(identity_node_output_channels)]
        requant_scale_uint14 = 2 ** (FRACTIONAL_BITS+1+4) # This will cause zero shift in the RQ stage
        identity_node['frontend']['folded_requant_scale_uint14'] = [requant_scale_uint14 for i in range(identity_node_output_channels)]
        mac_rough_shift_mux = 0
        identity_node['frontend']['folded_mac_rough_shift_mux'] = [mac_rough_shift_mux for i in range(identity_node_output_channels)]
        identity_node['frontend']['folded_requant_scale_float'] = 0.25
        if TFLITE_REQUANT:
            identity_node['frontend']['folded_padding'] = identity_node['attributes']['pads']
        else:
            identity_node['frontend']['folded_padding'] = 0
    else:
        identity_node['frontend']['weights_tensor'] = weights_tensor
        identity_node['frontend']['biases_tensor'] = biases_tensor
        identity_node['frontend']['requant_scale_shift'] = [(MAX_REDUCE_BUS_WIDTH - 3) for i in range(identity_node_output_channels)] # This will cause zero shift in the mac stage
        requant_scale_uint14 = 2 ** (FRACTIONAL_BITS+1+4) # This will cause zero shift in the RQ stage
        identity_node['frontend']['requant_scale_uint14'] = [requant_scale_uint14 for i in range(identity_node_output_channels)]
        mac_rough_shift_mux = 0
        identity_node['frontend']['mac_rough_shift_mux'] = [mac_rough_shift_mux for i in range(identity_node_output_channels)]
        identity_node['frontend']['requant_bias_int12'] = [0 for i in range(identity_node_output_channels)]
        identity_node['frontend']['requant_scale_float'] = 0.25
        if TFLITE_REQUANT:
            identity_node['frontend']['padding'] = identity_node['attributes']['pads']
        else:
            identity_node['frontend']['padding'] = 0

    if TFLITE_REQUANT:
        acc = None
        if ('_fold_x_1' in new_name) and ir.uint8_int8_conversion and ((len(ir.mean) > 1) or (len(ir.scale) > 1)):
            output_multiplier = [0 for i in range(identity_node_output_channels)]
            cInputH = [0 for i in range(identity_node_output_channels)]
            cInputL = [0 for i in range(identity_node_output_channels)]
            o_shift = [0 for i in range(identity_node_output_channels)]
            w_int8 = weights_tensor.data
            weights_per_channel_scale = weights_tensor.scale
            for och in range(identity_node_output_channels):
                w_sum = np.sum(w_int8[och])
                requant_scale = weights_per_channel_scale[och] / original_input_tensor.scale.item()
                requant_bias = biases_tensor.data[och]
                requant_bias -= w_sum*scale_zero_point
                requant_bias = requant_bias.astype(np.int32)

                acc = None # This is the accumulator
                bias_data = int(requant_bias)
                scale = requant_scale
                output_offset = original_input_tensor.zero_point.item()
                output_activation_min = -128
                output_activation_max = 127
                output_multiplier[och], cInputH[och], cInputL[och], o_shift[och] = \
                    quantize2MathBlock(acc,bias_data,scale,output_offset,output_activation_min,output_activation_max)

            identity_node['frontend']['output_multiplier'] = output_multiplier
            identity_node['frontend']['cInputH'] = cInputH
            identity_node['frontend']['cInputL'] = cInputL
            identity_node['frontend']['o_shift'] = o_shift
        else:
            bias_data = int(-8*original_input_tensor.zero_point)
            scale = 1/8.0
            output_offset = original_input_tensor.zero_point.item()
            output_activation_min = -128
            output_activation_max = 127
            output_multiplier, cInputH, cInputL, o_shift = \
                    quantize2MathBlock(acc,bias_data,scale,output_offset,output_activation_min,output_activation_max)
            identity_node['frontend']['output_multiplier'] = [output_multiplier for i in range(identity_node_output_channels)]
            identity_node['frontend']['cInputH'] = [cInputH for i in range(identity_node_output_channels)]
            identity_node['frontend']['cInputL'] = [cInputL for i in range(identity_node_output_channels)]
            identity_node['frontend']['o_shift'] = [o_shift for i in range(identity_node_output_channels)]

    # TODO: Maybe these can be False. This function was based on the ordering conv function.
    #identity_node['reorder_node'] = False
    #identity_node['y_unfolding_reorder_node'] = False
    ir.graph.add_nodes_from([(identity_node_name, identity_node)])

    # Remove predecessors of the given input node, and instead make them predecessors
    # of the identity node. Then make the identity a predecessor of the input node.
    for i_name in original_input_tensor.consumers:
        original_node_predecessors = list(ir.graph.predecessors(i_name))
        for original_node_predecessor_name in original_node_predecessors:
            ir.graph.remove_edge(original_node_predecessor_name,i_name)

    # Update consumer of the original input tensor
    for i_name in original_input_tensor.consumers:
        ir.graph.add_edge(identity_node_name,i_name)
        index = original_input_tensor.consumers.index(i_name)
        original_input_tensor.consumers[index] = identity_node_name
        if i_name not in post_identity_tensor.consumers:
            post_identity_tensor.consumers.append(i_name)
        next_node_info = ir.graph.nodes[i_name]
        ir.switch_input_name(next_node_info,original_input_name=original_input_tensor_name,new_input_name=post_identity_tensor_name)
        ir.switch_input_tensor(next_node_info,original_input_tensor=original_input_tensor,new_input_tensor=post_identity_tensor)
    original_input_tensor.consumers = list(set(original_input_tensor.consumers))
    
    # Update following nodes params
    node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(node) # We need to update it so that following nodes order is according to execution order
    following_nodes_params = ir.get_updated_following_nodes(identity_node)
    identity_node['frontend']['following_nodes_params'] = following_nodes_params # We need to update it so that following nodes order is according to execution order
    for following_node_params in following_nodes_params:
        following_node_name = following_node_params[0]
        following_node = ir.graph.nodes()[following_node_name]
        following_node['frontend']['preceding_nodes_params'] = ir.get_updated_preceding_nodes(following_node)

    identity_node['frontend']['preceding_nodes_params'] = ir.get_updated_preceding_nodes(identity_node)

    if 'tiling_blob_idx' in node['frontend']:
        # Put the identity node in the same blob as the input node
        identity_node_blob_idx = node['frontend']['tiling_blob_idx']
        identity_node['frontend']['tiling_blob_idx'] = identity_node_blob_idx
        # Update this blob to add the identity node
        current_blob = ir.tiling_blobs[identity_node_blob_idx]
        nodes_in_blob = current_blob.nodes_in_blob
        node_idx = nodes_in_blob.index(node_name)
        # Insert the identity node before the given node
        nodes_in_blob.insert(node_idx, identity_node_name)
        current_blob.num_of_nodes_in_blob += 1
        update_node_tiling_info(ir,identity_node)

    # Update execution order
    pre_identity_node_execution_index = ir.lexicographical_topological_sorted_graph.index(node_name)
    ir.lexicographical_topological_sorted_graph.insert(pre_identity_node_execution_index,identity_node_name)
    return identity_node_name

# Add identity convolution before k3 nodes at the start of 1-tile blobs
def add_identity_before_k3_starting_1T_blob(ir:internal_representation.IR) -> internal_representation.IR:
    for tiling_blob_idx,tiling_blob in ir.tiling_blobs.items():
        first_node_name = tiling_blob.nodes_in_blob[0]
        first_node = ir.graph.nodes[first_node_name]
        if (first_node['frontend']['y_tiles'] != 1):
            continue
        if (first_node['frontend']['input_folding_factor_x'] != 0):
            continue
        if not ir.is_k3_node(first_node):
            continue
        insert_identity_conv_before_node(ir,first_node,new_name=f'{first_node_name}_identity')
    return ir

def insert_identity_conv_after_node(ir: internal_representation.IR,node,new_name):
    node_name = node['name']
    identity_node_name = new_name
    identity_node = {}
    identity_node['name'] = identity_node_name
    identity_node['attributes']={}
    identity_node['op_type'] = 'Conv'
    identity_node['attributes']['kernel_shape'] = [1,1]
    identity_node['attributes']['pads'] = [0,0,0,0]
    identity_node['outputs'] = node['outputs']
    if len(node['outputs'])!=1:
        raise ValueError ('We currently dont support output nodes with multiple output tensors')
    # Rename the node's output tensor name
    original_output_tensor = node['frontend']['output_tensor']
    pre_identity_tensor_name = new_name + '_input'
    assert pre_identity_tensor_name not in ir.tensors.keys()

    input_folding_factor_y = original_output_tensor.folding_factor_y
    output_folding_factor_y = original_output_tensor.folding_factor_y
    following_nodes_params = node['frontend']['following_nodes_params']
    following_node_is_concat = False
    for following_node_params in following_nodes_params:
        if ('CONCATENATION' in following_node_params[0]):
            following_node_is_concat = True
            break

    if following_node_is_concat and (output_folding_factor_y > 0): # We can unfold y if output folding factor is already 0
        input_folding_factor_y-=1
        output_folding_factor_y-=1
    
    input_folding_coef_y = math.pow(2,input_folding_factor_y)
    folded_y_size = original_output_tensor.shape[2] // input_folding_coef_y
    num_y_tiles = math.ceil(folded_y_size / MAX_GRID_HEIGHT)

    # calculate folding factor 
    identity_node_input_folding_x = original_output_tensor.folding_factor_x
    identity_node_output_folding_x = identity_node_input_folding_x - 1

    #new version for the x is not always divisible by 16
    identity_node_input_xslices =  math.ceil(math.ceil(original_output_tensor.shape[3]/16)/(2**identity_node_input_folding_x))
    identity_node_output_xslices = math.ceil(math.ceil(original_output_tensor.shape[3]/16)/(2**identity_node_output_folding_x))
    identity_node_input_channels = original_output_tensor.shape[1]

    #old version for the x is divisible by 16
    # identity_node_input_xslices = int(original_output_tensor.shape[3]/16)
    # identity_node_input_xslices = int(identity_node_input_xslices/(2**identity_node_input_folding_x))
    # identity_node_output_xslices = int(identity_node_input_xslices*2)
    # identity_node_input_channels = original_output_tensor.shape[1]

    original_output_tensor.folding_factor_x = identity_node_output_folding_x
    original_output_tensor.x_slices = identity_node_output_xslices
    
    updated_consumers = [identity_node_name]
    pre_identity_tensor = Tensor(pre_identity_tensor_name,None,producer=node_name,consumers=updated_consumers,
                                 is_constant=False,shape = original_output_tensor.shape,scale = original_output_tensor.scale,
                                 zero_point=original_output_tensor.zero_point,folding_factor_x=identity_node_input_folding_x,
                                 folding_factor_y=original_output_tensor.folding_factor_y, x_slices=identity_node_input_xslices)
    ir.tensors[pre_identity_tensor_name] = pre_identity_tensor

    node['frontend']['output_tensor'] = pre_identity_tensor
    node['outputs'] = [pre_identity_tensor_name]
    identity_conv_weights_tensor_name = new_name + '_conv_w'
    identity_conv_biases_tensor_name = new_name + '_conv_b'
    identity_node['inputs'] = [pre_identity_tensor_name,identity_conv_weights_tensor_name,identity_conv_biases_tensor_name]
    original_node_output_shape = node['frontend']['output_tensor'].get_folded_shape()
    identity_node_output_channels = identity_node_input_channels

    identity_node['backend'] = {}
    identity_node['backend']['input_channels'] = original_node_output_shape[1]
    identity_node['backend']['output_channels'] = (original_node_output_shape[1] // 2)
    identity_node['backend']['ic_splits'] = 1
    identity_node['frontend'] = {}
    if following_node_is_concat:
        identity_node['frontend']['force_unfolding_y'] = True
    identity_node['frontend']['following_nodes_params'] = []
    y_tiles = num_y_tiles
    identity_node['frontend']['y_tiles'] = y_tiles
    current_grid_mode, _, _,current_op_num_grids,_,z_tiles,\
        output_padding_start_x = get_op_grid_config(identity_node,original_node_output_shape[2],original_node_output_shape[3],original_node_output_shape[1],
                                                    original_node_output_shape[3],kernel_size=1,y_tiles=y_tiles,x_slices=identity_node_input_xslices,k3_nodes_in_blob=0)
    identity_node['backend']['output_padding_start_x'] = output_padding_start_x

    identity_node['backend']['gridmode'] = current_grid_mode
    identity_node['backend']['grid_count'] = current_op_num_grids
    identity_node['backend']['z_tiles'] = z_tiles
    output_channels_split = get_num_virtual_grids(current_grid_mode)
    identity_node['backend']['oc_splits'] = output_channels_split
    identity_node['backend']['ic_groups'] = [[ic for ic in range(original_node_output_shape[1])]]
    identity_node['backend']['ic_lookup_dicts'] = [[ic for ic in range(original_node_output_shape[1])]]

    identity_node['backend']['oc_groups'] = [[oc for oc in range(0+i,(original_node_output_shape[1] // 2),output_channels_split)] for i in range(output_channels_split)]
    identity_node['backend']['deepconv'] = True
    identity_node['frontend']['input_tensor'] = pre_identity_tensor
    identity_node['frontend']['output_tensor'] = original_output_tensor
    identity_node['frontend']['input_tensor_zp'] = node['frontend']['output_tensor_zp']
    identity_node['frontend']['output_tensor_zp'] = node['frontend']['output_tensor_zp']
    identity_node['frontend']['input_tensor_scale'] = node['frontend']['output_tensor_scale']
    identity_node['frontend']['output_tensor_scale'] = node['frontend']['output_tensor_scale']
    identity_node['frontend']['kernel_size'] = 1
    identity_node['frontend']['stride'] = 1
    original_folded_output_tensor_shape = original_output_tensor.get_folded_shape()
    identity_node['frontend']['sparse_macs'] = identity_node_output_channels*original_folded_output_tensor_shape[2]*original_folded_output_tensor_shape[3]
    identity_node['frontend']['input_folding_factor_x'] = identity_node_input_folding_x
    identity_node['frontend']['input_folding_factor_y'] = input_folding_factor_y
    identity_node['frontend']['input_tensor'].folding_factor_x = identity_node_input_folding_x
    identity_node['frontend']['input_tensor'].folding_factor_y = node['frontend']['output_tensor'].folding_factor_y
    identity_node['frontend']['input_tensor'].x_slices = identity_node_input_xslices
    identity_node['frontend']['output_folding_factor_x'] = identity_node_output_folding_x
    identity_node['frontend']['output_folding_factor_y'] = output_folding_factor_y
    identity_node['frontend']['output_tensor'].folding_factor_x = identity_node_output_folding_x
    identity_node['frontend']['output_tensor'].folding_factor_y = output_folding_factor_y
    identity_node['frontend']['output_tensor'].x_slices = identity_node_output_xslices
    identity_node['frontend']['x_slices'] = identity_node_input_xslices
    if (current_grid_mode == GridConfig.H14xW16):
        identity_node['frontend']['x_wrapping'] = 1
    elif (current_grid_mode == GridConfig.H14xW8):
        identity_node['frontend']['x_wrapping'] = 0
    else:
        raise ValueError ("Grid Mode greater than 14x16 isn't supported")
    identity_node['frontend']['input_channels'] = identity_node_input_channels
    identity_node['frontend']['output_channels'] = identity_node_output_channels

    identity_weights_shape = [identity_node_input_channels,identity_node_input_channels,1,1]
    identity_biases_shape = [identity_node_input_channels]
    
    conv_W = np.zeros(identity_weights_shape, dtype=np.int8)
    conv_B = np.zeros(identity_biases_shape, dtype=np.int32)
    conv_W_scale = np.zeros(identity_node_input_channels, dtype=np.float32)
    conv_W_zp = np.zeros(identity_node_input_channels, dtype=np.int8)
    conv_B_scale = np.ones(identity_node_input_channels, dtype=np.float32)
    conv_B_zp = np.zeros(identity_node_input_channels, dtype=np.int8)
    for current_cout in range(identity_node_input_channels):
        conv_W[current_cout,current_cout,0,0] = 8 # value in int8
        conv_W_scale[current_cout] = 1/8

    # Add:
    # 1) new output tensor for node (the original output will be used by the identity node)
    # 2) identity node weights tensor
    # 3) identity node biases tensor
    weights_tensor = Tensor(identity_conv_weights_tensor_name,conv_W,is_constant=True, shape = identity_weights_shape, consumers=[identity_node_name], scale=conv_W_scale, zero_point=conv_W_zp)
    biases_tensor = Tensor(identity_conv_biases_tensor_name,conv_B,is_constant=True, shape = identity_biases_shape, consumers=[identity_node_name], scale=conv_B_scale, zero_point=conv_B_zp)
    ir.tensors[identity_conv_weights_tensor_name] = weights_tensor
    ir.tensors[identity_conv_biases_tensor_name] = biases_tensor
    identity_node['frontend']['weights_tensor'] = weights_tensor
    identity_node['frontend']['biases_tensor'] = biases_tensor
    identity_node['frontend']['weights_per_channel_scale'] = conv_W_scale
    identity_node['frontend']['weights_per_channel_zp'] = conv_W_zp
    identity_node['frontend']['weights_tensor'] = weights_tensor
    identity_node['frontend']['biases_tensor'] = biases_tensor
    if TFLITE_REQUANT:
        acc = None
        bias_data = int(-8*original_output_tensor.zero_point)
        scale = 1/8.0
        output_offset = original_output_tensor.zero_point.item()
        output_activation_min = -128
        output_activation_max = 127
        output_multiplier, cInputH, cInputL, o_shift = \
                quantize2MathBlock(acc,bias_data,scale,output_offset,output_activation_min,output_activation_max)
        identity_node['frontend']['output_multiplier'] = [output_multiplier for i in range(identity_node_input_channels)]
        identity_node['frontend']['cInputH'] = [cInputH for i in range(identity_node_input_channels)]
        identity_node['frontend']['cInputL'] = [cInputL for i in range(identity_node_input_channels)]
        identity_node['frontend']['o_shift'] = [o_shift for i in range(identity_node_input_channels)]


    if (identity_node['frontend']['input_folding_factor_x']>0 or identity_node['frontend']['input_folding_factor_y']>0):
        folded_weights = folding_algo.get_asym_folded_weights(weights_tensor.data,input_folding_factor_x=identity_node_input_folding_x,input_folding_factor_y=input_folding_factor_y, \
                                                            stride_x=1,stride_y=1, asymmetric_padding=False)
        kernel_shape = folded_weights.shape
        folded_weights_tensor_name = weights_tensor.name+'_folded'
        folded_weights_tensor = Tensor(folded_weights_tensor_name,folded_weights,is_constant=True,shape = folded_weights.shape,scale = weights_tensor.scale,
                                 zero_point=weights_tensor.zero_point)
        identity_node['frontend']['folded_weights_tensor'] = folded_weights_tensor

        folded_biases = folding_algo.get_asym_folded_per_oc_params(biases_tensor.data,input_folding_factor_x=identity_node_input_folding_x,input_folding_factor_y=input_folding_factor_y,stride_x=1,stride_y=1)
        folded_biases_tensor_name = biases_tensor.name+'_folded'
        folded_biases_tensor = Tensor(folded_biases_tensor_name,folded_biases,is_constant=True,shape = folded_biases.shape,scale = biases_tensor.scale,
                                 zero_point=biases_tensor.zero_point)
        identity_node['frontend']['folded_biases_tensor'] = folded_biases_tensor
        identity_node['frontend']['folded_kernel_size'] = folded_weights.shape[3]

        if TFLITE_REQUANT:
            identity_node['frontend']['folded_padding'] = identity_node['attributes']['pads']
        else:    
            identity_node['frontend']['folded_padding'] = 0

        if TFLITE_REQUANT:
            # Take params list of length [output channels] and replicate it based on folding
            # Currently using the same key name, can later prefix with 'folding_' like above if needed
            identity_node['frontend']['output_multiplier'] = folding_algo.get_asym_folded_per_oc_params(identity_node['frontend']['output_multiplier'],
                                                                            input_folding_factor_x=identity_node_input_folding_x,
                                                                            input_folding_factor_y=input_folding_factor_y,
                                                                            stride_x=1,
                                                                            stride_y=1)
            identity_node['frontend']['cInputH'] = folding_algo.get_asym_folded_per_oc_params(identity_node['frontend']['cInputH'],
                                                                            input_folding_factor_x=identity_node_input_folding_x,
                                                                            input_folding_factor_y=input_folding_factor_y,
                                                                            stride_x=1,
                                                                            stride_y=1)
            identity_node['frontend']['cInputL'] = folding_algo.get_asym_folded_per_oc_params(identity_node['frontend']['cInputL'],
                                                                            input_folding_factor_x=identity_node_input_folding_x,
                                                                            input_folding_factor_y=input_folding_factor_y,
                                                                            stride_x=1,
                                                                            stride_y=1)
            identity_node['frontend']['o_shift'] = folding_algo.get_asym_folded_per_oc_params(identity_node['frontend']['o_shift'],
                                                                            input_folding_factor_x=identity_node_input_folding_x,
                                                                            input_folding_factor_y=input_folding_factor_y,
                                                                            stride_x=1,
                                                                            stride_y=1)
    else:
        identity_node['frontend']['requant_scale_shift'] = [(MAX_REDUCE_BUS_WIDTH - 3) for i in range(identity_node_input_channels)] # This will cause zero shift in the mac stage
        requant_scale_uint14 = 2 ** (FRACTIONAL_BITS+1+4) # This will cause zero shift in the RQ stage
        identity_node['frontend']['requant_scale_uint14'] = [requant_scale_uint14 for i in range(identity_node_input_channels)]
        mac_rough_shift_mux = 0
        identity_node['frontend']['mac_rough_shift_mux'] = [mac_rough_shift_mux for i in range(identity_node_input_channels)]
        identity_node['frontend']['requant_bias_int12'] = [0 for i in range(identity_node_input_channels)]
        identity_node['frontend']['requant_scale_float'] = 0.25
        if TFLITE_REQUANT:
            identity_node['frontend']['padding'] = identity_node['attributes']['pads']
        else:
            identity_node['frontend']['padding'] = 0

    
    # These remain True so the weights can be filled later
    #identity_node['reorder_node'] = False
    #identity_node['y_unfolding_reorder_node'] = False
    ir.graph.add_nodes_from([(identity_node_name, identity_node)])
    # Update producer of the original output tensor to be the identity node
    original_output_tensor.producer = identity_node_name
    original_node_successors = list(ir.graph.successors(node['name']))
    for original_node_successor_name in original_node_successors:
        ir.graph.remove_edge(node['name'],original_node_successor_name)
        ir.graph.add_edge(identity_node_name,original_node_successor_name)
    ir.graph.add_edge(node_name,identity_node_name)
    node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(node) # We need to update it so that following nodes order is according to execution order
    following_nodes_params = ir.get_updated_following_nodes(identity_node)
    identity_node['frontend']['following_nodes_params'] = following_nodes_params # We need to update it so that following nodes order is according to execution order
    for following_node_params in following_nodes_params:
        following_node_name = following_node_params[0]
        following_node = ir.graph.nodes()[following_node_name]
        following_node['frontend']['preceding_nodes_params'] = ir.get_updated_preceding_nodes(following_node)

    identity_node['frontend']['preceding_nodes_params'] = ir.get_updated_preceding_nodes(identity_node)

    # Check if tiling was done already
    if 'tiling_blob_idx' in node['frontend']:
        identity_node_blob_idx = node['frontend']['tiling_blob_idx']
        identity_node['frontend']['tiling_blob_idx'] = identity_node_blob_idx

        # Update blob with identity node
        current_blob = ir.tiling_blobs[identity_node_blob_idx]
        nodes_in_blob = current_blob.nodes_in_blob
        node_idx = nodes_in_blob.index(node_name)
        nodes_in_blob.insert(node_idx+1,identity_node_name)
        current_blob.num_of_nodes_in_blob+=1
        update_node_tiling_info(ir,identity_node)
    return identity_node_name

# Insert x folding convolutions if needed
def add_x_folding(ir):
    original_graph_node_names = list(ir.graph.nodes).copy()
    for node_name in original_graph_node_names:
        node = ir.graph.nodes[node_name]
        # Check if the output tensor is either an output to the graph or sent to MXP
        input_folding_factor_x = node['frontend']['input_folding_factor_x']
        if (len(node['frontend']['preceding_nodes_params']) == 0) and (input_folding_factor_x > 0):
            if ('_split' in node_name):
                orig_node_name = node_name.split('_split')[0]
                orig_node = ir.graph.nodes[orig_node_name]
                in_tensor = ir.tensors[orig_node['inputs'][0]]
                in_tensor.consumers.append(node_name)
                node['frontend']['input_tensor'] = in_tensor
                orig_node['frontend']['input_tensor'] = in_tensor
                node['frontend']['preceding_nodes_params'] = orig_node['frontend']['preceding_nodes_params']
                orig_in_tensor = ir.tensors[node['inputs'][0]]
                orig_in_tensor.consumers.remove(node_name)
                node['inputs'][0] = orig_node['inputs'][0]
            else:
                prev_node = node
                for fold_idx in range(input_folding_factor_x, 0, -1):
                    identity_node_name = insert_identity_conv_before_node(ir,prev_node,new_name=f'{node_name}_fold_x_{fold_idx-1}')
                    identity_node = ir.graph.nodes[identity_node_name]
                    identity_node['frontend']['force_folding_x'] = True
                    prev_node = identity_node
    return ir

# Add x unfolding parameter to graph nodes and insert x unfolding convolutions if needed
def add_x_unfolding(ir):
    original_graph_node_names = list(ir.graph.nodes).copy()
    for node_name in original_graph_node_names:
        node = ir.graph.nodes[node_name]

        # Check if the output tensor is either an output to the graph or sent to MXP
        output_tensor_name = node['frontend']['output_tensor'].name
        is_output = output_tensor_name in ir.outputs
        to_mxp = output_tensor_name in ir.tensors_to_mxp

        output_folding_factor_x = node['frontend']['output_folding_factor_x']
        if (is_output or to_mxp) and output_folding_factor_x > 0:
            prev_node = node
            for unfold_idx in range(output_folding_factor_x):
                identity_node_name = insert_identity_conv_after_node(ir, prev_node, new_name=f'{node_name}_unfold_x_{unfold_idx}')
                identity_node = ir.graph.nodes[identity_node_name]
                identity_node['frontend']['force_unfolding_x'] = True
                prev_node = identity_node
    ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    return ir

def allocate_input_tensors(ir: internal_representation.IR) -> internal_representation.IR:
    input_tensor_names = ir.inputs
    inputs_ddr = ir.inputs_ddr
    for current_input_tensor_name in input_tensor_names:
        current_tensor = ir.tensors[current_input_tensor_name]
        tensor_shape = current_tensor.shape_real_x16     
        current_tensor.data = np.zeros(tensor_shape,dtype=np.uint8)
        tensor_bytearray = create_tsnp_tensor_byte_array(current_tensor.data)
        ddr_entry_description = 'Input tensor: %s' % (current_input_tensor_name)
        current_input_tensor_ddr_entry = TensorDDREntry(tensor_bytearray, type = DDREntryType.INPUT_TENSOR, description = ddr_entry_description,shape = tensor_shape)
        current_tensor.ddr_entry = current_input_tensor_ddr_entry
        inputs_ddr.add_entry(current_input_tensor_ddr_entry) 
    return ir
def allocate_output_tensors(ir: internal_representation.IR) -> internal_representation.IR:
    output_tensor_names = ir.outputs
    outputs_ddr = ir.outputs_ddr
    for current_output_tensor_name in output_tensor_names:
        current_tensor = ir.tensors[current_output_tensor_name]
        tensor_shape = current_tensor.shape_real_x16
        current_tensor.data = np.zeros(tensor_shape,dtype=np.uint8)
        tensor_bytearray = create_tsnp_tensor_byte_array(current_tensor.data)
        ddr_entry_description = 'Output tensor: %s' % (current_output_tensor_name)
        current_output_tensor_ddr_entry = TensorDDREntry(tensor_bytearray, type = DDREntryType.OUTPUT_TENSOR, description = ddr_entry_description,shape = tensor_shape)
        current_tensor.ddr_entry = current_output_tensor_ddr_entry
        outputs_ddr.add_entry(current_output_tensor_ddr_entry) 
    return ir

def allocate_ddr_for_offloaded_tensors(ir: internal_representation.IR,offloaded_tensors_names) -> bool:
    program_and_intermediate_tensors_ddr = ir.ddr
    for current_intermediate_tensor_name in offloaded_tensors_names:
        if current_intermediate_tensor_name not in ir.tensors:
            # The code below seems out of date. For example, current_intermediate_tensor_name is
            # 'onnx::MaxPool_275_blob13_tile0', but ir.tensors only contains tensors such as
            # 'onnx::MaxPool_275'. It does not contain any tensors with 'blob' or 'tile' in the name.
            # The offloading code therefore might need to be updated in order to work.
            # For now, return that this is a failure, otherwise the code will crash in the next line
            # with a KeyError.
            return False
        current_tensor = ir.tensors[current_intermediate_tensor_name]
        tensor_shape = current_tensor.real_shape_x16
        if current_tensor.y_tiles!=1:
            raise ValueError ('Offloading of tiled tensor not supported yet') # Need to see if we offload the complete tensor or just a single tile
        current_tensor.data = np.zeros(tensor_shape,dtype=np.uint8)
        tensor_bytearray = create_tsnp_tensor_byte_array(current_tensor.data)
        ddr_entry_description = 'Intermediate tensor: %s' % (current_intermediate_tensor_name)
        current_intermediate_tensor_ddr_entry = TensorDDREntry(tensor_bytearray, type = DDREntryType.INTERMEDIATE_TENSOR, description = ddr_entry_description,shape = tensor_shape)
        current_tensor.ddr_entry = current_intermediate_tensor_ddr_entry
        program_and_intermediate_tensors_ddr.add_entry(current_intermediate_tensor_ddr_entry) 
    return True

def fold_next_op_ic_lookup_dicts(next_op_ic_lookup_dicts,input_channels_after_folding): # In case of a y folding op we need to update
    if input_channels_after_folding % 2 !=0:
        raise ValueError ('This is not logical (odd number of folded channels). Please check...')
    input_channels_before_folding = input_channels_after_folding // 2
    folded_next_op_ic_lookup_dicts = [OrderedDict() for i in range(len(next_op_ic_lookup_dicts))]
    for ic_group_index,current_ic_group_dict in enumerate(next_op_ic_lookup_dicts):
        for current_pair in current_ic_group_dict.items():
            folded_next_op_ic_lookup_dicts[ic_group_index][current_pair[0]] = current_pair[1]
        for current_pair in current_ic_group_dict.items():
            folded_next_op_ic_lookup_dicts[ic_group_index][current_pair[0]+input_channels_before_folding] = current_pair[1]+input_channels_before_folding
    return folded_next_op_ic_lookup_dicts

def unfold_next_op_ic_lookup_dicts(next_op_ic_lookup_dicts,input_channels_after_folding):
    if input_channels_after_folding % 2 !=0:
        raise ValueError ('This is not logical (odd number of folded channels). Please check...')
    input_channels_before_folding = input_channels_after_folding // 2
    folded_next_op_ic_lookup_dicts = [OrderedDict() for i in range(len(next_op_ic_lookup_dicts))]
    for ic_group_index,current_ic_group_dict in enumerate(next_op_ic_lookup_dicts):
        for current_pair in current_ic_group_dict.items():
            folded_next_op_ic_lookup_dicts[ic_group_index][current_pair[0]] = current_pair[1]
        for current_pair in current_ic_group_dict.items():
            folded_next_op_ic_lookup_dicts[ic_group_index][current_pair[0]+input_channels_before_folding] = current_pair[1]+input_channels_before_folding
    return folded_next_op_ic_lookup_dicts

def calc_concat_oc_processing_order(ir,node):
    following_nodes_params = node['frontend']['following_nodes_params']
    input_dicts = node['backend']['ic_lookup_dicts']
    simulator_oc_processing_order = []
    if len(input_dicts[0])>1:
        raise ValueError ('Concat of ic split inputs not supported yet. need to check here if we did right creation of oc processing order')
    input0_channels = node['frontend']['input_tensors'][0].get_folded_shape()[1]
    current_input_start_index = 0
    # Calculate updated processing order (oc_processing_order) in order to fix concat of folded tensors
    new_dicts=[]
    concat_output_yfold_channels = int(node['frontend']['output_tensor'].get_original_shape()[1]*math.pow(2,node['frontend']['output_tensor'].folding_factor_x))
    for input_index,per_ic_group_dict in enumerate(input_dicts):
        new_per_ic_group_dicts=[]
        for input_dict in per_ic_group_dict:
            current_input_new_dict={}
            input_tensor = node['frontend']['input_tensors'][input_index]
            per_yfold_channels = int(input_tensor.get_original_shape()[1]*math.pow(2,input_tensor.folding_factor_x))
            input_folding_factor_y = input_tensor.folding_factor_y
            for pre_concat_actual_channel in input_dict.keys():
                if pre_concat_actual_channel<per_yfold_channels:
                    current_input_new_dict[pre_concat_actual_channel+input_index*per_yfold_channels] = input_dict[pre_concat_actual_channel]
                else:
                    current_input_new_dict[pre_concat_actual_channel-per_yfold_channels+concat_output_yfold_channels+input_index*per_yfold_channels] = input_dict[pre_concat_actual_channel]
            new_per_ic_group_dicts.append(current_input_new_dict)
        new_dicts.append(new_per_ic_group_dicts)
    all_channels=set()
    for per_input_dicts in new_dicts:
        for per_ic_group_dict in per_input_dicts:
            all_channels = all_channels.union(set(list(per_ic_group_dict.keys())))
    
    #if len(all_channels)!= node['frontend']['output_tensor'].get_folded_shape()[1]:
    #    raise ValueError ('Something wrong with folded dicts update')

    y_folds = int(math.pow(2,node['frontend']['output_tensor'].folding_factor_y)) # We expect all inputs y folding factor including  input folding/unfoldingto be equal to output's y folding factor
    updated_oc_processing_order_dict={}
    current_input_offset_in_y_fold=0
    current_input_processing_order_offset=0
    concat_output_channels_per_y_fold = int(node['frontend']['output_tensor'].get_original_shape()[1] * math.pow(2,node['frontend']['output_tensor'].folding_factor_x))
    for input_index,current_input_ic_groups_dicts in enumerate(input_dicts):
        current_input_original_channels = node['frontend']['input_tensors'][input_index].get_original_shape()[1]
        current_input_y_fold_channels = int(current_input_original_channels*math.pow(2,node['frontend']['output_tensor'].folding_factor_x))
        current_input_all_ic_groups_dict={}
        for input_group_dict in current_input_ic_groups_dicts:
            for key in input_group_dict:
                if key in current_input_all_ic_groups_dict:
                    raise ValueError ('Two ic groups contain same channel, this is illegal')
                else:
                    current_input_all_ic_groups_dict[key] = input_group_dict[key]
        sorted_current_input_all_ic_groups_dict = OrderedDict(sorted(current_input_all_ic_groups_dict.items(),key=lambda x:x[1])) # We need to make sure the dict is sorted by dict value which is actual input channel
        for current_input_output_channel in list(sorted_current_input_all_ic_groups_dict.keys()):
            current_channel_y_fold = int(current_input_output_channel /current_input_y_fold_channels)
            current_input_original_channel = current_input_output_channel % current_input_y_fold_channels
            output_y_fold_offset = current_channel_y_fold * concat_output_channels_per_y_fold
            current_input_processing_order = sorted_current_input_all_ic_groups_dict[current_input_output_channel]
            updated_current_input_processing_order = current_input_processing_order + current_input_processing_order_offset
            updated_oc_processing_order_dict[current_input_original_channel+output_y_fold_offset+current_input_offset_in_y_fold] = updated_current_input_processing_order
        current_input_offset_in_y_fold+=current_input_y_fold_channels
        current_input_processing_order_offset+=len(sorted_current_input_all_ic_groups_dict)
    sorted_updated_oc_processing_order_dict = OrderedDict(sorted(updated_oc_processing_order_dict.items(),key=lambda x:x[1]))

    #Alex comment it out, because it is not used
    # Calculate actual processing order (simulator_oc_processing_order), this will be used only by simulator to generate the concat output nxo file
    # for input_index,input_dict in enumerate(input_dicts):
    #     all_ic_groups_dict={}
    #     for input_group_dict in input_dict:
    #         for key in input_group_dict:
    #             if key in all_ic_groups_dict:
    #                 raise ValueError ('Two ic groups contain same channel, this is illegal')
    #             else:
    #                 all_ic_groups_dict[key] = input_group_dict[key]
    #     sorted_input_group_dict = OrderedDict(sorted(all_ic_groups_dict.items(),key=lambda x:x[1])) # We need to make sure the dict is sorted by dict value which is actual input channel
    #     list_of_channels = [(channel+current_input_start_index) for channel in list(sorted_input_group_dict.keys())]
    #     simulator_oc_processing_order+=list_of_channels
    #     current_input_max_channels = node['frontend']['input_tensors'][input_index].get_folded_shape()[1] # Max since if oc is all zeroes it will not be in calculated at all
    #     if 'force_folding_y' in node['frontend']:
    #         current_input_max_channels=current_input_max_channels*2
    #     if current_input_max_channels!= len(list_of_channels):
    #         # Implementing the below means:
    #         # 1) current_input_start_index should be added len(current_input_oc_processing_order) and not current_input_max_channels
    #         # 2) In the below code of fixing expected input channel (convert_expected_input_channel) will need to change current_input_num_of_input_channels to actual number of non empty channels
    #         raise ValueError ('Need to add support to concat of tensors which include empty(all zeros) output channels')
    #     current_input_start_index+=current_input_max_channels

    node['backend']['simulator_oc_order'] = simulator_oc_processing_order
    #Alex: also del this
    # if DEBUG_FIX_CONCAT_OUTPUT_PROCESSING_ORDER:
    #     old_fix_oc_processing_order=[]
    #     for new_ic_groups_dicts in new_dicts:
    #         for ic_group_dict in new_ic_groups_dicts:
    #             old_fix_oc_processing_order.extend(list(ic_group_dict.keys()))
    #     if len(old_fix_oc_processing_order)!= node['frontend']['output_tensor'].get_folded_shape()[1]:
    #         raise ValueError ('Something wrong with folded dicts update')
    #     oc_processing_order = list(sorted_updated_oc_processing_order_dict.keys())
    #     if len(oc_processing_order)!= node['frontend']['output_tensor'].get_folded_shape()[1]:
    #         raise ValueError ('Something wrong with folded dicts update')
    # else:
    oc_processing_order = simulator_oc_processing_order

    node['backend']['oc_order'] = oc_processing_order
    # When 2 inputs of concat are to be y folded we get wrong order of channels. input0even,input0odd,input1even,input1odd but it needs to be input0even,input1even,input0even,input1odd
    # We fix this by having a conversion dict that will be used in the following ops of that concat that will re-direct to right input channels
    inputs_folding_factor_y = node['frontend']['input_tensors'][0].folding_factor_y
    if not DEBUG_FIX_CONCAT_OUTPUT_PROCESSING_ORDER:
        if 'force_folding_y' in node['frontend']: 
            folded_concat_input_channels = node['backend']['input_channels']
            original_concat_input_channels = []
            for per_input_folded_input_channels in folded_concat_input_channels:
                original_concat_input_channels.append(per_input_folded_input_channels // 2)
            conversion_dict={}
            for i in range(2): # When i=0 we look on even y channels, when its 1 we look on odd y channels
                for input_index,current_input_num_of_input_channels in enumerate(original_concat_input_channels):
                    current_input_start_index = sum(original_concat_input_channels[0:input_index])
                    total_concat_channels = sum(original_concat_input_channels)
                    for channel in range(current_input_num_of_input_channels):
                        expected_input_channel = current_input_start_index+channel+i*total_concat_channels
                        actual_input_channel = current_input_start_index*2+original_concat_input_channels[input_index]*i+channel
                        conversion_dict[actual_input_channel] = expected_input_channel
            for node_params in following_nodes_params: # Note that when we created the node['backend']['following_op_ic_groups'] we already checked that all following ops have the same ic groups
                following_node = ir.graph.nodes[node_params[0]]
                following_node['backend']['input_channels_reorder_dict'] = conversion_dict

def create_ic_dicts(ir,node): # This creates the ops ic dicts based on input producer's oc_processing_order
    current_op_ic_splits = node['backend']['ic_splits']
    if node['op_type'] in MULTIPLE_INPUT_OPS:
        
        input_tensors = node['frontend']['input_tensors']
        
    else:
        input_tensors = [node['frontend']['input_tensor']]
    force_folding_y = 'force_folding_y' in node['frontend']        
    force_unfolding_y = 'force_unfolding_y' in node['frontend']

    for input_idx,input_tensor in enumerate(input_tensors):
        pre_y_folding_input_channels = input_tensor.get_folded_shape()[1]
        post_y_unfolding_input_channels = input_tensor.get_folded_shape(folding_conv_y=force_folding_y,unfolding_conv_y=force_unfolding_y)[1]

        current_input_ic_dict = node['backend']['ic_lookup_dicts'][input_idx]
        input_producing_node_name = input_tensor.producer
        if input_producing_node_name == None:
            # If there is no producer (start of the graph), keep its ic_lookup_dicts the same.
            # If any of the dicts are a list, convert them to a dict.
            # They are created as lists in e.g., create_ordering_conv, and might still be a
            # list if the node was created after the call to get_ops_channel_balancing.
            for index, ic_lookup_dict in enumerate(node['backend']['ic_lookup_dicts']):
                if isinstance(ic_lookup_dict, list):
                    node['backend']['ic_lookup_dicts'][index] = {x: x for x in ic_lookup_dict}
            continue
        input_producing_node = ir.graph.nodes()[input_producing_node_name]
        
        # Alex for ph2 changed this line to this
        #current_input_real_oc_processing_order = input_producing_node['backend']['oc_order']
        current_input_real_oc_processing_order = current_input_ic_dict
        if node['op_type'] == 'Add': # in MULTIPLE_INPUT_OPS:
            current_input_real_oc_processing_order = current_input_ic_dict[0][0] 
        elif node['op_type'] == 'Concat': # in MULTIPLE_INPUT_OPS:
            current_input_real_oc_processing_order = current_input_ic_dict[0] 

        # if ('folding_conv' in input_producing_node_name) and ('folding_conv' not in node['name']):
        #     x_folding_factor = pow(2, node['frontend']['input_folding_factor_x'])
        #     y_folding_factor = pow(2, node['frontend']['input_folding_factor_y'])
        #     mapping_ic_groups = [0] * pre_y_folding_input_channels
        #     for curr_i in range(int(len(current_input_real_oc_processing_order)/(y_folding_factor*x_folding_factor))):
        #         for curr_y in range(y_folding_factor):
        #             ic_offset = curr_y*int(pre_y_folding_input_channels/y_folding_factor) + curr_i * x_folding_factor
        #             for curr_x in range(x_folding_factor):
        #                 mapping_ic_groups[ic_offset+curr_x] = y_folding_factor*x_folding_factor*curr_i + curr_y*x_folding_factor + curr_x

        if node['op_type'] in MULTIPLE_INPUT_OPS:
            current_input_ic_groups = node['backend']['ic_groups'][input_idx]
        else:
            current_input_ic_groups = node['backend']['ic_groups']
        per_ic_group_counter = [0 for i in range(current_op_ic_splits)]
        current_input_ic_lookup_dicts = [{} for i in range(current_op_ic_splits)]
        # Alex commit it out, because we do not need the ordering of ic_chanal any more
        # for oc_idx,current_oc in enumerate(current_input_real_oc_processing_order): #<- alex: this is dict
        #     next_op_ic_group_found = False
        #     actual_ic_in_current_input=current_oc
        #     if force_unfolding_y:
        #         if current_oc>=post_y_unfolding_input_channels:
        #             if current_oc-post_y_unfolding_input_channels!=current_input_real_oc_processing_order[oc_idx-post_y_unfolding_input_channels]:
        #                 raise ValueError ('In an y unfolding op the oc processing order channel[x] must be equal to channel[x+pre_y_folding_input_channels]')
        #             continue
        #     for ic_group_idx,ic_group in enumerate(current_input_ic_groups):
        #         if actual_ic_in_current_input in ic_group:
        #             #if ('folding_conv' in input_producing_node_name) and ('folding_conv' not in node['name']):
        #             #    current_input_ic_lookup_dicts[ic_group_idx][actual_ic_in_current_input] = mapping_ic_groups[actual_ic_in_current_input]
        #             #else:
        #             current_input_ic_lookup_dicts[ic_group_idx][actual_ic_in_current_input] = per_ic_group_counter[ic_group_idx]
        #             if force_folding_y:
        #                 if (actual_ic_in_current_input+pre_y_folding_input_channels)  in ic_group:
        #                     current_input_ic_lookup_dicts[ic_group_idx][actual_ic_in_current_input+pre_y_folding_input_channels] = per_ic_group_counter[ic_group_idx]+pre_y_folding_input_channels
        #                 else:
        #                     raise ValueError('Creating ic_dict for op: %s, cant find folded channel in ic group' % node['name'])
        #             per_ic_group_counter[ic_group_idx]+=1
        #             next_op_ic_group_found = True
        #             break
        #     if not next_op_ic_group_found:
        #         raise ValueError ('at update_next_op_ic_groups: oc %d not found in next ops ic groups!' % current_oc)
        if node['op_type'] not in MULTIPLE_INPUT_OPS:
            node['backend']['ic_lookup_dicts'] = current_input_ic_lookup_dicts
        else:
            node['backend']['ic_lookup_dicts'][input_idx] = current_input_ic_lookup_dicts

        
    '''for node_params in following_nodes_params: # Note that when we created the node['backend']['following_op_ic_groups'] we already checked that all following ops have the same ic groups
        following_node = ir.graph.nodes[node_params[0]]
        if convert_expected_input_channel:
            following_node['backend']['input_channels_reorder_dict'] = conversion_dict
        following_node_input_index = node_params[1]
        if 'force_unfolding_y' in following_node['frontend']:
            raise ValueError ('This is not supported yet')
        if 'force_folding_y' in following_node['frontend']:
            if following_node['op_type'] in MULTIPLE_INPUT_OPS:
                input_channels_after_folding = following_node['backend']['input_channels'][following_node_input_index]
            else:
                input_channels_after_folding = following_node['backend']['input_channels']

            folded_next_op_ic_lookup_dicts = fold_next_op_ic_lookup_dicts(next_op_ic_lookup_dicts,input_channels_after_folding)
        else:
            folded_next_op_ic_lookup_dicts = next_op_ic_lookup_dicts
        if following_node['op_type'] in MULTIPLE_INPUT_OPS:
            following_node['backend']['ic_lookup_dicts'][following_node_input_index] = folded_next_op_ic_lookup_dicts
        else:
            following_node['backend']['ic_lookup_dicts'] = folded_next_op_ic_lookup_dicts'''

def update_next_op_ic_groups(ir,node):
    next_op_ic_splits = node['backend']['following_op_ic_split']
    following_nodes_params = node['frontend']['following_nodes_params']
    input_dicts = node['backend']['ic_lookup_dicts']
    if node['op_type'] == 'Concat': # In case of concat we need to calculate its oc processing order
        oc_processing_order = []
        if len(input_dicts[0])>1:
            raise ValueError ('Concat of ic split inputs not supported yet. need to check here if we did right creation of oc processing order')
        input0_channels = node['frontend']['input_tensors'][0].get_folded_shape()[1]
        current_input_start_index = 0
        
        for input_index,input_dict in enumerate(input_dicts):
            current_input_oc_processing_order = []
            for input_group_dict in input_dict:
                sorted_input_group_dict = OrderedDict(sorted(input_group_dict.items(),key=lambda x:x[1])) # We need to make sure the dict is sorted by dict value which is actual input channel
                list_of_channels = [(channel+current_input_start_index) for channel in list(sorted_input_group_dict.keys())]
                current_input_oc_processing_order+=list_of_channels
            oc_processing_order+=current_input_oc_processing_order
            current_input_max_channels = node['frontend']['input_tensors'][input_index].get_folded_shape()[1] # Max since if oc is all zeroes it will not be in calculated at all
            if 'force_folding_y' in node['frontend']:
                current_input_max_channels=current_input_max_channels*2
            if current_input_max_channels!= len(current_input_oc_processing_order):
                # Implementing the below means:
                # 1) current_input_start_index should be added len(current_input_oc_processing_order) and not current_input_max_channels
                # 2) In the below code of fixing expected input channel (convert_expected_input_channel) will need to change current_input_num_of_input_channels to actual number of non empty channels
                raise ValueError ('Need to add support to concat of tensors which include empty(all zeros) output channels')
            current_input_start_index+=current_input_max_channels
        node['backend']['oc_order'] = oc_processing_order
    else: 
        oc_processing_order = node['backend']['oc_order']
    next_op_ic_lookup_dicts = [{} for i in range(next_op_ic_splits)]
    per_ic_group_counter = [0 for i in range(next_op_ic_splits)]
    convert_expected_input_channel = False
    # When 2 inputs of concat are to be y folded we get wrong order of channels. input0even,input0odd,input1even,input1odd but it needs to be input0even,input1even,input0even,input1odd
    # We fix this by having a conversion dict that will be used in the following ops of that concat that will re-direct to right input channels
    if 'force_folding_y' in node['frontend']: 
        if node['op_type'] == 'Concat':
            original_concat_input_channels = node['backend']['input_channels']
            conversion_dict={}
            for i in range(2): # When i=0 we look on even y channels, when its 1 we look on odd y channels
                for input_index,current_input_num_of_input_channels in enumerate(original_concat_input_channels):
                    current_input_start_index = sum(original_concat_input_channels[0:input_index])
                    total_concat_channels = sum(original_concat_input_channels)
                    for channel in range(current_input_num_of_input_channels):
                        expected_input_channel = current_input_start_index+channel+i*total_concat_channels
                        actual_input_channel = current_input_start_index*2+original_concat_input_channels[input_index]*i+channel
                        conversion_dict[actual_input_channel] = expected_input_channel
            convert_expected_input_channel = True

    for node_params in following_nodes_params: # Note that when we created the node['backend']['following_op_ic_groups'] we already checked that all following ops have the same ic groups
        following_node = ir.graph.nodes[node_params[0]]
        if convert_expected_input_channel:
            following_node['backend']['input_channels_reorder_dict'] = conversion_dict
    
    # 

    '''for current_oc in oc_processing_order:
        next_op_ic_group_found = False
        for ic_group_idx,ic_group in enumerate(next_op_ic_groups):
            if current_oc in ic_group:
                next_op_ic_lookup_dicts[ic_group_idx][current_oc] = per_ic_group_counter[ic_group_idx]
                per_ic_group_counter[ic_group_idx]+=1
                next_op_ic_group_found = True
                break
        if not next_op_ic_group_found:
            raise ValueError ('at update_next_op_ic_groups: oc %d not found in next ops ic groups!' % current_oc)
        
    for node_params in following_nodes_params: # Note that when we created the node['backend']['following_op_ic_groups'] we already checked that all following ops have the same ic groups
        following_node = ir.graph.nodes[node_params[0]]
        if convert_expected_input_channel:
            following_node['backend']['input_channels_reorder_dict'] = conversion_dict
        following_node_input_index = node_params[1]
        if 'force_unfolding_y' in following_node['frontend']:
            raise ValueError ('This is not supported yet')
        if 'force_folding_y' in following_node['frontend']:
            if following_node['op_type'] in MULTIPLE_INPUT_OPS:
                input_channels_after_folding = following_node['backend']['input_channels'][following_node_input_index]
            else:
                input_channels_after_folding = following_node['backend']['input_channels']

            folded_next_op_ic_lookup_dicts = fold_next_op_ic_lookup_dicts(next_op_ic_lookup_dicts,input_channels_after_folding)
        else:
            folded_next_op_ic_lookup_dicts = next_op_ic_lookup_dicts
        if following_node['op_type'] in MULTIPLE_INPUT_OPS:
            following_node['backend']['ic_lookup_dicts'][following_node_input_index] = folded_next_op_ic_lookup_dicts
        else:
            following_node['backend']['ic_lookup_dicts'] = folded_next_op_ic_lookup_dicts'''

def update_reorder_node_wights_and_biases_tensors(node,current_op_weights_tensor,current_op_biases_tensor):
    ic_lookup_dict = node['backend']['ic_lookup_dicts']
    if len(ic_lookup_dict)!=1:
        raise ValueError('We currently dont support re-ordering conv for ic groups>1')
    output_channels = node['frontend']['output_tensor'].get_folded_shape()[1]

    weights_tensor = np.zeros((output_channels,output_channels,1,1)).astype(np.int8)
    for output_channel in range(output_channels): # This reorders output channels 0,1,2,3.... The actual reordering is done in CBC creation since we look on the ic_lookup_dict
        weights_tensor[output_channel,output_channel,0,0] = 8 # Dans The reason we put 8 instead of 1 is to push bits left so we can reduce bus width between MAC and RQ
    biases_tensor = np.zeros((output_channels)).astype(np.int32)

    current_op_weights_tensor.data = weights_tensor
    current_op_weights_tensor.shape = weights_tensor.shape
    current_op_biases_tensor.data = biases_tensor
    current_op_biases_tensor.shape = biases_tensor.shape

def following_node_needs_dual_allocation(ir: internal_representation.IR,node,origin_blob_idx): # for example concat node need to allocate both of its input together so they are contiguous in mem
    following_nodes_params = node['frontend']['following_nodes_params']
    current_blob = ir.tiling_blobs[origin_blob_idx]
    for following_node_params in following_nodes_params:
        following_node = ir.graph.nodes[following_node_params[0]]
        if following_node['op_type'] in MULTIPLE_INPUT_OPS and following_node['name'] in current_blob.nodes_in_blob: # We need contigeous allocation only if concat node is in same blob
            dual_node_input_nodes_params = following_node['frontend']['preceding_nodes_params']
            if following_node['name'] in ir.offloaded_concat_ops: # If this is the far node from long concat
                if following_node['frontend']['preceding_nodes_params'][0][0]==node['name']:
                    return True,True,True,dual_node_input_nodes_params # its concat,its_long_concat,its_far_node,[]
                else:
                    return True,True,False,dual_node_input_nodes_params # its concat,its_long_concat,its_not_far_node,[]
            return True,False,False,dual_node_input_nodes_params # Its concat,not_long,
        #if following_node['op_type']  == 'Add' and following_node_params[1]==0: # If its input 0 of add node (which is inline) check if it has a following concat
        #    concat_following_add, long_concat, far_node,concat_inputs = following_node_needs_dual_allocation(ir,following_node,origin_blob_idx)
        #    if concat_following_add:
        #        return True,long_concat, far_node, concat_inputs
    return False,False,False,[] # Return values areL is_concat,is long_concat,is_far_node_of_long_concat,concat_inputs

def mark_input_tensors_deallocation(ir: internal_representation.IR,amms,node,current_tile_num): # This will deallocate after read from ddr allocation, at next excuted node after last consumer
    current_op_type = node['op_type']
    if current_op_type == 'Add':
        input_tensors = node['frontend']['input_tensors']
    else:
        input_tensors = [node['frontend']['input_tensor']]

    current_node_blob_idx = node['frontend']['tiling_blob_idx']
    for input_index,tensor in enumerate(input_tensors):
        tensor_name = tensor.name
        num_xslices = node['frontend']['x_slices']
        num_xslices = num_xslices // tensor.num_packed_xslices
        for current_x_slice in range(num_xslices):
            amm_tensor_name = tensor.get_amm_tensor_name(current_node_blob_idx, current_tile_num, current_x_slice)
            deallocating_node = amms.tensors_in_amm[amm_tensor_name].deallocating_node
            ir_tensor = amms.tensors_in_amm[amm_tensor_name].tensor # We want to get to graph tensor so we know if it feeds more nodes. we walnt to deallocate it in last executing node
            current_blob_idx = node['frontend']['tiling_blob_idx']
            current_blob = ir.tiling_blobs[current_blob_idx]
            if tensor_name in current_blob.get_blob_outputs_names(): # If its a blobs output it will be deallocated elsewhere (after it is writen to DDR)
                continue
            last_consumer_node_in_blob_found,last_consumer_node_in_blob_name = ir.get_last_tensor_consumer_in_blob(current_blob,ir_tensor)

            if last_consumer_node_in_blob_name!=node['name']: # If we are not in last consumer node we dont set its deallocation. It will be set in actual deallocating node
                continue
            last_consumer_node = ir.graph.nodes()[last_consumer_node_in_blob_name] # We assume that consumers list order is by execution
            inline_tensor=False
            if node['op_type'] in INLINE_OPS:
                if input_index in INLINE_OPS[node['op_type']]:
                    inline_tensor = True
                    ir_tensor.is_inline_tensor = True
            
            success,next_executed_node, next_executed_tile, next_executed_xslice = ir.get_next_executed_node(last_consumer_node,error_on_last_node=True,current_tile_num=current_tile_num,current_xslice_num=current_x_slice)
            deallocating_node = next_executed_node
            dealocation_tile_idx = next_executed_tile
            deallocation_xslice_idx = next_executed_xslice
            if amms.tensors_in_amm[amm_tensor_name].deallocating_node == None:
                amms.tensors_in_amm[amm_tensor_name].deallocating_node = deallocating_node
            else:
                raise ValueError ('Tensor %s already set for deallocation at. Illegal to set 2 deallocation points(new point is: %s)' % (tensor_name,last_consumer_node['name']))
            dellocated_tensor = TensorDeAllocationInfo(amm_tensor_name,input_index,current_tile_num,current_x_slice, dealocation_tile_idx,deallocation_xslice_idx,inline_tensor=inline_tensor)
            if success:
                if 'tensors_for_deallocation_after_ddr_read_allocation' in deallocating_node['backend']:
                    deallocating_node['backend']['tensors_for_deallocation_after_ddr_read_allocation'].add_tensor(dellocated_tensor)
                else: 
                    tensor_list = TensorDeAllocationList()
                    tensor_list.add_tensor(dellocated_tensor)
                    deallocating_node['backend']['tensors_for_deallocation_after_ddr_read_allocation'] = tensor_list
            else: # This means we are at last node in workload and we should allocate at current node after output tensor allocation
                if 'tensors_for_deallocation_after_output_allocation' in deallocating_node['backend']:
                    deallocating_node['backend']['tensors_for_deallocation_after_output_allocation'].add_tensor(dellocated_tensor)
                else: 
                    tensor_list = TensorDeAllocationList()
                    tensor_list.add_tensor(dellocated_tensor)
                    deallocating_node['backend']['tensors_for_deallocation_after_output_allocation'] = tensor_list
    return True

def mark_blob_outputs_deallocations(ir,amms,node,current_tile_num):
    # This sets deallocation node for tensors of tiles that are written to DDR at end of blob
    # If write to DDR is in parallel to convs then write of output of tile N will be executed in tile N+1
    # If not, Write will be done after node execution so deallocation can occur in that node right after output allocation

    output_tensor = node['frontend']['output_tensor']
    output_tensor_name = output_tensor.name
    current_node_blob_idx = node['frontend']['tiling_blob_idx']
    num_xslices = node['frontend']['output_tensor'].x_slices
    current_node_blob_idx = node['frontend']['tiling_blob_idx']
    
    for current_x_slice in range(num_xslices):
        amm_tensor_name = output_tensor.get_amm_tensor_name(current_node_blob_idx,current_tile_num,current_x_slice)
        #if DEBUG_AVOID_DDR_WRITE_WHILE_CONV: # Althoug in this state the tensor can be deallocated earlier we keep it here so in case of inline tensors the output wont be deallocated before input 
        #    deallocating_node = node
        #    dealocation_tile_idx = current_tile_num
        #else: 
        deallocating_node, deallocation_tile_idx, deallocation_xslice_idx = ir.get_blob_output_dealocating_node(current_node_blob_idx,current_tile_num,current_x_slice,num_xslices)

        if amms.tensors_in_amm[amm_tensor_name].deallocating_node == None:
            amms.tensors_in_amm[amm_tensor_name].deallocating_node = deallocating_node
        else:
            raise ValueError ('Tensor %s already set for deallocation at. Illegal to set 2 deallocation points(new point is: %s)' % (amm_tensor_name,deallocating_node['name']))

        tensor = ir.tensors[output_tensor_name]
        if len(tensor.consumers)>0:
            last_consumer_node_name = tensor.consumers[-1]
            last_consumer_node = ir.graph.nodes()[last_consumer_node_name]
            inline_tensor=False # Output of last node will always go to another blob so even if its inline we need to dealocate it and it will then be read by new blob
            # If a node is split due to its size, the following nodes will still use the original tensor
            tensor_name = output_tensor_name
            if tensor_name in ir.split_tensor_to_original_tensor_map:
               tensor_name = tensor_name.split('_split')[0]
            input_index = last_consumer_node['inputs'].index(tensor_name)
        else:
            input_index=0
            inline_tensor=False
        dellocated_tensor = TensorDeAllocationInfo(amm_tensor_name,input_index,current_tile_num,current_x_slice,deallocation_tile_idx,deallocation_xslice_idx,inline_tensor=inline_tensor)

        # Note: Here it is
        #   tensors_for_deallocation_after_output_allocation
        # and not
        #   tensors_for_deallocation_after_ddr_read_allocation
        # I think this is because if Blob N last tile output is deallocated from AMM as soon as
        # Blob N+1 tile 0 input is read to AMM from DDR, it might be deallocating before the
        # Blob N last tile computation is done.
        # TODO: add comment explaining why deallocating Blob N last tile output after allocating
        # Blob N+1 tile 0 output is guaranteed to work.

        if 'tensors_for_deallocation_after_output_allocation' in deallocating_node['backend']:
            deallocating_node['backend']['tensors_for_deallocation_after_output_allocation'].add_tensor(dellocated_tensor)
        else: 
            tensor_list = TensorDeAllocationList()
            tensor_list.add_tensor(dellocated_tensor)
            deallocating_node['backend']['tensors_for_deallocation_after_output_allocation'] = tensor_list
    return True

def convert_28x28_add_input0_allocations_to_14x14(node):
    even_allocated_blocks = copy.deepcopy(node['backend']['allocated_amm_blocks_for_input_even_grid'][0]) # Allocated for even grids of input0 of add
    odd_allocated_blocks = copy.deepcopy(node['backend']['allocated_amm_blocks_for_input_odd_grid'][0]) # Allocated for odd grids of input0 of add
    merged_allocated_blocks = []
    for amm_idx,allocated_blocks in enumerate(even_allocated_blocks):
        current_amm_merged_blocks = even_allocated_blocks[amm_idx]+odd_allocated_blocks[amm_idx]
        merged_allocated_blocks.append(current_amm_merged_blocks)
    return merged_allocated_blocks

def set_following_nodes_inputs_allocations(ir,allocating_node,following_nodes_params,allocated_blocks,current_tile_num):

    for current_following_node in following_nodes_params:
        following_node = ir.graph.nodes[current_following_node[0]]
        if (following_node['frontend']['tiling_blob_idx'] == allocating_node['frontend']['tiling_blob_idx']): # If following node is from different blob, its input allocation will be done by "allocate_next_tile_input_mem" or "" in case its immediate read
            input_index = current_following_node[1]
            for current_x_slice in range(following_node['frontend']['x_slices']):
                following_node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][current_x_slice][input_index] = allocated_blocks[current_x_slice]
                following_node['backend']['allocated_amm_blocks_for_input_odd_grid'][current_tile_num][current_x_slice][input_index] = allocated_blocks[current_x_slice]
        #else:
        #    if following_node['op_type'] in DUAL_CONTIGUOUS_ALLOCATION_OPS:
        #        print('here')

def read_missing_tensors_from_ddr(ir: internal_representation.IR,node,current_tile_num):
    current_op_type = node['op_type']
    current_op_input_tensor_name = node['inputs'][0]
    force_folding_y = 'force_folding_y' in node['frontend']
    force_unfolding_y = 'force_unfolding_y' in node['frontend']
    current_op_input_tensor_shape = ir.tensors[current_op_input_tensor_name].get_folded_shape(folding_conv_y = force_folding_y,unfolding_conv_y=force_unfolding_y)
    current_op_input_channels = current_op_input_tensor_shape[1]
    if current_op_input_channels!=node['backend']['input_channels'] and current_op_type!='Add': # In add the 'backend' 'input_channels' property relates to the conv input size
        raise ValueError ('calced input channels != stored input channels. Please check...')
    current_blob_idx = node['frontend']['tiling_blob_idx']
    num_xslices = node['frontend']['x_slices']
    # If node needs to read inputs from DDR allocate AMM mem for them
    if current_op_type == 'Add':
        input_tensors = node['frontend']['input_tensors']
    else:
        input_tensors = [node['frontend']['input_tensor']]
    # TODO: Dans need to handle a state where this input is alsan input to concat in the same blob. In such case we need to allocate the complete concat input and
    # not just the input of the regular node

    num_xslices = num_xslices // input_tensors[0].num_packed_xslices
    for input_index,input_tensor in enumerate(input_tensors):
        for current_x_slice in range(num_xslices):
            allocate_and_mark_for_read_input_tensor(ir,input_tensor,node,current_blob_idx,current_tile_num,current_tile_num,current_x_slice,load_before_op = True)
    

def allocate_and_mark_for_read_input_tensor(ir: internal_representation.IR,input_tensor,allocation_node,consumer_node_blob_idx,current_tile_idx,read_tensor_tile_idx,current_xslice_num,load_before_op = False):
    amms = ir.amms
    if load_before_op:
        sequencer_tensor_load_marker_name = 'tensors_to_load_immediately_from_ddr'
    else:
        sequencer_tensor_load_marker_name = 'next_tile_tensors_to_load_from_ddr'
    consumer_node_blob = ir.tiling_blobs[consumer_node_blob_idx]

    found_consumer_in_blob,current_input_tensor_dominant_consumer_node_name = ir.get_dominant_tensor_consumer_in_blob(consumer_node_blob,input_tensor)
    if not found_consumer_in_blob:
        raise ValueError ('Blobs input tensor consumer was not found in blob. please check...')
    current_input_tensor_dominant_consumer_node = ir.graph.nodes()[current_input_tensor_dominant_consumer_node_name]
    found_index,input_index_at_consumer = ir.get_consumer_input_index(current_input_tensor_dominant_consumer_node,input_tensor) # It is important to get index of the first consumer since we will use wmt of this node to actually load the tensor
    if not found_index:
        raise ValueError ('input not found at consumer node. please check...')

    current_op_type = current_input_tensor_dominant_consumer_node['op_type']
    force_folding_y = 'force_folding_y' in current_input_tensor_dominant_consumer_node['frontend']
    force_unfolding_y = 'force_unfolding_y' in current_input_tensor_dominant_consumer_node['frontend']
    current_op_input_tensor_shape = input_tensor.get_folded_shape(folding_conv_y = force_folding_y,unfolding_conv_y=force_unfolding_y)
    current_op_input_channels = current_op_input_tensor_shape[1]
    if current_op_input_channels!=current_input_tensor_dominant_consumer_node['backend']['input_channels'] and current_op_type not in MULTIPLE_INPUT_OPS: # In add the 'backend' 'input_channels' property relates to the conv input size
        raise ValueError ('calced input channels != stored input channels. Please check...')
    input_channels_split = current_input_tensor_dominant_consumer_node['backend']['ic_splits']
    current_tile_tensor_name = input_tensor.get_amm_tensor_name(consumer_node_blob_idx,read_tensor_tile_idx, current_xslice_num)
    tensor_was_already_in_amm = False
    if current_tile_tensor_name not in amms.tensors_in_amm:
        input_channels_per_amm = current_op_input_channels / input_channels_split
        if (input_tensor.num_packed_xslices > 1):
            input_channels_per_amm = math.ceil(input_channels_per_amm/16) * 16 * input_tensor.num_packed_xslices
        blob_input_consumer_name = input_tensor.consumers[0]
        next_ops_inputs_tensors = []
        if current_op_type in MULTIPLE_INPUT_OPS:
            next_ops_inputs_ic_per_amm = []
            # We need to prepare the output in AMM mem ready for the concatenation so we allocate output mem for all inputs of the Concat in contiguous mem
            inputs_nodes_params = current_input_tensor_dominant_consumer_node['frontend']['preceding_nodes_params']
            input_index_sorted_inputs_nodes_params = sorted(inputs_nodes_params, key=lambda x:x[1])
            num_slices = 1
            for input_node_params in input_index_sorted_inputs_nodes_params:
                input_node = ir.graph.nodes[input_node_params[0]]
                num_slices = max(num_slices, input_node['frontend']['x_slices'])
                needed_tensor_name = input_node['frontend']['output_tensor'].name
                if ('_split' in needed_tensor_name):
                    needed_tensor_name = needed_tensor_name.split('_split')[0]
                    next_ops_inputs_tensors.append(ir.tensors[needed_tensor_name])
                    current_input_node_output_channels = ir.tensors[needed_tensor_name].get_folded_shape()[1]
                else:
                    next_ops_inputs_tensors.append(input_node['frontend']['output_tensor']) # We use the y_folding/unfolding of the concat input
                    # Input node output channels will be input channels of the concat node
                    # Input node output tensor is actually the input tensor of consumer node. this is why we use it as below
                    current_input_node_output_channels = input_node['frontend']['output_tensor'].get_folded_shape(folding_conv_y = force_folding_y,unfolding_conv_y=force_unfolding_y)[1]
                next_ops_inputs_ic_per_amm.append(current_input_node_output_channels / input_channels_split)
            allocated_blocks,per_input_first_block_index = amms.allocate_contiguous_mem(allocation_node['name'],next_ops_inputs_tensors,
                                                                                        next_ops_inputs_ic_per_amm,tensors_blob_idx=consumer_node_blob_idx,
                                                                                        tensors_tile_num=read_tensor_tile_idx,current_tile_num=current_tile_idx, num_slices=num_slices, is_valid=False)
            if len(allocated_blocks[0]) == 0:
                print(allocation_node['name'])
                raise ValueError ('Not enough mem for next tile mem allocation')
        else:
            input_tensor_producer_name = input_tensor.producer
            if input_tensor_producer_name:
                input_tensor_producer = ir.graph.nodes()[input_tensor_producer_name]
                contiguous_inputs_allocation_node,long_concat,long_concat_far_node,inputs_nodes_params = following_node_needs_dual_allocation(ir,input_tensor_producer,consumer_node_blob_idx)
                if contiguous_inputs_allocation_node:
                    raise ValueError ('We currently dont take care in allocate_and_mark_for_read_input_tensor of cases where we need to allocate add that is input to concat.')
                    # When take care of the above we wont have cases of current_op_type not in DUAL_CONTIGUOUS_ALLOCATION_OPS but contiguous_inputs_allocation_node=True
                    # This state can occur if one of the inputs to an add op is input to a concat op and this add input is input to the blob
            allocated_blocks = amms.allocate_mem(allocation_node['name'],input_tensor,input_channels_per_amm,tensors_blob_idx=consumer_node_blob_idx,
                                                tensors_tile_num=read_tensor_tile_idx,current_tile_num=current_tile_idx, current_xslice_num = current_xslice_num, is_valid=False) # Is valid means  that in current node we took care of reading the tensor/it is in mem from last op. In this case we handle this below and then set valid to true
            if len(allocated_blocks[0]) == 0:
                raise ValueError ('Not enough mem for next tile mem allocation')
            per_input_first_block_index = [0]
            next_ops_inputs_tensors.append(input_tensor)

        if DEBUG_PRINT_AMM_ALLOCATION:
            print('Allocating %d for blob input tensor (read from ddr) of op: %s' % (input_channels_per_amm,blob_input_consumer_name))
        # TODO: Add comment of what this loop is for
        for input_index,current_input_tensor in enumerate(next_ops_inputs_tensors):
            # TODO: Add comment of what this is for
            current_allocated_blocks = list_of_lists_split_at_pos(allocated_blocks,input_index,per_index_start_pos=per_input_first_block_index)
            if current_op_type in MULTIPLE_INPUT_OPS:
                for xslice_num in range(num_slices):
                    if current_input_tensor.producer in consumer_node_blob.nodes_in_blob: # If we allocated input which its producer is in the blob (This can happen if a concat allocated tensor that is not input to the blob). We need to mark its producer as "output allocated"
                        # TODO: explain why
                        producer_node = ir.graph.nodes()[current_input_tensor.producer]
                        producer_node['backend']['allocated_amm_blocks_for_output_even_grid'][read_tensor_tile_idx][xslice_num] = current_allocated_blocks[xslice_num*AMM_COUNT:xslice_num*AMM_COUNT+AMM_COUNT]
                        producer_node['backend']['allocated_amm_blocks_for_output_odd_grid'][read_tensor_tile_idx][xslice_num] = current_allocated_blocks[xslice_num*AMM_COUNT:xslice_num*AMM_COUNT+AMM_COUNT]
                        if 'output_amm_allocated' in producer_node['backend']:
                            producer_node['backend']['output_amm_allocated'].append((read_tensor_tile_idx, xslice_num))
                        else:
                            producer_node['backend']['output_amm_allocated'] = [(read_tensor_tile_idx, xslice_num)]

                    # TODO: Add comment of what this loop is for
                    for consumer_name in current_input_tensor.consumers:
                        consumer_node = ir.graph.nodes()[consumer_name] # This should happen if current node gets its input from DDR or its an input node to the workload and its input address was never allocated. If it was already allocated then this tensor is used by 2 nodes and one of them used it before it was offloaded to ddr
                        if consumer_name in consumer_node_blob.nodes_in_blob:
                            found_tensor,consumer_node_input_index = ir.get_consumer_input_index(consumer_node,current_input_tensor)
                            consumer_node['backend']['allocated_amm_blocks_for_input_even_grid'][read_tensor_tile_idx][xslice_num][consumer_node_input_index] = current_allocated_blocks[xslice_num*AMM_COUNT:xslice_num*AMM_COUNT+AMM_COUNT]
                            consumer_node['backend']['allocated_amm_blocks_for_input_odd_grid'][read_tensor_tile_idx][xslice_num][consumer_node_input_index] = current_allocated_blocks[xslice_num*AMM_COUNT:xslice_num*AMM_COUNT+AMM_COUNT]    
            else:
                if current_input_tensor.producer in consumer_node_blob.nodes_in_blob: # If we allocated input which its producer is in the blob (This can happen if a concat allocated tensor that is not input to the blob). We need to mark its producer as "output allocated"
                    # TODO: explain why
                    producer_node = ir.graph.nodes()[current_input_tensor.producer]
                    producer_node['backend']['allocated_amm_blocks_for_output_even_grid'][read_tensor_tile_idx][current_xslice_num] = current_allocated_blocks
                    producer_node['backend']['allocated_amm_blocks_for_output_odd_grid'][read_tensor_tile_idx][current_xslice_num] = current_allocated_blocks
                    if 'output_amm_allocated' in producer_node['backend']:
                        producer_node['backend']['output_amm_allocated'].append((read_tensor_tile_idx, current_xslice_num))
                    else:
                        producer_node['backend']['output_amm_allocated'] = [(read_tensor_tile_idx, current_xslice_num)]

                # TODO: Add comment of what this loop is for
                for consumer_name in current_input_tensor.consumers:
                    consumer_node = ir.graph.nodes()[consumer_name] # This should happen if current node gets its input from DDR or its an input node to the workload and its input address was never allocated. If it was already allocated then this tensor is used by 2 nodes and one of them used it before it was offloaded to ddr
                    if consumer_name in consumer_node_blob.nodes_in_blob:
                        found_tensor,consumer_node_input_index = ir.get_consumer_input_index(consumer_node,current_input_tensor)
                        consumer_node['backend']['allocated_amm_blocks_for_input_even_grid'][read_tensor_tile_idx][current_xslice_num][consumer_node_input_index] = current_allocated_blocks
                        consumer_node['backend']['allocated_amm_blocks_for_input_odd_grid'][read_tensor_tile_idx][current_xslice_num][consumer_node_input_index] = current_allocated_blocks
    else:
        tensor_was_already_in_amm = True

    if not amms.tensors_in_amm[current_tile_tensor_name].is_valid: # Is valid will be true if tensor was already loaded (read from ddr) by preceding tile.
        amms.tensors_in_amm[current_tile_tensor_name].is_valid = True
        tensor_found,current_input_tensor_first_consumer_node_name = ir.get_first_tensor_consumer_in_blob(consumer_node_blob,input_tensor)
        if not tensor_found:
            raise ValueError ('Didnt find tensors first consumer, this shouldnt have happend. please check integrity')
        current_input_tensor_first_consumer_node = ir.graph.nodes()[current_input_tensor_first_consumer_node_name]
        tensor_found,first_consumer_node_input_index = ir.get_consumer_input_index(current_input_tensor_first_consumer_node,input_tensor)
        if not tensor_found:
            raise ValueError ('Didnt find tensors first consumer, this shouldnt have happend. please check integrity')
        # The first consumer node is used below in the input tensor info struct as it will be used when actually reading the tesnor for wmt generation. So we must specify the node which
        # its input tensor is actually read and not the allocating node which is the node/time where the tensor is being read. 
        input_tensor_info = InputTensorInfo(input_tensor.name,first_consumer_node_input_index,read_tensor_tile_idx,current_xslice_num,current_tile_idx,current_xslice_num,consumer_node=current_input_tensor_first_consumer_node)
        #input_tensor_info = InputTensorInfo(input_tensor.name,input_index_at_consumer,read_tensor_tile_idx,current_tile_idx,consumer_node=current_input_tensor_consumer_node)
        if sequencer_tensor_load_marker_name in allocation_node['backend']:
            allocation_node['backend'][sequencer_tensor_load_marker_name].append(input_tensor_info)
        else:
            allocation_node['backend'][sequencer_tensor_load_marker_name] = [input_tensor_info]

def read_concats_missing_tensors_from_ddr(ir: internal_representation.IR,node,current_tile_num):
    current_op_input_tensor_name = node['inputs'][0]
    current_op_input_tensor_shape = ir.tensors[current_op_input_tensor_name].get_folded_shape()
    node_blob_idx = node['frontend']['tiling_blob_idx']
    num_xslices = node['frontend']['x_slices']

    # If node needs to read inputs from DDR allocate AMM mem for them
    input_tensors = node['frontend']['input_tensors']
    for input_index,input_tensor in enumerate(input_tensors):
        for current_x_slice in range(num_xslices):
            allocate_and_mark_for_read_input_tensor(ir,input_tensor,node,node_blob_idx,current_tile_num,current_tile_num,current_x_slice,load_before_op = True) #Since its a concat this will allocate both tensors as contigous
                
def allocate_next_tile_input_mem(ir: internal_representation.IR,allocation_node,current_blob_idx,current_tile_num,current_blob_num_tiles,
                                 consecutive_2_tile_blobs, prev_blob_lines_in_ddr):
    if current_tile_num == current_blob_num_tiles-1:
        # If we need to read 1st tile of next blob we make sure all its lines already written to DDR.
        # If not we dont read it here.
        if not ir.is_next_tile_written_to_ddr(current_blob_idx, consecutive_2_tile_blobs, prev_blob_lines_in_ddr):
            if not DEBUG_AVOID_DDR_WRITE_WHILE_CONV:
                allocation_node['backend']['wait_for_last_tile_write_end'] = current_tile_num
            return True
        next_tile_blob_idx = current_blob_idx+1
        next_tile_idx = 0
    else:
        next_tile_blob_idx = current_blob_idx
        next_tile_idx = current_tile_num+1
        
    next_tile_blob = ir.tiling_blobs[next_tile_blob_idx]
    amms = ir.amms

    # If node needs to read inputs from DDR allocate AMM mem for them
    input_tensors = next_tile_blob.inputs
    dominant_tensor_consumer_node_names = []
    for input_tensor in input_tensors:
        # The below gets the "dominant" consumer node. This means that in case of few consumers we need to allocate contigeous mem if one of the consumers is concat.
        found_consumer_in_blob,current_input_tensor_consumer_node_name = ir.get_dominant_tensor_consumer_in_blob(next_tile_blob,input_tensor)
        if not found_consumer_in_blob:
            raise ValueError ('Blobs input tensor consumer was not found in blob. please check...')
        current_input_tensor_consumer_node = ir.graph.nodes()[current_input_tensor_consumer_node_name]
        found_index,input_index_at_consumer = ir.get_consumer_input_index(current_input_tensor_consumer_node,input_tensor) # It is important to get index of the first consumer since we will use wmt of this node to actually load the tensor
        if not found_index:
            raise ValueError ('input not found at consumer node. please check...')

        current_op_type = current_input_tensor_consumer_node['op_type']
        force_folding_y = 'force_folding_y' in current_input_tensor_consumer_node['frontend']
        force_unfolding_y = 'force_unfolding_y' in current_input_tensor_consumer_node['frontend']
        current_op_input_tensor_shape = input_tensor.get_folded_shape(folding_conv_y = force_folding_y,unfolding_conv_y=force_unfolding_y)
        
        current_op_input_channels = current_op_input_tensor_shape[1]
        if current_op_input_channels!=current_input_tensor_consumer_node['backend']['input_channels'] and current_op_type not in MULTIPLE_INPUT_OPS: # In add the 'backend' 'input_channels' property relates to the conv input size
            raise ValueError ('calced input channels != stored input channels. Please check...')
        input_channels_split = current_input_tensor_consumer_node['backend']['ic_splits']

        if (current_op_type in MULTIPLE_INPUT_OPS) and (current_input_tensor_consumer_node_name not in dominant_tensor_consumer_node_names):
            next_ops_inputs_tensors = []
            next_ops_inputs_ic_per_amm = []
            # We need to prepare the output in AMM mem ready for the concatenation so we allocate output mem for all inputs of the Concat in contiguous mem
            inputs_nodes_params = current_input_tensor_consumer_node['frontend']['preceding_nodes_params']
            num_slices = current_input_tensor_consumer_node['frontend']['x_slices']
            input_index_sorted_inputs_nodes_params = sorted(inputs_nodes_params, key=lambda x:x[1])
            for input_node_params in input_index_sorted_inputs_nodes_params:
                input_node = ir.graph.nodes[input_node_params[0]]
                tensor_name = input_node['frontend']['output_tensor'].name
                if ('_split' in tensor_name):
                    tensor_name = tensor_name.split('_split')[0]
                    next_ops_inputs_tensors.append(ir.tensors[tensor_name])
                    current_input_node_output_channels = ir.tensors[tensor_name].get_folded_shape()[1]
                else:
                    next_ops_inputs_tensors.append(input_node['frontend']['output_tensor'])
                    current_input_node_output_channels = input_node['frontend']['output_tensor'].get_folded_shape()[1]
                if 'force_folding_y' in current_input_tensor_consumer_node['frontend']:
                    current_input_node_output_channels = current_input_node_output_channels * 2
                next_ops_inputs_ic_per_amm.append(current_input_node_output_channels / input_channels_split)
            allocated_blocks,per_input_first_block_index = amms.allocate_contiguous_mem(allocation_node['name'],next_ops_inputs_tensors,
                                                                                        next_ops_inputs_ic_per_amm,tensors_blob_idx=next_tile_blob_idx,
                                                                                        tensors_tile_num=next_tile_idx,current_tile_num=current_tile_num, num_slices=num_slices, is_valid=False)
            if len(allocated_blocks[0]) == 0:
                return False
                raise ValueError ('Not enough mem for next tile mem allocation')
            if DEBUG_PRINT_AMM_ALLOCATION:
                print('Allocating %d for blob input tensor (read from ddr) of op: %s' % (input_channels_per_amm,blob_input_consumer_name))
            # After allocating the input mem we now need to update all its consumer nodes (not only the dominant in case of few consumers)
            for input_index,current_input_tensor in enumerate(next_ops_inputs_tensors):
                current_allocated_blocks = list_of_lists_split_at_pos(allocated_blocks,input_index,per_index_start_pos=per_input_first_block_index)
                for current_xslice_idx in range(current_input_tensor.x_slices):
                    if current_input_tensor.producer in next_tile_blob.nodes_in_blob: # If we allocated input which its producer is in the blob (This can happen if a concat allocated tensor that is not input to the blob). We need to mark its producer as "output allocated"
                        producer_node = ir.graph.nodes()[current_input_tensor.producer]
                        producer_node['backend']['allocated_amm_blocks_for_output_even_grid'][next_tile_idx][current_xslice_idx] = current_allocated_blocks[current_xslice_idx*AMM_COUNT:current_xslice_idx*AMM_COUNT+AMM_COUNT]
                        producer_node['backend']['allocated_amm_blocks_for_output_odd_grid'][next_tile_idx][current_xslice_idx] = current_allocated_blocks[current_xslice_idx*AMM_COUNT:current_xslice_idx*AMM_COUNT+AMM_COUNT]
                        if 'output_amm_allocated' in producer_node['backend']:
                            producer_node['backend']['output_amm_allocated'].append((next_tile_idx,current_xslice_idx))
                        else:
                            producer_node['backend']['output_amm_allocated'] = [(next_tile_idx,current_xslice_idx)]

                    for consumer_name in current_input_tensor.consumers:
                        consumer_node = ir.graph.nodes()[consumer_name] # This should happen if current node gets its input from DDR or its an input node to the workload and its input address was never allocated. If it was already allocated then this tensor is used by 2 nodes and one of them used it before it was offloaded to ddr
                        found_tensor,consumer_node_input_index = ir.get_consumer_input_index(consumer_node,current_input_tensor)
                        if consumer_name in next_tile_blob.nodes_in_blob:
                            consumer_node['backend']['allocated_amm_blocks_for_input_even_grid'][next_tile_idx][current_xslice_idx][consumer_node_input_index] = current_allocated_blocks[current_xslice_idx*AMM_COUNT:current_xslice_idx*AMM_COUNT+AMM_COUNT]
                            consumer_node['backend']['allocated_amm_blocks_for_input_odd_grid'][next_tile_idx][current_xslice_idx][consumer_node_input_index] = current_allocated_blocks[current_xslice_idx*AMM_COUNT:current_xslice_idx*AMM_COUNT+AMM_COUNT]
                                        
                    current_tile_tensor_name = current_input_tensor.get_amm_tensor_name(next_tile_blob_idx,next_tile_idx,current_xslice_idx)
                    if (not amms.tensors_in_amm[current_tile_tensor_name].is_valid):
                        amms.tensors_in_amm[current_tile_tensor_name].is_valid = True
                        if current_input_tensor.producer not in next_tile_blob.nodes_in_blob:
                            input_tensor_info = InputTensorInfo(current_input_tensor.name,input_index,next_tile_idx,current_xslice_idx,current_tile_num,current_xslice_idx,consumer_node=current_input_tensor_consumer_node)
                            if 'next_tile_tensors_to_load_from_ddr' in allocation_node['backend']:
                                allocation_node['backend']['next_tile_tensors_to_load_from_ddr'].append(input_tensor_info)
                            else:
                                allocation_node['backend']['next_tile_tensors_to_load_from_ddr'] = [input_tensor_info]
        else:
            num_xslices = input_tensor.x_slices
            num_xslices = num_xslices // input_tensor.num_packed_xslices
            for current_xslice_idx in range(num_xslices):
                current_tile_tensor_name = input_tensor.get_amm_tensor_name(next_tile_blob_idx,next_tile_idx,current_xslice_idx)
                tensor_was_already_in_amm = False
                if current_tile_tensor_name not in amms.tensors_in_amm: # There could be a case where the tensor is already in mem if it is part of a contigeous mem allocation which was already allocated
                    input_channels_per_amm = current_op_input_channels / input_channels_split
                    if (input_tensor.num_packed_xslices > 1):
                        input_channels_per_amm = math.ceil(input_channels_per_amm/16) * 16 * input_tensor.num_packed_xslices
                    blob_input_consumer_name = input_tensor.consumers[0]
                    next_ops_inputs_tensors = []
                    if 'force_folding_y' in current_input_tensor_consumer_node['frontend']:
                        input_channels_per_amm = input_channels_per_amm * 2
                    allocated_blocks = amms.allocate_mem(allocation_node['name'],input_tensor,input_channels_per_amm,tensors_blob_idx=next_tile_blob_idx,
                                                        tensors_tile_num=next_tile_idx,current_tile_num=current_tile_num, current_xslice_num=current_xslice_idx, is_valid=False) # Is valid means  that in current node we took care of reading the tensor/it is in mem from last op. In this case we handle this below and then set valid to true
                    if len(allocated_blocks[0]) == 0:
                        return True # If there is not enough mem maybe we can load it as immediate at blob start
                        raise ValueError ('Not enough mem for next tile mem allocation')
                    per_input_first_block_index = [0]
                    next_ops_inputs_tensors.append(input_tensor)
                    if DEBUG_PRINT_AMM_ALLOCATION:
                        print('Allocating %d for blob input tensor (read from ddr) of op: %s' % (input_channels_per_amm,blob_input_consumer_name))
                    # After allocating the input mem we now need to update all its consumer nodes (not only the dominant in case of few consumers)
                    for input_index,current_input_tensor in enumerate(next_ops_inputs_tensors):
                        current_allocated_blocks = list_of_lists_split_at_pos(allocated_blocks,input_index,per_index_start_pos=per_input_first_block_index)
                        if current_input_tensor.producer in next_tile_blob.nodes_in_blob: # If we allocated input which its producer is in the blob (This can happen if a concat allocated tensor that is not input to the blob). We need to mark its producer as "output allocated"
                            producer_node = ir.graph.nodes()[current_input_tensor.producer]
                            producer_node['backend']['allocated_amm_blocks_for_output_even_grid'][next_tile_idx][current_xslice_idx] = current_allocated_blocks
                            producer_node['backend']['allocated_amm_blocks_for_output_odd_grid'][next_tile_idx][current_xslice_idx] = current_allocated_blocks
                            if 'output_amm_allocated' in producer_node['backend']:
                                producer_node['backend']['output_amm_allocated'].append((next_tile_idx,current_xslice_idx))
                            else:
                                producer_node['backend']['output_amm_allocated'] = [(next_tile_idx,current_xslice_idx)]

                        for consumer_name in current_input_tensor.consumers:
                            consumer_node = ir.graph.nodes()[consumer_name] # This should happen if current node gets its input from DDR or its an input node to the workload and its input address was never allocated. If it was already allocated then this tensor is used by 2 nodes and one of them used it before it was offloaded to ddr
                            found_tensor,consumer_node_input_index = ir.get_consumer_input_index(consumer_node,current_input_tensor)
                            if consumer_name in next_tile_blob.nodes_in_blob:
                                consumer_node['backend']['allocated_amm_blocks_for_input_even_grid'][next_tile_idx][current_xslice_idx][consumer_node_input_index] = current_allocated_blocks
                                consumer_node['backend']['allocated_amm_blocks_for_input_odd_grid'][next_tile_idx][current_xslice_idx][consumer_node_input_index] = current_allocated_blocks
                else:
                    tensor_was_already_in_amm = True

                if (not amms.tensors_in_amm[current_tile_tensor_name].is_valid):
                    amms.tensors_in_amm[current_tile_tensor_name].is_valid = True
                    input_tensor_info = InputTensorInfo(input_tensor.name,input_index_at_consumer,next_tile_idx,current_xslice_idx,current_tile_num,current_xslice_idx,consumer_node=current_input_tensor_consumer_node)
                    if 'next_tile_tensors_to_load_from_ddr' in allocation_node['backend']:
                        allocation_node['backend']['next_tile_tensors_to_load_from_ddr'].append(input_tensor_info)
                    else:
                        allocation_node['backend']['next_tile_tensors_to_load_from_ddr'] = [input_tensor_info]
        dominant_tensor_consumer_node_names.append(current_input_tensor_consumer_node_name)

    return True

def get_concat_output_allocation_from_inputs_allocation(inputs_allocation):
    output_allocation=[]
    for idx,current_input_allocation in enumerate(inputs_allocation):
        if idx==0:
            output_allocation = copy.deepcopy(current_input_allocation)
        else:
            for amm_idx,amm_allocation in enumerate(current_input_allocation):
                output_allocation[amm_idx].extend(amm_allocation)
    return output_allocation

'''
def set_output_alocation_to_node(ir,node,even_grid_blocks_split,odd_grid_blocks_split,current_tile_num=0):
    grid_mode = node['backend']['gridmode']
    is_folding_conv = 'folding_conv' in node['backend']
    if 'output_amm_allocated' in node['backend'] and current_tile_num in node['backend']['output_amm_allocated']:
        return
    
    if 'output_amm_allocated' in node['backend']:
        node['backend']['output_amm_allocated'].append(current_tile_num)
    else:
        node['backend']['output_amm_allocated'] = [current_tile_num]
    if 'allocated_amm_blocks_for_output_even_grid' in node['backend']:
        raise ValueError ('Didnt expect this already set. Please check...')
    if 'allocated_amm_blocks_for_output_odd_grid' in node['backend']:
        raise ValueError ('Didnt expect this already set. Please check...')
    node['backend']['allocated_amm_blocks_for_output_even_grid'][current_tile_num] = even_grid_blocks_split
    node['backend']['allocated_amm_blocks_for_output_odd_grid'][current_tile_num] = odd_grid_blocks_split
    if node['op_type'] == 'Add':
        node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][0] = even_grid_blocks_split
        node['backend']['allocated_amm_blocks_for_input_odd_grid'][current_tile_num][0]= odd_grid_blocks_split
        add_input0_producer_node= ir.graph.nodes[node['frontend']['preceding_nodes_params'][0][0]]
        set_output_alocation_to_node(ir,add_input0_producer_node,even_grid_blocks_split,odd_grid_blocks_split,current_tile_num=current_tile_num)
    # Current op output amm blocks are following op's input amm block
    nodes_following_current_twin_node_params = node['frontend']['following_nodes_params']
    for current_following_node in nodes_following_current_twin_node_params:
        following_node = ir.graph.nodes[current_following_node[0]]
        input_index = current_following_node[1]
        if following_node['op_type'] == 'Add' and input_index==0 and current_tile_num in node['backend']['output_amm_allocated']: # If add node output is allocated it input0 is already allocated since its inline op for input 0
            continue
        following_node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][input_index] = even_grid_blocks_split
        following_node['backend']['allocated_amm_blocks_for_input_odd_grid'][current_tile_num][input_index] = odd_grid_blocks_split
'''

def set_input_alocation_to_following_far_node(ir,node,even_grid_blocks_split,odd_grid_blocks_split,current_tile_num=0):
    # Current op output amm blocks are following op's input amm block
    far_node_following_current_twin_node_params = node['frontend']['following_nodes_params'][-1] # The last node on the list is guranteed to be the last executed one
    following_node = ir.graph.nodes[far_node_following_current_twin_node_params[0]]
    if following_node['op_type'] != 'Concat':
        raise ValueError ('Expected it to be the concat. Please Check')
    input_index = far_node_following_current_twin_node_params[1]
    for current_xslice_num in range(following_node['frontend']['x_slices']):
        following_node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][current_xslice_num][input_index] = even_grid_blocks_split[current_xslice_num*AMM_COUNT:current_xslice_num*AMM_COUNT+AMM_COUNT]
        following_node['backend']['allocated_amm_blocks_for_input_odd_grid'][current_tile_num][current_xslice_num][input_index] = odd_grid_blocks_split[current_xslice_num*AMM_COUNT:current_xslice_num*AMM_COUNT+AMM_COUNT]

def move_outputs_write_to_amm_to_erliest_point(ir: internal_representation.IR):
    nodes_execution_order = ir.lexicographical_topological_sorted_graph

    output_tensors = ir.outputs
    # find producing node_per each output
    outputs_producing_nodes_names = []
    for output_tensor in output_tensors:
        # If this tensor was split, append both the split tensor producers
        if output_tensor in ir.original_tensor_to_split_tensor_map:
            for split_tensor in ir.original_tensor_to_split_tensor_map[output_tensor]:
                outputs_producing_nodes_names.append(ir.tensors[split_tensor].producer)
        # Otherwise append the original
        else:
            outputs_producing_nodes_names.append(ir.tensors[output_tensor].producer)

    for output_producing_node_name in outputs_producing_nodes_names:
        output_producing_node = ir.graph.nodes[output_producing_node_name] # This would be an ordering conv
        # Move each output producing node right after its last preceding node
        output_producing_node_preceding_nodes_params = output_producing_node['frontend']['preceding_nodes_params']

        # If this output has no preceding nodes, then skip it.
        # This would only happen for simple tests with 1-layer graphs, if there are no ordering convs.
        if not output_producing_node_preceding_nodes_params:
            continue

        last_preceding_node_name = output_producing_node_preceding_nodes_params[-1][0] # This would be the original op that produced the output
        last_preceding_node = ir.graph.nodes[last_preceding_node_name]
        ops_before_original_output_producing_node = last_preceding_node['frontend']['preceding_nodes_params']
        if len(ops_before_original_output_producing_node)==0: # If its the 1st op in the workload
            continue
        anchor_node_name = ops_before_original_output_producing_node[-1][0] # This is the node that preceeds the original output producing node . We want to move the producing node and its ordering conv right after this one so it doesnt stay in AMM
        anchor_node = ir.graph.nodes[anchor_node_name]
        anchor_node_following_nodes_params = anchor_node['frontend']['following_nodes_params']
        # Check if anchor node is followed by an ordering node. If so we dont want to get between them and we will move our output node after the ordering node
        is_anchor_followed_by_reorder_node = False
        for node_idx,anchor_node_following_node_params in enumerate(anchor_node_following_nodes_params):
            anchor_node_following_node = ir.graph.nodes[anchor_node_following_node_params[0]]
            if 'reorder_node' in anchor_node_following_node:
                is_anchor_followed_by_reorder_node = True
                if node_idx!=0:
                    # We count on the fact that the ordering node is executed right after its producing node and hence it must be first in 'following_nodes_params'
                    raise ValueError ('Expected ordering node to be the first executed node after anchor node. If its not check why')
                break
        #find index of last preceding node in following nodes params since we need to move it as first following node
        following_node_index = 0
        for anchor_node_following_node_params in anchor_node_following_nodes_params:
            if anchor_node_following_node_params[0] == last_preceding_node_name:
                break
            following_node_index+=1
        last_preceding_node_params = anchor_node_following_nodes_params.pop(following_node_index)
        if is_anchor_followed_by_reorder_node:
            anchor_node_following_nodes_params.insert(1,last_preceding_node_params)
        else:
            anchor_node_following_nodes_params.insert(0,last_preceding_node_params)
        anchor_node_index = nodes_execution_order.index(anchor_node_name)
        if is_anchor_followed_by_reorder_node:
            anchor_node_index+=1
        last_preceding_node_execution_index = nodes_execution_order.index(last_preceding_node_name)
        current_output_node_execution_index = nodes_execution_order.index(output_producing_node_name)
        #if current_output_node_execution_index!=last_preceding_node_execution_index+1:
        #    raise ValueError ('Expected ordering node to execure right after its preceding node. Please check...')
        if current_output_node_execution_index>anchor_node_index+2: # Move original output producing node and its ordering conv
            nodes_execution_order.pop(current_output_node_execution_index) # Remove ordering conv
            nodes_execution_order.pop(last_preceding_node_execution_index) # Remove original output producing node
            nodes_execution_order.insert(anchor_node_index+1,output_producing_node_name) # Add ordering node back
            nodes_execution_order.insert(anchor_node_index+1,last_preceding_node_name) # Add original output producing node back

def clear_previous_allocation_decisions(ir: internal_representation.IR):
    for node_name in ir.lexicographical_topological_sorted_graph:
        node = ir.graph.nodes()[node_name]
        fields_to_clear = ['tensors_to_offload_to_ddr','tensors_for_deallocation_after_output_allocation','next_tile_tensors_to_load_from_ddr','tensors_to_load_immediately_from_ddr',
                           'output_amm_allocated','tensors_for_deallocation_after_ddr_read_allocation']
        for field_name in fields_to_clear:
            if field_name in node['backend']:
                del(node['backend'][field_name])
        fields_for_input_allocations = ['allocated_amm_blocks_for_input_even_grid','allocated_amm_blocks_for_input_odd_grid']
        for field_name in fields_for_input_allocations:
            op_inputs_num = len(ir.get_non_constant_inputs(node))
            y_tiles = node['frontend']['y_tiles']
            x_slices = node['frontend']['x_slices']
            node['backend'][field_name] = [[[[] for i in range(op_inputs_num)] for j in range(x_slices)] for k in range(y_tiles)]
        fields_for_output_allocations = ['allocated_amm_blocks_for_output_even_grid','allocated_amm_blocks_for_output_odd_grid']
        for field_name in fields_for_output_allocations:
            op_inputs_num = len(ir.get_non_constant_inputs(node))
            y_tiles = node['frontend']['y_tiles']
            x_slices = node['frontend']['output_tensor'].x_slices
            node['backend'][field_name] = [[[] for j in range(x_slices)] for k in range(y_tiles)]
    # Clear DDR Intermediate tensors
    for ddr_entry in ir.ddr.entries.copy():
        if ddr_entry.type in [DDREntryType.INTERMEDIATE_TENSOR]:
            ir.ddr.remove_entry(ddr_entry)
    pass

def find_offloading_options(ir:internal_representation.IR,node): # This tries to find long concats for offloading tesnsors. If found they will be marked and we restart allocation
    current_tensors_in_amm = ir.amms.tensors_in_amm
    possible_long_concats=[]

    for tensor_name in current_tensors_in_amm:
        current_tensor_deallocation_node = ir.amms.tensors_in_amm[tensor_name].deallocating_node
        if current_tensor_deallocation_node!=None and current_tensor_deallocation_node['op_type'] == 'Concat':
            concat_execution_order = ir.lexicographical_topological_sorted_graph.index(current_tensor_deallocation_node['name'])
            possible_long_concats.append((current_tensor_deallocation_node['name'],concat_execution_order))
    # We sort possible concats by their offloading value
    sorted_possible_concats = sorted(possible_long_concats, key=lambda x:x[1],reverse=True)
    for concat_node_name,execution_index in sorted_possible_concats:
        if concat_node_name in ir.offloaded_concat_ops:
            continue
        else:
            ir.offloaded_concat_ops.append(concat_node_name)
            return True
    return False


# When mem allocation fails, a blob is split.
# This function determines which blob to split.
# In the future, this function might determine a specific node instead.
def get_blob_to_split(current_blob_idx, current_y_tile, ir):
    # If this is tile 0, try splitting the previous blob, if it
    # can still be split.
    if current_y_tile == 0 and current_blob_idx > 0:
        if len(ir.tiling_blobs[current_blob_idx - 1].nodes_in_blob) > 1:
            return current_blob_idx - 1

    # Otherwise, try splitting this blob, if it can still be split
    if len(ir.tiling_blobs[current_blob_idx].nodes_in_blob) > 1:
        return current_blob_idx

    # Otherwise, if this is the last tile, try splitting the next blob,
    # if it can still be split.
    num_y_tiles = ir.tiling_blobs[current_blob_idx].y_tiles
    num_blobs = len(ir.tiling_blobs)
    if current_y_tile == (num_y_tiles-1) and current_blob_idx < (num_blobs-1):
        if len(ir.tiling_blobs[current_blob_idx + 1].nodes_in_blob) > 1:
            return current_blob_idx + 1

    # Can add additional cases here
    return current_blob_idx

def mem_allocation_pass(ir: internal_representation.IR, debug_output_dir:str):
    amms = ir.amms
    amms.reset_mem()
    clear_previous_allocation_decisions(ir)

    # Allocate system-reserved tensors in DDR and AMM (e.g., zeros in last block of each AMM).
    # This needs to be done after the memory is reset above.
    ir = allocate_system_tensors(ir)

    allocation_succeeded = True
    last_blob_idx=0
    tiling_blobs=ir.tiling_blobs
    blobs_tqdm_iterator = tqdm(tiling_blobs.values())

    # Keep track of the number of consecutive 2 tile blobs
    consecutive_2_tile_blobs = 0

    # Each tiling blob will be updated to store this
    # Then at the end of the loop this will be updated based on the node
    prev_blob_lines_in_ddr = -1 # -1 means it's the input

    for current_blob_idx,current_tiling_blob in enumerate(blobs_tqdm_iterator):

        if current_tiling_blob.y_tiles == 2:
            consecutive_2_tile_blobs += 1
        else:
            consecutive_2_tile_blobs = 0

        blobs_tqdm_iterator.set_description('Mem allocation, at blob#%d:' % current_blob_idx)
        current_blob_num_tiles = current_tiling_blob.y_tiles
        tiles_tqdm_iterator = tqdm(range(current_blob_num_tiles))
        nodes_tqdm_iterator = tqdm(current_tiling_blob.nodes_in_blob)
        for current_y_tile in tiles_tqdm_iterator:
            tiles_tqdm_iterator.set_description('Tile#%d:' % current_y_tile)
            for current_node_idx,current_node_name in enumerate(nodes_tqdm_iterator):
                #if current_y_tile==0 and current_blob_idx==15 and current_node_idx==0:
                #    print('') # This if is for debug. to add a breakpoint in specific node/tile
                amms.check_mem_integrity()
                nodes_tqdm_iterator.set_description('At node:%s:' % current_node_name)
                node = ir.graph.nodes()[current_node_name]
                current_op_type = node['op_type']
                blob_outputs = current_tiling_blob.outputs
                outputs_producing_nodes = [output_tensor.producer for output_tensor in blob_outputs]
                if current_op_type in MULTIPLE_INPUT_OPS: # We need to update the tensors in amm structure so that inputs are no longer in AMM and output is. Also set it deallocating node
                    num_slices = node['frontend']['x_slices']
                    read_concats_missing_tensors_from_ddr(ir,node,current_y_tile)
                    input_tensors = node['frontend']['input_tensors']
                    output_tensor = node['frontend']['output_tensor']
                    xslices_allocated_blocks = []
                    for current_xslice_num in range(num_slices):
                        if current_op_type == 'Concat':
                            even_grid_allocated_blocks = get_concat_output_allocation_from_inputs_allocation(node['backend']['allocated_amm_blocks_for_input_even_grid'][current_y_tile][current_xslice_num])
                            odd_grid_allocated_blocks = get_concat_output_allocation_from_inputs_allocation(node['backend']['allocated_amm_blocks_for_input_odd_grid'][current_y_tile][current_xslice_num])    
                        else:
                            even_grid_allocated_blocks = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_y_tile][current_xslice_num][0]
                            odd_grid_allocated_blocks = node['backend']['allocated_amm_blocks_for_input_odd_grid'][current_y_tile][current_xslice_num][0]
                        node['backend']['allocated_amm_blocks_for_output_even_grid'][current_y_tile][current_xslice_num] = even_grid_allocated_blocks
                        node['backend']['allocated_amm_blocks_for_output_odd_grid'][current_y_tile][current_xslice_num] = odd_grid_allocated_blocks
                        amms.tensors_in_amm[output_tensor.get_amm_tensor_name(current_blob_idx,current_y_tile,current_xslice_num)] = AMMTensor(output_tensor,even_grid_allocated_blocks,odd_grid_allocated_blocks,current_y_tile,current_xslice_num,is_valid=True,allocated_at=(node['name'],current_y_tile,current_xslice_num))
                        xslices_allocated_blocks.append(even_grid_allocated_blocks)
                    # If there are tensors we need to deallocate here we do it before we add the input tensors to 'tensors_for_deallocation_after_output_allocation'
                    # This is because for the input tensors we just add them to the list but dont actually deallocate mem as it will be used by output tensor of concat
                    if 'tensors_for_deallocation_after_output_allocation' in node['backend']: 
                        deallocated_tensors = node['backend']['tensors_for_deallocation_after_output_allocation']
                        node['backend']['tensors_for_deallocation_after_output_allocation_report'] = deallocated_tensors# The 'tensors_for_deallocation_after_output_allocation' field is used to check if all tensors were deallocated this new one is for report
                        amms.deallocate_amm_tensors(node,deallocated_tensors,force_free_mem=False,current_tile_num=current_y_tile)
                    if 'tensors_for_deallocation_after_ddr_read_allocation' in node['backend']:
                        deallocated_tensors = node['backend']['tensors_for_deallocation_after_ddr_read_allocation']
                        node['backend']['tensors_for_deallocation_after_ddr_read_allocation_report'] = deallocated_tensors# The 'tensors_for_deallocation_after_output_allocation' field is used to check if all tensors were deallocated this new one is for report
                        amms.deallocate_amm_tensors(node,deallocated_tensors,force_free_mem=False,current_tile_num=current_y_tile)
                    for input_index,input_tensor in enumerate(input_tensors):
                        inline_tensor=True
                        input_tensor.is_inline_tensor = True
                        for current_xslice_num in range(num_slices):
                            amm_tensor_name = input_tensor.get_amm_tensor_name(current_blob_idx,current_y_tile,current_xslice_num)
                            dellocated_tensor = TensorDeAllocationInfo(amm_tensor_name,input_index,current_y_tile,current_xslice_num,current_y_tile,current_xslice_num,inline_tensor=inline_tensor)
                            if 'tensors_for_deallocation_after_output_allocation' in node['backend']:
                                node['backend']['tensors_for_deallocation_after_output_allocation'].add_tensor(dellocated_tensor)
                            else: 
                                tensor_list = TensorDeAllocationList()
                                tensor_list.add_tensor(dellocated_tensor)
                                node['backend']['tensors_for_deallocation_after_output_allocation'] = tensor_list

                            if amm_tensor_name not in amms.tensors_in_amm:
                                raise ValueError ('At Concat, input %s is not in mem, need to read from DDR but this is not supported yet.' % input_tensor.name)
                            if current_op_type == 'Concat':
                                del amms.tensors_in_amm[amm_tensor_name] #We only delete the tensors from AMM but dont deallocate the actual mem as it is still used by concat output

                    following_nodes_params = node['frontend']['following_nodes_params']
                    get_next_op_grid_config(ir,node,following_nodes_params) # We need to know the following op grid config (including input channel splitting) in order to write current op results in right places
                    allocated_blocks = xslices_allocated_blocks
                    set_following_nodes_inputs_allocations(ir,node,following_nodes_params,allocated_blocks,current_tile_num=current_y_tile)
                    
                    if (current_op_type == 'Add'):
                        # Set deallocation point for each input tensor
                        mark_input_tensors_deallocation(ir,amms,node,current_y_tile)
                
                    if current_node_name in outputs_producing_nodes: # output tensors in each blob should be de-allocated after they are written to DDR
                        mark_blob_outputs_deallocations(ir,amms,node,current_y_tile)
                else:
                    current_op_output_tensor_name = node['outputs'][0]
                    current_op_output_tensor_shape = ir.tensors[current_op_output_tensor_name].get_folded_shape()
                    current_op_output_channels = current_op_output_tensor_shape[1]
                    
                    following_nodes_params = node['frontend']['following_nodes_params']
                    get_next_op_grid_config(ir,node,following_nodes_params) # We need to know the following op grid config (including input channel splitting) in order to write current op results in right places

                    # If node needs to read inputs from DDR allocate AMM mem for them
                    read_missing_tensors_from_ddr(ir,node,current_y_tile)
                    # Set deallocation point for each input tensor
                    mark_input_tensors_deallocation(ir,amms,node,current_y_tile)
                    
                    if 'tensors_for_deallocation_after_ddr_read_allocation' in node['backend']:
                        deallocated_tensors = node['backend']['tensors_for_deallocation_after_ddr_read_allocation']
                        node['backend']['tensors_for_deallocation_after_ddr_read_allocation_report'] = deallocated_tensors# The 'tensors_for_deallocation_after_output_allocation' field is used to check if all tensors were deallocated this new one is for report
                        amms.deallocate_amm_tensors(node,deallocated_tensors,force_free_mem=False,current_tile_num=current_y_tile)

                    # Allocate AMM for output
                    # There are cases where an op's output AMM was already allocated.
                    # This happens if the output is going to a DUAL_CONTIGUOUS_ALLOCATION_OPS
                    # In such case the two inputs must be allocated together to make sure they are contiguous in mem
                    current_node_output_tensor = node['frontend']['output_tensor']
                    contiguous_inputs_allocation_node,long_concat,long_concat_far_node,inputs_nodes_params = following_node_needs_dual_allocation(ir,node,current_blob_idx) # Twin node is a node which produces one of the inputs of a dual input node. It is called twin since it has a twin node that produces the other input of that dual input node
                    allocated_blocks = []
                    even_grid_allocated_blocks = []
                    odd_grid_allocated_blocks = []
                    if long_concat_far_node:
                        offload_tensor = node['frontend']['output_tensor']
                        offloaded_tensor_name = offload_tensor.name
                        ir.intermediate_ddr_tensors[offloaded_tensor_name] = internal_representation.IntermediateDDRTensor(offloaded_tensor_name,offloading_node_name = node['name'])
                        raise ValueError ('Something went wrong AMM allocation, didnt find best solution long_concat_far_node')
                        allocate_ddr_for_offloaded_tensors(ir,[offloaded_tensor_name])
                        next_executing_node = ir.get_next_executed_node(node)
                        next_executing_node['backend']['tensors_to_offload_to_ddr'] = [offload_tensor] # Actual write to DDR will happen in next execution node since it guranttes that tensor was generated
                    if contiguous_inputs_allocation_node and not long_concat_far_node: # This means that current node is input to a DUAL_CONTIGUOUS_ALLOCATION_OPS and it was not offloaded to DDR (only the far node is offloaded to DDR)
                        num_slices=node['frontend']['output_tensor'].x_slices
                        if 'output_amm_allocated' in node['backend'] and (current_y_tile, 0) in node['backend']['output_amm_allocated']: # Since two input ops will try to allocate mem twice (once per each input) we need to make sure we dont allocate twice
                            for current_xslice_num in range(num_slices):
                                output_tensor_name = current_node_output_tensor.get_amm_tensor_name(current_blob_idx,current_y_tile,current_xslice_num)
                                if output_tensor_name not in amms.tensors_in_amm:
                                    # Seems like this happens becuase we have one of concat inputs is add in same blob. in such case the tensor name we need in amm is the add's input
                                    raise ValueError ('Need to revisit this. Since i moved tensors_in_amm setting into contigous_mem_all i would expect that its already set')
                                else:
                                    amms.tensors_in_amm[output_tensor_name].is_valid=True
                        else: # Need to allocate output amm for all inputs of multiple input op to make sure its contiguous in mem
                            next_ops_inputs_ic_per_amm = []
                            next_ops_inputs_tensors = []
                            # We need to prepare the output in AMM mem ready for the concatenation so we allocate output mem for all inputs of the Concat in contiguous mem
                            input_index_sorted_inputs_nodes_params = sorted(inputs_nodes_params, key=lambda x:x[1])
                            for input_node_params in input_index_sorted_inputs_nodes_params:
                                input_node = ir.graph.nodes[input_node_params[0]]
                                next_ops_inputs_tensors.append(input_node['frontend']['output_tensor'])
                                current_input_node_output_channels = input_node['frontend']['output_tensor'].get_folded_shape()[1]
                                
                                next_ops_inputs_ic_per_amm.append(current_input_node_output_channels / node['backend']['following_op_ic_split'])
                            
                            if DEBUG_PRINT_AMM_ALLOCATION:
                                print('Allocating %s for output tensor of %s, we allocated 2x since it goes to dual input op(e.g. add)' % (str(next_ops_inputs_ic_per_amm),current_node_name))
                            
                            allocated_blocks,per_input_first_block_index = amms.allocate_contiguous_mem(current_node_name,next_ops_inputs_tensors,
                                                                                                        next_ops_inputs_ic_per_amm,tensors_blob_idx=current_blob_idx,
                                                                                                        tensors_tile_num = current_y_tile,current_tile_num=current_y_tile, num_slices=num_slices, is_valid=False)
                            if len(allocated_blocks[0]) == 0:
                                return ir, False, get_blob_to_split(current_blob_idx, current_y_tile, ir)

                            for node_idx,current_input_node_params in enumerate(inputs_nodes_params): # setting allocations for each of the twin nodes
                                input_idx = current_input_node_params[1]
                                current_input_node = ir.graph.nodes[current_input_node_params[0]]
                                current_input_node_blob_idx = current_input_node['frontend']['tiling_blob_idx']
                                blocks_split=list_of_lists_split_at_pos(allocated_blocks,input_idx,per_index_start_pos=per_input_first_block_index)
                                if (current_input_node['op_type'] == 'Add'):
                                    following_add_node = ir.graph.nodes[current_input_node['frontend']['following_nodes_params'][0][0]]
                                    if (following_add_node['op_type'] == 'Concat') and node_idx==0: #inputs_nodes_params are sorted by execution order so node_idx==0 means its the far node
                                        set_input_alocation_to_following_far_node(ir,current_input_node,blocks_split,blocks_split,current_tile_num=current_y_tile)
                                xslices_allocated_blocks = []
                                for current_xslice_num in range(num_slices):
                                    if current_input_node_blob_idx==current_blob_idx: # If the source node is from different blob we dont need to set its output allocation
                                        current_input_node['backend']['allocated_amm_blocks_for_output_even_grid'][current_y_tile][current_xslice_num] = blocks_split[current_xslice_num*AMM_COUNT:current_xslice_num*AMM_COUNT+AMM_COUNT]
                                        current_input_node['backend']['allocated_amm_blocks_for_output_odd_grid'][current_y_tile][current_xslice_num] = blocks_split[current_xslice_num*AMM_COUNT:current_xslice_num*AMM_COUNT+AMM_COUNT]
                                        if 'output_amm_allocated' in current_input_node['backend']:
                                            current_input_node['backend']['output_amm_allocated'].append((current_y_tile,current_xslice_num))
                                        else:
                                            current_input_node['backend']['output_amm_allocated'] = [(current_y_tile, current_xslice_num)]
                                        output_tensor = current_input_node['frontend']['output_tensor']
                                        output_tensor_name = current_node_output_tensor.get_amm_tensor_name(current_blob_idx,current_y_tile,current_xslice_num)
                                        if output_tensor_name not in amms.tensors_in_amm:
                                            # Seems like this happens becuase we have one of concat inputs is add in same blob. in such case the tensor name we need in amm is the add's input
                                            raise ValueError ('Need to revisit this. Since i moved tensors_in_amm setting into contigous_mem_all i would expect that its already set')
                                        else:
                                            amms.tensors_in_amm[output_tensor_name].is_valid=True
                                    xslices_allocated_blocks.append(blocks_split[current_xslice_num*AMM_COUNT:current_xslice_num*AMM_COUNT+AMM_COUNT])
                                current_input_node_following_nodes_params = current_input_node['frontend']['following_nodes_params']
                                set_following_nodes_inputs_allocations(ir,node,current_input_node_following_nodes_params,xslices_allocated_blocks,current_tile_num=current_y_tile)

                    else: # In allocation of output tensor in AMM in case of non dual_contiguous_input_allocation_node
                        next_op_input_channels_per_amm = current_op_output_channels / node['backend']['following_op_ic_split'] # We need to prepare the output in AMM mem ready as the input of next op. So the next op input channels are actually the following op input channels
                        if DEBUG_PRINT_AMM_ALLOCATION:
                            print('Allocating %d for output tensor of %s' % (next_op_input_channels_per_amm,current_node_name))
                        if 'ddr_stall_test' in ir.model_name and node['name']=='X_Conv3': # Since in this ddr stall test we want to have read during conv we fail allocation so that tensors will be written and read from DDR
                            allocated_blocks = [[],[]]
                        else:
                            allocated_blocks = []
                            for current_x_slice in range(node['frontend']['output_tensor'].x_slices):
                                temp_allocated_blocks = amms.allocate_mem(current_node_name,current_node_output_tensor,next_op_input_channels_per_amm,
                                                                tensors_blob_idx=current_blob_idx,tensors_tile_num=current_y_tile,
                                                                current_tile_num=current_y_tile,current_xslice_num=current_x_slice,is_valid=True) # Is valid means  that in current node we took care of reading the tensor/it is in mem from last op. In this case tensor is in mem from current op
                                if (len(temp_allocated_blocks[0]) == 0):
                                    if DEBUG_PRINT_AMM_ALLOCATION:
                                        print('Failed in mem allocation. trying to set concats as long concat and re-trying')
                                    current_node_tensors = []
                                    input_tensors_names = ir.get_non_constant_inputs(node)
                                    current_node_tensors.extend(input_tensors_names)
                                    current_node_tensors.extend([node['frontend']['output_tensor'].name])
                                    current_node_tensors = amms.add_blob_and_tile_num_to_tensors(current_node_tensors,current_blob_idx,current_y_tile,current_x_slice)
                                    skip_tensor_names,skip_tensors =amms.get_amm_tensors_except_current_node_tensors(current_node_tensors)
                                    for skip_tensor_name in skip_tensor_names:
                                        ir.intermediate_ddr_tensors[skip_tensor_name] = internal_representation.IntermediateDDRTensor(skip_tensor_name,offloading_node_name = node['name'])
                                    if current_tiling_blob.y_tiles>1:
                                        return ir, False, get_blob_to_split(current_blob_idx, current_y_tile, ir)
                                        #raise ValueError ('Need to add support to tensor offloading with tiles. allocate DDR once')
                                    success = allocate_ddr_for_offloaded_tensors(ir,skip_tensor_names)
                                    if not success:
                                        return ir, False, get_blob_to_split(current_blob_idx, current_y_tile, ir)
                                    node['backend']['tensors_to_offload_to_ddr'] = skip_tensors
                                    amms.deallocate_amm_tensors(node,skip_tensor_names,force_free_mem=True,current_tile_num=current_y_tile)
                                    temp_allocated_blocks = amms.allocate_mem(current_node_name,current_node_output_tensor,next_op_input_channels_per_amm,
                                                                        tensors_blob_idx=current_blob_idx,tensors_tile_num=current_y_tile,
                                                                        current_tile_num=current_y_tile,current_xslice_num=current_x_slice,is_valid=True) # Is valid means  that in current node we took care of reading the tensor/it is in mem from last op. In this case tensor is in mem from current op
                                    if len(temp_allocated_blocks[0]) == 0:
                                        raise ValueError('Failed in amm allocation 2nd try after offloading skip tensors to amm at node: %s' % current_node_name)
                                allocated_blocks.append(temp_allocated_blocks)

                        for current_x_slice in range(node['frontend']['output_tensor'].x_slices):
                            node['backend']['allocated_amm_blocks_for_output_even_grid'][current_y_tile][current_x_slice] = allocated_blocks[current_x_slice]
                            node['backend']['allocated_amm_blocks_for_output_odd_grid'][current_y_tile][current_x_slice] = allocated_blocks[current_x_slice]
                        set_following_nodes_inputs_allocations(ir,node,following_nodes_params,allocated_blocks,current_tile_num=current_y_tile)
                        
                    
                    if current_node_name in outputs_producing_nodes: # output tensors in each blob should be de-allocated after they are written to DDR
                        mark_blob_outputs_deallocations(ir,amms,node,current_y_tile)
                    
                    # Deallocate blocks set for deallocation after current engine op allocation
                    if 'tensors_for_deallocation_after_output_allocation' in node['backend']:
                        deallocated_tensors = node['backend']['tensors_for_deallocation_after_output_allocation']
                        node['backend']['tensors_for_deallocation_after_output_allocation_report'] = deallocated_tensors# The 'tensors_for_deallocation_after_output_allocation' field is used to check if all tensors were deallocated this new one is for report
                        amms.deallocate_amm_tensors(node,deallocated_tensors,force_free_mem=False,current_tile_num=current_y_tile)

                    if current_blob_idx!=len(tiling_blobs)-1:
                        next_blob_num_tiles = tiling_blobs[current_blob_idx+1].y_tiles
                    else:
                        next_blob_num_tiles = 0
                    if current_node_idx == current_tiling_blob.next_tile_read_node_idx: # We are in the node that performs read of next tile
                        node['backend']['next_tile_read_node'] = True
                        if ((current_y_tile!=current_blob_num_tiles-1) and current_blob_num_tiles>1) or\
                            ((current_y_tile==current_blob_num_tiles-1) and next_blob_num_tiles>1): # If current blob has no tiles we dont need to read next tile
                            
                            if not allocate_next_tile_input_mem(ir,node,current_blob_idx,current_y_tile,current_blob_num_tiles,
                                                        consecutive_2_tile_blobs, prev_blob_lines_in_ddr):
                                return ir, False, get_blob_to_split(current_blob_idx, current_y_tile, ir)

                        if (current_y_tile==current_blob_num_tiles-1) and (current_blob_idx==len(tiling_blobs)-1): # If its last tile in last blob we dont need to read/allocate
                            # But we need to mark it for adding a custom wait for the last tile write. Normally this wait is done by next tile read but in this case
                            # There is no next tile read
                            node['backend']['wait_for_last_tile_write_end'] = current_y_tile
                    
                    
                if DEBUG_PRINT_AMM_ALLOCATION:
                    print(amms.amms[0].amm_allocator.allocated_blocks)
                node['backend']['post_op_mem_utilization'] = amms.get_current_amms_utilization()

        # Before moving to the next blob, store the number of lines the previous blob will write to DDR
        # when the next blob will start reading
        # NOTE: If current_tiling_blob.next_tile_read_node_idx is a concat, this is not set
        prev_blob_lines_in_ddr = current_tiling_blob.num_lines_written_to_ddr_before_next_blob_read
        #assert prev_blob_lines_in_ddr > 0

    return ir,allocation_succeeded,None

def create_statistics_atributes(node):
    node['backend']['statistics'] = {}
    node['backend']['statistics']['wloc_conv_cmd_count'] = 0 # will be updated
    node['backend']['statistics']['rq_waiting_nops']     = 0 # will be updated
    node['backend']['statistics']['nops_end_of_split']   = 0 # will be updated
    node['backend']['statistics']['stall_nops']          = 0 # will be updated
    node['backend']['statistics']['wloc_all_splits']     = 0 # will be updated
    node['backend']['statistics']['wloc_pairs']          = 0 # will be updated

def cbc_generation_pass(ir: internal_representation.IR, debug_output_dir:str) -> internal_representation.IR:

    tqdm_iterator = tqdm(ir.lexicographical_topological_sorted_graph)
    for node_name in tqdm_iterator:
        node = ir.graph.nodes()[node_name]
        tqdm_iterator.set_description('CBC Generation, at layer %s:' % node_name)
        current_op_type = node['op_type']
        
        #create the statistic atribute
        create_statistics_atributes(node)
        
        if current_op_type == 'Concat': # We need to update the tensors in amm structure so that inputs are no longer in AMM and output is. Also set it deallocating node
            if not DEBUG_SKIP_CBC_GENERATION:
                create_ic_dicts(ir,node)
            calc_concat_oc_processing_order(ir,node)
            num_of_grids = node['backend']['grid_count']
            grids_cbc = CBC_IR(num_of_grids) # In Concat op case the cbc_ir only contains amm_write_mask and only if one of its inputs need to be read from DDR
            node['backend']['grids_cbc'] = grids_cbc            

        elif OPTIMIZE_IDENTITY and ((current_op_type == "Identity") or ("fold_x" in node_name) or ("SPLIT" in node_name) or ("identity" in node_name) or (current_op_type == "Resize")):
            num_of_grids = node['backend']['grid_count']
            grids_cbc = CBC_IR(num_of_grids)
            grids_cbc = generate_grids_identity_cbc(ir, node, grids_cbc)
            num_of_grids = node['backend']['grid_count']            
            node['backend']['grids_cbc'] = grids_cbc
            node['backend']['nlf'] = grids_cbc.nlf
            if ir.debug:
                export_cbc_to_xls_alex(node,debug_output_dir+node_name+'_cbc',format=DebugFilesFormat.CSV) 

        elif OPTIMIZE_1x1_CONV and ((('kernel_shape' in node['attributes']) and (node['attributes']['kernel_shape'] == [1,1])) or (current_op_type == 'Add') or (current_op_type == "Resize") or (current_op_type == 'Identity')):
            current_op_weights_tensor,_ = internal_representation.get_node_weights_tensor(node)
            generated_wlocs = generate_grids_wloc_cbc(node, current_op_weights_tensor.data)
            num_of_grids = node['backend']['grid_count']
            grids_cbc = CBC_IR(num_of_grids) 
            grids_cbc = generate_grids_1x1_cbc(ir, node, grids_cbc, generated_wlocs)
            num_of_grids = node['backend']['grid_count']            
            node['backend']['grids_cbc'] = grids_cbc
            node['backend']['nlf'] = grids_cbc.nlf
            if ir.debug: 
                export_cbc_to_xls_alex(node,debug_output_dir+node_name+'_cbc',format=DebugFilesFormat.CSV) 
                #statistics report
                node['backend']['statistics']['wloc_conv_cmd_count'] = len(generated_wlocs[0].cmd_list)
                node['backend']['statistics']['nops_AMM_SYNC']       = 2*len(node['backend']['oc_groups'][0])

        elif current_op_type in GRID_OPS:
            current_op_output_tensor_name = node['outputs'][0]
            current_op_output_tensor_shape = ir.tensors[current_op_output_tensor_name].get_folded_shape()
            input_folding_factor_x = node['frontend']['input_folding_factor_x']
            input_folding_factor_y = node['frontend']['input_folding_factor_y']

            current_op_weights_tensor,_ = internal_representation.get_node_weights_tensor(node)

            if type(current_op_weights_tensor.data) == None: #Checking if conv has weights. If not, it might be a reordering node and we need to set its weights
                if 'reorder_node' in node:
                    if (input_folding_factor_x>0 or input_folding_factor_y>0)>0:
                        current_op_biases_tensor = node['frontend']['folded_biases_tensor']
                    else:
                        current_op_biases_tensor = node['frontend']['biases_tensor']

                       
                    update_reorder_node_wights_and_biases_tensors(node,current_op_weights_tensor,current_op_biases_tensor)
                else:
                    raise ValueError ('Conv node without weights. Node name: %s' % node_name)
            #ALex del this line for ph2     
            create_ic_dicts(ir,node)
            
            # Get actual total wloc size from last compile to better set per split wloc size limit (we want all splits to have same size so they hide DDR reads)
            # current_node_past_compile_wloc_sizes = None
            # current_node_past_balanced_wloc_entries_limit = None
            # if ir.past_wloc_data:
            #     if node_name in ir.past_wloc_data['past_compile_wloc_sizes_dict']:
            #         current_node_past_compile_wloc_sizes = ir.past_wloc_data['past_compile_wloc_sizes_dict'][node_name]
            #     if node_name in ir.past_wloc_data['past_balanced_wloc_entries_limit_dict']:
            #         current_node_past_balanced_wloc_entries_limit = ir.past_wloc_data['past_balanced_wloc_entries_limit_dict'][node_name]
            # node['backend']['past_compile_wloc_sizes'] = current_node_past_compile_wloc_sizes
            # node['backend']['balanced_wloc_entries_limit'] = current_node_past_balanced_wloc_entries_limit


            # Alex process pipe

            ################################################
            ########## this part is needed for comatability, it have to be removed by next cleaning ######################
        
            num_of_grids = 2
            grids_cbc = CBC_IR(num_of_grids)
            non_empty_output_channels = {}
            oc_groups = node['backend']['oc_groups']
            for current_oc in  oc_groups[0]:
                non_empty_output_channels[current_oc] = True

            total_minimal_oc_clocks_inserted_nops = 0
            total_first_wloc_entries = 0
            node['backend']['non_empty_output_channels'] = non_empty_output_channels
            node['backend']['total_minimal_oc_clocks_inserted_nops'] = total_minimal_oc_clocks_inserted_nops
            node['backend']['total_first_wloc_entries'] = total_first_wloc_entries

            ic_lookup_dicts = get_nodes_real_ic_lookup_dicts(node)
            per_ic_group_sorted_weight_activation_pairs = get_per_ic_group_sorted_weight_activation_pairs(ic_lookup_dicts)
            node['backend']['per_ic_group_sorted_weight_activation_pairs'] = per_ic_group_sorted_weight_activation_pairs
            ic_splits = node['backend']['ic_splits']
            oc_splits = node['backend']['oc_splits']
            per_grid_macs = [[[] for i in range(ic_splits)] for j in range(oc_splits)]
            node['backend']['per_grid_macs'] = per_grid_macs
            grids_cbc.per_oc_non_empty_ic_groups = {key: [0, 0] for key in non_empty_output_channels}
            ################################################

            generated_wlocs = generate_grids_wloc_cbc(node,current_op_weights_tensor.data)
            #update statistics
            node['backend']['statistics']['wloc_conv_cmd_count'] = len(generated_wlocs[0].cmd_list)
            node['backend']['statistics']['nops_AMM_SYNC']       = 2*len(node['backend']['oc_groups'][0]) #Nod_output * num_of_AMM_write_cmd
            
            max_mem_for_wloc_split = ((generated_wlocs[0].mem_output_depth*generated_wlocs[0].mem_output_width)//2 -
                                        3*2700) # this is the ~3 NOPS from EOC to Write
            # for tests
            #max_mem_for_wloc_split = (generated_wlocs[0].mem_output_depth*generated_wlocs[0].mem_output_width)//2//128
            split_nr = 0

            # max_mem_for_wloc_split = 1e40
            generated_wlocs[0].optimise_wloc_with_shorts()
            generated_wlocs[1].optimise_wloc_with_shorts()
            check_nop_balance(generated_wlocs)
            if ('optimal_offload_point_for_tensor' in node['backend']) or ('add_stall_nops' in node['backend'] and node['backend']['add_stall_nops'] == True): 
                add_stalls([generated_wlocs[0], generated_wlocs[1]], node)

                #update statistic for stalls
                node['backend']['statistics']['stall_nops']          = len(generated_wlocs[0].cmd_list) - node['backend']['statistics']['wloc_conv_cmd_count']     

            # create new tile in wloc
            grids_cbc.alex_wlocs.append([])
            
            # check the length and do the splits
            while (generated_wlocs[0].cmd_list and generated_wlocs[1].cmd_list):
                
                #Check the len             
                cmd_inx_split_0, mem_size_bit_0 = generated_wlocs[0].find_EOC_to_split(available_mem = max_mem_for_wloc_split) 
                cmd_inx_split_1, mem_size_bit_1 = generated_wlocs[1].find_EOC_to_split(available_mem = max_mem_for_wloc_split)
                cmd_inx_split = min (cmd_inx_split_0, cmd_inx_split_1)
                
                splited_wlocs = [generated_wlocs[0].cmd_list[:cmd_inx_split+1],generated_wlocs[1].cmd_list[:cmd_inx_split+1]]                 
        
                #rq params ets
                # num_of_AMM_write_cmd = 8 if (node['op_type'] == 'Resize') else 2 
                wlocs_part, rq_part, rt_part = generate_grids_rq_cbc_alex2(grids_wloc_commands=splited_wlocs, node = node)

                # cut the splited part from large  generated_wlocs
                generated_wlocs[0].cmd_list = generated_wlocs[0].cmd_list[cmd_inx_split+1:]
                generated_wlocs[1].cmd_list = generated_wlocs[1].cmd_list[cmd_inx_split+1:]

                # create mem of wloc (this will be done for each tile)
                wlocs_part[0][0].create_cmd_mem()
                wlocs_part[0][1].create_cmd_mem()                    
                grids_cbc.alex_wlocs[0].append(wlocs_part[0])
                
                # create mem of rq_param and rt_part  (this will be done ones for whole tile)
                rq_part[   0][0].create_cmd_mem()
                rt_part[      0].create_cmd_mem()

                grids_cbc.alex_rqParam.append( rq_part[0]   )
                grids_cbc.RTable.append (      rt_part[0]   )

                # statistic for the w_loc_len                        
                node['backend']['statistics']['wloc_all_splits'] +=(len(wlocs_part[0][0].cmd_list))

                split_nr +=1

            if ('_fold_x_0' in node_name and (len(node['frontend']['preceding_nodes_params']) == 0)):
                if ir.uint8_int8_lut is not None:
                    grids_cbc.nlf = [NonLinearFunctionList(ir.uint8_int8_lut)]
                elif ir.uint8_int8_conversion:
                    lut = [(i - 128) for i in range(256)]
                    ir.uint8_int8_lut = lut
                    grids_cbc.nlf = [NonLinearFunctionList(lut)]
                else:
                    grids_cbc.nlf = [LinearFunctionList()]
            elif ('lut_silu' in node['frontend']):
                grids_cbc.nlf = [NonLinearFunctionList(node['frontend']['lut_silu'])]
            else:
                grids_cbc.nlf = [LinearFunctionList()]

            grids_cbc.nlf[0].create_cmd_mem()
            
            ###########################################################            
            node['backend']['grids_cbc'] = grids_cbc
            node['backend']['nlf'] = grids_cbc.nlf

            #set statistic             
            node['backend']['statistics']['nops_end_of_split']   = split_nr*HARDWARE_SYNC_CMD_COMPLATE
            if ir.debug and DEBUG_CREATE_PER_NODE_DEBUG_FILES and not DEBUG_SKIP_CBC_GENERATION:
                if DEBUG_SPREADSHEET_FORMAT_XLSX:
                    export_cbc_to_xls(node,debug_output_dir+node_name+'_cbc')
                else:
                    # Alex for Ph2 replase
                    #export_cbc_to_xls(node,debug_output_dir+node_name+'_cbc',format=DebugFilesFormat.CSV)
                    export_cbc_to_xls_alex(node,debug_output_dir+node_name+'_cbc',format=DebugFilesFormat.CSV) 

                if DEBUG_CREATE_HEX_FILES: 
                    #prepare_wloc_ir(node)
                    #alex for phase 2 remove this
                    #write_wloc_hex_files(node,debug_output_dir+node_name+'_wloc')
                    #add this
                    #write_wloc_hex_files_alex(node,debug_output_dir+node_name+node_name+'_wloc_alex')

                    #prepare_rqloc_ir(node)
                    #write_rqloc_hex_file(node,debug_output_dir+node_name+'_rqloc')
                    #prepare_rqparams_ir(node)
                    #write_rqparams_hex_file(node,debug_output_dir+node_name+'_rqparams.hex')
                    pass
        
        ####################check if there are AMM+STALL together########################
        if  current_op_type in GRID_OPS:   
            for split_wloc in node['backend']['grids_cbc'].alex_wlocs[0]: # For each split wloc
                items = split_wloc[0].cmd_list
                for inx_check, cmd in  enumerate(items):
                    if "AMM" in cmd.nop_reason:
                        if items[inx_check+1].long_entry==False and items[inx_check+1].weight_value==0 and items[inx_check+1].weight_offset==0:
                            raise ValueError('Found AMM and STALL NOPs together in node %s' % node_name)
                        if items[inx_check+2].long_entry==False and items[inx_check+2].weight_value==0 and items[inx_check+2].weight_offset==0:
                            raise ValueError('Found AMM and STALL NOPs together in node %s' % node_name)
                        if items[inx_check-1].long_entry==False and items[inx_check-1].weight_value==0 and items[inx_check-1].weight_offset==0:
                            raise ValueError('Found AMM and STALL NOPs together in node %s' % node_name)
                        if items[inx_check-2].long_entry==False and items[inx_check-2].weight_value==0 and items[inx_check-2].weight_offset==0:
                            raise ValueError('Found AMM and STALL NOPs together in node %s' % node_name)
                                
  
        #     for i in range(len(items) - 1):
        #     # Check if aa == 0 in two neighboring items
        #     if items[i]["aa"] == 0 and items[i + 1]["aa"] == 0:
        #     # Check if neither has bb == 0
        #     if items[i]["bb"] != 0 and items[i + 1]["bb"] != 0:
        #         return True
        # return False
    ############################################    
            
    return ir

def prepare_intermediate_tensors_write(ir: internal_representation.IR) -> internal_representation.IR:
# During program_compiler.compile compiler pass (where AMM mem is allocated) we found places where we must offload intermediate tensors
# to DDR to free AMM mem.
# The below compiler pass, marks optimal places for writing these tensors to DDR and make sure the tensor is written only after its producing
# op has finished execution Optimal writing point would be in parallel to execution of the node which follows(by execution order) the
# intermediate tensor producer node

    for intermediate_tensor_name,intermediate_tensor_ir in ir.intermediate_ddr_tensors.items():
        offloading_node = intermediate_tensor_ir.offloading_node_name
        intermediate_tensor = ir.tensors[intermediate_tensor_name]
        intermediate_tensor_producer_name = intermediate_tensor.producer
        producer_node = ir.graph.nodes()[intermediate_tensor_producer_name]
        if 'tensor_to_offload_name' in producer_node['backend']:
            raise ValueError('Didnt expect node to already have this attribute. Please check!')
        else:
            producer_node['backend']['tensor_to_offload_name'] = intermediate_tensor_name
        producer_index = ir.lexicographical_topological_sorted_graph.index(intermediate_tensor_producer_name)
        write_node_index = producer_index+1
        write_node_name = ir.lexicographical_topological_sorted_graph[write_node_index]
        write_node = ir.graph.nodes()[write_node_name]
        if 'optimal_offload_point_for_tensor' in write_node['backend']:
            raise ValueError('Didnt expect node to already have optimal_offload_point_for_tensor. Please check!')
        else:
            write_node['backend']['optimal_offload_point_for_tensor'] = intermediate_tensor_name
    return ir

def compile_program(ir: internal_representation.IR) -> internal_representation.IR:
    sequencer_program = ir.sequencer_program.commands_list
    wait_flags_allocator = ir.wait_flags_allocator
    single_split_tables_buffer_allocator = ir.single_split_tables_buffer_allocator
    splitted_tables_buffer_allocator = ir.splitted_tables_buffer_allocator
    #TODO: Dans Must make sure the below loop is going through nodes by execution order
    last_single_split_tables_buffer_id = None
    # Allocate all flags
    axi0_tables_load_flag = wait_flags_allocator.allocate_flag(description='generic axi0 set/wait flag.')
    axi1_tables_load_flag = wait_flags_allocator.allocate_flag(description='generic axi1 set/wait flag.')
    local_ddr_transaction_flag = wait_flags_allocator.allocate_flag(description='generic local ddr transaction (read/write) set/wait flag.')
    global_ddr_transaction_flag = wait_flags_allocator.allocate_flag(description='generic global ddr transaction (read/write) set/wait flag.')
    #engine_command_set_flag = wait_flags_allocator.allocate_flag(description='generic engine command set/waitflag.')
    ir.axi0_tables_load_flag = axi0_tables_load_flag
    ir.axi1_tables_load_flag = axi1_tables_load_flag
    ir.local_ddr_transaction_flag = local_ddr_transaction_flag
    ir.global_ddr_transaction_flag = global_ddr_transaction_flag

    tiling_blobs=ir.tiling_blobs
    tile_read_in_process = False
    tile_write_in_process = False
    conv_in_process = False
    blobs_tqdm_iterator = tqdm(tiling_blobs.values())

    # Initial DDR commands before generating for the rest of the program
    layers_sequence = generate_initial_command_sequence(ir)
    sequencer_program.extend(layers_sequence)

    # Keep track of the number of consecutive 2 tile blobs
    consecutive_2_tile_blobs = 0

    # Each tiling blob will be updated to store this
    # Then at the end of the loop this will be updated based on the node
    prev_blob_lines_in_ddr = -1 # -1 means it's the input

    for blob_idx,tiling_blob in enumerate(blobs_tqdm_iterator):

        if (blob_idx in DEBUG_SKIP_BLOB_LIST):
            continue

        if tiling_blob.y_tiles == 2:
            consecutive_2_tile_blobs += 1
        else:
            consecutive_2_tile_blobs = 0

        last_blob = (blob_idx == (len(tiling_blobs)-1))
        blobs_tqdm_iterator.set_description('Sequencer code generation, at blob#%d:' % blob_idx)
        tiles_tqdm_iterator = tqdm(range(tiling_blob.y_tiles))
        for tile_idx,current_y_tile in enumerate(tiles_tqdm_iterator):
            last_tile = (tile_idx == (tiling_blob.y_tiles-1))
            tiles_tqdm_iterator.set_description('Tile#%d:' % current_y_tile)
            nodes_tqdm_iterator = tqdm(tiling_blob.nodes_in_blob)
            for node_idx,current_node_name in enumerate(nodes_tqdm_iterator):
                last_node = (node_idx == (len(tiling_blob.nodes_in_blob)-1))
                last_tile_of_last_node = (last_blob and last_tile and last_node)
                nodes_tqdm_iterator.set_description('At node:%s:' % current_node_name)
                node = ir.graph.nodes()[current_node_name]
                if (node['frontend']['run_op']):
                    current_op_type = node['op_type']
                    if current_op_type == 'Concat':
                        layers_sequence,tile_read_in_process,tile_write_in_process = generate_layer_command_sequence(ir,node, wait_flags_allocator,0,0,
                                                                            current_tile_num=current_y_tile,tile_read_in_process=tile_read_in_process,tile_write_in_process=tile_write_in_process,
                                                                            consecutive_2_tile_blobs=consecutive_2_tile_blobs,
                                                                            prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
                                                                            conv_in_process=conv_in_process) # Concat generates commands only if one of its inputs needs to be read from DDR
                        sequencer_program.extend(layers_sequence)
                    elif current_op_type == 'Sync':
                        pass
                    elif current_op_type in GRID_OPS:
                        single_split_tables_buffer_id = single_split_tables_buffer_allocator.allocate_buffer()
                        if not last_single_split_tables_buffer_id==None:
                            single_split_tables_buffer_allocator.deallocate_buffer(last_single_split_tables_buffer_id)
                        last_single_split_tables_buffer_id = single_split_tables_buffer_id
                        layer_sequence, tile_read_in_process, tile_write_in_process = generate_layer_command_sequence(ir,node, wait_flags_allocator,single_split_tables_buffer_id,splitted_tables_buffer_allocator,
                                                                                current_tile_num=current_y_tile,tile_read_in_process=tile_read_in_process,
                                                                                consecutive_2_tile_blobs=consecutive_2_tile_blobs,
                                                                                prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
                                                                                tile_write_in_process=tile_write_in_process,conv_in_process=conv_in_process) # Add current layer's sequencer commands to program list
                        sequencer_program.extend(layer_sequence)
                        output_tensor_name = node['frontend']['output_tensor'].name

        # Before moving to the next blob, store the number of lines the previous blob will write to DDR
        # when the next blob will start reading
        prev_blob_lines_in_ddr = tiling_blob.num_lines_written_to_ddr_before_next_blob_read
        # Not set for concat
        #assert prev_blob_lines_in_ddr > 0

    if DEBUG_PUSH_TABLES_READ_BEFORE_DDR_RW:
        ir.sequencer_program.push_tables_read_before_ddr_rw()
    wait_flags_allocator.deallocate_flag(ir.axi0_tables_load_flag)
    wait_flags_allocator.deallocate_flag(ir.axi1_tables_load_flag)
    wait_flags_allocator.deallocate_flag(ir.local_ddr_transaction_flag)
    wait_flags_allocator.deallocate_flag(ir.global_ddr_transaction_flag)
    return ir

def get_last_compile_per_layer_wloc_size(ir,debug_output_dir):
    last_compile_per_layer_wloc_size_filename = debug_output_dir+'past_wloc_data.pickle'
    if os.path.exists(last_compile_per_layer_wloc_size_filename):
        with open(last_compile_per_layer_wloc_size_filename, 'rb') as f:
            past_wloc_data=pickle.load(f)
        ir.past_wloc_data = past_wloc_data
    return ir

def save_per_layer_wloc_size(ir,debug_output_dir):
    past_balanced_wloc_entries_limit_dict={}
    past_wloc_sizes_dict = {}
    last_compile_per_layer_wloc_size_filename = debug_output_dir+'past_wloc_data.pickle'
    for node_name in ir.lexicographical_topological_sorted_graph:
        current_node = ir.graph.nodes[node_name]
        if 'per_split_wloc_size' in current_node['backend']:
            past_wloc_sizes_dict[node_name] = current_node['backend']['per_split_wloc_size']
        if 'balanced_wloc_entries_limit' in current_node['backend']:
            past_balanced_wloc_entries_limit_dict[node_name] = current_node['backend']['balanced_wloc_entries_limit']
    with open(last_compile_per_layer_wloc_size_filename, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        past_wloc_sizes_data={}
        past_wloc_sizes_data['past_compile_wloc_sizes_dict'] = past_wloc_sizes_dict
        past_wloc_sizes_data['past_balanced_wloc_entries_limit_dict'] = past_balanced_wloc_entries_limit_dict
        pickle.dump(past_wloc_sizes_data, f, pickle.HIGHEST_PROTOCOL)

def get_best_node_index_to_split_blob(ir:internal_representation.IR,sorted_nodes,node_index,nodes_in_blob):
    # The below will minimize the number of outputs from the blob
    nodes_in_blob_copy = copy.deepcopy(nodes_in_blob)
    min_blob_outputs=100
    next_k3_found = False
    best_idx=0
    current_node_idx=node_index
    while not next_k3_found:
        blob_inputs,blob_outputs = get_blob_inputs_and_outputs(ir,nodes_in_blob_copy)
        if len(blob_outputs)<=min_blob_outputs:
            min_blob_outputs = len(blob_outputs)
            best_idx = current_node_idx
        current_node_idx+=1
        if current_node_idx>=len(sorted_nodes):
            next_k3_found = True
            break
        next_node_name = sorted_nodes[current_node_idx]
        next_node = ir.graph.nodes()[next_node_name]
        nodes_in_blob_copy.append(next_node_name)
        next_k3_found = ir.is_k3_node(next_node)
    return best_idx

def _is_following_of_type(ir, node, search_type) -> bool:
    following_nodes_params = node['frontend']['following_nodes_params']
    is_following_of_type = False
    for following_node_params in following_nodes_params:
        following_node_name = following_node_params[0]
        following_node = ir.graph.nodes()[following_node_name]
        if following_node['op_type']==search_type:
            is_following_of_type = True
            break
    return is_following_of_type

def node_input_is_from_mxp(ir, node_name) -> bool:
    node = ir.graph.nodes()[node_name]

    # For now, handle case of 1 input tensor
    if 'input_tensor' in node['frontend']:
        producer_node_name = node['frontend']['input_tensor'].producer
        if producer_node_name:
            producer_node = ir.graph.nodes()[producer_node_name]
            if producer_node['op_type'] == "Sync":
                return True
    # If multiple input tensors and one input is Sync, fail for now
    else:
        assert 'input_tensors' in node['frontend']
        for input_tensor in node['frontend']['input_tensors']:
            if input_tensor.producer != None:
                producer_node = ir.graph.nodes()[input_tensor.producer]
                if producer_node['op_type'] == "Sync":
                    assert False, "Handle case of Sync input to multi-input node"
    return False

def should_start_new_blob(
    ir: internal_representation.IR,
    node_name: str,
    current_blob: internal_representation.TilingBlob,
) -> bool:

    start_new_blob = False
    node = ir.graph.nodes()[node_name]

    # When a node is split due to having too many output channels, for now
    # make a new blob between the two.
    if node_name.endswith('_split'):
        return True

    # Automatically start a new blob if this input requires sync with MXP
    if node['op_type'] == 'Sync':
        return True
    if node_input_is_from_mxp(ir, node_name):
        return True

    if DEBUG_TRY_TO_FIX_AUTO_SPLIT:

        # Start new blob at Concat
        if node['op_type']=='Concat':

            start_new_blob = True

            # In some cases, we're already splitting for this Concat so don't split again
            DONT_SPLIT_IF_OUTPUTS_FOLDED = True
            DONT_SPLIT_IF_INPUTS_FOLDED = True
            DONT_SPLIT_IF_INPUT_STARTED_THIS_BLOB = True

            if DONT_SPLIT_IF_OUTPUTS_FOLDED:
                # If outputs of the Concat are folded, then don't split here because a split will be needed
                # right after the concat anyway
                has_folded_output = False
                for consumer in node['frontend']['output_tensor'].consumers:
                    consumer_node = ir.graph.nodes()[consumer]
                    if 'force_folding_y' in consumer_node['frontend'] or 'force_unfolding_y' in consumer_node['frontend']:
                        has_folded_output=True
                        break
                if has_folded_output:
                    start_new_blob = False

            if DONT_SPLIT_IF_INPUTS_FOLDED:
                # If an input of the Concat from same blob is folded, then don't split here because we just split
                # STEFAN TODO: Maybe it's better to check all inputs instead of just find one
                has_folded_input_from_same_blob = False
                for input_tensor in node['frontend']['input_tensors']:
                    producer = input_tensor.producer
                    # Only look at nodes in the current blob
                    if producer not in current_blob.nodes_in_blob:
                        continue
                    # This producer is in the current blob
                    producer_node = ir.graph.nodes()[producer]
                    if 'force_folding_y' in producer_node['frontend'] or 'force_unfolding_y' in producer_node['frontend']:
                        has_folded_input_from_same_blob = True
                        break
                if has_folded_input_from_same_blob:
                    start_new_blob = False

            if DONT_SPLIT_IF_INPUT_STARTED_THIS_BLOB:
                # If an input of the Concat from the current blob is the only node in the current blob,
                # then don't split here because we just split. Consider the following example:
                #          ...     ...
                #           │       │
                #        ┌──▼──┐    │
                #        │Conv1│    │
                #        └┬──┬─┘    │
                #         │  │      │
                #         │┌─▼───┐  │
                #         ││Conv3│  │
                #         │└─┬───┘  │
                #       ┌─┘  └────┐ │
                #       │       ┌─▼─▼───┐
                #       │       │Concat4│
                #    ┌──▼──┐    └──┬────┘
                #    │Conv2│       │
                #    └─────┘       ▼
                #    (OUTPUT)     ...
                #
                # Currently, a final output such as Conv2 is always the last node in a blob, because it has
                # an ordering Conv. Since the lexicographical sort places Conv3 after Conv2, this means
                # that Conv3 will start a new blob. If Concat also starts a new blob, then Conv3 will
                # be in a blob on its own. One solution (below) is keep Concat4 in the same blob.

                # STEFAN TODO: Maybe it's better to check all inputs instead of just find one
                has_input_that_started_this_blob = False
                for input_tensor in node['frontend']['input_tensors']:
                    producer = input_tensor.producer
                    # Only look at nodes in the current blob
                    if producer not in current_blob.nodes_in_blob:
                        continue
                    # This producer is in the current blob
                    # Check if it is the only node in the current blob.
                    if len(current_blob.nodes_in_blob) == 1:
                        has_input_that_started_this_blob = True
                        break
                if has_input_that_started_this_blob:
                    start_new_blob = False

        # Folding and unfolding is done by DDR so need to start a new blob
        if 'force_folding_y' in node['frontend'] or 'force_unfolding_y' in node['frontend']:
            start_new_blob=True

        # Sometimes requant are from very different parts of the graph, so avoid error
        # "2 nodes in same blob cant have different number of tiles."
        if 'requantnode' in node_name:
            # Don't need to start a new blob if input is in the current blob already
            input_tensor = node['frontend']['input_tensor']
            if input_tensor.producer not in current_blob.nodes_in_blob:
                start_new_blob=True

    if DEBUG_MINIMIZE_Y_FOLDING:
        # If stride = 2, split so that there is not a blob with mismatched # tiles
        if node['op_type']=='Conv' and node['frontend']['stride'] == 2:
            start_new_blob=True
        # Similarly if next = Resize, split so there is not a mismatched # tiles
        # Split before the resize so the smaller amount is DMA
        #if _is_following_of_type(ir, node, 'Resize'):
        #    start_new_blob=True
        if node['op_type']=='Resize':
            start_new_blob=True

    if (node['op_type']=='Conv') and (current_blob != None) and not start_new_blob:
        producer = node['frontend']['input_tensor'].producer
        if producer not in current_blob.nodes_in_blob:
            first_node_in_current_blob_name = current_blob.nodes_in_blob[0]
            first_node_in_current_blob = ir.graph.nodes()[first_node_in_current_blob_name]
            if first_node_in_current_blob['op_type'] != 'Concat':
                if (node['frontend']['stride'] == 1) and (first_node_in_current_blob['frontend']['stride'] == 2):
                    start_new_blob=True
            else:
                start_new_blob=True
            
    if (node['op_type']=='Conv') and (current_blob != None) and not start_new_blob:
        consumers = node['frontend']['output_tensor'].consumers
        for idx in range(len(consumers)):
            if ('ADD' in consumers[idx]):
                add_node = ir.graph.nodes[consumers[idx]]
                for input_tensor in add_node['frontend']['input_tensors']:
                    if (input_tensor.producer != None) and (('SPLIT' in input_tensor.producer) or ('STRIDEDSLICE' in input_tensor.producer)):
                        for c_idx in range(len(input_tensor.consumers)):
                            if 'CONCATENATION' in input_tensor.consumers[c_idx]:
                                start_new_blob = True
                                break
    
    # if current_blob != None:
    #     if node['frontend']['y_tiles'] != current_blob.y_tiles:
    #         start_new_blob = True

    return start_new_blob
        

# Given a blob which was identified as causing an AMM allocation failure,
# select a node in the blob to split at.
def get_node_to_split_in_failing_blob(ir, failing_blob_idx):
    # Check '== None' here because the index can be 0
    if failing_blob_idx == None:
        return None, False
    # For now, return the half-way point.
    # This can be adjusted as more cases are discovered, e.g.,
    # can use get_best_node_index_to_split_blob instead.
    nodes_in_failing_blob = ir.tiling_blobs[failing_blob_idx].nodes_in_blob
    num_nodes_in_failing_blob = len(nodes_in_failing_blob)
    if num_nodes_in_failing_blob < 2:
        print(f"Blob {failing_blob_idx}:")
        print(nodes_in_failing_blob)
        node_name = nodes_in_failing_blob[0]
        node_name = node_name.split('_split')[0]
        if (node_name not in ir.marked_nodes_for_output_split):
            node = ir.graph.nodes[node_name]
            if node['op_type'] == 'Conv':
                if 'folded_weights_tensor' in node['frontend']:
                    del node['frontend']['folded_weights_tensor']
                split_conv_out_channels_in_two(ir,node_name,node)
                ir.marked_nodes_for_output_split.append(node_name)
                ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
                calc_conv_qparams(ir,node_name,node)
                node_split_name = node_name + '_split'
                node_split = ir.graph.nodes[node_split_name]
                calc_conv_qparams(ir,node_split_name,node_split)
                return node_name, True
            elif node['op_type'] == 'Resize':
                split_conv_out_channels_in_two(ir,node_name,node)
                ir.marked_nodes_for_output_split.append(node_name)
                ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
                node_split_name = node_name + '_split'
                node_split = ir.graph.nodes[node_split_name]
                return node_name, True
        raise ValueError("Cannot further split blob")
    
    # Split the blobs at the first concatenation node or in the middle of the blob
    for idx in range(1,num_nodes_in_failing_blob):
        node_info = ir.graph.nodes()[nodes_in_failing_blob[idx]]
        if (node_info['op_type']) == 'Concat':
            break
    if (idx < num_nodes_in_failing_blob-1):
        split_location = idx
    else:
        split_location = (num_nodes_in_failing_blob + 1) //2

    # If splitting at the half-way point would result in a Concat-only blob,
    # try to split one node later
    #if ir.graph.nodes()[nodes_in_failing_blob[split_location]]['op_type'] == 'Concat':
    #    split_location -= 1

    return nodes_in_failing_blob[split_location], False

def set_tiling_blobs(ir: internal_representation.IR, failing_blob_idx = None) -> internal_representation.IR:

    # See if there was a previous failing blob, and if so, determine
    # which node to split the blob at.
    node_to_split_in_failing_blob, needed_node_output_split = get_node_to_split_in_failing_blob(ir, failing_blob_idx)
    if needed_node_output_split:
        ir.nodes_to_split_failing_blobs = set()
        return ir, needed_node_output_split
    if node_to_split_in_failing_blob not in ir.nodes_to_split_failing_blobs:
        ir.nodes_to_split_failing_blobs.add(node_to_split_in_failing_blob)
    
    sorted_nodes = ir.lexicographical_topological_sorted_graph
    blob_idx=0
    first_node_in_blob = True
    k3_nodes_folded = 0
    current_blob = None
    print("Starting to split blobs")
    previous_y_tiles = 0
    for node_index in range(len(sorted_nodes)):
        node_name=sorted_nodes[node_index]
        start_new_blob=False
        if node_index in DEBUG_PER_NODE_BLOB or (node_name in ir.marked_nodes_for_folding_y \
            or node_name in ir.force_y_folding or node_name in ir.force_y_unfolding):
            start_new_blob=True

        # Check additional rules for starting new blob
        if should_start_new_blob(ir, node_name, current_blob):
            start_new_blob=True

        # If this was a previously failing blob, check if this is the node to split at
        if node_name in ir.nodes_to_split_failing_blobs:
            start_new_blob=True

        node = ir.graph.nodes()[node_name]
        y_tiles = node['frontend']['y_tiles']
        if y_tiles != previous_y_tiles:
            start_new_blob=True

        if start_new_blob:
            if not first_node_in_blob: # If its already a first node in blob no need to advance blob idx
                blob_idx+=1
                first_node_in_blob = True
        
        x_slices = node['frontend']['x_slices']
        
        y_size_folded = node['frontend']['output_tensor'].shape[2] / pow(2,node['frontend']['output_folding_factor_y'])
        #Calculate the maximum number of K3 nodes that would still not increase the overall number of tiles in this BLOB
        if blob_idx == 0 and LARGE_FIRST_BLOB:
            max_k3_nodes = MAX_FOLDED_K3_NODES_IN_BLOB # Allow first blob to be larger
        elif y_size_folded <= SNP_GRID_HEIGHT:
            max_k3_nodes = 10 #This is a single tile and we have no real K3 limit
        elif y_size_folded <= (SNP_GRID_HEIGHT-1)*2:
            max_k3_nodes = ((SNP_GRID_HEIGHT-1)*2-y_size_folded) // 2 + 1
        else:
            #min_tiles is the minimal number of tiles above 2 if we only run a single K3 node
            min_tiles = math.ceil((y_size_folded - (SNP_GRID_HEIGHT-1)*2) / (SNP_GRID_HEIGHT-2))
            extra_k3 = min_tiles * (SNP_GRID_HEIGHT-2) + (SNP_GRID_HEIGHT-1)*2 - y_size_folded
            max_k3_nodes = extra_k3 // (2*min_tiles+2) + 1
        # Temp improvement
        # if y_size in [36, 72]:
        #    max_k3_nodes = 1
        # else:
        #    max_k3_nodes = MAX_FOLDED_K3_NODES_IN_BLOB

        is_k3_node = ir.is_k3_node(node)
        input_folding_y = 2**node['frontend']['input_folding_factor_y']
        kernel_size = 1
        if is_k3_node:
            kernel_size = node['frontend']['kernel_size']
        kernel_field_reduction = kernel_size // 2
        if is_k3_node and not first_node_in_blob and (k3_nodes_folded+1/input_folding_y) > max_k3_nodes \
            and not (y_tiles == 1 and DEBUG_NO_K3_LIMIT_FOR_1T):
            blob_idx+=1
            first_node_in_blob=True
        node['frontend']['tiling_blob_idx'] = blob_idx
        if first_node_in_blob:
            k3_nodes_folded = is_k3_node/input_folding_y * kernel_field_reduction
            current_blob = internal_representation.TilingBlob(nodes_in_blob=[node_name],num_y_tiles=y_tiles,num_x_slices=x_slices,k3_nodes=int(is_k3_node))
            ir.tiling_blobs[blob_idx] = current_blob
        else:
            current_blob.nodes_in_blob.append(node_name)
            current_blob.num_of_nodes_in_blob+=1
            if is_k3_node:
                k3_nodes_folded += 1/input_folding_y * kernel_field_reduction
                current_blob.k3_nodes = math.ceil(k3_nodes_folded)
        '''
        first_node_in_blob = False
        if DEBUG_AUTO_BLOB_SPLITTING:
            if (k3_nodes_folded+1/input_folding_y)>MAX_FOLDED_K3_NODES_IN_BLOB:
                # If this is a 1-tile blob, we may not want to use k3 nodes as a deciding factor for blob splitting.
                if not (y_tiles == 1 and DEBUG_NO_K3_LIMIT_FOR_1T):
                    if node_index == get_best_node_index_to_split_blob(ir,sorted_nodes,node_index,current_blob.nodes_in_blob):
                        blob_idx+=1 # We set each blob to have single node
                        first_node_in_blob = True
            if node['op_type']=='Add': # Because of mem alloc issue we need to fix we dont allow concat after add in same blob
                if _is_following_of_type(ir, node, 'Concat'):

                    # STEFAN TODO: This assumes that the next node is the Concat, but sometimes there is a parallel
                    # path to the Add with a single Conv as the other input to the Concat. When the lexicographical
                    # sort puts that single Conv between the Add and the Concat, it will become a single-node blob
                    # (if the Concat also starts a new blob):
                    #   ...     ...
                    #   │  │     │
                    #  ┌▼──▼┐ ┌──▼──┐
                    #  │Add1│ │Conv2│   Add1    - part of Blob N
                    #  └──┬─┘ └─┬───┘   Conv2   - part of Blob N+1
                    #     │     │       Concat3 - part of Blob N+2
                    #    ┌▼─────▼┐
                    #    │Concat3│
                    #    └───┬───┘
                    #        ▼
                    # Possible solutions:
                    # - add logic here to check if the input to Conv2 is generated by Blob N, and if so, include
                    #   Conv2 into Blob N before incrementing blob_idx
                    #     - resulting blobs: [..., Add1, Conv2], [Concat3]
                    # - remove this logic here so that Add1 and Conv2 stay in the same blob, and instead add logic
                    #   for the Concat to always make a new Blob if an input from the current Blob is an Add
                    #     - resulting blobs: [..., Add1, Conv2], [Concat3]
                    # - keep this logic here the same but add logic for the Concat node to not create a new Blob N+2
                    #   and instead keep it in the Blob N+1 with Conv2
                    #     - resulting blobs: [..., Add1], [Conv2, Concat3]
                    #
                    # Can experiment to see which split is better. For now going to implement #3 in should_start_new_blob.

                    blob_idx+=1 # We set each blob to have single node
                    first_node_in_blob = True
        '''

        # Up to this point is where we forced a new BLOB before the node.
        # Now we check if we need to force a new blob after the node
        if (k3_nodes_folded+1/input_folding_y)>max_k3_nodes and not first_node_in_blob \
              and not (y_tiles == 1 and DEBUG_NO_K3_LIMIT_FOR_1T):
                if node_index == get_best_node_index_to_split_blob(ir,sorted_nodes,node_index,current_blob.nodes_in_blob):
                    blob_idx+=1 # We set each blob to have single node
                    first_node_in_blob = True
                else:
                    first_node_in_blob = False
        elif node['op_type']=='Add' and _is_following_of_type(ir, node, 'Concat'): 
            blob_idx+=1 # We set each blob to have single node
            first_node_in_blob = True
            # Because of mem alloc issue we need to fix we dont allow concat after add in same blob
        
        
            # STEFAN TODO: This assumes that the next node is the Concat, but sometimes there is a parallel
            # path to the Add with a single Conv as the other input to the Concat. When the lexicographical
            # sort puts that single Conv between the Add and the Concat, it will become a single-node blob
            # (if the Concat also starts a new blob):
            #   ...     ...
            #   │  │     │
            #  ┌▼──▼┐ ┌──▼──┐
            #  │Add1│ │Conv2│   Add1    - part of Blob N
            #  └──┬─┘ └─┬───┘   Conv2   - part of Blob N+1
            #     │     │       Concat3 - part of Blob N+2
            #    ┌▼─────▼┐
            #    │Concat3│
            #    └───┬───┘
            #        ▼
            # Possible solutions:
            # - add logic here to check if the input to Conv2 is generated by Blob N, and if so, include
            #   Conv2 into Blob N before incrementing blob_idx
            #     - resulting blobs: [..., Add1, Conv2], [Concat3]
            # - remove this logic here so that Add1 and Conv2 stay in the same blob, and instead add logic
            #   for the Concat to always make a new Blob if an input from the current Blob is an Add
            #     - resulting blobs: [..., Add1, Conv2], [Concat3]
            # - keep this logic here the same but add logic for the Concat node to not create a new Blob N+2
            #   and instead keep it in the Blob N+1 with Conv2
            #     - resulting blobs: [..., Add1], [Conv2, Concat3]
            #
            # Can experiment to see which split is better. For now going to implement #3 in should_start_new_blob.
        else:
            first_node_in_blob = False

        '''
        if node_name in last_node_in_blob_names:
            if node_index!=len(sorted_nodes)-1: # We always add ordering node to the last blob
                next_executed_node_name = sorted_nodes[node_index+1]
                next_executed_node = ir.graph.nodes()[next_executed_node_name]
                if 'reorder_node' in next_executed_node:
                   first_node_in_blob = False
                else:
                    blob_idx+=1 # For now, blob split is pre-set. Next stage we split blobs when res-change etc.
                    first_node_in_blob = True
            else:
                blob_idx+=1 # For now, blob split is pre-set. Next stage we split blobs when res-change etc.
                first_node_in_blob = True


        if 'reorder_node' in node and not first_node_in_blob: # Ordering node will always be last in blob
            blob_idx+=1 # For now, blob split is pre-set. Next stage we split blobs when res-change etc.
            first_node_in_blob = True

        '''
        previous_y_tiles = y_tiles

    return ir, needed_node_output_split
def update_node_tiling_info(ir,node):
    node_name = node['name']
    y_tiles,output_padding_start_y = get_node_tiling_info(ir,node)
    node['frontend']['y_tiles'] = y_tiles # We need to update number of tiles after blob boundries were set since the number of k=3 convs in each blob changes size of each tile and hence the number of tiles
    if y_tiles>1:
        node['backend']['add_stall_nops'] = True
        if DEBUG_AVOID_DDR_WRITE_WHILE_CONV and DEBUG_AVOID_DDR_WRITE_WHILE_CONV:
            node['backend']['add_stall_nops'] = False
    if node['op_type'] in MULTIPLE_INPUT_OPS:
        for input_tensor in node['frontend']['input_tensors']:
            input_tensor.y_tiles = y_tiles
    else:
        node['frontend']['input_tensor'].y_tiles = y_tiles
    node['frontend']['output_tensor'].y_tiles = y_tiles
    current_blob_idx = node['frontend']['tiling_blob_idx']
    current_blob = ir.tiling_blobs[current_blob_idx]
    if current_blob.nodes_in_blob.index(node_name)>0 and y_tiles!=current_blob.y_tiles:

        # Print out all nodes in thie blob and # y tiles
        print("-----------------------------------------")
        print("DEBUG: Nodes in this blob and their tiles")
        for n in current_blob.nodes_in_blob:
            print(n, end="\t")
            print(ir.graph.nodes()[n]['frontend']['y_tiles'])

        # Print out all nodes and their input and output folding
        print("-----------------------------------------")
        print("DEBUG: All nodes and their input/output Y folding")
        for n in ir.graph.nodes:
            print(ir.graph.nodes[n]['name'], end="\t")
            if 'input_folding_factor_y' in ir.graph.nodes[n]['frontend'].keys():
                print(ir.graph.nodes[n]['frontend']['input_folding_factor_y'], end="\t")
            else:
                print("n/a", end="\t")
            if 'output_folding_factor_y' in ir.graph.nodes[n]['frontend'].keys():
                print(ir.graph.nodes[n]['frontend']['output_folding_factor_y'])
            else:
                print("n/a")
        print("-----------------------------------------")

        raise ValueError ('2 nodes in same blob cant have different number of tiles. This can be caused by wrong blob splitting. Please check...')
    current_blob.y_tiles = y_tiles
    node['backend']['output_padding_start_y'] = output_padding_start_y

def set_node_tiling_info(ir: internal_representation.IR) -> internal_representation.IR:
    sorted_nodes = ir.lexicographical_topological_sorted_graph
    
    # Specifying the start node from hw_config
    start_node_idx = 0
    if NODES_LIST['Start']:
        start_node_idx = sorted_nodes.index(NODES_LIST['Start'][0])
        if (start_node_idx != 0):
            node = ir.graph.nodes()[sorted_nodes[start_node_idx]]
            ir.tensors_from_mxp = set()
            ir.tensors_from_mxp.add(node['frontend']['input_tensor'].name)
    # Specifying the end node from hw_config
    end_node_idx = len(sorted_nodes)-1
    if NODES_LIST['End']:
        end_node_idx = sorted_nodes.index(NODES_LIST['End'][0])
        if (end_node_idx != len(sorted_nodes)-1):
            node = ir.graph.nodes()[sorted_nodes[end_node_idx]]
            ir.tensors_to_mxp = set()
            ir.tensors_to_mxp.add(node['frontend']['output_tensor'].name)

    for node_index in range(len(sorted_nodes)):
        node_name=sorted_nodes[node_index]
        node = ir.graph.nodes()[node_name]
        # Specifying start and end nodes to be run on sequencer
        node['frontend']['run_op'] = False
        if (node_index >= start_node_idx) and (node_index <= end_node_idx):
            node['frontend']['run_op'] = True
        update_node_tiling_info(ir,node)

    for blob_idx,current_blob in ir.tiling_blobs.items():
        current_blob_nodes = current_blob.nodes_in_blob
        nodes_in_blob_num = len(current_blob_nodes)
        #In each blob we write current_tile-1 and read tile+1. 
        # Write will start at last node of blob and will end in next tile, before start of op in the node we mark as "next_tile_read_node_idx"
        # Read of tile+1 will start right after above write finishes and will finish at first node of next tile
        current_blob.next_tile_read_node_idx = (nodes_in_blob_num - 1) // 2 
        if current_blob.next_tile_read_node_idx > 0:
            current_blob.next_tile_read_node_idx -= 1
            if (ir.graph.nodes[current_blob.nodes_in_blob[current_blob.next_tile_read_node_idx]]['op_type'] == 'Concat') or \
                (ir.graph.nodes[current_blob.nodes_in_blob[current_blob.next_tile_read_node_idx]]['op_type'] == 'Add'):
                    current_blob.next_tile_read_node_idx += 1 
    return ir


def get_blob_inputs_and_outputs(ir:internal_representation.IR,blob_node_names):
    blob_inputs = []
    blob_outputs = []

    for node_name in blob_node_names:
        node = ir.graph.nodes()[node_name]
        if node['op_type'] in MULTIPLE_INPUT_OPS:
            input_tensors = node['frontend']['input_tensors']
        else:
            input_tensors = [node['frontend']['input_tensor']]
        for input_tensor in input_tensors:
            if input_tensor.producer not in blob_node_names:
                if input_tensor not in blob_inputs:
                    blob_inputs.append(input_tensor)
        output_tensor = node['frontend']['output_tensor']
        output_tensor_consumers = output_tensor.consumers
        external_output = False
        if len(output_tensor_consumers) == 0:
            blob_outputs.append(output_tensor)
        else:
            for consumer_node in output_tensor_consumers:
                if consumer_node not in blob_node_names:
                    if output_tensor not in blob_outputs:
                        blob_outputs.append(output_tensor)
                    break
    return blob_inputs,blob_outputs

def mark_blobs_inputs_and_outputs(ir:internal_representation.IR) -> internal_representation.IR:
    for blob_idx,current_blob in ir.tiling_blobs.items():
        current_blob_nodes = current_blob.nodes_in_blob
        blob_inputs,blob_outputs = get_blob_inputs_and_outputs(ir,current_blob_nodes)
#        if len(blob_inputs)>2:
#            raise ValueError ('Currently max blob inputs = 2. Blob nodes: %s, number of  inputs:%d' % (str(current_blob.nodes_in_blob),len(blob_inputs)))
        current_blob.inputs = blob_inputs
#        if len(blob_outputs)>2:
#            raise ValueError ('Currently max blob outputs = 2. Blob nodes: %s, number of  outputs:%d' % (str(current_blob.nodes_in_blob),len(blob_outputs)))
        current_blob.outputs = blob_outputs
    return ir

def verify_y_folding_on_blob_edges(ir: internal_representation.IR):
    sorted_nodes = ir.lexicographical_topological_sorted_graph
    for node_index in range(len(sorted_nodes)):
        node_name=sorted_nodes[node_index]
        node = ir.graph.nodes()[node_name]
        current_blob_idx = node['frontend']['tiling_blob_idx']
        current_blob = ir.tiling_blobs[current_blob_idx]
        current_op_type = node['op_type']
        if current_op_type in MULTIPLE_INPUT_OPS:
            input_tensors = node['frontend']['input_tensors']
        else:
            input_tensors = [node['frontend']['input_tensor']]
        if ('force_y_folding' in node['frontend']) or ('force_y_folding' in node['frontend']):
            all_inputs_are_blob_inputs = True
            for input_tensor in input_tensors:
                if input_tensor not in current_blob.inputs:
                    all_inputs_are_blob_inputs = False
                    break
            if not all_inputs_are_blob_inputs:
                raise ValueError ('Node %s, is y folding but its not a blob input. Please check...')
            


def allocate_ddr_for_blobs_output_nodes(ir: internal_representation.IR) -> internal_representation.IR:
    program_and_intermediate_tensors_ddr = ir.ddr
    for tiling_blob_idx,tiling_blob in ir.tiling_blobs.items():
        last_node_in_blob_name = tiling_blob.nodes_in_blob[-1]
        last_node_in_blob = ir.graph.nodes()[last_node_in_blob_name]
        last_node_in_blob_output_tensor = last_node_in_blob['frontend']['output_tensor']
        blob_output_tensors = tiling_blob.outputs
        for blob_output_tensor in blob_output_tensors:
            if blob_output_tensor.name not in ir.outputs: # If its an output tensor, its already has allocated mem in DDR
                blob_output_tensor_shape=blob_output_tensor.shape_real_x16
                # Remove extra line padding
                # if blob_output_tensor_shape[2] % MAX_GRID_HEIGHT != 0: # If we dont fix grid exactly we need to add extra line that will hold zero for k=3 convs
                #   blob_output_tensor_shape[2]+=1
                blob_output_tensor.data = np.zeros(blob_output_tensor_shape,dtype=np.uint8)
                tensor_bytearray = create_tsnp_tensor_byte_array(blob_output_tensor.data)
                ddr_entry_description = ddr_entry_description = (
                                        f"Blob {tiling_blob_idx} output mem, "
                                        f"Intermediate tensor name: {blob_output_tensor.name}, "
                                        f"producer node: {blob_output_tensor.producer}, "
                                        f"folded shape: {blob_output_tensor.get_folded_shape()}, "
                                        f"real shape: {blob_output_tensor.shape_real_x16}"
                                    )
                 
                current_intermediate_tensor_ddr_entry = TensorDDREntry(tensor_bytearray, type = DDREntryType.INTERMEDIATE_TENSOR, description = ddr_entry_description,shape = blob_output_tensor_shape)
                blob_output_tensor.ddr_entry = current_intermediate_tensor_ddr_entry

                # Check if this tensor is a split version of another tensor.
                # If so, do not add a new DDR entry, it will re-use the original entry.
                if blob_output_tensor.name in ir.split_tensor_to_original_tensor_map:
                    # Get the original tensor
                    original_output_tensor_name = ir.split_tensor_to_original_tensor_map[blob_output_tensor.name]
                    original_output_tensor = ir.tensors[original_output_tensor_name]

                    # Make sure the original tensor has a DDR entry. There are different cases:
                    # - it is an output tensor of the graph: already has a DDR entry
                    # - it is not an output tensor and current tensor is the first of the 2 splits: original has no DDR entry
                    # - it is not an output tensor and current tensor is the second of the 2 splits: original has a DDR entry
                    if original_output_tensor.ddr_entry is None:
                        original_output_tensor_shape = original_output_tensor.shape_real_x16
                        original_output_tensor.data = np.zeros(original_output_tensor_shape,dtype=np.uint8)
                        original_tensor_bytearray = create_tsnp_tensor_byte_array(original_output_tensor.data)
                        original_ddr_entry_description = 'Blob %d output mem, Intermediate tensor name: %s, producer node: %s, folded shape: %s' % (tiling_blob_idx,
                                                        original_output_tensor.name,original_output_tensor.producer,str(original_output_tensor.get_folded_shape()))
                        original_output_tensor_ddr_entry = TensorDDREntry(original_tensor_bytearray, type = DDREntryType.INTERMEDIATE_TENSOR, description = original_ddr_entry_description,shape = original_output_tensor_shape)
                        original_output_tensor.ddr_entry = original_output_tensor_ddr_entry
                        program_and_intermediate_tensors_ddr.add_entry(original_output_tensor_ddr_entry)

                    # Assign this tensor to that same type and address, plus an offset for split2
                    blob_output_tensor.ddr_entry.type = original_output_tensor.ddr_entry.type
                    blob_output_tensor.ddr_entry.address = original_output_tensor.ddr_entry.address
                    if 'split2' in blob_output_tensor.name:
                        original_size = original_output_tensor.ddr_entry.get_length()
                        current_split_size = current_intermediate_tensor_ddr_entry.get_length()
                        blob_output_tensor.ddr_entry.address += (original_size - current_split_size)
                else:
                    program_and_intermediate_tensors_ddr.add_entry(current_intermediate_tensor_ddr_entry) 
    return ir

# Allocate tensors reserved by the system
# Currently this is just a tensor of zeros to be loaded into the final AMM blocks
def allocate_system_tensors(ir: internal_representation.IR):
    # First create a tensor for the zeros to be loaded into AMM
    zero_tensor_name = 'SYSTEM_zero_tensor'
    zero_tensor_data = np.zeros((1, 14, 16, URAM_BLOCK_SIZE), dtype=np.int8)
    ir.tensors[zero_tensor_name] = Tensor(
        zero_tensor_name,
        zero_tensor_data,
        is_constant=True,
        shape=zero_tensor_data.shape)

    # Create DDR entry for the zero tensor
    zero_data_bytearray = bytearray([0]*zero_tensor_data.size)
    ddr_entry_description = "System-reserved zero region"
    # Assert this entry doesn't already exist
    for entry in ir.ddr.entries:
        assert entry.description != ddr_entry_description
    zero_tensor_ddr_entry = TensorDDREntry(
        zero_data_bytearray,
        type = DDREntryType.INTERMEDIATE_TENSOR,
        description = ddr_entry_description,
        shape = zero_tensor_data.shape)
    ir.tensors[zero_tensor_name].ddr_entry = zero_tensor_ddr_entry
    ir.ddr.add_entry(zero_tensor_ddr_entry)

    # Allocate the last block in each AMM
    ir.amms.allocate_blocks_in_all_amms([URAM_NUM_BLOCKS-1])
    allocated_blocks = [[URAM_NUM_BLOCKS-1],[URAM_NUM_BLOCKS-1],[URAM_NUM_BLOCKS-1],[URAM_NUM_BLOCKS-1]]
    ir.amms.tensors_in_amm[zero_tensor_name] = AMMTensor(ir.tensors[zero_tensor_name],allocated_blocks,allocated_blocks,tile_num=0,xslice_num=0)

    return ir

def compile(ir: internal_representation.IR, debug_output_dir:str) -> internal_representation.IR:
    ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    ir = get_ops_grid_config(ir) # This will scan the graph and calculate grid config for each of the grid related ops
    if DEBUG_CREATE_ORDERING_CONV:
        ir = add_output_ordering_conv(ir)
    if SUPPORT_Z_AXIS_TILING:
        ir = generate_z_tiling_ops(ir) # This will split ops with z tiling to separate ops
    if DEBUG_X_SLICING:
        ir = add_x_folding(ir)
    if DEBUG_ADD_UNFOLDING_X:
        ir = add_x_unfolding(ir) 
    ir = get_ops_channel_balancing(ir) # This will allocate each input/output channel to a grid in a balanced way
    ir = allocate_input_tensors(ir)
    ir = allocate_output_tensors(ir)
    if not DEBUG_SKIP_CBC_GENERATION:
        ir = get_last_compile_per_layer_wloc_size(ir,debug_output_dir)
    move_outputs_write_to_amm_to_erliest_point(ir) # This moves outputs generation to earliest execution point so its tensor can be de-allocated
    qparams_report_filename = os.path.join(ir.compiler_output_dir, ir.model_name+'_qparams_pretiling')
    reports.generate_qparams_report(qparams_report_filename,ir)

    allocation_succeeded = False
    failing_blob_idx = None
    # This is a copy of the IR which will be updated after each set_tiling_blobs.
    # It allows restoring to a version of the IR which does not include other
    # changes such as the insertion of ordering convolutions.
    latest_ir = copy.deepcopy(ir)
    while not allocation_succeeded:
        ir = copy.deepcopy(latest_ir) # Restore the previous iteration IR
        ir, needed_node_output_split = set_tiling_blobs(ir, failing_blob_idx)
        latest_ir = copy.deepcopy(ir) # Update the IR for this iteration
        
        if needed_node_output_split:
            failing_blob_idx = None
            continue

        if DEBUG_IDENTITY_CONV_BEFORE_K3_1T:
            ir = add_identity_before_k3_starting_1T_blob(ir)
        ir = set_node_tiling_info(ir)
        ir = mark_blobs_inputs_and_outputs(ir)
        verify_y_folding_on_blob_edges(ir) # Y folding and unfolding can happen only on blob inputs as we do it with dma read from ddr
        if DEBUG_CREATE_ORDERING_CONV:
            ir = add_pre_unfolding_ordering_conv(ir) # Since channels must be ordered before input y axis unfolding we insert a reordering node before each such node

            # Add ordering conv before folded nodes
            if DEBUG_TRY_TO_FIX_AUTO_FOLD:
                ir = add_pre_folding_ordering_conv(ir)

            ir = add_pre_sync_ordering_conv(ir)
        qparams_report_filename = os.path.join(ir.compiler_output_dir, ir.model_name+'_qparams')
        reports.generate_qparams_report(qparams_report_filename,ir)

        ir, allocation_succeeded, failing_blob_idx = mem_allocation_pass(ir,debug_output_dir)# This will include AMM and DDR offloading allocations

    ir = allocate_ddr_for_blobs_output_nodes(ir) # AlloThis must be done after mem_allocation_pass since mem_allocation_pass resets all mem allocations
    ir = prepare_intermediate_tensors_write(ir) # This part can be done after program_compiler.compile found where intermediate tensors need to be written to ddr
    ir = cbc_generation_pass(ir, debug_output_dir) # This will include clock by clock (cbc) generation to grid ops
    if not DEBUG_SKIP_CBC_GENERATION:
        save_per_layer_wloc_size(ir,debug_output_dir)
    return ir

def main ():
    ir = internal_representation.IR('')
    ir = ir.load('conv.nxf')
    compiler_augmented_ir = compile(ir)
    ir.save('conv.nxb')

if __name__ == "__main__":

    main()    
