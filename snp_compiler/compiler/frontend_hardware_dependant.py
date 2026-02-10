import sys
sys.path.append('.')
import numpy as np
import networkx as nx
import common.internal_representation as internal_representation
from common.tensor_ir import Tensor
from common.hw_config import FRACTIONAL_BITS, BIAS_FRACTIONAL_BITS, BIAS_MULITIPLIER, BIAS_ROUNDING_ADD, MAX_BIAS_BITS, MAX_GRID_HEIGHT,\
                    MAX_GRID_WIDTH, MINIMAL_ACTUAL_INPUT_WIDTH, MAX_REDUCE_BUS_WIDTH, REDUCED_MAC_RESCALE_BUS_WIDTH,GRID_CONFIGS,URAM_DEPTH,\
                    FORCED_OUTPUT_FOLDING_OPS, MCHP_NUMERICS, MCHP_MAC_TO_RQ_BUS_WIDTH, FINAL_RESULTS_BITS, OVERFLOW_EXTRA_BITS, INT_SCALE_BITS,\
                    MAC_ROUGH_SHIFT_GRANULARITY, MCHP_ADDED_SCALE_BITS, UINT17_ADDED_SCALE_BITS, SMALLEST_X_RESIZABLE_GRID_SIZE, LONG_ENTRY_BITS,\
                    TFLITE_REQUANT, URAM_BLOCK_SIZE, URAM_NUM_BLOCKS
from common.debug_flags import DEBUG_FORCE_FOLDCONV,DEBUG_CLIP_BIAS_TO_MAX_BIAS_BITS, DEBUG_SIMULATE_CONCAT_REQUANT, DEBUG_FORCE_FOLDED_INPUT,\
                                DEBUG_FORCE_FOLDED_FACTOR_X, DEBUG_FORCE_FOLDED_FACTOR_Y, DEBUG_AUTO_Y_FOLDING, DEBUG_TRY_TO_FIX_AUTO_FOLD,\
                                DEBUG_MINIMIZE_Y_FOLDING, DEBUG_X_SLICING, DEBUG_ADD_UNFOLDING_X, DEBUG_OPTIMIZE_FIRST_LAYER_DDR_ACCESS
from common.utils import LUTPopulateInt8, quantize2MathBlock, int8min, int8max
import folding_algo
import copy
import math
from tqdm import tqdm

def round_up_to_next_16(x):
    return ((x + 15) // 16) * 16

def calculate_prime_factors(N):
    prime_factors = []
    while N % 2 == 0:
        prime_factors.append(2)
        N = N // 2
    if N == 1:
        return prime_factors
    for factor in range(3, N + 1, 2):
        while N % factor == 0:
            prime_factors.append(factor)
            N = N // factor
        if N == 1:
            return prime_factors

def update_x_fold_slice(input_folding_factor, original_input_width, kernel_size, stride, is_stride2 = False):
    x_slices = 1
    if DEBUG_X_SLICING and (input_folding_factor >= 0):
        if (kernel_size == 1) and (stride == 1):
            #x_slices = pow(2, input_folding_factor)
            x_slices = original_input_width / 16
            folding_factor_x = 0
        elif (kernel_size == 1) and (stride == 2):
            x_slices = (original_input_width/2)/16
            folding_factor_x = 1
        elif (kernel_size == 3) and (stride == 1):
            x_slices = pow(2, input_folding_factor)
            folding_factor_x = 0
        elif (kernel_size == 3) and (stride == 2):
            if is_stride2:
                x_slices = (original_input_width/4)/16
                folding_factor_x = 2
            else:
                x_slices = (original_input_width/2)/16
                folding_factor_x = 1
        elif (kernel_size > 3) and (stride == 1):
            x_slices = (original_input_width/2)/16
            folding_factor_x = 1
        elif (kernel_size > 3) and (stride == 2):
            x_slices = (original_input_width/4)/16
            folding_factor_x = 2
        else:
            folding_factor_x = input_folding_factor
    else:
        folding_factor_x = input_folding_factor
    return int(folding_factor_x), math.ceil(x_slices)

def get_new_input_folding_factors(original_input_width,original_input_height,stride,kernel_size,original_input_channels,original_output_channels, is_stride2 = False):
    x_res_primals = calculate_prime_factors(original_input_width)
    x_res_primals
    #for primal in x_res_primals:
    #    if primal!=2:
    #        raise ValueError ('Non 2 primals currently not supported. To support this x/y folding factor need to be a list of primals instead of a single factor')
    folded_res=original_input_width
    last_idx_in_res=0
    for idx,x_primal in enumerate(x_res_primals):
        if folded_res<=MAX_GRID_WIDTH:
            break
        folded_res=folded_res // x_primal
        last_idx_in_res = idx
    for primal_idx in range(last_idx_in_res):
        primal = x_res_primals[primal_idx]
        if primal!=2:
            raise ValueError ('Non 2 primals currently not supported. To support this x/y folding factor need to be a list of primals instead of a single factor')
    input_x_folding_coef = original_input_width // folded_res
    input_folding_factor_x = math.ceil(math.log(input_x_folding_coef,2))

    y_res_primals = calculate_prime_factors(original_input_height)
    #y_res_primals.reverse()
    input_y_folding_coef=1
    #in order to calculate y folding we check that expected amm depth (of both input and output) doesnt exceed max amm depth
    get_last_folding_coef = False
    for idx,y_primal in enumerate(y_res_primals):
        # Calculatin
        if y_primal!=2:
            break
        if original_input_height//input_y_folding_coef<=MAX_GRID_HEIGHT:
            get_last_folding_coef = True
            break
        input_y_folding_coef=input_y_folding_coef*y_primal
        folded_input_channels = original_input_channels*input_x_folding_coef*input_y_folding_coef
        folded_output_channels = original_output_channels*input_x_folding_coef*input_y_folding_coef/(stride*stride)
        total_channels = folded_input_channels+folded_output_channels
        #if total_channels>URAM_DEPTH // 2:
        if (total_channels>2817) or (np.maximum(folded_input_channels, folded_output_channels) > 2**LONG_ENTRY_BITS):
            input_y_folding_coef = input_y_folding_coef // y_primal # We get back to the latest working coef
            get_last_folding_coef = True
            break
    
    input_folding_factor_y = math.ceil(math.log(input_y_folding_coef,2))
    if input_folding_factor_y!=math.log(input_y_folding_coef,2):
        raise ValueError ('Y folding factor must be a power of 2, actual value: %d' % input_y_folding_coef)
    
    input_folding_factor_x, x_slices = update_x_fold_slice(input_folding_factor_x, original_input_width, kernel_size, stride, is_stride2=is_stride2)

    return input_folding_factor_x,input_folding_factor_y, x_slices



def get_input_folding_factors(original_input_width,original_input_height,stride):
    input_folding_factor_x = math.ceil(math.log(original_input_width / MAX_GRID_WIDTH,2))
    if input_folding_factor_x<0: # If input resolution is less than MAX_GRID_WIDTH, folding is not needed
        input_folding_factor_x=0
    if DEBUG_FORCE_FOLDED_INPUT:
        input_folding_factor_x=DEBUG_FORCE_FOLDED_FACTOR_X
    input_folding_factor_y = math.ceil(math.log(original_input_height / MAX_GRID_HEIGHT,2))
    if input_folding_factor_y<0: # If input resolution is less than MAX_GRID_WIDTH, folding is not needed
        input_folding_factor_y=0
    if DEBUG_FORCE_FOLDED_INPUT:
        input_folding_factor_y=DEBUG_FORCE_FOLDED_FACTOR_Y
    return input_folding_factor_x,input_folding_factor_y

def get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size):
    input_folding_coef_y = math.pow(2,input_folding_factor_y)
    folded_y_size = original_input_height // input_folding_coef_y
    if original_kernel_size == 1:
        num_y_tiles = math.ceil(folded_y_size / MAX_GRID_HEIGHT)
    else:
        if folded_y_size<=MAX_GRID_HEIGHT:
            num_y_tiles = 1
        else:
            handled_lines=MAX_GRID_HEIGHT-1
            num_y_tiles=1
            while (folded_y_size>handled_lines):
                if (handled_lines+(MAX_GRID_HEIGHT-1))>=folded_y_size:
                    num_y_tiles+=1
                    break
                handled_lines+=(MAX_GRID_HEIGHT-2)
                num_y_tiles+=1
    return num_y_tiles

def is_amm_overflow(input_tensor_size: int, output_tensor_size: int, is_output: bool) -> bool:
    total_channels = input_tensor_size + output_tensor_size

    # If it's a workload output, then an ordering node will follow with input
    # channels=output_channels, so total wil be output_channels*2
    # STEFAN TODO: Account for other cases of ordering nodes (e.g., before folding/unfolding)
    if is_output:
        total_channels=max(total_channels, output_tensor_size*2)
    if (total_channels> URAM_DEPTH // 2) or (input_tensor_size> 2** LONG_ENTRY_BITS):
        return True
    return False

def is_y_unfolding_needed(ir:internal_representation.IR,node_name,node):

    # If its input to the workload we dont force unfolding.
    # inputs folding calc takes care of choosing the right folding factor to fit mem
    if not len(node['frontend']['preceding_nodes_params'])>0:
        return False

    # Resize folds currently, so unfold right after
    # TODO: Handle case where Conv is not directly after a Resize, e.g. check inputs
    # recursively until a Resize is found (and stop if Conv is found).
    producing_node_name = node['frontend']['input_tensor'].producer
    if ir.graph.nodes[producing_node_name]['op_type'] == 'Resize':
        return True

    input_tensor_size_on_amm = node['frontend']['input_tensor'].get_size_on_amm()
    output_tensor_size_on_amm = node['frontend']['output_tensor'].get_size_on_amm()
    is_output = node['frontend']['output_tensor'].name in ir.outputs
    if is_amm_overflow(input_tensor_size_on_amm, output_tensor_size_on_amm, is_output):
        return True

    # Because Concat Y unfolding is not implemented yet, check if this Conv output
    # is a Concat which will not fit in AMM. If it is, unfold here instead.
    if node['op_type'] == 'Conv':
        for consumer in node['frontend']['output_tensor'].consumers:
            consumer_node = ir.graph.nodes()[consumer]
            if consumer_node['op_type'] != 'Concat':
                continue

            # Found a Concat output. Get total input size and output size.
            concat = consumer_node
            total_input_tensor_size_on_amm = 0
            for input_tensor in concat['frontend']['input_tensors']:
                total_input_tensor_size_on_amm += input_tensor.get_size_on_amm()
            output_tensor_size_on_amm = concat['frontend']['output_tensor'].get_size_on_amm()
            is_output = concat['frontend']['output_tensor'].name in ir.outputs
            if is_amm_overflow(total_input_tensor_size_on_amm, output_tensor_size_on_amm, is_output):
                return True

    return False

def get_next_non_folding_conv(ir:internal_representation.IR,node):
    current_node = node
    while 'folding_conv' in current_node['name']:
        successors = list(ir.graph.successors(current_node['name']))
        # For now consider single successor case
        if len(successors) != 1:
            break
        current_node = ir.graph.nodes()[successors[0]]
    return current_node

def get_num_consecutive_stride2_conv(ir:internal_representation.IR,node):
    current_node = node
    num_consecutive_stride2_conv = 0
    while current_node['frontend']['stride'] == 2:
        num_consecutive_stride2_conv += 1
        successors = list(ir.graph.successors(current_node['name']))
        # For now consider single successor case
        if len(successors) != 1:
            break
        current_node = ir.graph.nodes()[successors[0]]
        if ('stride' not in current_node['frontend']):
            break
    return num_consecutive_stride2_conv

def get_required_folding_factor_for_kernel_size(kernel_size):
    if kernel_size >= 6:
        # For 6x6, there is a spread of 2 lines in one direction and 3 in the other.
        # Folding by a factor of 1 (folding 2x) would have a spread of 1 and 1.5.
        # This cannot be handled by a 3x3. So need to fold twice.
        return 2
    elif kernel_size >= 4:
        # For 4x4, there is a spread of 2 lines in one direction and 1 in the other.
        # Folding by a factor of 1 (folding 2x) would have a spread of 1 and 0.5.
        # This can be handled by a 3x3. So need to fold only once.
        return 1
    return 0
def calc_avgpool_folding_factor(ir:internal_representation.IR,node_name,node):
    calc_conv_folding_factor(ir,node_name,node)

def add_xfoldingConv_before_node(ir:internal_representation.IR, node):
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    channels = original_input_tensor_shape[1]
    op_input_name = node['frontend']['input_tensor'].name
    new_node_name = node['name'] + '_fold_x_0'
    op_output_name = new_node_name+'_out'
    conv_W_initializer_tensor_name = new_node_name+"_W"
    conv_B_initializer_tensor_name = new_node_name+"_B"
    ir.tensors[op_output_name] = Tensor(op_output_name,np.zeros(original_input_tensor_shape),is_constant=False, shape=original_input_tensor_shape)
    ir.tensors[op_output_name].scale = node['frontend']['input_tensor'].scale
    ir.tensors[op_output_name].zero_point = node['frontend']['input_tensor'].zero_point
    source_node = ir.graph.nodes[ir.tensors[op_input_name].producer]
    
    y_tiles= get_num_y_tiles(original_input_height=node['frontend']['input_tensor'].shape[2],input_folding_factor_y=0,original_kernel_size=1)
    
    # Create the new node
    new_node = {
        'name': new_node_name,
        'op_type': 'Conv',  # Use Conv instead of Identity for folding compatibility
        'attributes': {
            'kernel_shape': [1, 1],
            'pads': [0, 0, 0, 0]
        },
        'outputs': [op_output_name],
        'frontend': {}
    }

    new_node['frontend'].update({
        'input_channels': channels,
        'output_channels': channels,  # Same as input
        'input_folding_factor_x': 0,
        'input_folding_factor_y': source_node['frontend']['input_folding_factor_y'],
        'output_folding_factor_x': 1,
        'output_folding_factor_y': source_node['frontend']['input_folding_factor_y'],
        'preceding_nodes_params': [(source_node['name'], 0)],
        'input_tensor': ir.tensors[op_input_name],
        'output_tensor': ir.tensors[op_output_name],
        'input_tensor_scale': ir.tensors[op_input_name].scale,
        'input_tensor_zp': ir.tensors[op_input_name].zero_point,
        'output_tensor_scale': ir.tensors[op_output_name].scale,
        'output_tensor_zp': ir.tensors[op_output_name].zero_point,
        'kernel_size': 1,
        'stride': 1,
        'padding':new_node['attributes']['pads'],
        #'y_tiles':node['frontend']['y_tiles']
        'y_tiles':y_tiles
    })

    # Set up weights_tensor
    original_weights_tensor_shape = [channels,channels,1,1]
    original_weights_tensor = np.zeros(original_weights_tensor_shape,dtype=np.int64)
    weight_value = 32 # Changed from 8 to 32 in MCHP numerics so that scale will be smaller and right shift will be bigger
    for oc in range(channels):
            original_weights_tensor[oc,oc,0,0] = weight_value
    w_int8_np = original_weights_tensor.astype(np.int8)

    new_weights_tensor_name = conv_W_initializer_tensor_name
    per_channel_scale = np.full((channels),1/weight_value)
    per_channel_zp = np.full((channels),0)
    weights_tensor = Tensor(new_weights_tensor_name,w_int8_np,is_constant=True,shape = original_weights_tensor_shape,scale = per_channel_scale,zero_point=per_channel_zp)
    ir.tensors[new_weights_tensor_name] = weights_tensor

    new_node['frontend'].update({
        'weights_tensor':           weights_tensor,
        'weights_per_channel_scale':per_channel_scale,
        'weights_per_channel_zp':   per_channel_zp,
        'sparse_macs':              original_input_tensor_shape[1]*original_input_tensor_shape[2]*original_input_tensor_shape[3]
        })
    
    # Set up biases_tensor
    original_biases_tensor_shape = [channels]
    original_biases_tensor = np.zeros(original_biases_tensor_shape,dtype=np.int64)
    new_biases_tensor_name = conv_B_initializer_tensor_name
    biases_tensor = Tensor(new_biases_tensor_name,original_biases_tensor,is_constant=True,shape = original_biases_tensor_shape)
    ir.tensors[new_biases_tensor_name] = biases_tensor
    new_node['frontend']['biases_tensor'] = biases_tensor
    new_node['frontend']['force_folding_x'] = True
    new_node['frontend']['output_tensor'].folding_factor_x = 1
    new_node['frontend']['output_tensor'].folding_factor_y = source_node['frontend']['input_folding_factor_y']
    x_slices = node['frontend']['input_tensor'].x_slices
    new_node['frontend']['output_tensor'].x_slices = math.ceil(x_slices/2)
    new_node['frontend']['x_slices'] = x_slices
    #if (source_node['frontend']['output_folding_factor_y'] == 0) and (node['frontend']['input_folding_factor_y'] > 0):
    #    new_node['frontend']['force_folding_y'] = True
    
    #insert the node to the graph: update edges etc
    ir.graph.add_node(new_node_name,**new_node) # When the node is created it copies the dictionary attributes and create a new dict
    created_new_node = ir.graph.nodes[new_node_name]

    inputs = []
    inputs.append(op_input_name)
    inputs.append(conv_W_initializer_tensor_name)
    inputs.append(conv_B_initializer_tensor_name)
    outputs = []
    outputs.append(op_output_name)
    created_new_node['inputs'] = inputs
    created_new_node['outputs'] = outputs

    input_consumers = ir.tensors[op_input_name].consumers
    target_node_names = []
    if (len(input_consumers) > 0):
        for idx in range(len(input_consumers)):
            next_node_info = ir.graph.nodes[input_consumers[idx]]
            if ('stride' in next_node_info['frontend']) and (next_node_info['frontend']['stride'] == 2): 
                target_node_names.append(input_consumers[idx])
    ir.graph.add_edge(source_node['name'],new_node_name)
    for i_name in target_node_names:
        ir.graph.add_edge(new_node_name,i_name)
        ir.graph.remove_edge(source_node['name'],i_name)
        index = input_consumers.index(i_name)
        ir.tensors[op_input_name].consumers[index] = new_node_name
        ir.tensors[op_output_name].consumers.append(i_name)
    ir.tensors[op_input_name].consumers = list(set(ir.tensors[op_input_name].consumers))
    ir.tensors[op_output_name].producer = new_node_name
    
    # Update the 'inputs' and input_tensor field in target node
    for i_name in target_node_names:
        next_node_info = ir.graph.nodes[i_name]
        ir.switch_input_name(next_node_info,original_input_name=op_input_name,new_input_name=op_output_name)
        ir.switch_input_tensor(next_node_info,original_input_tensor=ir.tensors[op_input_name],new_input_tensor=ir.tensors[op_output_name])

    # Update following nodes params
    source_node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(source_node) # Update the following_nodes_params field in source node - It is important to also update following nodes according to execution order
    for i_name in target_node_names:
        next_node_info = ir.graph.nodes[i_name]
        next_node_info['frontend']['preceding_nodes_params'] = ir.get_updated_preceding_nodes(next_node_info) # Update the preceding_nodes_params field in target node
    created_new_node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(new_node)

    return ir

def calc_conv_folding_factor(ir:internal_representation.IR,node_name,node):
    # We check y force folding here since it affects the input folding
    need_to_fold_y_for_concat = (node['name'] in ir.marked_nodes_for_folding_y) and DEBUG_AUTO_Y_FOLDING
    force_y_folding = need_to_fold_y_for_concat or node_name in ir.force_y_folding
    force_y_unfolding = node_name in ir.force_y_unfolding

    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    stride = node['frontend']['stride']
    original_input_width = original_input_tensor_shape[3]
    original_input_height = original_input_tensor_shape[2]
    original_input_channels = original_input_tensor_shape[1]
    original_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape()
    original_output_channels = original_output_tensor_shape[1]
    original_kernel_size = node['frontend']['kernel_size']

    skip_y_unfolding = False
    if len(node['frontend']['preceding_nodes_params'])>0: # If its an input to the workload we calc folding factor based on resolution, otherwise we take it from input tensor
        input_folding_factor_x = node['frontend']['input_tensor'].folding_factor_x
        input_folding_factor_y = node['frontend']['input_tensor'].folding_factor_y
        x_slices = node['frontend']['input_tensor'].x_slices
        if force_y_folding and force_y_unfolding:
            # Undo both
            force_y_folding = False
            node['frontend']['force_folding_y'] = False
            if 'force_folding_y' in node['frontend']:
                del node['frontend']['force_folding_y']
            force_y_unfolding = False
            node['frontend']['force_unfolding_y'] = False
            if 'force_unfolding_y' in node['frontend']:
                del node['frontend']['force_unfolding_y']
            skip_y_unfolding = True
        else:
            if force_y_folding:
                node['frontend']['force_folding_y'] = True
                input_folding_factor_y+=1
            else: # There are cases where 1st pass of folding factor calc fails and at later pass a folding conv is change to non folding conv
                if 'force_folding_y' in node['frontend']:
                    del node['frontend']['force_folding_y']

            if force_y_unfolding:
                node['frontend']['force_unfolding_y'] = True
                input_folding_factor_y-=1
            else: # There are cases where 1st pass of folding factor calc fails and at later pass an unfolding conv is change to regular conv
                if 'force_unfolding_y' in node['frontend']:
                    del node['frontend']['force_unfolding_y']

        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
    else:
        is_stride2 = False
        if (len(node['frontend']['following_nodes_params']) > 0):
            following_node_name = node['frontend']['following_nodes_params'][0][0]
            following_node = ir.graph.nodes[following_node_name]
            if ('stride' in following_node['frontend'] and  following_node['frontend']['stride'] == 2):
                is_stride2 = True
        real_input_width  = round_up_to_next_16(original_input_width)

        real_input_height = round_up_to_next_16(original_input_height)
        input_folding_factor_x,input_folding_factor_y, x_slices = get_new_input_folding_factors(real_input_width,real_input_height,stride,original_kernel_size,original_input_channels,original_output_channels, is_stride2 = is_stride2)

        if 'forced_tiling' in ir.model_name:
            if stride == 1:
                input_folding_factor_y = 0
            else:
                input_folding_factor_y = 1
        if 'two_tile_test' in ir.model_name:
            input_folding_factor_y = 0
        if 'two_tile_alternate_test' in ir.model_name:
            # 2-tile for first blob
            input_folding_factor_y = 0
        if 'yolo128' in ir.model_name:
            if node_name == 'Conv_0':
                input_folding_factor_y = 2
        if 'yolo256' in ir.model_name:
            if node_name == 'Conv_0':
                input_folding_factor_y = 3
        if 'y_folding_test' in ir.model_name:
            input_folding_factor_y=0
        if 'y_unfolding_test' in ir.model_name:
            input_folding_factor_y=1
        if 'tsnp_x_folding' in ir.model_name and '2tiles' in ir.model_name:
            input_folding_factor_y=1
        if 'concat_channel_reordering_fold2_test' in ir.model_name:
            # Simulator result should be 0.003 if its working well or if folding factor=0 and 0.3 if not
            input_folding_factor_x=1 # Change this to 1 to check if concat channel reordering is working well, 0 will force no folding so reordering is not needed
            input_folding_factor_y=1# Change this to 1 to check if concat channel reordering is working well, 0 will force no folding so reordering is not needed
        if 'concat_channel_reordering_fold4_test' in ir.model_name:
            # Simulator result should be 0.003 if its working well or if folding factor=0 and 0.3 if not
            input_folding_factor_x=2 # Change this to 1 to check if concat channel reordering is working well, 0 will force no folding so reordering is not needed
            input_folding_factor_y=2# Change this to 1 to check if concat channel reordering is working well, 0 will force no folding so reordering is not needed
        elif ('Conv256x256_k3' in ir.model_name):
            input_folding_factor_y = 2
            num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)

        else:
            # Calculate Y Folding Factor
            # Get the first non-folding conv node
            first_conv = get_next_non_folding_conv(ir, node)
            # Make initial Y folding the number of initial s=2 Convs
            input_folding_factor_y = get_num_consecutive_stride2_conv(ir, first_conv)

            # Folding might also be required due to kernel size
            first_conv_kernel_size = first_conv['frontend']['kernel_size']
            kernel_required_folding_factor = get_required_folding_factor_for_kernel_size(first_conv_kernel_size)
            input_folding_factor_y = max(input_folding_factor_y, kernel_required_folding_factor)

            num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
        node['frontend']['input_tensor'].folding_factor_x = input_folding_factor_x
        node['frontend']['input_tensor'].folding_factor_y = input_folding_factor_y # We update input folding factor only if its re-calced from scratch so that input folding/unfolding in y will not change it
        node['frontend']['input_tensor'].x_slices = x_slices
    if stride == 1:
        output_folding_factor_x = input_folding_factor_x
        output_folding_factor_y = input_folding_factor_y
    elif stride == 2:
        output_folding_factor_x = input_folding_factor_x - 1
        output_folding_factor_y = input_folding_factor_y - 1
    else:
        raise ValueError ('Stride >2 is not supported')


    following_nodes_params = node['frontend']['following_nodes_params']
    max_following_node_stride = 0
    force_folding_x = False
    for following_node_params in following_nodes_params:
        current_following_node = ir.graph.nodes[following_node_params[0]] # [0] is node name, [1] is which input of folloing node is connected to current node
        if 'stride' in current_following_node['frontend'] and current_following_node['frontend']['stride']>max_following_node_stride:
            max_following_node_stride = current_following_node['frontend']['stride']
            if max_following_node_stride>2:
                raise ValueError ('Stride>2 is not currently supported')
        if (output_folding_factor_x == 0):
            if (('CONCATENATION' in following_node_params[0]) and (len(following_nodes_params) > 1)):
                force_folding_x = False
                break
            if ((current_following_node['op_type'] == 'Conv') and (current_following_node['frontend']['stride'] == 2)):
                if (len(following_nodes_params) == 1):
                    force_folding_x = True

            
    if 'force_folding_x' in node['frontend']: # There are cases where 1st pass of folding factor calc fails and at later pass a folding conv is change to non folding conv
        del node['frontend']['force_folding_x']

    # Check if we need to fold to match following concat grid params
    need_to_fold_x_for_concat = node['name'] in ir.marked_nodes_for_folding_x
    force_x_folding = need_to_fold_x_for_concat or node_name in ir.force_x_folding or force_folding_x

    output_xslices = x_slices
    if 'tsnp_x_folding' in ir.model_name and node_name=='conv1':
        node['frontend']['force_folding_x'] = True
        output_folding_factor_x += 1
        output_xslices = output_xslices // 2

    add_folding_conv_before_node = False
    # We change to folding conv if we see in following ops a stride=2 (In case of width>14 => 28x28 grid) or if current op is stride==2
    # if current op is stride==2 and the data is not folded we will perform a folding conv with stride=2 which means un-needed output channels will be calculated but dropped in RQ
    if DEBUG_FORCE_FOLDCONV: #if (output_folding_factor==0 and max_following_node_stride==2 and original_input_width>MINIMAL_ACTUAL_INPUT_WIDTH) or DEBUG_FORCE_FOLDCONV:
        node['frontend']['force_folding_x'] = True
        output_folding_factor_x += 1
        output_xslices = output_xslices // 2
    elif force_x_folding:
        node['frontend']['force_folding_x'] = True
        output_folding_factor_x+=1
        output_xslices = math.ceil(output_xslices / 2)
    elif 'force_unfolding_x' in node['frontend']:
        output_folding_factor_x -= 1
        check_x_slices = output_xslices * 2
        output_xslices = round_up_to_next_16(original_input_width)//16

        if check_x_slices-1>output_xslices:
            raise ("Check this case, x_slices increase more than 2x")      
        
    elif ((output_folding_factor_x==-1 or output_folding_factor_y==-1) and stride==2): # In that case the output folding factor is not increased since we drop 3/4 of the output channels
        if output_folding_factor_x==-1:
            input_folding_factor_x+=1
            output_folding_factor_x+=1
            add_folding_conv_before_node = True
            output_xslices = math.ceil(output_xslices / 2)          
        if output_folding_factor_y==-1:
            input_folding_factor_y+=1
            output_folding_factor_y+=1
            node['frontend']['force_folding_y'] = True

    node['frontend']['output_tensor'].folding_factor_x = output_folding_factor_x 
    node['frontend']['output_tensor'].folding_factor_y = output_folding_factor_y # It is important to update the output tensor folding factor since it is used by below "is_y_unfolding_needed"
    node['frontend']['output_tensor'].x_slices = output_xslices
    node['frontend']['output_folding_factor_x'] = output_folding_factor_x
    node['frontend']['output_folding_factor_y'] = output_folding_factor_y
    if (add_folding_conv_before_node):
        node['frontend']['x_slices'] = output_xslices
    else:
        node['frontend']['x_slices'] = x_slices
    if DEBUG_AUTO_Y_FOLDING and not skip_y_unfolding and is_y_unfolding_needed(ir,node_name,node,): #Check if there is expected mem overflow that can be solved by y unfolding(which reduce the number of channels)
        if not force_y_unfolding:
            if output_folding_factor_y>0: # We can unfold y if output folding factor is already 0
                ir.force_y_unfolding.append(node_name)
                node['frontend']['force_unfolding_y'] = True
                input_folding_factor_y-=1
                output_folding_factor_y-=1
                num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
                node['frontend']['output_tensor'].folding_factor_y = output_folding_factor_y
                node['frontend']['output_folding_factor_y'] = output_folding_factor_y
            else:
                print('Warning: Tried to y unfold for mem reduction but failed since y is not folded')

    num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
    
    if not add_folding_conv_before_node:
        node['frontend']['input_tensor'].y_tiles = num_y_tiles
    node['frontend']['y_tiles'] = num_y_tiles
    node['frontend']['input_folding_factor_x'] = input_folding_factor_x
    node['frontend']['input_folding_factor_y'] = input_folding_factor_y
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    
    if add_folding_conv_before_node:
        add_xfoldingConv_before_node(ir, node)
    
    if DEBUG_OPTIMIZE_FIRST_LAYER_DDR_ACCESS:
        if (len(node['frontend']['preceding_nodes_params']) == 0) and (input_folding_factor_x == 0) and (x_slices % 4 == 0):
            node['frontend']['input_tensor'].num_packed_xslices = 4


def calc_conv_qparams(ir,node_name,node):
    output_channels = node['frontend']['output_channels']
    input_channels = node['frontend']['input_channels']
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    kernel_size = node['frontend']['kernel_size']
    stride = node['frontend']['stride']
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y'] # This already includes input forced folding/unfolding
    is_folding_conv_x =  'force_folding_x' in node['frontend']
    
    if ('activation_silu' in node['attributes']) and (node['attributes']['activation_silu'] != None):
        input_scale = node['attributes']['activation_silu']['input_scale'][0]
        input_zp = node['attributes']['activation_silu']['input_zp'][0]
        output_scale = node['attributes']['activation_silu']['output_scale'][0]
        output_zp = node['attributes']['activation_silu']['output_zp'][0]
        if ('lut' in node['attributes']['activation_silu']):
            node['frontend']['lut_silu'] = node['attributes']['activation_silu']['lut']
        else:
            node['frontend']['lut_silu'] = LUTPopulateInt8(input_scale, input_zp, output_scale, output_zp)
    # If its a conv with stride we implement this by removing output channels of folded tensor. We have 2 options to implement it:
    # 1) if the tensor is already folded, reducing output channel by weights config. This has the advantage of reduced calculations (we dont calculate channels that will be dropped)
    # 2) if the tensor is not folded in x or y axis, we need to do output folding and after that, in CBC we output only part of the folded channels. In TSNP this is only possible in X axis
    weights_stride_x = 1 # If this equals 2 it means that stride is implemented by removing weights. This can be done only if input folding factor>0 if not the conv must be folding 
    weights_stride_y = 1
    if stride == 2: # If its a conv with
        weights_stride_x = 2
        weights_stride_y = 2
        if is_folding_conv_x:
            weights_stride_x=1
        if input_folding_factor_y<1:
                raise ValueError ('stride=2 conv with input folding factor_y <1 cant be executed')
    node['frontend']['weights_stride_x'] = weights_stride_x
    original_weights_tensor = node['frontend']['weights_tensor']
    w_int8 = original_weights_tensor.data
    original_biases_tensor = node['frontend']['biases_tensor']
    bias =  original_biases_tensor.data * original_biases_tensor.scale
    kernel_shape = w_int8.shape
    if (input_folding_factor_x>0 or input_folding_factor_y>0) and ('folded_weights_tensor' not in node['frontend']):
        asymmetric_padding = False
        if TFLITE_REQUANT:
            padding = node['attributes']['pads']
            if (len(padding) > 0) and (padding[0] != padding[1]):
                asymmetric_padding = True
        folded_weights = folding_algo.get_asym_folded_weights(w_int8,input_folding_factor_x=input_folding_factor_x,input_folding_factor_y=input_folding_factor_y, \
                                                            stride_x=weights_stride_x,stride_y=weights_stride_y, asymmetric_padding=asymmetric_padding)
        kernel_shape = folded_weights.shape
        folded_weights_tensor_name = original_weights_tensor.name+'_folded'
        folded_weights_tensor = Tensor(folded_weights_tensor_name,folded_weights,is_constant=True,shape = folded_weights.shape,scale = original_weights_tensor.scale,
                                 zero_point=original_weights_tensor.zero_point)
        node['frontend']['folded_weights_tensor'] = folded_weights_tensor

        folded_biases = folding_algo.get_asym_folded_per_oc_params(bias,input_folding_factor_x=input_folding_factor_x,input_folding_factor_y=input_folding_factor_y,stride_x=weights_stride_x,stride_y=weights_stride_y)
        folded_biases_tensor_name = original_biases_tensor.name+'_folded'
        folded_biases_tensor = Tensor(folded_biases_tensor_name,folded_biases,is_constant=True,shape = folded_biases.shape,scale = original_biases_tensor.scale,
                                 zero_point=original_biases_tensor.zero_point)
        node['frontend']['folded_biases_tensor'] = folded_biases_tensor
        node['frontend']['folded_kernel_size'] = folded_weights.shape[3]
        if TFLITE_REQUANT:
            if asymmetric_padding:
                node['frontend']['folded_padding'] = [int(value*2) for value in padding]
            else:
                if (w_int8.shape[2] == 6):
                    node['frontend']['folded_padding'] = [int(value/2) for value in padding]
                elif (w_int8.shape[2] == 7):
                    node['frontend']['folded_padding'] = [int(value/3) for value in padding]
                else:
                    node['frontend']['folded_padding'] = padding
        else:
            node['frontend']['folded_padding'] = folded_weights.shape[3] // 2

    if kernel_shape[3]!=kernel_shape[2]:
        raise ValueError ('Asymetric kernel is not supported (%s)' % str(kernel_shape))
    if kernel_shape[3] not in [1,3]:
        raise ValueError ('folded kernel shape is not supported (%s)' % str(kernel_shape))
    input_tensor_scale = node['frontend']['input_tensor_scale']
    input_tensor_zp = node['frontend']['input_tensor_zp']
    weights_per_channel_scale = node['frontend']['weights_per_channel_scale']
    weights_per_channel_zp = node['frontend']['weights_per_channel_zp']
    output_tensor_scale = node['frontend']['output_tensor_scale']
    output_tensor_zp = node['frontend']['output_tensor_zp']

    # TODO: These are not needed for TFLITE_REQUANT
    requant_bias_int12      = [0 for i in range(output_channels)]
    # New MAC shift logic
    requant_scale_shift   = [0 for i in range(output_channels)]
    requant_scale_float   = [0 for i in range(output_channels)]

    # MCHP numerics
    requant_scale_uint14   = [0 for i in range(output_channels)]
    mac_rough_shift_mux = [0 for i in range(output_channels)]

    if MCHP_NUMERICS and TFLITE_REQUANT:
        output_multiplier = [0 for i in range(output_channels)]
        cInputH = [0 for i in range(output_channels)]
        cInputL = [0 for i in range(output_channels)]
        o_shift = [0 for i in range(output_channels)]
    
    for och in range(output_channels):
        w_sum = np.sum(w_int8[och])
        zeroed_oc = np.all(w_int8[och,:]==0)
        if zeroed_oc:
            requant_scale = input_tensor_scale*0.0039/output_tensor_scale.item() #TODO: We can set requant_scale to 0.02 in case all weights are zero (see neuronix_models capture code)
        else:
            try:
                requant_scale = input_tensor_scale*weights_per_channel_scale[och].item()/output_tensor_scale.item() #TODO: We can set requant_scale to 0.02 in case all weights are zero (see neuronix_models capture code)
            except:
                # yolov5 has per-channel scale, but yolov8 has only 1
                assert weights_per_channel_scale.size == 1
                requant_scale = input_tensor_scale*weights_per_channel_scale[0].item()/output_tensor_scale.item()
        # Dans: note to self: in the below, if we change the +2 to +1 we sometimes exceed the expected max mac bits. . This could be logical if using histogram calibration but we got it in min/max calibration. need to check why...
        # Note that this value in the simulator is calculated in neuronix_ops.py so if changed need to change also there
        if REDUCED_MAC_RESCALE_BUS_WIDTH:
            assert False
        elif MCHP_NUMERICS:
            requant_scale_f = min(math.trunc(-math.log(requant_scale, 2)+BIAS_FRACTIONAL_BITS) + FRACTIONAL_BITS, 31) # See above note
        else:
            assert False
        # New MAC shift logic
        if zeroed_oc:
            mac_rough_shift_mux[och] = 0
            requant_scale_uint14[och] = 0
        else:
            if REDUCED_MAC_RESCALE_BUS_WIDTH:
                assert False
            elif MCHP_NUMERICS:
                expected_mac_bits = requant_scale_f + FINAL_RESULTS_BITS + OVERFLOW_EXTRA_BITS - INT_SCALE_BITS # This is the expected number of bits in the mac result
                # We split the shift right to 2 parts. We shift as minimal as possible in the mac in order reduce the bus size between MAC and RQ
                # The rest will be shifted in RQ after we multiply in integer scale.
                # NOTE: Negative shift is possible if its -1 or -2 since rough shift will be 0 and fine shift left will be done by integer scale.
                per_och_mac_shift = np.array(expected_mac_bits - MCHP_MAC_TO_RQ_BUS_WIDTH) # We shift right to limit MAC to MCHP_MAC_TO_RQ_BUS_WIDTH bits
                # We achieve the per och mac shift by rough shift right between MAC and RQ (multiplexer) and fine shift left by adding bits to the int scale (UINT13 instead of UINT10)
                rough_shift_right = int(np.ceil(per_och_mac_shift/MAC_ROUGH_SHIFT_GRANULARITY))
                rough_shift_right_bits = rough_shift_right * MAC_ROUGH_SHIFT_GRANULARITY
                fine_shift_left = rough_shift_right_bits - per_och_mac_shift

                requant_scale_uint14[och]  = math.trunc(requant_scale*math.pow(2,(requant_scale_f+fine_shift_left)) + 0.0) # Was +0.5
                mac_rough_shift_mux[och] = rough_shift_right
            else:
                assert False

        requant_scale_shift[och] = requant_scale_f
        requant_scale_float[och] = requant_scale

        if MCHP_NUMERICS and (not TFLITE_REQUANT):
            requant_bias  = bias[och].item()/output_tensor_scale.item() + output_tensor_zp.item()
            try:
                requant_bias -= w_sum*input_tensor_zp*weights_per_channel_scale[och].item()*input_tensor_scale/output_tensor_scale.item()
            except:
                assert weights_per_channel_scale.size == 1
                requant_bias -= w_sum*input_tensor_zp*weights_per_channel_scale[0].item()*input_tensor_scale/output_tensor_scale.item()
            requant_bias_int12[och] = math.trunc(BIAS_MULITIPLIER*requant_bias+BIAS_ROUNDING_ADD)
        elif MCHP_NUMERICS and TFLITE_REQUANT:
            requant_bias  = original_biases_tensor.data[och].item()
            requant_bias -= w_sum*input_tensor_zp 
            requant_bias = requant_bias.astype(np.int32)
            
            # c_input = (bias_data*output_multiplier + output_offset)<<o_shift + 1<<(o_shift-1)
            acc = None # This is the accumulator
            bias_data = int(requant_bias)
            scale = requant_scale
            output_offset = output_tensor_zp.item()
            
            output_activation_min = int8min
            output_activation_max = int8max

            try:
                if ir.graph.nodes[node_name]['attributes']['fused_activation_function'] == 'RELU':
                    output_activation_min = output_offset
                elif ir.graph.nodes[node_name]['attributes']['fused_activation_function'] == 'RELU6':
                    output_activation_min = output_offset
                    output_activation_max = round((6 / output_tensor_scale.item()) + output_offset)
                    output_activation_max = min(output_activation_max, int8max)
            except:
                pass

            # These are used after the function returns:
            # Inputs:
            #   acc, output_multiplier, cInputH, cInputL, o_shift, output_activation_min, output_activation_max
            # Outputs:
            #   acc, output_activation_min, output_activation_max
            output_multiplier[och], cInputH[och], cInputL[och], o_shift[och] = \
                quantize2MathBlock(acc,bias_data,scale,output_offset,output_activation_min,output_activation_max)

    if MCHP_NUMERICS and TFLITE_REQUANT:
        node['frontend']['output_multiplier'] = output_multiplier
        node['frontend']['cInputH'] = cInputH
        node['frontend']['cInputL'] = cInputL
        node['frontend']['o_shift'] = o_shift

    if DEBUG_CLIP_BIAS_TO_MAX_BIAS_BITS and np.any(np.abs(np.array(requant_bias_int12))>=np.power(2,MAX_BIAS_BITS)):
        max_bias_int12 = 2 ** MAX_BIAS_BITS-1
        min_bias_int12 = -1*max_bias_int12
        requant_bias_int12 = list(np.clip(np.array(requant_bias_int12),min_bias_int12,max_bias_int12)) # Clip bias to MAX_BIAS_BITS
        print('At layer %s, bias exceeded INT%d ' % (node_name,MAX_BIAS_BITS+1))

    # TODO: Not needed for TFLITE_REQUANT
    node['frontend']['requant_bias_int12'] = requant_bias_int12
    if REDUCED_MAC_RESCALE_BUS_WIDTH:
        assert False
    elif MCHP_NUMERICS:
        max_scale = max(requant_scale_uint14)
        if max_scale!=0:
            if math.log(max_scale,2)>=(INT_SCALE_BITS+MCHP_ADDED_SCALE_BITS):
                raise ValueError('max_scale: %d is more than 10 bits' % (max_scale))
    else:
        assert False

    # New MAC shift logic
    if not TFLITE_REQUANT:
        node['frontend']['requant_scale_shift'] = requant_scale_shift
        node['frontend']['requant_scale_float'] = requant_scale_float
        # MCHP numerics
        node['frontend']['requant_scale_uint14'] = requant_scale_uint14
        node['frontend']['mac_rough_shift_mux'] = mac_rough_shift_mux

    if (input_folding_factor_x>0 or input_folding_factor_y>0):
        if not TFLITE_REQUANT:
            node['frontend']['folded_requant_scale_uint14'] = folding_algo.get_asym_folded_per_oc_params(requant_scale_uint14,
                                                                                                        input_folding_factor_x=input_folding_factor_x,
                                                                                                        input_folding_factor_y=input_folding_factor_y,stride_x=weights_stride_x,stride_y=weights_stride_y)
            node['frontend']['folded_requant_scale_shift'] = folding_algo.get_asym_folded_per_oc_params(requant_scale_shift,
                                                                                                        input_folding_factor_x=input_folding_factor_x,
                                                                                                        input_folding_factor_y=input_folding_factor_y,stride_x=weights_stride_x,stride_y=weights_stride_y)
            node['frontend']['folded_mac_rough_shift_mux'] = folding_algo.get_asym_folded_per_oc_params(mac_rough_shift_mux,
                                                                                                        input_folding_factor_x=input_folding_factor_x,
                                                                                                        input_folding_factor_y=input_folding_factor_y,stride_x=weights_stride_x,stride_y=weights_stride_y)
            node['frontend']['folded_requant_scale_float'] = folding_algo.get_asym_folded_per_oc_params(requant_scale_float,
                                                                                                        input_folding_factor_x=input_folding_factor_x,
                                                                                                        input_folding_factor_y=input_folding_factor_y,stride_x=weights_stride_x,stride_y=weights_stride_y)

        # TODO: Not needed for TFLITE_REQUANT
        node['frontend']['folded_requant_bias_int12'] = folding_algo.get_asym_folded_per_oc_params(requant_bias_int12,
                                                                                                   input_folding_factor_x=input_folding_factor_x,
                                                                                                   input_folding_factor_y=input_folding_factor_y,stride_x=weights_stride_x,stride_y=weights_stride_y)

        if TFLITE_REQUANT:
            # Take params list of length [output channels] and replicate it based on folding
            # Currently using the same key name, can later prefix with 'folding_' like above if needed
            node['frontend']['output_multiplier'] = folding_algo.get_asym_folded_per_oc_params(output_multiplier,
                                                                                            input_folding_factor_x=input_folding_factor_x,
                                                                                            input_folding_factor_y=input_folding_factor_y,stride_x=weights_stride_x,
                                                                                            stride_y=weights_stride_y)
            node['frontend']['cInputH'] = folding_algo.get_asym_folded_per_oc_params(cInputH,
                                                                                    input_folding_factor_x=input_folding_factor_x,
                                                                                    input_folding_factor_y=input_folding_factor_y,
                                                                                    stride_x=weights_stride_x,
                                                                                    stride_y=weights_stride_y)
            node['frontend']['cInputL'] = folding_algo.get_asym_folded_per_oc_params(cInputL,
                                                                                    input_folding_factor_x=input_folding_factor_x,
                                                                                    input_folding_factor_y=input_folding_factor_y,
                                                                                    stride_x=weights_stride_x,
                                                                                    stride_y=weights_stride_y)
            node['frontend']['o_shift'] = folding_algo.get_asym_folded_per_oc_params(o_shift,
                                                                                    input_folding_factor_x=input_folding_factor_x,
                                                                                    input_folding_factor_y=input_folding_factor_y,
                                                                                    stride_x=weights_stride_x,
                                                                                    stride_y=weights_stride_y)
    
    # Identifying average_pool layer from the regular convolutions
    is_avgPool = False
    if (kernel_shape[3] == 3):
        avgPool_weights = np.zeros((3,3), dtype=np.int8)
        avgPool_weights[1, 1] = 127
        avgPool_weights[1, 2] = 127
        avgPool_weights[2, 1] = 127
        avgPool_weights[2, 2] = 127
        is_avgPool = True
        for i in range(kernel_shape[0]):
            for j in range(kernel_shape[1]):
                if j == i:
                    if not np.array_equal(w_int8[i,j,:,:], avgPool_weights):
                        is_avgPool = False
                        break
            if not is_avgPool:
                break
    node['frontend']['is_avgPool'] = is_avgPool
    if is_avgPool:
        node['frontend']['output_tensor'].is_avgPool_output = True
        node['frontend']['output_tensor'].shape_real_x16[2] += 1
        

def calc_gemm_folding_factor(ir,node_name,node):
    if len(node['frontend']['preceding_nodes_params'])>0: # If its an input to the workload we calc folding factor based on resolution, otherwise we take it from input tensor
        input_folding_factor = node['frontend']['input_tensor'].folding_factor
    else:
        input_folding_factor=0
    if input_folding_factor!=0:
        raise ValueError ('Gemm with folding factor!=0 not supported currently')

    node['frontend']['input_tensor'].folding_factor = input_folding_factor
    node['frontend']['input_folding_factor'] = input_folding_factor
    output_folding_factor=0
    node['frontend']['output_tensor'].folding_factor = output_folding_factor
    node['frontend']['output_folding_factor'] = output_folding_factor

def calc_gemm_qparams(ir,node_name,node):
    output_channels = node['frontend']['output_channels']
    input_channels = node['frontend']['input_channels']

    original_weights_tensor = node['frontend']['weights_tensor']
    w_int8 = original_weights_tensor.data
    original_biases_tensor = node['frontend']['biases_tensor']
    bias =  original_biases_tensor.data * original_biases_tensor.scale

    input_tensor_scale = node['frontend']['input_tensor_scale']
    input_tensor_zp = node['frontend']['input_tensor_zp']
    weights_per_channel_scale = node['frontend']['weights_per_channel_scale']
    weights_per_channel_zp = node['frontend']['weights_per_channel_zp']
    output_tensor_scale = node['frontend']['output_tensor_scale']
    output_tensor_zp = node['frontend']['output_tensor_zp']

    requant_bias_int12      = [0 for i in range(output_channels)]
    # New MAC shift logic
    requant_scale_shift   = [0 for i in range(output_channels)]
    requant_scale_float   = [0 for i in range(output_channels)]

    max_bias = 2 ** MAX_BIAS_BITS -1
    min_bias = -1 * max_bias

    
    for och in range(output_channels):
        w_sum = 0
        for ich in range(input_channels):
                    w_sum += w_int8[ich][och] # Weights tensor in Gemm is (#ic,#oc) and not (#oc,#ic,kx,ky)
        requant_scale = input_tensor_scale*weights_per_channel_scale[och].item()/output_tensor_scale.item() #TODO: We can set requant_scale to 0.02 in case all weights are zero (see neuronix_models capture code)
        # Dans: note to self: in the below, if we change the +2 to +1 we sometimes exceed the expected max mac bits. . This could be logical if using histogram calibration but we got it in min/max calibration. need to check why...
        # Note that this value in the simulator is calculated in neuronix_ops.py so if changed need to change also there
        if REDUCED_MAC_RESCALE_BUS_WIDTH:
            requant_scale_f = min(math.trunc(-math.log(requant_scale, 2)+BIAS_FRACTIONAL_BITS) + FRACTIONAL_BITS, 31) # See above note
        else:
            requant_scale_f = min(math.trunc(-math.log(requant_scale, 2)+1) + FRACTIONAL_BITS, 31) # See above note
        requant_scale_shift[och] = requant_scale_f
        requant_scale_float[och] = requant_scale

        #requant_bias  = np.clip(bias[och].item()/output_tensor_scale.item() + output_tensor_zp.item(),-255,255)
        requant_bias  = bias[och].item()/output_tensor_scale.item() + output_tensor_zp.item()
        requant_bias -= w_sum*input_tensor_zp*weights_per_channel_scale[och].item()*input_tensor_scale/output_tensor_scale.item()
        requant_bias_int12[och] = math.trunc(BIAS_MULITIPLIER*requant_bias+BIAS_ROUNDING_ADD)
    
    if DEBUG_CLIP_BIAS_TO_MAX_BIAS_BITS and np.any(np.log2(np.abs(np.array(requant_bias_int12)))>=MAX_BIAS_BITS):
        max_bias_int12 = 2 ** MAX_BIAS_BITS-1
        min_bias_int12 = -1*max_bias_int12
        requant_bias_int12 = list(np.clip(np.array(requant_bias_int12),min_bias_int12,max_bias_int12)) # Clip bias to MAX_BIAS_BITS
        print('At layer %s, bias exceeded INT%d ' % (node_name,MAX_BIAS_BITS+1))

    node['frontend']['requant_bias_int12'] = requant_bias_int12
    # New MAC shift logic
    node['frontend']['requant_scale_shift'] = requant_scale_shift
    node['frontend']['requant_scale_float'] = requant_scale_float

def calc_maxpool_folding_factor(ir,node_name,node):
    # First we make sure that both input tensors have same shape
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    original_input_height = original_input_tensor_shape[2]
    original_input_channels = original_input_tensor_shape[1]
    original_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape()
    original_output_channels = original_output_tensor_shape[1]
    stride = node['frontend']['stride']
    original_input_width = original_input_tensor_shape[3]
    original_kernel_size = node['frontend']['kernel_size']
    if len(node['frontend']['preceding_nodes_params'])>0: # If its an input to the workload we calc folding factor based on resolution, otherwise we take it from input tensor
        input_folding_factor_x = node['frontend']['input_tensor'].folding_factor_x
        input_folding_factor_y = node['frontend']['input_tensor'].folding_factor_y
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
        x_slices = node['frontend']['input_tensor'].x_slices
    else:
        input_folding_factor_x,input_folding_factor_y, x_slices = get_new_input_folding_factors(original_input_width,original_input_height,stride,original_kernel_size,original_input_channels,original_output_channels)
        if 'MaxPool16x16' in ir.model_name:
            if node_name == 'MaxPool1':
                input_folding_factor_x = 0
                input_folding_factor_y = 0
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
        node['frontend']['input_tensor'].folding_factor_x = input_folding_factor_x
        node['frontend']['input_tensor'].folding_factor_y = input_folding_factor_y
        node['frontend']['input_tensor'].y_tiles = num_y_tiles
        node['frontend']['input_tensor'].x_slices = x_slices

    if stride == 1:
        output_folding_factor_x = input_folding_factor_x
        output_folding_factor_y = input_folding_factor_y
    elif stride == 2:
        output_folding_factor_x = input_folding_factor_x - 1
        output_folding_factor_y = input_folding_factor_y - 1
    else:
        raise ValueError ('Stride >2 is not supported')

    if node_name in ir.force_y_folding:
        output_folding_factor_y = output_folding_factor_y +1
        node['frontend']['force_folding_y'] = True

    if node_name in ir.force_y_unfolding:
        output_folding_factor_y = output_folding_factor_y -1
        node['frontend']['force_unfolding_y'] = True

    following_nodes_params = node['frontend']['following_nodes_params']
    output_xslices = x_slices
    max_following_node_stride = 0
    for following_node_params in following_nodes_params:
        current_following_node = ir.graph.nodes[following_node_params[0]] # [0] is node name, [1] is which input of folloing node is connected to current node
        if 'stride' in current_following_node['frontend'] and current_following_node['frontend']['stride']>max_following_node_stride:
            max_following_node_stride = current_following_node['frontend']['stride']
            if max_following_node_stride>2:
                raise ValueError ('Stride>2 is not currently supported')
    
    add_folding_conv_before_node = False
    # We change to folding conv if we see in following ops a stride=2 (In case of width>14 => 28x28 grid) or if current op is stride==2
    # if current op is stride==2 and the data is not folded we will perform a folding conv with stride=2 which means un-needed output channels will be calculated but dropped in RQ
    if DEBUG_FORCE_FOLDCONV: #if (output_folding_factor==0 and max_following_node_stride==2 and original_input_width>MINIMAL_ACTUAL_INPUT_WIDTH) or DEBUG_FORCE_FOLDCONV:
        node['frontend']['force_folding_x'] = True
        output_folding_factor_x += 1
        output_xslices = output_xslices // 2
    elif ((output_folding_factor_x==-1 or output_folding_factor_y==-1) and stride==2): # In that case the output folding factor is not increased since we drop 3/4 of the output channels
        if output_folding_factor_x==-1:
            input_folding_factor_x+=1
            output_folding_factor_x+=1
            add_folding_conv_before_node = True
            output_xslices = math.ceil(output_xslices / 2)          
        if output_folding_factor_y==-1:
            input_folding_factor_y+=1
            output_folding_factor_y+=1
            node['frontend']['force_folding_y'] = True
    else:
        if 'force_folding_x' in node['frontend']: # There are cases where 1st pass of folding factor calc fails and at later pass a folding conv is change to non folding conv
            del node['frontend']['force_folding_x']
        if 'force_folding_y' in node['frontend']: # There are cases where 1st pass of folding factor calc fails and at later pass a folding conv is change to non folding conv
            del node['frontend']['force_folding_y']
    
    node['frontend']['output_tensor'].folding_factor_x = output_folding_factor_x 
    node['frontend']['output_tensor'].folding_factor_y = output_folding_factor_y # It is important to update the output tensor folding factor since it is used by below "is_y_unfolding_needed"
    node['frontend']['output_tensor'].x_slices = output_xslices
    node['frontend']['output_folding_factor_x'] = output_folding_factor_x
    node['frontend']['output_folding_factor_y'] = output_folding_factor_y
    if (add_folding_conv_before_node):
        node['frontend']['x_slices'] = output_xslices
    else:
        node['frontend']['x_slices'] = x_slices
    num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
    node['frontend']['input_tensor'].y_tiles = num_y_tiles
    node['frontend']['y_tiles'] = num_y_tiles
    node['frontend']['input_folding_factor_x'] = input_folding_factor_x
    node['frontend']['input_folding_factor_y'] = input_folding_factor_y
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    node['frontend']['y_tiles'] = num_y_tiles
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    node['frontend']['output_tensor'].x_slices = output_xslices
    
    if add_folding_conv_before_node:
        add_xfoldingConv_before_node(ir, node)
    
def calc_avgpool_qparams(ir,node_name,node):
    calc_fake_conv_weights(ir,node_name,node)
    num_outputs = node['frontend']['output_channels']
    
    # We set the requant params to have 1 to 1
    node['frontend']['output_multiplier'] = node['frontend']['output_multiplier']//8
    node['frontend']['cInputH'] = 0
    node['frontend']['cInputL'] = 0
    # node['frontend']['o_shift'] = 0

    node['frontend']['weights_tensor'].data = 0*node['frontend']['weights_tensor'].data
    for c in range(num_outputs):
        # node['frontend']['weights_tensor'].data[c, c, 0, 0]  = 8//4
        # node['frontend']['weights_tensor'].data[c, c, 0, 1]  = 8//4
        # node['frontend']['weights_tensor'].data[c, c, 1, 0]  = 8//4
        # node['frontend']['weights_tensor'].data[c, c, 1, 1]  = 8//4

        node['frontend']['weights_tensor'].data[c, c, 1, 1]  = 8//4
        node['frontend']['weights_tensor'].data[c, c, 1, 2]  = 8//4
        node['frontend']['weights_tensor'].data[c, c, 2, 1]  = 8//4
        node['frontend']['weights_tensor'].data[c, c, 2, 2]  = 8//4
        
    calc_folded_weights_and_sparsity(ir,node_name,node)    



def calc_maxpool_qparams(ir,node_name,node):
    calc_fake_conv_weights(ir,node_name,node)
    calc_folded_weights_and_sparsity(ir,node_name,node)

def calc_fake_conv_weights(ir,node_name,node):    
    # First we make sure that both input tensors have same shape
    
    kernel_size = node['frontend']['kernel_size']
    stride = node['frontend']['stride']

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    
    requant_bias_int12 = 0
    requant_scale_uint14 = 2 ** (FRACTIONAL_BITS+1+4) # This will cause zero shift in the RQ stage

    requant_scale_f = MAX_REDUCE_BUS_WIDTH - 3 # This is so we dont shift right at mac
    requant_scale_shift = requant_scale_f
        
    node['frontend']['requant_bias_int12'] = requant_bias_int12

    # New MAC shift logic
    node['frontend']['requant_scale_uint14'] = requant_scale_uint14
    node['frontend']['mac_rough_shift_mux'] = 0
    node['frontend']['requant_scale_shift'] = requant_scale_shift
    
    if MCHP_NUMERICS and TFLITE_REQUANT:
        acc = None # This is the accumulator
        bias_data = 0
        scale = 1.0
        output_offset = 0
        output_activation_min = int8min
        output_activation_max = int8max
        output_multiplier, cInputH, cInputL, o_shift = \
            quantize2MathBlock(acc,bias_data,scale,output_offset,output_activation_min,output_activation_max)
            
        node['frontend']['output_multiplier'] = output_multiplier
        node['frontend']['cInputH'] = cInputH
        node['frontend']['cInputL'] = cInputL
        node['frontend']['o_shift'] = o_shift
        
    if (input_folding_factor_x>0 or input_folding_factor_y>0):
        node['frontend']['folded_requant_scale_uint14'] = requant_scale_uint14
        node['frontend']['folded_mac_rough_shift_mux'] = 0
        node['frontend']['folded_requant_scale_shift'] = requant_scale_shift
        node['frontend']['folded_requant_bias_int12'] = requant_bias_int12

    # Create the weights tensor
    output_channels = node['frontend']['output_tensor'].get_original_shape()[1] # This will be folded output channels but without output folding
    input_channels = node['frontend']['input_tensor'].get_original_shape()[1]

    weights_tensor_shape = [output_channels,input_channels,kernel_size,kernel_size]
    weights_tensor = np.zeros(weights_tensor_shape,dtype=np.int64)
    for oc in range(output_channels):
        for k_y in range(kernel_size):
            for k_x in range(kernel_size):
                weights_tensor[oc,oc,k_y,k_x] = 8 # This in addition to requant_scale_uint10 and requant_scale_shift will cause copy of input to output

    w_int8_np = weights_tensor.astype(np.int8)
    weights_tensor_name = node_name + '_fake_weights_tensor'
    original_weights_tensor = Tensor(weights_tensor_name,w_int8_np,is_constant=True,shape = weights_tensor_shape)
    ir.tensors[weights_tensor_name] = original_weights_tensor
    node['frontend']['weights_tensor'] = original_weights_tensor

def calc_folded_weights_and_sparsity(ir,node_name,node): 
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    weights_tensor_name = node_name + '_fake_weights_tensor'
    original_weights_tensor = ir.tensors[weights_tensor_name]
    weights_tensor = original_weights_tensor
    stride = node['frontend']['stride']

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']

    dense_weights = original_weights_tensor.data.size
    sparse_weights = np.count_nonzero(weights_tensor.data)
    node['frontend']['dense_weights'] = dense_weights
    node['frontend']['sparse_weights'] = sparse_weights
    node['frontend']['weights_sparsity'] = 1- (sparse_weights/dense_weights)
    sparse_macs = sparse_weights*original_input_tensor_shape[2]*original_input_tensor_shape[3]
    node['frontend']['dense_macs'] = sparse_macs
    node['frontend']['sparse_macs'] = sparse_macs
    node['frontend']['macs_sparsity'] = 0

    if (input_folding_factor_x>0 or input_folding_factor_y>0):
        folded_weights = folding_algo.get_asym_folded_weights(weights_tensor.data,input_folding_factor_x=input_folding_factor_x,input_folding_factor_y=input_folding_factor_y,stride_x=stride,stride_y=stride)
        folded_weights_tensor_name = weights_tensor_name+'_folded'
        folded_weights_tensor = Tensor(folded_weights_tensor_name,folded_weights,is_constant=True,shape = folded_weights.shape,scale = original_weights_tensor.scale,
                                 zero_point=original_weights_tensor.zero_point)
        node['frontend']['folded_weights_tensor'] = folded_weights_tensor

        node['frontend']['folded_kernel_size'] = folded_weights.shape[3]
        node['frontend']['folded_padding'] = folded_weights.shape[3] // 2


def calc_resize_folding_factor(ir,node_name,node):

    # First we make sure that both input tensors have same shape
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    original_input_height = original_input_tensor_shape[2]
    original_input_width = original_input_tensor_shape[3]

    # We check y force folding here since it affects the input folding
    need_to_fold_y_for_concat = node['name'] in ir.marked_nodes_for_folding_y
    force_y_folding = need_to_fold_y_for_concat or node_name in ir.force_y_folding
    force_y_unfolding = node_name in ir.force_y_unfolding

    node['frontend']['kernel_size'] = 1
    node['frontend']['stride'] = 1
    node['frontend']['padding'] = 0
    
    if force_y_folding:
        node['frontend']['force_folding_y'] = True

    if force_y_unfolding:
        node['frontend']['force_unfolding_y'] = True

    if len(node['frontend']['preceding_nodes_params'])>0: # If its an input to the workload we calc folding factor based on resolution, otherwise we take it from input tensor
        input_folding_factor_x = node['frontend']['input_tensor'].folding_factor_x
        input_folding_factor_y = node['frontend']['input_tensor'].folding_factor_y
        if force_y_folding:
            input_folding_factor_y+=1
        if force_y_unfolding:
            input_folding_factor_y-=1
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
        num_x_slices = node['frontend']['input_tensor'].x_slices
    else:
        # Check original input width and height
        input_folding_factor_x = math.ceil(math.log(original_input_width / MAX_GRID_WIDTH,2))
        if input_folding_factor_x<0: # If input resolution is less than MAX_GRID_WIDTH, folding is not needed
            input_folding_factor_x=0
        input_folding_factor_y = math.ceil(math.log(original_input_height / MAX_GRID_HEIGHT,2)) # TODO: Should be original_input_height?
        if input_folding_factor_y<0: # If input resolution is less than MAX_GRID_WIDTH, folding is not needed
            input_folding_factor_y=0
        original_kernel_size = node['frontend']['kernel_size']
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
        num_x_slices = int(original_input_width/(16*(2**input_folding_factor_x)))
        node['frontend']['input_tensor'].folding_factor_x = input_folding_factor_x # We update the tensor's folding factor only if it was generated here
        node['frontend']['input_tensor'].folding_factor_y = input_folding_factor_y

    node['frontend']['input_folding_factor_x'] = input_folding_factor_x # Node's input folding factor will contain effects of forced folding at input at y axis
    node['frontend']['input_folding_factor_y'] = input_folding_factor_y # This already includes input forced folding/unfolding
    node['frontend']['x_slices'] = num_x_slices
    node['frontend']['y_tiles'] = num_y_tiles
    folded_input_width = node['frontend']['input_tensor'].get_folded_shape()[3]
    smallest_x_hw_resizable_grid_size = SMALLEST_X_RESIZABLE_GRID_SIZE
    if folded_input_width>=8: # In >7x7 config (14x14) we use channel duplication to impement the folding
        output_folding_factor_x = input_folding_factor_x + 1
    else:
        is_hw_x_resize = True
        node['frontend']['is_hw_x_resize'] = 1
        output_folding_factor_x = input_folding_factor_x

    if DEBUG_MINIMIZE_Y_FOLDING:
        # This is an attempt to reduce cases of Y folding, in this case to not do Y folding
        # for resize. However, this can cause a problems when the number of tiles changes.
        # The case that failed was as follows:
        #
        # The number of lines changed from 16 to 32 as a result of the resize, but there was
        # no folding, so there were 3 output tiles. The number of lines written for each tile
        # was [14, 4, 14], with start lines [0, 14, 18]. However, get_y_tile_sizes expects the
        # input and output number of lines to be the same, and normally 16 lines would be 2 tiles,
        # not 3. So when it was called by add_input_load_from_ddr_commands to get the read start
        # lines, it returned [2, 14] for the number of lines and start lines [0, 2]. But the
        # calling function add_input_load_from_ddr_commands needed the read start line for the
        # 3rd tile too, and this resulted in an error.
        output_folding_factor_y = input_folding_factor_y
    else:
        output_folding_factor_y = input_folding_factor_y + 1

    node['frontend']['output_tensor'].y_tiles = num_y_tiles

    max_following_node_stride = 0
    following_nodes_params = node['frontend']['following_nodes_params']
    add_unfolding_nodes = False
    for following_node_params in following_nodes_params:
        current_following_node = ir.graph.nodes[following_node_params[0]] # [0] is node name, [1] is which input of folloing node is connected to current node
        if 'stride' in current_following_node['frontend'] and current_following_node['frontend']['stride']>max_following_node_stride:
            max_following_node_stride = current_following_node['frontend']['stride']
            if max_following_node_stride>2:
                raise ValueError ('Stride>2 is not currently supported')
        if (current_following_node['op_type'] == 'Concat') or (current_following_node['op_type'] == 'Add'):
            add_unfolding_nodes = True
        if ('_requant' in following_node_params[0]):
            current_following_node['frontend']['force_unfolding_x'] = True

    if output_folding_factor_x==0 and max_following_node_stride==2:
        node['frontend']['force_folding_x'] = True
        output_folding_factor_x = output_folding_factor_x + 1
    if output_folding_factor_y==0 and max_following_node_stride==2:
        node['frontend']['force_folding_y'] = True
        output_folding_factor_y = output_folding_factor_y + 1

    node['frontend']['output_tensor'].folding_factor_x = output_folding_factor_x
    node['frontend']['output_tensor'].folding_factor_y = output_folding_factor_y
    node['frontend']['output_tensor'].x_slices = num_x_slices
    node['frontend']['output_folding_factor_x'] = output_folding_factor_x
    node['frontend']['output_folding_factor_y'] = output_folding_factor_y

    if DEBUG_ADD_UNFOLDING_X:
        if add_unfolding_nodes:
            prev_node = node
            for unfold_idx in range(output_folding_factor_x):
                name_sufix = 'unfold_x_'+str(unfold_idx)
                insert_identity_node(ir=ir, param_source_node = prev_node, name_sufix = name_sufix)
                identity_node_name = prev_node['name']+'_'+ name_sufix
                identity_node = ir.graph.nodes[identity_node_name]
                identity_node['frontend']['force_unfolding_x'] = True
                prev_node = identity_node

def calc_resize_qparams(ir,node_name,node):
    # First we make sure that both input tensors have same shape
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    kernel_size = 1
    stride = 1
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    is_hw_x_resize = 'is_hw_x_resize' in node['frontend']
    #requant
    requant_scale = 0.25
    requant_bias  = 0
    
    requant_bias_int12 = 0
    requant_scale_uint14 = 2 ** (FRACTIONAL_BITS+1+4) # This will cause zero shift in the RQ stage

    requant_scale_f = MAX_REDUCE_BUS_WIDTH - 3 # This is so we dont shift right at mac
    requant_scale_shift = requant_scale_f
    requant_scale_float = requant_scale
        
    node['frontend']['requant_bias_int12'] = requant_bias_int12

    # New MAC shift logic
    node['frontend']['requant_scale_uint14'] = requant_scale_uint14
    node['frontend']['mac_rough_shift_mux'] = 0
    node['frontend']['requant_scale_shift'] = requant_scale_shift
    node['frontend']['requant_scale_float'] = requant_scale_float
    if (input_folding_factor_x>0 or input_folding_factor_y>0):
        node['frontend']['folded_requant_scale_uint14'] = requant_scale_uint14
        node['frontend']['folded_mac_rough_shift_mux'] = 0
        node['frontend']['folded_requant_scale_shift'] = requant_scale_shift
        node['frontend']['folded_requant_bias_int12'] = requant_bias_int12
        node['frontend']['folded_requant_scale_float'] = requant_scale_float

    if MCHP_NUMERICS and TFLITE_REQUANT:
        acc = None # This is the accumulator
        bias_data = int(requant_bias)
        scale = (1/8.0) #requant_scale
        output_offset = 0
        output_activation_min = int8min
        output_activation_max = int8max
        output_multiplier, cInputH, cInputL, o_shift = \
            quantize2MathBlock(acc,bias_data,scale,output_offset,output_activation_min,output_activation_max)
            
        node['frontend']['output_multiplier'] = output_multiplier
        node['frontend']['cInputH'] = cInputH
        node['frontend']['cInputL'] = cInputL
        node['frontend']['o_shift'] = o_shift

    # Create the weights tensor
    input_channels = node['frontend']['input_tensor'].get_original_shape()[1]
    scales_tensor_name = node['inputs'][1]
    scales_tensor = ir.tensors[scales_tensor_name].data
    if np.any(scales_tensor != np.array([1,1,2,2])):
        # Found in lpd_sim_13 workload case where resize is specified with empt roi as tensor[1]. Maybe in onnx parser need to pin point to scales input and make it tensor[1]
        # for now, removing this check
        #raise ValueError ('At Resize op, currently only scales=[1,1,2,2] is supported')
        pass
    
    if is_hw_x_resize:
        weights_tensor_shape = [input_channels*2,input_channels,kernel_size,kernel_size] # In "is_hw_x_resize" resize the x resizing is done in rq and not by duplicating channels the *2 is because we still use channel duplication for y axis resizing
    else:
        weights_tensor_shape = [input_channels*4,input_channels,kernel_size,kernel_size] # The *4 is the way we implement scales = [2,2] we duplicate each input channel 4 times at output
    weights_tensor = np.zeros(weights_tensor_shape,dtype=np.int64)

    for ic in range(input_channels):
        if is_hw_x_resize:
            for i in range(2):
                weights_tensor[ic+i*input_channels,ic,0,0] = 8 # # In "is_hw_x_resize" resize the resizing in x axis is done in rq and not by duplicating channels
        else:
            for i in range(4):
                #oc=ic*2+(i&1)+input_channels*2*(i//2) # old concept no slices 
                oc = ic+i*input_channels               #new resize concept with slices 
                weights_tensor[oc,ic,0,0] = 8 # This in addition to requant_scale_uint10 and requant_scale_shift will cause copy of input to output

    w_int8_np = weights_tensor.astype(np.int8)
    weights_tensor_name = node_name + '_fake_weights_tensor'
    original_weights_tensor = Tensor(weights_tensor_name,w_int8_np,is_constant=True,shape = weights_tensor_shape)
    ir.tensors[weights_tensor_name] = original_weights_tensor
    node['frontend']['weights_tensor'] = original_weights_tensor

    dense_weights = original_weights_tensor.data.size
    sparse_weights = np.count_nonzero(weights_tensor.data)
    node['frontend']['dense_weights'] = dense_weights
    node['frontend']['sparse_weights'] = sparse_weights
    node['frontend']['weights_sparsity'] = 1- (sparse_weights/dense_weights)
    sparse_macs = sparse_weights*original_input_tensor_shape[2]*original_input_tensor_shape[3]/(stride*stride)
    node['frontend']['dense_macs'] = 0
    node['frontend']['sparse_macs'] = sparse_macs
    node['frontend']['macs_sparsity'] = 0

    if (input_folding_factor_x>0 or input_folding_factor_y>0):
        folded_weights = folding_algo.get_asym_folded_weights(w_int8_np,input_folding_factor_x=input_folding_factor_x,input_folding_factor_y=input_folding_factor_y,stride_x=stride,stride_y=stride)
        folded_weights_tensor_name = weights_tensor_name+'_folded'
        folded_weights_tensor = Tensor(folded_weights_tensor_name,folded_weights,is_constant=True,shape = folded_weights.shape,scale = original_weights_tensor.scale,
                                 zero_point=original_weights_tensor.zero_point)
        node['frontend']['folded_weights_tensor'] = folded_weights_tensor

        node['frontend']['folded_kernel_size'] = folded_weights.shape[3]
        node['frontend']['folded_padding'] = folded_weights.shape[3] // 2
        # Below is a sanity check for the resize weights calculations
        # We expect each of the ocs to apear only once in the weights 
        ocs_found=[]
        for ic in range(folded_weights.shape[1]):
            target_oc = np.where(folded_weights[:,ic,0,0]==8)[0]
            for oc in target_oc:
                if oc in ocs_found:
                    raise ValueError('Found same oc twice')
                ocs_found.append(oc)
        if len(ocs_found)!= folded_weights.shape[0]:
            raise ValueError ('Not all target ocs found')

def copy_folding_factor_from_prev_node(node):
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    original_input_height = original_input_tensor_shape[2]
    original_kernel_size = 1
    assert len(node['frontend']['preceding_nodes_params'])>0
    input_folding_factor_x = node['frontend']['input_tensor'].folding_factor_x
    input_folding_factor_y = node['frontend']['input_tensor'].folding_factor_y
    num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
    node['frontend']['input_tensor'].folding_factor_x = input_folding_factor_x
    node['frontend']['input_tensor'].folding_factor_y = input_folding_factor_y
    node['frontend']['input_tensor'].y_tiles = num_y_tiles
    node['frontend']['input_folding_factor_x'] = input_folding_factor_x
    node['frontend']['input_folding_factor_y'] = input_folding_factor_y
    node['frontend']['output_tensor'].folding_factor_x = input_folding_factor_x
    node['frontend']['output_tensor'].folding_factor_y = input_folding_factor_y
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    node['frontend']['output_folding_factor_x'] = input_folding_factor_x
    node['frontend']['output_folding_factor_y'] = input_folding_factor_y
    node['frontend']['y_tiles'] = num_y_tiles

def calc_sync_qparams(ir,node_name,node):
    # First we make sure that both input tensors have same shape
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()

    node['frontend']['kernel_size'] = 1
    node['frontend']['stride'] = 1
    node['frontend']['padding'] = 0
    kernel_size = 1 # Like with Add, this is just the default
    stride = 1      # Like with Add, this is just the default
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    #requant
    requant_scale = 0.25
    requant_bias  = 0

    requant_bias_int12 = 0
    requant_scale_uint14 = 2 ** (FRACTIONAL_BITS+1+4) # This will cause zero shift in the RQ stage

    requant_scale_f = MAX_REDUCE_BUS_WIDTH - 3 # This is so we dont shift right at mac
    requant_scale_shift = requant_scale_f
    requant_scale_float = requant_scale

    node['frontend']['requant_bias_int12'] = requant_bias_int12

    # New MAC shift logic
    node['frontend']['requant_scale_uint14'] = requant_scale_uint14
    node['frontend']['mac_rough_shift_mux'] = 0
    node['frontend']['requant_scale_shift'] = requant_scale_shift
    node['frontend']['requant_scale_float'] = requant_scale_float
    if (input_folding_factor_x>0 or input_folding_factor_y>0):
        node['frontend']['folded_requant_scale_uint14'] = requant_scale_uint14
        node['frontend']['folded_mac_rough_shift_mux'] = 0
        node['frontend']['folded_requant_scale_shift'] = requant_scale_shift
        node['frontend']['folded_requant_bias_int12'] = requant_bias_int12
        node['frontend']['folded_requant_scale_float'] = requant_scale_float

    if MCHP_NUMERICS and TFLITE_REQUANT:
        node['frontend']['output_multiplier'] = 1
        node['frontend']['cInputH'] = 0
        node['frontend']['cInputL'] = 0
        node['frontend']['o_shift'] = 8

    # Create the weights tensor
    input_channels = node['frontend']['input_tensor'].get_original_shape()[1]

    weights_tensor_shape = [input_channels*4,input_channels,kernel_size,kernel_size] # The *4 is the way we implement scales = [2,2] we duplicate each input channel 4 times at output
    weights_tensor = np.zeros(weights_tensor_shape,dtype=np.int64)

    for ic in range(input_channels):
        for i in range(4):
            oc=ic*2+(i&1)+input_channels*2*(i//2)
            weights_tensor[oc,ic,0,0] = 8 # This in addition to requant_scale_uint10 and requant_scale_shift will cause copy of input to output

    w_int8_np = weights_tensor.astype(np.int8)
    weights_tensor_name = node_name + '_fake_weights_tensor'
    original_weights_tensor = Tensor(weights_tensor_name,w_int8_np,is_constant=True,shape = weights_tensor_shape)
    ir.tensors[weights_tensor_name] = original_weights_tensor
    node['frontend']['weights_tensor'] = original_weights_tensor

    dense_weights = original_weights_tensor.data.size
    sparse_weights = np.count_nonzero(weights_tensor.data)
    node['frontend']['dense_weights'] = dense_weights
    node['frontend']['sparse_weights'] = sparse_weights
    node['frontend']['weights_sparsity'] = 1- (sparse_weights/dense_weights)
    sparse_macs = sparse_weights*original_input_tensor_shape[2]*original_input_tensor_shape[3]/(stride*stride)
    node['frontend']['dense_macs'] = 0
    node['frontend']['sparse_macs'] = sparse_macs
    node['frontend']['macs_sparsity'] = 0

    if (input_folding_factor_x>0 or input_folding_factor_y>0):
        folded_weights = folding_algo.get_asym_folded_weights(w_int8_np,input_folding_factor_x=input_folding_factor_x,input_folding_factor_y=input_folding_factor_y,stride_x=stride,stride_y=stride)
        folded_weights_tensor_name = weights_tensor_name+'_folded'
        folded_weights_tensor = Tensor(folded_weights_tensor_name,folded_weights,is_constant=True,shape = folded_weights.shape,scale = original_weights_tensor.scale,
                                 zero_point=original_weights_tensor.zero_point)
        node['frontend']['folded_weights_tensor'] = folded_weights_tensor

        node['frontend']['folded_kernel_size'] = folded_weights.shape[3]
        node['frontend']['folded_padding'] = folded_weights.shape[3] // 2
        # Below is a sanity check for the resize weights calculations
        # We expect each of the ocs to apear only once in the weights
        ocs_found=[]
        for ic in range(folded_weights.shape[1]):
            target_oc = np.where(folded_weights[:,ic,0,0]==8)[0]
            for oc in target_oc:
                if oc in ocs_found:
                    raise ValueError('Found same oc twice')
                ocs_found.append(oc)
        if len(ocs_found)!= folded_weights.shape[0]:
            raise ValueError ('Not all target ocs found')

def calc_globalavg_pooling_folding_factor(ir,node_name,node):
    # TODO: need to add support for forced input folding/unfolding on y. note that node['frontend]['input_folding_factor_y'] should include y folding/unfolding effect see conv/resize for implementation


    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    original_input_width = original_input_tensor_shape[3]
    original_input_height = original_input_tensor_shape[2]

    if len(node['frontend']['preceding_nodes_params'])>0: # If its an input to the workload we calc folding factor based on resolution, otherwise we take it from input tensor
        input_folding_factor_x = node['frontend']['input_tensor'].folding_factor_x
        input_folding_factor_y = node['frontend']['input_tensor'].folding_factor_y
        num_y_tiles = node['frontend']['input_tensor'].y_tiles
        num_x_slices = node['frontend']['input_tensor'].x_slices
    else:
        input_folding_factor_x = math.ceil(math.log(original_input_width / MAX_GRID_WIDTH,2))
        if input_folding_factor_x<0: # If input resolution is less than MAX_GRID_WIDTH, folding is not needed
            input_folding_factor_x=0
        input_folding_factor_y = math.ceil(math.log(original_input_width / MAX_GRID_HEIGHT,2))
        if input_folding_factor_y<0: # If input resolution is less than MAX_GRID_WIDTH, folding is not needed
            input_folding_factor_y=0
        original_kernel_size = 1
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)

    if (input_folding_factor_x!=0) or (input_folding_factor_y!=0):
        raise ValueError ('Global avarage pooling is supported only in case of 0 folding factor')

    node['frontend']['input_tensor'].folding_factor_x = input_folding_factor_x
    node['frontend']['input_tensor'].folding_factor_y = input_folding_factor_y
    node['frontend']['input_folding_factor_x'] = input_folding_factor_x
    node['frontend']['input_folding_factor_y'] = input_folding_factor_y
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    node['frontend']['output_tensor'].x_slices = num_x_slices
    node['frontend']['x_slices'] = num_x_slices

    node['frontend']['output_tensor'].folding_factor_x = input_folding_factor_x
    node['frontend']['output_tensor'].folding_factor_y = input_folding_factor_y
    node['frontend']['output_folding_factor_x'] = input_folding_factor_x
    node['frontend']['output_folding_factor_y'] = input_folding_factor_y

def calc_identity_folding_factor(ir,node_name,node):
    # TODO: need to add support for forced input folding/unfolding on y. note that node['frontend]['input_folding_factor_y'] should include y folding/unfolding effect see conv/resize for implementation

    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    original_input_height = original_input_tensor_shape[2]
    original_kernel_size = 1

    stride = 1
    original_input_channels = original_input_tensor_shape[1]
    original_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape()
    original_output_channels = original_output_tensor_shape[1]

    if len(original_input_tensor_shape)<4:
        original_input_width=1
    else:
        original_input_width = original_input_tensor_shape[3]
    if len(node['frontend']['preceding_nodes_params'])>0: # If its an input to the workload we calc folding factor based on resolution, otherwise we take it from input tensor
        input_folding_factor_x = node['frontend']['input_tensor'].folding_factor_x
        input_folding_factor_y = node['frontend']['input_tensor'].folding_factor_y
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
        x_slices = node['frontend']['input_tensor'].x_slices
    else:
        input_folding_factor_x,input_folding_factor_y, x_slices = get_new_input_folding_factors(original_input_width,original_input_height,stride,original_kernel_size,original_input_channels,original_output_channels)
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
        node['frontend']['input_tensor'].folding_factor_x = input_folding_factor_x
        node['frontend']['input_tensor'].folding_factor_y = input_folding_factor_y
        node['frontend']['input_tensor'].y_tiles = num_y_tiles
        node['frontend']['input_tensor'].x_slices = x_slices

    node['frontend']['input_folding_factor_x'] = input_folding_factor_x
    node['frontend']['input_folding_factor_y'] = input_folding_factor_y

    node['frontend']['output_tensor'].folding_factor_x = input_folding_factor_x
    node['frontend']['output_tensor'].folding_factor_y = input_folding_factor_y
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    node['frontend']['output_tensor'].x_slices = x_slices
    node['frontend']['output_folding_factor_x'] = input_folding_factor_x
    node['frontend']['output_folding_factor_y'] = input_folding_factor_y
    node['frontend']['y_tiles'] = num_y_tiles
    node['frontend']['x_slices'] = x_slices

                
def calc_general_folding_factor(ir,node_name,node):
    # TODO: need to add support for forced input folding/unfolding on y. note that node['frontend]['input_folding_factor_y'] should include y folding/unfolding effect see conv/resize for implementation

    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    original_input_height = original_input_tensor_shape[2]
    original_kernel_size = node['frontend']['kernel_size']

    if len(original_input_tensor_shape)<4:
        original_input_width=1
    else:
        original_input_width = original_input_tensor_shape[3]
    if len(node['frontend']['preceding_nodes_params'])>0: # If its an input to the workload we calc folding factor based on resolution, otherwise we take it from input tensor
        input_folding_factor_x = node['frontend']['input_tensor'].folding_factor_x
        input_folding_factor_y = node['frontend']['input_tensor'].folding_factor_y
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
        x_slices = node['frontend']['input_tensor'].x_slices
    else:
        input_folding_factor_x,input_folding_factor_y, x_slices = get_new_input_folding_factors(original_input_width,original_input_height,stride,original_kernel_size,original_input_channels,original_output_channels)
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1)
    node['frontend']['input_tensor'].folding_factor_x = input_folding_factor_x
    node['frontend']['input_tensor'].folding_factor_y = input_folding_factor_y
    node['frontend']['input_tensor'].y_tiles = num_y_tiles
    node['frontend']['input_tensor'].x_slices = x_slices

    node['frontend']['input_folding_factor_x'] = input_folding_factor_x
    node['frontend']['input_folding_factor_y'] = input_folding_factor_y

    node['frontend']['output_tensor'].folding_factor_x = input_folding_factor_x
    node['frontend']['output_tensor'].folding_factor_y = input_folding_factor_y
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    node['frontend']['output_tensor'].x_slices = x_slices
    node['frontend']['output_folding_factor_x'] = input_folding_factor_x
    node['frontend']['output_folding_factor_y'] = input_folding_factor_y
    node['frontend']['y_tiles'] = num_y_tiles
    node['frontend']['x_slices'] = x_slices

def calc_concat_folding_factor(ir,node_name,node):
    # TODO: need to add support for forced input folding/unfolding on y. note that node['frontend]['input_folding_factor_y'] should include y folding/unfolding effect see conv/resize for implementation
    
    inputs_folding_factors_x = []
    inputs_folding_factors_y = []
    inputs_y_tiles = []
    inputs_x_slices = []
    for input_tensor in node['frontend']['input_tensors']:
        inputs_folding_factors_x.append(input_tensor.folding_factor_x)
        inputs_folding_factors_y.append(input_tensor.folding_factor_y)
        inputs_y_tiles.append(input_tensor.y_tiles)
        inputs_x_slices.append(input_tensor.x_slices)
    max_inputs_folding_factor_x = max(inputs_folding_factors_x)
    max_inputs_folding_factor_y = max(inputs_folding_factors_y)
    if (max_inputs_folding_factor_x > 0):
        raise ValueError ("Currently compiler doesn't support concatenating folded inputs")
    # for input_idx,input_folding_factor_x in enumerate(inputs_folding_factors_x):
    #     if input_folding_factor_x<max_inputs_folding_factor_x:
    #         if input_folding_factor_x<(max_inputs_folding_factor_x-1):
    #             raise ValueError ('Diff >1 of x folding factor between inputs is not supported')
    #         else:
    #             input_tensor_to_fix = node['frontend']['input_tensors'][input_idx]
    #             producing_node_name = input_tensor_to_fix.producer
    #             if ir.graph.nodes[producing_node_name]['op_type'] not in FORCED_OUTPUT_FOLDING_OPS:
    #                 raise ValueError ('Need to fold here but op type!=Conv. Not supported yet....')
    #             ir.marked_nodes_for_folding_x.append(producing_node_name)
    #             return False

    # for input_idx,input_folding_factor_y in enumerate(inputs_folding_factors_y):
    #     if input_folding_factor_y<max_inputs_folding_factor_y:
    #         if input_folding_factor_y<(max_inputs_folding_factor_y-1):

    #             if not DEBUG_TRY_TO_FIX_AUTO_FOLD:

    #                 # This loop prints out the input and output y folding for each node
    #                 # Can delete this if too verbose
    #                 for n in ir.graph.nodes:
    #                     print(ir.graph.nodes[n]['name'], end="\t")
    #                     if 'input_folding_factor_y' in ir.graph.nodes[n]['frontend'].keys():
    #                         print(ir.graph.nodes[n]['frontend']['input_folding_factor_y'], end="\t")
    #                     else:
    #                         print("n/a", end="\t")
    #                     if 'output_folding_factor_y' in ir.graph.nodes[n]['frontend'].keys():
    #                         print(ir.graph.nodes[n]['frontend']['output_folding_factor_y'])
    #                     else:
    #                         print("n/a")

    #                 raise ValueError ('Diff >1 of y folding factor between inputs is not supported')

    #             # Found a difference > 1 of Y folding factor for the Concat
    #             # First check this has 2 inputs and the difference is 2. Other cases can be handled later.
    #             if (max_inputs_folding_factor_y - input_folding_factor_y) != 2:
    #                 assert False, "Only handled folding difference of 2 currently"
    #             if (len(inputs_folding_factors_y)) != 2:
    #                 assert False, "Only handled 2 concat inputs currently"

    #             # Found diff of 2, so force folding on input with less folding
    #             input_tensor_to_fix = node['frontend']['input_tensors'][input_idx]
    #             producing_node_name = input_tensor_to_fix.producer
    #             producing_node = ir.graph.nodes[producing_node_name]
    #             if producing_node['op_type'] not in FORCED_OUTPUT_FOLDING_OPS:
    #                 raise ValueError ('Need to fold here but op type!=Conv. Not supported yet....')
    #             ir.update_marked_nodes_for_folding_y(producing_node_name)

    #             # And force unfolding on other more folded index
    #             if input_idx == 0:
    #                 folded_input_idx = 1
    #             else:
    #                 folded_input_idx = 0
    #             input_tensor_to_fix = node['frontend']['input_tensors'][folded_input_idx]
    #             producing_node_name = input_tensor_to_fix.producer
    #             producing_node = ir.graph.nodes[producing_node_name]
    #             # Keep searching until Conv is found
    #             while producing_node['op_type'] not in FORCED_OUTPUT_FOLDING_OPS:
    #                 if 'input_tensors' in producing_node['frontend']:
    #                     input_tensor_to_fix = producing_node['frontend']['input_tensors'][0]
    #                 else:
    #                     input_tensor_to_fix = producing_node['frontend']['input_tensor']
    #                 producing_node_name = input_tensor_to_fix.producer
    #                 producing_node = ir.graph.nodes[producing_node_name]
    #             # Unfold this node
    #             ir.force_y_unfolding.append(producing_node_name)

    #             # Re-calculate folding
    #             return False

    #         else:
    #             input_tensor_to_fix = node['frontend']['input_tensors'][input_idx]
    #             producing_node_name = input_tensor_to_fix.producer
    #             if ir.graph.nodes[producing_node_name]['op_type'] not in FORCED_OUTPUT_FOLDING_OPS:
    #                 raise ValueError ('Need to fold here but op type!=Conv. Not supported yet....')
    #             ir.update_marked_nodes_for_folding_y(producing_node_name)
    #             return False

    input_folding_factor_x = node['frontend']['input_tensors'][0].folding_factor_x
    input_folding_factor_y = 0
    output_folding_factor_x = input_folding_factor_x
    output_folding_factor_y = input_folding_factor_y

    if node_name in ir.force_y_folding:
        output_folding_factor_y = output_folding_factor_y +1
        node['frontend']['force_folding_y'] = True

    if node_name in ir.force_y_unfolding:
        output_folding_factor_y = output_folding_factor_y -1
        node['frontend']['force_unfolding_y'] = True

    if (max_inputs_folding_factor_y > 0):
        node['frontend']['force_unfolding_y'] = True
    
    node['frontend']['input_folding_factor_x'] = input_folding_factor_x
    node['frontend']['input_folding_factor_y'] = input_folding_factor_y

    node['frontend']['output_tensor'].folding_factor_x = output_folding_factor_x
    node['frontend']['output_tensor'].folding_factor_y = output_folding_factor_y
    node['frontend']['output_tensor'].x_slices = inputs_x_slices[0]
    node['frontend']['output_folding_factor_x'] = output_folding_factor_x
    node['frontend']['output_folding_factor_y'] = output_folding_factor_y
    num_y_tiles = inputs_y_tiles[0]
    for current_input_y_tiles in inputs_y_tiles:
        if current_input_y_tiles!= num_y_tiles:
            pass
            #raise ValueError ('All Concat inputs must have same number of tiles. Please check...')
    node['frontend']['y_tiles'] = num_y_tiles        
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    node['frontend']['x_slices'] = inputs_x_slices[0]
    return True


def calc_concat_qparams(ir,node_name,node):

    # Note: We need to make sure both inputs have same quant params (scale and zp) and they are equal to output quant params
    input_tensors_scales = node['frontend']['input_tensors_scale']
    input_tensors_zps = node['frontend']['input_tensors_zp']
    output_tensor_scale = node['frontend']['output_tensor_scale']
    output_tensor_zp = node['frontend']['output_tensor_zp']
    for scale in input_tensors_scales:
        if scale!=output_tensor_scale and not DEBUG_SIMULATE_CONCAT_REQUANT:
            raise ValueError ('input scale is different from output scale. Need to check')
    for zp in input_tensors_zps:
        if zp!=output_tensor_zp and not DEBUG_SIMULATE_CONCAT_REQUANT:
            raise ValueError ('input zp is different from output zp. Need to check')

def calc_add_folding_factor(ir,node_name,node):
    # TODO: need to add support for forced input folding/unfolding on y. note that node['frontend]['input_folding_factor_y'] should include y folding/unfolding effect see conv/resize for implementation

    # First we make sure that both input tensors have same shape
    original_input0_tensor_shape = node['frontend']['input_tensors'][0].get_original_shape()
    original_input1_tensor_shape = node['frontend']['input_tensors'][1].get_original_shape()
    original_output_shape = node['frontend']['output_tensor'].get_original_shape()
    if original_input0_tensor_shape != original_input1_tensor_shape:
        raise ValueError ('Currently Add op does not support broadcasting. Expected both inputs to have same shape but got %s!=%s' % (str(original_input0_tensor_shape),str(original_input1_tensor_shape)))

    original_input_width = original_input0_tensor_shape[3]

    if (node['frontend']['input_tensors'][0].folding_factor_x!=node['frontend']['input_tensors'][1].folding_factor_x) or\
        (node['frontend']['input_tensors'][0].folding_factor_y!=node['frontend']['input_tensors'][1].folding_factor_y):
        raise ValueError ('Add input tensors folding factor must be the same')
    input_folding_factor_x = node['frontend']['input_tensors'][0].folding_factor_x
    input_folding_factor_y = node['frontend']['input_tensors'][0].folding_factor_y
    num_y_tiles = node['frontend']['input_tensors'][0].y_tiles
    num_x_slices = node['frontend']['input_tensors'][0].x_slices
    #if (node['frontend']['input_tensors'][0].y_tiles != node['frontend']['input_tensors'][1].y_tiles):
    #    raise ValueError ('Adds inputs must have same # of tiles')
    node['frontend']['y_tiles'] = num_y_tiles
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    node['frontend']['output_tensor'].x_slices = num_x_slices
    node['frontend']['x_slices'] = num_x_slices

    output_folding_factor_x = input_folding_factor_x
    output_folding_factor_y = input_folding_factor_y
    node['frontend']['input_folding_factor_x'] = input_folding_factor_x
    node['frontend']['input_folding_factor_y'] = input_folding_factor_y

    following_nodes_params = node['frontend']['following_nodes_params']
    max_following_node_stride = 0
    for following_node_params in following_nodes_params:
        current_following_node = ir.graph.nodes[following_node_params[0]] # [0] is node name, [1] is which input of folloing node is connected to current node
        if 'stride' in current_following_node['frontend'] and current_following_node['frontend']['stride']>max_following_node_stride:
            max_following_node_stride = current_following_node['frontend']['stride']
            if max_following_node_stride>2:
                raise ValueError ('Stride>2 is not currently supported')

    # We change to folding conv if we see in following ops a stride=2 (In case of width>14 => 28x28 grid) or if current op is stride==2
    # if current op is stride==2 and the data is not folded we will perform a folding conv with stride=2 which means un-needed output channels will be calculated but dropped in RQ
    # Disabled folding output in add conv since it takes double mem and we need to allocate double output mem (so inline add will not work). Instead folding ill occur in actual stride=2 convs
    if 0: #(output_folding_factor==0 and max_following_node_stride==2 and original_input_width>MINIMAL_ACTUAL_INPUT_WIDTH) or DEBUG_FORCE_FOLDCONV:
        node['frontend']['folding_conv'] = True
        node['frontend']['output_folding_factor_x'] = output_folding_factor_x + 1
        node['frontend']['output_folding_factor_y'] = output_folding_factor_y + 1
        node['frontend']['output_tensor'].folding_factor_x = output_folding_factor_x + 1
        node['frontend']['output_tensor'].folding_factor_y = output_folding_factor_y + 1
    else: 
        node['frontend']['output_tensor'].folding_factor_x = output_folding_factor_x
        node['frontend']['output_tensor'].folding_factor_y = output_folding_factor_y
        node['frontend']['output_folding_factor_x'] = output_folding_factor_x
        node['frontend']['output_folding_factor_y'] = output_folding_factor_y

def calc_add_qparams(ir,node_name,node):
    # First we make sure that both input tensors have same shape
    original_input0_tensor_shape = node['frontend']['input_tensors'][0].get_original_shape()
    original_input_channels = original_input0_tensor_shape[1]
    original_output_shape = node['frontend']['output_tensor'].get_original_shape()
    original_output_channels = original_output_shape[1]

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    input_tensors_scale = node['frontend']['input_tensors_scale']
    input_tensors_zp = node['frontend']['input_tensors_zp']
    output_tensor_scale = node['frontend']['output_tensor_scale']
    output_tensor_zp = node['frontend']['output_tensor_zp']

    is_folding_conv_x = 'force_folding_x' in node['frontend']
    is_folding_conv_y = 'force_folding_y' in node['frontend']
    if (is_folding_conv_x or is_folding_conv_y):
        raise ValueError ('Currently folding add conv is not supported') # If need to support look on next line to update shape calc.
    folded_output_channels = node['frontend']['output_tensor'].get_folded_shape()[1] # This will be folded output channels but without output folding
    folded_input_channels = node['frontend']['input_tensors'][0].get_folded_shape()[1]

    #quantized weights
    w_input      = (128*input_tensors_scale[0])/(input_tensors_scale[0]+input_tensors_scale[1])
    w_shortcut   = (128*input_tensors_scale[1])/(input_tensors_scale[0]+input_tensors_scale[1])
    q_w_input    = math.trunc(w_input    + 0.5)
    q_w_shortcut = math.trunc(w_shortcut + 0.5)
    
    #requant
    requant_scale = (input_tensors_scale[0]+input_tensors_scale[1]) / (128*output_tensor_scale)
    requant_bias  = output_tensor_zp.astype(np.float32)
    requant_bias -= input_tensors_zp[0]*input_tensors_scale[0]/output_tensor_scale
    requant_bias -= input_tensors_zp[1]*input_tensors_scale[1]/output_tensor_scale
    
    requant_bias_int12 = math.trunc(BIAS_MULITIPLIER*requant_bias + BIAS_ROUNDING_ADD)
    if DEBUG_CLIP_BIAS_TO_MAX_BIAS_BITS and math.log(abs(requant_bias_int12),2)>=MAX_BIAS_BITS:
        max_bias_int12 = 2 ** MAX_BIAS_BITS-1
        min_bias_int12 = -1*max_bias_int12
        requant_bias_int12 = min(max_bias_int12,max(min_bias_int12,requant_bias_int12)) # Clip bias to MAX_BIAS_BITS
        print('At layer %s, bias exceeded INT%d ' % (node_name,MAX_BIAS_BITS+1))

    if REDUCED_MAC_RESCALE_BUS_WIDTH:
        assert False
    elif MCHP_NUMERICS:
        requant_scale_f = min(math.trunc(-math.log(requant_scale, 2)+2) + FRACTIONAL_BITS, 31)
    else:
        assert False

    if not TFLITE_REQUANT:
        # MCHP numerics
        expected_mac_bits = requant_scale_f + FINAL_RESULTS_BITS + OVERFLOW_EXTRA_BITS - INT_SCALE_BITS # This is the expected number of bits in the mac result
        per_och_mac_shift = np.array(expected_mac_bits - MCHP_MAC_TO_RQ_BUS_WIDTH) # We shift right to limit MAC to MCHP_MAC_TO_RQ_BUS_WIDTH bits
        # We achieve the per och mac shift by rough shift right between MAC and RQ (multiplexer) and fine shift left by adding bits to the int scale (UINT13 instead of UINT10)
        rough_shift_right = int(np.ceil(per_och_mac_shift/MAC_ROUGH_SHIFT_GRANULARITY))
        rough_shift_right_bits = rough_shift_right * MAC_ROUGH_SHIFT_GRANULARITY
        fine_shift_left = rough_shift_right_bits - per_och_mac_shift
        requant_scale_uint14  = math.trunc(requant_scale*math.pow(2,requant_scale_f+fine_shift_left) + 0.5) # uint14 scale is used if not REDUCED_MAC_RESCALE_BUS_WIDTH and MCHP_NUMERICS is True
        mac_rough_shift_mux = rough_shift_right

        requant_scale_shift = requant_scale_f
        requant_scale_float = requant_scale
        
    node['frontend']['requant_bias_int12'] = requant_bias_int12
    if not TFLITE_REQUANT:
        # New MAC shift logic
        node['frontend']['requant_scale_uint14'] = requant_scale_uint14
        node['frontend']['mac_rough_shift_mux'] = mac_rough_shift_mux
        node['frontend']['requant_scale_shift'] = requant_scale_shift
        node['frontend']['requant_scale_float'] = requant_scale_float
    if (input_folding_factor_x>0 or input_folding_factor_y>0):
        if not TFLITE_REQUANT:
            node['frontend']['folded_requant_scale_uint14'] = requant_scale_uint14
            node['frontend']['folded_mac_rough_shift_mux'] = mac_rough_shift_mux
            node['frontend']['folded_requant_scale_shift'] = requant_scale_shift
            node['frontend']['folded_requant_scale_float'] = requant_scale_float
        node['frontend']['folded_requant_bias_int12'] = requant_bias_int12

    node['frontend']['q_w_input'] = q_w_input
    node['frontend']['q_w_shortcut'] = q_w_shortcut
    # Create the weights tensor
    original_weights_tensor_shape = [original_output_channels,original_input_channels*2,1,1]
    original_weights_tensor = np.zeros(original_weights_tensor_shape,dtype=np.int64)
    for oc in range(original_output_channels):
            original_weights_tensor[oc,oc,0,0] = q_w_input
            original_weights_tensor[oc,oc+original_input_channels,0,0] = q_w_shortcut
    w_int8_np = original_weights_tensor.astype(np.int8)
    original_weights_tensor_name = node_name + '_original_weights_tensor'
    weights_tensor = Tensor(original_weights_tensor_name,w_int8_np,is_constant=True,shape = original_weights_tensor_shape)
    ir.tensors[original_weights_tensor_name] = weights_tensor
    node['frontend']['weights_tensor'] = weights_tensor
    node['frontend']['kernel_size'] = 1

    dense_weights = weights_tensor.data.size
    sparse_weights = np.count_nonzero(weights_tensor.data)
    node['frontend']['dense_weights'] = dense_weights
    node['frontend']['sparse_weights'] = sparse_weights
    node['frontend']['weights_sparsity'] = 1- (sparse_weights/dense_weights)
    sparse_macs = sparse_weights*original_input0_tensor_shape[2]*original_input0_tensor_shape[3]
    node['frontend']['dense_macs'] = 0
    node['frontend']['sparse_macs'] = sparse_macs
    node['frontend']['macs_sparsity'] = 0

    if (input_folding_factor_x>0 or input_folding_factor_y>0):
        folded_weights_tensor_shape = [folded_output_channels,folded_input_channels*2,1,1]
        folded_weights_tensor = np.zeros(folded_weights_tensor_shape,dtype=np.int64)
        for oc in range(folded_output_channels):
                folded_weights_tensor[oc,oc,0,0] = q_w_input
                folded_weights_tensor[oc,oc+folded_input_channels,0,0] = q_w_shortcut
        folded_w_int8_np = folded_weights_tensor.astype(np.int8)
        folded_weights_tensor_name = node_name + '_folded_weights_tensor'
        folded_tensor = Tensor(folded_weights_tensor_name,folded_w_int8_np,is_constant=True,shape = folded_weights_tensor_shape)
        ir.tensors[folded_weights_tensor_name] = folded_tensor
        node['frontend']['folded_kernel_size'] = 1
        node['frontend']['folded_weights_tensor'] = folded_tensor

    node['frontend']['stride'] = 1
    input_tensors_shape = node['frontend']['input_tensors'][0].get_folded_shape() # TODO Dans, need to revisit for folded add
    node['frontend']['sparse_macs'] = 2*folded_output_channels*input_tensors_shape[2]*input_tensors_shape[3]

    if MCHP_NUMERICS and TFLITE_REQUANT:
        # c_input = (bias_data*output_multiplier + output_offset)<<o_shift + 1<<(o_shift-1)
        acc = None # This is the accumulator
        bias_data = 0 - (q_w_input * input_tensors_zp[0]) - (q_w_shortcut * input_tensors_zp[1])
        scale = requant_scale
        output_offset = output_tensor_zp.item()

        output_activation_min = int8min
        output_activation_max = int8max

        output_multiplier, cInputH, cInputL, o_shift = \
            quantize2MathBlock(acc,bias_data,scale,output_offset,output_activation_min,output_activation_max)

        node['frontend']['output_multiplier'] = output_multiplier
        node['frontend']['cInputH'] = cInputH
        node['frontend']['cInputL'] = cInputL
        node['frontend']['o_shift'] = o_shift

def calc_identity_qparams(ir,node_name,node):

    # First we make sure that both input tensors have same shape
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    original_input_channels = original_input_tensor_shape[1]
    original_output_shape = node['frontend']['output_tensor'].get_original_shape()
    original_output_channels = original_output_shape[1]
    if original_input_channels!=original_output_channels:
        # For Strided Slice, implement as identity but allow input and output channels to differ
        if 'STRIDEDSLICE' not in node_name and 'SPLIT' not in node_name:
            raise ValueError ('At node: %s. input channels!=output chanels. This is unexpected for identity op!' % node_name)

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    input_tensor_scale = node['frontend']['input_tensor_scale']
    input_tensor_zp = node['frontend']['input_tensor_zp']
    output_tensor_scale = node['frontend']['output_tensor_scale']
    output_tensor_zp = node['frontend']['output_tensor_zp']

    is_folding_conv_x = 'force_folding_x' in node['frontend']
    is_folding_conv_y = 'force_folding_y' in node['frontend']
    if (is_folding_conv_x or is_folding_conv_y):
        raise ValueError ('Currently folding add conv is not supported') # If need to support look on next line to update shape calc.
    folded_output_channels = node['frontend']['output_tensor'].get_folded_shape()[1] # This will be folded output channels but without output folding
    folded_input_channels = node['frontend']['input_tensor'].get_folded_shape()[1]

    #quantized weights
    weight_value      = 8
    node['frontend']['weight_value'] = weight_value
    # Create the weights tensor
    original_weights_tensor_shape = [original_output_channels,original_input_channels,1,1]
    original_weights_tensor = np.zeros(original_weights_tensor_shape,dtype=np.int64)
    if ('STRIDEDSLICE' in node_name) or ('SPLIT' in node_name):
        begin_input_channel = node['attributes']['begin_input_channel']
        end_input_channel = node['attributes']['end_input_channel']
        for ic in range(begin_input_channel, end_input_channel):
            original_weights_tensor[ic - begin_input_channel, ic, 0, 0] = weight_value
    else:
        for oc in range(original_output_channels):
            original_weights_tensor[oc,oc,0,0] = weight_value
    w_int8_np = original_weights_tensor.astype(np.int8)
    original_weights_tensor_name = node_name + '_original_weights_tensor'
    tensor = Tensor(original_weights_tensor_name,w_int8_np,is_constant=True,shape = original_weights_tensor_shape)
    ir.tensors[original_weights_tensor_name] = tensor
    node['frontend']['weights_tensor'] = tensor
    
    if (input_folding_factor_x>0 or input_folding_factor_y>0):
        folded_weights_tensor_shape = [folded_output_channels,folded_input_channels,1,1]
        if ('STRIDEDSLICE' in node_name) or ('SPLIT' in node_name):
            folded_weights_tensor = folding_algo.get_asym_folded_weights(original_weights_tensor,input_folding_factor_x=input_folding_factor_x,input_folding_factor_y=input_folding_factor_y,stride_x=1,stride_y=1)
        else:
            folded_weights_tensor = np.zeros(folded_weights_tensor_shape,dtype=np.int64)
            for oc in range(folded_output_channels):
                folded_weights_tensor[oc,oc,0,0] = weight_value
        folded_w_int8_np = folded_weights_tensor.astype(np.int8)
        folded_weights_tensor_name = node_name + '_folded_weights_tensor'
        folded_tensor = Tensor(folded_weights_tensor_name,folded_w_int8_np,is_constant=True,shape = folded_weights_tensor_shape)
        ir.tensors[folded_weights_tensor_name] = folded_tensor
        node['frontend']['folded_kernel_size'] = 1
        node['frontend']['folded_weights_tensor'] = folded_tensor
        node['frontend']['folded_padding'] = 0

    requant_scale = input_tensor_scale/(output_tensor_scale*weight_value) #TODO: We can set requant_scale to 0.02 in case all weights are zero (see neuronix_models capture code)
    if REDUCED_MAC_RESCALE_BUS_WIDTH:
        assert False
    elif MCHP_NUMERICS:
        requant_scale_f = min(math.trunc(-math.log(requant_scale, 2)+BIAS_FRACTIONAL_BITS) + FRACTIONAL_BITS, 31) # See above note
        expected_mac_bits = requant_scale_f + FINAL_RESULTS_BITS + OVERFLOW_EXTRA_BITS - INT_SCALE_BITS # This is the expected number of bits in the mac result
        per_och_mac_shift = np.array(expected_mac_bits - MCHP_MAC_TO_RQ_BUS_WIDTH) # We shift right to limit MAC to MCHP_MAC_TO_RQ_BUS_WIDTH bits
        # We achieve the per och mac shift by rough shift right between MAC and RQ (multiplexer) and fine shift left by adding bits to the int scale (UINT13 instead of UINT10)
        rough_shift_right = int(np.ceil(per_och_mac_shift/MAC_ROUGH_SHIFT_GRANULARITY))
        rough_shift_right_bits = rough_shift_right * MAC_ROUGH_SHIFT_GRANULARITY
        fine_shift_left = rough_shift_right_bits - per_och_mac_shift
        requant_scale_uint14  = math.trunc(requant_scale*math.pow(2,requant_scale_f+fine_shift_left) + 0.5) # uint14 scale is used if not REDUCED_MAC_RESCALE_BUS_WIDTH and MCHP_NUMERICS is True
        mac_rough_shift_mux = rough_shift_right
    else:
        assert False

    #requant_bias  = output_tensor_zp.astype(np.float64)
    requant_bias  = float(output_tensor_zp)
    requant_bias -= weight_value*input_tensor_zp*input_tensor_scale/output_tensor_scale
    requant_bias_int12 = math.trunc(BIAS_MULITIPLIER*requant_bias+BIAS_ROUNDING_ADD)

    requant_scale_shift = requant_scale_f
    requant_scale_float = requant_scale
   
    node['frontend']['requant_bias_int12'] = requant_bias_int12
    # New MAC shift logic
    node['frontend']['requant_scale_shift'] = requant_scale_shift
    node['frontend']['requant_scale_float'] = requant_scale_float
    # MCHP Numerics
    node['frontend']['requant_scale_uint14'] = requant_scale_uint14
    node['frontend']['mac_rough_shift_mux'] = mac_rough_shift_mux


    if input_folding_factor_x>0 or input_folding_factor_y>0:
        node['frontend']['folded_requant_scale_uint14'] = requant_scale_uint14
        node['frontend']['folded_mac_rough_shift_mux'] = mac_rough_shift_mux
        node['frontend']['folded_requant_scale_shift'] = requant_scale_shift
        node['frontend']['folded_requant_scale_float'] = requant_scale_float
        node['frontend']['folded_requant_bias_int12'] = requant_bias_int12

    node['frontend']['kernel_size'] = 1
    node['frontend']['padding'] = 0

    node['frontend']['stride'] = 1
    input_tensors_shape = node['frontend']['input_tensor'].get_folded_shape() # TODO Dans, need to revisit for folded add
    node['frontend']['sparse_macs'] = folded_output_channels*input_tensors_shape[2]*input_tensors_shape[3]

    if MCHP_NUMERICS and TFLITE_REQUANT:
        # c_input = (bias_data*output_multiplier + output_offset)<<o_shift + 1<<(o_shift-1)
        acc = None # This is the accumulator
        bias_data = int(-8*output_tensor_zp)
        scale = requant_scale
        output_offset = output_tensor_zp.item()

        output_activation_min = int8min
        output_activation_max = int8max

        output_multiplier, cInputH, cInputL, o_shift = \
            quantize2MathBlock(acc,bias_data,scale,output_offset,output_activation_min,output_activation_max)

        node['frontend']['output_multiplier'] = output_multiplier
        node['frontend']['cInputH'] = cInputH
        node['frontend']['cInputL'] = cInputL
        node['frontend']['o_shift'] = o_shift

def convert_maxpoolk5_to_2xmaxpoolk3(ir: internal_representation.IR,original_node_name,node):
    node['frontend']['kernel_size'] = 3 # We change original node kernel from 5 to 3 and duplicate it to have 2 consecutive maxpool nodes with k=3
    node['frontend']['padding'] = 1
    node['attributes']['kernel_shape'] = [3,3]
    node['attributes']['pads'] = [1,1,1,1]

    dup_node = copy.deepcopy(node)
    dup_node['frontend']['output_tensor'] = node['frontend']['output_tensor'] # We override the deepcopy to have pointer to the original tensor and not create a new one
    dup_node['frontend']['following_nodes_params'] = node['frontend']['following_nodes_params'] # We override the deepcopy to have pointer to the original node and not create a new one

    dup_node_name = original_node_name+'_dupk3'
    dup_node['name'] = dup_node_name

    maxpoolout_to_maxpoolin_tensor = copy.deepcopy(ir.tensors[node['outputs'][0]])
    maxpoolout_to_maxpoolin_tensor_name = node['outputs'][0]+'_dupk3'
    ir.tensors[maxpoolout_to_maxpoolin_tensor_name] = maxpoolout_to_maxpoolin_tensor
    original_node_output_tensor_name = node['outputs'][0]
    node['outputs'][0] = maxpoolout_to_maxpoolin_tensor_name
    maxpoolout_to_maxpoolin_tensor.name = maxpoolout_to_maxpoolin_tensor_name
    maxpoolout_to_maxpoolin_tensor.consumers = [dup_node_name]
    original_node_output_tensor = ir.tensors[original_node_output_tensor_name]
    original_node_output_tensor.producer = dup_node_name
    dup_node['inputs'] = [maxpoolout_to_maxpoolin_tensor_name]
    dup_node['frontend']['input_tensor'] = maxpoolout_to_maxpoolin_tensor
    dup_node['frontend']['preceding_nodes_params'] = [(original_node_name,0)]
    node['frontend']['output_tensor'] = maxpoolout_to_maxpoolin_tensor
    node['outputs'] = [maxpoolout_to_maxpoolin_tensor_name]
    ir.graph.add_node(dup_node_name,**dup_node) # When the node is created it copies the dictionary attributes and create a new dict
    dup_node = ir.graph.nodes[dup_node_name] # very important to get the dict created by node
    node['frontend']['following_nodes_params'] = [(dup_node_name,0)]
    # update original maxpool following nodes "preceding_nodes_params" to point to dup node
    for following_node_params in dup_node['frontend']['following_nodes_params']:
        following_node = ir.graph.nodes[following_node_params[0]]
        following_node_preceding_nodes_params = following_node['frontend']['preceding_nodes_params']
        for preceding_node_index,following_node_preceding_node_params in enumerate(following_node_preceding_nodes_params.copy()):
            if following_node_preceding_node_params[0] == original_node_name:
                new_params=(dup_node_name,following_node_preceding_node_params[1])
                following_node_preceding_nodes_params[preceding_node_index] = new_params
    
    edges_from_original_node = list(ir.graph.out_edges(original_node_name))
    for edge in edges_from_original_node:
        ir.graph.remove_edge(edge[0],edge[1])
        ir.graph.add_edge(dup_node_name,edge[1])
    ir.graph.add_edge(original_node_name,dup_node_name)

def duplicate_maxpool_nodes(ir: internal_representation.IR) -> internal_representation.IR:
    nodes_in_graph = list(ir.graph.nodes(data=True))
    for (node_name, node) in nodes_in_graph:
       if node['op_type'] == 'MaxPool':
            folded_input = node['frontend']['input_folding_factor_x']>0 and node['frontend']['input_folding_factor_y']>0 # We need both x and y inputs to be folded so that folded kernel size will be 3
            if node['frontend']['kernel_size'] == 5 and not folded_input: # We support maxpool with k=5 by 2 maxpool ops with k=3 in series or a single maxpool if its folded to 3x3 kernel
                convert_maxpoolk5_to_2xmaxpoolk3(ir,node_name,node)
    return ir

################################################
def insert_identity_node (ir:internal_representation.IR, param_target_node = None, param_source_node = None, name_sufix = 'new_node'): 
    source_node = param_source_node
    target_node = param_target_node

    if param_target_node is None:
        # we have to insert the NODE after source, check if source has only one output        
        if len(source_node['outputs'])==1:
            target_tensor_name = source_node['outputs'][0]
            target_node_names   = ir.tensors[target_tensor_name].consumers
            # if len(target_node_names)!=1:
            #     #raise ValueError ('Insert is not properly defined; too many inputs')
            #     target_nodes = [ir.graph.nodes[name] for name in target_node_names]
            # else:
            #     target_nodes = ir.graph.nodes[target_node_names]    
                  
        elif len(source_node['inputs'])!=1:
            raise ValueError ('Insert is not properly defined; it allows for more than one interpretation')

    if source_node is None:
        # we have to insert the NODE before target, check if target has only one output        
        if   len(target_node['inputs'])==1:
            source_tensor_name = target_node['inputs'][0]
            source_node_name   = ir.tensors[source_tensor_name].producer
            if len(source_node_name)!=1:
                raise ValueError ('Insert is not properly defined; too many outputs')               
            else:
                source_node        = ir.graph.nodes[source_node_name]

        elif len(target_node['outputs'])!=1:
            raise ValueError ('Insert is not properly defined; it allows for more than one interpretation')
    
    #### normal case, there is source_node and source_node
    
    #this is the input/output Tensor from node-to-node
    target_node_names   = ir.tensors[target_tensor_name].consumers
    original_tensor_name = source_node['outputs'][0] 
    original_tensor = ir.tensors[original_tensor_name]
    new_tensor_name = original_tensor_name+'_'+name_sufix
    if new_tensor_name in ir.tensors:
        raise ValueError ('Tensor name %s already exists' % new_tensor_name)
    
    input_folding_factor_x = original_tensor.folding_factor_x
    output_folding_factor_x = original_tensor.folding_factor_x
    input_folding_factor_y = original_tensor.folding_factor_y
    output_folding_factor_y = original_tensor.folding_factor_y    
    x_slices = original_tensor.x_slices
    num_y_tiles = original_tensor.y_tiles
    if 'unfold_x' in name_sufix:
        output_folding_factor_x -= 1
        x_slices *= 2

        following_nodes_params = source_node['frontend']['following_nodes_params']
        following_node_is_concat = False
        for following_node_params in following_nodes_params:
            if ('CONCATENATION' in following_node_params[0]) or ('ADD' in following_node_params[0]):
                following_node_is_concat = True
                break
        if following_node_is_concat and (output_folding_factor_y > 0): # We can unfold y if output folding factor is already 0
            input_folding_factor_y-=1
            output_folding_factor_y-=1
    
        input_folding_coef_y = math.pow(2,input_folding_factor_y)
        folded_y_size = original_tensor.shape[2] // input_folding_coef_y
        num_y_tiles = math.ceil(folded_y_size / MAX_GRID_HEIGHT)
    
    new_node_name = source_node['name']+'_'+ name_sufix
    new_node_name = ''.join(e for e in new_node_name if (e.isalnum() or e=='_')) # Removing special characters from node name as it is used as filename for debug file name (.xlsx)
    new_tensor = Tensor(new_tensor_name,None,producer=new_node_name,consumers=target_node_names,
                            is_constant=False,shape = original_tensor.shape,scale = original_tensor.scale,
                            zero_point=original_tensor.zero_point,folding_factor_x=output_folding_factor_x,
                            folding_factor_y=output_folding_factor_y, x_slices=x_slices)
    new_tensor.y_tiles = num_y_tiles

    # Register the new tensor
    ir.tensors[new_tensor_name] = new_tensor

    # Create the new node
    new_node = {
        'name': new_node_name,
        'op_type': 'Conv',  # Use Conv instead of Identity for folding compatibility
        'attributes': {
            'kernel_shape': [1, 1],
            'pads': [0, 0, 0, 0]
        },
        'outputs': [new_tensor_name],
        'frontend': {}
    }

    # Set up frontend information
    new_node_input_tensor = source_node['frontend']['output_tensor']
    original_input_shape  = new_node_input_tensor.get_original_shape()
    original_input_channels = original_input_shape[1]  # Assuming NCHW format
    original_output_channels = original_input_channels

    input_channels = original_input_channels
    output_channels = original_output_channels
    
    new_node['frontend'].update({
        'input_channels': input_channels,
        'output_channels': output_channels,  # Same as input
        'input_folding_factor_x': input_folding_factor_x,
        'input_folding_factor_y': input_folding_factor_y,
        'output_folding_factor_x': output_folding_factor_x,
        'output_folding_factor_y': output_folding_factor_y,
        'y_tiles': num_y_tiles,
        'x_slices': new_node_input_tensor.x_slices,
        'preceding_nodes_params': [(source_node['name'], 0)],
        'input_tensor': new_node_input_tensor,
        'output_tensor': new_tensor,
        'input_tensor_scale': new_node_input_tensor.scale,
        'input_tensor_zp': new_node_input_tensor.zero_point,
        'output_tensor_scale': new_tensor.scale,
        'output_tensor_zp': new_tensor.zero_point,
        'kernel_size': 1,
        'stride': 1,
        'padding':new_node['attributes']['pads']
    })

    if following_node_is_concat:
        new_node['frontend']['force_unfolding_y'] = True

    # Set up weights_tensor
    original_weights_tensor_shape = [original_output_channels,original_input_channels,1,1]
    original_weights_tensor = np.zeros(original_weights_tensor_shape,dtype=np.int64)
    weight_value = 32 # Changed from 8 to 32 in MCHP numerics so that scale will be smaller and right shift will be bigger
    for oc in range(original_output_channels):
            original_weights_tensor[oc,oc,0,0] = weight_value
    w_int8_np = original_weights_tensor.astype(np.int8)

    new_weights_tensor_name = new_tensor_name + '_weights_tensor'
    per_channel_scale = np.full((original_output_channels),1/weight_value)
    per_channel_zp = np.full((original_output_channels),0)
    weights_tensor = Tensor(new_weights_tensor_name,w_int8_np,is_constant=True,shape = original_weights_tensor_shape,scale = per_channel_scale,zero_point=per_channel_zp)
    ir.tensors[new_weights_tensor_name] = weights_tensor

    new_node['frontend'].update({
        'weights_tensor':           weights_tensor,
        'weights_per_channel_scale':per_channel_scale,
        'weights_per_channel_zp':   per_channel_zp,
        'sparse_macs':              original_input_shape[1]*original_input_shape[2]*original_input_shape[3]
        })
 
    # Set up weights_tensor
    original_biases_tensor_shape = [original_output_channels]
    original_biases_tensor = np.zeros(original_biases_tensor_shape,dtype=np.int64)
    new_biases_tensor_name = new_tensor_name + '_biases_tensor'
    per_channel_scale = np.full((original_output_channels),1)
    per_channel_zp = np.full((original_output_channels),0)
    biases_tensor = Tensor(new_biases_tensor_name,original_biases_tensor,is_constant=True,shape = original_biases_tensor_shape, scale = per_channel_scale, zero_point = per_channel_zp)
    ir.tensors[new_biases_tensor_name] = biases_tensor
    new_node['frontend']['biases_tensor'] = biases_tensor
    new_node['inputs'] = [original_tensor_name,new_weights_tensor_name,new_biases_tensor_name]

    #insert the node to the graph: update edges etc
    ir.graph.add_node(new_node_name,**new_node) # When the node is created it copies the dictionary attributes and create a new dict
    created_new_node = ir.graph.nodes[new_node_name]
    ir.graph.add_edge(source_node['name'],new_node_name)
    for i_name in target_node_names:
        ir.graph.add_edge(new_node_name,i_name)
        ir.graph.remove_edge(source_node['name'],i_name)

    #ir.switch_tensor_consumer(original_tensor,original_node_name=target_node['name'],new_node_name=new_node_name)
    original_tensor.consumers = [new_node_name]

    # Update the 'inputs' and input_tensor field in target node
    for i_name in target_node_names:
        i_target_node = ir.graph.nodes[i_name]
        ir.switch_input_name(i_target_node,original_input_name = original_tensor_name,new_input_name=new_tensor_name)
        ir.switch_input_tensor(i_target_node,original_input_tensor = original_tensor,new_input_tensor=new_tensor)

    source_node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(source_node) # Update the following_nodes_params field in source node - It is important to also update following nodes according to execution order
    for i_name in target_node_names:
        i_target_node = ir.graph.nodes[i_name]
        i_target_node['frontend']['preceding_nodes_params'] = ir.get_updated_preceding_nodes(i_target_node) # Update the preceding_nodes_params field in target node
    
    created_new_node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(new_node)

def insert_padding_whole_grid_node_between(ir:internal_representation.IR, node) -> internal_representation.IR:

    # we use this function on changing the mod from 16 to 8, because we need to pad the whole G1 grid
    insert_identity_node(ir=ir, param_source_node = node, name_sufix = 'padding_grid')

    # TODO this function have to be upated here  for nodes that have more than one output  


#This function adds the additional folding node if there is SIZE of picture is 8xN or less
def add_16_to_8_grid_folding(ir: internal_representation.IR) -> internal_representation.IR:
 
    nodes_in_graph = list(ir.graph.nodes(data=True))
    for (node_name, node) in nodes_in_graph:
        if ('force_folding_x' in node['frontend'] and node['frontend']['force_folding_x'] == True and node['frontend']['output_tensor'].get_original_shape()[3]<=8):     
            insert_padding_whole_grid_node_between(ir, node)

    return ir

# Split a conv with N output channels to N/2 and N/2, or (N/2 - 1) if odd.
# The original output tensor is kept in the IR. The 2 new (split) output tensors are made
# to be contiguous in DDR and given the same address locations as the original tensor.
#
# Note:
# - One option is to make a Concat after the split conv nodes, but this may just run
#   into the same AMM problem.
# - If this is an output to a Sync node, should only send the sync once the second
#   half of the tensor is written
def split_conv_out_channels_in_two(ir: internal_representation.IR,original_node_name,node):
    dup_node = copy.deepcopy(node)
    # Override the deepcopy to have pointer to the original tensor and not create a new one.
    # Otherwise it will be a new tensor object (e.g., new address) but just with the same data (name, etc.)
    dup_node['frontend']['input_tensor'] = node['frontend']['input_tensor']
    # Override the deepcopy to have pointer to the original node and not create a new one
    dup_node['frontend']['following_nodes_params'] = node['frontend']['following_nodes_params']
    dup_node['frontend']['preceding_nodes_params'] = node['frontend']['preceding_nodes_params']

    # Set the new node name
    dup_node_name = original_node_name + '_split'
    dup_node['name'] = dup_node_name

    # Update the output channels
    orig_output_channels = node['frontend']['output_channels']
    half_output_channels = orig_output_channels - (orig_output_channels // 2)
    remaining_output_channels = orig_output_channels - half_output_channels
    node['frontend']['output_channels'] = half_output_channels
    dup_node['frontend']['output_channels'] = remaining_output_channels
    
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']    
    folded_half_output_channels = half_output_channels * (2 ** input_folding_factor_x) * (2 ** input_folding_factor_y)
    
    # Update weights
    if 'weights_tensor' in node['frontend']:
        orig_weights_tensor = node['frontend']['weights_tensor']
        orig_weights_tensor_name = orig_weights_tensor.name
        orig_weights_tensor_scale = orig_weights_tensor.scale
        orig_weights_tensor_zp = orig_weights_tensor.zero_point
        # Split the weights by output channel
        weights_tensor_name_split1 = orig_weights_tensor_name + '_split1'
        weights_tensor_name_split2 = orig_weights_tensor_name + '_split2'
        weights_tensor_data_split1 = orig_weights_tensor.data[:half_output_channels, :, :, :]
        weights_tensor_data_split2 = orig_weights_tensor.data[half_output_channels:, :, :, :]
        # Create the new tensors for the weights
        if (isinstance(orig_weights_tensor_scale, int) or isinstance(orig_weights_tensor_scale, float)):
            ir.tensors[weights_tensor_name_split1] = Tensor(
                weights_tensor_name_split1, weights_tensor_data_split1, is_constant=True,
                shape=list(weights_tensor_data_split1.shape),
                scale=orig_weights_tensor_scale,
                zero_point=orig_weights_tensor_zp)
            ir.tensors[weights_tensor_name_split2] = Tensor(
                weights_tensor_name_split2, weights_tensor_data_split2, is_constant=True,
                shape=list(weights_tensor_data_split2.shape),
                scale=orig_weights_tensor_scale,
                zero_point=orig_weights_tensor_zp)
        else:
            ir.tensors[weights_tensor_name_split1] = Tensor(
                weights_tensor_name_split1, weights_tensor_data_split1, is_constant=True,
                shape=list(weights_tensor_data_split1.shape),
                scale=orig_weights_tensor_scale[:half_output_channels],
                zero_point=orig_weights_tensor_zp[:half_output_channels])
            ir.tensors[weights_tensor_name_split2] = Tensor(
                weights_tensor_name_split2, weights_tensor_data_split2, is_constant=True,
                shape=list(weights_tensor_data_split2.shape),
                scale=orig_weights_tensor_scale[half_output_channels:],
                zero_point=orig_weights_tensor_zp[half_output_channels:])
        # Update the weights tensors
        node['frontend']['weights_tensor'] = ir.tensors[weights_tensor_name_split1]
        dup_node['frontend']['weights_tensor'] = ir.tensors[weights_tensor_name_split2]

        if ('biases_tensor' in node['frontend']):
            # Update biases
            orig_biases_tensor = node['frontend']['biases_tensor']
            orig_biases_tensor_name = orig_biases_tensor.name
            orig_biases_tensor_scale = orig_biases_tensor.scale
            orig_biases_tensor_zp = orig_biases_tensor.zero_point
            # Split the biases by output channel
            biases_tensor_name_split1 = orig_biases_tensor_name + '_split1'
            biases_tensor_name_split2 = orig_biases_tensor_name + '_split2'
            biases_tensor_data_split1 = orig_biases_tensor.data[:half_output_channels]
            biases_tensor_data_split2 = orig_biases_tensor.data[half_output_channels:]
            # Create the new tensors for the biases
            if (isinstance(orig_biases_tensor_scale, int) or isinstance(orig_biases_tensor_scale, float)):
                ir.tensors[biases_tensor_name_split1] = Tensor(
                    biases_tensor_name_split1, biases_tensor_data_split1, is_constant=True,
                    shape=list(biases_tensor_data_split1.shape),
                    scale=orig_biases_tensor_scale,
                    zero_point=orig_biases_tensor_zp)
                ir.tensors[biases_tensor_name_split2] = Tensor(
                    biases_tensor_name_split2, biases_tensor_data_split2, is_constant=True,
                    shape=list(biases_tensor_data_split2.shape),
                    scale=orig_biases_tensor_scale,
                    zero_point=orig_biases_tensor_zp)
            else:
                ir.tensors[biases_tensor_name_split1] = Tensor(
                    biases_tensor_name_split1, biases_tensor_data_split1, is_constant=True,
                    shape=list(biases_tensor_data_split1.shape),
                    scale=orig_biases_tensor_scale[:half_output_channels],
                    zero_point=orig_biases_tensor_zp[:half_output_channels])
                ir.tensors[biases_tensor_name_split2] = Tensor(
                    biases_tensor_name_split2, biases_tensor_data_split2, is_constant=True,
                    shape=list(biases_tensor_data_split2.shape),
                    scale=orig_biases_tensor_scale[half_output_channels:],
                    zero_point=orig_biases_tensor_zp[half_output_channels:])
            # Update the biases tensors
            node['frontend']['biases_tensor'] = ir.tensors[biases_tensor_name_split1]
            dup_node['frontend']['biases_tensor'] = ir.tensors[biases_tensor_name_split2]

        # Update scale, zp, and dense/sparse parameters
        for n in [node, dup_node]:
            weights_tensor = n['frontend']['weights_tensor']
            # TODO: May need to adjust this for each node type
            dense_weights = weights_tensor.data.size
            sparse_weights = np.count_nonzero(weights_tensor.data)
            n['frontend']['dense_weights'] = dense_weights
            n['frontend']['sparse_weights'] = sparse_weights
            n['frontend']['weights_sparsity'] = 1- (sparse_weights/dense_weights)

            current_op_input_tensor_name = n['inputs'][0]
            stride = n['frontend']['stride']
            current_op_input_tensor_shape = ir.tensors[current_op_input_tensor_name].get_original_shape()
            dense_macs = dense_weights*current_op_input_tensor_shape[2]*current_op_input_tensor_shape[3]/(stride*stride)
            sparse_macs = sparse_weights*current_op_input_tensor_shape[2]*current_op_input_tensor_shape[3]/(stride*stride)
            n['frontend']['dense_macs'] = dense_macs
            n['frontend']['sparse_macs'] = sparse_macs
            n['frontend']['macs_sparsity'] = 1- (sparse_macs/dense_macs)
            n['frontend']['weights_per_channel_scale'] = weights_tensor.scale
            n['frontend']['weights_per_channel_zp'] = weights_tensor.zero_point

    if 'backend' in node:
        output_channels_split = node['backend']['oc_splits']
        node['backend']['output_channels'] = folded_half_output_channels
        node['backend']['oc_groups'] = [[oc for oc in range(0+i, folded_half_output_channels, output_channels_split)] for i in range(output_channels_split)] 
        dup_node['backend']['output_channels'] = folded_half_output_channels
        dup_node['backend']['oc_groups'] = [[oc for oc in range(0+i, folded_half_output_channels, output_channels_split)] for i in range(output_channels_split)] 

    # Create two separate output tensors, but store the pair in the IR and force them to be
    # side-by-side in DDR. The original output tensor is also kept and in the same DDR location
    # as the 2 split tensors. This is because the consumer of the original output tensor still
    # needs to have a single input tensor, and adding a Concat may result in AMM errors.
    orig_output_tensor_name = node['outputs'][0]
    orig_output_tensor = ir.tensors[orig_output_tensor_name]

    output_tensor_split1 = copy.deepcopy(orig_output_tensor)
    output_tensor_name_split1 = orig_output_tensor_name + '_split1'
    if output_tensor_split1.data is not None:
        output_tensor_data_split1 = output_tensor_split1.data[:, :half_output_channels, :, :]
        output_tensor_split1.data = output_tensor_data_split1
        output_tensor_split1.shape = list(output_tensor_data_split1.shape)
        output_tensor_split1.shape_real_x16[1] = half_output_channels
    else:
        output_tensor_split1.data = None
        output_tensor_split1.shape = [
            orig_output_tensor.shape[0],
            half_output_channels,
            orig_output_tensor.shape[2],
            orig_output_tensor.shape[3]
        ]
        output_tensor_split1.shape_real_x16 = [
            orig_output_tensor.shape_real_x16[0],
            half_output_channels,
            orig_output_tensor.shape_real_x16[2],
            orig_output_tensor.shape_real_x16[3]
        ]
    output_tensor_split1.name = output_tensor_name_split1
    output_tensor_split1.producer = node['name']
    node['outputs'][0] = output_tensor_name_split1
    node['frontend']['output_tensor'] = output_tensor_split1

    output_tensor_split2 = copy.deepcopy(orig_output_tensor)
    output_tensor_name_split2 = orig_output_tensor_name + '_split2'
    if output_tensor_split2.data is not None:
        output_tensor_data_split2 = output_tensor_split2.data[:, half_output_channels:, :, :]
        output_tensor_split2.data = output_tensor_data_split2
        output_tensor_split2.shape = list(output_tensor_data_split2.shape)
        output_tensor_split2.shape_real_x16[1] = orig_output_tensor.shape[1] - half_output_channels
    else:
        output_tensor_split2.data = None
        output_tensor_split2.shape = [
            orig_output_tensor.shape[0],
            orig_output_tensor.shape[1] - half_output_channels,
            orig_output_tensor.shape[2],
            orig_output_tensor.shape[3]
        ]
        output_tensor_split2.shape_real_x16 = [
            orig_output_tensor.shape_real_x16[0],
            orig_output_tensor.shape[1] - half_output_channels,
            orig_output_tensor.shape_real_x16[2],
            orig_output_tensor.shape_real_x16[3]
        ]
    output_tensor_split2.name = output_tensor_name_split2
    output_tensor_split2.producer = dup_node['name']
    dup_node['outputs'][0] = output_tensor_name_split2
    dup_node['frontend']['output_tensor'] = output_tensor_split2

    ir.tensors[output_tensor_name_split1] = output_tensor_split1
    ir.tensors[output_tensor_name_split2] = output_tensor_split2

    # Now add the new node to the IR
    # When the node is created it copies the dictionary attributes and create a new dict
    ir.graph.add_node(dup_node_name,**dup_node)
    dup_node = ir.graph.nodes[dup_node_name] # Dan: very important to get the dict created by node

    # If consumers are being updated in the future, may need to update the preceding nodes params
    # of the consumer node to be e.g. the duplicate node instead. See the bottom of
    # convert_maxpoolk5_to_2xmaxpoolk3 for how this can be done.

    # Similarly, add output edges from original node to duplicate node if there were any
    edges_from_original_node = list(ir.graph.out_edges(original_node_name))
    for edge in edges_from_original_node:
        ir.graph.add_edge(dup_node_name,edge[1])

    # Add an edge from the preceding node of the original node to the duplicate
    if (len(node['frontend']['preceding_nodes_params']) > 0):
        preceding_node_of_original_node_name = node['frontend']['preceding_nodes_params'][0][0]
        ir.graph.add_edge(preceding_node_of_original_node_name, dup_node_name)
        # Update following nodes of this preceding node as well
        preceding_node_of_original_node = ir.graph.nodes[preceding_node_of_original_node_name]
        preceding_node_of_original_node['frontend']['following_nodes_params'].append((dup_node_name,0))

    # Update consumers of the input tensor
    node['frontend']['input_tensor'].consumers.append(dup_node_name)

    # Update data structures to associate the split tensors with the original tensors
    ir.split_tensor_to_original_tensor_map[output_tensor_name_split1] = orig_output_tensor_name
    ir.split_tensor_to_original_tensor_map[output_tensor_name_split2] = orig_output_tensor_name
    # The order [split2, split1] is used because move_outputs_write_to_amm_to_erliest_point
    # iterates over the outputs sequentially and moves them earlier in the graph so this ensures
    # the correct final order.
    ir.original_tensor_to_split_tensor_map[orig_output_tensor_name] = [
        output_tensor_name_split2, output_tensor_name_split1
    ]

    # Also update the tensors to MXP. If the original tensor was going to the MXP, then the
    # 2 split tensors also need to.
    # Note: the Sync will only happen when the second writes (this is done in the sequencer)
    if orig_output_tensor_name in ir.tensors_to_mxp:
        ir.tensors_to_mxp.add(output_tensor_name_split1)
        ir.tensors_to_mxp.add(output_tensor_name_split2)
        ir.mxp_tensor_to_offset[output_tensor_name_split1] = ir.mxp_tensor_to_offset[orig_output_tensor_name]
        ir.mxp_tensor_to_offset[output_tensor_name_split2] = ir.mxp_tensor_to_offset[orig_output_tensor_name] + np.prod(output_tensor_split1.shape)

# If a Conv node has too many output channels, split it into 2 with half the output channels each
def split_large_conv_nodes(ir: internal_representation.IR) -> internal_representation.IR:
    nodes_in_graph = list(ir.graph.nodes(data=True))
    for (node_name, node) in nodes_in_graph:

        # No point to split concat
        if node['op_type'] == "Concat" or node['op_type'] == "Add":
            continue

        # Check if this node will exceed 2k AMM
        # Note: This is a heuristic which uses the folded input + output channels and checks
        # if > 1024, half the AMM size. But this does not consider:
        # - the last tile of a blob loads the first tile tensors of the next blob
        # - concat needs to load contiguous memory

        # TODO: Handle Add later
        input_channels = node['frontend']['input_channels']
        input_x_fold = node['frontend']['input_folding_factor_x']
        input_y_fold = node['frontend']['input_folding_factor_y']
        input_total_channels = input_channels * pow(2, input_x_fold + input_y_fold)

        output_channels = node['frontend']['output_channels']
        output_x_fold = node['frontend']['output_folding_factor_x']
        output_y_fold = node['frontend']['output_folding_factor_y']
        output_total_channels = output_channels * pow(2, output_x_fold + output_y_fold)

        input_total_channels = input_total_channels * node['frontend']['input_tensor'].x_slices
        output_total_channels = output_total_channels * node['frontend']['output_tensor'].x_slices
        if (input_total_channels + output_total_channels > ((URAM_BLOCK_SIZE * URAM_NUM_BLOCKS) // 2)):
            split_conv_out_channels_in_two(ir,node_name,node)
    return ir

def mark_nodes_for_folding_factor_change():
    for (node_name, node) in ir.graph.nodes(data=True):
        pass

def calc_nodes_folding_factor(ir):
    set_folding_factor_attempt = 0
    successful_folding_factor = False
    sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    MAX_FOLDING_FACTOR_SETTINGS_RETRIES = 10
    while set_folding_factor_attempt<MAX_FOLDING_FACTOR_SETTINGS_RETRIES and not successful_folding_factor:
        successful_folding_factor = True
        tqdm_iterator = tqdm(sorted_graph)
        for node_name in tqdm_iterator:
            node = ir.graph.nodes[node_name]
            tqdm_iterator.set_description('Folding factor propogation, at layer %s:' % node_name)
            if node['op_type'] == 'Conv':
                calc_conv_folding_factor(ir,node_name,node)
            elif node['op_type'] == 'AveragePool':
                calc_avgpool_folding_factor(ir,node_name,node)
            elif node['op_type'] == 'Gemm':
                calc_gemm_folding_factor(ir,node_name,node)
            elif node['op_type'] == 'MaxPool':
                calc_maxpool_folding_factor(ir,node_name,node)
            elif node['op_type'] == 'Add':
                calc_add_folding_factor(ir,node_name,node)
            elif node['op_type'] == 'GlobalAveragePool':
                calc_globalavg_pooling_folding_factor(ir,node_name,node)
            elif node['op_type'] == 'Concat':
                if not calc_concat_folding_factor(ir,node_name,node): # If we encountered concat inputs with different folding factor we mark its source conv for folding and re-try
                    successful_folding_factor = False
                    print('\nUpdated folding factor because of Concat op. Re-propogating folding factor...')
                    break
            elif node['op_type'] == 'Resize':
                calc_resize_folding_factor(ir,node_name,node)
            elif node['op_type'] == 'Identity':
                calc_identity_folding_factor(ir,node_name,node)
            elif node['op_type'] == 'Sync':
                # Copying these values so report writing does not fail. Alternative solution
                # is to edit report function to print blank if these values are not present.
                copy_folding_factor_from_prev_node(node)
            else:
                raise ValueError ('WARNING: Found an unsupported op of type ' + node['op_type'])
                calc_general_folding_factor(ir,node_name,node)
        set_folding_factor_attempt+=1
    if set_folding_factor_attempt==MAX_FOLDING_FACTOR_SETTINGS_RETRIES:
        raise ValueError ('Failed to set good folding factor... please check')
    return ir

def calc_nodes_quantization_params(ir):
    sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    tqdm_iterator = tqdm(sorted_graph)
    for node_name in tqdm_iterator:
        node = ir.graph.nodes[node_name]
        tqdm_iterator.set_description('Calculating quantization params and folding weights, at layer %s:' % node_name)
        if node['op_type'] == 'Conv':
            calc_conv_qparams(ir,node_name,node)
        elif node['op_type'] == 'AveragePool':
            calc_avgpool_qparams(ir,node_name,node) 
            node['op_type'] = 'Conv' # Alex: We change the op type to conv so that report writing works properly
        elif node['op_type'] == 'Gemm':
            calc_gemm_qparams(ir,node_name,node)
        elif node['op_type'] == 'MaxPool':
            calc_maxpool_qparams(ir,node_name,node)
        elif node['op_type'] == 'Add':
            calc_add_qparams(ir,node_name,node)
        elif node['op_type'] == 'Identity':
            calc_identity_qparams(ir,node_name,node)
        elif node['op_type'] == 'Concat':
            calc_concat_qparams(ir,node_name,node)
        elif node['op_type'] == 'Resize':
            calc_resize_qparams(ir,node_name,node)
        elif node['op_type'] == 'Sync':
            calc_sync_qparams(ir,node_name,node)
        else:
            raise ValueError ('Found an unsupported op of type ' + node['op_type'])
    return ir
