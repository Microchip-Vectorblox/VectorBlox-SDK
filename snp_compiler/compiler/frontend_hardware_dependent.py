import sys
sys.path.append('.')
import numpy as np
import networkx as nx
import common.internal_representation as internal_representation
from common.tensor_ir import Tensor
from common.ddr_ir import TensorDDREntry, create_tsnp_tensor_byte_array
from common.hw_config import FRACTIONAL_BITS, BIAS_MULITIPLIER, BIAS_ROUNDING_ADD, MAX_BIAS_BITS, \
                    MAX_GRID_WIDTH, MAX_REDUCE_BUS_WIDTH, URAM_DEPTH, URAM_BLOCK_SIZE, LONG_ENTRY_BITS, OpType, MULTIPLE_INPUT_OPS, \
                    amm_capacity, generic_op_footprint
from common.debug_flags import DEBUG_FORCE_FOLDCONV,DEBUG_CLIP_BIAS_TO_MAX_BIAS_BITS, DEBUG_SIMULATE_CONCAT_REQUANT
from common.utils import LUTPopulateInt8, quantize2MathBlock, int8min, int8max, get_xslices_per_htile, greedy_packed_xslice_groups, pad_k3_folding_output_xslices_per_htile
import folding_algo
import copy
import math
from tqdm import tqdm

def round_up_to_next_16(x):
    return ((x + (MAX_GRID_WIDTH-1)) // MAX_GRID_WIDTH) * MAX_GRID_WIDTH

# Mirrors program_compiler.py's _add_tensor_to_memo (duplicated, not imported, to avoid a
# circular import). Safe to hand deepcopy the original array by reference: no split path here
# mutates a constant tensor's .data/.scale/.zero_point in place, only reassigns to new arrays.
def _add_tensor_to_memo(memo, tensor):
    if getattr(tensor, 'is_constant', False):
        for arr in (tensor.data, getattr(tensor, 'scale', None), getattr(tensor, 'zero_point', None)):
            if isinstance(arr, np.ndarray):
                memo[id(arr)] = arr

def _memo_for_node(node):
    memo = {}
    for value in node.get('frontend', {}).values():
        if isinstance(value, Tensor):
            _add_tensor_to_memo(memo, value)
    return memo

def set_tflite_rq_params(node, acc, bias_data, scale, output_offset, output_activation_min=int8min, output_activation_max=int8max):
    output_multiplier, cInputH, cInputL, o_shift = \
        quantize2MathBlock(acc, bias_data, scale, output_offset, output_activation_min, output_activation_max)
    node['frontend']['output_multiplier'] = output_multiplier
    node['frontend']['cInputH'] = cInputH
    node['frontend']['cInputL'] = cInputL
    node['frontend']['o_shift'] = o_shift

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
    if (kernel_size <= 3) and (stride == 1):
        x_slices = original_input_width / MAX_GRID_WIDTH
        folding_factor_x = 0
    elif (kernel_size <= 3) and (stride == 2):
        if is_stride2:
            x_slices = (original_input_width/4)/MAX_GRID_WIDTH
            folding_factor_x = 2
        else:
            x_slices = (original_input_width/2)/MAX_GRID_WIDTH
            folding_factor_x = 1
    elif (kernel_size > 3) and (kernel_size <= 5):
        x_slices = (original_input_width/2)/MAX_GRID_WIDTH
        folding_factor_x = 1
    else:
        x_slices = (original_input_width/4)/MAX_GRID_WIDTH
        folding_factor_x = 2
    
    return int(folding_factor_x), math.ceil(x_slices)

def get_new_input_folding_factors(original_input_width,original_input_height,stride,kernel_size,original_input_channels,original_output_channels,amm_height,is_stride2 = False):
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
        if original_input_height//input_y_folding_coef<=amm_height:
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

def get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size,amm_height):
    input_folding_coef_y = math.pow(2,input_folding_factor_y)
    folded_y_size = math.ceil(original_input_height / input_folding_coef_y)
    if original_kernel_size == 1:
        num_y_tiles = math.ceil(folded_y_size / amm_height)
    else:
        if folded_y_size<=amm_height:
            num_y_tiles = 1
        else:
            handled_lines=amm_height-1
            num_y_tiles=1
            while (folded_y_size>handled_lines):
                if (handled_lines+(amm_height-1))>=folded_y_size:
                    num_y_tiles+=1
                    break
                handled_lines+=(amm_height-2)
                num_y_tiles+=1
    return num_y_tiles

def get_num_h_tiles(node):
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
    amm_total_channels = input_total_channels + output_total_channels

    num_h_tiles = 1
    if (amm_total_channels > 1500):
        num_h_tiles = math.ceil(amm_total_channels / 1500)
        # h_tiles can never exceed the node's own x_slices (it splits x_slices across h-tiles, see
        # get_xslices_per_htile): e.g. x_slices=1 can't split into 2, forcing h_tiles=2 anyway gives
        # a degenerate xslices_per_htile=[1,0] mem alloc/sequencer code doesn't expect. Same invariant
        # as apply_blob_htiling (force_folding_x blobs) and update_htiling_info (incremental growth).
        max_h_tiles = max(1, min(node['frontend']['input_tensor'].x_slices, node['frontend']['output_tensor'].x_slices))
        num_h_tiles = min(num_h_tiles, max_h_tiles)

    return num_h_tiles

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
    if ir.graph.nodes[producing_node_name]['op_type'] == OpType.RESIZE:
        return True

    input_tensor_size_on_amm = node['frontend']['input_tensor'].get_size_on_amm(ir.amm_block_size)
    output_tensor_size_on_amm = node['frontend']['output_tensor'].get_size_on_amm(ir.amm_block_size)
    is_output = node['frontend']['output_tensor'].name in ir.outputs
    if is_amm_overflow(input_tensor_size_on_amm, output_tensor_size_on_amm, is_output):
        return True

    # Because Concat Y unfolding is not implemented yet, check if this Conv output
    # is a Concat which will not fit in AMM. If it is, unfold here instead.
    if node['op_type'] == OpType.CONV:
        for consumer in node['frontend']['output_tensor'].consumers:
            consumer_node = ir.graph.nodes()[consumer]
            if consumer_node['op_type'] != OpType.CONCAT:
                continue

            # Found a Concat output. Get total input size and output size.
            concat = consumer_node
            total_input_tensor_size_on_amm = 0
            for input_tensor in concat['frontend']['input_tensors']:
                total_input_tensor_size_on_amm += input_tensor.get_size_on_amm(ir.amm_block_size)
            output_tensor_size_on_amm = concat['frontend']['output_tensor'].get_size_on_amm(ir.amm_block_size)
            is_output = concat['frontend']['output_tensor'].name in ir.outputs
            if is_amm_overflow(total_input_tensor_size_on_amm, output_tensor_size_on_amm, is_output):
                return True

    return False

def _is_identity_conv(node):
    # True only for a synthesized 1x1/stride-1 identity conv (e.g. the dtype-cast conv
    # build_identity_conv_for_quantize inserts at the graph entry): diagonal weight tensor,
    # OC==IC, zero off-diagonal, nonzero diagonal. A real weighted 1x1 conv (e.g. a MobileNet
    # expand/project conv) fails this and must not be treated as a transparent pass-through
    # when determining folding requirements, since it does real work at its blob.
    fe = node['frontend']
    if fe.get('kernel_size') != 1 or fe.get('stride') != 1:
        return False
    weights_tensor = fe.get('weights_tensor')
    if weights_tensor is None:
        return False
    data = weights_tensor.data
    if data.ndim != 4 or data.shape[0] != data.shape[1] or data.shape[2] != 1 or data.shape[3] != 1:
        return False
    oc = data.shape[0]
    square = data.reshape(oc, oc)
    eye_mask = np.eye(oc, dtype=bool)
    return bool(np.all(square[~eye_mask] == 0) and np.all(square[eye_mask] != 0))

def get_next_non_folding_conv(ir:internal_representation.IR,node):
    current_node = node
    while _is_identity_conv(current_node):
        successors = list(ir.graph.successors(current_node['name']))
        # For now consider single successor case
        if len(successors) != 1:
            break
        next_node = ir.graph.nodes()[successors[0]]
        if 'kernel_size' not in next_node['frontend'] or 'stride' not in next_node['frontend']:
            break
        current_node = next_node
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
    if kernel_size > 5:
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

def absorb_x_fold_step_on_preceding_node(preceding_node):
    # Instead of inserting a synthetic 1x1 identity fold conv, have preceding_node's own
    # already-computed output absorb the extra x-fold step directly - safe since preceding_node
    # is a real conv known to have only this one consumer. Mid-graph, single-consumer
    # counterpart of _apply_one_x_fold_step (program_compiler.py), which does the same for
    # entry-chain nodes.
    frontend = preceding_node['frontend']
    output_tensor = frontend['output_tensor']
    input_folding_x = frontend['output_folding_factor_x']
    input_xslices = output_tensor.x_slices

    output_folding_x = input_folding_x + 1
    output_xslices = math.ceil(input_xslices / 2)

    frontend['force_folding_x'] = True
    frontend['is_x_fold_node'] = True
    frontend['output_folding_factor_x'] = output_folding_x
    output_tensor.folding_factor_x = output_folding_x
    output_tensor.x_slices = output_xslices
    h_tiles = frontend['h_tiles'] if 'h_tiles' in frontend else get_num_h_tiles(preceding_node)
    kernel_size = frontend['kernel_size'] if 'kernel_size' in frontend else 1  # identity convs are 1x1
    output_tensor.xslices_per_htile = get_xslices_per_htile(output_xslices, h_tiles, kernel_size)

def add_xfoldingConv_before_node(ir:internal_representation.IR, node, step=0):
    # step is which x-fold step (0-indexed) this inserted conv performs: it takes an
    # input already folded `step` times and produces output folded `step+1` times.
    # Defaults to 0 (the original single-step behavior: unfolded in, folded-once out).
    # Multiple steps for the same node are chained by calling this repeatedly with
    # increasing step - each call's node name is suffixed with the step index so
    # they don't collide in ir.graph.
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    channels = original_input_tensor_shape[1]
    op_input_name = node['frontend']['input_tensor'].name
    new_node_name = node['name'] + '_fold_x_' + str(step)
    op_output_name = new_node_name+'_out'
    conv_W_initializer_tensor_name = new_node_name+"_W"
    conv_B_initializer_tensor_name = new_node_name+"_B"
    ir.tensors[op_output_name] = Tensor(op_output_name,np.zeros(original_input_tensor_shape),is_constant=False, shape=original_input_tensor_shape)
    # No producer means the input tensor is a genuine raw graph input (no preceding IR node,
    # e.g. plain INT8 input with no MXP/quantize preprocessing) - nothing to fuse an activation
    # from, key qparams/folding off of, or add a graph edge to/from.
    _producer_name = ir.tensors[op_input_name].producer
    source_node = ir.graph.nodes[_producer_name] if _producer_name else None

    # The input tensor stores the producer's PRE-activation quantization, but when the producer
    # fuses a silu/requant activation the data physically arriving here is the activation OUTPUT
    # (same post-activation scale/zp the real consumer reads via update_input_scale_zp). Use
    # post-activation qparams when present, else fall back to the raw tensor; cast the silu/requant
    # float/int attrs to the raw tensor's numpy dtype since calc_conv_qparams expects numpy scalars.
    _raw_scale = node['frontend']['input_tensor'].scale
    _raw_zp = node['frontend']['input_tensor'].zero_point
    _prod_attrs = (source_node.get('attributes', {}) or {}) if source_node is not None else {}
    _act = _prod_attrs.get('activation_silu') or _prod_attrs.get('requant_info')
    if _act is not None:
        eff_input_scale = np.asarray(_act['output_scale'][0], dtype=np.asarray(_raw_scale).dtype)[()]
        eff_input_zp = np.asarray(_act['output_zp'][0], dtype=np.asarray(_raw_zp).dtype)[()]
    else:
        eff_input_scale = _raw_scale
        eff_input_zp = _raw_zp

    ir.tensors[op_output_name].scale = eff_input_scale
    ir.tensors[op_output_name].zero_point = eff_input_zp
    
    # This node is a pure X-fold passthrough (no Y-folding of its own), so y_tiles must reflect
    # whatever Y-folding the earlier global pass already decided, not assume unfolded (0) -
    # hardcoding 0 mismatches the determining conv's real y_tiles whenever it needs Y-folding,
    # causing later DDR-tile-read indexing to go out of range.
    y_tiles= get_num_y_tiles(original_input_height=node['frontend']['input_tensor'].shape[2],input_folding_factor_y=node['frontend']['input_tensor'].folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
    
    # Create the new node
    # Only this folding conv reads the padded concat layout, so the producer concat's block maps go on it.
    if source_node is not None and source_node['op_type'] == OpType.CONCAT:
        _maps = source_node['frontend'].get('concat_block_maps')
    else:
        _maps = None
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

    # When there's no producer node (raw graph input), fall back to the input tensor's own
    # existing folding_factor_y (set by the earlier global Y-folding pass) rather than assuming
    # unfolded (0) - this passthrough node must not silently change the decided Y-folding.
    _source_input_folding_factor_y = (source_node['frontend']['input_folding_factor_y'] if source_node is not None
                                       else node['frontend']['input_tensor'].folding_factor_y)
    if _maps is not None:
        new_node['frontend']['aligned_input_ch'] = _maps
    new_node['frontend'].update({
        'input_channels': channels,
        'output_channels': channels,  # Same as input
        # Marks that input_channels/output_channels already have the y-fold multiplier baked in
        # (channels = raw * 2^y_fold * 2^step), unlike a real conv's output_channels which stays
        # the raw weight-tensor count. calculate_AMM_write_add_for_folding_case's caller divides
        # by 2**output_folding_factor_y to recover per-fold-group count - only correct when this
        # flag is set, else that division would floor a non-y-folded count to 0 (divide-by-zero).
        'input_folding_factor_x': step,
        'input_folding_factor_y': _source_input_folding_factor_y,
        'output_folding_factor_x': step + 1,
        'output_folding_factor_y': _source_input_folding_factor_y,
        'preceding_nodes_params': [(source_node['name'], 0)] if source_node is not None else [],
        'input_tensor': ir.tensors[op_input_name],
        'output_tensor': ir.tensors[op_output_name],
        'input_tensor_scale': eff_input_scale,
        'input_tensor_zp': eff_input_zp,
        'output_tensor_scale': eff_input_scale,
        'output_tensor_zp': eff_input_zp,
        'kernel_size': 1,
        'stride': 1,
        'padding':new_node['attributes']['pads'],
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
    # See _apply_one_x_fold_step's identical flag: every fold step must be structurally recognized
    # as a fold node so apply_blob_htiling's interior-degeneracy check (single x-slice in a
    # non-last h-tile, which can't hold a fold pair) fires; otherwise an invalid split like
    # [2,1,1] can survive.
    new_node['frontend']['is_x_fold_node'] = True
    new_node['frontend']['output_tensor'].folding_factor_x = step + 1
    new_node['frontend']['output_tensor'].folding_factor_y = _source_input_folding_factor_y
    x_slices = node['frontend']['input_tensor'].x_slices
    new_node['frontend']['output_tensor'].x_slices = math.ceil(x_slices/2)
    new_node['frontend']['x_slices'] = x_slices
    new_node['frontend']['h_tiles'] = get_num_h_tiles(new_node)
    new_node['frontend']['xslices_per_htile'] = get_xslices_per_htile(new_node['frontend']['x_slices'], new_node['frontend']['h_tiles'], new_node['frontend']['kernel_size'])
    fold_output_xslices_per_htile = get_xslices_per_htile(new_node['frontend']['output_tensor'].x_slices, new_node['frontend']['h_tiles'], new_node['frontend']['kernel_size'])
    # The real consumer `node` already computed its own xslices_per_htile (calc_conv_folding_factor,
    # just before this call) including the extra halo slice its kernel_size needs, and this fold
    # node's output feeds it directly, so it must produce at least as many slices per h-tile. The
    # fold's own kernel_size=1 computation under-counts by exactly one slice, but ONLY for h-tile 0
    # - later h-tiles already carry the halo slice via get_xslices_per_htile's last-h-tile
    # accumulation. Only bump h-tile 0 to avoid over-provisioning AMM/DDR-offload lanes elsewhere.
    consumer_xslices_per_htile = node['frontend']['xslices_per_htile']
    if consumer_xslices_per_htile:
        fold_output_xslices_per_htile[0] = max(fold_output_xslices_per_htile[0], consumer_xslices_per_htile[0])
    new_node['frontend']['output_tensor'].xslices_per_htile = fold_output_xslices_per_htile

    # Mirror calc_conv_folding_factor's/_apply_one_x_fold_step's entry-packing: when this synthetic
    # fold node IS the model's true entry (step 0, source_node is None), pack input x-slices into
    # AMM lanes greedily (largest-first) instead of a uniform divisor - same as the trivial-chain
    # entry-absorption path. Needed for models whose determining conv needs x-folding immediately
    # at the entry (see distribute_entry_x_folding's deficit>0 branch, the only caller passing
    # step==0 with no real predecessor), which would otherwise silently skip packing.
    if step == 0 and source_node is None and channels < 16 and (original_input_tensor_shape[3] % 16 == 0):
        ir.tensors[op_input_name].num_packed_xslices = greedy_packed_xslice_groups(new_node['frontend']['xslices_per_htile'][0])

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
    # `node` must always be redirected to the new fold node regardless of its own stride (not
    # just the original stride-2 self-referential case - also covers inserting missing fold
    # steps ahead of a stride-1 real conv, or the true graph entry). Other stride-2 sibling
    # consumers of the same tensor are also redirected.
    target_node_names = [node['name']]
    for idx in range(len(input_consumers)):
        consumer_name = input_consumers[idx]
        if consumer_name == node['name']:
            continue
        next_node_info = ir.graph.nodes[consumer_name]
        if ('stride' in next_node_info['frontend']) and (next_node_info['frontend']['stride'] == 2):
            target_node_names.append(consumer_name)
    if source_node is not None:
        ir.graph.add_edge(source_node['name'],new_node_name)
    for i_name in target_node_names:
        ir.graph.add_edge(new_node_name,i_name)
        if source_node is not None:
            ir.graph.remove_edge(source_node['name'],i_name)
        index = input_consumers.index(i_name)
        ir.tensors[op_input_name].consumers[index] = new_node_name
        ir.tensors[op_output_name].consumers.append(i_name)
    ir.tensors[op_input_name].consumers = list(set(ir.tensors[op_input_name].consumers))
    ir.tensors[op_output_name].producer = new_node_name

    # Update the 'inputs' and input_tensor field in target node
    for i_name in target_node_names:
        next_node_info = ir.graph.nodes[i_name]
        # The folding conv now carries the padded concat layout; the consumers it feeds must not keep it.
        next_node_info['frontend'].pop('aligned_input_ch', None)
        ir.switch_input_name(next_node_info,original_input_name=op_input_name,new_input_name=op_output_name)
        ir.switch_input_tensor(next_node_info,original_input_tensor=ir.tensors[op_input_name],new_input_tensor=ir.tensors[op_output_name])

    # Update following nodes params
    if source_node is not None:
        source_node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(source_node) # Update the following_nodes_params field in source node - It is important to also update following nodes according to execution order
    for i_name in target_node_names:
        next_node_info = ir.graph.nodes[i_name]
        next_node_info['frontend']['preceding_nodes_params'] = ir.get_updated_preceding_nodes(next_node_info) # Update the preceding_nodes_params field in target node
    created_new_node['frontend']['following_nodes_params'] = ir.get_updated_following_nodes(new_node)

    return ir

def calc_conv_folding_factor(ir:internal_representation.IR,node_name,node):
    # We check y force folding here since it affects the input folding
    need_to_fold_y_for_concat = (node['name'] in ir.marked_nodes_for_folding_y) 
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

        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
    else:
        # This node's own kernel/stride may be a trivial 1x1/stride-1 identity conv that doesn't
        # itself determine x-folding need - that's driven by the first "real" conv reached by
        # walking forward through such identity nodes.
        determining_conv = get_next_non_folding_conv(ir, node)
        determining_kernel_size = determining_conv['frontend']['kernel_size']
        determining_stride = determining_conv['frontend']['stride']

        is_stride2 = False
        if (len(determining_conv['frontend']['following_nodes_params']) > 0):
            following_node_name = determining_conv['frontend']['following_nodes_params'][0][0]
            following_node = ir.graph.nodes[following_node_name]
            if ('stride' in following_node['frontend'] and  following_node['frontend']['stride'] == 2):
                is_stride2 = True
        real_input_width  = round_up_to_next_16(original_input_width)

        real_input_height = round_up_to_next_16(original_input_height)
        input_folding_factor_x,input_folding_factor_y, x_slices = get_new_input_folding_factors(real_input_width,real_input_height,determining_stride,determining_kernel_size,original_input_channels,original_output_channels,ir.amm_height,is_stride2 = is_stride2)

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
            num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
        elif ('Conv16x32_k1_identity' in ir.model_name):
            input_folding_factor_x = 1
            x_slices = x_slices // 2
        elif ('Conv32x16_k1_identity' in ir.model_name):
            input_folding_factor_y = 1
            num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
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

            num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)

            # X folding mirrors the Y case: a chain of N consecutive stride-2 convs at the entry
            # needs N x-fold levels. get_new_input_folding_factors/update_x_fold_slice above only
            # look at the determining conv's own kernel/stride (plus a 1-node is_stride2
            # lookahead), so they under-count a chain longer than 2.
            num_consecutive_stride2 = get_num_consecutive_stride2_conv(ir, first_conv)
            if num_consecutive_stride2 > input_folding_factor_x:
                input_folding_factor_x = num_consecutive_stride2
                x_slices = math.ceil((real_input_width / (2 ** input_folding_factor_x)) / MAX_GRID_WIDTH)
        node['frontend']['input_tensor'].folding_factor_x = input_folding_factor_x
        node['frontend']['input_tensor'].folding_factor_y = input_folding_factor_y # We update input folding factor only if its re-calced from scratch so that input folding/unfolding in y will not change it
        node['frontend']['input_tensor'].x_slices = x_slices

    # A kernel >=4x4 needs a folded input regardless of this node's own stride or
    # graph position (unlike the stride==2 underflow fixup below, which only fires
    # for stride==2) - see get_required_folding_factor_for_kernel_size.
    kernel_fold_req = get_required_folding_factor_for_kernel_size(original_kernel_size)
    force_kernel_x_folding = kernel_fold_req > input_folding_factor_x
    absorbed_fold_on_preceding = False
    if force_kernel_x_folding:
        preceding_nodes_params = node['frontend'].get('preceding_nodes_params', [])
        if len(preceding_nodes_params) == 1:
            preceding_node = ir.graph.nodes[preceding_nodes_params[0][0]]
            preceding_frontend = preceding_node.get('frontend', {})
            # Only absorb onto a real conv that exists solely to feed this node (single
            # follower) and whose output isn't already spoken for by some other folding
            # requirement - otherwise fall back to inserting the synthetic fold conv below.
            if (preceding_node.get('op_type') == OpType.CONV
                    and len(preceding_frontend.get('following_nodes_params', [])) == 1
                    and preceding_frontend.get('output_folding_factor_x', 0) == 0
                    and 'force_folding_x' not in preceding_frontend
                    and 'force_unfolding_x' not in preceding_frontend):
                absorb_x_fold_step_on_preceding_node(preceding_node)
                absorbed_fold_on_preceding = True
    if force_kernel_x_folding:
        # The x-fold this kernel size needs is local to this node's own compute - the
        # next node (e.g. a 1x1 conv) has no need for x-folded input, so unfold right
        # after this node instead of letting the fold ride all the way to the graph's
        # output (which is what the generic is_output/to_mxp unfold-insertion below does).
        node['frontend']['unfold_x_after_node'] = True
    if kernel_fold_req > input_folding_factor_y:
        input_folding_factor_y = kernel_fold_req
        node['frontend']['force_folding_y'] = True
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
    if force_kernel_x_folding:
        input_folding_factor_x = kernel_fold_req
        x_slices = math.ceil((round_up_to_next_16(original_input_width) / (2 ** input_folding_factor_x)) / MAX_GRID_WIDTH)

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
            if ((current_following_node['op_type'] == OpType.CONV) and (current_following_node['frontend']['stride'] == 2)):
                if (len(following_nodes_params) == 1):
                    force_folding_x = True

            
    if 'force_folding_x' in node['frontend']: # There are cases where 1st pass of folding factor calc fails and at later pass a folding conv is change to non folding conv
        del node['frontend']['force_folding_x']

    # If the preceding node needs x-folding and this node is its only follower, propagate the
    # x-folding forward onto this node (unless already forced). Skip if the preceding node's
    # force_folding_x came from absorb_x_fold_step_on_preceding_node for THIS node's own
    # force_kernel_x_folding (already accounted for below) - or if this node's own
    # input_folding_factor_x is already non-zero (already accounted for via the stride formula
    # above) - either way propagating again would fold the output a second time.
    force_folding_x_from_preceding = False
    if 'force_folding_x' not in node['frontend'] and not absorbed_fold_on_preceding and input_folding_factor_x == 0:
        for preceding_node_params in node['frontend'].get('preceding_nodes_params', []):
            preceding_node = ir.graph.nodes[preceding_node_params[0]]
            if preceding_node['frontend'].get('force_folding_x', False):
                preceding_following_nodes_params = preceding_node['frontend'].get('following_nodes_params', [])
                if len(preceding_following_nodes_params) == 1:
                    force_folding_x_from_preceding = True
                    break

    # Check if we need to fold to match following concat grid params
    need_to_fold_x_for_concat = node['name'] in ir.marked_nodes_for_folding_x
    force_x_folding = need_to_fold_x_for_concat or node_name in ir.force_x_folding or force_folding_x or force_folding_x_from_preceding

    output_xslices = x_slices
    if 'tsnp_x_folding' in ir.model_name and node_name=='conv1':
        node['frontend']['force_folding_x'] = True
        output_folding_factor_x += 1
        output_xslices = output_xslices // 2

    add_folding_conv_before_node = force_kernel_x_folding
    insert_synthetic_fold_node_before = force_kernel_x_folding and not absorbed_fold_on_preceding
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
        output_xslices = round_up_to_next_16(original_input_width)//MAX_GRID_WIDTH

        if check_x_slices-1>output_xslices:
            raise ValueError("Check this case, x_slices increase more than 2x")      
        
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
    if  not skip_y_unfolding and kernel_fold_req < input_folding_factor_y and is_y_unfolding_needed(ir,node_name,node,): #Check if there is expected mem overflow that can be solved by y unfolding(which reduce the number of channels)
        if not force_y_unfolding:
            if output_folding_factor_y>0: # We can unfold y if output folding factor is already 0
                ir.force_y_unfolding.append(node_name)
                node['frontend']['force_unfolding_y'] = True
                input_folding_factor_y-=1
                output_folding_factor_y-=1
                num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
                node['frontend']['output_tensor'].folding_factor_y = output_folding_factor_y
                node['frontend']['output_folding_factor_y'] = output_folding_factor_y
            else:
                print('Warning: Tried to y unfold for mem reduction but failed since y is not folded')

    num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
    
    if not add_folding_conv_before_node:
        node['frontend']['input_tensor'].y_tiles = num_y_tiles
    node['frontend']['y_tiles'] = num_y_tiles
    node['frontend']['input_folding_factor_x'] = input_folding_factor_x
    node['frontend']['input_folding_factor_y'] = input_folding_factor_y
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    node['frontend']['h_tiles'] = get_num_h_tiles(node)
    node['frontend']['xslices_per_htile'] = get_xslices_per_htile(node['frontend']['x_slices'], node['frontend']['h_tiles'], original_kernel_size)
    if insert_synthetic_fold_node_before or (add_folding_conv_before_node and not absorbed_fold_on_preceding):
        add_xfoldingConv_before_node(ir, node)
    node['frontend']['input_tensor'].xslices_per_htile = node['frontend']['xslices_per_htile']
    output_xslices_per_htile = get_xslices_per_htile(node['frontend']['output_tensor'].x_slices, node['frontend']['h_tiles'], original_kernel_size)
    if original_kernel_size != 1 and 'force_folding_x' in node['frontend']:
        # k3 folding-conv boundary halo: the sum of output xslices_per_htile must equal
        # output_x_slices + h_tiles - 1 (one extra halo slice at each h-tile boundary). If the
        # plain split falls short, top up individual h-tiles by 1, starting at index 0.
        output_xslices_per_htile = pad_k3_folding_output_xslices_per_htile(
            output_xslices_per_htile, node['frontend']['output_tensor'].x_slices, node['frontend']['h_tiles'])
    node['frontend']['output_tensor'].xslices_per_htile = output_xslices_per_htile

    folded_input_channels = original_input_channels * (2 ** input_folding_factor_x) * (2 ** input_folding_factor_y)
    if (len(node['frontend']['preceding_nodes_params']) == 0) and (input_folding_factor_x == 0) and (folded_input_channels < 16) and (original_input_width % 16 == 0):
        # Pack the input layer's x-slices into AMM lanes as a greedy largest-first combination of
        # {8,4,2,1} (e.g. 12 -> [8,4]) instead of a single uniform divisor, so non-power-of-2 slice
        # counts use fewer/wider lanes. num_packed_xslices becomes a per-lane list (one entry per lane).
        node['frontend']['input_tensor'].num_packed_xslices = greedy_packed_xslice_groups(node['frontend']['xslices_per_htile'][0])

def apply_activation_silu_lut(node):
    # Populate node['frontend']['lut'] from an 'activation_silu' attribute (a SiLU or Requant
    # pattern detected during tflite parsing). Shared by every op-type qparams calculator so any op
    # carrying the attribute gets its LUT, not just Conv. No-op when the attribute is absent/None.
    activation_silu = node['attributes'].get('activation_silu')
    if activation_silu is not None:
        if 'lut' in activation_silu:
            node['frontend']['lut'] = activation_silu['lut']
        else:
            node['frontend']['lut'] = LUTPopulateInt8(activation_silu['input_scale'][0],
                                                      activation_silu['input_zp'][0],
                                                      activation_silu['output_scale'][0],
                                                      activation_silu['output_zp'][0])
        return

    # requant_info: a back-to-back-LUT feeding a Concat that got fused onto this node's own output
    # (fuse_concat_requant_into_folding_producers) or attached to a Resize producer directly
    # (insert_concat_input_requant_nodes). Resize applies this itself in calc_resize_qparams, so
    # skip here to avoid clobbering it a second time.
    requant_info = node['attributes'].get('requant_info')
    if requant_info is None or node['op_type'] == OpType.RESIZE:
        return
    input_scale, input_zp = requant_info['input_scale'][0], requant_info['input_zp'][0]
    output_scale, output_zp = requant_info['output_scale'][0], requant_info['output_zp'][0]
    if 'lut' in requant_info:
        node['frontend']['lut'] = requant_info['lut']
    else:
        node['frontend']['lut'] = LUTPopulateInt8(input_scale, input_zp, output_scale, output_zp)
    node['frontend']['output_tensor'].scale = output_scale
    node['frontend']['output_tensor'].zero_point = output_zp
    node['frontend']['output_tensor_scale'] = output_scale
    node['frontend']['output_tensor_zp'] = output_zp

def calc_conv_qparams(ir,node_name,node):
    output_channels = node['frontend']['output_channels']
    input_channels = node['frontend']['input_channels']
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    kernel_size = node['frontend']['kernel_size']
    stride = node['frontend']['stride']
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y'] # This already includes input forced folding/unfolding
    is_folding_conv_x =  'force_folding_x' in node['frontend']
    
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

    # Handle 2x2 Conv converted to 3x3: pad weights from [OC, IC, 2, 2] to [OC, IC, 3, 3].
    # snp_compiler does not support 2x2 convs, so this applies to converted transpose convs too.
    if w_int8.shape[2] == 2 and w_int8.shape[3] == 2:
        # Pad weights: add zeros on top and left, keep original weights in bottom-right.
        # Original 2x2:        Padded 3x3:
        # [w11, w12]    →      [  0,   0,   0]
        # [w21, w22]           [  0, w11, w12]
        #                      [  0, w21, w22]
        # This bottom-right placement is what the is_conv2x2 detector below expects (top row and
        # left column zero) so the output-width bookkeeping fires correctly.
        w_int8_padded = np.pad(w_int8, ((0,0), (0,0), (1,0), (1,0)), mode='constant', constant_values=0)
        w_int8 = w_int8_padded
        # Update original_weights_tensor with padded weights
        original_weights_tensor.data = w_int8
        original_weights_tensor.shape = w_int8.shape
        node['frontend']['kernel_size'] = 3
        # Input padding to recover the original 2x2 receptive field (bottom-right weight
        # placement reads forward window in[i..i+1, j..j+1]): genuine 2x2 SAME conv uses symmetric
        # pad [1,1,1,1]; converted transpose conv has 180-degree-flipped phase kernels
        # (calc_nodes_folding_factor), needing pad [2,0,2,0] (top,bottom,left,right). Both verified
        # against tf.nn.conv2d / tf.nn.conv2d_transpose.
        is_converted_transpose_conv = node.get('attributes', {}).get('is_transpose_conv', False)
        if is_converted_transpose_conv:
            padding = [2, 0, 2, 0]
        else:
            padding = [1, 1, 1, 1]
        node['frontend']['padding'] = padding

    # Dilated conv: materialize the atrous kernel as an explicit larger dense kernel by
    # inserting zero rows/columns between the original taps. kernel_shape/pads were already
    # expanded to the dilated footprint at parse time (tflite_parser.py), so padding here is
    # already correct - only the weight data itself needs to grow to match.
    dilation_h = node['attributes'].get('dilation_h_factor', 1)
    dilation_w = node['attributes'].get('dilation_w_factor', 1)
    if dilation_h > 1 or dilation_w > 1:
        oc, ic, kh, kw = w_int8.shape
        new_kh = 1 + (kh - 1) * dilation_h
        new_kw = 1 + (kw - 1) * dilation_w
        w_int8_dilated = np.zeros((oc, ic, new_kh, new_kw), dtype=w_int8.dtype)
        w_int8_dilated[:, :, ::dilation_h, ::dilation_w] = w_int8
        w_int8 = w_int8_dilated
        original_weights_tensor.data = w_int8
        original_weights_tensor.shape = list(w_int8.shape)
        node['frontend']['kernel_size'] = new_kh
        if (new_kh > 9):
            raise ValueError (f"Unstructured sparsity doesn't support kernel {new_kh} x {new_kh}. It supports only upto 9x9 kernels")

    kernel_shape = w_int8.shape
    if (input_folding_factor_x>0 or input_folding_factor_y>0) and ('folded_weights_tensor' not in node['frontend']):
        asymmetric_padding = False
        padding = node['attributes']['pads']
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
        # The folded conv runs stride-1 in folded space and must preserve folded spatial size, so
        # it needs "SAME" padding = folded_kernel//2 regardless of original padding/kernel size or
        # fold amount. The old fold-amount-independent formula only worked at
        # input_folding_factor_x==1 and gave wildly wrong padding at ==2 (e.g. CONV2D_3: kernel=7
        # folded to 3 got [6,4,6,4] instead of ~1); copying original padding was only ever correct
        # for SAME convs, since VALID drops a boundary row in folded space (e.g. squeezenet
        # CONV2D_0: 56->54, cascading to an NX->MXP size mismatch). folded_padding is sim-only
        # (neuronix_ops nx_conv).
        folded_pad = folded_weights.shape[3] // 2
        node['frontend']['folded_padding'] = [folded_pad, folded_pad, folded_pad, folded_pad]

    if kernel_shape[3]!=kernel_shape[2]:
        raise ValueError ('Asymetric kernel is not supported (%s)' % str(kernel_shape))
    if kernel_shape[3] not in [1,3,5,7]:
        raise ValueError ('folded kernel shape is not supported (%s)' % str(kernel_shape))
    input_tensor_scale = node['frontend']['input_tensor_scale']
    input_tensor_zp = node['frontend']['input_tensor_zp']
    weights_per_channel_scale = node['frontend']['weights_per_channel_scale']
    output_tensor_scale = node['frontend']['output_tensor_scale']
    output_tensor_zp = node['frontend']['output_tensor_zp']

    output_multiplier = [0 for i in range(output_channels)]
    cInputH = [0 for i in range(output_channels)]
    cInputL = [0 for i in range(output_channels)]
    o_shift = [0 for i in range(output_channels)]

    # Hoist values that don't depend on och out of the per-channel loop.
    output_tensor_scale_item = output_tensor_scale.item()
    output_offset = output_tensor_zp.item()
    try:
        fused_act = ir.graph.nodes[node_name]['attributes']['fused_activation_function']
    except (KeyError, TypeError):
        fused_act = None
    base_act_min = int8min
    base_act_max = int8max
    if fused_act == 'RELU':
        base_act_min = output_offset
    elif fused_act == 'RELU6':
        base_act_min = output_offset
        base_act_max = min(round((6 / output_tensor_scale_item) + output_offset), int8max)
    has_per_channel_scale = (weights_per_channel_scale.size > 1)
    fallback_scale = weights_per_channel_scale[0].item() if not has_per_channel_scale else None

    for och in range(output_channels):
        w_sum = np.sum(w_int8[och])
        zeroed_oc = np.all(w_int8[och,:]==0)
        if zeroed_oc:
            requant_scale = input_tensor_scale*0.0039/output_tensor_scale_item #TODO: We can set requant_scale to 0.02 in case all weights are zero (see neuronix_models capture code)
        elif has_per_channel_scale:
            requant_scale = input_tensor_scale*weights_per_channel_scale[och].item()/output_tensor_scale_item
        else:
            requant_scale = input_tensor_scale*fallback_scale/output_tensor_scale_item

        requant_bias  = original_biases_tensor.data[och].item()
        requant_bias -= w_sum*input_tensor_zp
        requant_bias = requant_bias.astype(np.int32)
        if zeroed_oc:
            # All weights are zero, so the true bias was quantized against TFLite's own degenerate
            # near-zero bias_scale (~1e-9), not the substituted 0.0039-based requant_scale - the
            # raw int32 bias would be a ~1e6x scale mismatch in quantize2MathBlock. Treat as 0
            # instead (matches the dead channel's actual near-zero real output).
            requant_bias = np.int32(0)

        acc = None
        bias_data = int(requant_bias)
        scale = requant_scale

        output_multiplier[och], cInputH[och], cInputL[och], o_shift[och] = \
            quantize2MathBlock(acc,bias_data,scale,output_offset,base_act_min,base_act_max)

    node['frontend']['output_multiplier'] = output_multiplier
    node['frontend']['cInputH'] = cInputH
    node['frontend']['cInputL'] = cInputL
    node['frontend']['o_shift'] = o_shift

    if (input_folding_factor_x>0 or input_folding_factor_y>0):
        # Take params list of length [output channels] and replicate it based on folding
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

    # Identifying 2x2 Conv converted to 3x3 (same pattern as avgPool but for regular conv)
    is_conv2x2 = False
    if (kernel_shape[3] == 3) and not is_avgPool:
        # Check if only the bottom-right 2x2 sub-kernel (indices [1:3, 1:3]) is non-zero, the
        # pattern left by a converted 2x2.
        is_conv2x2 = True
        for i in range(kernel_shape[0]):
            for j in range(kernel_shape[1]):
                if j == i:
                    # Check if top-left corner (indices [0, 0]) and edges are zero
                    if not (np.all(w_int8[i,j,0,:] == 0) and  # Top row all zeros
                            np.all(w_int8[i,j,:,0] == 0)):    # Left column all zeros
                        is_conv2x2 = False
                        break
            if not is_conv2x2:
                break
    node['frontend']['is_conv2x2'] = is_conv2x2
    if is_conv2x2:
        node['frontend']['output_tensor'].is_conv2x2_output = True
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
    # Gemm/fully-connected operates on flattened activations and has no spatial x/y folding, but the
    # shared post-folding code (and report writer) read the _x/_y folding factors on every node.
    # Set them to 0 so Gemm nodes don't raise KeyError('output_folding_factor_x').
    node['frontend']['input_folding_factor_x'] = 0
    node['frontend']['input_folding_factor_y'] = 0
    node['frontend']['output_folding_factor_x'] = 0
    node['frontend']['output_folding_factor_y'] = 0

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
    output_tensor_scale = node['frontend']['output_tensor_scale']
    output_tensor_zp = node['frontend']['output_tensor_zp']

    for och in range(output_channels):
        w_sum = 0
        for ich in range(input_channels):
                    w_sum += w_int8[ich][och] # Weights tensor in Gemm is (#ic,#oc) and not (#oc,#ic,kx,ky)
        requant_scale = input_tensor_scale*weights_per_channel_scale[och].item()/output_tensor_scale.item() #TODO: We can set requant_scale to 0.02 in case all weights are zero (see neuronix_models capture code)
        requant_bias  = bias[och].item()/output_tensor_scale.item() + output_tensor_zp.item()
        requant_bias -= w_sum*input_tensor_zp*weights_per_channel_scale[och].item()*input_tensor_scale/output_tensor_scale.item()
    

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
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
        x_slices = node['frontend']['input_tensor'].x_slices
    else:
        input_folding_factor_x,input_folding_factor_y, x_slices = get_new_input_folding_factors(original_input_width,original_input_height,stride,original_kernel_size,original_input_channels,original_output_channels,ir.amm_height)
        # Same entry y-fold rule as calc_conv_folding_factor: consecutive stride-2 nodes or the kernel-required
        # fold, instead of the max that fits the AMM channel cap (which over-folds a lone stride-2 pool).
        input_folding_factor_y = max(get_num_consecutive_stride2_conv(ir, node),
                                     get_required_folding_factor_for_kernel_size(original_kernel_size))
        if 'MaxPool16x16' in ir.model_name:
            if node_name == 'MaxPool1':
                input_folding_factor_x = 0
                input_folding_factor_y = 0
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
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
        input_folding_factor_y += 1
        node['frontend']['force_folding_y'] = True

    if node_name in ir.force_y_unfolding:
        output_folding_factor_y -= 1
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
    num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
    node['frontend']['input_tensor'].y_tiles = num_y_tiles
    node['frontend']['y_tiles'] = num_y_tiles
    node['frontend']['input_folding_factor_x'] = input_folding_factor_x
    node['frontend']['input_folding_factor_y'] = input_folding_factor_y
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    node['frontend']['y_tiles'] = num_y_tiles
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    node['frontend']['output_tensor'].x_slices = output_xslices
    node['frontend']['h_tiles'] = get_num_h_tiles(node)
    node['frontend']['xslices_per_htile'] = get_xslices_per_htile(node['frontend']['x_slices'], node['frontend']['h_tiles'], original_kernel_size)
    node['frontend']['input_tensor'].xslices_per_htile = node['frontend']['xslices_per_htile']
    node['frontend']['output_tensor'].xslices_per_htile = get_xslices_per_htile(node['frontend']['output_tensor'].x_slices, node['frontend']['h_tiles'], original_kernel_size)

    if add_folding_conv_before_node:
        # Absorb the extra x-fold step onto the preceding conv when it exists solely to feed this MaxPool
        # (same eligibility as calc_conv_folding_factor's absorption); otherwise insert the fold conv.
        absorbed = False
        preceding_nodes_params = node['frontend'].get('preceding_nodes_params', [])
        if len(preceding_nodes_params) == 1:
            preceding_node = ir.graph.nodes[preceding_nodes_params[0][0]]
            preceding_frontend = preceding_node.get('frontend', {})
            if (preceding_node.get('op_type') in (OpType.CONV, OpType.IDENTITY)
                    and len(preceding_frontend.get('following_nodes_params', [])) == 1
                    and preceding_frontend.get('output_folding_factor_x', 0) == 0
                    and 'force_folding_x' not in preceding_frontend
                    and 'force_unfolding_x' not in preceding_frontend):
                absorb_x_fold_step_on_preceding_node(preceding_node)
                absorbed = True
        if not absorbed:
            add_xfoldingConv_before_node(ir, node)
    
def calc_avgpool_qparams(ir,node_name,node):
    calc_fake_conv_weights(ir,node_name,node)
    num_outputs = node['frontend']['output_channels']
    
    # We set the requant params to have 1 to 1
    node['frontend']['output_multiplier'] = node['frontend']['output_multiplier']//8
    node['frontend']['cInputH'] = 0
    node['frontend']['cInputL'] = 0

    node['frontend']['weights_tensor'].data = 0*node['frontend']['weights_tensor'].data
    for c in range(num_outputs):
        node['frontend']['weights_tensor'].data[c, c, 1, 1]  = 8//4
        node['frontend']['weights_tensor'].data[c, c, 1, 2]  = 8//4
        node['frontend']['weights_tensor'].data[c, c, 2, 1]  = 8//4
        node['frontend']['weights_tensor'].data[c, c, 2, 2]  = 8//4
        
    calc_folded_weights_and_sparsity(ir,node_name,node)    



def calc_maxpool_qparams(ir,node_name,node):
    calc_fake_conv_weights(ir,node_name,node)
    # Handle 2x2 Conv converted to 3x3: pad weights from [OC, IC, 2, 2] to [OC, IC, 3, 3].
    # snp_compiler does not support 2x2 convs, so this applies to converted transpose convs too.
    w_int8 = node['frontend']['weights_tensor'].data
    if node['frontend']['kernel_size'] == 2:
        # Pad weights to bottom-right 2x2 (zeros top/left) - what the is_conv2x2 detector below
        # expects so output-width bookkeeping fires correctly. For a real Conv this
        # 2x2->3x3+pad[1,1,1,1] conversion is exact since the zero weights mask the padded
        # window. MaxPool has no weights to mask with though - nx_maxpool runs a literal kernel=3
        # torch.nn.MaxPool2d, so the padded window would wrongly pull in a neighboring ADJACENT
        # 2x2 window's cells. So save the ORIGINAL native kernel_size/padding here for the numeric
        # simulator to keep using; the 3x3 conversion below is only valid for CBC/WLOC weight-shape
        # bookkeeping, not the actual max reduction.
        node['frontend']['maxpool_native_kernel_size'] = node['frontend']['kernel_size']
        node['frontend']['maxpool_native_padding'] = node['frontend']['padding']
        w_int8_padded = np.pad(w_int8, ((0,0), (0,0), (1,0), (1,0)), mode='constant', constant_values=0)
        w_int8 = w_int8_padded
        # Update original_weights_tensor with padded weights
        node['frontend']['weights_tensor'].data = w_int8
        node['frontend']['weights_tensor'].shape = w_int8.shape
        node['frontend']['kernel_size'] = 3
        node['frontend']['padding'] = [0, 1, 0, 1]
        weights_tensor_name = node_name + '_fake_weights_tensor'
        ir.tensors[weights_tensor_name] =  node['frontend']['weights_tensor']
    elif node['frontend'].get('asym_padding', False) == True:
        #  replace left column and top row with zeros
        # Save original kernel_size and padding for numeric simulator
        node['frontend']['maxpool_native_kernel_size'] = node['frontend']['kernel_size']-1
        node['frontend']['maxpool_native_padding'] = node['frontend']['padding']
        # Zero out left column (index 0) and top row (index 0)
        w_int8_padded = w_int8.copy()
        w_int8_padded[:, :, 0, :] = 0  # Top row
        w_int8_padded[:, :, :, 0] = 0  # Left column
        w_int8 = w_int8_padded
        # Update weights tensor with padded weights
        node['frontend']['weights_tensor'].data = w_int8
        node['frontend']['weights_tensor'].shape = w_int8.shape
        weights_tensor_name = node_name + '_fake_weights_tensor'
        ir.tensors[weights_tensor_name] = node['frontend']['weights_tensor']
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
    set_tflite_rq_params(node, acc=None, bias_data=0, scale=1.0, output_offset=0)

    # Create the weights tensor
    output_channels = node['frontend']['output_tensor'].get_original_shape()[1] # This will be folded output channels but without output folding
    input_channels = node['frontend']['input_tensor'].get_original_shape()[1]

    weights_tensor_shape = [output_channels,input_channels,kernel_size,kernel_size]
    weights_tensor = np.zeros(weights_tensor_shape,dtype=np.int64)
    # Diagonal fill: weights_tensor[oc, oc, :, :] = 8 (in addition to requant_scale_uint10 and requant_scale_shift this will cause copy of input to output)
    diag = np.arange(output_channels)
    weights_tensor[diag, diag, :, :] = 8

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
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
        num_x_slices = node['frontend']['input_tensor'].x_slices
    else:
        input_folding_factor_x=0
        input_folding_factor_y=0
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
        num_x_slices = int(original_input_width/(MAX_GRID_WIDTH*(2**input_folding_factor_x)))
        node['frontend']['input_tensor'].folding_factor_x = input_folding_factor_x # We update the tensor's folding factor only if it was generated here
        node['frontend']['input_tensor'].folding_factor_y = input_folding_factor_y
        node['frontend']['input_tensor'].x_slices = num_x_slices

    node['frontend']['input_folding_factor_x'] = input_folding_factor_x # Node's input folding factor will contain effects of forced folding at input at y axis
    node['frontend']['input_folding_factor_y'] = input_folding_factor_y # This already includes input forced folding/unfolding
    node['frontend']['x_slices'] = num_x_slices
    node['frontend']['y_tiles'] = num_y_tiles
    node['frontend']['is_hw_x_resize'] = 1
    output_folding_factor_x = input_folding_factor_x
    output_folding_factor_y = input_folding_factor_y
    node['frontend']['output_tensor'].folding_factor_x = output_folding_factor_x
    node['frontend']['output_tensor'].folding_factor_y = output_folding_factor_y
    if ('_y' in node_name):
        # The _y node only resizes on Y; its input inherited the _x node's 2*num_x_slices (needed
        # by _x's RESIZE1_2 scaling: two output slices per input slice), but when the x-resized
        # width isn't a multiple of MAX_GRID_WIDTH the trailing slice is empty, so only
        # ceil(width/MGW) slices are useful (e.g. w20 -> x-resized 40 -> 3 slices, not 4). This _y
        # output feeds downstream ops from DDR, so set the geometric useful count here as the
        # source of truth, or every consumer inherits _x's inflated count.
        output_width = node['frontend']['output_tensor'].get_original_shape()[3]
        node['frontend']['output_tensor'].x_slices = math.ceil(output_width / (MAX_GRID_WIDTH * (2 ** output_folding_factor_x)))
        node['frontend']['output_tensor'].y_tiles = get_num_y_tiles(original_input_height*2,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
    else:
        node['frontend']['output_tensor'].x_slices = 2*num_x_slices
        node['frontend']['output_tensor'].y_tiles = num_y_tiles
    node['frontend']['output_folding_factor_x'] = output_folding_factor_x
    node['frontend']['output_folding_factor_y'] = output_folding_factor_y
    node['frontend']['h_tiles'] = get_num_h_tiles(node)
    node['frontend']['xslices_per_htile'] = get_xslices_per_htile(node['frontend']['x_slices'], node['frontend']['h_tiles'], 1)
    node['frontend']['input_tensor'].xslices_per_htile = node['frontend']['xslices_per_htile']
    node['frontend']['output_tensor'].xslices_per_htile = get_xslices_per_htile(node['frontend']['output_tensor'].x_slices, node['frontend']['h_tiles'], 1)

    if ('requant_info' in node['attributes']) and (node['attributes']['requant_info'] is not None):
        input_scale = node['attributes']['requant_info']['input_scale'][0]
        input_zp = node['attributes']['requant_info']['input_zp'][0]
        output_scale = node['attributes']['requant_info']['output_scale'][0]
        output_zp = node['attributes']['requant_info']['output_zp'][0]
        if ('_x' in node_name):
            if ('lut' in node['attributes']['requant_info']):
                node['frontend']['lut'] = node['attributes']['requant_info']['lut']
            else:
                node['frontend']['lut'] = LUTPopulateInt8(input_scale, input_zp, output_scale, output_zp)
            node['frontend']['input_tensor'].scale = input_scale
            node['frontend']['input_tensor'].zero_point = input_zp
        node['frontend']['output_tensor'].scale = output_scale
        node['frontend']['output_tensor'].zero_point = output_zp
    
def calc_resize_qparams(ir,node_name,node):
    # First we make sure that both input tensors have same shape
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    kernel_size = 1
    stride = 1
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    is_hw_x_resize = 'is_hw_x_resize' in node['frontend']
    
    set_tflite_rq_params(node, acc=None, bias_data=int(0), scale=1/32.0, output_offset=0)

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
                weights_tensor[ic+i*input_channels,ic,0,0] = 32 # # In "is_hw_x_resize" resize the resizing in x axis is done in rq and not by duplicating channels
        else:
            for i in range(4):
                #oc=ic*2+(i&1)+input_channels*2*(i//2) # old concept no slices 
                oc = ic+i*input_channels               #new resize concept with slices 
                weights_tensor[oc,ic,0,0] = 32 # This in addition to requant_scale_uint10 and requant_scale_shift will cause copy of input to output

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
            target_oc = np.where(folded_weights[:,ic,0,0]==32)[0]
            for oc in target_oc:
                if oc in ocs_found:
                    raise ValueError('Found same oc twice')
                ocs_found.append(oc)
        if len(ocs_found)!= folded_weights.shape[0]:
            raise ValueError ('Not all target ocs found')

def copy_folding_factor_from_prev_node(node, amm_height, ir=None):
    # A Sync node is treated as a pure DMA hop, exactly like Concat: no grid/kernel compute happens
    # on the NX side, so it never carries a folding factor (input or output is always 0) regardless
    # of whatever the wrapped MXP-side op (e.g. MEAN) actually does to the shape.
    assert len(node['frontend']['preceding_nodes_params'])>0

    node['frontend']['input_tensor'].folding_factor_x = 0
    node['frontend']['input_tensor'].folding_factor_y = 0
    node['frontend']['input_folding_factor_x'] = 0
    node['frontend']['input_folding_factor_y'] = 0
    node['frontend']['output_tensor'].folding_factor_x = 0
    node['frontend']['output_tensor'].folding_factor_y = 0
    node['frontend']['output_folding_factor_x'] = 0
    node['frontend']['output_folding_factor_y'] = 0

    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    original_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape()
    num_y_tiles = get_num_y_tiles(original_input_tensor_shape[2], 0, original_kernel_size=1, amm_height=amm_height)
    node['frontend']['input_tensor'].y_tiles = num_y_tiles
    node['frontend']['y_tiles'] = num_y_tiles
    node['frontend']['output_tensor'].y_tiles = get_num_y_tiles(
        original_output_tensor_shape[2], 0, original_kernel_size=1, amm_height=amm_height)

    # set_tiling_blobs requires x_slices/h_tiles/xslices_per_htile on every node, but Sync never
    # populates them -> KeyError during blob splitting. Copy from the preceding node; fall back to
    # width-based x_slices if unavailable.
    prev_frontend = None
    if ir is not None:
        # Sync may wrap a multi-input MXP op; frontend['input_tensor'] (chosen by
        # frontend_hardware_agnostic.py's highest-rank-shape heuristic) isn't necessarily
        # preceding_nodes_params[0], so find the actual producer to copy matching geometry from.
        input_tensor_name = node['frontend']['input_tensor'].name
        prev_name = None
        for preceding_node_name, _ in node['frontend']['preceding_nodes_params']:
            preceding_node = ir.graph.nodes.get(preceding_node_name)
            if preceding_node is not None and input_tensor_name in preceding_node.get('outputs', []):
                prev_name = preceding_node_name
                break
        if prev_name is None:
            prev_name = node['frontend']['preceding_nodes_params'][0][0]
        if prev_name in ir.graph.nodes:
            prev_frontend = ir.graph.nodes[prev_name].get('frontend')
    if prev_frontend is not None and 'x_slices' in prev_frontend:
        input_x_slices = prev_frontend['x_slices']
    else:
        input_x_slices = getattr(node['frontend']['input_tensor'], 'x_slices', None)
        if input_x_slices is None:
            input_x_slices = int(original_input_tensor_shape[3] / MAX_GRID_WIDTH)
    node['frontend']['x_slices'] = input_x_slices
    node['frontend']['input_tensor'].x_slices = input_x_slices
    # The output tensor's own width (post spatial-reduction, if any) determines its x_slices - it is
    # not necessarily the same grid geometry as the input.
    output_x_slices = max(1, int(math.ceil(original_output_tensor_shape[3] / MAX_GRID_WIDTH)))
    node['frontend']['output_tensor'].x_slices = output_x_slices
    if prev_frontend is not None and 'h_tiles' in prev_frontend:
        node['frontend']['h_tiles'] = prev_frontend['h_tiles']
        node['frontend']['xslices_per_htile'] = list(prev_frontend['xslices_per_htile'])
    else:
        node['frontend']['h_tiles'] = 1
        node['frontend']['xslices_per_htile'] = get_xslices_per_htile(input_x_slices, 1, 1)

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
        input_folding_factor_y = math.ceil(math.log(original_input_width / ir.amm_height,2))
        if input_folding_factor_y<0: # If input resolution is less than MAX_GRID_WIDTH, folding is not needed
            input_folding_factor_y=0
        original_kernel_size = 1
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)

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
    node['frontend']['h_tiles'] = get_num_h_tiles(node)
    node['frontend']['xslices_per_htile'] = get_xslices_per_htile(node['frontend']['x_slices'], node['frontend']['h_tiles'], 1)
    node['frontend']['input_tensor'].xslices_per_htile = node['frontend']['xslices_per_htile']
    node['frontend']['output_tensor'].xslices_per_htile = get_xslices_per_htile(node['frontend']['output_tensor'].x_slices, node['frontend']['h_tiles'], 1)

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
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
        x_slices = node['frontend']['input_tensor'].x_slices
    else:
        input_folding_factor_x,input_folding_factor_y, x_slices = get_new_input_folding_factors(original_input_width,original_input_height,stride,original_kernel_size,original_input_channels,original_output_channels,ir.amm_height)
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
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
    node['frontend']['h_tiles'] = get_num_h_tiles(node)
    node['frontend']['xslices_per_htile'] = get_xslices_per_htile(node['frontend']['x_slices'], node['frontend']['h_tiles'], original_kernel_size)
    node['frontend']['input_tensor'].xslices_per_htile = node['frontend']['xslices_per_htile']
    node['frontend']['output_tensor'].xslices_per_htile = get_xslices_per_htile(node['frontend']['output_tensor'].x_slices, node['frontend']['h_tiles'], original_kernel_size)

                
def calc_general_folding_factor(ir,node_name,node):
    # TODO: need to add support for forced input folding/unfolding on y. note that node['frontend]['input_folding_factor_y'] should include y folding/unfolding effect see conv/resize for implementation

    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    original_input_height = original_input_tensor_shape[2]
    original_kernel_size = node['frontend']['kernel_size']
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
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
        x_slices = node['frontend']['input_tensor'].x_slices
    else:
        input_folding_factor_x,input_folding_factor_y, x_slices = get_new_input_folding_factors(original_input_width,original_input_height,stride,original_kernel_size,original_input_channels,original_output_channels,ir.amm_height)
        num_y_tiles = get_num_y_tiles(original_input_height,input_folding_factor_y,original_kernel_size=1,amm_height=ir.amm_height)
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
    node['frontend']['h_tiles'] = 1
    node['frontend']['xslices_per_htile'] = get_xslices_per_htile(node['frontend']['x_slices'], node['frontend']['h_tiles'], original_kernel_size)
    node['frontend']['input_tensor'].xslices_per_htile = node['frontend']['xslices_per_htile']
    node['frontend']['output_tensor'].xslices_per_htile = get_xslices_per_htile(node['frontend']['output_tensor'].x_slices, node['frontend']['h_tiles'], original_kernel_size)

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

    input_folding_factor_x = node['frontend']['input_tensors'][0].folding_factor_x
    input_folding_factor_y = 0
    output_folding_factor_x = input_folding_factor_x
    output_folding_factor_y = input_folding_factor_y

    if node_name in ir.force_y_folding:
        input_folding_factor_y += 1
        node['frontend']['force_folding_y'] = True

    if node_name in ir.force_y_unfolding:
        output_folding_factor_y -= 1
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
    # If concat unfolds inputs on Y (output_folding_factor_y < inputs' folding), output is taller
    # than the folded inputs, so their y_tiles no longer applies; recompute from unfolded output
    # height (kernel_size=1). Otherwise inheriting folded y_tiles=1 tripped AMM_HEIGHT overflow.
    if max_inputs_folding_factor_y > output_folding_factor_y:
        original_output_height = node['frontend']['output_tensor'].get_original_shape()[2]
        num_y_tiles = get_num_y_tiles(original_output_height, output_folding_factor_y,
                                      original_kernel_size=1, amm_height=ir.amm_height)
    else:
        num_y_tiles = inputs_y_tiles[0]
        for current_input_y_tiles in inputs_y_tiles:
            if current_input_y_tiles!= num_y_tiles:
                pass
                #raise ValueError ('All Concat inputs must have same number of tiles. Please check...')
    node['frontend']['y_tiles'] = num_y_tiles
    node['frontend']['output_tensor'].y_tiles = num_y_tiles
    _block_maps = concat_amm_block_map([it.shape_real_x16[1] for it in node['frontend']['input_tensors']])
    if _block_maps is not None:
        _out_channels = node['frontend']['output_tensor'].shape_real_x16[1]
        assert len(_block_maps) == _out_channels, (
            f"concat block map covers {len(_block_maps)} channels but the concat has {_out_channels} output channels")
        node['frontend']['concat_block_maps'] = _block_maps
        # Every consumer reads the padded concat layout, so each one gets the maps; a consumer that inserts a
        # folding conv also gets them on that conv (add_xfoldingConv_before_node).
        for consumer in node['frontend']['output_tensor'].consumers:
            ir.graph.nodes()[consumer]['frontend']['aligned_input_ch'] = _block_maps
    node['frontend']['x_slices'] = inputs_x_slices[0]
    node['frontend']['h_tiles'] = 1
    node['frontend']['xslices_per_htile'] = get_xslices_per_htile(node['frontend']['x_slices'], node['frontend']['h_tiles'], 1)
    node['frontend']['output_tensor'].xslices_per_htile = get_xslices_per_htile(node['frontend']['output_tensor'].x_slices, node['frontend']['h_tiles'], 1)
    for input_tensor in node['frontend']['input_tensors']:
        input_tensor.xslices_per_htile = get_xslices_per_htile(input_tensor.x_slices, node['frontend']['h_tiles'], 1)

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

    # Calculate y_tiles if not already set on input tensors
    if hasattr(node['frontend']['input_tensors'][0], 'y_tiles'):
        num_y_tiles = node['frontend']['input_tensors'][0].y_tiles
    else:
        # Calculate y_tiles for ADD operation (kernel size = 1 for element-wise ops)
        original_input_height = original_input0_tensor_shape[2]  # Index 2 for height
        num_y_tiles = get_num_y_tiles(original_input_height, input_folding_factor_y, original_kernel_size=1, amm_height=ir.amm_height)

    # Tensor.__init__ always defaults x_slices=0, so the old hasattr() guard was always true and
    # silently used that default for a workload-input Add, giving x_slices=0/xslices_per_htile=[0]
    # and no engine/DMW commands (sequencer loops degenerate to range(0)). Decide by lineage instead
    # (as in calc_identity_folding_factor): a produced input inherits its producer's x_slices; a
    # workload input has no producer, so derive x_slices from resolution (kernel=1/stride=1).
    if len(node['frontend']['preceding_nodes_params']) > 0:
        num_x_slices = node['frontend']['input_tensors'][0].x_slices
    else:
        num_x_slices = math.ceil(original_input_width / (MAX_GRID_WIDTH * (2 ** input_folding_factor_x)))
        node['frontend']['input_tensors'][0].x_slices = num_x_slices
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

    node['frontend']['output_tensor'].folding_factor_x = output_folding_factor_x
    node['frontend']['output_tensor'].folding_factor_y = output_folding_factor_y
    node['frontend']['output_folding_factor_x'] = output_folding_factor_x
    node['frontend']['output_folding_factor_y'] = output_folding_factor_y
    node['frontend']['h_tiles'] = 1
    node['frontend']['xslices_per_htile'] = get_xslices_per_htile(node['frontend']['x_slices'], node['frontend']['h_tiles'], 1)
    node['frontend']['output_tensor'].xslices_per_htile = get_xslices_per_htile(node['frontend']['output_tensor'].x_slices, node['frontend']['h_tiles'], 1)
    for input_tensor in node['frontend']['input_tensors']:
        input_tensor.xslices_per_htile = get_xslices_per_htile(input_tensor.x_slices, node['frontend']['h_tiles'], 1)

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

    # Convert scale/zp to scalars if they are single-element arrays
    # This can happen with constants that have per-channel quantization with 1 channel
    scale_0 = input_tensors_scale[0].item() if isinstance(input_tensors_scale[0], np.ndarray) and input_tensors_scale[0].size == 1 else input_tensors_scale[0]
    scale_1 = input_tensors_scale[1].item() if isinstance(input_tensors_scale[1], np.ndarray) and input_tensors_scale[1].size == 1 else input_tensors_scale[1]
    zp_0 = input_tensors_zp[0].item() if isinstance(input_tensors_zp[0], np.ndarray) and input_tensors_zp[0].size == 1 else input_tensors_zp[0]
    zp_1 = input_tensors_zp[1].item() if isinstance(input_tensors_zp[1], np.ndarray) and input_tensors_zp[1].size == 1 else input_tensors_zp[1]

    is_folding_conv_x = 'force_folding_x' in node['frontend']
    is_folding_conv_y = 'force_folding_y' in node['frontend']
    if (is_folding_conv_x or is_folding_conv_y):
        raise ValueError ('Currently folding add conv is not supported') # If need to support look on next line to update shape calc.
    folded_output_channels = node['frontend']['output_tensor'].get_folded_shape()[1] # This will be folded output channels but without output folding
    folded_input_channels = node['frontend']['input_tensors'][0].get_folded_shape()[1]

    #quantized weights
    w_input      = (128*scale_0)/(scale_0+scale_1)
    w_shortcut   = (128*scale_1)/(scale_0+scale_1)
    q_w_input    = math.trunc(w_input    + 0.5)
    q_w_shortcut = math.trunc(w_shortcut + 0.5)

    #requant
    requant_scale = (scale_0+scale_1) / (128*output_tensor_scale)
    # output_tensor_zp is a per-tensor zero point (requant_bias must stay scalar for the math.trunc below).
    # It may arrive as a numpy array (size 1), a numpy scalar, or a plain Python int (per-tensor quant, e.g.
    # when the Add recombines the pieces of a combined output+input conv split); normalize to a float scalar.
    requant_bias = float(output_tensor_zp.reshape(-1)[0]) if isinstance(output_tensor_zp, np.ndarray) else float(output_tensor_zp)
    requant_bias -= zp_0*scale_0/output_tensor_scale
    requant_bias -= zp_1*scale_1/output_tensor_scale
    
    requant_bias_int12 = math.trunc(BIAS_MULITIPLIER*requant_bias + BIAS_ROUNDING_ADD)
    # Check if bias exceeds MAX_BIAS_BITS and clip if needed.
    # Skip check if bias is zero: log(0) is undefined, and zero always fits in any bit width.
    if DEBUG_CLIP_BIAS_TO_MAX_BIAS_BITS and requant_bias_int12 != 0 and math.log(abs(requant_bias_int12),2)>=MAX_BIAS_BITS:
        max_bias_int12 = 2 ** MAX_BIAS_BITS-1
        min_bias_int12 = -1*max_bias_int12
        requant_bias_int12 = min(max_bias_int12,max(min_bias_int12,requant_bias_int12)) # Clip bias to MAX_BIAS_BITS
        print('At layer %s, bias exceeded INT%d ' % (node_name,MAX_BIAS_BITS+1))

    node['frontend']['requant_bias_int12'] = requant_bias_int12
    node['frontend']['q_w_input'] = q_w_input
    node['frontend']['q_w_shortcut'] = q_w_shortcut
    # Create the weights tensor (original, unaligned)
    original_weights_tensor_shape = [original_output_channels,original_input_channels*2,1,1]
    original_weights_tensor = np.zeros(original_weights_tensor_shape,dtype=np.int64)
    oc_range = np.arange(original_output_channels)
    original_weights_tensor[oc_range, oc_range, 0, 0] = q_w_input
    original_weights_tensor[oc_range, oc_range + original_input_channels, 0, 0] = q_w_shortcut
    w_int8_np = original_weights_tensor.astype(np.int8)
    original_weights_tensor_name = node_name + '_original_weights_tensor'
    weights_tensor = Tensor(original_weights_tensor_name,w_int8_np,is_constant=True,shape = original_weights_tensor_shape)
    ir.tensors[original_weights_tensor_name] = weights_tensor
    node['frontend']['weights_tensor'] = weights_tensor

    # Create URAM-aligned weights tensor for CBC generation
    # If input_channels is not divisible by URAM_BLOCK_SIZE, align to next boundary
    aligned_input_channels = ((original_input_channels + URAM_BLOCK_SIZE - 1) // URAM_BLOCK_SIZE) * URAM_BLOCK_SIZE
    aligned_weights_tensor_shape = [original_output_channels, aligned_input_channels*2, 1, 1]
    aligned_weights_tensor = np.zeros(aligned_weights_tensor_shape, dtype=np.int64)
    # Place q_w_input in [0:original_input_channels]
    aligned_weights_tensor[oc_range, oc_range, 0, 0] = q_w_input
    # Padding [original_input_channels:aligned_input_channels] remains zeros
    # Place q_w_shortcut in [aligned_input_channels:aligned_input_channels+original_input_channels]
    aligned_weights_tensor[oc_range, oc_range + aligned_input_channels, 0, 0] = q_w_shortcut
    # Padding [aligned_input_channels+original_input_channels:aligned_input_channels*2] remains zeros
    w_int8_np_aligned = aligned_weights_tensor.astype(np.int8)
    aligned_weights_tensor_name = node_name + '_aligned_uram_weights_tensor'
    weights_tensor_aligned = Tensor(aligned_weights_tensor_name, w_int8_np_aligned, is_constant=True, shape=aligned_weights_tensor_shape)
    ir.tensors[aligned_weights_tensor_name] = weights_tensor_aligned
    node['frontend']['weights_tensor_aligned_uram'] = weights_tensor_aligned
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
        # Create folded weights tensor (original, unaligned)
        folded_weights_tensor_shape = [folded_output_channels,folded_input_channels*2,1,1]
        folded_weights_tensor = np.zeros(folded_weights_tensor_shape,dtype=np.int64)
        foc_range = np.arange(folded_output_channels)
        folded_weights_tensor[foc_range, foc_range, 0, 0] = q_w_input
        folded_weights_tensor[foc_range, foc_range + folded_input_channels, 0, 0] = q_w_shortcut
        folded_w_int8_np = folded_weights_tensor.astype(np.int8)
        folded_weights_tensor_name = node_name + '_folded_weights_tensor'
        folded_tensor = Tensor(folded_weights_tensor_name,folded_w_int8_np,is_constant=True,shape = folded_weights_tensor_shape)
        ir.tensors[folded_weights_tensor_name] = folded_tensor
        node['frontend']['folded_kernel_size'] = 1
        node['frontend']['folded_weights_tensor'] = folded_tensor

        # Create URAM-aligned folded weights tensor for CBC generation
        aligned_folded_input_channels = ((folded_input_channels + URAM_BLOCK_SIZE - 1) // URAM_BLOCK_SIZE) * URAM_BLOCK_SIZE
        aligned_folded_weights_tensor_shape = [folded_output_channels, aligned_folded_input_channels*2, 1, 1]
        aligned_folded_weights_tensor = np.zeros(aligned_folded_weights_tensor_shape, dtype=np.int64)
        aligned_folded_weights_tensor[foc_range, foc_range, 0, 0] = q_w_input
        aligned_folded_weights_tensor[foc_range, foc_range + aligned_folded_input_channels, 0, 0] = q_w_shortcut
        aligned_folded_w_int8_np = aligned_folded_weights_tensor.astype(np.int8)
        aligned_folded_weights_tensor_name = node_name + '_folded_aligned_uram_weights_tensor'
        folded_tensor_aligned = Tensor(aligned_folded_weights_tensor_name, aligned_folded_w_int8_np, is_constant=True, shape=aligned_folded_weights_tensor_shape)
        ir.tensors[aligned_folded_weights_tensor_name] = folded_tensor_aligned
        node['frontend']['folded_weights_tensor_aligned_uram'] = folded_tensor_aligned

    node['frontend']['stride'] = 1
    input_tensors_shape = node['frontend']['input_tensors'][0].get_folded_shape() # TODO Dans, need to revisit for folded add
    node['frontend']['sparse_macs'] = 2*folded_output_channels*input_tensors_shape[2]*input_tensors_shape[3]

    bias_data = 0 - (q_w_input * input_tensors_zp[0]) - (q_w_shortcut * input_tensors_zp[1])
    # Convert output_tensor_zp to scalar if it's a single-element array
    output_zp_scalar = output_tensor_zp.item() if isinstance(output_tensor_zp, np.ndarray) and output_tensor_zp.size == 1 else (output_tensor_zp[0] if isinstance(output_tensor_zp, np.ndarray) else output_tensor_zp)
    set_tflite_rq_params(node, acc=None, bias_data=bias_data, scale=requant_scale, output_offset=output_zp_scalar)

def calc_identity_qparams(ir,node_name,node):

    # First we make sure that both input tensors have same shape
    original_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
    original_input_channels = original_input_tensor_shape[1]
    original_output_shape = node['frontend']['output_tensor'].get_original_shape()
    original_output_channels = original_output_shape[1]
    if original_input_channels!=original_output_channels:
        # For Strided Slice, implement as identity but allow input and output channels to differ
        if 'STRIDEDSLICE' not in node_name and 'SPLIT' not in node_name:
            raise ValueError ('At node: %s. input channels!=output channels. This is unexpected for identity op!' % node_name)

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    input_tensor_scale = node['frontend']['input_tensor_scale']
    input_tensor_zp = node['frontend']['input_tensor_zp']
    output_tensor_scale = node['frontend']['output_tensor_scale']
    output_tensor_zp = node['frontend']['output_tensor_zp']

    is_folding_conv_x = 'force_folding_x' in node['frontend']
    is_folding_conv_y = 'force_folding_y' in node['frontend']
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

    requant_scale = input_tensor_scale/(output_tensor_scale*weight_value)
    node['frontend']['kernel_size'] = 1
    node['frontend']['padding'] = 0

    node['frontend']['stride'] = 1
    input_tensors_shape = node['frontend']['input_tensor'].get_folded_shape()
    node['frontend']['sparse_macs'] = folded_output_channels*input_tensors_shape[2]*input_tensors_shape[3]
    # Convert output_tensor_zp to scalar if it's a single-element array
    output_zp_scalar = output_tensor_zp.item() if isinstance(output_tensor_zp, np.ndarray) and output_tensor_zp.size == 1 else (output_tensor_zp[0] if isinstance(output_tensor_zp, np.ndarray) else output_tensor_zp)

    # Check if this is ADD converted to Identity (has add_constant_real)
    if 'add_constant_real' in node['frontend']:
        constant_real = node['frontend']['add_constant_real']

        # NEW FORMULA: bias_data = round(weight * (-input_zp + constant_real / input_scale))
        bias_data_array = np.round(weight_value * (-input_tensor_zp + constant_real / input_tensor_scale))

        # Check if per-channel or scalar
        unique_bias = np.unique(bias_data_array)
        if len(unique_bias) > 1:
            # Per-channel: call quantize2MathBlock for each channel
            bias_data = bias_data_array.astype(int)
            num_channels = len(bias_data)
            output_multiplier_arr = np.zeros(num_channels, dtype=np.int32)
            cInputH_arr = np.zeros(num_channels, dtype=np.int32)
            cInputL_arr = np.zeros(num_channels, dtype=np.int32)
            o_shift_arr = np.zeros(num_channels, dtype=np.int32)

            for ch in range(num_channels):
                # Get output_offset for this channel
                ch_output_offset = output_zp_scalar[ch] if isinstance(output_zp_scalar, np.ndarray) else output_zp_scalar

                # Call quantize2MathBlock for this channel
                mult, cH, cL, shift = quantize2MathBlock(
                    acc=None,
                    bias_data=int(bias_data[ch]),
                    scale=requant_scale,
                    output_offset=ch_output_offset,
                    output_activation_min=int8min,
                    output_activation_max=int8max
                )

                output_multiplier_arr[ch] = mult
                cInputH_arr[ch] = cH
                cInputL_arr[ch] = cL
                o_shift_arr[ch] = shift

            # Store per-channel arrays
            node['frontend']['output_multiplier'] = output_multiplier_arr
            node['frontend']['cInputH'] = cInputH_arr
            node['frontend']['cInputL'] = cInputL_arr
            node['frontend']['o_shift'] = o_shift_arr
        else:
            # Scalar bias_data (same for all channels)
            bias_data = int(bias_data_array[0] if isinstance(bias_data_array, np.ndarray) else bias_data_array)
            set_tflite_rq_params(node, acc=None, bias_data=bias_data, scale=requant_scale, output_offset=output_zp_scalar)
    else:
        # OLD FORMULA for regular Identity (not from ADD)
        bias_data = int(-8*output_zp_scalar)
        set_tflite_rq_params(node, acc=None, bias_data=bias_data, scale=requant_scale, output_offset=output_zp_scalar)

def convert_maxpoolk5_to_2xmaxpoolk3(ir: internal_representation.IR,original_node_name,node):
    node['frontend']['kernel_size'] = 3 # We change original node kernel from 5 to 3 and duplicate it to have 2 consecutive maxpool nodes with k=3
    node['frontend']['padding'] = 1
    node['attributes']['kernel_shape'] = [3,3]
    node['attributes']['pads'] = [1,1,1,1]

    dup_node = copy.deepcopy(node, memo=_memo_for_node(node))
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

def fix_max_pool_kernel_for_async_padding(ir: internal_representation.IR) -> internal_representation.IR:
    """
    Fix MaxPool kernel size for asynchronous padding mode.
    When padding is asymmetric (e.g., [0,1,0,1]), increase kernel size N -> N+1
    and set asym_padding flag for hardware to use pad_mode=1 (2 active AMMs).
    """
    nodes_in_graph = list(ir.graph.nodes(data=True))
    for (node_name, node) in nodes_in_graph:
        if node['op_type'] == OpType.MAXPOOL:
            # Check for asynchronous padding (asymmetric padding indicating pad_mode=1)
            has_async_padding = False
            if 'padding' in node['attributes']:
                pads = node['attributes']['padding'] if isinstance(node['attributes']['padding'], list) else node['attributes'].get('pads', [])
                # Asynchronous padding occurs when left!=right or top!=bottom padding
                if len(pads) >= 4:
                    # Check if asymmetric: NOT all elements are equal
                    is_asymmetric = len(set(pads)) > 1 and not (pads == [1, 0, 1, 0] )
                    if is_asymmetric:
                        if pads == [0,1,0,1]:
                           has_async_padding   = True
                           pads                = [1,1,1,1]
                        else:
                            raise ValueError(f"The asymmetric padding {pads} is not supported")
                    # else: symmetric padding, no async needed

            # For any kernel with asynchronous padding: set flag and increase kernel size N -> N+1
            if has_async_padding:
                node['frontend']['asym_padding'] = True
                current_kernel_size = node['frontend']['kernel_size']
                new_kernel_size = current_kernel_size + 1
                node['frontend']['kernel_size'] = new_kernel_size
                node['attributes']['kernel_shape'] = [new_kernel_size, new_kernel_size]
    return ir

def duplicate_maxpool_nodes(ir: internal_representation.IR) -> internal_representation.IR:
    nodes_in_graph = list(ir.graph.nodes(data=True))
    for (node_name, node) in nodes_in_graph:
       if node['op_type'] == OpType.MAXPOOL:
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
        # Recompute x_slices from the true unfolded width at the new fold factor rather than
        # doubling the previous grid-rounded x_slices, which would compound rounding excess
        # (e.g. width 112 -> 7 slices rounds to 8; doubling that gives 128 instead of 7).
        # Matches the fold-aware x_slices formula used in insert_identity_conv_before_node.
        unfolded_grid_xslices = int(math.ceil(original_tensor.shape[3] / MAX_GRID_WIDTH))
        x_slices = int(math.ceil(unfolded_grid_xslices / (2 ** output_folding_factor_x)))

        y_unfolding = False
        if (output_folding_factor_y > 0): # We can unfold y if output folding factor is already 0
            input_folding_factor_y-=1
            output_folding_factor_y-=1
            y_unfolding = True
    
        input_folding_coef_y = math.pow(2, input_folding_factor_y)
        folded_y_size = original_tensor.shape[2] // input_folding_coef_y
        num_y_tiles = math.ceil(folded_y_size / ir.amm_height)
    
    new_node_name = source_node['name']+'_'+ name_sufix
    new_node_name = ''.join(e for e in new_node_name if (e.isalnum() or e=='_')) # Removing special characters from node name as it is used as filename for debug file name (.xlsx)
    new_tensor = Tensor(new_tensor_name,None,producer=new_node_name,consumers=target_node_names,
                            is_constant=False,shape = original_tensor.shape,scale = original_tensor.scale,
                            zero_point=original_tensor.zero_point,folding_factor_x=output_folding_factor_x,
                            folding_factor_y=output_folding_factor_y, x_slices=x_slices)
    new_tensor.y_tiles = num_y_tiles
    node_h_tiles = source_node['frontend']['h_tiles']
    if ('unfold_x' in name_sufix) and ((x_slices % 2) != 0) and (node_h_tiles > 1):
        # Dummy/leftover slice (for odd unfolded x_slices) must land in h-tile 0 since
        # sequencer.py's is_dummy_unfolding only special-cases current_htile_num == 0.
        new_xslices_per_htile = get_xslices_per_htile(x_slices - 1, node_h_tiles, 1)
        new_xslices_per_htile[node_h_tiles-1] += 1
    else:
        new_xslices_per_htile = get_xslices_per_htile(x_slices, node_h_tiles, 1)
    new_tensor.xslices_per_htile = new_xslices_per_htile


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
        'h_tiles': source_node['frontend']['h_tiles'],
        'xslices_per_htile': source_node['frontend']['xslices_per_htile'],
        'x_slices': new_node_input_tensor.x_slices,
        'out_xslices_per_htile': new_xslices_per_htile,
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
    
    if y_unfolding:
        new_node['frontend']['force_unfolding_y'] = True

    if 'unfold_x' in name_sufix:
        if ((x_slices % 2) != 0):
            new_node['frontend']['is_dummy_unfolding'] = True

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
        if ('force_folding_x' in node['frontend'] and node['frontend']['force_folding_x'] and node['frontend']['output_tensor'].get_original_shape()[3]<=8):     
            insert_padding_whole_grid_node_between(ir, node)

    return ir

# Split a per-output-channel array (output_multiplier/cInputH/cInputL/o_shift, or a folded biases
# vector) along output channels. The array has length F * orig_output_channels, where F = 1 when no
# input folding is active and F > 1 when it is (calc_conv_qparams tiles the per-oc params F times via
# asym_fold_biases_continousychannels -> np.tile, keeping the original output-channel order within each
# tile). Reshaping to (F, orig_output_channels) and slicing the OC columns yields, per half, exactly the
# array that folding the half-OC params from scratch would produce. Accepts and returns the same type
# (list stays list, ndarray stays ndarray).
def _split_per_oc_param_by_output_channel(param, orig_output_channels, half_output_channels):
    is_list = isinstance(param, list)
    arr = np.asarray(param)
    fold = arr.shape[0] // orig_output_channels
    mat = arr.reshape(fold, orig_output_channels)
    first = mat[:, :half_output_channels].reshape(-1)
    second = mat[:, half_output_channels:].reshape(-1)
    if is_list:
        return first.tolist(), second.tolist()
    return first, second

# Split a folded weights tensor along its tiled output-channel axis (F*orig_output_channels, laid out
# as F contiguous blocks of orig_output_channels; see asym_fold_weights_continousychannels). Reshaping
# to (F, orig_output_channels, ...) and slicing OC columns matches folding already-split weights.
def _split_folded_tensor_by_output_channel(folded_data, orig_output_channels, half_output_channels):
    fold = folded_data.shape[0] // orig_output_channels
    reshaped = folded_data.reshape(fold, orig_output_channels, *folded_data.shape[1:])
    first = reshaped[:, :half_output_channels].reshape(-1, *folded_data.shape[1:])
    second = reshaped[:, half_output_channels:].reshape(-1, *folded_data.shape[1:])
    return first, second

# N-way generalizations of the two per-output-channel splitters above. `bounds` is the N+1 output-
# channel boundaries; chunk i owns [bounds[i]:bounds[i+1]]. Both preserve the tiled (F, OC) fold layout
# via reshape+column-slice per chunk, same as the two-way helpers. Returns a list of N slices.
def _split_per_oc_param_by_output_channel_n(param, orig_output_channels, bounds):
    is_list = isinstance(param, list)
    arr = np.asarray(param)
    fold = arr.shape[0] // orig_output_channels
    mat = arr.reshape(fold, orig_output_channels)
    pieces = []
    for i in range(len(bounds) - 1):
        piece = mat[:, bounds[i]:bounds[i + 1]].reshape(-1)
        pieces.append(piece.tolist() if is_list else piece)
    return pieces

def _split_folded_tensor_by_output_channel_n(folded_data, orig_output_channels, bounds):
    fold = folded_data.shape[0] // orig_output_channels
    reshaped = folded_data.reshape(fold, orig_output_channels, *folded_data.shape[1:])
    return [reshaped[:, bounds[i]:bounds[i + 1]].reshape(-1, *folded_data.shape[1:])
            for i in range(len(bounds) - 1)]

def _slice_fake_identity_weights(data, orig_output_channels, block_count, bounds):
    """Slice a MaxPool/Add fake identity weights tensor (row axis = OC, column axis = block_count
    concatenated IC blocks, each block the same width as OC since these ops are elementwise and map
    output channel c from input channel c in every input) into per-output-channel-chunk pieces. Row
    axis and every column block share the SAME [lo:hi] bounds so the diagonal identity structure survives
    the split; each axis may additionally carry its own folding factor (rows folded by the op's output
    folding, each column block folded by its input's folding), handled generically via reshape."""
    out_fold = data.shape[0] // orig_output_channels
    col_total = data.shape[1]
    col_block = col_total // block_count
    in_fold = col_block // orig_output_channels
    row_reshaped = data.reshape(out_fold, orig_output_channels, col_total, *data.shape[2:])
    pieces = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        row_piece = row_reshaped[:, lo:hi].reshape(-1, col_total, *data.shape[2:])
        col_pieces = []
        for b in range(block_count):
            block = row_piece[:, b * col_block:(b + 1) * col_block]
            block = block.reshape(row_piece.shape[0], in_fold, orig_output_channels, *data.shape[2:])
            col_pieces.append(block[:, :, lo:hi].reshape(row_piece.shape[0], -1, *data.shape[2:]))
        pieces.append(np.concatenate(col_pieces, axis=1))
    return pieces

def _output_channel_bounds(total_channels, n):
    """N monotone output/input-channel chunk boundaries summing to total_channels, each chunk >= 1."""
    bounds = [round(k * total_channels / n) for k in range(n + 1)]
    bounds[0] = 0
    bounds[n] = total_channels
    # Guarantee strictly increasing (each chunk at least 1 channel) even for awkward ratios.
    for i in range(1, n + 1):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = bounds[i - 1] + 1
    # Clamp the tail back so the last boundary is exactly total_channels.
    for i in range(n, 0, -1):
        if bounds[i] > total_channels:
            bounds[i] = total_channels
        if bounds[i - 1] >= bounds[i]:
            bounds[i - 1] = bounds[i] - 1
    return bounds

# AMM blocks hold 16 channels each, so channel splits are done in whole-block (multiple-of-16) chunks
# to avoid a partially-filled leading block wasting a whole block per piece. `max_chunk_channels` is
# the largest chunk fitting the AMM budget; we take the largest multiple of 16 <= it (at least 1 block).
# Returns [0, ..., total_channels]; a single-chunk result means no split was needed.
AMM_CHANNELS_PER_BLOCK = 16

def _block_aligned_bounds(total_channels, max_chunk_channels, align=AMM_CHANNELS_PER_BLOCK, pow2_pieces=False):
    chunk = (int(max_chunk_channels) // align) * align
    if chunk < align:
        chunk = align
    if chunk >= total_channels:
        return [0, total_channels]
    num_pieces = -(-total_channels // chunk)  # ceil div: minimal piece count under the chunk cap
    if pow2_pieces:
        # Output folding needs a power-of-two piece count so the fold pairs line up across pieces.
        num_pieces = 1 << (num_pieces - 1).bit_length()
    return _balanced_bounds(total_channels, num_pieces, align)


# Split total_channels into exactly num_pieces block-aligned pieces, evenly (front pieces absorb the
# +1-block remainder), instead of stepping by fixed-width chunk and leaving an undersized trailing
# piece (e.g. 336+336+336+16 vs 352+336+336 for 1024ch/336-cap) - same count and cap, better spread.
def _balanced_bounds(total_channels, num_pieces, align=AMM_CHANNELS_PER_BLOCK):
    num_pieces = max(1, int(num_pieces))
    total_units = -(-total_channels // align)  # ceil div into align-sized units
    num_pieces = min(num_pieces, total_units)  # can't split finer than one block per piece
    base_units, extra = divmod(total_units, num_pieces)
    bounds = [0]
    remaining = total_channels
    for i in range(num_pieces):
        units = base_units + (1 if i < extra else 0)
        size = min(units * align, remaining)
        bounds.append(bounds[-1] + size)
        remaining -= size
    return bounds

# Cutoff split_large_conv_nodes' output_total_channels<=amm branch uses to prefer input-only splitting
# over the combined output+input cascade. Modest cap so a genuinely oversized input still falls
# through to the combined cascade rather than building a deep Add tree.
_SMALL_PURE_INPUT_SPLIT_MAX_PIECES = 4

# Bounds for a pure input-channel split (no output split), sized so each slice-conv's recombine Add
# fits AMM: input_slice + (add_dup-1)*output_total_channels <= amm. Returns None if the output alone
# leaves no room for one input slice, or if fitting needs more than _SMALL_PURE_INPUT_SPLIT_MAX_PIECES
# pieces (then the output should be split too via the caller's combined cascade).
def _small_pure_input_split_bounds(amm, input_channels, output_total_channels, add_dup, in_mult, identity_amm_cap,
                                    add_htile_relief=1):
    # The Add's own footprint (add_dup*output_total_channels, two full output-sized operands at once)
    # is fixed regardless of input slicing - only checked/shrunk via `piece` below for the slice-conv.
    # If that fixed footprint already overflows AMM, only h-tiling the Add's blob to 2 can relieve it
    # (add_htile_relief=2, passed only when min(input_x_slices, output_x_slices) >= 2, matching
    # get_num_h_tiles' precondition); when x_slices==1 on either side that relief is impossible
    # (h_tiles can't go below 1 x_slice/h-tile), so the overflow is real -> return None and let the
    # caller fall through to the combined output+input cascade instead.
    if -(-(add_dup * output_total_channels) // add_htile_relief) > amm:
        return None
    piece = amm - (add_dup - 1) * output_total_channels  # single-input-buffer, no h-tile relief assumed
    if piece <= 0:
        return None
    max_chunk = piece // in_mult if in_mult else piece
    if identity_amm_cap is not None:
        max_chunk = min(max_chunk, identity_amm_cap // in_mult if in_mult else identity_amm_cap)
    if max_chunk <= 0:
        return None
    bounds = _block_aligned_bounds(input_channels, max_chunk)
    n_pieces = len(bounds) - 1
    if n_pieces < 2 or n_pieces > _SMALL_PURE_INPUT_SPLIT_MAX_PIECES:
        return None
    return bounds

# Split a conv with N output channels into N/2 and N/2 (or N/2-1 if odd). Original output tensor
# stays in the IR; the 2 new split output tensors are made contiguous in DDR at its address. If
# quant params (output_multiplier/cInputH/cInputL/o_shift, folded weights/biases) are already
# computed, they're split per output channel here; otherwise the normal pass computes them later.
# Note: a Concat after the split nodes could hit the same AMM problem; a Sync-node output should
# only fire once the second half is written.
# Generate a graph-node name based on `base` that isn't taken. Repeated/nested splitting can re-split
# a node whose deterministic dup name already exists; reusing it would collide with the prior split
# and, for the input-split Add, create a self-loop (non-topologically-sortable graph). A counter keeps
# generated names distinct; tensor names derive from the node's already-unique tensors.
def _unique_graph_node_name(ir, base):
    if base not in ir.graph.nodes:
        return base
    i = 2
    while f'{base}_{i}' in ir.graph.nodes:
        i += 1
    return f'{base}_{i}'

# Split a Conv along its INPUT channels into two convs recombined with an Add (input-channel
# counterpart of the removed 2-way output-channel split, reducing input-channel AMM contribution).
# Conv is linear over input channels: conv(x) = conv(x[:,:h],W[:,:h],bias) + conv(x[:,h:],W[:,h:],0);
# bias kept on first half only to avoid double-counting.
#
# Requant: both halves and the Add share the original conv's output scale/zp, so calc_add_qparams
# reduces to a plain scale-matched integer add (q = in0+in1-zp), reusing the original requant. This is
# functionally exact but not bit-exact (each half requantizes to int8 before the Add, so a partial sum
# near int8 range can saturate/round differently than the full sum).
#
# DDR/AMM: the two slices share the producer's output region (registered like the output split's
# tensors), since channel-contiguous DDR makes [0:h] and [h:] two contiguous reads.
#
# RMS/N-sigma bound on a half-conv's achievable output range (used by
# _recompute_input_split_conv_requant's headroom widening): treats zero-point-corrected input codes as
# uniform over int8 range (std=255/sqrt(12)); CLT makes the accumulator ~Gaussian over many weight
# elements, so N=5 sigma (empirically where max abs error bottoms out on a 1024-in/1024-out conv2d
# test) covers the realistic range with low saturation risk, unlike the theoretical worst-case
# weight-sum bound which forces K into 60-80 and destroys far more precision than it saves.
_SPLIT_REQUANT_N_SIGMA = 5
_SPLIT_REQUANT_UNIFORM_STD = 255.0 / math.sqrt(12)

# A "dead"/pruned output channel: TFLite quantizes it against a degenerate near-zero weight scale
# (calc_conv_qparams's zeroed_oc guard, e.g. 3.9e-9), whose real magnitude is negligible even when
# int8 codes aren't exactly 0 (observed: CONV2D_40 och=630, scale=3.9e-9, codes up to 11 -> real
# weight ~4e-8) - the exact-all-zero check (np.all(w_row==0)) misses this. Its bias, quantized
# against an equally tiny bias_scale, is huge in raw int32 terms but near-zero in real value; feeding
# it unscaled overflows the requant by ~1e6x and saturates int8 on recombine. 1e-6 sits well below
# real weight scales (~1e-3 to ~1e-1) and above the sentinel (~1e-9), so it won't misfire.
_DEAD_CHANNEL_WEIGHT_SCALE_EPS = 1e-6


def _compute_split_requant_K(node):
    """Statistical-bound K for one half-conv slice (see _SPLIT_REQUANT_N_SIGMA docstring above).

    Read-only: does not mutate node. Callers that split one conv into multiple sibling slices should call
    this once per sibling and share max(K_i) across all of them (see _recompute_input_split_conv_requant's
    K parameter) - siblings feed the same recombine Add, and giving each an independently-sized scale (even
    though calc_add_qparams handles mismatched scales correctly) forces the Add's own block-diagonal weights
    off a clean 64/64 split, adding avoidable rounding noise on top of the two halves' own quantization
    noise. A shared K removes that extra term for free.
    """
    output_channels = node['frontend']['output_channels']
    w_int8 = node['frontend']['weights_tensor'].data
    biases_tensor = node['frontend']['biases_tensor']
    input_tensor_scale = node['frontend']['input_tensor_scale']
    input_tensor_zp = node['frontend']['input_tensor_zp']
    weights_per_channel_scale = node['frontend']['weights_per_channel_scale']
    output_tensor_scale_item = node['frontend']['output_tensor_scale'].item()
    has_per_channel_scale = (weights_per_channel_scale.size > 1)
    fallback_scale = weights_per_channel_scale[0].item() if not has_per_channel_scale else None

    K = 1
    for och in range(output_channels):
        w_row = w_int8[och].astype(np.int64)
        w_sum = int(np.sum(w_row))
        oc_weight_scale = weights_per_channel_scale[och].item() if has_per_channel_scale else fallback_scale
        # Same fix as _recompute_input_split_conv_requant's zeroed_oc branch: only the fallback SCALE
        # guards a degenerate/near-zero per-channel scale. All-zero weight DATA with a valid scale (a
        # block-sparse/grouped conv slice carrying a real bias) must still have that bias's true
        # contribution counted in this slice's headroom bound, or K ends up under-provisioned once the
        # requant step above correctly includes the bias.
        invalid_scale = abs(oc_weight_scale) < _DEAD_CHANNEL_WEIGHT_SCALE_EPS
        requant_scale = (input_tensor_scale*0.0039/output_tensor_scale_item) if invalid_scale \
            else (input_tensor_scale*oc_weight_scale/output_tensor_scale_item)
        requant_bias = int(biases_tensor.data[och]) - w_sum*input_tensor_zp

        std_mac = math.sqrt(float(np.sum(w_row.astype(np.float64)**2))) * _SPLIT_REQUANT_UNIFORM_STD
        bound = _SPLIT_REQUANT_N_SIGMA*std_mac*requant_scale + abs(requant_bias*requant_scale)
        K = max(K, int(math.ceil(bound/127.0)))
    return K


def _recompute_input_split_conv_requant(node, fused_act, K=None):
    """Recompute a half-conv's per-output-channel requant params after an input-channel split.

    calc_conv_qparams folds the input zero-point correction into cInput as a sum over ALL input
    channels; after splitting, each half sees only its own slice (second half's bias zeroed), so the
    inherited cInput over/double-counts the correction. Recomputing from the half's own sliced
    weights/bias makes it a standalone quantized conv, so the recombine Add (out = s1+s2-out_zp)
    reconstructs the original output. Mirrors calc_conv_qparams's per-channel loop/fold-replication
    but skips its side effects (no folded_weights_tensor, no avgPool probe).

    Headroom widening: naively quantizing a half at the ORIGINAL output scale/zp saturates int8 before
    the two halves are summed (empirically verified: max abs error 136 on a 1024-in/1024-out conv2d
    test, ~15-25% elements wrong, isolated to per-half saturation, not to calc_conv_qparams/
    quantize2MathBlock/calc_add_qparams). Fix: quantize at a coarser scale (output_scale*K, zp=0)
    bounded statistically (see _compute_split_requant_K) rather than the generic int30 bound. The
    recombine Add already supports independently-scaled inputs, so no changes needed there.

    K: precomputed widening factor shared across sibling slices feeding the same Add (see
    _compute_split_requant_K). If None, computed standalone - only correct with no siblings to share.
    """
    output_channels = node['frontend']['output_channels']
    w_int8 = node['frontend']['weights_tensor'].data
    biases_tensor = node['frontend']['biases_tensor']
    input_tensor_scale = node['frontend']['input_tensor_scale']
    input_tensor_zp = node['frontend']['input_tensor_zp']
    weights_per_channel_scale = node['frontend']['weights_per_channel_scale']
    original_output_tensor_scale_item = node['frontend']['output_tensor_scale'].item()
    original_output_offset = node['frontend']['output_tensor_zp'].item()
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    stride = node['frontend']['stride']

    # Recompute weights_stride_x/y exactly as calc_conv_qparams does, so the fold replication of the
    # per-oc params below lays them out identically.
    weights_stride_x = 1
    weights_stride_y = 1
    if stride == 2:
        weights_stride_x = 2
        weights_stride_y = 2
        if 'force_folding_x' in node['frontend']:
            weights_stride_x = 1

    has_per_channel_scale = (weights_per_channel_scale.size > 1)
    fallback_scale = weights_per_channel_scale[0].item() if not has_per_channel_scale else None

    if K is None:
        K = _compute_split_requant_K(node)

    output_tensor_scale_item = original_output_tensor_scale_item*K
    output_offset = original_output_offset

    base_act_min = int8min
    base_act_max = int8max
    if K > 1:
        # Widen this half's own output tensor scale (zero_point recentred to 0, since a partial sum has
        # no reason to share the full conv's output zero-point) so calc_add_qparams sees the real
        # scale/zp when it builds the recombine Add and rescales back to the original output exactly.
        output_offset = 0
        half_output_tensor = node['frontend']['output_tensor']
        half_output_tensor.scale = np.array([output_tensor_scale_item], dtype=half_output_tensor.scale.dtype) \
            if isinstance(half_output_tensor.scale, np.ndarray) else output_tensor_scale_item
        half_output_tensor.zero_point = np.array([0], dtype=half_output_tensor.zero_point.dtype) \
            if isinstance(half_output_tensor.zero_point, np.ndarray) else 0
        node['frontend']['output_tensor_scale'] = half_output_tensor.scale
        node['frontend']['output_tensor_zp'] = half_output_tensor.zero_point
    elif fused_act == 'RELU':
        base_act_min = output_offset
    elif fused_act == 'RELU6':
        base_act_min = output_offset
        base_act_max = min(round((6 / output_tensor_scale_item) + output_offset), int8max)

    # Pass 2: recompute the real per-channel requant params at the (possibly widened) half scale/zp.
    output_multiplier = [0 for _ in range(output_channels)]
    cInputH = [0 for _ in range(output_channels)]
    cInputL = [0 for _ in range(output_channels)]
    o_shift = [0 for _ in range(output_channels)]
    for och in range(output_channels):
        w_sum = np.sum(w_int8[och])
        oc_weight_scale = weights_per_channel_scale[och].item() if has_per_channel_scale else fallback_scale
        # invalid_scale guards only a degenerate/near-zero PER-CHANNEL SCALE, not all-zero weight
        # DATA (e.g. a valid block-sparse channel with a real bias) - that case's mac is already 0
        # via w_sum==0, and zeroing requant_bias here would wrongly drop its real bias term.
        invalid_scale = abs(oc_weight_scale) < _DEAD_CHANNEL_WEIGHT_SCALE_EPS
        requant_scale = (input_tensor_scale*0.0039/output_tensor_scale_item) if invalid_scale \
            else (input_tensor_scale*oc_weight_scale/output_tensor_scale_item)
        requant_bias = biases_tensor.data[och].item()
        requant_bias -= w_sum*input_tensor_zp
        requant_bias = np.int32(requant_bias)
        output_multiplier[och], cInputH[och], cInputL[och], o_shift[och] = \
            quantize2MathBlock(None, int(requant_bias), requant_scale, output_offset, base_act_min, base_act_max)

    node['frontend']['output_multiplier'] = output_multiplier
    node['frontend']['cInputH'] = cInputH
    node['frontend']['cInputL'] = cInputL
    node['frontend']['o_shift'] = o_shift
    if (input_folding_factor_x > 0 or input_folding_factor_y > 0):
        for key in ('output_multiplier', 'cInputH', 'cInputL', 'o_shift'):
            node['frontend'][key] = folding_algo.get_asym_folded_per_oc_params(
                node['frontend'][key], input_folding_factor_x=input_folding_factor_x,
                input_folding_factor_y=input_folding_factor_y,
                stride_x=weights_stride_x, stride_y=weights_stride_y)


# Slice a folded weights tensor along a range of ORIGINAL (pre-fold) input channels. The folded IC
# axis is fold*orig_input_channels, laid out as `fold` contiguous blocks each in original IC order
# (asym_fold_weights_continousychannels). A contiguous original-channel range is thus `fold` strided
# blocks on the folded axis; reshape to (OC, fold, orig_input_channels, ...) and slice IC columns to
# match. Re-folding an already-sliced tensor would use a different stride and produce wrong data, so
# splits must slice the already-folded tensor (fold-then-split), never fold a pre-split slice.
def _slice_folded_weights_by_input_channel(folded_weights_data, orig_input_channels, lo, hi):
    oc, folded_ic = folded_weights_data.shape[0], folded_weights_data.shape[1]
    fold = folded_ic // orig_input_channels
    reshaped = folded_weights_data.reshape(oc, fold, orig_input_channels, *folded_weights_data.shape[2:])
    sliced = reshaped[:, :, lo:hi]
    return np.ascontiguousarray(sliced.reshape(oc, fold * (hi - lo), *folded_weights_data.shape[2:]))


def split_conv_in_channels_in_two(ir: internal_representation.IR, original_node_name, node):
    # Capture the original conv's graph connectivity before we mutate it. These are reused verbatim
    # when wiring the new nodes (the get_updated_* IR helpers can't be used here: they key on input
    # names matching producer output names, but this split renames inputs[0] to a channel slice).
    original_preceding_params = list(node['frontend']['preceding_nodes_params'])
    original_following_params = list(node['frontend']['following_nodes_params'])

    # Capture the original conv's backend (built by get_ops_grid_config) before we mutate it, so the
    # new Add node can be given a valid backend dict derived from it. When this split runs in the late
    # blob-failure path (after grid config), the Add is created here and never passes through
    # get_ops_grid_config, so without this it would have no 'backend' and crash update_node_tiling_info
    # with KeyError: 'backend'. None when the split runs before backend construction.
    original_backend = copy.deepcopy(node['backend']) if 'backend' in node else None

    # Capture the conv's fused activation (if any) before splitting; both halves recompute their requant
    # params from it. (A fused activation is only exactly reproducible by an input-channel split if it is
    # applied after the recombine Add; with no fused activation - the common case for split candidates -
    # the per-half requant is exact.)
    fused_act = node['attributes'].get('fused_activation_function') if isinstance(node.get('attributes'), dict) else None

    dup_node = copy.deepcopy(node, memo=_memo_for_node(node))
    dup_node['frontend']['preceding_nodes_params'] = list(original_preceding_params)

    # Set the new node name (unique, so re-splitting a node doesn't collide with a prior '_split_ic')
    dup_node_name = _unique_graph_node_name(ir, original_node_name + '_split_ic')
    dup_node['name'] = dup_node_name

    # Split the input channels in two (first half gets the extra channel when odd)
    orig_input_channels = node['frontend']['input_channels']
    half_input_channels = orig_input_channels - (orig_input_channels // 2)
    remaining_input_channels = orig_input_channels - half_input_channels
    node['frontend']['input_channels'] = half_input_channels
    dup_node['frontend']['input_channels'] = remaining_input_channels
    # Marks a node as one piece of an input-channel-split conv (recombined downstream by an Add) - lets
    # other code (e.g. reports, later split passes) tell it apart from a plain, unsplit conv.
    node['frontend']['is_input_channel_split'] = True
    dup_node['frontend']['is_input_channel_split'] = True
    # Output channels and tiling are unchanged - carry them over like the output split does
    dup_node['frontend']['h_tiles'] = node['frontend']['h_tiles']
    dup_node['frontend']['xslices_per_htile'] = node['frontend']['xslices_per_htile']

    # Split the weights along axis 1 (input channels).
    if 'weights_tensor' in node['frontend']:
        orig_weights_tensor = node['frontend']['weights_tensor']
        orig_weights_tensor_name = orig_weights_tensor.name
        # Weight scale/zp are per OUTPUT channel (axis 0), which is unchanged by an input-channel
        # split, so both halves keep the full, identical scale/zero_point (do NOT slice them).
        orig_weights_tensor_scale = orig_weights_tensor.scale
        orig_weights_tensor_zp = orig_weights_tensor.zero_point
        weights_tensor_name_split1 = orig_weights_tensor_name + '_split_ic1'
        weights_tensor_name_split2 = orig_weights_tensor_name + '_split_ic2'
        weights_tensor_data_split1 = orig_weights_tensor.data[:, :half_input_channels, :, :]
        weights_tensor_data_split2 = orig_weights_tensor.data[:, half_input_channels:, :, :]
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
        node['frontend']['weights_tensor'] = ir.tensors[weights_tensor_name_split1]
        dup_node['frontend']['weights_tensor'] = ir.tensors[weights_tensor_name_split2]
        node['inputs'][1] = weights_tensor_name_split1
        dup_node['inputs'][1] = weights_tensor_name_split2

        # Folded weights/biases must be sliced from the already-folded tensor, not re-folded from the
        # smaller unfolded slices above - see _slice_folded_weights_by_input_channel for why.
        input_folding_factor_x = node['frontend'].get('input_folding_factor_x', 0)
        input_folding_factor_y = node['frontend'].get('input_folding_factor_y', 0)
        if (input_folding_factor_x > 0 or input_folding_factor_y > 0) and 'folded_weights_tensor' in node['frontend']:
            orig_folded_weights_tensor = node['frontend']['folded_weights_tensor']
            folded_weights_tensor_name_split1 = orig_folded_weights_tensor.name + '_split_ic1'
            folded_weights_tensor_name_split2 = orig_folded_weights_tensor.name + '_split_ic2'
            folded_weights_data_split1 = _slice_folded_weights_by_input_channel(
                orig_folded_weights_tensor.data, orig_input_channels, 0, half_input_channels)
            folded_weights_data_split2 = _slice_folded_weights_by_input_channel(
                orig_folded_weights_tensor.data, orig_input_channels, half_input_channels, orig_input_channels)
            ir.tensors[folded_weights_tensor_name_split1] = Tensor(
                folded_weights_tensor_name_split1, folded_weights_data_split1, is_constant=True,
                shape=list(folded_weights_data_split1.shape),
                scale=orig_folded_weights_tensor.scale,
                zero_point=orig_folded_weights_tensor.zero_point)
            ir.tensors[folded_weights_tensor_name_split2] = Tensor(
                folded_weights_tensor_name_split2, folded_weights_data_split2, is_constant=True,
                shape=list(folded_weights_data_split2.shape),
                scale=orig_folded_weights_tensor.scale,
                zero_point=orig_folded_weights_tensor.zero_point)
            node['frontend']['folded_weights_tensor'] = ir.tensors[folded_weights_tensor_name_split1]
            dup_node['frontend']['folded_weights_tensor'] = ir.tensors[folded_weights_tensor_name_split2]
            node['frontend']['folded_kernel_size'] = folded_weights_data_split1.shape[3]
            dup_node['frontend']['folded_kernel_size'] = folded_weights_data_split1.shape[3]
            folded_pad = folded_weights_data_split1.shape[3] // 2
            node['frontend']['folded_padding'] = [folded_pad, folded_pad, folded_pad, folded_pad]
            dup_node['frontend']['folded_padding'] = [folded_pad, folded_pad, folded_pad, folded_pad]
            
        if 'biases_tensor' in node['frontend']:
            # First half keeps the original biases; second half gets a zeroed biases tensor so the
            # bias term is added exactly once when the two halves are summed.
            orig_biases_tensor = node['frontend']['biases_tensor']
            zero_biases_tensor_name = orig_biases_tensor.name + '_split_ic_zero'
            zero_biases_data = np.zeros_like(orig_biases_tensor.data)
            ir.tensors[zero_biases_tensor_name] = Tensor(
                zero_biases_tensor_name, zero_biases_data, is_constant=True,
                shape=list(zero_biases_data.shape),
                scale=orig_biases_tensor.scale,
                zero_point=orig_biases_tensor.zero_point)
            dup_node['frontend']['biases_tensor'] = ir.tensors[zero_biases_tensor_name]
            dup_node['inputs'][2] = zero_biases_tensor_name

            if (input_folding_factor_x > 0 or input_folding_factor_y > 0) and 'folded_biases_tensor' in node['frontend']:
                # Biases are per OUTPUT channel, unaffected by an input-channel split: first half keeps
                # the original folded biases, second half gets a zeroed folded biases tensor.
                orig_folded_biases_tensor = node['frontend']['folded_biases_tensor']
                zero_folded_biases_tensor_name = orig_folded_biases_tensor.name + '_split_ic_zero'
                zero_folded_biases_data = np.zeros_like(orig_folded_biases_tensor.data)
                ir.tensors[zero_folded_biases_tensor_name] = Tensor(
                    zero_folded_biases_tensor_name, zero_folded_biases_data, is_constant=True,
                    shape=list(zero_folded_biases_data.shape),
                    scale=orig_folded_biases_tensor.scale,
                    zero_point=orig_folded_biases_tensor.zero_point)
                dup_node['frontend']['folded_biases_tensor'] = ir.tensors[zero_folded_biases_tensor_name]

        # Update sparsity/macs for each half. Output channels and spatial dims are unchanged; only the
        # input-channel count (and therefore the weight/mac count) differs between halves.
        for n in [node, dup_node]:
            weights_tensor = n['frontend']['weights_tensor']
            dense_weights = weights_tensor.data.size
            sparse_weights = np.count_nonzero(weights_tensor.data)
            n['frontend']['dense_weights'] = dense_weights
            n['frontend']['sparse_weights'] = sparse_weights
            n['frontend']['weights_sparsity'] = 1 - (sparse_weights / dense_weights)

            stride = n['frontend']['stride']
            input_tensor_shape = n['frontend']['input_tensor'].get_original_shape()
            dense_macs = dense_weights * input_tensor_shape[2] * input_tensor_shape[3] / (stride * stride)
            sparse_macs = sparse_weights * input_tensor_shape[2] * input_tensor_shape[3] / (stride * stride)
            n['frontend']['dense_macs'] = dense_macs
            n['frontend']['sparse_macs'] = sparse_macs
            n['frontend']['macs_sparsity'] = 1 - (sparse_macs / dense_macs)
            n['frontend']['weights_per_channel_scale'] = weights_tensor.scale
            n['frontend']['weights_per_channel_zp'] = weights_tensor.zero_point

    if 'backend' in node:
        # backend['input_channels'] is FOLDED (from get_folded_shape), frontend['input_channels'] is
        # not. Folding multiplies both halves by the same 2^(x_fold+y_fold) factor, so scale the
        # unfolded halves by that ratio to stay consistent with the sliced folded input shapes
        # (checked in read_missing_tensors_from_ddr). Output channels/oc_groups are unchanged.
        fold_ratio = original_backend['input_channels'] // orig_input_channels
        node['backend']['input_channels'] = half_input_channels * fold_ratio
        dup_node['backend']['input_channels'] = remaining_input_channels * fold_ratio

    # Split the input tensor along channels so each half-conv reads its own slice. The two slices are
    # views of the producer's output region (mirrors the output split's two output tensors).
    orig_input_tensor = node['frontend']['input_tensor']
    orig_input_tensor_name = orig_input_tensor.name

    input_tensor_split1 = copy.deepcopy(orig_input_tensor)
    input_tensor_name_split1 = orig_input_tensor_name + '_split_ic1'
    if input_tensor_split1.data is not None:
        input_tensor_data_split1 = input_tensor_split1.data[:, :half_input_channels, :, :]
        input_tensor_split1.data = input_tensor_data_split1
        input_tensor_split1.shape = list(input_tensor_data_split1.shape)
        input_tensor_split1.shape_real_x16[1] = half_input_channels
    else:
        input_tensor_split1.shape = [
            orig_input_tensor.shape[0],
            half_input_channels,
            orig_input_tensor.shape[2],
            orig_input_tensor.shape[3]
        ]
        input_tensor_split1.shape_real_x16 = [
            orig_input_tensor.shape_real_x16[0],
            half_input_channels,
            orig_input_tensor.shape_real_x16[2],
            orig_input_tensor.shape_real_x16[3]
        ]
    input_tensor_split1.name = input_tensor_name_split1
    input_tensor_split1.consumers = [node['name']]

    input_tensor_split2 = copy.deepcopy(orig_input_tensor)
    input_tensor_name_split2 = orig_input_tensor_name + '_split_ic2'
    if input_tensor_split2.data is not None:
        input_tensor_data_split2 = input_tensor_split2.data[:, half_input_channels:, :, :]
        input_tensor_split2.data = input_tensor_data_split2
        input_tensor_split2.shape = list(input_tensor_data_split2.shape)
        input_tensor_split2.shape_real_x16[1] = orig_input_tensor.shape[1] - half_input_channels
    else:
        input_tensor_split2.shape = [
            orig_input_tensor.shape[0],
            orig_input_tensor.shape[1] - half_input_channels,
            orig_input_tensor.shape[2],
            orig_input_tensor.shape[3]
        ]
        input_tensor_split2.shape_real_x16 = [
            orig_input_tensor.shape_real_x16[0],
            orig_input_tensor.shape[1] - half_input_channels,
            orig_input_tensor.shape_real_x16[2],
            orig_input_tensor.shape_real_x16[3]
        ]
    input_tensor_split2.name = input_tensor_name_split2
    input_tensor_split2.consumers = [dup_node_name]

    ir.tensors[input_tensor_name_split1] = input_tensor_split1
    ir.tensors[input_tensor_name_split2] = input_tensor_split2

    node['frontend']['input_tensor'] = input_tensor_split1
    node['inputs'][0] = input_tensor_name_split1
    node['frontend']['input_tensor_scale'] = orig_input_tensor.scale
    node['frontend']['input_tensor_zp'] = orig_input_tensor.zero_point
    dup_node['frontend']['input_tensor'] = input_tensor_split2
    dup_node['inputs'][0] = input_tensor_name_split2
    dup_node['frontend']['input_tensor_scale'] = orig_input_tensor.scale
    dup_node['frontend']['input_tensor_zp'] = orig_input_tensor.zero_point

    # Associate the input slices with the producer's original output tensor so the allocator places
    # them contiguously in the same region (same maps the output split uses).
    ir.split_tensor_to_original_tensor_map[input_tensor_name_split1] = orig_input_tensor_name
    ir.split_tensor_to_original_tensor_map[input_tensor_name_split2] = orig_input_tensor_name
    # Merge, don't overwrite - see the identical fix/comment at the other split_tensor_to_
    # original_tensor_map registration site below (split_conv_in_channels_in_m) for why an
    # unconditional `=` here can silently drop an already-registered split-descendant name.
    existing = ir.original_tensor_to_split_tensor_map.get(orig_input_tensor_name, [])
    for name in [input_tensor_name_split2, input_tensor_name_split1]:
        if name not in existing:
            existing.append(name)
    ir.original_tensor_to_split_tensor_map[orig_input_tensor_name] = existing
    # The producer's output now also (logically) feeds the duplicate half (mirrors the output split,
    # which appends the dup to the shared input tensor's consumers).
    if dup_node_name not in orig_input_tensor.consumers:
        orig_input_tensor.consumers.append(dup_node_name)

    # Each half-conv gets its own output tensor (FULL output channels), both at the original output
    # scale/zp; the Add then sums them into the original output tensor.
    orig_output_tensor_name = node['outputs'][0]
    orig_output_tensor = ir.tensors[orig_output_tensor_name]

    output_tensor_split1 = copy.deepcopy(orig_output_tensor)
    output_tensor_name_split1 = orig_output_tensor_name + '_split_ic1'
    output_tensor_split1.name = output_tensor_name_split1
    output_tensor_split1.producer = node['name']
    output_tensor_split1.consumers = []
    node['outputs'][0] = output_tensor_name_split1
    node['frontend']['output_tensor'] = output_tensor_split1

    output_tensor_split2 = copy.deepcopy(orig_output_tensor)
    output_tensor_name_split2 = orig_output_tensor_name + '_split_ic2'
    output_tensor_split2.name = output_tensor_name_split2
    output_tensor_split2.producer = dup_node_name
    output_tensor_split2.consumers = []
    dup_node['outputs'][0] = output_tensor_name_split2
    dup_node['frontend']['output_tensor'] = output_tensor_split2

    # Recompute each half's requant params (cInput etc.) from its own sliced weights/bias now that
    # both are in place - otherwise both keep the original full-channel cInput, over/double-counting
    # the zero-point correction and bias, and the recombine Add can't reconstruct the output. Share
    # one K (max of both bounds) rather than per-half, to reduce recombine-Add rounding noise.
    shared_K = max(_compute_split_requant_K(node), _compute_split_requant_K(dup_node))
    _recompute_input_split_conv_requant(node, fused_act, K=shared_K)
    _recompute_input_split_conv_requant(dup_node, fused_act, K=shared_K)

    # NOTE: output_tensor_split1/split2 are deliberately NOT registered in the split maps - they're
    # distinct full-output-channel tensors summed by the Add, not slices of orig_output. A previous
    # version registered them (with Tensor-object keys) and overwrote the correct input-slice entry
    # above, corrupting name-based lookups in mark_blob_outputs_deallocations.

    ir.tensors[output_tensor_name_split1] = output_tensor_split1
    ir.tensors[output_tensor_name_split2] = output_tensor_split2

    # Build the Add node that recombines the two halves into the original output tensor (unique name,
    # so re-splitting a node doesn't reuse a prior Add and rewire it onto itself).
    add_node_name = _unique_graph_node_name(ir, original_node_name + '_split_ic_add')
    add_node = {
        'name': add_node_name,
        'op_type': OpType.ADD,
        'attributes': {},
        'inputs': [output_tensor_name_split1, output_tensor_name_split2],
        'outputs': [orig_output_tensor_name],
        'frontend': {}
    }
    add_node['frontend']['input_tensors'] = [output_tensor_split1, output_tensor_split2]
    add_node['frontend']['input_tensors_scale'] = [output_tensor_split1.scale, output_tensor_split2.scale]
    add_node['frontend']['input_tensors_zp'] = [output_tensor_split1.zero_point, output_tensor_split2.zero_point]
    add_node['frontend']['output_tensor'] = orig_output_tensor
    add_node['frontend']['output_tensor_scale'] = orig_output_tensor.scale
    add_node['frontend']['output_tensor_zp'] = orig_output_tensor.zero_point
    add_node['frontend']['output_channels'] = orig_output_tensor.get_original_shape()[1]
    # input_channels matches get_add_params: the channel count of an input (both Add inputs are the
    # half-conv outputs, which carry the full output channels). Sequencer/CBC read frontend
    # input_channels for every node, so it must be present on this synthesized Add.
    add_node['frontend']['input_channels'] = output_tensor_split1.get_original_shape()[1]
    add_node['frontend']['following_nodes_params'] = []
    add_node['frontend']['preceding_nodes_params'] = []
    add_node['frontend']['x_wrapping'] = 1

    # Give the Add a backend dict derived from the conv's (backend values are FOLDED counts; output
    # geometry is untouched by an input-channel split, so output_channels/z_tiles/output_padding_start_x
    # carry over verbatim). Add is MULTIPLE_INPUT_OPS: get_ops_grid_config would set input_channels to
    # 2 * output_channels since both inputs are the conv's full folded output - replicated here.
    if original_backend is not None:
        add_backend = copy.deepcopy(original_backend)
        add_backend['input_channels'] = 2 * add_backend['output_channels']
        add_node['backend'] = add_backend

    # The original output tensor is now produced by the Add and consumed by the same downstream nodes.
    orig_output_tensor.producer = add_node_name
    output_tensor_split1.consumers = [add_node_name]
    output_tensor_split2.consumers = [add_node_name]

    # Add the new nodes to the graph.
    ir.graph.add_node(dup_node_name, **dup_node)
    dup_node = ir.graph.nodes[dup_node_name]
    ir.graph.add_node(add_node_name, **add_node)
    add_node = ir.graph.nodes[add_node_name]

    # Rewire edges: producer -> {node, dup}; {node, dup} -> add; add -> original consumers.
    edges_from_original_node = list(ir.graph.out_edges(original_node_name))
    for edge in edges_from_original_node:
        ir.graph.add_edge(add_node_name, edge[1])
        ir.graph.remove_edge(original_node_name, edge[1])
    ir.graph.add_edge(original_node_name, add_node_name)
    ir.graph.add_edge(dup_node_name, add_node_name)
    if len(original_preceding_params) > 0:
        preceding_node_of_original_node_name = original_preceding_params[0][0]
        ir.graph.add_edge(preceding_node_of_original_node_name, dup_node_name)

    # Set following/preceding params manually: get_updated_* IR helpers assume input names equal
    # producers' output names, but this split renames inputs[0] to a channel slice, so they'd crash
    # or drop edges. Both halves keep the conv's preceding params and feed the Add; Add inherits consumers.
    node['frontend']['preceding_nodes_params'] = list(original_preceding_params)
    node['frontend']['following_nodes_params'] = [(add_node_name, 0)]
    dup_node['frontend']['preceding_nodes_params'] = list(original_preceding_params)
    dup_node['frontend']['following_nodes_params'] = [(add_node_name, 1)]
    add_node['frontend']['preceding_nodes_params'] = [(original_node_name, 0), (dup_node_name, 1)]
    add_node['frontend']['following_nodes_params'] = list(original_following_params)

    # The producer now also feeds the duplicate half (input index 0 = activation).
    if len(original_preceding_params) > 0:
        preceding_node = ir.graph.nodes[original_preceding_params[0][0]]
        if (dup_node_name, 0) not in preceding_node['frontend']['following_nodes_params']:
            preceding_node['frontend']['following_nodes_params'].append((dup_node_name, 0))

    # Downstream consumers are now preceded by the Add instead of the original conv.
    for (following_node_name, _) in original_following_params:
        following_node = ir.graph.nodes[following_node_name]
        following_node['frontend']['preceding_nodes_params'] = [
            (add_node_name if p[0] == original_node_name else p[0], p[1])
            for p in following_node['frontend']['preceding_nodes_params']
        ]

    # Folding/quant params were already computed before this split, so populate the Add's fields from
    # its folded conv inputs; sharing the conv's output scale/zp makes calc_add_qparams a plain sum.
    ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    calc_add_folding_factor(ir, add_node_name, add_node)
    calc_add_qparams(ir, add_node_name, add_node)

def _consumer_slice_bounds(ir: internal_representation.IR, output_tensor_name, output_channels):
    """If output_tensor_name already has downstream-consumer-fabricated read-channel-slice
    descendants (split_node_out_channels_in_n's generic non-conv input-slicing path, run on a
    consumer BEFORE this producer got independently split - e.g. force_split_oversized_wloc_adds
    splitting an Add ahead of its Conv producer's own AMM-overflow split), return the channel
    bounds those slices already impose - so the producer can be split at those SAME boundaries and
    have split_node_out_channels_in_n's output-tensor loop reuse/promote them - instead of creating
    independently-bounded chunks under colliding names. Returns None when there are no such slices,
    or when they don't contiguously and exactly tile [0, output_channels) (nothing to safely reuse).
    """
    descendants = ir.original_tensor_to_split_tensor_map.get(output_tensor_name, [])
    slices = [t for t in (ir.tensors.get(name) for name in descendants)
              if t is not None and getattr(t, 'is_read_channel_slice', False) and t.split_channel_lo is not None]
    if not slices:
        return None
    slices.sort(key=lambda t: t.split_channel_lo)
    bounds = [0]
    for t in slices:
        lo = t.split_channel_lo
        cs = t.shape[1]
        if lo != bounds[-1]:
            return None  # gap or overlap between slices - not a clean, reusable tiling
        bounds.append(lo + cs)
    if bounds[-1] != output_channels:
        return None  # slices don't cover the full output range
    return bounds


# One-shot N-way OUTPUT-channel split, generalizing the removed 2-way split to `num_splits` pieces in
# one call. Piece 0 stays on the original node; 1..N-1 are dup nodes named <name>_split1.._split{N-1}.
# All N output tensors are forced side-by-side in the original output's DDR region, so the downstream
# consumer still reads one contiguous tensor (no Concat needed). Conv pieces slice weights/biases/
# requant per output channel, keeping the full shared input; elementwise ops (Add/MaxPool/AveragePool/
# Resize) map output channel c from input channel c, so each piece also slices the matching input
# range. Concat/Gemm are excluded by the caller.
def split_node_out_channels_in_n(ir: internal_representation.IR, original_node_name, node, bounds):
    op_type = node['op_type']
    # Detect conv by op_type, NOT by weights_tensor presence: calc_add_qparams synthesizes a
    # weights_tensor on Add nodes, so a weights_tensor check would misclassify an Add as a conv.
    is_conv = (op_type == OpType.CONV)
    is_multi_input = op_type in MULTIPLE_INPUT_OPS  # Add: two input tensors that must both be sliced
    # MaxPool/Add carry a synthetic diagonal identity-like weights_tensor (calc_fake_conv_weights /
    # calc_add_qparams), consumed downstream, so it needs reslicing per output-channel chunk too - but
    # with OC==IC layout, so the OC axis and each input's IC block must share the SAME [lo:hi] bounds
    # to keep the diagonal identity intact.
    is_fake_identity_weights_op = op_type in (OpType.MAXPOOL, OpType.ADD)

    orig_output_channels = node['frontend']['output_channels']
    num_splits = len(bounds) - 1

    # Capture originals from `node` BEFORE mutating it in place for chunk 0 (dups deepcopy full copies).
    orig_weights_tensor = node['frontend'].get('weights_tensor') if (is_conv or is_fake_identity_weights_op) else None
    orig_biases_tensor = node['frontend'].get('biases_tensor') if is_conv else None
    has_qparams = 'output_multiplier' in node['frontend']
    orig_per_oc = {p: node['frontend'][p] for p in ('output_multiplier', 'cInputH', 'cInputL', 'o_shift')} \
        if (is_conv and has_qparams) else {}
    orig_folded_weights = node['frontend'].get('folded_weights_tensor') if (is_conv or is_fake_identity_weights_op) else None
    orig_folded_biases = node['frontend'].get('folded_biases_tensor') if is_conv else None

    orig_output_tensor_name = node['outputs'][0]
    orig_output_tensor = ir.tensors[orig_output_tensor_name]

    if not is_conv:
        if is_multi_input:
            orig_input_tensors = list(node['frontend']['input_tensors'])
            orig_input_names = list(node['inputs'][:len(orig_input_tensors)])
        else:
            orig_input_tensors = [node['frontend']['input_tensor']]
            orig_input_names = [node['inputs'][0]]

    # Build the N chunk nodes: chunk 0 == node (mutated in place), 1..N-1 dups. orig_*_tensor fields
    # get resliced into fresh per-chunk Tensors below, so avoid wastefully deepcopying them (can be
    # arbitrarily large with folding) - share by reference via the memo instead.
    _dup_node_memo = {}
    for _v in (orig_weights_tensor, orig_biases_tensor, orig_folded_weights, orig_folded_biases):
        if _v is not None and getattr(_v, 'is_constant', False):
            for _arr in (_v.data, getattr(_v, 'scale', None), getattr(_v, 'zero_point', None)):
                if isinstance(_arr, np.ndarray):
                    _dup_node_memo[id(_arr)] = _arr
    dup_nodes = []
    for i in range(1, num_splits):
        dup_node = copy.deepcopy(node, memo=dict(_dup_node_memo))
        # Fresh copies, not aliases: chunk 0 and this dup grow following_nodes_params independently
        # below, so sharing the list would leak one chunk's edit into the other's.
        dup_node['frontend']['following_nodes_params'] = list(node['frontend']['following_nodes_params'])
        dup_node['frontend']['preceding_nodes_params'] = list(node['frontend']['preceding_nodes_params'])
        dup_node['name'] = _unique_graph_node_name(ir, '%s_split%d' % (original_node_name, i))
        if is_conv:
            # CONV output-channel split: all chunks read the SAME input tensors as chunk 0.
            # Don't create aliases - just reuse the same input tensor references.
            # Mark this dup for later graph wiring (done after dup_node is added to ir.graph)
            dup_node['_needs_input_producer_graph_update'] = True
        dup_nodes.append(dup_node)
    all_nodes = [node] + dup_nodes

    orig_h_tiles = node['frontend']['h_tiles']
    orig_xslices = node['frontend']['xslices_per_htile']
    for i, n in enumerate(all_nodes):
        n['frontend']['output_channels'] = bounds[i + 1] - bounds[i]
        if is_multi_input:
            # Add: output channel c derives from input channel c, so input_channels must match the
            # output range too, or a retry re-splitting this chunk uses the stale full value and
            # cascades splits down to the 16-channel alignment floor.
            n['frontend']['input_channels'] = bounds[i + 1] - bounds[i]
        n['frontend']['h_tiles'] = orig_h_tiles
        n['frontend']['xslices_per_htile'] = orig_xslices
        # Marks this node so split_large_conv_nodes is never re-entered on it (chunk 0 reuses
        # original_node_name; re-splitting would nest a second split on top of this wiring).
        n['frontend']['was_channel_split'] = True

    # ---- Conv-specific: split weights / biases / requant per output channel ----
    if is_conv and orig_weights_tensor is not None:
        ow = orig_weights_tensor
        scalar_w = isinstance(ow.scale, (int, float)) or (len(ow.scale) == 1)
        for i, n in enumerate(all_nodes):
            lo, hi = bounds[i], bounds[i + 1]
            wname = '%s_split%d' % (ow.name, i)
            wdata = ow.data[lo:hi, :, :, :]
            ir.tensors[wname] = Tensor(
                wname, wdata, is_constant=True, shape=list(wdata.shape),
                scale=(ow.scale if scalar_w else ow.scale[lo:hi]),
                zero_point=(ow.zero_point if scalar_w else ow.zero_point[lo:hi]))
            n['frontend']['weights_tensor'] = ir.tensors[wname]
        if orig_biases_tensor is not None:
            ob = orig_biases_tensor
            scalar_b = isinstance(ob.scale, (int, float)) or (len(ob.scale) == 1)
            for i, n in enumerate(all_nodes):
                lo, hi = bounds[i], bounds[i + 1]
                bname = '%s_split%d' % (ob.name, i)
                bdata = ob.data[lo:hi]
                ir.tensors[bname] = Tensor(
                    bname, bdata, is_constant=True, shape=list(bdata.shape),
                    scale=(ob.scale if scalar_b else ob.scale[lo:hi]),
                    zero_point=(ob.zero_point if scalar_b else ob.zero_point[lo:hi]))
                n['frontend']['biases_tensor'] = ir.tensors[bname]
        for n in all_nodes:
            weights_tensor = n['frontend']['weights_tensor']
            dense_weights = weights_tensor.data.size
            sparse_weights = np.count_nonzero(weights_tensor.data)
            n['frontend']['dense_weights'] = dense_weights
            n['frontend']['sparse_weights'] = sparse_weights
            n['frontend']['weights_sparsity'] = 1 - (sparse_weights / dense_weights)
            current_op_input_tensor_shape = ir.tensors[n['inputs'][0]].get_original_shape()
            stride = n['frontend']['stride']
            dense_macs = dense_weights * current_op_input_tensor_shape[2] * current_op_input_tensor_shape[3] / (stride * stride)
            sparse_macs = sparse_weights * current_op_input_tensor_shape[2] * current_op_input_tensor_shape[3] / (stride * stride)
            n['frontend']['dense_macs'] = dense_macs
            n['frontend']['sparse_macs'] = sparse_macs
            n['frontend']['macs_sparsity'] = 1 - (sparse_macs / dense_macs)
            n['frontend']['weights_per_channel_scale'] = weights_tensor.scale
            n['frontend']['weights_per_channel_zp'] = weights_tensor.zero_point

    # ---- MaxPool/Add-specific: split fake identity weights (+ folded) per output channel ----
    # No biases_tensor/per-oc requant split needed: MaxPool/Add have no biases_tensor, and their
    # requant scalars are already copied as-is to every piece by the earlier dup-node deepcopy.
    if is_fake_identity_weights_op and orig_weights_tensor is not None:
        ow = orig_weights_tensor
        block_count = 2 if is_multi_input else 1
        scalar_w = isinstance(ow.scale, (int, float)) or (len(ow.scale) == 1)
        w_pieces = _slice_fake_identity_weights(ow.data, orig_output_channels, block_count, bounds)
        for i, n in enumerate(all_nodes):
            lo, hi = bounds[i], bounds[i + 1]
            wname = '%s_split%d' % (ow.name, i)
            wdata = w_pieces[i]
            ir.tensors[wname] = Tensor(
                wname, wdata, is_constant=True, shape=list(wdata.shape),
                scale=(ow.scale if scalar_w else ow.scale[lo:hi]),
                zero_point=(ow.zero_point if scalar_w else ow.zero_point[lo:hi]))
            n['frontend']['weights_tensor'] = ir.tensors[wname]
        if orig_folded_weights is not None:
            fw = orig_folded_weights
            fw_pieces = _slice_fake_identity_weights(fw.data, orig_output_channels, block_count, bounds)
            for i, n in enumerate(all_nodes):
                fwdata = fw_pieces[i].astype(fw.data.dtype)
                n['frontend']['folded_weights_tensor'] = Tensor(
                    '%s_split%d' % (fw.name, i), fwdata, is_constant=True, shape=list(fwdata.shape),
                    scale=n['frontend']['weights_tensor'].scale,
                    zero_point=n['frontend']['weights_tensor'].zero_point)
        for n in all_nodes:
            weights_tensor = n['frontend']['weights_tensor']
            dense_weights = weights_tensor.data.size
            sparse_weights = np.count_nonzero(weights_tensor.data)
            n['frontend']['dense_weights'] = dense_weights
            n['frontend']['sparse_weights'] = sparse_weights
            n['frontend']['weights_sparsity'] = 1 - (sparse_weights / dense_weights)
            n['frontend']['weights_per_channel_scale'] = weights_tensor.scale
            n['frontend']['weights_per_channel_zp'] = weights_tensor.zero_point

    # ---- Backend output/input channel counts per chunk ----
    if 'backend' in node:
        ofx = node['frontend']['output_folding_factor_x']
        ofy = node['frontend']['output_folding_factor_y']            
        for i, n in enumerate(all_nodes):
            cs = bounds[i + 1] - bounds[i]
            if not node['frontend'].get('force_folding_x',False):
                cs = cs * (2 ** (ofx + ofy))
            n['backend']['output_channels'] = cs
            n['backend']['oc_groups'] = [[oc for oc in range(cs)]]
        

    # ---- Conv-specific: per-output-channel requant params + folded tensors ----
    if is_conv and has_qparams:
        for param_name in ('output_multiplier', 'cInputH', 'cInputL', 'o_shift'):
            pieces = _split_per_oc_param_by_output_channel_n(orig_per_oc[param_name], orig_output_channels, bounds)
            for i, n in enumerate(all_nodes):
                n['frontend'][param_name] = pieces[i]
        for field in ('weights_stride_x', 'is_avgPool', 'folded_kernel_size', 'folded_padding', 'lut'):
            if field in node['frontend']:
                for n in dup_nodes:
                    n['frontend'][field] = copy.deepcopy(node['frontend'][field])
        if orig_folded_weights is not None:
            fw_pieces = _split_folded_tensor_by_output_channel_n(orig_folded_weights.data, orig_output_channels, bounds)
            for i, n in enumerate(all_nodes):
                fw = fw_pieces[i].astype(orig_folded_weights.data.dtype)
                n['frontend']['folded_weights_tensor'] = Tensor(
                    '%s_split%d' % (orig_folded_weights.name, i), fw, is_constant=True, shape=list(fw.shape),
                    scale=n['frontend']['weights_tensor'].scale,
                    zero_point=n['frontend']['weights_tensor'].zero_point)
        if orig_folded_biases is not None:
            fb_pieces = _split_per_oc_param_by_output_channel_n(orig_folded_biases.data, orig_output_channels, bounds)
            for i, n in enumerate(all_nodes):
                fb = np.asarray(fb_pieces[i], dtype=orig_folded_biases.data.dtype)
                n['frontend']['folded_biases_tensor'] = Tensor(
                    '%s_split%d' % (orig_folded_biases.name, i), fb, is_constant=True, shape=list(fb.shape),
                    scale=n['frontend']['biases_tensor'].scale,
                    zero_point=n['frontend']['biases_tensor'].zero_point)

    # ---- Generic (non-conv): slice each input tensor's channel range per chunk ----
    if not is_conv:
        for in_idx, orig_in in enumerate(orig_input_tensors):
            orig_in_name = orig_input_names[in_idx]
            # This input may ALREADY be output-channel-split by its own producer: orig_in_name is then a
            # stale pre-split name still carrying the ORIGINAL tensor's shape/producer, not a real
            # per-chunk region (nothing rewrites node['inputs'] when a producer splits). Fabricating new
            # '<name>_splitN' slices here would collide by name with the producer's genuine chunk output
            # tensors and orphan them (never get a ddr_entry -> crash). When the producer's chunk
            # count/bounds match this split exactly, reuse its real chunk tensors directly.
            producer_chunk_names = None
            if orig_in_name in ir.original_tensor_to_split_tensor_map:
                candidate = list(reversed(ir.original_tensor_to_split_tensor_map[orig_in_name]))
                if len(candidate) == num_splits and all(
                        ir.tensors[c].split_channel_lo == bounds[i] for i, c in enumerate(candidate)):
                    producer_chunk_names = candidate
            slice_names_rev = []
            for i, n in enumerate(all_nodes):
                lo, hi = bounds[i], bounds[i + 1]
                cs = hi - lo
                if producer_chunk_names is not None:
                    it_name = producer_chunk_names[i]
                    it = ir.tensors[it_name]
                    if n['name'] not in it.consumers:
                        it.consumers.append(n['name'])
                    n['inputs'][in_idx] = it_name
                    if is_multi_input:
                        n['frontend']['input_tensors'][in_idx] = it
                        if in_idx == 0:
                            n['frontend']['input_tensor'] = it
                    else:
                        n['frontend']['input_tensor'] = it
                    slice_names_rev.insert(0, it_name)
                    continue
                it = copy.deepcopy(orig_in)
                it_name = '%s_split%d' % (orig_in_name, i)
                if it_name in ir.tensors and ir.tensors[it_name] is not orig_in and it_name != orig_in_name:
                    # Name collision: '<orig_in_name>_split<i>' is already a real, differently-owned tensor
                    # (e.g. a genuine chunk from orig_in_name's own producer split). Reusing that name would
                    # overwrite it, orphaning it from every node that looks it up by name. Disambiguate with
                    # this splitting consumer's own name.
                    it_name = '%s_split%d_for_%s' % (orig_in_name, i, node['name'])
                if it.data is not None:
                    it.data = it.data[:, lo:hi, :, :]
                    it.shape = [orig_in.shape[0], cs, orig_in.shape[2], orig_in.shape[3]]
                    it.shape_real_x16[1] = cs
                else:
                    it.shape = [orig_in.shape[0], cs, orig_in.shape[2], orig_in.shape[3]]
                    it.shape_real_x16 = [orig_in.shape_real_x16[0], cs,
                                         orig_in.shape_real_x16[2], orig_in.shape_real_x16[3]]
                it.name = it_name
                it.consumers = [n['name']]
                # Deliberately do NOT remove n['name'] from orig_in.consumers: get_blob_inputs_and_outputs
                # decides blob-output status purely from orig_in.consumers, so dropping it would make
                # orig_in invisible there (no ddr_entry -> crash). get_dominant_tensor_consumer_in_blob
                # already filters stale consumers itself, so leaving it is harmless.
                # Channel start of this slice within the merged input, so allocate_ddr can alias it to the
                # producer's full-output region at byte offset lo*H*W (channel-major layout).
                it.split_channel_lo = lo
                # This is a channel-slice the chunk READS from the producer's full output, NOT an output
                # chunk - the multi-input contiguous allocator uses this flag to read cs channels at this
                # slice's DDR offset and size the AMM to cs. Output chunks stay unflagged so a downstream
                # concat still resolves them to the merged tensor.
                it.is_read_channel_slice = True
                # Eagerly alias this slice's ddr_entry only when orig_in already has one (i.e. this runs
                # from the AMM-failure retry loop, after allocate_ddr_for_blobs_output_nodes already ran) -
                # _alias_ddr_entries_to_split_descendants only walks the map at that pass's execution time,
                # so a registration added after the fact would otherwise leave ddr_entry None -> crash.
                # In the common proactive-split case (no ddr_entry yet), leave it unset; the later
                # blob-output pass will find and alias it via the map as before.
                if orig_in.ddr_entry is not None:
                    bytes_per_channel = int(orig_in.ddr_entry.shape[2]) * int(orig_in.ddr_entry.shape[3])
                    it.data = np.zeros(it.shape_real_x16, dtype=np.uint8)
                    slice_entry = TensorDDREntry(create_tsnp_tensor_byte_array(it.data),
                                                 type=orig_in.ddr_entry.type,
                                                 description='Channel slice of %s' % orig_in.name,
                                                 shape=it.shape_real_x16, amm_height=ir.amm_height)
                    slice_entry.address = orig_in.ddr_entry.address + lo * bytes_per_channel
                    it.ddr_entry = slice_entry
                ir.tensors[it_name] = it
                n['inputs'][in_idx] = it_name
                if is_multi_input:
                    n['frontend']['input_tensors'][in_idx] = it
                    if in_idx == 0:
                        n['frontend']['input_tensor'] = it
                else:
                    n['frontend']['input_tensor'] = it
                ir.split_tensor_to_original_tensor_map[it_name] = orig_in_name
                slice_names_rev.insert(0, it_name)
            # Merge, don't overwrite: orig_in_name may already have descendant entries here from an
            # earlier split of one of ITS OWN consumers (e.g. a conv input-channel split via
            # split_conv_in_channels_in_m registers '<name>_split_icN' entries keyed by this same
            # original name). An unconditional `=` here wiped those out, so mark_blob_outputs_
            # deallocations' lineage/sibling_tree walk from this tensor never reached the other
            # consumer's actual '_split_icN' input name -> ValueError('... is not in list').
            existing_in = ir.original_tensor_to_split_tensor_map.get(orig_in_name, [])
            merged_in = list(slice_names_rev)
            for name in existing_in:
                if name not in merged_in:
                    merged_in.append(name)
            ir.original_tensor_to_split_tensor_map[orig_in_name] = merged_in

    # ---- Output tensors: one per chunk, side-by-side in the original output DDR region ----
    to_mxp = orig_output_tensor_name in ir.tensors_to_mxp
    mxp_base = ir.mxp_tensor_to_offset[orig_output_tensor_name] if to_mxp else 0
    mxp_cumulative = 0
    split_out_names = []
    for i, n in enumerate(all_nodes):
        lo, hi = bounds[i], bounds[i + 1]
        cs = hi - lo
        ot_name = '%s_split%d' % (orig_output_tensor_name, i)
        # A downstream consumer may have already fabricated this exact channel-range slice as its own
        # input read-view before this producer's own split ran (_consumer_slice_bounds guarantees
        # `bounds` lines up with it). Promote that same tensor object into this chunk's real output
        # instead of creating a second, colliding tensor under the same name.
        existing = ir.tensors.get(ot_name)
        if (existing is not None and getattr(existing, 'is_read_channel_slice', False)
                and existing.split_channel_lo == lo and existing.shape[1] == cs):
            ot = existing
            ot.is_read_channel_slice = False  # now a real produced output chunk, not a read-view
        else:
            ot = copy.deepcopy(orig_output_tensor)
            if ot.data is not None:
                ot.data = ot.data[:, lo:hi, :, :]
                # Logical shape spatial dims from orig.shape, not the grid-padded data slice.
                ot.shape = [orig_output_tensor.shape[0], cs, orig_output_tensor.shape[2], orig_output_tensor.shape[3]]
                ot.shape_real_x16[1] = cs
            else:
                ot.data = None
                ot.shape = [orig_output_tensor.shape[0], cs, orig_output_tensor.shape[2], orig_output_tensor.shape[3]]
                ot.shape_real_x16 = [orig_output_tensor.shape_real_x16[0], cs,
                                     orig_output_tensor.shape_real_x16[2], orig_output_tensor.shape_real_x16[3]]
        ot.name = ot_name
        ot.producer = n['name']
        # Channel start within the merged output; allocate_ddr writes this chunk at byte offset lo*H*W of
        # the merged original's DDR region (channel-major layout). Generalizes the 2-way split2 offset to N.
        ot.split_channel_lo = lo
        ot.xslices_per_htile = orig_output_tensor.xslices_per_htile
        n['outputs'][0] = ot_name
        n['frontend']['output_tensor'] = ot
        ir.tensors[ot_name] = ot
        ir.split_tensor_to_original_tensor_map[ot_name] = orig_output_tensor_name
        split_out_names.append(ot_name)
        if to_mxp:
            ir.tensors_to_mxp.add(ot_name)
            ir.mxp_tensor_to_offset[ot_name] = mxp_base + mxp_cumulative
            mxp_cumulative += int(np.prod(ot.shape))
    # Reverse order matches move_outputs_write_to_amm_to_erliest_point's sequential earliest-point move.
    # Merge, don't overwrite: orig_output_tensor_name may already have descendant entries from an earlier
    # input-channel split of a downstream consumer ('_split_icN' names, keyed by this same original name).
    # An unconditional `=` would wipe those out, breaking mark_blob_outputs_deallocations' lineage walk.
    existing_out = ir.original_tensor_to_split_tensor_map.get(orig_output_tensor_name, [])
    merged_out = list(reversed(split_out_names))
    for name in existing_out:
        if name not in merged_out:
            merged_out.append(name)
    ir.original_tensor_to_split_tensor_map[orig_output_tensor_name] = merged_out

    # ---- Graph wiring for the dup chunks ----
    for dup_node in dup_nodes:
        dup_node_name = dup_node['name']
        ir.graph.add_node(dup_node_name, **dup_node)
        for edge in list(ir.graph.out_edges(original_node_name)):
            consumer_name = edge[1]
            ir.graph.add_edge(dup_node_name, consumer_name)
            # chunk 0 and this dup chunk TOGETHER reconstitute the full pre-split output, so a downstream
            # consumer's preceding_nodes_params entry for original_node_name must gain a sibling entry for
            # this dup at the same input slot - otherwise a dual-alloc resolution walk only ever sees
            # chunk 0's output, even when the consumer's cached input_tensors reference this chunk's tensor.
            consumer_preceding = ir.graph.nodes[consumer_name]['frontend'].get('preceding_nodes_params', [])
            for prev_name, idx in list(consumer_preceding):
                if prev_name == original_node_name and (dup_node_name, idx) not in consumer_preceding:
                    consumer_preceding.append((dup_node_name, idx))
        for (prev_name, _) in node['frontend']['preceding_nodes_params']:
            ir.graph.add_edge(prev_name, dup_node_name)
            prev_node = ir.graph.nodes[prev_name]
            if (dup_node_name, 0) not in prev_node['frontend']['following_nodes_params']:
                prev_node['frontend']['following_nodes_params'].append((dup_node_name, 0))
        # For CONV output-channel splits: dup_node reads the SAME input tensors as the original node,
        # so also add it to the producers' following_nodes_params to ensure it stays registered
        if is_conv and dup_node.get('_needs_input_producer_graph_update'):
            for input_idx, input_name in enumerate(dup_node.get('inputs', [])):
                if input_name in ir.tensors:
                    actual_input_tensor = ir.tensors[input_name]
                    producer_name = actual_input_tensor.producer
                    if producer_name and producer_name in ir.graph.nodes:
                        producer_node = ir.graph.nodes[producer_name]
                        if (dup_node_name, input_idx) not in producer_node['frontend']['following_nodes_params']:
                            producer_node['frontend']['following_nodes_params'].append((dup_node_name, input_idx))

    ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))

    # CONV output-channel split dups read the same input tensors as the original node, so register each
    # dup on the consumers of the tensor it actually reads. Do NOT rebuild the producer's output_tensor
    # consumers from following_nodes_params: the producer's current output tensor can be a different
    # tensor (e.g. an input-channel-split partial sum) than the one the dup reads.
    if is_conv:
        for dup_node in dup_nodes:
            if dup_node.get('_needs_input_producer_graph_update'):
                for input_name in dup_node.get('inputs', [])[:1]:
                    actual_input_tensor = ir.tensors.get(input_name)
                    if actual_input_tensor is not None and dup_node['name'] not in actual_input_tensor.consumers:
                        actual_input_tensor.consumers.append(dup_node['name'])

    # Add's synthesized [OC, IC*2, 1, 1] weights depend on the chunk's OC/IC, so recompute folding/qparams
    # per chunk now that tensors are wired. Each chunk is isolated into its own blob so it writes its
    # output slice to DDR at the right channel offset instead of sharing a blob with a downstream concat.
    if is_multi_input:
        for n in all_nodes:
            g = ir.graph.nodes[n['name']]
            g['frontend']['force_isolate_blob'] = True
            calc_add_folding_factor(ir, n['name'], g)
            calc_add_qparams(ir, n['name'], g)


# One-shot M-way conv INPUT-channel split + Add tree (generalizes the old 2-way split+single-Add) into
# `num_splits` slices recombined by a balanced Add tree (num_splits-1 Adds, depth ceil(log2(M))):
#     conv(x) = sum_j conv(x[:, sj:ej], W[:, sj:ej], bias_j), bias kept on slice 0 only.
# All partials/Add tensors share the conv's output scale/zp, so calc_add_qparams is a scale-matched
# integer sum - functionally exact but not bit-exact (each partial requantized to int8 before summing);
# the balanced tree minimizes requant depth/accumulated saturation.
def split_conv_in_channels_in_m(ir: internal_representation.IR, original_node_name, node, ibounds):
    original_preceding_params = list(node['frontend']['preceding_nodes_params'])
    original_following_params = list(node['frontend']['following_nodes_params'])
    original_backend = copy.deepcopy(node['backend']) if 'backend' in node else None
    fused_act = node['attributes'].get('fused_activation_function') if isinstance(node.get('attributes'), dict) else None
    # A non-linear activation LUT (frontend['lut']) is only correct on the FULL conv output, and an
    # input-channel split computes it as a sum of partials, so the LUT must apply ONCE at the root
    # recombine Add - sum(SiLU(partial)) != SiLU(sum). Capture it here, strip from every slice, re-attach
    # to the root Add.
    # The conv output_tensor scale is the LUT-INPUT (pre-activation) scale: slices/intermediate Adds sum
    # linearly at that scale, the root Add requants to it, then its LUT maps to the activation OUTPUT
    # scale. Conv consumers use their own independently-cached input_tensor_scale (already post-activation)
    # so they're unaffected - confirmed empirically (mutating the shared tensor's scale/zp regressed
    # already-correct nodes; do not reintroduce it).
    orig_lut = node['frontend'].get('lut')
    _act_silu = node['attributes'].get('activation_silu') if isinstance(node.get('attributes'), dict) else None

    orig_aligned_input_ch = node['frontend'].get('aligned_input_ch')  # parent's padded-block maps, sliced per piece below
    orig_input_channels = node['frontend']['input_channels']
    num_splits = len(ibounds) - 1

    orig_weights_tensor = node['frontend'].get('weights_tensor')
    orig_biases_tensor = node['frontend'].get('biases_tensor')
    # Folded weights/biases (present when input folding is active) must be sliced from the already-
    # folded tensor rather than independently re-folded from the (smaller) unfolded slices below - see
    # _slice_folded_weights_by_input_channel for why.
    input_folding_factor_x = node['frontend'].get('input_folding_factor_x', 0)
    input_folding_factor_y = node['frontend'].get('input_folding_factor_y', 0)
    is_input_folded = input_folding_factor_x > 0 or input_folding_factor_y > 0
    orig_folded_weights_tensor = node['frontend'].get('folded_weights_tensor') if is_input_folded else None
    orig_folded_biases_tensor = node['frontend'].get('folded_biases_tensor') if is_input_folded else None
    orig_input_tensor = node['frontend']['input_tensor']
    orig_input_tensor_name = orig_input_tensor.name
    orig_output_tensor_name = node['outputs'][0]
    orig_output_tensor = ir.tensors[orig_output_tensor_name]
    fold_ratio = (original_backend['input_channels'] // orig_input_channels) if original_backend is not None else 1

    # Capture the original conv's downstream out-edges NOW, before any tree edge (including the
    # original -> leaf-Add edge) is added. Rewiring live out_edges later would also catch that new
    # original -> leaf-Add edge and re-point it to the root Add, forming a root -> ... -> root cycle.
    original_out_edges = list(ir.graph.out_edges(original_node_name))

    orig_h_tiles = node['frontend']['h_tiles']
    orig_xslices = node['frontend']['xslices_per_htile']

    # Slice 0 == node (mutated in place); slices 1..M-1 == fresh dups.
    dup_nodes = []
    _dn_memo = _memo_for_node(node)
    for j in range(1, num_splits):
        dn = copy.deepcopy(node, memo=dict(_dn_memo))
        dn['frontend']['preceding_nodes_params'] = list(original_preceding_params)
        dn['name'] = _unique_graph_node_name(ir, '%s_split_ic%d' % (original_node_name, j))
        dup_nodes.append(dn)
    conv_slices = [node] + dup_nodes

    for j, n in enumerate(conv_slices):
        lo, hi = ibounds[j], ibounds[j + 1]
        cs = hi - lo
        n['frontend']['input_channels'] = cs
        # Each piece reads only its own channel slice, so the padded-block map is recomputed from that slice
        # (never inherited from the parent). A slice with no padded input has no map.
        # Take the parent's block map for this piece's channel range [lo, hi), keeping each channel's original AMM
        # block and renumbering channels from 0 for the slice. Pieces that cover no padded input get no map.
        if orig_aligned_input_ch is not None:
            _piece_map = {ch - lo: blk for ch, blk in orig_aligned_input_ch.items() if lo <= ch < hi}
            if _piece_map:
                n['frontend']['aligned_input_ch'] = _piece_map
            else:
                n['frontend'].pop('aligned_input_ch', None)
        else:
            n['frontend'].pop('aligned_input_ch', None)
        n['frontend']['h_tiles'] = orig_h_tiles
        n['frontend']['xslices_per_htile'] = orig_xslices
        # Marks a node as one slice of an input-channel-split conv (recombined downstream by a tree of
        # Adds) - lets other code (e.g. reports, later split passes) tell it apart from a plain conv.
        n['frontend']['is_input_channel_split'] = True
        # Mark the producer of this conv's input so the producer knows its consumer is input-channel split
        # (checked when the producer's h-tiling is grown, before any split is visible on the consumer).
        if orig_input_tensor.producer in ir.graph.nodes:
            ir.graph.nodes[orig_input_tensor.producer]['frontend']['consumed_by_input_channel_split'] = True
        # split_large_conv_nodes must never be re-entered on this node name: slice 0 reuses
        # original_node_name and keeps its OWN outputs[0] pointed at this slice's '_split_ic0' tensor
        # (only the root Add's output gets redirected back to the true original tensor, below) - a second
        # split call here would build a nested Add tree off that slice tensor instead of the true original
        # output, silently repointing a tensor the FIRST tree's root Add already holds a live reference to.
        n['frontend']['was_channel_split'] = True
        if orig_weights_tensor is not None:
            wname = '%s_split_ic%d' % (orig_weights_tensor.name, j)
            wdata = orig_weights_tensor.data[:, lo:hi, :, :]
            # Weight scale/zp are per OUTPUT channel (axis 0), unchanged by an input-channel split.
            ir.tensors[wname] = Tensor(
                wname, wdata, is_constant=True, shape=list(wdata.shape),
                scale=orig_weights_tensor.scale, zero_point=orig_weights_tensor.zero_point)
            n['frontend']['weights_tensor'] = ir.tensors[wname]
            n['inputs'][1] = wname
        if orig_folded_weights_tensor is not None:
            fwname = '%s_split_ic%d' % (orig_folded_weights_tensor.name, j)
            fwdata = _slice_folded_weights_by_input_channel(orig_folded_weights_tensor.data, orig_input_channels, lo, hi)
            ir.tensors[fwname] = Tensor(
                fwname, fwdata, is_constant=True, shape=list(fwdata.shape),
                scale=orig_folded_weights_tensor.scale, zero_point=orig_folded_weights_tensor.zero_point)
            n['frontend']['folded_weights_tensor'] = ir.tensors[fwname]
        if orig_biases_tensor is not None:
            if j == 0:
                n['frontend']['biases_tensor'] = orig_biases_tensor
            else:
                # Zeroed bias so the bias term is added exactly once when the slices are summed.
                zname = '%s_split_ic%d_zero' % (orig_biases_tensor.name, j)
                zdata = np.zeros_like(orig_biases_tensor.data)
                ir.tensors[zname] = Tensor(
                    zname, zdata, is_constant=True, shape=list(zdata.shape),
                    scale=orig_biases_tensor.scale, zero_point=orig_biases_tensor.zero_point)
                n['frontend']['biases_tensor'] = ir.tensors[zname]
                n['inputs'][2] = zname
        if orig_folded_biases_tensor is not None:
            # Biases are per OUTPUT channel, unaffected by an input-channel split: slice 0 keeps the
            # original folded biases, every other slice gets a zeroed folded biases tensor.
            if j == 0:
                n['frontend']['folded_biases_tensor'] = orig_folded_biases_tensor
            else:
                fzname = '%s_split_ic%d_zero' % (orig_folded_biases_tensor.name, j)
                fzdata = np.zeros_like(orig_folded_biases_tensor.data)
                ir.tensors[fzname] = Tensor(
                    fzname, fzdata, is_constant=True, shape=list(fzdata.shape),
                    scale=orig_folded_biases_tensor.scale, zero_point=orig_folded_biases_tensor.zero_point)
                n['frontend']['folded_biases_tensor'] = ir.tensors[fzname]

    # Split the input tensor along channels; each slice reads its own contiguous channel range.
    input_slice_names_rev = []
    for j, n in enumerate(conv_slices):
        lo, hi = ibounds[j], ibounds[j + 1]
        cs = hi - lo
        it = copy.deepcopy(orig_input_tensor)
        # Disambiguated by the CONSUMING node, not just the shared input tensor name: two different conv
        # nodes independently input-channel-splitting the SAME producer tensor previously reused identical
        # '<tensor>_split_ic<j>' names, letting the second split silently overwrite the first's slice
        # tensor under that name - later name-based lookups (e.g. the numeric sim) then resolved to the
        # wrong split, causing a channel-count mismatch (seen on onnx_resnet101-v1: CONV2D_128/CONV2D_131
        # both wrote '..._split_ic0'). '_split_ic<digits>' must stay at the very end of the name -
        # neuronix_ops.py's nx_conv parses it with a trailing regex (r'_split_ic(\d+)$').
        it_name = '%s_for_%s_split_ic%d' % (orig_input_tensor_name, original_node_name, j)
        # it.data may already be padded to shape_real_x16 (DMA width alignment); slicing it and then
        # taking .shape from the (possibly-padded) array leaks that padding into the true-shape metadata.
        # Derive shape from orig_input_tensor.shape (the true, unpadded shape) instead.
        if it.data is not None:
            it.data = it.data[:, lo:hi, :, :]
            it.shape = [orig_input_tensor.shape[0], cs, orig_input_tensor.shape[2], orig_input_tensor.shape[3]]
            it.shape_real_x16[1] = cs
        else:
            it.shape = [orig_input_tensor.shape[0], cs, orig_input_tensor.shape[2], orig_input_tensor.shape[3]]
            it.shape_real_x16 = [orig_input_tensor.shape_real_x16[0], cs,
                                 orig_input_tensor.shape_real_x16[2], orig_input_tensor.shape_real_x16[3]]
        it.name = it_name
        it.consumers = [n['name']]
        # Channel start of this input-channel slice; allocate_ddr aliases the slice's DDR entry to the
        # producer's full-output region at byte offset lo*H*W (channel-major). Replaces the 2-way _split_ic2
        # tail offset so M>2 input splits (e.g. CONV2D_96 ic1/ic2/ic3) read the correct channel ranges.
        it.split_channel_lo = lo
        ir.tensors[it_name] = it
        n['frontend']['input_tensor'] = it
        n['inputs'][0] = it_name
        n['frontend']['input_tensor_scale'] = orig_input_tensor.scale
        n['frontend']['input_tensor_zp'] = orig_input_tensor.zero_point
        # Unfolded channel offset of this slice within the full input, so the numeric simulator can pick
        # this slice's (possibly-scattered, under folding) channels out of the full producer output it
        # loads via split_tensor_to_original_tensor_map. Count is frontend['input_channels'] (= cs).
        n['frontend']['input_channel_slice_offset'] = lo
        if 'backend' in n:
            # backend['input_channels'] must equal the sliced input tensor's FOLDED channel count exactly
            # as read_missing_tensors_from_ddr recomputes it (via get_folded_shape with this node's
            # force-folding flags). A cs*fold_ratio shortcut is wrong when force_folding_y adds a fold
            # step (get_folded_shape then bumps folding_factor_y), so derive it from the slice tensor.
            ffy = 'force_folding_y' in n['frontend']
            ufy = 'force_unfolding_y' in n['frontend']
            n['backend']['input_channels'] = it.get_folded_shape(folding_conv_y=ffy, unfolding_conv_y=ufy)[1]
        ir.split_tensor_to_original_tensor_map[it_name] = orig_input_tensor_name
        input_slice_names_rev.insert(0, it_name)
    # Merge, don't overwrite: orig_input_tensor_name may already have entries from an earlier
    # output-channel split (alias_it names for dup nodes). An unconditional `=` would wipe those out,
    # breaking _alias_ddr_entries_to_split_descendants' BFS -> ddr_entry stays None -> sequencer crash.
    existing = ir.original_tensor_to_split_tensor_map.get(orig_input_tensor_name, [])
    for name in input_slice_names_rev:
        if name not in existing:
            existing.append(name)
    ir.original_tensor_to_split_tensor_map[orig_input_tensor_name] = existing
    for n in dup_nodes:
        if n['name'] not in orig_input_tensor.consumers:
            orig_input_tensor.consumers.append(n['name'])

    # Sparsity/macs per slice (output channels/spatial unchanged; only input-channel count differs).
    for n in conv_slices:
        weights_tensor = n['frontend']['weights_tensor']
        dense_weights = weights_tensor.data.size
        sparse_weights = np.count_nonzero(weights_tensor.data)
        n['frontend']['dense_weights'] = dense_weights
        n['frontend']['sparse_weights'] = sparse_weights
        n['frontend']['weights_sparsity'] = 1 - (sparse_weights / dense_weights)
        stride = n['frontend']['stride']
        input_tensor_shape = n['frontend']['input_tensor'].get_original_shape()
        dense_macs = dense_weights * input_tensor_shape[2] * input_tensor_shape[3] / (stride * stride)
        sparse_macs = sparse_weights * input_tensor_shape[2] * input_tensor_shape[3] / (stride * stride)
        n['frontend']['dense_macs'] = dense_macs
        n['frontend']['sparse_macs'] = sparse_macs
        n['frontend']['macs_sparsity'] = 1 - (sparse_macs / dense_macs)
        n['frontend']['weights_per_channel_scale'] = weights_tensor.scale
        n['frontend']['weights_per_channel_zp'] = weights_tensor.zero_point

    # Each slice gets its own FULL-output-channel output tensor at the original output scale/zp; the Add
    # tree sums them. NOT registered in the split maps (they are summed, not contiguous slices).
    conv_out_tensors = []
    for j, n in enumerate(conv_slices):
        ot = copy.deepcopy(orig_output_tensor)
        ot_name = '%s_split_ic%d' % (orig_output_tensor_name, j)
        ot.name = ot_name
        ot.producer = n['name']
        ot.consumers = []
        # orig_output_tensor may carry a stale is_read_channel_slice/split_channel_lo from an earlier
        # pass that treated it as a downstream consumer's channel-slice read-view (see the analogous
        # reset in split_node_out_channels_in_n). This tensor is a genuinely full-channel, freshly
        # produced conv output, not a read-view - an unreset flag makes nx_add wrongly truncate it.
        ot.is_read_channel_slice = False
        n['outputs'][0] = ot_name
        n['frontend']['output_tensor'] = ot
        ir.tensors[ot_name] = ot
        conv_out_tensors.append(ot)

    # All M slices feed the same Add tree, so they share one K (max of each slice's own statistical
    # bound) rather than each picking its own - see _compute_split_requant_K's docstring.
    shared_K = max(_compute_split_requant_K(n) for n in conv_slices)
    for n in conv_slices:
        # Strip the non-linear activation LUT from each slice: a slice produces only a partial (pre-
        # activation) sum, so it must stay LINEAR. The LUT is re-attached to the root Add below. Also drop
        # the 'activation_silu' attribute so nothing re-populates frontend['lut'] on the slice later.
        n['frontend'].pop('lut', None)
        if isinstance(n.get('attributes'), dict):
            n['attributes'].pop('activation_silu', None)
        _recompute_input_split_conv_requant(n, fused_act, K=shared_K)

    # ---- Build a balanced Add tree over the M half-conv outputs (M-1 Adds). ----
    add_records = []   # (add_dict, add_name, producer0_name, producer1_name)
    feeds_into = {}    # producer_name -> (add_name, input_index)
    level = [(conv_out_tensors[j], conv_slices[j]['name']) for j in range(num_splits)]
    add_counter = 0
    while len(level) > 1:
        next_level = []
        i = 0
        while i < len(level):
            if i + 1 < len(level):
                (t0, p0), (t1, p1) = level[i], level[i + 1]
                add_counter += 1
                add_name = _unique_graph_node_name(ir, '%s_split_ic_add%d' % (original_node_name, add_counter))
                out_tensor = copy.deepcopy(orig_output_tensor)
                out_name = '%s_split_ic_addout%d' % (original_node_name, add_counter)
                out_tensor.name = out_name
                out_tensor.producer = add_name
                out_tensor.consumers = []
                # See conv_out_tensors above: reset the stale read-view marker so nx_add doesn't
                # truncate this genuinely full-channel intermediate Add output.
                out_tensor.is_read_channel_slice = False
                ir.tensors[out_name] = out_tensor
                add_node = {
                    'name': add_name,
                    'op_type': OpType.ADD,
                    'attributes': {},
                    'inputs': [t0.name, t1.name],
                    'outputs': [out_name],
                    'frontend': {},
                }
                add_node['frontend']['input_tensors'] = [t0, t1]
                add_node['frontend']['input_tensors_scale'] = [t0.scale, t1.scale]
                add_node['frontend']['input_tensors_zp'] = [t0.zero_point, t1.zero_point]
                add_node['frontend']['output_tensor'] = out_tensor
                add_node['frontend']['output_tensor_scale'] = out_tensor.scale
                add_node['frontend']['output_tensor_zp'] = out_tensor.zero_point
                add_node['frontend']['output_channels'] = out_tensor.get_original_shape()[1]
                add_node['frontend']['input_channels'] = t0.get_original_shape()[1]
                add_node['frontend']['following_nodes_params'] = []
                add_node['frontend']['preceding_nodes_params'] = []
                add_node['frontend']['x_wrapping'] = 1
                if original_backend is not None:
                    add_backend = copy.deepcopy(original_backend)
                    add_backend['input_channels'] = 2 * add_backend['output_channels']
                    add_node['backend'] = add_backend
                t0.consumers = [add_name]
                t1.consumers = [add_name]
                feeds_into[p0] = (add_name, 0)
                feeds_into[p1] = (add_name, 1)
                add_records.append((add_node, add_name, p0, p1))
                next_level.append((out_tensor, add_name))
                i += 2
            else:
                next_level.append(level[i])
                i += 1
        level = next_level

    root_out_tensor, root_add_name = level[0]
    root_add = next(rec[0] for rec in add_records if rec[1] == root_add_name)
    # Redirect the root Add to produce the ORIGINAL output tensor (drop its unused intermediate tensor).
    if root_out_tensor.name in ir.tensors:
        del ir.tensors[root_out_tensor.name]
    root_add['outputs'][0] = orig_output_tensor_name
    root_add['frontend']['output_tensor'] = orig_output_tensor
    root_add['frontend']['output_tensor_scale'] = orig_output_tensor.scale
    root_add['frontend']['output_tensor_zp'] = orig_output_tensor.zero_point
    orig_output_tensor.producer = root_add_name
    # Re-attach the conv's non-linear activation LUT to the ROOT recombine Add only: intermediate Adds sum
    # linearly (no LUT), and the root Add applies it once after reconstructing the full pre-activation
    # output, matching the original fused conv+activation (both HW codegen and numeric sim key off
    # frontend['lut']).
    if orig_lut is not None:
        root_add['frontend']['lut'] = orig_lut

    # ---- Add all new nodes, then wire graph edges + following/preceding params. ----
    for dn in dup_nodes:
        ir.graph.add_node(dn['name'], **dn)
    for (add_node, add_name, p0, p1) in add_records:
        ir.graph.add_node(add_name, **add_node)

    # Root Add -> the original conv's downstream consumers (rewire the ORIGINAL out-edges captured at
    # entry). Do this BEFORE adding tree edges so the freshly added original -> leaf-Add edge is not
    # itself rewired (which would create a cycle).
    for edge in original_out_edges:
        ir.graph.add_edge(root_add_name, edge[1])
        ir.graph.remove_edge(original_node_name, edge[1])
    # Producer -> dup slices; also register the dup on the preceding node's following list.
    for dn in dup_nodes:
        for (prev_name, _) in original_preceding_params:
            ir.graph.add_edge(prev_name, dn['name'])
            prev_node = ir.graph.nodes[prev_name]
            if (dn['name'], 0) not in prev_node['frontend']['following_nodes_params']:
                prev_node['frontend']['following_nodes_params'].append((dn['name'], 0))
    # Slice/Add producers -> their consuming Add (adds original -> leaf-Add, dup -> Add, Add -> Add).
    for (add_node, add_name, p0, p1) in add_records:
        ir.graph.add_edge(p0, add_name)
        ir.graph.add_edge(p1, add_name)

    # following/preceding params. Each conv slice + each non-root Add feeds exactly one Add; the root Add
    # inherits the original conv's consumers.
    for j, n in enumerate(conv_slices):
        gname = n['name']
        g = ir.graph.nodes[gname]
        g['frontend']['preceding_nodes_params'] = list(original_preceding_params)
        g['frontend']['following_nodes_params'] = [feeds_into[gname]]
    for (add_node, add_name, p0, p1) in add_records:
        g = ir.graph.nodes[add_name]
        g['frontend']['preceding_nodes_params'] = [(p0, 0), (p1, 1)]
        if add_name in feeds_into:
            g['frontend']['following_nodes_params'] = [feeds_into[add_name]]
        else:
            g['frontend']['following_nodes_params'] = list(original_following_params)

    # Downstream consumers are now preceded by the root Add instead of the original conv.
    for (following_node_name, _) in original_following_params:
        following_node = ir.graph.nodes[following_node_name]
        following_node['frontend']['preceding_nodes_params'] = [
            (root_add_name if p[0] == original_node_name else p[0], p[1])
            for p in following_node['frontend']['preceding_nodes_params']
        ]

    ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    for (add_node, add_name, p0, p1) in add_records:
        g = ir.graph.nodes[add_name]
        calc_add_folding_factor(ir, add_name, g)
        calc_add_qparams(ir, add_name, g)
    # Re-attach the activation LUT to the root recombine Add AFTER calc_add_qparams (which recomputes the
    # Add's requant params against the LUT-input scale and would otherwise not preserve it).
    if orig_lut is not None:
        _rg = ir.graph.nodes[root_add_name]['frontend']
        _rg['lut'] = orig_lut
        # Also copy the 'activation_silu' ATTRIBUTE onto the root Add: the numeric sim's
        # _get_output_pad_value keys off node['attributes']['activation_silu']['output_zp'] (not
        # frontend['lut']) to pick the POST-activation zero-point for edge/width padding. Without this the
        # Add's padding used the pre-activation zp while its data is post-activation, corrupting any
        # padded region a downstream conv (kernel_size>1, same-padding) reads.
        root_add_node = ir.graph.nodes[root_add_name]
        if not isinstance(root_add_node.get('attributes'), dict):
            root_add_node['attributes'] = {}
        root_add_node['attributes']['activation_silu'] = _act_silu
        # Do NOT re-declare the output tensor's scale/zp to the activation OUTPUT scale here - conv
        # followers read their own independently-cached input_tensor_scale (already post-activation), and
        # mutating the shared tensor regressed already-correct nodes (e.g. CONV2D_96); confirmed CONCAT has
        # the same independent-cache mechanism, so there is no concat-scale bug here either.
        # NOTE: recombine Adds get the clean-AMM discipline stamped uniformly by _stamp_split_markers in
        # split_large_conv_nodes, the only caller of this function.


# Split a Conv (or other op) that an AMM-failing blob is stuck on. Called from the late blob-failure
# path: the blob has ALREADY failed to fit, so a split must be performed to make progress. Makes the
# WHOLE splitting decision in one call (no repeated two-way halving): output-split into N pieces when the
# double-buffered input fits, else input-split into M pieces + an Add tree when the output fits, else
# raise. Returns True if a split was performed, False if the node cannot / should not be split here.
def scatter_concat_inputs(ir: internal_representation.IR, concat_node):
    # Turns an oversized channel-axis CONCAT into a no-op: each input's producer writes directly into the
    # concat OUTPUT's DDR region at its channel offset (via split_tensor_to_original_tensor_map + the
    # existing output-split DDR aliasing), since a concat needs no compute - only AMM co-residency of its
    # inputs, which is what overflows for a wide concat. Returns True if scattered, False if ineligible.
    frontend = concat_node['frontend']
    if frontend.get('is_scatter_concat'):
        return False
    output_tensor = frontend['output_tensor']
    input_tensors = frontend['input_tensors']
    out_shape = output_tensor.shape_real_x16
    # Eligibility: a pure channel-axis concat (every input shares output H,W) whose cumulative channel
    # boundaries are all grid-aligned (multiples of MAX_GRID_WIDTH) - a non-aligned boundary would desync
    # the per-input write address (lo*H*W bytes) from the consumer's grid-strided read of the merged tensor.
    # Extra-line padding is per producer: either every input of this concat writes the extra line, or none does.
    # A mix cannot be scattered (the concat then starts its own blob, with more than one node).
    extra_line_flags = []
    for it in input_tensors:
        prod = it.producer
        extra_line_flags.append(bool(prod in ir.graph.nodes and ir.graph.nodes[prod]['frontend'].get('k3_1T_boundary_extra_line_write')))
    if any(extra_line_flags) and not all(extra_line_flags):
        return False
    offset = 0
    offsets = []
    for it in input_tensors:
        if it.shape_real_x16[2] != out_shape[2] or it.shape_real_x16[3] != out_shape[3]:
            return False
        if offset % MAX_GRID_WIDTH != 0:
            return False
        # If this same input tensor is already being scattered into a DIFFERENT concat's output, its
        # single physical write can't land in two different regions - bail out and keep this concat's
        # normal path instead of clobbering the other concat's registered alias for the tensor.
        existing_target = ir.split_tensor_to_original_tensor_map.get(it.name)
        if existing_target is not None and existing_target != output_tensor.name:
            return False
        offsets.append(offset)
        offset += it.shape_real_x16[1]
    if offset != out_shape[1]:
        return False
    slice_names = []
    for it, off in zip(input_tensors, offsets):
        it.split_channel_lo = off
        ir.split_tensor_to_original_tensor_map[it.name] = output_tensor.name
        slice_names.append(it.name)
    # Merge, don't overwrite - see the identical fix/comment at the other split map registration
    # sites (split_conv_in_channels_in_m etc): an unconditional `=` here can silently drop an
    # already-registered split-descendant name if output_tensor.name was split by an earlier pass.
    existing = ir.original_tensor_to_split_tensor_map.get(output_tensor.name, [])
    for name in slice_names:
        if name not in existing:
            existing.append(name)
    ir.original_tensor_to_split_tensor_map[output_tensor.name] = existing
    frontend['is_scatter_concat'] = True
    return True

def concat_amm_block_map(input_channels):
    """Build the AMM block map of a concat's OUTPUT, across all of its input tensors.

    AMM is allocated in blocks of AMM_CHANNELS_PER_BLOCK (16) channels. Each concat input is laid out in its own
    ceil(channels / 16) blocks, so a partly filled block belongs to that input alone. The output channel of each
    input's channel is that input's running offset plus its own channel index, and its block is the block the
    input starts at plus the channel's block within that input.

    input_channels: list of per-input channel counts, in concat input order.
    returns: None if every count is a multiple of 16, else a dict {output_channel: amm_block} covering all
             concat output channels (the sum of the input counts).
    """
    if all(c % AMM_CHANNELS_PER_BLOCK == 0 for c in input_channels):
        return None

    out_map = {}
    out_ch = 0
    block_offset = 0
    for c in input_channels:
        for ch in range(c):
            out_map[out_ch] = block_offset*AMM_CHANNELS_PER_BLOCK + ch
            out_ch += 1
        block_offset += math.ceil(c / AMM_CHANNELS_PER_BLOCK)
    return out_map

def _slice_resize_tensor(ir, orig_tensor, new_name, lo, hi):
    # Channel-slice [lo:hi] of a resize tensor (spatial dims unchanged - resize is per-channel).
    cs = hi - lo
    t = copy.deepcopy(orig_tensor)
    if t.data is not None:
        t.data = t.data[:, lo:hi, :, :]
        t.shape = [orig_tensor.shape[0], cs, orig_tensor.shape[2], orig_tensor.shape[3]]
        t.shape_real_x16[1] = cs
    else:
        t.shape = [orig_tensor.shape[0], cs, orig_tensor.shape[2], orig_tensor.shape[3]]
        t.shape_real_x16 = [orig_tensor.shape_real_x16[0], cs,
                            orig_tensor.shape_real_x16[2], orig_tensor.shape_real_x16[3]]
    t.name = new_name
    ir.tensors[new_name] = t
    return t

def split_resize_pair(ir: internal_representation.IR, rx_node, num_splits=2):
    # Channel-split a resize x-pass + paired y-pass into num_splits half-pairs to fit the AMM. Resize is
    # per-channel/spatial-preserving, so a channel slice keeps folding/x-slices/qparams unchanged - only
    # tensors and channel counts are sliced. The y-pass ALIASES the x-pass output in AMM, so each (rx_i,
    # ry_i) must stay co-blobbed: use '_c{i}' suffix (not '_split', which triggers blob isolation).
    rx_name = rx_node['name']
    ry_name = None
    for (cons, _) in rx_node['frontend']['following_nodes_params']:
        c = ir.graph.nodes[cons]
        if c['op_type'] == OpType.RESIZE and '_y' in cons:
            ry_name, ry_node = cons, c
            break
    if ry_name is None:
        return False  # no paired y-pass found; leave it

    orig_out_ch = rx_node['frontend']['output_channels']
    bounds = _block_aligned_bounds(orig_out_ch, (orig_out_ch + num_splits - 1) // num_splits)
    if len(bounds) - 1 < 2:
        return False

    rx_in = rx_node['frontend']['input_tensor']
    rx_in_name = rx_in.name
    rx_out = rx_node['frontend']['output_tensor']          # == ry input tensor
    rx_out_name = rx_out.name
    ry_out = ry_node['frontend']['output_tensor']          # final resize output (merged target)
    ry_out_name = ry_out.name
    rx_in_producer = rx_in.producer
    ry_out_consumers = list(ry_out.consumers)
    rx_ofx = rx_node['frontend']['output_folding_factor_x']; rx_ofy = rx_node['frontend']['output_folding_factor_y']
    rx_ifx = rx_node['frontend']['input_folding_factor_x'];  rx_ify = rx_node['frontend']['input_folding_factor_y']
    ry_ofx = ry_node['frontend']['output_folding_factor_x']; ry_ofy = ry_node['frontend']['output_folding_factor_y']
    ry_ifx = ry_node['frontend']['input_folding_factor_x'];  ry_ify = ry_node['frontend']['input_folding_factor_y']

    num = len(bounds) - 1
    ry_out_slice_names = []
    for i in range(num):
        lo, hi = bounds[i], bounds[i + 1]
        cs = hi - lo
        rxi = rx_node if i == 0 else copy.deepcopy(rx_node)
        ryi = ry_node if i == 0 else copy.deepcopy(ry_node)
        rxi['name'] = rx_name if i == 0 else _unique_graph_node_name(ir, '%s_c%d' % (rx_name, i))
        ryi['name'] = ry_name if i == 0 else _unique_graph_node_name(ir, '%s_c%d' % (ry_name, i))

        # rx_i input: channel slice of the x-pass input, read from the producer's DDR output at the offset.
        rxin_i = _slice_resize_tensor(ir, rx_in, '%s_c%d' % (rx_in_name, i), lo, hi)
        rxin_i.consumers = [rxi['name']]
        rxin_i.split_channel_lo = lo
        rxin_i.is_read_channel_slice = True
        ir.split_tensor_to_original_tensor_map[rxin_i.name] = rx_in_name
        # Forward registration too (merge, don't overwrite): without this, _alias_ddr_entries_to_
        # split_descendants' BFS never discovers rxin_i as rx_in_name's descendant, leaving ddr_entry
        # None -> AttributeError in the sequencer.
        existing_rxin = ir.original_tensor_to_split_tensor_map.get(rx_in_name, [])
        if rxin_i.name not in existing_rxin:
            existing_rxin.append(rxin_i.name)
        ir.original_tensor_to_split_tensor_map[rx_in_name] = existing_rxin
        rxi['frontend']['input_tensor'] = rxin_i
        rxi['inputs'][0] = rxin_i.name

        # rx_i output: its own in-AMM intermediate (consumed by ry_i in the same blob; NOT merged).
        rxout_i = _slice_resize_tensor(ir, rx_out, '%s_c%d' % (rx_out_name, i), lo, hi)
        rxout_i.producer = rxi['name']
        rxout_i.consumers = [ryi['name']]
        rxi['frontend']['output_tensor'] = rxout_i
        rxi['outputs'][0] = rxout_i.name

        # ry_i input == rx_i output.
        ryi['frontend']['input_tensor'] = rxout_i
        ryi['inputs'][0] = rxout_i.name

        # ry_i output: channel slice written into the MERGED final output's DDR region at the channel offset.
        ryout_i = _slice_resize_tensor(ir, ry_out, '%s_c%d' % (ry_out_name, i), lo, hi)
        ryout_i.producer = ryi['name']
        ryout_i.consumers = list(ry_out_consumers)
        ryout_i.split_channel_lo = lo
        ir.split_tensor_to_original_tensor_map[ryout_i.name] = ry_out_name
        ryi['frontend']['output_tensor'] = ryout_i
        ryi['outputs'][0] = ryout_i.name
        ry_out_slice_names.append(ryout_i.name)

        # Channel counts (folded) on both frontends and backends.
        for n, ofx, ofy, ifx, ify in ((rxi, rx_ofx, rx_ofy, rx_ifx, rx_ify), (ryi, ry_ofx, ry_ofy, ry_ifx, ry_ify)):
            n['frontend']['input_channels'] = cs
            n['frontend']['output_channels'] = cs
            if 'backend' in n and n['backend'] is not None:
                n['backend']['output_channels'] = cs * (2 ** (ofx + ofy))
                n['backend']['input_channels'] = cs * (2 ** (ifx + ify))

        if i != 0:
            ir.graph.add_node(rxi['name'], **rxi)
            ir.graph.add_node(ryi['name'], **ryi)
        # Wire rx_i -> ry_i, ry_i -> the original final-output consumers, and the x-pass input producer -> rx_i.
        ir.graph.add_edge(rxi['name'], ryi['name'])
        rxi['frontend']['following_nodes_params'] = [(ryi['name'], 0)]
        ryi['frontend']['preceding_nodes_params'] = [(rxi['name'], 0)]
        if rx_in_producer:
            ir.graph.add_edge(rx_in_producer, rxi['name'])
            if rxi['name'] not in ir.tensors[rx_in_name].consumers:
                ir.tensors[rx_in_name].consumers.append(rxi['name'])
        rxi['frontend']['preceding_nodes_params'] = list(rx_node['frontend']['preceding_nodes_params']) if i == 0 else \
            list(rx_node['frontend']['preceding_nodes_params'])
        ry_following = []
        for cons in ry_out_consumers:
            ir.graph.add_edge(ryi['name'], cons)
            ry_following.append((cons, 0))
        ryi['frontend']['following_nodes_params'] = ry_following

    # Merge, don't overwrite - see the identical fix/comment at the other split map registration
    # sites: an unconditional `=` here can silently drop an already-registered split-descendant name.
    existing_ry_out = ir.original_tensor_to_split_tensor_map.get(ry_out_name, [])
    merged_ry_out = list(reversed(ry_out_slice_names))
    for name in existing_ry_out:
        if name not in merged_ry_out:
            merged_ry_out.append(name)
    ir.original_tensor_to_split_tensor_map[ry_out_name] = merged_ry_out
    ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    # Mark every resize piece so avoid_ddr discipline etc. treat them as split-produced.
    for nm in [rx_name, ry_name] + ['%s_c%d' % (rx_name, i) for i in range(1, num)] + ['%s_c%d' % (ry_name, i) for i in range(1, num)]:
        if nm in ir.graph.nodes:
            ir.graph.nodes[nm]['frontend']['was_split'] = True
    return True

# An Add node alone in its own blob with >=1024 input channels overflows the wloc hardware table
# during cbc_generation_pass (MAX_WLOC_128BIT_ENTRIES, hw_config.py) even though it fits AMM on its
# own - the AMM-failure-driven split_large_conv_nodes trigger (program_compiler.py's failing-blob
# retry loop) never sees a node whose own blob allocation already succeeds. Runs proactively,
# independent of AMM allocation outcome. Each split at least halves input_channels, so re-checking
# the resulting pieces on the caller's next pass converges in a few iterations.
def force_split_oversized_wloc_adds(ir: internal_representation.IR):
    graph_nodes = ir.graph.nodes()
    did_split = False
    for blob in list(ir.tiling_blobs.values()):
        if len(blob.nodes_in_blob) != 1:
            continue
        node_name = blob.nodes_in_blob[0]
        node = graph_nodes[node_name]
        if node['op_type'] != OpType.ADD:
            continue
        # x_slices > 1 means h-tiling can still relieve this Add (shrinks per-tile AMM footprint without
        # touching channels); estimate_min_h_tiles returns h_tiles=1 once x_slices<=1, i.e. no headroom
        # left, so only then is a channel split the remaining option.
        input_tensors = node['frontend'].get('input_tensors', [])
        x_slices = input_tensors[0].x_slices if input_tensors else node['frontend'].get('x_slices', 1)
        if x_slices > 1:
            continue
        # split_large_conv_nodes itself checks the Add's actual per-input AMM footprint
        # (in_max_chunk_total, block-aligned) and returns False when the node already fits
        # without splitting - so calling it here at x_slices==1 already encodes "h-tiling
        # doesn't help fit in AMM" as its no-op condition.
        if split_large_conv_nodes(ir, node_name, node):
            did_split = True
    return ir, did_split


def split_large_conv_nodes(ir: internal_representation.IR, node_name, node):

    # A node already split once must never be split again: split_conv_in_channels_in_m /
    # split_node_out_channels_in_n mutate `node` in place and reuse `node_name`, so re-splitting would
    # corrupt cached tensor references - further fit issues go through h-tiling/re-blobbing instead.
    # was_consumer_bounds_split marks a node whose only prior split was boundary-matching (not AMM-sized);
    # pop it so that one-time allowance lets a real footprint-driven split still happen here.
    if node['frontend'].get('was_channel_split') and not node['frontend'].pop('was_consumer_bounds_split', False):
        print("split_large_conv_nodes: DECLINE %s - already was_channel_split" % node_name)
        return False

    # A CONCAT cannot be channel-split, but a wide one that overflows the AMM (its inputs must all be
    # co-resident contiguously) can be made a no-op via input-scatter: each input's producer writes straight
    # into the concat output DDR region, so the concat needs no AMM. Try that before giving up.
    if node['op_type'] == OpType.CONCAT:
        return scatter_concat_inputs(ir, node)
    # Gemm is out of scope for channel splitting; it surfaces via the caller's "Cannot further split blob".
    if node['op_type'] == OpType.GEMM:
        return False
    # A Sync node is an MXP sync marker with no AMM footprint, so it can't be the cause of a blob's AMM
    # overflow; channel-splitting it would only fabricate pointless pieces. Decline so the caller looks
    # elsewhere for the real cause.
    if node['op_type'] == OpType.SYNC:
        return False
    # A RESIZE must not be naively output-split (its x-pass/y-pass are an AMM-aliased pair). Channel-split
    # the WHOLE pair via split_resize_pair when the x-pass overflows AMM; the y-pass is handled through
    # its x-pass, so decline to split it directly.
    if node['op_type'] == OpType.RESIZE:
        if '_x' in node_name:
            # Size num_splits so one shot covers any overflow multiple (a hardcoded 2 would only halve
            # once, and re-invoking on the already-split '..._c0' piece would nest names as '..._c0_c0').
            rx_in_total = (node['frontend']['input_channels']
                           * pow(2, node['frontend']['input_folding_factor_x'] + node['frontend']['input_folding_factor_y'])
                           * node['frontend']['input_tensor'].x_slices)
            rx_out_total = (node['frontend']['output_channels']
                            * pow(2, node['frontend']['output_folding_factor_x'] + node['frontend']['output_folding_factor_y'])
                            * node['frontend']['output_tensor'].x_slices)
            rx_footprint = rx_in_total + 2 * rx_out_total
            amm = amm_capacity(ir)
            num_splits = max(2, math.ceil(rx_footprint / amm)) if rx_footprint > amm else 2
            return split_resize_pair(ir, node, num_splits)
        return False

    # A recombine Add from a conv input-channel split ('..._split_ic_add..') must NOT be channel-split -
    # its inputs are already conv slices, so re-splitting nests unresolved DDR aliasing. It fits via
    # H-TILING to 2 instead; decline here so the retry loop keeps it h-tiled.
    if node['op_type'] == OpType.ADD and '_split_ic_add' in node_name:
        return False

    # Folded input/output channel footprint, scaled by x_slices - the co-resident AMM demand the split
    # must bring under budget. Multi-input ops (Add) carry an input_tensors list, not 'input_tensor'.
    input_channels = node['frontend']['input_channels']
    input_x_fold = node['frontend']['input_folding_factor_x']
    input_y_fold = node['frontend']['input_folding_factor_y']

    # Multi-input ops (Add) count all their inputs. Key on op_type, NOT presence of 'input_tensor': an
    # output-split Add chunk carries both a stray 'input_tensor' and the real 'input_tensors' list.
    if node['op_type'] in MULTIPLE_INPUT_OPS:
        input_tensors = node['frontend']['input_tensors']
    else:
        input_tensors = [node['frontend']['input_tensor']]
    input_x_slices = input_tensors[0].x_slices
    num_inputs = len(input_tensors)

    # Footprint of ONE input tensor (folded channels * x_slices).
    input_total_channels = input_channels * pow(2, input_x_fold + input_y_fold) * input_x_slices

    output_channels = node['frontend']['output_channels']
    output_x_fold = node['frontend']['output_folding_factor_x']
    output_y_fold = node['frontend']['output_folding_factor_y']
    output_total_channels = output_channels * pow(2, output_x_fold + output_y_fold) \
        * node['frontend']['output_tensor'].x_slices

    # AMM channels available for a single blob (full AMM minus the reserved SYSTEM_zero block).
    amm = amm_capacity(ir)

    # add_identity_before_k3_starting_1T_blob later prepends a 1x1 pass-through identity (its own
    # single-node blob, peak AMM 3*input_total) before any k3 node starting a 1-tile blob. Fold that
    # into this split's chunk sizing now (3*piece_input_total <= AMM), since the multi-node failing-blob
    # path can't split a plain identity conv. Only applies in blob 0 (later blobs use boundary-extra-line
    # treatment instead, with no AMM footprint).
    needs_k3_identity = (node['op_type'] == OpType.CONV) and node['frontend'].get('tiling_blob_idx') == 0 \
        and ir.node_needs_k3_1T_identity(node, ir.amm_height)
    identity_amm_cap = (amm // 3) if needs_k3_identity else None

    # One-shot decision (no repeated two-way halving):
    #  - If the double-buffered input fits (2*in <= AMM), OUTPUT-split into N pieces (clean path, no
    #    recombine Add).
    #  - Else if the output alone fits (out <= AMM), INPUT-split (conv only) into M pieces recombined by
    #    an Add tree.
    #  - Else neither fits; raise so the case is addressed explicitly instead of spinning the loop.

    # This node reached the split with pre-split markers already set and allocation STILL failed.
    # Snapshot which markers are live so every node the split creates gets stamped too - the recombine
    # Add is a fresh dict that would otherwise lose them, and evict_amm_except_system_before_alloc is
    # only honored on a blob's first node, so each piece must carry it in case it lands first.
    presplit_markers = {marker: True for marker in (
        'avoid_ddr_read_while_conv', 'avoid_ddr_write_while_conv', 'evict_amm_except_system_before_alloc')
        if node['frontend'].get(marker, False)}
    nodes_before_split = set(ir.graph.nodes())

    # A downstream consumer may have ALREADY channel-sliced this node's output. If this producer then
    # output-splits independently with its own bounds, it would create colliding '<output_name>_splitN'
    # tensors and orphan the consumer's wired tensors (ddr_entry unresolved -> crash). Avoid this by
    # reusing the consumer's existing channel boundaries as this producer's split bounds when they exist
    # and cleanly tile the full output range.
    orig_output_tensor_name_for_reuse_check = node['outputs'][0]
    forced_bounds = _consumer_slice_bounds(
        ir, orig_output_tensor_name_for_reuse_check, node['frontend']['output_channels'])
    if forced_bounds is not None and len(forced_bounds) - 1 >= 2:
        print("split_large_conv_nodes: %s - reusing existing downstream channel-slice bounds %s "
              "(a consumer already fixed these channel boundaries before this producer split)" % (
                  node_name, forced_bounds))
        split_node_out_channels_in_n(ir, node_name, node, forced_bounds)

        def _stamp_split_markers_reuse(n_frontend):
            n_frontend['was_split'] = True
            n_frontend['avoid_ddr_read_while_conv'] = True
            n_frontend['avoid_ddr_write_while_conv'] = True
            n_frontend['evict_amm_except_system_before_alloc'] = True
            # This chunk's size came from matching a consumer's boundary, not from an AMM-fit
            # decision - let a later AMM-overflow retry on this exact chunk still attempt a real
            # split (see the DECLINE check at the top of this function).
            n_frontend['was_consumer_bounds_split'] = True
            if presplit_markers:
                n_frontend.update(presplit_markers)
        _stamp_split_markers_reuse(node['frontend'])
        for new_node_name in set(ir.graph.nodes()) - nodes_before_split:
            _stamp_split_markers_reuse(ir.graph.nodes[new_node_name]['frontend'])
        return True

    # Folded/x_slices multiplier per unfolded channel on each axis (input_total = input_channels * mult).
    # Chunk sizes are chosen in UNFOLDED channels and aligned to whole AMM blocks (multiples of 16), so
    # the per-chunk budget below is converted from the folded footprint back to unfolded channels.
    in_mult = (input_total_channels // input_channels) if input_channels else 1
    out_mult = (output_total_channels // output_channels) if output_channels else 1

    if node['op_type'] == OpType.CONV:
        # Footprint model: INPUT is single-buffered (avoid_ddr_read_while_conv). OUTPUT is double-buffered
        # unless avoid_ddr_write_while_conv is also set (single_buffered), in which case it's 1x. out_dup/
        # add_dup scale the output term: a plain piece is input(1x) + out_dup*output; a recombine-Add
        # piece needs add_dup*output (2 operands normally 3, the Add's two inputs plus its double-buffered
        # output write - or 2 when the output write isn't double-buffered). Always model the output as
        # single-buffered here since a split node's output need not stay double-buffered with the next tile.
        single_buffered = True
        out_dup = 1 if single_buffered else 2
        add_dup = 2 if single_buffered else 3
        # h-tiling-to-2 (program_compiler.py's update_htiling_info/apply_blob_htiling) can halve a
        # recombine Add's x_slices-driven footprint, but only when min(input_x_slices, output_x_slices)>=2
        # (get_num_h_tiles's own precondition). This relief is NEVER assumed for SIZING (piece/max_chunk
        # always use the full, un-relieved output_total_channels - trusting it caused a real overflow:
        # 2048 needed against amm=2032 with no relief materialized). It's used only to decide WHICH
        # strategy to attempt: an Add whose fixed add_dup*output_total overflows can't be rescued by
        # finer input splitting, so unless this structural relief is achievable, fall through to the
        # output+input combined cascade instead.
        add_htile_relief = 2 if min(input_x_slices, node['frontend']['output_tensor'].x_slices) >= 2 else 1
        output_total_for_add_check = output_total_channels

        print("split_large_conv_nodes: %s CONV amm=%d input_total=%d output_total=%d in_mult=%d out_mult=%d "
              "add_dup=%d identity_amm_cap=%s" % (
                  node_name, amm, input_total_channels, output_total_channels, in_mult, out_mult,
                  add_dup, identity_amm_cap))

        if input_total_channels <= amm and (identity_amm_cap is None or input_total_channels <= identity_amm_cap):
            remaining = amm - input_total_channels           # AMM left for the output
            if remaining <= 0:
                print("split_large_conv_nodes: DECLINE %s - output branch, remaining<=0" % node_name)
                return False
            max_chunk = remaining // (out_dup * out_mult)    # unfolded output channels per piece
            bounds = _block_aligned_bounds(output_channels, max_chunk, pow2_pieces=node['frontend']['output_folding_factor_x'] > 0 or node['frontend']['output_folding_factor_y'] > 0)
            if len(bounds) - 1 < 2:
                print("split_large_conv_nodes: DECLINE %s - output branch, max_chunk=%d gives <2 pieces" % (
                    node_name, max_chunk))
                return False
            split_node_out_channels_in_n(ir, node_name, node, bounds)
        elif -(-(add_dup * output_total_for_add_check) // add_htile_relief) <= amm:
            # Input too big, but the output is small enough - outright, or via achievable h-tiling relief
            # on the recombine Add (add_htile_relief, gated above on the structural x_slices precondition,
            # not assumed for SIZING below) - that a single recombine Add fits -> input-split directly,
            # recombining the slices with an Add tree. A slice-conv holds one input slice + the output, so
            # it fits when input_slice + (add_dup-1)*output <= AMM.
            piece = amm - (add_dup - 1) * output_total_for_add_check  # single-input-buffer
            if piece <= 0:
                print("split_large_conv_nodes: DECLINE %s - add-fits branch, piece<=0" % node_name)
                return False
            max_chunk = piece // in_mult                  # unfolded input channels per slice
            if identity_amm_cap is not None:
                max_chunk = min(max_chunk, identity_amm_cap // in_mult)
            # Stored input folding factors of 1 (fold of 2 in x and y) need a power-of-two input piece count.
            in_pow2 = input_x_fold > 0 and input_y_fold > 0
            bounds = _block_aligned_bounds(input_channels, max_chunk, pow2_pieces=in_pow2)
            if len(bounds) - 1 < 2:
                print("split_large_conv_nodes: DECLINE %s - add-fits branch, max_chunk=%d gives <2 pieces" % (
                    node_name, max_chunk))
                return False
            split_conv_in_channels_in_m(ir, node_name, node, bounds)
        elif output_total_channels <= amm and _small_pure_input_split_bounds(
                amm, input_channels, output_total_channels, add_dup, in_mult, identity_amm_cap,
                add_htile_relief) is not None:
            # Output fits AMM outright even though it misses the add_dup-margined bar above. Still prefer
            # a PURE input-channel split over the output+input combined cascade below when it needs only
            # a small number of input pieces (<= _SMALL_PURE_INPUT_SPLIT_MAX_PIECES) - splitting the
            # output too would needlessly multiply total pieces (input x output) instead of input alone
            # (regressed a known case, Conv40x40_i256_o512_k3_s2, without this check).
            bounds = _small_pure_input_split_bounds(
                amm, input_channels, output_total_channels, add_dup, in_mult, identity_amm_cap,
                add_htile_relief)
            split_conv_in_channels_in_m(ir, node_name, node, bounds)
        else:
            # Input is too big AND a single recombine Add would overflow (every Add level needs
            # ~3*piece_output_total, and an oversized Add's output can't feed a literal Concat either).
            # Fix: OUTPUT-split first into pieces small enough that each piece's own recombine Add fits
            # (via DDR channel-offset scatter, no literal Concat), then input-channel-split each piece.
            # Scan n_in upward from 2 for the minimal input-piece count whose worst-case piece, combined
            # with some output-piece size, lets every combo fit its own Add - a larger n_in only shrinks
            # the input budget further so the first fit found is minimal.
            n_in = 2
            in_bounds = None
            out_max_chunk = None
            max_n_in = -(-input_channels // AMM_CHANNELS_PER_BLOCK)  # can't split finer than 1 block/piece
            while n_in <= max_n_in:
                candidate_in_max_chunk = -(-input_channels // n_in)
                candidate_in_bounds = _block_aligned_bounds(input_channels, candidate_in_max_chunk)
                if len(candidate_in_bounds) - 1 < 2:
                    n_in += 1
                    continue
                # Worst-case (largest) input piece actually produced by this bound set - later pieces
                # from _block_aligned_bounds are never larger than earlier ones, so the first is worst-case.
                worst_piece_input_channels = candidate_in_bounds[1] - candidate_in_bounds[0]
                # identity_amm_cap bounds the input piece alone (3*input_piece<=amm for the
                # prepended k3 identity's own single-node-blob footprint) - it is NOT an
                # additional joint cap on (input_piece + output budget), so it must not be
                # subtracted from piece_budget a second time (that previously crushed
                # piece_budget to near-zero even when the input piece already satisfied the
                # cap on its own, e.g. 512<=677 still left only 677-512=165 "budget").
                if identity_amm_cap is not None and worst_piece_input_channels * in_mult > identity_amm_cap:
                    n_in += 1
                    continue
                piece_budget = amm - worst_piece_input_channels * in_mult
                if piece_budget > 0:
                    candidate_out_max_chunk = (piece_budget // add_dup) // out_mult if out_mult else piece_budget // add_dup
                    if candidate_out_max_chunk > 0:
                        in_bounds = candidate_in_bounds
                        out_max_chunk = candidate_out_max_chunk
                        break
                n_in += 1
            if in_bounds is None:
                raise Exception(
                    "%s: input (%d) and output (%d) both exceed AMM (%d); cannot find an input/output "
                    "split combination where every piece's recombine Add fits" % (
                        node_name, input_total_channels, output_total_channels, amm))
            out_bounds = _block_aligned_bounds(output_channels, out_max_chunk, pow2_pieces=node['frontend']['output_folding_factor_x'] > 0 or node['frontend']['output_folding_factor_y'] > 0)
            if len(out_bounds) - 1 < 2:
                raise Exception(
                    "%s: input (%d) and output (%d) both exceed AMM (%d)/3-per-recombine-Add; cannot "
                    "split output further" % (node_name, input_total_channels, output_total_channels, amm))

            # in_bounds above came from a conservative equal-guess chunk just to find a self-consistent
            # n_in; it never checked whether the now-fixed output split leaves more input headroom. So
            # recompute the true per-slice budget from the largest real output piece and rebalance into
            # the minimal piece count under that looser cap (avoids e.g. 4 pieces when 3 would fit).
            # Still un-relieved (no h-tiling-to-2 assumed), consistent with the sizing above.
            largest_out_piece = max(out_bounds[i + 1] - out_bounds[i] for i in range(len(out_bounds) - 1))
            largest_out_piece_total_for_add = largest_out_piece * out_mult
            real_piece_budget = amm - (add_dup - 1) * largest_out_piece_total_for_add
            if identity_amm_cap is not None:
                real_piece_budget = min(real_piece_budget, identity_amm_cap)
            real_max_chunk = real_piece_budget // in_mult if in_mult else real_piece_budget
            if real_max_chunk > 0:
                num_in_pieces = min(len(in_bounds) - 1, -(-input_channels // real_max_chunk))
                num_in_pieces = max(2, num_in_pieces)
                # Input-folded nodes split into a power-of-two number of input pieces.
                if input_x_fold > 0 and input_y_fold > 0:
                    num_in_pieces = 1 << (num_in_pieces - 1).bit_length()
                in_bounds = _balanced_bounds(input_channels, num_in_pieces)

            nodes_before_out_split = set(ir.graph.nodes())
            split_node_out_channels_in_n(ir, node_name, node, out_bounds)
            new_piece_names = (set(ir.graph.nodes()) - nodes_before_out_split) | {node_name}

            # Resolve each piece's actual graph node name (chunk 0 keeps node_name; dup chunks are
            # named '<node_name>_split<i>' by split_node_out_channels_in_n) and input-split it using
            # the SAME minimal-n_in input bounds for every piece, so the input-split count stays
            # uniform and minimal across pieces rather than being recomputed (and potentially
            # widened) independently per output piece.
            for i in range(len(out_bounds) - 1):
                piece_name = node_name if i == 0 else ('%s_split%d' % (node_name, i))
                if piece_name not in new_piece_names or piece_name not in ir.graph.nodes():
                    continue
                piece_node = ir.graph.nodes[piece_name]
                split_conv_in_channels_in_m(ir, piece_name, piece_node, in_bounds)
    else:
        # A MaxPool with input x/y folding must not be input-channel split (its folded input layout can't be
        # sliced per channel range); fail loudly rather than silently produce an invalid split.
        if node['op_type'] == OpType.MAXPOOL and input_x_fold > 0 and input_y_fold > 0:
            raise ValueError('MaxPool %s has input folding (x=%d, y=%d) and must not be input-channel split'
                             % (node_name, input_x_fold, input_y_fold))
        # Generic op (single-input MaxPool/Resize/AveragePool, or a MULTIPLE_INPUT_OPS Add): a channel-range
        # chunk slices the output (and each input's matching channel range) to lower the footprint.
        # Shared with estimate_min_h_tiles's per-h_tiles-hypothesis loop (program_compiler.py) so the two
        # never drift apart again - see generic_op_footprint's comment for the Add-aliases-input[0] model.
        if node['op_type'] in MULTIPLE_INPUT_OPS:
            # Elementwise Add: each of the 2 inputs must be independently co-resident in AMM (allocated as
            # separate block-aligned regions, not a combined byte sum), and the output aliases input[0]
            # with no separate buffer. So the real per-input budget is HALF the AMM, block-aligned DOWN -
            # not a combined num_inputs*input_total footprint vs the WHOLE amm, which misses per-input
            # block-rounding and can wrongly decline a split the real allocator would reject.
            in_max_chunk_total = (amm // 2 // AMM_CHANNELS_PER_BLOCK) * AMM_CHANNELS_PER_BLOCK
            if input_total_channels <= in_max_chunk_total:
                return False
            # input_total_channels/output_channels share the same mult (fold*x_slices) for an elementwise
            # Add (input and output shapes match), so a bound in "total" units converts directly to a
            # bound in raw output-channel units for _block_aligned_bounds below.
            num_pieces = math.ceil(input_total_channels / in_max_chunk_total)
            # Balance pieces evenly (ceil(output_channels/num_pieces)) rather than floor-dividing -
            # floor division leaves a remainder that _block_aligned_bounds appends as an extra, uneven
            # piece (e.g. 2048 channels / 3 floor-pieces of 682 leaves a stray 4th 32-channel piece).
            # Round the even share UP to the next block so num_pieces block-aligned chunks exactly cover
            # output_channels in that many pieces.
            max_chunk = -(-math.ceil(output_channels / num_pieces) // AMM_CHANNELS_PER_BLOCK) * AMM_CHANNELS_PER_BLOCK
        else:
            footprint = generic_op_footprint(node['op_type'], num_inputs, input_total_channels, output_total_channels)
            if footprint <= amm:
                return False
            # Single-input op: minimize the number of output pieces.
            # chunk_out/output_channels * footprint <= amm -> chunk_out <= output_channels*amm/footprint
            max_chunk = (output_channels * amm) // footprint
        bounds = _block_aligned_bounds(output_channels, max_chunk, pow2_pieces=node['frontend']['output_folding_factor_x'] > 0 or node['frontend']['output_folding_factor_y'] > 0)
        if len(bounds) - 1 < 2:
            return False
        split_node_out_channels_in_n(ir, node_name, node, bounds)

    # Mark every node this split touched as split-produced (was_split) so later passes recognize them
    # (e.g. add_identity_before_k3_starting_1T_blob arms eviction only when the blob's first node came
    # from a split). Each split node runs DDR fully serialized against conv, via three markers (live on
    # the frontend so they survive the set_tiling_blobs re-blob that follows):
    #  - avoid_ddr_read_while_conv: no input pre-read double-buffer; free the previous tile's input at
    #    tile start instead of holding two tiles' inputs co-resident.
    #  - avoid_ddr_write_while_conv: output DDR write is blocking, completing before the next blob's read.
    #  - evict_amm_except_system_before_alloc: evict every resident AMM tensor except system-zero at blob
    #    entry, so AMM starts clean (avoids last-block-short failures from a leftover co-resident tensor).
    def _stamp_split_markers(n_frontend):
        n_frontend['was_split'] = True
        n_frontend['avoid_ddr_read_while_conv'] = True
        n_frontend['avoid_ddr_write_while_conv'] = True
        n_frontend['evict_amm_except_system_before_alloc'] = True
        if presplit_markers:
            n_frontend.update(presplit_markers)
    _stamp_split_markers(node['frontend'])
    for new_node_name in set(ir.graph.nodes()) - nodes_before_split:
        _stamp_split_markers(ir.graph.nodes[new_node_name]['frontend'])
    return True

def calc_nodes_folding_factor(ir):
    set_folding_factor_attempt = 0
    successful_folding_factor = False
    sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    MAX_FOLDING_FACTOR_SETTINGS_RETRIES = 10
    while set_folding_factor_attempt<MAX_FOLDING_FACTOR_SETTINGS_RETRIES and not successful_folding_factor:
        successful_folding_factor = True
        # Clear any unfold-onto-following-node absorption decided by a previous attempt -
        # it will be freshly re-decided (or not) as this attempt's nodes are reprocessed in
        # topological order, strictly before the consumer node it targets is reached.
        for _node_name in sorted_graph:
            _node = ir.graph.nodes[_node_name]
            if _node['frontend'].pop('_absorbed_unfold_x', False):
                _node['frontend'].pop('force_unfolding_x', None)
                if _node_name in ir.force_y_unfolding:
                    ir.force_y_unfolding.remove(_node_name)
        tqdm_iterator = tqdm(sorted_graph)
        for node_name in tqdm_iterator:
            node = ir.graph.nodes[node_name]
            tqdm_iterator.set_description('Folding factor propogation, at layer %s:' % node_name)
            if node['op_type'] == OpType.CONV:
                calc_conv_folding_factor(ir,node_name,node)
            elif node['op_type'] == OpType.TRANSPOSE_CONV:
                # Distinct op type but folds like a conv (flipped to OpType.CONV at end of qparams
                # stage). Run conv folding calc to populate scaffolding, then override transpose-conv
                # specifics: input is already-upsampled so pin folding to 0/stride to 1, output is
                # folded (factor 1) then unfolded via force_unfolding flags below.
                node['frontend']['input_folding_factor_x'] = 0
                node['frontend']['input_folding_factor_y'] = 0
                node['frontend']['stride'] = 1
                # 3. Extract and reshape original weights and biases
                orig_weights = node['frontend']['weights_tensor'].data # Shape: [in_c, out_c, k, k]
                kernel_size = node['frontend']['kernel_size']
                input_channels = node['frontend']['input_channels']
                output_channels = node['frontend']['output_channels']
                if (node['frontend']['kernel_size'] % 2 != 0):
                    orig_weights = np.pad(orig_weights, ((0, 0), (0, 0), (0, 1), (0, 1)), mode='constant', constant_values=0)
                    kernel_size += 1
                new_kernel_size = kernel_size // 2
                node['frontend']['kernel_size'] = new_kernel_size
                # Stride-phase split: each kernel axis (length 2*new_kernel_size) splits as
                # (new_kernel_size, 2) so the phase index is k % 2 (sub-pixel stride sampling:
                # phase p uses kernel taps p, p+2, ...), NOT k // new_kernel_size (a contiguous
                # block split, which mis-decomposes the transpose conv when new_kernel_size > 1).
                new_weights = orig_weights.reshape(output_channels, input_channels, new_kernel_size, 2, new_kernel_size, 2)
                # Phase-major channel ordering (channel = phase*output_channels + oc, phase = 2*ky + kx)
                # to match the fold/unfold reconstruction convention in
                # folding_algo.asym_unfold_resfold_conv_output_continousychannels (folded channel =
                # phase*C + ic). An oc-major layout here scatters phases to the wrong positions.
                new_weights = np.transpose(new_weights, (3, 5, 0, 1, 2, 4)).reshape(output_channels * 4, input_channels, new_kernel_size, new_kernel_size)
                # Transpose-conv 180-degree kernel flip: in the equivalent forward conv each phase
                # reads in[m-1..m] with the taps in reversed spatial order (output_sub[m] =
                # W[phase]*in[m] + W[phase+2]*in[m-1]), so flip the phase kernel's spatial axes. The
                # subsequent 2x2->3x3 conversion (top-left placement + pad [1,1,1,1]) then supplies
                # the correct in[m-1..m] window. Identity for new_kernel_size==1 (the 2x2 case).
                new_weights = np.flip(new_weights, axis=(2, 3))
                node['frontend']['weights_tensor'].data = np.ascontiguousarray(new_weights)
                node['frontend']['weights_tensor'].shape = new_weights.shape
                orig_biases = node['frontend']['biases_tensor'].data
                # Phase-major: tile (not repeat) so channel c -> orig[c % output_channels] = orig[oc].
                new_biases = np.tile(orig_biases, 4)
                
                weights_scale = node['frontend']['weights_tensor'].scale
                weights_zp = node['frontend']['weights_tensor'].zero_point
                inputs_scale = node['frontend']['input_tensor'].scale
                inputs_zp = node['frontend']['input_tensor'].zero_point
                scale_acc = weights_scale * inputs_scale
    
                elements_old = input_channels * kernel_size * kernel_size
                elements_new = input_channels * new_kernel_size * new_kernel_size
    
                # Per-channel quant params (length-output_channels arrays) must be tiled 4x to match the
                # phase-major expansion; per-tensor quant (scalar/size-1) is left as-is. Gate on
                # ".size > 1" so a size-1 array isn't wrongly tiled into a length-4 array.
                def _is_per_channel(v):
                    return isinstance(v, np.ndarray) and v.size > 1

                # Zero-point / element-count bias correction. int() on a >1-element array raises
                # "only size-1 arrays can be converted to Python scalars", so tile per-channel and
                # add element-wise; scalar/size-1 keeps the original int() cast.
                zp_bias_correction = (weights_zp * inputs_zp * (elements_new - elements_old)) / scale_acc
                if _is_per_channel(zp_bias_correction):
                    new_biases = new_biases + np.tile(zp_bias_correction.astype(np.int64), 4)
                else:
                    new_biases = new_biases + int(zp_bias_correction)
                node['frontend']['biases_tensor'].data = new_biases
                node['frontend']['biases_tensor'].shape = new_biases.shape
                # The bias data was tiled 4x above, so its per-channel scale/zp must be tiled to
                # match, or the later `data * scale` in calc_conv_qparams fails to broadcast.
                biases_scale = node['frontend']['biases_tensor'].scale
                if _is_per_channel(biases_scale):
                    node['frontend']['biases_tensor'].scale = np.tile(biases_scale, 4)
                biases_zp = node['frontend']['biases_tensor'].zero_point
                if _is_per_channel(biases_zp):
                    node['frontend']['biases_tensor'].zero_point = np.tile(biases_zp, 4)
                node['frontend']['output_channels'] *= 4
                # Phase-major: tile per-channel params to match the weight layout above, and update the
                # frontend per-channel copies too - calc_conv_qparams indexes weights_per_channel_scale
                # over the 4x output channels, so a non-tiled array would be out of bounds.
                if _is_per_channel(weights_scale):
                    tiled_scale = np.tile(weights_scale, 4)
                    node['frontend']['weights_tensor'].scale = tiled_scale
                    node['frontend']['weights_per_channel_scale'] = tiled_scale
                else:
                    node['frontend']['weights_tensor'].scale = weights_scale
                if _is_per_channel(weights_zp):
                    tiled_zp = np.tile(weights_zp, 4)
                    node['frontend']['weights_tensor'].zero_point = tiled_zp
                    node['frontend']['weights_per_channel_zp'] = tiled_zp
                else:
                    node['frontend']['weights_tensor'].zero_point = weights_zp
                calc_conv_folding_factor(ir,node_name,node)
                # Output side: folded by 1 in x and y.
                node['frontend']['output_folding_factor_x'] = 1
                node['frontend']['output_folding_factor_y'] = 1
                node['frontend']['output_tensor'].folding_factor_x = 1
                node['frontend']['output_tensor'].folding_factor_y = 1
                node['frontend']['output_folding_factor_x'] = 1
                node['frontend']['output_folding_factor_y'] = 1
                # This fold is local to the phase-decomposition just done - no downstream node wants
                # it still folded, so unfold right after this node instead of letting it ride further.
                node['frontend']['unfold_x_after_node'] = True
            elif node['op_type'] == OpType.AVERAGEPOOL:
                calc_avgpool_folding_factor(ir,node_name,node)
            elif node['op_type'] == OpType.GEMM:
                calc_gemm_folding_factor(ir,node_name,node)
            elif node['op_type'] == OpType.MAXPOOL:
                calc_maxpool_folding_factor(ir,node_name,node)
            elif node['op_type'] == OpType.ADD:
                calc_add_folding_factor(ir,node_name,node)
            elif node['op_type'] == 'GlobalAveragePool':
                calc_globalavg_pooling_folding_factor(ir,node_name,node)
            elif node['op_type'] == OpType.CONCAT:
                if not calc_concat_folding_factor(ir,node_name,node): # If we encountered concat inputs with different folding factor we mark its source conv for folding and re-try
                    successful_folding_factor = False
                    print('\nUpdated folding factor because of Concat op. Re-propogating folding factor...')
                    break
            elif node['op_type'] == OpType.RESIZE:
                calc_resize_folding_factor(ir,node_name,node)
            elif node['op_type'] == OpType.IDENTITY:
                calc_identity_folding_factor(ir,node_name,node)
            elif node['op_type'] == OpType.SYNC:
                # Copying these values so report writing does not fail. Alternative solution
                # is to edit report function to print blank if these values are not present.
                copy_folding_factor_from_prev_node(node, ir.amm_height, ir)
            else:
                raise ValueError ('WARNING: Found an unsupported op of type ' + node['op_type'])
                calc_general_folding_factor(ir,node_name,node)
            
            # Check if the output tensor is either an output to the graph or sent to MXP
            output_tensor_name = node['frontend']['output_tensor'].name
            is_output = output_tensor_name in ir.outputs
            to_mxp = output_tensor_name in ir.tensors_to_mxp
            output_folding_factor_x = node['frontend']['output_folding_factor_x']
            unfold_locally = node['frontend'].get('unfold_x_after_node', False)

            # If this node's need to unfold x can be satisfied by the single conv consuming its output,
            # do the unfold there directly instead of inserting a separate identity unfold conv here.
            # absorbed_unfold_on_following = False
            # if (unfold_locally and not is_output and not to_mxp and output_folding_factor_x == 1):
            #     following_nodes_params = node['frontend'].get('following_nodes_params', [])
            #     if len(following_nodes_params) == 1:
            #         following_node_name = following_nodes_params[0][0]
            #         following_node = ir.graph.nodes[following_node_name]
            #         following_frontend = following_node.get('frontend', {})
            #         if (following_node.get('op_type') == OpType.CONV
            #                 and len(following_frontend.get('preceding_nodes_params', [])) == 1
            #                 and following_frontend.get('kernel_size') == 1
            #                 and 'force_folding_x' not in following_frontend):
            #             following_frontend['force_unfolding_x'] = True
            #             following_frontend['_absorbed_unfold_x'] = True
            #             if node['frontend']['output_folding_factor_y'] > 0 and following_node_name not in ir.force_y_unfolding:
            #                 ir.force_y_unfolding.append(following_node_name)
            #             absorbed_unfold_on_following = True

            if (is_output or to_mxp or unfold_locally) and output_folding_factor_x > 0:
                prev_node = node
                for unfold_idx in range(output_folding_factor_x):
                    name_sufix = 'unfold_x_'+str(unfold_idx)
                    new_tensor_name = output_tensor_name + '_' + name_sufix
                    if new_tensor_name not in ir.tensors:
                        insert_identity_node(ir=ir, param_source_node = prev_node, name_sufix = name_sufix)
                        identity_node_name = prev_node['name']+'_'+ name_sufix
                        identity_node = ir.graph.nodes[identity_node_name]
                        identity_node['frontend']['force_unfolding_x'] = True
                        prev_node = identity_node
                if to_mxp:
                    ir.tensors_to_mxp.discard(output_tensor_name)
                    ir.tensors_to_mxp.add(new_tensor_name)
                    ir.mxp_tensor_to_offset[new_tensor_name] = ir.mxp_tensor_to_offset[output_tensor_name]
                if is_output:
                    ir.outputs.remove(output_tensor_name)
                    ir.outputs.append(new_tensor_name)

        set_folding_factor_attempt+=1
    if set_folding_factor_attempt==MAX_FOLDING_FACTOR_SETTINGS_RETRIES:
        raise ValueError ('Failed to set good folding factor... please check')
    return ir

def try_replace_add_with_identity(ir, node_name, node):
    """
    Replace ADD with constant input as Identity/Requant by adjusting output zero_point.

    Based on instruction from add_replasment/istruction_romt.txt:
    If ADD has one constant input with same quantization as dynamic input,
    we can remove ADD and encode the constant addition as a zero_point shift.

    Formula: zero_point_new = zero_point_old - (q_constant - zero_point_old)
    """
    # Check if we have exactly 2 inputs
    if len(node['frontend']['input_tensors']) != 2:
        return False

    input0 = node['frontend']['input_tensors'][0]
    input1 = node['frontend']['input_tensors'][1]

    # Determine which input is constant
    constant_input = None
    dynamic_input = None
    constant_idx = -1

    if input0.is_constant and not input1.is_constant:
        constant_input = input0
        dynamic_input = input1
        constant_idx = 0
    elif input1.is_constant and not input0.is_constant:
        constant_input = input1
        dynamic_input = input0
        constant_idx = 1
    else:
        # Both constant or both dynamic - cannot apply this optimization
        return False

    # Get quantization parameters
    input_scales = node['frontend']['input_tensors_scale']
    input_zps = node['frontend']['input_tensors_zp']
    output_scale = node['frontend']['output_tensor_scale']
    output_zp = node['frontend']['output_tensor_zp']

    dynamic_idx = 1 - constant_idx
    dynamic_scale = input_scales[dynamic_idx]
    dynamic_zp = input_zps[dynamic_idx]
    constant_scale = input_scales[constant_idx]
    constant_zp = input_zps[constant_idx]

    # Get constant data and shape info
    q_constant = constant_input.data  # int8 quantized values
    constant_shape = constant_input.get_original_shape()
    dynamic_shape = dynamic_input.get_original_shape()

    # ADD in VNNX requires same shape (no broadcasting support in compiler yet)
    # So both tensors should have format [batch, channels, height, width]
    if constant_shape != dynamic_shape:
        # Different shapes - cannot apply optimization (would need broadcasting support)
        return False

    # Extract number of channels (axis 1 in NCHW format)
    if len(constant_shape) < 2:
        return False
    num_channels = constant_shape[1]

    # Strategy: We convert everything to output scale/zp
    # 1. Dynamic input: will be requantized via Identity/Requant node
    # 2. Constant: we dequantize and re-quantize to output scale/zp
    # Then both are in same scale and we can apply zero_point shift

    # Reshape constant to [channels, -1] to work per-channel
    q_constant_reshaped = q_constant.reshape(constant_shape[0], constant_shape[1], -1)

    # For each channel, calculate the constant value in output quantization
    zero_point_new_per_channel = np.zeros(num_channels, dtype=np.int32)

    for ch in range(num_channels):
        # Get constant values for this channel (all spatial locations)
        ch_constants = q_constant_reshaped[0, ch, :]  # Assume batch=1 for now

        # Check if this channel has uniform constant value
        unique_vals = np.unique(ch_constants)
        if len(unique_vals) != 1:
            # This channel has non-uniform constant - cannot optimize with simple zero_point shift
            # (would need per-pixel zero_point which is not supported)
            return False

        q_constant_ch_original = unique_vals[0]

        # Dequantize constant from its original scale/zp
        real_constant_ch = constant_scale * (q_constant_ch_original - constant_zp)

        # Re-quantize constant to output scale/zp
        q_constant_ch_output = np.round(real_constant_ch / output_scale + output_zp).astype(np.int32)

        # Validate: must be in int8 range
        if q_constant_ch_output < -128 or q_constant_ch_output > 127:
            return False

        # New zero_point for this channel (dynamic input requantized to output scale/zp via Identity):
        # zero_point_new = 2*output_zp - q_constant_output
        zero_point_new_ch = 2 * output_zp - q_constant_ch_output

        # Validate: must be in int8 range
        if zero_point_new_ch < -128 or zero_point_new_ch > 127:
            return False

        zero_point_new_per_channel[ch] = zero_point_new_ch

    # Success! Convert ADD to Identity with per-channel zero_point

    # Save constant real values for use in calc_identity_qparams
    # Calculate real values for all channels
    constant_real_values = np.zeros(num_channels, dtype=np.float32)
    for ch in range(num_channels):
        # Get constant values for this channel (all spatial locations)
        ch_constants = q_constant_reshaped[0, ch, :]  # Assume batch=1
        unique_vals = np.unique(ch_constants)
        q_constant_ch = unique_vals[0]
        # Dequantize to get real value
        constant_real_values[ch] = constant_scale * (q_constant_ch - constant_zp)

    # Update node to be Identity instead of Add
    # The dynamic input will be requantized from its scale/zp to output scale/zp
    node['frontend']['input_tensor'] = dynamic_input
    node['frontend']['input_tensor_scale'] = dynamic_scale
    node['frontend']['input_tensor_zp'] = dynamic_zp

    # Save constant real values for NEW FORMULA in calc_identity_qparams
    node['frontend']['add_constant_real'] = constant_real_values

    # Output uses output scale and ORIGINAL zero_point from TFLite
    # REMOVED: per-channel zero_point transformation (not needed with NEW FORMULA)
    # node['frontend']['output_tensor_zp'] = zero_point_new_per_channel  # OLD: was overwriting original output_zp
    # Keep original output_zp from TFLite (already set in _set_output_tensor_params)
    node['frontend']['output_tensor_scale'] = output_scale  # Important: use OUTPUT scale, not dynamic scale!

    # Remove the old input_tensors (plural) references
    del node['frontend']['input_tensors']
    del node['frontend']['input_tensors_scale']
    del node['frontend']['input_tensors_zp']

    # Update graph: remove edge from constant tensor to this node
    # The constant tensor name is the producer of constant_input
    constant_tensor_name = constant_input.name
    if ir.graph.has_edge(constant_tensor_name, node_name):
        ir.graph.remove_edge(constant_tensor_name, node_name)

    # Update node['inputs'] to only include the dynamic input
    # This is needed for reports and other code that checks node['inputs']
    node['inputs'] = [dynamic_input.name]

    return True

def _elide_concat_requant_if_noop(ir, node_name, node):
    # A concat-input requant Identity (inserted at parse time when the producer's baked scale
    # differed from the Concat's target) can become a no-op once an earlier node in this same pass
    # already propagated the real output scale/zp onto the producer. Splice it out when so.
    in_name = node['inputs'][0]
    out_name = node['outputs'][0]
    in_tensor = ir.tensors[in_name]
    out_tensor = ir.tensors[out_name]
    if in_tensor.scale != out_tensor.scale or in_tensor.zero_point != out_tensor.zero_point:
        return False

    graph = ir.graph
    producer_name = in_tensor.producer
    producer_node = graph.nodes[producer_name] if (producer_name is not None and graph.has_node(producer_name)) else None
    for consumer_name in list(out_tensor.consumers):
        consumer_node = graph.nodes[consumer_name]
        consumer_node['inputs'] = [in_name if t == out_name else t for t in consumer_node['inputs']]
        # program_compiler reads cached Tensor objects off node['frontend']['input_tensor']/
        # ['input_tensors'] directly, not via node['inputs'] names - patch those too or they keep
        # pointing at the stale out_tensor / this removed node.
        consumer_frontend = consumer_node.get('frontend', {})
        if consumer_frontend.get('input_tensor') is out_tensor:
            consumer_frontend['input_tensor'] = in_tensor
        if 'input_tensors' in consumer_frontend:
            consumer_frontend['input_tensors'] = [
                in_tensor if t is out_tensor else t for t in consumer_frontend['input_tensors']
            ]
        # Likewise, preceding/following_nodes_params are cached tuples that don't track later
        # graph.remove_node()/add_edge() calls, so should_start_new_blob would KeyError on this
        # removed node's name. Repoint them at the real producer.
        if 'preceding_nodes_params' in consumer_frontend:
            consumer_frontend['preceding_nodes_params'] = [
                (producer_name, idx) if n == node_name else (n, idx)
                for (n, idx) in consumer_frontend['preceding_nodes_params']
            ]
        if consumer_name not in in_tensor.consumers:
            in_tensor.consumers.append(consumer_name)
        if producer_node is not None:
            graph.add_edge(producer_name, consumer_name)
            producer_frontend = producer_node.get('frontend', {})
            if 'following_nodes_params' in producer_frontend:
                if not any(n == consumer_name for n, _ in producer_frontend['following_nodes_params']):
                    input_idx = None
                    for i, t in enumerate(consumer_node['inputs']):
                        if t == in_name:
                            input_idx = i
                            break
                    producer_frontend['following_nodes_params'] = [
                        (n, idx) for (n, idx) in producer_frontend['following_nodes_params'] if n != node_name
                    ] + [(consumer_name, input_idx)]

    in_tensor.consumers = [c for c in in_tensor.consumers if c != node_name]
    del ir.tensors[out_name]
    graph.remove_node(node_name)
    return True


def calc_nodes_quantization_params(ir):
    sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    tqdm_iterator = tqdm(sorted_graph)
    for node_name in tqdm_iterator:
        node = ir.graph.nodes[node_name]
        tqdm_iterator.set_description('Calculating quantization params and folding weights, at layer %s:' % node_name)
        # Every op-type branch below computes qparams; afterwards we uniformly attach any
        # activation_silu LUT so the SiLU/Requant computation is no longer Conv-only. Concat and Sync
        # are excluded (they never carry a fused non-linearity). Resize splits into _x/_y nodes and
        # only the _x half has qparams computed, so scope the LUT to it too.
        applied_qparams = node['op_type'] not in (OpType.CONCAT, OpType.SYNC)
        if node['op_type'] == OpType.CONV:
            calc_conv_qparams(ir,node_name,node)
        elif node['op_type'] == OpType.TRANSPOSE_CONV:
            calc_conv_qparams(ir,node_name,node)
            # Flip to Conv (AveragePool pattern above) so the sequencer, program_compiler and
            # numeric simulator reuse the entire conv backend unchanged. Everything downstream of
            # this stage only ever sees a Conv node.
            node['op_type'] = OpType.CONV
        elif node['op_type'] == OpType.AVERAGEPOOL:
            calc_avgpool_qparams(ir,node_name,node)
            node['op_type'] = OpType.CONV # Alex: We change the op type to conv so that report writing works properly
        elif node['op_type'] == OpType.GEMM:
            calc_gemm_qparams(ir,node_name,node)
        elif node['op_type'] == OpType.MAXPOOL:
            calc_maxpool_qparams(ir,node_name,node)
        elif node['op_type'] == OpType.ADD:
            # Try to replace ADD with constant as Identity
            if try_replace_add_with_identity(ir, node_name, node):
                # Successfully replaced with Identity - calculate identity params
                node['op_type'] = OpType.IDENTITY
                calc_identity_qparams(ir,node_name,node)
            else:
                # Keep as ADD - calculate normal ADD params
                calc_add_qparams(ir,node_name,node)
        elif node['op_type'] == OpType.IDENTITY:
            if node.get('attributes', {}).get('concat_requant') and _elide_concat_requant_if_noop(ir, node_name, node):
                applied_qparams = False
            else:
                calc_identity_qparams(ir,node_name,node)
        elif node['op_type'] == OpType.CONCAT:
            calc_concat_qparams(ir,node_name,node)
        elif node['op_type'] == OpType.RESIZE:
            if '_x' in node_name:
                calc_resize_qparams(ir,node_name,node)
            else:
                applied_qparams = False
        elif node['op_type'] == OpType.SYNC:
            pass
        else:
            raise ValueError ('Found an unsupported op of type ' + node['op_type'])
        if applied_qparams:
            apply_activation_silu_lut(node)
    return ir
