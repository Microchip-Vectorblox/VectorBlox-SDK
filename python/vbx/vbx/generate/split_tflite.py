import argparse
from .utils import existing_file, existing_dir, json_load, json_dump
from .transform_tflite import is_forked, is_multi_input, is_singleton_op, json2tflite
from .transform_tflite import graph_pattern, get_scale_factor, get_numpy_data, create_graph
import json
import copy
import subprocess
import shlex
import os.path
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import sys, shutil, glob, tempfile
from tqdm import tqdm 
import numpy as np
import cv2
import vbx.postprocess.dataset as rgb_color

#LUT related optimizations options
OPTIMIZED_WITH_LUT = 1
VCI_LUT = 1
MAX_LUTs = 4

@contextmanager
def exception_catcher(unsupported_ops):
    try:
        yield
    except AssertionError as e:
        sys.stderr.write("\n\033[31m######################## VECTORBLOX ERROR! #############################\033[0m\n")
        sys.stderr.write("\033[31mSome layers have parameters/arguments that we do not currently support.\033[0m")
        sys.stderr.write("\nPlease check the folder \033[31m unsupported_ops \033[0m in your working directory.")
        sys.stderr.write("\nFor futher assistance, please send the layer(s) to the vectorblox team at:\n\033[31mvectorblox@microchip.com\033[0m\n")

        sys.stderr.write("\nError unsupported ops:")
        for i,op,opcode, error_param, param_value, input_param in unsupported_ops:
            if len(param_value) == 0:
                print("\t {} at location {} is not currently supported.\n".format(opcode, i))   
            else:
                print("\t {} at location {}, \033[31m {}={} \033[0m is not currently supported.".format(opcode, i, param_value[0], input_param[0]))     
                print("\t\t-->Supported values are = {}\n".format(error_param[0]))
        # print('\nThese operators have been extracted and saved to ./unsupported_ops')
        # print('Please send to VectorBlox team for support')
        print()
        sys.exit(1)


# Check for case of Non-NX -> Pad -> Conv, where current op is the Pad
def fuse_pad_into_next_op(engine_op_types, prev_op, curr_op, next_op):
    # Make sure this is a PAD
    if curr_op != 'PAD':
        return False

    # If the previous op is already on NX then this isn't relevant
    if prev_op in engine_op_types.nx_op_types:
        return False

    # If next is Conv, split at the Pad so it can be combined into the next Conv
    if next_op in engine_op_types.nx_op_types:
        return True

    return False


# Whether to force a split because two ops must run on different engines
def force_split_due_to_engine(engine_op_types, prev_op, curr_op, next_op) -> bool:
    # If this op is agnostic, it might be between 2 ops which are on separate engines.
    # These cases are checked below. Otherwise, this op is agnostic so there is no need
    # to force a split.
    if curr_op in engine_op_types.agnostic_op_types:
        if fuse_pad_into_next_op(engine_op_types, prev_op, curr_op, next_op):
            return True
        return False

    # Otherwise, this op is not agnostic, so split if the previous op is on a different
    # engine than this op. If the previous op is agnostic, do not force a split since
    # those cases were handled above.
    if prev_op in engine_op_types.agnostic_op_types:
        return False
    prev_offloaded = prev_op in engine_op_types.nx_op_types
    curr_offloaded = curr_op in engine_op_types.nx_op_types
    if prev_offloaded != curr_offloaded:
        return True
    return False


def get_splits_vbx2(jname, split_every_op=False, size_config='V1000', engine_op_types=None):

    if type(jname) is str:
        graph = json_load(jname)
        assert(len(graph['subgraphs']) == 1)
    else:
        graph = jname

    buffers = graph['buffers']
    subgraph = graph['subgraphs'][0]
    outputs = subgraph['outputs']
    operators = subgraph['operators'] # ops = subgraph['operators']
    tensors = subgraph['tensors']
    codes = [_['builtin_code'] for _ in graph['operator_codes']]
    splits = []
    errors = []
    specials = []
    current = []
    current_lut_count = 0
    prev_op = None
    prev_opcode = None
    prev_inputs = None
    prev_outputs = None

    i = 0
    while i < len(operators):
        op = operators[i]
        if 'builtin_options' in op:
            opts = op['builtin_options']
        opcode = codes[op['opcode_index']]

        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]
        op_ishapes = [tensors[_].get('shape', []) for _ in op_inputs]
        op_oshapes = [tensors[_].get('shape', []) for _ in op_outputs]
        #op_ishapes = [tensors[_]['shape'] for _ in op_inputs]
        #op_oshapes = [tensors[_]['shape'] for _ in op_outputs]
      
        multi_input = is_multi_input(op, tensors, buffers)
        singleton = is_singleton_op(op, tensors)

        connected, forked = True, False
        if prev_op != None:
            connected = any([_ in prev_outputs for _ in op_inputs])
            forked, next_op_ids = is_forked(prev_op, operators, tensors)
            next_opcodes = [codes[operators[_]['opcode_index']] for _ in next_op_ids]

        output = any([_ for _ in op_inputs if _ in outputs])

        pattern_type, pattern = graph_pattern(operators[i:], codes, tensors, buffers)
        pattern = [i+_ for _ in pattern]
        partial_pattern = False
        if len(pattern) and forked:
            partial_pattern = not all([_ in pattern for _ in next_op_ids]) and forked

        if prev_opcode in ['PAD', 'PADV2']:
            if np.sum(get_numpy_data(tensors[prev_op['inputs'][1]], buffers)[-1]) > 0: #expands maps
                if len(current):
                    splits.append(current)
                current = []
                current_lut_count = 0
            elif len(next_op_ids) and next_opcodes[0] in ['MAX_POOL_2D', 'AVERAGE_POOL_2D']: #TODO onnx r18
                if len(current):
                    splits.append(current)
                current = []
                current_lut_count = 0

        if prev_opcode in ['CONCATENATION', 'STRIDED_SLICE']:
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0

        # If a pattern is found, check if it should be processed
        if len(pattern):
            if pattern_type == "TRANSFORM" or (pattern_type == "LUT" and (current_lut_count or partial_pattern)) or not connected:
                if len(current):
                    splits.append(current)
                current = []         
                current_lut_count = 0

            current += pattern
            if pattern_type == "LUT":
                current_lut_count += 1
            i += len(pattern) - 1
            op = operators[i]
            op_outputs = [_ for _ in op['outputs'] if _ != -1]

            # Split after pixel shuffle TRANSFORM if followed by LUT pattern to allow independent tiling
            # Only for ["RESHAPE", "TRANSPOSE", "RESHAPE"] pattern for PIXEL_SHUFFLE which changes tensor size significantly
            if pattern_type == "TRANSFORM":
                pattern_opcodes = [codes[operators[p]['opcode_index']] for p in pattern]
                if pattern_opcodes == ["RESHAPE", "TRANSPOSE", "RESHAPE"]:
                    next_pattern_type, next_pattern = graph_pattern(operators[i+1:], codes, tensors, buffers)
                    if next_pattern_type == "LUT":
                        if len(current):
                            splits.append(current)
                        current = []
                        current_lut_count = 0

        elif forked or output or not connected or split_every_op:
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        elif opcode in ['BATCH_MATMUL', 'CONCATENATION', 'DEPTHWISE_CONV_2D', 'CONV_2D', 'TRANSPOSE_CONV', 'FULLY_CONNECTED', 'UNIDIRECTIONAL_SEQUENCE_LSTM', 'SPLIT', 'SPLIT_V', 'PACK', 'UNPACK', 'TRANSPOSE', 'SOFTMAX']: #, 'ARG_MAX']: # start a new graph before key subgraph OP
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        elif opcode in ['RESIZE_BILINEAR'] and not get_scale_factor(operators, tensors, i) in [(2.,2.), (4.,4.), (8.,8.)]: #TODO optimized scales can fuse
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        elif opcode in ['RESHAPE'] and not ([_ for _ in op_ishapes[0] if _ != 1] == [_ for _ in op_oshapes[0] if _ != 1]):
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        elif opcode in ['ADD', 'SUB', 'MUL', 'DIV', 'SQUARED_DIFFERENCE', "GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL", "EQUAL", "NOT_EQUAL"] and multi_input and singleton: #split if singleton channelwise input
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        else:
            current.append(i)

        prev_op = op
        prev_opcode = opcode

        prev_inputs = op_inputs
        prev_outputs = op_outputs
        i = i + 1

    splits.append(current)
    return splits, errors, specials


# Helper function for optimized splitting
def is_broadcastable_constant(tensor_idx, tensors, buffers):
    """Check if tensor is a constant that can be broadcast (e.g., bias, scale)."""
    if tensor_idx < 0 or tensor_idx >= len(tensors):
        return False
    tensor = tensors[tensor_idx]
    buf_idx = tensor.get('buffer', -1)
    if buf_idx < 0 or buf_idx >= len(buffers):
        return False
    buf = buffers[buf_idx]
    # Has data = is a constant
    if 'data' not in buf:
        return False
    # Check if shape is broadcastable (1D or scalar)
    shape = tensor.get('shape', [])
    if len(shape) <= 1:
        return True
    # Check if all dims except last are 1 (channel-wise broadcast)
    return all(d == 1 for d in shape[:-1])


def get_splits_vbx2_optimized(jname, split_every_op=False, size_config='V1000', engine_op_types=None):
    """
    Optimized graph splitting with improved fusion for better runtime performance.

    KEY CONSTRAINT: CONV_OPS (CONV_2D, DEPTHWISE_CONV_2D, TRANSPOSE_CONV) must ALWAYS
    start a new subgraph - they are "main ops", not "subops". Only one main op per subgraph.

    Valid optimizations (subops that can be fused AFTER a main op):
    1. Activation fusion: RELU/RELU6/etc after CONV stay in same subgraph
    2. Pool fusion: POOL after CONV->ACT chain stays in same subgraph
    3. Element-wise fusion: ADD/MUL with constant after CONV stays in same subgraph
    4. Optimized resize: 2X/4X/8X RESIZE_BILINEAR can be fused
    
    """
    if type(jname) is str:
        graph = json_load(jname)
        assert(len(graph['subgraphs']) == 1)
    else:
        graph = jname

    buffers = graph['buffers']
    subgraph = graph['subgraphs'][0]
    outputs = subgraph['outputs']
    operators = subgraph['operators']
    tensors = subgraph['tensors']
    codes = [_['builtin_code'] for _ in graph['operator_codes']]
    splits = []
    errors = []
    specials = []
    current = []
    current_lut_count = 0
    prev_op = None
    prev_opcode = None
    prev_prev_opcode = None
    prev_inputs = None
    prev_outputs = None

    # === OPERATOR CATEGORIES ===
    # Subops: can be fused AFTER a main op in the same subgraph
    FUSEABLE_ACTIVATIONS = ['RELU', 'RELU6', 'RELU_N1_TO_1', 'LEAKY_RELU', 'PRELU',
                           'HARD_SWISH', 'LOGISTIC', 'TANH', 'ELU', 'GELU']
    POOL_OPS = ['MAX_POOL_2D', 'AVERAGE_POOL_2D']

    # Main ops: MUST start a new subgraph (only one per subgraph)
    CONV_OPS = ['CONV_2D', 'DEPTHWISE_CONV_2D', 'TRANSPOSE_CONV']
    KEY_SUBGRAPH_OPS = ['BATCH_MATMUL', 'FULLY_CONNECTED', 'UNIDIRECTIONAL_SEQUENCE_LSTM',
                        'SPLIT', 'SPLIT_V', 'PACK', 'UNPACK', 'TRANSPOSE', 'SOFTMAX', 'ARG_MAX']

    ELTWISE_OPS = ['ADD', 'SUB', 'MUL', 'DIV', 'SQUARED_DIFFERENCE',
                   'GREATER', 'GREATER_EQUAL', 'LESS', 'LESS_EQUAL', 'EQUAL', 'NOT_EQUAL']

    i = 0
    while i < len(operators):
        op = operators[i]
        if 'builtin_options' in op:
            opts = op['builtin_options']
        opcode = codes[op['opcode_index']]

        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]
        op_ishapes = [tensors[_].get('shape', []) for _ in op_inputs]
        op_oshapes = [tensors[_].get('shape', []) for _ in op_outputs]

        multi_input = is_multi_input(op, tensors, buffers)
        singleton = is_singleton_op(op, tensors)

        connected, forked = True, False
        next_op_ids = []
        next_opcodes = []
        if prev_op is not None:
            connected = any([_ in prev_outputs for _ in op_inputs])
            forked, next_op_ids = is_forked(prev_op, operators, tensors)
            next_opcodes = [codes[operators[_]['opcode_index']] for _ in next_op_ids]

        output = any([_ for _ in op_inputs if _ in outputs])

        pattern_type, pattern = graph_pattern(operators[i:], codes, tensors, buffers)
        pattern = [i + _ for _ in pattern]
        partial_pattern = False
        if len(pattern) and forked:
            partial_pattern = not all([_ in pattern for _ in next_op_ids]) and forked

        # === POST-OPERATION SPLITS (check previous op) ===

        # PAD handling: split after PAD if it expands maps or next is pooling
        if prev_opcode in ['PAD', 'PADV2']:
            if np.sum(get_numpy_data(tensors[prev_op['inputs'][1]], buffers)[-1]) > 0:
                if len(current):
                    splits.append(current)
                current = []
                current_lut_count = 0
            elif len(next_op_ids) and next_opcodes[0] in POOL_OPS:
                if len(current):
                    splits.append(current)
                current = []
                current_lut_count = 0

        # CONCATENATION/STRIDED_SLICE: always split after (same as original)
        if prev_opcode in ['CONCATENATION', 'STRIDED_SLICE']:
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0

        # === PATTERN HANDLING ===
        if len(pattern):
            if pattern_type == "TRANSFORM" or (pattern_type == "LUT" and (current_lut_count or partial_pattern)) or not connected:
                if len(current):
                    splits.append(current)
                current = []
                current_lut_count = 0

            current += pattern
            if pattern_type == "LUT":
                current_lut_count += 1
            i += len(pattern) - 1
            op = operators[i]
            op_outputs = [_ for _ in op['outputs'] if _ != -1]

            # Split after pixel shuffle TRANSFORM if followed by LUT pattern to allow independent tiling
            # Only for ["RESHAPE", "TRANSPOSE", "RESHAPE"] pattern which changes tensor size significantly (PIXEL SHUFFLE)
            if pattern_type == "TRANSFORM":
                pattern_opcodes = [codes[operators[p]['opcode_index']] for p in pattern]
                if pattern_opcodes == ["RESHAPE", "TRANSPOSE", "RESHAPE"]:
                    next_pattern_type, next_pattern = graph_pattern(operators[i+1:], codes, tensors, buffers)
                    if next_pattern_type == "LUT":
                        if len(current):
                            splits.append(current)
                        current = []
                        current_lut_count = 0

        # === FORK/DISCONNECT: Always split (residual patterns have multiple CONVs) ===
        elif forked or output or not connected or split_every_op:
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        # === MAIN OPS: Always start new subgraph ===
        elif opcode in CONV_OPS or opcode in KEY_SUBGRAPH_OPS or opcode == 'CONCATENATION':
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        # === OPTIMIZATION: Activation fusion with preceding CONV ===
        elif opcode in FUSEABLE_ACTIVATIONS and prev_opcode in CONV_OPS and connected:
            # Activation after CONV can be a subop - fuse (don't split)
            current.append(i)

        # === OPTIMIZATION: Pool fusion with CONV->ACT chain ===
        elif opcode in POOL_OPS and prev_opcode in FUSEABLE_ACTIVATIONS and prev_prev_opcode in CONV_OPS and connected:
            # POOL after CONV->ACT can be a subop - fuse (don't split)
            current.append(i)

        # === OPTIMIZATION: Element-wise fusion with preceding CONV ===
        elif opcode in ['ADD', 'MUL'] and prev_opcode in CONV_OPS and connected:
            second_input = op['inputs'][1] if len(op['inputs']) > 1 else -1
            if is_broadcastable_constant(second_input, tensors, buffers):
                # Bias-like ADD or scale-like MUL after CONV - fuse
                current.append(i)
            elif multi_input and singleton:
                # Singleton channelwise broadcast - split
                if len(current):
                    splits.append(current)
                current = []
                current_lut_count = 0
                current.append(i)
            else:
                # Other element-wise - add to current
                current.append(i)

        # === RESIZE_BILINEAR: Optimized scales can fuse ===
        elif opcode == 'RESIZE_BILINEAR':
            # scale = get_scale_factor(operators, tensors, i)
            # if scale not in [(0., 0.)]:
                # Non-optimized scale - needs isolation
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        # === RESHAPE: Split if actual shape change ===
        elif opcode == 'RESHAPE' and not ([_ for _ in op_ishapes[0] if _ != 1] == [_ for _ in op_oshapes[0] if _ != 1]):
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        # === Element-wise with singleton broadcast ===
        elif opcode in ELTWISE_OPS and multi_input and singleton:
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        # === DEFAULT: Add to current subgraph ===
        else:
            current.append(i)

        # Update previous state
        prev_prev_opcode = prev_opcode
        prev_op = op
        prev_opcode = opcode
        prev_inputs = op_inputs
        prev_outputs = op_outputs
        i = i + 1

    splits.append(current)
    return splits, errors, specials


def get_splits(jname, split_every_op=False, engine_op_types=None):
    if type(jname) is str:
        with open(jname) as f:
            graph = json.load(f)
        assert(len(graph['subgraphs']) == 1)
    else:
        graph = jname

    buffers = graph['buffers']
    subgraph = graph['subgraphs'][0]
    outputs = subgraph['outputs']
    operators = subgraph['operators'] # ops = subgraph['operators']
    tensors = subgraph['tensors']
    codes = [_['builtin_code'] for _ in graph['operator_codes']]
    splits = []
    errors = []
    specials = []
    current = []
    current_lut_count = 0
    prev_op = None
    prev_opcode = None
    prev_inputs = None
    prev_outputs = None

    i = 0
    while i < len(operators):
        op = operators[i]
        if 'builtin_options' in op:
            opts = op['builtin_options']
        opcode = codes[op['opcode_index']]

        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]
      
        multi_input = is_multi_input(op, tensors, buffers)
        singleton = is_singleton_op(op, tensors)

        connected, forked = True, False
        if prev_op != None:
            connected = any([_ in prev_outputs for _ in op_inputs])
            forked, next_ops = is_forked(prev_op, operators, tensors)

        output = any([_ for _ in op_inputs if _ in outputs])

        pattern_type, pattern = graph_pattern(operators[i:], codes, tensors, buffers)
        pattern = [i+_ for _ in pattern]
        partial_pattern = False
        if len(pattern) and forked:
            partial_pattern = not all([_ in pattern for _ in next_ops]) and forked

        # If using another engine with limited op support, force a split when the current and prev
        # op mismatch in which engine supports them.
        # Example: Conv -> Pool would normally be 1 subgraph, but not in 3.0.
        engine_split = False
        next_op, next_opcode = None, ''
        if i < len(operators)-1:
            next_op = operators[i+1]
            next_opcode = codes[next_op['opcode_index']]
        if engine_op_types and prev_op != None:
            engine_split = force_split_due_to_engine(engine_op_types, prev_opcode, opcode, next_opcode)

        if prev_opcode in ['CONV_2D', 'DEPTHWISE_CONV_2D'] and 'stride_w' in opts and opts['stride_w'] > 1: #TODO
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0

        if prev_opcode in ['PAD'] and np.sum(get_numpy_data(tensors[prev_op['inputs'][1]], buffers)[-1]) > 0: #expands maps
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0

        if prev_opcode in ['CONCATENATION']:
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0

        if len(pattern):
            if pattern_type == "LUT" and (current_lut_count or partial_pattern):
                if len(current):
                    splits.append(current)
                current = []         
                current_lut_count = 0

            current += pattern
            if pattern_type == "LUT":
                current_lut_count += 1
            i += len(pattern) - 1
            # update outputs to be from last op in pattern
            op = operators[i]
            op_outputs = [_ for _ in op['outputs'] if _ != -1]
            
        elif prev_opcode in ['UNPACK', 'RESHAPE','TRANSPOSE']: #TODO start a new graph after OP
            if len(current):
                splits.append(current)
            current = []
            current.append(i)

        elif forked or output or not connected or split_every_op:
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        elif opcode in ['BATCH_MATMUL', 'CONCATENATION', 'DEPTHWISE_CONV_2D', 'CONV_2D', 'TRANSPOSE_CONV', 'FULLY_CONNECTED', 'UNIDIRECTIONAL_SEQUENCE_LSTM', 'SOFTMAX', 'ARG_MAX', 'CAST', 'TILE', 'SPLIT', 'SPLIT_V', 'PACK', 'UNPACK', 'RESHAPE','TRANSPOSE', 'AVERAGE_POOL_2D', 'MEAN']: # start a new graph before key subgraph OP
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        elif opcode in ['RESIZE_NEAREST_NEIGHBOR']:
            sf_h, sf_w = get_scale_factor(operators, tensors, i)
            if sf_h > 2 or sf_w > 2 or prev_opcode in ['RESIZE_NEAREST_NEIGHBOR']:
                if len(current):
                    splits.append(current)
                current = []
                current_lut_count = 0
            current.append(i)
        elif opcode in ['RESIZE_BILINEAR']:
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        elif opcode in ['ADD', 'SUB', 'MUL', 'DIV', 'SQUARED_DIFFERENCE', "GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL", "EQUAL", "NOT_EQUAL"] and multi_input and singleton: #split if singleton channelwise input
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        elif engine_split:
            if len(current):
                splits.append(current)
            current = []
            current_lut_count = 0
            current.append(i)

        else:
            current.append(i)

        prev_op = op
        prev_opcode = opcode

        prev_inputs = op_inputs
        prev_outputs = op_outputs
        i = i + 1

    splits.append(current)
    return splits, errors, specials

# Enum to represent engine types
from enum import Enum
class Engine(Enum):
    MXP = 0
    NX = 1
    Agnostic = 2

# Iterate over all of the ops in this split to get the split's target engine
# Assert every op in this split is for one engine
def get_engine_from_split_ops(split, operators, codes, engine_op_types) -> Engine:
    current_split_target = None

    # Special case: Pad nodes are usually done as subnodes. However, if there
    # is a case of [Non-NX] -> Pad -> Conv, the Pad is currently done on the NX
    # as part of the Conv. This means there would be a split before the Pad. But
    # Conv also always makes a new split, resulting in a split with Pad on its own.
    # In such cases, put the split on NX.
    if len(split) == 1 and codes[operators[split[0]]['opcode_index']] == 'PAD':
        return Engine.NX

    for op_idx in split:
        op = operators[op_idx]
        opcode = codes[op['opcode_index']]

        # Check where this op will be executed and assign it for the
        # first op, or assert it for the remaining.
        # For ops on either engine (e.g., ADD), do not set the target yet.
        if opcode in engine_op_types.agnostic_op_types:
            pass
        elif opcode in engine_op_types.nx_op_types:
            if current_split_target == None:
                current_split_target = Engine.NX
            else:
                assert current_split_target == Engine.NX
        else:
            if current_split_target == None:
                current_split_target = Engine.MXP
            else:
                assert current_split_target == Engine.MXP

    # Sometimes, every op in the split is agnostic (can run on either engine),
    # e.g., a single Concat or a Pad at the beginning of the graph.
    # These can run on the same engine as the previous or next engine.
    if current_split_target == None:
        return Engine.Agnostic
    return current_split_target

# Simple class to store which operation types are on each engine
class EngineOpTypes:
    def __init__(self, nx_op_types, agnostic_op_types):
        self.nx_op_types = nx_op_types
        self.agnostic_op_types = agnostic_op_types

# Returns 2 lists specifying which splits go on each engine (NX and MXP).
# These are lists of lists with indices that reference into the subgraphs in splits.
#
# Example:
#   Consider a Conv -> SiLU -> Conv -> ReLU -> MaxPool -> MaxPool
#   Assume the Pools will be in separate subgraphs, e.g. because of forked outputs (like in YOLOv5).
#   The splits would be [ [0, 1, 2], [0], [0], [0] ], meaning there are 4 MXP/FIA subgraphs.
#   The engine graphs would be NX = [ [0, 1] ] and MXP = [ [2, 3] ], so 1 engine graph on each.
#   The first 2 Conv are on NX, and 2 Pool on MXP (assumes NX can do SiLU).
def get_splits_per_engine(jname: str, splits: list, engine_op_types: EngineOpTypes) -> (list, list):

    # Read graph from JSON file
    with open(jname) as f:
        graph = json.load(f)
    assert(len(graph['subgraphs']) == 1)

    # Get graph information
    subgraph = graph['subgraphs'][0]
    operators = subgraph['operators']
    tensors = subgraph['tensors']
    codes = [_['builtin_code'] for _ in graph['operator_codes']]

    # Initialize data structures
    engine_graphs_nx = []
    engine_graphs_mxp = []
    current_engine_graph = []
    current_engine_graph_target = None

    # Iterate over all of the splits and assign them to an engine graph
    for split_idx, split in enumerate(splits):

        # Get the target engine for this split
        current_split_target = get_engine_from_split_ops(split, operators, codes, engine_op_types)

        # If Agnostic is returned, it could run on either engine (e.g., PAD, CONCATENATION)
        if current_split_target == Engine.Agnostic:
            # If there is a current engine, put it in that one too
            if current_engine_graph_target:
                current_split_target = current_engine_graph_target
            # If there is no current target (e.g., this is the first split, such as a Pad at the
            # beginning of the graph), decide on the next iteration
            else:
                current_engine_graph.append(split_idx)
                continue

        assert current_split_target in [Engine.NX, Engine.MXP]

        # Assign this split to the correct graph
        if current_engine_graph_target == None:
            current_engine_graph_target = current_split_target
        # If current engine graph type matches current split, add it to current graph
        if current_engine_graph_target == current_split_target:
            current_engine_graph.append(split_idx)
        # Otherwise, end the current engine graph and start a new one
        else:
            if current_engine_graph_target == Engine.NX:
                engine_graphs_nx.append(current_engine_graph)
            else:
                engine_graphs_mxp.append(current_engine_graph)
            current_engine_graph = [split_idx]
            # Update the current graph type
            current_engine_graph_target = current_split_target

    # Add whatever is remaining to the correct graph
    assert current_engine_graph_target == current_split_target
    if current_engine_graph_target == Engine.NX:
        engine_graphs_nx.append(current_engine_graph)
    else:
        engine_graphs_mxp.append(current_engine_graph)

    return engine_graphs_nx, engine_graphs_mxp

# Given a split index and list of engine graphs, check whether this split is
# in one of the engine graphs
def is_split_idx_in_engine_graphs(split_idx, engine_graphs) -> bool:
    for engine_graph in engine_graphs:
        if split_idx in engine_graph:
            return True
    return False

# Given a split index, return which engine it is in
def get_engine_from_split_idx(split_idx, engine_graphs_nx, engine_graphs_mxp) -> Engine:
    nx  = is_split_idx_in_engine_graphs(split_idx, engine_graphs_nx)
    mxp = is_split_idx_in_engine_graphs(split_idx, engine_graphs_mxp)
    assert(nx != mxp) # Must be in exactly one
    return Engine.NX if nx else Engine.MXP

# Helper function for combine_nx_splits
def merge_splits(splits, indices_to_merge):
    # Assert indices are sorted and no duplicates
    assert indices_to_merge == sorted(indices_to_merge)
    assert len(indices_to_merge) == len(set(indices_to_merge))

    # Iterate over splits and extend the merged split while copying the unmerged ones
    new_splits = []
    merged_split = []
    for i, split in enumerate(splits):
        if i in indices_to_merge:
            merged_split.extend(split)  # Extend the merged split
        else:
            new_splits.append(split)    # Copy the unmerged splits

    # Insert the merged split at the index where the merge started
    new_splits.insert(indices_to_merge[0], merged_split)
    return new_splits

# Helper function for combine_nx_splits
def renumber_splits(engine_graphs, merged_split_idx, removed_split_idxs, num_splits_removed):
    for egi in range(len(engine_graphs)):
        for si in range(len(engine_graphs[egi])):
            # Assert the recently removed splits are not in the graph
            assert engine_graphs[egi][si] not in removed_split_idxs
            # Update the split number
            if engine_graphs[egi][si] > merged_split_idx:
                engine_graphs[egi][si] -= num_splits_removed
    return engine_graphs

# For NX, there is no need to make a separate JSON for each subgraph, can combine all the
# NX subgraphs into 2 subgraph and update the splits to reflect this.
#
# Example input:
#   splits = [ [0, 1], [2], [3, 4, 5], [6, 7], [8], [9], [10, 11], [12, 13, 14, 15], [16, 17], [18] ]
#   engine_graphs_nx  = [ [0, 1], [3, 4], [7] ]
#   engine_graphs_mxp = [ [2], [5, 6], [8, 9] ]
# Example output:
#   splits = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9], [10, 11], [12, 13, 14, 15], [16, 17], [18]]
#   engine_graphs_nx  = [[0], [2], [5]]
#   engine_graphs_mxp = [[1], [3, 4], [6, 7]]
def combine_nx_splits(splits, engine_graphs_nx, engine_graphs_mxp):
    split_idx = -1
    while True:
        split_idx += 1
        if split_idx >= len(splits):
            break
        split = splits[split_idx]

        # Check if this split is in NX or MXP
        engine = get_engine_from_split_idx(split_idx, engine_graphs_nx, engine_graphs_mxp)

        # If on MXP, keep the numbering the same
        if engine == Engine.MXP:
            continue
        assert engine == Engine.NX

        # Get the engine graph containing this split
        engine_graph = None
        engine_graph_idx = None
        for egi, eg in enumerate(engine_graphs_nx):
            if split_idx in eg:
                engine_graph_idx = egi
                engine_graph = eg.copy()
                break
        assert engine_graph

        # If it contains only one split, then keep numbering the same (like MXP case)
        if len(engine_graph) == 1:
            continue

        # Merge all the splits into one split
        # Example:
        #   engine_graph = [0, 1, 2], meaning this engine graph contains splits 0, 1, 2
        #   If splits = [ [0, 1], [2, 3], [4], [5, 6] ], then after the merge,
        #   splits = [ [0, 1, 2, 3, 4], [5, 6] ]
        num_splits_before = len(splits)
        splits = merge_splits(splits, engine_graph)

        # Some sanity checks
        num_splits_after = len(splits)
        assert num_splits_after < num_splits_before
        num_splits_removed = num_splits_before - num_splits_after
        assert num_splits_removed == len(engine_graph) - 1

        # Update this NX engine graph
        merged_split_idx = engine_graph[0]
        removed_split_idxs = engine_graph[1:]
        engine_graphs_nx[engine_graph_idx] = [merged_split_idx]

        # Update the NX and MXP engine graphs to renumber all later splits
        engine_graphs_nx  = renumber_splits(engine_graphs_nx,  merged_split_idx, removed_split_idxs, num_splits_removed)
        engine_graphs_mxp = renumber_splits(engine_graphs_mxp, merged_split_idx, removed_split_idxs, num_splits_removed)

    # Final check that every NX engine graph is 1 split now
    for eg in engine_graphs_nx:
        assert len(eg) == 1

    return splits, engine_graphs_nx, engine_graphs_mxp


def gen_signature(operators, tensors, buffers):
    signature = [{'inputs':None, 'outputs':None, 'signature_key':'serving_default', 'subgraph_index':0}]

    all_input_tensors = []
    all_output_tensors = []

    for op in operators:
        all_input_tensors += op['inputs']
        all_output_tensors += op['outputs']
        
    input_tensors = [_ for _ in all_input_tensors if _ not in all_output_tensors]
    output_tensors = [_ for _ in all_output_tensors if _ not in all_input_tensors]

    input_names = []
    input_idx = []
    for io in input_tensors:
        if io != -1:
            t = tensors[io]
            buf = buffers[t['buffer']]
            if 'data' not in buf:
                input_names.append(t['name'])
                input_idx.append(io)

    output_names = []
    output_idx = []
    for io in output_tensors:
        if io != -1:
            t = tensors[io]
            buf = buffers[t['buffer']]
            if 'data' not in buf:
                output_names.append(t['name'])
                output_idx.append(io)

    graph_inputs = [{'name': n, 'tensor_index': i} for n, i in zip(input_names, input_idx)]
    graph_outputs = [{'name': n, 'tensor_index': i} for n, i in zip(output_names, output_idx)]

    signature[0]['inputs'] = graph_inputs
    signature[0]['outputs'] = graph_outputs

    return signature


def clear_unused(operators, tensors, buffers, forced_shape=None):
    stensors, sbuffers = {}, {}

    # remove unused buffers/tensors
    used_buffers = []
    used_tensors = []
    for op in operators:
        
        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]
        op_intermediates = []

        if 'intermediates' in op:
            op_intermediates = [_ for _ in op['intermediates'] if _ != -1]

        for io in op_inputs + op_outputs + op_intermediates:
            used_buffers.append(tensors[io]['buffer'])
            used_tensors.append(io)

    sbuffers = []
    stensors = []

    for b in range(len(buffers)):
        if b not in used_buffers:
            sbuffers.append({})
        else:
            sbuffers.append(buffers[b])

    for t in range(len(tensors)):
        if t not in used_tensors:
            stensors.append({})
        else:
            if not forced_shape is None:
                buf = buffers[tensors[t]['buffer']]
                if 'data' not in buf:
                    tensors[t]['shape'] = forced_shape
            stensors.append(tensors[t])

    return stensors, sbuffers


def gen_subgraph(idir, odir, jname, splits, vnnx, sim, forced_shape=None):

    g = json_load(jname)

    buffers = g['buffers']
    subgraph = g['subgraphs'][0]
    operators = subgraph['operators']
    tensors = subgraph['tensors']

    show_progress = False
    if len(splits) > 100:
        show_progress = True

    json_subgraphs = []
    for s, split_indices in tqdm(enumerate(splits), total=len(splits), disable=(not show_progress)):
        subgraph_jname = os.path.join(odir, os.path.basename(jname).replace('.json', '.{}.json'.format(s)))
        json_subgraphs.append(subgraph_jname)

        selected_operators = []
        for i in split_indices:
            selected_operators.append(operators[i])

        sg = create_graph(selected_operators, tensors, buffers, g['operator_codes'], forced_shape)
        
        json_dump(sg, subgraph_jname)
        json2tflite(subgraph_jname)

        if vnnx:
            subgraph_tflite = subgraph_jname.replace('.json', '.tflite')
            subgraph_vnnx = subgraph_jname.replace('.json', '.vnnx')
            cmd = 'vnnx_compile -s {} -c {} -t {} -o {}'.format(os.environ.get('HW_SIZE'), os.environ.get('HW_COMP'),subgraph_tflite, subgraph_vnnx)
            subprocess.run(cmd, shell=True, check=True)

            if sim:
                cmd = 'python -m vbx.sim {} -d'.format(subgraph_vnnx)
                subprocess.run(cmd, shell=True, check=True)

    return json_subgraphs

# Given list of subgraphs, and engine graphs for nx and mxp,
# write the nx engine graphs to json files
def write_nx_engine_graphs(json_subgraphs, nx_dirname, engine_graphs_nx, engine_graphs_mxp) -> None:

    # Remove previous nx_engine dir and create a new one
    nx_engine_graphs = 0

    # Iterate over all splits
    assert os.path.isdir(nx_dirname)
    for split_idx, j in enumerate(json_subgraphs):
        # Check if this split is on NX or MXP
        engine = get_engine_from_split_idx(split_idx, engine_graphs_nx, engine_graphs_mxp)

        # If MXP, do not write JSON for this engine graph
        if engine == Engine.MXP:
            continue

        # For NX, write subgraphs to disk for now so NX can use later
        # Eventually can call NX directly
        assert engine == Engine.NX
        with open(j) as f:
            split = json.load(f)
        engine_graph_fname = os.path.join(nx_dirname, str(nx_engine_graphs) + ".json")
        nx_engine_graphs += 1
        with open(engine_graph_fname, 'w') as json_file:
            json.dump(split, json_file, indent=4)

# Write the graph to a .json file as well as a list of which ops are on MXP
def write_json_graph_and_mxp_ops_for_nx(jname: str, nx_dirname: str, engine_graphs_mxp: list[list],\
    splits: list[list]) -> None:

    # Read full graph from JSON
    with open(jname) as f:
        graph = json.load(f)
    
    # Write the JSON graph to a file
    assert os.path.isdir(nx_dirname)
    graph_fname = os.path.join(nx_dirname, os.path.basename(jname))
    with open(graph_fname, 'w') as json_file:
        json.dump(graph, json_file, indent=4)

    # Write all the MXP ops to a file
    mxp_ops_fname = os.path.join(nx_dirname, "mxp_op_idx.txt")
    with open(mxp_ops_fname, 'w') as txt_file:
        for eg in engine_graphs_mxp:
            for split_idx in eg:
                for op in splits[split_idx]:
                    txt_file.write(str(op) + "\n")

def write_json_per_nx_engine_graph(jname, tmp_dir, splits, nx_dirname,\
    engine_graphs_nx, engine_graphs_mxp):

    # For separate JSON, work on copies because these are modified in place below
    splits_copy  = copy.deepcopy(splits)
    engine_graphs_nx_copy  = copy.deepcopy(engine_graphs_nx)
    engine_graphs_mxp_copy = copy.deepcopy(engine_graphs_mxp)

    # For NX, combine all splits in an engine graph to a single split, so each NX engine graph
    # is one split instead of multiple. This is used to make one JSON per engine graph for NX.
    #
    # Note that the MXP compiler will not see these merged subgraphs for NX, it will still see the
    # original smaller subgraphs. This is because merging into larger subgraphs results in
    # set_io_nodes failing to find graph output tensors inside the merged NX subgraphs.
    merged_splits, merged_engine_graphs_nx, merged_engine_graphs_mxp = combine_nx_splits(splits_copy,
        engine_graphs_nx_copy, engine_graphs_mxp_copy)

    # Write temporary NX files like previous calls to gen_subgraph
    nx_subdir = os.path.join(tmp_dir, 'subgraphs_nx_merged')
    if not os.path.exists(nx_subdir):
        os.mkdir(nx_subdir)
    json_subgraphs_merged_nx = gen_subgraph(nx_subdir, nx_subdir, jname, merged_splits, False, False)

    # Write the JSON for each graph engine
    # It might be possible to do this more efficiently. E.g., instead of load and write the
    # NX JSON files, just copy them from their location in json_subgraphs_merged_nx.
    nx_unused_dir = os.path.join(nx_dirname, 'partitions')
    os.mkdir(nx_unused_dir)
    write_nx_engine_graphs(json_subgraphs_merged_nx, nx_unused_dir, merged_engine_graphs_nx, merged_engine_graphs_mxp)

def get_op_list_from_engine_graphs(splits, engine_graphs_mxp):
    mxp_ops = []
    for eg in engine_graphs_mxp:
        for split_idx in eg:
            for op_idx in splits[split_idx]:
                mxp_ops.append(op_idx)
    return mxp_ops

def get_op_list_from_splits(splits):
    all_ops = []
    for split in splits:
        for op_idx in split:
            all_ops.append(op_idx)
    return all_ops

# Get a list of tflite op indexes which are on the MXP and are part of
# pre-processing (at the inputs of the graph).
#
# Note: Currently this assumes that MXP ops processing the graph inputs are part
# of the pre-processing, i.e., there is not an MXP subgraph later in the graph
# which has inputs from both NX as well as a graph input. If this is not the
# case, this DFS can be updated to search in both directions (see implementation
# in get_mxp_postprocessing_ops_forwards_backwards)
def get_mxp_preprocessing_ops(jname, mxp_ops):
    # Read the graph from JSON
    with open(jname) as f:
        graph = json.load(f)
    subgraph = graph['subgraphs'][0]

    # Start with the ops that consume the graph inputs
    visited = []
    input_tensors = subgraph['inputs']
    for input_tensor_idx in input_tensors:
        for op_idx, op in enumerate(subgraph['operators']):
            if input_tensor_idx in op['inputs']:
                if op_idx in mxp_ops:
                    visited.append(op_idx)

    # Search forwards until no nodes left to visit
    mxp_preprocessing_ops = set()
    while visited:
        op_idx = visited.pop()
        if op_idx in mxp_preprocessing_ops:
            continue
        mxp_preprocessing_ops.add(op_idx)

        # For each output, find its consumer ops and add them to visited if on MXP
        outputs = subgraph['operators'][op_idx]['outputs']
        for output_tensor_idx in outputs:
            for consumer_op_idx, consumer_op in enumerate(subgraph['operators']):
                if output_tensor_idx in consumer_op['inputs']:
                    if consumer_op_idx in mxp_ops and consumer_op_idx not in mxp_preprocessing_ops:
                        visited.append(consumer_op_idx)
                    else:
                        break # Stop tracing if op not on MXP

    return list(mxp_preprocessing_ops)

# Get a list of tflite op indexes which are on the MXP and are part of
# post-processing (at the outputs of the graph).
def get_mxp_postprocessing_ops(jname, mxp_ops):
    # Read the graph from JSON
    with open(jname) as f:
        graph = json.load(f)
    subgraph = graph['subgraphs'][0]

    # Start with the ops that produce the graph outputs
    visited = []
    output_tensors = subgraph['outputs']
    for output_tensor_idx in output_tensors:
        for op_idx, op in enumerate(subgraph['operators']):
            if output_tensor_idx in op['outputs']:
                if op_idx in mxp_ops:
                    visited.append(op_idx)

    # Search backwards until no nodes left to visit
    mxp_postprocessing_ops = set()
    while visited:
        op_idx = visited.pop()
        if op_idx in mxp_postprocessing_ops:
            continue
        mxp_postprocessing_ops.add(op_idx)

        # For each input, find its producer op and add it to visited if on MXP
        inputs = subgraph['operators'][op_idx]['inputs']
        for input_tensor_idx in inputs:
            for producer_op_idx, producer_op in enumerate(subgraph['operators']):
                if input_tensor_idx in producer_op['outputs']:
                    if producer_op_idx in mxp_ops and producer_op_idx not in mxp_postprocessing_ops:
                        visited.append(producer_op_idx)
                    else:
                        break # Stop tracing if op not on MXP

    return list(mxp_postprocessing_ops)

# Get a list of tflite op indexes which are on the MXP and are part of
# post-processing (at the outputs of the graph).
#
# Note this DFS searches in both directions to find cases like this:
#
#               ...
#                │
#              ┌─▼─┐
#              │MXP│
#              └─┬─┘
#         ┌──────┤
#         │      │
#       ┌─▼─┐    │
#       │NX │    │
#       └┬──┘    │
#        │ ┌─────┘
#        │ │
#       ┌▼─▼┐
#       │MXP│
#       └─┬─┘
#         │
#         ▼
#       Output
#
# Here the bottom MXP op is part of post-processing and does not need to be sent
# to NX, but the top MXP op produces an output used by NX so it is not part of
# the post-processing.
#
# This was originally added for YOLOv8/9 where the final nodes in the graph
# contain a Conv2D which creates this pattern, and Conv2D is usually done on NX.
# But that Conv2D is actually very simple (1x1 and single-channel output) so it
# is better to do on the RISC-V where there is conditional execution and it can
# be pipelined with the next input. Therefore, currently there is no example
# where the DFS in both directions is needed, although it may be in the future.
def get_mxp_postprocessing_ops_forwards_backwards(jname, mxp_ops):
    # Read the graph from JSON
    with open(jname) as f:
        graph = json.load(f)
    subgraphs = graph['subgraphs']
    operators = subgraphs[0]['operators']
    tensors = subgraphs[0]['tensors']

    # Mapping from tensor index to the operator producing it
    tensor_to_producer = {}
    for idx, op in enumerate(operators):
        for tensor_idx in op['outputs']:
            tensor_to_producer[tensor_idx] = idx

    # Mapping from tensor index to the list of operators consuming it
    tensor_to_consumers = {}
    for idx, op in enumerate(operators):
        for tensor_idx in op['inputs']:
            if tensor_idx not in tensor_to_consumers:
                tensor_to_consumers[tensor_idx] = []
            tensor_to_consumers[tensor_idx].append(idx)

    # Forward DFS to check if any path leads to a non-mxp operator
    def forward_dfs(tensor_idx, visited_tensors):
        if tensor_idx in visited_tensors:
            return True  # Already visited
        visited_tensors.add(tensor_idx)

        if tensor_idx in tensor_to_consumers:
            for consumer_op_idx in tensor_to_consumers[tensor_idx]:
                if consumer_op_idx not in mxp_ops:
                    return False  # Found a non-mxp operator
                # Check further downstream tensors
                for output_tensor_idx in operators[consumer_op_idx]['outputs']:
                    if not forward_dfs(output_tensor_idx, visited_tensors):
                        return False
        return True

    # Backward DFS to find valid mxp operators
    def backward_dfs(tensor_idx, visited_ops):
        if tensor_idx not in tensor_to_producer:
            return True  # No producer, continue
        producer_op_idx = tensor_to_producer[tensor_idx]

        if producer_op_idx in visited_ops or producer_op_idx not in mxp_ops:
            return True  # Already visited or not a mxp op

        # Check all forward paths from this operator's outputs
        for output_tensor_idx in operators[producer_op_idx]['outputs']:
            if not forward_dfs(output_tensor_idx, set()):
                return False  # Found an invalid path

        visited_ops.add(producer_op_idx)

        # Continue DFS for all inputs of the current operator
        for input_tensor_idx in operators[producer_op_idx]['inputs']:
            if not backward_dfs(input_tensor_idx, visited_ops):
                return False  # Stop if any input path is invalid

        return True

    # Search backwards from the graph outputs
    visited_ops = set()
    for tensor_idx in set(subgraphs[0]['outputs']):
        backward_dfs(tensor_idx, visited_ops)

    return list(visited_ops)

def update_mxp_ops_with_removed_ops(mxp_ops, removed_ops):
    num_removed = 0
    num_to_remove = len(removed_ops)
    while num_removed < num_to_remove:
        # Pop the next one from the front
        removed_op_idx = removed_ops.pop(0)
        # If this is on MXP, remove it from MXP ops
        if removed_op_idx in mxp_ops:
            mxp_ops.remove(removed_op_idx)
        # Now renumber the remaining ops that come after this one
        # by subtracting 1 from their op index
        for i in range(len(removed_ops)):
            if removed_ops[i] > removed_op_idx:
                removed_ops[i] -= 1
        for i in range(len(mxp_ops)):
            if mxp_ops[i] > removed_op_idx:
                mxp_ops[i] -= 1
        num_removed += 1

# Write the graph to JSON without MXP nodes for pre and post processing
# Also write a .txt file of the op IDs inside the remaining graph which are on MXP
#
# Note: Another way of removing these nodes for NX would be to send the entire graph,
# and then remove the pre/post-processing in the NX TFLite parser before creating NX IR.
# This solution of truncating the graph before sending was chosen instead because it
# makes more sense that the handoff from MXP -> NX compiler should not include extra
# nodes which NX does not care about. But it requires the ops on MXP to be renumbered
# to match the new graph (done below). If this is too complicated, then this node removal
# can be done in the NX TFLite parser, although then other changes will be needed to not
# use the TFLite graph inputs/outputs as the IR inputs/outputs but instead the new tensors
# that become inputs/outputs after the pre/post-processing nodes are removed.
def write_json_graph_without_mxp_pre_post_processing(jname, tmp_dir, splits, nx_dirname, engine_graphs_mxp):

    # Get a list of all ops and all ops on MXP
    # The engine graph concept (list of lists) is not useful and can be removed.
    mxp_ops = get_op_list_from_engine_graphs(splits, engine_graphs_mxp)
    all_ops = get_op_list_from_splits(splits)

    # Get a list of ops to remove which are from MXP preprocessing
    removed_ops = get_mxp_preprocessing_ops(jname, mxp_ops)

    # Also get MXP post-processing ops.
    # Note this is usually not needed because the graph is cut for the RISC-V.
    removed_ops += get_mxp_postprocessing_ops_forwards_backwards(jname, mxp_ops)

    # If nothing to remove, no need to update files
    if not removed_ops:
        return

    # For every removed op, remove it from all_ops
    for removed_op_idx in removed_ops:
        if removed_op_idx in all_ops:
            all_ops.remove(removed_op_idx)

    # For every removed op, remove it from mxp_ops too,
    # but also renumber the ops to match the new graph
    update_mxp_ops_with_removed_ops(mxp_ops, removed_ops)

    # Use gen_subgraph to write a new .json for the truncated graph
    nx_subdir = os.path.join(tmp_dir, 'subgraphs_no_mxp_pre_post_proc')
    if not os.path.exists(nx_subdir):
        os.mkdir(nx_subdir)

    # If the graph is entirely on MXP, no need to write JSON for NX
    if all_ops:
        json_subgraphs_no_pre_post = gen_subgraph(nx_subdir, nx_subdir, jname, [all_ops], False, False)
        assert len(json_subgraphs_no_pre_post) == 1
        jname_truncated = json_subgraphs_no_pre_post[0]
        
        # Read graph from JSON and write it to a file.
        # It might be possible to do this more efficiently. E.g., instead of load and write the
        # NX JSON files, just copy them from their location in json_subgraphs_merged_nx.
        with open(jname_truncated) as f:
            graph = json.load(f)
        assert os.path.isdir(nx_dirname)
        graph_fname = os.path.join(nx_dirname, os.path.basename(jname))
        with open(graph_fname, 'w') as json_file:
            json.dump(graph, json_file, indent=4)

    # Write all the MXP ops to a file too
    mxp_ops_fname = os.path.join(nx_dirname, "mxp_op_idx.txt")  # mxp_op_idx_truncated.txt
    with open(mxp_ops_fname, 'w') as txt_file:
        for op in mxp_ops:
            txt_file.write(str(op) + "\n")

def delete_and_remake_dir(dirname):
    import shutil
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)

def write_json_for_nx(jname, tmp_dir, splits, engine_graphs_nx, engine_graphs_mxp):

    # Make an nx_dir for the final engine graph JSON files
    nx_dirname = os.path.join(tmp_dir, "nx_engine")
    delete_and_remake_dir(nx_dirname)

    # Also make a sync dir
    #nx_sync_dir = os.path.join(nx_dirname, 'sync')
    #os.mkdir(nx_sync_dir)

    # Write the entire graph to JSON, along with a list of which ops are on MXP.
    write_json_graph_and_mxp_ops_for_nx(jname, nx_dirname, engine_graphs_mxp, splits)

    # Write the JSON graph without the pre and post processing ops (on MXP).
    write_json_graph_without_mxp_pre_post_processing(jname, tmp_dir, splits, nx_dirname, engine_graphs_mxp)

    # This next function is unused and can be deleted, along with the helper functions
    # it calls. It writes each individual NX engine graph to a separete JSON file.
    # This may be useful for testing, but saving the entire graph as a single JSON and
    # also including a list of ops on MXP is more accurate because there can be connections
    # from one NX engine graph to another, e.g., two NX engine graphs does not mean two
    # completely separate graphs.
    write_json_per_nx_engine_graph(jname, tmp_dir, splits, nx_dirname,\
        engine_graphs_nx, engine_graphs_mxp)

def generate_split_graphs(tflite_model, dir, split_every_op=False, cuts=None, vnnx=False, sim=False, compression_vbx='ncomp', size_config='V1000', output_name=None, gen_subgraphs_from_splits=False):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    jname = os.path.join(dir, os.path.basename(tflite_model).replace('.tflite','.json'))
    
    cmd = 'flatc -t --strict-json --defaults-json -o {} {} -- {}'.format(dir, schema_path, tflite_model)
    subprocess.run(shlex.split(cmd))

    #update operator builtin_code
    schema = json_load(schema_path.replace('.fbs', '.json'))
    operators = schema["definitions"]["tflite_BuiltinOperator"]["enum"]
    with open(jname) as f:
        graph = json.load(f)
        for i,g in enumerate(graph['operator_codes']):
            if 'deprecated_builtin_code' in g and not g['deprecated_builtin_code'] == 127:
                graph['operator_codes'][i]['builtin_code'] = operators[g['deprecated_builtin_code']]
    json_dump(graph, jname)

    subdir = os.path.join(dir, 'subgraphs') 
    if not os.path.exists(subdir):
        os.mkdir(subdir)

    # errors.append([i,op,opcode])
    if compression_vbx == 'ucomp':
        nx_op_types = ['CONV_2D', 'MAX_POOL_2D']
        agnostic_op_types = ['QUANTIZE', 'RESIZE_NEAREST_NEIGHBOR', 'PAD', 'ADD', 'CONCATENATION',\
                             'STRIDED_SLICE', 'LOGISTIC', 'MUL', 'SPLIT', 'CAST', 'GATHER']
        engine_op_types = EngineOpTypes(nx_op_types, agnostic_op_types)

        # Pass in nx_op_types in case need to add additional splits at engine graph boundaries
        splits, error_ops, special_ops = get_splits(jname, split_every_op, engine_op_types)
    else:
        splits, error_ops, special_ops = get_splits_vbx2_optimized(jname, split_every_op, size_config=size_config)

    if len(error_ops)>0: 
        errors_dir = os.path.join(os.path.join(os.getcwd(), 'unsupported_ops'))
        if os.path.exists(errors_dir):
            shutil.rmtree(errors_dir)
            os.mkdir(errors_dir)
        else:
            os.mkdir(errors_dir)
        error_locations = [[_[0]] for _ in error_ops]
        
        gen_subgraph(subdir, errors_dir, jname, error_locations, False)
        for i, (error_op, fname) in enumerate(zip(error_ops, sorted(glob.glob(os.path.join(errors_dir, '*.tflite'))))):
            id, op, op_code = error_op[:3]
            dst = fname.replace('.{}.tflite'.format(i), '.{}.{}.tflite'.format(op_code,id))
            shutil.move(fname, dst)
        for tmp in glob.glob(os.path.join(errors_dir, '*.json')):
            os.remove(tmp)
        with exception_catcher(error_ops):
            assert(0)

    if len(special_ops):
        special_dir = os.path.join(subdir, 'special')
        if not os.path.exists(special_dir):
            os.makedirs(special_dir)
        gen_subgraph(subdir, special_dir, jname, special_ops, False) #, [256,])
    
    if gen_subgraphs_from_splits:
        json_subgraphs = gen_subgraph(subdir, subdir, jname, splits, vnnx, sim)

        return json_subgraphs, None

    if compression_vbx == 'ucomp':
        # Map each split to an engine
        # TODO: Currently these are only being used as lists of splits. The engine graph concept
        # (list of lists) is not used anywhere. If it will never be needed, can simplify this.
        engine_graphs_nx, engine_graphs_mxp = get_splits_per_engine(jname, splits, engine_op_types)
        
        # For NX, write the required engine graphs to JSON files
        write_json_for_nx(jname, dir, splits, engine_graphs_nx, engine_graphs_mxp)

        # For MXP, return all subgraphs but also which are on NX.
        # It may be possible to just return the subgraphs which are on MXP, rather than return all
        # subgraphs. But currently this would result in set_io_nodes failing to find the graph
        # input and output tensors which are only in the NX subgraphs.
        return splits, engine_graphs_nx

    return splits, None


def join_graph(graphs):
    graph = copy.deepcopy(graphs[0])

    subgraph_inputs = []
    subgraph_outputs = []
    for g in graphs:
        subgraph_inputs += g['signature_defs'][0]['inputs']
        subgraph_outputs += g['signature_defs'][0]['outputs']
    graph_inputs = [_ for _ in subgraph_inputs if _ not in subgraph_outputs]
    graph_outputs = [_ for _ in subgraph_outputs if _ not in subgraph_inputs]

    operators = []
    for g in graphs:
        operators += g['subgraphs'][0]['operators']

    graph['subgraphs'][0]['operators'] = operators
    graph['signature_defs'][0]['inputs'] = graph_inputs
    graph['signature_defs'][0]['outputs'] = graph_outputs
    graph['subgraphs'][0]['inputs'] = [_['tensor_index'] for _ in graph_inputs]
    graph['subgraphs'][0]['outputs'] = [_['tensor_index'] for _ in graph_outputs]

    return graph


def generate_join_graphs(tflite_model, dir, debug=False):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    json_subgraphs = []
    for subgraph_jname in sorted([_ for _ in os.listdir(dir) if '.json' in _]):
        with open(os.path.join(dir, subgraph_jname)) as f:
            json_subgraphs.append(json.load(f))

    graph = join_graph(json_subgraphs)
    jname = os.path.join(os.path.dirname(tflite_model), os.path.basename(tflite_model).replace('.tflite','.json'))
    with open(jname, 'w') as f:
        json.dump(graph, f)

    if debug:
        cmd = 'flatc -b --strict-json --defaults-json -o {} {} {}'.format(os.path.dirname(tflite_model), schema_path, jname)
        subprocess.run(shlex.split(cmd))

    return graph


def cut():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite", type=existing_file)
    parser.add_argument("-c", "--cuts", type=int, nargs='*')
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    tmp_tflite = os.path.join(tmp_dir, os.path.basename(args.tflite))
    shutil.copyfile(args.tflite, tmp_tflite)

    generate_cut_graphs(tmp_tflite, args.cuts, args.vnnx, args.sim)
    graphs = list(glob.glob(os.path.join(tmp_dir, "*.tflite")))
    graphs += list(glob.glob(os.path.join(tmp_dir, "*.vnnx")))
    for src in graphs:
        if src != tmp_tflite:
            dst = os.path.join(os.path.dirname(args.tflite),os.path.basename(src))
            shutil.copyfile(src, dst)
    tmp_dir_obj.cleanup()

    args = parser.parse_args()

    graph, dir_obj = load_graph(args.tflite)

    if args.post_process_layer in ['PIXEL_VOC', 'PIXEL_COCO', 'PIXEL_CITYSCAPES', 'PIXEL_DEPTH']:
        tgraph = postprocess_graph(graph, args.post_process_layer.split('_')[-1], args.opacity, args.height, args.width)
    else:
        tgraph = transform_graph(graph, ['YOLO_ARG_MAX'], args.verbose)
    save_graph(args.tflite.replace('.tflite', '.post.tflite'), dir_obj, tgraph, copy=True)

    if not dir_obj is None:
        dir_obj.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite", type=existing_file)
    parser.add_argument("dir")
    parser.add_argument("-j", "--join", action='store_true')
    parser.add_argument("-d", "--debug", action='store_true')
    parser.add_argument("-e", "--split-every-op", action='store_true')
    parser.add_argument("-c", "--cuts", type=int, nargs='*')
    parser.add_argument("--vnnx", action='store_true')
    parser.add_argument("--sim", action='store_true')
    args = parser.parse_args()

    if not args.join:
        generate_split_graphs(args.tflite, args.dir, args.split_every_op, args.cuts, args.vnnx, args.sim, gen_subgraphs_from_splits=True)
    else:
        generate_join_graphs(args.tflite, args.dir, args.debug)

if __name__ == "__main__":
    main()
