import argparse
from .utils import existing_file, existing_dir
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
from .transform_tflite import get_splits2

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


def forks(op0, op0_inputs, op0_outputs, operators, tensors):
    if len(op0_outputs) == 2:
        return True

    count = 0
    tensor = op0_outputs[0]
    for i, op in enumerate(operators):
        if op != op0:
            op_inputs = [_ for _ in op['inputs'] if _ != -1]
            for t in op_inputs:
                if t == tensor:
                    count += 1

    return count > 1


def in_inputs(op0, op):

    idx_input_tensors = [_ for _ in op0['inputs'] if _ != -1]
    op_outputs = [_ for _ in op['outputs'] if _ != -1]
    for o in op_outputs:
        if o in idx_input_tensors:
            return True

    return False


def is_input(idx, operators, cuts, prev_nodes=[]):
    nodes = [idx]

    if idx != -1:
        iop = operators[idx]

        for i, op in enumerate(operators):
            if i not in prev_nodes:
                if in_inputs(iop, op) and i not in cuts:
                    nodes.append(i)
                    nodes += is_input(i, operators, cuts, nodes+prev_nodes)

    return nodes


def get_cuts(jname, cuts, include_outputs=True):
    with open(jname) as f:
        graph = json.load(f)
    # assert(len(graph['subgraphs']) == 1)

    buffers = graph['buffers']
    subgraph = graph['subgraphs'][0]
    outputs = subgraph['outputs']
    operators = subgraph['operators']
    if include_outputs:
        for i, op in enumerate(operators):
            for o in op['outputs']:
                if o in outputs:
                    cuts.append(i)
    tensors = subgraph['tensors']
    codes = [_['builtin_code'] for _ in graph['operator_codes']]
    splits = []
    current = []
    prev_op = None
    prev_opcode = None
    prev_inputs = None
    prev_outputs = None

    cut_groups = {}
    for c in cuts:
        cut_groups[c] = set(is_input(c, operators, cuts))

    while len(cut_groups.keys()):
        modified = False
        keys = list(cut_groups.keys())
        k0 = keys[0]

        for k in keys:
            if k != k0:
                s0 = cut_groups[k0]
                s = cut_groups[k]

                if len(s) + len(s0) > len(s0.union(s)): #shared nodes
                    cut_groups[k0] = s0.union(s)
                    cut_groups.pop(k)
                    modified = True

        if not modified:
            splits.append(sorted(list(cut_groups.pop(k0))))

    return splits

def get_unique_values(values):
    uniq_values = []

    for v in values:
        if type(v) == str:
            uniq_values.append(v.upper())
        else:
            uniq_values.append(v)

    return list(set(uniq_values))


def get_negative_axis(axis, dims=4):
    if(axis>=0):
        return axis-dims

    return axis

def check_valid(valid, op_code, op, tensors):
    input_param = []
    error_param = []
    param_value = []

    if not op_code in valid:
        return False, error_param, param_value, input_param 

    if op_code == "SPLIT":
        i_tensor = tensors[op['inputs'][1]]
    elif op_code =="TRANSPOSE_CONV":
        i_tensor = tensors[op['inputs'][2]]
    else:
        i_tensor = tensors[op['inputs'][0]]
    idims = len(i_tensor['shape'])
    opts = None
    if 'builtin_options' in op:
        opts = op['builtin_options']
    
    if i_tensor is not None: 
        if "type" in i_tensor:
            values = valid[op_code]["type"] 
            dtype = i_tensor["type"]  
            if dtype.upper() not in values:
                error_param.append(get_unique_values(values))
                param_value.append("Input data type")
                input_param.append(dtype)
                return False, error_param, param_value, input_param
        
    if opts is not None:    
        if "axis" in opts:
            values = valid[op_code]["axis"]
            axis = get_negative_axis(opts["axis"], idims)
            if axis not in values:
                error_param.append(get_unique_values(values))
                param_value.append("axis")
                input_param.append(axis)
                return False, error_param, param_value, input_param
                
        if "fused_activation_function" in opts: 
            values = valid[op_code]["fused_activation_function"]
            faf = opts["fused_activation_function"]
            if type(faf) == str:
                faf = faf.upper()
            if faf not in values:
                error_param.append(get_unique_values(values))
                param_value.append("fused_activation_function")
                input_param.append(faf)
                return False, error_param, param_value, input_param
                
        if "mode" in opts:  
            values = valid[op_code]["mode"]
            mode = opts["mode"]

            if type(mode) == str:
                mode = mode.upper()

            if mode not in values:
                error_param.append(get_unique_values(values))
                param_value.append("mode")
                input_param.append(mode)
                return False, error_param, param_value, input_param
                
        if "dim" in opts: 
            dims = valid[op_code]
            values = dims["dim"]
            dim = get_negative_axis(opts["mode"], idims)
            if dim not in values:
                error_param.append(get_unique_values(values))
                param_value.append("dim")
                input_param.append(dim)
                return False, error_param, param_value, input_param
        
        if "padding" in opts: 
            values = valid[op_code]["padding"]
            pad = opts["padding"]

            if type(pad) == str:
                pad = pad.upper()
                
            if pad not in values:
                error_param.append(get_unique_values(values))
                param_value.append("padding")
                input_param.append(pad)
                return False, error_param, param_value, input_param

    return True, error_param, param_value, input_param


def get_scale_factor(operators, tensors, idx):
    op = operators[idx]
    ishape = tensors[op['inputs'][0]]['shape']
    oshape = tensors[op['outputs'][0]]['shape']
    return oshape[-3]/ishape[-3], oshape[-2]/ishape[-2]


def reshape_pattern(operators, codes, tensors, buffers, idx, opcode):
    patterns = []

    prev_op = None
    prev_inputs = None
    prev_outputs = None

    max_idx = len(operators)
    while idx < max_idx:
        op = operators[idx]
        opcode = codes[op['opcode_index']]
        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]

        ibuffers = []
        for _ in op_inputs:
            if 'buffer' in tensors[_]:
                ibuffers += [buffers[tensors[_]['buffer']]]
        multi_input = len(ibuffers) > 1 and not any(['data' in _ for _ in ibuffers])
        filters = [_ for _ in op_inputs if 'data' in buffers[tensors[_]['buffer']]]

        next_op, next_opcode = None, ''
        if idx < max_idx -1:
            next_op = operators[idx+1]
            next_opcode = codes[next_op['opcode_index']]
        next_next_op, next_next_opcode = None, ''
        if idx < max_idx -2:
            next_next_op = operators[idx+2]
            next_next_opcode = codes[next_next_op['opcode_index']]

        connected = True
        if prev_op != None:
            connected = any([_ in prev_outputs for _ in op_inputs])

        forked = False
        if prev_op != None:
            forked = forks(prev_op, prev_inputs, prev_outputs, operators, tensors)


        if not connected:
            break
        
        #PIXEL_SHUFFLE pattern
        elif opcode == "RESHAPE" and next_opcode == "TRANSPOSE" and next_next_opcode =="RESHAPE": 
            patterns.append(idx)
            patterns.append(idx+1)
            patterns.append(idx+2)
            idx += 2

        elif forked:
            break
        
        else:
            break

        idx += 1

        prev_op = op
        prev_inputs = op_inputs
        prev_outputs = op_outputs
        
    return patterns


def lut_pattern(operators, codes, tensors, buffers, idx, opcode):
    if not OPTIMIZED_WITH_LUT:
        return []

    patterns = []

    prev_op = None
    prev_inputs = None
    prev_outputs = None

    max_idx = len(operators)
    while idx < max_idx:
        op = operators[idx]
        opcode = codes[op['opcode_index']]
        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]

        ibuffers = []
        for _ in op_inputs:
            if 'buffer' in tensors[_]:
                ibuffers += [buffers[tensors[_]['buffer']]]
        multi_input = len(ibuffers) > 1 and not any(['data' in _ for _ in ibuffers])
        filters = [_ for _ in op_inputs if 'data' in buffers[tensors[_]['buffer']]]

        next_op, next_opcode = None, ''
        if idx < max_idx -1:
            next_op = operators[idx+1]
            next_opcode = codes[next_op['opcode_index']]
        next_next_op, next_next_opcode = None, ''
        if idx < max_idx -2:
            next_next_op = operators[idx+2]
            next_next_opcode = codes[next_next_op['opcode_index']]

        connected = True
        if prev_op != None:
            connected = any([_ in prev_outputs for _ in op_inputs])

        forked = False
        if prev_op != None:
            forked = forks(prev_op, prev_inputs, prev_outputs, operators, tensors)


        if not connected:
            break
        
        #SILU pattern
        elif opcode == "LOGISTIC" and next_opcode == "MUL": 
            patterns.append(idx)
            patterns.append(idx+1)
            idx += 1

        elif opcode == "RESHAPE" and next_opcode == "CAST" and next_next_opcode == "GATHER": 
            patterns.append(idx)
            patterns.append(idx+1)
            patterns.append(idx+2)
            lut_count = 4 #for RGBA uint32
            idx += 2

        elif opcode == "CAST" and next_opcode == "GATHER": 
            patterns.append(idx)
            patterns.append(idx+1)
            lut_count = 4 #for RGBA uint32
            idx += 1

        elif forked:
            break
        
        elif opcode in ["HARD_SWISH", "LOGISTIC", "QUANTIZE"]:
            patterns.append(idx)

        elif opcode in ['LEAKY_RELU', 'RELU', 'RELU6', 'RELU_N1_TO_1', 'RELU_0_TO_1']:
            patterns.append(idx)
          
        elif opcode in ["MUL", "ADD", "SUB", "SQUARED_DIFFERENCE"] and len(filters) == 1:
            weight_tensor = tensors[filters[0]]
            if np.prod(weight_tensor['shape']) <= 4:
                if 'shape' in weight_tensor and len(weight_tensor['shape'])>0 :
                    weight_shape = weight_tensor['shape']
                    lut_count = weight_shape[-1] #channels last
                else:
                    lut_count = 1

                if lut_count <= MAX_LUTs:
                    patterns.append(idx)
                else:
                    break
            else:
                break
        else:
            break

        idx += 1

        prev_op = op
        prev_inputs = op_inputs
        prev_outputs = op_outputs
        
    return patterns


# Check for case of Non-NX -> Pad -> Conv, where current op is the Pad
def fuse_pad_into_next_op(engine_op_types, prev_op, curr_op, next_op):
    # Make sure this is a PAD
    if curr_op != 'PAD':
        return False

    # If the previous op is already on NX then this isn't relevant
    if prev_op in engine_op_types.nx_op_types:
        return False

    # If next is Conv, split at the Pad so it can be combined into the next Conv
    if next_op == 'CONV_2D' and 'CONV_2D' in engine_op_types.nx_op_types:
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


def get_input_dtype(op, tensor):
    op_inputs = [_ for _ in op['inputs'] if _ != -1]
    dtype = tensor[op_inputs[0]]['type']

    return dtype


def get_output_dtype(op, tensor):
    op_outputs = [_ for _ in op['outputs'] if _ != -1]
    dtype = tensor[op_outputs[0]]['type']

    return dtype


def channels_first_shape(shape):
    s = list(shape)
    # if len(shape) > 3 or (s[0] > 1 and len(shape) == 3):
    if len(shape) >= 3:
        axis = 3
        s = s[:-axis] + s[-1:] + s[-axis:-1]
    return tuple(s), len(s)


def is_singleton(shapes):
    for shape in shapes:
        _shape, dims = channels_first_shape(shape)
        if len(_shape) > 1 and _shape[-1] == 1 and _shape[-2] == 1:
            return True
    return False


def is_sigmoid(a,b):
    if a == 'LOGISTIC' and b == 'MUL':
        return True
    return False


def is_lookup(a,b):
    if a == 'CAST' and b == 'GATHER':
        return True
    return False



def get_splits(jname, split_every_op=False, engine_op_types=None):
    valid_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'supported_ops.json') 
    with open(valid_path) as f:
        valid = json.load(f)

    with open(jname) as f:
        graph = json.load(f)
    assert(len(graph['subgraphs']) == 1)

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

        next_op, next_opcode = None, ''
        if i < len(operators)-1:
            next_op = operators[i+1]
            next_opcode = codes[next_op['opcode_index']]

        next_next_op, next_next_opcode = None, ''
        if i < len(operators)-2:
            next_next_op = operators[i+2]
            next_next_opcode = codes[next_next_op['opcode_index']]

        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]
      
        input_shapes=[]
        for op_input in op_inputs:
            if 'shape' in tensors[op_input]:
                input_shapes.append(tensors[op_input]['shape'])
        # input_shapes = [tensors[_]['shape'] for _ in op_inputs]
        input_buffers = [buffers[tensors[_]['buffer']] for _ in op_inputs]
        multi_input = len(input_buffers) > 1 and not any(['data' in _ for _ in input_buffers]) 
        output_shapes = [tensors[_]['shape'] for _ in op_outputs]
        output_buffers = [buffers[tensors[_]['buffer']] for _ in op_outputs]

        connected = True
        if prev_op != None:
            connected = any([_ in prev_outputs for _ in op_inputs])

        forked = False
        if prev_op != None:
            forked = forks(prev_op, prev_inputs, prev_outputs, operators, tensors)

        # If using another engine with limited op support, force a split when the current and prev
        # op mismatch in which engine supports them.
        # Example: Conv -> Pool would normally be 1 subgraph, but not in 3.0.
        engine_split = False
        if engine_op_types and prev_op != None:
            engine_split = force_split_due_to_engine(engine_op_types, prev_opcode, opcode, next_opcode)

        valid_op, error_param, param_value, input_param = check_valid(valid, opcode, op, tensors)

        if not valid_op:
            errors.append([i,op,opcode, error_param, param_value, input_param]) 

        # used for debugging, add ops to capture and save as solo graphs
        if opcode in ['QUANTIZE']:
            specials.append([i])

        if split_every_op:
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
        elif prev_opcode in ['UNPACK', 'RESHAPE','TRANSPOSE']: #TODO start a new graph after OP
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
              
        # elif not connected or forked:
        elif not connected:
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
        elif is_sigmoid(opcode,next_opcode) and len(lut_pattern(operators, codes, tensors, buffers, i, opcode)):
            patterns = lut_pattern(operators, codes, tensors, buffers, i, opcode) 
            current += patterns
            i += len(patterns) - 1

            # update outputs to be from last op in pattern
            op = operators[i]
            op_outputs = [_ for _ in op['outputs'] if _ != -1]
        elif (is_lookup(opcode,next_opcode) or is_lookup(next_opcode,next_next_opcode)) and len(lut_pattern(operators, codes, tensors, buffers, i, opcode)):
            patterns = lut_pattern(operators, codes, tensors, buffers, i, opcode) 
            if len(current):
                splits.append(current)
            current =[]         
            current += patterns
            i += len(patterns) - 1

            # update outputs to be from last op in pattern
            op = operators[i]
            op_outputs = [_ for _ in op['outputs'] if _ != -1]
        elif forked:
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
        elif any([_ for _ in op_inputs if _ in outputs]):
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
        elif len(reshape_pattern(operators, codes, tensors, buffers, i, opcode)):
            patterns = reshape_pattern(operators, codes, tensors, buffers, i, opcode)

            current += patterns
            i += len(patterns) - 1
            # update outputs to be from last op in pattern
            op = operators[i]
            op_outputs = [_ for _ in op['outputs'] if _ != -1]

            # @TODO: Maybe inject an identity if we can fit full map  --> suppose to be only FIA layers
        elif opcode in ['CONV_2D', 'TRANSPOSE_CONV', 'FULLY_CONNECTED', 'UNIDIRECTIONAL_SEQUENCE_LSTM', 'SOFTMAX', 'ARG_MAX', 'CAST', 'TILE', 'SPLIT', 'SPLIT_V', 'PACK', 'UNPACK', 'RESHAPE','TRANSPOSE', 'AVERAGE_POOL_2D', 'MEAN']: # start a new graph before key subgraph OP
            if len(current):
                splits.append(current)
            current = []
            current.append(i)

        elif opcode in ['RESIZE_NEAREST_NEIGHBOR']:
            sf_h, sf_w = get_scale_factor(operators, tensors, i)
            if sf_h > 2 or sf_w > 2 or prev_opcode in ['RESIZE_NEAREST_NEIGHBOR']:
                if len(current):
                    splits.append(current)
                current = []
            current.append(i)
        elif opcode in ['RESIZE_BILINEAR']:
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
        elif opcode in ['CONCATENATION']: # start a new graph before key subgraph OP
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
        elif len(lut_pattern(operators, codes, tensors, buffers, i, opcode)):
            patterns = lut_pattern(operators, codes, tensors, buffers, i, opcode)

            current += patterns
            i += len(patterns) - 1
            # update outputs to be from last op in pattern
            op = operators[i]
            op_outputs = [_ for _ in op['outputs'] if _ != -1]
            
        # elif opcode in ['DEPTHWISE_CONV_2D'] and 'padding' in opts and opts['padding'] != 'VALID': # start a new graph before key subgraph OP
        elif opcode in ['DEPTHWISE_CONV_2D']: # start a new graph before key subgraph OP
            if len(current):
                splits.append(current)
            current = []
            current.append(i)

        elif opcode in ['ADD', 'SUB', 'MUL', 'DIV', 'SQUARED_DIFFERENCE', "GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL", "EQUAL", "NOT_EQUAL"] and multi_input and is_singleton(input_shapes): #split if singleton channelwise input
            if len(current):
                splits.append(current)
            current = []
            current.append(i)

        elif engine_split:
            if len(current):
                splits.append(current)
            current = []
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

def gen_subgraph(idir, odir, jname, splits, vnnx, forced_shape=None):

    with open(jname) as f:
        g = json.load(f)

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

        sg = {}
        sg['subgraphs'] = [{}]
        sg['signature_defs'] = [{'inputs':None, 'outputs':None, 'signature_key':'serving_default', 'subgraph_index':0}]
        ssubgraph = sg['subgraphs'][0]

        if 'name' in subgraph:
            ssubgraph['name'] = subgraph['name']

        for key in ['version', 'description', 'metadata', 'operator_codes']:
            try:
                sg[key] = g[key]
            except:                
                sg[key] = []

        selected_operators = []
        for i in split_indices:
            selected_operators.append(operators[i])

        input_tensors = selected_operators[0]['inputs']
        output_tensors = selected_operators[-1]['outputs']

        
        all_input_tensors = []
        all_output_tensors = []

        for op in selected_operators:
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
                    if t['name'] not in input_names:
                        input_names.append(t['name'])
                        input_idx.append(io)

        output_names = []
        output_idx = []
        for io in output_tensors:
            if io != -1:
                t = tensors[io]
                buf = buffers[t['buffer']]
                if 'data' not in buf:
                    if t['name'] not in output_names:
                        output_names.append(t['name'])
                        output_idx.append(io)

        # remove unused buffers/tensors
        
        used_buffers = []
        used_tensors = []
        for op in selected_operators:
            
            op_inputs = [_ for _ in op['inputs'] if _ != -1]
            op_outputs = [_ for _ in op['outputs'] if _ != -1]
            op_intermediates = []

            if 'intermediates' in op:
                op_intermediates = [_ for _ in op['intermediates'] if _ != -1]

            for io in op_inputs + op_outputs + op_intermediates:
                used_buffers.append(tensors[io]['buffer'])
                used_tensors.append(io)

        sbuffers = []
        for b in range(len(buffers)):
            if b not in used_buffers:
                sbuffers.append({})
            else:
                sbuffers.append(buffers[b])
        sg['buffers'] = sbuffers

        stensors = []
        for t in range(len(tensors)):
            if t not in used_tensors:
                stensors.append({})
            else:
                if not forced_shape is None:
                    buf = buffers[tensors[t]['buffer']]
                    if 'data' not in buf:
                        tensors[t]['shape'] = forced_shape
                stensors.append(tensors[t])
        ssubgraph['tensors'] = stensors

        ssubgraph['operators'] = selected_operators
        ssubgraph['inputs'] = input_idx
        ssubgraph['outputs'] = output_idx

        graph_inputs = [{'name': n, 'tensor_index': i} for n, i in zip(input_names, input_tensors)]
        graph_outputs = [{'name': n, 'tensor_index': i} for n, i in zip(output_names, output_tensors)]

        sg['signature_defs'][0]['inputs'] = graph_inputs
        sg['signature_defs'][0]['outputs'] = graph_outputs

        schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
        
        with open(subgraph_jname, 'w+') as f:
            json.dump(sg, f)
        cmd = 'flatc -b --strict-json --defaults-json -o {} {} {}'.format(odir, schema_path, subgraph_jname)
        subprocess.run(shlex.split(cmd))

        if vnnx:
            subgraph_tflite = subgraph_jname.replace('.json', '.tflite')
            subgraph_vnnx = subgraph_jname.replace('.json', '.vnnx')
            cmd = 'vnnx_compile -c {} -t {} -o {}'.format(os.environ.get('HW_CONFIG'),subgraph_tflite, subgraph_vnnx)
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
    graph_fname = os.path.join(nx_dirname, "graph.json")
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
    json_subgraphs_merged_nx = gen_subgraph(nx_subdir, nx_subdir, jname, merged_splits, vnnx=False)

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
def write_json_graph_without_mxp_pre_post_processing(jname, tmp_dir, splits, nx_dirname,\
    engine_graphs_mxp, output_name):

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
        json_subgraphs_no_pre_post = gen_subgraph(nx_subdir, nx_subdir, jname, [all_ops], vnnx=False)
        assert len(json_subgraphs_no_pre_post) == 1
        jname_truncated = json_subgraphs_no_pre_post[0]

        # Read graph from JSON and write it to a file.
        # It might be possible to do this more efficiently. E.g., instead of load and write the
        # NX JSON files, just copy them from their location in json_subgraphs_merged_nx.
        with open(jname_truncated) as f:
            graph = json.load(f)
        assert os.path.isdir(nx_dirname)
        assert output_name != None
        output_name = output_name.replace('-', '_').replace('.', '_')
        graph_fname = os.path.join(nx_dirname, f"{output_name}.json")
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

def write_json_for_nx(jname, tmp_dir, splits, engine_graphs_nx, engine_graphs_mxp, output_name):

    # Make an nx_dir for the final engine graph JSON files
    nx_dirname = "nx_engine"
    delete_and_remake_dir(nx_dirname)

    # Also make a sync dir
    nx_sync_dir = os.path.join(nx_dirname, 'sync')
    os.mkdir(nx_sync_dir)

    # Write the entire graph to JSON, along with a list of which ops are on MXP.
    write_json_graph_and_mxp_ops_for_nx(jname, nx_dirname, engine_graphs_mxp, splits)

    # Write the JSON graph without the pre and post processing ops (on MXP).
    write_json_graph_without_mxp_pre_post_processing(jname, tmp_dir, splits, nx_dirname,\
        engine_graphs_mxp, output_name)

    # This next function is unused and can be deleted, along with the helper functions
    # it calls. It writes each individual NX engine graph to a separete JSON file.
    # This may be useful for testing, but saving the entire graph as a single JSON and
    # also including a list of ops on MXP is more accurate because there can be connections
    # from one NX engine graph to another, e.g., two NX engine graphs does not mean two
    # completely separate graphs.
    write_json_per_nx_engine_graph(jname, tmp_dir, splits, nx_dirname,\
        engine_graphs_nx, engine_graphs_mxp)

def generate_split_graphs(tflite_model, dir, split_every_op=False, cuts=None, vnnx=False, vbx_version=2,\
    output_name=None):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    jname = os.path.join(dir, os.path.basename(tflite_model).replace('.tflite','.json'))

    cmd = 'flatc -t --strict-json --defaults-json -o {} {} -- {}'.format(dir, schema_path, tflite_model)
    subprocess.run(shlex.split(cmd))

    #update operator builtin_code
    with open(schema_path.replace('.fbs', '.json')) as f:
        schema = json.load(f)
        operators = schema["definitions"]["tflite_BuiltinOperator"]["enum"]
        with open(jname) as f:
            graph = json.load(f)
            for i,g in enumerate(graph['operator_codes']):
                if 'deprecated_builtin_code' in g and not g['deprecated_builtin_code'] == 127:
                    graph['operator_codes'][i]['builtin_code'] = operators[g['deprecated_builtin_code']]
        with open(jname, 'w') as f:
            json.dump(graph, f)


    subdir = os.path.join(dir, 'subgraphs') 
    if not os.path.exists(subdir):
        os.mkdir(subdir)

    # errors.append([i,op,opcode])
    if vbx_version == 3:
        nx_op_types = ['CONV_2D', ]
        agnostic_op_types = ['QUANTIZE', 'RESIZE_NEAREST_NEIGHBOR', 'PAD', 'ADD', 'CONCATENATION',\
                             'STRIDED_SLICE', ]
        engine_op_types = EngineOpTypes(nx_op_types, agnostic_op_types)

        # Pass in nx_op_types in case need to add additional splits at engine graph boundaries
        splits, error_ops, special_ops = get_splits(jname, split_every_op, engine_op_types)
    else:
        # splits, error_ops, special_ops = get_splits(jname, split_every_op)
        splits, error_ops, special_ops = get_splits2(jname, split_every_op)

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
    json_subgraphs = gen_subgraph(subdir, subdir, jname, splits, vnnx)

    if vbx_version == 3:
        # Map each split to an engine
        # TODO: Currently these are only being used as lists of splits. The engine graph concept
        # (list of lists) is not used anywhere. If it will never be needed, can simplify this.
        engine_graphs_nx, engine_graphs_mxp = get_splits_per_engine(jname, splits, engine_op_types)

        # For NX, write the required engine graphs to JSON files
        write_json_for_nx(jname, dir, splits, engine_graphs_nx, engine_graphs_mxp, output_name)

        # For MXP, return all subgraphs but also which are on NX.
        # It may be possible to just return the subgraphs which are on MXP, rather than return all
        # subgraphs. But currently this would result in set_io_nodes failing to find the graph
        # input and output tensors which are only in the NX subgraphs.
        return json_subgraphs, engine_graphs_nx

    return json_subgraphs, None


def generate_cut_graphs(tflite_model, cuts=None, vnnx=False):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    dir = os.path.dirname(tflite_model)
    if dir == '':
        dir = './'
    jname = os.path.join(dir, os.path.basename(tflite_model).replace('.tflite','.json'))

    cmd = 'flatc -t --strict-json --defaults-json -o {} {} -- {}'.format(dir, schema_path, tflite_model)
    subprocess.run(shlex.split(cmd))

    #update operator builtin_code
    with open(schema_path.replace('.fbs', '.json')) as f:
        schema = json.load(f)
        operators = schema["definitions"]["tflite_BuiltinOperator"]["enum"]
        with open(jname) as f:
            graph = json.load(f)
            for i,g in enumerate(graph['operator_codes']):
                if 'deprecated_builtin_code' in g and not g['deprecated_builtin_code'] == 127:
                    graph['operator_codes'][i]['builtin_code'] = operators[g['deprecated_builtin_code']]
        with open(jname, 'w') as f:
            json.dump(graph, f)

    cmd = 'flatc -b --strict-json --defaults-json -o {} {} {}'.format(dir, schema_path, jname)
    subprocess.run(shlex.split(cmd))

    cuts = get_cuts(jname, cuts)
    return gen_subgraph(dir, dir, jname, cuts, vnnx)


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


def get_output_scale(scale_factor, dtype='INT8'):
    if dtype.upper()=='UINT8':
        q_min = 0
        q_max = 255
        min_output_fp = (q_min)*(1/scale_factor)
        max_output_fp = (q_max)*(1/scale_factor)
    else:
        q_min = -128
        q_max = 127
        min_output_fp = (q_min +128)*(1/scale_factor)
        max_output_fp = (q_max +128)*(1/scale_factor)

    output_scale = (max_output_fp-min_output_fp)/255

    output_zero_point = q_min - (min_output_fp/output_scale)

    return output_scale, output_zero_point


def get_details_from_constants(var_factor, dtype='INT8'):
    max_value = max(var_factor)
    min_value = min(var_factor)
    if max_value > 0 :
        if dtype.upper()=='UINT8':
            zero_point =  0
            q_value = 255

        else:
            zero_point =  -128
            q_value = 127
        fp_value = max_value
    else:
        if dtype.upper()=='UINT8':
            zero_point = 255
            q_value = 0
        else:
            zero_point = 127
            q_value = -128
        fp_value = min_value

    scale = fp_value/(q_value-zero_point)

    q_value = [round((i/scale) + zero_point) for i in var_factor]
    # print(q_value)
    
    return scale, zero_point, q_value


def get_quantize_values(scale, mean, dtype='INT8'):
    if isinstance(mean, list) and isinstance(scale, list):
        #scaling mean and scale inputs
        scale_factor = [1 / j for j in scale]

        mean_factor = [i / j for i, j in zip(mean, scale)]
        mean_factor = [-1 * j for j in mean_factor]

        mean_scale, mean_zero_point, mean_q_value = get_details_from_constants(mean_factor)
        scale, scale_zero_point, scale_q_value = get_details_from_constants(scale_factor)

        return [scale, scale_zero_point, scale_q_value], [mean_scale, mean_zero_point, mean_q_value]

    elif isinstance(scale, list) and mean ==0.0:
        if dtype.upper()=='UINT8':
            scale_q_value = 255
            scale_zero_point = 254
        else:
            scale_q_value = 127
            scale_zero_point = 126
        scale = [1 / j for j in scale]

        return [scale, scale_zero_point, scale_q_value], []
    

def inject_preprocess(i, o0, t0, opcodes, tensors, buffers, graph_inputs, inputs, operators, scale, mean, dtype='INT8'):

    # add QUANTIZE only for
    ops_num = i
    if dtype.upper() != "UINT8":
        buffers.append({'offset': 0, 'size': 0})
        tensors.append({'shape': t0['shape'],
            'type': 'UINT8',
            'buffer': len(buffers)-1,
            'name': 'preprocess_data:{}'.format(i),
            'quantization': {'scale': [1.0], 'zero_point': [0], 'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})
        graph_inputs[i]['tensor_index'] = len(tensors)-1
        inputs[i] = len(tensors)-1 

        inject_op = {'opcode_index': opcodes.index('QUANTIZE'),
                    'inputs': [len(tensors)-1],
                    'outputs': [o0]}
        operators = operators[:ops_num] + [inject_op] + operators[ops_num:]
        ops_num += 1

    do_mul = isinstance(scale, list) or (scale != 1.0)
    do_add = isinstance(mean, list) or (mean != 0.0)
    output_scale = 1.0
    output_zeropoint = -128
    channel = 1
    mul_offset = 0

    if do_mul:
        mul_offset = 1
        scale_details, shift_details = get_quantize_values(scale, mean, dtype)
        scale_factor = max(scale)
        scale = scale_details[0]
        scale_zero_point = scale_details[1]
        scale_q_value = scale_details[2]
        
        if isinstance(scale_q_value, list):
            channel = len(scale_q_value)
            scale_q_value = np.array(scale_q_value).astype(np.uint8) 
        else:
            scale_q_value = np.array([scale_q_value]).astype(np.uint8) 

        scale_q_value = scale_q_value.tolist()

        output_scale, output_zeropoint = get_output_scale(scale_factor, dtype)
        output_zeropoint = round(output_zeropoint)

        if isinstance(scale, list):
            scale = scale[0]
        if dtype.upper() == "UINT8":
            mul_zp = 0
        else:
            mul_zp = -128
        buffers.append({'offset': 0, 'size': 0})
        tensors.append({'shape': t0['shape'],
            'type': dtype.upper(),
            'buffer': len(buffers)-1,
            'name': 'scale_data:{}'.format(i),
            'quantization': {'scale': [1.0], 'zero_point': [mul_zp], 'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})
        
        if dtype.upper() != "UINT8":
            operators[i]['outputs'] = [len(tensors)-1]
        else:
            graph_inputs[i]['tensor_index'] = len(tensors)-1
            inputs[i] = len(tensors)-1 
        buffers.append({'data': scale_q_value, 'offset': 0, 'size': 0})
        tensors.append({'shape': [1,1,1,channel],
            'type': dtype.upper(),
            'buffer': len(buffers)-1,
            'name': 'mul_data:{}'.format(i),
            'quantization': {'scale': [scale], 'zero_point': [scale_zero_point], 'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})
        inject_op = {'opcode_index': opcodes.index('MUL'),
                    'builtin_options_type': 'MulOptions',
                    'builtin_options': {'fused_activation_function': 'NONE'},
                    'custom_options_format': 'FLEXBUFFERS',
                    'large_custom_options_offset': 0,
                    'large_custom_options_size': 0,
                    'inputs': [len(tensors)-2, len(tensors)-1], 
                    'outputs': [o0]}

        operators = operators[:ops_num] + [inject_op] + operators[ops_num:]
        ops_num = ops_num+mul_offset

    if do_add:
        # add ADD
        buffers.append({'offset': 0, 'size': 0})
        tensors.append({'shape': t0['shape'],
            'type': dtype.upper(),
            'buffer': len(buffers)-1,
            'name': 'shift_data:{}'.format(i),
            'quantization': {'scale': [output_scale], 'zero_point': [output_zeropoint], 'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

        operators[i+mul_offset]['outputs'] = [len(tensors)-1]
        
        channel = 1
        scale = shift_details[0]
        mean_zero_point = shift_details[1]
        mean_q_value = shift_details[2]

        if isinstance(mean_q_value, list):
            channel = len(mean_q_value)
            mean_q_value = np.array(mean_q_value).astype(np.uint8) 
        else:
            mean_q_value = np.array([mean_q_value]).astype(np.uint8) 

        mean_q_value = mean_q_value.tolist()

        if isinstance(scale, list):
            scale = scale[0]
        
        buffers.append({'data': mean_q_value, 'offset': 0, 'size': 0})
        tensors.append({'shape': [channel],
            'type': dtype.upper(),
            'buffer': len(buffers)-1,
            'name': 'add_data:{}'.format(i),
            # 'quantization': {'scale': [1/255*(1/scale * 128/127)], 'zero_point': [-128], 'details_type': 'NONE', 'quantized_dimension': 0},
            'quantization': {'scale': [scale], 'zero_point': [mean_zero_point], 'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})
        inject_op = {'opcode_index': opcodes.index('ADD'),
                    'builtin_options_type': 'AddOptions',
                    'builtin_options': {'fused_activation_function': 'NONE'},
                    'custom_options_format': 'FLEXBUFFERS',
                    'large_custom_options_offset': 0,
                    'large_custom_options_size': 0,
                    'inputs': [len(tensors)-2, len(tensors)-1],
                    'outputs': [o0]}

        operators = operators[:ops_num] + [inject_op] + operators[ops_num:]

    return tensors, buffers, graph_inputs, inputs, operators


def preprocess_graphs(tflite_model, scale=1.0, mean=0):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    dir = os.path.dirname(tflite_model)
    if dir == '':
        dir = './'
    jname = os.path.join(dir, os.path.basename(tflite_model).replace('.tflite','.json'))
    cmd = 'flatc -t --strict-json --defaults-json -o {} {} -- {}'.format(dir, schema_path, tflite_model)
    subprocess.run(shlex.split(cmd))
    jname = os.path.join(os.path.dirname(tflite_model), os.path.basename(tflite_model).replace('.tflite','.json'))

    with open(jname) as f:
        graph = json.load(f)

        assert(len(graph['subgraphs']) == 1)
        subgraph = graph['subgraphs'][0]
        operators = subgraph['operators']
        tensors = subgraph['tensors']

        opcodes = [_['builtin_code'] for _ in graph['operator_codes']]
        if not 'QUANTIZE' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 114, 'version': 1, 'builtin_code': 'QUANTIZE'})
        if not 'MUL' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 18, 'version': 2, 'builtin_code': 'MUL'})
        if not 'ADD' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 0, 'version': 2, 'builtin_code': 'ADD'})

        opcodes = [_['builtin_code'] for _ in graph['operator_codes']]
        inputs = subgraph['inputs']

        buffers = graph['buffers']
        if 'signature_defs' in graph and len(graph['signature_defs']) > 0:
            graph_inputs = graph['signature_defs'][0]['inputs'] 
        else:
            if 'inputs' in subgraph and 'outputs' in subgraph:
                input_tensors = subgraph['inputs']
                output_tensors = subgraph['outputs']
            else:
                output_tensors = operators[-1]['outputs']
                input_tensors = operators[0]['inputs']
                all_input_tensor = []
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
            graph['signature_defs'] = [{'inputs':None, 'outputs':None, 'signature_key':'serving_default', 'subgraph_index':0}]
            graph['signature_defs'][0]['inputs'] = graph_inputs
            graph['signature_defs'][0]['outputs'] = graph_outputs

            inputs = []
            for t in input_tensors:
                for i,op in enumerate(operators):
                    if t in op['inputs']:
                        inputs.append(i)
                        break
            subgraph['inputs'] = inputs

        for i, idx in enumerate(inputs[:1]):
            if len(subgraph['operators']) < idx:
                for ops in subgraph['operators']:
                    if idx in ops['inputs']:
                        op = ops
                        break
            else:
                op = subgraph['operators'][idx]
            dtype = get_input_dtype(op, tensors)
            for o, o0 in enumerate(op['inputs']):
                t0 = subgraph['tensors'][o0]
                if 'buffer' in t0 and not ('data' in buffers[t0['buffer']]):
                    b0 = buffers[t0['buffer']]
                    q0 = t0['quantization']
                    tensors, buffers, graph_inputs, inputs, operators = inject_preprocess(i, o0, t0, opcodes, tensors, buffers, graph_inputs, inputs, operators, scale, mean, dtype)

        graph['signature_defs'][0]['inputs'] = graph_inputs
        graph['buffers'] = buffers
        subgraph['inputs'] = inputs
        subgraph['operators'] = operators
        subgraph['tensors'] = tensors

        jname = os.path.join(os.path.dirname(tflite_model), os.path.basename(tflite_model).replace('.tflite','.pre.json'))
        with open(jname, 'w') as f:
            json.dump(graph, f)
        cmd = 'flatc -b --strict-json --defaults-json -o {} {} {}'.format('./', schema_path, jname)
        subprocess.run(shlex.split(cmd))

def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite", type=existing_file)
    parser.add_argument("-s", "--scale", type=float, nargs='+', default=1.0)
    parser.add_argument("-m", "--mean", type=float, nargs='+', default=0.)
    args = parser.parse_args()

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    tmp_tflite = os.path.join(tmp_dir, os.path.basename(args.tflite))
    shutil.copyfile(args.tflite, tmp_tflite)

    preprocess_graphs(tmp_tflite, args.scale, args.mean)
    graphs = glob.glob(os.path.join(tmp_dir, "*.tflite"))
    for src in graphs:
        if src != tmp_tflite:
            dst = os.path.join(os.path.dirname(args.tflite),os.path.basename(src))
            shutil.copyfile(src, dst)
    tmp_dir_obj.cleanup()

def get_optimal_resize_shape(src_h, src_w, dest_h, dest_w):

    sc_w = dest_w/src_w
    sc_h = dest_h/src_h
    heights = []
    widths = []
    
    tmp_h = dest_h
    tmp_w = dest_w
    while sc_h>=2 and sc_w>=2:
        if sc_w>=4:
            tmp_h = tmp_h//4
            tmp_w = tmp_w//4
        elif sc_w>=2 and sc_w<4:  
            tmp_h = tmp_h//2
            tmp_w = tmp_w//2
        else:
            break
        heights.append(tmp_h)
        widths.append(tmp_w)
        sc_w = tmp_w/src_w
        sc_h = tmp_h/src_h

    heights.reverse()
    widths.reverse()

    return heights, widths

def add_quantize_layer(tensors, buffers, opcodes, operators, i, current_output, current_shape, current_type, current_quant, pos):
    #add 

    if current_type in ['INT8']: #inject QUANTIZE
        current_type = 'UINT8'
        zp = 128 
        buffers.append({'offset': 0, 'size': 0})
        tensors.append({'shape': current_shape,
                'type': "UINT8",
                'buffer': len(buffers)-1,
                'name': 'quant_{}'.format(i),
                'quantization': {'scale': [current_quant['scale'][0]], 'zero_point': [0], 'details_type': 'NONE', 'quantized_dimension': 0},
                'is_variable': False,
                'has_rank': True}) 
    else:
        current_type = 'INT8'
        buffers.append({'offset': 0, 'size': 0})
        zp = -128 
        tensors.append({'shape': current_shape,
                'type': "INT8",
                'buffer': len(buffers)-1,
                'name': 'quant_{}'.format(i),
                'quantization': {'scale': [current_quant['scale'][0]], 'zero_point': [-128], 'details_type': 'NONE', 'quantized_dimension': 0},
                'is_variable': False,
                'has_rank': True}) 
       
    inject_op = {'opcode_index': opcodes.index('QUANTIZE'),
                    'inputs': [current_output],
                    'outputs': [len(tensors)-1]}
    operators = operators + [inject_op]
    current_output = len(tensors)-1
    
    return tensors, buffers, operators, current_output, current_type, zp

def op_quantize(tensors, buffers, opcode_idx, i, inject_before, dtype, current_quant, zp):
    t = tensors[i]
    #add QUANTIZE
    print(current_quant['scale'])
    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy() ,
            'type': dtype,
            'buffer': len(buffers)-1,
            'name': 'quant_{}'.format(i),
            'quantization': {'scale': [current_quant['scale'][0]], 'zero_point': [zp], 'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})
    input_tensor, output_tensor = i, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, i
    
    inject_op = {'opcode_index': opcode_idx,
                'inputs': [input_tensor],
                'outputs': [output_tensor]}
    
    return inject_op, tensors, buffers

def inject_post_process(i, op_idx, o0, t0, opcodes, tensors, buffers, graph_outputs, outputs, operators, dataset, opacity, input_height, input_width, height, width):
  
    tx = tensors[outputs[0]].copy()
    current_type = tx['type']
    current_shape = tx['shape']
    current_output = outputs[0]
    current_quant = tx['quantization']
    double = 0

    #inject ARG_MAX
    if current_shape[-1] > 1 and len(current_shape) > 3: #perform ARG_MAX injection on non-HxW dimension.

        while (current_shape[-3] < height/4 and current_shape[-2] < width/4):
            # current_shape = [current_shape[-4], current_shape[-3]*2, current_shape[-2]*2, current_shape[-1]]
            while (current_shape[-3] < height/4 and current_shape[-2] < width/4):
                current_shape = [current_shape[-4], current_shape[-3]*2, current_shape[-2]*2, current_shape[-1]]

            data = np.frombuffer(np.asarray([current_shape[-3], current_shape[-2]]).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
            buffers.append({'data': data, 'offset': 0, 'size': 0})
            tensors.append({'shape':[2],
                    'type': 'INT32',
                    'buffer': len(buffers)-1,
                    'name': 'resize_double{}_{}'.format(double,i),
                    'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
                    'is_variable': False,
                    'has_rank': True})

            buffers.append({'offset': 0, 'size': 0})
            tensors.append({'shape': current_shape,
                    'type': current_type,
                    'buffer': len(buffers)-1,
                    'name': 'resize_double{}_output_{}'.format(double,i),
                    'quantization': current_quant,
                    'is_variable': False,
                    'has_rank': True})

            resize = 'RESIZE_BILINEAR'
            inject_op = {'opcode_index': opcodes.index(resize),
                        'builtin_options_type': 'ReshapeOptions',
                        'inputs': [current_output, len(tensors)-2], 
                        'outputs': [len(tensors)-1]}
            operators = operators + [inject_op]
            current_output = len(tensors)-1
            double += 1

        data = np.frombuffer(np.asarray([3]).astype(np.int64).tobytes(), dtype=np.uint8).tolist()
        buffers.append({'data': data, 'offset': 0, 'size': 0})
        tensors.append({'shape': None,
                'type': 'INT64',
                'buffer': len(buffers)-1,
                'name': 'arg_max_dim_{}'.format(i),
                'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
                'is_variable': False,
                'has_rank': True})
        buffers.append({'offset': 0, 'size': 0})

        current_shape = current_shape[:-1]
        current_type = 'INT32'
        tensors.append({'shape': current_shape,
                'type': current_type,
                'buffer': len(buffers)-1,
                'name': 'arg_max_{}'.format(i),
                'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
                'is_variable': False,
                'has_rank': True})
        inject_op = {'opcode_index': opcodes.index('ARG_MAX'),
                    'builtin_options_type': "ArgMaxOptions",
                    'builtin_options': {'output_type': 'INT32'},
                    'inputs': [current_output, len(tensors)-2], 
                    'outputs': [len(tensors)-1]}
        
        operators = operators + [inject_op]
        current_output = len(tensors)-1

        current_type = 'UINT8'
        buffers.append({'offset': 0, 'size': 0})
        tensors.append({'shape': current_shape,
                'type': current_type,
                'buffer': len(buffers)-1,
                'name': 'cast_{}'.format(i),
                'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
                'is_variable': False,
                'has_rank': True})
        current_quant = {'details_type': 'NONE', 'quantized_dimension': 0}

        inject_op = {'opcode_index': opcodes.index('CAST'),
                    'inputs': [current_output], 
                    'outputs': [len(tensors)-1]}
        operators = operators + [inject_op]
        current_output = len(tensors)-1

    elif current_type in ['INT8']: #inject QUANTIZE
        current_type = 'UINT8'
        current_quant['zero_point'][0] += 128
        buffers.append({'offset': 0, 'size': 0})
        tensors.append({'shape': current_shape,
            'type': current_type,
            'buffer': len(buffers)-1,
            'name': 'quant_{}'.format(i),
            'quantization': current_quant,
            'is_variable': False,
            'has_rank': True})
        inject_op = {'opcode_index': opcodes.index('QUANTIZE'),
                    'inputs': [current_output],
                    'outputs': [len(tensors)-1]}
        operators = operators + [inject_op]
        current_output = len(tensors)-1

    #inject RESHAPE, adding back channels TODO remove if alread NHWC
    if len(current_shape) == 3:
        current_shape = current_shape + [1]
        data = np.frombuffer(np.asarray(current_shape).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
        buffers.append({'data': data, 'offset': 0, 'size': 0})
        tensors.append({'shape': [4],
                'type': 'INT32',
                'buffer': len(buffers)-1,
                'name': 'expand_shape_{}'.format(i),
                'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
                'is_variable': False,
                'has_rank': True})

        buffers.append({'offset': 0, 'size': 0})
        tensors.append({'shape': current_shape,
                'type': current_type,
                'buffer': len(buffers)-1,
                'name': 'expand_{}'.format(i),
                'quantization': current_quant,
                'is_variable': False,
                'has_rank': True})

        inject_op = {'opcode_index': opcodes.index('RESHAPE'),
                    'builtin_options_type': 'ReshapeOptions',
                    'inputs': [current_output, len(tensors)-2], 
                    'outputs': [len(tensors)-1]}
        operators = operators + [inject_op]
        current_output = len(tensors)-1

    # inject RESIZE 
    inject_dequantize = False
    
    resize = 'RESIZE_NEAREST_NEIGHBOR'
    if dataset in ['depth']:
        resize = 'RESIZE_BILINEAR'

    zp = current_quant['zero_point'][0] if 'zero_point' in current_quant and len(current_quant['zero_point']) > 0 else 0

    if current_shape[-3] != height or current_shape[-2] != width:
        heights, widths = get_optimal_resize_shape(current_shape[-3], current_shape[-2], height, width)
        if current_type in ['UINT8'] and not inject_dequantize and resize in ['RESIZE_BILINEAR']: #inject QUANTIZE
            # inject_op, tensors, buffers = op_quantize(tensors, buffers, opcodes.index('QUANTIZE'), i, False, current_type, current_quant, -128)
            tensors, buffers, operators, current_output, current_type, zp = add_quantize_layer(tensors, buffers, opcodes, operators, i, current_output, current_shape, current_type, current_quant, 1)
            inject_dequantize = True
                      

        # while (current_shape[-3] < height/4 and current_shape[-2] < width/4):
        for(_, (h, w)) in enumerate(zip(heights, widths)):
            #current_shape = [current_shape[-4], current_shape[-3]*2, current_shape[-2]*2, current_shape[-1]]
            current_shape = [current_shape[-4], h, w, current_shape[-1]]

            data = np.frombuffer(np.asarray([current_shape[-3], current_shape[-2]]).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
            buffers.append({'data': data, 'offset': 0, 'size': 0})
            tensors.append({'shape':[2],
                    'type': 'INT32',
                    'buffer': len(buffers)-1,
                    'name': 'resize_double{}_{}'.format(double,i),
                    'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
                    'is_variable': False,
                    'has_rank': True})
            if 'scale' in current_quant and len(current_quant['scale']) > 0:
                sc = current_quant['scale'][0]
            else:
                sc = 1.0
            buffers.append({'offset': 0, 'size': 0})
            tensors.append({'shape': current_shape,
                    'type': current_type,
                    'buffer': len(buffers)-1,
                    'name': 'resize_double{}_output_{}'.format(double,i),
                    'quantization': {'scale': [sc], 'zero_point': [zp], 'details_type': 'NONE', 'quantized_dimension': 0},
                    'is_variable': False,
                    'has_rank': True})

            inject_op = {'opcode_index': opcodes.index(resize),
                        'builtin_options_type': 'ReshapeOptions',
                        'inputs': [current_output, len(tensors)-2], 
                        'outputs': [len(tensors)-1]}
            operators = operators + [inject_op]
            current_output = len(tensors)-1
            double += 1

        
        current_shape = [current_shape[-4], height, width, current_shape[-1]]
        data = np.frombuffer(np.asarray([height,width]).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
        buffers.append({'data': data, 'offset': 0, 'size': 0})
        tensors.append({'shape':[2],
                'type': 'INT32',
                'buffer': len(buffers)-1,
                'name': 'resize_{}'.format(i),
                'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
                'is_variable': False,
                'has_rank': True})

        buffers.append({'offset': 0, 'size': 0})
        tensors.append({'shape': current_shape,
                'type': current_type,
                'buffer': len(buffers)-1,
                'name': 'resize_output_{}'.format(i),
                'quantization': {'scale': [sc], 'zero_point': [zp], 'details_type': 'NONE', 'quantized_dimension': 0},
                'is_variable': False,
                'has_rank': True})

        resize = 'RESIZE_NEAREST_NEIGHBOR'
        inject_op = {'opcode_index': opcodes.index(resize),
                    'builtin_options_type': 'ReshapeOptions',
                    'inputs': [current_output, len(tensors)-2], 
                    'outputs': [len(tensors)-1]}
        operators = operators + [inject_op]
        current_output = len(tensors)-1

        if inject_dequantize == True and current_type in ['INT8']: #inject DEQUANTIZE
            tensors, buffers, operators, current_output, current_type, zp = add_quantize_layer(tensors, buffers, opcodes, operators, i, current_output, current_shape, current_type, current_quant, 2)
            inject_dequantize = False
            current_quant = tx['quantization']

    #inject CAST if
    if current_type != 'INT32':
        buffers.append({'offset': 0, 'size': 0})
        tensors.append({'shape': current_shape,
                'type': 'INT32',
                'buffer': len(buffers)-1,
                'name': 'cast_{}'.format(i),
                'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
                'is_variable': False,
                'has_rank': True})
        inject_op = {'opcode_index': opcodes.index('CAST'),
                    'inputs': [current_output],
                    'outputs': [len(tensors)-1]}
        operators = operators + [inject_op]
        current_output = len(tensors)-1

    colors = []
    if dataset == "VOC":
        colors = [[0,0,0]] + rgb_color.voc_colors
    elif dataset == "COCO":
        colors = [[0,0,0]] + rgb_color.coco_colors
    elif dataset == "cityscapes":
        rgb2bgr = lambda x: (x[2],x[1],x[0])
        colors = np.asarray([rgb2bgr(_["color"]) for _ in rgb_color.city_groups], dtype="uint8")   
    elif dataset == "depth":
        colors = cv2.applyColorMap(np.arange(256).astype('uint8'), cv2.COLORMAP_PLASMA).reshape((256,3))

    colors = [np.asarray(_).astype('uint8') for _ in colors]
    if len(colors) < 256: 
        colors += [np.asarray([0,0,0]).astype('uint8') for _ in range(256-len(colors))]
    if len(colors) > 256:
        colors = colors[:256]
    # covert to BGR colormap to RGB, then add alpha
    colors = np.array([_[2]*(2**16) + _[1]*(2**8) + _[0]*(2**0) for _ in colors]).astype(np.uint32)
    alpha = max(0,min(255,int(opacity*255)))*(2**24)
    colors += alpha

    # make NULL category transparent
    if dataset in ["VOC", "COCO", "cityscapes"]:
        colors[0] -= alpha

    #inject Gather
    buffers.append({'data': np.frombuffer(colors.tobytes(), dtype=np.uint8).tolist(), 'offset': 0, 'size': 0})
    tensors.append({'shape':[len(colors)],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'class_rgba_{}'.format(i),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': current_shape,
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'output_{}'.format(i),
            'quantization': {'scale': [0.0], 'zero_point': [0], 'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    inject_op = {'opcode_index': opcodes.index('GATHER'),
                    'builtin_options_type': 'GatherOptions',
                    "builtin_options": {
                        "axis": 0,
                        "batch_dims": 0
                    },
                    "custom_options_format": "FLEXBUFFERS",
                    "large_custom_options_offset": 0,
                    "large_custom_options_size": 0,
                    'inputs': [len(tensors)-2, current_output],
                    'outputs': [len(tensors)-1]}
    operators = operators + [inject_op]

    current_output = len(tensors)-1
    outputs[i] = len(tensors)-1

    graph_outputs[i]['tensor_index'] = len(tensors)-1
    graph_outputs[i]['name'] = tensors[-1]['name']
    

    return tensors, buffers, graph_outputs, outputs, operators


def post_processing_graphs(tflite_model, datatset, opacity, height, width):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    dir = os.path.dirname(tflite_model)
    if dir == '':
        dir = './'
    jname = os.path.join(dir, os.path.basename(tflite_model).replace('.tflite','.json'))
    cmd = 'flatc -t --strict-json --defaults-json -o {} {} -- {}'.format(dir, schema_path, tflite_model)
    subprocess.run(shlex.split(cmd))
    jname = os.path.join(os.path.dirname(tflite_model), os.path.basename(tflite_model).replace('.tflite','.json'))

    with open(jname) as f:
        graph = json.load(f)

        assert(len(graph['subgraphs']) == 1)
        subgraph = graph['subgraphs'][0]
        operators = subgraph['operators']
        tensors = subgraph['tensors']

        opcodes = [_['builtin_code'] for _ in graph['operator_codes']]
        if not 'QUANTIZE' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 114, 'version': 1, 'builtin_code': 'QUANTIZE'})
        if not 'GATHER' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 36, 'version': 2, 'builtin_code': 'GATHER'})
        if not 'CAST' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 53, 'version': 1, 'builtin_code': 'CAST'})
        if not 'RESHAPE' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 22, 'version': 1, 'builtin_code': 'RESHAPE'})
        if not 'RESIZE_BILINEAR' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 23, 'version': 2, 'builtin_code': 'RESIZE_BILINEAR'})
        if not 'RESIZE_NEAREST_NEIGHBOR' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 97, 'version': 2, 'builtin_code': 'RESIZE_NEAREST_NEIGHBOR'})
        if not 'EMBEDDING_LOOKUP' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 7, 'version': 2, 'builtin_code': 'EMBEDDING_LOOKUP'})
        if not 'ARG_MAX' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 56, 'version': 2, 'builtin_code': 'ARG_MAX'})

        opcodes = [_['builtin_code'] for _ in graph['operator_codes']]
        inputs = subgraph['inputs']
        outputs = subgraph['outputs']
        
        buffers = graph['buffers']
        if 'signature_defs' in graph: # graph['signature_defs']:
            graph_inputs = graph['signature_defs'][0]['inputs'] 
            graph_outputs = graph['signature_defs'][0]['outputs'] 
        else:
            if 'inputs' in subgraph and 'outputs' in subgraph:
                input_tensors = subgraph['inputs']
                output_tensors = subgraph['outputs']
            else:
                output_tensors = operators[-1]['outputs']
                input_tensors = operators[0]['inputs']
                all_input_tensor = []
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
            graph['signature_defs'] = [{'inputs':None, 'outputs':None, 'signature_key':'serving_default', 'subgraph_index':0}]
            graph['signature_defs'][0]['inputs'] = graph_inputs
            graph['signature_defs'][0]['outputs'] = graph_outputs

        # grab input shape
        input_shape = subgraph['tensors'][graph_inputs[0]['tensor_index']]['shape']
        input_height, input_width = input_shape[-3], input_shape[-2]

        for i, idx in enumerate(outputs[:1]):
            if len(subgraph['operators']) < idx:
                for op_idx,ops in enumerate(subgraph['operators']):
                    if idx in ops['outputs']:
                        op = ops
                        break
            else:
                op = subgraph['operators'][idx]
                op_idx = idx
            dtype = get_output_dtype(op, tensors)
            for o, o0 in enumerate(op['outputs']):
                t0 = subgraph['tensors'][o0]
                if 'buffer' in t0 and not ('data' in buffers[t0['buffer']]):
                    b0 = buffers[t0['buffer']]
                    q0 = t0['quantization']
                    tensors, buffers, graph_outputs, outputs, operators = inject_post_process(i, op_idx, o0, t0, opcodes, tensors, buffers, graph_outputs, outputs, operators, datatset, opacity, input_height, input_width, height, width)

        graph['signature_defs'][0]['outputs'] = graph_outputs
        graph['buffers'] = buffers
        subgraph['outputs'] = outputs
        subgraph['operators'] = operators
        subgraph['tensors'] = tensors
        # print(graph)
        jname = os.path.join(os.path.dirname(tflite_model), os.path.basename(tflite_model).replace('.tflite','.post.json'))
        with open(jname, 'w') as f:
            json.dump(graph, f)
        cmd = 'flatc -b --strict-json --defaults-json -o {} {} {}'.format('./', schema_path, jname)
        subprocess.run(shlex.split(cmd))

def postprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite", type=existing_file)
    parser.add_argument("-d", "--dataset", choices=['VOC', 'COCO', 'cityscapes', 'depth'], default='VOC')
    parser.add_argument("-o", "--opacity", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    args = parser.parse_args()

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    tmp_tflite = os.path.join(tmp_dir, os.path.basename(args.tflite))
    shutil.copyfile(args.tflite, tmp_tflite)
    post_processing_graphs(tmp_tflite, args.dataset, args.opacity, args.height, args.width)
    graphs = glob.glob(os.path.join(tmp_dir, "*.tflite"))
    for src in graphs:
        if src != tmp_tflite:
            dst = os.path.join(os.path.dirname(args.tflite),os.path.basename(src))
            shutil.copyfile(src, dst)
    tmp_dir_obj.cleanup()

def cut():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite", type=existing_file)
    parser.add_argument("-c", "--cuts", type=int, nargs='*')
    parser.add_argument("--vnnx", action='store_true')
    args = parser.parse_args()

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    tmp_tflite = os.path.join(tmp_dir, os.path.basename(args.tflite))
    shutil.copyfile(args.tflite, tmp_tflite)

    generate_cut_graphs(tmp_tflite, args.cuts, args.vnnx)
    graphs = list(glob.glob(os.path.join(tmp_dir, "*.tflite")))
    graphs += list(glob.glob(os.path.join(tmp_dir, "*.vnnx")))
    for src in graphs:
        if src != tmp_tflite:
            dst = os.path.join(os.path.dirname(args.tflite),os.path.basename(src))
            shutil.copyfile(src, dst)
    tmp_dir_obj.cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite", type=existing_file)
    parser.add_argument("dir")
    parser.add_argument("-j", "--join", action='store_true')
    parser.add_argument("-d", "--debug", action='store_true')
    parser.add_argument("-e", "--split-every-op", action='store_true')
    parser.add_argument("-c", "--cuts", type=int, nargs='*')
    parser.add_argument("--vnnx", action='store_true')
    args = parser.parse_args()

    if not args.join:
        generate_split_graphs(args.tflite, args.dir, args.split_every_op, args.cuts, args.vnnx)
    else:
        generate_join_graphs(args.tflite, args.dir, args.debug)

if __name__ == "__main__":
    main()
