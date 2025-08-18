import argparse
from .utils import existing_file, existing_dir, generate_inputs_outputs
import json
import copy
import subprocess
import shlex
import os.path
from contextlib import contextmanager
import sys, shutil, glob, tempfile
from tqdm import tqdm 
import numpy as np
import cv2
import vbx.postprocess.dataset as rgb_color
from math import floor, ceil, log2, frexp, copysign, exp, tanh, pow, log, sqrt


MAX_LUTS = 4

@contextmanager
def exception_catcher(unsupported_ops):
    try:
        yield
    except AssertionError as e:
        sys.stderr.write("\n\033[31m######################## VECTORBLOX ERROR! #############################\033[0m\n")
        sys.stderr.write("\033[31mSome layers have parameters/arguments that we do not currently support.\033[0m")
        sys.stderr.write("\nFor futher assistance, please send the layer(s) to the vectorblox team at:\n\033[31mvectorblox@microchip.com\033[0m\n")

        sys.stderr.write("\nError unsupported ops:")
        for i,op,opcode, error_param, param_value, input_param in unsupported_ops:
            if len(param_value) == 0:
                print("\t {} at location {} is not currently supported.\n".format(opcode, i))   
            else:
                print("\t {} at location {}, \033[31m {}={} \033[0m is not currently supported.".format(opcode, i, param_value[0], input_param[0]))     
                print("\t\t-->Supported values are = {}\n".format(error_param[0]))
     
        print()
        sys.exit(1)

default_passes = [
        'REMOVE_FP32_IO',
        'CLEAN_LOGISTIC',
        'LUT',
        'REMOVE_CONSTANTS',
        'EXPLICIT_PAD',
        'GROUP_CONV',
        'STRIDED_DEPTHWISE',
        'TRANSPOSE_CONV',
        ]

all_passes = default_passes + [
        'YOLO_ARG_MAX',
        ]

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


def channels_first_shape(shape):
    s = list(shape)
    # if len(shape) > 3 or (s[0] > 1 and len(shape) == 3):
    if len(shape) >= 3:
        axis = 3
        s = s[:-axis] + s[-1:] + s[-axis:-1]
    return tuple(s), len(s)


def is_multi_input(op, tensors, buffers):
    op_inputs = [_ for _ in op['inputs'] if _ != -1]
    input_buffers = [buffers[tensors[_]['buffer']] for _ in op_inputs]
    multi_input = len(input_buffers) > 1 and not any(['data' in _ for _ in input_buffers]) 
    return multi_input


def is_singleton(op, tensors):
    shapes=[]
    op_inputs = [_ for _ in op['inputs'] if _ != -1]
    for op_input in op_inputs:
        if 'shape' in tensors[op_input]:
            shapes.append(tensors[op_input]['shape'])

    for shape in shapes:
        _shape, dims = channels_first_shape(shape)
        if len(_shape) > 1 and _shape[-1] == 1 and _shape[-2] == 1:
            return True
    return False


def get_splits2(jname, split_every_op=False, engine_op_types=None):
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
        singleton = is_singleton(op, tensors)

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

        if prev_opcode in ['CONV_2D', 'DEPTHWISE_CONV_2D'] and 'stride_w' in opts and opts['stride_w'] > 1: #TODO
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

        elif opcode in ['CONCATENATION', 'DEPTHWISE_CONV_2D', 'CONV_2D', 'TRANSPOSE_CONV', 'FULLY_CONNECTED', 'UNIDIRECTIONAL_SEQUENCE_LSTM', 'SOFTMAX', 'ARG_MAX', 'CAST', 'TILE', 'SPLIT', 'SPLIT_V', 'PACK', 'UNPACK', 'RESHAPE','TRANSPOSE', 'AVERAGE_POOL_2D', 'MEAN']: # start a new graph before key subgraph OP
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

        else:
            current.append(i)

        prev_op = op
        prev_opcode = opcode

        prev_inputs = op_inputs
        prev_outputs = op_outputs
        i = i + 1

    splits.append(current)
    return splits, errors, specials


def is_forked(op0, operators, tensors):
    if len(op0['outputs']) > 1:
        next_ops = []
        for i, op in enumerate(operators):
            if op != op0:
                op_inputs = [_ for _ in op['inputs'] if _ != -1]
                for t in op_inputs:
                    if t in op0['outputs']:
                        next_ops.append(i)
        return True, next_ops

    count = 0
    tensor = op0['outputs'][0]
    next_ops = []
    for i, op in enumerate(operators):
        if op != op0:
            op_inputs = [_ for _ in op['inputs'] if _ != -1]
            for t in op_inputs:
                if t == tensor:
                    next_ops.append(i)
                    count += 1

    return count > 1, next_ops


def is_graph_input(op, operators, tensors, buffers):
    outputs = []
    for _op in operators:
        outputs += [_ for _ in _op['outputs'] if 'data' not in buffers[tensors[_]['buffer']]]
    return not any([_ in outputs for _ in op['inputs']])


def is_graph_output(op, operators, tensors, buffers):
    inputs = []
    for _op in operators:
        inputs += [_ for _ in _op['inputs'] if 'data' not in buffers[tensors[_]['buffer']]]  
    return not any([_ in inputs for _ in op['outputs']])


valid_lut_patterns = []
valid_lut_patterns += [["QUANTIZE", "CAST", "GATHER"]]
valid_lut_patterns += [["QUANTIZE", "CAST", "SPLIT", "GATHER", "GATHER", "CONCATENATION"]]
valid_lut_patterns += [["QUANTIZE", "CAST", "SPLIT", "GATHER", "GATHER", "GATHER", "CONCATENATION"]]
valid_lut_patterns += [["QUANTIZE", "CAST", "SPLIT", "GATHER", "GATHER", "GATHER", "GATHER", "CONCATENATION"]]

valid_reshape_patterns = []
valid_reshape_patterns += [["RESHAPE", "TRANSPOSE", "RESHAPE"]]


def is_pattern(operators, codes, tensors, buffers, valid_patterns):
    patterns = []

    for pattern in valid_patterns:
        if len(pattern)-1 < len(operators):
            if all([pattern[i] == codes[operators[i]['opcode_index']] for i in range(len(pattern))]):
                for i in range(len(pattern)):
                    patterns.append(i)

    return patterns


def graph_pattern(operators, codes, tensors, buffers):
    rp = is_pattern(operators, codes, tensors, buffers, valid_reshape_patterns)
    if len(rp):
        return "TRANSFORM", rp

    lp = is_pattern(operators, codes, tensors, buffers, valid_lut_patterns)
    if len(lp):
        return "LUT", lp

    return None, []


def num_lut_tensor(filter, tensors, buffers):
    weight_tensor = tensors[filter]
    data = get_numpy_data(weight_tensor, buffers)
    single_value = np.all(data.flatten() == data.flatten()[0])
    if 'shape' not in weight_tensor or np.prod(weight_tensor['shape']) <= MAX_LUTS or single_value:
        if single_value:
            return 1
        elif 'shape' in weight_tensor and len(weight_tensor['shape']) > 0 :
            if weight_tensor['shape'][-1] > MAX_LUTS:
                return -1
            return weight_tensor['shape'][-1]
        else:
            return 1
    return -1


def lut_pattern(operators, codes, tensors, buffers, idx):
    patterns = []
    lut_count = 1

    prev_op = None
    prev_inputs = None
    prev_outputs = None

    max_idx = len(operators)
    while idx < max_idx:
        op = operators[idx]
        opcode = codes[op['opcode_index']]
        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]

        multi_input = is_multi_input(op, tensors, buffers)
        filters = [_ for _ in op_inputs if 'data' in buffers[tensors[_]['buffer']]]

        next_op, next_opcode = None, ''
        if idx < max_idx -1:
            next_op = operators[idx+1]
            next_opcode = codes[next_op['opcode_index']]
            next_op_inputs = [_ for _ in next_op['inputs'] if _ != -1]
            next_filters = [_ for _ in next_op_inputs if 'data' in buffers[tensors[_]['buffer']]]

        next_next_op, next_next_opcode = None, ''
        if idx < max_idx -2:
            next_next_op = operators[idx+2]
            next_next_opcode = codes[next_next_op['opcode_index']]
            next_next_op_inputs = [_ for _ in next_next_op['inputs'] if _ != -1]
            next_next_filters = [_ for _ in next_next_op_inputs if 'data' in buffers[tensors[_]['buffer']]]

        next_next_next_op, next_next_next_opcode = None, ''
        if idx < max_idx -3:
            next_next_next_op = operators[idx+3]
            next_next_next_opcode = codes[next_next_next_op['opcode_index']]

        connected, forked = True, False
        if prev_op != None:
            connected = any([_ in prev_outputs for _ in op_inputs])
            forked, _ = is_forked(prev_op, operators, tensors)

        if not connected:
            break

        elif opcode == "DEQUANTIZE" and (next_opcode in ["RSQRT", "EXP", "LOG", "ELU", "POW", "COS", "SIN"] or (next_opcode in ["MUL", "DIV", "ADD", "SUB", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM"] and len(next_filters) == 1)) and (next_next_opcode in ["RSQRT", "EXP", "LOG", "ELU", "POW", "COS", "SIN"] or (next_next_opcode in ["MUL", "DIV", "ADD", "SUB", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM"] and len(next_next_filters) == 1)) and next_next_next_opcode == "QUANTIZE": 
            if next_opcode in ["MUL", "DIV", "ADD", "SUB", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM"]:
                required = num_lut_tensor(next_filters[0], tensors, buffers)
                if required < 0:
                    break
                if required > lut_count:
                    lut_count = required
            if next_next_opcode in ["MUL", "DIV", "ADD", "SUB", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM"]:
                required = num_lut_tensor(next_next_filters[0], tensors, buffers)
                if required < 0:
                    break
                if required > lut_count:
                    lut_count = required
            patterns.append(idx)
            patterns.append(idx+1)
            patterns.append(idx+2)
            patterns.append(idx+3)
            idx += 3
        
        elif opcode == "DEQUANTIZE" and (next_opcode in ["RSQRT", "EXP", "LOG", "ELU", "POW", "COS", "SIN"] or (next_opcode in ["MUL", "DIV", "ADD", "SUB", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM"] and len(next_filters) == 1)) and next_next_opcode == "QUANTIZE": 
            if next_opcode in ["MUL", "DIV", "ADD", "SUB", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM"]:
                required = num_lut_tensor(next_filters[0], tensors, buffers)
                if required < 0:
                    break
                if required > lut_count:
                    lut_count = required
            patterns.append(idx)
            patterns.append(idx+1)
            patterns.append(idx+2)
            idx += 2

        elif opcode == "LOGISTIC" and next_opcode == "MUL" and (op['inputs'][0] in next_op['inputs']): #SILU pattern
            patterns.append(idx)
            patterns.append(idx+1)
            idx += 1

        elif opcode == "RESHAPE" and next_opcode == "CAST" and next_next_opcode == "GATHER": 
            patterns.append(idx)
            patterns.append(idx+1)
            patterns.append(idx+2)
            # lut_count = 4 #for RGBA uint32
            idx += 2

        elif opcode == "CAST" and next_opcode == "GATHER": 
            patterns.append(idx)
            patterns.append(idx+1)
            # lut_count = 4 #for RGBA uint32
            idx += 1

        elif forked:
            break
        
        elif opcode in ["QUANTIZE"] and tensors[op_inputs[0]]['type'] in ['INT8', 'UINT8']:
            patterns.append(idx)

        elif opcode in ["DEQUANTIZE"] and tensors[op_outputs[0]]['type'] in ['INT8', 'UINT8']:
            patterns.append(idx)

        elif opcode in ["HARD_SWISH"]:
            patterns.append(idx)

        elif opcode in ["LOGISTIC"]:
            patterns.append(idx)

        elif opcode in ["TANH"]:
            patterns.append(idx)

        elif opcode in ['LEAKY_RELU', 'RELU', 'RELU6', 'RELU_N1_TO_1', 'RELU_0_TO_1']:
            patterns.append(idx)

        elif opcode in ["MUL", "ADD", "SUB", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM"] and len(filters) == 1 and tensors[op_outputs[0]]['type'] in ['INT8', 'UINT8']:
            required = num_lut_tensor(filters[0], tensors, buffers)
            if required < 0:
                break
            if required > lut_count:
                lut_count = required
            patterns.append(idx)
        else:
            break

        idx += 1

        prev_op = op
        prev_inputs = op_inputs
        prev_outputs = op_outputs
        
    return patterns, lut_count


def op_mul(tensors, buffers, opcode_idx, tensor_idx, inject_before, scale, zero_point, data, data_shape, data_dtype, data_scale, data_zero_point):
    t = tensors[tensor_idx]

    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': data_shape,
        'type': data_dtype.upper(),
        'buffer': len(buffers)-1,
        'name': 'add_data:{}'.format(tensor_idx),
        'quantization': {'scale': data_scale, 'zero_point': data_zero_point, 'details_type': 'NONE', 'quantized_dimension': 0},
        'is_variable': False,
        'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
        'type': t['type'],
        'buffer': len(buffers)-1,
        'name': 'add_{}'.format(tensor_idx),
        'quantization': {'scale': scale, 'zero_point': zero_point, 'details_type': 'NONE', 'quantized_dimension': 0},
        'is_variable': False,
        'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    inject_op = {'opcode_index': opcode_idx,
                'builtin_options_type': 'MulOptions',
                'builtin_options': {'fused_activation_function': 'NONE'},
                'custom_options_format': 'FLEXBUFFERS',
                'large_custom_options_offset': 0,
                'large_custom_options_size': 0,
                'inputs': [input_tensor, len(tensors)-2],
                'outputs': [output_tensor]}

    return inject_op, tensors, buffers


def op_add(tensors, buffers, opcode_idx, tensor_idx, inject_before, scale, zero_point, data, data_shape, data_dtype, data_scale, data_zero_point):
    t = tensors[tensor_idx]

    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': data_shape,
        'type': data_dtype.upper(),
        'buffer': len(buffers)-1,
        'name': 'add_data:{}'.format(tensor_idx),
        'quantization': {'scale': data_scale, 'zero_point': data_zero_point, 'details_type': 'NONE', 'quantized_dimension': 0},
        'is_variable': False,
        'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
        'type': t['type'],
        'buffer': len(buffers)-1,
        'name': 'add_{}'.format(tensor_idx),
        'quantization': {'scale': scale, 'zero_point': zero_point, 'details_type': 'NONE', 'quantized_dimension': 0},
        'is_variable': False,
        'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    inject_op = {'opcode_index': opcode_idx,
                'builtin_options_type': 'AddOptions',
                'builtin_options': {'fused_activation_function': 'NONE'},
                'custom_options_format': 'FLEXBUFFERS',
                'large_custom_options_offset': 0,
                'large_custom_options_size': 0,
                'inputs': [input_tensor, len(tensors)-2],
                'outputs': [output_tensor]}


    return inject_op, tensors, buffers


def op_argmax(tensors, buffers, opcode_idx, tensor_idx, inject_before, shape, dtype):
    t = tensors[tensor_idx]

    data = np.frombuffer(np.asarray([3]).astype(np.int64).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': None,
            'type': 'INT64',
            'buffer': len(buffers)-1,
            'name': 'arg_max_dim_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})
    buffers.append({'offset': 0, 'size': 0})

    tensors.append({'shape': shape,
            'type': dtype,
            'buffer': len(buffers)-1,
            'name': 'arg_max_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    inject_op = {'opcode_index': opcode_idx,
                'builtin_options_type': "ArgMaxOptions",
                'builtin_options': {'output_type': 'INT32'},
                'inputs': [input_tensor, len(tensors)-2], 
                'outputs': [output_tensor]}
    return inject_op, tensors, buffers


def op_gather(tensors, buffers, opcode_idx, tensor_idx, inject_before, data, dtype):
    t = tensors[tensor_idx]

    buffers.append({'data': np.frombuffer(data.tobytes(), dtype=np.uint8).tolist(), 'offset': 0, 'size': 0})
    tensors.append({'shape': [len(data)],
            'type': dtype,
            'buffer': len(buffers)-1,
            'name': 'gather_data_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'gather_{}'.format(tensor_idx),
            'quantization': {'scale': [0.0], 'zero_point': [0], 'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    inject_op = {'opcode_index': opcode_idx,
                'builtin_options_type': 'GatherOptions',
                "builtin_options": {
                    "axis": 0,
                    "batch_dims": 0
                },
                "custom_options_format": "FLEXBUFFERS",
                "large_custom_options_offset": 0,
                "large_custom_options_size": 0,
                'inputs': [len(tensors)-2, input_tensor],
                'outputs': [output_tensor]}

    return inject_op, tensors, buffers


def op_cast(tensors, buffers, opcode_idx, tensor_idx, inject_before, dtype):
    t = tensors[tensor_idx]
    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
            'type': dtype,
            'buffer': len(buffers)-1,
            'name': 'cast_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    inject_op = {'opcode_index': opcode_idx,
                'inputs': [input_tensor],
                'outputs': [output_tensor]}
    return inject_op, tensors, buffers


def op_resize(tensors, buffers, opcode_idx, tensor_idx, inject_before, shape):
    t = tensors[tensor_idx]

    data = np.frombuffer(np.asarray([shape[-3], shape[-2]]).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape':[2],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'resize_shape_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': shape,
            'type': t['type'],
            'buffer': len(buffers)-1,
            'name': 'resize_{}'.format(tensor_idx),
            'quantization': t['quantization'],
            'is_variable': False,
            'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    inject_op = {'opcode_index': opcode_idx,
                'builtin_options_type': 'ReshapeOptions',
                'inputs': [input_tensor, len(tensors)-2], 
                'outputs': [output_tensor]}
    return inject_op, tensors, buffers


def op_reshape(tensors, buffers, opcode_idx, tensor_idx, inject_before, shape):
    t = tensors[tensor_idx]

    data = np.frombuffer(np.asarray(shape).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [4],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'shape_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': shape,
            'type': t['type'],
            'buffer': len(buffers)-1,
            'name': 'reshape_{}'.format(tensor_idx),
            'quantization': t['quantization'],
            'is_variable': False,
            'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx


    ReshapeOptions = {'new_shape': shape}
    inject_op = {'opcode_index': opcode_idx,
                'builtin_options_type': 'ReshapeOptions',
                'builtin_options': ReshapeOptions,
                'inputs': [input_tensor, len(tensors)-2], 
                'outputs': [output_tensor]}

    return inject_op, tensors, buffers


def op_pad(tensors, buffers, opcode_idx, tensor_idx, inject_before, pad_h, pad_w):
    t = tensors[tensor_idx]
    pad = [0,0,floor(pad_h/2),ceil(pad_h/2),floor(pad_w/2),ceil(pad_w/2),0,0]

    data = np.frombuffer(np.asarray(pad).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [4,2],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'pad_shape_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
            'type': t['type'],
            'buffer': len(buffers)-1,
            'name': 'pad_{}'.format(tensor_idx),
            'quantization': t['quantization'],
            'is_variable': False,
            'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    inject_op = {'opcode_index': opcode_idx,
                'inputs': [input_tensor, len(tensors)-2], 
                'outputs': [output_tensor]}

    return inject_op, tensors, buffers


def op_dilate(tensors, buffers, opcode_idx, tensor_idx, inject_before, dilation_h, dilation_w):
    t = tensors[tensor_idx]

    data = np.frombuffer(np.asarray([1,dilation_h, dilation_w, 1]).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [4],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'dilate_shape_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    data = np.frombuffer(np.asarray(t['quantization']['zero_point']).astype(t['type'].lower()).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [],
            'type': t['type'],
            'buffer': len(buffers)-1,
            'name': 'dilate_padding_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    shape = t['shape'].copy()
    shape[-3] = (shape[-3] - 1) * dilation_h + 1
    shape[-2] = (shape[-2] - 1) * dilation_w + 1

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': shape,
            'type': t['type'],
            'buffer': len(buffers)-1,
            'name': 'dilate_{}'.format(tensor_idx),
            'quantization': t['quantization'],
            'is_variable': False,
            'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    inject_op = {'opcode_index': opcode_idx,
                'inputs': [input_tensor, len(tensors)-3,len(tensors)-2],
                'outputs': [output_tensor]}

    return inject_op, tensors, buffers


def op_quantize(tensors, buffers, opcode_idx, tensor_idx, inject_before, dtype, scale, zero_point):
    t = tensors[tensor_idx]

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
                    'type': dtype,
                    'buffer': len(buffers)-1,
                    'name': 'quantize:{}'.format(tensor_idx),
                    'quantization': {'scale': scale, 'zero_point': zero_point, 'details_type': 'NONE', 'quantized_dimension': 0},
                    'is_variable': False,
                    'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    inject_op = {'opcode_index': opcode_idx,
                 'inputs': [input_tensor],
                 'outputs': [output_tensor]}

    return inject_op, tensors, buffers


def op_split(tensors, buffers, opcode_idx, tensor_idx, inject_before, axis, num_splits):
    t = tensors[tensor_idx]

    data = np.frombuffer(np.asarray([axis]).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'split_shape_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    shape = t['shape'].copy()
    SplitOptions = {'num_splits': num_splits}
    assert(shape[axis] % num_splits == 0)
    shape[axis] = shape[axis] // num_splits

    for n in range(num_splits):
        buffers.append({'offset': 0, 'size': 0})
        tensors.append({'shape': shape,
                'type': t['type'],
                'buffer': len(buffers)-1,
                'name': 'split_{}_{}'.format(tensor_idx, n),
                'quantization': t['quantization'],
                'is_variable': False,
                'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        assert(num_splits == 1)
        input_tensor, output_tensor = len(tensors)-1, tensor_idx
        inject_op = {'opcode_index': opcode_idx,
                    'builtin_options_type': 'SplitOptions',
                    'builtin_options': SplitOptions,
                    'inputs': [len(tensors)-2, input_tensor], 
                    'outputs': [output_tensor]}
    else:
        inject_op = {'opcode_index': opcode_idx,
                    'builtin_options_type': 'SplitOptions',
                    'builtin_options': SplitOptions,
                    'inputs': [len(tensors)-1-num_splits, input_tensor], 
                    'outputs': [len(tensors) - num_splits + _ for _ in range(num_splits)]}

    return inject_op, tensors, buffers


def op_strided_slice(tensors, buffers, opcode_idx, tensor_idx, inject_before, begin, end, stride):
    t = tensors[tensor_idx]

    data = np.frombuffer(np.asarray(begin).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [4],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'slice_begin_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    data = np.frombuffer(np.asarray(end).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [4],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'slice_end_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    data = np.frombuffer(np.asarray(stride).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [4],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'slice_stride_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    shape = t['shape'].copy()
    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': shape,
            'type': t['type'],
            'buffer': len(buffers)-1,
            'name': 'slice_{}'.format(tensor_idx),
            'quantization': t['quantization'],
            'is_variable': False,
            'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    StridedSliceOptions = {'begin_mask': 0, 'end_mask': 0, 'ellipsis_mask': 0, 'new_axis_mask': 0, 'shrink_axis_mask': 0, 'offset': False}


    inject_op = {'opcode_index': opcode_idx,
                'builtin_options_type': 'StridedSliceOptions',
                'builtin_options': StridedSliceOptions,
                'inputs': [input_tensor, len(tensors)-4, len(tensors)-3, len(tensors)-2], 
                'outputs': [output_tensor]}

    return inject_op, tensors, buffers


def op_concat(tensors, buffers, opcode_idx, tensor_indices, tensor_idx, axis):
    t = tensors[tensor_idx]

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
            'type': t['type'],
            'buffer': len(buffers)-1,
            'name': 'concat_{}'.format(tensor_idx),
            'quantization': t['quantization'],
            'is_variable': False,
            'has_rank': True})

    inject_op = {'opcode_index': opcode_idx,
                'builtin_options_type': 'ConcatenationOptions',
                'builtin_options': {'axis': axis, 'fused_activation_function': 'NONE'},
                'inputs': tensor_indices,
                'outputs': [len(tensors) - 1]}

    return inject_op, tensors, buffers


def op_conv(tensors, buffers, opcode_idx, tensor_idx, inject_before, opts, output_shape, output_quantization, weights, weight_quantization, biases, bias_quantization):
    t = tensors[tensor_idx]

    data = np.frombuffer(np.asarray(weights).astype(np.int8).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': weights.shape,
            'type': 'INT8',
            'buffer': len(buffers)-1,
            'name': 'conv_weights_{}'.format(tensor_idx),
            'quantization': weight_quantization,
            'is_variable': False,
            'has_rank': True})

    if not biases is None:
        data = np.frombuffer(np.asarray(biases).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
        buffers.append({'data': data, 'offset': 0, 'size': 0})
        tensors.append({'shape': biases.shape,
                'type': 'INT32',
                'buffer': len(buffers)-1,
                'name': 'conv_biases_{}'.format(tensor_idx),
                'quantization': bias_quantization,
                'is_variable': False,
                'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': output_shape,
            'type': 'INT8',
            'buffer': len(buffers)-1,
            'name': 'conv_{}'.format(tensor_idx),
            'quantization': output_quantization,
            'is_variable': False,
            'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    if not biases is None:
        inject_op = {'opcode_index': opcode_idx,
                    'builtin_options_type': 'Conv2DOptions',
                    'builtin_options': opts,
                    'inputs': [input_tensor, len(tensors)-3, len(tensors)-2], 
                    'outputs': [output_tensor]}
    else:
        inject_op = {'opcode_index': opcode_idx,
                    'builtin_options_type': 'Conv2DOptions',
                    'builtin_options': opts,
                    # 'inputs': [input_tensor, len(tensors)-3, -1], 
                    'inputs': [input_tensor, len(tensors)-2], 
                    'outputs': [output_tensor]}

    return inject_op, tensors, buffers


def get_numpy_data(tensor, buffers):
    # shape = tensor['shape']
    shape = tensor.get('shape', (1))
    dtype = np.dtype(tensor['type'].lower())
    raw_data = bytearray(buffers[tensor['buffer']]['data'])
    data = np.frombuffer(raw_data, dtype=dtype).reshape(shape)

    return data


def lut_i8_to_u8(arr):
    arr2 = arr.copy()
    arr2[0:128] = arr[128:256]
    arr2[128:256] = arr[0:128]

    return arr2


def op_lut(tensors, buffers, opcode_quantize, opcode_cast, opcode_gather, opcode_split, opcode_concat, output_data, i_idx, o_idx):
    ops = []

    i_tensor = tensors[i_idx]
    o_tensor = tensors[o_idx]
    i_quant = None
    if 'quantization' in i_tensor:
        i_quant = i_tensor['quantization']
    if 'scale' not in i_quant:
        i_quant['scale'] = [1.0]
    if 'zero_point' not in i_quant:
        i_quant['zero_point'] = [0]
    i_type = i_tensor['type']

    o_quant = None
    if 'quantization' in o_tensor:
        o_quant = o_tensor['quantization']
    o_type = o_tensor['type']

    assert(i_tensor['shape'] == o_tensor['shape'])

    current_idx = i_idx

    if i_type in ['INT8']:
        output_data = lut_i8_to_u8(output_data)
        quant_op, tensor, buffers = op_quantize(tensors, buffers, opcode_quantize, current_idx, False, 'UINT8', i_quant['scale'], [_ + 128 for _ in i_quant['zero_point']])
    else:
        quant_op, tensor, buffers = op_quantize(tensors, buffers, opcode_quantize, current_idx, False, 'UINT8', i_quant['scale'], i_quant['zero_point'])

    ops.append(quant_op)
    current_idx = len(tensors)-1

    cast_op, tensors, buffers = op_cast(tensors, buffers, opcode_cast, current_idx, False, 'INT32')
    ops.append(cast_op)
    current_idx = len(tensors)-1

    lut_count = 1
    if len(output_data.shape) > 1:
        lut_count = output_data.shape[-1]

    if lut_count == 1:
        gather_op, tensors, buffers = op_gather(tensors, buffers, opcode_gather, o_idx, True, output_data, o_type)
        gather_op['inputs'][1] = ops[-1]['outputs'][0]
        ops.append(gather_op)
    else:
        #gen split
        num_splits = lut_count
        split_op, tensors, buffers = op_split(tensors, buffers, opcode_split, current_idx, False, -1, num_splits)
        ops.append(split_op)

        o_shape = o_tensor['shape'].copy()
        o_shape[-1] = 1

        concat_inputs = []
        for n in range(num_splits):
            buffers.append({'offset': 0, 'size': 0})
            tensors.append({'shape': o_shape,
                    'type': o_type,
                    'buffer': len(buffers)-1,
                    'name': 'gather_{}_{}'.format(n, o_idx),
                    'quantization': o_quant,
                    'is_variable': False,
                    'has_rank': True})
            gather_op, tensors, buffers = op_gather(tensors, buffers, opcode_gather, len(tensors)-1, True, output_data[:,n], o_type)
            gather_op['inputs'][1] = split_op['outputs'][n]

            concat_inputs.append(gather_op['outputs'][0])
            ops.append(gather_op)

        concat_op, tensors, buffers = op_concat(tensors, buffers, opcode_concat, concat_inputs, o_idx, -1)
        ops.append(concat_op)

    return ops, tensors, buffers


def op_group_conv(tensors, buffers, opcode_split, opcode_conv, opcode_concat, op):
    opts = op['builtin_options']
    ishape = tensors[op['inputs'][0]]['shape'].copy()
    oshape = tensors[op['outputs'][0]]['shape'].copy()
    oquant = tensors[op['outputs'][0]]['quantization']

    f_tensor = tensors[op['inputs'][1]]
    b_tensor = None
    if len(op['inputs']) > 2 and op['inputs'][2] != -1:
        b_tensor = tensors[op['inputs'][2]]

    filter_data = get_numpy_data(f_tensor, buffers)
    wquant = f_tensor['quantization']
    k, h, w, c = tuple(f_tensor['shape'])

    bias_data = np.zeros((k,), dtype=np.int64)
    bquant = {'scale': [1. for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 0}
    # bias_data, bquant = None, None
    if not (b_tensor is None):
        bias_data = get_numpy_data(b_tensor, buffers).astype(np.int64)
        bquant = b_tensor['quantization']

    tensor_idx = op['inputs'][0]

    ops = []
    num_splits = ishape[-1] // f_tensor['shape'][-1]
    group_size = ishape[-1] // num_splits
    oshape[-1] = oshape[-1] // num_splits

    #gen split
    split_op, tensors, buffers = op_split(tensors, buffers, opcode_split, tensor_idx, False, -1, num_splits)
    ops.append(split_op)

    concat_inputs = []
    for n in range(num_splits):
        weights = filter_data[n*group_size:(n+1)*group_size].copy()
        biases = bias_data[n*group_size:(n+1)*group_size].copy()

        bias_quant = bquant.copy()
        bias_quant['zero_point'] = bias_quant['zero_point'][n*group_size:(n+1)*group_size].copy()
        bias_quant['scale'] = bias_quant['scale'][n*group_size:(n+1)*group_size].copy()

        weight_quant = wquant.copy()
        weight_quant['zero_point'] = weight_quant['zero_point'][n*group_size:(n+1)*group_size].copy()
        weight_quant['scale'] = weight_quant['scale'][n*group_size:(n+1)*group_size].copy()

        conv_op, tensors, buffers = op_conv(tensors, buffers, opcode_conv, split_op['outputs'][n], False, opts, oshape, oquant, weights, weight_quant, biases, bias_quant)
        concat_inputs.append(conv_op['outputs'][0])
        ops.append(conv_op)

    concat_op, tensors, buffers = op_concat(tensors, buffers, opcode_concat, concat_inputs, op['outputs'][0], -1)
    ops.append(concat_op)

    return ops, tensors, buffers


def get_pad(op, tensors):
    opts = None
    if 'builtin_options' in op:
        opts = op['builtin_options']
    if opts['padding'] == 'VALID':
        return 0, 0

    stride_h, stride_w = opts['stride_h'], opts['stride_w']
    dilation_h, dilation_w = opts['dilation_h_factor'], opts['dilation_w_factor']
    f_tensor = tensors[op['inputs'][1]]
    k, h, w, c = tuple(f_tensor['shape'])
    kernel_h, kernel_w = h, w
    i_h, i_w = tensors[op['inputs'][0]]['shape'][-3], tensors[op['inputs'][0]]['shape'][-2]
    o_h, o_w = tensors[op['outputs'][0]]['shape'][-3], tensors[op['outputs'][0]]['shape'][-2]

    pad_h = max(stride_h * (o_h - 1) - i_h + kernel_h + (kernel_h-1)*(dilation_h-1), 0)
    pad_w = max(stride_w * (o_w - 1) - i_w + kernel_w + (kernel_w-1)*(dilation_w-1), 0)

    return pad_h,pad_w


def apply_transformation(transform, operators, tensors, buffers, opcodes, builtin_codes, debug):

    if transform == "IDENTITY_INJECTION":
        splits,_,_ = get_splits2(create_graph(operators, tensors, buffers, opcodes))

        prev_op = None
        for i in range(len(operators)):
            op = operators[i]
            for split in splits:
                if i == split[0]:
                    connected, forked = True, False
                    if prev_op != None:
                        connected = any([_ in prev_op['outputs'] for _ in op['inputs']])
                        forked, _ = is_forked(prev_op, operators, tensors)
                    opcode = builtin_codes[op['opcode_index']]
            prev_op = op

    id_count = 0
    pattern_count = 0
    i = 0
    while i < len(operators): 
        op = operators[i]
        opts = None
        if 'builtin_options' in op:
            opts = op['builtin_options']
        opcode = builtin_codes[op['opcode_index']]

        if transform == 'REMOVE_CONSTANTS':
            if opcode in ['QUANTIZE']:
                activations = [_ for _ in op['inputs'] if not ('data' in buffers[tensors[_]['buffer']])]
                if len(activations) == 0:
                    iq = tensors[op['inputs'][0]]['quantization']
                    oq = tensors[op['outputs'][0]]['quantization']
                    ot = tensors[op['outputs'][0]]

                    data = get_numpy_data(tensors[op['inputs'][0]], buffers)
                    data = (data - iq['zero_point'][0])* iq['scale'][0] # get real FP value
                    data = ((data / oq['scale'][0]) + oq['zero_point'][0]).astype(ot['type'].lower()) # quantize to constant

                    buffers.append({'data': np.frombuffer(data.tobytes(), dtype=np.uint8).tolist(), 'offset': 0, 'size': 0})
                    ot['buffer'] = len(buffers)-1
                    operators = operators[:i] + operators[i+1:]

            if opcode in ['MAXIMUM', 'MINIMUM'] and 'data' in tensors[op['inputs'][1]]:
                t = tensors[op['inputs'][1]]
                data = get_numpy_data(t, buffers)
                if t['type'] in ['FLOAT32']:
                    data = data.astype(np.int8)
                    t['type'] = 'INT8'
                    buffers.append({'data': np.frombuffer(data.tobytes(), dtype=np.uint8).tolist(), 'offset': 0, 'size': 0})
                    t['buffer'] = len(buffers) - 1

                if len(data.flatten()) > 1 and np.all(data.flatten() == data.flatten()[0]):
                    t['shape'] = []
                    data = data.flatten()[:1]
                    buffers.append({'data': np.frombuffer(data.tobytes(), dtype=np.uint8).tolist(), 'offset': 0, 'size': 0})
                    t['buffer'] = len(buffers) - 1

            if opcode in ['SHAPE']:
                pattern = [i] 
                _op = op
                _next_ops = None
                while True:
                    _next_ops = [_ for _ in range(len(operators)) if _op['outputs'][0] in operators[_]['inputs']]
                    if len(_next_ops) == 1:
                        next_inputs = operators[_next_ops[0]]['inputs']
                        num_inputs = [_ for _ in next_inputs if 'buffer' in tensors[_] and not 'data' in buffers[tensors[_]['buffer']]]
                        if len(num_inputs) == 1:
                            _op = operators[_next_ops[0]]
                            pattern.append(_next_ops[0])
                        else:
                            break
                    else:
                        pattern = []
                        break
                if len(pattern):
                    tmp_dir_obj = tempfile.TemporaryDirectory()
                    graph = create_graph([operators[_] for _ in pattern], tensors, buffers, opcodes)
                    save_graph('pattern.{}.tflite'.format(pattern_count), tmp_dir_obj, graph, copy=debug)

                    with open(os.path.join(tmp_dir_obj.name, 'pattern.{}.tflite'.format(pattern_count)), 'rb') as f:
                        inputs, outputs = generate_inputs_outputs(f.read(), int8_range=True)
                    output_data = outputs['o0'].squeeze()
                    if not tmp_dir_obj is None:
                        tmp_dir_obj.cleanup()
                    buffers.append({'data': np.frombuffer(output_data.tobytes(), dtype=np.uint8).tolist(), 'offset': 0, 'size': 0})
                    tensors[_op['outputs'][0]]['buffer'] = len(buffers)-1
                    for _ in reversed(pattern):
                        operators = operators[:_] + operators[_+1:]

        elif transform == 'CLEAN_LOGISTIC':
            if opcode in ['LOGISTIC']:
                tensors[op['outputs'][0]]['quantization']['scale'][0] =  1. / 256

        elif transform == 'REMOVE_FP32_IO':
            if opcode in ['DEQUANTIZE', 'QUANTIZE']:
                if is_graph_input(op, operators, tensors, buffers):
                    if 'FLOAT32' in [tensors[_]['type'] for _ in op['inputs']]:
                        operators = operators[:i] + operators[i+1:]
                        i -= 1 #rerun 
                elif is_graph_output(op, operators, tensors, buffers):
                    if 'FLOAT32' in [tensors[_]['type'] for _ in op['outputs']]:
                        operators = operators[:i] + operators[i+1:]
                        i -= 1 #rerun 

        elif transform == 'LUT':
            pattern, lut_count = lut_pattern(operators, builtin_codes, tensors, buffers, i)

            if len(pattern):
                last_op = operators[i + len(pattern)-1]

                tmp_dir_obj = tempfile.TemporaryDirectory()
                graph = create_graph([operators[_] for _ in pattern], tensors, buffers, opcodes, force_shape=[1,1,256,lut_count])
                save_graph('pattern.{}.tflite'.format(pattern_count), tmp_dir_obj, graph, copy=debug)

                with open(os.path.join(tmp_dir_obj.name, 'pattern.{}.tflite'.format(pattern_count)), 'rb') as f:
                    inputs, outputs = generate_inputs_outputs(f.read(), int8_range=True)
                output_data = outputs['o0'].squeeze()
                if not tmp_dir_obj is None:
                    tmp_dir_obj.cleanup()

                inputs = [_ for _ in op['inputs'] if not 'data' in buffers[tensors[_]['buffer']]]
                assert(len(inputs) == 1)
                lut_ops, tensors, buffers = op_lut(tensors, buffers, builtin_codes.index('QUANTIZE'), builtin_codes.index('CAST'), builtin_codes.index('GATHER'), builtin_codes.index('SPLIT'), builtin_codes.index('CONCATENATION'), output_data, inputs[0], last_op['outputs'][0])

                operators = operators[:i] + lut_ops + operators[i+len(pattern):]

                for n,next_op in enumerate(operators):
                    if last_op['outputs'][0] in next_op['inputs']:
                        next_op['inputs'] = [lut_ops[-1]['outputs'][0] if _ == last_op['outputs'][0] else _ for _ in next_op['inputs']]

                pattern_count += 1
                i += len(lut_ops) - 1
            elif debug:
                print("No LUT transformation applied within the graph")

        elif transform == 'TRANSPOSE_CONV': 
            if opcode in ['TRANSPOSE_CONV']:
                if opts['stride_w'] == opts['stride_h'] and opts['stride_w'] <= 2:
                    i_tensor = tensors[op['inputs'][2]]
                    o_tensor = tensors[op['outputs'][0]]

                    f_tensor = tensors[op['inputs'][1]]
                    b_tensor = None
                    if len(op['inputs']) > 3 and op['inputs'][3] != -1:
                        b_tensor = tensors[op['inputs'][3]]
                    weights = get_numpy_data(f_tensor, buffers)
                    weight_quant = f_tensor['quantization']
                    k, h, w, c = tuple(f_tensor['shape'])
                    kernel_h, kernel_w = h, w

                    biases = np.zeros((k,), dtype=np.int64)
                    bias_quant = {'scale': [1. for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 0}
                    # biases, bias_quant = None, None
                    if not (b_tensor is None):
                        biases = get_numpy_data(b_tensor, buffers).astype(np.int64)
                        bias_quant = b_tensor['quantization']

                    pad_h, pad_w = 0,0
                    if opts['padding'] == 'VALID':
                        kh, kw = kernel_h, kernel_w
                        pl = kw - 1
                        pr = kw - 1
                        pu = kh - 1
                        pd = kh - 1
                        pad_w = pl + pr
                        pad_h = pu + pd

                    elif opts['padding'] == 'SAME':
                        stride_h, stride_w, = opts['stride_h'], opts['stride_w']

                        i_h, i_w = i_tensor['shape'][-3], i_tensor['shape'][-2]
                        o_h, o_w = o_tensor['shape'][-3], o_tensor['shape'][-2]

                        pad_h = ((i_h * stride_h) + (kernel_h-stride_h)) - o_h
                        pad_w = ((i_w * stride_w) + (kernel_w-stride_w)) - o_w

                        if (pad_w > 0 or pad_h > 0):
                            pl = (kernel_w - (pad_w//2) - 1)
                            pr = (kernel_w - ((pad_w//2) + (pad_w%2)) - 1)
                            pu = (kernel_h - (pad_h//2) - 1)
                            pd = (kernel_h - ((pad_h//2) + (pad_h%2)) - 1)
                            pad_w = pl + pr
                            pad_h = pu + pd

                    tensor_idx = op['inputs'][2]
                    ops = []

                    conv_opts = opts.copy()
                    if opts['stride_h'] > 1 or opts['stride_w'] > 1:
                        dilate_op, tensors, buffers = op_dilate(tensors, buffers, builtin_codes.index('DILATE'), tensor_idx, False, opts['stride_h'], opts['stride_w'])
                        ops.append(dilate_op)

                        conv_opts['stride_h'] = 1
                        conv_opts['stride_w'] = 1

                    if pad_h > 0 or pad_w > 0:
                        pad_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PAD'), len(tensors)-1, False, pad_h, pad_w)
                        pshape = tensors[len(tensors)-1]['shape']
                        pshape[-3] += pad_h
                        pshape[-2] += pad_w

                        ops.append(pad_op)

                    oquant = o_tensor['quantization']
                    oshape = tensors[len(tensors)-1]['shape'].copy()
                    oshape[-3] -= kernel_h-1
                    oshape[-2] -= kernel_w-1

                    conv_op, tensors, buffers = op_conv(tensors, buffers, builtin_codes.index('CONV_2D'), len(tensors)-1, False, conv_opts, oshape, oquant, weights, weight_quant, biases, bias_quant)
                    ops.append(conv_op)

                    # adjust
                    for n,next_op in enumerate(operators):
                        if op['outputs'][0] in next_op['inputs']:
                            next_op['inputs'] = [conv_op['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]

                    #inject DILATE-PAD-CONV and skip ahead n
                    operators = operators[:i] + ops + operators[i+1:]

                    i += len(ops)-1
                else:
                    print("TRANSPOSE_CONV with stride_h != stride_w or stride > 2 is not supported")

        elif transform == 'GROUP_CONV': 
            if opcode in ['CONV_2D']:
                ishape = tensors[op['inputs'][0]]['shape']
                fshape = tensors[op['inputs'][1]]['shape']
                if fshape[-1] < ishape[-1]:
                    inject_ops, tensors, buffers = op_group_conv(tensors, buffers, builtin_codes.index('SPLIT'), builtin_codes.index('CONV_2D'), builtin_codes.index('CONCATENATION'), op)

                    # adjust next op inputs, to go from CONCATENATION instead of CONV
                    for n,next_op in enumerate(operators):
                        if op['outputs'][0] in next_op['inputs']:
                            next_op['inputs'] = [inject_ops[-1]['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]

                    # inject SPLIT-CONV-CONCAT and skip ahead n
                    operators = operators[:i] + inject_ops + operators[i+1:]

                    i += len(inject_ops)-1


        elif transform == 'STRIDED_DEPTHWISE': 
            if opcode in ['DEPTHWISE_CONV_2D']:
                if opts['stride_w'] > 1:
                    _, pad_w = get_pad(op, tensors)
                    stride_w, dilation_w = opts['stride_w'], opts['dilation_w_factor']
                    filter_w, input_w = tensors[op['inputs'][1]]['shape'][-2], tensors[op['inputs'][0]]['shape'][-2]
                    adjusted_conv_width = input_w + pad_w - (((filter_w-1)*dilation_w)+1) + 1 

                    #create strided slice
                    t = tensors[op['outputs'][0]]
                    begin = [0, 0, 0, 0]
                    end = t['shape'].copy()
                    end[-2] = adjusted_conv_width
                    stride = [1, 1, opts['stride_w'], 1]

                    tensor_idx = op['outputs'][0]
                    inject_op, tensors, buffers = op_strided_slice(tensors, buffers, builtin_codes.index('STRIDED_SLICE'), tensor_idx, False, begin, end, stride)

                    #adjust DEPTHWISE_CONV_2D
                    t = tensors[op['outputs'][0]]
                    t['shape'][-2] = adjusted_conv_width
                    opts['stride_w'] = 1

                    #adjust inputs to use STRIDED_SLICE not CONV
                    for n,next_op in enumerate(operators):
                        if op['outputs'][0] in next_op['inputs']:
                            next_op['inputs'] = [inject_op['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]

                    # inject STRIDED_SLICE and skip ahead 1
                    operators = operators[:i+1] + [inject_op] + operators[i+1:]
                    i += 1
                elif debug:
                    print("STRIDED_DEPTHWISE with stride_w <= 1 or not supported configuration")

        elif transform == 'EXPLICIT_PAD': 
            if opcode in ['CONV_2D', 'DEPTHWISE_CONV_2D']:
                pad_h, pad_w = get_pad(op, tensors)
                if pad_h == 0 and pad_w == 0:
                    opts['padding'] = 'VALID'
                else:
                    tensor_idx = op['inputs'][0]
                    shared_ops = []
                    shared_indices = []
                    for s,op_ in enumerate(operators):
                        if s != i:
                            if tensor_idx in op_['inputs']:
                                shared_ops.append(op_)
                                shared_indices.append(s)

                    inject_shared = False
                    if len(shared_ops):
                        if all([builtin_codes[_['opcode_index']] in ['CONV_2D', 'DEPTHWISE_CONV_2D'] for _ in shared_ops]):
                            if all(get_pad(_, tensors) == (pad_h,pad_w) for _ in shared_ops):
                                inject_shared = True

                    if inject_shared:
                        # create PAD
                        inject_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PAD'), tensor_idx, True, pad_h, pad_w)

                        # adjust previous op outputs, to go to PAD instead of CONV
                        for p,prev_op in enumerate(operators):
                            if tensor_idx in prev_op['outputs']:
                                prev_op['outputs'] = [inject_op['inputs'][0] if _ == tensor_idx else _ for _ in prev_op['outputs']]

                        # set CONVs who share padding to VALID
                        for s in shared_indices:
                            operators[s]['builtin_options']['padding'] = 'VALID'
                    else:
                        # create PAD
                        inject_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PAD'), tensor_idx, False, pad_h, pad_w)
                        injected_tensor_idx = inject_op['outputs'][0]
                        operators[i]['inputs'] = [injected_tensor_idx if _ == tensor_idx else _ for _ in operators[i]['inputs']]
                        tensor_idx = injected_tensor_idx

                    # adjust CONV
                    ishape = tensors[tensor_idx]['shape'].copy()
                    ishape[-3] += pad_h
                    ishape[-2] += pad_w
                    tensors[tensor_idx]['shape'] = ishape
                    operators[i]['builtin_options']['padding'] = 'VALID'

                    # inject PAD and skip ahead 1
                    operators = operators[:i] + [inject_op] + operators[i:]
                    i += 1
            
        elif transform == 'YOLO_ARG_MAX':
            if is_graph_output(op, operators, tensors, buffers):
                if tensors[op['outputs'][0]]['shape'][-1] in [80]:
                    id_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PAD'), op['outputs'][0], False, 0, 0)

                    _shape = tensors[op['outputs'][0]]['shape'].copy()[:-1]
                    arg_max_op, tensors, buffers = op_argmax(tensors, buffers, builtin_codes.index('ARG_MAX'), op['outputs'][0], False, _shape, 'INT32')

                    _reshape = _shape.copy() + [1]
                    reshape_op, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), len(tensors)-1, False, _reshape)
        
                    cast_op, tensors, buffers = op_cast(tensors, buffers, builtin_codes.index('CAST'), len(tensors)-1, False, 'UINT8')
                    operators = operators[:i+1] + [id_op, arg_max_op, reshape_op, cast_op] + operators[i+1:]
                    i += 4
                elif debug:
                    print("YOLO_ARG_MAX: Output tensor shape does not match expected class count (80).")

        elif transform == 'IDENTITY_INJECTION':
            for split in splits:
                # if i == (split[0] + id_count):
                if False:
                    if opcode not in ['CONCATENATION', 'DEPTHWISE_CONV_2D', 'CONV_2D', 'TRANSPOSE_CONV', 'FULLY_CONNECTED', 'UNIDIRECTIONAL_SEQUENCE_LSTM', "PACK", "SPLIT", "SPLIT_V", "TILE", "STRIDED_SLICE"]:
                        tensor_idx = [_ for _ in op['inputs'] if 'data' not in tensors[_]][0]
                        id_op, tensors, buffers = op_split(tensors, buffers, builtin_codes.index('SPLIT'), tensor_idx, True, -1, 1)
                        # adjust previous op outputs, to go to SPLIT instead of current
                        for p,prev_op in enumerate(operators):
                            if tensor_idx in prev_op['outputs']:
                                prev_op['outputs'] = [id_op['inputs'][1] if _ == tensor_idx else _ for _ in prev_op['outputs']]
                        operators = operators[:i] + [id_op] + operators[i:]
                        i += 1
                        id_count += 1
                    break

        elif transform == 'CHANNELS_FIRST': 
            pass
        else:
            print('warning: unknown transformation', transform)
        i += 1

    return operators, tensors, buffers


def load_graph(src_tflite, copy=False):
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name

    tmp_tflite = os.path.join(tmp_dir, os.path.basename(src_tflite))
    shutil.copyfile(src_tflite, tmp_tflite)

    # Convert TFLITE to JSON, read graph from JSON
    jname = tflite2json(tmp_tflite)
    if copy:
        shutil.copyfile(jname, src_tflite.replace('.tflite', '.json'))
    with open(jname) as f:
        graph = json.load(f)

    subgraph = graph['subgraphs'][0]
    operators, tensors = subgraph['operators'], subgraph['tensors']
    for t in tensors:
        if 'shape' in t and 'shape_signature' in t:
            t['shape_signature'] = t['shape']
    buffers, opcodes = graph['buffers'], clean_operator_codes(graph['operator_codes'])
    input_tensors, output_tensors = get_io_tensors(operators, tensors, buffers, opcodes)
    graph = create_signature_defs(graph, tensors, input_tensors, output_tensors)

    return graph, tmp_dir_obj


def save_graph(dst_tflite, tmp_dir_obj, graph, copy=False, debug=False):
    if tmp_dir_obj is None:
        tmp_dir = 'temp'
    else:
        tmp_dir = tmp_dir_obj.name
    tmp_tflite = os.path.join(tmp_dir, os.path.basename(dst_tflite))
    tmp_json = tmp_tflite.replace('.tflite', '.json')

    with open(tmp_json, 'w') as f:
        json.dump(graph, f)
    tname = json2tflite(tmp_json)
    if copy:
        if debug:
            shutil.copyfile(tmp_json, dst_tflite.replace('.tflite', '.json'))
        shutil.copyfile(tname, dst_tflite)
    return tmp_tflite


def clean_operator_codes(operator_codes):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    with open(schema_path.replace('.fbs', '.json')) as f:
        schema = json.load(f)
        operators = schema["definitions"]["tflite_BuiltinOperator"]["enum"]
        for i,g in enumerate(operator_codes):
            if 'deprecated_builtin_code' in g and not g['deprecated_builtin_code'] >= 127:
                operator_codes[i]['builtin_code'] = operators[g['deprecated_builtin_code']]
    return operator_codes


def tflite2json(tflite_model):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    dir = os.path.dirname(tflite_model)
    if dir == '':
        dir = './'
    jname = os.path.join(dir, os.path.basename(tflite_model).replace('.tflite','.json'))

    cmd = 'flatc -t --strict-json --defaults-json -o {} {} -- {}'.format(dir, schema_path, tflite_model)
    subprocess.run(shlex.split(cmd))

    return jname


def json2tflite(json_model):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    dir = os.path.dirname(json_model)
    if dir == '':
        dir = './'
    tname = os.path.join(dir, os.path.basename(json_model).replace('.json','.tflite'))
        
    cmd = 'flatc -b --strict-json --defaults-json -o {} {} {}'.format(dir, schema_path, json_model)
    subprocess.run(shlex.split(cmd))

    return tname


def visit(n, operators, L, P, T):
    # function visit(node n)
    #     if n has a permanent mark then
    #         return
    #     if n has a temporary mark then
    #         stop   (graph has at least one cycle)

    #     mark n with a temporary mark

    #     for each node m with an edge from n to m do
    #         visit(m)

    #     mark n with a permanent mark
    #     add n to head of L

    if n in P:
        return L, P, T
    if n in T:
        return None, P, T

    T.append(n)
    m = []
    for o, op_ in enumerate(operators):
        for i in op_['inputs']:
            if i in operators[n]['outputs']:
                m.append(o)
                break
    for o in m:
        L, P, T = visit(o, operators, L, P, T)

    P.append(n)
    L = [n] + L
    return L, P, T


def topological_sort_dfs(operators, tensors, buffers):
    # L ← Empty list that will contain the sorted nodes
    # while exists nodes without a permanent mark do
    #     select an unmarked node n
    #     visit(n)
    L, P, T = [], [], []
    while len(P) < len(operators):
        n = [o for o in range(len(operators)) if o not in P][0]
        L, P, T = visit(n, operators, L, P, T)
        if L is None:
            break
    return L

def topological_sort_kahn(operators, tensors, buffers):
    # L ← Empty list that will contain the sorted elements
    # S ← Set of all nodes with no incoming edge

    # while S is not empty do
    #   remove a node n from S
    #   add n to L
    #   for each node m with an edge e from n to m do
    #     remove edge e from the graph
    #     if m has no other incoming edges then
    #         insert m into S

    # if graph has edges then
    #   return error   (graph has at least one cycle)
    # else 
    #   return L   (a topologically sorted order)
    L = []
    S = set()

    # get S
    input_tensors = []
    output_tensors = []
    for op in operators:
        input_tensors += op['inputs']
        output_tensors += op['outputs']
    io_input = [_ for _ in set(input_tensors) if _ not in output_tensors and not 'data' in buffers[tensors[_]['buffer']]]
    io_output = [_ for _ in set(output_tensors) if _ not in input_tensors and not 'data' in buffers[tensors[_]['buffer']]]


    for o, op in enumerate(operators):
        for i in op['inputs']:
            if i in io_input:
                S.add(o)
                break

    edges = set([_ for _ in set(input_tensors+output_tensors) if not 'data' in buffers[tensors[_]['buffer']] and _ not in io_input])

    while len(S):
        n = S.pop()
        L.append(n)
        op = operators[n]
        op_edges = [_ for _ in op['outputs'] if _ in edges]
        m = []
        for o,_op in enumerate(operators):
            for i in _op['inputs']:
                if i in op_edges:
                    m.append(o)
                    break
        for e in op_edges:
            edges.remove(e)
        for o in m:
            if all([not _ in edges for _ in operators[o]['inputs']]):
                S.add(o)

    if len(edges) == 0:
        return L
    else:
        return None
        



def clean_operators(operators, tensors, buffers, rename=False, force_shape=None):
    used_buffers = []
    used_tensors = []

    sbuffers = [{}]
    stensors = []
    soperators = []

    for op in operators:
        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]
        op_intermediates = []

        if 'intermediates' in op:
            op_intermediates = [_ for _ in op['intermediates'] if _ != -1]
        for io in op_inputs + op_outputs + op_intermediates:
            used_buffers.append(tensors[io]['buffer'])
            used_tensors.append(io)

        soperators.append(op)

    s = None
    # try:
    #     import time
    #     start = time.time()
    #     # s = topological_sort_dfs(soperators, tensors, buffers)
    #     s = topological_sort_kahn(soperators, tensors, buffers)
    #     end = time.time()
    #     print(end - start)
    # except:
    #     pass
    if not s is None:
        soperators = [soperators[_] for _ in s]
        
    remap_tensors = {}
    rt = 0
    for t in range(len(tensors)):
        if t not in used_tensors:
            pass
        else:
            stensors.append(tensors[t])
            remap_tensors[t] = rt
            rt += 1

    # for op_idx in range(len(soperators)):
    #     for t in soperators[op_idx]['inputs']:
    #         if t != -1 and t in used_tensors and not t in remap_tensors:
    #             stensors.append(tensors[t])
    #             remap_tensors[t] = rt
    #             rt += 1
    #     for t in soperators[op_idx]['outputs']:
    #         if t != -1 and t in used_tensors and not t in remap_tensors:
    #             stensors.append(tensors[t])
    #             remap_tensors[t] = rt
    #             rt += 1

    # for op_idx in range(len(soperators)):
    #     for t in soperators[op_idx]['inputs']:
    #         if t != -1 and t in used_tensors and not t in remap_tensors:
    #             stensors.append(tensors[t])
    #             remap_tensors[t] = rt
    #             rt += 1

    # for op_idx in range(len(soperators)):
    #     for t in soperators[op_idx]['outputs']:
    #         if t != -1 and t in used_tensors and not t in remap_tensors:
    #             stensors.append(tensors[t])
    #             remap_tensors[t] = rt
    #             rt += 1

    for op_idx in range(len(soperators)):
        soperators[op_idx]['inputs'] = [remap_tensors[_] for _ in soperators[op_idx]['inputs'] if _ != -1]
        soperators[op_idx]['outputs'] = [remap_tensors[_] for _ in soperators[op_idx]['outputs'] if _ != -1]

    remap_buffers = {}
    rb = 1
    for b in range(len(buffers)):
        if b not in used_buffers:
            pass
        else:
            sbuffers.append(buffers[b])
            remap_buffers[b] = rb
            rb += 1

    for t_idx in range(len(stensors)):
        if 'buffer' in stensors[t_idx] and stensors[t_idx]['buffer'] in remap_buffers.keys():
            stensors[t_idx]['buffer'] = remap_buffers[stensors[t_idx]['buffer']]

    if not force_shape is None:
        for t in stensors:
            if 'buffer' in t and not 'data' in sbuffers[t['buffer']]:
                t['shape'] = force_shape

    return soperators, stensors, sbuffers


def get_io_tensors(operators, tensors, buffers, opcodes):
    ignored_output_codes = []
    builtin_codes = [_['builtin_code'] for _ in opcodes]
    for opcode in ['SPLIT']:
        if opcode in builtin_codes:
            ignored_output_codes.append(builtin_codes.index(opcode))

    all_input_tensors, all_output_tensors, ignored_output_tensors = [], [], []
    for op in operators:
        all_input_tensors += op['inputs']
        all_output_tensors += op['outputs']
        if op['opcode_index'] in ignored_output_codes:
            ignored_output_tensors += op['outputs']

    input_tensors = []
    for i in all_input_tensors:
        if not i in input_tensors and not i in all_output_tensors:
            input_tensors.append(i)

    output_tensors = []
    for o in all_output_tensors:
        if not o in output_tensors and not o in all_input_tensors:
            if not o in ignored_output_tensors:
                output_tensors.append(o)

    valid_input_tensors = []
    for idx in input_tensors:
        t = tensors[idx]
        buf = buffers[t['buffer']]
        if 'data' not in buf:
            valid_input_tensors.append(idx)

    valid_output_tensors = []
    for idx in output_tensors:
        t = tensors[idx]
        buf = buffers[t['buffer']]
        if 'data' not in buf:
            valid_output_tensors.append(idx)

    return valid_input_tensors, valid_output_tensors


def create_signature_defs(graph, tensors, input_tensors, output_tensors):
    graph['signature_defs'] = [{'inputs': None, 'outputs': None, 'signature_key':'serving_default', 'subgraph_index':0}]
    graph['signature_defs'][0]['inputs'] = [{'name': tensors[idx]['name'], 'tensor_index': idx} for idx in input_tensors]
    graph['signature_defs'][0]['outputs'] = [{'name': tensors[idx]['name'], 'tensor_index': idx} for idx in output_tensors]
    return graph


def create_graph(operators, tensors, buffers, opcodes, force_shape=None):
    operators = [_.copy() for _ in operators]
    tensors = [_.copy() for _ in tensors]
    # buffers = [_.copy() for _ in buffers]
    buffers = buffers.copy()

    operators, tensors, buffers = clean_operators(operators, tensors, buffers, force_shape=force_shape)
    input_tensors, output_tensors = get_io_tensors(operators, tensors, buffers, opcodes)

    g = {'description': 'VectorBlox transform', 'version': 3, 'metadata':[], 'operator_codes': opcodes, 'subgraphs': [{}]}
    subg = g['subgraphs'][0]
    subg['inputs'], subg['outputs'] = input_tensors, output_tensors
    g['buffers'], subg['tensors'], subg['operators'] = buffers, tensors, operators

    g = create_signature_defs(g, tensors, input_tensors, output_tensors)
    if g['signature_defs'][0]['inputs'] == []:
        print('WARNING: NO INPUTS')

    return g


def verify_graph(graph, core, accel, debug, optimized=False):
    subgraph = graph['subgraphs'][0]
    operators, tensors = subgraph['operators'], subgraph['tensors']
    buffers, opcodes = graph['buffers'], graph['operator_codes']
    builtin_codes = [_['builtin_code'] for _ in opcodes]

    if not optimized:
        id_count = 0
        pattern_count = 0
        i = 0
        while i < len(operators): 
            op = operators[i]
            opts = None
            if 'builtin_options' in op:
                opts = op['builtin_options']
            opcode = builtin_codes[op['opcode_index']]

            if opcode in ['CONV_2D']:
                if 'FLOAT32' in [tensors[_]['type'] for _ in op['inputs']]:
                    print('FLOAT32 inputs not supported (for operator {} {})'.format(i, opcode))
                    return False

            if opcode in ['CONV_2D','SUM']:
                if 'FLOAT32' in [tensors[_]['type'] for _ in op['outputs']]:
                    print('FLOAT32 outputs not supported (operator {} {})'.format(i, opcode))
                    return False

            if opcode in ['DIV']:
                if is_multi_input(op, tensors, buffers):
                    if 'FLOAT32' in [tensors[_]['type'] for _ in op['outputs']]:
                        print('FLOAT32 DIV w/ multiple inputs is not supported (operator {} {})'.format(i, opcode))
                        return False

            i += 1
    else:
        errors = []
        
        codes = [_['builtin_code'] for _ in graph['operator_codes']]
        valid_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'supported_ops.json') 
        with open(valid_path) as f:
            valid = json.load(f)

        i = 0
        while i < len(operators): 
            op = operators[i]
            if 'builtin_options' in op:
                opts = op['builtin_options']
            opcode = codes[op['opcode_index']]

            valid_op, error_param, param_value, input_param = check_valid(valid, opcode, op, tensors)
            if not valid_op:
                errors.append([i,op,opcode, error_param, param_value, input_param])

            i += 1

        if len(errors) >0:
            with exception_catcher(errors):
                assert(0)
    return True


def transform_graph(graph, passes, debug):
    subgraph = graph['subgraphs'][0]
    operators, tensors = subgraph['operators'], subgraph['tensors']
    buffers, opcodes = graph['buffers'], graph['operator_codes']

    builtin_codes = [_['builtin_code'] for _ in opcodes]

    # add opcodes
    for transform in passes:
        if transform == 'LUT':
            if not 'CAST' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 53, 'version': 1, 'builtin_code': 'CAST'})
                builtin_codes = [_['builtin_code'] for _ in opcodes]
            if not 'GATHER' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 36, 'version': 2, 'builtin_code': 'GATHER'})
            if not 'QUANTIZE' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 114, 'version': 1, 'builtin_code': 'QUANTIZE'})
            if not 'SPLIT' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 49, 'version': 2, 'builtin_code': 'SPLIT'})
            if not 'CONCATENATION' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 2, 'version': 1, 'builtin_code': 'CONCATENATION'})
        elif transform == 'EXPLICIT_PAD':
            if not 'PAD' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 34, 'version': 1, 'builtin_code': 'PAD'})
        elif transform == 'GROUP_CONV':
            if not 'SPLIT' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 49, 'version': 2, 'builtin_code': 'SPLIT'})
            if not 'CONCATENATION' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 2, 'version': 1, 'builtin_code': 'CONCATENATION'})
        elif transform == 'STRIDED_DEPTHWISE': 
            if not 'STRIDED_SLICE' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 45, 'version': 1, 'builtin_code': 'STRIDED_SLICE'})
        elif transform == 'TRANSPOSE_CONV': 
            if not 'CONV_2D' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 3, 'version': 1, 'builtin_code': 'CONV_2D'})
            if not 'DILATE' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 127, 'version': 1, 'builtin_code': 'DILATE'})
            if not 'PAD' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 34, 'version': 1, 'builtin_code': 'PAD'})
        elif transform == 'YOLO_ARG_MAX': 
            if not 'CAST' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 53, 'version': 1, 'builtin_code': 'CAST'})
            if not 'ARG_MAX' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 56, 'version': 2, 'builtin_code': 'ARG_MAX'})
            if not 'RESHAPE' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 22, 'version': 1, 'builtin_code': 'RESHAPE'})
            if not 'PAD' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 34, 'version': 1, 'builtin_code': 'PAD'})
        elif transform == 'IDENTITY_INJECTION': 
            if not 'SPLIT' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 49, 'version': 2, 'builtin_code': 'SPLIT'})

        builtin_codes = [_['builtin_code'] for _ in opcodes]

    for transform in passes:
        operators, tensors, buffers = apply_transformation(transform, operators, tensors, buffers, opcodes, builtin_codes, debug)

    return create_graph(operators, tensors, buffers, opcodes)


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

def postprocess_inject(tensor_idx, operators, tensors, buffers, opcodes, dataset, opacity, input_height, input_width, height, width):
  
    tx = tensors[tensor_idx].copy()
    current_type = tx['type']
    current_shape = tx['shape'].copy()
    current_output = tensor_idx
    current_quant = tx['quantization']
    double = 0
    i = 0
    
    #inject ARG_MAX
    if current_shape[-1] > 1 and len(current_shape) > 3: #perform ARG_MAX injection on non-HxW dimension.

        while (current_shape[-3] < height/4 and current_shape[-2] < width/4):
            # current_shape = [current_shape[-4], current_shape[-3]*2, current_shape[-2]*2, current_shape[-1]]
            while (current_shape[-3] < height/4 and current_shape[-2] < width/4):
                current_shape = [current_shape[-4], current_shape[-3]*2, current_shape[-2]*2, current_shape[-1]]

            resize = 'RESIZE_BILINEAR'
            inject_op, tensors, buffers = op_resize(tensors, buffers, opcodes.index(resize), current_output, False, current_shape)
            operators = operators + [inject_op]
            current_output = len(tensors)-1
            double += 1

        current_shape = current_shape[:-1]
        current_type = 'INT32'
        inject_op, tensors, buffers = op_argmax(tensors, buffers, opcodes.index('ARG_MAX'), current_output, False, current_shape, current_type)
        
        operators = operators + [inject_op]
        current_output = len(tensors)-1

        current_type = 'UINT8'
        inject_op, tensors, buffers = op_cast(tensors, buffers, opcodes.index('CAST'), current_output, False, current_type)
        operators = operators + [inject_op]
        current_output = len(tensors)-1

    elif current_type in ['INT8']: #inject QUANTIZE
        current_type = 'UINT8'
        current_quant['zero_point'][0] += 128

        inject_op, tensor, buffers = op_quantize(tensors, buffers, opcodes.index('QUANTIZE'),
                                                 current_output, False, 'UINT8', current_quant['scale'], current_quant['zero_point'])
        operators = operators + [inject_op]
        current_output = len(tensors)-1

    #inject RESHAPE, adding back channels TODO remove if alread NHWC
    if len(current_shape) == 3:
        current_shape = current_shape + [1]

        inject_op, tensors, buffers = op_reshape(tensors, buffers, opcodes.index('RESHAPE'), current_output, False, current_shape)

        operators = operators + [inject_op]
        current_output = len(tensors)-1

    # inject RESIZE 
    inject_dequantize = False
    
    resize = 'RESIZE_NEAREST_NEIGHBOR'
    if dataset in ['DEPTH']:
        resize = 'RESIZE_BILINEAR'

    if current_shape[-3] != height or current_shape[-2] != width:
        heights, widths = get_optimal_resize_shape(current_shape[-3], current_shape[-2], height, width)
        if current_type in ['UINT8'] and not inject_dequantize and resize in ['RESIZE_BILINEAR']: #inject QUANTIZE
            
            inject_op, tensors, buffers = op_quantize(tensors, buffers, opcodes.index('QUANTIZE'),
                                                 current_output, False, 'INT8', current_quant['scale'], [-128])
            operators = operators + [inject_op]
            current_type = 'INT8'
            inject_dequantize = True
            current_output = len(tensors)-1

        for(_, (h, w)) in enumerate(zip(heights, widths)):    
            current_shape = [current_shape[-4], h, w, current_shape[-1]]

            inject_op, tensors, buffers = op_resize(tensors, buffers, opcodes.index(resize), current_output, False, current_shape)

            operators = operators + [inject_op]
            current_output = len(tensors)-1
            double += 1

        current_shape = [current_shape[-4], height, width, current_shape[-1]]

        resize = 'RESIZE_NEAREST_NEIGHBOR'
        inject_op, tensors, buffers = op_resize(tensors, buffers, opcodes.index(resize), current_output, False, current_shape)

        operators = operators + [inject_op]
        current_output = len(tensors)-1

        if inject_dequantize == True and current_type in ['INT8']: #inject DEQUANTIZE
            
            inject_op, tensors, buffers = op_quantize(tensors, buffers, opcodes.index('QUANTIZE'),
                                                 current_output, False, 'UINT8', current_quant['scale'], [0])
            operators = operators + [inject_op]
            current_type = 'UINT8'
            inject_dequantize = False
            current_output = len(tensors)-1


    #inject CAST if
    if current_type != 'INT32':
        current_type = 'INT32'
        inject_op, tensors, buffers = op_cast(tensors, buffers, opcodes.index('CAST'), current_output, False, current_type)
        operators = operators + [inject_op]
        current_output = len(tensors)-1

    colors = []
    if dataset == "VOC":
        colors = [[0,0,0]] + rgb_color.voc_colors
    elif dataset == "COCO":
        colors = [[0,0,0]] + rgb_color.coco_colors
    elif dataset == "CITYSCAPES":
        rgb2bgr = lambda x: (x[2],x[1],x[0])
        colors = np.asarray([rgb2bgr(_["color"]) for _ in rgb_color.city_groups], dtype="uint8")   
    elif dataset == "DEPTH":
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
    if dataset in ["VOC", "COCO", "CITYSCAPES"]:
        colors[0] -= alpha
    
    inject_op, tensors, buffers = op_gather(tensors, buffers, opcodes.index('GATHER'), current_output, False, colors, 'INT32')
    operators = operators + [inject_op]
    current_output = len(tensors)-1

    return operators, tensors, buffers


def postprocess_graph(graph, datatset, opacity, height, width):

    subgraph = graph['subgraphs'][0]
    operators, tensors = subgraph['operators'], subgraph['tensors']
    buffers, opcodes = graph['buffers'], graph['operator_codes']

    if not 'QUANTIZE' in opcodes:
        opcodes.append({'deprecated_builtin_code': 114, 'version': 1, 'builtin_code': 'QUANTIZE'})
    if not 'GATHER' in opcodes:
        opcodes.append({'deprecated_builtin_code': 36, 'version': 2, 'builtin_code': 'GATHER'})
    if not 'CAST' in opcodes:
        opcodes.append({'deprecated_builtin_code': 53, 'version': 1, 'builtin_code': 'CAST'})
    if not 'RESHAPE' in opcodes:
        opcodes.append({'deprecated_builtin_code': 22, 'version': 1, 'builtin_code': 'RESHAPE'})
    if not 'RESIZE_BILINEAR' in opcodes:
        opcodes.append({'deprecated_builtin_code': 23, 'version': 2, 'builtin_code': 'RESIZE_BILINEAR'})
    if not 'RESIZE_NEAREST_NEIGHBOR' in opcodes:
        opcodes.append({'deprecated_builtin_code': 97, 'version': 2, 'builtin_code': 'RESIZE_NEAREST_NEIGHBOR'})
    if not 'ARG_MAX' in opcodes:
        opcodes.append({'deprecated_builtin_code': 56, 'version': 2, 'builtin_code': 'ARG_MAX'})


    # grab input shape
    input_shape = subgraph['tensors'][graph['signature_defs'][0]['inputs'][0]['tensor_index']]['shape']
    input_height, input_width = input_shape[-3], input_shape[-2]

    tensor_idx = graph['signature_defs'][0]['outputs'][0]['tensor_index']

    builtin_codes = [_['builtin_code'] for _ in opcodes]

    operators, tensors, buffers = postprocess_inject(tensor_idx, operators, tensors, buffers, builtin_codes, datatset, opacity, input_height, input_width, height, width)

    return create_graph(operators, tensors, buffers, opcodes)


def get_splits(graph, cuts, include_outputs=True):
    subgraph = graph['subgraphs'][0]
    outputs = subgraph['outputs']
    operators = subgraph['operators']
    if include_outputs:
        for i, op in enumerate(operators):
            for o in op['outputs']:
                if o in outputs:
                    cuts.append(i)
    splits = []
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


def select_graph(graph, indices):
    subgraph = graph['subgraphs'][0]
    operators, tensors = subgraph['operators'], subgraph['tensors']
    buffers, opcodes = graph['buffers'], graph['operator_codes']

    select_operators = []

    for i in indices:
        select_operators.append(operators[i])

    return create_graph(select_operators, tensors, buffers, opcodes)


def preprocess_graph(graph, scale=1.0, mean=0):

    subgraph = graph['subgraphs'][0]
    operators, tensors = subgraph['operators'], subgraph['tensors']
    buffers, opcodes = graph['buffers'], graph['operator_codes']

    if not 'QUANTIZE' in opcodes:
        opcodes.append({'deprecated_builtin_code': 114, 'version': 1, 'builtin_code': 'QUANTIZE'})
    if not 'DEQUANTIZE' in opcodes:
        opcodes.append({'deprecated_builtin_code': 6, 'version': 1, 'builtin_code': 'DEQUANTIZE'})
    if not 'MUL' in opcodes:
        opcodes.append({'deprecated_builtin_code': 18, 'version': 2, 'builtin_code': 'MUL'})
    if not 'ADD' in opcodes:
        opcodes.append({'deprecated_builtin_code': 0, 'version': 2, 'builtin_code': 'ADD'})

    builtin_codes = [_['builtin_code'] for _ in opcodes]
    tensor_idx = graph['signature_defs'][0]['inputs'][0]['tensor_index'] 

    operators, tensors, buffers = preprocess_inject(tensor_idx, operators, tensors, buffers, builtin_codes, scale, mean)

    return create_graph(operators, tensors, buffers, opcodes)


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
    

def preprocess_inject(tensor_idx, operators, tensors, buffers, opcodes, scale, mean):
    i = 0
    op0 = None
    for o,op in enumerate(operators):
        if tensor_idx in op['inputs']:
            op0 = op
            i = o
            break
    t0 = tensors[tensor_idx]
    o0 = tensor_idx
    dtype = t0['type']

    # add QUANTIZE only for
    ops_num = i
    if dtype.upper() != "UINT8":
        inject_op, tensor, buffers = op_quantize(tensors, buffers, opcodes.index('QUANTIZE'), o0, True, 'UINT8', [1.0], [0])
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

        inject_op, tensors, buffers = op_mul(tensors, buffers, opcodes.index('MUL'), o0, True,
               [1.0], [mul_zp],
               scale_q_value, [1,1,1,channel], dtype.upper(), [scale], [scale_zero_point])

        if dtype.upper() != "UINT8":
            operators[i]['outputs'] = [len(tensors)-1]
        else:
            pass
        operators = operators[:ops_num] + [inject_op] + operators[ops_num:]
        ops_num = ops_num+mul_offset

    if do_add:
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
        
        inject_op, tensors, buffers = op_add(tensors, buffers, opcodes.index('ADD'), o0, True,
               [output_scale], [output_zeropoint],
               mean_q_value, [channel], dtype.upper(), [scale], [mean_zero_point])

        operators = operators[:ops_num] + [inject_op] + operators[ops_num:]
        operators[i+mul_offset]['outputs'] = [len(tensors)-1]

    return operators, tensors, buffers 


def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite", type=existing_file)
    parser.add_argument("-s", "--scale", type=float, nargs='+', default=1.0)
    parser.add_argument("-m", "--mean", type=float, nargs='+', default=0.)
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    tmp_tflite = os.path.join(tmp_dir, os.path.basename(args.tflite))
    shutil.copyfile(args.tflite, tmp_tflite)

    graph, dir_obj = load_graph(args.tflite)

    graph_ = preprocess_graph(graph, args.scale, args.mean)

    save_graph(args.tflite.replace('.tflite', '.pre.tflite'), dir_obj, graph_, copy=True)

    if not dir_obj is None:
        dir_obj.cleanup()


def postprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite", type=existing_file)
    parser.add_argument("-p", "--post-process-layer", choices=['YOLO_ARG_MAX', 'PIXEL_VOC', 'PIXEL_COCO', 'PIXEL_CITYSCAPES', 'PIXEL_DEPTH'], default='PIXEL_DEPTH')
    parser.add_argument("-o", "--opacity", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("-v", "--verbose", action='store_true')
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
    parser.add_argument("-c", "--core", choices=['MXP', 'RISCV'], default='MXP')
    parser.add_argument("-a", "--accel", choices=['FIA', 'NEU'], default='FIA')
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("-p", "--passes", nargs='+', choices=['ALL', 'DEFAULT']+all_passes, default=['DEFAULT'])
    args = parser.parse_args()

    graph, dir_obj = load_graph(args.tflite)

    passes = args.passes
    if 'ALL' in args.passes:
        if args.core == 'MXP' and args.accel == 'FIA':
            passes = all_passes
    elif 'DEFAULT' in args.passes:
        if args.core == 'MXP' and args.accel == 'FIA':
            passes = default_passes

    if verify_graph(graph, args.core, args.accel, args.verbose):
        graph_ = transform_graph(graph, passes, args.verbose)
        save_graph(args.tflite.replace('.tflite', '.tr.tflite'), dir_obj, graph_, copy=True)

    if not dir_obj is None:
        dir_obj.cleanup()


if __name__ == "__main__":
    main()
