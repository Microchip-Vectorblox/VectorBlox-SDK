import argparse
from .utils import json_load, json_dump, generate_inputs_outputs
import copy
import subprocess
import shlex
import os.path
import shutil, glob, tempfile
from tqdm import tqdm 
import tensorflow as tf
import numpy as np
import cv2
import vbx.postprocess.dataset as rgb_color
from math import floor, ceil, log2, frexp, copysign, exp, tanh, pow, log, sqrt


MAX_LUTS = 4

passes = [
        'REMOVE_FP32_IO',
        'CLEAN_LOGISTIC',
        'LUT',
        'REMOVE_CONSTANTS',
        'IMPLICIT_PAD',
        'PADV2',
        'SHARED_PAD',
        'EXPLICIT_PAD',
        'GROUP_DEPTH',
        'GROUP_DEPTH5x2',
        'GROUP_CONV',
        'TRANSPOSE_CONV',
        'FC_CONV_2D',
        'STRIDED_DEPTHWISE',
        'YOLO_ARG_MAX',
        'AVERAGE_POOL_2D',
        'FULL_DEPTH',
        'STRIDED_CONV',
        'NCHW_RESHAPE',
        'REDUCE_MAX_POOL',
        'REDUCE_RESIZE',
        'SUB_AS_ADD',
        'DIV_AS_MUL',
        'MUL_AS_CONV',
        'CHANNEL_SPLIT',
        'FUSE_CONV',
        'REWRITE_NORM',
        'REWRITE_ATTN',
        'REMOVE_NOPS',
        ]


def create_tensor_data(dtype, shape, min_value=-100, max_value=100, int8_range=False):
  """Build tensor data spreading the range [min_value, max_value)."""

  # print(dtype.numpy())
  # if dtype in MAP_TF_TO_NUMPY_TYPE:
  #   dtype = MAP_TF_TO_NUMPY_TYPE[dtype]
  # print(dtype)

  if dtype in (tf.float32, tf.float16, tf.float64):
    value = (max_value - min_value) * np.random.random_sample(shape) + min_value
  elif dtype in (tf.complex64, tf.complex128):
    real = (max_value - min_value) * np.random.random_sample(shape) + min_value
    imag = (max_value - min_value) * np.random.random_sample(shape) + min_value
    value = real + imag * 1j
  elif dtype in (tf.uint32, tf.int32, tf.uint8, tf.int8, tf.int64, tf.uint16,tf.int16):
    value = np.random.randint(min_value, max_value + 1, shape)
    if int8_range: #Generate consecutive values uint8
        arr = np.zeros(np.prod(shape))
        channels = shape[-1]
        map_size = np.prod(shape) // channels
        arr_inc = [arr[0]+ (i%256) for i in range(map_size) for c in range(channels)]
        value = np.array(arr_inc).reshape(shape)
  elif dtype == tf.bool:
    value = np.random.choice([True, False], size=shape)
  elif dtype == np.string_:
    # Not the best strings, but they will do for some basic testing.
    letters = list(string.ascii_uppercase)
    return np.random.choice(letters, size=shape).astype(dtype)
  return np.dtype(dtype).type(value) if np.isscalar(value) else value.astype(dtype)


def get_tensor_details(tflite_model):
    if isinstance(tflite_model, bytes):
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
    elif isinstance(tflite_model, str):
        interpreter = tf.lite.Interpreter(model_path=tflite_model, experimental_preserve_all_tensors=True)
    else:
        print('WARNING: invalid tflite model', type(tflite_model))
        return None

    interpreter.allocate_tensors()
    return interpreter.get_tensor_details()


def update_shapes(tflite_path, graph):

    subgraph_tensors = graph['subgraphs'][0]['tensors']
    old_shapes = {_['name']: {'shape': tuple(_['shape']), 'index': i} for i,_ in enumerate(subgraph_tensors) if ('shape' in _ and _['shape'])}

    need_updated_shapes = False
    details = get_tensor_details(tflite_path)
    new_shapes = {_['name']: tuple(_['shape'].tolist()) for _ in details if 'shape' in _}
    for key in new_shapes:
        if key in old_shapes:
            if old_shapes[key]['shape'] != new_shapes[key]:
                subgraph_tensors[old_shapes[key]['index']]['shape'] = new_shapes[key]
                need_updated_shapes = True

    if need_updated_shapes:
        tmp_json = tflite_path.replace('.tflite', '.json')
        json_dump(graph, tmp_json)
        json2tflite(tmp_json)


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
    input_buffers = [buffers[tensors[_]['buffer']] for _ in op['inputs'] if _ != -1]
    return len(input_buffers) > 1 and not any(['data' in _ for _ in input_buffers]) 


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


def num_lut_tensor(filter, tensors, buffers, max_luts=MAX_LUTS):
    weight_tensor = tensors[filter]
    data = get_numpy_data(weight_tensor, buffers)
    single_value = np.all(data.flatten() == data.flatten()[0])
    if 'shape' not in weight_tensor or np.prod(weight_tensor['shape']) <= max_luts or single_value:
        if single_value:
            return 1
        elif 'shape' in weight_tensor and len(weight_tensor['shape']) > 0 :
            if weight_tensor['shape'][-1] > max_luts:
                return -1
            return weight_tensor['shape'][-1]
        else:
            return 1
    return -1


def nop_pattern(operators, codes, tensors, buffers, idx):
    patterns = []
    max_idx = len(operators)
    start_idx = idx
    ishape = tensors[operators[idx]['inputs'][0]]['shape']

    prev_op = None
    prev_inputs = None
    prev_outputs = None
    
    is_pattern = False

    while True:
        if not idx < max_idx:
            break
        op = operators[idx]
        opcode = codes[op['opcode_index']]
        if not opcode in ['TRANSPOSE', 'RESHAPE']:
            break
        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]
        connected, forked = True, False
        if prev_op != None:
            connected = any([_ in prev_outputs for _ in op_inputs])
            forked, _ = is_forked(prev_op, operators, tensors)

        if not connected:
            break

        if forked:
            break

        if opcode in 'RESHAPE':
            filters = [_ for _ in op_inputs if 'data' in buffers[tensors[_]['buffer']]]
            if not len(filters) == 1:
                break
            filter_data = get_numpy_data(tensors[filters[0]], buffers).copy()
            filter_data = [_ for _ in filter_data if _ not in [-1,1]]
            if len(filter_data) != 1:
                break

        oshape = tensors[operators[idx]['outputs'][0]]['shape']

        if oshape == ishape:
            is_pattern = True
            break

        idx += 1
        prev_op = op
        prev_outputs = op_outputs


    if is_pattern:
        for i in range(start_idx, idx+1):
            patterns.append(i)
    return patterns


def lut_pattern(operators, codes, tensors, buffers, idx):
    patterns = []
    lut_count = 1

    prev_op = None
    prev_inputs = None
    prev_outputs = None

    weighted_opcodes = ["MUL", "DIV", "ADD", "SUB", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM", "PRELU"]
    activation_opcodes = ["HARD_SWISH", "LOGISTIC", "TANH", 'LEAKY_RELU', 'RELU', 'RELU6', 'RELU_N1_TO_1', 'RELU_0_TO_1']
    activation_opcodes += ["RSQRT", "EXP", "LOG", "ELU", "GELU", "POW", "COS", "SIN", "NEG"]

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

        elif is_pattern(operators[idx:], codes, tensors, buffers, valid_lut_patterns):
            lp = is_pattern(operators[idx:], codes, tensors, buffers, valid_lut_patterns)
            if len(lp) > 3:
                required = len(lp) - 4 
                if required > MAX_LUTS:
                    break
                elif required > lut_count:
                    lut_count = required

            for i in range(len(lp)):
                patterns.append(idx+i)
            idx += len(lp) - 1

        elif opcode == "DEQUANTIZE" and (next_opcode in activation_opcodes or (next_opcode in weighted_opcodes and len(next_filters) == 1)) and (next_next_opcode in activation_opcodes or (next_next_opcode in weighted_opcodes and len(next_next_filters) == 1)) and next_next_next_opcode == "QUANTIZE": 
            if next_opcode in weighted_opcodes:
                required = num_lut_tensor(next_filters[0], tensors, buffers)
                if required < 0:
                    break
                if required > lut_count:
                    lut_count = required
            if next_next_opcode in weighted_opcodes:
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
        
        elif opcode == "DEQUANTIZE" and (next_opcode in activation_opcodes or (next_opcode in weighted_opcodes and len(next_filters) == 1)) and next_next_opcode == "QUANTIZE": 
            if next_opcode in weighted_opcodes:
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

        elif opcode in activation_opcodes:
            patterns.append(idx)

        elif opcode in weighted_opcodes and len(filters) == 1 and tensors[op_outputs[0]]['type'] in ['INT8', 'UINT8']:
            required = num_lut_tensor(filters[0], tensors, buffers)
            if required < 0:
                break
            if required > lut_count:
                lut_count = required
            patterns.append(idx)
        else:
            break

        prev_op = operators[idx]
        idx += 1

        # prev_op = op
        prev_inputs = [_ for _ in prev_op['inputs'] if _ != -1]
        prev_outputs = [_ for _ in prev_op['outputs'] if _ != -1]
        
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

def op_div(tensors, buffers, opcode_idx, tensor_idx, inject_before, swap_inputs, quantization, data, data_shape, data_dtype, data_quantization):
    t = tensors[tensor_idx]

    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': data_shape,
        'type': data_dtype.upper(),
        'buffer': len(buffers)-1,
        'name': 'div_data:{}'.format(tensor_idx),
        'quantization': data_quantization,
        'is_variable': False,
        'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
        'type': t['type'],
        'buffer': len(buffers)-1,
        'name': 'div_{}'.format(tensor_idx),
        'quantization': quantization,
        'is_variable': False,
        'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    if swap_inputs:
        inject_op = {'opcode_index': opcode_idx,
                    'builtin_options_type': 'DivOptions',
                    'builtin_options': {'fused_activation_function': 'NONE'},
                    'custom_options_format': 'FLEXBUFFERS',
                    'large_custom_options_offset': 0,
                    'large_custom_options_size': 0,
                    'inputs': [len(tensors)-2, input_tensor],
                    'outputs': [output_tensor]}
    else:
        inject_op = {'opcode_index': opcode_idx,
                    'builtin_options_type': 'DivOptions',
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


def op_sub(tensors, buffers, opcode_idx, tensor_idx, inject_before, swap_inputs, scale, zero_point, data, data_shape, data_dtype, data_scale, data_zero_point):
    t = tensors[tensor_idx]

    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': data_shape,
        'type': data_dtype.upper(),
        'buffer': len(buffers)-1,
        'name': 'sub_data:{}'.format(tensor_idx),
        'quantization': {'scale': data_scale, 'zero_point': data_zero_point, 'details_type': 'NONE', 'quantized_dimension': 0},
        'is_variable': False,
        'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
        'type': t['type'],
        'buffer': len(buffers)-1,
        'name': 'sub_{}'.format(tensor_idx),
        'quantization': {'scale': scale, 'zero_point': zero_point, 'details_type': 'NONE', 'quantized_dimension': 0},
        'is_variable': False,
        'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    if swap_inputs:
        inject_op = {'opcode_index': opcode_idx,
                    'builtin_options_type': 'SubOptions',
                    'builtin_options': {'fused_activation_function': 'NONE'},
                    'custom_options_format': 'FLEXBUFFERS',
                    'large_custom_options_offset': 0,
                    'large_custom_options_size': 0,
                    'inputs': [len(tensors)-2, input_tensor],
                    'outputs': [output_tensor]}
    else:
        inject_op = {'opcode_index': opcode_idx,
                    'builtin_options_type': 'SubOptions',
                    'builtin_options': {'fused_activation_function': 'NONE'},
                    'custom_options_format': 'FLEXBUFFERS',
                    'large_custom_options_offset': 0,
                    'large_custom_options_size': 0,
                    'inputs': [input_tensor, len(tensors)-2],
                    'outputs': [output_tensor]}


    return inject_op, tensors, buffers


def op_elemwise(tensors, buffers, opcode_idx, tensor_idx, tensor2_idx, shape, dtype, quant):

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': shape,
        'type': dtype,
        'buffer': len(buffers)-1,
        'name': 'elem_{}'.format(tensor_idx),
        'quantization': quant,
        'is_variable': False,
        'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1

    inject_op = {'opcode_index': opcode_idx,
                'inputs': [tensor_idx, tensor2_idx],
                'outputs': [len(tensors)-1]}


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


def op_mean(tensors, buffers, opcode_idx, tensor_idx, inject_before, axis):
    assert(not inject_before)
    t = tensors[tensor_idx]

    data = np.frombuffer(np.asarray(axis).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [len(axis)],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'mean_axis_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    oshape = t['shape'].copy()
    for x in axis:
        oshape[x] = 1

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': oshape,
            'type': t['type'],
            'buffer': len(buffers)-1,
            'name': 'mean_{}'.format(tensor_idx),
            'quantization': t['quantization'].copy(),
            'is_variable': False,
            'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    inject_op = {'opcode_index': opcode_idx,
                'builtin_options_type': "ReducerOptions",
                'builtin_options': {'keep_dims': True},
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
            'quantization': {'scale': [1.0], 'zero_point': [0], 'details_type': 'NONE', 'quantized_dimension': 0},
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


def op_resize(tensors, buffers, opcode_idx, tensor_idx, inject_before, shape, isPostprocessing=False):
    t = tensors[tensor_idx]
    tensor_name = 'resize_shape_{}'.format(tensor_idx)
    if isPostprocessing:
        tensor_name = 'resize_shape_postprocessing_{}'.format(tensor_idx)
    data = np.frombuffer(np.asarray([shape[-3], shape[-2]]).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape':[2],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': tensor_name,
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


def op_max_pool(tensors, buffers, opcode_idx, tensor_idx, inject_before, shape, opts):
    t = tensors[tensor_idx]

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': shape,
            'type': t['type'],
            'buffer': len(buffers)-1,
            'name': 'max_pool_{}'.format(tensor_idx),
            'quantization': t['quantization'],
            'is_variable': False,
            'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    inject_op = {'opcode_index': opcode_idx,
                'builtin_options_type': 'Pool2DOptions',
                'builtin_options':opts,
                'inputs': [input_tensor],
                'outputs': [output_tensor]}
    return inject_op, tensors, buffers


def op_transpose(tensors, buffers, opcode_idx, tensor_idx, inject_before, shape):
    t = tensors[tensor_idx]

    data = np.frombuffer(np.asarray(shape).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [len(shape)],
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


    inject_op = {'opcode_index': opcode_idx,
                'inputs': [input_tensor, len(tensors)-2], 
                'outputs': [output_tensor]}

    return inject_op, tensors, buffers


def op_reshape(tensors, buffers, opcode_idx, tensor_idx, inject_before, shape):
    t = tensors[tensor_idx]

    data = np.frombuffer(np.asarray(shape).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [len(shape)],
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


def op_pad(tensors, buffers, opcode_idx, tensor_idx, inject_before, pad=None, constant_value=None):
    t = tensors[tensor_idx]

    if pad is None:
        pad = [0 for _ in range(8)]
    data = np.frombuffer(np.asarray(pad).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [4,2],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'pad_shape_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    if not constant_value is None:
        constant_data = np.frombuffer(np.asarray(constant_value).astype(t['type'].lower()).tobytes(), dtype=np.uint8).tolist()
        buffers.append({'data': constant_data, 'offset': 0, 'size': 0})
        tensors.append({'shape': [],
                'type': t['type'],
                'buffer': len(buffers)-1,
                'name': 'pad_consant_{}'.format(tensor_idx),
                'quantization': t['quantization'],
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

    if constant_value is None:
        inject_op = {'opcode_index': opcode_idx,
                    'inputs': [input_tensor, len(tensors)-2], 
                    'outputs': [output_tensor]}
    else:
        inject_op = {'opcode_index': opcode_idx,
                    'inputs': [input_tensor, len(tensors)-3, len(tensors)-2], 
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


def op_type(tensors, buffers, opcode_idx, tensor_idx, inject_before, dtype, quant):
    t = tensors[tensor_idx]

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
                    'type': dtype,
                    'buffer': len(buffers)-1,
                    'name': 'quantize:{}'.format(tensor_idx),
                    'quantization': quant,
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
    shape = t['shape'].copy()
    SplitOptions = {'num_splits': num_splits}
    assert(shape[axis] % num_splits == 0)
    shape[axis] = shape[axis] // num_splits

    data = np.frombuffer(np.asarray([axis]).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'split_shape_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

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


def op_split_v(tensors, buffers, opcode_idx, tensor_idx, inject_before, axis, splits):
    t = tensors[tensor_idx]

    num_splits = len(splits)
    SplitOptions = {'num_splits': num_splits}

    data = np.frombuffer(np.asarray(splits).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [len(splits)],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'split_shape_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    data = np.frombuffer(np.asarray([axis]).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'split_axis{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    for n,split in enumerate(splits):
        buffers.append({'offset': 0, 'size': 0})
        shape = t['shape'].copy()
        shape[axis] = int(split)
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
                    'builtin_options_type': 'SplitVOptions',
                    'builtin_options': SplitOptions,
                    'inputs': [input_tensor, len(tensors)-3, len(tensors)-2], 
                    'outputs': [output_tensor]}
    else:
        inject_op = {'opcode_index': opcode_idx,
                    'builtin_options_type': 'SplitVOptions',
                    'builtin_options': SplitOptions,
                    'inputs': [input_tensor, len(tensors)-2-num_splits, len(tensors)-1-num_splits],
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
            'quantization': t['quantization'].copy(),
            'is_variable': False,
            'has_rank': True})

    inject_op = {'opcode_index': opcode_idx,
                'builtin_options_type': 'ConcatenationOptions',
                'builtin_options': {'axis': axis, 'fused_activation_function': 'NONE'},
                'inputs': tensor_indices,
                'outputs': [len(tensors) - 1]}

    return inject_op, tensors, buffers


def op_conv(tensors, buffers, opcode_idx, tensor_idx, inject_before, opts_name, opts, output_shape, output_quantization, weights, weight_quantization, biases, bias_quantization):
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
                    'builtin_options_type': opts_name,
                    'builtin_options': opts,
                    'inputs': [input_tensor, len(tensors)-3, len(tensors)-2], 
                    'outputs': [output_tensor]}
    else:
        inject_op = {'opcode_index': opcode_idx,
                    'builtin_options_type': opts_name,
                    'builtin_options': opts,
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


def op_group_conv(tensors, buffers, opcode_split, opcode_split_v, opcode_conv, opcode_concat, op, splits):
    opts = op['builtin_options']
    ishape = tensors[op['inputs'][0]]['shape'].copy()
    oquant = tensors[op['outputs'][0]]['quantization']

    f_tensor = tensors[op['inputs'][1]]
    b_tensor = None
    if len(op['inputs']) > 2 and op['inputs'][2] != -1:
        b_tensor = tensors[op['inputs'][2]]

    filter_data = get_numpy_data(f_tensor, buffers)
    wquant = f_tensor['quantization']
    k, h, w, c = tuple(f_tensor['shape'])
    conv_opts = opts.copy()

    if op['opcode_index'] != opcode_conv: #if depthwise, adjust k and opts
        k == c
        conv_opts.pop('depth_multiplier', None)

    bias_data = np.zeros((k,), dtype=np.int64)
    bquant = {'scale': [1. for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 3}
    # bias_data, bquant = None, None
    if not (b_tensor is None):
        bias_data = get_numpy_data(b_tensor, buffers).astype(np.int64)
        bquant = b_tensor['quantization']

    tensor_idx = op['inputs'][0]

    ops = []

    if len(splits) > 1:
        if all([_ == splits[0] for _ in splits]):
            split_op, tensors, buffers = op_split(tensors, buffers, opcode_split, tensor_idx, False, -1, len(splits))
        else:
            split_op, tensors, buffers = op_split_v(tensors, buffers, opcode_split_v, tensor_idx, False, -1, splits)
        ops.append(split_op)

    concat_inputs = []
    offset = 0
    for n,split in enumerate(splits):
        if len(splits) > 1:
            tensor_idx = split_op['outputs'][n]
        if op['opcode_index'] != opcode_conv: #if depthwise, adjust weights
            weights = np.zeros((split,h,w,split), dtype=np.int64)
            for x in range(split):
                weights[x,:,:,x % split] = filter_data[0,:,:,offset+x].copy()
        else:
            weights = filter_data[offset:offset+split].copy()
        biases = bias_data[offset:offset+split].copy()

        bias_quant = bquant.copy()
        bias_quant['zero_point'] = bias_quant['zero_point'][offset:offset+split].copy()
        bias_quant['scale'] = bias_quant['scale'][offset:offset+split].copy()

        weight_quant = wquant.copy()
        weight_quant['zero_point'] = weight_quant['zero_point'][offset:offset+split].copy()
        weight_quant['scale'] = weight_quant['scale'][offset:offset+split].copy()

        oshape = tensors[op['outputs'][0]]['shape'].copy()
        oshape[-1] = split
        conv_op, tensors, buffers = op_conv(tensors, buffers, opcode_conv, tensor_idx, False, 'Conv2DOptions', conv_opts.copy(),
                                            oshape, oquant,
                                            weights, weight_quant,
                                            biases, bias_quant)
        concat_inputs.append(conv_op['outputs'][0])
        ops.append(conv_op)
        offset += split

    if len(splits) > 1:
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
    dilation_h, dilation_w = 1,1
    if 'dilation_h_factor' in opts and 'dilation_w_factor' in opts:
        dilation_h, dilation_w = opts['dilation_h_factor'], opts['dilation_w_factor']
    if len(op['inputs']) > 1:
        f_tensor = tensors[op['inputs'][1]]
        k, h, w, c = tuple(f_tensor['shape'])
        kernel_h, kernel_w = h, w
    else:
        kernel_h, kernel_w = opts['filter_height'], opts['filter_width']
    i_h, i_w = tensors[op['inputs'][0]]['shape'][-3], tensors[op['inputs'][0]]['shape'][-2]
    o_h, o_w = tensors[op['outputs'][0]]['shape'][-3], tensors[op['outputs'][0]]['shape'][-2]

    pad_h = max(stride_h * (o_h - 1) - i_h + kernel_h + (kernel_h-1)*(dilation_h-1), 0)
    pad_w = max(stride_w * (o_w - 1) - i_w + kernel_w + (kernel_w-1)*(dilation_w-1), 0)

    return pad_h,pad_w


def channel_order(i, operators, tensors, buffers):
    return "NHWC"


def apply_transformation(transform, operators, tensors, buffers, opcodes, builtin_codes, debug):

    if transform in ['SORT_DFS', 'SORT_KHAN']:
        sort = transform.lower().split('_')[-1]

        operators, tensors, buffers = [_.copy() for _ in operators], [_.copy() for _ in tensors], buffers.copy()

        return clean_operators(operators, tensors, buffers, sort=sort)

    id_count = 0
    pattern_count = 0
    pool_count = 0
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

        elif transform == 'REMOVE_NOPS':
            pattern = nop_pattern(operators, builtin_codes, tensors, buffers, i)
            if len(pattern):
                for n,next_op in enumerate(operators):
                    pattern_input = operators[pattern[0]]['inputs'][0]
                    pattern_output = operators[pattern[-1]]['outputs'][0]
                    if pattern_output in next_op['inputs']:
                        next_op['inputs'] = [pattern_input if _ == pattern_output else _ for _ in next_op['inputs']]
                operators = operators[:i] + operators[i+len(pattern):]


        elif transform == 'CLEAN_LOGISTIC':
            if opcode in ['LOGISTIC']: #TODO and 'FLOAT32' in [tensors[_]['type'] for _ in op['inputs']]:
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

        elif transform == 'REDUCE_RESIZE':
            # if opcode in ['RESIZE_NEAREST_NEIGHBOR', 'RESIZE_BILINEAR']:
            if opcode in ['RESIZE_NEAREST_NEIGHBOR']:
                ishape = tensors[op['inputs'][0]]['shape']
                oshape = tensors[op['outputs'][0]]['shape']
                shape = get_numpy_data(tensors[op['inputs'][1]], buffers)
                ih,iw = ishape[-3], ishape[-2]
                oh,ow = oshape[-3], oshape[-2]

                if ((oh/ih), (ow/iw)) in [(4.0,4.0), (8.0,8.0)]:
                    resize = opcode
                    inject_ops = []

                    current_shape = ishape.copy()
                    tensor_idx = op['inputs'][0]
                    current_shape[-3] = current_shape[-3] * 2
                    current_shape[-2] = current_shape[-2] * 2

                    resize_op0, tensors, buffers = op_resize(tensors, buffers, builtin_codes.index(resize), tensor_idx, False, current_shape)
                    inject_ops.append(resize_op0)

                    current_shape = current_shape.copy()
                    tensor_idx = len(tensors)-1
                    current_shape[-3] = current_shape[-3] * 2
                    current_shape[-2] = current_shape[-2] * 2

                    resize_op1, tensors, buffers = op_resize(tensors, buffers, builtin_codes.index(resize), tensor_idx, False, current_shape)
                    inject_ops.append(resize_op1)

                    if ((oh/ih), (ow/iw)) in [(8.0,8.0)]:
                        current_shape = current_shape.copy()
                        tensor_idx = len(tensors)-1
                        current_shape[-3] = current_shape[-3] * 2
                        current_shape[-2] = current_shape[-2] * 2

                        resize_op2, tensors, buffers = op_resize(tensors, buffers, builtin_codes.index(resize), tensor_idx, False, current_shape)
                        inject_ops.append(resize_op2)

                    for n,next_op in enumerate(operators):
                        if op['outputs'][0] in next_op['inputs']:
                            next_op['inputs'] = [inject_ops[-1]['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]

                    operators = operators[:i] + inject_ops + operators[i+1:]

                    i += len(inject_ops)-1

        elif transform == 'REDUCE_MAX_POOL':
            if opcode in ['MAX_POOL_2D']:
                fh, fw = opts['filter_height'], opts['filter_width'] 
                if (fh,fw) in [(5,5), (7,7)]:
                    tensor_idx = op['inputs'][0]
                    oshape = tensors[op['outputs'][0]]['shape']
                    inject_ops = []
                    opts3 = opts.copy()
                    opts3['filter_height'], opts3['filter_width'] = 3,3

                    # first 3x3
                    oshape = oshape.copy()
                    if opts3['padding'] == 'VALID':
                        oshape[-3] -= 2
                        oshape[-2] -= 2

                    max_pool_op0, tensors, buffers = op_max_pool(tensors, buffers, builtin_codes.index('MAX_POOL_2D'), tensor_idx, False, oshape, opts3.copy())
                    inject_ops.append(max_pool_op0)

                    # second 3x3
                    tensor_idx = len(tensors)-1
                    oshape = oshape.copy()
                    if opts3['padding'] == 'VALID':
                        oshape[-3] -= 2
                        oshape[-2] -= 2
                    max_pool_op1, tensors, buffers = op_max_pool(tensors, buffers, builtin_codes.index('MAX_POOL_2D'), tensor_idx, False, oshape, opts3.copy())
                    inject_ops.append(max_pool_op1)

                    if (fw,fh) == (7,7): # potential third 3x3
                        tensor_idx = len(tensors)-1
                        oshape = oshape.copy()
                        if opts3['padding'] == 'VALID':
                            oshape[-3] -= 2
                            oshape[-2] -= 2
                        max_pool_op2, tensors, buffers = op_max_pool(tensors, buffers, builtin_codes.index('MAX_POOL_2D'), tensor_idx, False, oshape, opts3.copy())
                        inject_ops.append(max_pool_op2)


                    for n,next_op in enumerate(operators):
                        if op['outputs'][0] in next_op['inputs']:
                            next_op['inputs'] = [inject_ops[-1]['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]

                    operators = operators[:i] + inject_ops + operators[i+1:]

                    i += len(inject_ops)-1

        elif transform == 'NCHW_RESHAPE':
            if opcode in ['RESHAPE'] and channel_order(i, operators, tensors, buffers) == "NHWC":
                ishape = tensors[op['inputs'][0]]['shape']
                shape = get_numpy_data(tensors[op['inputs'][1]], buffers)

                iarr = np.random.randint(0, 10, ishape)
                oarr = np.reshape(iarr, shape)
                print(iarr.shape, shape, oarr.shape)

                _shape = np.random.randint(0, 10, shape).transpose([0,2,3,1]).shape

                _iarr = np.transpose(iarr, [0,2,3,1])
                _t = np.reshape(_iarr, _shape)
                _oarr = np.transpose(_t, [0,2,3,1])
                assert((_oarr == oarr).all())

                
                # c_oarr = channels_first_array(oarr)
                # cishape = c_iarr.shape
                # coshape = c_oarr.shape

                transpose_nchw_op, tensors, buffers = op_transpose(tensors, buffers, builtin_codes.index('TRANSPOSE'), tensor_idx, False, [0,3,1,2])
                inject_op.append(transpose_nchw_op)
                tensor_idx = len(tensors)-1

                reshape_nchw, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), tensor_idx, False, _shape)
                inject_op.append(reshape_nchw_op)
                tensor_idx = len(tensors)-1

                transpose_nhwc_op, tensors, buffers = op_transpose(tensors, buffers, builtin_codes.index('TRANSPOSE'), tensor_idx, False, [0,2,3,1])
                inject_op.append(transpose_nhwc_op)
                tensor_idx = len(tensors)-1

                for n,next_op in enumerate(operators):
                    if op['outputs'][0] in next_op['inputs']:
                        next_op['inputs'] = [inject_ops[-1]['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]

                # inject TRANSPOSE-RESHAPE-TRANSPOSE ops and skip ahead n
                operators = operators[:i] + inject_ops + operators[i+1:]

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

                        if pad_w > 0 or pad_h > 0:
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
                        tensor_idx = len(tensors)-1

                        conv_opts['stride_h'] = 1
                        conv_opts['stride_w'] = 1


                    if pad_h > 0 or pad_w > 0:
                        pad = [0,0,floor(pad_h/2),ceil(pad_h/2),floor(pad_w/2),ceil(pad_w/2),0,0]
                        pad_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PAD'), tensor_idx, False, pad)
                        pshape = tensors[len(tensors)-1]['shape']
                        pshape[-3] += pad_h
                        pshape[-2] += pad_w

                        ops.append(pad_op)
                        tensor_idx = len(tensors)-1

                    oquant = o_tensor['quantization']
                    oshape = tensors[tensor_idx]['shape'].copy()
                    oshape[-3] -= kernel_h-1
                    oshape[-2] -= kernel_w-1
                    conv_opts['padding'] = 'VALID'

                    conv_op, tensors, buffers = op_conv(tensors, buffers, builtin_codes.index('CONV_2D'), tensor_idx, False, 'Conv2DOptions', conv_opts,
                                                        oshape, oquant,
                                                        weights, weight_quant,
                                                        biases, bias_quant)
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

        elif transform in ['GROUP_DEPTH', 'FULL_DEPTH', 'GROUP_DEPTH5x2']:
            if opcode in ['DEPTHWISE_CONV_2D']:
                ishape = tensors[op['inputs'][0]]['shape']
                fshape = tensors[op['inputs'][1]]['shape']
                if not transform == 'GROUP_DEPTH5x2' or (fshape[-2] == 5 and opts['stride_w'] > 1):
                    group_size = ishape[-1]
                    if transform in ['GROUP_DEPTH', 'GROUP_DEPTH5x2']:
                        group_size = 8

                    num_splits = ishape[-1] // group_size
                    splits = [group_size for _ in range(num_splits)]
                    if ishape[-1] % group_size:
                        splits.append(ishape[-1] % group_size)

                    inject_ops, tensors, buffers = op_group_conv(tensors, buffers, builtin_codes.index('SPLIT'), builtin_codes.index('SPLIT_V'), builtin_codes.index('CONV_2D'), builtin_codes.index('CONCATENATION'), op, splits)
                    # adjust next op inputs, to go from CONCATENATION instead of CONV
                    for n,next_op in enumerate(operators):
                        if op['outputs'][0] in next_op['inputs']:
                            next_op['inputs'] = [inject_ops[-1]['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]

                    # inject SPLIT-CONV-CONCAT and skip ahead n
                    operators = operators[:i] + inject_ops + operators[i+1:]

        elif transform == 'GROUP_CONV': 
            if opcode in ['CONV_2D']:
                ishape = tensors[op['inputs'][0]]['shape']
                fshape = tensors[op['inputs'][1]]['shape']
                f_tensor = tensors[op['inputs'][1]]
                if fshape[-1] < ishape[-1]:

                    num_splits = ishape[-1] // f_tensor['shape'][-1]
                    group_size = ishape[-1] // num_splits
                    assert(ishape[-1] % f_tensor['shape'][-1] == 0)

                    splits = [group_size for _ in range(num_splits)]
                    inject_ops, tensors, buffers = op_group_conv(tensors, buffers, builtin_codes.index('SPLIT'), builtin_codes.index('SPLIT_V'), builtin_codes.index('CONV_2D'), builtin_codes.index('CONCATENATION'), op, splits)

                    # adjust next op inputs, to go from CONCATENATION instead of CONV
                    for n,next_op in enumerate(operators):
                        if op['outputs'][0] in next_op['inputs']:
                            next_op['inputs'] = [inject_ops[-1]['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]

                    # inject SPLIT-CONV-CONCAT and skip ahead n
                    operators = operators[:i] + inject_ops + operators[i+1:]

                    i += len(inject_ops)-1

        elif transform == 'AVERAGE_POOL_2D': 
            inject_ops = []

            # if opcode in ['MEAN']:
            #     tensor_idx = op['inputs'][0]
            #     ishape = tensors[tensor_idx]['shape']
            #     axis = get_numpy_data(tensors[op['inputs'][1]], buffers)
            #     axis = [_ - len(ishape) for _ in axis]
            #     kh,kw,k = ishape[-3], ishape[-2],ishape[-1]
            #     if tuple(axis) == (-3,-2) and (kh,kw) in [(7,7)]:
            #         w_val = 127
            #         if (kh,kw) == (7,7):
            #             w_scale = 0.0001607 #1/127/(kh*kw)

            #         dconv_opts = {'depth_multiplier': 1, 'stride_h': 1, 'stride_w': 1, 'dilation_h_factor': 1, 'dilation_w_factor': 1, 'padding': 'VALID'}
            #         weights = np.zeros((1,kh,kw,k), dtype=np.int64)
            #         weights[0,:,:,:] += w_val
            #         weight_quant = {'scale': [w_scale for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 3}

            #         bias_data = np.zeros((k,), dtype=np.int64)
            #         bquant = {'scale': [1. for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 0}

            #         oshape = tensors[op['outputs'][0]]['shape']
            #         oquant = tensors[op['outputs'][0]]['quantization'].copy()
            #         conv_op, tensors, buffers = op_conv(tensors, buffers, builtin_codes.index('DEPTHWISE_CONV_2D'), tensor_idx, False, 'DepthwiseConv2DOptions', dconv_opts,
            #                                             oshape, oquant,
            #                                             weights, weight_quant,
            #                                             bias_data, bquant)
            #         inject_ops.append(conv_op)

            if opcode in ['AVERAGE_POOL_2D']:
                if opts['filter_height'] == 2 and opts['filter_width'] == 2 and opts['stride_h'] == 1 and opts['stride_w'] == 1:
                    pad_row, pad_col = 1, 1
                    w_val = 127

                    if pool_count % 2: # swap between w_scale that rounds down vs rounds up int8 values
                        w_scale = 0.001969 #1/127/(kh*kw)
                    else:
                        w_scale = 0.001968
                    pool_count += 1

                    inject_ops = []
                    tensor_idx = op['inputs'][0]
                    # pad = [0,0,0,pad_row,0,pad_col,0,0]
                    pad = [0,0,pad_row,0,pad_col,0,0,0]
                    pad_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PAD'), op['inputs'][0], False, pad)
                    pshape = tensors[len(tensors)-1]['shape']
                    pshape[-3] += pad_row
                    pshape[-2] += pad_col
                    inject_ops.append(pad_op)
                    tensor_idx = len(tensors)-1

                    ishape = tensors[tensor_idx]['shape']
                    k = ishape[-1]
                    kh, kw = 3, 3

                    oshape = ishape.copy()
                    oshape[-3] = oshape[-3] - (kh - 1)
                    oshape[-2] = oshape[-2] - (kw - 1)
                    oquant = tensors[op['inputs'][0]]['quantization'].copy()

                    dconv_opts = {'depth_multiplier': 1, 'stride_h': 1, 'stride_w': 1, 'dilation_h_factor': 1, 'dilation_w_factor': 1, 'padding': 'VALID'}
                    weights = np.zeros((1,kh,kw,k), dtype=np.int64)
                    # weights[0,0:2,0:2,:] += w_val
                    weights[0,1:3,1:3,:] += w_val

                    weight_quant = {'scale': [w_scale for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 3}

                    bias_data = np.zeros((k,), dtype=np.int64)
                    bquant = {'scale': [1. for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 0}

                    conv_op, tensors, buffers = op_conv(tensors, buffers, builtin_codes.index('DEPTHWISE_CONV_2D'), tensor_idx, False, 'DepthwiseConv2DOptions', dconv_opts,
                                                        oshape, oquant,
                                                        weights, weight_quant,
                                                        bias_data, bquant)
                    inject_ops.append(conv_op)

            if len(inject_ops):
                for n,next_op in enumerate(operators):
                    if op['outputs'][0] in next_op['inputs']:
                        next_op['inputs'] = [inject_ops[-1]['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]

                operators = operators[:i] + inject_ops + operators[i+1:]

                i += len(inject_ops)-1

        elif transform in ['STRIDED_DEPTHWISE','STRIDED_CONV']:
            if (transform == 'STRIDED_DEPTHWISE' and opcode == 'DEPTHWISE_CONV_2D') or (transform == 'STRIDED_CONV' and opcode == 'CONV_2D'):
                if opts['stride_w'] > 1:
                    pad_h, pad_w = get_pad(op, tensors)
                    stride_w, dilation_w = opts['stride_w'], opts['dilation_w_factor']
                    filter_w, input_w = tensors[op['inputs'][1]]['shape'][-2], tensors[op['inputs'][0]]['shape'][-2]
                    adjusted_conv_width = input_w + pad_w - (((filter_w-1)*dilation_w)+1) + 1 

                    stride_h, dilation_h = opts['stride_h'], opts['dilation_h_factor']
                    filter_h, input_h = tensors[op['inputs'][1]]['shape'][-3], tensors[op['inputs'][0]]['shape'][-3]
                    adjusted_conv_height = input_h + pad_h - (((filter_h-1)*dilation_h)+1) + 1 

                    #create strided slice
                    t = tensors[op['outputs'][0]]
                    begin = [0, 0, 0, 0]
                    end = t['shape'].copy()
                    end[-2] = adjusted_conv_width
                    stride = [1, 1, opts['stride_w'], 1]

                    # end[-3] = adjusted_conv_height
                    # stride = [1, opts['stride_h'], opts['stride_w'], 1]

                    tensor_idx = op['outputs'][0]
                    inject_op, tensors, buffers = op_strided_slice(tensors, buffers, builtin_codes.index('STRIDED_SLICE'), tensor_idx, False, begin, end, stride)

                    #adjust CONV
                    t = tensors[op['outputs'][0]]
                    t['shape'][-2] = adjusted_conv_width

                    op['builtin_options'] = opts.copy()
                    op['builtin_options']['stride_w'] = 1
                    # op['builtin_options']['stride_h'] = 1

                    #adjust inputs to use STRIDED_SLICE not CONV
                    for n,next_op in enumerate(operators):
                        if op['outputs'][0] in next_op['inputs']:
                            next_op['inputs'] = [inject_op['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]

                    # inject STRIDED_SLICE and skip ahead 1
                    operators = operators[:i+1] + [inject_op] + operators[i+1:]
                    i += 1

        elif transform == 'CHANNEL_SPLIT': 
            if opcode in ['STRIDED_SLICE']:
                ishape = tensors[op['inputs'][0]]['shape']
                oshape = tensors[op['outputs'][0]]['shape']
                begin = get_numpy_data(tensors[op['inputs'][1]], buffers)
                end = get_numpy_data(tensors[op['inputs'][2]], buffers)

                axis = None 
                same_dim = [i == o for i,o in zip(tuple(ishape),tuple(oshape))]
                if len(tuple(ishape)) == len(tuple(oshape)):
                    if len(same_dim) > 1 and all(same_dim[:-1]):
                        axis = -1
                    elif len(same_dim) > 2 and all(same_dim[:-2]) and all(same_dim[-1:]):
                        axis = -2
                    elif len(same_dim) > 3 and all(same_dim[:-3]) and all(same_dim[-2:]):
                        axis = -3
                    elif len(same_dim) > 4 and all(same_dim[:-4]) and all(same_dim[-3:]):
                        axis = -4
                
                if axis in [-1]:
                    split_idx = 0
                    splits = []
                    if begin[axis] != 0:
                        splits.append(begin[axis])
                        split_idx += 1
                    splits.append(end[axis]-begin[axis])
                    if ishape[axis] > end[axis]:
                        splits.append(ishape[axis] - end[axis])

                    tensor_idx = op['inputs'][0]
                    split_op, tensors, buffers = op_split_v(tensors, buffers, builtin_codes.index('SPLIT_V'), tensor_idx, False, axis, splits)
                    operators = operators[:i] + [split_op] + operators[i+1:]

                    # adjust
                    for n,next_op in enumerate(operators):
                        if op['outputs'][0] in next_op['inputs']:
                            next_op['inputs'] = [split_op['outputs'][split_idx] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]

        elif transform == 'IMPLICIT_PAD': 
            if opcode in ['CONV_2D', 'DEPTHWISE_CONV_2D'] and opts['padding'] == 'VALID':
                tensor_idx = op['inputs'][0]
                prev_ops = [] 
                for p,prev_op in enumerate(operators):
                    if tensor_idx in prev_op['outputs']:
                        prev_ops.append(p)
                if len(prev_ops) == 1 and builtin_codes[operators[prev_ops[0]]['opcode_index']] in ['PAD', 'PADV2']:
                    pad_idx = prev_ops[0]
                    pad_op = operators[pad_idx]

                    opts['padding'] = 'SAME'
                    op['inputs'][0] = pad_op['inputs'][0]

                    operators = operators[:pad_idx] + operators[pad_idx+1:]

        elif transform == 'PADV2':
            if opcode in ['PAD']:
                tensor_idx = op['inputs'][0]
                if tensors[tensor_idx]['quantization']['zero_point']:
                    constant_value = tensors[tensor_idx]['quantization']['zero_point'][-1]
                    pad = get_numpy_data(tensors[op['inputs'][1]], buffers).flatten().tolist()
                    inject_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PADV2'), tensor_idx, False, pad, constant_value)

                    inject_op['outputs'][0] = op['outputs'][0]
                    operators = operators[:i] + [inject_op] + operators[i+1:]

        elif transform == 'SHARED_PAD':
            output_ops = []
            output_idx = []
            for n,next_op in enumerate(operators):
                if next_op['inputs'][0] == op['outputs'][0]:
                    output_ops.append(next_op)
                    output_idx.append(n)
            if len(output_ops) > 1:
                if all([builtin_codes[_['opcode_index']] in ['PADV2'] for _ in output_ops]):
                    pads = [get_numpy_data(tensors[_['inputs'][1]], buffers).flatten().tolist() for _ in output_ops]
                    vals = [get_numpy_data(tensors[_['inputs'][2]], buffers).flatten().tolist() for _ in output_ops]
                    if all([pads[0] == _ for _ in pads[1:]]) and all([vals[0] == _ for _ in vals[1:]]):
                        for idx in reversed(sorted(output_idx[1:])):
                            for n,next_op in enumerate(operators):
                                if next_op['inputs'][0] == operators[idx]['outputs'][0]:
                                    next_op['inputs'][0] = operators[output_idx[0]]['outputs'][0]
                        for idx in reversed(sorted(output_idx[1:])):
                            operators = operators[:idx] + operators[idx+1:]
                        i -= len(output_idx) - 1

        elif transform == 'EXPLICIT_PAD': 
            if opcode in ['CONV_2D', 'DEPTHWISE_CONV_2D', 'MAX_POOL_2D']:
                pad_h, pad_w = get_pad(op, tensors)
                pad = [0,0,floor(pad_h/2),ceil(pad_h/2),floor(pad_w/2),ceil(pad_w/2),0,0]
                if pad_h == 0 and pad_w == 0:
                    opts['padding'] = 'VALID'
                else:
                    tensor_idx = op['inputs'][0]
                    constant_value = 0
                    if tensors[tensor_idx]['quantization']['zero_point']:
                        constant_value = tensors[tensor_idx]['quantization']['zero_point'][-1]
                    if opcode in ['MAX_POOL_2D']:
                        constant_value = -9999
                        if tensors[tensor_idx]['type'] == 'INT8':
                            constant_value = -128
                        elif tensors[tensor_idx]['type'] == 'UINT8':
                            constant_value = 0 
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
                        # inject_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PAD'), tensor_idx, True, pad)
                        inject_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PADV2'), tensor_idx, True, pad, constant_value)

                        # adjust previous op outputs, to go to PAD instead of CONV
                        for p,prev_op in enumerate(operators):
                            if tensor_idx in prev_op['outputs']:
                                prev_op['outputs'] = [inject_op['inputs'][0] if _ == tensor_idx else _ for _ in prev_op['outputs']]

                        # set NEXT who share padding to VALID
                        for s in shared_indices:
                            operators[s]['builtin_options']['padding'] = 'VALID'
                    else:
                        # create PAD
                        inject_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PADV2'), tensor_idx, False, pad, constant_value)
                        # if opcode in ['MAX_POOL_2D']:
                        #     inject_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PADV2'), tensor_idx, False, pad, constant_value)
                        # else:
                        #     inject_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PAD'), tensor_idx, False, pad)
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
        elif transform == 'MUL_AS_CONV':
            if opcode in ['MUL']:
                t0 = tensors[op['inputs'][0]]
                t1 = tensors[op['inputs'][1]]
                if 'data' in buffers[t0['buffer']] or 'data' in buffers[t1['buffer']]:
                    if 'data' in buffers[t0['buffer']]:
                        f_tensor = t0
                        itensor = t1
                        tensor_idx = op['inputs'][1]
                    else:
                        f_tensor = t1
                        itensor = t0
                        tensor_idx = op['inputs'][0]

                    filter_data = get_numpy_data(f_tensor, buffers)
                    if all([filter_data.shape[s] == 1 for s in range(len(filter_data.shape)-1)]):
                        ishape = itensor['shape'].copy()
                        k = ishape[-1]
                        oshape = tensors[op['outputs'][0]]['shape'].copy()
                        oquant = tensors[op['outputs'][0]]['quantization']
                        f_quant = f_tensor['quantization'].copy()

                        w_scale = f_quant['scale'][0]
                        w_zero = f_quant['zero_point'][0]
                        weight_quant = {'scale': [w_scale for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 3}
                        weights = filter_data.copy()

                        dconv_opts = {'depth_multiplier': 1, 'stride_h': 1, 'stride_w': 1, 'dilation_h_factor': 1, 'dilation_w_factor': 1, 'padding': 'VALID'}

                        bias_data = np.zeros((k,), dtype=np.int64)
                        bquant = {'scale': [1. for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 0}


                        weights = np.zeros((1,1,1,k), dtype=np.int64)
                        for k_ in range(k):
                            if filter_data.shape[-1] == 1:
                                w_val = filter_data[0,0,0,0]
                            else:
                                w_val = filter_data[0,0,0,k_]

                            if w_val - w_zero < 0:
                                w_val_next = -127
                            else:
                                w_val_next = 127

                            w_scale_next = w_scale / (w_val_next / (w_val - w_zero))

                            weights[0,0,0,k_] += w_val_next
                            weight_quant['scale'][k_] = float(w_scale_next)

                        conv_op, tensors, buffers = op_conv(tensors, buffers, builtin_codes.index('DEPTHWISE_CONV_2D'), tensor_idx, False, 'DepthwiseConv2DOptions', dconv_opts,
                                                        oshape, oquant,
                                                        weights, weight_quant,
                                                        bias_data, bquant)
                        # adjust
                        for n,next_op in enumerate(operators):
                            if op['outputs'][0] in next_op['inputs']:
                                next_op['inputs'] = [conv_op['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]
                        operators = operators[:i] + [conv_op] + operators[i+1:]

        elif transform == 'SUB_AS_ADD':
            if opcode in ['SUB']:
                t0 = tensors[op['inputs'][0]]
                t1 = tensors[op['inputs'][1]]
                if 'data' in buffers[t0['buffer']] or 'data' in buffers[t1['buffer']]:
                    if 'data' in buffers[t0['buffer']]:
                        f_tensor = tensors[op['inputs'][0]]
                        tensor_idx = op['inputs'][1]
                    else:
                        f_tensor = tensors[op['inputs'][1]]
                        tensor_idx = op['inputs'][0]
                    o_tensor = tensors[op['outputs'][0]]
                    o_quant = o_tensor['quantization']
                    f_quant = f_tensor['quantization'].copy()
                    filter_data = get_numpy_data(f_tensor, buffers)

                    filter_data *= -1
                    f_quant['zero_point'][0] *= -1
                    data = np.frombuffer(filter_data.tobytes(), dtype=np.uint8).tolist()

                    add_op, tensors, buffers = op_add(tensors, buffers, builtin_codes.index('ADD'), tensor_idx, False,
                           o_quant['scale'], o_quant['zero_point'],
                           data, filter_data.shape, f_tensor['type'], f_quant['scale'], f_quant['zero_point'])
                    # adjust
                    for n,next_op in enumerate(operators):
                        if op['outputs'][0] in next_op['inputs']:
                            next_op['inputs'] = [add_op['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]
                    operators = operators[:i] + [add_op] + operators[i+1:]

        elif transform == 'DIV_AS_MUL':
            if opcode in ['DIV']:
                t1 = tensors[op['inputs'][1]]
                if 'data' in buffers[t1['buffer']]:
                    # inject_op, tensors, buffers = op_mul(tensors, buffers, opcodes.index('MUL'), o0, True,
                    #        [1.0], [mul_zp],
                    #        scale_q_value, [1,1,1,channel], dtype.upper(), [scale], [scale_zero_point])
                    # operators = operators[:i] + [inject_op] + operators[i+1:]
                    pass

        elif transform == 'FUSE_CONV':

            if opcode in ['CONV_2D']:
                _next_ops = [_ for _ in range(len(operators)) if _op['outputs'][0] in operators[_]['inputs']]
                if len(_next_ops) == 1:
                    next_opcode = builtin_codes[_next_ops[0]['opcode_index']]
                    if next_opcode in ['CONV_2D']:
                        pass


        elif transform == 'FC_CONV_2D':
            if opcode in ['FULLY_CONNECTED']:
                ops = []
                ishape = tensors[op['inputs'][0]]['shape'].copy()
                oshape = tensors[op['outputs'][0]]['shape'].copy()
                oquant = tensors[op['outputs'][0]]['quantization']
                tensor_idx = op['inputs'][0]

                f_tensor = tensors[op['inputs'][1]]
                filter_data = get_numpy_data(f_tensor, buffers)
                weight_quant = f_tensor['quantization'].copy()
                k,c = filter_data.shape[0], filter_data.shape[1]

                b_tensor = None
                bias_data = np.zeros((k,), dtype=np.int64)
                bquant = {'scale': [1. for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 3}
                if len(op['inputs']) > 2 and op['inputs'][2] != -1:
                    b_tensor = tensors[op['inputs'][2]]
                    bias_data = get_numpy_data(b_tensor, buffers)
                    bias_quant = b_tensor['quantization'].copy()


                weight_data = np.zeros((k,1,1,c), dtype=np.int64)
                for k_ in range(k):
                    for c_ in range(c):
                        weight_data[k_,0,0,c_] = filter_data[k_,c_]

                _reshape = [ishape[-2],1,1,c]
                reshape_op, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), tensor_idx, False, _reshape)

                ops.append(reshape_op)
                tensor_idx = len(tensors)-1

                conv_opts = {'stride_h': 1, 'stride_w': 1, 'dilation_h_factor': 1, 'dilation_w_factor': 1, 'padding': 'VALID'}
                conv_op, tensors, buffers = op_conv(tensors, buffers, builtin_codes.index('CONV_2D'), tensor_idx, False, 'Conv2DOptions', conv_opts,
                                                    oshape, oquant,
                                                    weight_data, weight_quant,
                                                    bias_data, bias_quant)
                ops.append(conv_op)
                tensor_idx = len(tensors)-1

                # adjust
                for n,next_op in enumerate(operators):
                    if op['outputs'][0] in next_op['inputs']:
                        next_op['inputs'] = [ops[-1]['outputs'][0] if _ == op['outputs'][0] else _ for _ in next_op['inputs']]

                operators = operators[:i] + ops + operators[i+1:]
                i += len(ops) - 1

        elif transform == 'REWRITE_NORM':
            if find_pattern(operators[i:], tensors, buffers, builtin_codes, 'NORM'):
                ops = []
                tensor_idx = op['inputs'][0]
                ishape = tensors[op['inputs'][0]]['shape']
                itype = tensors[op['inputs'][0]]['type']
                oshape = tensors[op['outputs'][0]]['shape']
                otype = tensors[op['outputs'][0]]['type']
                num_splits = ishape[-1] // oshape[-1]
                i_quant = tensors[op['inputs'][0]]['quantization']

                grab_op = lambda x : (operators[x].copy(), tensors[operators[x]['outputs'][0]]['quantization'].copy(), tensors[operators[x]['outputs'][0]]['type'])
                wvals = lambda x : (get_numpy_data(tensors[x], buffers), tensors[x]['quantization'].copy(), tensors[x]['type'])

                mean0_op, mean0_quant, mean0_type = grab_op(i+1)
                sub0_op, sub0_quant, sub0_type = grab_op(i+2)
                mul0_op, mul0_quant, mul0_type = grab_op(i+3)
                mean1_op, mean1_quant, mean1_type = grab_op(i+4)
                add0_op, add0_quant, add0_type = grab_op(i+5)
                dequant_op, dequant_quant, dequant_type = grab_op(i+6)
                sqrt_op, sqrt_quant, sqrt_type = grab_op(i+7)
                div_op, div_quant, div_type = grab_op(i+8)
                quant_op, quant_quant, quant_type = grab_op(i+9)
                mul1_op, mul1_quant, mul1_type = grab_op(i+10)
                mul2_op, mul2_quant, mul2_type = grab_op(i+11)
                mul3_op, mul3_quant, mul3_type = grab_op(i+12)
                sub1_op, sub1_quant, sub1_type = grab_op(i+13)
                add1_op, add1_quant, add1_type = grab_op(i+14)
                final_op, final_quant, final_type = grab_op(i+15)

                add0_val, add0_fquant, add0_ftype = wvals(add0_op['inputs'][1])
                div_val, div_fquant, div_ftype = wvals(div_op['inputs'][0])
                mul1_val, mul1_fquant, mul1_ftype = wvals(mul1_op['inputs'][1])
                sub1_val, sub1_fquant, sub1_ftype = wvals(sub1_op['inputs'][0])

                split_op, tensors, buffers = op_split(tensors, buffers, builtin_codes.index('SPLIT'), tensor_idx, False, -1, num_splits)
                ops.append(split_op)

            
                mean_outputs = []
                for n in range(num_splits):
                    mean_op, tensors, buffers = op_mean(tensors, buffers, builtin_codes.index('MEAN'), split_op['outputs'][n], False, [1,2,3])
                    tensors[mean_op['outputs'][0]]['quantization'] = mean0_quant.copy()

                    mean_outputs.append(mean_op['outputs'][0])
                    ops.append(mean_op)

                sub_outputs = []
                for n in range(num_splits):
                    tensor_idx = split_op['outputs'][n]
                    tensor2_idx = mean_outputs[n]

                    sub_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('SUB'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], sub0_quant.copy())
                    sub_outputs.append(sub_op['outputs'][0])
                    ops.append(sub_op)

                mul_outputs = []
                for n in range(num_splits):
                    tensor_idx = sub_outputs[n]
                    tensor2_idx = sub_outputs[n]

                    mul_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('MUL'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], mul0_quant.copy())
                    mul_outputs.append(mul_op['outputs'][0])
                    ops.append(mul_op)

                mean1_outputs = []
                for n in range(num_splits):
                    mean_op, tensors, buffers = op_mean(tensors, buffers, builtin_codes.index('MEAN'), mul_outputs[n], False, [1,2,3])
                    tensors[mean_op['outputs'][0]]['quantization'] = mean1_quant.copy()

                    mean1_outputs.append(mean_op['outputs'][0])
                    ops.append(mean_op)

                add0_val = add0_val.reshape((1,1,1,1))
                add0_outputs = []
                for n in range(num_splits):

                    data = np.frombuffer(add0_val.tobytes(), dtype=np.uint8).tolist()

                    add0_op, tensors, buffers = op_add(tensors, buffers, builtin_codes.index('ADD'), mean1_outputs[n], False,
                           add0_quant['scale'], add0_quant['zero_point'],
                           data, add0_val.shape, add0_ftype, add0_fquant['scale'], add0_fquant['zero_point'])
                    add0_outputs.append(add0_op['outputs'][0])
                    ops.append(add0_op)

                dequant_outputs = []
                for n in range(num_splits):

                    d_op, tensors, buffers = op_type(tensors, buffers, builtin_codes.index('DEQUANTIZE'), add0_outputs[n], False, dequant_type, dequant_quant.copy())

                    dequant_outputs.append(d_op['outputs'][0])
                    ops.append(d_op)

                sqrt_outputs = []
                for n in range(num_splits):

                    sqrt_op, tensors, buffers = op_type(tensors, buffers, builtin_codes.index('SQRT'), dequant_outputs[n], False, sqrt_type, sqrt_quant.copy())

                    sqrt_outputs.append(sqrt_op['outputs'][0])
                    ops.append(sqrt_op)

                div_outputs = []
                for n in range(num_splits):
                    data = np.frombuffer(div_val.tobytes(), dtype=np.uint8).tolist()
                    div_op, tensors, buffers = op_div(tensors, buffers, builtin_codes.index('DIV'), sqrt_outputs[n], False, True, div_quant.copy(), data, div_val.shape, div_ftype, div_fquant.copy())

                    div_outputs.append(div_op['outputs'][0])
                    ops.append(div_op)

                quant_outputs = []
                for n in range(num_splits):

                    q_op, tensors, buffers = op_type(tensors, buffers, builtin_codes.index('QUANTIZE'), div_outputs[n], False, quant_type, quant_quant.copy())

                    quant_outputs.append(q_op['outputs'][0])
                    ops.append(q_op)

                mul1_outputs = []
                for n in range(num_splits):
                    val = mul1_val[:,:,:,n,:].reshape([1,1,1,-1])
                    data = np.frombuffer(val.tobytes(), dtype=np.uint8).tolist()

                    mul1_op, tensors, buffers = op_mul(tensors, buffers, builtin_codes.index('MUL'), quant_outputs[n], False,
                           mul1_quant['scale'], mul1_quant['zero_point'],
                           data, val.shape, mul1_ftype, mul1_fquant['scale'], mul1_fquant['zero_point'])
                    mul1_outputs.append(mul1_op['outputs'][0])
                    ops.append(mul1_op)

                mul2_outputs = []
                for n in range(num_splits):
                    tensor_idx = split_op['outputs'][n]
                    tensor2_idx = mul1_outputs[n]

                    mul_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('MUL'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], mul2_quant.copy())
                    mul2_outputs.append(mul_op['outputs'][0])
                    ops.append(mul_op)

                mul3_outputs = []
                for n in range(num_splits):
                    tensor_idx = mul1_outputs[n]
                    tensor2_idx = mean_outputs[n]

                    mul_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('MUL'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], mul3_quant.copy())
                    mul3_outputs.append(mul_op['outputs'][0])
                    ops.append(mul_op)

                sub1_outputs = []
                for n in range(num_splits):
                    val = sub1_val[:,:,:,n,:].reshape([1,1,1,-1])
                    data = np.frombuffer(val.tobytes(), dtype=np.uint8).tolist()

                    sub1_op, tensors, buffers = op_sub(tensors, buffers, builtin_codes.index('SUB'), mul3_outputs[n], False, True,
                           sub1_quant['scale'], sub1_quant['zero_point'],
                           data, val.shape, sub1_ftype, sub1_fquant['scale'], sub1_fquant['zero_point'])
                    sub1_outputs.append(sub1_op['outputs'][0])
                    ops.append(sub1_op)

                add1_outputs = []
                for n in range(num_splits):
                    tensor_idx = mul2_outputs[n]
                    tensor2_idx = sub1_outputs[n]

                    add_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('ADD'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], add1_quant.copy())
                    add1_outputs.append(add_op['outputs'][0])
                    ops.append(add_op)

                concat_op, tensors, buffers = op_concat(tensors, buffers, builtin_codes.index('CONCATENATION'), add1_outputs, tensor_idx, -1)
                ops.append(concat_op)

                concat_inputs = []
                for n,next_op in enumerate(operators):
                    if final_op['outputs'][0] in next_op['inputs']:
                        next_op['inputs'] = [ops[-1]['outputs'][0] if _ == final_op['outputs'][0] else _ for _ in next_op['inputs']]

                tensors[ops[-1]['outputs'][0]]['quantization'] = tensors[ops[-2]['outputs'][0]]['quantization'].copy()
                # tensors[ops[-1]['outputs'][0]]['shape'] = [1,1,1,num_splits]

                operators = operators[:i] + ops + operators[i+15+1:]
                i += len(ops) - 1

        elif transform == 'YOLO_ARG_MAX':
            if is_graph_output(op, operators, tensors, buffers):
                if tensors[op['outputs'][0]]['shape'][-1] in [80]:
                    id_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PAD'), op['outputs'][0], False)

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


def find_pattern(operators, tensors, buffers, opcodes, pattern):
    if pattern == 'NORM':
        codes = [opcodes[op['opcode_index']] for i,op in zip(range(16), operators)]
        target = ['RESHAPE', 'MEAN', 'SUB', 'MUL', 'MEAN', 'ADD', 'DEQUANTIZE', 'SQRT',
                  'DIV', 'QUANTIZE', 'MUL', 'MUL', 'MUL', 'SUB', 'ADD', 'RESHAPE']
        if len(codes) == 16 and codes == target:
            return True
    return False


def load_graph(src_tflite, copy=False):
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name

    tmp_tflite = os.path.join(tmp_dir, os.path.basename(src_tflite))
    shutil.copyfile(src_tflite, tmp_tflite)

    # Convert TFLITE to JSON, read graph from JSON
    jname = tflite2json(tmp_tflite)
    if copy:
        shutil.copyfile(jname, src_tflite.replace('.tflite', '.json'))
    graph = json_load(jname)

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

    json_dump(graph, tmp_json)
    tname = json2tflite(tmp_json)

    update_shapes(tname, graph)

    if copy:
        if debug:
            shutil.copyfile(tmp_json, dst_tflite.replace('.tflite', '.json'))
        shutil.copyfile(tname, dst_tflite)
    return tmp_tflite


def clean_operator_codes(operator_codes):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    schema = json_load(schema_path.replace('.fbs', '.json'))
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
        



def clean_operators(operators, tensors, buffers, rename=False, force_shape=None, sort=None):
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
    if not sort is None:
        try:
            if sort == 'dfs':
                s = topological_sort_dfs(soperators, tensors, buffers)
            elif sort == 'kahn':
                s = topological_sort_kahn(soperators, tensors, buffers)
        except:
            print("WARNING: sort {} failed".format(sort))
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


def unused_partial_outputs(output_tensors, all_input_tensors, operators):
    unused_tensors = []

    for idx in output_tensors:
        source_op = None
        for op in operators:
            if idx in op['outputs']:
                source_op = op
                break
        assert(source_op)
        if len(source_op['outputs']) > 1:
            for output in [_ for _ in source_op['outputs'] if _ != idx]:
                if output in all_input_tensors:
                    unused_tensors.append(idx)
                    break

    return unused_tensors


def get_io_tensors(operators, tensors, buffers, opcodes):

    all_input_tensors, all_output_tensors = [], []
    for op in operators:
        all_input_tensors += op['inputs']
        all_output_tensors += op['outputs']

    input_tensors = []
    for i in all_input_tensors:
        if not i in input_tensors and not i in all_output_tensors:
            input_tensors.append(i)

    output_tensors = []
    for o in all_output_tensors:
        if not o in output_tensors and not o in all_input_tensors:
            output_tensors.append(o)

    ignored_tensors = unused_partial_outputs(output_tensors, all_input_tensors, operators)

    valid_input_tensors = []
    for idx in input_tensors:
        buf = buffers[tensors[idx]['buffer']]
        if 'data' in buf:
            continue
        valid_input_tensors.append(idx)

    valid_output_tensors = []
    for idx in output_tensors:
        if idx in ignored_tensors:
            continue
        buf = buffers[tensors[idx]['buffer']]
        if 'data' in buf:
            continue
        valid_output_tensors.append(idx)

    return valid_input_tensors, valid_output_tensors


def create_signature_defs(graph, tensors, input_tensors, output_tensors):
    graph['signature_defs'] = [{'inputs': None, 'outputs': None, 'signature_key':'serving_default', 'subgraph_index':0}]
    graph['signature_defs'][0]['inputs'] = [{'name': tensors[idx]['name'], 'tensor_index': idx} for idx in input_tensors]
    graph['signature_defs'][0]['outputs'] = [{'name': tensors[idx]['name'], 'tensor_index': idx} for idx in output_tensors]
    return graph


def create_graph(operators, tensors, buffers, opcodes, force_shape=None):
    operators, tensors, buffers = [_.copy() for _ in operators], [_.copy() for _ in tensors], buffers.copy()

    operators, tensors, buffers = clean_operators(operators, tensors, buffers, force_shape=force_shape)
    input_tensors, output_tensors = get_io_tensors(operators, tensors, buffers, opcodes)

    g = {'description': 'VectorBlox transform', 'version': 3, 'metadata':[], 'operator_codes': opcodes, 'subgraphs': [{}]}
    subg = g['subgraphs'][0]
    subg['inputs'], subg['outputs'] = input_tensors, output_tensors
    g['buffers'], subg['tensors'], subg['operators'] = buffers, tensors, operators

    g = create_signature_defs(g, tensors, input_tensors, output_tensors)
    if g['signature_defs'][0]['inputs'] == []:
        print('WARNING: NO INPUTS')
    if g['signature_defs'][0]['outputs'] == []:
        print('WARNING: NO OUTPUTS')

    return g


def all_close_graphs(t0, t1):
    with open(t0, 'rb') as f:
        i0, o0 = generate_inputs_outputs(f.read(), int8_range=True)
    with open(t1, 'rb') as f:
        i1, o1 = generate_inputs_outputs(f.read(), int8_range=True)

    close = [np.allclose(o0[_],o1[_]) for _ in o0.keys()]
    max_diff = 0

    if not all(close):
        try:
            for _ in o0.keys():
                a,b = o0[_], o1[_]
                heat = a-b
                if np.max(np.abs(heat)) > 1:
                    while len(heat.shape) < 3:
                        heat = np.expand_dims(heat, axis=0)
                    np.save("heatmap.{}.npy".format(_), heat)
                max_diff = max(max_diff, np.max(np.abs(heat)))
        except:
            pass

    return all(close), max_diff
    

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
            if not 'PADV2' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 60, 'version': 1, 'builtin_code': 'PADV2'})
        elif transform == 'PADV2':
            if not 'PADV2' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 60, 'version': 1, 'builtin_code': 'PADV2'})
        elif transform == 'CHANNEL_SPLIT':
            if not 'SPLIT_V' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 102, 'version': 2, 'builtin_code': 'SPLIT_V'})
        elif transform in ['GROUP_CONV', 'GROUP_DEPTH', 'GROUP_DEPTH5x2', 'FULL_DEPTH']:
            if not 'CONV_2D' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 3, 'version': 3, 'builtin_code': 'CONV_2D'})
            if not 'SPLIT' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 49, 'version': 2, 'builtin_code': 'SPLIT'})
            if not 'SPLIT_V' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 102, 'version': 2, 'builtin_code': 'SPLIT_V'})
            if not 'CONCATENATION' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 2, 'version': 1, 'builtin_code': 'CONCATENATION'})
        elif transform in ['STRIDED_DEPTHWISE', 'STRIDED_CONV']: 
            if not 'STRIDED_SLICE' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 45, 'version': 1, 'builtin_code': 'STRIDED_SLICE'})
        elif transform == 'NCHW_RESHAPE':
            if not 'TRANSPOSE' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 39, 'version': 1, 'builtin_code': 'TRANSPOSE'})
        elif transform == 'MUL_AS_CONV':
            if not 'DEPTHWISE_CONV_2D' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 4, 'version': 3, 'builtin_code': 'DEPTHWISE_CONV_2D'})
        elif transform == 'SUB_AS_ADD':
            if not 'ADD' in opcodes:
                opcodes.append({'deprecated_builtin_code': 0, 'version': 2, 'builtin_code': 'ADD'})
        elif transform == 'DIV_AS_MUL':
            if not 'MUL' in opcodes:
                opcodes.append({'deprecated_builtin_code': 18, 'version': 2, 'builtin_code': 'MUL'})
        elif transform == 'FC_CONV_2D':
            if not 'CONV_2D' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 3, 'version': 3, 'builtin_code': 'CONV_2D'})
            if not 'RESHAPE' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 22, 'version': 1, 'builtin_code': 'RESHAPE'})
        elif transform == 'TRANSPOSE_CONV': 
            if not 'CONV_2D' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 3, 'version': 3, 'builtin_code': 'CONV_2D'})
            if not 'DILATE' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 127, 'version': 1, 'builtin_code': 'DILATE'})
            if not 'PAD' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 34, 'version': 1, 'builtin_code': 'PAD'})
        elif transform == 'AVERAGE_POOL_2D':
            if not 'DEPTHWISE_CONV_2D' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 4, 'version': 3, 'builtin_code': 'DEPTHWISE_CONV_2D'})
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
        elif transform in ['REWRITE_NORM', 'REWRITE_ATTN']:
            if not 'SPLIT' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 49, 'version': 2, 'builtin_code': 'SPLIT'})
            if not 'CONCATENATION' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 2, 'version': 1, 'builtin_code': 'CONCATENATION'})
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
            isPostprocessing = dataset in ['DEPTH']
            inject_op, tensors, buffers = op_resize(tensors, buffers, opcodes.index(resize), current_output, False, current_shape, isPostprocessing=isPostprocessing)

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
    parser.add_argument("tflite")
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
    parser.add_argument("tflite")
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
    parser.add_argument("tflite")
    parser.add_argument("-p", "--passes", nargs='+', choices=passes)
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()

    graph, dir_obj = load_graph(args.tflite)

    transformed_graph = transform_graph(graph, args.passes, args.verbose)
    save_graph(args.tflite.replace('.tflite', '.tr.tflite'), dir_obj, transformed_graph, copy=True)

    if not dir_obj is None:
        dir_obj.cleanup()

    if args.verbose:
        all_close, max_diff = all_close_graphs(args.tflite, args.tflite.replace('.tflite', '.tr.tflite'))
        if not all_close and max_diff > 1:
            print("WARNING: transformed graph doesn't match. Max diff == {}".format(max_diff))


if __name__ == "__main__":
    main()
