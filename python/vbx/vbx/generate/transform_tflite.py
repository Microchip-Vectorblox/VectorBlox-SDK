import argparse
from .utils import json_load, json_dump, generate_inputs_outputs
import copy
import subprocess
import shlex
import os.path
import shutil, glob, tempfile
import tensorflow as tf
import numpy as np
import cv2
import vbx.postprocess.dataset as rgb_color
from math import floor, ceil, log2, frexp, copysign, exp, tanh, pow, log, sqrt


MAX_LUTS = 4

clone_count = 0
unique_count = 0

passes = [
        'REMOVE_FP32_IO',
        'CLEAN_LOGISTIC',
        'LUT',
        'NORMALIZE_ADD_INPUTS',
        'REMOVE_CONSTANTS',
        'IMPLICIT_PAD',
        'PADV2',
        'SHARED_PAD',
        'EXPLICIT_PAD',
        'FUSE_GROUP1',
        'GROUP_DEPTH',
        'GROUP_DEPTH5x2',
        'GROUP_CONV',
        'TRANSPOSE_CONV',
        'FC_CONV_2D',
        'FC2_CONV_2D',
        'STRIDED_DEPTHWISE',
        'YOLO_ARG_MAX',
        'AVERAGE_POOL_2D',
        'FULL_DEPTH',
        'STRIDED_CONV',
        'NCHW_RESHAPE',
        'REDUCE_MAX_POOL',
        'MAXPOOL_INPUT_CHANNEL_SPLIT',
        'REDUCE_RESIZE',
        'SUB_AS_ADD',
        'DIV_AS_MUL',
        'MUL_AS_CONV',
        'CHANNEL_SPLIT',
        'UNPACK_SLICE',
        'FUSE_CONV',
        'SPNV2_TRANSPOSE',
        'REWRITE_NORM',
        'Y26',
        'FUSE_ADJ',
        'FUSE_ADD',
        'FUSE_SLICES',
        'FRAC_BATCH',
        'FRAC_FC',
        'YOLO_ATTN',
        'REWRITE_RESHAPE',
        'REMOVE_NOPS',
        'SPLIT_XL_POOL',
        'EXPAND_BROADCAST',
        'EXPAND_INPUT_DIMS',
        'SORT_DFS',
        'SORT_KHAN',
        ]


def create_tensor_data(dtype, shape, min_value=-100, max_value=100, int8_range=False):
  """Build tensor data spreading the range [min_value, max_value)."""

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


def get_interp(tflite_model):
    if isinstance(tflite_model, bytes):
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
    elif isinstance(tflite_model, str):
        interpreter = tf.lite.Interpreter(model_path=tflite_model, experimental_preserve_all_tensors=True)
    else:
        print('WARNING: invalid tflite model', type(tflite_model))
        return None

    interpreter.allocate_tensors()
    return interpreter


def update_shapes(tflite_path, graph):

    subgraph_tensors = graph['subgraphs'][0]['tensors']
    old_shapes = {_['name']: {'shape': tuple(_['shape']), 'index': i} for i,_ in enumerate(subgraph_tensors) if ('shape' in _ and _['shape'])}

    need_updated_shapes = False
    interp = get_interp(tflite_path)
    if interp is None:
        details = None
    else:
        details = interp.get_tensor_details()

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


def transpose_io_to_vnnx(x):
    # if len(x.shape) == 3:
    #     x = x.transpose((2,0,1))
    if len(x.shape) == 4:
        x = x.transpose((0,3,1,2))
    elif len(x.shape) == 5:
        x = x.transpose((0,1,4,2,3))
    elif len(x.shape) == 6:
        x = x.transpose((0,1,2,5,3,4))
    return x


def channels_first_shape(shape):
    s = list(shape)
    # if len(shape) >= 3:
    if len(shape) > 3 or (len(shape) == 3 and s[0] > 1):
        axis = 3
        s = s[:-axis] + s[-1:] + s[-axis:-1]
    return tuple(s), len(s)


def is_singleton(shapes):
    for shape in shapes:
        _shape, dims = channels_first_shape(shape)
        if len(_shape) > 1 and _shape[-1] == 1 and _shape[-2] == 1:
            return True
    return False


def is_singleton_op(op, tensors):
    shapes=[]
    op_inputs = [_ for _ in op['inputs'] if _ != -1]
    for op_input in op_inputs:
        if 'shape' in tensors[op_input]:
            shapes.append(tensors[op_input]['shape'])

    return is_singleton(shapes)


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
# Order patterns longest-first so longer patterns match before shorter ones
valid_reshape_patterns += [["RESHAPE", "TRANSPOSE", "RESHAPE"]]  # 3 ops
valid_reshape_patterns += [["ARG_MAX", "CAST", "RESHAPE"]]       # 3 ops
valid_reshape_patterns += [["ARG_MAX", "RESHAPE", "CAST"]]       # 3 ops
valid_reshape_patterns += [["ARG_MAX", "CAST"]]                  # 2 ops
valid_reshape_patterns += [["ARG_MAX", "RESHAPE"]]               # 2 ops (must be last)

valid_norm_patterns = []
valid_norm_patterns += [["DEQUANTIZE", "MEAN", "SQUARED_DIFFERENCE", "MEAN", "ADD", "RSQRT", "MUL", "MUL", "MUL", "SUB", "ADD", "QUANTIZE"]]
valid_norm_patterns += [["DEQUANTIZE", "MEAN", "RESHAPE", "SQUARED_DIFFERENCE", "MEAN", "ADD", "RSQRT", "MUL", "SUB", "RESHAPE", "RESHAPE", "MUL", "ADD", "QUANTIZE"]]


def is_pattern(operators, codes, tensors, buffers, valid_patterns):
    # Return on first matching pattern (patterns should be ordered longest-first)
    for pattern in valid_patterns:
        if len(pattern)-1 < len(operators):
            if all([pattern[i] == codes[operators[i]['opcode_index']] for i in range(len(pattern))]):
                return list(range(len(pattern)))

    return []


def graph_pattern(operators, codes, tensors, buffers, extra_patterns=None):
    rp = is_pattern(operators, codes, tensors, buffers, valid_reshape_patterns)
    if len(rp):
        return "TRANSFORM", rp

    lp = is_pattern(operators, codes, tensors, buffers, valid_lut_patterns)
    if len(lp):
        return "LUT", lp

    np = is_pattern(operators, codes, tensors, buffers, valid_norm_patterns)
    if len(np):
        return "NORM", np

    if not extra_patterns is None and len(extra_patterns):
        ep = is_pattern(operators, codes, tensors, buffers, extra_patterns)
        if len(ep):
            return "EXTRA", ep

    return None, []


def num_lut_tensor(filter, tensors, buffers, max_luts=MAX_LUTS):
    weight_tensor = tensors[filter]
    data = get_numpy_data(weight_tensor, buffers)
    single_value = np.all(data.flatten() == data.flatten()[0])
    single_value = False #TODO reduce weights
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


def get_activations(op, tensors, buffers):
    inputs = [idx for idx in op['inputs'] if 'data' not in buffers[tensors[idx]['buffer']]]
    outputs = [idx for idx in op['outputs'] if 'data' not in buffers[tensors[idx]['buffer']]]
    return inputs, outputs


def get_connected_ops(operators, idx):
    op = operators[idx]

    prev_ops = []
    for t in op['inputs']:
        prev_ops += [_ for _ in range(len(operators)) if t in operators[_]['outputs']]

    next_ops = []
    for t in op['outputs']:
        next_ops += [_ for _ in range(len(operators)) if t in operators[_]['inputs']]

    return prev_ops, next_ops


def get_next_op(operators, idx):
    if idx is None:
        return None

    _op = operators[idx]
    _next_ops = [_ for _,o in enumerate(operators) if _op['outputs'][0] in o['inputs']]
    if len(_next_ops) >= 1:
        return _next_ops[0]
    return None


def get_next_op(operators, idx):
    if idx is None:
        return None

    _op = operators[idx]
    _next_ops = [_ for _ in range(len(operators)) if _op['outputs'][0] in operators[_]['inputs']]
    if len(_next_ops) >= 1:
        return _next_ops[0]
    return None


def pattern_lut_table(pattern_ops, tensors, buffers, opcodes, lut_count, pattern_count):
    tmp_dir_obj = tempfile.TemporaryDirectory()
    graph = create_graph(pattern_ops, tensors, buffers, opcodes, force_shape=[1,1,256,lut_count], clean_graph=True)
    save_graph('pattern.{}.tflite'.format(pattern_count), tmp_dir_obj, graph, copy=False)
    with open(os.path.join(tmp_dir_obj.name, 'pattern.{}.tflite'.format(pattern_count)), 'rb') as f:
        inputs, outputs = generate_inputs_outputs(f.read(), int8_range=True)
    table = outputs['o0'].squeeze()
    tmp_dir_obj.cleanup()
    return table


def lut_pattern(operators, codes, tensors, buffers, idx, compression_vbx=None):
    patterns = []
    lut_count = 1

    prev_op = None
    prev_inputs = None
    prev_outputs = None

    # Graph-input preprocessing chains (e.g. QUANTIZE -> MUL -> ADD for
    # uint8-in / per-channel-mean-subtract models) are simulated end-to-end
    # and, on ucomp, affine-fit into a single synthetic conv_uint8 CONV_2D
    # (see try_build_conv_uint8 / is_graph_input gating below); a trailing
    # per-channel ADD there is real signal (mean subtraction) and must be
    # folded in, unlike a mid-graph residual ADD which hardware handles via
    # its own fused-activation path and which must NOT be pulled into a LUT.
    is_input_pattern = idx < len(operators) and is_graph_input(operators[idx], operators, tensors, buffers)

    weighted_opcodes = ["MUL", "DIV", "ADD", "SUB", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM", "PRELU"]
    activation_opcodes = ["HARD_SWISH", "LOGISTIC", "TANH", 'LEAKY_RELU', 'RELU', 'RELU6', 'RELU_N1_TO_1', 'RELU_0_TO_1']
    activation_opcodes += ["SQRT", "RSQRT", "EXP", "LOG", "ELU", "GELU", "POW", "COS", "SIN", "NEG"]

    max_idx = len(operators)
    while idx < max_idx:
        op = operators[idx]
        opcode = codes[op['opcode_index']]
        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]

        multi_input = is_multi_input(op, tensors, buffers)
        filters = [_ for _ in op_inputs if 'data' in buffers[tensors[_]['buffer']]]

        idx_1 = get_next_op(operators, idx)
        next_op, next_opcode = None, ''
        if idx_1:
            next_op = operators[idx_1]
            next_opcode = codes[next_op['opcode_index']]
            next_op_inputs = [_ for _ in next_op['inputs'] if _ != -1]
            next_filters = [_ for _ in next_op_inputs if 'data' in buffers[tensors[_]['buffer']]]

        idx_2 = get_next_op(operators, idx_1)
        next_next_op, next_next_opcode = None, ''
        if idx_2:
            next_next_op = operators[idx_2]
            next_next_opcode = codes[next_next_op['opcode_index']]
            next_next_op_inputs = [_ for _ in next_next_op['inputs'] if _ != -1]
            next_next_filters = [_ for _ in next_next_op_inputs if 'data' in buffers[tensors[_]['buffer']]]

        idx_3 = get_next_op(operators, idx_2)
        next_next_next_op, next_next_next_opcode = None, ''
        if idx_3:
            next_next_next_op = operators[idx_3]
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
            if is_pattern(operators[patterns[-1]:], codes, tensors, buffers, valid_lut_patterns):
                lp = is_pattern(operators[idx:], codes, tensors, buffers, valid_lut_patterns)
                if len(lp) == 3:
                    idx += len(lp) - 1

        elif opcode == "DEQUANTIZE" and next_opcode == "LOGISTIC" and next_next_opcode == "QUANTIZE" and next_next_next_opcode == "MUL" and (op['inputs'][0] in next_next_next_op['inputs']): #SILU pattern
            patterns.append(idx)
            patterns.append(idx_1)
            patterns.append(idx_2)
            patterns.append(idx_3)
            idx += 3


        elif opcode == "DEQUANTIZE" and (next_opcode in activation_opcodes or (next_opcode in weighted_opcodes and len(next_filters) == 1)) and (next_next_opcode in activation_opcodes or (next_next_opcode in weighted_opcodes and len(next_next_filters) == 1)) and next_next_next_opcode == "QUANTIZE":
            # Skip ADD patterns for ucomp - ADD should remain as-is with fused activation
            if compression_vbx == 'ucomp' and (next_opcode == 'ADD' or next_next_opcode == 'ADD'):
                idx += 1
                continue
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
            patterns.append(idx_1)
            patterns.append(idx_2)
            patterns.append(idx_3)
            idx += 3
        
        elif opcode == "DEQUANTIZE" and (next_opcode in activation_opcodes or (next_opcode in weighted_opcodes and len(next_filters) == 1)) and next_next_opcode == "QUANTIZE":
            # Skip ADD patterns for ucomp - ADD should remain as-is with fused activation
            if compression_vbx == 'ucomp' and next_opcode == 'ADD':
                idx += 1
                continue
            if next_opcode in weighted_opcodes:
                required = num_lut_tensor(next_filters[0], tensors, buffers)
                if required < 0:
                    break
                if required > lut_count:
                    lut_count = required
            patterns.append(idx)
            patterns.append(idx_1)
            patterns.append(idx_2)
            idx += 2

        elif opcode == "LOGISTIC" and next_opcode == "MUL" and (op['inputs'][0] in next_op['inputs']): #SILU pattern
            patterns.append(idx)
            patterns.append(idx_1)
            idx += 1

        elif opcode == "RESHAPE" and next_opcode == "CAST" and next_next_opcode == "GATHER": 
            patterns.append(idx)
            patterns.append(idx_1)
            patterns.append(idx_2)
            # lut_count = 4 #for RGBA uint32
            idx += 2

        elif opcode == "CAST" and next_opcode == "GATHER": 
            patterns.append(idx)
            patterns.append(idx_1)
            # lut_count = 4 #for RGBA uint32
            idx += 1

        elif forked:
            break
        
        elif opcode in ["QUANTIZE"] and tensors[op_inputs[0]]['type'] in ['INT8', 'UINT8']:
            patterns.append(idx)

        elif opcode in ["DEQUANTIZE"] and tensors[op_outputs[0]]['type'] in ['INT8', 'UINT8']:
            patterns.append(idx)

        elif opcode in activation_opcodes and tensors[op_outputs[0]]['type'] in ['INT8', 'UINT8']:
            patterns.append(idx)

        elif opcode in weighted_opcodes and len(filters) == 1 and tensors[op_outputs[0]]['type'] in ['INT8', 'UINT8']:
            # Skip ADD operations for ucomp - ADD should be handled by hardware with
            # fused activation, EXCEPT at the very start of the graph (e.g. a
            # QUANTIZE+MUL+ADD input-preprocessing chain such as per-channel mean
            # subtraction), where the ADD carries real per-channel signal that must
            # be folded into the synthetic conv_uint8 fit rather than left as a
            # separate node (which downstream only tracks a scalar rescale).
            if compression_vbx == 'ucomp' and opcode == 'ADD' and not is_input_pattern:
                break
            required = num_lut_tensor(filters[0], tensors, buffers)
            if required < 0:
                break
            if required > lut_count:
                lut_count = required
            patterns.append(idx)
        else:
            break

        prev_op = operators[idx]
        idx_next = get_next_op(operators, idx)
        if idx_next is None:
            break
        else:
            idx = idx_next

        # prev_op = op
        prev_inputs = [_ for _ in prev_op['inputs'] if _ != -1]
        prev_outputs = [_ for _ in prev_op['outputs'] if _ != -1]
        
    return patterns, lut_count


def op_clone(op, tensors, buffers):
    global clone_count

    _op = {}
    for key in op.keys():
        if isinstance(op[key], dict) or isinstance(op[key], list):
            _op[key] = op[key].copy()
        else:
            _op[key] = op[key]

    for i,input in enumerate(op['inputs']):
        tensors.append(tensors[input].copy())
        if 'buffers' in tensors[input]:
            buffers.append(buffers[tensors[input]['buffer']].copy())
            tensors[-1]['buffer'] == len(buffers)-1
            print(tensors[input]['buffer'], tensors[-1]['buffer'])
        tensors[-1]['name'] = tensors[-1]['name'] + '_clone.{}.{}'.format(clone_count, i)
        clone_count += 1

        _op['inputs'][i] = len(tensors)-1

    for o,output in enumerate(op['outputs']):
        tensors.append(tensors[output].copy())
        if 'buffers' in tensors[output]:
            buffers.append(buffers[tensors[output]['buffer']].copy())
            tensors[-1]['buffer'] == len(buffers)-1
        tensors[-1]['name'] = tensors[-1]['name'] + '_clone.{}.{}'.format(clone_count, o)
        clone_count += 1

        _op['outputs'][o] = len(tensors)-1

    return _op, tensors, buffers


def op_mul(tensors, buffers, opcode_idx, tensor_idx, inject_before, scale, zero_point, data, data_shape, data_dtype, data_scale, data_zero_point):
    t = tensors[tensor_idx]

    data_quantization = {'details_type': 'NONE', 'quantized_dimension': 0}
    if data_dtype.upper() != 'FLOAT32':
        data_quantization['scale'] =  data_scale
        data_quantization['zero_point'] = data_zero_point

    io_quantization = {'details_type': 'NONE', 'quantized_dimension': 0}
    if t['type'] != 'FLOAT32':
        io_quantization['scale'] =  scale
        io_quantization['zero_point'] = zero_point

    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': data_shape,
        'type': data_dtype.upper(),
        'buffer': len(buffers)-1,
        'name': 'add_data:{}'.format(tensor_idx),
        'quantization': data_quantization,
        'is_variable': False,
        'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
        'type': t['type'],
        'buffer': len(buffers)-1,
        'name': 'add_{}'.format(tensor_idx),
        'quantization': io_quantization,
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

    data_quantization = {'details_type': 'NONE', 'quantized_dimension': 0}
    if data_dtype.upper() != 'FLOAT32':
        data_quantization['scale'] =  data_scale
        data_quantization['zero_point'] = data_zero_point

    io_quantization = {'details_type': 'NONE', 'quantized_dimension': 0}
    if t['type'] != 'FLOAT32':
        io_quantization['scale'] =  scale
        io_quantization['zero_point'] = zero_point

    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': data_shape,
        'type': data_dtype.upper(),
        'buffer': len(buffers)-1,
        'name': 'add_data:{}'.format(tensor_idx),
        'quantization': data_quantization,
        'is_variable': False,
        'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
        'type': t['type'],
        'buffer': len(buffers)-1,
        'name': 'add_{}'.format(tensor_idx),
        'quantization': io_quantization,
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

    data_quantization = {'details_type': 'NONE', 'quantized_dimension': 0}
    if data_dtype.upper() != 'FLOAT32':
        data_quantization['scale'] =  data_scale
        data_quantization['zero_point'] = data_zero_point

    io_quantization = {'details_type': 'NONE', 'quantized_dimension': 0}
    if t['type'] != 'FLOAT32':
        io_quantization['scale'] =  scale
        io_quantization['zero_point'] = zero_point

    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': data_shape,
        'type': data_dtype.upper(),
        'buffer': len(buffers)-1,
        'name': 'sub_data:{}'.format(tensor_idx),
        'quantization': data_quantization,
        'is_variable': False,
        'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
        'type': t['type'],
        'buffer': len(buffers)-1,
        'name': 'sub_{}'.format(tensor_idx),
        'quantization': io_quantization,
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


def op_elemwise(tensors, buffers, opcode_idx, tensor_idx, tensor2_idx, shape, dtype, quant, builtin_options_type=None, builtin_options=None):

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

    if not builtin_options is None and not builtin_options_type is None:
        inject_op['builtin_options_type'] = builtin_options_type
        inject_op['builtin_options'] = builtin_options



    return inject_op, tensors, buffers


def op_softmax(tensors, buffers, opcode_idx, tensor_idx, dtype, quant):
    t = tensors[tensor_idx]

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t['shape'].copy(),
        'type': dtype,
        'buffer': len(buffers)-1,
        'name': 'soft_{}'.format(tensor_idx),
        'quantization': quant,
        'is_variable': False,
        'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1

    inject_op = {'opcode_index': opcode_idx,
                'inputs': [tensor_idx],
                'outputs': [len(tensors)-1],
                'builtin_options_type': 'SoftmaxOptions',
                 'builtin_options': {'beta': 1}}

    return inject_op, tensors, buffers


def op_batchmatmul(tensors, buffers, opcode_idx, tensor_idx, tensor2_idx, shape, dtype, quant, builtin_options_type=None, builtin_options=None):

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

    if not builtin_options is None and not builtin_options_type is None:
        inject_op['builtin_options_type'] = builtin_options_type
        inject_op['builtin_options'] = builtin_options



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


def op_pool(tensors, buffers, opcode_idx, tensor_idx, inject_before, shape, opts):
    t = tensors[tensor_idx]

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': shape,
            'type': t['type'],
            'buffer': len(buffers)-1,
            'name': 'pool_{}'.format(tensor_idx),
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
    tshape = list(np.zeros(t['shape']).transpose(shape).shape)

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
    tensors.append({'shape': tshape,
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


def op_pack(tensors, buffers, opcode_idx, tensor_idx, inject_before, axis, values_count):
    t = tensors[tensor_idx]

    shape = t['shape'].copy()
    shape[axis] *= values_count

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': shape,
            'type': t['type'],
            'buffer': len(buffers)-1,
            'name': 'pack_{}'.format(tensor_idx),
            'quantization': t['quantization'],
            'is_variable': False,
            'has_rank': True})

    input_tensor, output_tensor = tensor_idx, len(tensors)-1
    if inject_before:
        input_tensor, output_tensor = len(tensors)-1, tensor_idx

    PackOptions = {'axis': axis, 'values_count': values_count}

    inject_op = {'opcode_index': opcode_idx,
                'builtin_options_type': 'PackOptions',
                'builtin_options': PackOptions,
                'inputs': [input_tensor for _ in range(values_count)],
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
    tensors.append({'shape': [len(begin)],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'slice_begin_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    data = np.frombuffer(np.asarray(end).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [len(end)],
            'type': 'INT32',
            'buffer': len(buffers)-1,
            'name': 'slice_end_{}'.format(tensor_idx),
            'quantization': {'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

    data = np.frombuffer(np.asarray(stride).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': [len(stride)],
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


def op_concat(tensors, buffers, opcode_idx, tensor_indices, tensor_idx, axis, force_shape=None):
    t = tensors[tensor_idx]

    shape = t['shape'].copy()
    if force_shape:
        shape = force_shape
    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': shape,
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


def op_fully_connected(tensors, buffers, opcode_idx, tensor_idx, inject_before, opts_name, opts, output_shape, output_quantization, weights, weight_quantization, biases, bias_quantization):
    t = tensors[tensor_idx]

    data = np.frombuffer(np.asarray(weights).astype(np.int8).tobytes(), dtype=np.uint8).tolist()
    buffers.append({'data': data, 'offset': 0, 'size': 0})
    tensors.append({'shape': weights.shape,
            'type': 'INT8',
            'buffer': len(buffers)-1,
            'name': 'fc_weights_{}'.format(tensor_idx),
            'quantization': weight_quantization,
            'is_variable': False,
            'has_rank': True})

    if not biases is None:
        data = np.frombuffer(np.asarray(biases).astype(np.int32).tobytes(), dtype=np.uint8).tolist()
        buffers.append({'data': data, 'offset': 0, 'size': 0})
        tensors.append({'shape': biases.shape,
                'type': 'INT32',
                'buffer': len(buffers)-1,
                'name': 'fc_biases_{}'.format(tensor_idx),
                'quantization': bias_quantization,
                'is_variable': False,
                'has_rank': True})

    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': output_shape,
            'type': 'INT8',
            'buffer': len(buffers)-1,
            'name': 'fc_{}'.format(tensor_idx),
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


# def remove_ops(operators, ops):
#     op_ids = sorted([operators.index(op) for op in ops])
#     for id in reversed(op_ids):
#         operators = operators[:id] + operators[id+1:]
#     return operators

def remove_ops(operators, op_ids):
    for id in reversed(sorted(op_ids)):
        operators = operators[:id] + operators[id+1:]
    return operators


def get_numpy_data(tensor, buffers):
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


def _minimax_affine_fit(x, y, m0):
    # L2 (lstsq) minimizes total squared error, but try_build_conv_uint8 only cares
    # about the worst-case (L-infinity) error against the 1-LSB tolerance, and an L2
    # fit can leave a larger max deviation than the best achievable line. The
    # Chebyshev max-error of a line, as a function of slope m alone (with the
    # intercept set to the midpoint of the residual range for that m), is convex in
    # m, so a ternary search around the lstsq slope m0 finds the minimax-optimal line.
    def residual_span(m):
        r = y - m * x
        return r.max() - r.min()

    lo = m0 - (abs(m0) + 1e-9) * 0.5
    hi = m0 + (abs(m0) + 1e-9) * 0.5
    for _ in range(200):
        if hi - lo < 1e-13 * (abs(m0) + 1e-9):
            break
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if residual_span(m1) < residual_span(m2):
            hi = m2
        else:
            lo = m1
    m = (lo + hi) / 2
    r = y - m * x
    b = (r.max() + r.min()) / 2
    return float(m), float(b)


def try_build_conv_uint8(tensors, buffers, builtin_codes, i_tensor, o_tensor, i_idx, output_data, c, lut_count):
    # Fit a per-channel affine map (weight=slope, bias=intercept) in real-value
    # space against the calibrated 0..255 -> quantized-output table, then quantize
    # it into a diagonal 1x1 CONV_2D ("conv_uint8"). Bail out (return None) if the
    # fit isn't exact, since the chain being replaced may not actually be affine.
    # op_conv() always builds an INT8 input/output tensor, so a cheap leading/trailing
    # dtype-cast QUANTIZE (not a LUT) is added on whichever side isn't already INT8.
    output_data = output_data.reshape(256, -1)
    if lut_count == 1 and output_data.shape[1] != c:
        output_data = np.tile(output_data[:, 0:1], (1, c))

    i_quant = i_tensor['quantization']
    i_scale = i_quant['scale'][0] if i_quant.get('scale') else 1.0
    i_zp = i_quant['zero_point'][0] if i_quant.get('zero_point') else 0

    o_quant = o_tensor['quantization']
    o_scales = o_quant['scale'] if o_quant.get('scale') else [1.0]
    o_zps = o_quant['zero_point'] if o_quant.get('zero_point') else [0]

    xs = np.arange(256, dtype=np.float64)
    real_x = i_scale * (xs - i_zp)
    A = np.vstack([real_x, np.ones_like(real_x)]).T

    weight_data = np.zeros((c, 1, 1, c), dtype=np.int64)
    bias_data = np.zeros((c,), dtype=np.int64)
    weight_scales, bias_scales = [], []

    for ch in range(c):
        o_scale = o_scales[ch] if len(o_scales) > 1 else o_scales[0]
        o_zp = o_zps[ch] if len(o_zps) > 1 else o_zps[0]

        y = output_data[:, ch].astype(np.float64)
        real_y = o_scale * (y - o_zp)

        m0, _ = np.linalg.lstsq(A, real_y, rcond=None)[0]
        m, b = _minimax_affine_fit(real_x, real_y, float(m0))

        weight_scale = max(abs(m), 1e-12) / 127.0
        q_weight = int(np.clip(round(m / weight_scale), -127, 127))
        real_weight = weight_scale * q_weight
        bias_scale = i_scale * weight_scale

        # The minimax fit picks the real-valued bias that centers the worst-case
        # error, but q_bias must land on an integer bias_scale step, and the final
        # output is itself rounded to an integer code, so the nearest integer bias
        # isn't guaranteed to be the one that minimizes the resulting max code error.
        # Search a small integer window around it and keep whichever reproduces the
        # calibrated table best.
        q_bias_center = round(b / bias_scale) if bias_scale > 0 else 0
        best_q_bias, best_err = 0, None
        for q_bias_candidate in range(q_bias_center - 2, q_bias_center + 3):
            real_bias = bias_scale * q_bias_candidate
            pred_real_y = real_weight * real_x + real_bias
            pred_q_y = np.round(pred_real_y / o_scale + o_zp)
            err = np.max(np.abs(pred_q_y - y))
            if best_err is None or err < best_err:
                best_err, best_q_bias = err, q_bias_candidate
        q_bias = best_q_bias

        # verify the quantized affine map reproduces the calibrated table exactly
        if best_err > 1:
            return None, tensors, buffers

        weight_data[ch, 0, 0, ch] = q_weight
        bias_data[ch] = q_bias
        weight_scales.append(weight_scale)
        bias_scales.append(bias_scale)

    weight_quant = {'scale': weight_scales, 'zero_point': [0]*c, 'details_type': 'NONE', 'quantized_dimension': 0}
    bias_quant = {'scale': bias_scales, 'zero_point': [0]*c, 'details_type': 'NONE', 'quantized_dimension': 0}

    ops = []
    conv_input_idx = i_idx
    if i_tensor['type'] != 'INT8':
        quantize_op, tensors, buffers = op_quantize(tensors, buffers, builtin_codes.index('QUANTIZE'), i_idx, False,
                                                     'INT8', [i_scale], [i_zp - 128])
        ops.append(quantize_op)
        conv_input_idx = quantize_op['outputs'][0]

    conv_opts = {'stride_h': 1, 'stride_w': 1, 'dilation_h_factor': 1, 'dilation_w_factor': 1, 'padding': 'VALID'}
    conv_output_type = o_tensor['type'] if o_tensor['type'] == 'INT8' else 'INT8'
    conv_output_shape = o_tensor['shape'].copy()
    if o_tensor['type'] == 'INT8':
        conv_output_quant = o_tensor['quantization']
    else:
        conv_output_quant = {'scale': o_scales, 'zero_point': [_ - 128 for _ in o_zps], 'details_type': 'NONE', 'quantized_dimension': 0}
    conv_op, tensors, buffers = op_conv(tensors, buffers, builtin_codes.index('CONV_2D'), conv_input_idx, False,
                                         'Conv2DOptions', conv_opts, conv_output_shape, conv_output_quant,
                                         weight_data, weight_quant, bias_data, bias_quant)
    # Mark this CONV_2D as synthesized during input preprocessing (replacing the
    # old quantize+cast+gather LUT), so downstream passes (split engine
    # assignment, IR construction) can identify it structurally via the op
    # itself rather than by tensor name. custom_options is schema-legal on any
    # operator (builtin or not) and survives the flatbuffer round-trip untouched
    # by the CONV_2D interpreter, which only reads builtin_options.
    conv_op['custom_options'] = [1]
    conv_op['custom_options_format'] = 'FLEXBUFFERS'
    ops.append(conv_op)

    if o_tensor['type'] != 'INT8':
        cast_op, tensors, buffers = op_quantize(tensors, buffers, builtin_codes.index('QUANTIZE'), conv_op['outputs'][0], False,
                                                 o_tensor['type'], o_scales, o_zps)
        ops.append(cast_op)

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


def replace_input(operators, buf, target_buf):
    for n,next_op in enumerate(operators):
        if buf in next_op['inputs']:
            next_op['inputs'] = [target_buf if _ == buf else _ for _ in next_op['inputs']]


def replace_output(operators, buf, target_buf):
    for p,prev_op in enumerate(operators):
        if buf in prev_op['outputs']:
            prev_op['outputs'] = [target_buf if _ == buf else _ for _ in prev_op['outputs']]
            

def replace_input(operators, buf, target_buf):
    for n,next_op in enumerate(operators):
        if buf in next_op['inputs']:
            next_op['inputs'] = [target_buf if _ == buf else _ for _ in next_op['inputs']]


def replace_output(operators, buf, target_buf):
    for p,prev_op in enumerate(operators):
        if buf in prev_op['outputs']:
            prev_op['outputs'] = [target_buf if _ == buf else _ for _ in prev_op['outputs']]


def create_reshape_op_for_add(input_tensor_idx, new_shape, tensors, buffers, opcodes, builtin_codes):
    """
    Create a RESHAPE operation to convert low-dim tensor to 4D for ADD operation.

    Args:
        input_tensor_idx: Index of input tensor to reshape
        new_shape: Target 4D shape (list of 4 integers)
        tensors: List of tensors in the graph
        buffers: List of buffers in the graph
        opcodes: List of opcodes
        builtin_codes: List of builtin operation codes

    Returns:
        (new_tensor_idx, reshape_op): Index of reshaped tensor and the RESHAPE operation dict
    """
    # Create new tensor for reshaped output
    original_tensor = tensors[input_tensor_idx]
    new_tensor = copy.deepcopy(original_tensor)
    new_tensor['shape'] = list(new_shape)
    new_tensor['name'] = original_tensor.get('name', f'tensor_{input_tensor_idx}') + '_reshaped'
    new_tensor_idx = len(tensors)
    tensors.append(new_tensor)

    # Create shape constant buffer (int32 array)
    shape_data = np.array(new_shape, dtype=np.int32)
    shape_buffer_data = shape_data.tobytes()
    # Convert to plain Python list of integers (not numpy types)
    shape_buffer = {'data': [int(x) for x in np.frombuffer(shape_buffer_data, dtype=np.uint8)]}
    shape_buffer_idx = len(buffers)
    buffers.append(shape_buffer)

    # Create shape tensor (contains target shape as constant)
    shape_tensor = {
        'shape': [len(new_shape)],
        'type': 'INT32',
        'buffer': shape_buffer_idx,
        'name': f'reshape_shape_{new_tensor_idx}',
        'quantization': {'scale': [], 'zero_point': [], 'min': [], 'max': []}
    }
    shape_tensor_idx = len(tensors)
    tensors.append(shape_tensor)

    # Find RESHAPE opcode index
    reshape_opcode_idx = None
    for idx, opcode in enumerate(opcodes):
        if opcode['builtin_code'] == 'RESHAPE':
            reshape_opcode_idx = idx
            break

    if reshape_opcode_idx is None:
        # Add RESHAPE opcode if not present
        opcodes.append({'deprecated_builtin_code': 22, 'version': 1, 'builtin_code': 'RESHAPE'})
        reshape_opcode_idx = len(opcodes) - 1
        builtin_codes.append('RESHAPE')

    # Create RESHAPE operation
    reshape_op = {
        'opcode_index': reshape_opcode_idx,
        'inputs': [input_tensor_idx, shape_tensor_idx],
        'outputs': [new_tensor_idx],
        'builtin_options_type': 'ReshapeOptions',
        'builtin_options': {'new_shape': list(new_shape)}
    }

    return new_tensor_idx, reshape_op


_concat_pad_axis_pos = {-4: 0, -3: 1, -2: 2, -1: 3}


def concat_pad_info(op, opcode, opts, tensors, buffers):
    # Detects a CONCATENATION that is really just zero(-point)-padding one side of
    # the H or W axis with a constant tensor, so it can be folded into a PADV2.
    if opcode != 'CONCATENATION' or opts is None or len(op['inputs']) != 2:
        return None
    axis = opts['axis']
    if axis not in (-2, -3):
        return None

    in0, in1 = op['inputs']
    const0 = 'data' in buffers[tensors[in0]['buffer']]
    const1 = 'data' in buffers[tensors[in1]['buffer']]
    if const0 == const1:
        return None

    const_idx, data_idx = (in0, in1) if const0 else (in1, in0)
    const_data = get_numpy_data(tensors[const_idx], buffers)
    if const_data.size == 0 or not np.all(const_data == const_data.flat[0]):
        return None
    const_value = const_data.flat[0].item()

    zp = tensors[data_idx]['quantization'].get('zero_point')
    if not zp or const_value != zp[-1]:
        return None

    return {'axis': axis, 'before': const0, 'size': tensors[const_idx]['shape'][axis],
            'tensor_idx': data_idx, 'const_value': const_value}


def apply_transformation(transform, operators, tensors, buffers, opcodes, builtin_codes, debug, compression_vbx=None):

    if transform in ['SORT_DFS', 'SORT_KHAN']:
        sort = transform.lower().split('_')[-1]

        operators, tensors, buffers = [_.copy() for _ in operators], [_.copy() for _ in tensors], buffers.copy()

        return clean_operators(operators, tensors, buffers, sort=sort)

    grab_op = lambda x: (operators[x].copy(), tensors[operators[x]['outputs'][0]]['quantization'].copy(), tensors[operators[x]['outputs'][0]]['type'],operators[x].get('builtin_options', None))
    wvals = lambda x: (get_numpy_data(tensors[x], buffers), tensors[x]['quantization'].copy(), tensors[x]['type'])

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
                        inputs, outputs, _ = generate_inputs_outputs(f.read(), int8_range=True)
                    output_data = outputs['o0'].squeeze()
                    if not tmp_dir_obj is None:
                        tmp_dir_obj.cleanup()
                    buffers.append({'data': np.frombuffer(output_data.tobytes(), dtype=np.uint8).tolist(), 'offset': 0, 'size': 0})
                    tensors[_op['outputs'][0]]['buffer'] = len(buffers)-1
                    for _ in reversed(pattern):
                        operators = operators[:_] + operators[_+1:]

        elif transform == 'REMOVE_NOPS':
            if opcode in ['RESHAPE']:
                ishape = tensors[op['inputs'][0]]['shape']
                oshape = tensors[op['outputs'][0]]['shape']
                if ishape == oshape:
                    replace_input(operators, op['outputs'][0], op['inputs'][0])
                    operators = operators[:i] + operators[i+1:]

        elif transform == 'CLEAN_LOGISTIC':
            if opcode in ['LOGISTIC']: #TODO and 'FLOAT32' in [tensors[_]['type'] for _ in op['inputs']]:
                if tensors[op['inputs'][0]]['type'] == 'INT8':
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
        elif transform == 'EXPAND_INPUT_DIMS':
                if is_graph_input(op, operators, tensors, buffers):
                    ishape = tensors[op['inputs'][0]]['shape']
                    oshape = tensors[op['inputs'][0]]['shape']
                    if len(ishape) == 3:
                        if ishape[0] != 1:
                            tensors[op['inputs'][0]]['shape'] = [1] + ishape
                        else:
                            tensors[op['inputs'][0]]['shape'] = ishape + [1]
                    if len(oshape) == 3:
                        if oshape[0] != 1:
                            tensors[op['outputs'][0]]['shape'] = [1] + oshape
                        else:
                            tensors[op['outputs'][0]]['shape'] = oshape + [1]

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

                    replace_input(operators, op['outputs'][0], inject_ops[-1]['outputs'][0])
                    operators = operators[:i] + inject_ops + operators[i+1:]

                    i += len(inject_ops)-1

        elif transform == 'SPLIT_XL_POOL':
            ishape = tensors[op['inputs'][0]]['shape']
            oshape = tensors[op['outputs'][0]]['shape']
            tensor_idx = op['inputs'][0]
            if opcode in ['MEAN'] and (ishape[-2] > 64 or ishape[-3] > 64):
                filter_sz = -1
                if ishape[-2] % 8 == 0 and ishape[-3] % 8 == 0:
                    filter_sz = 8
                elif ishape[-2] % 4 == 0 and ishape[-3] % 4 == 0:
                    filter_sz = 4
                elif ishape[-2] % 2 == 0 and ishape[-3] % 2 == 0:
                    filter_sz = 2

                if filter_sz > 0:
                    opts = {}
                    opts['filter_height'], opts['filter_width'] = filter_sz,filter_sz
                    opts['stride_h'], opts['stride_w'] = filter_sz,filter_sz
                    opts['padding'] = 'VALID'

                    pshape = oshape.copy()
                    pshape[-3] //= filter_sz
                    pshape[-2] //= filter_sz

                    inject_op, tensors, buffers = op_pool(tensors, buffers, builtin_codes.index('AVERAGE_POOL_2D'), tensor_idx, False, pshape, opts)
                    op['inputs'][0] = inject_op['outputs'][0]

                    # inject POOL and skip ahead 1
                    operators = operators[:i] + [inject_op] + operators[i:]
                    i += 1

        elif transform == 'MAXPOOL_INPUT_CHANNEL_SPLIT':
            if compression_vbx == 'ucomp' and opcode in ['MAX_POOL_2D']:
                AMM_CHANNELS_CAP = 1536
                in_t = tensors[op['inputs'][0]]
                out_t = tensors[op['outputs'][0]]
                in_c = in_t['shape'][-1]
                out_c = out_t['shape'][-1]
                in_w = in_t['shape'][2]  # NHWC: width
                x_slices = -(-in_w // 16)  # ceil(width / 16)
                out_w = out_t['shape'][2]  # NHWC: width
                out_x_slices = -(-out_w // 16)  # ceil(width / 16)
                sh, sw = opts['stride_h'], opts['stride_w']
                # Output channels are folded only for stride 1 with filter larger than 3.
                fold_in = 1 if (sh == 2 and sw == 2) else 0
                folded_in_c = in_c * (2 ** fold_in)
                if (x_slices == 1) and (fold_in == 1):
                    folded_in_c *= 2
                footprint = (folded_in_c * x_slices) + (out_c * out_x_slices)
                num_splits = max(1, -(-footprint // AMM_CHANNELS_CAP))
                # Even pieces only: round up to the next count that divides in_c into 16-channel-aligned pieces.
                while num_splits < in_c and (in_c % num_splits != 0 or (in_c // num_splits) % 16 != 0):
                    num_splits += 1
                if num_splits > 1 and in_c % num_splits == 0:
                    opcode_split = builtin_codes.index('SPLIT')
                    opcode_concat = builtin_codes.index('CONCATENATION')
                    opcode_pool = builtin_codes.index('MAX_POOL_2D')

                    # If the MaxPool input is a PAD used only by this MaxPool, split before the PAD so every
                    # split MaxPool keeps its own PAD.
                    pad_idx = None
                    pad_op_pos = None
                    producer = [k for k, o in enumerate(operators[:i]) if op['inputs'][0] in o['outputs']]
                    if producer and builtin_codes[operators[producer[-1]]['opcode_index']] == 'PAD':
                        pad_op_pos = producer[-1]
                        pad_op = operators[pad_op_pos]
                        consumers = [k for k, o in enumerate(operators) if op['inputs'][0] in o['inputs']]
                        if consumers == [i]:
                            pad_idx = pad_op_pos

                    if pad_idx is not None:
                        pad_op = operators[pad_idx]
                        pre_tensor = pad_op['inputs'][0]
                        pad_const = pad_op['inputs'][2:] if len(pad_op['inputs']) > 2 else []
                        split_op, tensors, buffers = op_split(tensors, buffers, opcode_split, pre_tensor, False, -1, num_splits)
                        inject_ops = [split_op]
                        piece_inputs = split_op['outputs']
                    else:
                        split_op, tensors, buffers = op_split(tensors, buffers, opcode_split, op['inputs'][0], False, -1, num_splits)
                        inject_ops = [split_op]
                        piece_inputs = split_op['outputs']

                    pool_outs = []
                    piece_shape = out_t['shape'].copy()
                    piece_shape[-1] = out_c // num_splits
                    for n in range(num_splits):
                        piece_in = piece_inputs[n]
                        if pad_idx is not None:
                            # Re-create this piece's PAD with the original pad spec, then pool it.
                            pad_spec = list(pad_op['inputs'])
                            piece_pad_in_shape = tensors[piece_in]['shape'].copy()
                            pre_piece_shape = piece_pad_in_shape
                            pad_out_shape = [pre_piece_shape[k] for k in range(len(pre_piece_shape))]
                            pad_dim_tensor = tensors[pad_op['inputs'][1]]
                            pad_vals = get_numpy_data(pad_dim_tensor, buffers).reshape(-1).tolist()
                            pad_out_shape = [pre_piece_shape[k] + pad_vals[2*k] + pad_vals[2*k+1] for k in range(len(pre_piece_shape))]
                            buffers.append({'offset': 0, 'size': 0})
                            tensors.append({'shape': pad_out_shape, 'type': tensors[pre_tensor]['type'],
                                            'buffer': len(buffers)-1, 'name': 'pad_piece_{}_{}'.format(pad_idx, n),
                                            'quantization': tensors[pre_tensor]['quantization'].copy(),
                                            'is_variable': False, 'has_rank': True})
                            pad_out_idx = len(tensors)-1
                            pad_piece_op = {'opcode_index': pad_op['opcode_index'],
                                            'inputs': [piece_in, pad_op['inputs'][1]] + pad_const,
                                            'outputs': [pad_out_idx]}
                            inject_ops.append(pad_piece_op)
                            piece_in = pad_out_idx
                        pool_op, tensors, buffers = op_pool(tensors, buffers, opcode_pool, piece_in, False,
                                                            piece_shape, opts.copy())
                        inject_ops.append(pool_op)
                        pool_outs.append(pool_op['outputs'][0])

                    # If the MaxPool output feeds a LUT pattern, run that pattern on each piece after its MaxPool
                    # and concat the per-piece LUT outputs into the pattern's final output instead.
                    lut_after = []
                    if i + 1 < len(operators):
                        lut_after, _ = lut_pattern(operators, builtin_codes, tensors, buffers, i + 1, compression_vbx)
                    if len(lut_after):
                        # The trailing pattern is the LUT chain (QUANTIZE/CAST/...); its per-channel table is
                        # sliced to each piece's channel range, and op_lut is applied to each piece tensor.
                        last_pat_op = operators[i + 1 + lut_after[-1]]
                        final_out = last_pat_op['outputs'][0]
                        pat_ops = [operators[i + 1 + k] for k in lut_after]
                        pat_out_tensor = tensors[final_out]
                        out_c_total = pat_out_tensor['shape'][-1]
                        pattern_inputs = [_ for _ in pat_ops[0]['inputs'] if not 'data' in buffers[tensors[_]['buffer']]]
                        lut_table = np.asarray(pattern_lut_table(pat_ops, tensors, buffers, opcodes, out_c_total, pattern_count))
                        lut_pieces = []
                        piece_c = out_c_total // num_splits
                        for n_piece, piece_out in enumerate(pool_outs):
                            table = lut_table[:, n_piece*piece_c:(n_piece+1)*piece_c] if lut_table.ndim == 2 else lut_table
                            piece_pat_out = copy.deepcopy(tensors[final_out])
                            piece_pat_out['shape'] = list(piece_pat_out['shape'][:-1]) + [piece_c]
                            piece_pat_out['name'] = '{}_piece{}'.format(final_out, n_piece)
                            tensors.append(piece_pat_out)
                            piece_out_idx = len(tensors)-1
                            lut_ops, tensors, buffers = op_lut(
                                tensors=tensors, buffers=buffers,
                                opcode_quantize=builtin_codes.index('QUANTIZE'),
                                opcode_cast=builtin_codes.index('CAST'),
                                opcode_gather=builtin_codes.index('GATHER'),
                                opcode_split=builtin_codes.index('SPLIT'),
                                opcode_concat=builtin_codes.index('CONCATENATION'),
                                output_data=table, i_idx=piece_out, o_idx=piece_out_idx)
                            inject_ops.extend(lut_ops)
                            lut_pieces.append(piece_out_idx)
                        concat_op, tensors, buffers = op_concat(tensors, buffers, opcode_concat, lut_pieces, final_out, -1)
                        # op_concat allocates a new output tensor; rebind it to final_out so consumers stay wired.
                        concat_op['outputs'] = [final_out]
                        inject_ops.append(concat_op)
                        consumed = set(range(i + 1, i + 2 + lut_after[-1]))
                    else:
                        concat_op, tensors, buffers = op_concat(tensors, buffers, opcode_concat, pool_outs, op['outputs'][0], -1)
                        # Keep the original MaxPool output id so its existing consumers remain connected.
                        concat_op['outputs'] = [op['outputs'][0]]
                        inject_ops.append(concat_op)
                        consumed = set()

                    # Replace the MaxPool (at i) with inject_ops, drop the consumed trailing ops and the
                    # PAD (if any) in a single rebuild, so indices are computed once against the original list.
                    drop = set(consumed)
                    if pad_idx is not None:
                        drop.add(pad_idx)
                    new_ops = []
                    for k, o in enumerate(operators):
                        if k == i:
                            new_ops.extend(inject_ops)
                        elif k in drop:
                            continue
                        else:
                            new_ops.append(o)
                    removed_before_i = len([k for k in drop if k < i])
                    operators = new_ops
                    i = i - removed_before_i + len(inject_ops) - 1

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

                    max_pool_op0, tensors, buffers = op_pool(tensors, buffers, builtin_codes.index('MAX_POOL_2D'), tensor_idx, False, oshape, opts3.copy())
                    inject_ops.append(max_pool_op0)

                    # second 3x3
                    tensor_idx = len(tensors)-1
                    oshape = oshape.copy()
                    if opts3['padding'] == 'VALID':
                        oshape[-3] -= 2
                        oshape[-2] -= 2
                    max_pool_op1, tensors, buffers = op_pool(tensors, buffers, builtin_codes.index('MAX_POOL_2D'), tensor_idx, False, oshape, opts3.copy())
                    inject_ops.append(max_pool_op1)

                    if (fw,fh) == (7,7): # potential third 3x3
                        tensor_idx = len(tensors)-1
                        oshape = oshape.copy()
                        if opts3['padding'] == 'VALID':
                            oshape[-3] -= 2
                            oshape[-2] -= 2
                        max_pool_op2, tensors, buffers = op_pool(tensors, buffers, builtin_codes.index('MAX_POOL_2D'), tensor_idx, False, oshape, opts3.copy())
                        inject_ops.append(max_pool_op2)


                    replace_input(operators, op['outputs'][0], inject_ops[-1]['outputs'][0])
                    operators = operators[:i] + inject_ops + operators[i+1:]

                    i += len(inject_ops)-1

        elif transform == 'NCHW_RESHAPE':
            if opcode in ['RESHAPE'] and channel_order(i, operators, tensors, buffers) == "NHWC":
                ishape = tensors[op['inputs'][0]]['shape']
                shape = get_numpy_data(tensors[op['inputs'][1]], buffers)

                iarr = np.random.randint(0, 10, ishape)
                oarr = np.reshape(iarr, shape)

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

                reshape_nchw_op, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), tensor_idx, False, _shape)
                inject_op.append(reshape_nchw_op)
                tensor_idx = len(tensors)-1

                transpose_nhwc_op, tensors, buffers = op_transpose(tensors, buffers, builtin_codes.index('TRANSPOSE'), tensor_idx, False, [0,2,3,1])
                inject_op.append(transpose_nhwc_op)
                tensor_idx = len(tensors)-1

                replace_input(operators, op['outputs'][0], inject_ops[-1]['outputs'][0])
                operators = operators[:i] + inject_ops + operators[i+1:]

        elif transform == 'LUT':
            pattern, lut_count = lut_pattern(operators, builtin_codes, tensors, buffers, i, compression_vbx)

            if len(pattern):
                assert pattern[0] == i, "LUT match must start at the current operator"
                # Derive the boundary from the complete pattern. In a SiLU
                # pattern, MUL consumes both the original activation and the
                # LOGISTIC result, but only the former is external.
                pattern_produced = {
                    tensor
                    for pattern_idx in pattern
                    for tensor in operators[pattern_idx]['outputs']
                    if tensor != -1
                }
                pattern_consumed = {
                    tensor
                    for pattern_idx in pattern
                    for tensor in operators[pattern_idx]['inputs']
                    if tensor != -1
                }
                pattern_inputs = []
                pattern_outputs = []
                for pattern_idx in pattern:
                    for tensor in operators[pattern_idx]['inputs']:
                        if tensor != -1 and tensor not in pattern_produced:
                            if 'data' not in buffers[tensors[tensor]['buffer']]:
                                pattern_inputs.append(tensor)
                    for tensor in operators[pattern_idx]['outputs']:
                        if tensor != -1 and tensor not in pattern_consumed:
                            if 'data' not in buffers[tensors[tensor]['buffer']]:
                                pattern_outputs.append(tensor)

                pattern_inputs = list(dict.fromkeys(pattern_inputs))
                pattern_outputs = list(dict.fromkeys(pattern_outputs))
                if len(pattern_inputs) != 1 or len(pattern_outputs) != 1:
                    raise ValueError(
                        "LUT pattern must have one external input and output: "
                        f"{pattern} -> {pattern_inputs} -> {pattern_outputs}"
                    )

                i_tensor = tensors[pattern_inputs[0]]
                o_tensor = tensors[pattern_outputs[0]]
                c = i_tensor['shape'][-1]
                pattern_opcodes = [builtin_codes[operators[idx]['opcode_index']] for idx in pattern]

                # A lone QUANTIZE whose input is already INT8 is just the fixed raw
                # int8<->uint8 byte-shift (i - 128 for i in 0..255); no need to
                # simulate it, the table is known in closed form.
                if compression_vbx == 'ucomp' and pattern_opcodes == ["QUANTIZE"] and i_tensor['type'] == 'UINT8':
                    output_data = np.tile((np.arange(256) - 128).reshape(256, 1), (1, max(lut_count, 1)))
                else:
                    tmp_dir_obj = tempfile.TemporaryDirectory()
                    graph = create_graph([operators[_] for _ in pattern], tensors, buffers, opcodes, force_shape=[1,1,256,lut_count], clean_graph=True)
                    save_graph('pattern.{}.tflite'.format(pattern_count), tmp_dir_obj, graph, copy=debug)

                    with open(os.path.join(tmp_dir_obj.name, 'pattern.{}.tflite'.format(pattern_count)), 'rb') as f:
                        inputs, outputs, _ = generate_inputs_outputs(f.read(), int8_range=True)
                    output_data = outputs['o0'].squeeze()
                    if not tmp_dir_obj is None:
                        tmp_dir_obj.cleanup()

                # A uint8 model input feeding a LUT (whether matched from raw
                # QUANTIZE/ADD/SUB/MUL/DIV ops, or from an already-materialized
                # quantize+cast+(split+)gather(+concat) sequence) may still be a pure
                # per-channel affine remap (no LUT nonlinearity needed), so on ucomp
                # it's worth trying to express it as a 1x1 CONV_2D ("conv_uint8")
                # instead of the split/gather/concat sequence hardware handles poorly.
                # try_build_conv_uint8 does its own affine-fit verification and
                # returns None (falling back to op_lut below) if the map isn't exact.
                conv_ops = None
                if (compression_vbx == 'ucomp' and i_tensor['type'] in ('UINT8', 'INT8')
                        and is_graph_input(op, operators, tensors, buffers)
                        and lut_count in (1, c)):
                    conv_ops, tensors, buffers = try_build_conv_uint8(tensors, buffers, builtin_codes, i_tensor, o_tensor,
                                                                       pattern_inputs[0], output_data, c, lut_count)

                if conv_ops is not None:
                    # replace input w/ conv input
                    tensors[pattern_inputs[0]] = tensors[conv_ops[0]['inputs'][0]].copy()
                    conv_ops[0]['inputs'][0] = pattern_inputs[0]

                    # replace output w/ conv output
                    tensors[pattern_outputs[0]] = tensors[conv_ops[-1]['outputs'][0]].copy()
                    conv_ops[-1]['outputs'][0] = pattern_outputs[0]

                    for idx in reversed(sorted(pattern)):
                        operators = operators[:idx] + operators[idx+1:]
                    operators = operators[:i] + conv_ops + operators[i:]

                    pattern_count += 1
                    i += len(conv_ops) - 1

                else:
                    lut_ops, tensors, buffers = op_lut(
                    tensors         = tensors, 
                    buffers         = buffers, 
                    opcode_quantize = builtin_codes.index('QUANTIZE'), 
                    opcode_cast     = builtin_codes.index('CAST'), 
                    opcode_gather   = builtin_codes.index('GATHER'), 
                    opcode_split    = builtin_codes.index('SPLIT'), 
                    opcode_concat   = builtin_codes.index('CONCATENATION'), 
                    output_data     = output_data, 
                    i_idx           = pattern_inputs[0], 
                    o_idx           = pattern_outputs[0])


                    # replace input w/ lut input
                    tensors[pattern_inputs[0]] = tensors[lut_ops[0]['inputs'][0]].copy()
                    lut_ops[0]['inputs'][0] = pattern_inputs[0]

                    # replace output w/ lut output
                    tensors[pattern_outputs[0]] = tensors[lut_ops[-1]['outputs'][0]].copy()
                    lut_ops[-1]['outputs'][0] = pattern_outputs[0]

                    for idx in reversed(sorted(pattern)):
                        operators = operators[:idx] + operators[idx+1:]
                    operators = operators[:i] + lut_ops + operators[i:]

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
                    oshape[-1] = weights.shape[0]
                    conv_opts['padding'] = 'VALID'

                    conv_op, tensors, buffers = op_conv(tensors, buffers, builtin_codes.index('CONV_2D'), tensor_idx, False, 'Conv2DOptions', conv_opts,
                                                        oshape, oquant,
                                                        weights, weight_quant,
                                                        biases, bias_quant)
                    ops.append(conv_op)

                    replace_input(operators, op['outputs'][0], conv_op['outputs'][0])
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
                    replace_input(operators, op['outputs'][0], inject_ops[-1]['outputs'][0])
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
                    replace_input(operators, op['outputs'][0], inject_ops[-1]['outputs'][0])
                    operators = operators[:i] + inject_ops + operators[i+1:]

                    i += len(inject_ops)-1
        elif transform == 'FUSE_GROUP1':
            if opcode in ['SPLIT']:
                f_tensor = tensors[op['inputs'][0]]
                filter_data = get_numpy_data(f_tensor, buffers)
                if filter_data == [-1]:
                    next_ops = []
                    next_next_ops = []
                    for n,next_op in enumerate(operators):
                        if next_op['inputs'][0] in op['outputs']:
                            next_ops.append(next_op)
                            for nn,next_next_op in enumerate(operators):
                                if next_op['outputs'][0] in next_next_op['inputs']:
                                    if not next_next_op in next_next_ops: 
                                        next_next_ops.append(next_next_op)

                    if len(next_ops) and all([builtin_codes[_['opcode_index']] in ['CONV_2D'] for _ in next_ops]):
                        weights = [get_numpy_data(tensors[_['inputs'][1]], buffers) for _ in next_ops]
                        biases = [get_numpy_data(tensors[_['inputs'][2]], buffers) for _ in next_ops]
                        kh,kw = weights[0].shape[1], weights[0].shape[2]
                        nopts = next_ops[0]['builtin_options']
                        if all([_.shape[-1] == 1 and _.shape[0] == 1 for _ in weights]):
                            tensor_idx = op['inputs'][1]
                            inject_ops = []
                            num_groups = len(next_next_ops)
                            num_splits = opts['num_splits'] // (len(next_ops) // num_groups)

                            if num_groups >= 1:
                                if num_splits > 1:
                                    split_op, tensors, buffers = op_split(tensors, buffers, builtin_codes.index('SPLIT'), tensor_idx, False, -1, num_splits)
                                    inject_ops.append(split_op)

                                for o in range(num_groups):
                                    if num_splits > 1:
                                        tensor_idx = split_op['outputs'][o]
                                    concat_op = next_next_ops[o]
                                    oshape = tensors[concat_op['outputs'][0]]['shape'].copy()
                                    oquant = tensors[concat_op['inputs'][0]]['quantization'].copy()
                                    dconv_opts = {'depth_multiplier': 1, 'stride_h': 1, 'stride_w': 1, 'dilation_h_factor': 1, 'dilation_w_factor': 1, 'padding': nopts['padding']}
                                    k = len(concat_op['inputs'])
                                    dweights = np.zeros((1,kh,kw,k), dtype=np.int64)
                                    weight_quant = {'scale': [1. for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 3}
                                    bias_data = np.zeros((k,), dtype=np.int64)
                                    bquant = {'scale': [1. for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 0}
                                    for n,idx in enumerate(concat_op['inputs']):
                                        prev_ops = []
                                        for p,prev_op in enumerate(operators):
                                            if idx in prev_op['outputs']:
                                                prev_ops.append(p)
                                        assert(len(prev_ops) == 1)
                                        conv_op = operators[prev_ops[0]]
                                        weight_quant['scale'][n] = tensors[conv_op['inputs'][1]]['quantization']['scale'][0]
                                        weight_quant['zero_point'][n] = tensors[conv_op['inputs'][1]]['quantization']['zero_point'][0]
                                        bquant['scale'][n] = tensors[conv_op['inputs'][2]]['quantization']['scale'][0]
                                        bquant['zero_point'][n] = tensors[conv_op['inputs'][2]]['quantization']['zero_point'][0]
                                        dweights[0,:,:,n:n+1] = get_numpy_data(tensors[conv_op['inputs'][1]], buffers)
                                        bias_data[n:n+1] = get_numpy_data(tensors[conv_op['inputs'][2]], buffers)



                                
                                    conv_op, tensors, buffers = op_conv(tensors, buffers, builtin_codes.index('DEPTHWISE_CONV_2D'), tensor_idx, False, 'DepthwiseConv2DOptions', dconv_opts,
                                                                        oshape, oquant,
                                                                        dweights, weight_quant,
                                                                        bias_data, bquant)
                                    inject_ops.append(conv_op)

                                    replace_input(operators, concat_op['outputs'][0], conv_op['outputs'][0])

                                remove_op_ids = []
                                for n,next_op in enumerate(operators):
                                    if next_op in next_ops or next_op in next_next_ops:
                                        if not n in remove_op_ids:
                                            remove_op_ids.append(n)
                                for idx in reversed(sorted(remove_op_ids)):
                                    operators = operators[:idx] + operators[idx+1:]

                                operators = operators[:i] + inject_ops + operators[i+1:]

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
                elif opts['filter_height'] == 3 and opts['filter_width'] == 3:
                    kh, kw = 3, 3
                    w_val = 127

                    if pool_count % 2: # swap between w_scale that rounds down vs rounds up int8 values
                        w_scale = 0.000875 #1/127/(kh*kw)
                    else:
                        w_scale = 0.000874
                    pool_count += 1

                    ishape = tensors[op['inputs'][0]]['shape']
                    k = ishape[-1]

                    oshape = tensors[op['outputs'][0]]['shape'].copy()
                    oquant = tensors[op['outputs'][0]]['quantization'].copy()

                    dconv_opts = {'depth_multiplier': 1, 'stride_h': opts['stride_h'], 'stride_w': opts['stride_w'], 'dilation_h_factor': 1, 'dilation_w_factor': 1, 'padding': opts['padding']}
                    # uniform 3x3 kernel (all positions non-zero) so this is never mistaken for the
                    # bottom-right-2x2-only pattern that frontend_hardware_dependent.py's is_avgPool
                    # detection matches against
                    weights = np.zeros((1,kh,kw,k), dtype=np.int64)
                    weights[0,:,:,:] += w_val

                    weight_quant = {'scale': [w_scale for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 3}

                    bias_data = np.zeros((k,), dtype=np.int64)
                    bquant = {'scale': [1. for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 0}

                    conv_op, tensors, buffers = op_conv(tensors, buffers, builtin_codes.index('DEPTHWISE_CONV_2D'), op['inputs'][0], False, 'DepthwiseConv2DOptions', dconv_opts,
                                                        oshape, oquant,
                                                        weights, weight_quant,
                                                        bias_data, bquant)
                    inject_ops.append(conv_op)

            if len(inject_ops):
                replace_input(operators, op['outputs'][0], inject_ops[-1]['outputs'][0])
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
                    replace_input(operators, op['outputs'][0], inject_op['outputs'][0])
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
                    replace_input(operators, op['outputs'][0], split_op['outputs'][split_idx])

        elif transform == 'UNPACK_SLICE': 
            if opcode in ['UNPACK']:
                next_ops = []
                active_outputs = []
                for o, _op in enumerate(operators):
                    if len([_ for _ in _op['inputs'] if _ in op['outputs']]):
                           next_ops.append(o)
                           active_outputs += [o for o,_ in enumerate(op['outputs']) if _ in _op['inputs']]
                if len(next_ops) == 1:
                    axis = opts['axis']

                    ishape = tensors[op['inputs'][0]]['shape']
                    begin = [0 for _ in range(len(ishape))]
                    end = ishape.copy()
                    stride = [1 for _ in range(len(ishape))]

                    begin[axis] = active_outputs[0]
                    end[axis] = active_outputs[0] + 1
                    split_op, tensors, buffers = op_strided_slice(tensors, buffers, builtin_codes.index('STRIDED_SLICE'), op['inputs'][0], False, begin, end, stride)

                    # split_op, tensors, buffers = op_split_v(tensors, buffers, builtin_codes.index('SPLIT_V'), op['inputs'][0], False, axis, [1 for _ in range(opts['num'])])

                    operators = operators[:i] + [split_op] + operators[i+1:]
                    replace_input(operators, op['outputs'][0], split_op['outputs'][0])



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

        elif transform == 'CONCAT_PAD':
            info1 = concat_pad_info(op, opcode, opts, tensors, buffers)
            if info1 is not None:
                forked, _ = is_forked(op, operators, tensors)
                next_idx = get_next_op(operators, i)
                if not forked and next_idx is not None:
                    op2 = operators[next_idx]
                    opcode2 = builtin_codes[op2['opcode_index']]
                    opts2 = op2.get('builtin_options', None)
                    info2 = concat_pad_info(op2, opcode2, opts2, tensors, buffers)
                    if (info2 is not None and info2['tensor_idx'] == op['outputs'][0]
                            and info2['axis'] != info1['axis']
                            and info2['const_value'] == info1['const_value']):
                        pad = [0 for _ in range(8)]
                        for info in (info1, info2):
                            pos = _concat_pad_axis_pos[info['axis']]
                            pad[pos*2 + (0 if info['before'] else 1)] = info['size']

                        inject_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PAD'),
                                                              info1['tensor_idx'], False, pad, info1['const_value'])
                        inject_op['outputs'][0] = op2['outputs'][0]

                        operators = operators[:i] + [inject_op] + operators[i+1:next_idx] + operators[next_idx+1:]

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
                        inject_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PADV2'), tensor_idx, True, pad, constant_value)

                        # adjust previous op outputs, to go to PAD instead of CONV
                        replace_output(operators, tensor_idx, inject_op['inputs'][0])

                        # set NEXT who share padding to VALID
                        for s in shared_indices:
                            operators[s]['builtin_options']['padding'] = 'VALID'
                    else:
                        # create PAD
                        inject_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index('PADV2'), tensor_idx, False, pad, constant_value)
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
        elif transform == 'PAD_TO_MULTIPLE_OF_4':
            # UCOMP only. If the graph input H and W are both odd and the first stride-2 CONV_2D is within the first
            # 6 ops, insert a PAD between the graph input and the first op so the conv sees an even-sized input.
            # Pad multiple: 2 when the conv kernel is <= 5, otherwise 4. Runs once, at op index 0.
            if compression_vbx == 'ucomp' and i == 0:
                graph_input_idx = next((t for t in range(len(tensors)) if t in operators[0]['inputs']
                                        and is_graph_input({'inputs': [t]}, operators, tensors, buffers)), None)
                first_conv_idx = next((k for k, _ in enumerate(operators[:6])
                                       if builtin_codes[_['opcode_index']] == 'CONV_2D'
                                       and _['builtin_options'].get('stride_h') == 2
                                       and _['builtin_options'].get('stride_w') == 2), None)
                if graph_input_idx is not None and first_conv_idx is not None:
                    ishape = tensors[graph_input_idx]['shape']
                    i_h, i_w = ishape[-3], ishape[-2]
                    if i_h % 2 == 1 and i_w % 2 == 1:
                        # Conv kernel from its weight tensor, layout [out, kh, kw, in].
                        _, kh, kw, _ = tensors[operators[first_conv_idx]['inputs'][1]]['shape']
                        kernel = max(kh, kw)
                        pad_multiple = 2 if kernel <= 5 else 4
                        new_h = -(-i_h // pad_multiple) * pad_multiple
                        new_w = -(-i_w // pad_multiple) * pad_multiple
                        if (new_h, new_w) != (i_h, i_w):
                            pad = [0,0, 0, new_h - i_h, 0, new_w - i_w, 0,0]
                            zp = tensors[graph_input_idx]['quantization']['zero_point']
                            constant_value = zp[-1] if zp else 0
                            # PAD fills with zero; a non-zero zero-point needs PADV2 with the constant as its input.
                            pad_opcode = 'PADV2' if constant_value else 'PAD'
                            inject_op, tensors, buffers = op_pad(tensors, buffers, builtin_codes.index(pad_opcode), graph_input_idx,
                                                                 False, pad, constant_value if constant_value else None)
                            # Every op that read the graph input now reads the padded tensor.
                            for _op in operators:
                                _op['inputs'] = [inject_op['outputs'][0] if _ == graph_input_idx else _ for _ in _op['inputs']]
                            # The conv reads the padded input with VALID padding.
                            operators[first_conv_idx]['builtin_options']['padding'] = 'VALID'
                            padded_shape = tensors[inject_op['outputs'][0]]['shape']
                            padded_shape[-3], padded_shape[-2] = new_h, new_w
                            # The PAD sits at the front, directly between the graph input and the first op.
                            operators = [inject_op] + operators
        elif transform == 'EXPAND_BROADCAST':
            if opcode in ['MUL', 'ADD', 'SUB']:
                t0 = tensors[op['inputs'][0]]
                t1 = tensors[op['inputs'][1]]
                if 'data' in buffers[t0['buffer']] or 'data' in buffers[t1['buffer']]:
                    if 'data' in buffers[t0['buffer']]:
                        f_tensor = t0
                        itensor = t1
                        tensor_idx = op['inputs'][1]
                        non_data_idx = 1
                    else:
                        f_tensor = t1
                        itensor = t0
                        tensor_idx = op['inputs'][0]
                        non_data_idx = 0
                    if any([a>b for a,b in zip(f_tensor['shape'], itensor['shape'])]):
                        count = [(i,a//b) for i,(a,b) in enumerate(zip(f_tensor['shape'], itensor['shape'])) if (a > b) and i > 0]
                        if len(count) >= 1:
                            axis, values_count = count[0]
                            concat_inputs = [op['inputs'][0] for _ in range(values_count)]
                            concat_op, tensors, buffers = op_concat(tensors, buffers, builtin_codes.index('CONCATENATION'), concat_inputs, tensor_idx, axis)
                            op['inputs'][non_data_idx] = concat_op['outputs'][0]
                            operators = operators[:i] + [concat_op] + operators[i:]
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
                        replace_input(operators, op['outputs'][0], conv_op['outputs'][0])
                        operators = operators[:i] + [conv_op] + operators[i+1:]

        elif transform == 'POOL_AS_MEAN':
            if opcode in ['AVERAGE_POOL_2D']:
                oshape = tensors[op['outputs'][0]]['shape']
                ishape = tensors[op['inputs'][0]]['shape']
                if opts['filter_height'] >= ishape[-3] and opts['filter_width'] >= ishape[-2]:
                    if oshape[-2] == 1 and oshape[-3] == 1:
                        tensor_idx = op['inputs'][0]
                        mean_op, tensors, buffers = op_mean(tensors, buffers, builtin_codes.index('MEAN'), tensor_idx, False, [1,2])
                        replace_input(operators, op['outputs'][0], mean_op['outputs'][0])
                        operators = operators[:i] + [mean_op] + operators[i+1:]

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
                    replace_input(operators, op['outputs'][0], add_op['outputs'][0])
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

        elif transform == 'NORMALIZE_ADD_INPUTS':
            # Normalize ADD inputs to 4D and broadcast constants to full output shape for ucomp
            # This avoids broadcasting issues in the compiler
            if opcode in ['ADD']:
                def normalize_shape_to_4d(shape):
                    """
                    Normalize tensor shape to 4D NHWC format.

                    Rules:
                      1D: [C]           -> [1, 1, 1, C]   (Batch=1, H=1, W=1, Channels=C)
                      2D: [H, C]        -> [1, H, 1, C]   (Batch=1, Height=H, W=1, Channels=C)
                      3D: [B, H, C]     -> [B, H, 1, C]   (Batch=B, Height=H, W=1, Channels=C)
                      4D: [B, H, W, C]  -> [B, H, W, C]   (unchanged)

                    Key insight: Width (W) is inserted BEFORE the last dimension (Channels).
                    """
                    if len(shape) == 1:
                        # [C] -> [1, 1, 1, C]
                        return [1, 1, 1, shape[0]]
                    elif len(shape) == 2:
                        # [H, C] -> [1, H, 1, C]
                        return [1, shape[0], 1, shape[1]]
                    elif len(shape) == 3:
                        # [B, H, C] -> [B, H, 1, C]
                        return [shape[0], shape[1], 1, shape[2]]
                    elif len(shape) == 4:
                        # Already 4D
                        return shape
                    else:
                        # 5D or higher - shouldn't happen, return as-is
                        return shape

                t0 = tensors[op['inputs'][0]]
                t1 = tensors[op['inputs'][1]]
                t_out = tensors[op['outputs'][0]]

                shape0 = t0['shape']
                shape1 = t1['shape']
                output_shape = t_out['shape']

                # Normalize output shape to 4D if needed
                if len(output_shape) < 4:
                    normalized_output = normalize_shape_to_4d(output_shape)
                    t_out['shape'] = normalized_output
                    output_shape = normalized_output

                # CRITICAL: output_shape is in NHWC format here, but will be transposed to NCHW later
                # For broadcasting to work correctly after transpose, we need to:
                # 1. Convert output_shape from NHWC to what it will be in NCHW
                # 2. Broadcast constants in NCHW space
                # 3. Store the result (which will be transposed back to match)
                #
                # NHWC [B,H,W,C] -> NCHW [B,C,H,W] via transpose (0,3,1,2)
                output_shape_nchw = [output_shape[0], output_shape[3], output_shape[1], output_shape[2]]

                # Track if we need to insert RESHAPE ops
                reshape_ops_to_insert = []

                # Process both inputs
                input_tensor_indices = [op['inputs'][0], op['inputs'][1]]
                for idx, tensor_idx in enumerate(input_tensor_indices):
                    tensor = tensors[tensor_idx]
                    shape = tensor['shape']

                    # Check if this is a constant tensor that needs processing
                    is_constant = 'data' in buffers[tensor['buffer']]

                    # Normalize shape to 4D if needed, OR if it's already 4D constant but needs broadcasting
                    if len(shape) < 4:
                        # Normalize shape to 4D
                        normalized_shape = normalize_shape_to_4d(shape)
                        needs_processing = True
                    elif len(shape) == 4 and is_constant:
                        # Already 4D, but check if constant needs broadcasting
                        # Convert to NCHW to check against output
                        shape_nchw = [shape[0], shape[3], shape[1], shape[2]]
                        if shape_nchw != output_shape_nchw:
                            # 4D constant that needs broadcasting (e.g., [1,1,1,256] -> [1,80,80,256])
                            normalized_shape = shape
                            needs_processing = True
                        else:
                            needs_processing = False
                    else:
                        needs_processing = False

                    if needs_processing:
                        # Check if this is a constant tensor
                        if is_constant:
                            # Constant tensor: reshape data and broadcast if needed
                            old_data = get_numpy_data(tensor, buffers)

                            # Reshape to normalized 4D NHWC (or keep if already 4D)
                            reshaped_data = old_data.reshape(normalized_shape)

                            # Convert normalized shape to what it will be in NCHW after transpose
                            normalized_shape_nchw = [normalized_shape[0], normalized_shape[3], normalized_shape[1], normalized_shape[2]]

                            # Check if broadcasting is needed in NCHW space
                            if normalized_shape_nchw != output_shape_nchw:
                                try:
                                    # Transpose data to NCHW for broadcasting
                                    data_nchw = np.transpose(reshaped_data, (0, 3, 1, 2))

                                    # Broadcast in NCHW space
                                    broadcasted_nchw = np.broadcast_to(data_nchw, output_shape_nchw)

                                    # Transpose back to NHWC for storage
                                    broadcasted_nhwc = np.transpose(broadcasted_nchw, (0, 2, 3, 1))

                                    # Update tensor shape and buffer data (shape stored in NHWC)
                                    tensor['shape'] = list(broadcasted_nhwc.shape)
                                    data_bytes = np.frombuffer(broadcasted_nhwc.tobytes(), dtype=np.uint8).tolist()
                                    buffers[tensor['buffer']]['data'] = data_bytes
                                except Exception as e:
                                    # If broadcasting fails, just normalize to 4D without broadcasting
                                    tensor['shape'] = normalized_shape
                                    data_bytes = np.frombuffer(reshaped_data.tobytes(), dtype=np.uint8).tolist()
                                    buffers[tensor['buffer']]['data'] = data_bytes
                            else:
                                # Already matches output shape after normalization
                                tensor['shape'] = normalized_shape
                                data_bytes = np.frombuffer(reshaped_data.tobytes(), dtype=np.uint8).tolist()
                                buffers[tensor['buffer']]['data'] = data_bytes
                        else:
                            # Non-constant tensor (model input): insert RESHAPE operation

                            # Create RESHAPE operation
                            new_tensor_idx, reshape_op = create_reshape_op_for_add(
                                tensor_idx, normalized_shape, tensors, buffers, opcodes, builtin_codes)

                            # Queue RESHAPE op to be inserted before current ADD op
                            reshape_ops_to_insert.append(reshape_op)

                            # Update ADD operation to use reshaped tensor
                            op['inputs'][idx] = new_tensor_idx

                # Insert queued RESHAPE operations BEFORE current ADD op
                if reshape_ops_to_insert:
                    # Insert in reverse order so indices don't shift
                    for reshape_op in reversed(reshape_ops_to_insert):
                        operators.insert(i, reshape_op)

        elif transform == 'FUSE_CONV':

            if opcode in ['CONV_2D']:
                _next_ops = [_ for _ in range(len(operators)) if _op['outputs'][0] in operators[_]['inputs']]
                if len(_next_ops) == 1:
                    next_opcode = builtin_codes[_next_ops[0]['opcode_index']]
                    if next_opcode in ['CONV_2D']:
                        pass


        elif transform in ['FC_CONV_2D', 'FC2_CONV_2D']:
            if opcode in ['FULLY_CONNECTED'] and not is_multi_input(op, tensors, buffers): #could be batch mat mul
                ops = []
                ishape = tensors[op['inputs'][0]]['shape'].copy()
                oshape = tensors[op['outputs'][0]]['shape'].copy()
                oquant = tensors[op['outputs'][0]]['quantization']
                tensor_idx = op['inputs'][0]

                f_tensor = tensors[op['inputs'][1]]
                filter_data = get_numpy_data(f_tensor, buffers)
                weight_quant = f_tensor['quantization'].copy()
                k,c = filter_data.shape[0], filter_data.shape[1]

                is_2D = ishape[-2] > 1
                is_2K = c > 1024
                if (is_2D and transform == 'FC_CONV_2D') or (not is_2K and transform == 'FC2_CONV_2D'):
                    i += 1
                    continue

                b_tensor = None
                bias_data = np.zeros((k,), dtype=np.int64)
                bias_quant = {'scale': [1. for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 0}
                if len(op['inputs']) > 2 and op['inputs'][2] != -1:
                    b_tensor = tensors[op['inputs'][2]]
                    bias_data = get_numpy_data(b_tensor, buffers)
                    bias_quant = b_tensor['quantization'].copy()


                weight_data = np.zeros((k,1,1,c), dtype=np.int64)
                for k_ in range(k):
                    for c_ in range(c):
                        weight_data[k_,0,0,c_] = filter_data[k_,c_]

                conv_opts = {'stride_h': 1, 'stride_w': 1, 'dilation_h_factor': 1, 'dilation_w_factor': 1, 'padding': 'VALID'}
                if 'fused_activation_function' in opts:
                    conv_opts['fused_activation_function'] = opts['fused_activation_function']

                if not is_2D:
                    _reshape = [1,1,1,c]
                    reshape_op, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), tensor_idx, False, _reshape)

                    ops.append(reshape_op)
                    tensor_idx = len(tensors)-1

                    _oshape = [1,1,1,oshape[-1]]
                    conv_op, tensors, buffers = op_conv(tensors, buffers, builtin_codes.index('CONV_2D'), tensor_idx, False, 'Conv2DOptions', conv_opts,
                                                        _oshape, oquant.copy(),
                                                        weight_data, weight_quant,
                                                        bias_data, bias_quant)
                    ops.append(conv_op)
                    tensor_idx = len(tensors)-1

                    _reshape1 = [1,k]
                    reshape_op1, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), tensor_idx, False, _reshape1)

                    ops.append(reshape_op1)
                    tensor_idx = len(tensors)-1

                else:

                    _reshape = [1,1,ishape[-2],c]
                    reshape_op, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), tensor_idx, False, _reshape)

                    ops.append(reshape_op)
                    tensor_idx = len(tensors)-1

                    _oshape = [1,1,oshape[-2],oshape[-1]]
                    conv_op, tensors, buffers = op_conv(tensors, buffers, builtin_codes.index('CONV_2D'), tensor_idx, False, 'Conv2DOptions', conv_opts,
                                                        _oshape, oquant.copy(),
                                                        weight_data, weight_quant,
                                                        bias_data, bias_quant)
                    ops.append(conv_op)
                    tensor_idx = len(tensors)-1

                    _reshape1 = oshape.copy()
                    _reshape1 = [oshape[-2],oshape[-1]] #TODO
                    reshape_op1, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), tensor_idx, False, _reshape1)

                    ops.append(reshape_op1)
                    tensor_idx = len(tensors)-1


                # adjust
                replace_input(operators, op['outputs'][0], ops[-1]['outputs'][0])
                operators = operators[:i] + ops + operators[i+1:]
                i += len(ops) - 1

        elif transform == 'REWRITE_RESHAPE':
            is_pattern, pattern = find_pattern(operators[i:], tensors, buffers, builtin_codes, 'RESHAPE')
            if is_pattern:
                ops = []
                tensor_idx = op['inputs'][0]
                ishape = tensors[op['inputs'][0]]['shape']
                b,n,h,w,c = ishape


                for code, shape in [('R', [1,n,h*w,c]), ('T', [0,2,3,1]), ('R',[1,h,w,c*n]), ('T',[0,1,3,2]), ('R', [1,n,h,w*c])]:
                    if code == 'R':
                        op, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), tensor_idx, False, shape)
                    else:
                        op, tensors, buffers = op_transpose(tensors, buffers, builtin_codes.index('TRANSPOSE'), tensor_idx, False, shape)

                    ops.append(op)
                    tensor_idx = len(tensors)-1

                replace_input(operators, operators[i+len(pattern)-1]['outputs'][0], ops[-1]['outputs'][0]) 
                operators = operators[:i] + ops + operators[i+len(pattern):]
                i += len(ops) - 1

        elif transform == 'SPNV2_TRANSPOSE':
            is_pattern, pattern = find_pattern(operators[i:], tensors, buffers, builtin_codes, 'SPNV2_TRANSPOSE')
            if is_pattern:
                ops = []
                t0_op, _, _, _ = grab_op(i+0)
                reshape_op, reshape_quant, _, _ = grab_op(i+1)
                mul0_op, _, _, _ = grab_op(i+2)
                assert('data' in buffers[tensors[mul0_op['inputs'][1]]['buffer']])
                if 'ADD' in pattern:
                    add_op, _, _, _ = grab_op(i+3)
                    assert('data' in buffers[tensors[add_op['inputs'][1]]['buffer']])
                    logi_op, _, _, _ = grab_op(i+4)
                    mul1_op, _, _, _ = grab_op(i+5)
                    t1_op, _, _, _ = grab_op(i+6)
                    ops = [t0_op, reshape_op, t1_op, mul0_op, add_op, logi_op, mul1_op]
                else:
                    logi_op, _, _, _ = grab_op(i+3)
                    mul1_op, _, _, _ = grab_op(i+4)
                    t1_op, _, _, _ = grab_op(i+5)
                    ops = [t0_op, reshape_op, t1_op, mul0_op, logi_op, mul1_op]

            
                # replace output tensor w/ reshaped tensor
                for op in ops[3:]:
                    tensors, t_idx = create_transposed_tensor(tensors, op['outputs'][0], (0,2,3,1))
                    op['outputs'][0] = t_idx

                # replace input data tensor w/ reshaped tensor
                tensors, t_idx = create_transposed_tensor(tensors, mul0_op['inputs'][1], (0,2,3,1))
                mul0_op['inputs'][1] = t_idx
                if 'ADD' in pattern:
                    tensors, t_idx = create_transposed_tensor(tensors, add_op['inputs'][1], (0,2,3,1))
                    add_op['inputs'][1] = t_idx

                # reassign connections to new tensors
                replace_input(operators, t1_op['outputs'][0], mul1_op['outputs'][0])
                mul0_op['inputs'][0] = t1_op['outputs'][0]
                if 'ADD' in pattern:
                    add_op['inputs'][0] = mul0_op['outputs'][0]
                    logi_op['inputs'][0] = add_op['outputs'][0]
                    mul1_op['inputs'][0] = logi_op['outputs'][0]
                    mul1_op['inputs'][1] = add_op['outputs'][0]
                else:
                    logi_op['inputs'][0] = mul0_op['outputs'][0]
                    mul1_op['inputs'][0] = logi_op['outputs'][0]
                    mul1_op['inputs'][1] = mul0_op['outputs'][0]

                t1_op['inputs'][0] = reshape_op['outputs'][0]
                tensors[t1_op['outputs'][0]]['quantization'] = reshape_quant.copy()
                operators = operators[:i] + ops + operators[i+len(pattern):]


        elif transform == 'REWRITE_NORM':
            is_pattern, pattern = find_pattern(operators[i:], tensors, buffers, builtin_codes, 'NORM')
            if is_pattern:
                ops = []
                tensor_idx = op['inputs'][0]
                ishape = tensors[op['inputs'][0]]['shape']
                itype = tensors[op['inputs'][0]]['type']
                oshape = tensors[op['outputs'][0]]['shape']
                otype = tensors[op['outputs'][0]]['type']
                num_splits = ishape[-1] // oshape[-1]
                i_quant = tensors[op['inputs'][0]]['quantization']

                mean0_op, mean0_quant, mean0_type, mean0_opts = grab_op(i+1)
                sub0_op, sub0_quant, sub0_type, sub0_opts = grab_op(i+2)
                mul0_op, mul0_quant, mul0_type, mul0_opts = grab_op(i+3)
                mean1_op, mean1_quant, mean1_type, mean1_opts = grab_op(i+4)
                add0_op, add0_quant, add0_type, add0_opts = grab_op(i+5)
                dequant_op, dequant_quant, dequant_type, dequant_opts = grab_op(i+6)
                sqrt_op, sqrt_quant, sqrt_type, sqrt_opts = grab_op(i+7)
                div_op, div_quant, div_type, div_opts = grab_op(i+8)
                quant_op, quant_quant, quant_type, quant_opts = grab_op(i+9)
                mul1_op, mul1_quant, mul1_type, mul1_opts = grab_op(i+10)
                mul2_op, mul2_quant, mul2_type, mul2_opts = grab_op(i+11)
                mul3_op, mul3_quant, mul3_type, mul3_opts = grab_op(i+12)
                sub1_op, sub1_quant, sub1_type, sub1_opts = grab_op(i+13)
                if len(pattern) == 16:
                    add1_op, add1_quant, add1_type, add1_opts = grab_op(i+14)
                    final_op, final_quant, final_type, final_opts = grab_op(i+15)

                    sub1_val, sub1_fquant, sub1_ftype = wvals(sub1_op['inputs'][0])
                else:
                    final_op, final_quant, final_type, final_opts = grab_op(i+14)

                add0_val, add0_fquant, add0_ftype = wvals(add0_op['inputs'][1])
                div_val, div_fquant, div_ftype = wvals(div_op['inputs'][0])
                mul1_val, mul1_fquant, mul1_ftype = wvals(mul1_op['inputs'][1])

                concat_inputs = []

                split_op, tensors, buffers = op_split(tensors, buffers, builtin_codes.index('SPLIT'), tensor_idx, False, -1, num_splits)
                ops.append(split_op)

            
                for n in range(num_splits):
                    mean0_op, tensors, buffers = op_mean(tensors, buffers, builtin_codes.index('MEAN'), split_op['outputs'][n], False, [1,2,3])
                    tensors[mean0_op['outputs'][0]]['quantization'] = mean0_quant.copy()

                    ops.append(mean0_op)

                    tensor_idx = split_op['outputs'][n]
                    tensor2_idx = mean0_op['outputs'][0]
                    sub0_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('SUB'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], sub0_quant.copy(), 'SubOptions', sub0_opts)
                    ops.append(sub0_op)

                    tensor_idx = sub0_op['outputs'][0]
                    tensor2_idx = sub0_op['outputs'][0]
                    mul0_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('MUL'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], mul0_quant.copy())
                    ops.append(mul0_op)

                    mean1_op, tensors, buffers = op_mean(tensors, buffers, builtin_codes.index('MEAN'), mul0_op['outputs'][0], False, [1,2,3])
                    tensors[mean1_op['outputs'][0]]['quantization'] = mean1_quant.copy()
                    ops.append(mean1_op)

                    add0_val = add0_val.reshape((1,1,1,1))
                    data = np.frombuffer(add0_val.tobytes(), dtype=np.uint8).tolist()
                    add0_op, tensors, buffers = op_add(tensors, buffers, builtin_codes.index('ADD'), mean1_op['outputs'][0], False,
                           add0_quant['scale'], add0_quant['zero_point'],
                           data, add0_val.shape, add0_ftype, add0_fquant['scale'], add0_fquant['zero_point'])
                    add0_op['builtin_options']['pot_scale_int16'] = add0_opts['pot_scale_int16']
                    ops.append(add0_op)


                    d_op, tensors, buffers = op_type(tensors, buffers, builtin_codes.index('DEQUANTIZE'), add0_op['outputs'][0], False, dequant_type, dequant_quant.copy())
                    ops.append(d_op)


                    sqrt_op, tensors, buffers = op_type(tensors, buffers, builtin_codes.index('SQRT'), d_op['outputs'][0], False, sqrt_type, sqrt_quant.copy())
                    ops.append(sqrt_op)

                    data = np.frombuffer(div_val.tobytes(), dtype=np.uint8).tolist()
                    div_op, tensors, buffers = op_div(tensors, buffers, builtin_codes.index('DIV'), sqrt_op['outputs'][0], False, True, div_quant.copy(), data, div_val.shape, div_ftype, div_fquant.copy())
                    ops.append(div_op)

                    q_op, tensors, buffers = op_type(tensors, buffers, builtin_codes.index('QUANTIZE'), div_op['outputs'][0], False, quant_type, quant_quant.copy())
                    ops.append(q_op)

                    val = mul1_val[:,:,:,n,:].reshape([1,1,1,-1])
                    data = np.frombuffer(val.tobytes(), dtype=np.uint8).tolist()

                    mul1_op, tensors, buffers = op_mul(tensors, buffers, builtin_codes.index('MUL'), q_op['outputs'][0], False,
                           mul1_quant['scale'], mul1_quant['zero_point'],
                           data, val.shape, mul1_ftype, mul1_fquant['scale'], mul1_fquant['zero_point'])
                    ops.append(mul1_op)

                    tensor_idx = split_op['outputs'][n]
                    tensor2_idx = mul1_op['outputs'][0]

                    mul2_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('MUL'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], mul2_quant.copy())
                    ops.append(mul2_op)

                    tensor_idx = mul1_op['outputs'][0]
                    tensor2_idx = mean0_op['outputs'][0]

                    mul3_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('MUL'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], mul3_quant.copy())
                    ops.append(mul3_op)


                    if len(pattern) == 16:
                        val = sub1_val[:,:,:,n,:].reshape([1,1,1,-1])
                        data = np.frombuffer(val.tobytes(), dtype=np.uint8).tolist()

                        sub1_op, tensors, buffers = op_sub(tensors, buffers, builtin_codes.index('SUB'), mul3_op['outputs'][0], False, True,
                               sub1_quant['scale'], sub1_quant['zero_point'],
                               data, val.shape, sub1_ftype, sub1_fquant['scale'], sub1_fquant['zero_point'])
                        ops.append(sub1_op)

                        tensor_idx = mul2_op['outputs'][0]
                        tensor2_idx = sub1_op['outputs'][0]
                        add1_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('ADD'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], add1_quant.copy(), 'AddOptions', add1_opts.copy())
                        ops.append(add1_op)
                        concat_inputs.append(add1_op['outputs'][0])
                    else:
                        tensor_idx = mul2_op['outputs'][0]
                        tensor2_idx = mul3_op['outputs'][0]
                        sub1_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('SUB'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], sub1_quant.copy(), 'SubOptions', sub1_opts.copy())
                        ops.append(sub1_op)
                        concat_inputs.append(sub1_op['outputs'][0])

                concat_op, tensors, buffers = op_concat(tensors, buffers, builtin_codes.index('CONCATENATION'), concat_inputs, tensor_idx, -1)
                ops.append(concat_op)

                replace_input(operators, final_op['outputs'][0], ops[-1]['outputs'][0])

                tensors[ops[-1]['outputs'][0]]['quantization'] = tensors[ops[-2]['outputs'][0]]['quantization'].copy()

                operators = operators[:i] + ops + operators[i+len(pattern)-1+1:]
                i += len(ops) - 1

            is_pattern, pattern = find_pattern(operators[i:], tensors, buffers, builtin_codes, 'SPNV2_NORM')
            if is_pattern:
                ops = []
                tensor_idx = op['inputs'][0]
                ishape = tensors[op['inputs'][0]]['shape']
                itype = tensors[op['inputs'][0]]['type']
                oshape = tensors[op['outputs'][0]]['shape']
                otype = tensors[op['outputs'][0]]['type']
                reshape_op, _, _, _ = grab_op(i+1)
                reshape_val, _, _ = wvals(reshape_op['inputs'][1])
                num_splits = int(reshape_val[1])
                i_quant = tensors[op['inputs'][0]]['quantization']


                mean0_op, mean0_quant, mean0_type, mean0_opts = grab_op(i+3)
                diff0_op, diff0_quant, diff0_type, diff0_opts = grab_op(i+4)
                mean1_op, mean1_quant, mean1_type, mean1_opts = grab_op(i+5)
                add0_op, add0_quant, add0_type, add0_opts = grab_op(i+6)
                add0_val, add0_fquant, add0_ftype = wvals(add0_op['inputs'][1])
                rsqrt_op, rsqrt_quant, rsqrt_type, rsqrt_opts = grab_op(i+7)
                sub0_op, sub0_quant, sub0_type, sub0_opts = grab_op(i+8)
                mul0_op, mul0_quant, mul0_type, mul0_opts = grab_op(i+9)

                final_op, final_quant, final_type, final_opts = grab_op(i+len(pattern)-1)

                if True: # alternative to splitting 10x is batch 10x
                    concat_inputs = []
                    split_op, tensors, buffers = op_split(tensors, buffers, builtin_codes.index('SPLIT'), tensor_idx, False, -1, num_splits)
                    ops.append(split_op)

                
                    for n in range(num_splits):
                        mean0_op, tensors, buffers = op_mean(tensors, buffers, builtin_codes.index('MEAN'), split_op['outputs'][n], False, [1,2,3])
                        tensors[mean0_op['outputs'][0]]['quantization'] = mean0_quant.copy()
                        ops.append(mean0_op)

                        tensor_idx = split_op['outputs'][n]
                        tensor2_idx = mean0_op['outputs'][0]
                        diff_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('SQUARED_DIFFERENCE'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], diff0_quant.copy(), 'SquaredDifferenceOptions', diff0_opts)
                        ops.append(diff_op)

                        mean1_op, tensors, buffers = op_mean(tensors, buffers, builtin_codes.index('MEAN'), diff_op['outputs'][0], False, [1,2,3])
                        tensors[mean1_op['outputs'][0]]['quantization'] = mean1_quant.copy()
                        ops.append(mean1_op)

                        add0_val = add0_val.reshape((1,1,1,1))

                        data = np.frombuffer(add0_val.tobytes(), dtype=np.uint8).tolist()
                        add0_op, tensors, buffers = op_add(tensors, buffers, builtin_codes.index('ADD'), mean1_op['outputs'][0], False,
                               add0_quant['scale'], add0_quant['zero_point'],
                               data, add0_val.shape, add0_ftype, add0_fquant['scale'], add0_fquant['zero_point'])
                        add0_op['builtin_options']['pot_scale_int16'] = add0_opts['pot_scale_int16']
                        ops.append(add0_op)


                        rsqrt_op, tensors, buffers = op_type(tensors, buffers, builtin_codes.index('RSQRT'), add0_op['outputs'][0], False, rsqrt_type, rsqrt_quant.copy())
                        ops.append(rsqrt_op)


                        tensor_idx = split_op['outputs'][n]
                        tensor2_idx = mean0_op['outputs'][0]
                        sub_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('SUB'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], sub0_quant.copy(), 'SubOptions', sub0_opts)
                        ops.append(sub_op)

                        tensor_idx = rsqrt_op['outputs'][0]
                        tensor2_idx = sub_op['outputs'][0]

                        mul_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('MUL'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], mul0_quant.copy())
                        ops.append(mul_op)
                        concat_inputs.append(mul_op['outputs'][0])

                    concat_op, tensors, buffers = op_concat(tensors, buffers, builtin_codes.index('CONCATENATION'), concat_inputs, tensor_idx, -1)
                    ops.append(concat_op)


                    replace_input(operators, final_op['outputs'][0], ops[-1]['outputs'][0])
                    tensors[ops[-1]['outputs'][0]]['quantization'] = tensors[ops[-2]['outputs'][0]]['quantization'].copy()

                    operators = operators[:i] + ops + operators[i+len(pattern)-1+1:]
                    i += len(ops) - 1
                else:
                    split_op, tensors, buffers = op_split(tensors, buffers, builtin_codes.index('SPLIT'), tensor_idx, False, -1, num_splits)
                    ops.append(split_op)

                    tensor_idx = split_op['outputs'][0]
                    concat_shape = tensors[tensor_idx]['shape'].copy()
                    concat_shape[-4] = num_splits

                    concat_op, tensors, buffers = op_concat(tensors, buffers, builtin_codes.index('CONCATENATION'), split_op['outputs'], tensor_idx, -4, force_shape=concat_shape)

                    ops.append(concat_op)

                    mean0_op, tensors, buffers = op_mean(tensors, buffers, builtin_codes.index('MEAN'), concat_op['outputs'][0], False, [1,2,3])
                    tensors[mean0_op['outputs'][0]]['quantization'] = mean0_quant.copy()
                    ops.append(mean0_op)

                    tensor_idx = concat_op['outputs'][0]
                    tensor2_idx = mean0_op['outputs'][0]
                    diff_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('SQUARED_DIFFERENCE'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], diff0_quant.copy(), 'SquaredDifferenceOptions', diff0_opts)
                    ops.append(diff_op)

                    mean1_op, tensors, buffers = op_mean(tensors, buffers, builtin_codes.index('MEAN'), diff_op['outputs'][0], False, [1,2,3])
                    tensors[mean1_op['outputs'][0]]['quantization'] = mean1_quant.copy()
                    ops.append(mean1_op)

                    add0_val = add0_val.reshape((1,1,1,1))

                    data = np.frombuffer(add0_val.tobytes(), dtype=np.uint8).tolist()
                    add0_op, tensors, buffers = op_add(tensors, buffers, builtin_codes.index('ADD'), mean1_op['outputs'][0], False,
                           add0_quant['scale'], add0_quant['zero_point'],
                           data, add0_val.shape, add0_ftype, add0_fquant['scale'], add0_fquant['zero_point'])
                    add0_op['builtin_options']['pot_scale_int16'] = add0_opts['pot_scale_int16']
                    ops.append(add0_op)


                    rsqrt_op, tensors, buffers = op_type(tensors, buffers, builtin_codes.index('RSQRT'), add0_op['outputs'][0], False, rsqrt_type, rsqrt_quant.copy())
                    ops.append(rsqrt_op)


                    tensor_idx = concat_op['outputs'][0]
                    tensor2_idx = mean0_op['outputs'][0]
                    sub_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('SUB'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], sub0_quant.copy(), 'SubOptions', sub0_opts)
                    ops.append(sub_op)

                    tensor2_idx = rsqrt_op['outputs'][0]
                    tensor_idx = sub_op['outputs'][0]

                    mul_op, tensors, buffers = op_elemwise(tensors, buffers, builtin_codes.index('MUL'), tensor_idx, tensor2_idx, tensors[tensor_idx]['shape'].copy(), tensors[tensor_idx]['type'], mul0_quant.copy())
                    ops.append(mul_op)

                    tensor_idx = mul_op['outputs'][0]

                    split_op, tensors, buffers = op_split(tensors, buffers, builtin_codes.index('SPLIT'), tensor_idx, False, -4, num_splits)
                    ops.append(split_op)

                    tensor_idx = split_op['outputs'][0]
                    concat_shape = tensors[tensor_idx]['shape'].copy()
                    concat_shape[-1] *= num_splits

                    concat_op, tensors, buffers = op_concat(tensors, buffers, builtin_codes.index('CONCATENATION'), split_op['outputs'], tensor_idx, -1, force_shape=concat_shape)

                    ops.append(concat_op)


                    replace_input(operators, final_op['outputs'][0], ops[-1]['outputs'][0])
                    tensors[ops[-1]['outputs'][0]]['quantization'] = tensors[ops[-2]['outputs'][0]]['quantization'].copy()

                    operators = operators[:i] + ops + operators[i+len(pattern)-1+1:]
                    i += len(ops) - 1

        elif transform == 'Y26':
            p = ["TRANSPOSE", "BATCH_MATMUL", "MUL", "TRANSPOSE", "SOFTMAX"]
            p2 = ["TRANSPOSE", "STRIDED_SLICE", "BATCH_MATMUL","TRANSPOSE", "RESHAPE"]
            if len(operators[i:i+len(p)]) == len(p) and [builtin_codes[_['opcode_index']] for _ in operators[i:i+len(p)]] == p:
                t0_op = operators[i+0]
                batch_op = operators[i+1]
                mul_op = operators[i+2]
                t1_op = operators[i+3]
                soft_op = operators[i+4]


                batch_op['inputs'][1] = t0_op['inputs'][0]
                mul_op['outputs'][0] = soft_op['inputs'][0]
                tensors[batch_op['outputs'][0]]['shape'] = tensors[soft_op['outputs'][0]]['shape']

                ops = [batch_op, mul_op, soft_op]
                operators = operators[:i] + ops + operators[i+len(p):]
                i += len(ops) - 1
        
            elif len(operators[i:i+len(p2)]) == len(p2) and [builtin_codes[_['opcode_index']] for _ in operators[i:i+len(p2)]] == p2:
                    t0_op = operators[i+0]
                    slice_op = operators[i+1]
                    batch_op = operators[i+2]
                    t1_op = operators[i+3]
                    reshape_op = operators[i+4]

                    batch_op['inputs'][1] = t0_op['inputs'][0]
                    batch_op['outputs'][0] = reshape_op['inputs'][0]
                    tensors[batch_op['outputs'][0]]['shape'] = tensors[reshape_op['outputs'][0]]['shape']

                    ops = [slice_op, batch_op, reshape_op]
                    operators = operators[:i] + ops + operators[i+len(p2):]
                    i += len(ops) - 1

        elif transform == 'FUSE_ADJ':
            if opcode in ["TRANSPOSE"]:
                ni = get_next_op(operators, i)
                if ni and builtin_codes[operators[ni]['opcode_index']] == "BATCH_MATMUL":
                    batch_op = operators[ni]
                    if batch_op['inputs'][0] == i: 
                        batch_op['inputs'][0] = op['inputs'][0]
                        batch_op['builtin_options']['adj_x'] = not batch_op['builtin_options']['adj_x'] 
                    else:
                        batch_op['inputs'][1] = op['inputs'][0]
                        batch_op['builtin_options']['adj_y'] = not batch_op['builtin_options']['adj_y'] 
                    operators = operators[:i] + operators[i+1:]
                    i -= 1

        elif transform == 'FUSE_ADD': #TODO
            if opcode in ["CONV_2D", "FULLY_CONNECTED"]:
                ni = get_next_op(operators, i)
                if ni and builtin_codes[operators[ni]['opcode_index']] == "ADD" and not is_multi_input(operators[ni], tensors, buffers):

                    m_op, m_quant, m_type, m_opts = grab_op(i)
                    add_op, add_quant, add_type, add_opts = grab_op(ni)
                    # add_val, add_quant, add_type = wvals(add_op['inputs'][1])

        elif transform == 'FUSE_SLICES':
            forked, next_ids = is_forked(op, operators, tensors)
            if forked and len(next_ids):
                next_ops = [operators[_] for _ in next_ids]


                def all_slices(ops, tensors, buffers, codes):
                    if all([codes[_['opcode_index']] == "STRIDED_SLICE" for _ in ops]):
                        begin = [get_numpy_data(tensors[_['inputs'][1]], buffers) for _ in ops]
                        end = [get_numpy_data(tensors[_['inputs'][2]], buffers) for _ in ops]
                        strides = [get_numpy_data(tensors[_['inputs'][3]], buffers) for _ in ops]
                        dims = begin[0].size
                        axis = np.argmax([np.sum([b[i] for b in begin]) for i in range(dims)])
                        axis = axis - dims

                        splits = [(end[i]-begin[i])[axis] for i,_ in enumerate(ops)]
                        indices = np.argsort([begin[i][axis] for i,_ in enumerate(ops)])
                        ishape = tensors[ops[-1]['inputs'][0]]['shape']

                        for s,split in enumerate(splits):
                            if split < 0:
                                splits[s] += ishape[axis]

                        if not all([_ > 0 for _ in splits]):
                            return None, None, None
                        if all([np.all(np.ones(_.shape) == _) for _ in strides]):
                            return splits, indices, axis
                    return None, None, None

                splits, indices, axis = all_slices(next_ops, tensors, buffers, builtin_codes)

                if not splits is None:
                    splits = [splits[i] for i in indices]
                    next_ops = [next_ops[i] for i in indices]

                    split_op, tensors, buffers = op_split_v(tensors, buffers, builtin_codes.index('SPLIT_V'), op['outputs'][0], False, axis, splits)
                    for o,_op in enumerate(next_ops):
                        replace_input(operators, _op['outputs'][0], split_op['outputs'][o])

                    operators = remove_ops(operators, next_ids)
                    operators = operators[:i+1] + [split_op] + operators[i+1:]

        elif transform == 'FRAC_FC':
            if opcode in ['FULLY_CONNECTED']:
                forked, _ = is_forked(op, operators, tensors)
                if not forked:
                    remove_op_ids = []
                    nidx = get_next_op(operators, i)
                    if nidx:
                        next_op = operators[nidx]
                        next_opcode = builtin_codes[next_op['opcode_index']]
                        remove_op_ids.append(nidx)

                        has_add = False
                        if next_opcode == 'ADD':
                            has_add = True
                            add_op, add_quant, add_type, add_opts = grab_op(nidx)
                            nnidx = get_next_op(operators, nidx)
                            if nnidx:
                                next_op = operators[nnidx]
                                next_opcode = builtin_codes[next_op['opcode_index']]
                                remove_op_ids.append(nnidx)

                        if next_opcode == 'SPLIT_V':
                            num_splits = next_op['builtin_options']['num_splits']
                            split_tensor = tensors[next_op['inputs'][1]]
                            splits = get_numpy_data(split_tensor, buffers)

                            f_tensor = tensors[op['inputs'][1]]
                            filter_data = get_numpy_data(f_tensor, buffers)
                            weight_quant = f_tensor['quantization'].copy()
                            k,c = filter_data.shape[0], filter_data.shape[1]

                            b_tensor = None
                            bias_data = np.zeros((k,), dtype=np.int64)
                            bias_quant = {'scale': [1. for _ in range(k)], 'zero_point': [0 for _ in range(k)], 'details_type': 'NONE', 'quantized_dimension': 3}
                            if len(op['inputs']) > 2 and op['inputs'][2] != -1:
                                b_tensor = tensors[op['inputs'][2]]
                                bias_data = get_numpy_data(b_tensor, buffers)
                                bias_quant = b_tensor['quantization'].copy()

                            oquant = tensors[op['outputs'][0]]['quantization']

                            if has_add:
                                add_val, add_fquant, add_ftype = wvals(add_op['inputs'][1])
                            
                            inject_ops = []
                            offset = 0
                            for s,split in enumerate(splits):
                                o = next_op['outputs'][s]
                                oshape = tensors[o]['shape']
                                sweights = filter_data[offset:offset+split]
                                wquant = weight_quant.copy()
                                if len(wquant['scale']) > 1:
                                    wquant['scale'] = wquant['scale'][offset:offset+split]
                                    wquant['zero_point'] = wquant['zero_point'][offset:offset+split]
                                sbias_data = None
                                bquant = None
                                if not b_tensor is None:
                                    sbias_data = bias_data[offset:offset+split]
                                    bquant = bias_quant.copy()
                                    if len(bquant['scale']) > 1:
                                        bquant['scale'] = bquant['scale'][offset:offset+split]
                                        bquant['zero_point'] = bquant['zero_point'][offset:offset+split]

                                fc_op, tensors, buffers = op_fully_connected(tensors, buffers, builtin_codes.index('FULLY_CONNECTED'), op['inputs'][0], False, 'FullyConnectedOptions', opts,
                                                                        oshape, oquant,
                                                                        sweights, wquant,
                                                                        sbias_data, bquant)
                                inject_ops.append(fc_op)
                                if has_add:
                                    _add_val = add_val[offset:offset+split]
                                    data = np.frombuffer(_add_val.tobytes(), dtype=np.uint8).tolist()

                                    _add_op, tensors, buffers = op_add(tensors, buffers, builtin_codes.index('ADD'), fc_op['outputs'][0], False, add_quant['scale'], add_quant['zero_point'], data, _add_val.shape, add_ftype, add_fquant['scale'], add_fquant['zero_point'])
                                    _add_op['builtin_options']['pot_scale_int16'] = add_opts['pot_scale_int16']
                                    inject_ops.append(_add_op)


                                replace_input(operators, o, inject_ops[-1]['outputs'][0])
                                offset += split

                            operators = remove_ops(operators, remove_op_ids)
                            operators = operators[:i] + inject_ops + operators[i+1:]
                            i -= len(remove_op_ids)

        elif transform == 'FRAC_BATCH':
            attn_chain = find_chain(operators[i:], tensors, buffers, builtin_codes, ['BATCH_MATMUL', 'MUL', 'SOFTMAX', 'BATCH_MATMUL'])
            attn_chain_add = find_chain(operators[i:], tensors, buffers, builtin_codes, ['BATCH_MATMUL', 'MUL', 'SOFTMAX','ADD', 'BATCH_MATMUL'])
            attn_chain_quant = find_chain(operators[i:], tensors, buffers, builtin_codes, ['BATCH_MATMUL', 'MUL', 'SOFTMAX','QUANTIZE', 'BATCH_MATMUL'])

            if attn_chain or attn_chain_add or attn_chain_quant:
                if attn_chain:
                    attn_chain = [i+_ for _ in attn_chain]
                elif attn_chain_add:
                    attn_chain = [i+_ for _ in attn_chain_add]
                elif attn_chain_quant:
                    attn_chain = [i+_ for _ in attn_chain_quant]

                b1_op = operators[attn_chain[0]]
                mul_op = operators[attn_chain[1]]
                soft_op = operators[attn_chain[2]]
                if attn_chain_add or attn_chain_quant:
                    extra_op = operators[attn_chain[3]]
                    b2_op = operators[attn_chain[4]]
                else:
                    extra_op = None
                    b2_op = operators[attn_chain[3]]

                attn_heads = tensors[b1_op['inputs'][0]]['shape'][-3]
                if attn_heads > 1:
                    allowed_codes = ['TRANSPOSE', 'RESHAPE']
                    ops_0 = get_prev_thread(b1_op['inputs'][0], operators, tensors, buffers, builtin_codes, allowed_codes)
                    ops_1 = get_prev_thread(b1_op['inputs'][1], operators, tensors, buffers, builtin_codes, allowed_codes)
                    ops_2 = get_prev_thread(b2_op['inputs'][0], operators, tensors, buffers, builtin_codes, allowed_codes)
                    ops_o = get_next_thread(b2_op['outputs'][0], operators, tensors, buffers, builtin_codes, allowed_codes)

                    codes_0 = [builtin_codes[operators[_]['opcode_index']] for _ in ops_0]
                    codes_1 = [builtin_codes[operators[_]['opcode_index']] for _ in ops_1]
                    codes_2 = [builtin_codes[operators[_]['opcode_index']] for _ in ops_2]
                    o_codes = [builtin_codes[operators[_]['opcode_index']] for _ in ops_o]

                    is_transposed = False
                    if codes_0 == ['RESHAPE', 'TRANSPOSE']:
                        assert(codes_0 == codes_1)
                        assert(codes_0 == codes_2)
                        assert(codes_0 == list(reversed(o_codes)))
                        is_transposed = True

                    
                    if is_transposed:
                        inject_ops = []

                        axis = -1
                        idx_0 = operators[ops_0[0]]['inputs'][0]
                        splits_0 = [tensors[idx_0]['shape'][-1]//attn_heads for _ in range(attn_heads)]
                        split_op_0, tensors, buffers = op_split_v(tensors, buffers, builtin_codes.index('SPLIT_V'), idx_0, False, axis, splits_0)
                        inject_ops.append(split_op_0)

                        idx_1 = operators[ops_1[0]]['inputs'][0]
                        splits_1 = [tensors[idx_1]['shape'][-1]//attn_heads for _ in range(attn_heads)]
                        split_op_1, tensors, buffers = op_split_v(tensors, buffers, builtin_codes.index('SPLIT_V'), idx_1, False, axis, splits_1)
                        inject_ops.append(split_op_1)

                        idx_2 = operators[ops_2[0]]['inputs'][0]
                        splits_2 = [tensors[idx_2]['shape'][-1]//attn_heads for _ in range(attn_heads)]
                        split_op_2, tensors, buffers = op_split_v(tensors, buffers, builtin_codes.index('SPLIT_V'), idx_2, False, axis, splits_2)
                        inject_ops.append(split_op_2)

                        b1_oshape = [1] + tensors[b1_op['outputs'][0]]['shape'][-2:]
                        b2_oshape = [1] + tensors[b2_op['outputs'][0]]['shape'][-2:]

                        concat_inputs = []
                        for h in range(attn_heads):
                            _b1_op, tensors, buffers = op_clone(b1_op, tensors, buffers)
                            _m_op, tensors, buffers = op_clone(mul_op, tensors, buffers)
                            _s_op, tensors, buffers = op_clone(soft_op, tensors, buffers)
                            if extra_op != None:
                                _extra_op, tensors, buffers = op_clone(extra_op, tensors, buffers)
                            _b2_op, tensors, buffers = op_clone(b2_op, tensors, buffers)

                            _b1_op['inputs'][0] = split_op_0['outputs'][h]
                            _b1_op['inputs'][1] = split_op_1['outputs'][h]
                            b1_i0 = tensors[_b1_op['inputs'][0]]['shape']
                            b1_i1 = tensors[_b1_op['inputs'][1]]['shape']
                            _b1_op['builtin_options']['adj_y'] = not _b1_op['builtin_options']['adj_y']

                            #c = a b (or n.k = n.m x m.k)
                            #adj_x = m.n
                            #adj_y = k.m

                            # c = n=100,k=64 (m=100)


                            #ct = bt at (or k.n = k.m x m.n)

                            tensors[_b1_op['outputs'][0]]['shape'] = b1_oshape
                            tensors[_s_op['inputs'][0]]['shape'] = b1_oshape
                            tensors[_s_op['outputs'][0]]['shape'] = b1_oshape
                            if extra_op:
                                tensors[_extra_op['inputs'][0]]['shape'] = b1_oshape
                                tensors[_extra_op['outputs'][0]]['shape'] = b1_oshape

                            _m_op['inputs'][0] = _b1_op['outputs'][0]

                            _m_op['outputs'][0] = _s_op['inputs'][0]

                            if extra_op:
                                _s_op['outputs'][0] = _extra_op['inputs'][0]

                            _b2_op['inputs'][0] = split_op_2['outputs'][h]
                            _b2_op['inputs'][1] = _s_op['outputs'][0]
                            tensors[_b2_op['outputs'][0]]['shape'] = b2_oshape

                            tensors[_b2_op['outputs'][0]]['shape'] = [1, b2_oshape[-1], b2_oshape[-2]]
                            _b2_op['inputs'][1] = split_op_2['outputs'][h]
                            if extra_op:
                                _b2_op['inputs'][0] = _extra_op['outputs'][0]
                            else:
                                _b2_op['inputs'][0] = _s_op['outputs'][0]

                            # _b2_op['builtin_options']['adj_x'] = not _b2_op['builtin_options']['adj_x']
                            _b2_op['builtin_options']['adj_x'] = True
                            _b2_op['builtin_options']['adj_x'] = False
                            _b2_op['builtin_options']['adj_y'] = not _b2_op['builtin_options']['adj_y']




                            concat_inputs.append(_b2_op['outputs'][0])

                            inject_ops.append(_b1_op)
                            inject_ops.append(_m_op)
                            inject_ops.append(_s_op)
                            if extra_op:
                                inject_ops.append(_extra_op)
                            inject_ops.append(_b2_op)

                        c_out = operators[ops_o[-1]]['outputs'][0]
                        concat_op, tensors, buffers = op_concat(tensors, buffers, builtin_codes.index('CONCATENATION'), concat_inputs, c_out, axis)
                        replace_input(operators, c_out, concat_op['outputs'][0])

                        inject_ops.append(concat_op)

                        all_ops = attn_chain + ops_0 + ops_1 + ops_2 + ops_o
                        prior_ops = list(sorted([_ for _ in all_ops if _ < i]))
                        later_ops = list(sorted([_ for _ in all_ops if _ >= i]))

                        operators = remove_ops(operators, all_ops)
                        i = i-len(prior_ops)

                        operators = operators[:i] + inject_ops + operators[i:]

        elif transform == 'YOLO_ATTN':

            is_pattern, pattern = find_pattern(operators[i:], tensors, buffers, builtin_codes, 'YOLO_ATTN')
            if is_pattern:
                ops = []
                tensor_idx = op['inputs'][0]
                ishape = tensors[op['inputs'][0]]['shape']
                axis = -1
                slice_idx = [o['outputs'][0] for o,p in zip(operators[i:], pattern) if p == 'STRIDED_SLICE']
                slice_shapes = [tensors[_]['shape'] for _ in slice_idx]

                batch_ops = [o for o,p in zip(operators[i:], pattern) if p == 'BATCH_MATMUL']
                batch_op0 = batch_ops[0]
                batch_opts0 = batch_op0['builtin_options'].copy()

                batch_t0 = tensors[batch_op0['outputs'][0]]
                batch_shape0 = batch_t0['shape']

                batch_op1 = batch_ops[1]
                batch_opts1 = batch_op1['builtin_options'].copy()
                batch_t1 = tensors[batch_op1['outputs'][0]]
                batch_shape1 = batch_t1['shape']

                soft_ops = [o for o,p in zip(operators[i:], pattern) if p == 'SOFTMAX']
                soft_op0 = soft_ops[0]
                soft_t0 = tensors[soft_op0['outputs'][0]]

                reshape_ops = [o for o,p in zip(operators[i:], pattern) if p == 'RESHAPE']
                for _op in reshape_ops:
                    if batch_op1['outputs'][0] in _op['inputs']:
                        batch_reshape_op = _op

                batch_reshape_t = tensors[batch_reshape_op['outputs'][0]]
                batch_reshape_shape = batch_reshape_t['shape']

                t_ops = [o for o,p in zip(operators[i:], pattern) if p == 'TRANSPOSE']
                for _op in t_ops:
                    if batch_reshape_op['outputs'][0] in _op['inputs']:
                        batch_t_op = _op

                conv_reshape_op = reshape_ops[-1]
                conv_reshape_t = tensors[conv_reshape_op['outputs'][0]]
                conv_reshape_shape = conv_reshape_t['shape']

                conv_ops = [o for o,p in zip(operators[i:], pattern) if p == 'CONV_2D']
                conv_op = conv_ops[0]

                #channel heads c2psa
                reshape_shape = tensors[operators[i+1]['outputs'][0]]['shape']
                channel_splits = [reshape_shape[-2] for _ in range(reshape_shape[-3])]
                channel_op, tensors, buffers = op_split_v(tensors, buffers, builtin_codes.index('SPLIT_V'), tensor_idx, False, axis, channel_splits)
                ops.append(channel_op)

                mul1_op, mul1_quant, mul1_type, mul1_opts = grab_op(i+pattern.index('MUL'))
                mul1_val, mul1_fquant, mul1_ftype = wvals(mul1_op['inputs'][1])

                add_ops = [o for o,p in zip(operators[i:], pattern) if p == 'ADD']
                final_add_op = add_ops[-1]
                if len(add_ops) > 1:
                    add0_op, add0_quant, add0_type, add0_opts = grab_op(i+pattern.index('ADD'))
                    add0_val, add0_fquant, add0_ftype = wvals(add0_op['inputs'][1])

                batch_opts0['adj_x'] = not batch_opts0['adj_x']
                batch_opts0['adj_y'] = not batch_opts0['adj_y']
                batch_opts1['adj_x'] = not batch_opts1['adj_x']
                concat_inputs = []
                bypass_inputs = []
                for c,c_idx in enumerate(channel_op['outputs']):
                    tensor_idx = channel_op['outputs'][c]
                    splits = [_[-2] for _ in slice_shapes]
                    split_op, tensors, buffers = op_split_v(tensors, buffers, builtin_codes.index('SPLIT_V'), tensor_idx, False, axis, splits)
                    ops.append(split_op)


                    batch_inputs0 = []
                    batch_inputs1 = []
                    for s,split_idx in enumerate(split_op['outputs']):
                        _reshape_idx = split_idx
                        _reshape = [1,slice_shapes[s][-1], slice_shapes[s][-2]]

                        reshape_op, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), _reshape_idx, False, _reshape)
                        ops.append(reshape_op)

                        if s < 2:
                            batch_inputs0.append(ops[-1]['outputs'][0])
                        else:
                            batch_inputs1.append(ops[-1]['outputs'][0])
                            bypass_inputs.append(split_idx)

                    _shape = batch_shape0.copy()[-3:]
                    _shape[-3] = 1
                    batch_op, tensors, buffers = op_batchmatmul(tensors, buffers, builtin_codes.index('BATCH_MATMUL'), batch_inputs0[0], batch_inputs0[1], _shape,
                                                                batch_t0['type'], batch_t0['quantization'], "BatchMatMulOptions", batch_opts0)
                    ops.append(batch_op)

                    data = np.frombuffer(mul1_val.tobytes(), dtype=np.uint8).tolist()
                    mul_op, tensors, buffers = op_mul(tensors, buffers, builtin_codes.index('MUL'), ops[-1]['outputs'][0], False,
                           mul1_quant['scale'], mul1_quant['zero_point'],
                           data, [], mul1_ftype, mul1_fquant['scale'], mul1_fquant['zero_point'])
                    ops.append(mul_op)

                    soft_op, tensors, buffers = op_softmax(tensors, buffers, builtin_codes.index('SOFTMAX'), ops[-1]['outputs'][0], soft_t0['type'], soft_t0['quantization'])
                    ops.append(soft_op)

                    soft_shape = tensors[soft_op['outputs'][0]]['shape']
                    _split_shape = [1,400,20,20]
                    if len(add_ops) > 1:
                        data = np.frombuffer(add0_val.tobytes(), dtype=np.uint8).tolist()
                        add_op, tensors, buffers = op_add(tensors, buffers, builtin_codes.index('ADD'), ops[-1]['outputs'][0], False, add0_quant['scale'], add0_quant['zero_point'], data, add0_val.shape, add0_ftype, add0_fquant['scale'], add0_fquant['zero_point'])
                        ops.append(add_op)



                    # reshape_op, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), ops[-1]['outputs'][0], False, _split_shape)
                    # transpose_nhwc_op, tensors, buffers = op_transpose(tensors, buffers, builtin_codes.index('TRANSPOSE'), reshape_op['outputs'][0], False, [0,2,3,1])
                    # ops.append(reshape_op)
                    # ops.append(transpose_nhwc_op)

                    # transpose_nchw_op, tensors, buffers = op_transpose(tensors, buffers, builtin_codes.index('TRANSPOSE'), transpose_nhwc_op['outputs'][0], False, [0,3,1,2])
                    # reshape_op, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), transpose_nchw_op['outputs'][0], False, soft_shape)
                    # ops.append(transpose_nchw_op)
                    # ops.append(reshape_op)


                    next_batch_shape = batch_shape1.copy()[-3:]
                    next_batch_shape[-3] = 1
                    batch_op, tensors, buffers = op_batchmatmul(tensors, buffers, builtin_codes.index('BATCH_MATMUL'), batch_inputs1[0], ops[-1]['outputs'][0], next_batch_shape,
                                                                batch_t1['type'], batch_t1['quantization'], "BatchMatMulOptions", batch_opts1)
                    ops.append(batch_op)

                    half_batch_shape = batch_reshape_shape.copy()
                    half_batch_shape[-3] //= 2

                    if 0:
                        reshape_op, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), ops[-1]['outputs'][0], False, half_batch_shape)
                        ops.append(reshape_op)
                        transpose_nhwc_op, tensors, buffers = op_transpose(tensors, buffers, builtin_codes.index('TRANSPOSE'), ops[-1]['outputs'][0], False, [0,2,3,1])
                        ops.append(transpose_nhwc_op)

                    else:
                        half_batch_shape = [half_batch_shape[0], half_batch_shape[2], half_batch_shape[3], half_batch_shape[1]]
                        transpose_nhwc_op, tensors, buffers = op_transpose(tensors, buffers, builtin_codes.index('TRANSPOSE'), ops[-1]['outputs'][0], False, [0,2,1])
                        ops.append(transpose_nhwc_op)
                        reshape_op, tensors, buffers = op_reshape(tensors, buffers, builtin_codes.index('RESHAPE'), ops[-1]['outputs'][0], False, half_batch_shape)
                        ops.append(reshape_op)

                    concat_inputs.append(ops[-1]['outputs'][0])


                concat_op, tensors, buffers = op_concat(tensors, buffers, builtin_codes.index('CONCATENATION'), concat_inputs, batch_t_op['outputs'][0], -1)
                ops.append(concat_op)
                final_add_op['inputs'][0] = concat_op['outputs'][0]

                conv_concat_op, tensors, buffers = op_concat(tensors, buffers, builtin_codes.index('CONCATENATION'), bypass_inputs, conv_op['inputs'][0], -1)
                ops.append(conv_concat_op)

                conv_op['inputs'][0] = ops[-1]['outputs'][0]
                ops.append(conv_op)

                ops.append(final_add_op)

                operators = operators[:i] + ops + operators[i+len(pattern):]
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
                        replace_output(operators, tensor_idx, id_op['inputs'][1])
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


def get_thread(tid, operators, tensors, buffers, opcodes, allowed_opcodes, prev=False):
    ops = []

    if prev:
        ids = [_ for _ in range(len(operators)) if tid in operators[_]['outputs']]
    else:
        ids = [_ for _ in range(len(operators)) if tid in operators[_]['inputs']]

    if len(ids) == 1 and opcodes[operators[ids[0]]['opcode_index']] in allowed_opcodes:
        id = ids[0]
        ops.append(id)
    else:
        return ops

    while True:
        prev_ids, next_ids = get_connected_ops(operators, id)
        ids = next_ids
        if prev:
            ids = prev_ids
        if len(ids) != 1:
            break
        id = ids[0]

        if not opcodes[operators[id]['opcode_index']] in allowed_opcodes:
            break
        ops.append(id)

    return ops


def get_prev_thread(id, operators, tensors, buffers, opcodes, allowed_opcodes):
    return list(reversed(get_thread(id, operators, tensors, buffers, opcodes, allowed_opcodes, prev=True)))


def get_next_thread(id, operators, tensors, buffers, opcodes, allowed_opcodes):
    return get_thread(id, operators, tensors, buffers, opcodes, allowed_opcodes, prev=False)


def find_chain(operators, tensors, buffers, opcodes, pattern):
    if len(operators) < len(pattern):
        return None

    chain = []
    i = 0
    p = 0
    while p < len(pattern):
        if i is None:
            return None

        op = operators[i]
        code = opcodes[op['opcode_index']]
        if code != pattern[p]:
            return None
        else:
            chain.append(i)

        i = get_next_op(operators, i)
        p += 1

    return chain



def find_pattern(operators, tensors, buffers, opcodes, pattern):
    if pattern == 'NORM':
        codes = [opcodes[op['opcode_index']] for i,op in zip(range(16), operators)]
        target = ['RESHAPE', 'MEAN', 'SUB', 'MUL', 'MEAN', 'ADD', 'DEQUANTIZE', 'SQRT',
                  'DIV', 'QUANTIZE', 'MUL', 'MUL', 'MUL', 'SUB', 'ADD', 'RESHAPE']
        target2 = ['RESHAPE', 'MEAN', 'SUB', 'MUL', 'MEAN', 'ADD', 'DEQUANTIZE', 'SQRT',
                  'DIV', 'QUANTIZE', 'MUL', 'MUL', 'MUL', 'SUB', 'RESHAPE']
        if codes[:16] == target:
            return True, target
        if codes[:15] == target2:
            return True, target2

    elif pattern == 'SPNV2_NORM':
        codes = [opcodes[op['opcode_index']] for i,op in zip(range(13), operators)]
        target = ['TRANSPOSE', 'RESHAPE', 'TRANSPOSE', 'MEAN', 'SQUARED_DIFFERENCE',
                   'MEAN', 'ADD', 'RSQRT','SUB', 'MUL', 'TRANSPOSE', 'RESHAPE', 'TRANSPOSE']
        if codes[:13] == target:
            return True, target

    elif pattern == 'SPNV2_TRANSPOSE':
        codes = [opcodes[op['opcode_index']] for i,op in zip(range(7), operators)]
        target = ['TRANSPOSE', 'RESHAPE', 'MUL', 'LOGISTIC', 'MUL', 'TRANSPOSE']
        target2 = ['TRANSPOSE', 'RESHAPE', 'MUL', 'ADD', 'LOGISTIC', 'MUL', 'TRANSPOSE']
        if codes[:6] == target:
            return True, target
        if codes[:7] == target2:
            return True, target2

    elif pattern == 'RESHAPE':
        codes = [opcodes[op['opcode_index']] for i,op in zip(range(2), operators)]
        target = ['TRANSPOSE', 'RESHAPE']
        if codes[:2] == target:
            ishape = tensors[operators[0]['inputs'][0]]['shape']
            oshape = tensors[operators[0]['outputs'][0]]['shape']
            if len(ishape) == 5 and len(oshape) == 5:
                return True, target

    elif pattern == 'YOLO_ATTN':
        target = ['TRANSPOSE', 'RESHAPE', 'STRIDED_SLICE', 'STRIDED_SLICE', 'BATCH_MATMUL', 'MUL', 'SOFTMAX',
                  'STRIDED_SLICE','BATCH_MATMUL', 'RESHAPE', 'TRANSPOSE', 'RESHAPE', 'TRANSPOSE', 'CONV_2D', 'ADD']
        target2 = ['TRANSPOSE', 'RESHAPE', 'STRIDED_SLICE', 'STRIDED_SLICE', 'STRIDED_SLICE', 'BATCH_MATMUL', 'MUL', 'SOFTMAX', 'ADD', 'BATCH_MATMUL', 'RESHAPE', 'RESHAPE', 'TRANSPOSE', 'CONV_2D', 'TRANSPOSE', 'ADD']
        codes = [opcodes[op['opcode_index']] for i,op in zip(range(max(len(target),len(target2))), operators)]
        if codes[:len(target)] == target:
            return True, target
        if codes[:len(target2)] == target2:
            return True, target2


    return False, codes


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
    if copy:
        if debug:
            shutil.copyfile(tmp_json, dst_tflite.replace('.tflite', '.json'))

    tname = json2tflite(tmp_json)
    update_shapes(tname, graph)

    if copy:
        shutil.copyfile(tname, dst_tflite)
    return tmp_tflite


def clean_operator_codes(operator_codes): 
         """Normalize operator codes for graph processing and TFLite serialization.

         Compiler codes 0-126 determine the builtin name. Otherwise, a named
         builtin gets the legacy sentinel 127.
         """                                                         
         schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "schema.fbs")          
         schema = json_load(schema_path.replace(".fbs", ".json"))                                       
         operators = schema["definitions"]["tflite_BuiltinOperator"]["enum"]                            

         for code in operator_codes:                                                                   
             bcode, legacy = code.get("builtin_code"), code.get("deprecated_builtin_code")                                               
             if bcode and bcode in operators and 0 < operators.index(bcode) <= 127:
                 code["deprecated_builtin_code"] = operators.index(bcode)
             elif legacy is not None and 0 <= legacy < 127:
                 code["builtin_code"] = operators[legacy]                                              
             elif "builtin_code" in code:                                                              
                 code["deprecated_builtin_code"] = 127                                                      
                                                                                                            
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
        

def clear_unused_operators(operators, tensors, buffers):
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
            stensors.append(tensors[t])

    return stensors, sbuffers


def clean_operators(operators, tensors, buffers, force_shape=None, sort=None):
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
            import sys
            sys.setrecursionlimit(10000)
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

    # keep same order
    for t in range(len(tensors)):
        if t not in used_tensors:
            pass
        else:
            stensors.append(tensors[t])
            remap_tensors[t] = rt
            rt += 1

    # # reorder as tensors encountered
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

    for op_idx in range(len(soperators)):
        soperators[op_idx]['inputs'] = [remap_tensors[_] for _ in soperators[op_idx]['inputs'] if _ != -1]
        soperators[op_idx]['outputs'] = [remap_tensors[_] for _ in soperators[op_idx]['outputs'] if _ != -1]

    # keep same order
    remap_buffers = {}
    rb = 1 # leave buffer 0
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
        if i != -1 and not i in input_tensors and not i in all_output_tensors:
            input_tensors.append(i)

    output_tensors = []
    for o in all_output_tensors:
        if o != -1 and not o in output_tensors and not o in all_input_tensors:
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


def create_transposed_tensor(tensors, tensor_idx, transpose):
    swap = lambda x,y: [x[y[0]],x[y[1]],x[y[2]],x[y[3]]]

    tensors.append(tensors[tensor_idx].copy())
    idx = len(tensors)-1
    tensors[idx]['shape'] = swap(tensors[idx]['shape'], transpose)

    return tensors, idx


def create_signature_defs(graph, tensors, input_tensors, output_tensors):
    graph['signature_defs'] = [{'inputs': None, 'outputs': None, 'signature_key':'serving_default', 'subgraph_index':0}]
    graph['signature_defs'][0]['inputs'] = [{'name': tensors[idx]['name'], 'tensor_index': idx} for idx in input_tensors]
    graph['signature_defs'][0]['outputs'] = [{'name': tensors[idx]['name'], 'tensor_index': idx} for idx in output_tensors]
    return graph


def clean_opcodes(operators, opcodes):
    opcodes_used = set()
    for o in operators:
        if o['opcode_index'] not in opcodes_used:
            opcodes_used.add(o['opcode_index'])
    opcodes_used = sorted(list(opcodes_used))
    for o in operators:
        o['opcode_index'] = opcodes_used.index(o['opcode_index'])
    opcodes = [opcodes[_] for _ in opcodes_used]
    return operators, opcodes


def create_graph(operators, tensors, buffers, opcodes, force_shape=None, clean_graph=False):
    global unique_count

    operators, tensors, buffers = [_.copy() for _ in operators], [_.copy() for _ in tensors], buffers.copy()

    if clean_graph:
        operators, tensors, buffers = clean_operators(operators, tensors, buffers, force_shape=force_shape)
        operators, opcodes = clean_opcodes(operators, opcodes)
    else:
        tensors, buffers = clear_unused_operators(operators, tensors, buffers)
        if not force_shape is None:
            for t in tensors:
                if 'buffer' in t and not 'data' in buffers[t['buffer']]:
                    t['shape'] = force_shape

    unique_names = []
    for t in tensors:
        if 'name' in t:
            while t['name'] in unique_names:
                t['name'] += '.{}'.format(unique_count)
                unique_count += 1
            unique_names.append(t['name'])

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


def create_graph_from_split(graph, splits):
    subgraph = graph['subgraphs'][0]
    operators, tensors = subgraph['operators'], subgraph['tensors']
    buffers, opcodes = graph['buffers'], graph['operator_codes']

    return create_graph([operators[_] for _ in splits], tensors, buffers, opcodes)



def all_close_graphs(t0, t1):
    seed = 42
    with open(t0, 'rb') as f:
        i0, o0, _ = generate_inputs_outputs(f.read(), int8_range=True, seed=seed)
    with open(t1, 'rb') as f:
        i1, o1, _ = generate_inputs_outputs(f.read(), int8_range=True, seed=seed)

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
    

def transform_graph(graph, passes, debug, compression_vbx=None):
    subgraph = graph['subgraphs'][0]
    operators, tensors = subgraph['operators'], subgraph['tensors']
    buffers, opcodes = graph['buffers'], graph['operator_codes']

    builtin_codes = [_['builtin_code'] for _ in opcodes]

    # add opcodes
    for transform in passes:
        if transform == 'CONCAT_PAD':
            if not 'PAD' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 34, 'version': 1, 'builtin_code': 'PAD'})
                builtin_codes = [_['builtin_code'] for _ in opcodes]
        if transform == 'PAD_TO_MULTIPLE_OF_4':
            if not 'PAD' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 34, 'version': 1, 'builtin_code': 'PAD'})
                builtin_codes = [_['builtin_code'] for _ in opcodes]
            if not 'PADV2' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 60, 'version': 1, 'builtin_code': 'PADV2'})
                builtin_codes = [_['builtin_code'] for _ in opcodes]
        if transform == 'LUT':
            if not 'CAST' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 53, 'version': 1, 'builtin_code': 'CAST'})
            if not 'GATHER' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 36, 'version': 2, 'builtin_code': 'GATHER'})
            if not 'QUANTIZE' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 114, 'version': 1, 'builtin_code': 'QUANTIZE'})
            if not 'SPLIT' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 49, 'version': 2, 'builtin_code': 'SPLIT'})
            if not 'CONCATENATION' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 2, 'version': 1, 'builtin_code': 'CONCATENATION'})
            if compression_vbx == 'ucomp' and not 'CONV_2D' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 3, 'version': 3, 'builtin_code': 'CONV_2D'})
        elif transform == 'EXPLICIT_PAD':
            if not 'PAD' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 34, 'version': 1, 'builtin_code': 'PAD'})
            if not 'PADV2' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 60, 'version': 1, 'builtin_code': 'PADV2'})
        elif transform == 'PADV2':
            if not 'PADV2' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 60, 'version': 1, 'builtin_code': 'PADV2'})
        elif transform == 'FRAC_BATCH':
            if not 'SPLIT_V' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 102, 'version': 2, 'builtin_code': 'SPLIT_V'})
            if not 'CONCATENATION' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 2, 'version': 1, 'builtin_code': 'CONCATENATION'})
        elif transform == 'FRAC_FC':
            if not 'SPLIT_V' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 102, 'version': 2, 'builtin_code': 'SPLIT_V'})
            if not 'CONCATENATION' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 2, 'version': 1, 'builtin_code': 'CONCATENATION'})
        elif transform == 'FUSE_SLICES':
            if not 'SPLIT_V' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 102, 'version': 2, 'builtin_code': 'SPLIT_V'})
        elif transform == 'CHANNEL_SPLIT':
            if not 'SPLIT_V' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 102, 'version': 2, 'builtin_code': 'SPLIT_V'})
        elif transform == 'UNPACK_SLICE':
            if not 'STRIDED_SLICE' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 45, 'version': 1, 'builtin_code': 'STRIDED_SLICE'})
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
        elif transform == 'EXPAND_BROADCAST':
            if not 'CONCATENATION' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 2, 'version': 1, 'builtin_code': 'CONCATENATION'})
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
        elif transform == 'FC2_CONV_2D':
            if not 'CONV_2D' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 3, 'version': 3, 'builtin_code': 'CONV_2D'})
            if not 'RESHAPE' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 22, 'version': 1, 'builtin_code': 'RESHAPE'})
            if not 'SPLIT' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 49, 'version': 2, 'builtin_code': 'SPLIT'})
            if not 'SPLIT_V' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 102, 'version': 2, 'builtin_code': 'SPLIT_V'})
            if not 'CONCATENATION' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 2, 'version': 1, 'builtin_code': 'CONCATENATION'})
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
        elif transform == 'FUSE_GROUP1':
            if not 'DEPTHWISE_CONV_2D' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 4, 'version': 3, 'builtin_code': 'DEPTHWISE_CONV_2D'})
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
        elif transform in ['REWRITE_NORM', 'YOLO_ATTN']:
            if not 'SPLIT' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 49, 'version': 2, 'builtin_code': 'SPLIT'})
            if not 'SPLIT_V' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 102, 'version': 2, 'builtin_code': 'SPLIT_V'})
            if not 'CONCATENATION' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 2, 'version': 1, 'builtin_code': 'CONCATENATION'})
        elif transform in ['SPLIT_XL_POOL']:
            if not 'AVERAGE_POOL_2D' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 1, 'version': 2, 'builtin_code': 'AVERAGE_POOL_2D'})
        elif transform in ['MAXPOOL_INPUT_CHANNEL_SPLIT']:
            if compression_vbx == 'ucomp':
                if not 'SPLIT' in builtin_codes:
                    opcodes.append({'deprecated_builtin_code': 49, 'version': 2, 'builtin_code': 'SPLIT'})
                if not 'CONCATENATION' in builtin_codes:
                    opcodes.append({'deprecated_builtin_code': 2, 'version': 1, 'builtin_code': 'CONCATENATION'})
        elif transform in ['POOL_AS_MEAN']:
            if not 'MEAN' in builtin_codes:
                opcodes.append({'deprecated_builtin_code': 40, 'version': 1, 'builtin_code': 'MEAN'})

        builtin_codes = [_['builtin_code'] for _ in opcodes]

    for transform in passes:
        operators, tensors, buffers = apply_transformation(transform, operators, tensors, buffers, opcodes, builtin_codes, debug, compression_vbx)

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
    colors = np.array([int(_[2])*(2**16) + int(_[1])*(2**8) + int(_[0])*(2**0) for _ in colors]).astype(np.uint32)
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


def get_cut_indices(graph, cuts, include_outputs=True):
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
    if not 'SUB' in opcodes:
        opcodes.append({'deprecated_builtin_code': 41, 'version': 1, 'builtin_code': 'SUB'})
    if not 'DIV' in opcodes:
        opcodes.append({'deprecated_builtin_code': 42, 'version': 1, 'builtin_code': 'DIV'})

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
    o0 = tensor_idx
    t0 = tensors[tensor_idx]
    tensors.append(tensors[tensor_idx].copy())
    tensor_idx = len(tensors)-1

    do_scale = isinstance(scale, list) or (scale != 1.0)
    if do_scale:
        if not isinstance(scale, list):
            scale = [scale]
        scale_val = np.asarray(scale, dtype=np.float32)
        scale_data = np.frombuffer(scale_val.tobytes(), dtype=np.uint8).tolist()

    do_mean = isinstance(mean, list) or (mean != 0.0)
    if do_mean:
        if not isinstance(mean, list):
            mean = [mean]
        mean_val = np.asarray(mean, dtype=np.float32)
        mean_data = np.frombuffer(mean_val.tobytes(), dtype=np.uint8).tolist()

    preprocess_ops = []
    u_op, tensor, buffers = op_quantize(tensors, buffers, opcodes.index('QUANTIZE'), tensor_idx, True, 'UINT8', [1.0], [0])

    tensor_idx = u_op['outputs'][0]
    preprocess_ops.append(u_op)

    if do_mean or do_scale:
        if t0['type'] == 'INT8':
            tensors[tensor_idx]['quantization'] = {'scale': [1.0], 'zero_point': [-128], 'details_type': 'NONE', 'quantized_dimension': 0}
        elif t0['type'] == 'UINT8':
            tensors[tensor_idx]['quantization'] = {'scale': [1.0], 'zero_point': [0], 'details_type': 'NONE', 'quantized_dimension': 0}

        d_op, tensors, buffers = op_type(tensors, buffers, opcodes.index('DEQUANTIZE'), tensor_idx, False,
                                         'FLOAT32', {'details_type': 'NONE', 'quantized_dimension': 0})
        tensor_idx = len(tensors)-1
        preprocess_ops.append(d_op)
        if do_mean:
            sub_op, tensors, buffers = op_sub(tensors, buffers, opcodes.index('SUB'), tensor_idx, False, False,
                                              None, None,
                                              mean_data, mean_val.shape, 'FLOAT32',
                                              None, None)
            tensor_idx = len(tensors)-1
            preprocess_ops.append(sub_op)
        if do_scale:
            div_op, tensors, buffers = op_div(tensors, buffers, opcodes.index('DIV'), tensor_idx, False, False,
                                              {'details_type': 'NONE', 'quantized_dimension': 0},
                                              scale_data, scale_val.shape, 'FLOAT32',
                                              {'details_type': 'NONE', 'quantized_dimension': 0})
            tensor_idx = len(tensors)-1
            preprocess_ops.append(div_op)

        q_op, tensors, buffers = op_type(tensors, buffers, opcodes.index('QUANTIZE'), tensor_idx, False,
                                         t0['type'], t0['quantization'].copy())
        preprocess_ops.append(q_op)
        tensor_idx = len(tensors)-1

    replace_input(operators, o0, tensor_idx)
    operators = operators[:i] + preprocess_ops + operators[i:]


    return operators, tensors, buffers 


def convert():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    if args.model.endswith('.tflite'):
        tflite2json(args.model)
    elif args.model.endswith('.json'):
        json2tflite(args.model)


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
        tgraph = transform_graph(graph, ['YOLO_ARG_MAX'], args.verbose, None)
    save_graph(args.tflite.replace('.tflite', '.post.tflite'), dir_obj, tgraph, copy=True)

    if not dir_obj is None:
        dir_obj.cleanup()


def cut():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite")
    parser.add_argument("-c", "--cuts", type=int, nargs='*')
    args = parser.parse_args()

    graph, dir_obj = load_graph(args.tflite)

    cuts = get_cut_indices(graph, args.cuts)
    for i,indices in enumerate(cuts):
        graph_ = select_graph(graph, indices)

        save_graph(args.tflite.replace('.tflite', '.{}.tflite').format(i), dir_obj, graph_, copy=True)

    if not dir_obj is None:
        dir_obj.cleanup()

def parse_range(value):
    try:
        start, end = map(int, value.split('-'))
        if start > end:
            raise ValueError("Start must be less than or equal to end")
        return list(range(start, end + 1))
    except Exception:
        raise argparse.ArgumentTypeError("Must be an integer range in the format start-end (e.g., 5-10)")


def extract():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-e", "--extracted", type=int, nargs='+')
    group.add_argument("-r", "--extracted-range", type=parse_range)
    args = parser.parse_args()
    extracted = args.extracted
    if extracted is None:
        extracted = args.extracted_range

    graph, dir_obj = load_graph(args.tflite)

    graph_ = select_graph(graph, extracted)

    save_graph(args.tflite.replace('.tflite', '.extracted.tflite'), dir_obj, graph_, copy=True)

    if not dir_obj is None:
        dir_obj.cleanup()


def inline_buffers(interp, buffers, tensors):
    for b,buf in enumerate(buffers):
        if 'data' not in buf and buf['size'] > 0 and buf['offset'] > 0:
            for t,tensor in enumerate(tensors):
                if b == tensor['buffer']:
                    arr = interp.get_tensor(t)
                    data = np.frombuffer(arr.tobytes(), dtype=np.uint8).tolist()
                    buf['data'] = data
                    buf['offset'] = 0
                    buf['size'] = 0
                    break
    return buffers


def clean():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite")
    parser.add_argument("-i", "--inline-weights", action='store_true')
    args = parser.parse_args()

    graph, dir_obj = load_graph(args.tflite)

    subgraph = graph['subgraphs'][0]
    operators, tensors = subgraph['operators'], subgraph['tensors']
    buffers, opcodes = graph['buffers'], graph['operator_codes']

    if args.inline_weights:
        interp = get_interp(args.tflite)
        buffers = inline_buffers(interp, buffers, tensors)

    graph_ = create_graph(operators, tensors, buffers, opcodes, clean_graph=True)
    save_graph(args.tflite.replace('.tflite', '.clean.tflite'), dir_obj, graph_, copy=True)

    if not dir_obj is None:
        dir_obj.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite")
    parser.add_argument("-p", "--passes", nargs='+', choices=passes)
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()

    graph, dir_obj = load_graph(args.tflite)

    transformed_graph = transform_graph(graph, args.passes, args.verbose, None)
    save_graph(args.tflite.replace('.tflite', '.tr.tflite'), dir_obj, transformed_graph, copy=True)

    if not dir_obj is None:
        dir_obj.cleanup()

    if args.verbose:
        all_close, max_diff = all_close_graphs(args.tflite, args.tflite.replace('.tflite', '.tr.tflite'))
        if not all_close and max_diff > 1:
            print("WARNING: transformed graph doesn't match. Max diff == {}".format(max_diff))


if __name__ == "__main__":
    main()
