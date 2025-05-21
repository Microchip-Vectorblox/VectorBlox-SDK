import json
import struct
import sys
import os.path
from math import floor, ceil, log2, frexp, copysign, exp, tanh, pow, log, sqrt
from contextlib import contextmanager
import numpy as np
import os
import glob
import silence_tensorflow.auto
import tensorflow as tf

import vbx.sim
from .vnnx_types import *
from .vnnx_allocation import *
from .split_tflite import lut_pattern, VCI_LUT, is_split_idx_in_engine_graphs, is_singleton, channels_first_shape


VERBOSE = 0
USE_CONV_VECTOR = 1
PRECALC_OUTPUT = 0
USE_CONV_FIA = 1
USE_FIA_DB = 1

USE_PRECALC = 1
QMULT = 31
QTRUNCATE = 0

OPTIMIZED_MUL = 1
OPTIMIZED_ADD = 1
OPTIMIZED_DEPTHWISE = 0
ALL_WEIGHTS = 1


@contextmanager
def exception_catcher( node, node_index, tensor_index=None, subnode=None, subnode_index=None):
    try:
        yield
    except AssertionError as e:
        print(e)
        print("Error in VNNX node index {} type {}".format(node_index, node.name))
        if tensor_index is not None:
            print("Tensor index {}".format(tensor_index))
        if subnode_index is not None and subnode is not None:
            print("Subnode index {} type {}".format(subnode_index, subnode))
        sys.exit(1)
        

def sigmoid(x):
  if x > 0:   
    z = np.exp(-x)
    return 1/(1+z)
  else:
    z = np.exp(x)
    return z/(1+z)




def allocation(node, preset, opcode, debug=False):
    tile = None
    return tile_subgraph(node, preset, opcode)


def get_subop_parameters(subop, tensors, buffers):
    i_tensor = tensors[subop['inputs'][0]]
    f_tensor = tensors[subop['inputs'][1]]
    if 'buffer' in i_tensor and 'data' in buffers[i_tensor['buffer']]:
        i_tensor = tensors[subop['inputs'][1]]
        f_tensor = tensors[subop['inputs'][0]]
    filter_data = get_numpy_data(f_tensor, buffers)
    
    filter_offset = f_tensor['quantization']['zero_point'][0]
    input_offset = i_tensor['quantization']['zero_point'][0]

    input_scale = i_tensor['quantization']['scale']
    filter_scale = f_tensor['quantization']['scale']
            
    k, h, w, c = 1, 1, 1, 1
    if len(f_tensor['shape']) == 1:
        c = tuple(f_tensor['shape'])[0]
    elif len(f_tensor['shape']) == 2:
        k, c = tuple(f_tensor['shape'])
    elif len(f_tensor['shape']) == 3:
        h, w, c = tuple(f_tensor['shape'])
        filter_data = filter_data.transpose((2, 0, 1))
    elif len(f_tensor['shape']) == 4:
        k, h, w, c = tuple(f_tensor['shape'])
        filter_data = filter_data.transpose((0, 3, 1, 2))
    elif len(f_tensor['shape']) == 5:
        assert (f_tensor['shape'][0] == 1)
        k, h, w, c = f_tensor['shape'][1:]
        filter_data = filter_data.transpose((0, 1, 4, 2, 3))
    filter_shape_dims = [k, c, h, w]

    filter_data = filter_data.flatten()
    
    return i_tensor, f_tensor, input_scale[0], filter_scale, input_offset, filter_offset, filter_shape_dims, filter_data


# Use the same LUT generation code for both uint8_t and int8_t. Int8_t indexes
# will be directly casted to uint8_t, the int8 LUT will thus be ordered as [0,
# 1, ..., 127, -128, ..., -2, -1] instead of [-128, -127, ..., -1, 0, 1, ...,
# 126, 127].
def LUTPopulateInt8(input_scale, input_zero_point, output_scale, output_zero_point, transform, transform_params=None, itype=calc_type.INT8, otype=calc_type.INT8):
    lut_uint8 = np.zeros((256,), dtype=np.uint8)
    inverse_scale = 1. / output_scale

    max_idx, min_idx = 127, -128
    if itype==calc_type.UINT8:
        max_idx, min_idx = 255, 0

    max_out, min_out = 127, -128
    if otype==calc_type.UINT8:
        max_out, min_out = 255, 0
    
        
    for idx in range(min_idx, max_idx+1):
        dequantized = input_scale * (idx - input_zero_point)
        transformed = LUTTransform(transform, dequantized, transform_params)
        quantized = -128
        
        if otype==calc_type.UINT8:
            quantized = 0

        if not tf.math.is_nan(transformed) and not tf.math.is_inf(transformed):
            rescaled = TfLiteRound(transformed * inverse_scale)
            quantized = (rescaled + output_zero_point)
        lut_uint8[np.int32(idx).astype(np.uint8)] = np.int32(max(min(max_out, quantized), min_out)).astype(np.uint8)

    vals = {}
    step_vals = [] 
    step_indices = [] 
    prev_val = None
    for idx in range(min_idx,max_idx+1):
        val = lut_uint8[np.int32(idx).astype(np.uint8)]
        if prev_val is None or val != prev_val:
            step_vals.append(np.int8(val))
            step_indices.append(idx)
        if val in vals:
            vals[val].append(idx)
        else:
            vals[val] = [idx]
        prev_val = val

    min_val = [vals[k] for k in vals if min_idx in vals[k]][0]
    if len(min_val) == 1:
        first_unique_idx = min_idx
    else:
        for i in range(min_idx+1, max_idx):
            if not i in min_val:
                first_unique_idx = i
                break

    max_val = [vals[k] for k in vals if max_idx in vals[k]][0]
    if len(max_val) == 1:
        last_unique_idx = max_idx
    else:
        for i in range(max_idx-1, min_idx-1, -1):
            if not i in max_val:
                last_unique_idx = i
                break

    first_unique_idx = min_idx
    last_unique_idx = max_idx

    return lut_uint8, first_unique_idx, last_unique_idx, step_vals, step_indices


def LUTPopulate(input_scale, input_zero_point, output_scale, output_zero_point, transform, transform_params=None, bytes=4, itype=calc_type.INT8):
    lut_dtype = np.uint8
    max_val, min_val = 127, -128

    if bytes == 4:
        lut_dtype = np.uint32
        max_val, min_val = 2**31-1, -2**31

    lut_uint = np.zeros((256,), dtype=lut_dtype)
    inverse_scale = 1. / output_scale

    max_idx, min_idx = 127, -128
    if itype==calc_type.UINT8:
        max_idx, min_idx = 255, 0

    for idx in range(min_idx, max_idx+1):
        dequantized = input_scale * (idx - input_zero_point)
        transformed = LUTTransform(transform, dequantized, transform_params)
        quantized = min_val
        if not tf.math.is_nan(transformed) and not tf.math.is_inf(transformed):
            rescaled = TfLiteRound(transformed * inverse_scale)
            quantized = (rescaled + output_zero_point)
        lut_uint[np.int32(idx).astype(np.uint8)] = np.int32(max(min(max_val, quantized), min_val)).astype(lut_dtype)

    vals = {}
    step_vals = [] 
    step_indices = [] 
    prev_val = None
    for idx in range(min_idx,max_idx+1):
        val = lut_uint[np.int32(idx).astype(np.uint8)]
        if prev_val is None or val != prev_val:
            if bytes == 4:
                step_vals.append(np.int32(val))
            else:
                step_vals.append(np.int8(val))
            step_indices.append(idx)
        if val in vals:
            vals[val].append(idx)
        else:
            vals[val] = [idx]
        prev_val = val

    min_val = [vals[k] for k in vals if min_idx in vals[k]][0]
    if len(min_val) == 1:
        first_unique_idx = min_idx
    else:
        for i in range(min_idx+1, max_idx):
            if not i in min_val:
                first_unique_idx = i
                break

    max_val = [vals[k] for k in vals if max_idx in vals[k]][0]
    if len(max_val) == 1:
        last_unique_idx = max_idx
    else:
        for i in range(max_idx-1, min_idx-1, -1):
            if not i in max_val:
                last_unique_idx = i
                break

    first_unique_idx = min_idx
    last_unique_idx = max_idx

    return lut_uint, first_unique_idx, last_unique_idx, step_vals, step_indices




def LUTTransform(transform, dequantized, transform_params=None):
    return transform(dequantized)


def pad_list(x, len_pad, value=1):
    l = list(x)
    if len(l) < len_pad:
        l = [value for _ in range(len_pad-len(l))] + l
    return l


def conv_pack_shift(data, maps):
    kernels = len(data)
    flat = np.asarray(data, dtype=np.int32)
    packed = np.zeros(flat.shape, dtype=flat.dtype)

    for k in range(ceil(kernels/maps)):
        bytes = flat[k*maps:(k+1)*maps].astype(np.int8)
        zeros = np.zeros(bytes.shape, dtype=bytes.dtype) 
        words = np.concatenate((bytes,zeros,zeros,zeros), dtype=np.int8)
        packed[k*maps:(k+1)*maps] = np.frombuffer(words.tobytes(), dtype=np.int32)

    return packed.tolist()


def conv_pack_biases(data, maps):
    kernels = len(data)
    flat = np.asarray(data, dtype=np.int32)
    packed = np.zeros(flat.shape, dtype=flat.dtype)

    for k in range(ceil(kernels/maps)):
        bytes = np.frombuffer(flat[k*maps:(k+1)*maps].flatten().tobytes(), dtype=np.uint8)
        shuffled = bytes.reshape((-1,4)).transpose((1,0)).flatten()
        packed[k*maps:(k+1)*maps] = np.frombuffer(shuffled.tobytes(), dtype=np.int32)

    return packed.tolist()


def conv_pack_weights(data, maps, use_depthwise=False, is_transpose=False, weight_pad=0):
    # k,c,h,w -> c,h,w,k
    s = data.shape
    t = (1,2,3,0)
    kernels = s[0]
    channels = s[1]
    kernel_height = s[2]
    kernel_width = s[3]
    chunk_size = s[1]*s[2]*s[3]

    if s[0] == 1 and not is_transpose:
        kernels = s[1]
        channels = 1
        chunk_size = s[2]*s[3]
        data = data.squeeze(axis=0)
        t = (1,2,0)
        if use_depthwise:
            data = np.flip(data, axis=1) # along height axis
            t = (0,2,1)
            new_data_shape = (1, kernels, kernel_width, kernel_height+(2*weight_pad))
            chunk_size = new_data_shape[2] * new_data_shape[3]
            data = np.pad(data, ((0,0), (weight_pad, weight_pad), (0,0)), mode='constant', constant_values=0)

    flat = np.zeros(data.flatten().shape, dtype=data.dtype)
    for k in range(ceil(kernels/maps)):
        chunk = data[k*maps:(k+1)*maps].transpose(t)
        flat[k*chunk_size*maps:(k+1)*chunk_size*maps] = chunk.flatten()

    return flat.reshape(s) if not use_depthwise else flat.reshape(new_data_shape)


def get_quantized_multiplier(scale):
    multiplier = []
    shift = []
    for s in scale:
        significand, channel_shift = quantize_multiplier(s)
        multiplier.append(significand)
        shift.append(channel_shift)
        
    return multiplier, shift


def get_quantized_multiplier_from_tensor(tensor):
    scale = tensor['quantization']['scale']
    multiplier, shift = get_quantized_multiplier(scale)
    return scale, multiplier, shift



def get_effective_quantized_multiplier_from_tensors(i_tensor, o_tensor, f_tensor, filter_scale=None):

    input_scale = i_tensor['quantization']['scale']
    output_scale = o_tensor['quantization']['scale']
    if filter_scale is None:
        filter_scale = f_tensor['quantization']['scale']

    output_multiplier = []
    output_shift = []
    effective_output_scale = [input_scale[0] * f / output_scale[0] for f in filter_scale]
    output_multiplier, output_shift = get_quantized_multiplier(effective_output_scale)

    return effective_output_scale, output_multiplier, output_shift


def get_effective_quantized_multiplier_from_op(op, tensors, filter_scale=None):

    i_tensor = tensors[op['inputs'][0]]
    o_tensor = tensors[op['outputs'][0]]
    if filter_scale is None:
        f_tensor = tensors[op['inputs'][1]]
    else:
        f_tensor = None
    return get_effective_quantized_multiplier_from_tensors(i_tensor, o_tensor, f_tensor, filter_scale)


def MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift):
    total_shift = QMULT - shift
    round = 2 ** (total_shift - 1)
    result = int(x * quantized_multiplier + round)
    result = result // (2**total_shift)

    return int(result)


def DivideByQuantizedMultiplier(x, quantized_multiplier, shift):
    total_shift = QMULT - shift
    round = 2 ** (total_shift - 1)

    result = int(x * (2**total_shift))
    result = (result + round) // quantized_multiplier

    return int(result)


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return floor(n*multiplier + 0.5) / multiplier


def TfLiteRound(n, decimals=0):
    rounded_abs = round_half_up(abs(n), decimals)
    return copysign(rounded_abs, n)


def CalculateInputRadius(input_integer_bits, input_left_shift, total_signed_bits):
    return [((2 ** input_integer_bits) - 1) * (2**(total_signed_bits - input_integer_bits)) // (2**i) for i in input_left_shift]
    # else:
    #     max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) * (1ll << (total_signed_bits - input_integer_bits)) / (1ll << input_left_shift)
    #     return floor(max_input_rescaled)


def get_quantized_multiplier_gemmlowp(scale):

    assert(scale > 0)
    assert(scale < 1)
    s = 0
    while(scale < 0.5):
        scale *= 2.
        s += 1
    q = TfLiteRound(scale * (2**31))
    assert(q <= (2**31))
    if q == (2**31):
        q /= 2
        s -= 1
    assert(s >= 0)
    assert(q <= np.iinfo(np.int32).max)
    return q,s


def quantize_multiplier(double_multiplier, TFLITE_SINGLE_ROUNDING=False):

    if TFLITE_SINGLE_ROUNDING:
        pass
    assert(double_multiplier >= 0), "Double multiplier of quantize_multiplier is not greater or equal to 0"
    if double_multiplier == 0.:
        return 0, 0 
    q, shift = frexp(double_multiplier)
    q_fixed = np.int64(TfLiteRound(q * (2**QMULT)))
    if QTRUNCATE > 0:
        q_fixed = np.int64(TfLiteRound(q * (2**(QMULT-QTRUNCATE))))

    assert(q_fixed <= (2**QMULT)), "q_fixed in quantize_multiplier is not less or equal to 2^QMULT"
    if q_fixed == (2**QMULT):
        q_fixed /= 2
        shift += 1
    assert(q_fixed <= np.iinfo(np.int32).max), "q_fized in quantize_multiplier is not less or equal to numpy int32 maximum"

    if shift < -QMULT:
        shift = 0
        q_fixed = 0
    if TFLITE_SINGLE_ROUNDING:
        if shift > QMULT-1:
            shift = QMULT-1 
            q_fixed = (2**QMULT) - 1

    quantized_multiplier = np.int32(q_fixed)
    return quantized_multiplier, shift 


def adjust_transform(transform, dims):
    s = list(transform)
    while 1:
        if s[0] == 0:
            s = s[1:] - 1
        else:
            break

    if dims > len(s):
        outer = dims - len(s)
        transform = tuple([_ for _ in range(outer)] + [outer + _ for _ in s])

    return transform


def flatten_n(ishape, oshape):
    idim = len(ishape)
    odim = len(oshape)

    if idim - odim > 0:
        flatn = 1 + idim - odim
        if oshape[-1] == np.prod(ishape[-flatn:]):
            return flatn
    return 0


def check_unpack(ishape, oshape, axis, c_axis, count):

    iarr = np.random.randint(0, 10, ishape)
    c_iarr = channels_first_array(iarr)

    oarr = tf.unstack(iarr, axis=axis)[0]
    c_oarr = channels_first_array(oarr)

    if c_axis == -1:
        if len(ishape) >= 4: #axis = -1    NHWC -> NHC   NCHW->CNH  single col then 1,0,2
            tmp = tf.unstack(c_iarr, axis=-1)[0]
            tmp = np.transpose(tmp, adjust_transform([1,0,2], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 3: #axis = -1 HWC -> HC   CHW->HC  single col then 0,2,1
            tmp = tf.unstack(c_iarr, axis=-1)[0]
            tmp = np.transpose(tmp, adjust_transform([1,0], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 2: #axis = -1 HW -> H   HW->H  single col
            tmp = tf.unstack(c_iarr, axis=-1)[0]
            assert(np.all(tmp == c_oarr))
            return True
    elif c_axis == -2:
        if len(ishape) >= 4: #axis = -2    NHWC -> NWC   NCHW->CNW  single row, then 1,0,2
            tmp = tf.unstack(c_iarr, axis=-2)[0]
            tmp = np.transpose(tmp, adjust_transform([1,0,2], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 3: #axis = -2  HWC -> WC   CHW->WC  single row, then 0,2,1
            tmp = tf.unstack(c_iarr, axis=-2)[0]
            tmp = np.transpose(tmp, adjust_transform([1,0], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 2: #axis = -2 HW -> W   HW->W  single row
            tmp = tf.unstack(c_iarr, axis=-2)[0]
            assert(np.all(tmp == c_oarr))
            return True
    elif c_axis == -3: #axis = -3    NHWC -> NHW   NCHW->WNH single channel then 2,0,1
        if len(ishape) >= 4:
            tmp = tf.unstack(c_iarr, axis=-3)[0]
            tmp = np.transpose(tmp, adjust_transform([2,0,1], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 3: #axis = -3  HWC -> HW   CHW->HW  single channel
            tmp = tf.unstack(c_iarr, axis=-3)[0]
            assert(np.all(tmp == c_oarr))
            return True
    elif c_axis == -4: #axis = -4    NHWC -> HWC   NCHW->CHW  single batch
        tmp = tf.unstack(c_iarr, axis=-4)[0]
        assert(np.all(tmp == c_oarr))
        return True

    return False

def check_pack(ishape, oshape, axis, c_axis, count):

    iarrs = [np.random.randint(0, 10, ishape) for _ in range(count)]
    oarr = np.stack(iarrs, axis=axis)
    assert(tuple(oshape) == oarr.shape)

    c_iarrs = [channels_first_array(_) for _ in iarrs]
    c_oarr = channels_first_array(oarr)

    if c_axis == -1:
        if len(ishape) >= 3: #axis = -1    HWC -> HWxC   CHW->HCWx  1,0,2 then multi-col
            tmp = [np.transpose(_, adjust_transform([1,0,2], _.ndim)) for _ in c_iarrs]
            tmp = np.stack(tmp, axis=-1)
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 2: #axis = -1    HW -> HWx   HW->xHW  multi-maps
            pass
        elif len(ishape) == 1: #axis = -1    W -> Wx   W->Wx  multi-col?
            pass
    elif c_axis == -2:
        if len(ishape) >= 3: #axis = -2    HWC -> HxWC   CHW->HCxW  1,0,2 then multi-row
            tmp = [np.transpose(_, adjust_transform([1,0,2], _.ndim)) for _ in c_iarrs]
            tmp = np.stack(tmp, axis=-2)
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 2: #axis = -2    HW -> HxW   HW->WxH  1,0 then multi-row
            pass
    elif c_axis == -3: #axis = -3    HWC -> HWCx   CHW->HxWC 1,2,0 then multi-map
        tmp = [np.transpose(_, adjust_transform([1,2,0], _.ndim)) for _ in c_iarrs]
        tmp = np.stack(tmp, axis=-3)
        assert(np.all(tmp == c_oarr))
        return True
    elif c_axis == -4: #axis = -4    NHWC -> XNHWC   NCHW->NxCHW  multi-batch
        tmp = np.stack(c_iarrs, axis=-4)
        assert(np.all(tmp == c_oarr))
        return True

    return False


def channels_first_array_reshape(ishape, transform):
    if tuple(ishape) ==  transform:
        return 0

    iarr = np.random.randint(0, 10, ishape)
    oarr = np.reshape(iarr, transform)
    oshape = oarr.shape

    c_iarr = channels_first_array(iarr)
    c_oarr = channels_first_array(oarr)
    cishape = c_iarr.shape
    coshape = c_oarr.shape

    dims = len(transform)
    idims = len(ishape)
    odims = len(oshape)

    mode = -1
    ones = lambda x: len([_ for _ in x if _ == 1])
    squeezed_shape = lambda sh, ni: tuple([_ for i,_ in enumerate(sh) if i != len(sh) + ni])


    squeeze = False
    if ones(ishape) - 1 == ones(oshape):
        squeeze = True

        squeeze_axis = None
        for i in range(-1, -(len(ishape)+1), -1):
            if squeezed_shape(ishape, i) == tuple(oshape):
                squeeze_axis = i
                break

    if squeeze:
        assert(squeeze_axis)
        if squeeze_axis < -4:
            print('squeeze axis < -4 not supported')
            mode = -1
        elif squeeze_axis == -1: # n,h,w aka w,n,h  2,0,1
            tmp = np.squeeze(c_iarr, axis=-3)
            tmp = np.transpose(tmp, adjust_transform([2,0,1], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            mode = 4 + 1
        elif squeeze_axis == -2: # n,h,c aka c,n,h  1,0,2
            tmp = np.squeeze(c_iarr, axis=-1)
            tmp = np.transpose(tmp, adjust_transform([1,0,2], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            mode = 4 + 2
        elif squeeze_axis == -3: # n,w,c aka c,n,w  1,0,2
            tmp = np.squeeze(c_iarr, axis=-2)
            tmp = np.transpose(tmp, adjust_transform([1,0,2], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            mode = 4 + 3
        elif squeeze_axis == -4:  #1,h,w,c -> c,h,w NOP
            mode = 0
        else:
            print('squeeze unknown axis ', squeeze_axis) 
        return mode

    expand = False
    if ones(ishape) + 1 == ones(oshape):
        expand = True
        expand_axis = None

        for i in range(-1, -(len(oshape)+1), -1):
            if squeezed_shape(oshape, i) == tuple(ishape):
                expand_axis = i
                break
    if expand:
        assert(expand_axis)
        if expand_axis < -3:
            print('expand axis < -3 not supported')
            mode = -1
        elif expand_axis == -1 and len(ishape) == 3: # n,h,w aka n,h,w,1  
            mode = 0
        elif expand_axis == -1: # n,h,w,c,1 aka n,1,h,w,c  
            tmp = np.transpose(c_iarr, adjust_transform([1,2,0], c_iarr.ndim))
            tmp = np.expand_dims(tmp, axis=-3)
            assert(np.all(tmp == c_oarr))
            mode = 8 + 1
        elif expand_axis == -2: # n,h,w,1,c aka n,h,c,w,1
            tmp = np.transpose(c_iarr, adjust_transform([1,0,2], c_iarr.ndim))
            tmp = np.expand_dims(tmp, axis=-1)
            assert(np.all(tmp == c_oarr))
            mode = 8 + 2
        elif expand_axis == -3: # n,h,1,w,c aka n,h,c,1,w
            tmp = np.transpose(c_iarr, adjust_transform([1,0,2], c_iarr.ndim))
            tmp = np.expand_dims(tmp, axis=-2)
            assert(np.all(tmp == c_oarr))
            mode = 8 + 3
        else:
            print('expand unknown axis ', expand_axis) 
        return mode

    join = False
    if len(ishape) - 1 == len(oshape):
        join = True
        join_axis = None
        for i in range(1, len(ishape)):
            if ishape[-i]*ishape[-i-1] == oshape[-i]:
                join_axis = -i
                break

    if join:
        if join_axis < -3:
            print('join axis < -3 not supported')
            mode = -1
        elif join_axis == -1: # n,h,w*c aka w*c,n,h
            mode = 12 + 1
            tmp = np.transpose(c_iarr, adjust_transform([2,0,1], c_iarr.ndim)) #2,0,1   n,c,h,w -> n,w,c,h
            tshape = list(tmp.shape)[:-3] + [tmp.shape[-3]*tmp.shape[-2]] + [tmp.shape[-1]]
            tmp = np.reshape(tmp, tshape) # n,w*c,h
            tmp = np.transpose(tmp, adjust_transform([1,0,2], tmp.ndim)) #1,0,2
            assert(np.all(tmp == c_oarr))
        elif join_axis == -2: # n,h*w,c aka c,n,h*w
            mode = 12 + 2
        elif join_axis == -3: # n*h,w,c aka c,n*h,w
            mode = 12 + 3
        else:
            print('join unknown axis ', join_axis) 
        return mode

    seperate = False
    if len(ishape) + 1 == len(oshape):
        seperate = True
        seperate_axis = 0
        for i in range(1, len(oshape)):
            if oshape[-i]*oshape[-i-1] == ishape[-i]:
                seperate_axis = -i
                break

    if seperate:
        if seperate_axis < -3:
            print('seperate axis < -3 not supported')
            mode = -1
        elif seperate_axis == -1: # n,h,w*c aka
            mode = 16 + 1
        elif seperate_axis == -2: # n,h*w,c aka
            mode = 16 + 2
        elif seperate_axis == -3: # n*h,w,c aka
            mode = 16 + 3
        else:
            print('seperate unknown axis ', seperate_axis) 
        return mode

    if idims == odims:
        if oshape[-3] == ishape[-1] and oshape[-2] == ishape[-3] and oshape[-1] == ishape[-2]: # swap channels first to last aka 2,0,1
            return 2


    flatten  = flatten_n(ishape, oshape)
    if flatten:
        # print('flatten', flatten)
        return 0

    if len(ishape) - len(oshape) > 1:
        print("cannot handle multi-axis squeeze")
    if len(oshape) - len(ishape) > 1:
        print("cannot handle multi-axis expand")

    return mode


def channels_first_array(arr):
    s = list(arr.shape)
    if len(s) >= 3:
        carr = np.transpose(arr, adjust_transform([2,0,1], arr.ndim))
    else:
        carr = np.copy(arr)
    return carr


def channels_first_axis(axis, dims):
    if axis < 0:
        axis = dims + axis

    if dims >= 3:
        if axis == dims - 1:
            axis = dims - 3
        elif axis >= dims - 3:
            axis += 1

    return axis


def get_numpy_data_from_index(index, tensors, buffers):
    tensor = tensors[index]
    # shape = tensor['shape']
    shape = tensor.get('shape', (1))
    dtype = np.dtype(tensor['type'].lower())
    raw_data = bytearray(buffers[tensor['buffer']]['data'])
    data = np.frombuffer(raw_data, dtype=dtype).reshape(shape)

    return data


def get_numpy_data(tensor, buffers):
    # shape = tensor['shape']
    shape = tensor.get('shape', (1))
    dtype = np.dtype(tensor['type'].lower())
    raw_data = bytearray(buffers[tensor['buffer']]['data'])
    data = np.frombuffer(raw_data, dtype=dtype).reshape(shape)

    return data


def op_activations(op, activations):
    input_activations =  [_ for _ in op['inputs'] if _ in activations] 
    output_activations =  [_ for _ in op['outputs'] if _ in activations] 
    return input_activations, output_activations


def op_params(op, activations):
    input_params =  [_ for _ in op['inputs'] if not (_ in activations)] 
    output_params =  [_ for _ in op['outputs'] if not (_ in activations)] 
    return input_params, output_params


def get_graph_activations(json_subgraphs):
    activations = {}
    subgraph_inputs = []
    subgraph_outputs = []

    for g, graph in enumerate(json_subgraphs):
        buffers = graph['buffers']
        subgraph = graph['subgraphs'][0]
        tensors = subgraph['tensors']
        ops = subgraph['operators']

        subgraph_inputs += subgraph['inputs']
        subgraph_outputs += subgraph['outputs']

        subgraph_activations = []
        for op in ops:
            for _ in op['inputs']:
                if 'buffer' in tensors[_]:
                    if not 'data' in buffers[tensors[_]['buffer']]:
                        subgraph_activations += [_]   

        for op in ops:
            for _ in op['outputs']:
                if 'buffer' in tensors[_]:
                    if not 'data' in buffers[tensors[_]['buffer']]:
                        subgraph_activations += [_]
                
        for a in subgraph_activations:
            if a not in activations:
                activations[a] = tensors[a]

    inputs = [_ for _ in subgraph_inputs if _ not in subgraph_outputs]
    outputs = [_ for _ in subgraph_outputs if _ not in subgraph_inputs]

    return inputs, outputs, activations

 
def precalculate_filter_input_bias(filters, input_offset, filter_offset=0, reduce=True):
    if filters.shape == (): # numpy array 0-dimensional, single element filter
        return np.array( [(np.sum(filters) * -input_offset) + (-filter_offset * -input_offset) ])
    if filters.shape[0] == 1 and reduce and len(filters.shape)!=1: #TODO cover when group > 1, but not depthwise
        filters = filters[0]

    precalc = np.zeros((filters.shape[0],),dtype=np.int32)
    for i,f in enumerate(filters):
        precalc[i] += (np.sum(f) * -input_offset) + (-filter_offset * -input_offset)
    return precalc


def precalculate_output_offset_bias(scale, offset):
    precalc = []
    for s in scale:
        if s == 0:
            x = 0
        else:
            x = 1.0 * offset / s
            round = 0.5
            if x < 0:
                round = -0.5
            x = int(x + round)
        precalc.append(x)
    return precalc


def find_tensors_by_id(tensors, id):
    x = []
    for t in tensors:
        if t.id == id:
            x.append(t)
    return x


def find_tensors_by_name(tensors, name):
    x = []
    for t in tensors:
        if t.name == name:
            x.append(t)
    return x


def get_node_and_offset(Nodes, id):
    for i,n in enumerate(Nodes):
        for j,t in enumerate(n.tensor_array):
            # if t.id == id:
            if t.name == id:
                return i, j
    print(id, 'not found')
    return None


def mod_shape(shape, x):
    if len(shape) > 1:
        w = (shape[-1] + (x-1)) // x * x 
        h = (shape[-2] + (x-1)) // x * x 
        return tuple(list(shape)[:-2] + [h, w])
    return shape

def update_tensor_shapes(Nodes):
    all_tensor_array = [_ for n in Nodes for _ in n.tensor_array]

    for t in all_tensor_array:
        while len(t.shape) < SHAPE_DIMS: 
            t.shape = list(t.shape) + [0]

# create a mapping of nodes, inputs, outputs to be used for pad sublayer injection
# key: input_id, value: list of output_ids that are created from the input_id
# if input_id has more than one output_id, means the input_id is taken as input to different Nodes
def tensor_in_id_to_out_id_mapping(json_subgraphs):
    in_id_to_out_ids = {}
    for graph in json_subgraphs:
        for op in graph['subgraphs'][0]['operators']:
            g_inputs = op['inputs']
            g_outputs = op['outputs']
            for in_id in g_inputs:
                current_value = in_id_to_out_ids.get(in_id, [])
                current_value += g_outputs
                in_id_to_out_ids[in_id] = current_value
    in_id_to_out_ids = {key: value for key, value in in_id_to_out_ids.items() if len(value) > 1}
    return in_id_to_out_ids

# this function is to be called after populate_nodes and used for pad sublayer injection
def final_check_pad_sublayer_injection(in_id_to_out_ids, Nodes, ids_with_dummies):
    # using the in_id_to_out_ids dict, go through Nodes again, and find the node with output id matching the input id key (source node), 
    # as well as the nodes with output id matching with the output id values (destination nodes)
    for in_id, out_ids in in_id_to_out_ids.items():
        source_node = None
        dest_nodes = []
        for node in Nodes:
            if node.tensor_array[node.num_inputs].id == in_id:
                source_node = node
            elif node.tensor_array[0].id == in_id and node.tensor_array[node.num_inputs].id in out_ids:
                dest_nodes.append(node)

        # Must confirm that the nodes that share the input tensor (destination nodes) are:
            # a) Conv nodes
            # b) check their conv8 pad attributes (shows that they use SAME padding) and check that they're exact
            # c) have the exact same pad value (input_offset)
        # if all conditions are satisfied, then the pad sublayer can be injected directly to the source node
        # else, identity+pad must be injected for each of the applicable destination nodes (conv w/ SAME padding)
        if len(dest_nodes) > 0:
            # verify all dest nodes have the same attributes
            all_same_type = all(dnode.type == BuiltinOperator.CONV_2D for dnode in dest_nodes)
            all_same_input_offset = all(dnode.input_offset == dest_nodes[0].input_offset for dnode in dest_nodes)
            pad_attributes = ['padding_width', 'padding_height']
            all_same_pad_hw = all_same_type and all(all(getattr(dnode.Conv2DOptions, attr) == getattr(dest_nodes[0].Conv2DOptions, attr) for attr in pad_attributes) for dnode in dest_nodes)
            all_matching_attr = all_same_input_offset and all_same_pad_hw

            if all_matching_attr:
            # if 0:
                # inject pad sublayer to source node
                opcode=''
                # TODO update with correct arguments when calling function is fixed
                # inject_pad_subnode_to_previous_node(source_node, dest_nodes, ids_with_dummies, preset, opcode)
            else:
                # go through dest nodes and inject identity+pad where applicable
                for dnode in dest_nodes:
                    pad_hw = [dnode.Conv2DOptions.padding_height, dnode.Conv2DOptions.padding_width]
                    if dnode.type == BuiltinOperator.CONV_2D and (dnode.Conv2DOptions.padding_height > 0 or dnode.Conv2DOptions.padding_width > 0):
                        # print('2 ', dnode.tensor_array[0].id)
                        Nodes = inject_dummy_identity(Nodes, ids_with_dummies, 
                                                        dnode,
                                                        preset, 
                                                        pad_hw,
                                                        inject_strided=False)

                        # reset dest_nodes conv padding attributes as node is injected explicitly
                        dnode.Conv2DOptions.padding_height = 0
                        dnode.Conv2DOptions.padding_width = 0

    # for g, node in enumerate(Nodes):
    #     if node.type == BuiltinOperator.CONV_2D:
    #         if node.Conv2DOptions.padding_height != 0 or node.Conv2DOptions.padding_width != 0:
    #             import pdb; pdb.set_trace()

    # print(ids_with_dummies)

# Write all the tensors to a file with their offset
def write_tensor_offset_map(Nodes, activations_offset, activations_size):
    # Note: Some tensors have multiple offsets now (bug)
    nx_dirname = "nx_engine"
    offset_map_fname = os.path.join(nx_dirname, "mxp_tensor_offset_map.txt")
    with open(offset_map_fname, 'w') as table:
        table.write(f"Intermediates base address: {activations_offset}\n")
        table.write(f"Intermediates size: {activations_size}\n\n")
        table.write("\t".join(["id", "offset", "name"]) + "\n")
        printed = []
        for n in Nodes:
            for t in n.tensor_array:
                # Skip duplicate tensors from identity nodes
                # For example, Conv -> Pool may be 1 subgraph in 2.0, but 2 subgraphs in 3.0,
                # where Conv is on NX but Pool on MXP. In the 3.0 case, the Pool is a subnode
                # of an Identity. This means a new tensor with ".id" at the end of the name
                # will be made, but currently its tensor.id is not updated so it looks like a
                # conflicting id / offset mapping. Since this identity tensor is not seen by
                # NX can just omit it for now.
                if t.name.endswith('.id'):
                    continue
                assert t.buffer[0] == 0
                assert t.buffer[1] == t.direct
                entry = (t.id, t.name, t.direct)
                if entry not in printed:
                    printed.append(entry)
                    table.write("\t".join([str(t.id), str(t.direct), t.name]) + "\n")

# Write a table of each input/output tensor and its size/offset in the vnnx file
# Similar to set_io_buffers
def write_io_offset_map(Nodes, weights, test_inputs, test_outputs, io_vnnx_offset):
    all_tensor_array = []
    for n in Nodes:
        all_tensor_array += n.tensor_array

    unique_names = []
    for t in all_tensor_array:
        if t.name not in unique_names:
            unique_names.append(t.name)

    offset = io_vnnx_offset
    nx_dirname = "nx_engine"
    offset_map_fname = os.path.join(nx_dirname, "vnnx_io_offsets.txt")
    with open(offset_map_fname, 'w') as table:
        table.write("\t".join(["type", "name", "id", "size", "offset"]) + "\n")
        for name in unique_names:
            ts = find_tensors_by_name(all_tensor_array, name)
            ts_io = [t for t in ts if (t.name in test_inputs.keys() or t.name in test_outputs.keys())]
            for idx, t in enumerate(ts_io):
                if t.name in test_inputs.keys():
                    data = test_inputs[t.name].astype(np.int8).tobytes()
                    io_type = "input"
                elif t.name in test_outputs.keys():
                    data = test_outputs[t.name].astype(np.int8).tobytes()
                    io_type = "output"
                table.write("\t".join([io_type, t.name, str(t.id), str(len(data)), str(offset)]) + "\n")
                offset += np.prod(mod_shape(t.shape[:4], 4))*2 # See set_tensor_buffer


def generate_vnnx_from_json_subgraphs(json_subgraphs, preset, test_inputs, test_outputs, include_io_data=0, tmp_dir=None,\
    engine_graphs_nx=None):
    graph_inputs, graph_outputs, graph_activations = get_graph_activations(json_subgraphs)
    graph_inputs = list(test_inputs.keys())
    graph_outputs = list(test_outputs.keys())

    aliased_ids = {}
    ids_with_dummies = dict()

    weights = weight_array()

    # Get a list of tensors that are inputs and outputs to external (NX) ops.
    external_inputs = None
    external_outputs = None
    if engine_graphs_nx:
        external_inputs, external_outputs = get_external_io(json_subgraphs, engine_graphs_nx)

    in_id_to_out_ids = tensor_in_id_to_out_id_mapping(json_subgraphs)
    # if while populating nodes we see an input id in this dict, we will inject identity+pad. otherwise, inject pad sublayer to previous node.
    # TODO it will need to change such that we don't inject identity+pad in populate_nodes. it should be done after and if we can't get away with pad sublayer injection.

    Nodes = populate_nodes(json_subgraphs, preset, graph_activations, weights, aliased_ids, ids_with_dummies, tmp_dir,\
        in_id_to_out_ids, engine_graphs_nx, external_inputs, external_outputs)

    # TODO
    # final_check_pad_sublayer_injection(in_id_to_out_ids, Nodes, ids_with_dummies)

    # setup graph object
    vnnx_graph = Graph()
    vnnx_graph.num_inputs = len(graph_inputs)
    vnnx_graph.num_outputs = len(graph_outputs)

    vnnx_graph.version = graph_version()
    vnnx_graph.vbx_nn_preset = preset_select['PRESET'][preset]
    vnnx_graph.num_layers = len(Nodes)

    while len(vnnx_graph.description) < DESCRIPTION_CHARS:
        vnnx_graph.description += b"\0"

    set_io_nodes(vnnx_graph, Nodes, graph_inputs, graph_outputs, weights)
    set_skip_concat(Nodes, test_inputs, test_outputs, external_inputs)
    set_skip_channel_slice(Nodes, test_inputs, test_outputs, external_inputs)
    act_buffer_size, io_buffer_size = set_io_buffers(Nodes, weights, test_inputs, test_outputs)
    update_tensor_shapes(Nodes)

    vnnx_graph.replay_buffer = len(weights)
    vnnx_graph.replay_buffer_size = 1024*1024*512
    replay = bytearray(vnnx_graph.replay_buffer_size)

    node_data, subnode_data, tensor_data, align1 = update_offsets(vnnx_graph, Nodes)

    # do twice, once to find allocate_length, once for real
    vnnx_graph.include_io_data = include_io_data
    vnnx_graph.allocate_bytes = 0
    vnnx_graph.data_bytes = 0
    graph_data = [vnnx_graph.get_structured_data()]
    data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[weights]+[replay])

    replay_start = len(b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[weights]))
    vnnx_graph.replay_buffer = replay_start
    graph_data = [vnnx_graph.get_structured_data()]
    data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[weights]+[replay])

    node_start = len(b"".join(graph_data))
    node_stop = len(b"".join(graph_data+node_data))

    vnnx_graph.data_bytes = len(data) # - act_buffer_size
    vnnx_graph.allocate_bytes = len(data)
    graph_data = [vnnx_graph.get_structured_data()]
    data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[weights]+[replay])

    replay_first = 1
    trim_act_buffer = 1

    data = data[:vnnx_graph.data_bytes]
    m = vbx.sim.Model(data)
    try:
        sim_outputs = m.run(m.test_input)

        node_data = [m.model_bytes[node_start : node_stop]]
        replay_size = int.from_bytes(m.model_bytes[4*(14+4):4*(14+1+4)], byteorder='little')
        vnnx_graph.replay_buffer_size = replay_size

        if not replay_first:
            replay = m.model_bytes[replay_start : replay_start + vnnx_graph.replay_buffer_size]
            vnnx_graph.fixed_replay_buffer0 = int.from_bytes(m.model_bytes[4*(4):4*(4+1)], byteorder='little')
            vnnx_graph.fixed_replay_buffer1 = int.from_bytes(m.model_bytes[4*(5):4*(5+1)], byteorder='little')
            vnnx_graph.fixed_replay_buffer2 = int.from_bytes(m.model_bytes[4*(6):4*(6+1)], byteorder='little')
            vnnx_graph.fixed_replay_buffer3 = int.from_bytes(m.model_bytes[4*(7):4*(7+1)], byteorder='little')
        else:
            vnnx_graph.replay_buffer_size += 128
            replay = bytearray(vnnx_graph.replay_buffer_size)
            replay_start = len(b"".join(graph_data+node_data+subnode_data+tensor_data+align1))
            vnnx_graph.replay_buffer = replay_start

            node_data, subnode_data, tensor_data, align2 = update_offsets(vnnx_graph, Nodes, vnnx_graph.replay_buffer_size, vnnx_graph.replay_buffer)
            vnnx_graph.replay_buffer = replay_start
            graph_data = [vnnx_graph.get_structured_data()]
            data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[replay]+align2+[weights])

            vnnx_graph.data_bytes = len(data)
            if trim_act_buffer:
                vnnx_graph.data_bytes = len(data) - act_buffer_size
                # if not vnnx_graph.include_io_data:
                #     vnnx_graph.data_bytes -= io_buffer_size
            vnnx_graph.allocate_bytes = len(data)
            graph_data = [vnnx_graph.get_structured_data()]
            data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[replay]+align2+[weights])

            data = data[:vnnx_graph.data_bytes]
            m = vbx.sim.Model(data)
            sim_outputs = m.run(m.test_input)

            node_data = [m.model_bytes[node_start : node_stop]]
            replay_size = int.from_bytes(m.model_bytes[4*(14+4):4*(14+1+4)], byteorder='little')
            replay = m.model_bytes[replay_start : replay_start + vnnx_graph.replay_buffer_size]
            vnnx_graph.fixed_replay_buffer0 = int.from_bytes(m.model_bytes[4*(4):4*(4+1)], byteorder='little')
            vnnx_graph.fixed_replay_buffer1 = int.from_bytes(m.model_bytes[4*(5):4*(5+1)], byteorder='little')
            vnnx_graph.fixed_replay_buffer2 = int.from_bytes(m.model_bytes[4*(6):4*(6+1)], byteorder='little')
            vnnx_graph.fixed_replay_buffer3 = int.from_bytes(m.model_bytes[4*(7):4*(7+1)], byteorder='little')
    except:
        sys.stderr.write('ERROR: model failed to run\n')
        sys.exit(1)

    # print('observed fixed_replay_buffer values {},{},{},{}'.format(
    #     vnnx_graph.fixed_replay_buffer0,
    #     vnnx_graph.fixed_replay_buffer1,
    #     vnnx_graph.fixed_replay_buffer2,
    #     vnnx_graph.fixed_replay_buffer3))
    # print()
    # print('replay_size {}'.format(vnnx_graph.replay_buffer_size))
    # print('act buffers {}'.format(act_buffer_size))
    # print('io buffers {}'.format(io_buffer_size))
    # print()

    if not replay_first:
        data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[weights]+[replay])
        vnnx_graph.data_bytes = len(data)
        vnnx_graph.allocate_bytes = len(data)
        graph_data = [vnnx_graph.get_structured_data()]
        data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[weights]+[replay])
    else:
        data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[replay]+align2+[weights])
        vnnx_graph.data_bytes = len(data)
        if trim_act_buffer:
            vnnx_graph.data_bytes = len(data) - act_buffer_size
            # if not vnnx_graph.include_io_data:
            #     vnnx_graph.data_bytes -= io_buffer_size
        vnnx_graph.allocate_bytes = len(data)
        graph_data = [vnnx_graph.get_structured_data()]
        data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[replay]+align2+[weights])
        
        if 0: # enable for debug
            offset = len(b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[replay]+align2))
            all_tensor_array = []
            for n in Nodes:
                all_tensor_array += n.tensor_array

            for t in all_tensor_array:
                print(t.name, (t.buffer[1] + offset), hex((t.buffer[1] + offset)))

    # For 3.0, when graphs are being run in another engine, need also to write the
    # physical memory address of each tensor. Currently, for each (relevant) tensor,
    # write the address offset from the graph. Later, the address of the graph will also
    # be known from the reference design (e.g., 0x30_0010_0000).
    if engine_graphs_nx is not None:
        write_tensor_offset_map(Nodes, vnnx_graph.data_bytes, act_buffer_size)
        # Also write offset of inputs and outputs in .vnnx file, and their sizes
        write_io_offset_map(Nodes, weights, test_inputs, test_outputs, vnnx_graph.data_bytes - io_buffer_size)

    return data[:vnnx_graph.data_bytes]


def update_offsets(vnnx_graph, Nodes, weights_offset=None, min_addr=None):
    # All buffers allocated, now adjust offsets, and dump to bytearray
    node_size = Node().get_structure_size()
    subnode_size = Subnode().get_structure_size()
    tensor_size = Tensor().get_structure_size()
    num_subnodes = sum([len(n.subnode_array) for n in Nodes])
    num_tensors = sum([len(n.tensor_array) for n in Nodes])
    node_offset = vnnx_graph.get_structure_size()
    subnode_offset = vnnx_graph.get_structure_size() + node_size * len(Nodes)
    tensor_offset = subnode_offset + num_subnodes*subnode_size

    if weights_offset is None:
        weights_offset = tensor_offset + num_tensors*tensor_size
    
    #align weights_offset
    align = [bytearray(0)]
    if weights_offset % 16 != 0:
        align_len = 16 - (weights_offset % 16)
        align = [bytearray(align_len)]
        weights_offset += align_len

    vnnx_graph.update_offsets(weights_offset, min_addr)
    for n in Nodes:
        n.update_offsets(weights_offset, min_addr)

    all_subnode_array = []
    for n in Nodes:
        all_subnode_array += n.subnode_array
        n.sublayers = subnode_offset
        n.num_sublayers = len(n.subnode_array)
        subnode_offset += len(n.subnode_array)*subnode_size

    all_tensor_array = []
    for n in Nodes:
        all_tensor_array += n.tensor_array
        n.tensors = tensor_offset
        tensor_offset += len(n.tensor_array)*tensor_size

    for sn in all_subnode_array:
        sn.update_offsets(weights_offset, min_addr)

    for tn in all_tensor_array:
        tn.update_offsets(weights_offset, min_addr)
        tn.buffer[1] += weights_offset #TODO fix data structure


    tensor_data = [t.get_structured_data() for t in all_tensor_array]
    subnode_data = [sn.get_structured_data() for sn in all_subnode_array]
    node_data = [n.get_structured_data() for n in Nodes]

    return node_data, subnode_data, tensor_data, align


def set_concat_tensor_buffer(tname, all_tensor_array, weights, concat_map, concat_offset):
    t = find_tensors_by_name(all_tensor_array, tname)[0]
    if t.buffer[1] == -1:
        if t.name in concat_map.keys():
            t_ = find_tensors_by_name(all_tensor_array, concat_map[t.name])[0]
            if t_.buffer[1] == -1:
                size = np.prod(mod_shape(t_.shape, 4))*2 #TODO fix for proper sized maps (w/ output_strides)
                t_.buffer[1], t_.direct = len(weights), len(weights)
                weights += bytearray(size)

            offset = concat_offset[t.name]
            t.buffer[1], t.direct = t_.buffer[1] + offset, t_.direct + offset


def set_tensor_buffer(idx, t, ts, weights, test_inputs, test_outputs):
    if t.buffer[1] == -1:
        for prev in ts[:idx]:
            if prev.buffer[1] and prev.shape == t.shape:
                t.buffer[1], t.direct = prev.buffer[1], prev.direct
                break
        if t.buffer[1] == -1:
            size = np.prod(mod_shape(t.shape, 4))*2*sizeof_calc_type(t.type) #TODO fix for proper sized maps (w/ output_strides)
            t.buffer[1] = len(weights)
            t.direct = len(weights)
            weights += bytearray(size)


def set_skip_concat(Nodes, test_inputs, test_outputs, external_inputs):
    for n,node in enumerate(Nodes):
        if node.type == BuiltinOperator.CONCATENATION:
            concat_io = [t for t in node.tensor_array if (t.name in test_inputs.keys() or t.name in test_outputs.keys())]
            idims = len(node.tensor_array[0].shape)

            if (node.ConcatOptions.axis - idims == -3) and len(node.subnode_array) == 0 and len(concat_io) == 0:
                node.skip = 1

            # If this concat is output to NX then need to keep it
            if external_inputs:
                assert node.num_outputs == 1
                output_tensor = node.tensor_array[node.num_inputs]
                if output_tensor.id in external_inputs:
                    node.skip = 0

def set_skip_channel_slice(Nodes, test_inputs, test_outputs, external_inputs):
    for n,node in enumerate(Nodes):
        if node.type == VNNXOperator.IDENTITY:
            if len(node.subnode_array) == 1 and node.subnode_array[0].type == BuiltinOperator.SLICE:
                sn = node.subnode_array[0]
                width_matches = sn.tensor_array[0].shape[-1] == sn.tensor_array[-1].shape[-1]
                height_matches = sn.tensor_array[0].shape[-2] == sn.tensor_array[-1].shape[-2]
                channel_matches = sn.tensor_array[0].shape[-3] == sn.tensor_array[-1].shape[-3]
                is_channel_slice = width_matches and height_matches and not channel_matches
                if is_channel_slice:
                    # node.skip = 1
                    pass


def set_io_buffers(Nodes, weights, test_inputs, test_outputs):
    '''
    allocate buffers
    '''
    init_len = len(weights)

    all_tensor_array = []
    for n in Nodes:
        all_tensor_array += n.tensor_array

    unique_names = []
    for t in all_tensor_array:
        if t.name not in unique_names: 
            unique_names.append(t.name)

    concat_offset = {}
    concat_map = {}

    for n, node in enumerate(Nodes):
        if node.type == BuiltinOperator.CONCATENATION and node.skip:
            names = [_.name for _ in node.tensor_array]
            shapes = [_.shape for _ in node.tensor_array]

            offset = 0
            for name, shape in zip(names[:-1],shapes[:-1]):
                concat_map[name] = names[-1]
                concat_offset[name] = offset
                offset += np.prod(shape)

    for name in unique_names:
        ts = find_tensors_by_name(all_tensor_array, name)
        ts_io = [t for t in ts if (t.name in test_inputs.keys() or t.name in test_outputs.keys())]
        for idx, t in enumerate(ts_io):
            set_concat_tensor_buffer(t.name, all_tensor_array, weights, concat_map, concat_offset)
            set_tensor_buffer(idx, t, ts_io, weights, test_inputs, test_outputs)

            if t.name in test_inputs.keys():
                data = test_inputs[t.name].astype(np_type(t.type)).tobytes()
                weights[t.direct:t.direct+len(data)] = data
            elif t.name in test_outputs.keys():
                dtype = t.type
                data = test_outputs[t.name].astype(np_type(t.type)).tobytes()
                weights[t.direct:t.direct+len(data)] = data

    io_len = len(weights)
    io_buffer_size = io_len - init_len

    for name in unique_names:
        ts = find_tensors_by_name(all_tensor_array, name)
        for idx, t in enumerate(ts):
            set_concat_tensor_buffer(t.name, all_tensor_array, weights, concat_map, concat_offset)
            set_tensor_buffer(idx, t, ts, weights, test_inputs, test_outputs)


    act_buffer_size = len(weights) - io_len

    return act_buffer_size, io_buffer_size


def set_io_nodes(vnnx_graph, Nodes, graph_inputs, graph_outputs, weights):
    node_inputs = []
    offset_inputs = []
    node_outputs = []
    offset_outputs = []

    for i in graph_inputs + graph_outputs:
        n, off = get_node_and_offset(Nodes, i)
        node_inputs.append(n)
        offset_inputs.append(off)
    
        if type(Nodes[n].tensor_array[off].scale) == type([]):
            Nodes[n].tensor_array[off].scale_f16 = [int(_ * 2**16) for _ in Nodes[n].tensor_array[off].scale]
        else:
            Nodes[n].tensor_array[off].scale_f16 = int(Nodes[n].tensor_array[off].scale * 2**16)

    io_nodes = np.array(node_inputs + node_outputs, dtype=np.int32)
    io_offsets = np.array(offset_inputs + offset_outputs, dtype=np.int32)
    vnnx_graph.io_nodes = len(weights)
    weights += io_nodes.tobytes()
    vnnx_graph.io_offsets = len(weights)
    weights += io_offsets.tobytes()


def inject_dummy_identity(Nodes, ids_with_dummies, reference_node, preset, opcode, pad_hw=None, inject_strided=0, transpose_dilate=[1,1]):

    dummy_nodes = []
    
    for i in range(reference_node.num_inputs):
        input_shape = reference_node.tensor_array[i].shape
        reference_shape = input_shape
        output_shape = reference_shape

        node = Node()

        node.type = VNNXOperator.IDENTITY

        node.input_data_type = reference_node.input_data_type
        node.output_data_type = reference_node.input_data_type

        if inject_strided == 1:
            node.output_strides = reference_node.input_strides

        if pad_hw is not None:
            sn = Subnode()
            sn.num_inputs = 0 #TODO proper fix
            sn.num_outputs = 0

            pad_h, pad_w = pad_hw[0], pad_hw[1]

            new_h = reference_shape[-2] + pad_h + ((transpose_dilate[0]-1) * (reference_shape[-2] - 1))
            new_w = reference_shape[-1] + pad_w + ((transpose_dilate[1]-1) * (reference_shape[-1] - 1))
            reference_shape = tuple(list(reference_shape)[:-2] + [new_h, new_w])
            output_shape = reference_shape
            reference_node.m, reference_node.n = reference_shape[-2], reference_shape[-1]

            sn.input_data_type = node.input_data_type
            sn.output_data_type = node.output_data_type

            sn.type = BuiltinOperator.PAD
            sn.pads = [0, floor(pad_h/2), floor(pad_w/2), 
                       0, ceil(pad_h/2), ceil(pad_w/2)]
            if reference_node.type == BuiltinOperator.TRANSPOSE_CONV:
                sn.pads = [0, ceil(pad_h/2), ceil(pad_w/2), 
                           0, floor(pad_h/2), floor(pad_w/2)]
            sn.PadOptions.value = reference_node.input_offset
            sn.PadOptions.transpose_dilate_h = transpose_dilate[0]
            sn.PadOptions.transpose_dilate_w = transpose_dilate[1]

            if inject_strided == 1:
                sn.strides = node.output_strides
                osh, osw = node.output_strides[0], node.output_strides[1]

                strided_h = int(ceil(output_shape[-2] / osh)) * osh
                strided_w = int(ceil(output_shape[-1] / osw)) * osw
                # output_shape = tuple(list(output_shape)[:-2] + [strided_h, strided_w])

            # update padding to 0 because padding node will be injected explicitly with identity
            reference_node.Conv2DOptions.padding_width = 0
            reference_node.Conv2DOptions.padding_height = 0
            node.subnode_array.append(sn)

        # if strides == 2,2 then identity+pad/pad were injected to do transpose dilation
        # so set the stride to 1
        if transpose_dilate == [2,2]:
            reference_node.Conv2DOptions.stride_height = 1
            reference_node.Conv2DOptions.stride_width = 1

        node.channels, node.m, node.n = 1, 1, 1
        if len(input_shape) >= 3:
            node.channels = input_shape[-3]
        if len(input_shape) >= 2:
            node.m = input_shape[-2]
        node.n = input_shape[-1]


        node.activation_max = -1
        node.activation_min = -1
        node.input_multiplier = -1
        node.input_shift = -1
        node.output_multiplier = -1
        node.output_shift = -1
        node.input_offset = reference_node.input_offset
        node.output_offset = reference_node.input_offset        

        ref_id = reference_node.tensor_array[i].id
        dummy_id_list = ids_with_dummies.get(ref_id, [])
        dummy_id_name = str(ref_id) + '.' + str(len(dummy_id_list)+1)
        dummy_id_list.append(dummy_id_name)
        ids_with_dummies[ref_id] = dummy_id_list

        # new input tensor for Identity node that is a copy of the reference node input tensor
        ref_tensor = reference_node.tensor_array[i]
        t = Tensor()
        t.type = ref_tensor.type
        t.shape = ref_tensor.shape
        t.dims = ref_tensor.dims
        t.scale = ref_tensor.scale
        t.zero = ref_tensor.zero
        t.id = ref_tensor.id
        t.name = ref_tensor.name
        node.tensor_array.append(t)
        if len(node.subnode_array) == 1: # pad subnode injected
            sn.tensor_array.append(t)

        # new output tensor for Identity node that is what would now be fed into the reference node
        out_t = Tensor()
        out_t.type = ref_tensor.type
        out_t.scale = ref_tensor.scale
        out_t.zero = ref_tensor.zero
        out_t.shape = output_shape
        out_t.dims = ref_tensor.dims
        out_t.id = float(dummy_id_name)
        out_t.name = dummy_id_name
        node.tensor_array.append(out_t)
        if len(node.subnode_array) == 1: # pad subnode injected
            sn.tensor_array.append(out_t)

        if len(node.subnode_array) == 1: # pad subnode injected
            sn.num_tensors = len(sn.tensor_array)

        # node.tensor_array += sn.tensor_array #TODO proper fix.
        node.num_tensors = len(node.tensor_array)

        ref_tensor.shape = reference_shape
        ref_tensor.id = float(dummy_id_name)
        ref_tensor.name = dummy_id_name
        reference_node.tensor_array[i] = ref_tensor

        tile = allocation(node, preset, opcode)

        # keep description scheme here just for referencing injected nodes
        node.input_description = reference_node.input_description
        while len(node.input_description) < DESCRIPTION_CHARS:
            node.input_description += b"\0"
        node.output_description = dummy_id_name.encode()[:DESCRIPTION_CHARS]
        while len(node.output_description) < DESCRIPTION_CHARS:
            node.output_description += b"\0"

        t = ''
        if ref_tensor.type == calc_type.INT8:
            t = 'INT8'
        elif ref_tensor.type == calc_type.UINT8:
            t = 'UINT8'
        else:
            print('warning: not 8bit')

        dummy_nodes.append(node)
    Nodes = Nodes + dummy_nodes
    return Nodes


def inject_strided_slice(conv_node, conv_o_tensor, subnode_tensor_array, ids_with_dummies):
    sn = Subnode()

    sn.input_offset = conv_o_tensor.zero
    sn.scale = conv_o_tensor.scale
    sn.output_offset = conv_o_tensor.zero
    sn.output_scale = conv_o_tensor.scale
 
    sn.input_data_type = conv_node.output_data_type
    sn.output_data_type = conv_node.output_data_type

    sn.activation_min = -128
    sn.activation_max = 127

    # assuming NCHW format, adjust width of conv node output tensor
    # ignore stride_width in computation as we stride with StridedSlice
    conv_in_w = conv_node.tensor_array[0].shape[-1]
    pad_w = conv_node.Conv2DOptions.padding_width
    filter_w = conv_node.Conv2DOptions.filter_shape_dims[-1]
    dilation_w = conv_node.Conv2DOptions.dilation_width_factor
    adjusted_conv_o_tensor_width = conv_in_w + pad_w - (((filter_w-1)*dilation_w)+1) + 1 
    adjusted_conv_o_tensor_shape = (conv_o_tensor.shape[0], conv_o_tensor.shape[1], conv_o_tensor.shape[2], adjusted_conv_o_tensor_width)

    sn.type = BuiltinOperator.SLICE

    sn.SliceOptions.begin = (0, 0, 0, 0)
    sn.SliceOptions.end = adjusted_conv_o_tensor_shape
    sn.SliceOptions.stride = (1, 1, 1, conv_node.Conv2DOptions.stride_width)

    # pre-sliced tensor ID and name init
    ref_id = conv_o_tensor.id
    dummy_id_list = ids_with_dummies.get(ref_id, [])
    dummy_id_name = str(ref_id) + '.' + str(len(dummy_id_list)+1)
    dummy_id_list.append(dummy_id_name)
    ids_with_dummies[ref_id] = dummy_id_list

    # create new tensor (pre-sliced) for strided slice input / conv new output
    tn = Tensor()
    tn.type = sn.input_data_type
    tn.shape = adjusted_conv_o_tensor_shape
    tn.dims = conv_o_tensor.dims
    tn.scale = sn.output_scale
    tn.zero = sn.output_offset
    tn.id = float(dummy_id_name)
    tn.name = dummy_id_name
    sn.tensor_array.append(tn)
    subnode_tensor_array.insert(0, tn)

    # post-sliced tensor (conv node former output, now is strided slice output)
    tn = Tensor()
    tn.type = sn.output_data_type
    tn.shape = conv_o_tensor.shape
    tn.dims = conv_o_tensor.dims
    tn.scale = sn.output_scale
    tn.zero = sn.output_offset
    tn.id = conv_o_tensor.id
    tn.name = conv_o_tensor.name
    sn.tensor_array.append(tn)
    subnode_tensor_array.insert(1, tn)

    sn.num_tensors = len(sn.tensor_array)
    conv_node.Conv2DOptions.stride_width = 1
    conv_o_tensor.id = float(dummy_id_name)
    conv_o_tensor.name = dummy_id_name
    conv_o_tensor.shape = adjusted_conv_o_tensor_shape
    conv_node.subnode_array.insert(0, sn)    


def inject_pad_subnode_to_previous_node(prev_node, nodes, ids_with_dummies, preset, opcode, weights, prev_node_graph, transpose_dilate=[1,1]):
    sn = Subnode()

    if isinstance(nodes, Node):
        node = nodes
    elif isinstance(nodes, list):
        node = nodes[0]

    # pad_h, pad_w = pad_hw[0], pad_hw[1]
    pad_h = node.Conv2DOptions.padding_height
    pad_w = node.Conv2DOptions.padding_width

    input_id = node.tensor_array[0].id
    input_shape = node.tensor_array[0].shape

    new_h = input_shape[-2] + pad_h + ((transpose_dilate[0]-1) * (input_shape[-2] - 1))
    new_w = input_shape[-1] + pad_w + ((transpose_dilate[1]-1) * (input_shape[-1] - 1))
    input_shape = tuple(list(input_shape)[:-2] + [new_h, new_w])
    prev_node_output_shape = input_shape
    node.m, node.n = input_shape[-2], input_shape[-1]

    sn.input_data_type = prev_node.input_data_type
    sn.output_data_type = prev_node.output_data_type

    sn.type = BuiltinOperator.PAD
    sn.pads = [0, floor(pad_h/2), floor(pad_w/2), 
                0, ceil(pad_h/2), ceil(pad_w/2)]
    if node.type == BuiltinOperator.TRANSPOSE_CONV:
        sn.pads = [0, ceil(pad_h/2), ceil(pad_w/2), 
                    0, floor(pad_h/2), floor(pad_w/2)]
    sn.PadOptions.value = node.input_offset
    sn.PadOptions.transpose_dilate_h = transpose_dilate[0]
    sn.PadOptions.transpose_dilate_w = transpose_dilate[1]

    dummy_id_list = ids_with_dummies.get(input_id, [])
    dummy_id_name = str(input_id) + '.' + str(len(dummy_id_list)+1)
    dummy_id_list.append(dummy_id_name)
    ids_with_dummies[input_id] = dummy_id_list

    # creating new input tensor for pad subnode, based off current node's input
    ref_tensor = node.tensor_array[0]
    t = Tensor()
    t.type = ref_tensor.type
    t.shape = ref_tensor.shape
    t.dims = ref_tensor.dims
    t.scale = ref_tensor.scale
    t.zero = ref_tensor.zero
    t.id = ref_tensor.id
    t.name = ref_tensor.name
    sn.tensor_array.append(t)

    # creating new output tensor for pad subnode, based off current node's padded input
    out_t = Tensor()
    out_t.type = ref_tensor.type
    out_t.scale = ref_tensor.scale
    out_t.zero = ref_tensor.zero
    out_t.shape = prev_node_output_shape
    out_t.dims = ref_tensor.dims
    out_t.id = float(dummy_id_name)
    out_t.name = dummy_id_name
    sn.tensor_array.append(out_t)

    sn.num_tensors = len(sn.tensor_array)

    # if strides == 2,2 then identity+pad/pad were injected to do transpose dilation
    # so set the stride to 1
    if transpose_dilate == [2,2]:
        node.Conv2DOptions.stride_height = 1
        node.Conv2DOptions.stride_width = 1

    # update current node's input to be padded
    ref_tensor.shape = prev_node_output_shape
    ref_tensor.id = float(dummy_id_name)
    ref_tensor.name = dummy_id_name
    node.tensor_array[0] = ref_tensor

    # keep description scheme here just for referencing injected nodes
    prev_node.output_description = dummy_id_name.encode()[:DESCRIPTION_CHARS]
    while len(prev_node.output_description) < DESCRIPTION_CHARS:
        prev_node.output_description += b"\0"
    node.input_description = dummy_id_name.encode()[:DESCRIPTION_CHARS]
    while len(node.input_description) < DESCRIPTION_CHARS:
        node.input_description += b"\0"

    # padding set to 0 because padding subnode is injected explicitly into prev node
    node.Conv2DOptions.padding_width = 0
    node.Conv2DOptions.padding_height = 0

    # if nodes was a list, go through it and update all dest nodes input tensor and other attributes
    # this is for when the previous node's output branches, but all receiving nodes want the same pad
    if isinstance(nodes, list):
        for dnode in nodes[1:]:
            dnode.m, dnode.n = input_shape[-2], input_shape[-1]
            
            in_tensor = dnode.tensor_array[0]
            in_tensor.shape = prev_node_output_shape
            in_tensor.id = float(dummy_id_name)
            in_tensor.name = dummy_id_name
            dnode.tensor_array[0] = in_tensor
            
            dnode.input_description = dummy_id_name.encode()[:DESCRIPTION_CHARS]
            while len(dnode.input_description) < DESCRIPTION_CHARS:
                dnode.input_description += b"\0"
            
            dnode.Conv2DOptions.padding_width = 0
            dnode.Conv2DOptions.padding_height = 0

    # append pad subnode and update prev node # of tensors
    prev_node.tensor_array += sn.tensor_array
    prev_node.subnode_array.append(sn)
    prev_node.num_tensors = len(prev_node.tensor_array)

    # re-allocate previous node due to pad subnode
    codes = [_['builtin_code'] for _ in prev_node_graph['operator_codes']]
    subgraph = prev_node_graph['subgraphs'][0]
    ops = subgraph['operators']
    opcodes = [str(codes[_['opcode_index']]) for _ in ops]
    prev_opcode = opcodes[0]
    tile = allocation(prev_node, preset, prev_opcode)
    assert(tile is not None)

    if prev_node.type == BuiltinOperator.CONV_2D:
        set_conv_attributes(prev_node, tile, preset, prev_opcode, weights, prev_node_graph)


# helper function to run code that is tile-dependent for previous Conv2D layers that were re-allocated
def set_conv_attributes(node, tile, preset, opcode, weights, prev_node_graph):
    subgraph = prev_node_graph['subgraphs'][0]
    ops = subgraph['operators']
    tensors = subgraph['tensors']
    op = ops[0]
    f_tensor = tensors[op['inputs'][1]]
    buffers = prev_node_graph['buffers']
    filter_data = get_numpy_data(f_tensor, buffers).transpose((0,3,1,2))

    conv8 = node.Conv2DOptions

    # if not full rows, then don't double buffer
    # previously disabling double buffer for k > 3 because the reduced scratchpad size causes
    # # the first k=6 layer of yolov5 to not fit all columns, and that gives incorrect results
    if node.n != tile[-1]:
        conv8.mxp_double_buffer = 0
        conv8.split_weight_shaper_buffers = 0
        tile = allocation(node, preset, opcode)

    assert not (tile is None)

    # TODO would need to re-determine prev_fia_node whenever collision is fixed
    # as check_node_for_collision only forces collisions, which would've been done for a previous node already,
    # no need to run it here again
    # if conv8.use_fia:
    #     check_node_for_collision(node, Nodes, prev_fia_node)

    #     prev_fia_node = node

    weight_pad = 0
    _, _, _, _, _, rows, columns = tile
    if conv8.use_fia and conv8.use_depthwise and node.m == rows and node.n == columns: # check if full maps, if so pad weights here for 1D DMA
        dilated_filter_height = ((h-1)*conv8.dilation_height_factor) + 1
        padded_input_height = node.m + conv8.padding_height
        output_shaper_height = (padded_input_height - dilated_filter_height) + 1                
        conv8.fit_weights = 0
        parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]
        weight_pad = min(parallel_output_maps - 1, output_shaper_height)
    
    # if not full input maps, then set weight pad to be parallel_output_maps-1. We want to avoid splat in FIA. Offset appropriately when DMAing in FIA code
    # TODO should fit all weights if possible (same for above case)
    elif conv8.use_fia and conv8.use_depthwise and (node.m != rows or node.n != columns): 
        conv8.fit_weights = 0
        parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]
        weight_pad = parallel_output_maps-1

    conv8.filter_data = len(weights)
    fmt = "{}b".format(len(filter_data.flatten()))
    if conv8.use_fia:
        filter_data = conv_pack_weights(filter_data, node.maps, conv8.use_depthwise, weight_pad=weight_pad)
        if weight_pad != 0:
            fmt = "{}b".format(len(filter_data.flatten()))
    weights += struct.pack(fmt, *filter_data.flatten())


def update_nodes_with_dequant_params(Nodes, dequantize_op_params):
    for idx,node in enumerate(Nodes):
        for t_idx, tensor in enumerate(node.tensor_array):
            '''
            if tensor.name in dequantize_op_params.keys():

                offset = dequantize_op_params[tensor.name][1]
                scale = dequantize_op_params[tensor.name][0]

                if dequantize_op_params[tensor.name][3] == 'input':
                    node.input_offset = offset
            '''
            if tensor.id in dequantize_op_params.keys():

                offset = dequantize_op_params[tensor.id][1]
                scale = dequantize_op_params[tensor.id][0]

                if dequantize_op_params[tensor.id][3] == 'input':
                    node.input_offset = offset
                else:
                    node.output_offset = offset
                tensor.zero = offset
                tensor.scale = scale
                node.tensor_array[t_idx] = tensor
                
                if node.type == BuiltinOperator.SPLIT:
                    for t_idx2, tensor2 in enumerate(node.tensor_array):
                        tensor2.zero = offset
                        tensor2.scale = scale
                        node.tensor_array[t_idx2] = tensor2
                Nodes[idx] = node
    return Nodes


# Get a set of all tensors which are inputs and outputs of NX (external to MXP) ops
def get_external_io(json_subgraphs: list, engine_graphs_nx: list) -> tuple[set, set]:
    external_inputs = set()
    external_outputs = set()
    if not engine_graphs_nx:
        return external_inputs, external_outputs

    # Collect all the inputs and outputs of NX ops
    # Note: can use name instead of ID, but ID might be faster
    for eg in engine_graphs_nx:
        for split in eg:
            g = json_subgraphs[split]
            subgraph = g['subgraphs'][0]
            for op in subgraph['operators']:
                for i in op['inputs']:
                    external_inputs.add(i)
                for o in op['outputs']:
                    external_outputs.add(o)

    return external_inputs, external_outputs


# Add external_producer and external_consumer attributes to a tensor
def add_external_producer_consumer_info(t, external_inputs, external_outputs,\
    already_external_producer, already_external_consumer, subgraph_idx):

    has_external_producer = (t.id in external_outputs)
    has_external_consumer = (t.id in external_inputs)

    already_marked_external_producer = (t.id in already_external_producer and already_external_producer[t.id] != subgraph_idx)
    already_marked_external_consumer = (t.id in already_external_consumer and already_external_consumer[t.id] != subgraph_idx)

    if has_external_producer and not already_marked_external_producer:
        t.external_producer = True
        already_external_producer[t.id] = subgraph_idx

    if has_external_consumer and not already_marked_external_consumer:
        t.external_consumer = True
        already_external_consumer[t.id] = subgraph_idx


def check_node_for_collision(node, Nodes, prev_fia_node):
    node.Conv2DOptions.fia_collision = 1
    node.FullyConnectedOptions.fia_collision = 1
    return
    # check if prev node(s) that feeds into this node, is non-FIA (current node must wait for signal from MXP), count it as collision
    if len(Nodes) != 0:
        for idx in range(node.num_inputs): 
            input_id = node.tensor_array[idx].id

            # check tensors of every node for a match for input, if so check if the node uses MXP (i.e. is not FIA)
            for prev_node in Nodes:
                for tensor in prev_node.tensor_array:
                    if tensor.id == input_id: 
                        if prev_node.Conv2DOptions.use_fia == 1 or prev_node.FullyConnectedOptions.use_fia == 1:
                            # FIA tile collision is handled separately below
                            pass 
                        else:
                            node.Conv2DOptions.fia_collision = 1
                            node.FullyConnectedOptions.fia_collision = 1
                            return

    # check FIA tile collision
    if prev_fia_node:

        prev_fia_o_tensor_id = prev_fia_node.tensor_array[-1].id
        node_i_tensor_id = node.tensor_array[0].id
        node_i_tensor_ids = [node.tensor_array[_].id for _ in range(node.num_inputs)]

        # if prev_fia_node feeds into current node, then it is collision
        # TODO determine if we can get no collisions on other layer specs / tilings
        if prev_fia_o_tensor_id in node_i_tensor_ids:
            # if prev_fia_node.type == BuiltinOperator.CONV_2D or prev_fia_node.type == BuiltinOperator.TRANSPOSE_CONV:
            #     prev_fia_node_omaps = prev_fia_node.Conv2DOptions.kernels
            # else:
            #     prev_fia_node_omaps = 1

            # last_tile_maps = prev_fia_node_omaps % prev_fia_node.maps
            # if last_tile_maps == 0: # if 0, then prev_fia_node_omaps == prev_fia_node.maps
            #     last_tile_maps = prev_fia_node.maps
            # prev_fia_node_completed_maps = prev_fia_node_omaps - last_tile_maps

            # row_collision_threshold = node.row_start + node.rows_0
            # if node.Conv2DOptions.use_db and node.Conv2DOptions.conv_rows:
            #     row_collision_threshold = node.Conv2DOptions.conv_rows

            # col_collision_threshold = node.col_start + node.cols_0

            # channel_collision_threshold = node.channels
            # if node.type == BuiltinOperator.CONV_2D:
            #     channel_collision_threshold = node.Conv2DOptions.imaps
            # if node.type == BuiltinOperator.FULLY_CONNECTED:
            #     channel_collision_threshold = 1

            # if channel_collision_threshold < prev_fia_node_completed_maps:
            #     return

            # if row_collision_threshold <= prev_fia_node.row_last // prev_fia_node.Conv2DOptions.stride_height:
            #     return

            # if col_collision_threshold <= prev_fia_node.col_last // prev_fia_node.Conv2DOptions.stride_width:
            #     return

            node.Conv2DOptions.fia_collision = 1
            node.FullyConnectedOptions.fia_collision = 1


def populate_nodes(json_subgraphs, preset, graph_activations, weights, aliased_ids, ids_with_dummies, tmp_dir,\
    in_id_to_out_ids, engine_graphs_nx=None, external_inputs=None, external_outputs=None):
    dequantize_op_params = dict() # old_id : (scale, offset, new_id, "input"/"output")
    Nodes = []
    Nodes_to_graph_dict = {} # can be used for referencing past Nodes while populating current/future nodes

    valid_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'supported_ops.json') 
    with open(valid_path) as f:
        valid = json.load(f)
    is_vectorized = lambda x: x in valid and valid[x]['vectorized'][0] >= 1

    # This dict is needed instead of a set because e.g. max pool is a subnode to an identity.
    # So the same input tensor is first checked on the pool and then on the identity.
    # But that means on the identity (the node, not subnode) the input sync will be false.
    # The solution below is to set the sync for all tensors in the node. If this causes any
    # issues, another solution is to ignore subnodes, and just do this for the main nodes.
    # Another possibility is to keep a data structure in the C/C++ like a cache of what tensor
    # is loaded / saved (e.g., by indirect / direct address) to avoid duplicates.
    already_external_producer = {}
    already_external_consumer = {}

    first_fia = False
    prev_fia_node = None

    for g, graph in enumerate(json_subgraphs):
        codes = [_['builtin_code'] for _ in graph['operator_codes']]
        subgraph = graph['subgraphs'][0]
        buffers = graph['buffers']
        sig = graph['signature_defs'][0]


        node = Node()
        Nodes_to_graph_dict[node] = graph
        ops = subgraph['operators']
        opcodes = [str(codes[_['opcode_index']]) for _ in ops]
        tensors = subgraph['tensors']
        opcode = opcodes[0]
        layer_codes = []

        if engine_graphs_nx:
            # Set whether or not this node is going to be offloaded to NX
            node.offloaded = is_split_idx_in_engine_graphs(g, engine_graphs_nx)
            # Most of the below can be skipped, but still need to add input and output tensors so
            # set_io_nodes can find the nodes for these tensors

        node.use_replay = all([is_vectorized(_) for _ in opcodes])
        if not node.use_replay:
            print('WARNING: replay cannot be used on subgraph {}'.format(g))
            for c, code in enumerate(opcodes):
                if not is_vectorized(code):
                    print('\t{} {} is not vectorized'.format(c, code))

        for sig_input in sig['inputs']:
            if sig_input['tensor_index'] in graph_activations:
                input_type = calc_type.from_str(graph_activations[sig_input['tensor_index']]['type'])
                break
        
        for sig_output in sig['outputs']:
            if sig_output['tensor_index'] in graph_activations:
                output_type = calc_type.from_str(graph_activations[sig_output['tensor_index']]['type'])
                break
        
        node.input_description = sig['inputs'][0]['name'].encode()[:DESCRIPTION_CHARS]
        while len(node.input_description) < DESCRIPTION_CHARS:
            node.input_description += b"\0"
        node.output_description = sig['outputs'][0]['name'].encode()[:DESCRIPTION_CHARS]
        while len(node.output_description) < DESCRIPTION_CHARS:
            node.output_description += b"\0"

        input_shapes = []
        for _ in sig['inputs']:
            if 'shape' in tensors[_['tensor_index']]:
                input_shapes += [tensors[_['tensor_index']]['shape']]
                break
        
        output_shapes = [tensors[_['tensor_index']]['shape'] for _ in sig['outputs']]

        input_buffers = []
        for _ in ops[0]['inputs']:
            if 'buffer' in tensors[_]:
                input_buffers += [buffers[tensors[_]['buffer']]]
        multi_input = len(input_buffers) > 1 and not any(['data' in _ for _ in input_buffers])

        REDUCTION = ["ARG_MIN", "ARG_MAX", "SUM", "REDUCE_PROD", "REDUCE_MAX", "REDUCE_MIN", "MEAN"]
        COMPARISON = ["GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL", "EQUAL", "NOT_EQUAL"]
        ACTIVATION = ["LEAKY_RELU", "SOFTMAX", "LOG_SOFTMAX", "RELU", "PRELU", "RELU6", "RELU_N1_TO_1", "RELU_0_TO_1"]
        LUT = ["TANH", "HARD_SWISH", "SILU", "LOGISTIC", "GELU", "ELU", "LOG", "POW", "EXP"]
        POINT = ['ABS', 'NEG', 'RSQRT', 'SQUARED_DIFFERENCE']
        DATA_MOVEMENT = ["SLICE", "STRIDED_SLICE", "GATHER", "TRANSPOSE"]
        DATA_MOVEMENT += ["UNPACK"]
        DATA_MOVEMENT += ["RESHAPE", "EXPAND_DIMS", "SQUEEZE"]
        DATA_MOVEMENT += ["DEPTH_TO_SPACE", "SPACE_TO_DEPTH", 'BATCH_TO_SPACE_ND', 'SPACE_TO_BATCH_ND']
        PADDING = ['PAD', 'PADV2', 'MIRROR_PAD']
        POOLING = ['MAX_POOL_2D', 'AVERAGE_POOL_2D']
        KERNEL_REGULARIZER = ['L2_NORMALIZATION']
        POST_PROCESSING = ['EMBEDDING_LOOKUP']
        BINARY = ['ADD', 'SUB', 'MUL', 'SUB', 'DIV', "SQUARED_DIFFERENCE"]
        TYPE_CAST = ['CAST']
        if g == 0: #TODO if input use eltwise, or take in input 1
            BINARY += ['ADD']
        VALID_SUBNODE = BINARY + PADDING + POOLING + REDUCTION + COMPARISON + ACTIVATION + LUT + POINT + DATA_MOVEMENT + KERNEL_REGULARIZER + POST_PROCESSING + TYPE_CAST
        if g != 0: #TODO
            VALID_SUBNODE += ['ADD']
        VALID_SUBNODE += ['DEPTHWISE_CONV_2D']
        VALID_SUBNODE += ['RESIZE_BILINEAR']
        VALID_SUBNODE += ['RESIZE_NEAREST_NEIGHBOR']
        VALID_SUBNODE += ['MINIMUM', 'MAXIMUM']
        VALID_SUBNODE += ['CAST']

        if opcode in ['CONV_2D', 'DEPTHWISE_CONV_2D']:
            node.type = BuiltinOperator.CONV_2D
            subops = ops[1:]
            subcodes = opcodes[1:]
        elif opcode in ['FULLY_CONNECTED']:
            node.type = BuiltinOperator.FULLY_CONNECTED
            subops = ops[1:]
            subcodes = opcodes[1:]
        elif opcode in ["CONCATENATION", "PACK", "SPLIT", "SPLIT_V", "TILE"]:
            node.type = getattr(BuiltinOperator, f"{opcode}")
            subops = ops[1:]
            subcodes = opcodes[1:]
        elif opcode in ['UNIDIRECTIONAL_SEQUENCE_LSTM']:
            node.type = BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM
            subops = ops[1:]
            subcodes = opcodes[1:]
        elif opcode in ['TRANSPOSE_CONV']:
            node.type = BuiltinOperator.TRANSPOSE_CONV
            subops = ops[1:]
            subcodes = opcodes[1:]
        elif opcode in BINARY + COMPARISON and multi_input:
            node.type = VNNXOperator.IDENTITY
            subops = ops
            subcodes = opcodes
        elif opcode in ["QUANTIZE", "DEQUANTIZE"] or opcode in VALID_SUBNODE or opcode in POST_PROCESSING:
            node.type = VNNXOperator.IDENTITY
            subops = ops
            subcodes = opcodes
        else:
            sys.stderr.write('ERROR: {} not implemented\n'.format(opcode))
            sys.exit(1)

        if node.type == VNNXOperator.IDENTITY:
            layer_codes.append('IDENTITY')
        else:
            layer_codes.append(opcode)

        prev_subop = None
        subnode_tensor_array = []
        
        i = 0
        lut_ops = None
        while i < len(subops):
            patterns = lut_pattern(subops, codes, tensors, buffers, i, opcode)
            if len(patterns) and patterns[0] == i:
                subops_ = [subops[_] for _ in patterns]
                subcodes_ = [subcodes[_] for _ in patterns]
                subcode = "LUT"
                lut_ops = list(zip(subops_, subcodes_, patterns))
            else:
                subops_ = [subops[i]]
                subcode = subcodes[i]
            layer_codes.append(subcode)
            subnode_array, subnode_tensors, prev_subop = populate_subnodes(subcode, lut_ops, subops_, prev_subop, graph_activations, tensors, buffers, weights, dequantize_op_params, aliased_ids, tmp_dir,\
                engine_graphs_nx, external_inputs, external_outputs, already_external_producer, already_external_consumer, node.offloaded, g)
           
            node.subnode_array += subnode_array
            subnode_tensor_array += subnode_tensors
            i += len(subops_)
        # print op and sub_ops as they are actually ran
        # print(g, layer_codes)

        op = ops[0]
        opts = None
        if 'builtin_options' in op:
            opts = op['builtin_options']

        if node.type == BuiltinOperator.SPLIT:
            i_tensor = tensors[op['inputs'][1]]
            
        elif node.type == BuiltinOperator.TRANSPOSE_CONV:
            i_tensor = tensors[op['inputs'][2]]
        else:
            i_tensor = tensors[op['inputs'][0]]
            for input in op['inputs']:
                tensor = tensors[input]
                if 'buffer' in tensor and 'data' in buffers[tensor['buffer']]:
                    pass
                else:
                    i_tensor = tensor
                    break

        o_tensor = tensors[op['outputs'][0]]
        input_offset = 0
        input_scale = 1.0
        if 'quantization' in i_tensor.keys() and 'zero_point' in i_tensor['quantization'].keys():
            input_offset = i_tensor['quantization']['zero_point'][0]
        if 'quantization' in i_tensor.keys() and 'scale' in i_tensor['quantization'].keys():
            input_scale = i_tensor['quantization']['scale'][0]

        output_offset = 0
        output_scale = 1.
        if 'quantization' in o_tensor.keys() and 'zero_point' in o_tensor['quantization'].keys():
            output_offset = o_tensor['quantization']['zero_point'][0]
        if 'quantization' in o_tensor.keys() and 'scale' in o_tensor['quantization'].keys():
            output_scale = o_tensor['quantization']['scale'][0]

        # force undefined batch to 1 TODO move to preprocess
        if i_tensor['shape'][0] == -1:
            i_tensor['shape'][0] = 1
        if o_tensor['shape'][0] == -1:
            o_tensor['shape'][0] = 1

        node.input_offset = input_offset
        node.output_offset = output_offset

        node.input_data_type = calc_type.from_str(i_tensor['type'])
        node.output_data_type = calc_type.from_str(o_tensor['type'])

        input_shape, idims = channels_first_shape(i_tensor['shape'])

        t = Tensor()
        t.type = calc_type.from_str(i_tensor['type'])
        t.shape = input_shape
        t.dims = idims
        t.scale = input_scale
        t.zero = input_offset
        t.id = i_tensor['buffer'] - 1
        t.name = i_tensor['name']
        if not node.offloaded and engine_graphs_nx:
            add_external_producer_consumer_info(t, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)
        node.tensor_array.append(t)

        node.channels, node.m, node.n = 1, 1, 1
        if idims >= 3:
            node.channels = t.shape[-3]
        if idims >= 2:
            node.m = t.shape[-2]
        node.n = t.shape[-1]

        if node.type == BuiltinOperator.CONV_2D:
            conv8 = node.Conv2DOptions
            conv8.direct_dma = 0

            f_tensor = tensors[op['inputs'][1]]
            b_tensor = None
            if len(op['inputs']) > 2 and op['inputs'][2] != -1:
                b_tensor = tensors[op['inputs'][2]]

            k, h, w, c = tuple(f_tensor['shape'])

            filter_shape_dims = [k, c, h, w]
            conv8.filter_shape_dims = filter_shape_dims

            conv8.kernels = k
            conv8.fit_weights = 0
            conv8.use_depthwise = 0
            conv8.group = 1
            if k == 1 and c > 1 and opcode == 'DEPTHWISE_CONV_2D':
                conv8.kernels = c
                conv8.group = c
                conv8.use_depthwise = 1
            # else:
            parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]
            if k*h*w*c <= (fia_weight_shaper_size_kb(parallel_output_maps)*1024):
                conv8.fit_weights = ALL_WEIGHTS

            node.input_strides = [opts['stride_h'], opts['stride_w']]
            conv8.stride_width = opts['stride_w']
            conv8.stride_height = opts['stride_h']
            conv8.dilation_height_factor = opts['dilation_h_factor']
            conv8.dilation_width_factor = opts['dilation_w_factor']
            conv8.fused_activation = 0
            node.activation_max = 127
            node.activation_min = -128
            if opts['fused_activation_function'] == 'RELU':
                node.activation_min = output_offset
            elif opts['fused_activation_function'] == 'RELU6':
                node.activation_min = output_offset
                node.activation_max = round((6 / output_scale) + output_offset)
                node.activation_max = min(node.activation_max, 127)

            conv8.use_strided = 0
            conv8.use_vector = USE_CONV_VECTOR
            conv8.use_fia = conv8.use_vector and USE_CONV_FIA
            conv8.use_db = USE_FIA_DB
            conv8.conv_rows = USE_FIA_DB

            if conv8.use_fia and conv8.stride_width != 1 and not conv8.use_depthwise:
                conv8.use_strided = 1

            pad_hw = None
            pad_h, pad_w = 0, 0

            if opts['padding'] == 'VALID':
                conv8.padding_width = 0
                conv8.padding_height = 0

            elif opts['padding'] == 'SAME':
                # inject identity node with padding before this conv node
                stride_h, stride_w = conv8.stride_height, conv8.stride_width
                dilation_h, dilation_w = conv8.dilation_height_factor, conv8.dilation_width_factor
                kernel_h, kernel_w = h, w

                i_h, i_w = node.m, node.n
                o_h, o_w = tensors[op['outputs'][0]]['shape'][-3], tensors[op['outputs'][0]]['shape'][-2]

                pad_h = max(stride_h * (o_h - 1) - i_h + kernel_h + (kernel_h-1)*(dilation_h-1), 0)
                pad_w = max(stride_w * (o_w - 1) - i_w + kernel_w + (kernel_w-1)*(dilation_w-1), 0)

                conv8.padding_width = 0
                conv8.padding_height = 0

                if pad_w > 0 or pad_h > 0:
                    # these values will be updated to 0 later if an identity+pad or just pad are injected explicitly
                    conv8.padding_width = pad_w
                    conv8.padding_height = pad_h

                    # for FIA, doing padding in injected layer is faster than doing padding in FIA which would force 2D or 3D DMA
                    pad_hw = [pad_h, pad_w]

            # identity should only be injected if any of these are true:
            # 1) pad_hw is set, and there is no previous subgraph, i.e. this Conv node is the first subgraph of the network
            # 2) it is strided FIA and there's no previous subgraph to re-map the inputs, or 
            #    if the prev subgraph's output is mapped to different subgraphs that do not want re-mapped inputs            
            inject_identity = False
            prev_node = None

            # find previous node
            input_id = node.tensor_array[0].id
            for pnode in Nodes:
                out_id = pnode.tensor_array[-1].id
                if input_id == out_id:
                    prev_node = pnode
                    break

            if pad_hw is not None:
                if prev_node is None:
                    inject_identity = True
                elif op['inputs'][0] in in_id_to_out_ids.keys():
                    inject_identity = True
                elif prev_node.num_outputs > 1:
                    inject_identity = True
                elif input_id in in_id_to_out_ids.keys():
                    # TODO the input to this node is shared with another node, should skip and determine after populate_nodes if pad can be injected or identity must be injected
                    # until implemented, just continue inject_identity for these cases
                    inject_identity = True
                else: #input id not in in_id_to_out_ids.keys                    
                    inject_pad_subnode_to_previous_node(prev_node, node, ids_with_dummies, preset, opcode, weights, Nodes_to_graph_dict[prev_node])
                # elif len(prev_node.subnode_array) == 0:
                #     inject_identity = True
                # elif prev_node.subnode_array[-1].type != BuiltinOperator.PAD:
                #     inject_identity = True

            if conv8.use_strided:
                if prev_node is None:
                    inject_identity = True
                else:
                    # get prev_node output buffer idand search for it among all other subgraphs inputs
                    # it should not occur more than once, otherwise inject identity
                    # TODO if other node it outputs to is strided as well, then an identity need not be injected (same idea as pad optimization)
                    output_id = prev_node.tensor_array[-1].id
                    found = False
                    for temp_g in json_subgraphs:
                        temp_g_inputs = temp_g['subgraphs'][0]['operators'][0]['inputs']
                        if output_id in temp_g_inputs:
                            if found is False:
                                found = True
                            else:
                                inject_identity = True
                                break
                        
                if inject_identity is False:
                    prev_node.output_strides = [conv8.stride_height, conv8.stride_width]

            # If offloading this node to another processor anyway, no need to add an identity
            if node.offloaded:
                inject_identity = False

            if inject_identity:
                with exception_catcher( node.type, len(Nodes)):
                    Nodes = inject_dummy_identity(Nodes, ids_with_dummies, 
                                                            node,
                                                            # op,
                                                            preset, 
                                                            # graph, 
                                                            opcode,
                                                            pad_hw,
                                                            inject_strided=conv8.use_strided)

            # Currently disabling weight shaper double buffer for FC
            is_fc = (node.n == 1 and node.m == 1)
            conv8.mxp_double_buffer = 1
            conv8.split_weight_shaper_buffers = 0
            if (not is_fc) and (not conv8.fit_weights) and conv8.mxp_double_buffer and (not conv8.use_depthwise):
                conv8.split_weight_shaper_buffers = 1
            tile = allocation(node, preset, opcode)

            # if not full rows, then don't double buffer
            # previously disabling double buffer for k > 3 because the reduced scratchpad size causes
            # # the first k=6 layer of yolov5 to not fit all columns, and that gives incorrect results
            if node.n != tile[-1]:
                conv8.mxp_double_buffer = 0
                conv8.split_weight_shaper_buffers = 0
                tile = allocation(node, preset, opcode)

            assert not (tile is None)

            conv8.first_fia = 0
            conv8.last_fia = 0
            conv8.fia_collision = 0
            if conv8.use_fia:
                if first_fia is False:
                    conv8.first_fia = 1
                    first_fia = True

                conv8.last_fia = 1
                
                if prev_fia_node:
                    prev_fia_node.Conv2DOptions.last_fia = 0
                    prev_fia_node.FullyConnectedOptions.last_fia = 0

                check_node_for_collision(node, Nodes, prev_fia_node)

                prev_fia_node = node

            with exception_catcher( node.type, len(Nodes), op['inputs'][0]):
                effective_scale, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(op, tensors)

            if len(output_multiplier) == 1 and conv8.kernels > 1:
                output_multiplier = [output_multiplier[0] for _ in range(conv8.kernels)]
            if len(output_shift) == 1 and conv8.kernels > 1:
                output_shift = [output_shift[0] for _ in range(conv8.kernels)]

            node.input_offset = input_offset
            node.output_offset = output_offset

            filter_data = get_numpy_data(f_tensor, buffers).transpose((0,3,1,2))

            if conv8.use_fia and conv8.use_depthwise and conv8.dilation_height_factor != 1 and h != 1:
                dilated_height = ((conv8.dilation_height_factor-1) * (h-1)) + h
                h = dilated_height
                filter_shape_dims = [k, c, h, w]
                conv8.filter_shape_dims = filter_shape_dims
                dilated_data = np.zeros(filter_shape_dims, dtype=np.int8)
                dilated_data[:, :, ::conv8.dilation_height_factor, :] = filter_data
                filter_data = dilated_data
                conv8.dilation_height_factor = 1

            node.output_multiplier = len(weights)
            fmt = "{}i".format(len(output_multiplier))
            weights += struct.pack(fmt, *output_multiplier)

            node.output_shift = len(weights)
            fmt = "{}i".format(len(output_shift))
            weights += struct.pack(fmt, *output_shift)

            bias_data = np.zeros((k,), dtype=np.int64)
            if not (b_tensor is None):
                bias_data = get_numpy_data(b_tensor, buffers).astype(np.int64)

            if conv8.use_fia or USE_PRECALC:
                if opcode == 'CONV_2D':
                    bias_data += precalculate_filter_input_bias(filter_data, input_offset, reduce=False)
                else:
                    bias_data += precalculate_filter_input_bias(filter_data, input_offset, reduce=True)
            if PRECALC_OUTPUT:
                bias_data += precalculate_output_offset_bias(effective_scale, output_offset)
                if conv8.use_fia:
                    node.output_offset = 0

            bias_data = bias_data.clip(np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            bias_data = bias_data.astype(np.int32)

            conv8.bias_data = len(weights)
            fmt = "{}i".format(len(bias_data))
            weights += struct.pack(fmt, *bias_data)

            _, _, _, _, _, rows, columns = tile
            weight_pad = 0
            if conv8.use_fia and conv8.use_depthwise and node.m == rows and node.n == columns: # check if full maps, if so pad weights here for 1D DMA
                dilated_filter_height = ((h-1)*conv8.dilation_height_factor) + 1
                padded_input_height = node.m + conv8.padding_height
                output_shaper_height = (padded_input_height - dilated_filter_height) + 1                
                conv8.fit_weights = 0
                parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]
                weight_pad = min(parallel_output_maps - 1, output_shaper_height)
            
            # if not full input maps, then set weight pad to be parallel_output_maps-1. We want to avoid splat in FIA. Offset appropriately when DMAing in FIA code
            # TODO should fit all weights if possible (same for above case)
            elif conv8.use_fia and conv8.use_depthwise and (node.m != rows or node.n != columns): 
                conv8.fit_weights = 0
                parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]
                weight_pad = parallel_output_maps-1

            conv8.filter_data = len(weights)
            if t.type==calc_type.UINT8:
                fmt = "{}B".format(len(filter_data.flatten()))
            else:
                fmt = "{}b".format(len(filter_data.flatten()))
            # fmt = "{}b".format(len(filter_data.flatten()))
            
            if conv8.use_fia:
                filter_data = conv_pack_weights(filter_data, node.maps, conv8.use_depthwise, weight_pad=weight_pad)
                if weight_pad != 0:                    
                    if t.type==calc_type.UINT8:
                        fmt = "{}B".format(len(filter_data.flatten()))
                    else:
                        fmt = "{}b".format(len(filter_data.flatten()))
            weights += struct.pack(fmt, *filter_data.flatten())

            # used for FIA
            # TODO maybe redundant as we pack bias, output multiplier, and output shift earlier
            # TODO future optimization if VNNX is too large then clean up redundancy
            # force 16-byte alignment for records
            if len(weights) % 16 != 0:
                filler = 16 - (len(weights) % 16)
                fmt = "{}x".format(filler)
                weights += struct.pack(fmt)
            conv8.quantization_records = len(weights)
            fmt = "<2iB7x" # little-endian 16 bytes 
            # bias int32 - 4 bytes
            # multiplier int32 - 4 bytes
            # shift uint8 - 1 byte
            # pad with 7 bytes for 16byte align
            for kernel in range(conv8.kernels):
                total_shift_minus1 = (31 - output_shift[kernel]) - 1
                weights += struct.pack(fmt, bias_data[kernel], output_multiplier[kernel], total_shift_minus1)

        elif node.type ==  BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM:

            allocation(node, preset, opcode)

        elif node.type ==  BuiltinOperator.TRANSPOSE_CONV:

            conv8 = node.Conv2DOptions
            conv8.direct_dma = 0
            conv8.fit_weights = 0
            conv8.use_depthwise = 0
            conv8.group = 1
            conv8.dilation_height_factor = 1
            conv8.dilation_width_factor = 1

            f_tensor = tensors[op['inputs'][1]]
            b_tensor = None
            if len(op['inputs']) > 3 and op['inputs'][3] != -1:
                b_tensor = tensors[op['inputs'][3]]

            k, h, w, c = tuple(f_tensor['shape'])
            filter_shape_dims = [k, c, h, w]
            conv8.filter_shape_dims = filter_shape_dims
            parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]
            if k*h*w*c <= (fia_weight_shaper_size_kb(parallel_output_maps)*1024):
                conv8.fit_weights = ALL_WEIGHTS

            conv8.kernels = k
            conv8.use_vector = USE_CONV_VECTOR
            conv8.use_fia = USE_CONV_VECTOR and USE_CONV_FIA
            conv8.use_db = USE_FIA_DB
            conv8.conv_rows = USE_FIA_DB

            conv8.use_strided = 0

            node.input_strides = [opts['stride_h'], opts['stride_w']]
            conv8.stride_width = opts['stride_w']
            conv8.stride_height = opts['stride_h']
            conv8.fused_activation = 0
            node.activation_max = 127
            node.activation_min = -128
            if opts['fused_activation_function'] == 'RELU':
                node.activation_min = output_offset
            elif opts['fused_activation_function'] == 'RELU6':
                node.activation_min = output_offset
                node.activation_max = round((6 / output_scale) + output_offset)
                node.activation_max = min(node.activation_max, 127)

            # pad (or modified pad) should only be injected if stride_check is True
            stride_check = (conv8.stride_height==conv8.stride_width and conv8.stride_width <= 2)
            pad_hw = None
            pad_h, pad_w = 0, 0
            if opts['padding'] == 'VALID':
                conv8.padding_width = 0
                conv8.padding_height = 0

                if conv8.use_fia and stride_check:
                    _, _, kh, kw = conv8.filter_shape_dims
                    pl = kw - 1
                    pr = kw - 1
                    pu = kh - 1
                    pd = kh - 1
                    pad_w = pl + pr
                    pad_h = pu + pd

                    # these values will be updated to 0 later if an identity+pad or just pad are injected explicitly
                    conv8.padding_width = pad_w
                    conv8.padding_height = pad_h

                    if pad_w != 0 or pad_h != 0:
                        pad_hw = [pad_h, pad_w]

            elif opts['padding'] == 'SAME':
                stride_h, stride_w = conv8.stride_height, conv8.stride_width
                kernel_h, kernel_w = conv8.filter_shape_dims[2], conv8.filter_shape_dims[3]

                i_h, i_w = i_tensor['shape'][-3], i_tensor['shape'][-2]
                o_h, o_w = o_tensor['shape'][-3], o_tensor['shape'][-2]

                if conv8.use_fia:
                    pad_h = ((i_h * stride_h) + (kernel_h-stride_h)) - o_h
                    pad_w = ((i_w * stride_w) + (kernel_w-stride_w)) - o_w
                    conv8.padding_width  = pad_w
                    conv8.padding_height = pad_h

                    if (pad_w > 0 or pad_h > 0) and stride_check:
                        pl = (kernel_w - (pad_w//2) - 1)
                        pr = (kernel_w - ((pad_w//2) + (pad_w%2)) - 1)
                        pu = (kernel_h - (pad_h//2) - 1)
                        pd = (kernel_h - ((pad_h//2) + (pad_h%2)) - 1)
                        pad_w = pl + pr
                        pad_h = pu + pd

                        # these values will be updated to 0 later if an identity+pad or just pad are injected explicitly
                        conv8.padding_width = pad_w
                        conv8.padding_height = pad_h

                        # for FIA, doing padding in injected layer is faster than doing padding in FIA which would force 2D or 3D DMA
                        if pad_w != 0 or pad_h != 0:
                            pad_hw = [pad_h, pad_w]

                else: # not FIA
                    pad_h = ((i_h * stride_h) + (kernel_h-1)) - o_h
                    pad_w = ((i_w * stride_w) + (kernel_w-1)) - o_w

                    conv8.padding_width = pad_w // 2
                    conv8.padding_height = pad_h // 2

            # identity+pad or just pad should only be injected if either:
            # 1) pad_hw is set and stride_check is True, and there is no previous subgraph, i.e. this Conv node is the first subgraph of the network
            # 2) strides are 2,2 and so the input must be dilated with a modified Pad
            inject_identity = False
            prev_node = None

            # find previous node
            input_id = node.tensor_array[0].id
            for pnode in Nodes:
                out_id = pnode.tensor_array[-1].id
                if input_id == out_id:
                    prev_node = pnode
                    break

            if pad_hw is not None or (stride_check and conv8.stride_width == 2):
                if prev_node is None:
                    inject_identity = True
                elif input_id in in_id_to_out_ids.keys():
                    # TODO the input to this node is shared with another node, should skip and determine after populate_nodes if pad can be injected or identity must be injected
                    # until implemented, just continue inject_identity for these cases
                    inject_identity = True
                else: #input id not in in_id_to_out_ids.keys                    
                    inject_pad_subnode_to_previous_node(prev_node, node, ids_with_dummies, preset, opcode, weights, Nodes_to_graph_dict[prev_node], transpose_dilate=[conv8.stride_height, conv8.stride_width])
                # elif len(prev_node.subnode_array) == 0:
                #     inject_identity = True
                # elif prev_node.subnode_array[-1].type != BuiltinOperator.PAD:
                #     inject_identity = True

            if inject_identity:
                with exception_catcher( node.type, len(Nodes)):
                    Nodes = inject_dummy_identity(Nodes, ids_with_dummies, 
                                                            node,
                                                            # op,
                                                            preset, 
                                                            # graph, 
                                                            opcode,
                                                            pad_hw,
                                                            inject_strided=0,
                                                            transpose_dilate=[conv8.stride_height, conv8.stride_width])

            # if doing FIA and stride_check is True, then TransposeConv can be run as regular Conv, and use Conv2D FIA C code
            if conv8.use_fia and stride_check:
                node.type = BuiltinOperator.CONV_2D

            effective_scale, output_multiplier, output_shift = get_effective_quantized_multiplier_from_tensors(i_tensor, o_tensor, f_tensor)

            if len(output_multiplier) == 1 and conv8.kernels > 1:
                output_multiplier = [output_multiplier[0] for _ in range(conv8.kernels)]
            if len(output_shift) == 1 and conv8.kernels > 1:
                output_shift = [output_shift[0] for _ in range(conv8.kernels)]

            node.input_offset = input_offset
            node.output_offset = output_offset

            conv8.imaps = -1
            conv8.mxp_double_buffer = 1
            conv8.split_weight_shaper_buffers = 0
            tile = allocation(node, preset, opcode)
            assert not (tile is None)

            conv8.first_fia = 0
            conv8.last_fia = 0
            conv8.fia_collision = 0
            if conv8.use_fia:
                if first_fia is False:
                    conv8.first_fia = 1
                    first_fia = True

                conv8.last_fia = 1
                
                if prev_fia_node:
                    prev_fia_node.Conv2DOptions.last_fia = 0
                    prev_fia_node.FullyConnectedOptions.last_fia = 0

                check_node_for_collision(node, Nodes, prev_fia_node)

                prev_fia_node = node

            filter_data = get_numpy_data(f_tensor, buffers).transpose((0,3,1,2))
            if conv8.use_fia:
                # rotate 2d weights 90 degrees clockwise
                filter_data = filter_data.transpose((0,1,3,2))
                filter_data = np.flip(filter_data, axis=3)
                # rotate 90 more degrees
                filter_data = filter_data.transpose((0,1,3,2))
                filter_data = np.flip(filter_data, axis=3)

            node.output_multiplier = len(weights)
            fmt = "{}i".format(len(output_multiplier))
            weights += struct.pack(fmt, *output_multiplier)

            node.output_shift = len(weights)
            fmt = "{}i".format(len(output_shift))
            weights += struct.pack(fmt, *output_shift)

            bias_data = np.zeros((k,), dtype=np.int64)
            if not (b_tensor is None):
                bias_data = get_numpy_data(b_tensor, buffers)

            if USE_PRECALC:
                # bias_data += precalculate_filter_input_bias(filter_data, input_offset, reduce=True)
                bias_data += precalculate_filter_input_bias(filter_data, input_offset, reduce=False)
            if PRECALC_OUTPUT:
                bias_data += precalculate_output_offset_bias(effective_scale, output_offset)
                if conv8.use_fia:
                    node.output_offset = 0
            conv8.bias_data = len(weights)
            fmt = "{}i".format(len(bias_data))
            weights += struct.pack(fmt, *bias_data)

            conv8.filter_data = len(weights)
            fmt = "{}b".format(len(filter_data.flatten()))
            if conv8.use_fia:
                filter_data = conv_pack_weights(filter_data, node.maps, is_transpose=True)
            weights += struct.pack(fmt, *filter_data.flatten())

            # used for FIA
            # TODO maybe redundant as we pack bias, output multiplier, and output shift earlier
            # TODO future optimization if VNNX is too large then clean up redundancy
            # force 16-byte alignment for records
            if len(weights) % 16 != 0:
                filler = 16 - (len(weights) % 16)
                fmt = "{}x".format(filler)
                weights += struct.pack(fmt)
            conv8.quantization_records = len(weights)
            fmt = "<2iB7x" # little-endian 16 bytes 
            # bias int32 - 4 bytes
            # multiplier int32 - 4 bytes
            # shift uint8 - 1 byte
            # pad with 7 more bytes for 16 byte aligned record
            for kernel in range(conv8.kernels):
                total_shift_minus1 = (31 - output_shift[kernel]) - 1
                weights += struct.pack(fmt, bias_data[kernel], output_multiplier[kernel], total_shift_minus1)

        elif node.type == BuiltinOperator.FULLY_CONNECTED:

            if idims >= 3:
                node.channels = t.shape[-2]
                node.m = t.shape[-1]
                node.n = t.shape[-3]
            elif idims >= 2:
                node.channels = 1
                node.m = t.shape[-2]
                node.n = t.shape[-1]
            elif idims >= 1:
                node.channels = 1
                node.m = 1
                node.n = t.shape[-1]

            f_tensor = tensors[op['inputs'][1]]
            b_tensor = None
            if len(op['inputs']) > 2 and op['inputs'][2] != -1:
                b_tensor = tensors[op['inputs'][2]]

            fc8 = node.FullyConnectedOptions

            output_depth, accum_depth = tuple(f_tensor['shape'])
            fc8.filter_shape_dims = [output_depth, accum_depth]
            fc8.use_fia = USE_CONV_VECTOR and USE_CONV_FIA
            fc8.mxp_double_buffer = 1

            node.activation_max = 127
            node.activation_min = -128
            if opts['fused_activation_function'] == 'RELU':
                node.activation_min = output_offset

            with exception_catcher( node.type, len(Nodes), op['inputs'][0]):
                effective_scale, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(op, tensors)

            node.input_offset = input_offset
            node.output_offset = output_offset

            node.output_multiplier = len(weights)
            fmt = "{}i".format(len(output_multiplier))
            weights += struct.pack(fmt, *output_multiplier)

            node.output_shift = len(weights)
            fmt = "{}i".format(len(output_shift))
            weights += struct.pack(fmt, *output_shift)

            tile = allocation(node, preset, opcode)
            assert not (tile is None)

            fc8.first_fia = 0
            fc8.last_fia = 0
            fc8.fia_collision = 0
            if fc8.use_fia:
                if first_fia is False:
                    fc8.first_fia = 1
                    first_fia = True

                fc8.last_fia = 1
                
                if prev_fia_node:
                    prev_fia_node.Conv2DOptions.last_fia = 0
                    prev_fia_node.FullyConnectedOptions.last_fia = 0

                check_node_for_collision(node, Nodes, prev_fia_node)

                prev_fia_node = node


            filter_data = get_numpy_data(f_tensor, buffers)

            bias_data = np.zeros((output_depth,), dtype=np.int64)
            if not (b_tensor is None):
                bias_data = get_numpy_data(b_tensor, buffers)

            if USE_PRECALC:
                res = precalculate_filter_input_bias(filter_data, input_offset)
                bias_data = bias_data + res
            if PRECALC_OUTPUT:
                bias_data += precalculate_output_offset_bias(effective_scale, output_offset)
                if fc8.use_fia:
                    node.output_offset = 0
            fc8.bias_data = len(weights)
            fmt = "{}i".format(len(bias_data))
            weights += struct.pack(fmt, *bias_data)

            fc8.filter_data = len(weights)
            if fc8.use_fia: #TODO handle case were cols varies or not evenly divisible 
                o,a = filter_data.shape
                o_ = (o + node.cols-1) // node.cols
                reshaped_data = np.zeros((o_ * node.cols * a), dtype=filter_data.dtype)
                for i in range(o_):
                    f = filter_data[i*node.cols: (i+1)*node.cols]
                    f_ = f.transpose((1,0)).flatten()
                    reshaped_data[i*node.cols*a:(i*node.cols+f.shape[0])*a] = f_
                filter_data = reshaped_data.reshape((o_, node.cols, a))

            fmt = "{}b".format(len(filter_data.flatten()))
            weights += struct.pack(fmt, *filter_data.flatten())

            # used for FIA
            # TODO maybe redundant as we pack bias, output multiplier, and output shift earlier
            # TODO future optimization if VNNX is too large then clean up redundancy
            # TODO for fully connected, clean up redundancy of storing same multiplier+shift for each record
            # force 16-byte alignment for records
            if len(weights) % 16 != 0:
                filler = 16 - (len(weights) % 16)
                fmt = "{}x".format(filler)
                weights += struct.pack(fmt)
            fc8.quantization_records = len(weights)
            fmt = "<2iB7x" # little-endian 16 bytes 
            # bias int32 - 4 bytes
            # multiplier int32 - 4 bytes
            # shift uint8 - 1 byte
            # pad with 7 more bytes for 16 byte aligned record
            total_shift_minus1 = (31 - output_shift[0]) - 1
            for out in range(output_depth):
                weights += struct.pack(fmt, bias_data[out], output_multiplier[0], total_shift_minus1)

        elif node.type == VNNXOperator.IDENTITY: #TODO not adding explicit tensor
            if multi_input:
                if node.subnode_array[0].type == VNNXOperator.ELTWISE:
                    node.subnode_array[0].eltwise8.swap = False

                for t in op['inputs'][1:]:
                    tensor = tensors[t]
                    offset = tensor['quantization']['zero_point'][0]
                    scale = tensor['quantization']['scale'][0]
                    shape, dims = channels_first_shape(tensor['shape'])

                    tn = Tensor()
                    tn.type = calc_type.from_str(tensor['type'])
                    tn.shape = shape
                    tn.dims = dims
                    tn.scale = scale
                    tn.zero = offset
                    tn.id = tensor['buffer'] - 1
                    tn.name = tensor['name']
                    if not node.offloaded and engine_graphs_nx:
                        add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)
                    node.tensor_array.append(tn)

                node.m, node.n = max(node.m, shape[-2]), max(node.n, shape[-1]) #increase size (to handle channelwise broadcast) 

            node.num_inputs = len(node.tensor_array)
            allocation(node, preset, opcode)

        elif node.type == BuiltinOperator.CONCATENATION:
            o_tensor = tensors[op['outputs'][0]]
            axis = op['builtin_options']['axis']
            node.ConcatOptions.axis = channels_first_axis(axis, len(o_tensor['shape']))

            for t in op['inputs'][1:]:
                tensor = tensors[t]
                offset = tensor['quantization']['zero_point'][0]
                scale = tensor['quantization']['scale'][0]
                shape, dims = channels_first_shape(tensor['shape'])

                tn = Tensor()
                tn.type = calc_type.from_str(tensor['type'])
                tn.shape = shape
                tn.dims = dims
                tn.scale = scale
                tn.zero = offset
                tn.id = tensor['buffer'] - 1
                tn.name = tensor['name']
                if not node.offloaded and engine_graphs_nx:
                    add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)
                node.tensor_array.append(tn)

            output_shape, odims = channels_first_shape(o_tensor['shape'])

            node.channels = 1
            if odims > 2:
                node.channels = output_shape[-3]
            node.m = 1
            if odims > 1:
                node.m = output_shape[-2]
            node.n = output_shape[-1]
            node.num_inputs = len(node.tensor_array)
            allocation(node, preset, opcode)

        elif node.type == BuiltinOperator.PACK:
            idims = len(i_tensor['shape'])
            odims = len(o_tensor['shape'])
            node.PackOptions.axis = channels_first_axis(opts['axis'], odims) - odims
            node.PackOptions.count = opts['values_count']
            node.PackOptions.dims = idims

            assert(check_pack(i_tensor['shape'], o_tensor['shape'], opts['axis'], node.PackOptions.axis, node.PackOptions.count))

            for t in op['inputs'][1:]:
                tensor = tensors[t]
                offset = tensor['quantization']['zero_point'][0]
                scale = tensor['quantization']['scale'][0]
                shape, dims = channels_first_shape(tensor['shape'])

                tn = Tensor()
                tn.type = calc_type.from_str(tensor['type'])
                tn.shape = shape
                tn.dims = dims
                tn.scale = scale
                tn.zero = offset
                tn.id = tensor['buffer'] - 1
                tn.name = tensor['name']
                if not node.offloaded and engine_graphs_nx:
                    add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)
                node.tensor_array.append(tn)

            node.num_inputs = len(node.tensor_array)

            allocation(node, preset, opcode)

        elif node.type == BuiltinOperator.UNPACK:
            input_shape, idims = channels_first_shape(i_tensor['shape'])
            node.PackOptions.axis = channels_first_axis(opts['axis'], idims) - idims
            node.PackOptions.count = opts['num']
            node.PackOptions.dims = idims

            assert(check_unpack(i_tensor['shape'], o_tensor['shape'],
                                opts['axis'], node.PackOptions.axis, opts['num']))
            allocation(node, preset, opcode)
       
        elif node.type in [BuiltinOperator.SPLIT, BuiltinOperator.SPLIT_V]:
            if node.type == BuiltinOperator.SPLIT_V:
                axis_tensor = tensors[op['inputs'][2]]
            else:
                axis_tensor = tensors[op['inputs'][0]]
            dtype = np.dtype(axis_tensor['type'].lower())
            raw_data = bytearray(buffers[axis_tensor['buffer']]['data'])
            axis = np.frombuffer(raw_data, dtype=dtype)[0]

            num_splits = len(op['outputs'])
            if node.type == BuiltinOperator.SPLIT_V:
                split_tensor = tensors[op['inputs'][1]]
                dtype = np.dtype(split_tensor['type'].lower())
                raw_data = bytearray(buffers[split_tensor['buffer']]['data'])
                splits = np.frombuffer(raw_data, dtype=dtype)
            else:
                num_splits = len(op['outputs'])
                splits = [i_tensor['shape'][axis] // num_splits for _ in range(num_splits)]

            idims = len(i_tensor['shape'])
            node.SplitOptions.axis = channels_first_axis(axis, idims) - idims
            node.num_outputs = num_splits

            node.SplitOptions.splits = len(weights)
            fmt = "{}i".format(len(splits))
            weights += struct.pack(fmt, *splits)

            allocation(node, preset, opcode)

        elif node.type == BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
            i_tensor = tensors[op['inputs'][0]]
            o_tensor = tensors[op['outputs'][0]]
            s_tensor = tensors[op['inputs'][1]]

            resize = node.ResizeOptions
            resize.mode = resize_mode.NEAREST 

            new_size_data = get_numpy_data(s_tensor, buffers)
            resize.scale = (new_size_data / i_tensor['shape'][1:3]).tolist() # height and width scales
            # resize.scale = ( i_tensor['shape'][1:3] /new_size_data).tolist()
            # print("resize.scale == ", resize.scale)

            allocation(node, preset, opcode)
        elif node.type == BuiltinOperator.RESIZE_BILINEAR:
            i_tensor = tensors[op['inputs'][0]]
            o_tensor = tensors[op['outputs'][0]]
            s_tensor = tensors[op['inputs'][1]]

            resize = node.ResizeOptions
            resize.mode = resize_mode.LINEAR

            new_size_data = get_numpy_data(s_tensor, buffers)
            resize.scale = (new_size_data / i_tensor['shape'][1:3]).tolist() # height and width scales
            # resize.scale = ( i_tensor['shape'][1:3] /new_size_data).tolist()


            # TODO m and n are currently expected to be shapes of the outputs in the c code
            node.m = output_shapes[0][-3]
            node.n = output_shapes[0][-2]
            allocation(node, preset, opcode)

        elif node.type == BuiltinOperator.TILE:
            i_tensor = tensors[op['inputs'][0]]
            o_tensor = tensors[op['outputs'][0]]
            m_tensor = tensors[op['inputs'][1]]
            multipliers = get_numpy_data(m_tensor, buffers) 
            if idims == 1:
                multipliers =  [1, 1, 1, multipliers[0]]
            elif idims == 2: 
                multipliers =  [1, 1, multipliers[0], multipliers[1]]
            elif idims == 3: #NHWC -> NCHW
                multipliers =  [1, multipliers[-1], multipliers[-3], multipliers[-2]]
            else: #NHWC -> NCHW
                multipliers =  [multipliers[-4], multipliers[-1], multipliers[-3], multipliers[-2]]
            node.TileOptions.tile = multipliers

            for t in op['inputs'][1:]:
                tensor = tensors[t]
                offset = 0
                scale = 0
                shape, dims = channels_first_shape(tensor['shape'])

                tn = Tensor()
                tn.type = calc_type.from_str(tensor['type'])
                tn.shape = shape
                tn.dims = dims
                tn.scale = 0
                tn.zero = 0
                tn.id = tensor['buffer'] - 1
                tn.name = tensor['name']
                if not node.offloaded and engine_graphs_nx:
                    add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)
                node.tensor_array.append(tn)

            node.num_inputs = len(node.tensor_array)

            allocation(node, preset, opcode)
        
        else:
            sys.stderr.write("ERROR: no parsing for {}\n".format(node.type))
            sys.exit(1)
                    
        node.num_outputs = len(op['outputs'])
        for t in op['outputs']:
            tensor = tensors[t]
            offset = tensor['quantization'].get('zero_point', [0])[0]
            scale = tensor['quantization'].get('scale', [0.0])[0]
            shape, dims = channels_first_shape(tensor['shape'])

            tn = Tensor()
            tn.type = calc_type.from_str(tensor['type'])
            tn.shape = shape
            tn.dims = dims
            tn.scale = scale
            tn.zero = offset
            tn.id = tensor['buffer'] - 1
            tn.name = tensor['name']
            if not node.offloaded and engine_graphs_nx:
                add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)

            if node.type == BuiltinOperator.CONV_2D and node.Conv2DOptions.use_fia:
                if node.Conv2DOptions.use_depthwise and node.Conv2DOptions.stride_width != 1:
                    inject_strided_slice(node, tn, subnode_tensor_array, ids_with_dummies)
                    
                    tile = allocation(node, preset, opcode) # re-allocate as stride_width forced to 1
                    assert not (tile is None)

            node.tensor_array.append(tn)

        assert(node.num_inputs + node.num_outputs == len(node.tensor_array))

        # TODO does this need to apply to injected nodes?
        if node.type == VNNXOperator.IDENTITY and node.num_inputs == 1: #injected identity, rename tensors TODO fix for multi-input
            node_input_name = node.tensor_array[0].name
            node.tensor_array[1].name = node_input_name + '.id'
            node.tensor_array[1].shape = node.tensor_array[0].shape
            for s, t in enumerate(subnode_tensor_array):
                if t.name == node_input_name:
                    subnode_tensor_array[s].name = node.tensor_array[1].name

        node.tensor_array += subnode_tensor_array
        node.num_tensors = len(node.tensor_array)
        
        Nodes.append(node)

    Nodes = update_nodes_with_dequant_params(Nodes, dequantize_op_params)
    return Nodes


def hard_swish(x):
    if x < -3:
        return 0
    elif x > 3:
        return x
    else:
        return x * (x + np.float32(3.)) * np.float32(1./ 6.)


def lut_func(code, scale=None, param=None):
    if code == "SILU":
        fn = lambda s: s*sigmoid(s)
    elif code == "QUANTIZE":
        fn = lambda s: s
    elif code == "LOGISTIC":
        fn = lambda s: sigmoid(s)
    elif code == "HARD_SWISH":
        fn = lambda s: s * tf.nn.relu6(s + np.float32(3.)) * np.float32(1./ 6.)
        # fn = hard_swish
    elif code == 'LEAKY_RELU':
        fn = lambda s: tf.nn.leaky_relu(s, scale)
    elif code == 'RELU':
        fn = lambda s: tf.nn.relu(s)
    elif code == 'RELU6':
        fn = lambda s: tf.nn.relu6(s)
    elif code == 'RELU_0_TO_1':
        fn = lambda s: min(1.0, tf.nn.relu(s))
    elif code == "MUL":
        fn = lambda s: tf.multiply(s, scale)
    elif code == "ADD":
        fn = lambda s: tf.add(s, scale)
    elif code == "SUB":
        fn = lambda s: tf.subtract(s, scale)
    elif code == "POST_PROCESSING":
        fn = lambda s: tf.gather(s, param, axis=1)
    elif code == "SQUARED_DIFFERENCE":
        fn = lambda s: tf.math.squared_difference(s, scale)
    return fn


def populate_subnodes(subcode, lut_ops, subops, prev_subop, graph_activations, tensors, buffers, weights, dequantize_op_params, aliased_ids, tmp_dir,\
engine_graphs_nx, external_inputs, external_outputs, already_external_producer, already_external_consumer, node_offloaded, subgraph_idx):
    subnode_array = []
    subnode_tensor_array = []
    sn = Subnode()
    subop = subops[0]
    input_buffers = []
    for _ in subop['inputs']:
        if 'buffer' in tensors[_]:
            input_buffers += [buffers[tensors[_]['buffer']]]
    multi_input = len(input_buffers) > 1 and not any(['data' in _ for _ in input_buffers])

    i_acts, _ = op_activations(subop, graph_activations)
    _, o_acts = op_activations(subops[-1], graph_activations)

    i_tensor = graph_activations[i_acts[0]]
    o_tensor = graph_activations[o_acts[0]]

    # force undefined batch to 1 TODO move to preprocess
    if i_tensor['shape'][0] == -1:
        i_tensor['shape'][0] = 1
    if o_tensor['shape'][0] == -1:
        o_tensor['shape'][0] = 1
    
    input_offset = 0
    input_scale = 1.
    if 'quantization' in i_tensor.keys() and 'zero_point' in i_tensor['quantization'].keys():
        input_offset = i_tensor['quantization']['zero_point'][0]
    if 'quantization' in i_tensor.keys() and 'scale' in i_tensor['quantization'].keys():
        input_scale = i_tensor['quantization']['scale'][0]

    output_offset = 0
    output_scale = 1.
    if 'quantization' in o_tensor.keys() and 'zero_point' in o_tensor['quantization'].keys():
        output_offset = o_tensor['quantization']['zero_point'][0]
    if 'quantization' in o_tensor.keys() and 'scale' in o_tensor['quantization'].keys():
        output_scale = o_tensor['quantization']['scale'][0]

    output_type = o_tensor['type']
    sn.output_data_type = calc_type.from_str(output_type)

    sn.output_offset = output_offset

    input_shape, idims = channels_first_shape(i_tensor['shape'])
    output_shape, odims = channels_first_shape(o_tensor['shape'])

    input_type = i_tensor['type']
    sn.input_data_type = calc_type.from_str(input_type)

    if not subcode in ['LUT', 'QUANTIZE', 'GATHER', 'CAST', 'TOPK_V2', 'RESHAPE', 'NEG']:
        assert(sn.input_data_type == calc_type.INT8 or sn.input_data_type == calc_type.UINT8)

    sn.activation_min = -128
    sn.activation_max = 127

    tn = Tensor()
    tn.type = sn.input_data_type
    tn.shape = input_shape
    tn.dims = idims
    tn.scale = input_scale
    tn.zero = input_offset
    tn.id = i_tensor['buffer'] - 1
    tn.name = i_tensor['name']
    if not node_offloaded and engine_graphs_nx:
        add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, subgraph_idx)
    sn.tensor_array.append(tn)

    opts = None
    if 'builtin_options' in subop:
        opts = subop['builtin_options']

    if subcode == 'AVERAGE_POOL_2D': # functions as MEAN operator
        if output_shape[-2] == 1 and output_shape[-1] == 1:
            subcode = 'MEAN'


    if subcode == 'DEPTHWISE_CONV_2D':
        sn.type = BuiltinOperator.DEPTHWISE_CONV_2D

        conv8 = sn.Conv2DOptions
        conv8.direct_dma = 0

        f_tensor = tensors[subop['inputs'][1]]
        b_tensor = None
        if len(op['inputs']) > 2 and op['inputs'][2] != -1:
            b_tensor = tensors[op['inputs'][2]]

        k, h, w, c = tuple(f_tensor['shape'])
        filter_shape_dims = [k, c, h, w]
        conv8.filter_shape_dims = filter_shape_dims

        conv8.kernels = k
        conv8.fit_weights = 0
        conv8.use_depthwise = 0
        conv8.group = 1
        if k == 1 and c > 1 and subcode == 'DEPTHWISE_CONV_2D':
            conv8.kernels = c
            conv8.group = c
            conv8.use_depthwise = 1


            sn.input_strides = [opts['stride_h'], opts['stride_w']]
            conv8.stride_width = opts['stride_w']
            conv8.stride_height = opts['stride_h']
            conv8.dilation_height_factor = opts['dilation_h_factor']
            conv8.dilation_width_factor = opts['dilation_w_factor']
            conv8.fused_activation = 0
            sn.activation_max = 127
            sn.activation_min = -128
            if opts['fused_activation_function'] == 'RELU':
                sn.activation_min = output_offset
            elif opts['fused_activation_function'] == 'RELU6':
                sn.activation_min = output_offset
                sn.activation_max = round((6 / output_scale) + output_offset)
                sn.activation_max = min(sn.activation_max, 127)

            conv8.use_strided = 0
            if OPTIMIZED_DEPTHWISE:
                conv8.use_vector = 2 #OPTIMIZED
            else:
                conv8.use_vector = 1
            conv8.use_fia = False
            conv8.use_db = 0
            conv8.conv_rows = 0

            assert(opts['padding'] == 'VALID')

            if opts['padding'] == 'VALID':
                conv8.padding_width = 0
                conv8.padding_height = 0
            

            conv8.imaps = -1
            effective_scale, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(subop, tensors)

            if len(output_multiplier) == 1 and conv8.kernels > 1:
                output_multiplier = [output_multiplier[0] for _ in range(conv8.kernels)]
            if len(output_shift) == 1 and k > 1:
                output_shift = [output_shift[0] for _ in range(conv8.kernels)]

            sn.input_offset = input_offset
            sn.output_offset = output_offset

            filter_data = get_numpy_data(f_tensor, buffers).transpose((0,3,1,2))

            sn.output_multiplier = len(weights)
            fmt = "{}i".format(len(output_multiplier))
            weights += struct.pack(fmt, *output_multiplier)

            sn.output_shift = len(weights)
            fmt = "{}i".format(len(output_shift))
            weights += struct.pack(fmt, *output_shift)

            bias_data = np.zeros((k,), dtype=np.int64)
            if not (b_tensor is None):
                bias_data = get_numpy_data(b_tensor, buffers).astype(np.int64)

            if USE_PRECALC:
                if subcode == 'CONV_2D':
                    bias_data += precalculate_filter_input_bias(filter_data, input_offset, reduce=False)
                else:
                    bias_data += precalculate_filter_input_bias(filter_data, input_offset, reduce=True)
            if PRECALC_OUTPUT:
                bias_data += precalculate_output_offset_bias(effective_scale, output_offset)

            bias_data = bias_data.clip(np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            bias_data = bias_data.astype(np.int32)

            conv8.bias_data = len(weights)
            fmt = "{}i".format(len(bias_data))
            weights += struct.pack(fmt, *bias_data)

            conv8.filter_data = len(weights)
            fmt = "{}b".format(len(filter_data.flatten()))
            weights += struct.pack(fmt, *filter_data.flatten())

            conv8.quantization_records = -1

        subnode_array.append(sn)

    elif subcode in ['ADD', 'SUB', 'MUL', 'DIV', "GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL", "EQUAL", "NOT_EQUAL", "SQUARED_DIFFERENCE"] and multi_input:  # >>>>>>>>> Main ELTWISE BLOCK with multi_inputs
       
        sn.type = VNNXOperator.ELTWISE
        eltwise8 = sn.eltwise8
        eltwise8.type = getattr(eltwise_type, f"ELTWISE_{subcode}")
        if eltwise8.type is None:
            raise ValueError(f"Invalid input string: {subcode}")

        i_tensor = tensors[subop['inputs'][0]] # left
        i2_tensor = tensors[subop['inputs'][1]] # right
        # Node brings in input2, assumed to be later 
        eltwise8.swap = subop['inputs'][1] < subop['inputs'][0]

        o_tensor = tensors[subop['outputs'][0]] 
        if subcode in ['MUL', 'ADD', 'SUB', 'DIV', 'MINIMUM', 'MAXIMUM', 'SQUARED_DIFFERENCE']:
            output_offset = o_tensor['quantization']['zero_point'][0] 
            output_scale = o_tensor['quantization']['scale']
        else:
            output_offset = 0
            output_scale = [1.]

        input_scale = i_tensor['quantization']['scale']
        input_offset = i_tensor['quantization']['zero_point'][0]
        input2_scale = i2_tensor['quantization']['scale']
        input2_offset = i2_tensor['quantization']['zero_point'][0]

        eltwise8.optimized = 0
        if subcode == 'ADD' and (2*input_scale[0]/output_scale[0]) < 2**(15-Q16) and (2*input2_scale[0]/output_scale[0]) < 2**(15-Q16):
            if len(i_tensor['shape']) > 3: # TODO fix post-procesing accuracy drop
                eltwise8.optimized = OPTIMIZED_ADD
        else:
            if OPTIMIZED_ADD and VERBOSE:
                print(subcode, input_scale[0]/output_scale[0], input2_scale[0]/output_scale[0], "WARNING can't optimize multi")

        max_input_scale = input_scale[0]
        for t in subop['inputs'][1:]:
            tensor = tensors[t]

            ishape, idims = channels_first_shape(tensor['shape'])
            tn = Tensor()
            tn.type = calc_type.from_str(tensor['type'])
            tn.shape = ishape
            tn.dims = idims
            tn.scale = tensor['quantization']['scale'][0]
            if tn.scale > max_input_scale:
                max_input_scale = tn.scale
            tn.zero = tensor['quantization']['zero_point'][0]
            tn.id = tensor['buffer'] - 1
            tn.name = tensor['name']
            if not node_offloaded and engine_graphs_nx:
                add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, subgraph_idx)
            sn.tensor_array.append(tn)

        twice_max_input_scale = 2 * max_input_scale

        if subcode in ['ADD', 'SUB']:
            left_shift = 15
            if 'pot_scale_int16' in opts and opts['pot_scale_int16']:
                left_shift = 20
            twice_max_input_scale = 2 * max(input_scale[0], input2_scale[0])
            real_input_multiplier = (np.asarray(input_scale) / twice_max_input_scale).tolist()
            real_input2_multiplier = (np.asarray(input2_scale) / twice_max_input_scale).tolist()
            real_output_multiplier = (twice_max_input_scale / (2**left_shift * np.asarray(output_scale))).tolist()

            input_multiplier, input_shift = get_quantized_multiplier(real_input_multiplier)
            input2_multiplier, input2_shift = get_quantized_multiplier(real_input2_multiplier)
            output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)

            sn.activation_max = 127
            sn.activation_min = -128

            if opts['fused_activation_function'] == 'RELU':
                sn.activation_min = output_offset

        elif subcode == 'SQUARED_DIFFERENCE':
            left_shift = 7
            twice_max_input_scale = 2 * max(input_scale[0], input2_scale[0])
            real_input_multiplier = (np.asarray(input_scale) / twice_max_input_scale).tolist()
            real_input2_multiplier = (np.asarray(input2_scale) / twice_max_input_scale).tolist()
            real_output_multiplier = ((twice_max_input_scale * twice_max_input_scale) /
                                        (2** left_shift*2 * np.asarray(output_scale))).tolist()
            # real_output_multiplier = (twice_max_input_scale / (2**left_shift * np.asarray(output_scale))).tolist()

            input_multiplier, input_shift = get_quantized_multiplier(real_input_multiplier)
            input2_multiplier, input2_shift = get_quantized_multiplier(real_input2_multiplier)
            output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)

            sn.activation_max = 127
            sn.activation_min = -128

        elif subcode in ['MUL', 'DIV']:
            left_shift = -1
            input_multiplier, input_shift = [], []
            input2_multiplier, input2_shift = [], []
            with exception_catcher( sn.type, len(subnode_array), subop['inputs'][0]):
                effective_scale, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(subop, tensors)

            sn.activation_max = 127
            sn.activation_min = -128

            if opts['fused_activation_function'] == 'RELU':
                sn.activation_min = output_offset

        elif subcode in ['MINIMUM', 'MAXIMUM']:
            input1_shape, idims = channels_first_shape(i_tensor['shape'])
            input2_shape, i2dims = channels_first_shape(i2_tensor['shape'])

            if input1_shape != input2_shape:
                if len(input2_shape) > 1:
                    sys.stderr.write("ERROR: SUBOP{} doesn't support difference tensor shapes: {}, {}.\n".format(subcode, input1_shape, len(input2_shape)))
                    sys.exit(1)

            left_shift = 0
            input_multiplier, input_shift = get_quantized_multiplier(input_scale)
            input2_multiplier, input2_shift = get_quantized_multiplier(input2_scale)
            output_multiplier, output_shift = get_quantized_multiplier(output_scale)

            sn.activation_max = 127
            sn.activation_min = -128
        else:
            left_shift = 8
            output_offset = 0
            output_scale = [1.]
            input_multiplier, input_shift = get_quantized_multiplier(input_scale)
            input2_multiplier, input2_shift = get_quantized_multiplier(input2_scale)
            output_multiplier, output_shift = get_quantized_multiplier(output_scale)

            sn.activation_max = 1
            sn.activation_min = 0
        
        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.input_multiplier = len(weights)
        fmt = "{}i".format(len(input_multiplier))
        weights += struct.pack(fmt, *input_multiplier)

        sn.input_shift = len(weights)
        fmt = "{}i".format(len(input_shift))
        weights += struct.pack(fmt, *input_shift)

        eltwise8.input2_multiplier = len(weights)
        fmt = "{}i".format(len(input2_multiplier))
        weights += struct.pack(fmt, *input2_multiplier)

        eltwise8.input2_shift = len(weights)
        fmt = "{}i".format(len(input2_shift))
        weights += struct.pack(fmt, *input2_shift)
    
        eltwise8.bias_data = -1
        sn.input_offset = input_offset
        sn.output_offset = output_offset
        eltwise8.input2_offset = input2_offset
        eltwise8.left_shift = left_shift

        subnode_array.append(sn)

    elif subcode == 'LUT':
        sn.type = VNNXOperator.LUT
        transform = None
        sn.ActivationOptions.lut_count = 1

        l = 0
        bytes = 1
        while l < len(lut_ops):
            op, code ,_ = lut_ops[l]
            next_op, next_code = None, ''
            if l < len(lut_ops)-1:
                next_op, next_code, _ = lut_ops[l+1]
            next_next_op, next_next_code = None, ''
            if l < len(lut_ops)-2:
                next_next_op, next_next_code, _ = lut_ops[l+2]

            if code == 'LOGISTIC' and next_code == 'MUL':
                l += 1
            elif code == 'RESHAPE' and next_code == 'CAST'and next_next_code == 'GATHER':
                l += 2
                bytes = 4
            elif code == 'CAST'and next_code == 'GATHER':
                l += 1
                bytes = 4
            elif code in ['MUL', 'ADD', 'SUB']:
                _, _, _, _, _, _, filter_shape_dims, _ = get_subop_parameters(op, tensors, buffers)
                if filter_shape_dims[-3] > sn.ActivationOptions.lut_count:
                    sn.ActivationOptions.lut_count = filter_shape_dims[-3]
            l += 1

        lut, step_vals, step_indices = [], [], []
        final_output_scale = None
        final_output_offset = None
        for x in range(sn.ActivationOptions.lut_count):
            l = 0
            lut_fn = []
            while l < len(lut_ops):
                op, code, _ = lut_ops[l]
                next_op, next_code = None, ''
                if l < len(lut_ops)-1:
                    next_op, next_code, _ = lut_ops[l+1]
                next_next_op, next_next_code = None, ''
                if l < len(lut_ops)-2:
                    next_next_op, next_next_code, _ = lut_ops[l+2]

                o_tensor = tensors[op['outputs'][0]]

                if code == 'LOGISTIC' and next_code == 'MUL':
                    fn = lut_func('SILU')
                    l += 1

                    o_tensor = tensors[next_op['outputs'][0]]
                    output_type = o_tensor['type']
                    sn.output_data_type = calc_type.from_str(output_type)

                elif code == 'RESHAPE' and next_code == 'CAST'and next_next_code == 'GATHER':
                    pixels = get_numpy_data_from_index(next_next_op['inputs'][0], tensors, buffers)
                    arr = np.zeros((256,), dtype=pixels.dtype)
                    arr[:len(pixels)] = pixels
                    fn = lambda s: float(arr[int(s)])
                    l += 2

                    o_tensor = tensors[next_next_op['outputs'][0]]
                    output_type = o_tensor['type']
                    sn.output_data_type = calc_type.from_str(output_type)

                elif code == 'CAST'and next_code == 'GATHER':
                    pixels = get_numpy_data_from_index(next_op['inputs'][0], tensors, buffers)
                    arr = np.zeros((256,), dtype=pixels.dtype)
                    arr[:len(pixels)] = pixels
                    fn = lambda s: float(arr[int(s)])
                    l += 1

                    o_tensor = tensors[next_op['outputs'][0]]
                    output_type = o_tensor['type']
                    sn.output_data_type = calc_type.from_str(output_type)

                elif code in ['HARD_SWISH', 'LOGISTIC', 'RELU', 'RELU6', 'RELU_0_TO_1']:
                    fn = lut_func(code)

                elif code in ['QUANTIZE']:
                    fn = lut_func(code)

                    #TODO cover other cases / zeros
                    if o_tensor['type'] == 'UINT8':
                        if l < len(lut_ops) - 1:
                            fn = lambda s: s + 128

                elif code in ['LEAKY_RELU']:
                    opts = op['builtin_options']
                    fn = lut_func(code, opts['alpha'])

                elif code in ['MUL', 'ADD', 'SUB', 'SQUARED_DIFFERENCE']:
                    _, _, _, filter_scale, _, filter_offset, filter_shape_dims, filter_data = get_subop_parameters(op, tensors, buffers)
                    if x < filter_shape_dims[-3]:
                        dequantized_filter = filter_scale[0] * (filter_data[x] - filter_offset)  
                    else:
                        dequantized_filter = filter_scale[0] * (filter_data[0] - filter_offset)  

                    fn = lut_func(code, dequantized_filter)
                else:
                    print('error unsupported LUT pattern', code)

                final_output_offset = o_tensor['quantization'].get('zero_point', [0])[0]
                final_output_scale = o_tensor['quantization'].get('scale', [1.0])[0]

                lut_fn.append(fn)

                l += 1
            lut_fn.reverse()
            transform = composite_function(*lut_fn)
            
            if bytes == 4:
                l, first, last, step_val, step_ind = LUTPopulate(1., 0., 1., 0, transform, bytes=4, itype=sn.input_data_type)
                
                lut_repacked = [0xff & _ for _ in l]
                lut_repacked += [(0xff * (2**8) & _) // (2**8) for _ in l]
                lut_repacked += [(0xff * (2**16) & _) // (2**16) for _ in l]
                lut_repacked += [(0xff * (2**24) & _) // (2**24) for _ in l]
                lut.extend(lut_repacked)
                step_vals.extend(step_val)
                step_indices.extend(step_ind)
            else:
                l, first, last, step_val, step_ind = LUTPopulateInt8(input_scale, input_offset, final_output_scale, final_output_offset, transform, itype=sn.input_data_type, otype=sn.output_data_type)

                lut.extend(l)
                step_vals.extend(step_val)
                step_indices.extend(step_ind)

        sn.input_offset = input_offset
        sn.output_offset = final_output_offset
        sn.ActivationOptions.input_range_radius = -1

        if VCI_LUT:
            sn.ActivationOptions.vci_int8 = len(weights)
            fmt = "{}B".format(len(lut))
            weights += struct.pack(fmt, *lut)
        else:
            sn.ActivationOptions.vci_int8 = -1 

        if bytes == 4:
            sn.ActivationOptions.lut_int8 = -1
            # step_vals = np.asarray(step_vals, dtype='int32')
            # sn.ActivationOptions.lut_int32 = len(weights)
            # fmt = "{}I".format(len(step_vals))
            # weights += struct.pack(fmt, *step_vals)
        else:
            sn.ActivationOptions.lut_int8 = len(weights)
            fmt = "{}b".format(len(step_vals))
            weights += struct.pack(fmt, *step_vals)

        if bytes == 4:
            sn.ActivationOptions.idx_int8 = -1
        else:
            sn.ActivationOptions.idx_int8 = len(weights)
            if sn.input_data_type == calc_type.UINT8:
                fmt = "{}B".format(len(step_indices))
            else:
                fmt = "{}b".format(len(step_indices))
            weights += struct.pack(fmt, *step_indices)
        sn.ActivationOptions.count = len(step_vals)

        sn.input_shift = -1
        sn.input_multiplier = 1 
        sn.output_multiplier = len(weights)
        sn.output_shift = len(weights)

        subnode_array.append(sn)

    elif subcode in ['PAD', 'PADV2']:
        sn.type = BuiltinOperator.PAD

        pads = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        c, h, w = [0, 0], [0, 0], [0, 0]
        if len(pads) == 4:
            c, h, w = pads[3], pads[1], pads[2]
        elif len(pads) == 3:
            c, h, w = pads[2], pads[0], pads[1]
        elif len(pads) == 2:
            h, w = pads[0], pads[1]
        elif len(pads) == 1:
            w = pads[0]
        sn.pads = [c[0], h[0], w[0], c[1], h[1], w[1]]
        assert len(sn.pads) == 6
        sn.PadOptions.value = input_offset
        sn.PadOptions.transpose_dilate_w = 1
        sn.PadOptions.transpose_dilate_h = 1
        if subcode == 'PADV2':
            value = get_numpy_data_from_index(subop['inputs'][2], tensors, buffers)
            sn.PadOptions.value = value

        subnode_array.append(sn)

    elif subcode == "RESIZE_NEAREST_NEIGHBOR":
        sn.type = BuiltinOperator.RESIZE_NEAREST_NEIGHBOR
        s_tensor = tensors[subop['inputs'][1]]

        resize = sn.ResizeOptions
        resize.mode = resize_mode.NEAREST 

        new_size_data = get_numpy_data(s_tensor, buffers)
        resize.scale = (new_size_data / i_tensor['shape'][1:3]).tolist() # height and width scales
        subnode_array.append(sn)

    elif subcode == "RESIZE_BILINEAR":
        sn.type = BuiltinOperator.RESIZE_BILINEAR
        s_tensor = tensors[subop['inputs'][1]]

        resize = sn.ResizeOptions
        resize.mode = resize_mode.LINEAR 
        resize.ratio =  ((1 << 10) * i_tensor['shape'][1] + o_tensor['shape'][1] / 2) / o_tensor['shape'][1]
       
        new_size_data = get_numpy_data(s_tensor, buffers)
        resize.scale = (new_size_data / i_tensor['shape'][1:3]).tolist() # height and width scales

        if i_tensor['shape'][1:3] == [1,1]:
            sn.type = BuiltinOperator.RESIZE_NEAREST_NEIGHBOR
            resize.mode = resize_mode.NEAREST 

        subnode_array.append(sn)

    elif subcode == 'MIRROR_PAD':
        print('WARNING: currently MIRROR_PAD running as PAD')
        sn.type = BuiltinOperator.MIRROR_PAD

        pads = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        c, h, w = [0, 0], [0, 0], [0, 0]
        if len(pads) == 4:
            c, h, w = pads[3], pads[1], pads[2]
        elif len(pads) == 3:
            c, h, w = pads[2], pads[0], pads[1]
        elif len(pads) == 2:
            h, w = pads[0], pads[1]
        elif len(pads) == 1:
            w = pads[0]
        sn.pads = [c[0], h[0], w[0], c[1], h[1], w[1]]
        assert len(sn.pads) == 6

        # 0 = SYMMETRIC, 1 = REFLECT
        pad_mode = opts['mode']
        sn.MirrorPadOptions.mode = 1 if pad_mode=='REFLECT' else 0

        subnode_array.append(sn)

    elif subcode == 'SOFTMAX':
        sn.type = BuiltinOperator.SOFTMAX

        assert (i_tensor['shape'] == o_tensor['shape'])
        beta = opts['beta']
        input_scale = i_tensor['quantization']['scale']

        sn.SoftmaxOptions.diff_min = -1
        output_scale = o_tensor['quantization']['scale']
        output_multiplier = 1. / (output_scale[0] * 2**16)
        quantized_multiplier, left_shift = get_quantized_multiplier([output_multiplier])

        sn.input_multiplier = len(weights)
        fmt = "{}i".format(len(quantized_multiplier))
        weights += struct.pack(fmt, *quantized_multiplier)

        sn.input_shift = len(weights)
        fmt = "{}i".format(len(left_shift))
        weights += struct.pack(fmt, *left_shift)

        axis = -1
        if 'axis' in opts:
            axis = opts['axis'] 
        axis = channels_first_axis(axis, len(input_shape)) - idims
        sn.SoftmaxOptions.axis = axis
        sn.SoftmaxOptions.depth = output_shape[axis]


        lut_scale = min(2**24, 2**31 / sn.SoftmaxOptions.depth)
        transform = lambda s: np.exp((s-127)*input_scale[0]) * lut_scale
        lut, first, last, step_vals, step_indices = LUTPopulate(1., 0., 1., 0, transform, bytes=4)

        sn.SoftmaxOptions.vci_int8 = len(weights)
        #repack
        lut_repacked = [0xff & _ for _ in lut]
        lut_repacked += [(0xff * (2**8) & _) // (2**8) for _ in lut]
        lut_repacked += [(0xff * (2**16) & _) // (2**16) for _ in lut]
        lut_repacked += [(0xff * (2**24) & _) // (2**24) for _ in lut]
        fmt = "{}B".format(len(lut_repacked))
        weights += struct.pack(fmt, *lut_repacked)

        sn.SoftmaxOptions.lut_int32 = len(weights)
        fmt = "{}I".format(len(step_vals))
        weights += struct.pack(fmt, *step_vals)

        sn.SoftmaxOptions.idx_int8 = len(weights)
        fmt = "{}b".format(len(step_indices))
        weights += struct.pack(fmt, *step_indices)
        sn.SoftmaxOptions.count = len(step_vals)

        subnode_array.append(sn)

    elif subcode == 'LOG_SOFTMAX':
        sn.type = BuiltinOperator.LOG_SOFTMAX

        kOutputZeroPoint = 127
        kOutputScale = 16.0 / 256
        kBeta = 1.0
        input_integer_bits = 5

        input_scale = i_tensor['quantization']['scale']
        input_offset = i_tensor['quantization']['zero_point'][0]

        max_real_multiplier = (1 << 31) - 1.0
        input_beta_real_multiplier = min(kBeta * input_scale[0] * (1 << (31 - input_integer_bits)), max_real_multiplier)
        assert (input_beta_real_multiplier > 1.0)
        quantized_multiplier, left_shift = get_quantized_multiplier([input_beta_real_multiplier])
        assert (all(x >= 0 for x in left_shift))
        assert (i_tensor['shape'] == o_tensor['shape'])

        real_reverse_scaling_divisor = (1 << (31 - left_shift[0])) / quantized_multiplier[0]
        assert (real_reverse_scaling_divisor < 1.0 and real_reverse_scaling_divisor > 0.0)
        reverse_scaling_divisor, reverse_scaling_left_shift = get_quantized_multiplier([real_reverse_scaling_divisor])

        reverse_scaling_right_shift = -1 * reverse_scaling_left_shift[0]
        assert (reverse_scaling_right_shift >= 0)

        output_multiplier, output_shift = get_quantized_multiplier([kOutputScale])

        max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) * (1 << (31 - input_integer_bits)) / (1 << left_shift[0])
        diff_min = -1 * floor(max_input_rescaled)

        sn.input_multiplier = len(weights)
        fmt = "{}i".format(len(quantized_multiplier))
        weights += struct.pack(fmt, *quantized_multiplier)

        sn.input_shift = len(weights)
        fmt = "{}i".format(len(left_shift))
        weights += struct.pack(fmt, *left_shift)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.output_offset = kOutputZeroPoint

        axis = -1
        if 'axis' in opts:
            axis = opts['axis']
        axis = channels_first_axis(axis, len(input_shape)) - idims

        sn.LogSoftmaxOptions.reverse_scaling_divisor = reverse_scaling_divisor[0]
        sn.LogSoftmaxOptions.reverse_scaling_right_shift = reverse_scaling_right_shift
        sn.LogSoftmaxOptions.diff_min = diff_min
        sn.LogSoftmaxOptions.axis = axis
        sn.LogSoftmaxOptions.depth = output_shape[axis]

        # assuming only one vector
        if len(o_tensor['shape']) > 1:
            sn.LogSoftmaxOptions.outer_size = o_tensor['shape'][0]
        else:
            sn.LogSoftmaxOptions.outer_size = 1

        sn.LogSoftmaxOptions.depth = o_tensor['shape'][-1]

        subnode_array.append(sn)

    elif subcode == 'MAX_POOL_2D':
        sn.type = BuiltinOperator.MAX_POOL_2D

        sn.kernel_shape = [opts['filter_height'], opts['filter_width']]
        sn.strides = [opts['stride_h'], opts['stride_w']]
        sn.pads = [0, 0, 0, 0, 0, 0]

        # inject Pad subnode before if necessary
        if opts['padding'] == "SAME":
            i_h, i_w = input_shape[-2:]
            o_h, o_w = output_shape[-2:]
            stride_h, stride_w = sn.strides
            kernel_h, kernel_w = sn.kernel_shape
            pad_h = max(0, stride_h * (o_h - 1) - i_h + kernel_h)
            pad_w = max(0, stride_w * (o_w - 1) - i_w + kernel_w)

            if pad_h > 0 or pad_w > 0:
                pad_sn = Subnode()
                pad_sn.type = BuiltinOperator.PAD

                pad_sn.input_data_type = calc_type.from_str(input_type)
                pad_sn.output_data_type = calc_type.from_str(output_type)


                pad_sn.pads = [0, floor(pad_h/2), floor(pad_w/2), 0, ceil(pad_h/2), ceil(pad_w/2)]
                pad_sn.PadOptions.value = -128
                pad_sn.PadOptions.transpose_dilate_w = 1
                pad_sn.PadOptions.transpose_dilate_h = 1

                pad_sn.num_inputs = 0
                pad_sn.num_outputs = 0

                subnode_array.append(pad_sn)

        subnode_array.append(sn)

    elif subcode == 'AVERAGE_POOL_2D':
        assert(sn.input_data_type == calc_type.INT8)
        i_h, i_w = input_shape[-2:]
        o_h, o_w = output_shape[-2:]

        sn.type = BuiltinOperator.AVERAGE_POOL_2D

        sn.kernel_shape = [opts['filter_height'], opts['filter_width']]
        if sn.kernel_shape[0] > i_h:
            sn.kernel_shape[0] = i_h
        if sn.kernel_shape[1] > i_w:
            sn.kernel_shape[1] = i_w

        sn.strides = [opts['stride_h'], opts['stride_w']]
        sn.pads = [0, 0, 0, 0, 0, 0]

        if opts['padding'] == "SAME":
            stride_h, stride_w = sn.strides
            kernel_h, kernel_w = sn.kernel_shape

            pad_h = max(0, stride_h * (o_h - 1) - i_h + kernel_h)
            pad_w = max(0, stride_w * (o_w - 1) - i_w + kernel_w)

            sn.pads = [0, floor(pad_h/2), floor(pad_w/2), 0, ceil(pad_h/2), ceil(pad_w/2)]

        subnode_array.append(sn)

    elif subcode == 'MUL': 
        sn.type = BuiltinOperator.MUL

        i_tensor = tensors[subop['inputs'][0]]
        f_tensor = tensors[subop['inputs'][1]]

        swap_input_order = 0
        if 'buffer' in i_tensor and 'data' in buffers[i_tensor['buffer']]:
            swap_input_order = 1
            i_tensor = tensors[subop['inputs'][1]]
            f_tensor = tensors[subop['inputs'][0]]

        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]
        filter_offset = f_tensor['quantization']['zero_point'][0]

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']
        filter_scale = f_tensor['quantization']['scale']


        effective_scale, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(subop, tensors)

        broad8 = sn.broadcast8
        broad8.iscale = input_scale[0]
        broad8.fscale = filter_scale[0]
        broad8.oscale = output_scale[0]

        broad8.iscale = sqrt(broad8.iscale*broad8.fscale)
        broad8.fscale = broad8.iscale

        broad8.swap_inputs = swap_input_order

        filter_data = get_numpy_data(f_tensor, buffers)
        if len(f_tensor['shape']) >= 3:
            if len(f_tensor['shape']) == len(i_tensor['shape']):
                pass

        k, h, w, c = 1, 1, 1, 1
        if len(f_tensor['shape']) == 1:
            c = tuple(f_tensor['shape'])[0]
        elif len(f_tensor['shape']) == 2:
            k, c = tuple(f_tensor['shape'])
        elif len(f_tensor['shape']) == 3:
            h, w, c = tuple(f_tensor['shape'])
            filter_data = filter_data.transpose((2,0,1))
        elif len(f_tensor['shape']) == 4:
            k, h, w, c = tuple(f_tensor['shape'])
            filter_data = filter_data.transpose((0,3,1,2))
        elif len(f_tensor['shape']) == 5:
            assert(f_tensor['shape'][0] == 1)
            k, h, w, c = f_tensor['shape'][1:]
            filter_data = filter_data.transpose((0,1,4,2,3))

        broad8.filter_shape_dims = [k, c, h, w]

        broad8.broadcast = 0
        ishape, _ = channels_first_shape(i_tensor['shape'])
        if broad8.filter_shape_dims[-2] == 1 and broad8.filter_shape_dims[-1] == 1: # channels
            broad8.broadcast = 1
        elif broad8.filter_shape_dims[-2] == 1 and broad8.filter_shape_dims[-1] == ishape[-1]: #row
            broad8.broadcast = 2
        elif broad8.filter_shape_dims[-2] == ishape[-2] and broad8.filter_shape_dims[-1] == ishape[-1]: #maps
            broad8.broadcast = 3

        broad8.optimized = 0
        sf = input_scale[0] * filter_scale[0] / output_scale[0] 
        scaled_filters = [sf * (_ - filter_offset) for _ in filter_data.flatten()]
        valid_filters = [_ < 2**(15-Q16) for _ in scaled_filters]
        if (np.all(valid_filters)):
            if broad8.broadcast == 1:
                broad8.optimized = OPTIMIZED_MUL
        else:
            if OPTIMIZED_MUL and VERBOSE:
                print(subcode, input_scale[0], filter_scale[0], output_scale[0], "WARNING can't optimize")


        sn.input_multiplier = -1
        sn.input_shift = -1

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        broad8.filter_multiplier = -1
        broad8.filter_shift = -1

        broad8.filter_data = len(weights)
        fmt = "{}b".format(len(filter_data.flatten()))
        weights += struct.pack(fmt, *filter_data.flatten())

        if USE_PRECALC:
            bias_data = precalculate_filter_input_bias(filter_data, input_offset, filter_offset)
            bias_data += precalculate_output_offset_bias(effective_scale, output_offset)
            broad8.bias_data = len(weights)
            fmt = "{}i".format(len(bias_data))
            weights += struct.pack(fmt, *bias_data)
        else:
            broad8.bias_data = -1

        sn.input_offset = input_offset
        sn.output_offset = output_offset
        broad8.filter_offset = filter_offset
        broad8.left_shift = -1
        broad8.sub = 0

        subnode_array.append(sn)

    elif subcode in ['ADD', 'SUB', 'SQUARED_DIFFERENCE', 'SQUARED_ROOT']:
        if subcode == 'ADD':
            sn.type = BuiltinOperator.ADD
        else:
            sn.type = BuiltinOperator.SUB
            
        i_tensor = tensors[subop['inputs'][0]]
        f_tensor = tensors[subop['inputs'][1]]

        swap_input_order = 0
        if 'buffer' in i_tensor and 'data' in buffers[i_tensor['buffer']]:
            swap_input_order = 1
            i_tensor = tensors[subop['inputs'][1]]
            f_tensor = tensors[subop['inputs'][0]]

        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]
        filter_offset = f_tensor['quantization']['zero_point'][0]

        filter_data = get_numpy_data(f_tensor, buffers)

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']
        filter_scale = f_tensor['quantization']['scale']

        left_shift = 15
        if subcode != 'SQUARED_DIFFERENCE':
            if 'pot_scale_int16' in opts and opts['pot_scale_int16']:
                left_shift = 20
        
        twice_max_input_scale = 2 * max(input_scale[0], filter_scale[0])
        real_input_multiplier = (np.asarray(input_scale) / twice_max_input_scale).tolist()
        real_filter_multiplier = (np.asarray(filter_scale) / twice_max_input_scale).tolist()
        real_output_multiplier = (twice_max_input_scale / (2**left_shift * np.asarray(output_scale))).tolist()

        input_multiplier, input_shift = get_quantized_multiplier(real_input_multiplier)
        filter_multiplier, filter_shift = get_quantized_multiplier(real_filter_multiplier)
        output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)


        broad8 = sn.broadcast8
        broad8.iscale = input_scale
        broad8.fscale = filter_scale
        broad8.oscale = output_scale

        broad8.sub = 0
        broad8.swap_inputs = swap_input_order
        if subcode == 'SUB':
            broad8.sub = 1
        
        if subcode == 'SQUARED_DIFFERENCE':
            sn.type = BuiltinOperator.SQUARED_DIFFERENCE
            left_shift = 7
            sn.activation_max = 127
            sn.activation_min = -128
            broad8.sub = -1
            twice_max_input_scale = 2 * max(input_scale[0], filter_scale[0])
            real_input_multiplier = (np.asarray(input_scale) / twice_max_input_scale).tolist()
            real_filter_multiplier = (np.asarray(filter_scale) / twice_max_input_scale).tolist()
            real_output_multiplier = (twice_max_input_scale*twice_max_input_scale / (1<<left_shift * 2) * np.asarray(output_scale)).tolist()

            input_multiplier, input_shift = get_quantized_multiplier(real_input_multiplier)
            filter_multiplier, filter_shift = get_quantized_multiplier(real_filter_multiplier)
            output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)
            
        k, h, w, c = 1, 1, 1, 1
        if len(f_tensor['shape']) == 1:
            c = tuple(f_tensor['shape'])[0]
        elif len(f_tensor['shape']) == 2:
            k, c = tuple(f_tensor['shape'])
        elif len(f_tensor['shape']) == 3:
            h, w, c = tuple(f_tensor['shape'])
            filter_data = filter_data.transpose((2,0,1))
        elif len(f_tensor['shape']) == 4:
            k, h, w, c = tuple(f_tensor['shape'])
            filter_data = filter_data.transpose((0,3,1,2))
        elif len(f_tensor['shape']) == 5:
            assert(f_tensor['shape'][0] == 1)
            k, h, w, c = f_tensor['shape'][1:]
            filter_data = filter_data.transpose((0,1,4,2,3))
        broad8.filter_shape_dims = [k, c, h, w]


        broad8.broadcast = 0
        ishape, _ = channels_first_shape(i_tensor['shape'])
        if broad8.filter_shape_dims[-2] == 1 and broad8.filter_shape_dims[-1] == 1: # channels
            broad8.broadcast = 1
        elif broad8.filter_shape_dims[-2] == 1 and broad8.filter_shape_dims[-1] == ishape[-1]: #row
            broad8.broadcast = 2
        elif broad8.filter_shape_dims[-2] == ishape[-2] and broad8.filter_shape_dims[-1] == ishape[-1]: #maps
            broad8.broadcast = 3

        broad8.optimized = 0
        if broad8.broadcast == 1 and (2*input_scale[0]/output_scale[0]) < 2**(15-Q16) and (2*filter_scale[0]/output_scale[0]) < 2**(15-Q16):
            broad8.optimized = OPTIMIZED_ADD
        else:
            if OPTIMIZED_ADD and VERBOSE:
                print(subcode, input_scale[0]/output_scale[0], filter_scale[0]/output_scale[0], "WARNING can't optimize")

        if subcode != 'SQUARED_DIFFERENCE':
            if opts['fused_activation_function'] == 'RELU':
                sn.activation_min = output_offset

        sn.input_multiplier = len(weights)
        fmt = "{}i".format(len(input_multiplier))
        weights += struct.pack(fmt, *input_multiplier)

        sn.input_shift = len(weights)
        fmt = "{}i".format(len(input_shift))
        weights += struct.pack(fmt, *input_shift)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        broad8.filter_multiplier = len(weights)
        fmt = "{}i".format(len(filter_multiplier))
        weights += struct.pack(fmt, *filter_multiplier)

        broad8.filter_shift = len(weights)
        fmt = "{}i".format(len(filter_shift))
        weights += struct.pack(fmt, *filter_shift)

        broad8.filter_data = len(weights)
        fmt = "{}b".format(len(filter_data.flatten()))
        weights += struct.pack(fmt, *filter_data.flatten())

        broad8.bias_data = -1
        sn.input_offset = input_offset
        sn.output_offset = output_offset
        broad8.filter_offset = filter_offset
        broad8.left_shift = left_shift

        subnode_array.append(sn)
    elif subcode in ["GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL", "EQUAL", "NOT_EQUAL"]: # > to remove
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        f_tensor = tensors[subop['inputs'][1]]
        if np.squeeze(np.array(f_tensor['shape'])) != ():
            sys.stderr.write('ERROR: {} comparison not on scalar\n'.format(subcode))
            sys.exit(1)
        filter_offset = f_tensor['quantization']['zero_point'][0]
        filter_data = get_numpy_data(f_tensor, buffers)
        filter_scale = f_tensor['quantization']['scale']
        output_offset = 0
        output_scale = [1.]

        left_shift = 8

        input_multiplier, input_shift = get_quantized_multiplier(input_scale)
        filter_multiplier, filter_shift = get_quantized_multiplier(filter_scale)
        output_multiplier, output_shift = get_quantized_multiplier(output_scale)

        broad8 = sn.broadcast8
        broad8.iscale = input_scale
        broad8.fscale = filter_scale
        broad8.oscale = output_scale

        sn.activation_max = 1
        sn.activation_min = 0

        sn.input_multiplier = len(weights)
        fmt = "{}i".format(len(input_multiplier))
        weights += struct.pack(fmt, *input_multiplier)

        sn.input_shift = len(weights)
        fmt = "{}i".format(len(input_shift))
        weights += struct.pack(fmt, *input_shift)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        broad8.filter_multiplier = len(weights)
        fmt = "{}i".format(len(filter_multiplier))
        weights += struct.pack(fmt, *filter_multiplier)

        broad8.filter_shift = len(weights)
        fmt = "{}i".format(len(filter_shift))
        weights += struct.pack(fmt, *filter_shift)

        broad8.filter_data = len(weights)
        fmt = "{}b".format(len(filter_data.flatten()))
        weights += struct.pack(fmt, *filter_data.flatten())

        broad8.bias_data = 0
        sn.input_offset = input_offset
        sn.output_offset = output_offset
        broad8.filter_offset = filter_offset
        broad8.left_shift = left_shift
        broad8.sub = 0
        broad8.swap_inputs = 0

        subnode_array.append(sn)

    elif subcode in ['MEAN', 'SUM', 'REDUCE_PROD', 'REDUCE_MAX', 'REDUCE_MIN']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']

        num_elements_in_axis = input_shape[-2]*input_shape[-1]

        if subcode == 'REDUCE_PROD':
            effective_scale, output_multiplier, output_shift = get_quantized_multiplier_from_tensor(o_tensor)
        elif subcode == 'MEAN':
            effective_scale, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(subop, tensors, [1.0/num_elements_in_axis,])
        else:
            effective_scale, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(subop, tensors, [1.0,])
        
        sn.input_multiplier = -1
        sn.input_shift = -1

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.input_offset = input_offset
        sn.output_offset = output_offset

        subnode_array.append(sn)

    elif subcode == 'TRANSPOSE':
        sn.type = BuiltinOperator.TRANSPOSE

        transform = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        transform_size = len(np.trim_zeros([_ if i == _ else 0 for i,_ in enumerate(transform)], 'f'))
        transform_size = max(3, transform_size)

        assert(np.all([_ == i for i,_ in enumerate(transform[:-3])]))
        transform = [t - (len(transform) - 3) for t in transform[-3:]]

        assert(len(transform) == 3)

        transform = [(transform[2] + 1) % 3, (transform[0] + 1) % 3, (transform[1] + 1) % 3]

        sn.TransposeOptions.permutation = transform
        sn.TransposeOptions.out_rows_at_once = 1
        sn.TransposeOptions.out_maps_at_once = 1

        subnode_array.append(sn)

    elif subcode in ['UNPACK']:
        sn.type = BuiltinOperator.UNPACK

        input_shape, idims = channels_first_shape(i_tensor['shape'])
        sn.PackOptions.axis = channels_first_axis(opts['axis'], idims) - idims
        sn.PackOptions.count = opts['num']
        sn.PackOptions.dims = idims

        assert(check_unpack(i_tensor['shape'], o_tensor['shape'],
                            opts['axis'], sn.PackOptions.axis, opts['num']))
        subnode_array.append(sn)

    elif subcode in ['SQUEEZE', 'EXPAND_DIMS', 'RESHAPE']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        transform = tuple(get_numpy_data_from_index(subop['inputs'][1], tensors, buffers))

        ishape = i_tensor['shape']
        mode = channels_first_array_reshape(ishape, transform)

        sn.nop = 0
        if mode == 0:
            sn.nop = 1

        if mode == -1:
            print('RESHAPE: not implemented')
            assert(0)

        sn.ReshapeOptions.mode = mode
        subnode_array.append(sn)
    elif subcode == 'QUANTIZE':
        if len(dequantize_op_params) % 2 != 0:
            input_id = subop['inputs'][0]
            output_id = subop['outputs'][0]
            o_tensor = tensors[subop['outputs'][0]]
            output_offset = o_tensor['quantization']['zero_point'][0]
            output_scale = o_tensor['quantization']['scale']
            dequantize_op_params[input_id] = (output_scale, output_offset, output_id, 'output')
        else:
            sn.type = BuiltinOperator.DEQUANTIZE

            i_tensor = tensors[subop['inputs'][0]]
            o_tensor = tensors[subop['outputs'][0]]

            input_offset = 0
            input_scale = 1.0
            if 'quantization' in i_tensor.keys():
                input_offset = i_tensor['quantization'].get('zero_point', [0])[0]
                input_scale = i_tensor['quantization'].get('scale', 1.0)
            output_offset = o_tensor['quantization']['zero_point'][0]
            output_scale = o_tensor['quantization']['scale']

            real_input_multiplier = [1.0]
            real_output_multiplier = (np.asarray(input_scale) / np.asarray(output_scale) ).tolist()

            output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)

            i_type = calc_type.from_str(i_tensor['type'])
            o_type = calc_type.from_str(o_tensor['type'])
            mixed_type_int8_uint8 = i_type == calc_type.INT8 and o_type == calc_type.UINT8
            mixed_type_uint8_int8 = i_type == calc_type.UINT8 and o_type == calc_type.INT8
            zero_point_diff = input_offset - output_offset
            if output_multiplier[0] == 2**30 and output_shift[0] == 1: 
                if mixed_type_int8_uint8 and zero_point_diff == -128: 
                    sn.activation_min = 0
                    sn.activation_max = 0
                elif mixed_type_uint8_int8 and zero_point_diff == 128:
                    sn.activation_min = 0
                    sn.activation_max = 0

            sn.output_multiplier = len(weights)
            fmt = "{}i".format(len(output_multiplier))
            weights += struct.pack(fmt, *output_multiplier)

            sn.output_shift = len(weights)
            fmt = "{}i".format(len(output_shift))
            weights += struct.pack(fmt, *output_shift)

            sn.input_offset = input_offset
            sn.output_offset = output_offset

            subnode_array.append(sn)

    elif subcode == 'DEQUANTIZE':
        # case where Dequantize followed by Split followed by Quantize (yolo-v4-tiny-tf)
        # skip Dequantize and Quantize ops
        # TODO raise error for other cases, as they will not be handled correctly
        input_id = subop['inputs'][0]
        output_id = subop['outputs'][0]
        i_tensor = tensors[subop['inputs'][0]]
        scale = i_tensor['quantization']['scale'][0]
        zero_point = i_tensor['quantization']['zero_point'][0]
        dequantize_op_params[output_id] = (scale, zero_point, input_id, 'input')

    # @TODO to move.
    elif subcode in ['POW','EXP','LOG','ELU','GELU', 'HARD_SWISH', 'SILU', 'RSQRT', 'TANH', 'LOGISTIC']:
        sn.ActivationOptions.lut_count = 1
        sn.ActivationOptions.vci_int8 = -1

        sn.type = getattr(BuiltinOperator, f"{subcode}")
        if subcode == 'POW': #TODO
            exponent_tensor = tensors[op['inputs'][1]]
            transform = lambda s: pow(s)
        elif subcode == 'LOG':
            transform = lambda s: tf.math.log(s)
        elif subcode == 'EXP':
            transform = lambda s: exp(s)
        elif subcode == 'ELU':
            transform = lambda s: tf.nn.elu(s)
        elif subcode == 'GELU':
            transform = lambda s: tf.nn.gelu(s)
        elif subcode == 'TANH':
            transform = lambda s: tf.nn.tanh(s)
        elif subcode == 'LOGISTIC':
            transform = lambda s: sigmoid(s)
        elif subcode == 'HARD_SWISH':
            transform = lambda s: s * tf.nn.relu6(s + np.float32(3.)) * np.float32(1./ 6.)
        elif subcode == 'SILU':
            transform = lambda s: s*sigmoid(s)
        elif subcode == 'RSQRT':
            # transform = lambda s: 1.0 / sqrt(s) if s > 0 else 127 * o_tensor['quantization']['scale'][0]
            transform = lambda s: 1.0 / sqrt(s) if s > 0 else o_tensor['quantization']['zero_point'][0] * 1.0

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']

        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]

        sn.input_offset = input_offset
        sn.output_offset = output_offset

        sn.ActivationOptions.input_range_radius = -1
        lut, first, last, step_vals, step_indices = LUTPopulateInt8(input_scale[0], input_offset, output_scale[0], output_offset, transform)

        if VCI_LUT:
            sn.ActivationOptions.vci_int8 = len(weights)
            fmt = "{}B".format(len(lut))
            weights += struct.pack(fmt, *lut)
        else:
            sn.ActivationOptions.vci_int8 = -1
        sn.ActivationOptions.lut_int8 = len(weights)
        fmt = "{}b".format(len(step_vals))
        weights += struct.pack(fmt, *step_vals)
        sn.ActivationOptions.idx_int8 = len(weights)
        fmt = "{}b".format(len(step_indices))
        weights += struct.pack(fmt, *step_indices)
        sn.ActivationOptions.count = len(step_vals)

        sn.input_shift = -1
        sn.input_multiplier = 1 
        sn.output_multiplier = len(weights)
        sn.output_shift = len(weights)

        subnode_array.append(sn)
                
    elif subcode == 'PRELU':
        sn.type = BuiltinOperator.PRELU

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        alpha_tensor = tensors[subop['inputs'][1]]
        alpha_data = get_numpy_data(alpha_tensor, buffers)

        b, h, w, c = 1, 1, 1, 1
        if len(alpha_tensor['shape']) == 3:
            h, w, c = tuple(alpha_tensor['shape'])
            alpha_data = alpha_data.transpose((2,0,1))
        else:
            sys.stderr.write('ERROR: broadcast on 2D or 3D input is not supported yet in {}\n'.format(subcode))
            sys.exit(1)
        sn.prelu.alpha_shape = [b, c, h, w]

        input_offset = i_tensor['quantization']['zero_point'][0]
        alpha_offset = alpha_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]

        input_scale = i_tensor['quantization']['scale']
        alpha_scale = alpha_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']
        sn.prelu.iscale = input_scale[0]
        sn.prelu.ascale = alpha_scale[0]
        sn.prelu.oscale = output_scale[0]

        sn.prelu.optimized = 0
        sn.prelu.vci_int8 = -1
        sn.prelu.maps_at_once = 1

        sf = input_scale[0] * alpha_scale[0] / output_scale[0] 
        scaled_filters = [sf * (_ - alpha_offset) for _ in alpha_data.flatten()]
        valid_filters = [_ < 2**(15-Q16) for _ in scaled_filters]
        if (np.all(valid_filters)):
            sn.prelu.optimized = 1
            sn.prelu.maps_at_once = 8

            transform = lambda s: s
            lut, first, last, step_vals, step_indices = LUTPopulateInt8(input_scale[0], input_offset, output_scale[0], output_offset, transform)

            if VCI_LUT:
                sn.prelu.vci_int8 = len(weights)
                fmt = "{}B".format(len(lut))
                weights += struct.pack(fmt, *lut)
        else:
            print(subcode, input_scale[0], filter_scale[0], output_scale[0], "WARNING can't optimize")

        real_input_multiplier = [1.0]
        real_output_multiplier = (np.asarray(input_scale) / np.asarray(output_scale) ).tolist()
        real_alpha_multiplier = (np.asarray(input_scale) * np.asarray(alpha_scale) / np.asarray(output_scale)).tolist()

        output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)
        alpha_multiplier, alpha_shift = get_quantized_multiplier(real_alpha_multiplier)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.prelu.alpha_multiplier = len(weights)
        fmt = "{}i".format(len(alpha_multiplier))
        weights += struct.pack(fmt, *alpha_multiplier)

        sn.prelu.alpha_shift = len(weights)
        fmt = "{}i".format(len(alpha_shift))
        weights += struct.pack(fmt, *alpha_shift)

        sn.prelu.alpha_data = len(weights)
        fmt = "{}b".format(len(alpha_data.flatten()))
        weights += struct.pack(fmt, *alpha_data.flatten())

        sn.input_offset = input_offset
        sn.output_offset = output_offset
        sn.prelu.alpha_offset = alpha_offset

        subnode_array.append(sn)

    elif subcode == 'LEAKY_RELU':
        sn.type = BuiltinOperator.LEAKY_RELU

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']

        alpha = opts['alpha']
        real_input_multiplier = [1.0]
        real_output_multiplier = (np.asarray(input_scale) / np.asarray(output_scale) ).tolist()
        real_alpha_multiplier = [o*alpha for o in real_output_multiplier]

        output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)
        alpha_multiplier, alpha_shift = get_quantized_multiplier(real_alpha_multiplier)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.leakyrelu.alpha_multiplier = len(weights)
        fmt = "{}i".format(len(alpha_multiplier))
        weights += struct.pack(fmt, *alpha_multiplier)

        sn.leakyrelu.alpha_shift = len(weights)
        fmt = "{}i".format(len(alpha_shift))
        weights += struct.pack(fmt, *alpha_shift)

        sn.input_offset = input_offset
        sn.output_offset = output_offset

        subnode_array.append(sn)

    elif subcode in ['RELU', 'RELU6', 'RELU_N1_TO_1', 'RELU_0_TO_1']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']


        # TODO
        # quantized_activation_min = max(-127,output_offset + int(roundf(act_min / output_scale)))
        # quantized_activation_max = min(128, int(roundf(act_max / output_scale)))
        sn.activation_min = max(-128, output_offset)
        if subcode == 'RELU6':
            sn.activation_max = min(127, output_offset + int(round(6.0 / output_scale[0])))
        elif subcode == 'RELU_0_TO_1':
            # sn.activation_min = max(-128, output_offset + int(round(0.0 / output_scale[0])))
            sn.activation_max = min(127, output_offset + int(round(1.0 / output_scale[0])))
        elif subcode == 'RELU_N1_TO_1':
            sn.activation_min = max(-128, output_offset + int(round(-1.0 / output_scale[0])))
            sn.activation_max = min(127, output_offset + int(round(1.0 / output_scale[0])))

        real_output_multiplier = (np.asarray(input_scale) / np.asarray(output_scale) ).tolist()
        output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.input_offset = input_offset
        sn.output_offset = output_offset

        subnode_array.append(sn)

    elif subcode in ['ARG_MIN', 'ARG_MAX', 'TOPK_V2']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")
        # sys.stderr.write('\nsubop=={}\n'.format(subop))
        axis = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        # tflite: n,y,x,c {0, -3, -2, -1}
        # tflite: n,y,x,c {0, 1, 2, 3}
        # vnnx: 1,c,y,x {0, 3, 1, 2}
        # neg vnnx: 1,c,y,x {-4, -3, -2, -1}
        if axis == -1 or axis == 3:
            sn.reduce8.axis =  -3 #vnnx channels first
            axis_list = np.array(range(input_shape[1]))
            sn.reduce8.axis_list = len(weights)
            fmt = "{}b".format(len(axis_list.flatten()))
            weights += struct.pack(fmt, *axis_list.flatten())
        elif axis == 0:
            sys.stderr.write('ERROR: reduction on axis {} is not supported in {}\n'.format(axis,subcode))
            sys.exit(1)
        elif axis == -2 or axis == 2:
            sn.reduce8.axis =  -2 #vnnx channels first
            sys.stderr.write('ERROR: reduction on axis {} is not supported in {}\n'.format(axis,subcode))
            sys.exit(1)
        elif axis == -3 or axis == 1:
            sn.reduce8.axis =  -1 #vnnx channels first
            sys.stderr.write('ERROR: reduction on axis {} is not supported in {}\n'.format(axis,subcode))
            sys.exit(1)
        else:
            sys.stderr.write('ERROR: reduction on axis {} is not supported in {}\n'.format(axis,subcode))
            sys.exit(1)

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        input_offset = 0
        output_offset = 0

        input_scale = 1.
        output_scale = 1.

        assert(len(o_tensor['shape']) == len(i_tensor['shape']) - 1)
        
        sn.reduce8.arg_max = 0
        if subcode == 'ARG_MAX':
            sn.reduce8.arg_max = 1
        
        subnode_array.append(sn)

    elif subcode == "SLICE":
        sn.type = BuiltinOperator.SLICE

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        begin = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        size = get_numpy_data_from_index(subop['inputs'][2], tensors, buffers)

        assert(len(begin) <= 4)
        assert(len(size) <= 4)

        begin, _ = channels_first_shape(begin)
        size, _ = channels_first_shape(size)
        ishape, _ = channels_first_shape(i_tensor['shape'])

        begin = pad_list(begin, 4, 0)
        size = pad_list(size, 4)
        ishape = pad_list(ishape, 4)
        for i,d in enumerate(size):
            if d == -1:
                size[i] = ishape[i]

        sn.SliceOptions.begin = begin
        sn.SliceOptions.end = [sum(x) for x in zip(sn.SliceOptions.begin, size)]
        sn.SliceOptions.stride = [1,1,1,1]
        subnode_array.append(sn)

    elif subcode == "STRIDED_SLICE":
        sn.type = BuiltinOperator.SLICE

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        begin = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        end = get_numpy_data_from_index(subop['inputs'][2], tensors, buffers)
        stride = get_numpy_data_from_index(subop['inputs'][3], tensors, buffers)

        oshape, _ = channels_first_shape(o_tensor['shape'])
        begin, _ = channels_first_shape(begin)
        end, _ = channels_first_shape(end)
        stride, _ = channels_first_shape(stride)

        oshape = pad_list(oshape, 4)
        begin = pad_list(begin, 4, 0)
        end = pad_list(end, 4)
        stride = pad_list(stride, 4)

        sn.SliceOptions.begin = begin
        sn.SliceOptions.stride = stride
        sn.SliceOptions.end = [_ if _ != 0 else oshape[idx] for idx, _ in enumerate(end)]

        subnode_array.append(sn)

    elif subcode == "GATHER":
        sn.type = BuiltinOperator.GATHER
        axis = opts['axis']
        batch_dims = opts['batch_dims']
        # TODO: generalize for batch_dims > 0
        if batch_dims != 0:
            sys.stderr.write('ERROR: batch_dim = {} for GATHER is not implemented\n'.format(batch_dims))

        
        coord_data_tensor = tensors[subop['inputs'][1]]
        coord_data = []
        swap_input_order = subop['inputs'][1] < subop['inputs'][0]
        # if 'buffer' in coord_data_tensor and 'params' not in coord_data_tensor:
        #     swap_input_order = 1
        #     coord_data_tensor = tensors[subop['inputs'][1]]
        #     coord_data = np.random.randint(0, 20 + 1, coord_data_tensor['shape'])#get_numpy_data_from_index(subop['inputs'][0], tensors, buffers)
        #     # sys.stderr.write("\n------------------- coord_data{} {}".format(coord_data, coord_data_tensor['shape']))
        #     coord_data = coord_data.flatten()
        # if swap_input_order:
            #Error here because coord_data are not const, so there is nothing inside the buffer
        coord_data = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)  
        # assert (len(coord_data_tensor['shape']) == 1)

        input_shape = tensors[subop['inputs'][0]]['shape']
        output_shape = tensors[subop['outputs'][0]]['shape']

        pos_axis = axis
        if axis < 0:
            pos_axis = len(input_shape) + axis
        # axis_size = input_shape[axis]
        # batch_size = int(np.prod(input_shape[:batch_dims]))
        # outer_size = int(np.prod(input_shape[batch_dims:axis]))
        # inner_size = int(np.prod(input_shape[pos_axis+1:]))
        # coord_size = coord_data_tensor['shape'][0]
        # print(batch_dims, pos_axis)
        # print('ax', axis_size, 'bat', batch_size, 'out', outer_size, 'in', inner_size, 'cord', coord_size)

        # # o(1, 3, 1, 2), i(1, 20, 1, 2)
        # print('batch gap', outer_size*coord_size*inner_size, 'map gap', coord_size*inner_size, 'elem gap', inner_size)
        # print('batch gap', outer_size*axis_size*inner_size, 'map gap', axis_size*inner_size, 'elem gap', inner_size)
    
        # sys.stderr.write("\n-------------------output_shape {}".format(output_shape))
        cinput_shape, _ = channels_first_shape(input_shape)
        coutput_shape, _ = channels_first_shape(output_shape)
        caxis = channels_first_axis(axis, len(input_shape))
        sn.GatherOptions.swap_input_order =swap_input_order
        sn.GatherOptions.axis_size = cinput_shape[caxis]
        sn.GatherOptions.batch_size = int(np.prod(cinput_shape[:batch_dims]))
        sn.GatherOptions.outer_size = int(np.prod(cinput_shape[batch_dims:caxis]))
        sn.GatherOptions.inner_size = int(np.prod(cinput_shape[caxis+1:]))
        sn.GatherOptions.coord_size = coord_data_tensor['shape'][0]

        sn.GatherOptions.batch_dims = batch_dims
        sn.GatherOptions.axis = caxis
        # print(cinput_shape, coutput_shape)
        # print(batch_dims, pos_axis)
        # print('ax', sn.GatherOptions.axis_size, 'bat', sn.GatherOptions.batch_size, 'out', sn.GatherOptions.outer_size, 'in', sn.GatherOptions.inner_size, 'cord', sn.GatherOptions.coord_size)

        # print('batch gap', sn.GatherOptions.outer_size*sn.GatherOptions.coord_size*sn.GatherOptions.inner_size, 'map gap', sn.GatherOptions.coord_size*sn.GatherOptions.inner_size, 'elem gap', sn.GatherOptions.inner_size)
        # print('batch gap', sn.GatherOptions.outer_size*sn.GatherOptions.axis_size*sn.GatherOptions.inner_size, 'map gap', sn.GatherOptions.axis_size*sn.GatherOptions.inner_size, 'elem gap', sn.GatherOptions.inner_size)

        sn.GatherOptions.coord_data = len(weights)
        fmt = "{}i".format(len(coord_data))
        weights += struct.pack(fmt, *coord_data)
        
        subnode_array.append(sn)

    elif subcode in ['DEPTH_TO_SPACE', 'SPACE_TO_DEPTH']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")
        subnode_array.append(sn)

    elif subcode in ['BATCH_TO_SPACE_ND', 'SPACE_TO_BATCH_ND']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        block_shape_data = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        paddings_data = get_numpy_data_from_index(subop['inputs'][2], tensors, buffers)
        
        if subcode == "SPACE_TO_BATCH_ND":
            sn.SpaceToBatchNDOptions.block_shape_data = len(weights)
            fmt = "{}i".format(len(block_shape_data))
            weights += struct.pack(fmt, *block_shape_data)

            sn.SpaceToBatchNDOptions.paddings_data = len(weights)
            fmt = "{}i".format(len(paddings_data.flatten()))
            weights += struct.pack(fmt, *paddings_data.flatten())

        elif subcode == "BATCH_TO_SPACE_ND":
            sn.BatchToSpaceNDOptions.block_shape_data = len(weights)
            fmt = "{}i".format(len(block_shape_data))
            weights += struct.pack(fmt, *block_shape_data)

            sn.BatchToSpaceNDOptions.crop_data = len(weights)
            fmt = "{}i".format(len(paddings_data.flatten()))
            weights += struct.pack(fmt, *paddings_data.flatten())
            # sys.stderr.write("\n-------------------BATCH_TO_SPACE_ND Test1")

        subnode_array.append(sn)

    elif subcode in ['ABS']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']


        real_output_multiplier = (np.asarray(input_scale) / np.asarray(output_scale) ).tolist()

        output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.input_offset = input_offset
        sn.output_offset = output_offset

        subnode_array.append(sn)

    elif subcode == "L2_NORMALIZATION":
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]

        input_zero_point = i_tensor['quantization']['zero_point'][0]

        sn.input_offset = input_zero_point

        subnode_array.append(sn)
    
    elif subcode == 'NEG':
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        # i_tensor = tensors[subop['inputs'][0]]
        # o_tensor = tensors[subop['outputs'][0]]

        # input_zero_point = i_tensor['quantization']['zero_point'][0]

        # sn.input_offset = input_zero_point

        subnode_array.append(sn)
    
    elif subcode == "EMBEDDING_LOOKUP":
        sn.type = getattr(BuiltinOperator, f"{subcode}")
        embedding = sn.embedding
        i_tensor = tensors[subop['inputs'][0]]
        i_tensor_value = tensors[subop['inputs'][1]]

        color_data = get_numpy_data(i_tensor_value, buffers)
        h, c = tuple(i_tensor_value['shape'])
        color_dims = [h, c]

        embedding.colar_map_data = len(weights)
        fmt = "{}i".format(len(color_data))
        weights += struct.pack(fmt, *color_data.flatten())
        
        embedding.colar_map_dims = len(weights)
        fmt = "{}i".format(len(color_dims))
        weights += struct.pack(fmt, *color_dims)

        sn.input_offset = input_zero_point

        subnode_array.append(sn)

    elif subcode in ['MAXIMUM', 'MINIMUM']:
        if subcode == 'MAXIMUM':
            sn.type = BuiltinOperator.MAXIMUM
            sn.MinMaxOptions.max = 1
        else:
            sn.type = BuiltinOperator.MINIMUM
            sn.MinMaxOptions.max = 0

        i_tensor = tensors[subop['inputs'][0]]
        f_tensor = tensors[subop['inputs'][1]]
        filter_data = get_numpy_data(f_tensor, buffers)

        k, h, w, c = 1, 1, 1, 1
        if len(f_tensor['shape']) == 1:
            c = tuple(f_tensor['shape'])[0]
        elif len(f_tensor['shape']) == 2:
            k, c = tuple(f_tensor['shape'])
        elif len(f_tensor['shape']) == 3:
            h, w, c = tuple(f_tensor['shape'])
            filter_data = filter_data.transpose((2,0,1))
        elif len(f_tensor['shape']) == 4:
            k, h, w, c = tuple(f_tensor['shape'])
            filter_data = filter_data.transpose((0,3,1,2))
        elif len(f_tensor['shape']) == 5:
            assert(f_tensor['shape'][0] == 1)
            k, h, w, c = f_tensor['shape'][1:]
            filter_data = filter_data.transpose((0,1,4,2,3))

        sn.MinMaxOptions.filter_shape_dims = [k, c, h, w]    

        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0] 
        output_scale = o_tensor['quantization']['scale']
        
        filter_offset = f_tensor['quantization']['zero_point'][0]
        input_scale = i_tensor['quantization']['scale']
        filter_scale = f_tensor['quantization']['scale']
        
        input_multiplier, input_shift = get_quantized_multiplier(input_scale)
        filter_multiplier, filter_shift = get_quantized_multiplier(filter_scale)
        output_multiplier, output_shift = get_quantized_multiplier(output_scale)

        sn.activation_max = 127
        sn.activation_min = -128

        sn.input_multiplier = len(weights)
        fmt = "{}i".format(len(input_multiplier))
        weights += struct.pack(fmt, *input_multiplier)

        sn.input_shift = len(weights)
        fmt = "{}i".format(len(input_shift))
        weights += struct.pack(fmt, *input_shift)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.MinMaxOptions.filter_multiplier = len(weights)
        fmt = "{}i".format(len(filter_multiplier))
        weights += struct.pack(fmt, *filter_multiplier)

        sn.MinMaxOptions.filter_shift = len(weights)
        fmt = "{}i".format(len(filter_shift))
        weights += struct.pack(fmt, *filter_shift)
        
        sn.MinMaxOptions.filter_offset = filter_offset

        sn.MinMaxOptions.filter_data = len(weights)
        fmt = "{}b".format(len(filter_data.flatten()))
        weights += struct.pack(fmt, *filter_data.flatten())

        sn.input_offset = input_offset
        sn.output_offset = output_offset

        subnode_array.append(sn)

    elif subcode == "CAST":

        sn.type = getattr(BuiltinOperator, f"{subcode}")

        subnode_array.append(sn)
    else:
        sys.stderr.write('ERROR: Subnode {} not implemented\n'.format(subcode))
        sys.exit(1)

    sn.num_inputs = len(sn.tensor_array)

    tn = Tensor()
    tn.type = sn.output_data_type
    tn.shape = output_shape
    tn.dims = odims
    tn.scale = output_scale
    tn.zero = output_offset
    tn.id = o_tensor['buffer'] - 1
    tn.name = o_tensor['name']
    if not node_offloaded and engine_graphs_nx:
        add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, subgraph_idx)
    sn.tensor_array.append(tn)
    sn.num_outputs = 1 #TODO currently must have 1 output

    sn.num_tensors = len(sn.tensor_array)

    assert(sn.num_outputs + sn.num_inputs == sn.num_tensors)

    return subnode_array, sn.tensor_array, subop
