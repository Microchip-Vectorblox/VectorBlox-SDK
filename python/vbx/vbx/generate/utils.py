import numpy as np
import collections
import cv2
import sys
import tensorflow as tf
import onnxruntime
import onnx
import openvino.inference_engine as ie
import argparse
import pathlib


def existing_dir(value):
    filepath = pathlib.Path(value)

    if not filepath.exists() or not filepath.is_dir():
        msg = f'Directory not found: {value}'
        raise argparse.ArgumentTypeError(msg)
    else:
        return value
    

def existing_file(value):
    filepath = pathlib.Path(value)

    if not filepath.exists():
        msg = f'File not found: {value}'
        raise argparse.ArgumentTypeError(msg)
    else:
        return value


MAP_TF_TO_NUMPY_TYPE = {
    tf.float32: np.float32,
    tf.float16: np.float16,
    tf.float64: np.float64,
    tf.complex64: np.complex64,
    tf.complex128: np.complex128,
    tf.int32: np.int32,
    tf.uint32: np.uint32,
    tf.uint8: np.uint8,
    tf.int8: np.int8,
    tf.uint16: np.uint16,
    tf.int16: np.int16,
    tf.int64: np.int64,
    tf.bool: np.bool_,
    tf.string: np.string_,
}


def create_tensor_data(dtype, shape, min_value=-100, max_value=100, int8_range=False):
  """Build tensor data spreading the range [min_value, max_value)."""

  if dtype in MAP_TF_TO_NUMPY_TYPE:
    dtype = MAP_TF_TO_NUMPY_TYPE[dtype]

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
        arr_inc = [arr[0]+ (i%255) for i in range(len(arr))]
        value = np.array(arr_inc).reshape(shape)
  elif dtype == tf.bool:
    value = np.random.choice([True, False], size=shape)
  elif dtype == np.string_:
    # Not the best strings, but they will do for some basic testing.
    letters = list(string.ascii_uppercase)
    return np.random.choice(letters, size=shape).astype(dtype)
  return np.dtype(dtype).type(value) if np.isscalar(value) else value.astype(dtype)


def one_elem(l):
    "get first element from  list of one element"
    assert len(l) == 1
    return l[0]
    

def pad_input(image, inputDims):
    img = cv2.imread(image)
    imgDims = np.array(img.shape[:2])

    resizeRatio = np.min(inputDims/imgDims)
    resizeDims = np.round(imgDims * resizeRatio).astype('int')
    imgResize = cv2.resize(img.astype('float32'), (resizeDims[1],resizeDims[0]), interpolation=cv2.INTER_LINEAR)
    padTop = int((inputDims[0]-resizeDims[0])/2)
    padBottom = inputDims[0]-resizeDims[0] - padTop
    padLeft = int((inputDims[1]-resizeDims[1])/2)
    padRight = inputDims[1]-resizeDims[1] - padLeft
    imgPad = cv2.copyMakeBorder(imgResize/255.0, padTop, padBottom, padLeft, padRight, cv2.BORDER_CONSTANT, value=[0.5,0.5,0.5])
    modelInput = imgPad.swapaxes(0,2).swapaxes(1,2)
    return modelInput


def convert_to_fixedpoint(data, dtype):
    # this should go away eventually, and always input uint8 rather than fixedpoint Q1.7
    if dtype == np.int16:
        shift_amt = 13
    elif dtype == np.int8:
        shift_amt = 7
    clip_max, clip_min = (1 << shift_amt)-1, -(1 << shift_amt)
    float_img = data.astype(np.float32)/255 * (1 << shift_amt) + 0.5

    fixedpoint_img = np.clip(float_img, clip_min, clip_max).astype(dtype)
    return fixedpoint_img


# used in unit tests also
def match_shape(act, ref_shape, to_tfl):
    while len(act.shape) < len(ref_shape):
        act = np.expand_dims(act, axis=0)
    while len(act.shape) > len(ref_shape):
        try:
            act = np.squeeze(act, axis=0)
        except ValueError as e:
            raise Exception("Axis 0 (batch) is not equal to one, cannot be squeezed! Activation shape {} reference shape {}".format(act.shape, ref_shape))

    # TODO may this still cause an issue if the shapes are same but still needs a transpose? -> yes it does
    # TODO are there cases where we wouldnt need to transpose?
    # if act.shape == tuple(ref_shape):
    #     return act 

    if len(act.shape)==5:
        if to_tfl:
            act = act.transpose(0,1,3,4,2)
        else:
            act = act.transpose(0,1,4,2,3)
    elif len(act.shape)==4:
        if to_tfl:
            act = act.transpose(0,2,3,1)
        else:
            act = act.transpose(0,3,1,2)
    elif len(act.shape)==3:
        if to_tfl:
            act = act.transpose(1,2,0)
        else:
            act = act.transpose(2,0,1)
    # elif len(act.shape) == 2:
    #     act = act.transpose((1,0))
    return act

# used in unit tests also
def calc_diff(src, dst, threshold=1):
    
    if len(dst.shape)==3:
        dst = dst.transpose(1,2,0)

    # all_within_threshold = np.allclose(src, dst, atol=threshold)
    all_within_threshold = not (np.max(np.abs(src-dst)) > threshold)
    abs_diff = np.abs(src - dst) 
    diff = src - dst

    # abs_counter = collections.Counter(abs_diff.flatten().tolist())
    counter = collections.Counter(diff.flatten().tolist())

    total_vals_diff = 0
    for diff, v in counter.items():
        if diff == 0:
            continue
        total_vals_diff += v

    return all_within_threshold, abs_diff, total_vals_diff, counter

# from onnx_infer
def load_input(src, scale, input_shape, rgb=False, norm=False):
    try:
        channels = input_shape[-3]
    except:
        channels = 1
    height = input_shape[-2]
    width = input_shape[-1]
    ext = src.split('.')[-1].lower()
    if ext in ['npy']:
        arr = np.load(src)
    elif ext in ['jpg', 'jpeg', 'png']:
        if channels == 3:
            img = cv2.imread(src)
            if rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if norm:
                img = img.astype(np.float32) / 255.
            if img is None:
                sys.stderr.write("Error Unable to read image file {}\n".format(src))
                sys.exit(1)
            if height and width and img.shape[:2] != [height, width]:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            arr = img.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
        else:
            img = cv2.imread(src, 0)
            if img is None:
                sys.stderr.write("Error Unable to read image file {}\n".format(src))
                sys.exit(1)
            if height and width and img.shape != [height, width]:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            arr = img.astype(np.float32)
            arr = np.expand_dims(arr, axis=0)
    if scale:
        arr = arr * scale
    arr = np.expand_dims(arr, axis=0)

    return arr


def onnx_input_shape(model_file):
    model = onnx.load(model_file)

    input_shapes = []
    for input in model.graph.input:
        tensor_type = input.type.tensor_type
        if (tensor_type.HasField("shape")):
            input_shapes.append([d.dim_value for d in tensor_type.shape.dim if d.HasField("dim_value") ])
    return input_shapes

def onnx_output_shape(model_file):
    model = onnx.load(model_file)

    output_shapes = []
    for output in model.graph.output:
        tensor_type = output.type.tensor_type
        if (tensor_type.HasField("shape")):
            output_shapes.append([d.dim_value for d in tensor_type.shape.dim if d.HasField("dim_value") ])
    return output_shapes

def onnx_infer(onnx_model, input_array, flatten=False):
    model = onnx.load(onnx_model)
    session_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(model.SerializeToString())
    input_name = session.get_inputs()[0].name
    
    if flatten:
        return [o.flatten() for o in session.run([], {input_name: input_array})]
    else:
        return [o for o in session.run([], {input_name: input_array})]


def openvino_input_shape(xml, weights):
    core = ie.IECore()
    net = core.read_network(model=xml, weights=weights)

    exec_net = core.load_network(network=net, device_name="CPU")
    inputs = [k for k in net.input_info.keys()]
    return [exec_net.requests[0].input_blobs[i].buffer.shape for i in inputs]


def openvino_infer(xml_model, input_array, flatten=False):
    weights=xml_model.replace('.xml', '.bin')
    core = ie.IECore()
    net = core.read_network(model=xml_model, weights=weights)
    exec_net = core.load_network(network=net, device_name="CPU")
    assert(len(net.input_info) == 1)
    i0 = [k for k in net.input_info.keys()][0]

    exec_net.requests[0].input_blobs[i0].buffer[:] = input_array
    exec_net.requests[0].infer()
    outputs = [k for k in net.outputs.keys()]
    if flatten:
        return [exec_net.requests[0].output_blobs[o].buffer.flatten() for o in outputs]
    else:
        return [exec_net.requests[0].output_blobs[o].buffer for o in outputs]
