import numpy as np
import collections
import cv2
import sys
import tensorflow as tf
import onnxruntime
import onnx
import argparse
import pathlib
import json
import orjson
import os
import sys, shutil, subprocess, glob
from contextlib import contextmanager


@contextmanager
def sp_invalid_exception_catcher(sp, opcode, tmp_dir, graph_idx, tmp_dir_obj):
    try:
        yield
    except AssertionError as e:
        errors_dir = os.path.join(os.path.join(os.getcwd(), 'failing_subgraphs'))
        if os.path.exists(errors_dir):
            shutil.rmtree(errors_dir)
            os.mkdir(errors_dir)
        else:
            os.mkdir(errors_dir)

        subgraph_name = ''
        if tmp_dir is not None and graph_idx is not None:
           list_graph = sorted(
                glob.glob(os.path.join(tmp_dir, 'subgraphs/*.tflite')),
                key=lambda x: int(x.split('.')[-2])
            )
           for i, subgraph_name in enumerate(list_graph):
               if i== graph_idx:
                   shutil.copy(subgraph_name, errors_dir)
                   subgraph_name = os.path.basename(subgraph_name)
                   break
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()

        error_msg = (
            "\n" + "="*75 + "\n"
            "\033[31m" + "CRITICAL ERROR: Memory Allocation Failure".center(75) + "\033[0m\n"
            "="*1 + "\n"
            f"\n\033[31mLayer Type:\033[0m {opcode}\n"
            f"\033[31mError:\033[0m Layer cannot fit within the memory scratchpad\n"
            f"\033[31mScratchpad Capacity:\033[0m {sp / 1024:.1f} KB\n"
            f"\n\033[31mFailing Subgraph:\033[0m {subgraph_name}\n"
            f"\033[31mLocation:\033[0m failing_subgraphs/{subgraph_name}\n"
            f"\nWe are continuously working to improve the SDK.\n"
            f"\n\033[36mFor further assistance:\033[0m\n"
            f"  Email: vectorblox@microchip.com\n"
            f"  Repository: https://github.com/Microchip-Vectorblox/\n"
            + "="*75 + "\n"
        )
        
        sys.stderr.write(error_msg)
        sys.exit(1)

def get_input_details(model):
    interpreter= tf.lite.Interpreter(model_path=model)
    return sorted(interpreter.get_input_details(), key=lambda s: s['index'])


def get_output_details(model):
    interpreter= tf.lite.Interpreter(model_path=model)
    return sorted(interpreter.get_output_details(), key=lambda s: s['index'])


def json_load(path):
    # with open(path) as f:
    #     return json.loads(f.read())

    with open(path, 'rb') as f:
        return orjson.loads(f.read())


def json_dump(obj, path):
    # with open(path, 'w') as f:
    #     return f.write(json.dumps(obj))

    with open(path, 'wb') as f:
        return f.write(orjson.dumps(obj))


# compress and decompress mux controls
# controlOptions = [sel01,zero01,sel23,zero23]
# controls are repeated for the second half of the mux (input activations 4-7)
D = 0   # DNC
controlOptions = [[0,0,D,1],[1,0,D,1],[D,1,0,0],[D,1,1,0],[0,0,0,0],[0,0,1,0],[1,0,0,0],[1,0,1,0]] # 8 possible options
controlDecompress = np.zeros(len(controlOptions),dtype=int)
for n,c in enumerate(controlOptions):
    controlDecompress[n] = np.dot(c,[1,2,4,8])
# constant DECODE_SELECT_LUT : MATRIX_1x8:=(x"8",x"9",x"2",x"6",x"0",x"4",x"1",x"5"]);

def compressControl(ctrl,sign):
    R = ctrl.shape[0]
    ctrlComp = np.zeros(R,dtype=int)
    for r in range(R):
        ctrlCompL = np.where((controlOptions==ctrl[r,:4]).all(1))[0][0]
        ctrlCompH = np.where((controlOptions==ctrl[r,4:]).all(1))[0][0]
        ctrlComp[r] = np.array([ctrlCompL + 8*ctrlCompH + 64*sign[r]],dtype=int)
    return ctrlComp

def reconstructWeights(ctrl,sign,k,R):
    w = np.zeros(8,dtype=int)
    for r in range(R):
        if ctrl[r,1]==0:
            w[ctrl[r,0]] = k[r,0]
        if ctrl[r,3]==0:
            if sign[r]:
                w[ctrl[r,2]+2] = k[r,0]
            else:
                w[ctrl[r,2]+2] = -k[r,0]
        if ctrl[r,5]==0:
            w[ctrl[r,4]+4] = k[r,1]
        if ctrl[r,7]==0:
            if sign[r]:
                w[ctrl[r,6]+6] = k[r,1]
            else:
                w[ctrl[r,6]+6] = -k[r,1]
    return w

def decompressControl(ctrlComp):
    R = ctrlComp.size
    muxControl = np.zeros((R,8),dtype=int)
    signControl = np.zeros((R),dtype=int)
    for r in range(R):
        ctrlCompL = ctrlComp[r] % 8           # bits 0,1,2
        ctrlCompH = (ctrlComp[r]//8) % 8      # bits 3,4,5
        signControl[r] = ctrlComp[r]//64      # bit 6
        controlL = controlDecompress[ctrlCompL]
        controlL = [int(b) for b in bin(controlL)[2:].zfill(4)]
        controlL.reverse()
        controlH = controlDecompress[ctrlCompH]
        controlH = [int(b) for b in bin(controlH)[2:].zfill(4)]
        controlH.reverse()
        muxControl[r,:] = controlL + controlH # concatenate
    return muxControl,signControl

# quick compress for repeat 1
def compress1(w):
    ctrl = np.zeros(8,dtype=int)
    k = np.zeros(2,dtype=int)
    ctrl[0::2] = w[1::2]!=0
    ctrl[1::2] = (w.reshape(-1,2)==0).all(1)
    wSel = w[ctrl[[0,2,4,6]] + [0,2,4,6]]
    if not ctrl[[1,3]].any():   # first 2 are paired
        sign = wSel[0]==wSel[1]
    elif not ctrl[[5,7]].any(): # second 2 are paired
        sign = wSel[2]==wSel[3]
    else:
        sign = 1    # no pairs
    if ctrl[1]==0:
        k[0] = wSel[0]
    elif ctrl[3]==0:
        if sign:
            k[0] = wSel[1]
        else:
            k[0] = -wSel[1]
    else:
        ctrl[1] = 0
    if ctrl[5]==0:
        k[1] = wSel[2]
    elif ctrl[7]==0:
        if sign:
            k[1] = wSel[3]
        else:
            k[1] = -wSel[3]
    else:
        ctrl[5] = 0
    ctrl = ctrl.reshape(1,8)
    sign = [sign]
    k = k.reshape(1,2)
    ctrlComp = compressControl(ctrl,sign)
    success = (w==reconstructWeights(ctrl,sign,k,1)).all()
    return success,ctrl,sign,ctrlComp,k

# quick compress for repeat 2
def compress2(w):
    pairA = np.array([[0,2,4,6],
             [0,2,4,7],
             [0,2,5,6],
             [0,2,5,7],
             [0,3,4,6],
             [0,3,4,7],
             [0,3,5,6],
             [0,3,5,7]],dtype=int)
    pairB = np.zeros((8,4),dtype=int)
    for n,p in enumerate(pairA):
        pairB[n,:] = np.setdiff1d(np.arange(8),p)

    success = False
    for n in range(pairA.shape[0]):
        wA = w[pairA[n,:]]
        wB = w[pairB[n,:]]
        signLA = -1
        signHA = -1
        signLB = -1
        signHB = -1
        if wA[0]!=0 and wA[1]!=0:
            if abs(wA[0])!=abs(wA[1]):
                continue    # failed compression with this combination
            signLA = int(wA[0]==wA[1])
        if wA[2]!=0 and wA[3]!=0:
            if abs(wA[2])!=abs(wA[3]):
                continue    # failed compression with this combination
            signHA = int(wA[2]==wA[3])
        if wB[0]!=0 and wB[1]!=0:
            if abs(wB[0])!=abs(wB[1]):
                continue    # failed compression with this combination
            signLB = int(wB[0]==wB[1])
        if wB[2]!=0 and wB[3]!=0:
            if abs(wB[2])!=abs(wB[3]):
                continue    # failed compression with this combination
            signHB = int(wB[2]==wB[3])
        # make sure signs match between lower and upper
        if signLA>=0 and signHA>=0:
            if signLA != signHA:
                continue    # failed compression with this combination
        if signLB>=0 and signHB>=0:
            if signLB != signHB:
                continue    # failed compression with this combination
        success = True
        break
    if not success:
        return success,None,None,None,None
    
    ctrl = np.zeros((2,8),dtype=int)
    ctrl[0,0::2] = pairA[n]-[0,2,4,6]   # select A
    ctrl[1,0::2] = pairB[n]-[0,2,4,6]   # select B
    ctrl[0,1::2] = w[pairA[n]]==0
    ctrl[1,1::2] = w[pairB[n]]==0
    for r in range(2):
        for n in range(0,8,4):
            if ctrl[r,n+1]==1 and ctrl[r,n+3]==1:   # if both zeros are set, unset one (otherwise this control will not be controlOptions)
                ctrl[r,n+1] = 0
        for n in range(0,8,2):
            if ctrl[r,n+1]==1: # if zero is set, set select to 0 (otherwise this control will not be controlOptions)
                ctrl[r,n] = 0

    sign = [int(signLA==1 or signHA==1), int(signLB==1 or signHB==1)]

    k = np.zeros((2,2),dtype=int)
    for r in range(2):
        for n in range(2):
            if not ctrl[r,1+4*n]:
                k[r,n] = w[4*n+ctrl[r,4*n]]
            elif sign[r]:
                k[r,n] = w[2+4*n+ctrl[r,2+4*n]]
            else:
                k[r,n] = -w[2+4*n+ctrl[r,2+4*n]]
    
    ctrlComp = compressControl(ctrl,sign)

    # assert (w==reconstructWeights(ctrl,sign,k,2)).all() # this should always pass, so consider this a Python-only test
    success = (w==reconstructWeights(ctrl,sign,k,2)).all()
    return success,ctrl,sign,ctrlComp,k

# quick compress for repeat 4
def compress4(w):
    # there's only one configuration needed for repeat 4, and it always succeeds
    sign = [1,1,1,1]
    k = w[[0,4,1,5,2,6,3,7]].reshape(-1,2)
    ctrl = np.array([[0, 0, 0, 1, 0, 0, 0, 1],
                     [1, 0, 0, 1, 1, 0, 0, 1],
                     [0, 1, 0, 0, 0, 1, 0, 0],
                     [0, 1, 1, 0, 0, 1, 1, 0]],dtype=int)
    ctrlComp = np.array([64, 73, 82, 91],dtype=int)
    return True,ctrl,sign,ctrlComp,k

def randWeights(R, shape):
    # shape expected in hwck (unit test format), and should return hwck
    kh,kw,c,kernels = shape
    
    c_div_8 = c + ((8 - (c % 8)) if (c % 8 != 0) else 0)

    channel_groups = kh*kw*kernels*(c_div_8//8)
    weights = np.zeros(kh*kw*kernels*c_div_8, dtype=int)

    for group in range(channel_groups):
        if R==4:
            w = np.random.randint(-128,high=128,size=(8))
        else:
            ctrlComp = np.random.randint(128, size=R)
            k = np.random.randint(-128,high=128,size=(R,2))
            muxControl,signControl = decompressControl(ctrlComp)
            w = reconstructWeights(muxControl,signControl,k,R)
        weights[group*8 : (group+1)*8] = w

    # reshaped into k,h,w,c , then transposed to h,w,c,k for TF
    weights = weights.reshape(kernels,kh,kw,c_div_8)[:,:,:,:c]
    return weights.transpose(1,2,3,0)

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


def generate_inputs_outputs(tflite_model, int8_range=False):
    if isinstance(tflite_model, bytes):
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
    elif isinstance(tflite_model, str):
        interpreter = tf.lite.Interpreter(model_path=tflite_model)
    else:
        print('WARNING: invalid tflite model')
        return None
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_values = {}
    min_value = 0
    max_value = 20
    for i,input_detail in enumerate(sorted(input_details, key=lambda x: x['index'])):
        if input_detail["dtype"] == np.float32:
            min_value = -1
            max_value = 1
        elif input_detail["dtype"] == np.int8:
            min_value = -128
            max_value = 127
        elif input_detail["dtype"] == np.uint8:
            min_value = 0
            max_value = 255
        input_value = create_tensor_data(
                input_detail["dtype"],
                input_detail["shape"],
                min_value=min_value,
                max_value=max_value,
                int8_range=int8_range)
        interpreter.set_tensor(input_detail["index"], input_value)
        input_values.update({"i{}".format(i): input_value})
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_values = {}
    for o, output_detail in enumerate(sorted(output_details, key=lambda x: x['index'])):
        output_values.update({ "o{}".format(o): interpreter.get_tensor(output_detail["index"])})
    return input_values, output_values


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
    # if len(dst.shape)==3:
    #     dst = dst.transpose(1,2,0)
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
    import openvino.inference_engine as ie
    core = ie.IECore()
    net = core.read_network(model=xml, weights=weights)

    exec_net = core.load_network(network=net, device_name="CPU")
    inputs = [k for k in net.input_info.keys()]
    return [exec_net.requests[0].input_blobs[i].buffer.shape for i in inputs]


def openvino_infer(xml_model, input_array, flatten=False):
    import openvino.inference_engine as ie
    core = ie.IECore()
    weights=xml_model.replace('.xml', '.bin')
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
