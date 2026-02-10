import numpy as np
import cv2
import os
import math
import json
import subprocess
import threading
import argparse
from numpy.random import RandomState
from itertools import chain
import struct 
import sys
import vbx.sim
from vbx.generate.utils import onnx_infer, onnx_input_shape, onnx_output_shape, openvino_infer, openvino_input_shape
from vbx.generate.utils import load_input, get_input_details, get_output_details


import vbx.postprocess.dataset as dataset
import vbx.postprocess.classifier as classifier
import vbx.postprocess.yolo as yolo

timeout_seconds = 500

def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_shape[0], model.output_shape


def preprocess_img_to_input_array(img, model, bgr, preprocess_scale=1, preprocess_mean=0):
    
    quantization_scale, quantization_zero = 0.0, 0
    input_dtype = np.float32 

    if model.endswith('.json'):
        with open(model) as f:
            data = json.load(f)
            input_shape = data['inputs'][0]['shape']            

            channels_last = input_shape[-1] < input_shape[-3]
            if channels_last:
                channels, input_height, input_width = input_shape[-1], input_shape[-3], input_shape[-2]
            else:
                channels, input_height, input_width = input_shape[-3], input_shape[-2], input_shape[-1]

            v_idx = [-1] + [i for i,_ in enumerate(model) if _ == 'v']
            t_idx = [-1] + [i for i,_ in enumerate(model) if _ == 't']
            if max(v_idx) > max(t_idx):
                channels_last = False
            elif max(v_idx) < max(t_idx):
                channels_last = True

            return None, input_height, input_width, channels_last


    elif model.endswith('.onnx'):
        input_shape = onnx_input_shape(model)[0]    
        
    elif model.endswith('.ucomp'):
        model_ucomp = model.replace('.ucomp','.vnnx')
        with open(model_ucomp, 'rb') as mf:
            model_ucomp = vbx.sim.Model(mf.read())
        input_shape = model_ucomp.input_shape[0]
        quantization_scale, quantization_zero = model_ucomp.input_scale_factor[0], model_ucomp.input_zeropoint[0]
        input_dtype = model_ucomp.input_dtypes[0]

    elif model.endswith('.vnnx'):
        with open(model, 'rb') as mf:
            vnnx_model = vbx.sim.Model(mf.read())
        input_shape = vnnx_model.input_shape[0]
        quantization_scale, quantization_zero = vnnx_model.input_scale_factor[0], vnnx_model.input_zeropoint[0]
        input_dtype = vnnx_model.input_dtypes[0]

    elif model.endswith('.xml'):
        weights=model.replace('.xml', '.bin')
        input_shape = openvino_input_shape(model, weights)[0] 
        
    elif model.endswith('.tflite'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=model)
        input_details = get_input_details(model)

        input_shape = input_details[0]["shape"]
        quantization_scale, quantization_zero = input_details[0].get('quantization', (0.0, 0))
        input_dtype = input_details[0]['dtype']

    channels_last = input_shape[-1] < input_shape[-3]

    if channels_last:
        channels, input_height, input_width = input_shape[-1], input_shape[-3], input_shape[-2]
    else:
        channels, input_height, input_width = input_shape[-3], input_shape[-2], input_shape[-1]
    

    if img.shape != (input_height, input_width, channels):
        img_resized = cv2.resize(img, (input_width, input_height)).clip(0, 255)
    else:
        img_resized = img
    if channels == 1:
        img_resized = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif not bgr:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    arr = (img_resized.astype(np.float32) - preprocess_mean) / preprocess_scale
    if channels == 1:
        arr = np.expand_dims(arr, axis=-1)
    if not channels_last:
        arr = arr.transpose(2,0,1)
    if len(input_shape) == 4:
        arr = np.expand_dims(arr, axis=0)
    
    if  quantization_scale != 0.0 and not model.endswith('.ucomp'):
        arr = (arr / quantization_scale) + quantization_zero  

    arr = arr.astype(input_dtype)

    return arr, input_height, input_width, channels_last
    

def transpose_outputs(outputs):
    t_outputs=[]
    for output in outputs:
        if len(output.shape) == 4:
            t_outputs.append(output.transpose((0,3,1,2)))
        elif len(output.shape) == 3:
            t_outputs.append(output.transpose((2,0,1)))
        else:
            t_outputs.append(output)
    return t_outputs

def nx_simulator(cmd):
    result = subprocess.run(cmd, timeout=timeout_seconds, capture_output=True, text=True)

def vnnx_simulator(arr, model, outputs, outputs_int8):
    with open(model, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    model_outputs = model.run([arr.flatten()])
    for o, output in enumerate(model_outputs):
        outputs_int8[o] = model_outputs[o]
        model_outputs[o] = model.output_scale_factor[o] * (model_outputs[o].astype(np.float32) - model.output_zeropoint[o])
        model_outputs[o] = model_outputs[o].reshape(model.output_shape[o])
        outputs[o] = model_outputs[o]
    
def model_run(arr, model):
    if model.endswith('.onnx'):
        outputs = onnx_infer(model, arr)
        output_shapes = onnx_output_shape(model)
    
    elif model.endswith('.ucomp'):
        prev_dir = os.getcwd()
        model = os.path.join(prev_dir, model)
        vnnx_model = model.replace('.ucomp', '.vnnx')
        model_basename = os.path.basename(vnnx_model).rsplit('.',1)[0]
        input_shape, output_shapes = get_vnnx_io_shapes(vnnx_model)
        
        if arr is None:
            with open(vnnx_model, 'rb') as mf:
                model_info = vbx.sim.Model(mf.read())
            arr = model_info.test_input[0]
        
        # Neuronix Simulator command'
        env = os.environ.copy()
        NX_SDK = env["NX_SDK"]
        os.chdir(os.path.normpath(NX_SDK))
        output_order = str(output_shapes)
        nx_simulator_cmd = [
            "python", "numeric_simulator/tflite_simulator.py",
            f"-m={model_basename}",
            f"-o={prev_dir}",
            "-r", output_order
        ]
        
        # Running MXP and Neuronix simulator concurrently using 
        outputs = [[] for _ in range(len(output_shapes))]
        outputs_int8 = [[] for _ in range(len(output_shapes))]
        thread1 = threading.Thread(target=vnnx_simulator, args=(arr, vnnx_model, outputs, outputs_int8))
        thread2 = threading.Thread(target=nx_simulator, args=(nx_simulator_cmd,))
        thread1.daemon = True
        thread2.daemon = True
        thread1.start()
        thread2.start()
        thread1.join(timeout=timeout_seconds)
        thread2.join(timeout=timeout_seconds)
        if thread1.is_alive():
            sys.exit()
        if thread2.is_alive():
            sys.exit()
        
        # Writing test input/outputs to the model binary file
        try:
            with open(model, 'r+b') as mf:
                header_size = struct.unpack('i', mf.read(4))[0]
                # writing user input to the .ucomp file if provided
                if (arr.shape[3] % 16 != 0):
                    padding_width = math.ceil(output.shape[3]/16)*16 - output.shape[3]
                    arr = np.pad(arr, ((0, 0), (0, 0), (0, 0), (0, padding_width)), mode="constant", constant_values=0)
                    input_size_offset = 20
                    input_size = int(np.prod(arr.shape))
                    mf.seek(input_size_offset, 0)
                    mf.write(input_size.to_bytes(4, 'little'))    
                input_bytearray = bytearray(arr.astype(np.int8))
                input_address_offset = 24
                mf.seek(input_address_offset, 0)
                input_address = header_size + struct.unpack('i', mf.read(4))[0]
                mf.seek(input_address, 0)
                mf.write(input_bytearray)
                print(f"Successfully modified {len(input_bytearray)} bytes at {hex(input_address)} offset in {model}")
                # writing folded_output of VBX3.0 simulator to the .ucomp and .nxo file
                num_output_offset = 16
                mf.seek(num_output_offset, 0)
                num_outputs = struct.unpack('i', mf.read(4))[0]
                # creating bytearray to be written to the .ucomp file
                used_idx = []
                for o, output in enumerate(outputs_int8):
                    output_bytearray = bytearray(outputs_int8[o].astype(np.int8))
                    output_size = len(output_bytearray)
                    for out_idx in range(num_outputs):
                        output_size_offset = 28 + (8 * out_idx)
                        mf.seek(output_size_offset, 0)
                        out_size = struct.unpack('i', mf.read(4))[0]
                        if (output_size == out_size) and (out_idx not in used_idx):
                            used_idx.append(out_idx)
                            break   
                    # Writing the test output to .ucomp file
                    output_address_offset = 32 + (8 * out_idx)
                    mf.seek(output_address_offset, 0)
                    output_address = header_size + struct.unpack('i', mf.read(4))[0]
                    mf.seek(output_address, 0)
                    mf.write(output_bytearray)
                    print(f"Successfully written Output #{o} {len(output_bytearray)} bytes at {hex(output_address)} offset in {model}")                
        except Exception as e:
            print(f"An error occurred: {e}")

        os.chdir(prev_dir)
                
    elif model.endswith('.vnnx'):
        input_shape, output_shapes = get_vnnx_io_shapes(model)
        with open(model, 'rb') as mf:
            model = vbx.sim.Model(mf.read())
        outputs = model.run([arr.flatten()])
        
        for o, output in enumerate(outputs):
            outputs[o] = model.output_scale_factor[o] * (outputs[o].astype(np.float32) - model.output_zeropoint[o])
            outputs[o] = outputs[o].reshape(model.output_shape[o])

    elif model.endswith('.xml'):
        output_shapes = None
        outputs = openvino_infer(model, arr)
    
    elif model.endswith('.tflite'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=model)
        interpreter.allocate_tensors()

        input_details = get_input_details(model)
        output_details = get_output_details(model)
        
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        outputs = []
        output_shapes =[]
        for o in range(len(output_details)):
            output_scale, output_zero_point = output_details[o].get('quantization', (0.0, 0))
            output = interpreter.get_tensor(output_details[o]['index'])
            if  output_scale != 0.0:
                output = output_scale * (output.astype(np.float32) - output_zero_point)
           
            outputs.append(output)
            output_shapes.append(output.shape)

    elif model.endswith('.json'):
        with open(model) as f:
            data = json.load(f)
            outputs, output_shapes = [], []
            for o, output in enumerate(data['outputs']):
                output_shapes.append(output['shape'])
                outputs.append((output['scale'] * (np.asarray(output['data']).astype(np.float32) - output['zero'])).reshape(output['shape']))

    return outputs, output_shapes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-i', '--input_data', default=None)
    args = parser.parse_args()
    
    # run inference
    outputs, _ = model_run(args.input_data, args.model)
    
    