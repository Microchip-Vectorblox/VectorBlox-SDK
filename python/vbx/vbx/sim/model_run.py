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
import re
import vbx.sim
from vbx.generate.utils import onnx_infer, onnx_input_shape, onnx_output_shape, openvino_infer, openvino_input_shape
from vbx.generate.utils import load_input, get_input_details, get_output_details


import vbx.postprocess.dataset as dataset
import vbx.postprocess.classifier as classifier
import vbx.postprocess.yolo as yolo

timeout_seconds = 600

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

            channels_last = False
            if len(input_shape) == 2:
                channels, input_height, input_width = 1, 1,input_shape[-1]
            elif len(input_shape) == 3:
                channels, input_height, input_width = 1, input_shape[-2],input_shape[-1]
            elif len(input_shape) == 4:
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
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    elif not bgr:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    arr = (img_resized.astype(np.float32) - preprocess_mean) / preprocess_scale
    if channels == 1:
        arr = np.expand_dims(arr, axis=-1)
    if not channels_last:
        arr = arr.transpose(2,0,1)
    if len(input_shape) == 4:
        arr = np.expand_dims(arr, axis=0)
    
    if  quantization_scale != 0.0:
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

def nx_simulator(cmd, status):
    # status is a shared dict used to report the outcome back to model_run so a
    # slow/failed numeric reference run is surfaced explicitly rather than as an
    # uncaught thread exception that silently continues with unverified output.
    #
    # DEBUG: stream NX's stdout/stderr live (prefixed) instead of buffering it via
    # capture_output, so a crash/traceback shows up immediately instead of being
    # hidden until this process exits or the MXP side's 600s join timeout fires.
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        lines = []
        for line in proc.stdout:
            print(f"[NX] {line}", end='')
            lines.append(line)
        returncode = proc.wait(timeout=timeout_seconds)
        if returncode != 0:
            status['state'] = 'error'
            status['detail'] = ''.join(lines).strip() or f'exit code {returncode}'
        else:
            status['state'] = 'ok'
    except subprocess.TimeoutExpired:
        proc.kill()
        status['state'] = 'timeout'
        status['detail'] = f'numeric reference simulation exceeded {timeout_seconds}s'

def vnnx_simulator(arr, model, outputs):
    # Handle single input (arr is numpy array) or multiple inputs (arr is list)
    if isinstance(arr, list):
        # Multiple inputs - flatten each one
        inputs = [a.flatten() for a in arr]
    else:
        # Single input - keep backward compatibility
        inputs = [arr.flatten()]

    model_outputs = model.run(inputs)
    for o, output in enumerate(model_outputs):
        outputs[o] = model_outputs[o].reshape(model.output_shape[o])
    
def model_run(arr, model, apply_sc_and_mean=True):
    if model.endswith('.onnx'):
        outputs = onnx_infer(model, arr)
        output_shapes = onnx_output_shape(model)
    
    elif model.endswith('.ucomp'):
        prev_dir = os.getcwd()
        model = os.path.join(prev_dir, model)
        vnnx_model = model.replace('.ucomp', '.vnnx')
        model_basename = os.path.basename(vnnx_model).rsplit('.',1)[0]
        input_shape, output_shapes = get_vnnx_io_shapes(vnnx_model)
        
        with open(vnnx_model, 'rb') as mf:
            vnnx_model = vbx.sim.Model(mf.read())

        # Handle multiple inputs
        num_inputs = vnnx_model.num_inputs
        if arr is None:
            if num_inputs == 1:
                arr = vnnx_model.test_input[0]
            else:
                arr = vnnx_model.test_input  # List of input arrays

        # Unique per-run id so the NX numeric simulator (spawned as a subprocess below)
        # and the MXP cmodel (run in-process via model.run -> fia_cmodel.c) agree on a
        # private set of POSIX shared-memory/semaphore names. Setting it in os.environ
        # makes the in-process C library see it via getenv() and the subprocess inherit
        # it, so multiple model_run instances can execute concurrently without clobbering
        # each other's sync objects.
        sim_run_id = re.sub(r'[^A-Za-z0-9_]', '_', f"{model_basename}_{os.getpid()}")
        os.environ['SIM_RUN_ID'] = sim_run_id

        # Neuronix Simulator command'
        env = os.environ.copy()
        NX_SDK = env["NX_SDK"]
        os.chdir(os.path.normpath(NX_SDK))
        nx_simulator_cmd = [
            "python", "numeric_simulator/tflite_simulator.py",
            f"-m={model_basename}",
            f"-o={prev_dir}",
        ]
        
        # Running MXP and Neuronix simulator concurrently using
        outputs = [[] for _ in range(len(output_shapes))]
        nx_status = {'state': 'unknown', 'detail': ''}
        thread1 = threading.Thread(target=vnnx_simulator, args=(arr, vnnx_model, outputs))
        thread2 = threading.Thread(target=nx_simulator, args=(nx_simulator_cmd, nx_status))
        thread1.daemon = True
        thread2.daemon = True
        thread1.start()
        thread2.start()
        thread1.join(timeout=timeout_seconds)
        thread2.join(timeout=timeout_seconds)
        if thread1.is_alive():
            sys.exit()
        # Surface a skipped/failed numeric reference run explicitly. The model was still
        # generated; only the NX-vs-MXP numeric equivalence check did not run.
        nx_skip_reason = None
        if thread2.is_alive():
            # Numeric reference simulator did not return even after the join timeout.
            nx_skip_reason = (f"did not complete within {timeout_seconds}s and is still "
                              f"running; skipping numeric verification")
        elif nx_status['state'] == 'timeout':
            nx_skip_reason = f"skipped — {nx_status['detail']}"
        elif nx_status['state'] == 'error':
            nx_skip_reason = f"failed — {nx_status['detail']}"
        if nx_skip_reason is not None:
            print(f"WARNING: NX numeric reference simulation {nx_skip_reason}. The model was "
                  f"generated successfully but NX-vs-MXP numeric equivalence was NOT verified.",
                  file=sys.stderr)

        # Writing test input/outputs to the model binary file
        try:
            with open(model, 'r+b') as mf:
                header_size = struct.unpack('i', mf.read(4))[0]

                # Handle multiple inputs
                if isinstance(arr, list):
                    # Multiple inputs - write each one
                    for input_idx, input_arr in enumerate(arr):
                        input_bytearray = bytearray(input_arr.astype(np.int8))
                        input_address_offset = 24 + (8 * input_idx)  # Each input has 8 bytes for offset
                        mf.seek(input_address_offset, 0)
                        input_address = header_size + struct.unpack('i', mf.read(4))[0]
                        mf.seek(input_address, 0)
                        mf.write(input_bytearray)
                        print(f"Successfully modified input #{input_idx} {len(input_bytearray)} bytes at {hex(input_address)} offset in {model}")
                else:
                    # Single input - original logic
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
                for o, output in enumerate(outputs):
                    output_bytearray = bytearray(outputs[o].astype(np.int8))
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

        for o, output in enumerate(outputs):
            # TODO Added to fix Midas-V2-Quantized which needs to be in int32 not FP32 to preserve RBGA structure for pixels
            # Would like to find a cleaner fix for this instead of adding extra 2 conditionals in model run
            # Tried converting back to int32 in segmentation.py but lost accurracy when converting FP32->Int32
            if apply_sc_and_mean:
                outputs[o] = vnnx_model.output_scale_factor[o] * (outputs[o].astype(np.float32) - vnnx_model.output_zeropoint[o])

            outputs[o] = outputs[o].reshape(vnnx_model.output_shape[o])
                
    elif model.endswith('.vnnx'):
        input_shape, output_shapes = get_vnnx_io_shapes(model)
        with open(model, 'rb') as mf:
            vnnx_model = vbx.sim.Model(mf.read())

        # Handle multiple inputs
        if isinstance(arr, list):
            inputs = [a.flatten() for a in arr]
        else:
            inputs = [arr.flatten()]

        outputs = vnnx_model.run(inputs)
        model = vnnx_model  # Keep for compatibility below
        
        for o, output in enumerate(outputs):
            # TODO Added to fix Midas-V2-Quantized which needs to be in int32 not FP32 to preserve RBGA structure for pixels
            # Would like to find a cleaner fix for this instead of adding extra 2 conditionals in model run
            # Tried converting back to int32 in segmentation.py but lost accurracy when converting FP32->Int32
            if apply_sc_and_mean:
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
            # TODO Added to fix Midas-V2-Quantized, needs cleaner fix, refer to first Midas-V2-Quantized comment for more info
            if output_scale != 0.0 and apply_sc_and_mean:
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