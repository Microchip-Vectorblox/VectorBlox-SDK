import numpy as np
import cv2
import os
import math
import json

import vbx.sim
from vbx.generate.utils import onnx_infer, onnx_input_shape, onnx_output_shape, openvino_infer, openvino_input_shape
from vbx.generate.utils import load_input


import vbx.postprocess.dataset as dataset
import vbx.postprocess.classifier as classifier
import vbx.postprocess.yolo as yolo


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
            return None, input_shape

    elif model.endswith('.onnx'):
        input_shape = onnx_input_shape(model)[0]    
        
    elif model.endswith('.vnnx'):
        with open(model, 'rb') as mf:
            model = vbx.sim.Model(mf.read())
        input_shape = model.input_shape[0]
        quantization_scale, quantization_zero = model.input_scale_factor[0], model.input_zeropoint[0]
        input_dtype = model.input_dtypes[0]

    elif model.endswith('.xml'):
        weights=model.replace('.xml', '.bin')
        input_shape = openvino_input_shape(model, weights)[0] 
        

    elif model.endswith('.tflite'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=model)
        input_details = interpreter.get_input_details()
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
    
    if  quantization_scale != 0.0:
        arr = (arr / quantization_scale) + quantization_zero  

    arr = arr.astype(input_dtype)

    return arr ,input_shape
    

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


def model_run(arr,model):
    if model.endswith('.onnx'):
        outputs = onnx_infer(model, arr)
        output_shapes = onnx_output_shape(model)

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

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
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
