import vbx.sim
import cv2
import numpy as np
import argparse
import os
import math

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import load_input

def cosine_distance(arr0, arr1):
    return np.sum(arr0*arr1)/(np.sqrt(np.sum(arr0*arr0)) * np.sqrt(np.sum(arr1*arr1)))


def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_shape[0], model.output_shape


def vnnx_infer(vnnx_model, input_array):
    with open(vnnx_model, 'rb') as mf:
        model = vbx.sim.Model(mf.read())

    input_array = input_array.astype(np.float32)
    input_array = (input_array / model.input_scale_factor[0]) + model.input_zeropoint[0]
    flattened = (input_array.flatten()).astype(model.input_dtypes[0])
    outputs = model.run([flattened])
    out_scaled = model.output_scale_factor[0] * (outputs[0].astype(np.float32) - model.output_zeropoint[0])
    output = out_scaled.reshape(model.output_shape[0])

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))    
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image1')
    parser.add_argument('image2')
    parser.add_argument('-b', '--bgr', action='store_true')
    args = parser.parse_args()
    
    if not os.path.isfile(args.image1):
        print('Error: {} could not be read'.format(args.image1))
        os._exit(1)
    if not os.path.isfile(args.image2):
        print('Error: {} could not be read'.format(args.image2))
        os._exit(1)

    if args.model.endswith('.vnnx'):
        input_shape, _ = get_vnnx_io_shapes(args.model)

        input_array = load_input(args.image1, 1., input_shape, (not args.bgr))
        image1_out = vnnx_infer(args.model,input_array)

        input_array = load_input(args.image2, 1., input_shape, (not args.bgr))
        image2_out = vnnx_infer(args.model,input_array)
        
    elif args.model.endswith('.xml'):
        weights=args.model.replace('.xml', '.bin')
        input_shape = openvino_input_shape(args.model, weights)[0]

        input_array = load_input(args.image1, 1., input_shape, (not args.bgr))
        image1_out = openvino_infer(args.model,input_array)[0]

        input_array = load_input(args.image2, 1., input_shape, (not args.bgr))
        image2_out = openvino_infer(args.model,input_array)[0]

    elif args.model.endswith('.tflite'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        h, w = input_details[0]['shape'][1:3]

        img1 = cv2.imread(args.image1)
        if img1.shape != (h, w, 3):
            img1 = cv2.resize(img1, (h, w)).clip(0, 255)
        if not args.bgr:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1_resized = img1.astype(np.float32)
        img1_resized = np.expand_dims(img1_resized, axis=0)
        input_scale, input_zero_point = input_details[0].get('quantization', (0.0, 0))
        if  input_scale != 0.0:
            img1_resized = (img1_resized / input_scale) + input_zero_point
        img1_resized = img1_resized.astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], img1_resized)
        interpreter.invoke()
        outputs = []
        for o in range(len(output_details)):
            output_scale, output_zero_point = output_details[o].get('quantization', (0.0, 0))
            output = interpreter.get_tensor(output_details[o]['index'])
            if  output_scale != 0.0:
                # output = output_scale * (output.astype(output_details[o]['dtype']) - output_zero_point)
                output = output_scale * (output.astype(np.float32) - output_zero_point)
            # output = output.transpose((0,3,1,2))
            outputs.append(output)
        image1_out = outputs[0]

        img2 = cv2.imread(args.image2)
        if img2.shape != (h, w, 3):
            img2 = cv2.resize(img2, (h, w)).clip(0, 255)
        if not args.bgr:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2_resized = img2.astype(np.float32)
        img2_resized = np.expand_dims(img2_resized, axis=0)
        input_scale, input_zero_point = input_details[0].get('quantization', (0.0, 0))
        if  input_scale != 0.0:
            img2_resized = (img2_resized / input_scale) + input_zero_point
        img2_resized = img2_resized.astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], img2_resized)
        interpreter.invoke()
        outputs = []
        for o in range(len(output_details)):
            output_scale, output_zero_point = output_details[o].get('quantization', (0.0, 0))
            output = interpreter.get_tensor(output_details[o]['index'])
            if  output_scale != 0.0:
                # output = output_scale * (output.astype(output_details[o]['dtype']) - output_zero_point)
                output = output_scale * (output.astype(np.float32) - output_zero_point)
            # output = output.transpose((0,3,1,2))
            outputs.append(output)
        image2_out = outputs[0]

    print("image similiarity = {:.3f}".format(cosine_distance(image2_out,image1_out)))
