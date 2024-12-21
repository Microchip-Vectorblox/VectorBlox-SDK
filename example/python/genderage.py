import vbx.sim
import argparse
import os
import numpy as np
import cv2
import json

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import load_input

def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_shape[0], model.output_shape


def vnnx_infer(vnnx_model, input_array):
    model = vbx.sim.model.Model(open(vnnx_model,"rb").read())

    input_array = input_array.astype(np.float32)
    inputs_resized = (input_array / model.input_scale_factor[0]) + model.input_zeropoint[0]
    flattened = inputs_resized.flatten().astype(model.input_dtypes[0])

    outputs = model.run([flattened])
    for idx, o in enumerate(outputs):
        out_scaled = model.output_scale_factor[idx] * (o.astype(np.float32) - model.output_zeropoint[idx])
        outputs[idx] = out_scaled

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('--norm', action='store_true')

    args = parser.parse_args()
    if '.vnnx' in args.model:
        input_shape, _ = get_vnnx_io_shapes(args.model)
        input_array = load_input(args.image, 1., input_shape, (not args.bgr), args.norm)
        outputs = vnnx_infer(args.model, input_array)
    elif '.xml' in args.model:
        weights=args.model.replace('.xml', '.bin')
        input_shape = openvino_input_shape(args.model, weights)[0]
        input_array = load_input(args.image, 1., input_shape, (not args.bgr))
        outputs = openvino_infer(args.model, input_array)

    elif args.model.endswith('.tflite'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = tuple(input_details[0]["shape"][-3:])
        h, w = input_shape[0], input_shape[1]
        img = cv2.imread(args.image)
        if img.shape != input_shape:
            img_resized = cv2.resize(img, (w, h))
        else:
            img_resized = img
        img_resized = img_resized.astype(np.float32)
        if not args.bgr:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        if args.norm:
            img_resized /= 255.
        img_resized = np.expand_dims(img_resized, axis=0)
        input_scale, input_zero_point = input_details[0].get('quantization', (0.0, 0))
        if  input_scale != 0.0:
            img_resized = (img_resized / input_scale) + input_zero_point
        img_resized = img_resized.astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], img_resized)
        interpreter.invoke()
        outputs = []
        for o in range(len(output_details)):
            output_scale, output_zero_point = output_details[o].get('quantization', (0.0, 0))
            output = interpreter.get_tensor(output_details[o]['index'])
            if  output_scale != 0.0:
                # output = output_scale * (output.astype(output_details[o]['dtype']) - output_zero_point)
                output = output_scale * (output.astype(np.float32) - output_zero_point)
            # output = output.transpose((0,3,1,2))
            outputs.append(output.flatten())

    if len(outputs) == 1: # concatenated age/gender outputs
        age = outputs[0][2]
        gender = outputs[0][:2]
    else: # seperated age/gender outputs
        if len(outputs[0]) == 1: # age is single value, gender array has length 2
            age = outputs[0][0]
            gender = outputs[1]
        else:
            age = outputs[1][0]
            gender = outputs[0]
    print('Age: {} Gender: {}'.format(int(100*age), 'M' if np.argmax(gender) == 1 else 'F'))


if __name__ == "__main__":
    main()
