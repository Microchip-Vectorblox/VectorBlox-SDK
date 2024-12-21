import numpy as np
import vbx.sim
import argparse
import os
import cv2
from vbx.postprocess.ocr import ctc_greedy_decode, lpr_chinese_characters, lpr_characters 
import json
from vbx.generate.utils import load_input

def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_shape[0], model.output_shape


def vnnx_infer(vnxx, input_array):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-c', '--characters', choices=['chinese', 'latin'], default='latin')
    parser.add_argument('-o', '--output', default="output.png")
    parser.add_argument('--io')

    args = parser.parse_args()

    if '.tflite' in args.model:
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
        img_resized = np.expand_dims(img_resized, axis=0)
        img_resized = img_resized.astype(np.float32)
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
            outputs.append(output)
            # outputs.append(output.flatten())
        output = np.squeeze(outputs[0])
        
    else:
        input_shape, _ = get_vnnx_io_shapes(args.model)
        input_array = load_input(args.image, 1., input_shape, (not args.bgr))
        output = vnnx_infer(args.model, input_array)[0]
        if args.characters == 'latin':
            output = output.reshape((1,37,1,106))
        else:   
            output = output.reshape((1,71,1,88))
        output = np.squeeze(np.transpose(output, [0,2,3,1]))

    characters = None
    if args.characters == 'latin':
        characters = lpr_characters
    elif args.characters == 'chinese':
        characters = lpr_chinese_characters

    print(ctc_greedy_decode(output, merge_repeated=True, characters=characters))
