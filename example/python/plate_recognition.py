import numpy as np
import vbx.sim
import argparse
import os
import cv2
import vbx.generate.onnx_infer as onnx_infer
import vbx.generate.onnx_helper as onnx_helper
from vbx.postprocess.ocr import ctc_greedy_decode, lpr_chinese_characters, lpr_characters 
from vbx.generate.onnx_infer import onnx_infer, load_input
import json


def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_dims[0], model.output_dims


def vnnx_infer(vnxx, input_array):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    flattened = input_array.flatten().astype('uint8')
    outputs = model.run([flattened])

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return [o.astype('float32') * sf for o,sf in zip(outputs, model.output_scale_factor)]


def convert_to_fixedpoint(data, dtype):
    # this should go away eventually, and always input uint8 rather than fixedpoint Q1.7
    if dtype == np.int16:
        shift_amt = 13
    elif dtype == np.int8:
        shift_amt = 7
    clip_max, clip_min = (1 << shift_amt)-1, -(1 << shift_amt)
    float_img = flattened.astype(np.float32)/255 * (1 << shift_amt) + 0.5

    fixedpoint_img = np.clip(float_img, clip_min, clip_max).astype(dtype)
    return fixedpoint_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-c', '--characters', choices=['chinese', 'latin'], default='latin')
    parser.add_argument('-o', '--output', default="output.png")
    parser.add_argument('--io')

    args = parser.parse_args()

    if '.onnx' in args.model:
        input_shape =  onnx_helper.get_model_input_shape(args.model)
        scale = None
        if args.io:
            with open(args.io) as f:
                input_scale_factors = json.load(f)['input_scale_factors']
                scale = 1./input_scale_factors[0]
        input_array = onnx_infer.load_input(args.image, scale, input_shape)
        output = onnx_infer.onnx_infer(args.model, input_array)
        if args.io:
            with open(args.io) as f:
                scale_factors = json.load(f)['output_scale_factors']
                output = scale_factors[0] * output
        output = np.squeeze(np.transpose(output, [0,2,3,1]))
    else:
        input_shape, _ = get_vnnx_io_shapes(args.model)
        input_array = load_input(args.image, 1., input_shape)
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
