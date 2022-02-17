import numpy as np
import vbx.sim
import argparse
import os
import cv2
import vbx.generate.onnx_infer as onnx_infer
import vbx.generate.onnx_helper as onnx_helper
from vbx.postprocess.ocr import ctc_greedy_decode, lpr_chinese_characters, lpr_characters 
import json


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
        with open(args.model, "rb") as mf:
            model = vbx.sim.Model(mf.read())
        input_width = 94
        if args.characters == 'latin':
            input_width = 112
        input_height = 24
        input_dtype = model.input_dtypes[0]
        if not os.path.isfile(args.image):
            print('Error: {} could not be read'.format(args.image))
            os._exit(1)
        img = cv2.imread(args.image)
        if img.shape != (input_width, input_height, 3):
            img_resized = cv2.resize(img, (input_width, input_height)).clip(0, 255)
        else:
            img_resized = img
        flattened = img_resized.swapaxes(1, 2).swapaxes(0, 1).flatten()
        if input_dtype != np.uint8:
            flattened = convert_to_fixedpoint(flattened, input_dtype)

        outputs = model.run([flattened])
        output = outputs[0].astype(np.float32)*model.output_scale_factor[0]
        if args.characters == 'latin':
            output = output.reshape((1,37,1,106))
        else:   
            output = output.reshape((1,71,1,88))
        output = np.squeeze(np.transpose(output, [0,2,3,1]))

        print("bandwidth per run = {}".format(model.get_bandwidth_per_run()))
        print("estimated {} ms at 133MHz".format(model.get_estimated_runtime(133*1E6)*1000))

    characters = None
    if args.characters == 'latin':
        characters = lpr_characters
    elif args.characters == 'chinese':
        characters = lpr_chinese_characters

    print(ctc_greedy_decode(output, merge_repeated=True, characters=characters))
