import vbx.sim
import cv2
import numpy as np
import argparse
import os
import math
import json
import vbx.sim.model_run as mr

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import onnx_infer, onnx_input_shape
from vbx.generate.utils import load_input
from vbx.generate.utils import existing_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=existing_file)
    parser.add_argument('image', type=existing_file)
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=[0.])
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1.])
    parser.add_argument('-i', '--invert', action='store_true')
    parser.add_argument('--output', '-o', default="output.png", help='output image to write labels to')
    args = parser.parse_args()

    # open image and preprocess
    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    
    img = cv2.imread(args.image)
    arr, input_height, input_width, channels_last = mr.preprocess_img_to_input_array(img, args.model, args.bgr, args.scale, args.mean)

    # run inference
    outputs, _ = mr.model_run(arr, args.model)
   
    output = outputs[0][0]
    if output.shape[-3] < output.shape[-1]:
        output = output.transpose((1,2,0))
    output = (output*(255./np.max(output))).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    input = arr[0]
    if input.shape[-3] < input.shape[-1]:
        input = input.transpose((1,2,0))
    input = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)

    # draw results
    cv2.imwrite("input.png", input)
    cv2.imwrite("output.png", output)
    print("Saved simulation result to ", args.output)
