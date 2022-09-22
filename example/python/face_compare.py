import vbx.sim
import cv2
import numpy as np
import argparse
import os
import math

from vbx.generate.openvino_infer import openvino_infer, get_model_input_shape as get_xml_input_shape
from vbx.generate.onnx_infer import onnx_infer, load_input
from vbx.generate.onnx_helper import get_model_input_shape as get_onnx_input_shape


def cosine_distance(arr0, arr1):
    return np.sum(arr0*arr1)/(np.sqrt(np.sum(arr0*arr0)) * np.sqrt(np.sum(arr1*arr1)))


def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_dims[0], model.output_dims


def vnnx_infer(vnnx_model, input_array):
    with open(vnnx_model, 'rb') as mf:
        model = vbx.sim.Model(mf.read())

    flattened = input_array.flatten().astype('uint8')
    outputs = model.run([flattened])

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))    
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs[0]*model.output_scale_factor[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image1')
    parser.add_argument('image2')
    args = parser.parse_args()
    
    if not os.path.isfile(args.image1):
        print('Error: {} could not be read'.format(args.image1))
        os._exit(1)
    if not os.path.isfile(args.image2):
        print('Error: {} could not be read'.format(args.image2))
        os._exit(1)

    if args.model.endswith('.vnnx'):
        input_shape, _ = get_vnnx_io_shapes(args.model)

        input_array = load_input(args.image1, 1., input_shape)
        image1_out = vnnx_infer(args.model,input_array)

        input_array = load_input(args.image2, 1., input_shape)
        image2_out = vnnx_infer(args.model,input_array)
        
    elif args.model.endswith('.xml'):
        weights=args.model.replace('.xml', '.bin')
        input_shape = get_xml_input_shape(args.model, weights)

        input_array = load_input(args.image1, 1., input_shape)
        image1_out = openvino_infer(args.model,input_array)[0]

        input_array = load_input(args.image2, 1., input_shape)
        image2_out = openvino_infer(args.model,input_array)[0]

    elif args.model.endswith('.onnx'):
        input_shape = get_onnx_input_shape(args.model)

        input_array = load_input(args.image1, 1., input_shape)
        image1_out = onnx_infer(args.model,input_array)[0]

        input_array = load_input(args.image2, 1., input_shape)
        image2_out = onnx_infer(args.model,input_array)[0]

    print("image similiarity = {:.3f}".format(cosine_distance(image2_out,image1_out)))
