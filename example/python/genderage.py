import vbx.sim
import argparse
import os
import numpy as np
import cv2
import json

from vbx.generate.openvino_infer import openvino_infer, get_model_input_shape as get_xml_input_shape
from vbx.generate.onnx_infer import onnx_infer, load_input
from vbx.generate.onnx_helper import get_model_input_shape as get_onnx_input_shape


def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_dims[0], model.output_dims


def vnnx_infer(vnnx_model, input_array):
    model = vbx.sim.model.Model(open(vnnx_model,"rb").read())

    input_array = input_array.astype(np.uint8)
    outputs = model.run([input_array.flatten()])
    outputs = [o/(1<<16) for o in outputs]

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')

    args = parser.parse_args()
    if '.vnnx' in args.model:
        input_shape, _ = get_vnnx_io_shapes(args.model)
        input_array = load_input(args.image, 1., input_shape)
        outputs = vnnx_infer(args.model, input_array)
    elif '.xml' in args.model:
        weights=args.model.replace('.xml', '.bin')
        input_shape = get_xml_input_shape(args.model, weights)
        input_array = load_input(args.image, 1., input_shape)
        outputs = openvino_infer(args.model, input_array)
    elif '.onnx' in args.model:
        input_shape = get_onnx_input_shape(args.model)
        input_array = load_input(args.image, 1., input_shape)  
        input_array = input_array[:,::-1,:,:]
        outputs = onnx_infer(args.model, input_array)

    if len(outputs) == 1: # concatenated age/gender outputs
        age = outputs[0][2]
        gender = outputs[0][:2]
    else: # seperated age/gender outputs
        if len(outputs[0]) == 1: # age is single value, gender array has length 2
            age = outputs[0]
            gender = outputs[1]
        else:
            age = outputs[1]
            gender = outputs[0]
    print('Age: {} Gender: {}'.format(int(100*age), 'M' if np.argmax(gender) == 1 else 'F'))


if __name__ == "__main__":
    main()
