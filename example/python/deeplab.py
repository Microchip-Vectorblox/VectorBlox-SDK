import vbx.postprocess.dataset as dataset
import vbx.sim
import argparse
import cv2
import os,os.path
import math
import numpy as np

from vbx.generate.openvino_infer import openvino_infer, get_model_input_shape as get_xml_input_shape
from vbx.generate.onnx_infer import onnx_infer, load_input
from vbx.generate.onnx_helper import get_model_input_shape as get_onnx_input_shape


def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_dims[0], model.output_dims


def vnnx_infer(vnnx_model, input_array):
    with open(vnnx_model, "rb") as mf:
        model = vbx.sim.Model(mf.read())

    flattened = input_array.flatten().astype('uint8')
    outputs = model.run([flattened])

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('--dataset',choices=['COCO','VOC'],default='VOC')
    parser.add_argument('-o', '--output', default="output.png")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    if args.model.endswith('.vnnx'):
        input_shape, _ = get_vnnx_io_shapes(args.model)
        input_array = load_input(args.image, 1., input_shape)
        outputs = vnnx_infer(args.model, input_array)
    elif args.model.endswith('.xml'):
        weights=args.model.replace('.xml', '.bin')
        input_shape = get_xml_input_shape(args.model, weights)
        input_array = load_input(args.image, 1., input_shape)
        outputs = openvino_infer(args.model, input_array)
    elif args.model.endswith('.onnx'):
        input_shape = get_onnx_input_shape(args.model)
        input_array = load_input(args.image, 1., input_shape)  
        outputs = onnx_infer(args.model, input_array)

    assert(len(outputs)==1)
    output = outputs[0].reshape((input_shape[2], input_shape[1])).astype('int32')
    #add None Colour at start of array
    colours = np.asarray([[0, 0, 0]] + dataset.voc_colors, dtype="uint8") #TODO; indexing bug with .post/.norm onnx

    #get top category, map that to colour
    mask = colours[output]

    img = cv2.imread(args.image)
    if img.shape != input_shape:
        img = cv2.resize(img, (input_shape[2], input_shape[1])).clip(0, 255)

    output_img=((0.3 * img) + (0.7 * mask)).astype("uint8")
    cv2.imwrite(args.output, output_img)
