import numpy as np
import vbx.sim
import argparse
import os
import cv2
from vbx.generate.onnx_infer import onnx_infer, load_input
import vbx.postprocess.lpr as lpr
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
    outputs = [o.astype(np.float32)/(1<<16) for o in outputs]
    output = outputs[0].reshape((37,18))
    
    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return output

    
  



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', default = 'lpr_eu_v3.vnnx')
    parser.add_argument('image', default = '../../test_images/A358CC82.jpg')
    args = parser.parse_args()


    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    img = cv2.imread(args.image)

    if '.vnnx' in args.model:
        input_shape , _ = get_vnnx_io_shapes(args.model)
        input_array = load_input(args.image, 1., input_shape) #automatically resizes img to input dims
        output = vnnx_infer(args.model,input_array)
    elif '.xml' in args.model:
        weights=args.model.replace('.xml', '.bin')
        input_shape = get_xml_input_shape(args.model, weights)
        input_array = load_input(args.image, 1., input_shape)
        output = openvino_infer(args.model, input_array)
    elif '.onnx' in args.model:
        input_shape = get_onnx_input_shape(args.model)
        input_array = load_input(args.image, 1., input_shape)  
        output = onnx_infer(args.model, input_array)
    plateID,conf = lpr.PlateDecodeCStyle(output)
    print("Plate ID: ", plateID,"    Recognition Score: {:3.4f}".format(conf))


if __name__ == "__main__":
    main()