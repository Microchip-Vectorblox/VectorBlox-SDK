import numpy as np
import vbx.sim
import argparse
import os
import cv2
import vbx.postprocess.lpr as lpr
import json
from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import load_input
import vbx.sim.model_run as mr
  
  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', default = 'lpr_eu_v3.vnnx')
    parser.add_argument('image', default = '../../test_images/A358CC82.jpg')
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1.])
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=[0.])
    args = parser.parse_args()


    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    img = cv2.imread(args.image)
    scale=args.scale
    
    arr, _, _, _ = mr.preprocess_img_to_input_array(img, args.model, args.bgr, scale, args.mean)
    outputs, output_shapes = mr.model_run(arr, args.model)
    output=outputs[0].squeeze()
    
    plateID,conf = lpr.PlateDecodeCStyle(output)
    print("Plate ID: ", plateID,"    Recognition Score: {:3.4f}".format(conf))


if __name__ == "__main__":
    main()
