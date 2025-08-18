import vbx.sim
import cv2
import numpy as np
import argparse
import os
import math
import vbx.sim.model_run as mr
from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import load_input

def cosine_distance(arr0, arr1):
    return np.sum(arr0*arr1)/(np.sqrt(np.sum(arr0*arr0)) * np.sqrt(np.sum(arr1*arr1)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image1')
    parser.add_argument('image2')
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1.])
    args = parser.parse_args()
    
    if not os.path.isfile(args.image1):
        print('Error: {} could not be read'.format(args.image1))
        os._exit(1)
    if not os.path.isfile(args.image2):
        print('Error: {} could not be read'.format(args.image2))
        os._exit(1)

    scale = args.scale
    
    img1 = cv2.imread(args.image1)
    arr1, _, _, _ = mr.preprocess_img_to_input_array(img1, args.model, args.bgr, scale)
    outputs1, output_shapes1 = mr.model_run(arr1, args.model)
    image1_out = outputs1[0]

    img2 = cv2.imread(args.image2)
    arr2, _, _, _ = mr.preprocess_img_to_input_array(img2, args.model, args.bgr, scale)
    outputs2, output_shapes2 = mr.model_run(arr2, args.model)
    image2_out = outputs2[0]
    


    print("image similiarity = {:.3f}".format(cosine_distance(image2_out,image1_out)))
