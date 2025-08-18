import vbx.sim
import vbx.postprocess.dataset as dataset
import vbx.postprocess.classifier as classifier
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
   
    # postprocess outputs and draw
    output = outputs[0].flatten()
    sorted_classes = classifier.topk(output)

    if len(output)==1001:
        classes = dataset.imagenet_classes_with_nul
    elif len(output)==1000:
        classes = dataset.imagenet_classes
    elif len(output)==8:
        classes = dataset.emotion_classes
    else:
        classes = None

    # draw results
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    print()
    i = 0
    for cls, score in list(zip(*sorted_classes))[:5]:
        p3 = (4, (i+1)*(32+4))
        if classes is None:
            print("{} {}".format(cls, score))
            cv2.putText(output_img, '{}'.format(cls), p3,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            class_name = classes[cls]
            short_name = class_name.split(',')[0]
            print(cls, short_name, score)
            cv2.putText(output_img, '{} {}'.format(cls, short_name), p3,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        i += 1

    cv2.imwrite(args.output, output_img)
    print("Saved simulation result to ", args.output)
    print()
