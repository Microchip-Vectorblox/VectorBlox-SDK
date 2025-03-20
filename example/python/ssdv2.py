import argparse
import numpy as np
import cv2
import vbx.postprocess.ssd as ssd
import vbx.postprocess.dataset as dataset
import vbx.sim
import os
import math
import sys
import model_run as mr

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import onnx_infer, onnx_input_shape
from vbx.generate.utils import load_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-nc', '--num-classes', type=int, default=91)
    parser.add_argument('-t', '--torch', action='store_true')
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('--mean', type=float, nargs='+', default=[0.])
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1.])
    parser.add_argument('--output', '-o', default="output.png", help='output image to write labels to')
    parser.add_argument('-m', '--modification', default=None, type=int,  nargs="+")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    img = cv2.imread(args.image)
    scale = args.scale

    arr, input_shape = mr.preprocess_img_to_input_array(img, args.model, args.bgr, scale, args.mean)
    outputs, output_shapes = mr.model_run(arr, args.model)

    channels_last = input_shape[-1] < input_shape[-3]
    h, w = input_shape[-2], input_shape[-1]
    if channels_last:
        h, w = input_shape[-3], input_shape[-2]
        outputs=mr.transpose_outputs(outputs)

    # outputs should be sorted in descending sets of coords and classes w/ size NxN
    reordered_outputs = []
    for box_size in [20,10,5,3,2,1]:
        for class_size in [2*3*4, 2*3*args.num_classes]:
            for output in outputs:
                if output.shape[-1] == box_size and output.shape[-3] == class_size:
                    reordered_outputs.append(output)
    outputs = reordered_outputs

    # scaling occurs in _infer methods
    output_scale_factor = len(outputs) * [1.0]
    if args.torch:
        predictions = ssd.ssd_torch_predictions(outputs, output_scale_factor, confidence_threshold=0.5, nms_threshold=0.4, top_k=1, num_classes=args.num_classes)
    else:
        predictions = ssd.ssdv2_predictions(outputs, output_scale_factor, confidence_threshold=0.5, nms_threshold=0.4, top_k=1)
    
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    output_scale_x = 1024. / w
    output_scale_y = 1024. / h

    classes = [str(_) for _ in range(args.num_classes)]
    if args.num_classes == 91:
        classes = ssd.coco91
    colors = dataset.coco91_colors
    for p in predictions:
        print("{}\t{}\t({}, {}, {}, {})".format(classes[p['class_id']],
                                                int(100*p['confidence']),
                                                int(p['xmin']), int(p['xmax']),
                                                int(p['ymin']), int(p['ymax'])))
        p1 = (int(p['xmin'] * output_scale_x), int(p['ymin'] * output_scale_y))
        p2 = (int(p['xmax'] * output_scale_x), int(p['ymax'] * output_scale_y))
        color = colors[p['class_id']]
        cv2.rectangle(output_img, p1, p2, color, 2)

        p3 = (max(p1[0]-4, 4), max(p1[1]-4, 4))
        class_name = classes[p['class_id']]
        short_name = class_name.split(',')[0]
        cv2.putText(output_img, short_name, p3,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(args.output, output_img)
