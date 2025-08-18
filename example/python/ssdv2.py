import argparse
import numpy as np
import cv2
import vbx.postprocess.ssd as ssd
import vbx.postprocess.dataset as dataset
import vbx.sim
import os
import math
import sys
import vbx.sim.model_run as mr

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import onnx_infer, onnx_input_shape
from vbx.generate.utils import load_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-nc', '--num-classes', type=int, default=91)
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-i', '--iou', type=float, default=0.4)
    parser.add_argument('--torch', action='store_true')
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

    arr, input_height, input_width, channels_last = mr.preprocess_img_to_input_array(img, args.model, args.bgr, scale, args.mean)
    outputs, output_shapes = mr.model_run(arr, args.model)

    if channels_last:
        outputs = mr.transpose_outputs(outputs)

    # outputs should be sorted in descending sets of classes and coords w/ size NxN
    outputs = sorted(outputs, key=lambda x: (x.shape[-1], x.shape[-3]))
    outputs.reverse()

    predictions = ssd.ssdv2_predictions(outputs, args.threshold, args.iou, top_k=1, size=input_width, torch=args.torch)
    
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    output_scale_x = 1024. / input_width
    output_scale_y = 1024. / input_height

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
