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
import onnxruntime as rt

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import onnx_infer, onnx_input_shape
from vbx.generate.utils import load_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-nc', '--num-classes', type=int, default=90)
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
    regression = []
    classes = []

    for output in outputs:
        if output.shape[1] == 810:
            output = np.transpose(output,(0,2,3,1)).reshape((1,-1,90))
            classes.append(output)
        else:
            output = np.transpose(output,(0,2,3,1)).reshape((1,-1,4))
            regression.append(output)
    regression = np.concatenate(regression, axis=1)
    classes = np.squeeze(np.concatenate(classes, axis=1))
    classes = 1 / (1 + np.exp(-classes))

    labels = np.argmax(classes, axis=-1)
    scores = np.max(classes, axis=-1)

    # post process boxes
    rt_sess = rt.InferenceSession('EfficientDet.anchors.onnx')
    outputs = rt_sess.run(None, {'regression/concat:0': regression})
    boxes = np.squeeze(outputs[0]) #[xmin,ymin,xmax,ymax]

    # threshold
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)
    scores = scores[indices]
    labels = labels[indices]
    boxes = boxes[indices]

    # sort
    sorted_indices = np.argsort(scores)[::-1]
    scores = scores[sorted_indices]
    labels = labels[sorted_indices]
    boxes = boxes[sorted_indices]

    boxes[:, 0] = np.clip(boxes[:, 0], 0, input_width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, input_height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, input_width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, input_height - 1)

    predictions = []
    for id, conf, box in zip(labels, scores,boxes):
        predictions.append(
                {'class_id': id,
                 'confidence': conf,
                 'xmin': box[0],
                 'ymin': box[1],
                 'xmax': box[2],
                 'ymax': box[3],
                 })
    
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    output_scale_x = 1024. / input_width
    output_scale_y = 1024. / input_height

    classes = [str(_) for _ in range(args.num_classes)]
    if args.num_classes == 91:
        classes = ssd.coco91
    elif args.num_classes == 90:
        classes = ssd.coco91[1:]
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
