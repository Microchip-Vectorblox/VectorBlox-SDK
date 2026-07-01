import cv2
import numpy as np
import math
import os
import onnx
import onnxruntime as rt
import vbx.postprocess.dataset as dataset
import argparse
import tensorflow as tf
import vbx.sim.model_run as mr
import vbx.generate.utils as utils
from vbx.postprocess.linemod import *


def nms(boxes, scores, max_detections=100, iou_threshold=0.5):

    indices = np.where(scores > 0.01)
    filtered_boxes = boxes[indices]
    filtered_scores = scores[indices]

    nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=iou_threshold)
    indices = [indices[0][_] for _ in nms_indices.numpy()]
    return np.asarray(indices, dtype='int64'),



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-t', '--task', choices=['occlusion'] + linemod_classes, default='occlusion')
    parser.add_argument('--mean', type=float, nargs='+', default=[0.])
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1.])
    parser.add_argument('-b', '--bgr', action='store_true')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    arr, input_height, input_width, channels_last = mr.preprocess_img_to_input_array(img, args.model, args.bgr, args.scale, args.mean)
    outputs, output_shapes = mr.model_run(arr, args.model)

    if channels_last:
        outputs = mr.transpose_outputs(outputs)
    
    classes = []
    regressions = []
    rotations = []
    translation2 = []
    translation1 = []

    if 'onnx' in args.model or '32.tflite' in args.model:
        class_indices = [_ for _ in range(10)]
    elif '.tflite' in args.model or '.vnnx' in args.model:
        class_indices = [3,8,13,18,23]

    num_classes = 1
    if args.task == 'occlusion':
        num_classes = 8
    for o,output in enumerate(outputs):
        if output.shape[1] == 9*num_classes and o in class_indices:
            classes.append(np.transpose(output,(0,2,3,1)).reshape((1,-1,num_classes)))
        elif output.shape[1] == 9*4*num_classes:
            regressions.append(np.transpose(output,(0,2,3,1)).reshape((1,-1,4)))
        elif output.shape[1] == 27:
            rotations.append(np.transpose(output,(0,2,3,1)).reshape((1,-1,3)))
        elif output.shape[1] == 18:
            translation2.append(np.transpose(output,(0,2,3,1)).reshape((1,-1,2)))
        elif output.shape[1] == 9:
            translation1.append(np.transpose(output,(0,2,3,1)).reshape((1,-1,1)))

    classes = sorted(classes, key=lambda x: -1*x.shape[-2])
    regressions = sorted(regressions, key=lambda x: -1*x.shape[-2])
    rotations = sorted(rotations, key=lambda x: -1*x.shape[-2])
    translation1 = sorted(translation1, key=lambda x: -1*x.shape[-2])
    translation2 = sorted(translation2, key=lambda x: -1*x.shape[-2])


    translations_raw = np.concatenate([np.concatenate([a,b], axis=2) for a,b in zip(translation2, translation1)], axis=1)
    regressions = np.concatenate(regressions, axis=1)
    rotations = np.squeeze(np.concatenate(rotations, axis=1))
    classes = np.squeeze(np.concatenate(classes, axis=1))
    classes = 1 / (1 + np.exp(-classes))

    scale = 1.
    camera_matrix = np.array([[572.4114, 0., 325.2611],[0., 573.57043, 242.04899],[0., 0., 1.]])

    camera = np.zeros((6,), dtype = np.float32)
    camera[0] = camera_matrix[0, 0]
    camera[1] = camera_matrix[1, 1]
    camera[2] = camera_matrix[0, 2]
    camera[3] = camera_matrix[1, 2]
    camera[4] = 1000.
    camera[5] = scale

    rt_sess2 = rt.InferenceSession('EfficientPose.trans.onnx')
    outputs2 = rt_sess2.run(None, {'translation_raw_outputs/concat:0': translations_raw, 'input_2': np.expand_dims(camera, axis=0)})
    translations = outputs2[0]

    rt_sess3 = rt.InferenceSession('EfficientPose.anchors.onnx')
    outputs3 = rt_sess3.run(None, {'regression/concat:0': regressions})
    boxes = outputs3[0]
    if len(classes.shape) > 1:
        labels = np.argmax(classes, axis=-1)
        scores = np.max(classes, axis=-1)
    else:
        labels = np.zeros(classes.shape, dtype=np.int32)
        scores = classes

    boxes, scores, labels, rotations, translations = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels), np.squeeze(rotations), np.squeeze(translations)

    nms_indices = nms(boxes, scores)
    scores = scores[nms_indices]
    labels = labels[nms_indices]
    boxes = boxes[nms_indices]
    rotations = rotations[nms_indices]
    translations = translations[nms_indices]

    score_threshold = 0.5
    boxes /= scale
    height, width = img.shape[-3], img.shape[-2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)

    rotations *= math.pi

    # threshold
    indices = np.where(scores[:] > score_threshold)
    scores = scores[indices]
    labels = labels[indices]
    boxes = boxes[indices]
    rotations = rotations[indices]
    translations = translations[indices]

    # sort
    sorted_indices = np.argsort(scores)[::-1]
    scores = scores[sorted_indices]
    labels = labels[sorted_indices]
    boxes = boxes[sorted_indices]
    rotations = rotations[sorted_indices]
    translations = translations[sorted_indices]

    img = cv2.imread(args.image)
    img = cv2.resize(img, (input_width, input_height))

    selection = np.where(scores > score_threshold)[0]
    for i in selection:
        c = dataset.coco91_colors[labels[i]]
        translation_vector = translations[i, :]
        label = args.task
        if args.task == 'occlusion':
            label = occlusion_classes[i]
        points_bbox_2D = project_bbox_2D(label, rotations[i, :], translation_vector, camera_matrix, append_centerpoint = True)
        draw_bbox_8_2D(img, points_bbox_2D, color = c)
    cv2.imwrite('output.png', img)
