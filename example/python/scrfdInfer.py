import vbx.sim
import argparse
import os
import numpy as np
import cv2
import json
import vbx.postprocess.scrfd
import sys

import vbx.sim.model_run as mr

np.set_printoptions(threshold=sys.maxsize)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1.])
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=[0.])
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-t', '--threshold', type=float, default=0.8)
    parser.add_argument('-nms', '--nms-threshold', type=float, default=0.4)
    
    

    args = parser.parse_args()
    img = cv2.imread(args.image)
    scale = args.scale

    arr, input_height, input_width, channels_last = mr.preprocess_img_to_input_array(img, args.model, args.bgr, scale, args.mean)
    outputs, _ = mr.model_run(arr, args.model)

    h, w = input_height, input_width
    
    if channels_last:
        outputs=mr.transpose_outputs(outputs)

    #sort the outputs for preprocessing
    # Sort total outputs(9) first Map_Channel by ascending order and Stride by descending order for each Map Channel
    # output_shapes: [1, Map_Channel, Stride, Predetect]
    # This is an example output order (idx can vary depending on outputs, but output_shapes should be the same order):
        #   (idx, output_shapes)
        #   (5, [1, 2, 36, 64])
        #   (4, [1, 2, 18, 32])
        #   (0, [1, 2, 9, 16])
        #   (3, [1, 8, 36, 64])
        #   (1, [1, 8, 18, 32])
        #   (7, [1, 8, 9, 16])
        #   (2, [1, 20, 36, 64])
        #   (8, [1, 20, 18, 32])
        #   (6, [1, 20, 9, 16])       
    ordered_outputs=[]

    if '.vnnx' in args.model or '.tflite' in args.model :
        idx = sorted(enumerate(_.shape for _ in outputs), key = lambda x: (x[1][-3],-x[1][-2]))           
    elif '.onnx' in args.model:
        idx = sorted(enumerate(_.shape for _ in outputs), key = lambda x: (x[1][-1], -x[1][-2]))

    for i,l in enumerate(idx):
        ordered_outputs.append(outputs[idx[i][0]].flatten().squeeze())

    faces = vbx.postprocess.scrfd.scrfd(ordered_outputs, input_width, input_height ,args.threshold, args.nms_threshold)
    if img.shape[:2] != (input_height, input_width):
        img = cv2.resize(img,(input_width, input_height))


    for f in faces:
        text = "{:.4f}".format(f['score'])
        box = list(map(int, f['box']))

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cx = box[0]
        cy = box[1] + 12
        cv2.putText(img, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        for l in f['landmarks']:
            cv2.circle(img, (int(l[0]), int(l[1])), 1, (0, 0, 255), 4)
        print("face found at", *box, 'w/ confidence {:3.4f}'.format(f['score']))
        for l in f['landmarks']:
            print("face feature at",*l)
        print()
    # save image
    print("{} faces found".format(len(faces)))
    name = "test.jpg"
    cv2.imwrite(name, img)

if __name__ == "__main__":
    main()
