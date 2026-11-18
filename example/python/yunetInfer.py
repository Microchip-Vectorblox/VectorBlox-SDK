import argparse
import sys

import cv2
import numpy as np

import vbx.sim
import vbx.sim.model_run as mr
import vbx.postprocess.yunet as yunet_post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    # YuNet is trained on raw 0-255 BGR pixels (OpenCV blobFromImage with
    # scalefactor=1.0, mean=(0,0,0)), so no scale/mean here and bgr on by default.
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1.])
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=[0.])
    parser.add_argument('-rgb', '--rgb', action='store_true',
                        help='feed RGB instead of YuNet\'s native BGR')
    parser.add_argument('-t', '--threshold', type=float, default=0.6)
    parser.add_argument('-nms', '--nms-threshold', type=float, default=0.1)
    parser.add_argument('-o', '--output', default='yunet_out.jpg')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    arr, input_height, input_width, channels_last = mr.preprocess_img_to_input_array(
        img, args.model, not args.rgb, args.scale, args.mean)
    outputs, output_shapes = mr.model_run(arr, args.model)
    if channels_last:
        outputs = mr.transpose_outputs(outputs)

    ordered_outputs = yunet_post.order_outputs(outputs)
    faces = yunet_post.yunet(ordered_outputs, input_width, input_height,
                                       args.threshold, args.nms_threshold)

    if img.shape[:2] != (input_height, input_width):
        img = cv2.resize(img, (input_width, input_height))
    for f in faces:
        l, t, r, b = map(int, f['box'])
        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)
        cv2.putText(img, "{:.4f}".format(f['score']), (l, t + 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        for x, y in f['landmarks']:
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 4)
        print("face at {} {} {} {} w/ confidence {:3.4f}".format(l, t, r, b, f['score']))

    print("{} faces found".format(len(faces)))
    cv2.imwrite(args.output, img)


if __name__ == "__main__":
    main()
