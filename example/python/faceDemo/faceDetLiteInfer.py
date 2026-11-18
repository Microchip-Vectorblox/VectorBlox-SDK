import argparse
import sys

import numpy as np
import cv2

import vbx.sim.model_run as mr
import vbx.postprocess.face_det_lite
from vbx.generate.utils import get_input_details

np.set_printoptions(threshold=sys.maxsize)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1])
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=[0.])
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-t', '--threshold', type=float, default=0.55)
    parser.add_argument('-nms', '--nms-threshold', type=float, default=0.4,
                        help='IoU threshold for NMS; -1 disables NMS (default=0.4)')

    args = parser.parse_args()
    img = cv2.imread(args.image)

    scale = args.scale[0] if len(args.scale) == 1 else np.asarray(args.scale)
    mean = args.mean[0] if len(args.mean) == 1 else np.asarray(args.mean)
    arr, input_height, input_width, channels_last = mr.preprocess_img_to_input_array(
        img, args.model, args.bgr, scale, mean)
    outputs, _ = mr.model_run(arr, args.model)

    # bring every output to NCHW so channel count identifies it
    if channels_last:
        outputs = mr.transpose_outputs(outputs)

    heatmap = box = landmark = None
    for out in outputs:
        #out = np.asarray(out)
        #if out.ndim == 3:
        #    out = out[np.newaxis, ...]
        channels = out.shape[1]
        if channels == 1:
            heatmap = out
        elif channels == 4:
            box = out
        elif channels == 10:
            landmark = out
    if heatmap is None or box is None or landmark is None:
        raise RuntimeError("could not match heatmap/bbox/landmark outputs; got "
                           "shapes {}".format([np.asarray(o).shape for o in outputs]))

    stride = input_height // heatmap.shape[2]
    faces = vbx.postprocess.face_det_lite.detect(
        heatmap, box, landmark, args.threshold, args.nms_threshold, stride)

    if img.shape[:2] != (input_height, input_width):
        img = cv2.resize(img, (input_width, input_height))

    for f in faces:
        text = "{:.4f}".format(f['score'])
        b = list(map(int, f['box']))

        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cv2.putText(img, text, (b[0], b[1] + 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        for l in f['landmarks']:
            cv2.circle(img, (int(l[0]), int(l[1])), 1, (0, 0, 255), 4)

        print("face found at", *b, 'w/ confidence {:3.4f}'.format(f['score']))
        for l in f['landmarks']:
            print("face feature at", int(l[0]), int(l[1]))
        print()

    print("{} faces found".format(len(faces)))
    name = "test.jpg"
    cv2.imwrite(name, img)


if __name__ == "__main__":
    main()