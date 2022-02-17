import vbx.sim
import argparse
import os
import numpy as np
import cv2
import json
import  vbx.postprocess.retinaface

from vbx.generate.openvino_infer import openvino_infer, get_model_input_shape as get_xml_input_shape
from vbx.generate.onnx_infer import onnx_infer, load_input
from vbx.generate.onnx_helper import get_model_input_shape as get_onnx_input_shape


def vnnx_infer(vnnx_model, input_array):
    model = vbx.sim.model.Model(open(vnnx_model,"rb").read())

    input_array = input_array.astype(np.uint8)
    outputs = model.run([input_array.flatten()])
    outputs = [o/(1<<16) for o in outputs]

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('--channels', type=int, default=3) 
    parser.add_argument('-t', '--threshold', type=float, default=0.8)
    parser.add_argument('-nms', '--nms-threshold', type=float, default=0.4)

    args = parser.parse_args()
    if '.vnnx' in args.model:
        input_shape = (args.channels, args.height, args.width)
        input_array = load_input(args.image, 1., input_shape)
        outputs = vnnx_infer(args.model, input_array)
    elif '.xml' in args.model:
        weights=args.model.replace('.xml', '.bin')
        input_shape = get_xml_input_shape(args.model, weights)
        input_array = load_input(args.image, 1., input_shape)
        outputs = openvino_infer(args.model, input_array)
    elif '.onnx' in args.model:
        input_shape = get_onnx_input_shape(args.model)
        input_array = load_input(args.image, 1., input_shape)  
        outputs = onnx_infer(args.model, input_array)

    faces = vbx.postprocess.retinaface.retinaface(outputs, args.width, args.height,args.threshold, args.nms_threshold)
    img = cv2.imread(args.image)
    if img.shape != (args.height,args.width,args.channels):
        img = cv2.resize(img,(args.width,args.height))

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
