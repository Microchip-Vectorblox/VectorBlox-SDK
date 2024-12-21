import vbx.sim
import argparse
import os
import numpy as np
import cv2
import json
import  vbx.postprocess.retinaface

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import load_input

def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_shape[0], model.output_shape


def vnnx_infer(vnnx_model, input_array):
    model = vbx.sim.model.Model(open(vnnx_model,"rb").read())

    input_array = input_array.astype(np.float32)
    inputs_resized = (input_array / model.input_scale_factor[0]) + model.input_zeropoint[0]
    flattened = inputs_resized.flatten().astype(model.input_dtypes[0])

    outputs = model.run([flattened])
    for idx, o in enumerate(outputs):
        out_scaled = model.output_scale_factor[idx] * (o.astype(np.float32) - model.output_zeropoint[idx])
        outputs[idx] = out_scaled

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-t', '--threshold', type=float, default=0.8)
    parser.add_argument('-nms', '--nms-threshold', type=float, default=0.4)

    args = parser.parse_args()
    if '.vnnx' in args.model:
        input_shape, _ = get_vnnx_io_shapes(args.model)
        input_array = load_input(args.image, 1., input_shape, (not args.bgr))
        h, w = input_shape[-2], input_shape[-1]
        outputs = vnnx_infer(args.model, input_array)
    elif '.xml' in args.model:
        weights=args.model.replace('.xml', '.bin')
        input_shape = openvino_input_shape(args.model, weights)[0]
        input_array = load_input(args.image, 1., input_shape, (not args.bgr))
        h, w = input_shape[-2], input_shape[-1]
        outputs = openvino_infer(args.model, input_array)
    elif args.model.endswith('.tflite'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = tuple(input_details[0]["shape"][-3:])
        h, w = input_shape[0], input_shape[1]
        img = cv2.imread(args.image)
        if img.shape != input_shape:
            img_resized = cv2.resize(img, (w, h))
        else:
            img_resized = img
        img_resized = np.expand_dims(img_resized, axis=0)
        img_resized = img_resized.astype(np.float32)
        if "quantization" in input_details[0]:
            input_scale, input_zero_point = input_details[0]["quantization"]
            if input_scale != 0:
                img_resized = (img_resized / input_scale) + input_zero_point
                img_resized = img_resized.astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], img_resized)
        interpreter.invoke()
        outputs = []
        for o in range(len(output_details)):
            output = interpreter.get_tensor(output_details[o]['index'])
            if "quantization" in output_details[o]:
                output_scale, output_zero_point = output_details[o]['quantization']
                if output_scale != 0:
                    output = output_scale * (output.astype(np.float32) - output_zero_point)
            output = output.transpose((0,3,1,2))
            outputs.append(output.flatten())

    faces = vbx.postprocess.retinaface.retinaface(outputs, w, h,args.threshold, args.nms_threshold)
    img = cv2.imread(args.image)
    if img.shape != input_shape:
        img = cv2.resize(img,(w,h))

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
