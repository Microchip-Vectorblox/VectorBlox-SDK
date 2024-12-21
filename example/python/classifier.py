import vbx.sim
import vbx.postprocess.dataset as dataset
import vbx.postprocess.classifier as classifier
import cv2
import numpy as np
import argparse
import os
import math
import json

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import onnx_infer, onnx_input_shape
from vbx.generate.utils import load_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=[0.])
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1.])
    parser.add_argument('--output', '-o', default="output.png", help='output image to write labels to')
    parser.add_argument('-g', '--grayscale', action='store_true')
    parser.add_argument('--norm', action='store_true')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)

    if args.model.endswith('.vnnx'):

        with open(args.model, 'rb') as mf:
            model = vbx.sim.Model(mf.read())

        input_size = model.input_shape[0][-1]
        def run_image(img):
            img = img.astype(np.float32)
            if not args.bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if args.norm:
                img /= 255.
            img = (img - args.mean) / args.scale
            img_scaled = (img / model.input_scale_factor[0]) + model.input_zeropoint[0]
            if not args.grayscale:
                flattened = img_scaled.swapaxes(1, 2).swapaxes(0, 1).flatten().astype(model.input_dtypes[0])
            else:
                flattened = img_scaled.flatten().astype(model.input_dtypes[0])

            outputs = model.run([flattened])
            for o, output in enumerate(outputs):
                outputs[o] = model.output_scale_factor[o] * (outputs[o].astype(np.float32) - model.output_zeropoint[o])
            return outputs[0]

    # TODO needs to be updated          
    elif args.model.endswith('.xml'):
        weights=args.model.replace('.xml', '.bin')
        input_shape = openvino_input_shape(args.model, weights)[0]
        input_size = input_shape[-1]
        input_array = load_input(args.image, 1. / args.scale[0], input_shape, (not args.bgr))
        output = openvino_infer(args.model, input_array)[0].flatten()

    elif args.model.endswith('.onnx'):
        input_shape = onnx_input_shape(args.model)[0]
        input_size = input_shape[-1]
        input_array = load_input(args.image, 1. / args.scale[0], input_shape, (not args.bgr))
        output = onnx_infer(args.model, input_array)[0].flatten()

    elif args.model.endswith('.tflite'):
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_size = input_details[0]['shape'][1]

        def run_image(img):
            img = img.astype(np.float32)
            if not args.bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if args.norm:
                img /= 255.
            img = (img - args.mean) / args.scale
            img = np.expand_dims(img, axis=0)
            input_scale, input_zero_point = input_details[0].get('quantization', (0.0, 0))
            if  input_scale != 0.0:
                img = (img / input_scale) + input_zero_point
            if args.grayscale:
                img = np.expand_dims(img, axis=3)
            img = img.astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()

            outputs = []
            for o in range(len(output_details)):
                output_scale, output_zero_point = output_details[o].get('quantization', (0.0, 0))
                output = interpreter.get_tensor(output_details[o]['index']).squeeze()
                if  output_scale != 0.0:
                    output = output_scale * (output.astype(np.float32) - output_zero_point)
                outputs.append(output)
            return outputs[0]

    elif args.model.endswith('.json'):
        input_size = 224
        def run_image(img):
            with open(args.model) as f:
                arr = json.load(f)
                outputs = []
                for o in range(len(arr)):
                    output = np.asarray(arr[o]['data'], dtype=np.uint8)
                    output = output.reshape(arr[o]['dims'][1:]).astype(np.int8)
                    output = arr[o]['scale'] * (output - arr[o]['zeropoint'])
                    outputs.append(output.transpose((0,3,1,2)).squeeze())
            return outputs[0]
    
    img = cv2.imread(args.image)
    if args.grayscale:
        img = cv2.imread(args.image, 0)
    if img.shape != (input_size, input_size, 3):
        if input_size == 224 or input_size == 227: # resize 256, then center crop
            resize_shape = (256, 256)
            img_resized = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR).clip(0, 255)
            h_offset = max(0, resize_shape[1]//2 - input_size//2)
            w_offset = max(0, resize_shape[0]//2 - input_size//2)
            img_resized = img_resized[h_offset:h_offset+input_size, w_offset:w_offset+input_size]
        elif input_size == 299:
            resize_shape = (320, 320)
            img_resized = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR).clip(0, 255)
            h_offset = max(0, resize_shape[1]//2 - input_size//2)
            w_offset = max(0, resize_shape[0]//2 - input_size//2)
            img_resized = img_resized[h_offset:h_offset+input_size, w_offset:w_offset+input_size]
        else: #TODO match openvino's central fraction if needed
            resize_shape = (input_size, input_size)
            img_resized = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR).clip(0, 255)
    else:
        img_resized = img

    if '.onnx' in args.model or '.xml' in args.model:
        pass
    else:
        output = run_image(img_resized)
    sorted_classes = classifier.topk(output)

    if len(output)==1001:
        classes = dataset.imagenet_classes_with_nul
    elif len(output)==1000:
        classes = dataset.imagenet_classes
    else: # mnist
        classes = None

    i = 0
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
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
