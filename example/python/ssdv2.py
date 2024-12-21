import argparse
import numpy as np
import cv2
import vbx.postprocess.ssd as ssd
import vbx.postprocess.dataset as dataset
import vbx.sim
import os
import math
import sys

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import onnx_infer, onnx_input_shape
from vbx.generate.utils import load_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
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

    if args.model.endswith('.vnnx'):
        with open(args.model, 'rb') as mf:
            model = vbx.sim.Model(mf.read())
        h, w = model.input_shape[0][-2], model.input_shape[0][-1]

        iimg = cv2.imread(args.image)
        resize_shape = (w, h)
        iimg = cv2.resize(iimg, resize_shape, interpolation=cv2.INTER_LINEAR)
        iimg = iimg.astype(np.float32)
        if not args.bgr:
            iimg = cv2.cvtColor(iimg, cv2.COLOR_BGR2RGB)
        iimg = (iimg - args.mean) / args.scale
        iimg = iimg.astype(np.float32)
        img_scaled = (iimg / model.input_scale_factor[0]) + model.input_zeropoint[0]
        flattened = img_scaled.swapaxes(1, 2).swapaxes(0, 1).flatten().astype(model.input_dtypes[0])
        outputs = model.run([flattened])
        for o, output in enumerate(outputs):
            outputs[o] = model.output_scale_factor[o] * (outputs[o].astype(np.float32) - model.output_zeropoint[o])
            outputs[o] = outputs[o].reshape(model.output_shape[o])
    elif args.model.endswith('.xml'):
        weights=args.model.replace('.xml', '.bin')
        input_shape = openvino_input_shape(args.model, weights)[0]
        input_array = load_input(args.image, 1., input_shape, (not args.bgr))
        h, w = input_shape[-2], input_shape[-1]
        outputs = openvino_infer(args.model, input_array)
    elif args.model.endswith('.onnx'):
        input_shape = onnx_input_shape(args.model)[0]
        h,w = input_shape[-2], input_shape[-1]
        iimg = cv2.imread(args.image)
        resize_shape = (w, h)
        iimg = cv2.resize(iimg, resize_shape, interpolation=cv2.INTER_LINEAR)
        if not args.bgr:
            iimg = cv2.cvtColor(iimg, cv2.COLOR_BGR2RGB)
        iimg = iimg.astype(np.float32)
        iimg = (iimg - args.mean) / args.scale
        iimg = np.expand_dims(iimg, axis=0)
        iimg = iimg.transpose((0,3,1,2))
        iimg = iimg.astype(np.float32)
        outputs = onnx_infer(args.model, iimg)

    elif args.model.endswith('.tflite'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = tuple(input_details[0]["shape"])
        h, w = input_shape[-3], input_shape[-2]
        if img.shape != input_shape:
            img_resized = cv2.resize(img, (w, h))
        else:
            img_resized = img
        if not args.bgr:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized = img_resized.astype(np.float32)
        img_resized = (img_resized - args.mean) / args.scale
        img_resized = np.expand_dims(img_resized, axis=0)
        img_resized = img_resized.astype(np.float32)
        input_scale, input_zero_point = input_details[0].get('quantization', (0.0, 0))
        if  input_scale != 0.0:
            img_resized = (img_resized / input_scale) + input_zero_point
        img_resized = img_resized.astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], img_resized)
        interpreter.invoke()
        outputs = []
        for o in range(len(output_details)):
            output_scale, output_zero_point = output_details[o].get('quantization', (0.0, 0))
            output = interpreter.get_tensor(output_details[o]['index'])
            if  output_scale != 0.0:
                # output = output_scale * (output.astype(output_details[o]['dtype']) - output_zero_point)
                output = output_scale * (output.astype(np.float32) - output_zero_point)
            output = output.transpose((0,3,1,2))
            outputs.append(output)

    if '.vnnx' in args.model or '.tflite' in args.model:
        #shuffle outputs
        _outputs = outputs.copy()
        if args.modification is not None:
            modification = args.modification
        elif args.torch:
            modification = [9,4,1,7,5,3,0,2,8,11,10,6] 
        else:
            modification = [5,11,0,8,1,3,9,2,6,10,7,4]

        for o in range(len(outputs)):
            outputs[o] = _outputs[modification.index(o)]

    # scaling occurs in _infer methods
    output_scale_factor = len(outputs) * [1.0]
    if args.torch:
        predictions = ssd.ssd_torch_predictions(outputs, output_scale_factor, confidence_threshold=0.5, nms_threshold=0.4, top_k=1)
    else:
        predictions = ssd.ssdv2_predictions(outputs, output_scale_factor, confidence_threshold=0.5, nms_threshold=0.4, top_k=1)
    
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    output_scale_x = 1024. / w
    output_scale_y = 1024. / h

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
