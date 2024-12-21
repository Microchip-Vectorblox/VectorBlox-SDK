import argparse
import numpy as np
import cv2
import vbx.postprocess.yolo as yolo
import vbx.postprocess.dataset as dataset
import os
import json

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import pad_input, load_input
import vbx.sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-j', '--json', default=None)
    parser.add_argument('-l', '--labels', default=None)
    parser.add_argument('-o', '--output', default="output.png")
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-i', '--iou', type=float, default=0.4)
    parser.add_argument('-p', '--padding', action='store_true')
    parser.add_argument('-v', '--version', type=int)
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=0.)
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=1.)
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--io')
    args = parser.parse_args()
    
    if not args.json:
        args.json = os.path.splitext(args.model)[0]+'.json'

    ioCfg = None
    if os.path.exists(args.json):
        with open(args.json) as f:
            ioCfg = json.load(f)

    colors = dataset.coco_colors
    if args.labels:
        with open(args.labels) as f:
            classLabel = f.readlines()
            classLabel = [x.strip() for x in classLabel]
        if 'voc' in args.labels:
            colors = dataset.voc_colors
    elif not ioCfg is None:
        classLabel = []
        for n in range(ioCfg[-1]['classes']):
            classLabel.append(str(n))
    else:
        print("Either labels or JSON containing 'classes' must be provided")
        sys.exit(0)
            
    img = cv2.imread(args.image)
    h, w, _ = img.shape
    imgOut = np.copy(img)
    img_height, img_width = img.shape[-3], img.shape[-2]
    
    # model
    if args.model.endswith('.vnnx'):

        with open(args.model, 'rb') as mf:
            model = vbx.sim.Model(mf.read())
        
        input_height, input_width = model.input_shape[0][-2], model.input_shape[0][-1]
        if img.shape != (input_height, input_width, 3):
            img = cv2.resize(img, (input_width, input_height)).clip(0, 255)
        img_resized = img.astype(np.float32)
        if not args.bgr:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        if args.norm:
            img_resized /= 255.
        img_resized = (img_resized - args.mean) / args.scale
        inputs_resized = (img_resized / model.input_scale_factor[0]) + model.input_zeropoint[0]
        flattened = inputs_resized.swapaxes(1, 2).swapaxes(0, 1).flatten().astype(model.input_dtypes[0])

        outputs = model.run([flattened])
        for idx, o in enumerate(outputs):
            out_scaled = model.output_scale_factor[idx] * (o.astype(np.float32) - model.output_zeropoint[idx])
            outputs[idx] = out_scaled.reshape(model.output_shape[idx])

    # TODO needs to be updated
    elif args.model.endswith('.xml'):
        weights=args.model.replace('.xml', '.bin')
        input_shape = openvino_input_shape(args.model, weights)[0]
        input_height, input_width = input_shape[-2], input_shape[-1]
        input_array = load_input(args.image, 1., input_shape, (not args.bgr))
        outputs = openvino_infer(args.model, input_array)

    elif args.model.endswith('.tflite'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_height, input_width = input_details[0]['shape'][1], input_details[0]['shape'][2]
        if img.shape != (input_height, input_width, 3):
            img = cv2.resize(img, (input_width, input_height)).clip(0, 255)
        img_resized = img.astype(np.float32)
        if not args.bgr:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        if args.norm:
            img_resized /= 255.
        img_resized = (img_resized - args.mean) / args.scale
        img_resized = np.expand_dims(img_resized, axis=0)
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
                output = output_scale * (output.astype(np.float32) - output_zero_point)
            if len(output.shape) == 4:
                output = output.transpose((0,3,1,2))
            outputs.append(output)

    elif args.model.endswith('.json'):
        with open(args.model) as f:
            arr = json.load(f)
            outputs = []
            for o in range(len(arr)):
                output = np.asarray(arr[o]['data'], dtype=np.uint8)
                output = output.reshape(arr[o]['dims'][1:]).astype(np.int8)
                output = arr[o]['scale'] * (output - arr[o]['zeropoint'])
                outputs.append(output.transpose((0,3,1,2)))

    if args.version:
        version = args.version
    elif ioCfg[-1]['type'] == 'region':
        version = 2
    else:
        version = 3

    if version == 8:
        if len(outputs) == 1:
            predictions = yolo.nms_post_process(outputs[0], args.threshold, args.iou, input_width, input_height)
        else:
            post = yolo.ultralytics_post_process(outputs, args.threshold)
            predictions = yolo.nms_post_process(post, args.threshold, args.iou, input_width, input_height)
    else:
        modelOutput = {}
        for output in outputs:
            for l,layer in enumerate(ioCfg):
                if layer['type'] in ['region','yolo']:
                    label = '{}'.format(l)
                    shape = (1,layer['c'],layer['h'],layer['w'])
                    if output.size == np.prod(shape): # match output by size
                        modelOutput[label] = output.reshape(shape) 
        
        # post-processing
        params = {}
        blobs = {}
        for l,layer in enumerate(ioCfg):
            if layer['type'] in ['region','yolo']:
                label = '{}'.format(l)
                ioScale = int(input_width/layer['w'])   # input width / output width
                if layer['type'] == 'region':
                    coords = layer['coords']
                    anchors = [ioScale*float(i) for i in layer['anchors'].split(',')]
                    num = layer['num']
                else:
                    coords = 4
                    allAnchors = [float(i) for i in layer['anchors'].split(',')]
                    mask = [int(i) for i in layer['mask'].split(',')]
                    anchors = []
                    for i in mask:
                        anchors += allAnchors[i*2:i*2+2]
                    num = len(mask)
                    
                params[label] = {
                    'side': layer['w'],
                    'anchors': anchors,
                    'classes': layer['classes'],
                    'num': num,
                    'coords': coords,
                    'shape': [1, layer['c'], layer['h'], layer['w']],
                    }
                blobs[label] = modelOutput[label].squeeze()
        predictions = yolo.yolo_post_process(blobs, params, input_height, input_width, args.threshold, args.iou, version, True, True)

    for p in predictions:
        p['class_label'] = classLabel[p['class_id']]
        if args.padding:
            p['xmin'] = int(round((p['xmin']-padLeft)/resizeRatio))
            p['xmax'] = int(round((p['xmax']-padLeft)/resizeRatio))
            p['ymin'] = int(round((p['ymin']-padTop)/resizeRatio))
            p['ymax'] = int(round((p['ymax']-padTop)/resizeRatio))
        else:
            p['xmin'] = int(round((p['xmin'])/(input_width/img_width)))
            p['xmax'] = int(round((p['xmax'])/(input_width/img_width)))
            p['ymin'] = int(round((p['ymin'])/(input_height/img_height)))
            p['ymax'] = int(round((p['ymax'])/(input_height/img_height)))
    
    print("class\tscore\t(x_min,x_max,y_min,y_max)")
    # output image imgOut
    for p in predictions:
        print("{}\t{}%\t({}, {}, {}, {})".format(p['class_label'], int(round(100*p['confidence'])),
                                                p['xmin'], p['xmax'], p['ymin'], p['ymax']))
        color = colors[p['class_id']]
        p1 = (p['xmin'], p['ymin'])
        p2 = (p['xmax'], p['ymax'])
        cv2.rectangle(imgOut, p1, p2, color, 2)
    
        pText = (max(p['xmin']-4, 4), max(p['ymin']-4, 4))
        class_name = p['class_label']
        short_name = class_name.split(',')[0]
        cv2.putText(imgOut, short_name, pText, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imwrite(args.output, imgOut)
    print("Saved simulation result to", args.output)
