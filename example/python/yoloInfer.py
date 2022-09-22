import argparse
import numpy as np
import cv2
import vbx.postprocess.yolo as yolo
import vbx.postprocess.dataset as dataset
import os
import json

from vbx.generate.openvino_infer import openvino_infer, get_model_input_shape as get_xml_input_shape
from vbx.generate.onnx_infer import onnx_infer, load_input
from vbx.generate.onnx_helper import get_model_input_shape as get_onnx_input_shape
from vbx.generate.utils import pad_input
import vbx.sim


def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_dims[0], model.output_dims


def vnnx_infer(vnxx_model, modelInput):
    with open(vnxx_model, "rb") as mf:
        model = vbx.sim.Model(mf.read())
    flattened = (modelInput.flatten()).astype('uint8')
    outputList = model.run([flattened])
    outputs = [out.astype('float32') * scale for out,scale in zip(outputList, model.output_scale_factor)]

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs

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
    parser.add_argument('--io')
    args = parser.parse_args()
    
    if not args.json:
        args.json = os.path.splitext(args.model)[0]+'.json'
    with open(args.json) as f:
        ioCfg = json.load(f)
    colors = dataset.coco_colors
    if args.labels:
        with open(args.labels) as f:
            classLabel = f.readlines()
            classLabel = [x.strip() for x in classLabel]
        if 'voc' in args.labels:
            colors = dataset.voc_colors
    else:
        classLabel = []
        for n in range(ioCfg[-1]['classes']):
            classLabel.append(str(n))
            
    img = cv2.imread(args.image)
    imgDims = np.array(img.shape[:2])
    inputDims = np.array([ioCfg[0]['height'], ioCfg[0]['width']])
    
    # model
    modelOutput = {}
    if args.padding:
        input_array = pad_input(args.image, inputDims)

    if args.model.endswith('.vnnx'):
        if not(args.padding):
            input_shape, _ = get_vnnx_io_shapes(args.model)
            input_array = load_input(args.image, 1., input_shape)  
        outputs = vnnx_infer(args.model, input_array)

    elif args.model.endswith('.xml'):
        if not(args.padding):
            weights=args.model.replace('.xml', '.bin')
            input_shape = get_xml_input_shape(args.model, weights)
            input_array = load_input(args.image, 1., input_shape)
        outputs = openvino_infer(args.model, input_array)

    elif args.model.endswith('.onnx'):
        if not(args.padding):
            input_shape = get_onnx_input_shape(args.model)
            input_array = load_input(args.image, 1., input_shape)       
        outputs = onnx_infer(args.model, input_array)

    for output in outputs:
        for layer in ioCfg[1:]:
            shape = (1,layer['c'],layer['h'],layer['w'])
            if output.size == np.prod(shape): # match output by size
                modelOutput[layer['outputName']] = output.reshape(shape) 
    
    # post-processing
    params = {}
    blobs = {}
    for layer in ioCfg:
        if layer['type'] in ['region','yolo']:
            n = layer['layerNum']
            label = layer['outputName']
            ioScale = int(ioCfg[0]['width']/layer['w'])   # input width / output width
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
    if args.version:
        version = args.version
    elif ioCfg[-1]['type'] == 'region':
        version = 2
    else:
        version = 3
    predictions = yolo.yolo_post_process(blobs, params, inputDims[0], inputDims[1], args.threshold, args.iou, version, True, True)
    for p in predictions:
        p['class_label'] = classLabel[p['class_id']]
        if args.padding:
            p['xmin'] = int(round((p['xmin']-padLeft)/resizeRatio))
            p['xmax'] = int(round((p['xmax']-padLeft)/resizeRatio))
            p['ymin'] = int(round((p['ymin']-padTop)/resizeRatio))
            p['ymax'] = int(round((p['ymax']-padTop)/resizeRatio))
        else:
            p['xmin'] = int(round((p['xmin'])/(inputDims[1]/imgDims[1])))
            p['xmax'] = int(round((p['xmax'])/(inputDims[1]/imgDims[1])))
            p['ymin'] = int(round((p['ymin'])/(inputDims[0]/imgDims[0])))
            p['ymax'] = int(round((p['ymax'])/(inputDims[0]/imgDims[0])))
    
    # output image
    imgOut = np.copy(img)
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
    
