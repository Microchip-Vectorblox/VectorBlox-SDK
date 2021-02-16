import argparse
import numpy as np
import cv2
import vbx.postprocess.yolo as yolo
import vbx.postprocess.dataset as dataset
import os
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-j', '--json', default=None)
    parser.add_argument('-l', '--labels', default=None)
    parser.add_argument('-o', '--output', default="output.png")
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
            
    # pre-processing
    img = cv2.imread(args.image)
    imgDims = np.array(img.shape[:2])
    inputDims = np.array([ioCfg[0]['height'], ioCfg[0]['width']])
    resizeRatio = np.min(inputDims/imgDims)
    resizeDims = np.round(imgDims * resizeRatio).astype('int')
    imgResize = cv2.resize(img.astype('float32'), (resizeDims[1],resizeDims[0]), interpolation=cv2.INTER_LINEAR)
    padTop = int((inputDims[0]-resizeDims[0])/2)
    padBottom = inputDims[0]-resizeDims[0] - padTop
    padLeft = int((inputDims[1]-resizeDims[1])/2)
    padRight = inputDims[1]-resizeDims[1] - padLeft
    imgPad = cv2.copyMakeBorder(imgResize/255.0, padTop, padBottom, padLeft, padRight, cv2.BORDER_CONSTANT, value=[0.5,0.5,0.5])
    modelInput = imgPad.swapaxes(0,2).swapaxes(1,2)
    modelInput = np.expand_dims(modelInput, axis=0)
    
    # model
    if args.model.endswith('.onnx'):
        import onnx
        import onnxruntime
        m = onnx.load(args.model)
        session = onnxruntime.InferenceSession(m.SerializeToString())
        inputName = session.get_inputs()[0].name
        outputList = session.run([],{inputName:modelInput})
        modelOutput = {}
        for n,out in enumerate(session.get_outputs()):
            modelOutput[out.name] = outputList[n]
    elif args.model.endswith('.vnnx'):
        import vbx.sim
        with open(args.model, "rb") as mf:
            model = vbx.sim.Model(mf.read())
        input_dtype = model.input_dtypes[0]
        flattened = (255*modelInput.flatten()).astype('uint8')
        outputList = model.run([flattened])
        modelOutput = {}
        for out,scale in zip(outputList, model.output_scale_factor):
            out = out.astype('float32') * scale
            for layer in ioCfg[1:]:
                shape = (1,layer['c'],layer['h'],layer['w'])
                if out.size == np.prod(shape):  # match output by size
                    modelOutput[layer['outputName']] = out.reshape(shape)
        print("bandwidth per run = {}".format(model.get_bandwidth_per_run()))
        print("estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    
    # post-processing
    threshold=0.5
    iou=0.4
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
    if ioCfg[-1]['type'] == 'region':
        version = 2
    else:
        version = 3
    predictions = yolo.yolo_post_process(blobs, params, inputDims[0], inputDims[1], threshold, iou, version, True, True)
    for p in predictions:
        p['class_label'] = classLabel[p['class_id']]
        p['xmin'] = int(round((p['xmin']-padLeft)/resizeRatio))
        p['xmax'] = int(round((p['xmax']-padLeft)/resizeRatio))
        p['ymin'] = int(round((p['ymin']-padTop)/resizeRatio))
        p['ymax'] = int(round((p['ymax']-padTop)/resizeRatio))
    
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
    
