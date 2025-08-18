import argparse
import numpy as np
import cv2
import vbx.postprocess.yolo as yolo
import vbx.postprocess.dataset as dataset
import os
import json

from vbx.generate.utils import onnx_infer, onnx_input_shape
from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import pad_input, load_input
from vbx.generate.utils import existing_file
import vbx.sim
import vbx.sim.model_run as mr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=existing_file)
    parser.add_argument('image', type=existing_file)
    parser.add_argument('-j', '--json', default=None, type=existing_file)
    parser.add_argument('-l', '--labels', default=None, type=existing_file)
    parser.add_argument('-o', '--output', default="output.png")
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-i', '--iou', type=float, default=0.4)
    parser.add_argument('-v', '--version', type=str, choices=['2','3','5','7','x','ultra5','8','ultra'], required='true')
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=0.)
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1.])
    parser.add_argument('--prescaled', action='store_true')
    args = parser.parse_args()
    
    # open image and preprocess
    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    img = cv2.imread(args.image)
    img_height, img_width = img.shape[-3], img.shape[-2]
    imgOut = np.copy(img)
    do_nms = True #TODO debug

    arr, input_height, input_width, channels_last = mr.preprocess_img_to_input_array(img, args.model, args.bgr, args.scale, args.mean)
    

    # run inference
    outputs, _ = mr.model_run(arr, args.model)
    # postprocess outputs
    if channels_last: #if model is NHWC outputs need to be converted to
        outputs=mr.transpose_outputs(outputs)

    if args.version in ['8', 'ultra']:
        predictions = []
        if len(outputs) == 1:
            predictions = yolo.nms_post_process(outputs[0], args.threshold, args.iou, input_width, input_height, prescaled=args.prescaled)
        else:
            post = yolo.ultralytics_post_process(outputs, input_height, input_width, args.threshold)
            if(len(post)>1):
                predictions = yolo.nms_post_process(post, args.threshold, args.iou, input_width, input_height, prescaled=args.prescaled)
    else:
        # load config
        config = None
        if not args.json is None and os.path.exists(args.json):
            with open(args.json) as f:
                config = json.load(f)
        else:
            print("WARNING: Anchors in the JSON are required")
            os._exit(1)

        if len(outputs) == 9:
            classes = [] #argmax needs to be done here to find highest class box
            combined_outputs = []

            for grid in set([_.shape[-2] for _ in outputs]):
                for o, output in enumerate(outputs):  
                    if output.shape[-2] == grid:
                        match output.shape[1]:
                            case 80:
                                class_scores = output
                            case 4:
                                coords = output
                            case 1:
                                object_scores = output           
                combined_outputs.append(np.concatenate((coords,object_scores,class_scores), axis = -3))
            outputs = combined_outputs

        # sort outputs by increasing map size (increasing fine-grained) 
        outputs.sort(reverse=False, key=lambda x: x.shape[-1]*x.shape[-2])

        output_anchors = []
        prev_max_anchor = None
        max_anchor = None
        for l,layer in enumerate([_ for _ in config if 'anchors' in _.keys()]):
            prev_max_anchor=max_anchor
            if args.version in ['2','3','5','7','x', 'ultra5']:
                if args.version == '2':
                    # excepting list of "anchors" entries for layers (in inc output map size)
                    ioScale = input_width/outputs[l].shape[-1]
                    anchors = [ioScale*float(i) for i in layer['anchors'].split(',')]
                elif args.version == 'x':
                    anchors = [1 for _ in  range(2*layer['num'])]
                else:
                    # excepting list of "anchors" + "masks" entries for layers (in inc output map size)
                    if "mask" in layer:
                        allAnchors = [float(i) for i in layer['anchors'].split(',')]
                        mask = [int(i) for i in layer['mask'].split(',')]
                        anchors = []
                        for i in mask:
                            anchors += allAnchors[i*2:i*2+2]
                    else: # assume "anchors" already masked
                        anchors = [float(i) for i in layer['anchors'].split(',')]
                max_anchor = max(anchors)
                if prev_max_anchor is not None and prev_max_anchor < max_anchor:
                    print("WARNING: Anchors in the JSON are expected to be ordered in increasing output map size (decreasing anchor size)")
                output_anchors.append(anchors)
        predictions = yolo.yolo_post_process(outputs, output_anchors, input_height, input_width, args.threshold, args.iou, args.version, do_nms, True)

    # draw results
    colors = dataset.coco_colors
    classLabel = None
    if args.labels:
        with open(args.labels) as f:
            classLabel = f.readlines()
            classLabel = [x.strip() for x in classLabel]
        if 'voc' in args.labels:
            colors = dataset.voc_colors

    for p in predictions:
        p['class_label'] = str(p['class_id'])
        if not classLabel is None:
            p['class_label'] = classLabel[p['class_id']]
        p['xmin'] = int(round((p['xmin'])/(input_width/img_width)))
        p['xmax'] = int(round((p['xmax'])/(input_width/img_width)))
        p['ymin'] = int(round((p['ymin'])/(input_height/img_height)))
        p['ymax'] = int(round((p['ymax'])/(input_height/img_height)))
    
    print("class\tscore\t(x_min,x_max,y_min,y_max)")

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
