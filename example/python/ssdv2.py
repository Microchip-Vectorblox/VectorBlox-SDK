import argparse
import numpy as np
import cv2
import vbx.postprocess.ssd as ssd
import vbx.postprocess.dataset as dataset
import vbx.sim
import os
import math

from vbx.generate.openvino_infer import openvino_infer, get_model_input_shape as get_xml_input_shape
from vbx.generate.onnx_infer import onnx_infer, load_input
from vbx.generate.onnx_helper import get_model_input_shape as get_onnx_input_shape


def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_dims[0], model.output_dims


def vnnx_infer(vnnx_model, input_array):
    with open(vnnx_model, "rb") as mf:
        model = vbx.sim.Model(mf.read())
    
    flattened = input_array.flatten().astype('uint8')
    outputs = model.run([flattened])

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return [o.astype('float32') * sf for o,sf in zip(outputs, model.output_scale_factor)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('--output', '-o', default="output.png", help='output image to write labels to')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    img = cv2.imread(args.image)

    if args.model.endswith('.vnnx'):
        input_shape, _ = get_vnnx_io_shapes(args.model)
        input_array = load_input(args.image, 1., input_shape)
        outputs = vnnx_infer(args.model, input_array)
    elif args.model.endswith('.xml'):
        weights=args.model.replace('.xml', '.bin')
        input_shape = get_xml_input_shape(args.model, weights)
        input_array = load_input(args.image, 1., input_shape)
        outputs = openvino_infer(args.model, input_array)
    elif args.model.endswith('onnx'):
        input_shape = get_onnx_input_shape(args.model)
        input_array = load_input(args.image, 1., input_shape)  
        outputs = onnx_infer(args.model, input_array)
        
    # scaling occurs in _infer methods
    output_scale_factor = len(outputs) * [1.0]
    predictions = ssd.ssdv2_predictions(outputs, output_scale_factor, confidence_threshold=0.5, nms_threshold=0.4, top_k=1)
    
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    output_scale_x = 1024. / input_shape[2]
    output_scale_y = 1024. / input_shape[1]

    classes = ssd.coco91
    colors = dataset.coco_colors
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
