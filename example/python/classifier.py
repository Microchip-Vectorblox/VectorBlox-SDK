import vbx.sim
import vbx.postprocess.dataset as dataset
import vbx.postprocess.classifier as classifier
import cv2
import numpy as np
import argparse
import os
import math

from vbx.generate.openvino_infer import openvino_infer, get_model_input_shape as get_xml_input_shape
from vbx.generate.onnx_infer import onnx_infer, load_input
from vbx.generate.onnx_helper import get_model_input_shape as get_onnx_input_shape


def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_dims[0], model.output_dims


def vnnx_infer(vnxx, input_array):
    with open(vnxx, 'rb') as mf:
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
        output = vnnx_infer(args.model, input_array)[0]
    elif args.model.endswith('.xml'):
        weights=args.model.replace('.xml', '.bin')
        input_shape = get_xml_input_shape(args.model, weights)
        input_array = load_input(args.image, 1., input_shape)
        output = openvino_infer(args.model, input_array)[0]
    elif args.model.endswith('.onnx'):
        input_shape = get_onnx_input_shape(args.model)
        input_array = load_input(args.image, 1., input_shape)  
        output = onnx_infer(args.model, input_array)[0]

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
