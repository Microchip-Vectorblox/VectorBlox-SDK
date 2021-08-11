import argparse
import json
import openvino.inference_engine as ie
import numpy as np
import cv2
import os
from vbx.postprocess import classifier, yolo,dataset
from .onnx_infer import load_input

np.set_printoptions(suppress=True, precision=4, linewidth=120)
np.random.seed(1337)


def get_model_input_shape(xml, weights):
    core = ie.IECore()
    net = core.read_network(model=xml, weights=weights)
    assert(len(net.input_info) == 1)

    i0 = [k for k in net.input_info.keys()][0]

    exec_net = core.load_network(network=net, device_name="CPU")

    return exec_net.requests[0].input_blobs[i0].buffer.shape


def openvino_activations(xml, weights, input_array):
    core = ie.IECore()
    net = core.read_network(model=xml, weights=weights)
    assert(len(net.input_info) == 1)

    i0 = [k for k in net.input_info.keys()][0]
    o0 = [k for k in net.outputs.keys()][0]

    layers = [layer for layer in net.layers]
    for layer in layers:
        if net.layers[layer].type != "Const":
            net.add_outputs(layer)

    exec_net = core.load_network(network=net, device_name="CPU")
    exec_net.requests[0].input_blobs[i0].buffer[:] = input_array
    exec_net.requests[0].infer()

    activations = {}
    for key in net.outputs.keys():
        activations[key] = exec_net.requests[0].output_blobs[key].buffer
    return activations


def openvino_infer(xml, weights, input_array, return_all=False):
    core = ie.IECore()
    net = core.read_network(model=xml, weights=weights)
    assert(len(net.input_info) == 1)

    i0 = [k for k in net.input_info.keys()][0]
    o0 = [k for k in net.outputs.keys()][0]

    exec_net = core.load_network(network=net, device_name="CPU")
    exec_net.requests[0].input_blobs[i0].buffer[:] = input_array
    exec_net.requests[0].infer()

    if return_all:
        return [exec_net.requests[0].output_blobs[_].buffer for _ in net.outputs.keys()]
    else:
        return exec_net.requests[0].output_blobs[o0].buffer


def openvino_random_input(xml, weights, scale=255.):
    net = ie.IENetwork(model=xml, weights=weights)
    assert(len(net.inputs) == 1)

    i0 = [k for k in net.inputs.keys()][0]
    plugin = ie.IEPlugin(device="CPU")
    exec_net = plugin.load(network=net)
    input_shape = exec_net.requests[0].inputs[i0].shape
    input_array = np.random.random(input_shape).astype(np.float32) * scale

    return input_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('xml')
    parser.add_argument('-i', '--input', default='../oreo.224.jpg')
    parser.add_argument('-y', '--yolo', action='store_true')

    args = parser.parse_args()
    weights=args.xml.replace('.xml', '.bin')

    input_shape = get_model_input_shape(args.xml, weights)
    input_array = load_input(args.input, 1., input_shape)

    if args.yolo:
        predictions = None
        outputs = openvino_infer(args.xml, weights, input_array, True)

        if 'voc' in args.onnx and '2' in args.onnx and 'tiny' not in args.onnx:
            scale_factors = [1.0]
            predictions = yolo.yolov2_voc(outputs, scale_factors, do_activations=True)
            classes = dataset.voc_classes

        for p in predictions:
            print("{}\t{}\t({}, {}, {}, {})".format(classes[p['class_id']],
                                                    int(100*p['confidence']),
                                                    int(p['xmin']), int(p['xmax']),
                                                    int(p['ymin']), int(p['ymax'])))
    else:
        output = openvino_infer(args.xml, weights, input_array)

        if len(output.flatten())==1001:
            classes = dataset.imagenet_classes_with_nul
            classifier.print_topk(output[0].flatten(),classes=classes)
        elif len(output.flatten())==1000:
            classes = dataset.imagenet_classes
            classifier.print_topk(output[0].flatten(),classes=classes)
        else:
            outputs = openvino_infer(args.xml, weights, input_array, True)
            scale_factors = [1. for _ in outputs]
            for o,sf in zip(outputs, scale_factors):
                print(sf*o.flatten()[:8])


if __name__ == "__main__":
    main()
