import argparse
import json
import openvino.inference_engine as ie
import numpy as np
import cv2
import os
from vbx.postprocess import classifier, yolo,dataset, ssd
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

    # layers = [layer for layer in net.layers]
    # for layer in layers:
    #     if net.layers[layer].type != "Const":
    #     net.add_outputs(layer)
    import ngraph as ng
    func_net = ng.function_from_cnn(net)
    ops_net = func_net.get_ordered_ops()
    ops_net_names = []
    for op in ops_net:
        if op.get_type_name() not in ['Parameter', 'Constant', 'Result']:
            ops_net_names.append(op.friendly_name)
    for layer in ops_net_names:
        net.add_outputs(layer)
    # layers_map = core.query_network(network=net, device_name="CPU")
    # layers = layers_map.keys()
    # layers = [l for l in layers if '_const' not in l]
    # layers = [l for l in layers if '_port' not in l]
    # for layer in layers:
        # net.add_outputs(layer)

    exec_net = core.load_network(network=net, device_name="CPU")
    exec_net.requests[0].input_blobs[i0].buffer[:] = input_array
    exec_net.requests[0].infer()

    activations = {}
    for key in net.outputs.keys():
        activations[key] = exec_net.requests[0].output_blobs[key].buffer
    return activations


def openvino_random_input(xml, weights, scale=255.):
    net = ie.IENetwork(model=xml, weights=weights)
    assert(len(net.inputs) == 1)

    i0 = [k for k in net.inputs.keys()][0]
    plugin = ie.IEPlugin(device="CPU")
    exec_net = plugin.load(network=net)
    input_shape = exec_net.requests[0].inputs[i0].shape
    input_array = np.random.random(input_shape).astype(np.float32) * scale

    return input_array

def openvino_infer(xml_model, input_array):
    weights=xml_model.replace('.xml', '.bin')
    core = ie.IECore()
    net = core.read_network(model=xml_model, weights=weights)
    exec_net = core.load_network(network=net, device_name="CPU")
    assert(len(net.input_info) == 1)
    i0 = [k for k in net.input_info.keys()][0]

    exec_net.requests[0].input_blobs[i0].buffer[:] = input_array
    exec_net.requests[0].infer()
    outputs = [k for k in net.outputs.keys()]
    return [exec_net.requests[0].output_blobs[o].buffer.flatten() for o in outputs]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('xml')
    parser.add_argument('-i', '--input', default='../oreo.224.jpg')
    parser.add_argument('-a', '--activations', action='store_true')
    parser.add_argument('-y', '--yolo', action='store_true')
    parser.add_argument('-s', '--ssd', action='store_true')
    parser.add_argument('-t', '--threshold', type=float, default=0.5)

    args = parser.parse_args()
    weights=args.xml.replace('.xml', '.bin')

    input_shape = get_model_input_shape(args.xml, weights)
    input_array = load_input(args.input, 1., input_shape)
    outputs = openvino_infer(args.xml, input_array)

    if args.yolo:
        predictions = None
        scale_factors = [1.0]

        if 'voc' in args.xml and '2' in args.xml and 'tiny' not in args.xml:
            predictions = yolo.yolov2_voc(outputs, scale_factors, do_activations=True)
        elif 'voc' in args.xml and '2' in args.xml and 'tiny' in args.xml:
            predictions = yolo.yolov2_tiny_voc(outputs, scale_factors)
        elif 'voc' not in args.xml and '2' in args.xml and 'tiny' not in args.xml:
            predictions = yolo.yolov2_coco(outputs, scale_factors, do_activations=True)
        elif 'voc' not in args.xml and '2' in args.xml and 'tiny' in args.xml:
            predictions = yolo.yolov2_tiny_coco(outputs, scale_factors)
        # TODO currently not working
        # elif 'voc' not in args.xml and '3' in args.xml and 'tiny' not in args.xml:
        #     predictions = yolo.yolov3_coco(outputs, scale_factors)
        # elif 'voc' not in args.xml and '3' in args.xml and 'tiny' in args.xml:
        #     predictions = yolo.yolov3_tiny_coco(outputs, scale_factors)

        if 'voc' in args.xml:
            classes = dataset.voc_classes
        else:
            classes = dataset.coco_classes

        for p in predictions:
            print("{}\t{}\t({}, {}, {}, {})".format(classes[p['class_id']],
                                                    int(100*p['confidence']),
                                                    int(p['xmin']), int(p['xmax']),
                                                    int(p['ymin']), int(p['ymax'])))
    elif args.activations:
        activations = openvino_activations(args.xml, weights, input_array)
        np.savez('openvino.npz', **activations)
    elif args.ssd:
        scale_factors = len(outputs) * [1.0]
        predictions = ssd.ssdv2_predictions(outputs, scale_factors, args.threshold, nms_threshold=0.4, top_k=1)
        classes = ssd.coco91

        for p in predictions:
            print("{}\t{}\t({}, {}, {}, {})".format(classes[p['class_id']],
                                                int(100*p['confidence']),
                                                int(p['xmin']), int(p['xmax']),
                                                int(p['ymin']), int(p['ymax'])))
    # classifier
    else:
        outputs = outputs[0]

        if len(outputs.flatten())==1001:
            classes = dataset.imagenet_classes_with_nul
            classifier.print_topk(outputs,classes=classes)
        elif len(outputs.flatten())==1000:
            classes = dataset.imagenet_classes
            classifier.print_topk(outputs,classes=classes)
        else:
            scale_factors = [1. for _ in outputs]
            for o,sf in zip(outputs, scale_factors):
                print(sf*o.flatten()[:8])


if __name__ == "__main__":
    main()
