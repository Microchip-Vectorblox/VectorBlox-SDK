import argparse
import json
import openvino.inference_engine as ie
import numpy as np
import cv2
import os
from vbx.postprocess import classifier, yolo,dataset
from .onnx_infer import load_image

np.set_printoptions(suppress=True, precision=4, linewidth=120)
np.random.seed(1337)


#def load_image(image_src, scale=255.0, channels=3):
#    if channels == 3:
#        img = cv2.imread(image_src)
#        arr = img.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
#    else:
#        img = cv2.imread(image_src, 0)
#        arr = img.astype(np.float32)
#        arr = np.expand_dims(arr, axis=0)
#    if scale != 255.0:
#        arr = arr / 255. * scale
#    arr = np.expand_dims(arr, axis=0)
#
#    return arr


def openvino_activations(xml, weights, input_array, extension=None):
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


def openvino_infer(xml, weights, input_array, extension=None, return_all=False):
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


def openvino_random_input(xml, weights, scale=255., extension=None):
    net = ie.IENetwork(model=xml, weights=weights)
    assert(len(net.inputs) == 1)

    i0 = [k for k in net.inputs.keys()][0]
    plugin = ie.IEPlugin(device="CPU")
    if extension:
        plugin.add_cpu_extension(extension)
    exec_net = plugin.load(network=net)
    input_shape = exec_net.requests[0].inputs[i0].shape
    input_array = np.random.random(input_shape).astype(np.float32) * scale

    return input_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('xml')
    parser.add_argument('-i', '--image', default='../oreo.224.jpg')
    parser.add_argument('-c', '--channels', type=int, default=3)
    parser.add_argument('-s', '--scale', type=float, default=255.)
    parser.add_argument('-y', '--yolo', action='store_true')
    parser.add_argument('-x', '--extension')

    args = parser.parse_args()

    weights=args.xml.replace('.xml', '.bin')
    input_array = load_image(args.image, args.scale)
    if args.yolo:
        outputs = openvino_infer(args.xml, weights, input_array, args.extension, True)
        if '2' in args.xml and 'tiny' not in args.xml:
            predictions = yolo.yolov2_voc(outputs, [1.0], do_activations=False)
            classes = dataset.voc_classes
        for p in predictions:
            print("{}\t{}\t({}, {}, {}, {})".format(classes[p['class_id']],
                                                    int(100*p['confidence']),
                                                    int(p['xmin']), int(p['xmax']),
                                                    int(p['ymin']), int(p['ymax'])))
    else:
        output = openvino_infer(args.xml, weights, input_array, args.extension)
        if len(output.flatten())==1001:
            classes = dataset.imagenet_classes_with_nul
        else:
            classes = dataset.imagenet_classes

        classifier.print_topk(output[0].flatten(),classes=classes)

if __name__ == "__main__":
    main()
