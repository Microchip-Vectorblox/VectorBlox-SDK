import argparse
import onnxruntime
import numpy as np
import tqdm
import json
import cv2
from vbx.postprocess import classifier, yolo,dataset
from vbx.sim.model import Fletcher32
import sys
import onnx
import re
from onnx import numpy_helper, helper, checker, shape_inference

from .onnx_kld import collect_histogram_data


np.set_printoptions(suppress=True, precision=4, linewidth=120)
np.random.seed(1337)


session_options = onnxruntime.SessionOptions()


def load_input(src, scale, input_shape):
    try:
        channels = input_shape[-3]
    except:
        channels = 1
    height = input_shape[-2]
    width = input_shape[-1]
    ext = src.split('.')[-1].lower()
    if ext in ['npy']:
        arr = np.load(src)
    elif ext in ['jpg', 'jpeg', 'png']:
        if channels == 3:
            img = cv2.imread(src)
            if img is None:
                sys.stderr.write("Error Unable to read image file {}\n".format(src))
                sys.exit(1)
            if height and width and img.shape[:2] != [width, height]:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            arr = img.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
        else:
            img = cv2.imread(src, 0)
            if img is None:
                sys.stderr.write("Error Unable to read image file {}\n".format(src))
                sys.exit(1)
            if height and width and img.shape != [width, height]:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            arr = img.astype(np.float32)
            arr = np.expand_dims(arr, axis=0)
    if scale:
        arr = arr * scale
    arr = np.expand_dims(arr, axis=0)

    return arr


def extend_model_outputs(model):
    model_ = shape_inference.infer_shapes(model)
    graph = model_.graph

    output_names = [_.name for _ in graph.output]
    for vi in graph.value_info:
        if vi.name not in output_names:
            graph.output.extend([vi])

    output_names = [_.name for _ in graph.output]
    for i in graph.input:        
        if i.name not in output_names:
            graph.output.extend([i])

    output_names = [_.name for _ in graph.output]
    for node in graph.node:
        for o in node.output:
            if o not in output_names:
                graph.output.extend([helper.make_tensor_value_info(o, 1, None)])

    return model_


def onnx_activations(model_name, input_array=None):
    model = onnx.load(model_name)
    onnx.checker.check_model(model)
    model_ = extend_model_outputs(model)

    session = onnxruntime.InferenceSession(model_.SerializeToString(), session_options)

    if input_array is None:
        inputs = dict()
        for session_input in  session.get_inputs():
            shape = tuple([1] + session_input.shape[1:])
            input_array = np.zeros(shape,dtype=np.float32)+128
            inputs[session_input.name] = input_array
    else:
        i0 = session.get_inputs()[0].name
        inputs = {i0: input_array}
    outputs = session.run([],inputs)

    activations = {}
    for node, arr in zip(session.get_outputs(), outputs):
        activations[node.name] = arr

    try:
        graph = model.graph
        output_names = [_.name for _ in graph.input]
        for node in graph.node:
            for i in node.input:
                if i not in output_names:
                    output_names.append(i)
        output_names += [_.name for _ in graph.node]
        output_names += [_.name for _ in graph.output]
        for node in graph.node:
            for o in node.output:
                if o not in output_names:
                    output_names.append(o)

        activations = dict(sorted(activations.items(), key=lambda x: output_names.index(x[0])))
    except:
        pass

    return activations


def onnx_activations_batched(model_name, input_array, batch=2, stats_only=False, histogram_dict={}):
    num_passes = int((input_array.shape[0] + batch-1) / batch)
    if num_passes == 1:
        return onnx_activations(model_name, input_array)

    model_ = extend_model_outputs(onnx.load(model_name))
    session = onnxruntime.InferenceSession(model_.SerializeToString(), session_options)
    i0 = session.get_inputs()[0].name

    activations = {}
    stats = {}
    
    for i in range(0, batch*num_passes, batch):
        outputs = session.run([], {i0: input_array[i:i+batch]})
        for out, arr in zip(session.get_outputs(), outputs):
            if stats_only:
                reduce_axis = tuple((i for i in range(len(arr.shape)) if i != 1))
                stat = {"mean":np.mean(arr,axis=reduce_axis) / (input_array.shape[0]/arr.shape[0]),
                        "min":np.min(arr,axis=reduce_axis),
                        "max":np.max(arr,axis=reduce_axis)}
                if out.name in histogram_dict.keys():
                    stat["hist"] = collect_histogram_data(arr, None)
                old_stat = stat
                if out.name in stats:
                    old_stat = stats[out.name]
                stats[out.name] = {"mean":stat['mean'] + old_stat['mean'],
                                   "min":np.minimum(stat['min'],old_stat['min']),
                                   "max":np.maximum(stat['max'],old_stat['max'])}
                if out.name in histogram_dict.keys():
                    stats[out.name]["hist"] = collect_histogram_data(arr, stat['hist'])
            else:
                if out.name in activations:
                    activations[out.name] = np.vstack((activations[out.name], arr))
                else:                    
                    activations[out.name] = arr
    
    return stats if stats_only else activations 


def onnx_infer_batched(onnx_name, input_array, batch=8):
    num_passes = int((input_array.shape[0] + batch-1) / batch)
    if num_passes == 1:
        return onnx_infer(onnx_name, input_array)

    session = onnxruntime.InferenceSession(onnx_name, session_options)
    input_name = session.get_inputs()[0].name
    outputs = None

    with tqdm.tqdm(total=batch*num_passes) as t:
        for i in range(0, batch*num_passes, batch):
            output = session.run([], {input_name: input_array[i:i+batch]})[0]
            if outputs is None:
                outputs = output
            else:
                outputs = np.vstack((outputs, output))
            t.update(batch)

    return outputs


def onnx_infer(onnx_name, input_array):
    session = onnxruntime.InferenceSession(onnx_name, session_options)
    input_name = session.get_inputs()[0].name

    return session.run([], {input_name: input_array})[0]

def onnx_infer_all(onnx_name, input_array):
    session = onnxruntime.InferenceSession(onnx_name, session_options)
    input_name = session.get_inputs()[0].name

    return session.run([], {input_name: input_array})


def onnx_random_infer(onnx_name, scale=255., batch=1):
    input_array = onnx_random_input(onnx_name, batch)
    session = onnxruntime.InferenceSession(onnx_name, session_options)

    return onnx_infer(onnx_name, input_array)


def onnx_random_activations(onnx_name, scale=255., batch=1):
    input_array = onnx_random_input(onnx_name, batch)
    session = onnxruntime.InferenceSession(onnx_name, session_options)

    return onnx_activations(onnx_name, input_array)


def onnx_random_input(onnx_name, scale=255., batch=1):
    session = onnxruntime.InferenceSession(onnx_name, session_options)
    input_shape = session.get_inputs()[0].shape
    input_shape[0] = batch
    input_array = np.random.random(input_shape).astype(np.float32) * scale

    return input_array


def onnx_allclose(model_a, model_b, scale_factor=1.0, atol=1e-05):
    input_array = onnx_random_input(model_a)
    output_a = onnx_infer(model_a, input_array)
    output_b = onnx_infer(model_b, input_array / scale_factor)

    return np.allclose(output_a, output_b, atol=atol)


def onnx_activation_stats(model_name, activations):
    graph = onnx.load(model_name).graph

    inames = [_.name for _ in graph.input]
    onames = [_.name for _ in graph.output]

    for layer, activation in activations.items():
        if layer not in inames and layer not in onames:
            node = [n for n in graph.node if n.name == layer][0]
            print(layer, node.op_type, np.max(activation), np.min(activation))

    for layer, activation in activations.items():
        if layer in onames:
            node = [n for n in graph.node if n.name == layer][0]
            print(layer, node.op_type, np.max(activation), np.min(activation))
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx')
    parser.add_argument('-i', '--input', default='../oreo.224.jpg')
    parser.add_argument('-n', '--normalized', action='store_true')
    parser.add_argument('-a', '--activations', action='store_true')
    parser.add_argument('-y', '--yolo', action='store_true')
    parser.add_argument('--io')

    args = parser.parse_args()

    from . import onnx_helper
    input_shape =  onnx_helper.get_model_input_shape(args.onnx)
    scale = None
    if args.normalized:
        scale = 1./255.
    input_array = load_input(args.input, scale, input_shape)

    if args.activations:
        activations = onnx_activations(args.onnx, input_array)
        np.savez('onnx.npz', **activations)
        for act in activations:
            if act[0] not in ['W', 'b']:
                arr = activations[act]
                print(act, Fletcher32(arr), arr.shape, np.min(arr), np.max(arr))

    elif args.yolo:
        predictions = None
        outputs = onnx_infer(args.onnx, input_array)

        if 'voc' in args.onnx and '2' in args.onnx and 'tiny' not in args.onnx:
            if 'post' in args.onnx:
                scale_factors = [19.08]
            else:
                scale_factors = [1.0]
            predictions = yolo.yolov2_voc(outputs, scale_factors, do_activations=True)
            classes = dataset.voc_classes

        for p in predictions:
            print("{}\t{}\t({}, {}, {}, {})".format(classes[p['class_id']],
                                                    int(100*p['confidence']),
                                                    int(p['xmin']), int(p['xmax']),
                                                    int(p['ymin']), int(p['ymax'])))
    else:
        output = onnx_infer(args.onnx, input_array)

        if len(output.flatten())==1001:
            classes = dataset.imagenet_classes_with_nul
            classifier.print_topk(output[0].flatten(),classes=classes)
        elif len(output.flatten())==1000:
            classes = dataset.imagenet_classes
            classifier.print_topk(output[0].flatten(),classes=classes)
        else:
            outputs = onnx_infer_all(args.onnx, input_array)
            scale_factors = [1. for _ in outputs]
            if args.io:
                with open(args.io) as f:
                    scale_factors = json.load(f)['output_scale_factors']

            for o,sf in zip(outputs, scale_factors):
                print(sf*o.flatten()[:8])


if __name__ == "__main__":
    main()
