import argparse
import onnxruntime
import numpy as np
import tqdm
import json
import cv2
from vbx.postprocess import classifier, yolo,dataset, ssd
from vbx.sim.model import Fletcher32
import sys
import os
import glob
import onnx
import re
from onnx import numpy_helper, helper, checker, shape_inference

from .onnx_kld import collect_histogram_data, get_optimal_threshold, get_valid_kld
from . import onnx_helper


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
            if height and width and img.shape[:2] != [height, width]:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            arr = img.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
        else:
            img = cv2.imread(src, 0)
            if img is None:
                sys.stderr.write("Error Unable to read image file {}\n".format(src))
                sys.exit(1)
            if height and width and img.shape != [height, width]:
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
        # i0 = session.get_inputs()[0].name
        # inputs = {i0: input_array}
        inputs = {}
        for input in session.get_inputs():
            if tuple(input.shape[1:]) == input_array.shape[1:]:
                inputs[input.name] = input_array
            else:
                inputs[input.name] = np.random.randn(1, *input.shape[1:]).astype(np.float32)
    outputs = session.run([], inputs)

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
    if num_passes == 1 and stats_only==False:
        return onnx_activations(model_name, input_array)

    model_ = extend_model_outputs(onnx.load(model_name))
    session = onnxruntime.InferenceSession(model_.SerializeToString(), session_options)

    activations = {}
    stats = {}
    
    for i in range(0, batch*num_passes, batch):
        input_feed = {}
        for input in session.get_inputs():
            if tuple(input.shape[1:]) == input_array.shape[1:]:
                input_feed[input.name] = input_array[i:i+batch]
            else:
                input_feed[input.name] = np.random.randn(batch, *input.shape[1:]).astype(np.float32) # TODO need to fix (was temp fix for gaze model)
        outputs = session.run([], input_feed)
        
        for out, arr in zip(session.get_outputs(), outputs):
            if stats_only:
                reduce_axis = tuple((i for i in range(len(arr.shape)) if i != 1))
                stat = {"mean":np.mean(arr,axis=reduce_axis) / (input_array.shape[0]/arr.shape[0]),
                        "min":np.min(arr,axis=reduce_axis),
                        "max":np.max(arr,axis=reduce_axis)}
                old_stat = None
                if out.name in stats:
                    old_stat = stats[out.name]

                if old_stat:
                    stats[out.name] = {"mean":stat['mean'] + old_stat['mean'],
                                       "min":np.minimum(stat['min'],old_stat['min']),
                                       "max":np.maximum(stat['max'],old_stat['max'])}
                else:
                    stats[out.name] = {"mean":stat['mean'],
                                       "min":stat['min'],
                                       "max":stat['max']}
                if out.name in histogram_dict.keys():
                    if old_stat:
                        stats[out.name]["hist"] = collect_histogram_data(arr, old_stat['hist'])
                    else:
                        stats[out.name]["hist"] = collect_histogram_data(arr, None)
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


def onnx_infer(onnx_model, input_array):
    onnxx = onnx.load(onnx_model)
    session_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(onnxx.SerializeToString())
    input_name = session.get_inputs()[0].name
    
    if onnx_model.endswith('.post.onnx') or onnx_model.endswith('.norm.onnx'):
        if onnx_model.endswith('.post.onnx'):
            io = onnx_model.replace('.post.onnx', '.io.json')
        else:
            io = onnx_model.replace('.norm.onnx', '.io.json')
        with open(io) as f:
            io_dict = json.load(f)
            output_scale_factors = io_dict['output_scale_factors']
            input_scale_factors = io_dict['input_scale_factors']
            input_array /= input_scale_factors[0]
    
    if onnx_model.endswith('.post.onnx') or onnx_model.endswith('.norm.onnx'):
        return [o.flatten() * sf for o,sf in zip(session.run([], {input_name: input_array}), output_scale_factors)]
    else:
        return [o.flatten() for o in session.run([], {input_name: input_array})]

def onnx_infer_multi(onnx_model, input_feed, match_names=False):
    onnxx = onnx.load(onnx_model)
    session = onnxruntime.InferenceSession(onnxx.SerializeToString())
    
    out_names = [o.name for o in session.get_outputs()]
    if match_names:
        if onnx_model.endswith('.norm.onnx'):
            key = 'norm'
            io_names_json = onnx_model.replace('.norm.onnx', '.ionames.json')
        elif onnx_model.endswith('.post.onnx'):
            key = 'post'
            io_names_json = onnx_model.replace('.post.onnx', '.ionames.json')
        elif onnx_model.endswith('.onnx'):
            key = 'onnx'
            io_names_json = onnx_model.replace('.onnx', '.ionames.json')
        with open(io_names_json) as fname:
            io_names_dict = json.load(fname)
        input_feed = {io_names_dict['inputs'][name][key]:val for name,val in input_feed.items()}

        out_names = onnx_helper.get_model_output_xml_names(io_names_json, onnx_model)


    if onnx_model.endswith('.post.onnx') or onnx_model.endswith('.norm.onnx'):
        if onnx_model.endswith('.post.onnx'):
            io = onnx_model.replace('.post.onnx', '.io.json')
        else:
            io = onnx_model.replace('.norm.onnx', '.io.json')
        with open(io) as f:
            io_dict = json.load(f)
            output_scale_factors = io_dict['output_scale_factors']
            input_scale_factors = io_dict['input_scale_factors']
        for i,sf in zip(input_feed.items(),input_scale_factors): # NOTE TODO scale factors may need to be aligned correctly to proper io (?)
            input_feed[i[0]] = i[1] / sf
        outputs = [o.flatten() * sf for o,sf in zip(session.run([], input_feed), output_scale_factors)]
    else:
        outputs = [o.flatten() for o in session.run([], input_feed)]
    
    return {name:val for name,val in zip(out_names, outputs)}

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


def onnx_gather_stats(onnx_model, nodes, input, count, scale, kld_threshold=False):

    valid_kld = {}
    if kld_threshold:
        valid_kld = get_valid_kld(onnx_model)

    if os.path.isdir(input):
        inputs = []
        extensions = ['*.jpg', '*.png', '*.jpeg', '*.npy']
        extensions += [e.upper() for e in extensions]
        for ext in extensions:
            inputs += sorted(glob.glob(os.path.join(input, ext)))
        if count:
            inputs = inputs[:count]
        input_shape = onnx_helper.get_model_input_shape(onnx_model)
        input_arrays = np.vstack([load_input(i, scale, input_shape) for i in inputs])
    else:
        input_arrays = np.load(input)
    stats = onnx_activations_batched(onnx_model, input_arrays, stats_only=True, histogram_dict=valid_kld)

    stats_list = []
    for output in sorted(stats.keys()):
        entry = {'id':output,
                'mean': stats[output]['mean'],
                'max': stats[output]['max'],
                'min': stats[output]['min']}
        if 'hist' in stats[output]:
            (hist, hist_edges, min_val, max_val, th) = stats[output]['hist']
            _min, _max, opt, min_div = get_optimal_threshold(stats[output]['hist'], 255)
            entry['kld'] = opt

        stats_list.append(entry)

    return stats_list


def onnx_save_stats(fname, stats_np):
    stats = []
    for entry_np in stats_np:
        entry = entry_np.copy()
        for key in entry:
            if isinstance(entry[key], np.ndarray):
                entry[key] = entry[key].tolist()
            elif isinstance(entry[key], np.generic):
                entry[key] = float(entry[key])
        stats.append(entry)

    with open(fname, 'w') as f:
        json.dump(stats, f)


def onnx_load_stats(fname, mode=None):
    stats = {}
    with open(fname) as f:
        j = json.load(f)
        for arr in j:
            channel_maximums = np.asarray(arr['max'],dtype=np.float32)
            channel_minimums = np.asarray(arr['min'],dtype=np.float32)
            channel_abs = np.max(np.stack((np.abs(channel_maximums), np.abs(channel_minimums)), axis=-1),axis=-1)

            stats[arr['id']] = {'abs': channel_abs, 'max': channel_maximums, 'min':channel_minimums} 
            if 'kld' in arr:
                stats[arr['id']]['kld'] = np.asarray(arr['kld'])
    return stats


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
    parser.add_argument('-a', '--activations', action='store_true')
    parser.add_argument('-y', '--yolo', action='store_true')
    parser.add_argument('-s', '--ssd', action='store_true')
    parser.add_argument('-t', '--threshold', type=float, default=0.5)

    args = parser.parse_args()

    input_shape =  onnx_helper.get_model_input_shape(args.onnx)
    input_array = load_input(args.input, 1., input_shape)
    outputs = onnx_infer(args.onnx, input_array)

    if args.activations:
        activations = onnx_activations(args.onnx, input_array)
        np.savez('onnx.npz', **activations)
        for act in activations:
            if act[0] not in ['W', 'b']:
                arr = activations[act]
                print(act, Fletcher32(arr), arr.shape, np.min(arr), np.max(arr))

    elif args.yolo:
        predictions = None
        scale_factors = [1.0]

        if 'voc' in args.onnx and '2' in args.onnx and 'tiny' not in args.onnx: 
            predictions = yolo.yolov2_voc(outputs, scale_factors, do_activations=True)
        elif 'voc' in args.onnx and '2' in args.onnx and 'tiny' in args.onnx:
            predictions = yolo.yolov2_tiny_voc(outputs, scale_factors)
        elif 'voc' not in args.onnx and '2' in args.onnx and 'tiny' not in args.onnx:
            predictions = yolo.yolov2_coco(outputs, scale_factors, do_activations=True)
        elif 'voc' not in args.onnx and '2' in args.onnx and 'tiny' in args.onnx:
            predictions = yolo.yolov2_tiny_coco(outputs, scale_factors)
        # TODO currently not working
        # elif 'voc' not in args.xml and '3' in args.xml and 'tiny' not in args.xml:
        #     predictions = yolo.yolov3_coco(outputs, scale_factors)
        # elif 'voc' not in args.xml and '3' in args.xml and 'tiny' in args.xml:
        #     predictions = yolo.yolov3_tiny_coco(outputs, scale_factors)
        
        if 'voc' in args.onnx:
            classes = dataset.voc_classes
        else:
            classes = dataset.coco_classes

        for p in predictions:
            print("{}\t{}\t({}, {}, {}, {})".format(classes[p['class_id']],
                                                    int(100*p['confidence']),
                                                    int(p['xmin']), int(p['xmax']),
                                                    int(p['ymin']), int(p['ymax'])))
    
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
        if len(outputs[0])==1001:
            classes = dataset.imagenet_classes_with_nul
            classifier.print_topk(outputs[0],classes=classes)
        elif len(outputs[0])==1000:
            classes = dataset.imagenet_classes
            classifier.print_topk(outputs[0],classes=classes)
        else:
            print(outputs[:8])


if __name__ == "__main__":
    main()
