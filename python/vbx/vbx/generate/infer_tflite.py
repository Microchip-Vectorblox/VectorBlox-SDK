import silence_tensorflow.auto
import tensorflow as tf
import cv2
import numpy as np
import argparse
from .utils import existing_dir, existing_file
import os
import glob
import vbx.sim
import pprint
import shutil
import json

from .utils import match_shape, calc_diff, create_tensor_data
from .split_tflite import generate_split_graphs, generate_join_graphs 

import subprocess
import shlex
import natsort

DEBUG=0

def get_input_details(model):
    interpreter= tf.lite.Interpreter(
             model_path=model)
    return interpreter.get_input_details()


def get_output_details(model):
    interpreter= tf.lite.Interpreter(
             model_path=model)
    return interpreter.get_output_details()

def get_input_data(input_file, width, height, channels, mean, scale, rgb):
    img = cv2.imread(input_file)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    if channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
    img = img.astype(np.float32)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - mean) / scale
    img = np.expand_dims(img, axis=0)
    return img


def default_input_arr(shape, dtype, seed=42):
    np.random.seed(seed)
    # arr = np.zeros(shape, dtype=np.uint8)+128
    arr = np.random.randint(256, size=shape)
    return arr.astype(dtype)


def get_tflite_io(tflite_model, input_files, subdir, mean=0., scale=1., rgb=False):

    interpreter= tf.lite.Interpreter(
             model_path=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inputs = {}
    for i,input_detail in enumerate(input_details):
        shape = input_detail['shape']
        index = input_detail['index']
        name = input_detail['name']
        dtype = np.dtype(input_detail['dtype'])

        if input_files is not None:
            if type(input_files) is type({}):
                if index not in input_files.keys():
                    arr = np.load(os.path.join(subdir, 'activations.{}.npy'.format(index)))
                else:
                    arr = input_files[index]
            elif len(input_files) > i: # for when input_files is a list of strings (and possibly more than one)
                input_file = input_files[i]

                if input_file is None:
                    arr = default_input_arr(shape, dtype)
                elif '.npy' in input_file:
                    arr = np.load(input_file)
                elif type(input_file) is np.ndarray:
                    arr = input_file
                else:
                    assert(len(shape) == 4), "Input image passing supported only for 4-dimension input TFLite"
                    channels, height, width = shape[3], shape[1], shape[2]
                    arr = get_input_data(input_file, width, height, channels, mean, scale, rgb)
                    input_scale, input_zero_point = input_detail.get('quantization', (0.0, 0))
                    if  input_scale != 0.0:
                        arr = (arr / input_scale) + input_zero_point
                    arr = arr.astype(dtype)
        else:
            arr = default_input_arr(shape, dtype)

        # expand if necessary
        if len(arr.shape) == len(shape) - 1:
            arr = np.expand_dims(arr, axis=0)

        # transpose if necessary
        if len(shape) == 4:
            if tuple(shape) == tuple(arr.shape):
                pass
            elif tuple(shape) == tuple(arr.transpose((0,3,1,2)).shape):
                arr = arr.transpose((0,3,1,2))
            elif tuple(shape) == tuple(arr.transpose((0,2,3,1)).shape):
                arr = arr.transpose((0,2,3,1))
        elif len(shape) == 3:
            if tuple(shape) == tuple(arr.shape):
                pass
            elif tuple(shape) == tuple(arr.transpose((2,0,1)).shape):
                arr = arr.transpose((2,0,1))
            elif tuple(shape) == tuple(arr.transpose((1,2,0)).shape):
                arr = arr.transpose((1,2,0))

        if tuple(shape) != tuple(arr.shape):
            print("input array {} doesn't match ({} != {})".format(i, shape, arr.shape))
            break


        interpreter.set_tensor(input_detail['index'], arr)
        # inputs[input_detail['index']] = interpreter.get_tensor(input_detail['index'])
        inputs[input_detail['name']] = interpreter.get_tensor(input_detail['index'])

    interpreter.invoke()

    outputs = {}
    for output_detail in output_details:
        # outputs[output_detail['index']] = interpreter.get_tensor(output_detail['index'])
        outputs[output_detail['name']] = interpreter.get_tensor(output_detail['index'])

    return inputs, outputs

def get_vnnx_io(vnnx_model, tflite_keys, input_files, tfin, subdir, mean=0., scale=1., rgb=False):
    with open(vnnx_model, 'rb') as mf:
        model = vbx.sim.Model(mf.read())    

    # get input(s)
    inputs = []
    flattened = []

    for i, k in enumerate(tflite_keys):
        if tfin and input_files is not None:
            if k not in input_files.keys(): # should only happen when substituing vnnx subgraphs in
                path = subdir+'/activations.{}.npy'.format(k)
                input_file = np.load(os.path.join(subdir, 'activations.{}.npy'.format(k)))
            else:
                input_file = input_files[k]
            arr = match_shape(input_file, model.input_dims[i], to_tfl=False) # from tfl shape to vnnx shape
        else:
            vnnx_inp_path = subdir+'/vnnx_activations.{}.npy'.format(k)
            shape = model.input_dims[i]
            if input_files is not None and len(input_files) > i:
                input_file = input_files[i]
                if type(input_file) is str and '.npy' not in input_file:
                    if shape[-1] not in [1,3]:
                        channels, height, width = shape
                        nchw = True
                    else:
                        height, width, channels = shape
                        nchw = False
                    img = get_input_data(input_file, width, height, channels, mean, scale, rgb)
                    if nchw:
                        img = img.swapaxes(-2, -1).swapaxes(-3, -2)
                    arr = ((img / model.input_scale_factor[i]) + model.input_zeropoint[i])
                elif type(input_file) is str and '.npy' in input_file:
                    arr = np.load(input_file)
                elif os.path.exists(vnnx_inp_path):
                    arr = np.load(vnnx_inp_path)
                elif type(input_file) is np.ndarray:
                    arr = input_file
                else: #type input file is None
                    arr = default_input_arr(shape, dtype)
            elif os.path.exists(vnnx_inp_path):
                arr = np.load(vnnx_inp_path)
            else:
                arr = default_input_arr(shape, dtype)
        arr = arr.astype(model.input_dtypes[i])
        inputs.append(arr)
        flattened.append(arr.flatten())
    
    outputs = []
    for o in range(model.num_outputs):
        output = model.run(flattened)[o]
        output = output.reshape(model.output_dims[o]).astype(model.output_dtypes[o])
        outputs.append(output)

    return inputs, outputs


def print_diff(src_name, dst_name, src, dst, info, verbose):

    all_within_threshold, abs_diff, total_vals_diff, counter = calc_diff(src, dst)
    error_rate = total_vals_diff / np.prod(src.shape) * 100
    
    if verbose or not all_within_threshold or (error_rate >= 1):
        print("{} vs {}".format(src_name, dst_name))
        print("Error rate for {}: {:3.2f}%".format(info, error_rate))
        print("Max diff: {}, Num of diff values: {}".format(np.max(abs_diff), total_vals_diff))                
        if not all_within_threshold:
            print("All {} values NOT within 1 of {}!".format(dst_name, src_name))
        if verbose:
            pprint.pprint([("Diff", "Count"), sorted(counter.items())], width=20, sort_dicts=True)
        print()


def generate_inputs_outputs(tflite_model_binary):
    interpreter = tf.lite.Interpreter(model_content=tflite_model_binary)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_values = {}
    min_value = 0
    max_value = 20
    for i,input_detail in enumerate(input_details):
        if input_detail["dtype"] == np.float32:
            min_value = -1
            max_value = 1
        elif input_detail["dtype"] == np.int8:
            min_value = -128
            max_value = 127
        elif input_detail["dtype"] == np.uint8:
            min_value = 0
            max_value = 255
        input_value = create_tensor_data(
                input_detail["dtype"],
                input_detail["shape"],
                min_value=min_value,
                max_value=max_value,
                int8_range=False)
        interpreter.set_tensor(input_detail["index"], input_value)
        input_values.update({"i{}".format(i): input_value})
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_values = {}
    for o, output_detail in enumerate(output_details):
        output_values.update({ "o{}".format(o): interpreter.get_tensor(output_detail["index"])})
    return input_values, output_values


def compare_tflite(tflite_graph_name, size_conf='V1000', error_rate_threshold=0, error_threshold=1, verbose=False):
    if verbose:
        print(tflite_graph_name)
    with open(tflite_graph_name, 'rb') as f:
        tflite_model_binary = f.read()
        # run tflite with tf inputs to get tflite outputs
        baseline_input_map_tfl, baseline_output_map_tfl = generate_inputs_outputs(tflite_model_binary)
        
        example = { "inputs": baseline_input_map_tfl, "outputs": baseline_output_map_tfl }

        tfl_in_files = []
        for (inp_name, inp) in example['inputs'].items():
            npy_input_name = tflite_graph_name.replace('.tflite', '.' + inp_name + '.npy')
            np.save(npy_input_name, inp)
            tfl_in_files.append(npy_input_name)
        tfl_in_files = ' '.join(tfl_in_files)
        if verbose:
            for (outp_name, outp) in example['outputs'].items():
                npy_output_name = tflite_graph_name.replace('.tflite', '.' + outp_name + '.npy')
                np.save(npy_output_name, outp)
        vnnx_graph_name = tflite_graph_name.replace('.tflite', '.vnnx')
        cmd = 'vnnx_compile -c {} -t {} -i {} -o {}'.format(size_conf, tflite_graph_name, tfl_in_files, vnnx_graph_name)
        log = []
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True)
            log.append(res.stderr)
        except:
            print(log)
            return
        if not os.path.exists(vnnx_graph_name):
            return
        cmd = 'python -m vbx.sim {} -d'.format(vnnx_graph_name)
        log = []
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True)
            log.append(res.stderr)
        except:
            print(log)
            return

        with open(vnnx_graph_name, "rb") as vnnx_bin:
            vnnx_model = vbx.sim.Model(vnnx_bin.read())
            flattened = []
            outputs = []
            vnnx_io = {'inputs':{}, 'outputs':{}}
            for (inp_name, inp), idx in zip(example['inputs'].items(), range(vnnx_model.num_inputs)):
                vnnx_i = match_shape(inp, vnnx_model.input_shape[idx], to_tfl=False) #from tfl shape to vnnx
                vnnx_io['inputs'][inp_name] = vnnx_i
                flattened.append(vnnx_i.flatten())

            outputs = vnnx_model.run(flattened)
            # outputs = vnnx_model.run(vnnx_model.test_input)

            for (out_name,tfl_out), idx in zip(example['outputs'].items(), range(vnnx_model.num_outputs)):
                output = outputs[idx]
                output = output.reshape(vnnx_model.output_shape[idx])
                vnnx_io['outputs'][out_name] = output
                if verbose:
                    npy_output_name = tflite_graph_name.replace('.tflite', '.' + out_name + '.vnnx.npy')
                    np.save(npy_output_name, output)
                #TODO use vbx.sim outputs
                output = np.load(os.path.join(os.path.dirname(tflite_graph_name), 'vnnx.output.{}.npy'.format(idx)))

            tfl_out = match_shape(tfl_out, vnnx_model.output_shape[idx], to_tfl=False) # from tfl shape to vnnx
            
            all_within_threshold, abs_diff, total_vals_diff, counter = calc_diff(output, tfl_out, error_threshold)
            error_rate = total_vals_diff / np.prod(output.shape) * 100
            if not all_within_threshold and error_rate > error_rate_threshold:
                print('\n{} #diff > {}: {} ({:3.2f}%), max_diff: {}'.format(os.path.basename(tflite_graph_name), error_threshold, total_vals_diff, error_rate, np.max(abs_diff)))
                if verbose:
                    pprint.pprint([("Diff", "Count"), sorted(counter.items())], width=20, sort_dicts=True)

                heatmap = np.abs(output-tfl_out)
                heatmap_name = tflite_graph_name.replace('.tflite','.heatmap.npy')
                if len(heatmap.shape) == 4 and heatmap.shape[0] == 1:
                    heatmap = heatmap.squeeze(axis=0)
                while len(heatmap.shape) < 3:
                    heatmap = np.expand_dims(heatmap, axis = 0)
                if len(heatmap.shape) != 3:
                    print('heatmap is > 3 dimensions')
                else:
                    for c,channel in enumerate(heatmap):
                        sum_diff_above_threshold = np.sum(channel > 1)
                        if sum_diff_above_threshold > 0 and verbose:
                            print('channel {} # diff > {}: {} ({:3.2f}%)'.format(c, error_threshold, sum_diff_above_threshold, sum_diff_above_threshold / np.prod(channel.shape) * 100))
                    
                    np.save(heatmap_name, heatmap)
                    print('heatmap for Numpy Viewer:', os.path.basename(heatmap_name))

def compare():
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite', type=existing_file)
    parser.add_argument('-c', '--size-conf', help='size configuration to build model for',
                        choices = ['V250','V500','V1000'], default='V1000')
    parser.add_argument('-e', '--error-rate', type=int, default=0)
    parser.add_argument('-t', '--error-threshold',  type=int, default=1)
    parser.add_argument('-s', '--split-every-op', action='store_true')
    parser.add_argument("-v", "--verbose", action='store_true', help='Prints all activations, even those with less than 1% off-by-one')
    args = parser.parse_args()

    if os.path.isdir('./subgraphs'):
        shutil.rmtree('./subgraphs', ignore_errors=True)
    generate_split_graphs(args.tflite, './', split_every_op=args.split_every_op)
    for t in natsort.natsorted(glob.glob(os.path.join('./subgraphs', '*.tflite'))):
        compare_tflite(t, args.size_conf, args.error_rate, args.error_threshold, args.verbose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=existing_file)
    parser.add_argument("-i", "--input", nargs='*', type=existing_file)
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=0.)
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=1.)
    parser.add_argument("--vnnx", action='store_true', help='Compare vnnxlite model against tflite')
    parser.add_argument("--tfin", action='store_true', help='When comparing vnnxlite, compare with tflite input activations instead')
    parser.add_argument("--count", type=int, default=8, help='Used when args.input is a directory of images')
    parser.add_argument("--layers_to_sub", nargs='+', default=[], help='List of layer IDs to substitute the tflite subgraph with corresponding vnnx subgraph (use : for contiguous sequence like in python syntax)')
    parser.add_argument("-v", "--verbose", action='store_true', help='Prints all activations, even those with less than 1% off-by-one')
    args = parser.parse_args()


    # Get models
    models = []
    vnnx_models = []
    layer_ids_subbed = []
    if os.path.isdir(args.model):
        models = sorted(glob.glob(os.path.join(args.model, "*.tflite")), key=lambda x: int(x.split('.')[-2]))
        if args.vnnx or args.layers_to_sub:
            vnnx_models = sorted(glob.glob(os.path.join(args.model, "*.vnnx")), key=lambda x: int(x.split('.')[-2]))
        model_dir = args.model
    elif ".tflite" in args.model:
        models = [args.model]
        model_dir = os.path.dirname(args.model)
        if args.vnnx:
            vnnx_models = [args.model.replace('.tflite', ".vnnx")]
    else:
        vnnx_models = [args.model]
        model_dir = os.path.dirname(args.model)

    # Get list of indices to sub in vnnx subgraphs
    for i in args.layers_to_sub:
        if ':' in i:
            start, end = i.split(':')
            if len(start) == 0:
                start = '0'
            elif len(end) == 0:
                end = str(len(models)-1)
            start_idx = int(start)
            end_idx = int(end)+1
            layer_ids_subbed += range(start_idx,end_idx)
        else:
            layer_ids_subbed.append(int(i))

    # Get inputs
    if args.input:
        if os.path.isdir(args.input[0]):
            images = sorted(glob.glob(os.path.join(args.input[0], '*.jpg')))
            images = images[:args.count]
        else:
            images = args.input
    else:
        images = [None]

    # Clean up old i/o activations data
    old_subdirs = glob.glob(os.path.join(model_dir, "input*"))
    for sub in old_subdirs:
        if os.path.exists(sub):
            shutil.rmtree(sub, ignore_errors=True)

    for num, img in enumerate(images):
        print(num, img)
        subdir = os.path.join(model_dir, 'input{}'.format(num)) 
        os.mkdir(subdir)

        if args.vnnx:
            inputs = [img]
            vnnx_i = [img]
            model_num = 0

            for model, vnnx_model in zip(models, vnnx_models):
                with open(model.replace('.tflite', '.json')) as f:
                    jmodel = json.load(f)
                op_codes = jmodel['operator_codes']
                subgraph = jmodel['subgraphs'][0]
                layer_types = [op_codes[_['opcode_index']]['builtin_code'] for _ in subgraph['operators']]
                tf_inputs, outputs = get_tflite_io(model, inputs, subdir, args.mean, args.scale, (not args.bgr))

                if args.tfin:
                    vnnx_i, vnnx_o = get_vnnx_io(vnnx_model, tf_inputs.keys(), tf_inputs, True, subdir, args.mean, args.scale, (not args.bgr))
                else:
                    vnnx_i, vnnx_o = get_vnnx_io(vnnx_model, tf_inputs.keys(), vnnx_i, False, subdir, args.mean, args.scale, (not args.bgr))

                for idx, k in enumerate(tf_inputs.keys()):
                    np.save(os.path.join(subdir, 'activations.{}.npy'.format(k)), tf_inputs[k])
                    
                    # do not save vnnx_i as .npy when passing in tflite acts (vnnx_i == tflite acts), overwrites vnnx_o acts
                    if not args.tfin or model_num == 0: 
                        np.save(os.path.join(subdir, 'vnnx_activations.{}.npy'.format(k)), vnnx_i[idx])

                with open(vnnx_model, 'rb') as mf:
                    vnnx_o_shapes = vbx.sim.Model(mf.read()).output_dims
                    
                for idx, k in enumerate(outputs.keys()):
                    np.save(os.path.join(subdir, 'activations.{}.npy'.format(k)), outputs[k])
                    np.save(os.path.join(subdir, 'vnnx_activations.{}.npy'.format(k)), vnnx_o[idx])

                    tf_o = match_shape(outputs[k], vnnx_o_shapes[idx], to_tfl=False) # from tfl shape to vnnx

                    img_name = os.path.basename(img) if img is not None else 'None'
                    info = '{} sublayer {} (i/o {}/{}) on image {}'.format(layer_types[-1], model_num, ','.join([str(_) for _ in tf_inputs.keys()]), k, img_name)
                    print_diff('tflite', 'vnnx', tf_o, vnnx_o[idx], info, args.verbose)

                model_num += 1
                vnnx_i = vnnx_o
                inputs = outputs

        else:
            inputs = [img]
            for idx, model in enumerate(models):
                tf_inputs, outputs = get_tflite_io(model, inputs, subdir)
                if idx in layer_ids_subbed:
                    v_inputs, v_outputs = get_vnnx_io(vnnx_models[idx], tf_inputs.keys(), inputs, True, subdir)
                    
                    for i, k in enumerate(tf_inputs.keys()):
                        v_input = match_shape(v_inputs[i], tf_inputs[k].shape, to_tfl=True) # from vnnx shape to tfl
                        np.save(os.path.join(subdir, 'activations.{}.npy'.format(k)), v_input)
                    for i, k in enumerate(outputs.keys()):
                        v_output = match_shape(v_outputs[i], outputs[k].shape, to_tfl=True) # from vnnx shape to tfl
                        np.save(os.path.join(subdir, 'activations.{}.npy'.format(k)), v_output)
                        outputs[k] = v_output
                else:
                    for k in tf_inputs.keys():
                        np.save(os.path.join(subdir, 'activations.{}.npy'.format(k)), tf_inputs[k])
                    for k in outputs.keys():
                        np.save(os.path.join(subdir, 'activations.{}.npy'.format(k)), outputs[k])
                inputs = outputs


    if args.vnnx:
        all_acts = sorted(glob.glob(os.path.join(model_dir, "input*/activations*")), key=lambda x: int(x.split('.')[-2]))
        all_vnnx_acts = sorted(glob.glob(os.path.join(model_dir, "input*/vnnx_activations*")), key=lambda x: int(x.split('.')[-2]))

        curr_input = 0
        activations = dict()
        for tfact, vact in zip(all_acts, all_vnnx_acts):
            v_act = np.load(vact)
            tf_act = np.load(tfact)
            tf_act = match_shape(tf_act, v_act.shape, to_tfl=False) # from tfl shape to vnnx

            if curr_input == 0:
                diff = np.zeros(tf_act.shape)
            
            diff += np.abs(tf_act - v_act)

            curr_input += 1
            if curr_input == len(images):
                np.save(os.path.join(model_dir, "act_diff_map.{}.npy".format(tfact.split('.')[-2])), diff)
                if np.any(diff):
                    activations[tfact.split('.')[-2]] = diff
                curr_input = 0
        
        np.savez(os.path.join(model_dir, 'diff.npz'), **activations)

if __name__ == "__main__":
    main()
