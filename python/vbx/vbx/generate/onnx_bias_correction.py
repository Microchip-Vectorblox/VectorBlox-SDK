import glob
import onnx
import onnxruntime
import tqdm
import cv2
import numpy as np
import os
import json
from multiprocessing import Pool
import argparse
from . import onnx_infer


np.set_printoptions(suppress=True, precision=4, linewidth=120)


Q32 = 16
Q16 = 13
Q8 = 7
U8 = 8


def convert_array(arr):
    q = 0
    if arr.dtype == np.int16:
        q = Q16
    elif arr.dtype == np.int32:
        q = Q32
    elif arr.dtype == np.int8:
        q = Q8
    elif arr.dtype == np.uint8:
        q = U8

    narr = arr.astype(np.float32) / 2**q

    return narr


def get_previous_layer_ids(js, current_id):
    prev_layer_ids = []
    input_id = js['layers'][current_id]['input_id']
    for l, layer in enumerate(js['layers']):
        if layer['output_id'] == input_id:
            prev_layer_ids.append(l)
    return prev_layer_ids


def get_current_subgraph_nodes(js, bias_layers_nums, bias_layer):
    layers = [bias_layer]

    prev = get_previous_layer_ids(js, bias_layer)
    for p in prev:
        if p in bias_layers_nums:
            continue
        else:
            layers = get_current_subgraph_nodes(js, bias_layers_nums, p) + layers
    return layers


def vnnx_load_input(input, input_shape, input_scale=1./255., input_dtype=np.uint8):
    arr = onnx_infer.load_input(input, input_scale, input_shape)
    arr = (arr*255.).clip(0, 255.).astype(np.uint8)
    flattened = arr.flatten()
    return [flattened]


def collect_inputs(samples_folder, samples_count):
    inputs = []
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.npy']
    extensions += [e.upper() for e in extensions]
    for ext in extensions:
        inputs += sorted(glob.glob(os.path.join(samples_folder, ext)))
    if samples_count:
        inputs = inputs[:samples_count]
    return inputs


def get_previous_weighted_layer_output_ids(js, output_id):
    previous_weighted_output_ids = []
    layers = [l for l in js['layers'] if l['output_id'] == output_id]
    for layer in layers:
        if is_weighted_layer(layer):
            previous_weighted_output_ids += [layer['output_id']]
        else:
            previous_weighted_output_ids += get_previous_weighted_layer_output_ids(js, layer['input_id'])
    return previous_weighted_output_ids


def is_weighted_layer(layer):
    return (layer['op_type'] in ['Conv', 'Gemm']) or (layer['op_type'] == "Identity" and len(layer['sublayers']) and layer['sublayers'][-1]['op_type'] == 'Add')


def jslayers_to_adjust(js):
    _, output_ids = get_io_ids(js)

    ignored_output_ids = []
    for output_id in output_ids:
        ignored_output_ids += get_previous_weighted_layer_output_ids(js, output_id)

    layers = []
    for l, layer in enumerate(js['layers']):
        if layer['op_type'] == "Conv":
            if layer['use_cvi']:
                if layer['output_id'] not in ignored_output_ids:
                    layers.append(l)
        elif layer['op_type'] == "Identity":
            if len(layer['sublayers']) and layer['sublayers'][-1]['op_type'] == 'Add':
                layers.append(l)
    return layers


def get_io_ids(js, subgraph_nodes=None):
    if subgraph_nodes:
        js_subgraph = []
        for s in subgraph_nodes:
            js_subgraph.append(js['layers'][s])
    else:
        js_subgraph = js['layers']

    inputs = [l['input_id'] for l in js_subgraph]
    outputs = [l['output_id'] for l in js_subgraph]
    output_indices = [n for n, l in enumerate(js_subgraph) if l['output_id'] not in inputs]
    input_indices = [n for n, l in enumerate(js_subgraph) if l['input_id'] not in outputs]
    output_ids = [js_subgraph[i]['output_id'] for i in output_indices]
    input_ids = [js_subgraph[i]['input_id'] for i in input_indices]

    return input_ids, output_ids


def update_subgraph_biases(js_node, output_id, inputs, tmp_dir):
    biases = {}
    mse = {} 

    obuf = js_node['output_description']
    ibuf = js_node['input_description']
    name = js_node['name']
    is_unsigned = js_node['output_unsigned']

    for input in inputs:
        oname = '{}.{}.npy'.format(os.path.join(tmp_dir, os.path.basename(input)), output_id)
        onnx_name = '{}.onnx.npz'.format(os.path.join(tmp_dir, os.path.basename(input)))

        with open(oname, 'rb') as f:
            vnnx_arr = np.load(f)
        onnx_arr = np.load(onnx_name)[obuf]
        vnnx_arr = convert_array(vnnx_arr)

        if is_unsigned:
            onnx_arr = np.clip(onnx_arr, 0., 1.0)
        else:
            onnx_arr = np.clip(onnx_arr, -1.0, 1.0)

        vnnx_arr = vnnx_arr.reshape(onnx_arr.shape)

        reduce_axis = tuple((i for i in range(len(onnx_arr.shape)) if i != 1))
        onnx_mean = np.mean(onnx_arr, axis=reduce_axis)
        vnnx_mean = np.mean(vnnx_arr, axis=reduce_axis)

        if name not in biases:
            biases[name] = (onnx_mean - vnnx_mean)
            mse[name] = np.mean((onnx_arr-vnnx_arr)**2, axis=reduce_axis)
        else:
            biases[name] += (onnx_mean - vnnx_mean)
            mse[name] += np.mean((onnx_arr-vnnx_arr)**2, axis=reduce_axis)

    for key in biases:
        biases[key] = biases[key] / len(inputs)
        mse[key] = mse[key] / len(inputs)
    return biases, mse


def vnnx_run_input(args):
    import vbx.sim

    vnnx_fname, input_ids, output_id, input_shape, input_scale, input, tmp_dir = args
    inames = ['{}.{}.npy'.format(os.path.join(tmp_dir, os.path.basename(input)), i) for i in input_ids]
    oname = '{}.{}.npy'.format(os.path.join(tmp_dir, os.path.basename(input)), output_id)

    with open(vnnx_fname, 'rb') as mf:
        vnnx_model = vbx.sim.Model(mf.read())
    input_dtype = vnnx_model.input_dtypes[0]


    vnnx_input = []
    for input_id, iname in zip(input_ids, inames):
        if input_id == 0:
            vnnx_input += vnnx_load_input(input, input_shape, input_scale, input_dtype)
        else:
            with open(iname, 'rb') as f:
                vnnx_input.append(np.load(f))

    vnnx_arr = vnnx_model.run(vnnx_input)[0]
    with open(oname, 'wb') as f:
        np.save(f, vnnx_arr)


def vnnx_remove_inputs(output_id, inputs, tmp_dir):
    for input in inputs:
        oname = '{}.{}.npy'.format(os.path.join(tmp_dir, os.path.basename(input)), output_id)
        if os.path.exists(oname):
            os.remove(oname)


def run_subgraph_biases(input_ids, output_id, vnnx_fname, tmp_dir, inputs, input_shape, input_scale):
    args = [(vnnx_fname, input_ids, output_id, input_shape, input_scale, input, tmp_dir) for input in inputs]
    with Pool() as p:
        p.map(vnnx_run_input, args)


def onnx_save_activations(onnx_fname, tmp_dir, inputs, input_shape, input_scale=1./255.):
    onnx_model = onnx.load(onnx_fname)
    onnx_graph = onnx_model.graph

    for input in inputs:
        oname = '{}.onnx.npz'.format(os.path.join(tmp_dir, os.path.basename(input)))

        onnx_input = onnx_infer.load_input(input, input_scale, input_shape)
        activations = onnx_infer.onnx_activations(onnx_fname, onnx_input)
        np.savez(oname, **activations)


def onnx_remove_activations(inputs, tmp_dir):
    for input in inputs:
        oname = '{}.onnx.npz'.format(os.path.join(tmp_dir, os.path.basename(input)))
        if os.path.exists(oname):
            os.remove(oname)


def vnnx_bias_corrections(json_string, onnx_model, size_conf, io_info, output_bytes, samples_folder, samples_count, tmp_dir):
    from . import onnx_helper
    from . import json_to_graph

    js = json.loads(json_string)
    bias_correction_nodes = jslayers_to_adjust(js)

    input_scale = 1. / io_info['input_scale_factors'][0]
    input_shape =  onnx_helper.get_model_input_shape(onnx_model)
    inputs = collect_inputs(samples_folder, samples_count)
    onnx_save_activations(onnx_model, tmp_dir, inputs, input_shape, input_scale)
    vnnx_fname = os.path.join(tmp_dir, 'tmp.vnnx')

    required_inputs = {}
    deleted_nodes = []
    for bl in bias_correction_nodes:
        current_nodes = get_current_subgraph_nodes(js, bias_correction_nodes, bl)
        input_ids, output_ids = get_io_ids(js, current_nodes)
        required_inputs[bl] = input_ids

    for bl in bias_correction_nodes:
        if bl == bias_correction_nodes[0]:
            bias_corrections = None
        else:
            bias_corrections = os.path.join(tmp_dir, 'bias_corrections.json')

        current_nodes = get_current_subgraph_nodes(js, bias_correction_nodes, bl)
        input_ids, output_ids = get_io_ids(js, current_nodes)
        assert(len(output_ids) == 1)
        output_id = output_ids[0]
        js_node = js['layers'][current_nodes[-1]]

        graph_binary = json_to_graph.json_to_graph(json_string, size_conf, io_info=io_info, output_bytes=output_bytes, bias_corrections=bias_corrections, bias_correction_nodes=(current_nodes, bias_correction_nodes))
        with open(vnnx_fname, "wb") as output_file:
            output_file.write(graph_binary)

        run_subgraph_biases(input_ids, output_id, vnnx_fname, tmp_dir, inputs, input_shape, input_scale)

        biases, mse = update_subgraph_biases(js_node, output_id, inputs, tmp_dir)
        if bl == bias_correction_nodes[0]:
            bias_corrections = os.path.join(tmp_dir, 'bias_corrections.json')
            current_biases = {}
            for key in biases:
                current_biases[key] = (biases[key]).tolist()
        else:
            with open(bias_corrections) as f:
                current_biases = json.load(f)
            for key in biases:
                current_biases[key] = (biases[key]).tolist()
        with open(bias_corrections, 'w') as f:
            json.dump(current_biases, f, indent=2)

        graph_binary = json_to_graph.json_to_graph(json_string, size_conf, io_info=io_info, output_bytes=output_bytes, bias_corrections=bias_corrections, bias_correction_nodes=(current_nodes, bias_correction_nodes))
        with open(vnnx_fname, "wb") as output_file:
            output_file.write(graph_binary)

        run_subgraph_biases(input_ids, output_id, vnnx_fname, tmp_dir, inputs, input_shape, input_scale)

        # correction phase (based of channel-wise mean squared error)
        if False:
            biases, current_mse = update_subgraph_biases(js_node, output_id, inputs, tmp_dir)
            with open(bias_corrections) as f:
                current_biases = json.load(f)
            for key in biases:
                current = np.asarray(current_biases[key])
                for i,_ in enumerate(biases[key]):
                    if mse[key][i] < current_mse[key][i]:
                        biases[key][i] = 0.
                current_biases[key] = (biases[key]).tolist()
            with open(bias_corrections, 'w') as f:
                json.dump(current_biases, f, indent=2)

            graph_binary = json_to_graph.json_to_graph(json_string, size_conf, io_info=io_info, output_bytes=output_bytes, bias_corrections=bias_corrections, bias_correction_nodes=(current_nodes, bias_correction_nodes))
            with open(vnnx_fname, "wb") as output_file:
                output_file.write(graph_binary)
            run_subgraph_biases(input_ids, output_id, vnnx_fname, tmp_dir, inputs, input_shape, input_scale)

        for prev_bl in bias_correction_nodes[:bias_correction_nodes.index(bl)]:
            if prev_bl not in deleted_nodes:
                prev_nodes = get_current_subgraph_nodes(js, bias_correction_nodes, prev_bl)
                prev_input_ids, prev_output_ids = get_io_ids(js, prev_nodes)
                assert(len(prev_output_ids) == 1)
                prev_output_id = prev_output_ids[0]
                can_delete = True
                for next_bl in bias_correction_nodes[bias_correction_nodes.index(bl):]:
                    if prev_output_id in required_inputs[next_bl]:
                        can_delete = False
                        break
                if can_delete:
                    vnnx_remove_inputs(prev_output_id, inputs, tmp_dir)
                    deleted_nodes.append(prev_bl)




    #clean up
    for bl in bias_correction_nodes:
        if bl not in deleted_nodes:
            current_nodes = get_current_subgraph_nodes(js, bias_correction_nodes, bl)
            input_ids, output_ids = get_io_ids(js, current_nodes)
            assert(len(output_ids) == 1)
            output_id = output_ids[0]
            vnnx_remove_inputs(output_id, inputs, tmp_dir)

    onnx_remove_activations(inputs, tmp_dir)
    if os.path.exists(vnnx_fname):
        os.remove(vnnx_fname)
