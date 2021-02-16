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


def vnnx_load_image(image, channels, input_shape):
    img = cv2.imread(image)
    ishape = tuple([input_shape[0], input_shape[1], channels])
    if img.shape != ishape:
        img = cv2.resize(img, input_shape).clip(0, 255)
    flattened = img.swapaxes(1, 2).swapaxes(0, 1).flatten()
    return [flattened]


def collect_images(samples_folder, samples_count):
    images = []
    extensions = ['*.jpg', '*.png', '*.jpeg']
    extensions += [e.upper() for e in extensions]
    for ext in extensions:
        images += sorted(glob.glob(os.path.join(samples_folder, ext)))
    if samples_count:
        images = images[:samples_count]
    return images


def jslayers_to_adjust(js):
    layers = []

    for l, layer in enumerate(js['layers']):
        if layer['op_type'] == "Conv":
            if layer['use_cvi']:
                if layer['m'] > 1 and layer['n'] > 1:
                    layers.append(l)
        elif layer['op_type'] == "Identity":
            if len(layer['sublayers']) and layer['sublayers'][-1]['op_type'] == 'Add':
                layers.append(l)

    return layers


def get_io_ids(js, subgraph_nodes):
    js_subgraph = []
    for s in subgraph_nodes:
        js_subgraph.append(js['layers'][s])

    inputs = [l['input_id'] for l in js_subgraph]
    outputs = [l['output_id'] for l in js_subgraph]
    output_indices = [n for n, l in enumerate(js_subgraph) if l['output_id'] not in inputs]
    input_indices = [n for n, l in enumerate(js_subgraph) if l['input_id'] not in outputs]
    output_ids = [js_subgraph[i]['output_id'] for i in output_indices]
    input_ids = [js_subgraph[i]['input_id'] for i in input_indices]

    return input_ids, output_ids


def update_subgraph_biases(js_node, output_id, images, tmp_dir):
    biases = {}

    obuf = js_node['output_description']
    ibuf = js_node['input_description']
    name = js_node['name']
    is_unsigned = js_node['output_unsigned']

    for image in images:
        oname = '{}.{}.npy'.format(os.path.join(tmp_dir, os.path.basename(image)), output_id)
        onnx_name = '{}.onnx.npz'.format(os.path.join(tmp_dir, os.path.basename(image)))

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
        else:
            biases[name] += (onnx_mean - vnnx_mean)

    for key in biases:
        biases[key] = biases[key] / len(images)
    return biases


def vnnx_run_image(args):
    import vbx.sim

    vnnx_fname, input_ids, output_id, input_shape, image, tmp_dir = args
    inames = ['{}.{}.npy'.format(os.path.join(tmp_dir, os.path.basename(image)), i) for i in input_ids]
    oname = '{}.{}.npy'.format(os.path.join(tmp_dir, os.path.basename(image)), output_id)

    vnnx_input = []
    for input_id, iname in zip(input_ids, inames):
        if input_id == 0:
            input_channels = input_shape[0]
            input_height = input_shape[1]
            input_width = input_shape[2]
            vnnx_input += vnnx_load_image(image, input_channels, (input_height, input_width))
        else:
            with open(iname, 'rb') as f:
                vnnx_input.append(np.load(f))

    with open(vnnx_fname, 'rb') as mf:
        vnnx_model = vbx.sim.Model(mf.read())
    vnnx_arr = vnnx_model.run(vnnx_input)[0]
    with open(oname, 'wb') as f:
        np.save(f, vnnx_arr)


def run_subgraph_biases(input_ids, output_id, vnnx_fname, input_shape, images, tmp_dir):
    args = [(vnnx_fname, input_ids, output_id, input_shape, image, tmp_dir) for image in images]
    with Pool() as p:
        p.map(vnnx_run_image, args)
    p.close()
    p.join()


def onnx_save_activations(onnx_fname, input_shape, images, tmp_dir):
    from . import onnx_infer

    onnx_model = onnx.load(onnx_fname)
    onnx_graph = onnx_model.graph

    input_channels = input_shape[0]
    input_height = input_shape[1]
    input_width = input_shape[2]

    for image in tqdm.tqdm(images):
        oname = '{}.onnx.npz'.format(os.path.join(tmp_dir, os.path.basename(image)))

        onnx_input = onnx_infer.load_image(image, 1.0, input_channels, input_shape=(input_height, input_width))
        activations = onnx_infer.onnx_activations(onnx_fname, onnx_input)
        np.savez(oname, **activations)


def vnnx_bias_corrections(json_string, onnx_model, size_conf, io_info, output_bytes, samples_folder, samples_count, tmp_dir):
    from . import onnx_helper
    from . import json_to_graph

    js = json.loads(json_string)
    bias_correction_nodes = jslayers_to_adjust(js)

    input_shape =  onnx_helper.get_model_input_shape(onnx_model)
    images = collect_images(samples_folder, samples_count)
    onnx_save_activations(onnx_model, input_shape, images, tmp_dir)

    vnnx_fname = os.path.join(tmp_dir, 'tmp.vnnx')

    print('Correcting biases...')
    for bl in tqdm.tqdm(bias_correction_nodes):
        if bl == bias_correction_nodes[0]:
            bias_corrections = None
        else:
            bias_corrections = 'biases_correction.json'

        current_nodes = get_current_subgraph_nodes(js, bias_correction_nodes, bl)
        input_ids, output_ids = get_io_ids(js, current_nodes)
        assert(len(output_ids) == 1)
        output_id = output_ids[0]
        js_node = js['layers'][current_nodes[-1]]

        with open('subgraph.json', 'w') as f:
            json.dump({'input_ids': input_ids, 'output_id': output_id,
                'vnnx_fname': vnnx_fname, 'input_shape': input_shape,
                'images': images, 'tmp_dir': tmp_dir}, f)

        graph_binary = json_to_graph.json_to_graph(json_string, size_conf, io_info=io_info, output_bytes=output_bytes, bias_corrections=bias_corrections, bias_correction_nodes=(current_nodes, bias_correction_nodes))
        with open(vnnx_fname, "wb") as output_file:
            output_file.write(graph_binary)

        run_subgraph_biases(input_ids, output_id, vnnx_fname, input_shape, images, tmp_dir)

        biases = update_subgraph_biases(js_node, output_id, images, tmp_dir)
        if bl == bias_correction_nodes[0]:
            current_biases = {}
            for key in biases:
                current_biases[key] = (biases[key]).tolist()
        else:
            with open('biases_correction.json') as f:
                current_biases = json.load(f)
            for key in biases:
                current_biases[key] = (biases[key]).tolist()
        with open('biases_correction.json', 'w') as f:
            json.dump(current_biases, f, indent=2)

        bias_corrections = 'biases_correction.json'
        graph_binary = json_to_graph.json_to_graph(json_string, size_conf, io_info=io_info, output_bytes=output_bytes, bias_corrections=bias_corrections, bias_correction_nodes=(current_nodes, bias_correction_nodes))
        with open(vnnx_fname, "wb") as output_file:
            output_file.write(graph_binary)

        run_subgraph_biases(input_ids, output_id, vnnx_fname, input_shape, images, tmp_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json')
    args = parser.parse_args()
    with open(args.json) as f:
        j = json.load(f)
        run_subgraph_biases(**j)

if __name__ == "__main__":
    main()
