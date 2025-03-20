import argparse
import tempfile
import os
import json
import subprocess
import numpy as np

from .split_tflite import generate_split_graphs
from .vnnx_tflite import generate_vnnx_from_json_subgraphs, get_graph_activations
from .infer_tflite import get_tflite_io


def transpose_io_to_vnnx(x):
    if len(x.shape) == 3:
        x = x.transpose((2,0,1))
    elif len(x.shape) == 4:
        x = x.transpose((0,3,1,2))
    elif len(x.shape) == 5:
        x = x.transpose((0,1,4,2,3))
    elif len(x.shape) == 6:
        x = x.transpose((0,1,2,5,3,4))
    return x


def tflite_to_vnnx(tflite, size_config, output_filename=None, start_layer=0, end_layer=None, inputs=None, mean=0., scale=1., rgb=False, debug=False,\
    vbx_version=2):
    include_io_data = 1
    if inputs is None:
        include_io_data = 0


    if debug:
        tmp_dir ='temp'
        try:
            os.mkdir(tmp_dir)
        except FileExistsError:
            pass

    else:
        tmp_dir_obj = tempfile.TemporaryDirectory()
        tmp_dir = tmp_dir_obj.name

    output_name = os.path.splitext(output_filename)[0]
    json_subgraphs, engine_graphs_nx = generate_split_graphs(tflite, tmp_dir, split_every_op=False, vbx_version=vbx_version,\
        output_name=output_name)
    if end_layer is None:
        end_layer = len(json_subgraphs)-1

    json_graph = []
    for layer_idx, j in enumerate(json_subgraphs):
        if layer_idx < start_layer:
            continue
        with open(j) as f:
            json_graph.append(json.load(f))
        if layer_idx == end_layer:
            break

    # get io and transpose for VNNX (NHWC -> CHW)
    if len(json_graph) == len(json_subgraphs): # no specifications for start layer and end layer, so skip the io of individual layers
        inputs, outputs = get_tflite_io(tflite, inputs, None, mean=mean, rgb=rgb,scale=scale)
        for i,input in enumerate(inputs):
            if debug:
                np.save(os.path.join(tmp_dir, 'tflite.input.{}.npy'.format(i)), inputs[input])
            inputs[input] = transpose_io_to_vnnx(inputs[input])
        for o,output in enumerate(outputs):
            if debug:
                np.save(os.path.join(tmp_dir, 'tflite.output.{}.npy'.format(o)), outputs[output])
            outputs[output] = transpose_io_to_vnnx(outputs[output])

        vnnx_graph_binary = generate_vnnx_from_json_subgraphs(json_graph, size_config, inputs, outputs, include_io_data, tmp_dir, engine_graphs_nx)
    else:
        graph_inputs, graph_outputs, _ = get_graph_activations(json_graph)

        input_final = dict()
        output_final = dict()
        curr_inputs = inputs
        for graph in json_subgraphs[start_layer:end_layer+1]:

            subgraph_tflite = graph.replace('.json', '.tflite')

            try:
                curr_inputs, outputs =  get_tflite_io(subgraph_tflite, curr_inputs, tmp_dir)
            except FileNotFoundError as e: # input subgraph of a multiinput graph, so its input is defined in inputs
                curr_inputs, outputs = get_tflite_io(subgraph_tflite, inputs, tmp_dir)
            for k in curr_inputs.keys():
                np.save(os.path.join(tmp_dir, 'activations.{}.npy'.format(k)), curr_inputs[k])
                if k in graph_inputs:
                    input_final[k] = curr_inputs[k]
            for k in outputs.keys():
                np.save(os.path.join(tmp_dir, 'activations.{}.npy'.format(k)), outputs[k])
                if k in graph_outputs:
                    output_final[k] = outputs[k]
            curr_inputs = outputs

        curr_inputs = input_final
        outputs = output_final
        for i in curr_inputs:
            curr_inputs[i] = transpose_io_to_vnnx(curr_inputs[i])
        for o in outputs:
            outputs[o] = transpose_io_to_vnnx(outputs[o])
        vnnx_graph_binary = generate_vnnx_from_json_subgraphs(json_graph, size_config, curr_inputs, outputs, include_io_data, tmp_dir,\
            engine_graphs_nx)

    if not debug:
        tmp_dir_obj.cleanup()

    with open(output_filename, "wb") as output_file:
        output_file.write(vnnx_graph_binary)
    hexfile = os.path.splitext(output_filename)[0]+".hex"
    subprocess.check_call(["objcopy", "-Ibinary","-Oihex", output_filename, hexfile])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tflite', help='tflite I8 model description (.tflite)', required=True)
    parser.add_argument('-c', '--size-conf', help='size configuration to build model for',
                        choices = ['V250','V500','V1000'], required=True)
    parser.add_argument('-o','--output', help="Name of vnnx output file",required=True)
    parser.add_argument('-s', '--start_layer', type=int, default=0)
    parser.add_argument('-e', '--end_layer', type=int)
    parser.add_argument('-i', '--inputs', nargs='*', help='provide test inputs for model')
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=0.)
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=1.)
    parser.add_argument('-b', '--bgr', action='store_true')
    #undocumented arguments
    parser.add_argument('-d', '--debug', help=argparse.SUPPRESS,action='store_true')
    parser.add_argument('-v', '--vbx_version', choices=[2, 3], type=int, default=2, help=argparse.SUPPRESS)
    args = parser.parse_args()

    tflite_to_vnnx(args.tflite,
                   args.size_conf,
                   output_filename=args.output,
                   start_layer=args.start_layer,
                   end_layer=args.end_layer,
                   inputs=args.inputs,
                   mean=args.mean,
                   scale=args.scale,
                   rgb=(not args.bgr),
                   debug=args.debug,
                   vbx_version=args.vbx_version)

if __name__ == "__main__":
    main()
