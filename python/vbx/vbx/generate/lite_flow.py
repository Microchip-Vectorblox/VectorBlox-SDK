import argparse
from .utils import existing_file, existing_dir
import tempfile
import os
import json
import subprocess
import numpy as np
import sys
import shutil

from .verify_tflite import verify_graph
from .transform_tflite import load_graph, save_graph, transform_graph
from .split_tflite import generate_split_graphs
from .infer_tflite import get_tflite_io
from .vnnx_tflite import generate_vnnx_from_json_subgraphs, get_graph_activations
from .utils import existing_file, existing_dir, json_load, json_dump, get_input_details, get_output_details
from .vnnx_types import preset_select

default_ncomp_passes = ['SORT_DFS',
                      'REMOVE_FP32_IO',
                      'REMOVE_CONSTANTS',
                      'CLEAN_LOGISTIC',
                      'LUT',
                      'PADV2',
                      'EXPLICIT_PAD',
                      'SHARED_PAD',
                      'GROUP_DEPTH5x2',
                      'STRIDED_DEPTHWISE',
                      'TRANSPOSE_CONV',
                      'FC_CONV_2D',
                      'GROUP_CONV',
                      ]

default_comp_passes = ['SORT_DFS',
                      'REMOVE_FP32_IO',
                      'REMOVE_CONSTANTS',
                      'CLEAN_LOGISTIC',
                      'LUT',
                      'PADV2',
                      'EXPLICIT_PAD',
                      'SHARED_PAD',
                      'GROUP_DEPTH',
                      'STRIDED_DEPTHWISE',
                      'TRANSPOSE_CONV',
                      'FC_CONV_2D',
                      'GROUP_CONV',
                      ]

default_ucomp_passes = ['LUT',
                       'AVERAGE_POOL_2D',
                       'FULL_DEPTH',
                       ]


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
    compression_vbx=None, tmp_dir=None, tmp_dir_obj=None):
    include_io_data = 1
    if inputs is None:
        include_io_data = 0


    output_name = os.path.splitext(output_filename)[0]
    json_subgraphs, engine_graphs_nx = generate_split_graphs(tflite, tmp_dir,
                                                             split_every_op=False,
                                                             compression_vbx=compression_vbx,
                                                             size_config=size_config,
                                                             output_name=output_name)
    if end_layer is None:
        end_layer = len(json_subgraphs)-1

    json_graph = []
    for layer_idx, j in enumerate(json_subgraphs):
        if layer_idx < start_layer:
            continue
        json_graph.append(json_load(j))
        if layer_idx == end_layer:
            break

    # get io and transpose for VNNX (NHWC -> CHW)
    if len(json_graph) == len(json_subgraphs): # no specifications for start layer and end layer, so skip the io of individual layers
        inputs, outputs = get_tflite_io(tflite, inputs, None, mean=mean, rgb=rgb,scale=scale)

        if debug:
            data = {'inputs': [], 'outputs': []}
            import tensorflow as tf
            interpreter= tf.lite.Interpreter(model_path=tflite)
            input_details, output_details = get_input_details(tflite), get_output_details(tflite)

        for i,input in enumerate(inputs):
            if debug:
                np.save(os.path.join(tmp_dir, 'tflite.input.{}.npy'.format(i)), inputs[input])
                scale,zero = input_details[i].get('quantization', (0.0, 0))
                data['inputs'].append({'data': inputs[input].tolist(), 'shape': inputs[input].shape, 'zero': zero, 'scale': scale, 'dtype': inputs[input].dtype.name.upper()})


            inputs[input] = transpose_io_to_vnnx(inputs[input])
        for o,output in enumerate(outputs):
            if debug:
                np.save(os.path.join(tmp_dir, 'tflite.output.{}.npy'.format(o)), outputs[output])
                scale,zero = output_details[o].get('quantization', (0.0, 0))
                data['outputs'].append({'data': outputs[output].tolist(), 'shape': outputs[output].shape, 'zero': zero, 'scale': scale, 'dtype': outputs[output].dtype.name.upper()})
            outputs[output] = transpose_io_to_vnnx(outputs[output])
        if debug:
            json_dump(data, os.path.join(tmp_dir, 'tflite.io.json'))

        vnnx_graph_binary = generate_vnnx_from_json_subgraphs(json_graph, size_config, inputs, outputs, include_io_data, tmp_dir, engine_graphs_nx, debug, \
                                                              compression_vbx=compression_vbx, tmp_dir_obj=tmp_dir_obj if not debug else None)
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
            engine_graphs_nx, debug, compression_vbx=compression_vbx, tmp_dir_obj=tmp_dir_obj if not debug else None)

    with open(output_filename, "wb") as output_file:
        output_file.write(vnnx_graph_binary)
    hexfile = os.path.splitext(output_filename)[0]+".hex"
    subprocess.check_call(["objcopy", "-Ibinary","-Oihex", output_filename, hexfile])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tflite', help='tflite I8 model description (.tflite)', required=True, type=existing_file)
    parser.add_argument('-s', '--size_conf', help='size configuration to build model for',
                        choices = ['V250','V500','V1000'], required=True)
    parser.add_argument('-c', '--compression-vbx', help='compression setting for VNNX model generation',
                        choices = ['ncomp', 'comp', 'ucomp'], required=True)
    parser.add_argument('-o','--output', help="Name of vnnx output file")
    parser.add_argument('--start_layer', type=int, default=0)
    parser.add_argument('-e', '--end_layer', type=int)
    parser.add_argument('-i', '--inputs', nargs='*', help='provide test inputs for model', type=existing_file)
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=0.)
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=1.)
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-u', '--uint8', action='store_true')
    #undocumented arguments
    parser.add_argument('-d', '--debug', help=argparse.SUPPRESS,action='store_true')
    
    args = parser.parse_args()
    
    env = os.environ.copy()
    VBX_SDK = env["VBX_SDK"]
    NX_SDK = env["NX_SDK"]

    if args.debug:
        tmp_dir ='temp'
        try:
            os.mkdir(tmp_dir)
        except FileExistsError:
            pass
    else:
        tmp_dir_obj = tempfile.TemporaryDirectory()
        tmp_dir = tmp_dir_obj.name

    if args.compression_vbx == 'ucomp':
        # Also update presets
        for preset in preset_select["SCRATCHPAD_KB"]:
            preset_select["SCRATCHPAD_KB"][preset] = 32
        for preset in preset_select["VECTOR_LANES"]:
            preset_select["VECTOR_LANES"][preset] = 2    
        model_jname = (os.path.basename(args.tflite)).replace('.tflite', '.json')
        
    graph, dir_obj = load_graph(args.tflite)

    default_passes = []
    if args.compression_vbx == 'ucomp':
        default_passes = default_ucomp_passes
        core = 'MXP'
        accel = 'TSNP'
    elif args.compression_vbx == 'comp':
        default_passes = default_comp_passes
        core = 'MXP'
        accel = 'FIA'
    else:
        default_passes = default_ncomp_passes
        core = 'MXP'
        accel = 'FIA'
    
    output_filename = args.output
    if not args.output:
        base_name = args.tflite.rsplit('.',1)[0]
        if args.compression_vbx == 'ucomp':
            output_filename = os.path.join(tmp_dir, base_name + ".vnnx")
        else:
            output_filename = base_name + "_" + args.size_conf + "_" + args.compression_vbx + ".vnnx"
    elif args.compression_vbx == 'ucomp':
        output_filename=os.path.join(tmp_dir,args.output.replace(".ucomp",".vnnx"))
    

    if verify_graph(graph, core, accel, False, transformed=False):  #Catch very high level errors
        optimized_graph = transform_graph(graph, default_passes, False)
        optimized_tflite = save_graph(args.tflite.replace('.tflite', '.tr.tflite'), dir_obj, optimized_graph, copy=args.debug)

        if verify_graph(optimized_graph, core, accel, False, transformed=True): #Catch detailed errors after optimization
            tflite_to_vnnx(optimized_tflite, #args.tflite
                        args.size_conf,
                        output_filename=output_filename,
                        start_layer=args.start_layer,
                        end_layer=args.end_layer,
                        inputs=args.inputs,
                        mean=args.mean,
                        scale=args.scale,
                        rgb=(not args.bgr),
                        debug=args.debug,
                        compression_vbx=args.compression_vbx,
                        tmp_dir=tmp_dir,
                        tmp_dir_obj=tmp_dir_obj if not args.debug else None)
            
            if args.compression_vbx == 'ucomp':
                model_jname = model_jname.replace('.json','.tr.json')

    if not dir_obj is None:
        dir_obj.cleanup()
    
    if args.compression_vbx == 'ucomp':
        prev_dir = os.getcwd()

        debug_flag = False
        if args.debug:
            tmp_dir = os.path.join(prev_dir, tmp_dir)
            debug_flag = True
        uint8_flag = False
        if args.uint8:
            uint8_flag = True

        # Running Neuronix compiler and generating combine binary file for VBX3.0
        os.chdir(NX_SDK)
        
        if isinstance(args.mean, float):
            mean = str(args.mean)
        else:
            mean = ", ".join(str(item) for item in args.mean)
        
        if isinstance(args.scale, float):
            scale = str(args.scale)
        else:
            scale = ", ".join(str(item) for item in args.scale)
        
        nx_compiler_cmd = [
            "python", "compiler/tflite_compiler.py",
            f"-m={os.path.join(tmp_dir, 'nx_engine', model_jname)}",
            f"-o={tmp_dir}",
            f"-u={uint8_flag}",
            f"-d={debug_flag}",
            f"--mean={mean}",
            f"--scale={scale}",
        ]
        result = subprocess.Popen(nx_compiler_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in result.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        result.wait()
        assert result.returncode == 0
        
        os.chdir(prev_dir)
        shutil.copy2(output_filename, prev_dir)
        ucomp_sim_file = output_filename.replace('.vnnx', '_numsim.nxir')
        shutil.copy2(ucomp_sim_file, prev_dir)
        ddr_info_txt_file = output_filename.replace('.vnnx', '_ddr_info.txt')
        shutil.copy2(ddr_info_txt_file, prev_dir)
        nx_bin_filename = output_filename.replace('.vnnx', '_ddr_content.bin')
        ucomp_filename = os.path.basename(output_filename.replace('.vnnx', '.ucomp'))
        generate_model_cmd = [
            "python", f"{VBX_SDK}/python/vbx/vbx/generate/generate_vbx3_model.py",
            "-v", output_filename, "-n", nx_bin_filename, "-o", ucomp_filename
        ]
        result = subprocess.Popen(generate_model_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in result.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        result.wait()
        assert result.returncode == 0

    if not args.debug:
        tmp_dir_obj.cleanup() 

if __name__ == "__main__":
    main()
