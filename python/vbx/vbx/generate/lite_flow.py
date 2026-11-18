import argparse
from .utils import existing_file, existing_dir
import tempfile
import os
import json
import subprocess
import numpy as np
import sys
import shutil
import signal

from .verify_tflite import verify_graph
from .transform_tflite import load_graph, save_graph, transform_graph, tflite2json, transpose_io_to_vnnx
from .split_tflite import generate_split_graphs
from .infer_tflite import get_tflite_io
from .vnnx_tflite import generate_vnnx_from_json_subgraphs
from .utils import existing_file, existing_dir, json_load, json_dump, get_input_details, get_output_details
from .vnnx_types import preset_select

default_ncomp_passes = ['SORT_DFS',
                      'REMOVE_FP32_IO',
                      'REMOVE_CONSTANTS',
                      'CLEAN_LOGISTIC',
                      'LUT',
                      'FUSE_GROUP1',
                      'PADV2',
                      'EXPLICIT_PAD',
                      'SHARED_PAD',
                      'STRIDED_DEPTHWISE',
                      'TRANSPOSE_CONV',
                    #   'FC_CONV_2D',
                      'GROUP_CONV',
                      'POOL_AS_MEAN',
                      'REMOVE_NOPS', 
                      'SORT_DFS', 
                      ]

default_comp_passes = ['SORT_DFS',
                      'REMOVE_FP32_IO',
                      'REMOVE_CONSTANTS',
                      'CLEAN_LOGISTIC',
                      'LUT',
                      'FUSE_GROUP1',
                      'PADV2',
                      'EXPLICIT_PAD',
                      'SHARED_PAD',
                      'STRIDED_DEPTHWISE',
                      'TRANSPOSE_CONV',
                      'FC_CONV_2D',
                      'GROUP_CONV',
                      'POOL_AS_MEAN',
                      'REMOVE_NOPS', 
                      'SORT_DFS', 
                      ]

default_ucomp_passes = ['REMOVE_FP32_IO',
                      'REMOVE_CONSTANTS',
                      'CLEAN_LOGISTIC',
                      'LUT',
                      'CONCAT_PAD',
                      'NORMALIZE_ADD_INPUTS',
                      'AVERAGE_POOL_2D',
                      'POOL_AS_MEAN',
                      'MAXPOOL_INPUT_CHANNEL_SPLIT',
                      'FULL_DEPTH',
                      'STRIDED_DEPTHWISE',
                      'GROUP_CONV',
                      'REMOVE_NOPS',
                      ]


def tflite_to_vnnx(tflite, size_config, output_filename=None, inputs=None, mean=0., scale=1., rgb=False, debug=False,\
    compression_vbx=None, tmp_dir=None, tmp_dir_obj=None):
    include_io_data = 1
    if inputs is None:
        include_io_data = 0


    output_name = os.path.splitext(output_filename)[0]
    initial_splits, engine_graphs_nx = generate_split_graphs(tflite, tmp_dir,
                                                             split_every_op=False,
                                                             compression_vbx=compression_vbx,
                                                             size_config=size_config,
                                                             output_name=output_name)

    json_graph = json_load(os.path.join(tmp_dir, os.path.basename(tflite).replace('.tflite','.json')))

    # get io and transpose for VNNX (NHWC -> CHW)
    if compression_vbx == "ucomp":
        inputs, outputs, activations = get_tflite_io(tflite, inputs, None, mean=mean, rgb=rgb,scale=scale, all_tensors=False)
    else:
        _, outputs, _ = get_tflite_io(tflite, inputs, None, mean=mean, rgb=rgb,scale=scale, all_tensors=False)
        inputs, _, activations = get_tflite_io(tflite, inputs, None, mean=mean, rgb=rgb,scale=scale, all_tensors=True)

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

    vnnx_graph_binary = generate_vnnx_from_json_subgraphs(json_graph, initial_splits, activations, size_config, inputs, outputs, include_io_data, tmp_dir, engine_graphs_nx, debug, \
                                                              compression_vbx=compression_vbx, tmp_dir_obj=tmp_dir_obj if not debug else None)

    with open(output_filename, "wb") as output_file:
        output_file.write(vnnx_graph_binary)
    hexfile = os.path.splitext(output_filename)[0]+".hex"
    subprocess.check_call(["objcopy", "-Ibinary","-Oihex", output_filename, hexfile])


def run_and_stream(cmd):
    """Run cmd, streaming its combined stdout/stderr live, and raise a descriptive
    RuntimeError on failure (including if it was killed by a signal) instead of a bare
    `assert returncode == 0`, which is silently stripped under `python -O` and, even when
    active, gives no indication of what failed or why after a long-running compile."""
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in result.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    result.wait()
    if result.returncode != 0:
        if result.returncode < 0:
            sig = signal.Signals(-result.returncode)
            reason = f"killed by signal {sig.name} ({-result.returncode})"
        else:
            reason = f"exited with code {result.returncode}"
        raise RuntimeError(f"Command {reason} in {os.getcwd()}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tflite', help='tflite I8 model description (.tflite)', required=True, type=existing_file)
    parser.add_argument('-s', '--size_conf', help='size configuration to build model for',
                        choices = ['V250','V500','V1000'], required=True)
    parser.add_argument('-c', '--compression-vbx', help='compression setting for VNNX model generation',
                        choices = ['ncomp', 'comp', 'ucomp'], required=True)
    parser.add_argument('-o','--output', help="Name of vnnx output file")
    parser.add_argument('-i', '--inputs', nargs='*', help='provide test inputs for model', type=existing_file)
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=0.)
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=1.)
    parser.add_argument('-b', '--bgr', action='store_true')
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
            output_filename = base_name + "_" + args.size_conf + ".vnnx"
        else:
            output_filename = base_name + "_" + args.size_conf + "_" + args.compression_vbx + ".vnnx"
    elif args.compression_vbx == 'ucomp':
        output_filename = args.output.replace(".ucomp",".vnnx")
    
    # batch is constant and 1 accross all inputs, so we can optimize the graph and verify it before generating the vnnx model
    # all ops in graph must have constant batch 2) 1st dim assumed to be batch

    # dir_obj comes from load_graph(). It owns the temporary copy of the TFLite model and its JSON graph.
    # We clean it up after either verification failure if it exists.
    # tmp_dir_obj owns the conversion workspace created when --debug is off. We clean it up only in that case

    if not verify_graph(graph, core, accel, False, transformed=False):
        if dir_obj is not None:
            dir_obj.cleanup()
        if not args.debug:
            tmp_dir_obj.cleanup()
        raise SystemExit(1)
    optimized_graph = transform_graph(graph, default_passes, False, args.compression_vbx)
    if not verify_graph(optimized_graph, core, accel, False, transformed=True):
        if dir_obj is not None:
            dir_obj.cleanup()
        if not args.debug:
            tmp_dir_obj.cleanup()
        raise SystemExit(1)
    optimized_tflite = save_graph(args.tflite.replace('.tflite', '.tr.tflite'), dir_obj, optimized_graph, copy=args.debug)

    tflite_to_vnnx(optimized_tflite,
                  args.size_conf,
                  output_filename=output_filename,
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

        # Running Neuronix compiler and generating combine binary file for VBX3.0
        os.chdir(NX_SDK)

        nx_compiler_cmd = [
            "python", "compiler/tflite_compiler.py",
            f"-m={os.path.join(tmp_dir, 'nx_engine', model_jname)}",
            f"-s={args.size_conf}",
            f"-o={output_filename.replace('.vnnx','')}",
            f"--tmp_dir={tmp_dir}",
            f"-d={debug_flag}",
        ]
        run_and_stream(nx_compiler_cmd)

        os.chdir(prev_dir)
        vnnx_filename = os.path.join(tmp_dir, output_filename)
        ucomp_sim_file = vnnx_filename.replace('.vnnx', '_numsim.nxir')
        shutil.copy2(ucomp_sim_file, prev_dir)
        ddr_info_txt_file = vnnx_filename.replace('.vnnx', '_ddr_info.txt')
        shutil.copy2(ddr_info_txt_file, prev_dir)
        nx_bin_filename = vnnx_filename.replace('.vnnx', '_ddr_content.bin')
        ucomp_filename = os.path.basename(output_filename.replace('.vnnx', '.ucomp'))
        generate_model_cmd = [
            "python", f"{VBX_SDK}/python/vbx/vbx/generate/generate_vbx3_model.py",
            "-v", output_filename, "-n", nx_bin_filename, "-o", ucomp_filename
        ]
        run_and_stream(generate_model_cmd)

    if not args.debug:
        tmp_dir_obj.cleanup() 

if __name__ == "__main__":
    main()
