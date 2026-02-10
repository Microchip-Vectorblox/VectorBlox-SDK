import sys
import json
sys.path.append('.')
import common.internal_representation as internal_representation
from frontend import compile_frontend_tflite
from backend import compile_backend
from file_writer import create_model_files,create_ddr_entries_from_ir
from common.debug_flags import DEBUG_REMOVE_ENGINE_COMMAND, DEBUG_FORCE_IC_SPLIT, DEBUG_CREATE_ORDERING_CONV,\
                                DEBUG_FORCE_DEEP_CONV, DEBUG_GENERATE_PERFORMANCE_REPORT,\
                                DEBUG_FORCE_FOLDCONV, DEBUG_GENERATE_AMM_ALLOCATION_REPORT
from common.hw_config import MAX_WLOC_128BIT_ENTRIES
import os
import reports
from common.utils import bcolors
import sys
import argparse
import shutil
import subprocess

def compile_model(ir:internal_representation.IR, json_fname: str, debug_output_dir:str, compiler_output_dir:str) -> internal_representation.IR:

    print('Loading TFLite model: %s' % json_fname)
    with open(json_fname, 'r') as json_file:
        graph = json.load(json_file)

    # In the same directory as the graph.json, search for a file called mxp_op_idx.txt
    # If it exists, also parse the list of op indices on the MXP
    #directory = os.path.dirname(tflite_fname)
    mxp_ops_file = os.path.join(compiler_output_dir, 'nx_engine/mxp_op_idx.txt')
    mxp_ops = []
    if os.path.exists(mxp_ops_file) and os.path.isfile(mxp_ops_file):
        print('Reading: %s' % mxp_ops_file)
        with open(mxp_ops_file, 'r') as file:
            mxp_ops = [int(line.strip()) for line in file.readlines()]
    
    # In the same directory as the graph.json, search for a file called vnnx_io_offsets.txt
    # If it exists, also parse the io
    model_io_file = os.path.join(compiler_output_dir, 'nx_engine/vnnx_io_offsets.txt')
    model_io_id = []
    model_io_offset = []
    io_tensor_names = []
    if os.path.exists(model_io_file) and os.path.isfile(model_io_file):
        print('Reading: %s' % model_io_file)
        output_offset = 0
        with open(model_io_file, 'r') as file:
            line_count = 0
            for line in file:
                line_count += 1
                # Line 1 is the table heading
                if line_count == 2:
                    # Parse the input
                    entry = line.strip().split('\t')
                    model_io_id.append(int(entry[2]))
                    model_io_offset.append(int(0))
                    io_tensor_names.append(f"T{int(entry[2])}")
                elif line_count > 2:
                    # Parse the outputs
                    entry = line.strip().split('\t')
                    model_io_id.append(int(entry[2]))
                    model_io_offset.append(int(output_offset))
                    io_tensor_names.append(f"T{int(entry[2])}")
                    output_offset += int(entry[3])
        ir.io_tensor_names = io_tensor_names

    # Similarly, search for a map from tensor index to address offset.
    # The base address is the VNNX model, e.g., 0x30_0010_0000
    mxp_tensor_file = os.path.join(compiler_output_dir, 'nx_engine/mxp_tensor_offset_map.txt')
    mxp_tensor_to_offset = {}
    mxp_offset_to_tensor = {}
    mxp_tensors_base = None
    mxp_tensors_size = None
    if os.path.exists(mxp_tensor_file) and os.path.isfile(mxp_tensor_file):
        ir.sync_with_MXP = True
        print('Reading: %s' % mxp_tensor_file)
        with open(mxp_tensor_file, 'r') as file:
            line_count = 0
            for line in file:
                line_count += 1
                # Line 1 is tensor region offset
                if line_count == 1:
                    mxp_tensors_base = int(line.strip().split()[-1])
                    continue
                # Line 1 is tensor region size
                if line_count == 2:
                    mxp_tensors_size = int(line.strip().split()[-1])
                    continue
                # Line 3 is empty, line 4 is the table heading
                if line_count < 5:
                    continue
                # Parse the remaining lines
                entry = line.strip().split('\t')
                # If this is a duplicate tensor from an identity on MXP, NX can ignore it since
                # that tensor should only be part of internal MXP operations. Tensors from
                # injected identity have 0.1 added to their ID.
                tensor_id = float(entry[0])
                if tensor_id != int(tensor_id):
                    continue
                tensor_id = int(tensor_id)
                offset = int(entry[1])

                # Assert every tensor has a single offset
                if tensor_id in mxp_tensor_to_offset:
                    assert offset == mxp_tensor_to_offset[tensor_id]
                else:
                    # if tensor_id in model_io_id:
                    #     mxp_tensor_to_offset[tensor_id] = model_io_offset[model_io_id.index(tensor_id)]
                    # else:
                    #     mxp_tensor_to_offset[tensor_id] = offset
                    mxp_tensor_to_offset[tensor_id] = offset
                # Assert no 2 tensors have the same offset
                #if offset in mxp_offset_to_tensor:
                #    assert tensor_id == mxp_offset_to_tensor[offset]
                #else:
                #    mxp_offset_to_tensor[offset] = tensor_id

    ir = compile_frontend_tflite(ir, graph, mxp_ops, mxp_tensor_to_offset)
    backend_augmented_ir = compile_backend(ir, debug_output_dir)

    return backend_augmented_ir

def main():
    parser = argparse.ArgumentParser(description='VBX 3.0 compiler')
    parser.add_argument('-m','--model', default=None, help='Quantized TFLite model file', required=True)
    parser.add_argument('-o','--output_dir', default=None, help='Output directory for compiler results.', required=True)
    parser.add_argument('-u', '--uint8', default=False, help='Enable uint8 to int8 conversion.')
    parser.add_argument('-d', '--debug', default=False, help='Writing debug files')
    parser.add_argument('--mean', nargs='+', default=0.)
    parser.add_argument('--scale', nargs='+', default=1.)
    global args
    args = parser.parse_args()

    if ('.json' not in args.model):
        print("Please run vnnx_compile command to generate the compiler files")
        sys.exit(0)

    if isinstance(args.mean, float):
        mean = args.mean
        scale = args.scale
    else:
        mean = [float(value) for value in args.mean[0].split(',')]
        scale = [float(value) for value in args.scale[0].split(',')]

    if DEBUG_FORCE_DEEP_CONV or DEBUG_FORCE_FOLDCONV or DEBUG_FORCE_IC_SPLIT:
        print(bcolors.WARNING+'Not currently supporting FORCE_IC_SPLIT, FORCE_DEEP_CONV, or FORCE_FOLDCONV'+bcolors.ENDC)
        sys.exit(0)
    if MAX_WLOC_128BIT_ENTRIES == 10:
        print(bcolors.WARNING+'Not currently supporting MAX_WLOC_128BIT_ENTRIES == 10'+bcolors.ENDC)
        sys.exit(0)

    # Make directories and create IR object
    model_name = os.path.basename(args.model).rsplit('.',2)[0]
    if DEBUG_CREATE_ORDERING_CONV:
        model_name=model_name+'_with_reordering'
    compiler_output_dir = args.output_dir
    if not os.path.exists(compiler_output_dir):
        os.makedirs(compiler_output_dir)
    if DEBUG_REMOVE_ENGINE_COMMAND:
        model_name+='_noengine'

    ir = internal_representation.IR(model_name,compiler_output_dir=compiler_output_dir,uint8_flag=args.uint8, debug=args.debug, mean=mean, scale=scale)
    debug_output_dir = os.path.join(compiler_output_dir, 'debug/')
    if not os.path.exists(debug_output_dir):
        os.makedirs(debug_output_dir)

    # This converts the TFLite model to nx internal representation including backend data.
    ir = compile_model(ir, args.model, debug_output_dir, compiler_output_dir)
    
    # TODO: The rest below is same as compiler.py, can refactor to a common file.
    if ir.debug:
        print('Writing sequencer debug and CSV files')
        sequencer_program = ir.sequencer_program
        sequencer_program_debug_filename =  os.path.join(debug_output_dir, model_name+'_sequencer_program_debug_info.txt')
        sequencer_program.save_sequencer_program_debug_file(sequencer_program_debug_filename)
        sequencer_program_csv_filename =  os.path.join(compiler_output_dir, model_name+'_sequencer_program.csv')
        sequencer_program.save_sequencer_program_csv_file(sequencer_program_csv_filename)
    print('Writing IR files')
    numsim_ir_filename = os.path.join(compiler_output_dir, model_name+'_numsim.nxir')
    ir.save_numsim_ir(numsim_ir_filename)
    print('Writing compiler output files to directory: %s' % compiler_output_dir)
    create_model_files(compiler_output_dir,model_name,ir) # This takes all ddr entries and creates a ddr file and model config file which are needed to run model on device.
    if DEBUG_GENERATE_PERFORMANCE_REPORT and ir.debug:
        performance_report_filename = os.path.join(compiler_output_dir, model_name+'_perf')
        reports.generate_performance_report(performance_report_filename,ir)
        wloc_balancing_report_filename = os.path.join(compiler_output_dir, model_name+'_wloc_balancing')
        reports.generate_wloc_balancing_report(wloc_balancing_report_filename,ir)
    if DEBUG_GENERATE_AMM_ALLOCATION_REPORT and ir.debug:
        amm_allocation_report_filename = os.path.join(compiler_output_dir, model_name+'_ammalloc')
        reports.generate_ammalloc_report(amm_allocation_report_filename,ir)
    if ir.debug:
        shutil.copyfile('common/debug_flags.py', os.path.join(compiler_output_dir, 'debug_flags.py'))

if __name__ == "__main__":
    main()
