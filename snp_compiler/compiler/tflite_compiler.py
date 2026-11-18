import sys
import json
sys.path.append('.')
import common.internal_representation as internal_representation
import tflite_parser
import frontend_hardware_agnostic
import frontend_hardware_dependent
from file_writer import create_model_files,create_ddr_entries_from_ir, create_ddr_entries_from_ir_alex, create_sequencer_ddr_entry_from_ir
from common.debug_flags import DEBUG_REMOVE_ENGINE_COMMAND, DEBUG_GENERATE_PERFORMANCE_REPORT,\
                                DEBUG_GENERATE_AMM_ALLOCATION_REPORT, DEBUG_SIMULATE_CONCAT_REQUANT
from common.hw_config import MAX_WLOC_128BIT_ENTRIES, OpType
import os
import reports
from common.utils import bcolors
import argparse
import shutil
import program_compiler
from sequencer import split_sequencer_program

def load_graph(json_fname: str) -> dict:
    print('Loading TFLite model: %s' % json_fname)
    with open(json_fname, 'r') as json_file:
        graph = json.load(json_file)
    return graph

def parse_mxp_ops(compiler_output_dir: str) -> list:
    mxp_ops_file = os.path.join(compiler_output_dir, 'nx_engine/mxp_op_idx.txt')
    mxp_ops = []
    if os.path.isfile(mxp_ops_file):
        print('Reading: %s' % mxp_ops_file)
        with open(mxp_ops_file, 'r') as file:
            mxp_ops = [int(line.strip()) for line in file.readlines()]
    return mxp_ops

def parse_mxp_sync_order(compiler_output_dir: str, ir: internal_representation.IR):
    # Read the NX->MXP sync wait order emitted by the vnnx generator (write_mxp_sync_order in
    # vnnx_tflite.py) and stash the ordered NX tensor ids on the IR so save_numsim_ir persists
    # them into the numsim .nxir. The numeric simulator then reads ir.mxp_sync_order instead of
    # re-opening this .txt file. Leaves ir.mxp_sync_order as None when the file is absent (e.g.
    # a model with no MXP syncing).
    sync_order_file = os.path.join(compiler_output_dir, 'nx_engine/mxp_sync_order.txt')
    if os.path.isfile(sync_order_file):
        print('Reading: %s' % sync_order_file)
        sync_order_ids = []
        with open(sync_order_file, 'r') as file:
            next(file)  # skip header (order\tid\tname)
            for line in file:
                cols = line.strip().split('\t')
                if len(cols) < 2:
                    continue
                sync_order_ids.append(int(cols[1]))  # id column
        ir.mxp_sync_order = sync_order_ids

def parse_model_io(compiler_output_dir: str, ir: internal_representation.IR):
    model_io_file = os.path.join(compiler_output_dir, 'nx_engine/vnnx_io_offsets.txt')
    model_io_id = []
    model_io_offset = []
    io_tensor_names = []
    if os.path.isfile(model_io_file):
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

def parse_mxp_tensor_offset(compiler_output_dir: str, ir: internal_representation.IR) -> dict:
    mxp_tensor_file = os.path.join(compiler_output_dir, 'nx_engine/mxp_tensor_offset_map.txt')
    mxp_tensor_to_offset = {}
    if os.path.isfile(mxp_tensor_file):
        ir.sync_with_MXP = True
        print('Reading: %s' % mxp_tensor_file)
        with open(mxp_tensor_file, 'r') as file:
            line_count = 0
            for line in file:
                line_count += 1
                # Line 1 is tensor region offset, Line 2 is tensor region size (skip both)
                if line_count <= 2:
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
                    mxp_tensor_to_offset[tensor_id] = offset
    return mxp_tensor_to_offset

def compile_model(ir:internal_representation.IR, json_fname: str, debug_output_dir:str, compiler_output_dir:str) -> internal_representation.IR:

    graph = load_graph(json_fname)
    mxp_ops = parse_mxp_ops(compiler_output_dir)
    parse_model_io(compiler_output_dir, ir)
    parse_mxp_sync_order(compiler_output_dir, ir)
    mxp_tensor_to_offset = parse_mxp_tensor_offset(compiler_output_dir, ir)
    ############# Frontend ###############################
    print('At TFLite Frontend:')
    ir = tflite_parser.tflite_to_ir(ir, graph, mxp_ops, mxp_tensor_to_offset)
    ir = frontend_hardware_agnostic.parse_nodes_params(ir)
    if not DEBUG_SIMULATE_CONCAT_REQUANT:
        ir = frontend_hardware_agnostic.update_nodes_qparams(ir)
    ir = frontend_hardware_dependent.calc_nodes_folding_factor(ir)
    ir = frontend_hardware_agnostic.fuse_concat_requant_into_folding_producers(ir)
    # Must run immediately after calc_nodes_folding_factor and before any other pass
    # (duplicate_maxpool_nodes, calc_nodes_quantization_params, get_ops_grid_config, ...)
    # reads folding_factor_x - those all lock in fold-derived values (channels, requant
    # params, backend config), so redistributing folding later leaves them stale.
    ir = program_compiler.distribute_entry_x_folding(ir)
    folding_report_filename = os.path.join(ir.compiler_output_dir, ir.model_name+'_folding_pretiling')
    reports.generate_folding_report(folding_report_filename,ir)
    ir = frontend_hardware_dependent.fix_max_pool_kernel_for_async_padding(ir)
    ir = frontend_hardware_dependent.duplicate_maxpool_nodes(ir)
    ir = frontend_hardware_dependent.calc_nodes_quantization_params(ir)

    print('At Backend:')
    ############# Backend ###############################
    # A CONCAT whose output crosses back to MXP (an NX->MXP->NX sync tensor) needs a real,
    # F4-flagged DMW to signal MXP the data is ready. The normal CONCAT op never itself emits a
    # DDR write (generate_layer_command_sequence only reads its inputs), so scatter it - each
    # input's producer then writes directly into the concat output's DDR region, and that write
    # picks up the F4 flag via the existing to-MXP tensor check in add_tensor_write_to_ddr_commands.
    # scatter_concat_inputs is otherwise only invoked reactively from the blob-failure path
    # (program_compiler.split_large_conv_nodes), which a well-fitting concat like this never hits.
    for node_name in list(ir.graph.nodes()):
        node = ir.graph.nodes()[node_name]
        if node['op_type'] == OpType.CONCAT and node['frontend']['output_tensor'].name in ir.tensors_to_mxp:
            frontend_hardware_dependent.scatter_concat_inputs(ir, node)
    ir = program_compiler.compile(ir, debug_output_dir)
    create_ddr_entries_from_ir_alex(ir)
    ir = program_compiler.compile_program(ir)


     # Split sequencer program commands list into groups of instructions of length
    # PROGRAM_MEMORY_NUM_COMMANDS (e.g., 1024 at a time).
    #
    # This will insert new DDR commands to load the next group of instructions.
    # These new DDR commands need to know the address of the sequencer program region in DDR.
    # However, that region has not been assigned yet. Therefore, an input to this function
    # needs to be that sequencer address. Currently, it is assumed to be the next free region in DDR,
    # meaning that the very next DDR entry added needs to be the sequencer region. So this
    # function is called right before create_sequencer_ddr_entry_from_ir.
    # Splitting the sequencer program needs to be done before creating the sequencer DDR entry
    # so the command list will be correct, which is used in create_sequencer_ddr_entry_from_ir
    # to create the sequencer_program_ddr_entry.

    sequencer_program_ddr_address = ir.ddr.next_free_address
    split_sequencer_program(ir, sequencer_program_ddr_address)
    create_sequencer_ddr_entry_from_ir(ir)

    return ir

def main():
    parser = argparse.ArgumentParser(description='VBX 3.0 compiler')
    parser.add_argument('-m','--model', default=None, help='Quantized TFLite model file', required=True)
    parser.add_argument('-s', '--size-conf', help='size configuration to build model for',
                        choices = ['V500','V1000'], required=True)
    parser.add_argument('-o','--output', default=None, help='Name of the ucomp output file', required=True)
    parser.add_argument('--tmp_dir', default=None, help='directory to save the debug files', required=True)
    parser.add_argument('-d', '--debug', default=False, help='Writing debug files')
    args = parser.parse_args()

    if ('.json' not in args.model):
        print("Please run vnnx_compile command to generate the compiler files")
        sys.exit(0)

    if MAX_WLOC_128BIT_ENTRIES == 10:
        print(bcolors.WARNING+'Not currently supporting MAX_WLOC_128BIT_ENTRIES == 10'+bcolors.ENDC)
        sys.exit(0)

    # Make directories and create IR object
    model_name = os.path.basename(args.model).rsplit('.',2)[0]
    compiler_output_dir = args.tmp_dir
    os.makedirs(compiler_output_dir, exist_ok=True)
    if DEBUG_REMOVE_ENGINE_COMMAND:
        model_name+='_noengine'

    ir = internal_representation.IR(model_name,size_config=args.size_conf,compiler_output_dir=compiler_output_dir,debug=args.debug)
    debug_output_dir = os.path.join(compiler_output_dir, 'debug/')
    os.makedirs(debug_output_dir, exist_ok=True)

    # This converts the TFLite model to nx internal representation including backend data.
    ir = compile_model(ir, args.model, debug_output_dir, compiler_output_dir)
    
    # TODO: The rest below is same as compiler.py, can refactor to a common file.
    if ir.debug:
        print('Writing sequencer debug and CSV files')
        sequencer_program = ir.sequencer_program
        sequencer_program_debug_filename =  os.path.join(debug_output_dir, args.output+'_sequencer_program_debug_info.txt')
        sequencer_program.save_sequencer_program_debug_file(sequencer_program_debug_filename)
        sequencer_program_csv_filename =  os.path.join(compiler_output_dir, args.output+'_sequencer_program.csv')
        sequencer_program.save_sequencer_program_csv_file(sequencer_program_csv_filename)
    print('Writing IR files')
    numsim_ir_filename = os.path.join(compiler_output_dir, args.output+'_numsim.nxir')
    ir.save_numsim_ir(numsim_ir_filename)
    print('Writing compiler output files to directory: %s' % compiler_output_dir)
    create_model_files(compiler_output_dir,args.output,ir) # This takes all ddr entries and creates a ddr file and model config file which are needed to run model on device.
    if DEBUG_GENERATE_PERFORMANCE_REPORT and ir.debug:
        performance_report_filename = os.path.join(compiler_output_dir, args.output+'_perf')
        reports.generate_performance_report(performance_report_filename,ir)
        wloc_balancing_report_filename = os.path.join(compiler_output_dir, args.output+'_wloc_balancing')
        reports.generate_wloc_balancing_report(wloc_balancing_report_filename,ir)
    if DEBUG_GENERATE_AMM_ALLOCATION_REPORT and ir.debug:
        amm_allocation_report_filename = os.path.join(compiler_output_dir, args.output+'_ammalloc')
        reports.generate_ammalloc_report(amm_allocation_report_filename,ir)
    if ir.debug:
        shutil.copyfile('common/debug_flags.py', os.path.join(compiler_output_dir, 'debug_flags.py'))

if __name__ == "__main__":
    main()
