import onnx
import onnx_parser
import common.internal_representation as internal_representation
from common.debug_flags import DEBUG_FORCE_IC_SPLIT, DEBUG_CREATE_ORDERING_CONV, DEBUG_SAVE_IR_AFTER_CBC_CREATION, DEBUG_LOAD_CBC_FROM_IR, DEBUG_SKIP_CBC_GENERATION
import program_compiler
from common.ddr_ir import DDR,DDREntry,TensorDDREntry
from common.enums import GridConfig,DDREntryType
from file_writer import create_ddr_entries_from_ir, create_ddr_entries_from_ir_alex, create_sequencer_ddr_entry_from_ir
from sequencer import split_sequencer_program
import os
import networkx as nx
import math

def compile_backend(ir : internal_representation.IR, debug_output_dir:str,model_name:str='',compiler_output_ir_dir:str='') -> internal_representation.IR:
    print('At Backend:')
    if DEBUG_LOAD_CBC_FROM_IR:
        if model_name!='' and compiler_output_ir_dir!='':
            ir_with_cbc_filename = compiler_output_ir_dir+model_name+'_withcbc.nxir'
            compiler_augmented_ir = ir.load(ir_with_cbc_filename)
        else:
            raise ValueError ('Can load ir with cbc as no directory and model names specified')
    else:
        compiler_augmented_ir = program_compiler.compile(ir, debug_output_dir)
    if DEBUG_SAVE_IR_AFTER_CBC_CREATION and model_name!='' and compiler_output_ir_dir!='':
        if DEBUG_SKIP_CBC_GENERATION:
            raise ValueError ('Cant save ir with cbc since it was not created')
        ir_with_cbc_filename = compiler_output_ir_dir+model_name+'_withcbc.nxir'
        compiler_augmented_ir.save(ir_with_cbc_filename)

    #Alex for phase 2
    #create_ddr_entries_from_ir(compiler_augmented_ir) # This allocates ddr entries from ir excluding program which can only be created after program creation
    create_ddr_entries_from_ir_alex(compiler_augmented_ir) # 

    compiler_augmented_ir = program_compiler.compile_program(compiler_augmented_ir) # This part can be done only after all ddr entries are allocated

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
    sequencer_program_ddr_address = compiler_augmented_ir.ddr.next_free_address
    split_sequencer_program(compiler_augmented_ir, sequencer_program_ddr_address)
    # Create the sequencer DDR entry
    create_sequencer_ddr_entry_from_ir(compiler_augmented_ir) #

    return compiler_augmented_ir

def main ():
    model_name = 'Conv14x14_simple'
    model_name = 'Add14x14'
    model_name = 'stage1_conv1'
    model_name = 'yolov5l_nx_s94sub'
    output_dir = 'C:/Users/dshir/Documents/tsnp_output/'
    if DEBUG_FORCE_IC_SPLIT and '28' not in model_name:
        model_name=model_name+'_icsplit'+str(DEBUG_FORCE_IC_SPLIT)
    if DEBUG_CREATE_ORDERING_CONV:
        model_name=model_name+'_with_reordering'

    compiler_output_dir = output_dir+'/'+model_name+'/'
    if not os.path.exists(compiler_output_dir):
        os.makedirs(compiler_output_dir)
    compiler_output_ir_dir = output_dir+'/nx_ir/'
    if not os.path.exists(compiler_output_ir_dir):
        os.makedirs(compiler_output_ir_dir)
    debug_output_dir = compiler_output_dir+'/debug/'
    if not os.path.exists(debug_output_dir):
        os.makedirs(debug_output_dir)
    ir_filename = compiler_output_ir_dir+model_name+'.nxir'
    ir = internal_representation.IR(model_name)
    ir = ir.load(ir_filename)
    backend_augmented_ir = compile_backend(ir, debug_output_dir=debug_output_dir,model_name=model_name,compiler_output_ir_dir=compiler_output_ir_dir)
    backend_augmented_ir.save('conv.nxb')

if __name__ == "__main__":

    main()    