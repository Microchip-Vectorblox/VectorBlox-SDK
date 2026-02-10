import common.internal_representation as internal_representation
from common.ddr_ir import DDREntry
from common.enums import DDREntryType
from common.debug_flags import DEBUG_PRINT_CBC, DEBUG_GENERATE_DDR_HEX
from common.hw_config import GRID_OPS, PER_TILE_CBC_OPS, WLOCS_PER_GRID, GridConfig, TFLITE_REQUANT
from collections import OrderedDict
import struct
from datetime import datetime
import os
# wloc/rq definitions: https://docs.google.com/spreadsheets/d/17kMTP18Vjo52BkP0z1pygQA0bY0Z70zj/edit#gid=1267198102



def prepare_wloc_ir(node):
    if node['op_type'] in PER_TILE_CBC_OPS:
        grids_wloc_splits = node['backend']['grids_cbc'][0].wlocs
    else:

        # alex ph2 del 
        #grids_wloc_splits = node['backend']['grids_cbc'].wlocs
        #put this
        current_tile_num =0
        grids_wloc_splits = node['backend']['grids_cbc'].alex_wlocs[current_tile_num]

    per_split_grids_wloc_128bit_words = []
    for grids_wloc_commands in grids_wloc_splits:
        num_grids = node['backend']['grid_count']
        grids_wloc_128bit_words = []
        num_wlocs=WLOCS_PER_GRID*num_grids
        #Alex add this line
        num_wlocs=2

        for current_grid in range(num_wlocs):
            current_grid_wloc_words = []
            current_grid_wloc = grids_wloc_commands[current_grid].cmd_list
            wloc_file_word = ''
            for idx,current_wloc in enumerate(current_grid_wloc):
                current_wloc_bits = current_wloc.to_bits()
                wloc_file_word = current_wloc_bits[2:] + wloc_file_word
                if len(wloc_file_word) == 120: # We pack all wloc entries in 128 bits words
                    current_grid_wloc_words.append('00000000'+wloc_file_word)
                    wloc_file_word=''
                elif len(wloc_file_word) > 120: # This is in case we added 24 bits long command to a command already populated with 60 bits.
                    sliced_word = wloc_file_word[12:] # We take the 12 LSB of the added 24 bits to the current word. and move the 12 MSB to next word
                    wloc_file_word = wloc_file_word[0:12]
                    current_grid_wloc_words.append('00000000'+sliced_word)
            if len (wloc_file_word) > 0: # Align last word to 72 bits
                wloc_file_word = wloc_file_word.zfill(128)
                current_grid_wloc_words.append(wloc_file_word)
            # Add additional empty wqloc entry to align to 32 Bytes
            if (len(current_grid_wloc_words) % 2 == 1):        
                wloc_file_word = ''.zfill(128)
                current_grid_wloc_words.append(wloc_file_word)
            grids_wloc_128bit_words.append(current_grid_wloc_words)
            if DEBUG_PRINT_CBC:
                print('Wloc 128 bit entries: %d' % len(current_grid_wloc_words))
        per_split_grids_wloc_128bit_words.append(grids_wloc_128bit_words)
    node['backend']['per_split_grids_wloc_128bit_words'] = per_split_grids_wloc_128bit_words

def write_wloc_hex_files(node,filename):
    if 'per_split_grids_wloc_128bit_words' not in node['backend']:
        raise ValueError ('wloc ir not available for node')
    per_split_grids_wloc_128bit_words = node['backend']['per_split_grids_wloc_128bit_words']
    for split_idx,current_split_grids_wloc_128bit_words in enumerate(per_split_grids_wloc_128bit_words):
        for wloc_idx,current_grid_wloc_bin in enumerate(current_split_grids_wloc_128bit_words):
            grid_idx = wloc_idx // WLOCS_PER_GRID
            wloc_num = wloc_idx % WLOCS_PER_GRID
            current_filename = filename+'_split'+str(split_idx)+'_grid'+str(grid_idx)+'_wloc'+str(wloc_num)+'.hex'
            with open(current_filename,'w') as f:
                for entry in current_grid_wloc_bin:
                    list_8bits = [entry[i:i+8] for i in range(0, len(entry), 8)]
                    line=''
                    for current_8bit in list_8bits:
                        current_int = int(current_8bit,2)
                        current_hex = hex(current_int)[2:].zfill(2)
                        line=line+current_hex
                    f.write(line+'\n')

def write_wloc_hex_files_alex(node,filename, current_tile_num = 0):

    #do the memeory
    node['backend']['grids_cbc'].alex_wlocs[current_tile_num][0][0].create_cmd_mem()
    node['backend']['grids_cbc'].alex_wlocs[current_tile_num][0][1].create_cmd_mem()

    #nr_wlocs for 14x16 only
    nr_wlocs = 2
    split_idx =0
    grid_idx = 0
    digits_per_line = 8  # Number of hexadecimal digits per line
    
    for wloc_idx in range(nr_wlocs):
 
        mem = node['backend']['grids_cbc'].alex_wlocs[current_tile_num][0][wloc_idx].cmd_mem
        current_filename = filename+'_split'+str(split_idx)+'_grid'+str(grid_idx)+'_wloc'+str(wloc_idx)+'.hex'
        with open(current_filename,'w') as f:

            # Initialize a temporary string to accumulate hex characters
            hex_string = ''

            # Iterate through each byte in the array
            for byte in mem:
                # Convert the byte to hex and add it to the string
                hex_string += f'{byte:02x}'  # Two characters per byte, e.g., '01', 'ff'
        
                # Check if the string has reached the required length (N)
                if len(hex_string) >= digits_per_line:
                    f.write(hex_string[:digits_per_line]+'\n')        
                    hex_string = hex_string[digits_per_line:]  # Keep the remaining characters for the next line

            # If there are remaining characters, print them
            if hex_string:
                f.write(hex_string)

                    

def prepare_rqloc_ir(node):
    if node['op_type'] in PER_TILE_CBC_OPS:
        per_split_grids_rq_commands = node['backend']['grids_cbc'][0].rqlocs
    else:
        per_split_grids_rq_commands = node['backend']['grids_cbc'].rqlocs
    per_split_rqloc_128bit_hex_words = []
    for current_split_grids_rq_commands in per_split_grids_rq_commands:
        grids_rq_commands = current_split_grids_rq_commands.cmd_list
        rqloc_128bit_hex_words = []
        rqloc_file_word = ''
        for current_rqloc in grids_rq_commands:
            current_rqloc_bin = current_rqloc.to_bits()
            rqloc_file_word = current_rqloc_bin[2:].zfill(12) + rqloc_file_word
            if len(rqloc_file_word) == 120: # We pack all rqloc entries in 128 bits words => 32 hex digits
                rqloc_128bit_hex = hex(int('0b'+rqloc_file_word,2))[2:].zfill(32)
                rqloc_128bit_hex_words.append(rqloc_128bit_hex)
                rqloc_file_word=''
            if len(rqloc_file_word) > 120:
                # alex ph2 del
                #raise ValueError ('Something went wrong, got a word longer than 120 bits')
                pass
        if len (rqloc_file_word) > 0: # Align last word to 32 hex digits (128 bits)
            rqloc_file_word = rqloc_file_word.zfill(128)
            rqloc_128bit_hex = hex(int('0b'+rqloc_file_word,2))[2:].zfill(32)
            rqloc_128bit_hex_words.append(rqloc_128bit_hex)
        
        # By Itamar request, add additional empty rqloc entry
        rqloc_file_word = ''.zfill(128)
        rqloc_128bit_hex = hex(int('0b'+rqloc_file_word,2))[2:].zfill(32)
        rqloc_128bit_hex_words.append(rqloc_128bit_hex)

        # Add additional empty rqloc entry to align to 32 Bytes
        if (len(rqloc_128bit_hex_words) % 2 == 1):        
            rqloc_file_word = ''.zfill(128)
            rqloc_128bit_hex = hex(int('0b'+rqloc_file_word,2))[2:].zfill(32)
            rqloc_128bit_hex_words.append(rqloc_128bit_hex)
        if DEBUG_PRINT_CBC:
            print('RQloc 128 bit entries: %d' % len(rqloc_128bit_hex_words))
        per_split_rqloc_128bit_hex_words.append(rqloc_128bit_hex_words)
    node['backend']['per_split_rqloc_128bit_hex_words'] = per_split_rqloc_128bit_hex_words

def write_rqloc_hex_file(node,rqloc_filename):
    if 'per_split_rqloc_128bit_hex_words' not in node['backend']:
        raise ValueError ('rqloc ir not available for node')
    per_split_rqloc_128bit_hex_words = node['backend']['per_split_rqloc_128bit_hex_words']
    for split_idx,current_split_rqloc_128bit_hex_words in enumerate(per_split_rqloc_128bit_hex_words):
        current_filename = rqloc_filename+'_split'+str(split_idx)+'.hex'
        with open(current_filename,'w') as f:
            for entry in current_split_rqloc_128bit_hex_words:
                f.write(entry+'\n')

def prepare_rqparams_ir(node):
    if node['op_type'] in PER_TILE_CBC_OPS:
        rqparams = node['backend']['grids_cbc'][0].rqparams.rqparams_list
    else:
        rqparams = node['backend']['grids_cbc'].rqparams.cmd_list
    rqparams_128bit_hex_words = []
    rqparams_file_word = ''
    for current_rqparams in rqparams:
        current_rqparams_bin = current_rqparams.to_bits()
        if TFLITE_REQUANT:
            # This used to be packed 3 x 36 into 128 bit words
            # Now it packs 2 x 64
            rqparams_file_word = current_rqparams_bin[2:] + rqparams_file_word
            if len(rqparams_file_word) == 128: # We pack all rqparams entries in 128 bits words => 32 hex digits
                rqparams_128bit_hex = hex(int('0b'+rqparams_file_word, 2))[2:].zfill(32)
                rqparams_128bit_hex_words.append(rqparams_128bit_hex)
                rqparams_file_word=''
            if len(rqparams_file_word) > 128:
                raise ValueError ('Something went wrong, got a word longer than 128 bits')
        else:
            rqparams_file_word = current_rqparams_bin[2:].zfill(36) + rqparams_file_word
            if len(rqparams_file_word) == 108: # We pack all rqparams entries in 128 bits words => 32 hex digits
                rqparams_128bit_hex = hex(int('0b'+rqparams_file_word, 2))[2:].zfill(32)
                rqparams_128bit_hex_words.append(rqparams_128bit_hex)
                rqparams_file_word=''
            if len(rqparams_file_word) > 108:
                # alex ph2 del
                #raise ValueError ('Something went wrong, got a word longer than 108 bits')
                pass
    if len (rqparams_file_word) > 0: # Align last word to 32 hex digits (128 bits)
        rqparams_file_word = rqparams_file_word.zfill(128)
        rqparams_128bit_hex = hex(int('0b'+rqparams_file_word,2))[2:].zfill(32)
        rqparams_128bit_hex_words.append(rqparams_128bit_hex)
    node['backend']['rqparams_128bit_hex_words'] = rqparams_128bit_hex_words

def write_rqparams_hex_file(node,rqparams_filename):
    if 'rqparams_128bit_hex_words' not in node['backend']:
        raise ValueError ('rqparams ir not available for node')
    rqparams_128bit_hex_words = node['backend']['rqparams_128bit_hex_words']
    with open(rqparams_filename,'w') as f:
        for entry in rqparams_128bit_hex_words:
            f.write(entry+'\n')

def create_ddr_entries_from_ir(ir: internal_representation.IR):
    ddr = ir.ddr
    for node_name,node in ir.graph.nodes.items():
        if node['op_type'] in GRID_OPS:
            if node['op_type'] in PER_TILE_CBC_OPS:
                cbc_per_op = node['frontend']['y_tiles']
            else:
                cbc_per_op = 1
            for tile_num in range(cbc_per_op):
                if node['op_type'] in PER_TILE_CBC_OPS:
                    grids_cbc = node['backend']['grids_cbc'][tile_num]
                else:
                    grids_cbc = node['backend']['grids_cbc']
                grids_cbc.create_wloc_mem() # This converts wloc ir for each wloc split and grid to bytesarray as it would be in DDR mem
                for split_idx,split_wlocs in enumerate(grids_cbc.wlocs):
                    for wloc_idx,wloc in enumerate(split_wlocs): # Allocate DDR mem for above wlocs
                        grid_idx = wloc_idx // WLOCS_PER_GRID
                        wloc_num = wloc_idx % WLOCS_PER_GRID
                        ddr_entry_description = 'Layer: %s, Tile %d, Split %d, CBC WLOC of grid #%d wloc#%d' % (node_name,tile_num,split_idx,grid_idx,wloc_num)
                        wloc_ddr_entry = DDREntry(wloc.wloc_mem, type = DDREntryType.WLOC, description = ddr_entry_description)
                        ddr.add_entry(wloc_ddr_entry)
                        wloc.wloc_mem_address = wloc_ddr_entry.address

                grids_cbc.create_rqloc_mem() # This converts rqloc ir to bytesarray as it would be in DDR mem
                # Below code allocates DDR for the CBC RQLOC Table
                for split_idx,split_rqloc in enumerate(grids_cbc.rqlocs):
                    ddr_entry_description = 'Layer: %s, Tile %d, split: %d, CBC RQLOC' % (node_name,tile_num,split_idx)
                    rq_param_ddr_entry = DDREntry(split_rqloc.cmd_mem,type = DDREntryType.RQLOC, description=ddr_entry_description)
                    ddr.add_entry(rq_param_ddr_entry)
                    split_rqloc.rqloc_mem_address = rq_param_ddr_entry.address

                grids_cbc.create_rqparams_mem() # This cpnverts rqparams it to bytesarray as it would be in DDR mem
                # Below code allocates DDR for the RQPARAMS Table
                ddr_entry_description = 'Layer: %s, RQPARAMS table, Tile %d' % (node_name,tile_num)
                rqparams_ddr_entry = DDREntry(grids_cbc.rqparams.cmd_mem,type = DDREntryType.RQPARAMS, description=ddr_entry_description)
                ddr.add_entry(rqparams_ddr_entry)
                grids_cbc.rqparams.rqparams_mem_address = rqparams_ddr_entry.address

def create_ddr_entries_from_ir_alex(ir: internal_representation.IR):

    ddr = ir.ddr
    for node_name,node in ir.graph.nodes.items():
        #concate is not conv comand
        if node['op_type'] == 'Concat':
            continue

        grids_cbc      = node['backend']['grids_cbc'] 
        tile_num_total = len(grids_cbc.alex_wlocs) 
        node_name      = node_name

        for current_tile_num in range(tile_num_total):
            current_tile = grids_cbc.alex_wlocs[current_tile_num]
            num_splits = len(current_tile) 
            
            for split_idx,split_wlocs in enumerate(current_tile):
                for wloc_idx in range(len(split_wlocs)):

                    grid_idx = wloc_idx // WLOCS_PER_GRID
                    wloc_num = wloc_idx % WLOCS_PER_GRID
                    wloc = current_tile[split_idx][wloc_num]
                    ddr_entry_description = 'Layer: %s, Tile %d, Split %d, CBC WLOC of grid #%d wloc#%d' % (node_name,current_tile_num,split_idx,grid_idx,wloc_num)
                    wloc_ddr_entry = DDREntry(wloc.cmd_mem, type = DDREntryType.WLOC, description = ddr_entry_description)
                    ddr.add_entry(wloc_ddr_entry)
                    wloc.cmd_mem_address = wloc_ddr_entry.address

            #RQ PARAMS
            tile_num = 0        
            # Example: rq_params = grids_cbc.alex_rqParam[split_idx][0] 
            for split_idx, split_rq_params in enumerate(grids_cbc.alex_rqParam):
                rq_params = split_rq_params[0]
                rq_params.create_cmd_mem()     
                ddr_entry_description = 'Layer: %s, Tile %d, split: %d, CBC RQPARAMS' % (node_name,tile_num,split_idx)
                rq_param_ddr_entry = DDREntry(rq_params.cmd_mem,type = DDREntryType.RQPARAMS, description=ddr_entry_description)
                ddr.add_entry(rq_param_ddr_entry)
                rq_params.cmd_mem_address = rq_param_ddr_entry.address

            # Result Table
            tile_num = 0
            for split_idx, split_rt in enumerate(grids_cbc.RTable):
                rt = split_rt
                rt.create_cmd_mem()    
                ddr_entry_description = 'Layer: %s, Tile %d, split: %d, CBC RESULTTABLE' % (node_name,tile_num,split_idx)
                rt_ddr_entry = DDREntry(rt.cmd_mem,type = DDREntryType.RESULTTABLE, description=ddr_entry_description)
                ddr.add_entry(rt_ddr_entry)
                rt.cmd_mem_address = rt_ddr_entry.address 

            # NLF
            tile_num = 0
            split_idx = 0
            nlf = grids_cbc.nlf[split_idx]    
            nlf.create_cmd_mem() # This converts rqparams it to bytesarray as it would be in DDR mem 
            ddr_entry_description = 'Layer: %s, Not Linear Function, Tile %d' % (node_name,tile_num)
            nlf_ddr_entry = DDREntry(nlf.cmd_mem,type = DDREntryType.NONLINEARFUNCTION, description=ddr_entry_description)
            ddr.add_entry(nlf_ddr_entry)
            nlf.cmd_mem_address = nlf_ddr_entry.address

   


def create_sequencer_ddr_entry_from_ir(ir: internal_representation.IR):
    ddr = ir.ddr
    ir.sequencer_program.create_program_mem()
    ddr_entry_description = 'This is the program sequence in DDR'
    sequencer_program_ddr_entry = DDREntry(ir.sequencer_program.commands_mem,type=DDREntryType.PROGRAM,description=ddr_entry_description)
    ddr.add_entry(sequencer_program_ddr_entry)
    ir.sequencer_program.commands_mem_address = sequencer_program_ddr_entry.address

    register_page_bytes = bytearray(0)
    register_page_bytes.extend((sequencer_program_ddr_entry.address).to_bytes(8,'little')) # Set ddr entry 0, address 0 to the start address of the program (8 bytes)
    register_page_bytes.extend((len(sequencer_program_ddr_entry.bytes)).to_bytes(4,'little')) # Set ddr entry 0, address 8 to the program length in bytes(4 bytes)
    number_of_outputs = len(ir.outputs)
    register_page_bytes.extend((number_of_outputs).to_bytes(4,'little')) # Set ddr entry 0, address 12 to number of outputs(4 bytes)
    t = datetime.now()
    register_page_bytes.extend((t.day).to_bytes(1,'little'))
    register_page_bytes.extend((t.month).to_bytes(1,'little'))
    register_page_bytes.extend((t.year).to_bytes(2,'little'))
    register_page_bytes.extend((t.minute).to_bytes(1,'little'))
    register_page_bytes.extend((t.hour).to_bytes(1,'little'))
    register_page_bytes.extend((0).to_bytes(6,'little'))
    if 'images' in ir.tensors:
        input_tensor = ir.tensors['images']
    elif 'X' in ir.tensors: # Used by many tests in nx_quantize_model.py
        input_tensor = ir.tensors['X']
    else:
        input_tensor = ir.tensors[ir.inputs[0]]
    input_tensor_shape = input_tensor.get_original_shape()
    model_input_length = ir.inputs_ddr.next_free_address
    model_output_length = ir.outputs_ddr.next_free_address
    register_page_bytes.extend((input_tensor_shape[2]).to_bytes(2,'little'))
    register_page_bytes.extend((input_tensor_shape[3]).to_bytes(2,'little'))
    register_page_bytes.extend((input_tensor.folding_factor_y).to_bytes(2,'little'))
    register_page_bytes.extend((input_tensor.folding_factor_x).to_bytes(2,'little'))
    register_page_bytes.extend((model_input_length).to_bytes(4,'little'))
    register_page_bytes.extend((model_output_length).to_bytes(4,'little'))
    register_page_bytes.extend((0).to_bytes(84,'little')) # Set ddr entry 0, address 0x2C to Reserved (84 bytes) so outputs start at 0x80
    for output_index,output_name in enumerate(ir.outputs):
        output_tensor = ir.tensors[output_name]
        output_tensor_shape = output_tensor.get_original_shape()
        output_producing_node_name = output_tensor.producer
        output_producing_node = ir.graph.nodes[output_producing_node_name]
        tensor_gridmode = output_producing_node['backend']['gridmode']
        if tensor_gridmode == GridConfig.H28xW28:
            mode_param = 0
        elif tensor_gridmode == GridConfig.H14xW14:
            mode_param = 1
        elif tensor_gridmode == GridConfig.H7xW7:
            mode_param = 2
        elif tensor_gridmode == GridConfig.H14xW8:
            mode_param = 3
        elif tensor_gridmode == GridConfig.H14xW16:
            mode_param = 4
        elif tensor_gridmode == GridConfig.H14xW32:
            mode_param = 5
        else:
            raise ValueError ('Gridmode %s not supported for register page output. Please check...' % (str(tensor_gridmode)))

        register_page_bytes.extend((output_tensor.ddr_entry.address).to_bytes(4,'little'))
        register_page_bytes.extend((output_tensor.ddr_entry.num_of_blocks).to_bytes(2,'little'))
        register_page_bytes.extend((mode_param).to_bytes(2,'little'))
        register_page_bytes.extend((output_tensor_shape[0]).to_bytes(2,'little'))
        register_page_bytes.extend((output_tensor_shape[1]).to_bytes(2,'little'))
        register_page_bytes.extend((output_tensor_shape[2]).to_bytes(2,'little'))
        register_page_bytes.extend((output_tensor_shape[3]).to_bytes(2,'little'))
        register_page_bytes.extend(int(output_tensor.zero_point).to_bytes(1,'little', signed=TFLITE_REQUANT))
        register_page_bytes.extend((0).to_bytes(1,'little')) # 1 byte reserved
        register_page_bytes.extend((output_tensor.folding_factor_y).to_bytes(1,'little'))
        register_page_bytes.extend((output_tensor.folding_factor_x).to_bytes(1,'little'))
        register_page_bytes.extend(bytearray(struct.pack("f", output_tensor.scale)))

        output_name_bytes = bytearray(output_name,'utf-8')
        if len(output_name_bytes)>64:
            raise ValueError ('output tensor names (%s) longer than 64 characters are not supported' % output_name)
        output_name_fill_to_64 = (0).to_bytes((64-len(output_name_bytes)),'little')
        register_page_bytes.extend(output_name_bytes)
        register_page_bytes.extend(output_name_fill_to_64)
        register_page_bytes.extend((0).to_bytes(168,'little')) # 176 bytes reserved so next output description starts at multiples of 0x100

    ddr.entries[0].bytes = (register_page_bytes) # Set ddr entry 0 to the start address of the program

def create_snp_host_cfg(snp_hos_cfg_filename,sequencer_program_ddr_address,sequencer_program_size):
    with open(snp_hos_cfg_filename,'w') as snp_host_cfg_file:
        program_adress = sequencer_program_ddr_address.to_bytes(8,'big')
        program_address_32bitlsb=program_adress.hex()[8:16]
        program_address_32bitmsb=program_adress.hex()[0:8]
        program_size = sequencer_program_size.to_bytes(4,'big').hex()
        line = '20 %s\n24 %s\n28 %s\n00 00000001\n00 00000000' % (program_address_32bitlsb,program_address_32bitmsb,program_size)
        snp_host_cfg_file.write(line)

def create_model_files(compiler_output_dir,model_name, ir: internal_representation.IR):
    ir.inputs_ddr.create_ddr_mem() # This creates the bytearray of the entire DDR
    ir.outputs_ddr.create_ddr_mem() # This creates the bytearray of the entire DDR
    ir.ddr.create_ddr_mem() # This creates the bytearray of the entire DDR
    if DEBUG_GENERATE_DDR_HEX:
        nx_model_hex_filename = os.path.join(compiler_output_dir, model_name+'_ddr_content.hex')
        ir.ddr.save_ddr_bytearray_as_text(nx_model_hex_filename)
    nx_model_bin_filename = os.path.join(compiler_output_dir, model_name+'_ddr_content.bin')
    ir.ddr.save_ddr_bytearray_as_bin(nx_model_bin_filename)
    nx_model_debug_info_filename = os.path.join(compiler_output_dir, model_name+'_ddr_info.txt')
    #ddr.save_ddr_info_file(nx_model_debug_info_filename)
    with open(nx_model_debug_info_filename,'w') as ddr_info_file:
        inputs_ddr_info = ir.inputs_ddr.get_ddr_info()
        ddr_info_file.write(inputs_ddr_info)
        outputs_ddr_info = ir.outputs_ddr.get_ddr_info()
        ddr_info_file.write(outputs_ddr_info)
        program_ddr_info = ir.ddr.get_ddr_info()
        ddr_info_file.write(program_ddr_info)

def write_oc_processing_order(node,filename):
    with open(filename,'w') as oc_order_file:
        for idx,oc in enumerate(node['backend']['oc_order']):
            oc_order_file.write('processing order:'+str(idx)+' Actual output channel:'+str(oc)+'\n')

