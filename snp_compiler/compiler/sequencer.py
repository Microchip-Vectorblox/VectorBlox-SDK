import numpy as np
from common.sequencer_ir import SequencerDDRReadEntry, SequencerDDRWriteEntry, SequencerEngineOperationEntry, DDRReadDstInfoAMM,\
                                DDRReadDstInfoWLOC, DDRReadDstInfoTables, DDRWriteSrcInfoAMM, SequencerCustomOperationEntry, CustomeOpCode,\
                                DDRReadDstInfoProgram
from common.enums import DDRCommandDstSrcType, EngineOpCode, DDRReadOffsetType, DDRWriteOffsetType, GridConfig, PadMode, ScalingMode, DDREntryType
from common.hw_config import DDR_MIN_READ_SIZE, DDR_READ_GRANULARITY_BYTES, DMR_TABLE_LENGTH_BYTES, URAM_BLOCK_SIZE, URAM_FIRST_AMM_INDEX,\
                            DUAL_NONCONTIGUOUS_ALLOCATION_OPS, GRID_OPS, TSNP_DDR_BYTES_PER_CHANNEL, TSNP_DDR_BOX_DEPTH, AMM_HEIGHT,\
                            AMM_WIDTH,BYTES_PER_LINE_OFFSET_UNIT, PER_TILE_CBC_OPS, WLOC_READ_AXI0_GRIDS, WLOC_READ_AXI1_GRIDS, MULTIPLE_INPUT_OPS,\
                            LIMITED_GRIDS_OPS, get_grids_per_line, get_num_virtual_grids, WLOCS_READ_ON_AXI0, WLOCS_READ_ON_AXI1, MAX_TABLE_LENGTH,\
                            URAM_NUM_BLOCKS, PROGRAM_MEMORY_NUM_COMMANDS, BYTES_IN_SEQUENCER_COMMAND, MAX_X_WRAPPING, NO_NEIGHBOR_SLICE
import math
from common.debug_flags import DEBUG_OPTIMIZE_DMA_TRANSACTIONS, DEBUG_REMOVE_ENGINE_COMMAND, DEBUG_FORCE_DEEP_CONV, DEBUG_AVOID_DDR_WRITE_WHILE_CONV, DEBUG_ADD_AXI1_WAIT_FLAG,\
      DEBUG_SKIP_CBC_GENERATION,DEBUG_AVOID_DDR_READ_WHILE_CONV, DEBUG_AVOID_OFFLOAD_TENSOR_DDR_WRITE_WHILE_CONV, DEBUG_MERGE_IDENTICAL_WLOC_READS,\
      DEBUG_ADD_WAIT_FOR_WMT_READ, DEBUG_2_TILE_RW_LINES
import common.internal_representation as internal_representation
from common.utils import get_y_tile_sizes, get_y_tile_sizes_2_tile_case
from common.tensor_ir import InputTensorInfo
import copy
import re

def get_line_offset(tensor_channels):
    rounded_length = int(math.ceil(tensor_channels/TSNP_DDR_BOX_DEPTH)*TSNP_DDR_BOX_DEPTH)
    line_offset = rounded_length*TSNP_DDR_BYTES_PER_CHANNEL // BYTES_PER_LINE_OFFSET_UNIT# line offset in bytes is line_offset*256
    return line_offset

def get_grid_offset(grid_num,tensor_channels=0,tensor_lines=0):
    rounded_length = int(math.ceil(tensor_channels/TSNP_DDR_BOX_DEPTH)*TSNP_DDR_BOX_DEPTH)
    grid_offset = int(grid_num*rounded_length*TSNP_DDR_BYTES_PER_CHANNEL*tensor_lines) # We need the int since tensor lines can be float if its a y folding conv and original y is odd number of lines
    return grid_offset

def get_grids_to_write(folded_output_tensor):
    folded_shape = folded_output_tensor.get_folded_shape()
    folded_width = folded_shape[3]
    num_grids_to_write = ((folded_width-1) // AMM_WIDTH)+1
    return num_grids_to_write


def add_input_load_from_ddr_commands(ir,tensor_info:InputTensorInfo,current_node_sequence,node, set_flags_mask=0,wait_flags_mask=0,target_grid = 0, ddr_read_offset_type = DDRReadOffsetType.INPUT_TENSOR,current_tile_num=0,read_padding_line=False, current_xslice_num=0):
    input_tensor_name = tensor_info.name
    input_index = tensor_info.input_index
    tile_to_load_idx = tensor_info.tensor_tile_num
    xslice_to_load_idx = tensor_info.tensor_xslice_num

    if node['op_type'] in ['Add','Concat']:
        input_tensor = node['frontend']['input_tensors'][input_index]
    else:
        input_tensor = node['frontend']['input_tensor']

    even_grid = not (target_grid % 2)
    #if input_tensor.ddr_entry: # Input tensor is in DDR
    current_blob_idx = node['frontend']['tiling_blob_idx']
    current_blob = ir.tiling_blobs[current_blob_idx]
    k3_nodes_in_blob = current_blob.k3_nodes
    
    if even_grid:
        amm_allocated_blocks_for_input = node['backend']['allocated_amm_blocks_for_input_even_grid'][tile_to_load_idx][xslice_to_load_idx][input_index]
    else:
        amm_allocated_blocks_for_input = node['backend']['allocated_amm_blocks_for_input_odd_grid'][tile_to_load_idx][xslice_to_load_idx][input_index]

    # We need to make sure that allocated blocks for input are same for all amms. Otherwise we need to find a way to write to different addresses per each amm (which is currently not supported)
    same_blocks = True
    allocated_blocks = amm_allocated_blocks_for_input[0]
    for current_amm_allocated_blocks in amm_allocated_blocks_for_input:
        if current_amm_allocated_blocks!= allocated_blocks:
            same_blocks = False
            break
    if not same_blocks:
        raise ValueError ('At read inputs from DDR to AMM. AMM allocation is not same for all AMMs. This is currently un-supported')
    amm_start_address = allocated_blocks[0] * URAM_BLOCK_SIZE
    input_tensor_length = input_tensor.shape[1]
    post_folding_input_tensor_length = input_tensor_length
    post_folding_tensor_lines = input_tensor.get_folded_shape()[2]
    y_folding_factor = input_tensor.folding_factor_y
    if 'force_folding_y' in node['frontend']:
        post_folding_input_tensor_length *= 2
        y_folding_factor += 1
        post_folding_tensor_lines = int(math.ceil(post_folding_tensor_lines / 2))
    if 'force_unfolding_y' in node['frontend']:
        if (input_tensor_length % 2) !=0:
            raise ValueError ('Y Unfolding of odd channel tensor is not supported yet')
        post_folding_input_tensor_length = int(post_folding_input_tensor_length / 2)
        post_folding_tensor_lines *= 2
        if (y_folding_factor > 0):
            y_folding_factor -= 1

    grid_offset = get_grid_offset(target_grid,tensor_channels=input_tensor_length,tensor_lines = input_tensor.shape[2]) # We calc grid offset with original (before folding) params (input_tensor_length,actual_tensor_lines) since in case of odd numbers the gird offset changes and we must have same grid offset as it was written to ddr    
    input_amm_info = DDRReadDstInfoAMM(amm_start_address=amm_start_address)

    
    #add_padding_line = (input_tensor.shape[2]%2==1) if (node['frontend']['input_folding_factor_y'] > 0) else False
    num_of_lines,read_start_line,write_start_line = get_num_of_lines_per_tile(post_folding_tensor_lines, k3_nodes_in_blob=k3_nodes_in_blob,
                                                    num_y_tiles=node['frontend']['y_tiles'],current_tile_num=tile_to_load_idx,add_padding_line=False) # add_padding_line=False since in read we always read MAX_GRID_HEIGHT lines
    read_size_bytes = int(math.ceil(post_folding_input_tensor_length/32)*32 * 14 * (256 // 32)) # 256 bytes per 32 channels of each grid , 14 lines per read
    if 'total_tensor_read_bytes' in node['backend']:
        if (current_tile_num, current_xslice_num) in node['backend']['total_tensor_read_bytes']:
            node['backend']['total_tensor_read_bytes'][(current_tile_num, current_xslice_num)]+=read_size_bytes
        else:
            node['backend']['total_tensor_read_bytes'][(current_tile_num, current_xslice_num)] = read_size_bytes
    else:
        node['backend']['total_tensor_read_bytes']={(current_tile_num, current_xslice_num):read_size_bytes}

    input_y = input_tensor.shape[2]
    if ('input_tensor' in node['frontend']) and node['frontend']['input_tensor'].is_avgPool_output:
        input_y += 1
    input_z = input_tensor.shape[1]*(2**input_tensor.folding_factor_x)
    y_folding = pow(2, y_folding_factor)
    grid_mode = node['backend']['gridmode']
    if grid_mode == GridConfig.H14xW8:
        input_x = 8
        if (node['backend']['oc_splits'] > 1):
            amm_mask = 3
            amm_write_mode = 1
        else:
            amm_mask = pow(2, target_grid)
            amm_write_mode = pow(2, target_grid)
        x_wrapping = 0
    else:
        input_x = math.ceil(input_tensor.shape[3]/(16*(2**input_tensor.folding_factor_x)))*16
        amm_mask = 3
        amm_write_mode = 0
        x_wrapping = 1
    
    if (input_tensor.num_packed_xslices > 1):
        amm_mask = 11
    amm_nchannels = input_z * y_folding * x_wrapping * input_tensor.num_packed_xslices

    # Check if this tensor is from MXP
    tensor_from_mxp = False
    if ir.sync_with_MXP and input_tensor.name in ir.tensors_from_mxp:
        if input_tensor.name in ir.io_tensor_names:
            ddr_read_offset_type = DDRReadOffsetType.INPUT_TENSOR
        else:
            ddr_read_offset_type = DDRReadOffsetType.MXP_BASE
        tensor_from_mxp = True

    #Yaron - DMA parameter change
#    if DEBUG_OPTIMIZE_DMA_TRANSACTIONS and (x_wrapping < 16) and (input_x > 8) and (ddr_read_offset_type == DDRReadOffsetType.MODEL_MEM):
#        input_z = int(input_z/(256/input_x))
#        input_x = 256
#        x_wrapping = 16
    if DEBUG_OPTIMIZE_DMA_TRANSACTIONS and (x_wrapping == 1) and (ddr_read_offset_type == DDRReadOffsetType.MODEL_MEM):
        x_wrapping = 1
        while (input_z > 1) and (x_wrapping <= (MAX_X_WRAPPING//2)) and (input_x <2048):
            input_z /= 2
            x_wrapping *= 2
            input_x *= 2
        input_z = int(math.ceil(input_z))
        if x_wrapping == 32:
            x_wrapping = 0
        y_tile_ddr_offset = read_start_line * input_x * y_folding
        x_slice_ddr_offset = current_xslice_num * 16 * x_wrapping
    else:
        x_wrapping = input_tensor.num_packed_xslices
        y_tile_ddr_offset = read_start_line * input_x * y_folding
        x_slice_ddr_offset = current_xslice_num * 16 * x_wrapping
    ddr_src_address = input_tensor.ddr_entry.address + y_tile_ddr_offset + grid_offset + x_slice_ddr_offset
    
    #if (x_wrapping == 32): # x-wrapping of 32 is 0 in the DMR tables
    #    x_wrapping = 0

    if tensor_from_mxp:
        if (tile_to_load_idx==0) and (xslice_to_load_idx==0):
            wait_flags_mask += 32 # TSNP wait for MXP on F5
        # Offset type still stays at 0, but address needs to change
        ddr_src_address = ir.mxp_tensor_to_offset[input_tensor.name] + y_tile_ddr_offset + grid_offset + x_slice_ddr_offset
        ir.input_ddr_offset.add(ddr_src_address)

    sequencer_entry = SequencerDDRReadEntry(src_address=ddr_src_address, ddr_read_offset_type = ddr_read_offset_type, amm_mask = amm_mask, amm_write_mode = amm_write_mode, amm_start_address=amm_start_address, 
                    nlines_to_read = 14, input_x = input_x, input_y = input_y, input_z = input_z, y_folding = y_folding, x_wrapping = x_wrapping, amm_nchannels = amm_nchannels,
                    dst_type = DDRCommandDstSrcType.AMM, dst_info = input_amm_info, set_flags = set_flags_mask, wait_flags = wait_flags_mask)
    current_node_sequence.append(sequencer_entry)
    return current_node_sequence
        
def get_num_of_lines_per_tile(
        folded_height,
        k3_nodes_in_blob=0,
        num_y_tiles=1,
        current_tile_num=0,
        add_padding_line=False,
        consecutive_2_tile_blobs=1,
        prev_blob_lines_in_ddr=-1,
        is_current_blob_folding=False,
        change_read_write_length=False):
    
    folded_height_padded = folded_height + (1 if add_padding_line else 0)

    if DEBUG_2_TILE_RW_LINES and num_y_tiles == 2 and change_read_write_length:
        per_tile_num_of_lines,per_tile_read_start_line,per_tile_write_start_line = get_y_tile_sizes_2_tile_case(
            folded_height_padded,
            k3_nodes_in_blob=k3_nodes_in_blob,
            consecutive_2_tile_blobs=consecutive_2_tile_blobs,
            prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
            is_first_node_folding=is_current_blob_folding,
        )
    else:
        per_tile_num_of_lines,per_tile_read_start_line,per_tile_write_start_line = get_y_tile_sizes(folded_height_padded,k3_nodes_in_blob=k3_nodes_in_blob,add_padding_line=False)
    num_of_lines = per_tile_num_of_lines[current_tile_num]
    read_start_line = per_tile_read_start_line[current_tile_num]
    write_start_line = per_tile_write_start_line[current_tile_num]
    return num_of_lines,read_start_line,write_start_line


def add_tensor_write_to_ddr_commands(ir,tensor_to_write,current_node_sequence,node, set_flags_mask = 0, wait_flags_mask = 0,
                                     consecutive_2_tile_blobs=1, prev_blob_lines_in_ddr=-1,
                                     source_grid=0, add_padding_line = False,current_tile_num=0, override_output_start_line = None, current_xslice_num=0):
    if tensor_to_write.ddr_entry.type == DDREntryType.INTERMEDIATE_TENSOR:
        ddr_write_offset_type = DDRWriteOffsetType.MODEL_MEM
    elif tensor_to_write.ddr_entry.type == DDREntryType.OUTPUT_TENSOR:
        ddr_write_offset_type = DDRWriteOffsetType.OUTPUT_TENSOR
    else:
        raise ValueError ('Unsupported tensor type for write. Please check...')

    even_grid = not (source_grid % 2)
    amm_idx = source_grid
    if even_grid:
        amm_allocated_blocks_for_output = node['backend']['allocated_amm_blocks_for_output_even_grid'][current_tile_num][current_xslice_num]
    else:
        amm_allocated_blocks_for_output = node['backend']['allocated_amm_blocks_for_output_odd_grid'][current_tile_num][current_xslice_num]

    # We need to make sure that allocated blocks for input are same for all amms. Otherwise we need to find a way to write to different addresses per each amm (which is currently not supported)
    same_blocks = True
    allocated_blocks = amm_allocated_blocks_for_output[0]
    for current_amm_allocated_blocks in amm_allocated_blocks_for_output:
        if current_amm_allocated_blocks!= allocated_blocks:
            same_blocks = False
            break
    if not same_blocks:
        raise ValueError ('At write outputs from AMM to DDR. AMM allocation is not same for all AMMs. This is currently un-supported')

    amm_start_address = allocated_blocks[0] * URAM_BLOCK_SIZE
    actual_tensor_shape = tensor_to_write.shape
    ouput_tensor_length = actual_tensor_shape[1]
    current_blob_idx = node['frontend']['tiling_blob_idx']
    current_blob = ir.tiling_blobs[current_blob_idx]
    k3_nodes_in_blob = current_blob.k3_nodes
    is_current_blob_folding = 'force_folding_y' in ir.graph.nodes()[current_blob.nodes_in_blob[0]]['frontend']
    grid_offset = get_grid_offset(source_grid,tensor_channels=ouput_tensor_length,tensor_lines = actual_tensor_shape[2])

    #add_padding_line = (actual_tensor_shape[2]%2==1) and (('is_avgPool' in node['frontend']) and node['frontend']['is_avgPool'])
    num_of_lines,read_start_line,write_start_line = get_num_of_lines_per_tile(tensor_to_write.get_folded_shape()[2],k3_nodes_in_blob=k3_nodes_in_blob,
                                                      num_y_tiles=node['frontend']['y_tiles'],current_tile_num=current_tile_num,
                                                      add_padding_line=False, consecutive_2_tile_blobs=consecutive_2_tile_blobs,
                                                      prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
                                                      is_current_blob_folding=is_current_blob_folding,
                                                      change_read_write_length=True)

    # Not set for concat
    #assert current_blob.num_lines_written_to_ddr_before_next_blob_read > 0
    
    write_size_bytes = int(math.ceil(ouput_tensor_length/32)*32 * num_of_lines * (256 // 32)) # 256 bytes per 32 channels
    if 'total_tensor_write_bytes' in node['backend']:
        if (current_tile_num, current_xslice_num) in node['backend']['total_tensor_write_bytes']:
            node['backend']['total_tensor_write_bytes'][(current_tile_num, current_xslice_num)]+=write_size_bytes
        else:
            node['backend']['total_tensor_write_bytes'][(current_tile_num, current_xslice_num)] = write_size_bytes
    else:
        node['backend']['total_tensor_write_bytes']={(current_tile_num, current_xslice_num):write_size_bytes}
    
    output_y = actual_tensor_shape[2]
    if (('is_avgPool' in node['frontend']) and node['frontend']['is_avgPool']):
        output_y += 1
    output_z = actual_tensor_shape[1]*(2**tensor_to_write.folding_factor_x)
    grid_mode = node['backend']['gridmode']
    if grid_mode == GridConfig.H14xW8:
        output_x = 8
        if (node['backend']['oc_splits'] > 1):
            amm_write_mode = 1
        else:
            amm_write_mode = pow(2, amm_idx)
        x_wrapping = 0
    else:
        output_x = math.ceil(actual_tensor_shape[3]/(16*(2**tensor_to_write.folding_factor_x)))*16
        amm_write_mode = 0
        x_wrapping = 1
    y_folding = pow(2, tensor_to_write.folding_factor_y)
    amm_start_address=amm_start_address
    amm_nchannels = output_z * y_folding * x_wrapping

    # Check if this tensor is to MXP
    tensor_to_mxp = False
    if ir.sync_with_MXP and tensor_to_write.name in ir.tensors_to_mxp:
        tensor_name = (tensor_to_write.name).split('_')[0]
        if tensor_name in ir.io_tensor_names:
            ddr_write_offset_type = DDRWriteOffsetType.OUTPUT_TENSOR
        else:
            ddr_write_offset_type = DDRWriteOffsetType.MXP_BASE
        tensor_to_mxp = True

    #Yaron - DMA parameter change
#    if DEBUG_OPTIMIZE_DMA_TRANSACTIONS and (x_wrapping < 16) and (output_x > 8) and (ddr_write_offset_type == DDRWriteOffsetType.MODEL_MEM):
#        output_z = int(output_z/(256/output_x))
#        output_x = 256
#        x_wrapping = 16
    if DEBUG_OPTIMIZE_DMA_TRANSACTIONS and (x_wrapping == 1) and (ddr_write_offset_type == DDRWriteOffsetType.MODEL_MEM):
        x_wrapping = 1
        while (output_z > 1) and (x_wrapping <= (MAX_X_WRAPPING//2)) and (output_x < 2048):
            output_z /= 2
            x_wrapping *= 2
            output_x *= 2
        output_z = int(math.ceil(output_z))
        if x_wrapping == 32:
            x_wrapping = 0
        y_tile_ddr_offset = write_start_line * output_x * y_folding
        x_slice_ddr_offset = current_xslice_num * 16 * x_wrapping
    else:
        y_tile_ddr_offset = write_start_line * output_x * y_folding
        x_slice_ddr_offset = current_xslice_num * 16 * x_wrapping
    if override_output_start_line != None:
        write_start_line = override_output_start_line
    in_grid_write_start_line = write_start_line - read_start_line
    ddr_dst_address = tensor_to_write.ddr_entry.address + y_tile_ddr_offset + grid_offset + x_slice_ddr_offset

    if tensor_to_mxp:
        if (current_tile_num == node['frontend']['y_tiles']-1) and (current_xslice_num == node['frontend']['output_tensor'].x_slices-1):
            send_sync_to_MXP = True

            # If this is a tensor that was split, then both splits will write to MXP memory.
            # However, only need to sync on the second.
            if tensor_to_write.name in ir.split_tensor_to_original_tensor_map:
                if 'split2' not in tensor_to_write.name:
                    send_sync_to_MXP = False

            # TSNP set F4 for MXP to wait
            if send_sync_to_MXP:
                set_flags_mask += 16

        # Offset type still stays at 0, but address needs to change
        ddr_dst_address = ir.mxp_tensor_to_offset[tensor_to_write.name] + y_tile_ddr_offset + grid_offset + x_slice_ddr_offset
        ir.output_ddr_offset.add(ddr_dst_address)


    sequencer_entry = SequencerDDRWriteEntry(dst_address=ddr_dst_address, ddr_write_offset_type = ddr_write_offset_type, nlines_to_write=num_of_lines, output_x = output_x, output_y = output_y, output_z = output_z, amm_start_address=amm_start_address,
                                            amm_nchannels = amm_nchannels, grid_start_line=in_grid_write_start_line, src_type = DDRCommandDstSrcType.AMM, y_folding = y_folding, x_wrapping = x_wrapping,
                                            amm_write_mode = amm_write_mode,set_flags = set_flags_mask, wait_flags = wait_flags_mask)
    current_node_sequence.append(sequencer_entry)
    return current_node_sequence

def get_wloc_mask_from_group(wloc_group):
    mask=0
    for grid_idx in wloc_group:
        mask+=1<<grid_idx
    return mask
def get_identical_wloc_groups(grids_wloc_commands):
    wloc_identical_groups_axi0=[]
    wloc_identical_groups_axi1=[]
    grids_to_check = list(range(len(grids_wloc_commands)))
    axi0_grids_to_check=list(set(WLOC_READ_AXI0_GRIDS) & set(grids_to_check))
    axi1_grids_to_check=list(set(WLOC_READ_AXI1_GRIDS) & set(grids_to_check))
    while len(axi0_grids_to_check)>0:
        current_wloc_bytearray = None
        current_group=[]
        axi0_grids_to_check_copy = copy.deepcopy(axi0_grids_to_check)
        for current_grid in axi0_grids_to_check_copy:
            if current_wloc_bytearray == None:
                current_wloc_bytearray = grids_wloc_commands[current_grid].wloc_mem
                current_group.append(current_grid)
                axi0_grids_to_check.remove(current_grid)
            else:
                if current_wloc_bytearray == grids_wloc_commands[current_grid].wloc_mem:
                    current_group.append(current_grid)
                    axi0_grids_to_check.remove(current_grid)
        wloc_identical_groups_axi0.append(current_group)
    while len(axi1_grids_to_check)>0:
        current_wloc_bytearray = None
        current_group=[]
        axi1_grids_to_check_copy = copy.deepcopy(axi1_grids_to_check)
        for current_grid in axi1_grids_to_check_copy:
            if current_wloc_bytearray == None:
                current_wloc_bytearray = grids_wloc_commands[current_grid].wloc_mem
                current_group.append(current_grid)
                axi1_grids_to_check.remove(current_grid)
            else:
                if current_wloc_bytearray == grids_wloc_commands[current_grid].wloc_mem:
                    current_group.append(current_grid)
                    axi1_grids_to_check.remove(current_grid)
        wloc_identical_groups_axi1.append(current_group)
    return wloc_identical_groups_axi0,wloc_identical_groups_axi1

def get_identical_wloc_groups_2wlocs_per_grid(grids_wloc_commands): # This is same as above "get_identical_wloc_groups" but for config of 2 wlocs per each grid (MCHP dot product setup)
    wloc_identical_groups_axi0=[]
    wloc_identical_groups_axi1=[]
    wlocs_to_check = list(range(len(grids_wloc_commands)))
    axi0_wlocs_to_check=list(set(WLOCS_READ_ON_AXI0) & set(wlocs_to_check))
    axi1_wlocs_to_check=list(set(WLOCS_READ_ON_AXI1) & set(wlocs_to_check))
    while len(axi0_wlocs_to_check)>0:
        current_wloc_bytearray = None
        current_group=[]
        axi0_wlocs_to_check_copy = copy.deepcopy(axi0_wlocs_to_check)
        for current_wloc in axi0_wlocs_to_check_copy:
            if current_wloc_bytearray == None:
                current_wloc_bytearray = grids_wloc_commands[current_wloc].wloc_mem
                current_group.append(current_wloc)
                axi0_wlocs_to_check.remove(current_wloc)
            else:
                if current_wloc_bytearray == grids_wloc_commands[current_wloc].wloc_mem:
                    current_group.append(current_wloc)
                    axi0_wlocs_to_check.remove(current_wloc)
        wloc_identical_groups_axi0.append(current_group)
    while len(axi1_wlocs_to_check)>0:
        current_wloc_bytearray = None
        current_group=[]
        axi1_wlocs_to_check_copy = copy.deepcopy(axi1_wlocs_to_check)
        for current_wloc in axi1_wlocs_to_check_copy:
            if current_wloc_bytearray == None:
                current_wloc_bytearray = grids_wloc_commands[current_wloc].wloc_mem
                current_group.append(current_wloc)
                axi1_wlocs_to_check.remove(current_wloc)
            else:
                if current_wloc_bytearray == grids_wloc_commands[current_wloc].wloc_mem:
                    current_group.append(current_wloc)
                    axi1_wlocs_to_check.remove(current_wloc)
        wloc_identical_groups_axi1.append(current_group)
    return wloc_identical_groups_axi0,wloc_identical_groups_axi1

def add_wloc_load_from_ddr_commands(current_node_sequence,node,tables_buffer_id,split_idx,axi0_tables_read_set_flag_mask,axi1_tables_read_set_flag_mask,current_tile_num=0):

        
    #Alex for phase 2 change this
    #TODO: Dans, need to add axi
    # if node['op_type'] in PER_TILE_CBC_OPS:
    #     grids_wloc_commands = node['backend']['grids_cbc'][current_tile_num].wlocs[split_idx]
    # else:
    #     grids_wloc_commands = node['backend']['grids_cbc'].wlocs[split_idx]

    # if DEBUG_MERGE_IDENTICAL_WLOC_READS:
    #     wloc_groups_axi0, wloc_groups_axi1 = get_identical_wloc_groups_2wlocs_per_grid(grids_wloc_commands)
    # else:
    #     wloc_groups_axi0 = list(range(0,len(grids_wloc_commands),2))
    #     wloc_groups_axi0 = list(map(lambda el:[el], wloc_groups_axi0))
    #     wloc_groups_axi1 = list(range(1,len(grids_wloc_commands),2))
    #     wloc_groups_axi1 = list(map(lambda el:[el], wloc_groups_axi1))
    
    
    # to this ----
    # Alex: current_tile_num - is only for Add used, have to do it more smart

    grids_wloc_commands = node['backend']['grids_cbc'].alex_wlocs[0][split_idx] 
      
#    wloc_groups_axi0 = [[0,2],[1,3]] # alex graoupe manualy
    wloc_groups_axi0 = [[0],[1]] # Single grid, 2 WLOC tables
    wloc_groups_axi1 = []
    # ----------



    
    for wloc_group_idx,wloc_group in enumerate(wloc_groups_axi0):
        wloc_mask= get_wloc_mask_from_group(wloc_group)

        #alex for phase 2 change this to 
        #current_wloc = grids_wloc_commands[wloc_group[0]]
        # this
        current_wloc = grids_wloc_commands[wloc_group_idx]

        current_wloc_ddr_address = current_wloc.cmd_mem_address
        wloc_length_bytes = len(current_wloc.cmd_mem)
        if wloc_length_bytes<DDR_MIN_READ_SIZE: # Due to HW limitation minimum read is DDR_MIN_READ_SIZE bytes
            wloc_length_bytes = DDR_MIN_READ_SIZE     
        align_size = math.ceil(wloc_length_bytes / DMR_TABLE_LENGTH_BYTES)
        if ((align_size % 2) == 1):
            align_size = align_size + 1
        wloc_length_bytes = align_size * DMR_TABLE_LENGTH_BYTES
        if ((wloc_length_bytes/DMR_TABLE_LENGTH_BYTES) > MAX_TABLE_LENGTH):
            raise ValueError ('WLOC0 Table Length exceeds MAX_TABLE_LENGTH')
        
        if 'total_wloc_tables_bytes' in node['backend']:
            if current_tile_num in node['backend']['total_wloc_tables_bytes']:
                node['backend']['total_wloc_tables_bytes'][current_tile_num]+=wloc_length_bytes
            else:
                node['backend']['total_wloc_tables_bytes'][current_tile_num] = wloc_length_bytes
        else:
            node['backend']['total_wloc_tables_bytes']={current_tile_num:wloc_length_bytes}
        if DEBUG_ADD_AXI1_WAIT_FLAG: # we set flag for last odd grid wloc read
            if wloc_group_idx == len(wloc_groups_axi0)-1:
                set_flags = axi0_tables_read_set_flag_mask
            else:
                set_flags = 0
        else:
            set_flags = 0
        target_wloctable_info = DDRReadDstInfoWLOC(wloc_mask=wloc_mask,wloc_buffer_id=tables_buffer_id)
        sequencer_entry = SequencerDDRReadEntry(src_address=current_wloc_ddr_address,length = wloc_length_bytes, buffer_id=tables_buffer_id,
                        dst_type = DDRCommandDstSrcType.WLOC, dst_info = target_wloctable_info, set_flags = set_flags, wait_flags = 0)
        current_node_sequence.append(sequencer_entry)
    for wloc_group_idx,wloc_group in enumerate(wloc_groups_axi1):
        wloc_mask= get_wloc_mask_from_group(wloc_group)
        current_wloc = grids_wloc_commands[wloc_group[0]]
        current_wloc_ddr_address = current_wloc.cmd_mem_address
        wloc_length_bytes = len(current_wloc.cmd_mem)
        if wloc_length_bytes<DDR_MIN_READ_SIZE: # Due to HW limitation minimum read is DDR_MIN_READ_SIZE bytes
            wloc_length_bytes = DDR_MIN_READ_SIZE
        align_size = math.ceil(wloc_length_bytes / DMR_TABLE_LENGTH_BYTES)
        if ((align_size % 2) == 1):
            align_size = align_size + 1
        wloc_length_bytes = align_size * DMR_TABLE_LENGTH_BYTES
        if ((wloc_length_bytes/DMR_TABLE_LENGTH_BYTES) > MAX_TABLE_LENGTH):
            raise ValueError ('WLOC1 Table Length exceeds MAX_TABLE_LENGTH')
        
        if 'total_wloc_tables_bytes' in node['backend']:
            if current_tile_num in node['backend']['total_wloc_tables_bytes']:
                node['backend']['total_wloc_tables_bytes'][current_tile_num]+=wloc_length_bytes
            else:
                node['backend']['total_wloc_tables_bytes'][current_tile_num] = wloc_length_bytes
        else:
            node['backend']['total_wloc_tables_bytes']={current_tile_num:wloc_length_bytes}
        if DEBUG_ADD_AXI1_WAIT_FLAG: # we set flag for last odd grid wloc read
            if wloc_group_idx == len(wloc_groups_axi1)-1:
                set_flags = axi1_tables_read_set_flag_mask
            else:
                set_flags = 0
        else:
            set_flags = 0
        target_wloctable_info = DDRReadDstInfoWLOC(wloc_mask=wloc_mask,wloc_buffer_id=tables_buffer_id)
        sequencer_entry = SequencerDDRReadEntry(src_address=current_wloc_ddr_address,length = wloc_length_bytes, buffer_id=tables_buffer_id,
                        dst_type = DDRCommandDstSrcType.WLOC, dst_info = target_wloctable_info, set_flags = set_flags, wait_flags = 0)
        current_node_sequence.append(sequencer_entry)
    return current_node_sequence


def add_table_load_from_ddr_command_alex(table_list, tableType, write_mask, current_node_sequence,node,tables_buffer_id,split_idx =0,set_flags_mask = 0,current_tile_num=0):
   
    
    #grids_rqloc_ir = node['backend']['grids_cbc'].RTable[0]   
    #current_rqloc_ddr_address = grids_rqloc_ir.cmd_mem_address
    #    
    table_list_a_ddr_address = table_list.cmd_mem_address
    
    table_list_length_bytes = len(table_list.cmd_mem)
    if table_list_length_bytes<DDR_MIN_READ_SIZE: # Due to HW limitation minimum read is DDR_MIN_READ_SIZE bytes
        table_list_length_bytes = DDR_MIN_READ_SIZE
    align_size = math.ceil(table_list_length_bytes / DMR_TABLE_LENGTH_BYTES)
    if ((align_size % 2) == 1):
        align_size = align_size + 1
    table_list_length_bytes = align_size * DMR_TABLE_LENGTH_BYTES
  
    #TODO: Dans, need to add axi
    target_read_dst_info_tables = DDRReadDstInfoTables(table_buffer_id=tables_buffer_id,axi_id=0)
    sequencer_entry = SequencerDDRReadEntry(src_address=table_list_a_ddr_address, length = table_list_length_bytes,
                                             buffer_id=tables_buffer_id,               dst_type = tableType, 
                                             dst_info = target_read_dst_info_tables,   set_flags = set_flags_mask, 
                                             wait_flags = 0, write_mask_alex = write_mask)
    current_node_sequence.append(sequencer_entry)
    return current_node_sequence

def add_rqloc_load_from_ddr_command(current_node_sequence,node,tables_buffer_id,split_idx,set_flags_mask = 0,current_tile_num=0):
    # if node['op_type'] in PER_TILE_CBC_OPS:
    #     grids_rqloc_ir = node['backend']['grids_cbc'][current_tile_num].rqlocs[split_idx]
    # else:
    #     grids_rqloc_ir = node['backend']['grids_cbc'].rqlocs[split_idx]
    #Alex for phase 2 change this
    # current_rqloc_ddr_address = grids_rqloc_ir.rqloc_mem_address
    
    # to this
    grids_rqloc_ir = node['backend']['grids_cbc'].RTable[0]   
    current_rqloc_ddr_address = grids_rqloc_ir.cmd_mem_address
    #    
    
    rqloc_length_bytes = len(grids_rqloc_ir.cmd_mem)
    if rqloc_length_bytes<DDR_MIN_READ_SIZE: # Due to HW limitation minimum read is DDR_MIN_READ_SIZE bytes
        rqloc_length_bytes = DDR_MIN_READ_SIZE
    align_size = math.ceil(rqloc_length_bytes / DMR_TABLE_LENGTH_BYTES)
    if ((align_size % 2) == 1):
        align_size = align_size + 1
    rqloc_length_bytes = align_size * DMR_TABLE_LENGTH_BYTES
    if ((rqloc_length_bytes/DMR_TABLE_LENGTH_BYTES) > MAX_TABLE_LENGTH):
        raise ValueError ('RQLOC Table Length exceeds MAX_TABLE_LENGTH')
        
    if 'total_rqloc_table_bytes' in node['backend']:
        if current_tile_num in node['backend']['total_rqloc_table_bytes']:
            node['backend']['total_rqloc_table_bytes'][current_tile_num]+=rqloc_length_bytes
        else:
            node['backend']['total_rqloc_table_bytes'][current_tile_num] = rqloc_length_bytes
    else:
        node['backend']['total_rqloc_table_bytes']={current_tile_num:rqloc_length_bytes}

    #TODO: Dans, need to add axi
    target_rqloctable_info = DDRReadDstInfoTables(table_buffer_id=tables_buffer_id,axi_id=0)
    sequencer_entry = SequencerDDRReadEntry(src_address=current_rqloc_ddr_address,length = rqloc_length_bytes, buffer_id=tables_buffer_id,
                    dst_type = DDRCommandDstSrcType.RQLOC, dst_info = target_rqloctable_info, set_flags = set_flags_mask, wait_flags = 0)
    current_node_sequence.append(sequencer_entry)
    return current_node_sequence

# def add_rqparams_load_from_ddr_command(current_node_sequence,node,tables_buffer_id=0,current_tile_num=0):
#     if node['op_type'] in PER_TILE_CBC_OPS:
#         grids_rqparams_ir = node['backend']['grids_cbc'][current_tile_num].rqparams
#     else:
#         grids_rqparams_ir = node['backend']['grids_cbc'].rqparams

#     #Alex for phase 2 change this
#     # current_rqloc_ddr_address = grids_rqparams_ir.rqparams_mem_address 
    
#     # to this
#     grids_rqparams_ir = node['backend']['grids_cbc'].alex_rqParam[0][0]   
#     current_rqloc_ddr_address = grids_rqparams_ir.cmd_mem_address
#     #

#     rqparams_length_bytes = len(grids_rqparams_ir.cmd_mem)
#     if rqparams_length_bytes<DDR_MIN_READ_SIZE: # Due to HW limitation minimum read is DDR_MIN_READ_SIZE bytes
#         rqparams_length_bytes = DDR_MIN_READ_SIZE
#     align_size = math.ceil(rqparams_length_bytes / DMR_TABLE_LENGTH_BYTES)
#     if ((align_size % 2) == 1):
#         align_size = align_size + 1
#     rqparams_length_bytes = align_size * DMR_TABLE_LENGTH_BYTES
#     if ((rqparams_length_bytes/DMR_TABLE_LENGTH_BYTES) > MAX_TABLE_LENGTH):
#         raise ValueError ('RQPARAMS Table Length exceeds MAX_TABLE_LENGTH')
        
#     if 'total_rqparams_table_bytes' in node['backend']:
#         if current_tile_num in node['backend']['total_rqparams_table_bytes']:
#             node['backend']['total_rqparams_table_bytes'][current_tile_num]+=rqparams_length_bytes
#         else:
#             node['backend']['total_rqparams_table_bytes'][current_tile_num] = rqparams_length_bytes
#     else:
#         node['backend']['total_rqparams_table_bytes']={current_tile_num:rqparams_length_bytes}
#     #TODO: Dans, need to add axi
#     target_rqloctable_info = DDRReadDstInfoTables(table_buffer_id=tables_buffer_id,axi_id=0)
#     sequencer_entry = SequencerDDRReadEntry(src_address=current_rqloc_ddr_address,length = rqparams_length_bytes, buffer_id=tables_buffer_id,
#                     dst_type = DDRCommandDstSrcType.RQPARAMS, dst_info = target_rqloctable_info, set_flags = 0, wait_flags = 0)
#     current_node_sequence.append(sequencer_entry)
#     return current_node_sequence


def add_engine_command(ir,current_node_sequence,node, set_flags_mask = 0, wait_flags_mask = 0,single_split_tables_buffer_id=0, splitted_tables_buffer_id=0,
                       reset_amm_address=1,description='',current_tile_num=0, current_xslice_num=0, dummy_folding_command=False):

    if node['op_type'] == 'Add' or ('AVERAGEPOOL2D' in node['name']):
        input_pad_value = 0
    else:
        input_pad_value = node['frontend']['input_tensor_zp']
    if ('activation_silu' in node['attributes']) and (node['attributes']['activation_silu'] != None):
        output_pad_value = node['attributes']['activation_silu']['output_zp'][0]
    else:
        output_pad_value = node['frontend']['output_tensor_zp']

    ### Output Padding X ####################
    if current_xslice_num == node['frontend']['x_slices']-1:
        output_padding_start_x = node['backend']['output_padding_start_x']
    else:
        output_padding_start_x = 15

    ## Output Padding Y  ####################    
    num_y_tiles = node['frontend']['y_tiles']
    if dummy_folding_command: # The dummy slice need for the next
        output_padding_start_y = 0
    elif current_tile_num==(num_y_tiles-1):
        output_padding_start_y = node['backend']['output_padding_start_y']
    else:
        output_padding_start_y = AMM_HEIGHT

    grid_mode = node['backend']['gridmode']
    if grid_mode == GridConfig.H14xW8:
        pad_mode = PadMode.GRIDH14XW8
    elif grid_mode == GridConfig.H14xW16:
        pad_mode = PadMode.GRIDH14XW16
    elif grid_mode == GridConfig.H14xW32:
        pad_mode = PadMode.GRIDH14XW32
    else:
        raise ValueError ('Grid mode not supported!!!')
    
    if ('input_tensors' in node['frontend']):
        input_tensor = node['frontend']['input_tensors'][0]
    else:
        input_tensor = node['frontend']['input_tensor']

    input_index=0
    current_blob_idx = node['frontend']['tiling_blob_idx']
    current_blob = ir.tiling_blobs[current_blob_idx]
    if (input_tensor.num_packed_xslices > 1):
        num_xslices = node['frontend']['x_slices'] // input_tensor.num_packed_xslices
        xslice_idx = current_xslice_num // input_tensor.num_packed_xslices
        xslice_idx_offset = current_xslice_num % input_tensor.num_packed_xslices
        amm_start_address_for_input_even_grid = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][xslice_idx][input_index][0][0]*URAM_BLOCK_SIZE + (xslice_idx_offset*16)
        if (current_xslice_num == 0 and num_xslices > 1):
            slice_backward_offset = NO_NEIGHBOR_SLICE
            slice_forward_offset = amm_start_address_for_input_even_grid + 16
        elif (current_xslice_num == 0 and num_xslices == 1):
            slice_backward_offset = NO_NEIGHBOR_SLICE
            slice_forward_offset = NO_NEIGHBOR_SLICE
        elif (current_xslice_num == node['frontend']['x_slices']-1) :
            slice_backward_offset = amm_start_address_for_input_even_grid - 16
            slice_forward_offset =  NO_NEIGHBOR_SLICE
        elif dummy_folding_command:
            slice_backward_offset = NO_NEIGHBOR_SLICE
            slice_forward_offset = NO_NEIGHBOR_SLICE
        else:
            if (xslice_idx_offset == 0):
                slice_backward_offset = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][xslice_idx-1][input_index][0][0]*URAM_BLOCK_SIZE + (input_tensor.num_packed_xslices-1)*16
            else:
                slice_backward_offset = amm_start_address_for_input_even_grid - 16
            if (xslice_idx_offset == input_tensor.num_packed_xslices - 1):
                slice_forward_offset = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][xslice_idx+1][input_index][0][0]*URAM_BLOCK_SIZE
            else:
                slice_forward_offset = amm_start_address_for_input_even_grid + 16
    else:
        amm_start_address_for_input_even_grid = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][current_xslice_num][input_index][0][0]*URAM_BLOCK_SIZE
        if (current_xslice_num == 0 and node['frontend']['x_slices'] > 1):
            slice_backward_offset = NO_NEIGHBOR_SLICE
            slice_forward_offset = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][current_xslice_num+1][input_index][0][0]*URAM_BLOCK_SIZE
        elif (current_xslice_num == 0 and node['frontend']['x_slices'] == 1):
            slice_backward_offset = NO_NEIGHBOR_SLICE
            slice_forward_offset = NO_NEIGHBOR_SLICE
        elif (current_xslice_num == node['frontend']['x_slices']-1) :
            slice_backward_offset = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][current_xslice_num-1][input_index][0][0]*URAM_BLOCK_SIZE
            slice_forward_offset =  NO_NEIGHBOR_SLICE
        elif dummy_folding_command:
            slice_backward_offset = NO_NEIGHBOR_SLICE
            slice_forward_offset = NO_NEIGHBOR_SLICE
        else:
            slice_backward_offset = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][current_xslice_num-1][input_index][0][0]*URAM_BLOCK_SIZE
            slice_forward_offset = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][current_xslice_num+1][input_index][0][0]*URAM_BLOCK_SIZE

    if node['backend']['deepconv'] or DEBUG_FORCE_DEEP_CONV:
        operation  = EngineOpCode.DEEPCONV
    elif node['op_type'] == 'MaxPool':
        operation = EngineOpCode.MAXPOOL
    else:
        operation  = EngineOpCode.CONV
    scaling_mode=ScalingMode.RQDIRECT
    if 'is_hw_x_resize' in node['frontend']:
        if scaling_mode!=ScalingMode.RQDIRECT:
            raise ValueError ('Op cant be both folding and resizing. Please check...')
        scaling_mode=ScalingMode.RESIZE1_2
    
    fold_left_right = []
    amm_start_address_for_output_even_grid = []
    if 'force_unfolding_x' in node['frontend']:
        scaling_mode=ScalingMode.UNFOLDING
        slices_ratio = int(node['frontend']['output_tensor'].x_slices / node['frontend']['x_slices'])
        for i in range(slices_ratio):
            amm_start_address_for_output_even_grid.append(node['backend']['allocated_amm_blocks_for_output_even_grid'][current_tile_num][slices_ratio*current_xslice_num+i][0][0]*URAM_BLOCK_SIZE)
            fold_left_right.append(int(i % 2))
        current_xslice_offset = slices_ratio*current_xslice_num
    elif 'force_folding_x' in node['frontend']:
        slices_ratio = 1
        scaling_mode=ScalingMode.FOLDING2_1
        amm_start_address_for_output_even_grid.append(node['backend']['allocated_amm_blocks_for_output_even_grid'][current_tile_num][int(current_xslice_num/2)][0][0]*URAM_BLOCK_SIZE)
        current_xslice_offset = int(current_xslice_num/2)
        fold_left_right.append(int((current_xslice_num+dummy_folding_command) % 2))
    else:
        slices_ratio = 1
        amm_start_address_for_output_even_grid.append(node['backend']['allocated_amm_blocks_for_output_even_grid'][current_tile_num][current_xslice_num][0][0]*URAM_BLOCK_SIZE)
        fold_left_right.append(0)
        current_xslice_offset = current_xslice_num

    for idx in range(len(amm_start_address_for_output_even_grid)):
        if slices_ratio != 1:
            slice_num = description.split('_')[-1]
            slice_num = re.findall(r'\d+', slice_num)
            index = description.rfind(slice_num[0])
            description = description[:index]
            description += str(int(current_xslice_offset+idx))
        
        if (int(slices_ratio*current_xslice_num+idx) != 0) or dummy_folding_command:
            wait_flags_mask = wait_flags_mask & ~(1 << 0)
                    
        sequencer_entry = SequencerEngineOperationEntry(operation=operation, AMM_URAM_EVEN_GRID_src_addr=amm_start_address_for_input_even_grid,
                AMM_URAM_EVEN_GRID_dst_addr=amm_start_address_for_output_even_grid[idx], output_padding_start_x=output_padding_start_x,
                output_padding_start_y=output_padding_start_y,output_pad_value=output_pad_value,input_pad_value=input_pad_value,pad_mode=pad_mode,
                slice_backward_offset = slice_backward_offset, slice_forward_offset = slice_forward_offset, fold_left_right=fold_left_right[idx],
                wloc_buffer_id = splitted_tables_buffer_id, rt_buffer_id = splitted_tables_buffer_id, rqparams_buffer_id = splitted_tables_buffer_id, nlt_buffer_id = single_split_tables_buffer_id,
                scaling_mode=scaling_mode,reset_amm_address=reset_amm_address,set_flags=set_flags_mask,wait_flags=wait_flags_mask,description=description)

        # We add the below to sequencer engine command metadata so that it can be used by FPGA simulator to gather intermediate tensors tiles and compare to numeric simulator nxo files
        current_blob_idx = node['frontend']['tiling_blob_idx']
        current_blob = ir.tiling_blobs[current_blob_idx]
        k3_nodes_in_blob = current_blob.k3_nodes
        output_tensor = node['frontend']['output_tensor']
        folded_tensor_shape = output_tensor.get_folded_shape()
        
        # check if the Padding line is needed - if output height is odd or input height is odd (for VALID padding)
        # add_padding_line = (node['op_type'] == "Conv"    and
        #                     (((output_tensor.shape[2]%2==1) and (('is_avgPool' in node['frontend']) and node['frontend']['is_avgPool'])) or 
        #                      (node['frontend']['input_tensor'].shape[2]%2==1))) 
        # if (node['op_type'] == "Conv") and \
        #     (output_tensor.shape[2]%2==1) and (('is_avgPool' in node['frontend']) and node['frontend']['is_avgPool']):
        #     folded_tensor_shape[2] += 1
        
        num_of_lines,read_start_line,write_start_line = get_num_of_lines_per_tile(folded_tensor_shape[2],k3_nodes_in_blob=k3_nodes_in_blob,
                                                        num_y_tiles=node['frontend']['y_tiles'],current_tile_num=current_tile_num,add_padding_line=False)
        sequencer_entry.num_of_lines = num_of_lines
        sequencer_entry.write_start_line = write_start_line
        sequencer_entry.read_start_line = read_start_line
        if node['op_type'] in MULTIPLE_INPUT_OPS:
            folded_input_channels = node['frontend']['input_tensors'][0].get_folded_shape()[1]
        else:
            folded_input_channels = node['frontend']['input_tensor'].get_folded_shape()[1]
        sequencer_entry.conv_input_channels = folded_input_channels
        current_node_sequence.append(sequencer_entry)
    return current_node_sequence

def get_latest_preceding_node(following_node,ir):
    lexicographical_topological_sorted_nodes = ir.lexicographical_topological_sorted_graph
    latest_preceding_node_name=''
    latest_index=-1
    for current_preceding_node_params in following_node['frontend']['preceding_nodes_params']:
        current_preceding_node_name = current_preceding_node_params[0] # param 0 is the nodes name
        current_preceding_node_index = lexicographical_topological_sorted_nodes.index(current_preceding_node_name)
        if current_preceding_node_index>latest_index:
            latest_index = current_preceding_node_index
            latest_preceding_node_name = current_preceding_node_name
    return latest_preceding_node_name

def read_needed_tensors_from_ddr(ir,tensors_info,node,wait_flags_allocator,current_node_sequence,single_split_tables_buffer_id,current_tile_num=0,current_xslice_num=0,
                                 read_ddr_wait_flag = 0,read_ddr_set_flag = 0,is_immediate_read = False, conv_in_process=False,tile_write_in_process=False):                                 
    # Check if any input tensors need to be loaded from ddr
    read_ddr_to_amm_set_flag_mask = 0
    read_in_process = False
    tensors_read_during_this_split=0
    number_of_tensors_read_from_ddr=0
    is_first_read_in_tile = True
    for tensor_idx,tensor_to_load_from_ddr in enumerate(tensors_info):
        if tensor_to_load_from_ddr.load_at_execution_of_tile_idx == current_tile_num:
            number_of_tensors_read_from_ddr+=1
            tensors_read_during_this_split+=1
            input_index = tensor_to_load_from_ddr.input_index
            read_in_process = True
            tensor_to_load_from_ddr_name = tensor_to_load_from_ddr.name
            if tensor_to_load_from_ddr_name in ir.inputs:
                ddr_read_offset_type=DDRReadOffsetType.INPUT_TENSOR
            else:
                ddr_read_offset_type=DDRReadOffsetType.MODEL_MEM
            # No need to condition intermediate tensor read by its write flag since we alreaedy made sure it was written to ddr at the offloading node
            # Since DDR read command is blocking we dont need wmt wait flag as the ddr read to amm will not start until wmt read ends anyway
            #wmt_read_set_flag = wait_flags_allocator.allocate_flag(description=node['name'] + ' ddr to wmt read.')
            #wmt_read_set_flag_mask = wait_flags_allocator.get_mask_from_flags([wmt_read_set_flag])
            ddr_read_wait_flag_mask = 0
            is_first_op_first_tile = (ir.lexicographical_topological_sorted_graph.index(node['name'])==0) and (current_tile_num==0) and (tensor_to_load_from_ddr.tensor_xslice_num==0)
            ddr_read_wait_flags = []
            if not is_first_op_first_tile: # If its first op first tile we dont need to wait since nothing happend before
                if is_immediate_read:
                    ddr_read_wait_flags.append(read_ddr_set_flag) # We need to wait for previous op to finish since mem is full (this is the reason we offloaded this tensor)
                    if tile_write_in_process and number_of_tensors_read_from_ddr==1: # If its an immediate read and tile write is in process we need to wait for it to finish before starting read. This is relevant only for the 1st tensor we read
                        ddr_read_wait_flags.append(ir.global_ddr_transaction_flag)
                        # We clean the tile_write_in_process at end of all tensors read since we want each read to wait for last read
                else:
                    if tile_write_in_process:
                        ddr_read_wait_flags.append(read_ddr_set_flag) #If its a tile read and write is in process we need to wait for it to finish
                        # We clean the tile_write_in_process at end of all tensors read since we want each read to wait for last read
            if DEBUG_AVOID_DDR_READ_WHILE_CONV and conv_in_process:
                ddr_read_wait_flags = [read_ddr_wait_flag]
                if tensors_read_during_this_split!=1:
                    ddr_read_wait_flags.append(read_ddr_set_flag) # If DDR reads are blocking and its not the 1st tensor we read we wait for last tensor read to finishe
            #if wait_for_prev_op:
            #    ddr_read_wait_flags.append(read_ddr_set_flag) # This will make sure we wait for last tile write to finish
            wmt_read_ready_flag = read_ddr_set_flag # We use same flag for both read ddr set flag and wmt read ready flag
            wmt_read_ready_flag_mask = wait_flags_allocator.get_mask_from_flags([wmt_read_ready_flag])

            read_ddr_to_amm_set_flag_mask = wait_flags_allocator.get_mask_from_flags([read_ddr_set_flag])
            read_ddr_to_amm_wait_flag_mask = wait_flags_allocator.get_mask_from_flags(ddr_read_wait_flags)
            node['backend']['read_ddr_to_amm_set_flag_mask'] = read_ddr_to_amm_set_flag_mask

            # We always wait for last op to complete or current op write to ddr to complete
            # We dont allow read during last op conv since we assume that tensors were offloaded to DDR because of lack of mem.
            # Only after last op is complete, its input tensors can be deallocated and allow space for current op input read from DDR
            consumer_node = tensor_to_load_from_ddr.consumer_node # Since load from ddr can be executed by node X(node) but actually load input of node Y(consumer_node) we need to get wmt and all tensor data from consumer node
            grids_to_read = get_grids_per_line(gridconfig=consumer_node['backend']['gridmode'])
            #grids_to_read = get_num_virtual_grids(gridconfig=consumer_node['backend']['gridmode'])
            for current_grid in range(grids_to_read):
                #Yaron - Only first grid in first tensor within each tile needs to have a wait mask
                if (current_grid==0) and is_first_read_in_tile:
                    wait_mask = read_ddr_to_amm_wait_flag_mask
                    if DEBUG_AVOID_DDR_READ_WHILE_CONV:
                        wmt_read_set_flag_mask = wait_flags_allocator.get_mask_from_flags([wmt_read_ready_flag,read_ddr_wait_flag]) # When DEBUG_AVOID_DDR_READ_WHILE_CONV we wait in first wmt read for conv to finish and also set local_wait to indicate that last command was finished
                    else:
                        wmt_read_set_flag_mask = wait_flags_allocator.get_mask_from_flags([wmt_read_ready_flag])
                else:
                    wait_mask = 0
                    wmt_read_set_flag_mask = wait_flags_allocator.get_mask_from_flags([wmt_read_ready_flag])
                #last_read_ddr_to_amm = (current_grid==(grids_to_read-1)) and (tensor_idx==(len(tensors_info)-1))
                #Yaron - Last ddr read is when last grid and last tensor in this specific tile
                last_read_ddr_to_amm = False
                if current_grid==(grids_to_read-1):
                    if tensor_idx==(len(tensors_info)-1):
                        last_read_ddr_to_amm = True
                    elif tensors_info[tensor_idx+1].load_at_execution_of_tile_idx!=current_tile_num:
                        last_read_ddr_to_amm = True
                if last_read_ddr_to_amm:
                    set_flags = read_ddr_to_amm_set_flag_mask
                else:
                    set_flags = 0
                if not DEBUG_ADD_WAIT_FOR_WMT_READ:
                    wmt_read_set_flag_mask = wmt_read_set_flag_mask
                    wmt_read_ready_flag_mask = wait_mask

                current_node_sequence = add_input_load_from_ddr_commands(ir, tensor_to_load_from_ddr,current_node_sequence,consumer_node, set_flags_mask = set_flags,
                        wait_flags_mask = wmt_read_ready_flag_mask,target_grid=current_grid,ddr_read_offset_type=ddr_read_offset_type,
                        current_tile_num=current_tile_num,read_padding_line=False, current_xslice_num=tensor_to_load_from_ddr.tensor_xslice_num) # If input to layer is in ddr add load ddr commands
            is_first_read_in_tile = False
    if DEBUG_AVOID_DDR_READ_WHILE_CONV and read_in_process:
        read_in_process = False
        write_ddr_from_amm_set_flag_mask = read_ddr_to_amm_set_flag_mask
        node['backend']['write_ddr_from_amm_set_flag_mask'] = write_ddr_from_amm_set_flag_mask
        if read_ddr_set_flag==2:
            sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags = read_ddr_to_amm_set_flag_mask, wait_flags = read_ddr_to_amm_set_flag_mask)
        else:
            sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags = 0, wait_flags = read_ddr_to_amm_set_flag_mask)
        current_node_sequence.extend([sequencer_entry])
    if tile_write_in_process and read_in_process: # We clean the tile_write_in_process at end of all tensors read since we want each read to wait for last read.
                                                  # We also check that read_in_process, to make sure that indeed at least one tensor was read (tensor_to_load_from_ddr.load_at_execution_of_tile_idx == current_tile_num)
        tile_write_in_process = False
                
    return read_in_process, tile_write_in_process          

# Get the command to load the next instructions from DDR
def get_command_to_load_next_instructions(set_mask, start_address, num_instructions, sequencer_program_ddr_address):
    return SequencerDDRReadEntry(
        src_address = start_address * BYTES_IN_SEQUENCER_COMMAND + sequencer_program_ddr_address,
        length = num_instructions * BYTES_IN_SEQUENCER_COMMAND,
        dst_type = DDRCommandDstSrcType.PROGRAM,
        dst_info = DDRReadDstInfoProgram(program_mem_start_address=start_address * BYTES_IN_SEQUENCER_COMMAND), # Seems to be unused
        set_flags = set_mask,
        wait_flags = 0)

# Add special operation / no op that waits on flag set above to complete
def get_nop_command_to_wait_for_flag(wait_mask):
    return SequencerCustomOperationEntry(
        operation=CustomeOpCode.NOP,
        set_flags = 0,
        wait_flags = wait_mask)

# Split sequencer program commands list into groups of instructions
def split_sequencer_program(ir, sequencer_program_ddr_address):
    instruction_limit = PROGRAM_MEMORY_NUM_COMMANDS
    original_sequencer_program = ir.sequencer_program.commands_list
    new_sequencer_program = []

    instruction_count = 0
    for original_instruction_index, original_instruction in enumerate(original_sequencer_program):
        new_sequencer_program.append(original_instruction)
        instruction_count += 1
        if instruction_count == instruction_limit-2:

            # Load the next (instruction_limit) instructions, or fewer if there are not that many
            num_instructions_to_load = instruction_limit
            num_original_instructions_remaining = len(original_sequencer_program) - original_instruction_index - 1
            if num_original_instructions_remaining < instruction_limit:
                num_instructions_to_load = num_original_instructions_remaining
                # Use multiple of 2 lines (required by hardware)
                if num_instructions_to_load % 2 == 1:
                    num_instructions_to_load += 1

            # Add DMA request to read the new instructions. Set F1.
            load_new_instructions_command = get_command_to_load_next_instructions(
                set_mask=2, start_address=len(new_sequencer_program)+2, num_instructions=num_instructions_to_load,
                sequencer_program_ddr_address=sequencer_program_ddr_address)
            new_sequencer_program.append(load_new_instructions_command)

            # Add special operation / no op that waits on flag set above (F1) to complete
            nop_command = get_nop_command_to_wait_for_flag(wait_mask=2)
            new_sequencer_program.append(nop_command)

            # Reset instruction count
            instruction_count = 0

    ir.sequencer_program.commands_list = new_sequencer_program

# Get the command to load zeros from DDR into the final block of each AMM grid
def get_load_zeros_from_ddr_command(ir):
    # Get the previously-created tensor with zeros
    zero_tensor = ir.tensors['SYSTEM_zero_tensor']

    # Write only the last block
    amm_start_address = (URAM_NUM_BLOCKS - 1) * URAM_BLOCK_SIZE
    input_amm_info = DDRReadDstInfoAMM(amm_start_address=amm_start_address)

    # Create the DDR read command
    sequencer_entry = SequencerDDRReadEntry(
        src_address = zero_tensor.ddr_entry.address,
        ddr_read_offset_type = DDRReadOffsetType.MODEL_MEM,
        amm_mask = 3,
        amm_write_mode = 0,
        amm_start_address = amm_start_address,
        nlines_to_read = 14,
        input_x = 16,
        input_y = 14,
        input_z = URAM_BLOCK_SIZE,
        y_folding = 1,
        x_wrapping = 1,
        amm_nchannels = URAM_BLOCK_SIZE,
        dst_type = DDRCommandDstSrcType.AMM,
        dst_info = input_amm_info,
        set_flags = 0,
        wait_flags = 0)

    return sequencer_entry

# Return a sequence of commands at the start of the sequencer program
def generate_initial_command_sequence(ir: internal_representation.IR):
    initial_command_sequence = []

    # Add command to load zeros from DDR into last (64th) block of each AMM
    load_zeros_command = get_load_zeros_from_ddr_command(ir)
    initial_command_sequence.append(load_zeros_command)

    return initial_command_sequence

def generate_layer_command_sequence(ir: internal_representation.IR,node,wait_flags_allocator,single_split_tables_buffer_id,
                                    splitted_tables_buffer_allocator,current_tile_num=0,tile_read_in_process = False, tile_write_in_process = False,
                                    consecutive_2_tile_blobs=1, prev_blob_lines_in_ddr=-1,
                                    conv_in_process = False):
    op_type = node['op_type']

    axi0_tables_load_flag = ir.axi0_tables_load_flag
    axi1_tables_load_flag = ir.axi1_tables_load_flag
    local_ddr_transaction_flag = ir.local_ddr_transaction_flag
    global_ddr_transaction_flag = ir.global_ddr_transaction_flag

    current_node_sequence = []
    # Need to add code for knowing which blob we are and if we are 1st node in blob, add read comman, if we are last add write command
    current_blob_idx = node['frontend']['tiling_blob_idx']
    current_tiling_blob = ir.tiling_blobs[current_blob_idx]
    current_tiling_blob_nodes = current_tiling_blob.nodes_in_blob
    first_node_in_blob = False
    last_node_in_blob = False
    if node['name'] == current_tiling_blob_nodes[0]:
        first_node_in_blob = True
    if node['name'] == current_tiling_blob_nodes[-1]:
        last_node_in_blob = True
    last_blob = current_blob_idx == (len(ir.tiling_blobs)-1)
    last_tile = current_tile_num == (current_tiling_blob.y_tiles-1)
    last_tile_of_last_node_in_last_blob = (last_node_in_blob and last_tile and last_blob)
    if op_type == 'Concat':
        current_op_wait_flags = []
        read_ddr_to_amm_set_flag = None            
        if 'tensors_to_load_immediately_from_ddr' in node['backend']:
            tensors_info = node['backend']['tensors_to_load_immediately_from_ddr']
            if len(tensors_info)>0:
                read_ddr_to_amm_set_flag = local_ddr_transaction_flag
                read_in_process, tile_write_in_process = read_needed_tensors_from_ddr(ir,tensors_info,node,wait_flags_allocator,current_node_sequence,0,
                                                current_tile_num=current_tile_num,read_ddr_wait_flag=read_ddr_to_amm_set_flag,
                                                read_ddr_set_flag=read_ddr_to_amm_set_flag,is_immediate_read=True,conv_in_process=conv_in_process,
                                                tile_write_in_process = tile_write_in_process)
                # In concat op there is no actual op so we insert a wait op to make sure needed tensors are in mem
                # We then set back the flag since next op will be waiting on it
                if read_in_process:
                    read_ddr_to_amm_set_flag_mask = wait_flags_allocator.get_mask_from_flags([read_ddr_to_amm_set_flag]) 
                    sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags = 0, wait_flags = read_ddr_to_amm_set_flag_mask)
                    current_node_sequence.extend([sequencer_entry])
                    current_op_wait_flags.append(read_ddr_to_amm_set_flag)
                    sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags = read_ddr_to_amm_set_flag_mask, wait_flags = 0)
                    current_node_sequence.extend([sequencer_entry])
                    current_op_wait_flags.append(read_ddr_to_amm_set_flag)
    else:
        axi0_tables_read_set_flag_mask = wait_flags_allocator.get_mask_from_flags([axi0_tables_load_flag])
        if DEBUG_ADD_AXI1_WAIT_FLAG:
            axi1_tables_read_set_flag_mask = wait_flags_allocator.get_mask_from_flags([axi1_tables_load_flag])
            if node['op_type'] in LIMITED_GRIDS_OPS and LIMITED_GRIDS_OPS[node['op_type']]==1:
                current_op_wait_flags = [axi0_tables_load_flag] # Single GRID ops use only grid 0 so we dont read tables to odd grids using axi1
            else:
                current_op_wait_flags = [axi0_tables_load_flag,axi1_tables_load_flag]
        else:
            axi1_tables_read_set_flag_mask = 0
            current_op_wait_flags = [axi0_tables_load_flag]
        intermediate_tensor_read_wait_flag = None
        read_ddr_to_amm_set_flag = None
        write_intermediate_from_amm_to_ddr_set_flag = None
        if DEBUG_SKIP_CBC_GENERATION:
            num_wloc_splits = 1
        else:
            num_wloc_splits = len(node['backend']['grids_cbc'].alex_wlocs[0])

        # Check if any tensors need to be offloaded to DDR
        if 'optimal_offload_point_for_tensor' in node['backend']:
            offloaded_tensor_name = node['backend']['optimal_offload_point_for_tensor']

            # We use the flag allocated for ddr write (making sure its producer op finished) and deallocate it
            ddr_write_intermediate_tensor_wait_flag_mask = wait_flags_allocator.get_mask_from_flags([local_ddr_transaction_flag])

            offloaded_tensor = ir.tensors[offloaded_tensor_name]
            offloaded_tensor_producer_node_name = offloaded_tensor.producer
            offloaded_tensor_producer_node = ir.graph.nodes()[offloaded_tensor_producer_node_name]
            offloaded_tensor_producer_node_grid_mode = offloaded_tensor_producer_node['backend']['gridmode']
            offloaded_tensor_producer_is_folding_conv = 'folding_conv' in offloaded_tensor_producer_node['frontend']
            if offloaded_tensor_producer_is_folding_conv:
                if offloaded_tensor_producer_node_grid_mode == GridConfig.H28xW28:
                    intermediate_tensor_grid_mode = GridConfig.H14xW14
                else:
                    intermediate_tensor_grid_mode = offloaded_tensor_producer_node_grid_mode
            else:
                intermediate_tensor_grid_mode = offloaded_tensor_producer_node_grid_mode
            
            write_intermediate_from_amm_to_ddr_set_flag_mask = wait_flags_allocator.get_mask_from_flags([local_ddr_transaction_flag])
            node['backend']['write_intermediate_from_amm_to_ddr_set_flag_mask'] = write_intermediate_from_amm_to_ddr_set_flag_mask

            grids_to_write = get_grids_per_line(gridconfig=node['backend']['gridmode'])
            #grids_to_write = get_num_virtual_grids(gridconfig=node['backend']['gridmode'])
            for current_grid in range(grids_to_write):
                if current_grid==0:
                    wait_flags = ddr_write_intermediate_tensor_wait_flag_mask
                else:
                    wait_flags = 0
                last_write_ddr_from_amm = (current_grid==(grids_to_write-1))
                if last_write_ddr_from_amm:
                    set_flags = write_intermediate_from_amm_to_ddr_set_flag_mask
                else:
                    set_flags = 0
                
                for current_xslice_num in range(offloaded_tensor_producer_node['frontend']['output_tensor'].x_slices):
                    set_mask = 0
                    if (current_xslice_num == offloaded_tensor_producer_node['frontend']['output_tensor'].x_slices-1):
                        set_mask = set_flags
                    wait_mask = 0
                    if (current_xslice_num == 0):
                        wait_mask = wait_flags
                    current_node_sequence = add_tensor_write_to_ddr_commands(ir,offloaded_tensor,current_node_sequence,offloaded_tensor_producer_node,set_flags_mask = set_mask,
                            consecutive_2_tile_blobs=consecutive_2_tile_blobs,
                            prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
                            wait_flags_mask = wait_mask,source_grid=current_grid,add_padding_line=False,current_xslice_num=current_xslice_num) # If output from layer should be written to ddr (e.g. workload output) add write ddr commands

            if DEBUG_AVOID_OFFLOAD_TENSOR_DDR_WRITE_WHILE_CONV: # If this flag is on we will not start the current conv until intermediate tensor written to DDR
                # We put write_intermediate_from_amm_to_ddr_set_flag_mask also as set_flags since its also used for previous conv execution finished signaling
                sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags = write_intermediate_from_amm_to_ddr_set_flag_mask, wait_flags = write_intermediate_from_amm_to_ddr_set_flag_mask)
                current_node_sequence.append(sequencer_entry)

        #alex del for Phase 2 
        #current_node_sequence = add_rqparams_load_from_ddr_command(current_node_sequence,node,tables_buffer_id=single_split_tables_buffer_id,current_tile_num=current_tile_num) # Add the commands to load RQPARAMS from DDR
        
        splitted_tables_buffer_id = None
        is_first_op = ir.lexicographical_topological_sorted_graph.index(node['name'])==0
        op_set_flag = local_ddr_transaction_flag

        for split_idx in range(num_wloc_splits):
            allocated_buffers_of_splitted_tables = splitted_tables_buffer_allocator.get_allocated_buffers_indexes()
            splitted_tables_buffer_id = splitted_tables_buffer_allocator.allocate_buffer()
            if len(allocated_buffers_of_splitted_tables)>1:
                raise ValueError ('Didnt expect all buffers to be allocated!')
            elif len(allocated_buffers_of_splitted_tables) == 1:
                    splitted_tables_buffer_allocator.deallocate_buffer(allocated_buffers_of_splitted_tables[0])

            #1xNLF (Itamar asks to load the NLF before all others tablesd)
            tableType = DDRCommandDstSrcType.NONLINEARFUNCTION
            table_list = node["backend"]["grids_cbc"].nlf[0]
            write_mask = 0
            if split_idx==0:
                current_node_sequence = add_table_load_from_ddr_command_alex(table_list, tableType, write_mask, current_node_sequence,node,tables_buffer_id=single_split_tables_buffer_id,current_tile_num=current_tile_num) # Add the commands to load RQPARAMS from DDR



            # We first add all the table read commands: wloc, rqloc, rqparams
            current_node_sequence = add_wloc_load_from_ddr_commands(current_node_sequence,node,splitted_tables_buffer_id,split_idx,0,axi1_tables_read_set_flag_mask, current_tile_num=current_tile_num) # Add the commands to load WLOC split
            #laex changed this to
            #current_node_sequence = add_rqloc_load_from_ddr_command(current_node_sequence,node,splitted_tables_buffer_id,split_idx,
            #                                                        set_flags_mask=axi0_tables_read_set_flag_mask,current_tile_num=current_tile_num) # Add the commands to load RQLOC split from DDR
            # this --------------------------------
            #2xRQPARAM
            tableType = DDRCommandDstSrcType.RQPARAMS
            table_list = node["backend"]["grids_cbc"].alex_rqParam[split_idx][0]
            write_mask = 1
            current_node_sequence = add_table_load_from_ddr_command_alex(table_list, tableType, write_mask, current_node_sequence,node,tables_buffer_id=splitted_tables_buffer_id,current_tile_num=current_tile_num) # Add the commands to load RQPARAMS from DDR

            #1xRT (Alex: have to be last because of set_flags_mask settings - they are important for last table)
            tableType = DDRCommandDstSrcType.RESULTTABLE
            table_list = node["backend"]["grids_cbc"].RTable[split_idx]
            write_mask = 0
            current_node_sequence = add_table_load_from_ddr_command_alex(table_list, tableType, write_mask, current_node_sequence,node,tables_buffer_id=splitted_tables_buffer_id, set_flags_mask = axi0_tables_read_set_flag_mask, current_tile_num=current_tile_num) # Add the commands to load RQPARAMS from DDR



            ### ----------------------------------

            if split_idx==0:
                # Check if any input tensors need to be loaded from ddr
                read_ddr_to_amm_set_flag = None            
                if 'tensors_to_load_immediately_from_ddr' in node['backend']:
                    tensors_info = node['backend']['tensors_to_load_immediately_from_ddr']
                    if len(tensors_info)>0:
                        read_ddr_to_amm_set_flag = local_ddr_transaction_flag
                        read_in_process, tile_write_in_process = read_needed_tensors_from_ddr(ir,tensors_info,node,wait_flags_allocator,current_node_sequence,single_split_tables_buffer_id,
                                                                                current_tile_num=current_tile_num,read_ddr_wait_flag=read_ddr_to_amm_set_flag,
                                                                                read_ddr_set_flag=read_ddr_to_amm_set_flag,is_immediate_read=True,
                                                                                conv_in_process=conv_in_process,tile_write_in_process = tile_write_in_process)
                        if read_in_process or not (is_first_op and current_tile_num==0):
                            current_op_wait_flags.append(read_ddr_to_amm_set_flag)

                if 'tensors_to_offload_to_ddr' in node['backend']: # If this node needs to offload tensors, make sure its execution starts only after offloaded tensors were written to DDR
                    tensors_to_offload_to_ddr = node['backend']['tensors_to_offload_to_ddr']
                    if len(tensors_to_offload_to_ddr)>1:
                        raise ValueError ('Currently only single tensor offload is supported')
                    offloaded_tensor_name = tensors_to_offload_to_ddr[0].name
                    if not DEBUG_AVOID_DDR_WRITE_WHILE_CONV: # if DEBUG_AVOID_DDR_WRITE_WHILE_CONV then DDR write was blocking (not in parallel to a conv) so it must have finished before reaching actual offloading point (The place where we actually need its AMM)
                        intermediate_tensor_read_wait_flag = local_ddr_transaction_flag
                        current_op_wait_flags.append(intermediate_tensor_read_wait_flag)

            # And finally we can perform the command
            current_op_wait_flags.extend([op_set_flag]) # We wait for first read or last op to finish
            engine_wait_flags_mask = wait_flags_allocator.get_mask_from_flags(current_op_wait_flags)

            if local_ddr_transaction_flag in current_op_wait_flags: # This gets wait flag back to having only tables axi0 and axi1 read wait, ready for nexe split
                current_op_wait_flags.remove(local_ddr_transaction_flag)

            if not DEBUG_REMOVE_ENGINE_COMMAND:
                if split_idx == (num_wloc_splits-1):

                    following_nodes_params = node['frontend']['following_nodes_params']
                    engine_command_set_flags = []

                    # If there are no following commands we set a single flag for the DDR read command to wait until current engine command is finished
                    #engine_command_set_flag = wait_flags_allocator.allocate_flag(description=node['name'] + ' engine command.')
                    engine_command_set_flag = local_ddr_transaction_flag
                    engine_command_set_flags.append(engine_command_set_flag)
                    if len(following_nodes_params) != 0: # TODO, when we implement write to DDR in middle of workload we will have to do this even when there are following nodes and move the per following node set flag to the DDR read
                        # We set a wait flag for the following executed node as it will be executed before all other nodes and hence gurantees for all following nodes that it finished its run
                        next_executed_grid_op_node_found = False
                        next_executed_node_index = ir.lexicographical_topological_sorted_graph.index(node['name'])+1
                        while not next_executed_grid_op_node_found:
                            next_executed_node_name = ir.lexicographical_topological_sorted_graph[next_executed_node_index]
                            next_executed_node = ir.graph.nodes()[next_executed_node_name]
                            if next_executed_node['op_type'] in GRID_OPS:
                                next_executed_grid_op_node_found=True
                            elif next_executed_node['op_type'] == 'Concat':
                                next_executed_grid_op_node_found=True
                            else:
                                next_executed_node_index+=1 # We assume that we will always have ordering conv at end of workload so we will always find such op

                        if 'tensor_to_offload_name' in node['backend']: #This node produces an intermediate tensor that will be written to ddr before next op
                            intermediate_tensor_name = node['backend']['tensor_to_offload_name']
                            engine_command_ddr_write_intermediate_tensor_set_flag = local_ddr_transaction_flag
                            engine_command_set_flags.append(engine_command_ddr_write_intermediate_tensor_set_flag)
                            current_node_execution_order_index = ir.lexicographical_topological_sorted_graph.index(node['name'])
                            execution_order_following_node_name = ir.lexicographical_topological_sorted_graph[current_node_execution_order_index+1]
                            execution_order_following_node = ir.graph.nodes()[execution_order_following_node_name]
                            ir.intermediate_ddr_tensors[intermediate_tensor_name].write_wait_flag = engine_command_ddr_write_intermediate_tensor_set_flag

                    engine_command_set_flag_mask = wait_flags_allocator.get_mask_from_flags(engine_command_set_flags)
                else: # If we are not in last split
                    engine_command_set_flags = [local_ddr_transaction_flag]
                    engine_command_set_flag_mask = wait_flags_allocator.get_mask_from_flags(engine_command_set_flags)

                if split_idx==0:
                    reset_amm_address = 1
                else:
                    reset_amm_address = 0
                
                for current_xslice_num in range(node['frontend']['x_slices']):
                    engine_command_description = node['name']+'_tile'+str(current_tile_num)+'_xslice'+str(current_xslice_num)
                    current_node_sequence = add_engine_command(ir,current_node_sequence,node, set_flags_mask = engine_command_set_flag_mask,
                                        wait_flags_mask = engine_wait_flags_mask,single_split_tables_buffer_id=single_split_tables_buffer_id,
                                        splitted_tables_buffer_id = splitted_tables_buffer_id,reset_amm_address=reset_amm_address,
                                        description=engine_command_description,current_tile_num=current_tile_num,current_xslice_num=current_xslice_num) # Add the command to initiate engine operation (e.g. Conv)
                # if there is a folding, and the additional dummy slice is needed, we add it here
                if node['frontend']['x_slices']%2==1 and ('force_folding_x' in  node['frontend']) and  node['frontend']['force_folding_x']: 
                    # We add a dummy command for the last slice
                    engine_command_description = node['name']+'_tile'+str(current_tile_num)+'_xslice'+str(current_xslice_num+1)                          
                    current_node_sequence = add_engine_command(ir,current_node_sequence,node, set_flags_mask = engine_command_set_flag_mask,
                                        wait_flags_mask = engine_wait_flags_mask,single_split_tables_buffer_id=single_split_tables_buffer_id,
                                        splitted_tables_buffer_id = splitted_tables_buffer_id,reset_amm_address=reset_amm_address,
                                        description=engine_command_description,current_tile_num=current_tile_num,current_xslice_num=current_xslice_num, dummy_folding_command=True)    

                conv_in_process = True
            else: # if DEBUG_REMOVE_ENGINE_COMMAND
                engine_command_set_flag_mask = 0
                following_nodes_params = node['frontend']['following_nodes_params']
                for following_node_params in following_nodes_params:
                    following_node = ir.graph.nodes[following_node_params[0]]
                    following_node['backend']['previous_op_set_flag'] = engine_wait_flags_mask
            node['backend']['engine_command_set_flag_mask'] = engine_command_set_flag_mask

        # We then add the read from DDR to AMM of next tile
        read_ddr_to_amm_set_flag = None            
        if 'next_tile_tensors_to_load_from_ddr' in node['backend']:
            tensors_info = node['backend']['next_tile_tensors_to_load_from_ddr']
            if len(tensors_info)>0:
                read_ddr_to_amm_set_flag = global_ddr_transaction_flag
                current_op_wait_flags.append(read_ddr_to_amm_set_flag)
                read_in_process, tile_write_in_process = read_needed_tensors_from_ddr(ir,tensors_info,node,wait_flags_allocator,current_node_sequence,single_split_tables_buffer_id,
                                                                current_tile_num=current_tile_num,read_ddr_wait_flag=local_ddr_transaction_flag,
                                                                read_ddr_set_flag=read_ddr_to_amm_set_flag,is_immediate_read=False,
                                                                conv_in_process=conv_in_process,tile_write_in_process = tile_write_in_process)
                if read_in_process:
                    tile_read_in_process = True
        if ('wait_for_last_tile_write_end' in node['backend']) and tile_write_in_process:
            tile_num_for_wait = node['backend']['wait_for_last_tile_write_end']
            if current_tile_num==tile_num_for_wait and not DEBUG_AVOID_DDR_WRITE_WHILE_CONV:
                tile_write_in_process = False
                write_ddr_from_amm_wait_flag_mask = wait_flags_allocator.get_mask_from_flags([global_ddr_transaction_flag])
                sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags = 0, wait_flags = write_ddr_from_amm_wait_flag_mask)
                current_node_sequence.extend([sequencer_entry])

        if ('next_tile_read_node' in node['backend']) and tile_write_in_process: #If we got to the point of next tile read and write is still in progress we need to wait for its end since in this point we deallocated the written tensor AMM
            tile_write_in_process = False
            write_ddr_from_amm_wait_flag_mask = wait_flags_allocator.get_mask_from_flags([global_ddr_transaction_flag])
            sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags = 0, wait_flags = write_ddr_from_amm_wait_flag_mask)
            current_node_sequence.extend([sequencer_entry])

        # We then add the write from AMM to DDR if needed
        if last_node_in_blob: # Output tensor is blob/workload output and needs to be written to DDR
            for output_tensor_index,blob_output_tensor in enumerate(current_tiling_blob.outputs):
                output_tensor_name = blob_output_tensor.name
                if not DEBUG_REMOVE_ENGINE_COMMAND:
                    if last_tile_of_last_node_in_last_blob:
                        write_ddr_from_amm_set_flag_mask = 0
                    else:
                        write_ddr_from_amm_set_flag_mask = wait_flags_allocator.get_mask_from_flags([local_ddr_transaction_flag])
                    write_ddr_from_amm_wait_flag_mask = wait_flags_allocator.get_mask_from_flags([local_ddr_transaction_flag])
                    if output_tensor_index==0:
                        sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags = 0, wait_flags = write_ddr_from_amm_wait_flag_mask)
                        current_node_sequence.extend([sequencer_entry])
                        if write_ddr_from_amm_set_flag_mask!=0:
                            sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags = write_ddr_from_amm_set_flag_mask, wait_flags = 0)
                            current_node_sequence.extend([sequencer_entry])
                    ddr_write_wait_flags = []
                else:         
                    ddr_write_wait_flags = [axi0_tables_load_flag,axi1_tables_load_flag]
                # There could be a case where tile write was in process in case of 2 tile blob followed by a 2 tile blob.
                    # In such case we read tile 0 input of blob only after preceding op write is done.
                    # then in middle of 1st 2 tile blob we read tile 1. Then at end of tile0 calc we write its output to ddr
                    # then at end of tile1 calc we write its output to ddr
                    # only then we can read input on 2nd 2 tile blob.
                    # This is the reason why we get 2 writes without read op in between.
                if tile_read_in_process or tile_write_in_process: 
                    ddr_write_wait_flags.append(global_ddr_transaction_flag)
                    tile_read_in_process=False
                if output_tensor_name in ir.outputs and last_tile_of_last_node_in_last_blob:
                    write_ddr_from_amm_set_flags = [local_ddr_transaction_flag]
                else:
                    write_ddr_from_amm_set_flags = [global_ddr_transaction_flag]
                    tile_write_in_process = True

                ddr_write_wait_flag_mask = wait_flags_allocator.get_mask_from_flags(ddr_write_wait_flags)

                #grids_to_write = get_grids_per_line(gridconfig=node['backend']['gridmode'])
                if node['backend']['gridmode'] == GridConfig.H14xW8:
                    grids_to_write = get_grids_to_write(blob_output_tensor)
                else:
                    grids_to_write = get_grids_per_line(gridconfig=node['backend']['gridmode'])
                #grids_to_write = get_num_virtual_grids(gridconfig=node['backend']['gridmode'])
                
                #add_padding_line = not(output_tensor_name in ir.outputs) # If its an output tensor and not last node in blob we dont need to add padding line
                add_padding_line = False
        
                for current_grid in range(grids_to_write):
                    if current_grid==0 and output_tensor_index==0:
                        wait_flags = ddr_write_wait_flag_mask
                    else:
                        wait_flags = 0
                    last_write_ddr_from_amm = (current_grid==(grids_to_write-1)) and (output_tensor_index==len(current_tiling_blob.outputs)-1)
                    set_flags=[]
                    if last_write_ddr_from_amm:
                        set_flags = write_ddr_from_amm_set_flags
                    set_flags_mask = wait_flags_allocator.get_mask_from_flags(set_flags)
                    output_tensor_producer_node_name = blob_output_tensor.producer
                    output_tensor_producer_node = ir.graph.nodes()[output_tensor_producer_node_name] # This is passed to the below "add_tensor_write_to_ddr_commands" in order to get the tensor allocation address in AMM

                    for current_xslice_num in range(output_tensor_producer_node['frontend']['output_tensor'].x_slices):
                        set_mask = 0
                        if (current_xslice_num == output_tensor_producer_node['frontend']['output_tensor'].x_slices-1):
                            set_mask = set_flags_mask
                        wait_mask = 0
                        if (current_xslice_num == 0):
                            wait_mask = wait_flags
                        current_node_sequence = add_tensor_write_to_ddr_commands(ir,blob_output_tensor,current_node_sequence,output_tensor_producer_node, set_flags_mask = set_mask,
                                wait_flags_mask = wait_mask,source_grid=current_grid,add_padding_line=False,
                                consecutive_2_tile_blobs=consecutive_2_tile_blobs,
                                prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
                                current_tile_num=current_tile_num, current_xslice_num=current_xslice_num) # If output from layer should be written to ddr (e.g. workload output) add write ddr commands

            if DEBUG_AVOID_DDR_WRITE_WHILE_CONV or last_tile_of_last_node_in_last_blob:
                tile_write_in_process = False
                write_ddr_from_amm_set_flag_mask = wait_flags_allocator.get_mask_from_flags(write_ddr_from_amm_set_flags)
                node['backend']['write_ddr_from_amm_set_flag_mask'] = write_ddr_from_amm_set_flag_mask
                sequencer_entry = SequencerCustomOperationEntry(operation=CustomeOpCode.NOP, set_flags = 0, wait_flags = write_ddr_from_amm_set_flag_mask)
                current_node_sequence.extend([sequencer_entry])


    return current_node_sequence,tile_read_in_process,tile_write_in_process

