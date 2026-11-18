import numpy as np

from common.debug_flags import DEBUG_OPTIMIZE_DMA_TRANSACTIONS
from common.enums import DDREntryType
from common.hw_config import DDR_ADDRESS_GRANULARITY, DDR_BOX_DEPTH, DDR_BOX_WIDTH, MAX_GRID_WIDTH, \
                            AMM_HEIGHT, MODEL_BIN_FILE_GRANULARITY, MAX_X_WRAPPING
import math
import os

DDR_TEXTFILE_BYTES_PER_LINE = 64

class DDREntry:
    def __init__(self,bytes, type = DDREntryType.INPUT_TENSOR, address = None, description = ''):
        self.bytes = bytes # type: bytearray
        self.type = type
        self.address = address
        self.description = description
    def set_description(self,description):
        self.description = description
    def get_length(self):
        return len(self.bytes)

class TensorDDREntry (DDREntry):
    def __init__(self,bytes, type = DDREntryType.INPUT_TENSOR, address = None, description = '',shape = None, amm_height = AMM_HEIGHT, blocks_offset=0):
        super().__init__(bytes, type = type, address = address, description = description)
        self.shape = shape
        if self.type not in [DDREntryType.INPUT_TENSOR, DDREntryType.OUTPUT_TENSOR,DDREntryType.INTERMEDIATE_TENSOR]:
            raise ValueError('Using tensor ddr entry for non tensor data')
        self.num_of_blocks, _ = get_tsnp_tensor_ddr_boxes_count(shape, amm_height)
        self.blocks_offset = blocks_offset

    def get_length_for_read(self):
        return int(bin(self.blocks_offset)[2:].zfill(23)+bin(self.num_of_blocks)[2:].zfill(12),2)

    def get_length_for_write(self):
        return int('0b'+ bin(self.blocks_offset)[2:].zfill(24)+bin(self.num_of_blocks)[2:].zfill(12),2)


class DDR:
    def __init__(self,ddr_base_name):
        self.entries = []
        self.next_free_address=0
        self.mem = bytearray()
        self.ddr_base_name = ddr_base_name
    def add_entry(self,entry: DDREntry):
        entry.address = self.next_free_address
        entry_length = len(entry.bytes)
        rounded_entry_length = DDR_ADDRESS_GRANULARITY * math.ceil(entry_length/DDR_ADDRESS_GRANULARITY)
        self.next_free_address += rounded_entry_length
        self.entries.append(entry)
    def remove_entry(self,entry_to_remove: DDREntry):
        entry_length = len(entry_to_remove.bytes)
        rounded_entry_length = DDR_ADDRESS_GRANULARITY * math.ceil(entry_length/DDR_ADDRESS_GRANULARITY)
        for current_entry in self.entries:
            if current_entry.address>entry_to_remove.address:
                current_entry.address = current_entry.address - rounded_entry_length
        self.next_free_address -= rounded_entry_length
        self.entries.remove(entry_to_remove)

    def create_ddr_mem(self):
        ddr_bytes_array = bytearray(0)
        amm_wm_72bit_word=''
        current_ddr_size = 0
        for current_ddr_entry in self.entries:
            current_ddr_entry_address = current_ddr_entry.address
            if current_ddr_size > current_ddr_entry.address:
                raise ValueError ('While creating ddr bytearray, entry address is not sorted')
            elif current_ddr_size < current_ddr_entry.address: # Need to fill gap between last entry to current entry with zeros
                filler_bytearray = bytearray(current_ddr_entry.address-current_ddr_size)
                ddr_bytes_array.extend(filler_bytearray)
                current_ddr_size += len(filler_bytearray)

            ddr_bytes_array.extend(current_ddr_entry.bytes)
            current_ddr_size += len(current_ddr_entry.bytes)
        if len(ddr_bytes_array) % MODEL_BIN_FILE_GRANULARITY !=0:
            alignment_bytes_array = bytearray((MODEL_BIN_FILE_GRANULARITY-(len(ddr_bytes_array) % MODEL_BIN_FILE_GRANULARITY)))
            ddr_bytes_array.extend(alignment_bytes_array)
            current_ddr_size += len(alignment_bytes_array)
        self.mem = ddr_bytes_array
    def save_ddr_bytearray_as_bin(self,ddr_bin_filename):
        with open(ddr_bin_filename,'wb') as bin_file:
            bin_file.write(self.mem)
    def save_ddr_bytearray_as_text(self,ddr_txt_filename):
        with open(ddr_txt_filename,'w') as txt_file:
            for i in range(0,len(self.mem),DDR_TEXTFILE_BYTES_PER_LINE):
                line = ''
                for j in range(DDR_TEXTFILE_BYTES_PER_LINE):
                    if (i+j)<len(self.mem): # This is to make sure that at the end of mem we dont read bytes until end of 64 byte line
                        line = hex(self.mem[i+j])[2:].zfill(2)+line
                    else:
                        break
                if len(line)<(DDR_TEXTFILE_BYTES_PER_LINE*2):
                    line=line.zfill(DDR_TEXTFILE_BYTES_PER_LINE*2)
                txt_file.write(line+'\n')
    def save_ddr_info_file(self,ddr_info_filename):
        with open(ddr_info_filename,'w') as ddr_info_file:
            ddr_info_file.write('All below entries are relative to: %s\n' % self.ddr_base_name)
            for current_ddr_entry in self.entries:
                entry_info = 'Address: 0x%X, Size(bytes): %s, Description: %s' % (current_ddr_entry.address,len(current_ddr_entry.bytes),current_ddr_entry.description)
                ddr_info_file.write(entry_info+'\n')
    def get_ddr_info(self) -> str:
        info_str = ''
        info_str+=('All below entries are relative to: %s\n' % self.ddr_base_name)
        for current_ddr_entry in self.entries:
            entry_info = 'Address: 0x%X, Size(bytes): %s, Description: %s' % (current_ddr_entry.address,len(current_ddr_entry.bytes),current_ddr_entry.description)
            info_str+=(entry_info+'\n')
        info_str+='\n'
        return info_str

def get_tsnp_tensor_ddr_boxes_count(shape, amm_height = AMM_HEIGHT):
    if len(shape)!=4:
        raise ValueError ('At get_tensor_ddr_boxes_count, we expect input shape to have dim=4')
    tensor_depth = shape[1]
    tensor_height = shape[2]
    tensor_width = shape[3]
    num_boxes_c = math.ceil(tensor_depth / DDR_BOX_DEPTH)
    num_boxes_y = math.ceil(tensor_height / amm_height)
    num_boxes_x = math.ceil(tensor_width / DDR_BOX_WIDTH)
    num_boxes = num_boxes_c*num_boxes_y*num_boxes_x
    size_in_bytes = num_boxes*DDR_BOX_DEPTH*amm_height*DDR_BOX_WIDTH
    return num_boxes, size_in_bytes

    
# The below converts numpy array to ddr bytearray according to TSNP architecture spec
def create_tsnp_tensor_byte_array(tensor_nparray):
    if len(tensor_nparray.shape)!=4:
        raise ValueError ('At create_tensor_byte_array, we expect input numpy array to have dim=4')
    
    # Creating bytearray in roster format for .nxo write
    tensor_bytearray = bytearray(0)
    for channel_idx in range(tensor_nparray.shape[1]):
        for line_idx in range(tensor_nparray.shape[2]):
            current_box = tensor_nparray[0, channel_idx, line_idx, :]
            current_box_bytearray = bytearray(current_box.astype(np.int8))
            tensor_bytearray.extend(current_box_bytearray)

    return tensor_bytearray

# The below converts numpy array to ddr bytearray according to TSNP architecture spec
def create_nxd_byte_array(tensor_nparray, is_intermediate_node = False, num_xslices = 1, y_folding = 0, zero_point = 0, filename = None):
    #Reordering the intermediate outputs to support Yaron's DMA parameter change
    if DEBUG_OPTIMIZE_DMA_TRANSACTIONS and is_intermediate_node:        
        x_wrapping = 1
        output_z = tensor_nparray.shape[1]
        output_x = math.ceil(tensor_nparray.shape[3] / MAX_GRID_WIDTH) * MAX_GRID_WIDTH
        while (output_z > 1) and (output_z % 2 != 1) and (x_wrapping <= (MAX_X_WRAPPING//2)) and (output_x < 1024):
            output_z /= 2
            x_wrapping *= 2
            output_x *= 2
        output_z = int(math.ceil(output_z))
        if x_wrapping == 32:
            x_wrapping = 0

        # Creating tensor with x_slices
        width = tensor_nparray.shape[3]
        height = tensor_nparray.shape[2]
        depth = tensor_nparray.shape[1]
        # Resize sets x_slices=2*input_x_slices, but a non-grid-multiple width leaves the last
        # slice empty (input 20->2 slices, resized 40->3, not 4). Recompute from width/MAX_GRID_WIDTH
        # and pad; no-op for already-grid-padded tensors, fixes resize-fed ones.
        num_xslices = math.ceil(width / MAX_GRID_WIDTH)
        padded_width = num_xslices * MAX_GRID_WIDTH
        if padded_width != width:
            tensor_nparray = np.pad(tensor_nparray, ((0, 0), (0, 0), (0, 0), (0, padded_width - width)),
                                    mode="constant", constant_values=zero_point)
            width = padded_width
        sliced_tensor = np.zeros((1, depth * num_xslices, height, width // num_xslices))
        for current_xslice_idx in range(num_xslices):
            for channel_idx in range(depth):
                for current_y_idx in range(height):
                    x_slice_start = current_xslice_idx * MAX_GRID_WIDTH
                    x_slice_end = (current_xslice_idx+1) * MAX_GRID_WIDTH
                    sliced_tensor[0, current_xslice_idx*depth + channel_idx, current_y_idx, :] = tensor_nparray[0, channel_idx, current_y_idx, x_slice_start:x_slice_end]  
        reshaped_tensor = sliced_tensor.reshape((1, (sliced_tensor.shape[1] // int(2 ** y_folding)), sliced_tensor.shape[2] * int(2 ** y_folding), sliced_tensor.shape[3]))

        if (reshaped_tensor.shape[3] != math.ceil(reshaped_tensor.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH):
            padding_width = math.ceil(reshaped_tensor.shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH - reshaped_tensor.shape[3]
            reshaped_tensor = np.pad(reshaped_tensor, ((0, 0), (0, 0), (0, 0), (0, padding_width)), mode="constant", constant_values=0)  

        # Changing the format to address the DMA optimization used in the sequencer
        tensor_width =  output_x
        tensor_height = reshaped_tensor.shape[2]
        tensor_depth = output_z
        ddr_tensor_nparray = np.zeros((1, tensor_depth, tensor_height, tensor_width))
        for output_channel_idx in range(tensor_depth):
            for current_y_idx in range(tensor_height):
                for channel_idx in range(reshaped_tensor.shape[1] // (x_wrapping*tensor_depth)):
                    for current_x_idx in range(x_wrapping):
                        x_start = ((x_wrapping*channel_idx + current_x_idx) * MAX_GRID_WIDTH)
                        x_end = x_start + MAX_GRID_WIDTH
                        ddr_tensor_nparray[0, output_channel_idx, current_y_idx, x_start:x_end] = reshaped_tensor[0, tensor_depth*x_wrapping*channel_idx + current_x_idx + x_wrapping*output_channel_idx, current_y_idx, :]

        # Creating bytearray for .nxd write
        ddr_tensor_bytearray = bytearray(ddr_tensor_nparray.astype(np.int8))        
        ddr_filename = filename.replace('.nxo', '.nxd')
        with open(ddr_filename,'wb') as bin_file:
            bin_file.write(ddr_tensor_bytearray)

# The below converts numpy array to grids by x_wrapping
def create_tsnp_tensor_xwrap_array(tensor_nparray, input_folding_factor_x=0):
    if len(tensor_nparray.shape)!=4:
        raise ValueError ('At create_tsnp_tensor_xwrap_array, we expect input numpy array to have dim=4')
    tensor_depth = tensor_nparray.shape[1]
    tensor_height = tensor_nparray.shape[2]
    tensor_width = tensor_nparray.shape[3]
    new_width = int(tensor_width / (2 ** input_folding_factor_x))
    num_boxes_x = int(tensor_width / new_width)
    output_tensor_depth = tensor_depth * num_boxes_x
    xwrap_tensor = np.zeros((1,output_tensor_depth, tensor_height, new_width))
    output_channel = 0
    for current_c_idx in range(tensor_depth):
        for current_x_idx in range(num_boxes_x):
            x_start = current_x_idx * new_width
            x_end = (current_x_idx+1) * new_width
            for current_y_idx in range(tensor_height):
                xwrap_tensor[0, output_channel, current_y_idx, :] = tensor_nparray[0, current_c_idx, current_y_idx, x_start:x_end]
            output_channel += 1
    return xwrap_tensor

# The below performs DDR mapping for the intermediate tensor
def tensor_xwrap_yfold(tensor_nparray, x_wrapping = 0, y_folding = 0):
    if len(tensor_nparray.shape)!=4:
        raise ValueError ('At create_tsnp_tensor_xunwrap_array, we expect input numpy array to have dim=4')
    tensor_depth = tensor_nparray.shape[1]
    tensor_height = tensor_nparray.shape[2]
    tensor_width = tensor_nparray.shape[3]
    output_tensor = np.zeros((1,tensor_depth, tensor_height, tensor_width))
    output_c_idx = 0
    output_y_idx = 0
    for current_c_idx in range(int(tensor_depth/(pow(2, y_folding)*pow(2,x_wrapping)))):
        for current_y_idx in range(tensor_height):
            for y_fold in range(pow(2, y_folding)):
                for current_x_idx in range(pow(2,x_wrapping)):
                    input_c_idx = y_fold * int(tensor_depth/pow(2,y_folding)) + current_c_idx*pow(2,x_wrapping) + current_x_idx
                    output_tensor[0, output_c_idx, output_y_idx, :] = tensor_nparray[0, input_c_idx, current_y_idx, :]
                    if (output_y_idx == tensor_height-1):
                        output_c_idx += 1
                        output_y_idx = 0
                    else:
                        output_y_idx += 1
    return output_tensor

