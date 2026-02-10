import numpy as np

from common.debug_flags import DEBUG_OPTIMIZE_DMA_TRANSACTIONS
from common.enums import DDREntryType
from common.hw_config import DDR_ADDRESS_GRANULARITY,DDR_BOX_DEPTH,DDR_BOX_HEIGHT,DDR_BOX_WIDTH, \
                            AMM_HEIGHT, MODEL_BIN_FILE_GRANULARITY,get_grid_config, get_grids_per_line, TFLITE_REQUANT, MAX_X_WRAPPING
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
    def __init__(self,bytes, type = DDREntryType.INPUT_TENSOR, address = None, description = '',shape = None, blocks_offset=0):
        super().__init__(bytes, type = type, address = address, description = description)
        self.shape = shape
        if self.type not in [DDREntryType.INPUT_TENSOR, DDREntryType.OUTPUT_TENSOR,DDREntryType.INTERMEDIATE_TENSOR]:
            raise ValueError('Using tensor ddr entry for non tensor data')
        self.num_of_blocks, _ = get_tsnp_tensor_ddr_boxes_count(shape)
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

def pad_to_ddr_box_size(array):
    target_shape = [DDR_BOX_DEPTH,DDR_BOX_HEIGHT,DDR_BOX_WIDTH]
    return np.pad(
        array,
        [(0, target_shape[i] - array.shape[i]) for i in range(len(array.shape))],
        "constant",
    )

'''
def pad_to_tsnp_ddr_box_size(array, pad_value=0):
    half_ddr_box_depth = TSNP_DDR_BOX_DEPTH // 2 # This is because even/odd channels are half depth and then we concatenate them
    target_shape = [half_ddr_box_depth,TSNP_DDR_BOX_HEIGHT,TSNP_DDR_BOX_WIDTH]
    transposed_array = np.transpose(array) # We need to transpose the array because we need to control value of pixels which are padded in both axis. We need axis 0(channel) to take precedence and not X axis
    padding = [(0, target_shape[i] - array.shape[i]) for i in reversed(range(len(array.shape)))]
    constant_values = ((0,pad_value),(0,pad_value),(0,0)) 
    padded_transposed = np.pad(transposed_array,padding,"constant",constant_values=constant_values)
    padded_array = np.transpose(padded_transposed)
    return padded_array

# The below returns the number of DDR boxes of size DDR_BOX_DEPTH*DDR_BOX_HEIGHT*DDR_BOX_WIDTH needed to store the provided nparray
def get_tensor_ddr_boxes_count(shape):
    if len(shape)!=4:
        raise ValueError ('At get_tensor_ddr_boxes_count, we expect input shape to have dim=4')
    tensor_depth = shape[1]
    tensor_height = shape[2]
    tensor_width = shape[3]
    num_boxes_c = math.ceil(tensor_depth / DDR_BOX_DEPTH)
    num_boxes_y = math.ceil(tensor_height / DDR_BOX_HEIGHT)
    num_boxes_x = math.ceil(tensor_width / DDR_BOX_WIDTH)
    num_boxes = num_boxes_c*num_boxes_y*num_boxes_x
    size_in_bytes = num_boxes*DDR_BOX_DEPTH*DDR_BOX_HEIGHT*DDR_BOX_WIDTH
    return num_boxes, size_in_bytes
'''

def get_tsnp_tensor_ddr_boxes_count(shape):
    if len(shape)!=4:
        raise ValueError ('At get_tensor_ddr_boxes_count, we expect input shape to have dim=4')
    tensor_depth = shape[1]
    tensor_height = shape[2]
    tensor_width = shape[3]
    if tensor_height<AMM_HEIGHT:
        tensor_height+=1
    num_boxes_c = math.ceil(tensor_depth / DDR_BOX_DEPTH)
    num_boxes_y = math.ceil(tensor_height / DDR_BOX_HEIGHT)
    num_boxes_x = math.ceil(tensor_width / DDR_BOX_WIDTH)
    num_boxes = num_boxes_c*num_boxes_y*num_boxes_x
    size_in_bytes = num_boxes*DDR_BOX_DEPTH*DDR_BOX_HEIGHT*DDR_BOX_WIDTH
    return num_boxes, size_in_bytes

'''
# The below gets a box (slice) of size 14(width)x14(height)x16(channels). axis 0 will be channel, axis 1 will be height and axis 2 will be width
def get_tensor_box(nparray,c,y,x):
    if len(nparray.shape)!=4:
        raise ValueError ('At get_tensor_box, we expect input numpy array to have dim=4')
    x_start = x*TENSOR_BOX_WIDTH
    x_end = (x+1)*TENSOR_BOX_WIDTH
    y_start = y * TENSOR_BOX_HEIGHT
    y_end = (y+1) * TENSOR_BOX_HEIGHT
    c_start = c * DDR_BOX_DEPTH
    c_end = (c+1) * DDR_BOX_DEPTH
    np_slice = nparray[0,c_start:c_end,y_start:y_end,x_start:x_end] # We assume batch is always 1
    padded_slice = pad_to_ddr_box_size(np_slice)
    return padded_slice

def get_tsnp_tensor_box(nparray,c,y,x,pad_value=0):
    if len(nparray.shape)!=4:
        raise ValueError ('At get_tensor_box, we expect input numpy array to have dim=4')
    x_start = x*TSNP_DDR_BOX_WIDTH
    x_end = (x+1)*TSNP_DDR_BOX_WIDTH
    y_start = y * TSNP_DDR_BOX_HEIGHT
    y_end = (y+1) * TSNP_DDR_BOX_HEIGHT
    c_start = c * TSNP_DDR_BOX_DEPTH
    c_end = (c+1) * TSNP_DDR_BOX_DEPTH
    slice_even_channels = nparray[0,c_start:c_end:2,y_start:y_end,x_start:x_end] # We assume batch is always 1
    slice_odd_channels = nparray[0,c_start+1:c_end:2,y_start:y_end,x_start:x_end] # We assume batch is always 1
    if x==3 and nparray.shape[3]<24: # If its a folded tensor of width < 24 (uses 3 out of 4 grids), we need to pad all the 4th grid(x==3) with zeros so it matches FPGA output
        actual_pad_value = 0
    else:
        actual_pad_value = pad_value
    padded_slice_even_channels = pad_to_tsnp_ddr_box_size(slice_even_channels,pad_value=actual_pad_value)
    padded_slice_odd_channels = pad_to_tsnp_ddr_box_size(slice_odd_channels,pad_value=actual_pad_value)
    padded_slice = np.concatenate([padded_slice_even_channels,padded_slice_odd_channels],axis=2)
    return padded_slice

# The below converts numpy array to ddr bytearray according to SNP architecture spec
# see paragraph 2.13 called "DDR Access"
def create_tensor_byte_array(tensor_nparray):
    if len(tensor_nparray.shape)!=4:
        raise ValueError ('At create_tensor_byte_array, we expect input numpy array to have dim=4')
    tensor_bytearray = bytearray(0)
    tensor_depth = tensor_nparray.shape[1]
    tensor_height = tensor_nparray.shape[2]
    tensor_width = tensor_nparray.shape[3]
    num_boxes_c = math.ceil(tensor_depth / DDR_BOX_DEPTH)
    num_boxes_y = math.ceil(tensor_height / DDR_BOX_HEIGHT)
    num_boxes_x = math.ceil(tensor_width / DDR_BOX_WIDTH)
    for current_y_box_idx in range(num_boxes_y):
        for current_x_box_idx in range(num_boxes_x):
            for current_c_box_idx in range(num_boxes_c):
                current_box = get_tensor_box(tensor_nparray,current_c_box_idx,current_y_box_idx,current_x_box_idx) # The box axis order is as in ddr. axis0 = channel, axis1 = height, axis2 = width
                if not TFLITE_REQUANT:
                    current_box_bytearray = bytearray(current_box.astype(np.uint8))
                else:
                    current_box_bytearray = bytearray(current_box.astype(np.int8))
                if len(current_box_bytearray)>DDR_BOX_ALIGNMENT:
                    raise ValueError('At create_tensor_byte_array, each box cant be more than %d bytes as we dont currently support this ' % DDR_BOX_ALIGNMENT)
                if len(current_box_bytearray)<DDR_BOX_ALIGNMENT: # Each box should take exactly DDR_BOX_ALIGNMENT bytes. if its less pad with zeros
                    filler_bytearray = bytearray(DDR_BOX_ALIGNMENT-len(current_box_bytearray))
                    current_box_bytearray.extend(filler_bytearray)
                tensor_bytearray.extend(current_box_bytearray)

    return tensor_bytearray

# The below converts numpy array to ddr bytearray according to TSNP architecture spec
def create_tsnp_tensor_byte_array(tensor_nparray, pad_extra_line = False,pad_to_full_grid=False,pad_value=0):
    if len(tensor_nparray.shape)!=4:
        raise ValueError ('At create_tensor_byte_array, we expect input numpy array to have dim=4')
    tensor_bytearray = bytearray(0)
    tensor_depth = tensor_nparray.shape[1]
    tensor_height = tensor_nparray.shape[2]
    tensor_width = tensor_nparray.shape[3]
    #grid_mode = get_grid_config(tensor_width)
    #num_boxes_x = get_grids_per_line(grid_mode)
   # If tensor height is smaller than AMM size we need to pad y axis in 1 line so that in case it is used as input to k=3 conv we will not use "garbage"
   # and extra line of padding will be loaded from DDR to AMM.

    if (pad_extra_line or pad_to_full_grid): 
        if pad_extra_line:
            y_padding=0 #Remove padding line
        elif pad_to_full_grid:
            if (tensor_height % AMM_HEIGHT)==0:
                y_padding=0
            else:
                y_padding=AMM_HEIGHT-(tensor_height % AMM_HEIGHT)
        else:
            raise ValueError ('Not supported. Please check')
        tensor_nparray = np.pad(tensor_nparray,[(0,0),(0,0),(0,y_padding),(0,0)],"constant",constant_values=pad_value)
        tensor_height+=y_padding
    num_boxes_c = math.ceil(tensor_depth / TSNP_DDR_BOX_DEPTH)
    num_boxes_y = math.ceil(tensor_height / TSNP_DDR_BOX_HEIGHT)
    num_boxes_x = math.ceil(tensor_width / TSNP_DDR_BOX_WIDTH)
    for current_x_box_idx in range(num_boxes_x):
        for current_y_box_idx in range(num_boxes_y):
            for current_c_box_idx in range(num_boxes_c):
                current_box = get_tsnp_tensor_box(tensor_nparray,current_c_box_idx,current_y_box_idx,current_x_box_idx, pad_value=pad_value) # The box axis order is as in ddr. axis0 = channel, axis1 = height, axis2 = width
                current_box_bytearray = bytearray(current_box.astype(np.uint8))
                if len(current_box_bytearray)>TSNP_DDR_BOX_ALIGNMENT:
                    raise ValueError('At create_tensor_byte_array, each box cant be more than %d bytes as we dont currently support this ' % DDR_BOX_ALIGNMENT)
                tensor_bytearray.extend(current_box_bytearray)
    return tensor_bytearray
'''
    
# The below converts numpy array to ddr bytearray according to TSNP architecture spec
def create_tsnp_tensor_byte_array(tensor_nparray):
    if len(tensor_nparray.shape)!=4:
        raise ValueError ('At create_tensor_byte_array, we expect input numpy array to have dim=4')
    
    # Creating bytearray in roster format for .nxo write
    tensor_bytearray = bytearray(0)
    for channel_idx in range(tensor_nparray.shape[1]):
        for line_idx in range(tensor_nparray.shape[2]):
            current_box = tensor_nparray[0, channel_idx, line_idx, :]
            if not TFLITE_REQUANT:
                current_box_bytearray = bytearray(current_box.astype(np.uint8))
            else:
                current_box_bytearray = bytearray(current_box.astype(np.int8))
            tensor_bytearray.extend(current_box_bytearray)

    return tensor_bytearray

# The below converts numpy array to ddr bytearray according to TSNP architecture spec
def create_nxd_byte_array(tensor_nparray, is_intermediate_node = False, num_xslices = 1, y_folding = 0, is_split_node = False, filename = None):
    #Reordering the intermediate outputs to support Yaron's DMA parameter change
    if DEBUG_OPTIMIZE_DMA_TRANSACTIONS and is_intermediate_node:        
        x_wrapping = 1
        output_z = tensor_nparray.shape[1]
        output_x = math.ceil(tensor_nparray.shape[3]/16)*16
        while ((output_z % 2) ==0) and (x_wrapping <= (MAX_X_WRAPPING // 2)) and (output_x <= 2048):
            output_z = output_z // 2
            x_wrapping *= 2
            output_x *= 2

        # Creating tensor with x_slices
        width = tensor_nparray.shape[3]
        height = tensor_nparray.shape[2]
        depth = tensor_nparray.shape[1]
        sliced_tensor = np.zeros((1, depth * num_xslices, height, width // num_xslices))
        for current_xslice_idx in range(num_xslices):
            for channel_idx in range(depth):
                for current_y_idx in range(height):
                    x_slice_start = current_xslice_idx * 16
                    x_slice_end = (current_xslice_idx+1) * 16
                    sliced_tensor[0, current_xslice_idx*depth + channel_idx, current_y_idx, :] = tensor_nparray[0, channel_idx, current_y_idx, x_slice_start:x_slice_end]  
        reshaped_tensor = sliced_tensor.reshape((1, (sliced_tensor.shape[1] // int(2 ** y_folding)), sliced_tensor.shape[2] * int(2 ** y_folding), sliced_tensor.shape[3]))

        if (reshaped_tensor.shape[3] != math.ceil(reshaped_tensor.shape[3]/16)*16):
            padding_width = math.ceil(reshaped_tensor.shape[3]/16)*16 - reshaped_tensor.shape[3]
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
                        x_start = ((x_wrapping*channel_idx + current_x_idx) * 16)
                        x_end = x_start + 16
                        ddr_tensor_nparray[0, output_channel_idx, current_y_idx, x_start:x_end] = reshaped_tensor[0, tensor_depth*x_wrapping*channel_idx + current_x_idx + x_wrapping*output_channel_idx, current_y_idx, :]

        # Creating bytearray for .nxd write
        ddr_tensor_bytearray = bytearray(0)
        for channel_idx in range(ddr_tensor_nparray.shape[1]):
            for line_idx in range(ddr_tensor_nparray.shape[2]):
                current_box = ddr_tensor_nparray[0, channel_idx, line_idx, :]
                if not TFLITE_REQUANT:
                    current_box_bytearray = bytearray(current_box.astype(np.uint8))
                else:
                    current_box_bytearray = bytearray(current_box.astype(np.int8))
                ddr_tensor_bytearray.extend(current_box_bytearray)
        
        ddr_filename = filename.replace('.nxo', '.nxd')
        if is_split_node:
            ddr_filename = ddr_filename.replace('_split','')
            if os.path.exists(ddr_filename):
                with open(ddr_filename,'ab') as bin_file:
                    bin_file.write(ddr_tensor_bytearray)
        else:
            with open(ddr_filename,'wb') as bin_file:
                bin_file.write(ddr_tensor_bytearray)

'''
def get_tensor_from_byte_array(byte_array,target_tensor_shape):
    if not TFLITE_REQUANT:
        numpy_array = np.frombuffer(byte_array,dtype=np.uint8)
    else:
        numpy_array = np.frombuffer(byte_array,dtype=np.int8)
    channels = numpy_array.size // (DDR_BOX_HEIGHT*DDR_BOX_WIDTH)
    ddr_format_tensor = numpy_array.reshape(channels,DDR_BOX_HEIGHT,DDR_BOX_WIDTH)
    tensor_depth = target_tensor_shape[1]
    tensor_height = target_tensor_shape[2]
    tensor_width = target_tensor_shape[3]
    output_tensor = np.zeros(target_tensor_shape)
    num_boxes_c = math.ceil(tensor_depth / DDR_BOX_DEPTH)
    num_boxes_y = math.ceil(tensor_height / TENSOR_BOX_HEIGHT)
    num_boxes_x = math.ceil(tensor_width / TENSOR_BOX_WIDTH)
    for current_y_box_idx in range(num_boxes_y):
        for current_x_box_idx in range(num_boxes_x):
            for current_c_box_idx in range(num_boxes_c):
                current_box = ddr_format_tensor[current_c_box_idx+num_boxes_c*current_x_box_idx+(num_boxes_c*num_boxes_x)*current_y_box_idx,0:TENSOR_BOX_HEIGHT,0:TENSOR_BOX_WIDTH] # The box axis order is as in ddr. axis0 = channel, axis1 = height, axis2 = width
                output_tensor_start_x = current_x_box_idx*TENSOR_BOX_WIDTH
                output_tensor_end_x = (current_x_box_idx+1)*TENSOR_BOX_WIDTH
                output_tensor_start_y = current_y_box_idx*TENSOR_BOX_HEIGHT
                output_tensor_end_y = (current_y_box_idx+1)*TENSOR_BOX_HEIGHT
                output_tensor[0,current_c_box_idx,output_tensor_start_y:output_tensor_end_y,output_tensor_start_x:output_tensor_end_x] = current_box

    return output_tensor
'''

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

'''
# The below performs DDR mapping for the intermediate tensor
def tensor_xslice_yfold(tensor_nparray, x_folding = 0, y_folding = 0):
    if len(tensor_nparray.shape)!=4:
        raise ValueError ('At create_tsnp_tensor_xunwrap_array, we expect input numpy array to have dim=4')
    tensor_depth = tensor_nparray.shape[1]
    tensor_height = tensor_nparray.shape[2]
    tensor_width = tensor_nparray.shape[3]
    output_tensor = np.zeros((1,tensor_depth, tensor_height, tensor_width))
    ratio = int(tensor_depth/(pow(2, y_folding)*pow(2, x_folding)))
    for current_x_fold in range(pow(2, x_folding)):
        for current_y_fold in range(pow(2, y_folding)):
            for current_c_idx in range(ratio):
                input_c_idx = current_y_fold*pow(2,x_folding)*ratio  + current_x_fold*ratio + current_c_idx
                output_c_idx = current_x_fold*pow(2,y_folding)*ratio + current_y_fold*ratio + current_c_idx
                output_tensor[0, output_c_idx, :, :] = tensor_nparray[0, input_c_idx, :, :]
    return output_tensor


# The below converts numpy array to ddr roster according to TSNP architecture spec
def create_tsnp_ddr_roster_array(tensor_nparray):
    if len(tensor_nparray.shape)!=4:
        raise ValueError ('At create_tsnp_ddr_roster_array, we expect input numpy array to have dim=4')
    tensor_depth = tensor_nparray.shape[1]
    tensor_height = tensor_nparray.shape[2]
    tensor_width = tensor_nparray.shape[3]
    num_boxes_x = int(tensor_width / VBX3_PHASE1_DDR_BOX_WIDTH)
    #num_boxes_y = math.ceil(tensor_height / VBX3_PHASE1_DDR_BOX_HEIGHT)
    #output_tensor_depth = tensor_depth * num_boxes_y * num_boxes_x
    #grid_tensor = np.zeros((1,output_tensor_depth, VBX3_PHASE1_DDR_BOX_HEIGHT, VBX3_PHASE1_DDR_BOX_WIDTH))
    output_tensor_depth = tensor_depth * num_boxes_x
    grid_tensor = np.zeros((1,output_tensor_depth, tensor_height, VBX3_PHASE1_DDR_BOX_WIDTH))
    output_channel = 0
    output_y_idx = 0
    for current_c_idx in range(tensor_depth):
        for current_y_idx in range(tensor_height):
            for current_x_idx in range(num_boxes_x):
                grid_x_start = current_x_idx * VBX3_PHASE1_DDR_BOX_WIDTH
                grid_x_end = (current_x_idx+1) * VBX3_PHASE1_DDR_BOX_WIDTH
                grid_tensor[0, output_channel, output_y_idx, :] = tensor_nparray[0, current_c_idx, current_y_idx, grid_x_start:grid_x_end]
                if (output_y_idx == tensor_height-1):
                    output_y_idx = 0
                    output_channel += 1
                else:
                    output_y_idx += 1
    return grid_tensor

def convert_tsnp_tensor_grid_fold(tensor_nparray, fold_counter = 0, original_depth = 0, original_width = 0):
    if len(tensor_nparray.shape)!=4:
        raise ValueError ('At convert_tsnp_tensor_grid_fold, we expect input numpy array to have dim=4')
    x_fold = pow(2, fold_counter)
    grid_tensor = np.zeros((1, original_depth, tensor_nparray.shape[2], original_width))
    grid_boxes = 2 * x_fold
    for current_c_idx in range(int(original_depth/x_fold)):
        for current_x_idx in range(x_fold):
            output_channel_idx = (current_c_idx * x_fold) + current_x_idx
            input_channel_idx = (grid_boxes * current_c_idx) + current_x_idx
            grid_tensor[0, output_channel_idx, :, 0:math.ceil(original_width/2)] = tensor_nparray[0, input_channel_idx, :, :]
            grid_tensor[0, output_channel_idx, :, math.ceil(original_width/2):original_width] = tensor_nparray[0, input_channel_idx+x_fold, :, 0:original_width-math.ceil(original_width/2)]
    return grid_tensor
'''