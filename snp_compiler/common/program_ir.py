# wloc/rq definitions: https://docs.google.com/spreadsheets/d/17kMTP18Vjo52BkP0z1pygQA0bY0Z70zj/edit#gid=1267198102

import textwrap
from common.debug_flags import SHORT_PAIRING_USED
from common.enums import RqDSPCommand, RqOP, AMMType
# from common.hw_config import SHORT_ENTRY_JUMP_BITS, MAX_OFFSET_VALUE_FOR_SHORT_ENTRY, MAX_ADDRESS_VALUE_FOR_LONG_ENTRY, MAX_ADDRESS_VALUE_FOR_DEEP_CONV_LONG_ENTRY,\
#                              LONG_ENTRY_BITS, MAX_WLOC_128BIT_ENTRIES, MAX_RQLOC_128BIT_ENTRIES, MAX_RQPARAMS_128BIT_ENTRIES,\
#                              REDUCED_MAC_RESCALE_BUS_WIDTH, MCHP_NUMERICS
from common.hw_config import *
from collections import OrderedDict
from common.utils import bin2c
import math
import copy
from abc import ABC
import numpy as np

"""
Function: create_mem
-----------------------------
This function takes a list of binary numbers (as strings) and combines them 
into mem_input_width (now 256-bit numbers), each of which consists of two mem_output_width (now 128-bit) sub-numbers. 
The resulting 256-bit numbers are stored in the list `mem_address`.

The memory in the FPGA can only be written in blocks of mem_input_width (currently 256-bit numbers) and not in shorter blocks. 
This is due to the DMA width, as each word in DMA must be 256 bits. Therefore, we need to organize the memory in 256-bit blocks.

Reading from the memory, however, can be done in smaller blocks, defined by mem_output_width, which can be as small as 128 bits. 
For this reason, we must arrange the memory in such a way that reading 128-bit words is also possible.


Parameters:
-----------
list_str_nummers : list of str
A list of binary numbers represented as strings.

Returns:
--------
list of str
A list containing the combined mem_input_width (now 256-bit) numbers in binary string format.
"""
def create_mem_old(list_str_nummers, mem_input_width=256, mem_output_width=128):        
    
    # Temporary variables for accumulating sub-numbers
    current_sub_bit = ""
    sub_line = []
    mem_created = []
    sub_line_number = mem_input_width // mem_output_width
    out_format_bit_length = mem_input_width//8

    # Process the input list 'list_str_nummers'
    for binary_str_cmd in list_str_nummers:

        binary_str = binary_str_cmd.to_bits()[2:]


        # Add the current binary string to the accumulator
        current_sub_bit = binary_str+current_sub_bit

        # Check if we have reached the length of mem_output_width
        while len(current_sub_bit) >= mem_output_width:
            # Extract the first mem_output_width (now 128) bits and add them to the sub-number list
            sub_bit = current_sub_bit[-mem_output_width:]
            sub_line.append(sub_bit)
            
            # Remove the added part from the accumulator
            current_sub_bit = current_sub_bit[:-mem_output_width]
            
            # If we have accumulated the correct number of sub-numbers (sub_line_number), add them to the main list
            if len(sub_line) == sub_line_number:
                # Combine the sub-numbers into one 256-bit number
                combined_256_bit_number = ''.join(sub_line[::-1])
                # Add this 256-bit number to mem_created list
                current_word_bytearray = int(combined_256_bit_number, 2).to_bytes(out_format_bit_length , byteorder='little')
                mem_created.extend(current_word_bytearray)
                # Clear the list of sub-numbers for the next 256-bit block
                sub_line = []

    # Process the remaining bits in the accumulator
    if len(current_sub_bit) > 0:
        # Pad the remaining bits to mem_output_width length with zeros
        current_sub_bit = current_sub_bit.rjust(mem_output_width, '0')
        sub_line.append(current_sub_bit)

    # If there are any unprocessed sub-numbers, pad them and store
    if len(sub_line) > 0:
        # Pad the second sub-number with zeros to complete the 256-bit number
        for i in range(sub_line_number - len(sub_line)):
            sub_line.append('0' * mem_output_width)

    # Add the final combined 256-bit number if available
    if len(sub_line) == sub_line_number:
        combined_256_bit_number = ''.join(sub_line[::-1])
        current_word_bytearray = int(combined_256_bit_number, 2).to_bytes(out_format_bit_length , byteorder='little')
        mem_created.extend(current_word_bytearray)

    return mem_created

def create_mem(list_str_nummers, mem_cmd_resolution, mem_input_width=256, mem_output_width=128):        
    
    # Temporary variables for accumulating sub-numbers
    current_sub_bit = ""
    sub_line = []
    mem_created = []
    sub_line_number = mem_input_width // mem_output_width
    out_format_bit_length = mem_input_width//8

    # Process the input list 'list_str_nummers'
    for inx, binary_str_cmd in enumerate (list_str_nummers):

        binary_str = binary_str_cmd.to_bits()[2:]

        if 'b' in binary_str:
            raise ValueError('string with b!')

        binary_str_parts = textwrap.wrap(binary_str, width=mem_cmd_resolution)[::-1]

        for i_part in binary_str_parts:
            # Add the current binary string to the accumulator
            candidate_current_sub_bit = i_part+current_sub_bit

            # Check if we have reached the length of mem_output_width
            if len(candidate_current_sub_bit) > mem_output_width:
                # Extract the first mem_output_width (now 128) bits and add them to the sub-number list
                sub_bit = current_sub_bit.zfill(mem_output_width)#current_sub_bit[-mem_output_width:]
                sub_line.append(sub_bit)
                
                # Remove the added part from the accumulator
                current_sub_bit = i_part#current_sub_bit[:-mem_output_width]
                
                # If we have accumulated the correct number of sub-numbers (sub_line_number), add them to the main list
                if len(sub_line) == sub_line_number:
                    # Combine the sub-numbers into one 256-bit number
                    combined_256_bit_number = ''.join(sub_line[::-1])
                    # Add this 256-bit number to mem_created list
                    current_word_bytearray = int(combined_256_bit_number, 2).to_bytes(out_format_bit_length , byteorder='little')
                    mem_created.extend(current_word_bytearray)
                    #return mem_created
                    # Clear the list of sub-numbers for the next 256-bit block
                    sub_line = []
            else:
                current_sub_bit = candidate_current_sub_bit        

    # Process the remaining bits in the accumulator
    if len(current_sub_bit) > 0:
        # Pad the remaining bits to mem_output_width length with zeros
        current_sub_bit = current_sub_bit.rjust(mem_output_width, '0')
        sub_line.append(current_sub_bit)

    # If there are any unprocessed sub-numbers, pad them and store
    if len(sub_line) > 0:
        # Pad the second sub-number with zeros to complete the 256-bit number
        for i in range(sub_line_number - len(sub_line)):
            sub_line.append('0' * mem_output_width)

    # Add the final combined 256-bit number if available
    if len(sub_line) == sub_line_number:
        combined_256_bit_number = ''.join(sub_line[::-1])
        current_word_bytearray = int(combined_256_bit_number, 2).to_bytes(out_format_bit_length , byteorder='little')
        mem_created.extend(current_word_bytearray)

    #debug 
    #      bin_line=''.join(f'{byte:08b}' for byte in mem_created)
    #      (bin_line[::-1])[-128:]
    return mem_created

'''
Alexander Logvinenko 11.07.2024
The command in WLOC is changed according to the V2.0 planning
'''
class WLOCEntry:

    ZERO_MUL_ENTRY = None

    def __init__(self, weight_value=0, weight_index=0, weight_offset=0, shift_down=False, shift_right=False, end_of_oc=False, long_entry=False, is_pair = False, nop=False, is_always_long = False,
                 freeze_wloc=False, ic=0, oc=0, add_op_source_buffer = 0, pair_add = 0, sub_sel_preadder_minus_mode = 0, nop_reason = '?', clc_nr=0, 
                 pad_line_first = False, pad_line_last = False, pad_column_first = False, pad_column_last = False):
        # Assign the provided parameters to the instance variables 
        

        #cmd mode
        self.long_entry = long_entry
        self.is_pair = is_pair
        self.is_always_long = is_always_long

        #command information
        self.weight_value = weight_value
        self.weight_index = weight_index # this is the AMM addr
        self.weight_offset = weight_offset # this is offset

        #Additional command information for long and pairing
        self.pair_add = pair_add                                       # address of the pair
        self.sub_sel_preadder_minus_mode = sub_sel_preadder_minus_mode # 0 if preadder "+", 1 for "-"

        #control grid param (in long mode only)
        self.shift_down = shift_down
        self.shift_right = shift_right
        self.pad_line_first = pad_line_first
        self.pad_line_last = pad_line_last
        self.pad_column_first = pad_column_first
        self.pad_column_last = pad_column_last
        self.end_of_oc = end_of_oc # Lock Result in WLOC Table for Convolution

        #Addition of pseudo commands and flags (not existing in CMD format)
        self.nop = nop
        self.nop_reason = nop_reason
        self.clc_nr = clc_nr
        self.freeze_wloc = freeze_wloc
        self.ic = ic #real input chanal, that is calculating (as in kernel)
        self.oc = oc

        
        # Check if the short cmd needs to be replaced by the long cmd
        if weight_offset > MAX_OFFSET_VALUE_FOR_SHORT_ENTRY:
            self.long_entry = True

#       # Check if the long cmd has the correct address    
        if self.long_entry: # Make sure that long entry index fits # of address(weight_index) bits
                if weight_index > MAX_ADDRESS_VALUE_FOR_LONG_ENTRY:
                    raise ValueError('Regular conv: Weight index (%d) too big cant fit long WLOC index bits (%d)' % (self.weight_index,LONG_ENTRY_BITS))
        
        # if freeze_wloc is requerd, make sure, there is weight_value, weight_index, long_entry
        if self.freeze_wloc:
            if sum(self.weight_value+self.weight_index+ (not self.long_entry)) !=0:
                raise ValueError('Alex, check the freeze comand insert')
            
            self.weight_value = 0
            self.weight_index = 0
            self.long_entry = True

    @classmethod
    def get_zero_mul_entry(cls):        
        if cls.ZERO_MUL_ENTRY is None:
            cls.ZERO_MUL_ENTRY = cls(weight_value=0, weight_index=1, weight_offset=1, long_entry=True)
        return copy.deepcopy(cls.ZERO_MUL_ENTRY)

    def is_oc_end(self):
        return self.end_of_oc
    
    def wloc_cmd_len(self):
        # Long Pair
        if self.long_entry and self.is_pair:
            return WLOC_PAIR_LONG_SIZE
        
        # Short Pair
        if not self.long_entry and self.is_pair:
            return WLOC_PAIR_SHORT_SIZE
        
        # Single Long 
        if self.long_entry and not self.is_pair:
            return  WLOC_SINGLE_LONG_SIZE

        # Single Short 
        if not self.long_entry and not self.is_pair:
            return  WLOC_SINGLE_SHORT_SIZE
    
    def set_long(self):
        if (self.nop==False) and (self.long_entry==False):
            self.long_entry=True

    def set_eof(self):
        if (self.weight_index==0 and self.weight_value==0 and self.long_entry==True):
            raise ValueError('EOF command cant be on NOP_STALL')
        self.long_entry=True
        self.end_of_oc=True
        self.is_always_long=True
        self.nop = False 
        if (self.weight_index==0 and self.weight_value==0):
            self.weight_index=1 # to avoid NOP       

    def __str__(self):

        # Long Pair
        if self.long_entry and self.is_pair:
            str = ('LP_ W:%d@%d offset:%d B@%d S_%d' % (self.weight_value, self.weight_index, 0, self.pair_add, self.sub_sel_preadder_minus_mode))
        
        # Short Pair
        if not self.long_entry and self.is_pair:
            str = ('SP_ W:%d@%d offset:%d B@%d S_%d' % (self.weight_value, self.weight_index,self.weight_offset,self.pair_add, self.sub_sel_preadder_minus_mode))
        
        # Single Long 
        if self.long_entry and not self.is_pair:
            str = ('LS_ W:%d@%d offset:%d' % (self.weight_value, self.weight_index, 0))

        # Single Short 
        if not self.long_entry and not self.is_pair:
            str = ('SS_  W:%d@%d offset:%d' % (self.weight_value, self.weight_index, self.weight_offset))

        if self.pad_line_first:
            str = str + '_PLin_0_'
        if self.pad_line_last:
            str = str + '_PLin_N_'   
        if self.pad_column_first:
            str = str + '_PCol_0_' 
        if self.pad_column_last:
            str = str + '_PCol_N_'
        if self.end_of_oc:
            str = str + ' EOC'
        if self.freeze_wloc:
            str = str + ' FREEZE'
        if True or self.nop:
            str = str + 'NOP ' +'REASON: ' + self.nop_reason
        if self.shift_down:
            str = str + 'SHD'  
        if self.shift_right:
            str = str + 'SHR'  

        
        #mark end of channal
        str = str + ('(%d)' % self.oc)

        return str
    
    def to_hex(self):
        return format(int(self.to_bits(), 2), "06X")

    def to_bits(self):
        weight_value = bin(self.weight_value & 0b11111111)[2:].zfill(
            8)  # Dans: I use the &0b11111111 so that negative numbers will have heading ones
        
        # the flags that are depending from W0 or W1 
        shift = (self.shift_down or self.shift_right)
        pad_line   = (self.pad_line_first   or self.pad_line_last  )
        pad_column = (self.pad_column_first or self.pad_column_last) 
        
        if self.nop==True:
            if self.long_entry==True:
                raise ValueError('NOP command must be long entry')
            if self.is_pair==True:
                raise ValueError('NOP command cant be pair entry')
            if self.weight_value!=0:
                raise ValueError('NOP command must have zero weight value, index and offset')
            if self.weight_offset!=0:
                raise ValueError('NOP command must have zero offset')
            
            # For NOP we will use a short CMD with zero offset
            bits = ('0b'                                                      + # Prefix to indicate that the string value is written in binary format                    
                    bin(0)[2:].zfill(SHORT_ENTRY_JUMP_BITS)                   + # Offset [2:0] 
                    '0'                                                       + # Code of the command
                    weight_value                                                # Weight
                    )
            return bits
        
        #for Pairs do the calculations
        if self.is_pair:
            pair_add_part     = bin(self.pair_add    )[2:].zfill(10+1)
            weight_index_part = bin(self.weight_index)[2:].zfill(10+1)
            if pair_add_part[-1]!="0" or weight_index_part[-1]!='1':
                raise ValueError('Check the odd-even logic') 
        
        # Long Pair
        if self.long_entry and self.is_pair:

            bits = ('0b'                                     + # Prefix to indicate that the string value is written in binary format
                    bin(0)[2:].zfill(2).zfill(6)             + # reserved                  
                    bin(self.sub_sel_preadder_minus_mode)[2:]+ # ADD_SUB_sel for preadder
                    bin(self.end_of_oc)[2:]                  + # lock result
                    bin(pad_column)[2:]                      + # WLOC 0/2 - column 7: subtract offset for AMM addr. Pad in first wrap
                                                               # WLOC 1/3 - column 0: Add offset for AMM addr. Pad in last wrap
                    bin(pad_line)[2:]                        + # LOC 0/2 - Pad Line 0 | WLOC 1/3 - Pad last line 

                    bin(shift)[2:]                           + # up_right for WLOC 0,2 and down for WLOC 1,3
                    pair_add_part[:-1]                       + # Address B [9:0] +1K
                    weight_index_part[:-1]                   + # Address A [9:0]
                    '111'                                    + # Code of the command
                    weight_value                               # Weight
                    )
            if len(bits)!=(WLOC_PAIR_LONG_SIZE+2):
                raise ValueError('PAIR_LONG WLOC entry must me 42 bits. Found %d bits' % (len(bits)-2)) 

       
        # Short Pair
        if not self.long_entry and self.is_pair:
            offest_A = bin(self.weight_offset)[2:].zfill(SHORT_PAIR_ENTRY_JUMP_BITS+1)
            if offest_A[-1]!='0':
                raise ValueError('Offset have to be even, because of even-odd logic')
            bits = ('0b'                    + # Prefix to indicate that the string value is written in binary format                    
                    bin(self.sub_sel_preadder_minus_mode)[2:]                + # ADD_SUB_sel for preadder
                    offest_A[     :-1]                                       + # Offset 7-bits 
                    pair_add_part[:-1]                                       + # Address B [9:0] +1K
                    '01'                                                     + # Code of the command
                    weight_value                               # Weight
                    )
            if len(bits)!=(WLOC_PAIR_SHORT_SIZE+2):
                raise ValueError('PAIR_SHORT WLOC entry must me 28 bits. Found %d bits' % (len(bits)-2)) 

        # Single Short 
        if not self.long_entry and not self.is_pair:
            bits = ('0b'                                                      + # Prefix to indicate that the string value is written in binary format                    
                    bin(self.weight_offset)[2:].zfill(SHORT_ENTRY_JUMP_BITS)  + # Offset 5-bits 
                    '0'                                                       + # Code of the command
                    weight_value                                                # Weight
                    )
            if len(bits)!=(WLOC_SINGLE_SHORT_SIZE+2):
                raise ValueError('SINGLE_SHORT WLOC entry must me 14 bits. Found %d bits' % (len(bits)-2)) 
             
        # Single Long 
        if self.long_entry and not self.is_pair:
            bits = ('0b'                                      + # Prefix to indicate that the string value is written in binary format                    
                    bin(0)[2:].zfill(2)                       + # reserved
                    bin(self.end_of_oc)[2:]                   + # lock result
                    bin(pad_column)[2:]                       + # WLOC 0/2 - column 7: subtract offset for AMM addr. Pad in first wrap
                                                                # WLOC 1/3 - column 0: Add offset for AMM addr. Pad in last wrap
                    bin(pad_line)[2:]                         + # LOC 0/2 - Pad Line 0 | WLOC 1/3 - Pad last line                   
                    bin(shift)[2:]                            + # up_right for WLOC 0,2 and down for WLOC 1,3
                    bin(self.weight_index)[2:].zfill(11)      + # Address AB [10:0]                    
                    '011'                                     + # Code of the command
                    weight_value                                # Weight
                    )
            if len(bits)!=(WLOC_SINGLE_LONG_SIZE+2):
                raise ValueError('SINGLE_LONG WLOC entry must me 28 bits. Found %d bits' % (len(bits)-2))

        
        return bits

class RQParamEntry:
    def __init__(self,  scale =1, bias = 0, rough_shift_sel = 0, nop_count = 0, shift_count = 0, clc_nr=0, is_config_cmd=False):
                 
                 self.is_config_cmd = is_config_cmd
                 self.shift = (shift_count>0)

                 ####
                 # for nope node
                 self.scale = scale
                 self.bias  = bias
                 self.rough_shift_sel = rough_shift_sel 
                 # for ~nope node
                 self.nop_count = nop_count
                 self.shift_count = shift_count
                 self.clc_nr = clc_nr       
                
    def __str__(self):
        if (self.is_config_cmd ==True):
            #set command
            str = ('SET_COMMAND: scale:%d  bias:%d shift:%d' % (self.scale, self.bias, self.rough_shift_sel))
            return str 
        if (self.shift_count == 0):
            # this is "nop" comand
            str = ('NOP: count:%d' % (self.nop_count))
            return str 
        if (self.shift_count != 0):            
            str = ('SHIFT: count:%d nop count: %d' % ( self.shift_count, self.nop_count))
            return str


        raise ValueError('RQ Command is not valide')        
           
            

    def to_bits(self):
        if (self.is_config_cmd ==True):
            bias_len_in_bits = 42
            bias_mask_str = bin((1 << bias_len_in_bits) - 1)
            bias_mask = int(bias_mask_str, 2)
            bias_value = (self.bias & bias_mask)
            # exampe for 13 bit by Dan: bias_value_bin = bin(self.bias & 0b1111111111111)[2:].zfill(13)  # Dans: I use the &0b111111111111 so that negative numbers will have heading ones

        
            # bits = bin(self.scale)[2:].zfill(14) + bias_value_bin + '00' + bin(self.rough_shift)[2:].zfill(2) + bin(
            #     self.write_mask)[2:].zfill(4)
            # bits = '0b' + bits.zfill(36)

            # this is "set" comand
            bits = ('0b'                                                 +
                    bin(0)[2:].zfill(3)                                  +  # reserved
                    bin(self.rough_shift_sel)[2:].zfill(2)               +  # shift
                    bin(bias_value)[2:].zfill(bias_len_in_bits)          +  # bias
                    bin(self.scale)[2:].zfill(16)                        +  # scale
                    '0'                                                     # CMD bit
                    )
        else:
            # this is "nop" comand
            bits = ('0b'                               + 
                    bin(0)[2:].zfill(45)               +  # reserved
                    bin(self.shift_count)[2:].zfill(8) +  # shift count
                    bin(self.nop_count)[2:].zfill(10)  +  # NOP count
                    '1'                                 # CMD bit             
                    )
            
        
        if len(bits)!=(64+2):
            raise ValueError('RQParam entry must be either 64  bits. Found %d bits' % (len(bits)-2))
        return bits
        
class RTEntry:
    def __init__(self, nop_count = 0, AMM_write_add = 0, AMM_write_mask = 0, scale = 0, resize_grid_sel =0, result_pipeline_reset =0,  CMD_complete = 0, clc_nr=0):
                 
              self.nop_count = nop_count
              self.AMM_write_add = AMM_write_add
              self.AMM_write_mask = AMM_write_mask
              self.scale = scale
              self.resize_grid_sel = resize_grid_sel
              self.result_pipeline_reset = result_pipeline_reset
              self.CMD_complete = CMD_complete
              self.clc_nr = clc_nr

    def to_bits(self):  # Converts to 32 bits entry
        bits = ('0b' + 
                    bin(0)[2:].zfill(3)                         +  # reserved
                    bin(self.nop_count)[2:].zfill(10)           +  # nop_count
                    bin(self.CMD_complete)[2:].zfill(1)         +  # CMD complete
                    bin(self.result_pipeline_reset)[2:].zfill(1)+  # pipeline_reset
                    bin(self.resize_grid_sel)[2:].zfill(1)      +  # resize grid write sel
                    bin(self.scale)[2:].zfill(1)                +  # scale grid write sel
                    bin(self.AMM_write_mask)[2:].zfill(4)       +  # AMM write mask                                                                                                                                                                                    +  # NOP bit
                    bin(self.AMM_write_add)[2:].zfill(11)          # AMM write address
                   )
        ddd = int(bits[2:]) # check if the number is valide
        if len(bits)!=(32+2):
            raise ValueError('RT entry must be 32 bits. Found %d bits' % (len(bits)-2))
        return bits
    
    def __str__(self):
        str_cmd = ''
        str = ''
        if (self.AMM_write_mask > 0):
            # this is "WRITE" command
            str_cmd = "WRITE "
            str = ('AMM_write_add:%d, AMM_write_mask:%d, scale:%d, CMD_complete:%d' % (self.AMM_write_add, self.AMM_write_mask, self.scale, self.CMD_complete))
              
        if (self.result_pipeline_reset==True):
            # this is "buffer enable" command
            str_cmd += "BUF_ACT "
            str += ('NOP_COUNTER: %d, PIPE_RESET %d' % (self.nop_count, self.result_pipeline_reset)) 

        if (self.CMD_complete==True):
            # this is "buffer enable" command
            str_cmd += "_EOS "
            str = ('AMM_write_add:%d, AMM_write_mask:%d, scale:%d, CMD_complete:%d' % (self.AMM_write_add, self.AMM_write_mask, self.scale, self.CMD_complete))    

        if (self.AMM_write_mask == 0) and (self.result_pipeline_reset==False) and (self.CMD_complete==False):   
            # this is "nop" command at the end
            str_cmd += "NOPS"
            str += ('NOP_COUNTER: %d' % (self.nop_count))  
          
        return str_cmd+str

class LinearFunctionEntry:
    def __init__(self, number):              
        self.number = number

    def to_bits(self):              
        bits = ('0b' +                     
                bin(self.number)[2:].zfill(8) # number                   
                )
        #ddd = int(bits[2:]) # check if the number is valide
        if len(bits)!=(8+2):
            raise ValueError('NLF entry must be 8 bits. Found %d bits' % (len(bits)-2))
        return bits

    def __str__(self):   
        return str(self.number) 
    
class NonLinearFunctionEntry:
    def __init__(self, number, lut_int8):
        self.number = lut_int8[number]
        
    def to_bits(self):              
        bits = ('0b' +                     
                bin2c(self.number, 8)
                )
        #ddd = int(bits[2:]) # check if the number is valide
        if len(bits)!=(8+2):
            raise ValueError('NLF entry must be 8 bits. Found %d bits' % (len(bits)-2))
        return bits

    def __str__(self):   
        return str(self.number)

class CommandList(ABC):

    def __init__(self):
        self.cmd_list = []
        self.cmd_mem = bytearray(0)
        self.cmd_mem_address = 0
        self.list_size = 0# In bits
        # These attributes must be declared in a non-abstract class
        self.mem_input_width = 0
        self.mem_output_width = 0
        self.mem_output_depth = 0 
        self.mem_cmd_resolution = 0

    def add_entry(self,entry):
        # is_last_entry_nop =  (len(self.cmd_list)!=0) and (self.cmd_list[-1].nop==True)        
                
        # if entry.nop and is_last_entry_nop:
        #     self.cmd_list[-1].nop_count+=1
        # else:
        self.cmd_list.append(copy.deepcopy(entry))

    def create_cmd_mem(self, is_clal_only = False):
        cmd_bytes_array = create_mem( list_str_nummers = self.cmd_list, mem_cmd_resolution = self.mem_cmd_resolution, mem_input_width=self.mem_input_width, mem_output_width=self.mem_output_width)

        if not is_clal_only:
            debug_whatch = len(cmd_bytes_array)
            if (len(cmd_bytes_array))>((self.mem_output_depth*self.mem_output_width)//8//2):
                    raise ValueError ('MEM_SIZE size (0x%d) exceeds max mem size (0x%d) - (size in bytes)' % (len(cmd_bytes_array),self.mem_output_depth))
            
        self.cmd_mem = cmd_bytes_array



class WLOCList (CommandList):
    def __init__(self, cmd_list = None):
            super().__init__()
            self.mem_input_width    = WLOC_INPUT_MEM_WIDTH
            self.mem_output_width   = WLOC_OUTPUT_MEM_WIDTH
            self.mem_output_depth   = WLOC_OUTPUT_MEM_DEPTH
            self.mem_cmd_resolution = WLOC_CMD_RESOLUTION

            self.cmd_list = copy.deepcopy(cmd_list) if cmd_list is not None else [] 

  
 
    def set_oc_end(self):
        self.set_long()
        self.cmd_list[-1].end_of_oc = True
        self.cmd_list[-1].nop = False
        # if the last cmd was NOP change it 
        if self.cmd_list[-1].weight_value == 0  and self.cmd_list[-1].weight_offset ==0:
            self.cmd_list[-1].weight_offset = 1 # the namber that is not 0, because zero used for nop
            self.cmd_list[-1].weight_index =  1

    def set_long(self):
        last_entry = self.cmd_list[-1]
        if not last_entry.long_entry: # If it wasnt originally a long entry we must update the per_clock_list_size and list_size variables
            last_entry.long_entry = True  # end of oc can only be signaled in long entry
        
    def add_entry(self,entry):
        super().add_entry(entry)
        # if entry.is_pair:
        #     raise ValueError('Pair not ready yet.  ')


    # if posible, replase long to short
    def add_entry_smart(self,entry):
        entry_to_add = copy.deepcopy(entry)
        if entry.is_pair:
            raise ValueError('Pair not ready yet.')
        if  (0<entry_to_add.weight_offset) and (entry_to_add.weight_offset <= MAX_OFFSET_VALUE_FOR_SHORT_ENTRY) and (entry_to_add.long_entry):
            # Check if the long cmd needs to be replaced by the short cmd
            entry_to_add.long_entry = False
        self.add_entry(entry_to_add)
    
    def del_last_n_cmd (self, num_cmd_to_del):
         self.cmd_list= self.cmd_list[:-num_cmd_to_del]


    def optimise_wloc_with_shorts(self):
        act_inx = self.cmd_list[0].weight_index
        ch_not_init = True
        for inx_el, act_el in enumerate(self.cmd_list):       
            
            
            # this case is to find NOPs
            if ((act_el.is_pair == False) and (act_el.is_always_long==False) and (act_el.long_entry)   and (act_el.end_of_oc == False) and (act_el.freeze_wloc==False) and (act_el.nop == False) and 
                (act_el.pad_column_first == False) and (act_el.pad_column_last== False) and (act_el.pad_line_first== False) and (act_el.pad_line_last == False) and 
                (act_el.shift_down == False) and (act_el.shift_right == False) and 
                (act_el.weight_offset == 1) and  (act_el.weight_value == 0)):

                act_el.nop           = True
                act_el.long_entry    = False
                act_el.weight_offset = 0
                act_el.weight_index  = 0

                
            # this is for short without pair
            elif ((act_el.is_pair == False) and  (act_el.long_entry)   and (act_el.end_of_oc == False) and (act_el.freeze_wloc==False) and (act_el.nop == False) and 
                (act_el.pad_column_first == False) and (act_el.pad_column_last== False) and (act_el.pad_line_first== False) and (act_el.pad_line_last == False) and 
                (act_el.shift_down == False) and (act_el.shift_right == False) and 
                (ch_not_init == False)):

                #check if jump in memory shorter than MAX_OFFSET_VALUE_FOR_SHORT_ENTRY
                cmd_offset  = act_el.weight_index - act_inx
                if (0 <= cmd_offset and  cmd_offset <= MAX_OFFSET_VALUE_FOR_SHORT_ENTRY):
                    act_el.long_entry    = False
                    act_el.weight_offset = cmd_offset
                    act_inx = act_el.weight_index
            
            #this is for pairing short            
            elif (SHORT_PAIRING_USED and
                  (act_el.is_pair == True) and  (act_el.long_entry)   and (act_el.end_of_oc == False) and (act_el.freeze_wloc==False) and 
                  (act_el.nop == False)    and  (act_el.pad_column_first == False) and               (act_el.pad_column_last== False) and 
                  (act_el.pad_line_first== False)                                  and                (act_el.pad_line_last == False) and 
                  (act_el.shift_down == False) and   (act_el.shift_right == False) and                (ch_not_init == False)):
                cmd_offset  = act_el.weight_index - act_inx
                if (0 <= cmd_offset and  cmd_offset <= MAX_OFFSET_VALUE_FOR_PIAR_ENTRY):
                    act_el.long_entry    = False
                    act_el.weight_offset = cmd_offset
                    act_inx = act_el.weight_index     


            # flags to set        
            if(act_el.long_entry):
                act_inx = act_el.weight_index
                ch_not_init = False

            if(act_el.end_of_oc ==True):
                ch_not_init = True


            # if (self.cmd_list[inx_el].end_of_oc== True): check the cases 
            #     pass             
                    
                  
    def find_EOC_to_split(self, available_mem, cmd_num_AMM_sync = 2, min_per_oc_channel=11):        
    
        # Temporary variables for accumulating sub-numbers
        mem_full_block_count      = available_mem        //self.mem_output_width        
        block_num_pro_input_block = self.mem_input_width // self.mem_output_width
        extra_cmd             = []
        sub_bit_len           = 0
        block_128_count       = 0
        ops_per_oc_channel    = 0 # min operation per oc have to be HARDWARE_NOP_REQ + shift_reg_length+num_of_AMM_write_cmd+extra_3x3_shifts

        if (block_num_pro_input_block==0):
            raise ValueError('This version do not support mem_input_width < mem_output_width')
        
        # Process the input list 'list_str_nummers'
        for cmd_inx, cand_cmd in enumerate (self.cmd_list):

            #count the operation per_output_channal
            ops_per_oc_channel    +=1

            # Mark the last EOC in block  
            if cand_cmd.end_of_oc == True:
                cmd_inx_split = cmd_inx
                # each output chanal have to be minimum 11 operation
                extra_nop_to_add      = max(0, min_per_oc_channel-ops_per_oc_channel) 
                extra_cmd = extra_cmd + [WLOC_SINGLE_SHORT_SIZE]*extra_nop_to_add
                ops_per_oc_channel    = 0
                
            cand_cmd_len = cand_cmd.wloc_cmd_len() 
                   

            list_parts_len = [self.mem_cmd_resolution]*(cand_cmd_len//self.mem_cmd_resolution) + extra_cmd

            for i_part in  list_parts_len:

                # Add the current binary string to the accumulator
                sub_bit_len = i_part+sub_bit_len

                # Check if we have reached the length of mem_output_width
                if sub_bit_len > self.mem_output_width:
                    
                    block_128_count += 1
                    sub_bit_len = i_part 
                    
                    # If memory is full stop
                    if block_128_count == mem_full_block_count-1:
                        mem_size_bit  = ((1+block_128_count)//block_num_pro_input_block)*self.mem_input_width//8 #+1 for round ceil
                        return cmd_inx_split, mem_size_bit
                     
            # each channal needs 2 NOP for write AMM and extra DMA sync nops for each chanal
            if (cand_cmd.end_of_oc == True):
                extra_cmd = [WLOC_SINGLE_SHORT_SIZE]*cmd_num_AMM_sync
            else:
                extra_cmd = []

        
        #if we are here, the whole wloc can be written in the mem
        if sub_bit_len>0:
            #we need extra one block of 128 to put the rest in it
            block_128_count += 1

        mem_size_bit  = ((1+block_128_count)//block_num_pro_input_block)*self.mem_input_width #+1 for round ceil
        
        return cmd_inx,mem_size_bit              
             
    def calc_mem_size_in_bit(self):

        # size_in_bit = 0
        # for i_cmd in self.cmd_list:
        #     if       i_cmd.long_entry and     i_cmd.is_pair:
        #         raise ValueError('No Pairing yet')
        #     elif not i_cmd.long_entry and     i_cmd.is_pair:
        #         raise ValueError('No Pairing yet')
        #     elif     i_cmd.long_entry and not i_cmd.is_pair:
        #         size_in_bit+=WLOC_SINGLE_LONG_SIZE
        #     elif not i_cmd.long_entry and not i_cmd.is_pair:
        #         size_in_bit+=WLOC_SINGLE_SHORT_SIZE  

        # TODO - write the calculation without the call of create_cmd_mem 
        self.create_cmd_mem(is_clal_only = True)             
        return  len(self.cmd_mem)
      

    def remove_last_entry(self):
        raise ValueError('This function is removed by Alex')

        
    def insert_entry(self,pos,entry):        
    
        if pos==len(self.cmd_list):
            self.add_entry(entry)
        elif pos>(len(self.cmd_list)):
            if entry.nop:
                return
            else:
                raise ValueError ('Tried to insert a wloc command in position which is bigger than list length')
        else:
            self.cmd_list.insert(pos,entry)
        
    def replace_posible_nop_with_zero_mul(self, pos):
        if self.cmd_list[pos].nop == False:
            # this is not nop
            return
        if pos>=len(self.cmd_list):
            #the pos is out of list len
            return
        
        self.cmd_list[pos] = WLOCEntry.get_zero_mul_entry()
        i_pos = pos+1
        find_next_valid_cmd = False
        while (i_pos< len (self.cmd_list)) and (find_next_valid_cmd == False):
            if self.cmd_list[i_pos].nop:
                i_pos+=1
            else:
                self.cmd_list[i_pos].set_long()
                find_next_valid_cmd = True   

    def get_size(self):
        raise ValueError('This function is removed by Alex')

    
    def get_size_at_clock(self,clock):
        raise ValueError('This function is removed by Alex')


    def create_wloc_mem(self,split_idx,wloc_idx):
        raise ValueError('This function is removed by Alex')



    def split_at(self,clock_idx):
        raise ValueError('This function is removed by Alex')


class RQParamList(CommandList): 
    def __init__(self):
        super().__init__()
        self.mem_input_width    = RQPARAMS_INPUT_MEM_WIDTH
        self.mem_output_width   = RQPARAMS_OUTPUT_MEM_WIDTH
        self.mem_output_depth   = RQPARAMS_OUTPUT_MEM_DEPTH
        self.mem_cmd_resolution = RQPARAMS_CMD_RESOLUTION
                
    def add_entry(self,entry):
        super().add_entry(entry)
            
class RTableList(CommandList):
    def __init__(self):
        super().__init__()
        self.mem_input_width    = RTABLE_INPUT_MEM_WIDTH
        self.mem_output_width   = RTABLE_OUTPUT_MEM_WIDTH
        self.mem_output_depth   = RTABLE_OUTPUT_MEM_DEPTH
        self.mem_cmd_resolution = RTABLE_CMD_RESOLUTION

class LinearFunctionList(CommandList):
    def __init__(self):
        super().__init__()
        self.mem_input_width    = RTABLE_INPUT_MEM_WIDTH
        self.mem_output_width   = RTABLE_OUTPUT_MEM_WIDTH
        self.mem_output_depth   = RTABLE_OUTPUT_MEM_DEPTH
        self.mem_cmd_resolution = RTABLE_CMD_RESOLUTION
        # Creating a list of integers from 0 to 255
        self.cmd_list = [LinearFunctionEntry(i) for i in range(256)]

class NonLinearFunctionList(CommandList):
    def __init__(self, lut_int8):
        super().__init__()
        self.mem_input_width    = RTABLE_INPUT_MEM_WIDTH
        self.mem_output_width   = RTABLE_OUTPUT_MEM_WIDTH
        self.mem_output_depth   = RTABLE_OUTPUT_MEM_DEPTH
        self.mem_cmd_resolution = RTABLE_CMD_RESOLUTION
        # Creating a list of integers from 0 to 255
        self.cmd_list = [NonLinearFunctionEntry(i, lut_int8) for i in range(256)]

class CBC_IR:
    def __init__(self, num_grids):
        self.num_grids = num_grids
        # this have to be removed
        #self.wlocs = [] # This contains a list of wlocs lists as each conv have several lists of wlocs. Each list has num_grids WLOCList
        #self.rqlocs = [] # This contains a list of rqlocs as each conv can have several RQLOCList
        #self.rqparams = RQParamList()
        self.per_oc_non_empty_ic_groups = {}
        #----------------------- alex new format    
           
        self.alex_wlocs     = [] # create of array of WLOCList
        self.alex_rqParam   = [] # create of array of 2*split RQParamList. For example 
        self.RTable         = [] 

    def create_wloc_mem(self):
        for split_idx,wloc_split in enumerate(self.wlocs):
            for wloc_idx,wloc in enumerate(wloc_split):
                wloc.create_wloc_mem(split_idx,wloc_idx)

    def create_rqloc_mem(self):
        for split_idx,rqloc_split in enumerate(self.rqlocs):
            rqloc_split.create_cmd_mem()

    def create_rqparams_mem(self):
        self.rqparams.create_cmd_mem()



class WLOC_spliter:
    def __init__(self, wlocs):
        """
        :param wlocs: list of wlocs (each string consists of '0' and '1')
        :param max_len: maximum allowed length of output subwlocs
        :param max_ones: maximum allowed contiguous '1's in a substring
        """
        self.wlocs = wlocs
        num_of_grids = len(wlocs)//2
        self.splited_wlocs = [[] for _ in range(2*num_of_grids)]
        self.len_of_dead_zone = 15 if kernel_size == 1 else 17
        self.max_wloc_size_bite = ((WLOC_OUTPUT_MEM_DEPTH*WLOC_OUTPUT_MEM_WIDTH)//8//2)

        # this size need to be here to put the freeze cmd
        self.reserved_memory_size = WLOC_OUTPUT_MEM_WIDTH//8

    def split(self):
        """Splits all wlocs according to the rules and returns a list of lists."""
        is_empty = all(not elem.cmd_list for elem in self.wlocs)
        while not is_empty:
            #calkulate the length of the split, to be doen
            line_num_to_cut = 1000 # TODO, automitize it

            #find the longest w_loc
            longest_wloc_index = 0
            max_wloc_length    = 0
            for inx_wloc, i_wloc in enumerate(self.wlocs):
                
                current_length = i_wloc.calc_mem_size_in_bit()
                if current_length > max_wloc_length:
                    max_wloc_length = current_length
                    longest_wloc_index = inx_wloc

            # line_num_to_cut
            parts_to_need = math.ceil(max_wloc_length / self.max_wloc_size_bite)
            each_part_mem_bites = math.ceil(max_wloc_length/parts_to_need)

            # if something went wrong, and the calculated size was not correct, correct it (this case must NOT happen)
            if (each_part_mem_bites>self.max_wloc_size_bite):
                each_part_mem_bites= self.max_wloc_size_bite - self.reserved_memory_size


            #read WLOC, till the the memory is not full
            for inx_wloc, i_wloc in enumerate (self.wlocs):
                self._split_wlocs(inx_wloc, line_num_to_cut)

            is_empty = all(not elem.cmd_list for elem in self.wlocs)

        return self.splited_wlocs



    def _split_wlocs(self, inx_wloc, split_index):
        """
        Splits a wloc in  the given line_num_to_cut.
        If split_index falls on ded_zone, it moves forward to the next cmd.
        
        :param index: The index of the string in self.wlocs
        :param split_index: The position where the split is intended to happen
        """
        wloc_cand = self.wlocs[inx_wloc]

        # Ensure split_index in the len
        if split_index >= len(wloc_cand.cmd_list):
            split_index = len(wloc_cand.cmd_list)


        # find EOC (if exixsts)
        i_EOF = split_index
        is_eof_found = False
        while i_EOF > 0 and is_eof_found ==False:
            if wloc_cand.cmd_list[i_EOF].end_of_oc == True:
                is_eof_found = True
            else:     
                i_EOF -= 1

        #find write_cmd if exists
        if is_eof_found and (split_index-i_EOF)<self.len_of_dead_zone:
            split_index = i_EOF 

  
        # Extract the left part and update the original string
        
        self.splited_wlocs.extend(wloc_cand.cmd_list[:split_index])
        del wloc_cand.cmd_list[:split_index]  # Remaining part after split

    def validate(self, split_wlocs):
        """
        Checks if the split result satisfies constraints.
        :param split_wlocs: list of lists (each inner list contains split parts of a string)
        :return: True if all conditions are met, otherwise False
        """
        for parts in split_wlocs:
            for part in parts:
                if len(part) > self.max_len:  # Проверка максимальной длины
                    return False
                if '1' * (self.max_ones + 1) in part:  # Проверка максимального блока '1'
                    return False
        return True
    ########################################

    
    




# TODO: Change name to DSP State and add values to it
# class Grid:
#     def __init__(self):
#         # TODO: 4 is hardcoded
#         self.shiftEntry = FirstWLOCEntry()
#         self.gridMacResults = []
#         self.gridMacOCResults = [[0] * 14 for _ in range(14)]
#         self.readShiftEntry = True
#         self.runningOffset = 0
#         self.offsetToAdvance = 0

#         self.frozenChannel = False

#         # for k=3 positioning, set to default
#         self.left = 1
#         self.right = 15
#         self.up = 15
#         self.down = 1


# class Clock:
#     def __init__(self):
#         #self.AMMWriteMask = 0
#         self.AMM0EvenCounter = 0
#         self.AMM0OddCounter = 0
#         self.AMM1EvenCounter = 0
#         self.AMM1OddCounter = 0
#         self.firstAddCommand = True
#         self.currentClockRQMult = False
#         self.nextClockRQMult = False


# class ConvolutionBitsToParse:
#     def __init__(self):
#         self.WlocBits = []
#         self.RQCMDEntriesBits = ""
#         self.paramEntriesBits = ""


# class SimulatorState:
#     def __init__(self, AMMGRIDSIZEX, AMMGRIDSIZEY, AMMROWS, NUMBEROFAMMS, DDR):
#         self.grids = [Grid() for i in range(4)]
#         self.AMMMemory = [["00" * AMMGRIDSIZEX * AMMGRIDSIZEY] * AMMROWS for i in range(NUMBEROFAMMS)]
#         #self.AMMWriteMask = []
#         self.clockData = Clock()
#         self.BitsToParse = ConvolutionBitsToParse()

#         self.DDR = DDR

#         # For debug
#         self.destResults = []

#         self.paramEntryList = []
#         self.paramEntryListCounterMult = 0
#         self.paramEntryListCounterWrite = 0
#         self.resetCounter = 0

#         self.ParseRQCommands = ""
#         self.ParseRQParams = ""

#         self.finalRQInAMM = []

    # def writeToAMMMemory(self, writeTo0Even, writeTo0Odd, writeTo1Even, writeTo1Odd, EvenOffset, OddOffset, grid):
    #     stringRows = ""
    #     for row in grid:
    #         tempRow = ""
    #         for element in row:
    #             tempRow += format(element, "02x")
    #         stringRows += tempRow

    #     if writeTo0Even:
    #         self.AMMMemory[0][EvenOffset + self.clockData.AMM0EvenCounter] = stringRows
    #         self.clockData.AMM0EvenCounter += 1

    #     if writeTo0Odd:
    #         self.AMMMemory[0][OddOffset + self.clockData.AMM0OddCounter] = stringRows
    #         self.clockData.AMM0OddCounter += 1

    #     if writeTo1Even:
    #         self.AMMMemory[1][EvenOffset + self.clockData.AMM1EvenCounter] = stringRows
    #         self.clockData.AMM1EvenCounter += 1
    #     if writeTo1Odd:
    #         self.AMMMemory[1][OddOffset + self.clockData.AMM1OddCounter] = stringRows
    #         self.clockData.AMM1OddCounter += 1

