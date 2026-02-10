from common.enums import SequencerOPCode, EngineOpCode, DDRCommandDstSrcType, CustomeOpCode, PadMode, ScalingMode, DDRReadOffsetType, DDRWriteOffsetType
from common.hw_config import BYTES_IN_SEQUENCER_COMMAND,  DMR_TABLE_LENGTH_BYTES, NLTABLE_HALF_LENGTH, RQPARAMS_INPUT_MEM_DEPTH, RQPARAMS_OUTPUT_MEM_DEPTH, \
      RTABLE_INPUT_MEM_DEPTH, RTABLE_OUTPUT_MEM_DEPTH, RTABLE_START_ADD_RESOLUTION, WLOC_INPUT_MEM_DEPTH, WLOC_OUTPUT_MEM_DEPTH, TFLITE_REQUANT, WLOC_OUTPUT_MEM_WIDTH, NO_NEIGHBOR_SLICE
from common.utils import bin2c
import math
FIRST_AMM_ADDR_MSB_BIT = 4
AMM_MSB_DIVIDER = 2 ** FIRST_AMM_ADDR_MSB_BIT

class DDRReadDstInfoProgram:
    def __init__(self,program_mem_start_address):
        self.program_mem_start_address = program_mem_start_address # 16 bits
    def to_hex(self):
        hex_output ='0x'+hex(self.program_mem_start_address)[2:].zfill(4)
        return hex_output

class DDRReadDstInfoAMM:
    def __init__(self,amm_start_address):
        self.amm_start_address = amm_start_address #12 bits
        
    def to_hex(self):
        hex_output = '0x'+hex(self.amm_start_address)[2:].zfill(4)
        return hex_output

class DDRReadDstInfoWLOC:
    def __init__(self,wloc_mask,wloc_buffer_id):
        if wloc_mask & 0b1010101 and wloc_mask & 0b10101010:
            raise ValueError ('Cant write wloc simultneously to even and odd grids since they use different axi ports')
        self.wloc_mask = wloc_mask #8 bits
        self.wloc_buffer_id = wloc_buffer_id # 1 bit , bit8
        
    def to_hex(self):
        hex_output = '0x'+hex(self.wloc_buffer_id)[2:].zfill(2)+hex(self.wloc_mask)[2:].zfill(2)
        return hex_output

class DDRReadDstInfoTables:
    def __init__(self,table_buffer_id,axi_id):
        self.table_buffer_id = table_buffer_id # 1 bit , bit8
        
    def to_hex(self):
        table_buffer_id_supported_values = [0,1]
        if self.table_buffer_id not in table_buffer_id_supported_values:
            raise ValueError('Bad value for table_buffer_id. Supported values are: %s, got %d' % (str(table_buffer_id_supported_values),self.table_buffer_id))
        hex_output = '0x'+hex(self.table_buffer_id)[2:].zfill(2)+hex(0)[2:].zfill(2)
        return hex_output

class DDRWriteSrcInfoAMM:
    def __init__(self,amm_start_address,amm_select):
        self.amm_start_address = amm_start_address #12 bits
        self.amm_select = amm_select # 2 bits
        
    def to_hex(self):
        hex_output = hex(int(bin(self.amm_select)[2:].zfill(2)+'00',2)) + hex(self.amm_start_address)[2:].zfill(3)
        return hex_output

class SequencerEntry:
    def __init__(self,opcode = SequencerOPCode.DMR_AMM, set_flags = 0, wait_flags = 0):
        self.opcode = opcode # 4 bits
        self.wait_flags = wait_flags # 4 bits
        self.set_flags = set_flags # 4 bits
    def __str__(self):

        str_value = ('Cmd:%s set:0x%X wait:0x%X' % (str(self.opcode), self.set_flags, self.wait_flags))
        return str_value
    
    def __repr__(self):
        if self.opcode == SequencerOPCode.DMR_TABLES:
            str_value = 'TABLES,'
        else:
            str_value = ('%s,' % ((str(self.opcode).split('_'))[0]))
        for flag in [32,16,8,4,2,1]:
            if self.wait_flags & self.set_flags & flag:
                str_value = str_value + "WS,"
            elif self.wait_flags & flag:
                str_value += "W,"
            elif self.set_flags & flag:
                str_value += "S,"
            else:
                str_value += " ,"
        return str_value


class SequencerDDRReadEntry(SequencerEntry):
    def __init__(self, src_address=0, length = 0, input_x = 0, ddr_read_offset_type = DDRReadOffsetType.MODEL_MEM, input_y = 0, input_z = 0, nlines_to_read = 0, amm_nchannels = 0, 
                 amm_mask = 0, amm_start_address = 0, amm_write_mode = 0, y_folding = 0, x_wrapping = 0, dst_info = None, set_flags = 0,wait_flags = 0,
                 dst_type = DDRCommandDstSrcType.AMM, buffer_id = 0, write_mask_alex = 0):
        if (dst_type == DDRCommandDstSrcType.AMM):
            super().__init__(opcode=SequencerOPCode.DMR_AMM, set_flags = set_flags, wait_flags=wait_flags)
        else:
            super().__init__(opcode=SequencerOPCode.DMR_TABLES, set_flags = set_flags, wait_flags=wait_flags)
        
        self.ddr_start_address = src_address # 30 bits
        self.ddr_read_offset_type = ddr_read_offset_type # 2 bit
        self.input_x = input_x # 12 bits
        self.input_y = input_y # 12 bits
        self.input_z = input_z # 12 bits
        self.nlines_to_read = nlines_to_read # 5 bits
        self.amm_nchannels = amm_nchannels # 12 bits
        self.amm_start_address = amm_start_address # 12 bits
        self.amm_mask = amm_mask # 4 bits
        self.amm_write_mode = amm_write_mode # 2 bits
        self.y_folding = y_folding # 4 bits
        self.x_wrapping = x_wrapping # 5 bits

        self.dst_type = dst_type
        self.dst_info = dst_info 
        
        self.table_ddr_start_address = math.ceil(src_address/4096) # 24 bits
        self.table_length = math.ceil(length/DMR_TABLE_LENGTH_BYTES) * DMR_TABLE_LENGTH_BYTES # 24 bits
        self.table_type = dst_type # 4 bits
        #self.table_start_address = buffer_id * 1024 #16 bits

        # Alex this was
        #self.table_start_address = buffer_id * 512 #16 bits
        #but the addres of the pin-pong buffer dependes from the table memmory allocation,
        #  so for WLOC 1K*256 -> 2K*128 and the half of it is 1K
        #         RQPA 1K*256 -> 4K*64  and the half of it is 2K
        #         RT   1K*256 -> 8K*32 and the half is 4K
        #         NL   1K*256 -> 512
        # so the write calculation is like this:
        if (dst_type == DDRCommandDstSrcType.WLOC) :
            half_length = WLOC_INPUT_MEM_DEPTH     //2
        elif (dst_type == DDRCommandDstSrcType.RQPARAMS):
            half_length = RQPARAMS_INPUT_MEM_DEPTH //2
        elif (dst_type == DDRCommandDstSrcType.RESULTTABLE):
            half_length = RTABLE_INPUT_MEM_DEPTH   //2
        elif (dst_type == DDRCommandDstSrcType.NONLINEARFUNCTION):
            half_length = NLTABLE_HALF_LENGTH
        elif (dst_type == DDRCommandDstSrcType.AMM):
            half_length = 0
        elif (dst_type == DDRCommandDstSrcType.PROGRAM):
            half_length = 0
        else:
            raise ('Strange type of the Command. This is not a table at all')
        self.table_start_address = buffer_id * half_length  


        if (dst_type == DDRCommandDstSrcType.WLOC) :
            self.table_param = dst_info.wloc_mask # 8 bits
        #add alex    
        elif (dst_type == DDRCommandDstSrcType.RQPARAMS):
            self.table_param = write_mask_alex
        else:        
            self.table_param = 0 # 8 bits

        
    def __str__(self):
        str_value = super().__str__().strip()
        str_value = str_value+(',%s'  % (str(self.dst_type)))
        if self.dst_type == DDRCommandDstSrcType.AMM:
            str_value = str_value+(',DDR_START_ADDR:0x%x' % (self.ddr_start_address))
            str_value = str_value+(',DDR_OFFSET:%d' % (self.ddr_read_offset_type.value))
            str_value = str_value+(',input_x:%d' % (self.input_x))
            str_value = str_value+(',input_y:%d' % (self.input_y))
            str_value = str_value+(',input_z:%d' % (self.input_z))
            str_value = str_value+(',nlines_to_read:%d' % (self.nlines_to_read))
            str_value = str_value+(',amm_nchannels:%d' % (self.amm_nchannels))
            str_value = str_value+(',amm_start_address:0x%x' % (self.amm_start_address))
            str_value = str_value+(',amm_mask:%d' % (self.amm_mask))
            str_value = str_value+(',amm_write_mode:%d' % (self.amm_write_mode))
            str_value = str_value+(',y_folding:%d' % (self.y_folding))
            str_value = str_value+(',x_wrapping:%d' % (self.x_wrapping))
        else:
            str_value = str_value+(',DDR_START_ADDR:0x%x' % (self.table_ddr_start_address))
            str_value = str_value+(',table_length:0x%x' % (self.table_length))
            str_value = str_value+(',table_type:%s' % (self.table_type.name))
            str_value = str_value+(',table_start_address:0x%x' % (self.table_start_address))
            str_value = str_value+(',table_param:0x%x' % (self.table_param))
        
        return str_value

    def __repr__(self):
        str_value = super().__repr__().strip()
       
        if self.dst_type in [DDRCommandDstSrcType.WLOC, DDRCommandDstSrcType.RQPARAMS, 
                             DDRCommandDstSrcType.NONLINEARFUNCTION, DDRCommandDstSrcType.RESULTTABLE,
                             DDRCommandDstSrcType.PROGRAM]:
            str_name = self.dst_type.name
            str_value += str_name + ', ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,0x%x, 0x%x, 0x%x, 0x%x' % (self.table_ddr_start_address, self.table_length, self.table_start_address, self.table_param)  
        

        elif self.dst_type == DDRCommandDstSrcType.AMM:
            str_value += 'AMM,,0x%x,%d,%d,%d,%d,%d,,%d,0x%x,%d,%d,%d,%d,,,,,,,,,,,,,,,,,,,,' % (self.ddr_start_address, self.ddr_read_offset_type.value, self.input_x, self.input_y, self.input_z, \
                                                          self.nlines_to_read, self.amm_nchannels, self.amm_start_address, self.amm_mask, self.amm_write_mode, \
                                                          self.y_folding, self.x_wrapping)
        
        return str_value
    
    def to_hex(self):
        if self.dst_type == DDRCommandDstSrcType.AMM:
            #DMR_AMM command see doc
            bits   = ( '0b'                                                +
                        bin(self.opcode.value              )[2:].zfill(4 ) +
                        bin(self.set_flags                 )[2:].zfill(6 ) +
                        bin(self.wait_flags                )[2:].zfill(6 ) + 
                        bin(self.x_wrapping                )[2:].zfill(5 ) +
                        bin(self.y_folding                 )[2:].zfill(4 ) + 
                        bin(self.amm_write_mode            )[2:].zfill(2 ) + 
                        bin(self.amm_mask                  )[2:].zfill(4 ) + 
                        bin(self.amm_start_address         )[2:].zfill(12) + 
                        bin(self.amm_nchannels             )[2:].zfill(12) +
                        bin(self.nlines_to_read            )[2:].zfill(5 ) + 
                        bin(self.input_z                   )[2:].zfill(12) +
                        bin(self.input_y                   )[2:].zfill(12) +
                        bin(self.input_x                   )[2:].zfill(12) +
                        bin(self.ddr_read_offset_type.value)[2:].zfill(2 ) +
                        bin(self.ddr_start_address         )[2:].zfill(30)
                    )

        else:
            # DMR_TABLES command see doc
            bits   = (  '0b'                                                + 
                        bin(self.opcode.value              )[2:].zfill(4 )  +
                        bin(self.set_flags                 )[2:].zfill(6 )  +
                        bin(self.wait_flags                )[2:].zfill(6 )  + 
                        bin(0                              )[2:].zfill(36)  +  # reserved 36 bits
                        bin(self.table_param               )[2:].zfill(8)   +
                        bin(self.table_start_address       )[2:].zfill(16)  +
                        bin(self.table_type.value          )[2:].zfill(4)   +
                        bin(self.table_length              )[2:].zfill(24)  +
                        bin(self.table_ddr_start_address   )[2:].zfill(24)
                    )
        if len(bits)!=(128+2):
            raise ValueError('Command have to be 128  bits. Found %d bits' % (len(bits)-2))
        
        hex_128bits = hex(int(bits, 2))

        return hex_128bits

class SequencerDDRWriteEntry(SequencerEntry):
    def __init__(self, dst_address=0, length = 0, ddr_write_offset_type = DDRWriteOffsetType.MODEL_MEM, output_x = 0, output_y = 0, output_z = 0, nlines_to_write = 0, amm_nchannels = 0,
                 src_type = DDRCommandDstSrcType.AMM , amm_start_address = 0, amm_write_mode = 0,
                 y_folding = 0, x_wrapping = 0, grid_start_line = 0, set_flags = 0,wait_flags = 0):
        super().__init__(opcode=SequencerOPCode.DMW_AMM, set_flags = set_flags, wait_flags=wait_flags)
        self.ddr_start_address = dst_address # 30 bits
        self.ddr_write_offset_type = ddr_write_offset_type # 2 bit
        self.output_x = output_x # 12 bits
        self.output_y = output_y # 12 bits
        self.output_z = output_z # 12 bits
        self.nlines_to_write = nlines_to_write # 5 bits
        self.amm_nchannels = amm_nchannels # 12 bits
        self.amm_start_address = amm_start_address # 12 bits
        self.amm_write_mode = amm_write_mode # 2 bits
        self.y_folding = y_folding # 4 bits
        self.x_wrapping = x_wrapping # 5 bits
        self.grid_start_line = grid_start_line # 4 bits

        self.src_type = src_type
        self.dst_address = dst_address
        self.length = length

    def __str__(self):
        str_value = super().__str__().strip()
        str_value = str_value+(',%s'  % (str(self.src_type)))
        if self.src_type == DDRCommandDstSrcType.AMM:
            str_value = str_value+(',DDR_START_ADDR:0x%x' % (self.ddr_start_address))
            str_value = str_value+(',DDR_OFFSET:%d' % (self.ddr_write_offset_type.value))
            str_value = str_value+(',output_x:%d' % (self.output_x))
            str_value = str_value+(',output_y:%d' % (self.output_y))
            str_value = str_value+(',output_z:%d' % (self.output_z))
            str_value = str_value+(',nlines_to_write:%d' % (self.nlines_to_write))
            str_value = str_value+(',amm_nchannels:%d' % (self.amm_nchannels))
            str_value = str_value+(',amm_start_address:%d' % (self.amm_start_address))
            str_value = str_value+(',amm_write_mode:%d' % (self.amm_write_mode))
            str_value = str_value+(',y_folding:%d' % (self.y_folding))
            str_value = str_value+(',x_wrapping:%d' % (self.x_wrapping))
            str_value = str_value+(',grid_start_line:%d' % (self.grid_start_line))
        else:
            str_value = str_value+(',DDR_START_ADDR:0x%x' % (self.dst_address))
        return str_value

    def __repr__(self):
        str_value = super().__repr__().strip()
        str_value += 'AMM,,0x%x,%d,%d,%d,%d,,%d,%d,0x%x,,%d,%d,%d,%d,,,,,,,,,,,,,,,,,,,' % (self.ddr_start_address, self.ddr_write_offset_type.value, self.output_x, self.output_y, self.output_z, \
                                                          self.nlines_to_write, self.amm_nchannels, self.amm_start_address, self.amm_write_mode, \
                                                          self.y_folding, self.x_wrapping, self.grid_start_line)
        
        return str_value
    
    def to_hex(self):
        flag_wait_bits_and_flag_set_hex = hex(int(bin(self.set_flags)[2:].zfill(6)+bin(self.wait_flags)[2:].zfill(6),2))[2:].zfill(3)
        if self.src_type == DDRCommandDstSrcType.AMM:
            temp_bins = bin(self.x_wrapping)[2:].zfill(5) + bin(self.y_folding)[2:].zfill(4) + bin(self.amm_write_mode)[2:].zfill(2) \
                        + bin(self.amm_start_address)[2:].zfill(12) + bin(self.amm_nchannels)[2:].zfill(12) \
                        + bin(self.nlines_to_write)[2:].zfill(5)
            temp_hex = hex(int(temp_bins, 2))[2:].zfill(10)
            hex_128bits = '0x'+hex(self.opcode.value)[2:] + flag_wait_bits_and_flag_set_hex + hex(self.grid_start_line)[2:] + temp_hex \
                            + hex(self.output_z)[2:].zfill(3) + hex(self.output_y)[2:].zfill(3) + hex(self.output_x)[2:].zfill(3) \
                            + hex(int(bin(self.ddr_write_offset_type.value)[2:].zfill(2) + bin(self.ddr_start_address)[2:].zfill(30), 2))[2:].zfill(8)
        else:
            hex_128bits = '0x'+hex(self.opcode.value)[2:] + flag_wait_bits_and_flag_set_hex \
                            + hex(0)[2:].zfill(20) + hex(self.dst_address)[2:].zfill(8)
        return hex_128bits

class SequencerEngineOperationEntry(SequencerEntry):
    def __init__(self, operation = EngineOpCode.CONV, AMM_URAM_EVEN_GRID_src_addr = 0, AMM_URAM_EVEN_GRID_dst_addr = 0,
                output_padding_start_x = 7, output_padding_start_y = 11, output_pad_value = 0, input_pad_value = 0, 
                pad_mode = PadMode.GRIDH14XW8, wloc_buffer_id = 0, rt_buffer_id = 0, rqparams_buffer_id = 0, nlt_buffer_id = 0, scaling_mode=ScalingMode.RQDIRECT, 
                slice_backward_offset = NO_NEIGHBOR_SLICE, slice_forward_offset = NO_NEIGHBOR_SLICE, fold_left_right = 0,
                reset_amm_address = 1, set_flags = 0, wait_flags = 0, description = ''):
        super().__init__(opcode=SequencerOPCode.ENGINE_OPERATION, set_flags = set_flags, wait_flags=wait_flags)
        self.operation = operation # 2 bits
        self.amm_src_address = AMM_URAM_EVEN_GRID_src_addr # 11 bits
        self.amm_dst_address = AMM_URAM_EVEN_GRID_dst_addr # 11 bits
        self.output_padding_start_x= output_padding_start_x # 4 bits
        self.output_padding_start_y= output_padding_start_y # 4 bits
        self.output_pad_value = output_pad_value # 8 bits
        self.input_pad_value = input_pad_value # 8 bits
        self.pad_mode = pad_mode # 4 bits
        #self.wloc_start_address = wloc_buffer_id * 1024 # 11 bits
        #self.rt_start_address = rt_buffer_id * 1024 # 11 bits
        #self.rqparams_start_address = rqparams_buffer_id * 1024 # 11 bits
        self.wloc_start_address     = wloc_buffer_id     * (WLOC_OUTPUT_MEM_DEPTH    //2)   # 11 bits
        self.rqparams_start_address = rqparams_buffer_id * (RQPARAMS_OUTPUT_MEM_DEPTH//4)   # 11 bits
        self.rt_start_address       = rt_buffer_id    * (RTABLE_OUTPUT_MEM_DEPTH  //4 //RTABLE_START_ADD_RESOLUTION) # 12 bits
        
        self.nlt_start_addr_flag    = nlt_buffer_id       # NLF is simle simaphore bit
        self.scale_mode = scaling_mode # 2 bits , bits 110-111 , '00' RQ direct 1:1 ,'01' - Folding 2:1 , '10' - Resize 1:2, '11' - unfolding
        self.reset_amm_address = reset_amm_address # 1 bit        

        self.description = description 

        self.num_of_lines=0 # This attribute is only used for enabling fpga simulator to generate intermediate nxo files
        self.read_start_line=0 # This attribute is only used for enabling fpga simulator to generate intermediate nxo files
        self.write_start_line=0 # This attribute is only used for enabling fpga simulator to generate intermediate nxo files
        self.conv_input_channels=0 # This attribute is only used for enabling fpga simulator to generate intermediate nxo files for input
        
        self.slice_backward_offset= slice_backward_offset
        self.slice_forward_offset = slice_forward_offset
        self.fold_left_right = int(fold_left_right)
        
    def __str__(self):
        str_value = super().__str__().strip()
        str_value = str_value + ',' + self.description
        str_value = str_value + (',wloc_addr:0x%x,rt_addr:0x%x,rqparams_addr:0x%x, nlt_addr_flag:0x%x' % (self.wloc_start_address, self.rt_start_address, self.rqparams_start_address, self.nlt_start_addr_flag))
        if self.scale_mode == ScalingMode.FOLDING2_1:
            str_value = str_value + (',scl_mode:fold')
        elif self.scale_mode == ScalingMode.RESIZE1_2:
            str_value = str_value + (',scl_mode:resize')
        elif self.scale_mode == ScalingMode.UNFOLDING:
            str_value = str_value + (',scl_mode:unfolding')
        str_value = str_value + (',out_pad:(%d,%d)' % (self.output_padding_start_y, self.output_padding_start_x))            
        str_value = str_value + (',rst_amm:%d' % self.reset_amm_address)
        str_value = str_value + (',AMM_Src:0x%x,AMM_Dst:0x%x' % (self.amm_src_address, self.amm_dst_address))
        str_value = str_value + (',out_pad_val:%d,in_pad_val:%d,pad_mode:%d' % (self.output_pad_value, self.input_pad_value, self.pad_mode.value))
        str_value = str_value + (',line num:%d,line start:%d' % (self.num_of_lines,(self.write_start_line-self.read_start_line)))
        str_value = str_value + (',input_channels:%d' % (self.conv_input_channels))
        str_value = str_value + (',fold_left_right:%d, slice_backward_offset%d, slice_forward_offset%d' % 
                                 (self.fold_left_right, self.slice_backward_offset, self.slice_forward_offset))
        
        return str_value
    
    def __repr__(self):
        str_value = super().__repr__().strip()
        str_value += ',%s,,,,,,,,,,,,,,, 0x%x, 0x%x, 0x%x, 0x%x, %d, %d, %d, %d, %d, %d, %d,0x%x,0x%x,0x%x,,,,' % (
            self.description, 
            self.amm_src_address, self.amm_dst_address,
            self.slice_backward_offset, self.slice_forward_offset, self.scale_mode.value, self.fold_left_right,
            (self.write_start_line-self.read_start_line), self.num_of_lines, 
            self.output_padding_start_y, self.output_padding_start_x, self.conv_input_channels,
            self.wloc_start_address, self.rt_start_address, self.rqparams_start_address
            )
        return str_value
    
    def to_hex(self):
        bits =  ( '0b'+
                   bin(self.opcode.value           )[2:].zfill(4 ) +
                   bin(self.set_flags              )[2:].zfill(6 ) +
                   bin(self.wait_flags             )[2:].zfill(6 ) +
                   bin(0                           )[2:].zfill(1 ) + # reserved 1 bit
                   bin(self.nlt_start_addr_flag    )[2:].zfill(1)  +
                   bin(self.fold_left_right        )[2:].zfill(1)  +          
                   bin(self.scale_mode.value       )[2:].zfill(2 ) +
                   bin(self.rqparams_start_address )[2:].zfill(11) +
                   bin(self.rt_start_address       )[2:].zfill(11) +
                   bin(self.wloc_start_address     )[2:].zfill(11) +
                   bin(self.pad_mode.value         )[2:].zfill(1 ) +
                   bin2c(self.input_pad_value, 8)                  +
                   bin2c(self.output_pad_value, 8)                 +
                   bin(self.output_padding_start_y )[2:].zfill(4 ) +
                   bin(self.output_padding_start_x )[2:].zfill(4 ) +
                   bin(self.slice_backward_offset  )[2:].zfill(12) +
                   bin(self.slice_forward_offset   )[2:].zfill(12) +
                   bin(self.amm_dst_address        )[2:].zfill(12) +
                   bin(self.amm_src_address        )[2:].zfill(12) +
                   bin(self.operation.value        )[2:].zfill(1 )
                   
               ) 
        if len(bits)!=(128+2):
            raise ValueError('Command have to be 128  bits. Found %d bits' % (len(bits)-2))
        
        hex_128bits = hex(int(bits, 2))
        if  not TFLITE_REQUANT:
            raise ('only tflite format is suported')
        return hex_128bits

class SequencerCustomOperationEntry(SequencerEntry):
    def __init__(self, operation = CustomeOpCode.NOP, set_flags = 0,wait_flags = 0):
        super().__init__(opcode=SequencerOPCode.CUSTOM_OPERATION, set_flags = set_flags, wait_flags=wait_flags)
        self.operation = operation # 8 bits

    def to_hex(self):
        flag_wait_bits_and_flag_set_hex = hex(int(bin(self.set_flags)[2:].zfill(6)+bin(self.wait_flags)[2:].zfill(6),2))[2:].zfill(3)
        hex_128bits = '0x'+hex(self.opcode.value)[2:] + flag_wait_bits_and_flag_set_hex \
                        + hex(0)[2:].zfill(26) + hex(self.operation.value)[2:].zfill(2)
        return hex_128bits

class SequencerProgram:
    def __init__(self):
        self.commands_list = []
        self.commands_mem = bytearray(0)
        self.commands_mem_address = 0

    def create_program_mem(self):
        sequencer_program_bytes_array = bytearray(0)
        for current_command in self.commands_list:
            current_command_hex = current_command.to_hex()
            current_command_bytearray = int(current_command_hex,16).to_bytes(BYTES_IN_SEQUENCER_COMMAND,byteorder='little')
            sequencer_program_bytes_array.extend(current_command_bytearray)
        self.commands_mem = sequencer_program_bytes_array
    def save_sequencer_program_debug_file(self,debug_filename):
        with open(debug_filename,'w') as sequencer_program_info_file:
            for command_index,current_sequencer_command in enumerate(self.commands_list):
                entry_info = '0x%X: %s' % (command_index,str(current_sequencer_command))
                sequencer_program_info_file.write(entry_info+'\n')
    def save_sequencer_program_csv_file(self,csv_filename):
        with open(csv_filename,'w') as sequencer_program_info_file:
            temp_str = 'PC,Operation,F5,F4,F3,F2,F1,F0,Type,Layer,'
            temp_str += 'ddr_start_address,ddr_offset,X,Y,Z,nlines_to_read,nlines_to_write,'
            temp_str += 'amm_nchannels,amm_start_address,amm_mask,amm_write_mode,y_folding,x_wrapping,grid_start_line,'
            temp_str += 'AMM Src,AMM Dst,back_X_offset,forw_X_offset,scale_mode,fold_left_right,1st line,#Lines,Pad Y,Pad X,Input Ch,wloc_addr,rt_addr,rqparams_addr,'
            temp_str += 'Table_ddr_start_address,Table_length,Table_start_address,Table_param \n'
            sequencer_program_info_file.write(temp_str)    
            for command_index,current_sequencer_command in enumerate(self.commands_list):
                entry_info = '0x%X, %s' % (command_index,repr(current_sequencer_command))
                sequencer_program_info_file.write(entry_info+'\n')
    def move_sequencer_block(self,target_pointer,source_start_pointer,source_end_pointer):                
        if target_pointer>=source_start_pointer:
            raise ValueError ('at: move_sequencer_block, Target pointer must be smaller than source pointer')
        num_ops_to_move = source_end_pointer-source_start_pointer+1
        table_read_ops=[]
        for i in range(num_ops_to_move):
            table_read_ops.append(self.commands_list.pop(source_start_pointer))
        self.commands_list[target_pointer:target_pointer] = table_read_ops


    def is_table_read_command(self,sequencer_command):
        if sequencer_command.opcode!=SequencerOPCode.DMR_AMM:
            return False
        if sequencer_command.dst_type not in [DDRCommandDstSrcType.WLOC, DDRCommandDstSrcType.RQPARAMS, DDRCommandDstSrcType.RESULTTABLE,DDRCommandDstSrcType.NONLINEARFUNCTION, DDRCommandDstSrcType.PROGRAM]:
            return False
        return True
    def push_tables_read_before_ddr_rw(self):
        sequencer_pointer = len(self.commands_list)-1
        op_found=False
        while not op_found and sequencer_pointer>=0: # We find the last engine op command in sequence
            if self.commands_list[sequencer_pointer].opcode==SequencerOPCode.ENGINE_OPERATION:
                op_found=True
            sequencer_pointer=sequencer_pointer-1
        while sequencer_pointer>=0:
            last_table_read_pointer = sequencer_pointer
            while self.is_table_read_command(self.commands_list[sequencer_pointer]) and sequencer_pointer>=0: # We find the 1st table read command
                sequencer_pointer=sequencer_pointer-1
            first_table_read_pointer=sequencer_pointer+1
            op_found=False
            while not op_found and sequencer_pointer>=0: # We find the previous op command in sequence
                if self.commands_list[sequencer_pointer].opcode==SequencerOPCode.ENGINE_OPERATION:
                    op_found=True
                sequencer_pointer=sequencer_pointer-1
            if sequencer_pointer==0:
                break
            target_pointer = sequencer_pointer+2
            if target_pointer!=first_table_read_pointer:
                self.move_sequencer_block(target_pointer,first_table_read_pointer,last_table_read_pointer)


