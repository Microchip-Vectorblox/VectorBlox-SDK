class DMR_AMM:
    def __init__(self):
        self.ddr_start_addr = 32
        self.input_x        = 12
        self.input_y        = 12
        self.input_z        = 12
        self.nlines_to_read = 5
        self.amm_nchannels  = 12
        self.amm_start_addr = 12
        self.amm_mask       = 4
        self.amm_write_mode = 2
        self.y_folding      = 4
        self.x_wrapping     = 5
        self.flag_wait      = 6
        self.flag_set       = 6
        self.opcode         = 4

        self.ddr_start_addr_offset = 0
        self.input_x_offset        = self.ddr_start_addr_offset + self.ddr_start_addr
        self.input_y_offset        = self.input_x_offset + self.input_x
        self.input_z_offset        = self.input_y_offset + self.input_y
        self.nlines_to_read_offset = self.input_z_offset + self.input_z
        self.amm_nchannels_offset  = self.nlines_to_read_offset + self.nlines_to_read
        self.amm_start_addr_offset = self.amm_nchannels_offset + self.amm_nchannels
        self.amm_mask_offset       = self.amm_start_addr_offset + self.amm_start_addr
        self.amm_write_mode_offset = self.amm_mask_offset + self.amm_mask
        self.y_folding_offset      = self.amm_write_mode_offset + self.amm_write_mode
        self.x_wrapping_offset     = self.y_folding_offset + self.y_folding
        self.flag_wait_offset      = self.x_wrapping_offset + self.x_wrapping
        self.flag_set_offset       = self.flag_wait_offset + self.flag_wait
        self.opcode_offset         = self.flag_set_offset + self.flag_set

class DMW_AMM:
    def __init__(self):
        self.ddr_start_addr  = 32
        self.output_x        = 12
        self.output_y        = 12
        self.output_z        = 12
        self.nlines_to_read  = 5
        self.amm_nchannels   = 12
        self.amm_start_addr  = 12
        self.amm_write_mode  = 2
        self.y_folding       = 4
        self.x_wrapping      = 5
        self.grid_start_line = 4
        self.flag_wait       = 6
        self.flag_set        = 6
        self.opcode          = 4

        self.ddr_start_addr_offset  = 0
        self.output_x_offset        = self.ddr_start_addr_offset + self.ddr_start_addr
        self.output_y_offset        = self.output_x_offset + self.output_x
        self.output_z_offset        = self.output_y_offset + self.output_y
        self.nlines_to_read_offset  = self.output_z_offset + self.output_z
        self.amm_nchannels_offset   = self.nlines_to_read_offset + self.nlines_to_read
        self.amm_start_addr_offset  = self.amm_nchannels_offset + self.amm_nchannels
        self.amm_write_mode_offset  = self.amm_start_addr_offset + self.amm_start_addr
        self.y_folding_offset       = self.amm_write_mode_offset + self.amm_write_mode
        self.x_wrapping_offset      = self.y_folding_offset + self.y_folding
        self.grid_start_line_offset = self.x_wrapping_offset + self.x_wrapping
        self.flag_wait_offset       = self.grid_start_line_offset + self.grid_start_line 
        self.flag_set_offset        = self.flag_wait_offset + self.flag_wait
        self.opcode_offset          = self.flag_set_offset + self.flag_set

class DMR_TABLES:
    def __init__(self):
        self.ddr_start_addr   = 24
        self.table_length     = 24
        self.table_type       = 4
        self.table_start_addr = 16
        self.table_param      = 8
        self.reserved         = 36
        self.flag_wait        = 6
        self.flag_set         = 6
        self.opcode           = 4

        self.ddr_start_addr_offset   = 0
        self.table_length_offset     = self.ddr_start_addr_offset + self.ddr_start_addr
        self.table_type_offset       = self.table_length_offset + self.table_length
        self.table_start_addr_offset = self.table_type_offset + self.table_type
        self.table_param_offset      = self.table_start_addr_offset + self.table_start_addr
        self.reserved_offset         = self.table_param_offset + self.table_param
        self.flag_wait_offset        = self.reserved_offset + self.reserved
        self.flag_set_offset         = self.flag_wait_offset + self.flag_wait
        self.opcode_offset           = self.flag_set_offset + self.flag_set


class RD_FROM_DDR:
    def __init__(self):
        self.src_address = 64
        self.length = 32
        self.dst_type = 3
        self.priority_pipe = 1
        self.dst_info = 16
        self.flag_wait = 4
        self.flag_set = 4
        self.opcode = 4

        self.src_address_offset = 0
        self.length_offset = self.src_address_offset + self.src_address
        self.dst_type_offset = self.length_offset + self.length
        self.priority_pipe_offset = self.dst_type_offset + self.dst_type
        self.dst_info_offset = self.priority_pipe_offset + self.priority_pipe
        self.flag_wait_offset = self.dst_info_offset + self.dst_info
        self.flag_set_offset = self.flag_wait_offset + self.flag_wait
        self.opcode_offset = self.flag_set_offset + self.flag_set


class WR_TO_DDR:

    def __init__(self):
        self.dst_address = 64
        self.length = 32
        self.src_type = 3
        self.priority_pipe = 1
        self.src_info = 16
        self.flag_wait = 4
        self.flag_set = 4
        self.opcode = 4

        self.dst_address_offset = 0
        self.length_offset = self.dst_address_offset + self.dst_address
        self.src_type_offset = self.length_offset + self.length
        self.priority_pipe_offset = self.src_type_offset + self.src_type
        self.src_info_offset = self.priority_pipe_offset + self.priority_pipe
        self.flag_wait_offset = self.src_info_offset + self.src_info
        self.flag_set_offset = self.flag_wait_offset + self.flag_wait
        self.opcode_offset = self.flag_set_offset + self.flag_set


class ENGINE_OPERATION:
    def __init__(self):
        self.operation = 2
        self.amm_src_addr = 12
        self.amm_dst_addr = 12
        self.output_padding_start_x = 4
        self.output_padding_start_y = 4
        self.output_pad_value = 8
        self.input_pad_value = 8
        self.pad_mode = 4
        self.wloc_start_addr = 12
        self.rqcmd_start_addr = 12
        self.rqparams_start_addr = 12
        self.scale_mode = 2
        self.reset_amm_address = 1
        self.reserved1 = 1
        self.stride = 3
        self.wrap_count = 6
        self.x_wrapping = 9
        self.flag_wait = 6
        self.flag_set = 6
        self.opcode = 4

        self.operation_offset = 0
        self.amm_src_addr_offset = self.operation_offset + self.operation
        self.amm_dst_addr_offset = self.amm_src_addr_offset + self.amm_src_addr
        self.output_padding_start_x_offset = self.amm_dst_addr_offset + self.amm_dst_addr
        self.output_padding_start_y_offset = self.output_padding_start_x_offset + self.output_padding_start_x
        self.output_pad_value_offset = self.output_padding_start_y_offset + self.output_padding_start_y
        self.input_pad_value_offset = self.output_pad_value_offset + self.output_pad_value
        self.pad_mode_offset = self.input_pad_value_offset + self.input_pad_value
        self.wloc_start_addr_offset = self.pad_mode_offset + self.pad_mode
        self.rqcmd_start_addr_offset = self.wloc_start_addr_offset + self.wloc_start_addr
        self.rqparams_start_addr_offset = self.rqcmd_start_addr_offset + self.rqcmd_start_addr
        self.scale_mode_offset = self.rqparams_start_addr_offset + self.rqparams_start_addr
        self.reset_amm_address_offset = self.scale_mode_offset + self.scale_mode
        self.reserved1_offset = self.reset_amm_address_offset + self.reset_amm_address
        self.stride_offset = self.reserved1_offset + self.reserved1
        self.wrap_count_offset = self.stride_offset + self.stride
        self.x_wrapping_offset = self.wrap_count_offset + self.wrap_count
        self.flag_wait_offset = self.x_wrapping_offset + self.x_wrapping
        self.flag_set_offset = self.flag_wait_offset + self.flag_wait
        self.opcode_offset = self.flag_set_offset + self.flag_set

