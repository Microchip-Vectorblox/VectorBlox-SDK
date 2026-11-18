from collections import OrderedDict
import math

# Operation type constants
class OpType:
    CONV = 'Conv'
    TRANSPOSE_CONV = 'TransposeConv'
    ADD = 'Add'
    CONCAT = 'Concat'
    RESIZE = 'Resize'
    SYNC = 'Sync'
    IDENTITY = 'Identity'
    MAXPOOL = 'MaxPool'
    AVERAGEPOOL = 'AveragePool'
    GEMM = 'Gemm'

    @classmethod
    def values(cls):
        # All op-type string constants declared on this class.
        return {v for k, v in vars(cls).items()
                if not k.startswith('_') and isinstance(v, str)}

SNP_GRID_HEIGHT = 16
ROUGH_SHIFT_BITS = 2 # 4 possible values, {8, 16, 24, 32} for TFLITE requant
BIAS_FRACTIONAL_BITS = 2
BIAS_MULITIPLIER = math.pow(2,BIAS_FRACTIONAL_BITS)
BIAS_ROUNDING_ADD = 2.5
MAX_MAC_SHIFT = 19
FRACTIONAL_BITS = 8
MAX_BIAS_BITS = 12 #  This excludes the sign bit. so 12 means INT13 bias

MAX_MAC_BITS = 26
REDUCED_MAC_RESCALE_BUS_WIDTH = False # This flag enables reduction of bus width between MAC OUTPUT to RQ. When enabled some of the shifting is done in mac and is setup by special wloc entries of activation and weight shifts
MAX_REDUCE_BUS_WIDTH = 15 # (This is X bits including sign bit) This takes effect only if REDUCED_MAC_RESCALE_BUS_WIDTH is True. it limits mac result after dynamic reduce to MAX_DYNAMIC_REDUCE_WIDTH bits including sign

MCHP_MAC_TO_RQ_BUS_WIDTH = 18
FINAL_RESULTS_BITS=8 # This means the final result is INT8
OVERFLOW_EXTRA_BITS = 4# These are extra bits we take for over/underflow of result
INT_SCALE_BITS = 10 # This means int scale is UINT10
MCHP_ADDED_SCALE_BITS = 4 # IN MCHP numerics we add bits to scale in order to allow shift left of result as fine shift
UINT17_ADDED_SCALE_BITS = 7 # IN UIT17 numerics we add bits to scale in order to allow shift left of result as fine shift
MAC_ROUGH_SHIFT_GRANULARITY = 4 # rough shift allows 0,3,6,9 right shift

# For now, adding Sync to grid ops to prevent many errors in the compilation.
# Alternatively would need to edit various places to add extra parameters or
# use different logic for Sync nodes. For example, if Sync is not a grid op,
# it's missing parameters such as ['backend']['output_padding_start_y'] and
# ['backend']['ic_groups'], which causes failures in update_node_tiling_info
# and get_next_op_grid_config. So e.g. set_node_tiling_info would need to change
# to not call update_node_tiling_info on Sync nodes.
GRID_OPS = ['Conv','Add','MaxPool','Resize','Identity', 'AveragePool']
MULTIPLE_INPUT_OPS = ['Add','Concat']

def amm_capacity(ir):
    # AMM channels available for a single blob (full AMM minus the reserved SYSTEM_zero block).
    return ir.amm_block_size * (ir.amm_num_blocks - 1)

def generic_op_footprint(op_type, num_inputs, input_total_channels, output_total_channels):
    # Peak co-resident AMM footprint for a "generic" (non-Conv) op: a MULTIPLE_INPUT_OP (Add) has its
    # output aliased directly onto input[0]'s already-allocated AMM blocks by the allocator
    # (mem_allocation_per_htile in program_compiler.py: output_even_grid = input_even_grid[...][0]), so it
    # needs single-buffered space for every input and NO separate output buffer. A true single-input op
    # (MaxPool/Resize/AveragePool) produces a distinct output tensor needing its own double-buffered
    # allocation, on top of a single-buffered input.
    if op_type in MULTIPLE_INPUT_OPS:
        return num_inputs * input_total_channels
    return input_total_channels + 2 * output_total_channels
SINGLE_QUANT_PARAMS_OPS = ['Add','MaxPool','Resize','Identity','Sync'] # Unlike conv which has per channel quanization params (scale,zp etc.)
REQUANTING_OPS = ['Add','Conv','Identity']
MULTIPLE_INPUT_NON_REQUANTING_OPS = ['Concat']
NON_REQUANTING_OPS = ['Concat','Resize','MaxPool','Sync'] # Resize and MaxPool can be made requanting ops as they are mapped to conv op that includes rq block. Just need to update rq params to support this
DUAL_CONTIGUOUS_ALLOCATION_OPS = ['Concat']
DUAL_NONCONTIGUOUS_ALLOCATION_OPS = ['Add']
FORCED_OUTPUT_FOLDING_OPS = ['Conv']
WLOC_SYMETRIC_OPS = ['Add'] # These ops must have symetric wloc which means nop insertion should be done in parallel to all grids wloc. This is to keep channels output order same as input order
INLINE_OPS = {'Add':[0]} # OP NAME, Writeinn to the same input
PER_TILE_CBC_OPS = ['Add'] # This is for each cbc to Add (not used)

MINIMAL_ACTUAL_INPUT_WIDTH = 8 
MINIMAL_ACTUAL_INPUT_HEIGHT = 8
MAX_GRID_HEIGHT = SNP_GRID_HEIGHT
MAX_GRID_WIDTH = 16
MAX_X_WRAPPING = 16

MAX_GRID_COUNT = 2 
WLOCS_PER_GRID = 2
WLOC_READ_AXI0_GRIDS=[0,1]
WLOC_READ_AXI1_GRIDS=[]
WLOCS_READ_ON_AXI0=[0,2] # WLOCS 0,1 are for AMM0/Grid0, WLOCS 2,3 are used for AMM1/Grid1
WLOCS_READ_ON_AXI1=[1,3]
AMM_COUNT = 4

AMM_WIDTH = 8
AMM_HEIGHT = SNP_GRID_HEIGHT

NUM_OF_BRAM_AMMS = 0 # if possible to del
BRAM_BLOCK_SIZE = 128 # if possible to del
BRAM_NUM_BLOCKS = 8 # if possible to del
BRAM_FIRST_AMM_INDEX = 0 # if possible to del

#URAM_BLOCK_SIZE = 32
#URAM_NUM_BLOCKS = 64
URAM_BLOCK_SIZE = MAX_GRID_WIDTH
URAM_NUM_BLOCKS = 128
URAM_DEPTH = URAM_BLOCK_SIZE * URAM_NUM_BLOCKS
URAM_SIZE = URAM_BLOCK_SIZE * URAM_NUM_BLOCKS
URAM_FIRST_AMM_INDEX = NUM_OF_BRAM_AMMS # AMM indexes are starting from BRAM AMMs and after that URAM amms. The 1st index of URAM amm will be the number of BRAM AMMs
NO_NEIGHBOR_SLICE = 0x7FF # This value means that there is no neighbor slice in that direction
MODEL_BIN_FILE_GRANULARITY = 4096 # In Lattice design model bin file must be aligned to 4096

DDR_ADDRESS_GRANULARITY = 4096
DDR_BOX_DEPTH = 1
DDR_BOX_HEIGHT = MAX_GRID_HEIGHT
DDR_BOX_WIDTH = MAX_GRID_WIDTH
TENSOR_BOX_HEIGHT = 14
TENSOR_BOX_WIDTH = 14
DDR_BOX_ALIGNMENT = 256
BYTES_PER_LINE_OFFSET_UNIT = 256
# TSNP architecture sizes
TSNP_DDR_BOX_HEIGHT = 1
TSNP_DDR_BOX_WIDTH = 8
TSNP_DDR_BOX_DEPTH = 32
TSNP_DDR_BOX_ALIGNMENT = 256
TSNP_DDR_BYTES_PER_CHANNEL = 8

VBX3_PHASE1_DDR_BOX_HEIGHT = 14
VBX3_PHASE1_DDR_BOX_WIDTH = 16
VBX3_PHASE1_DDR_BOX_DEPTH = 1

DDR_MIN_READ_SIZE = 32
DDR_READ_GRANULARITY_BYTES = 16
DMR_TABLE_LENGTH_BYTES = 32

WLOC_NOPS_AT_LAYER_START=3
STALL_NOPS_FREQ=28
STALLS_CLOCKS = 2

BYTES_IN_SEQUENCER_COMMAND = 16
PROGRAM_MEMORY_NUM_COMMANDS = 1024

NUM_TABLE_BUFFERS = 2
NUM_MUTEX_FLAGS = 4



    
# WLOC Params
SHORT_ENTRY_JUMP_BITS = 5
SHORT_PAIR_ENTRY_JUMP_BITS = 7
DIFF_LEN_BETWEEN_SHORT_AND_LONG = 14
LONG_ENTRY_BITS = 11
MAX_OFFSET_VALUE_FOR_SHORT_ENTRY = 2 ** SHORT_ENTRY_JUMP_BITS - 1
MAX_OFFSET_VALUE_FOR_PIAR_ENTRY  = 2 ** (SHORT_PAIR_ENTRY_JUMP_BITS+1) -1  # +1 because we dont send the last bit
MAX_ADDRESS_VALUE_FOR_LONG_ENTRY = 2 ** LONG_ENTRY_BITS - 1
#MAX_ADDRESS_VALUE_FOR_DEEP_CONV_LONG_ENTRY = 2 ** DEEP_CONV_LONG_ENTRY_BITS - 1


#All RQ is delayed by 12 clocks	
#From EOC to rq_data_read	-4
#From rq_data_read to dsp_op	2
#from dsp_op to dsp out	1
#from dsp_out to dest_write	1

# CBC Creator rq state machine clock definitions

WLOC_NOPS_BEFORE_NEW_OC_IN_CASE_OF_INPUT_SPLIT = 3
RQ_READ_CLOCKS = 1 # Number of clocks from rq start handle of grid until grid can overwrite output register
RQ_NOPS_AT_START = 17
RQ_NOPS_AT_START_MAXPOOL = 19
GRID_OC_END_TO_RQ_READ = -4 # Clocks from end of channel to RQ read command
FROM_RQ_READ_TO_RQ_START_MUL = 2 # From RQ_READ to first multiplication of grid output by re-scale
FROM_LAST_MUL_TO_DSP_OUT = 1 # From last multiplication to dsp_out 
FROM_LAST_MUL_TO_DSP_OUT_MAXPOOL = 1 # From last multiplication to dsp_out in case of maxpool command
RQ_BUSY_CLOCKS = 1 # Value of 1 means rq ops starts back to back
RQ_BUSY_CLOCKS_HW_X_RESIZE = 3

ADDITIONAL_CLOCKS_PER_OC_CALC_IN_CASE_OF_INPUT_SPLIT = WLOC_NOPS_BEFORE_NEW_OC_IN_CASE_OF_INPUT_SPLIT+1 # The +1 is for the FirstWLOCEntry
#ADDITIONAL_RQ_BUSY_FOR_FOLDING_CONV = 5 # Value of 5 means that there will be at least 5 nops between finish of folding rq flow and next rq flow
FROM_DSP_OUT_TO_DEST_WRITE = 1 # number of clocks from rq lock signal to destination write signal 
MULTIPLY_COMMAND_CYCLES = 1 # Multiply command takes X clocks
MULTIPLY_ADD_COMMAND_CYCLES = 3 # Multiply add command need few cycles for FPGA execution of multiply add
RQ_ADDITIONAL_CLOCKS_FOR_FOLDING_CONV_1CHANNEL = 2 # Additional clocks used by folding conv to complete folding scheme in rq for 14x14 folding conv
RQ_ADDITIONAL_CLOCKS_FOR_FOLDING_CONV_2CHANNELS = 5 # Additional clocks used by folding conv to complete folding scheme in rq for 12x16 or 12x32 convs
RQ_ADDITIONAL_CLOCKS_FOR_FOLDING_CONV_4CHANNELS = 5 # Additional clocks used by folding conv to complete folding scheme in rq for 28x28 folding conv
RQ_ADDITIONAL_CLOCKS_FOR_FOLDING_CONV = 3
ADDITIONAL_RQ_BUSY_FOR_FOLDING_CONV = 2
MINIMAL_EOC_TO_FREEZE_DISTANCE = 4 # A value of 4 means that In case FREEZE is right next to EOC, 3 nops will be inserted
MAX_WLOC_128BIT_ENTRIES = 1024 # Set to 10 for debug wloc split (otherwise should be 2048)
BALANCED_WLOC_ENTRIES_LIMIT = True
SPARE_72BIT_ENTRIES_FOR_SPLIT = 48 # These spare wlock entries are used to finish ongoing rq operations in split position
MAX_RQLOC_128BIT_ENTRIES = 1024
MAX_RQPARAMS_128BIT_ENTRIES = 1024
MAX_TABLE_LENGTH = 512

### all this constants are used in CBC generation 
WLOC_CMD_RESOLUTION       = 14 # This is the bits resolution in the wloc command. Like the command can be N*WLOC_CMD_RESOLUTION bits (N= 1,2,3,4)
WLOC_INPUT_MEM_WIDTH      = 128  # WLOC mem width in bits (WRITE acsess granularity)
WLOC_INPUT_MEM_DEPTH      = 1024 # WLOC mem depth in bits (WRITE acsess granularity)
WLOC_OUTPUT_MEM_WIDTH     = 128  # WLOC mem width in bits (READ(!) acsess granularity)
WLOC_OUTPUT_MEM_DEPTH     =  (WLOC_INPUT_MEM_WIDTH*WLOC_INPUT_MEM_DEPTH)//WLOC_OUTPUT_MEM_WIDTH # (READ(!) acsess granularity)

RQPARAMS_CMD_RESOLUTION   = 16   # This is the bits resolution in the RQPARAMS command.
RQPARAMS_INPUT_MEM_WIDTH  = 128  # RQ mem width in bits (WRITE acsess granularity)
RQPARAMS_INPUT_MEM_DEPTH  = 1024 # RQ mem depth in bits (WRITE acsess granularity)
RQPARAMS_OUTPUT_MEM_WIDTH = 64   # RQ mem width in bits (READ(!) acsess granularity)
RQPARAMS_OUTPUT_MEM_DEPTH =  (RQPARAMS_INPUT_MEM_WIDTH*RQPARAMS_INPUT_MEM_DEPTH)//RQPARAMS_OUTPUT_MEM_WIDTH # (READ(!) acsess granularity)

RTABLE_CMD_RESOLUTION     = 16   # This is the bits resolution in the RTABLE command.
RTABLE_INPUT_MEM_WIDTH    = 128  # RTABLE mem width in bits (WRITE acsess granularity)
RTABLE_INPUT_MEM_DEPTH    = 1024 # RTABLE mem depth in bits (WRITE acsess granularity)
RTABLE_OUTPUT_MEM_WIDTH   = 32   # RTABLE mem width in bits (READ(!) acsess granularity)
RTABLE_OUTPUT_MEM_DEPTH   =  (RTABLE_INPUT_MEM_WIDTH*RTABLE_INPUT_MEM_DEPTH)//RTABLE_OUTPUT_MEM_WIDTH # (READ(!) acsess granularity)

# In SequencerEngineOperationEntry, there are not enough bits in rt_start_address to use all the addresses,  
# we need to reduce the address resolution to get access to the whole memory.
RTABLE_START_ADD_RESOLUTION      = 2 # only 2 Addresses are possible: 0 and half of the memory. 

# for NLT we need only the address for pin-pong buffer
NLTABLE_HALF_LENGTH              = 512

#CONST FOR WLOC CMD in bits
WLOC_SINGLE_LONG_SIZE            = 28
WLOC_SINGLE_SHORT_SIZE           = 14
WLOC_PAIR_LONG_SIZE              = 42
WLOC_PAIR_SHORT_SIZE             = 28

# CONSTATNS FOR generate_grids_rq_cbc
USE_BEGIN_SYNC                   = True               # wait before start the RQ, to sinc it with the WLOC (False used for debug only)
RQ_BEGIN_SYNC_NOP                = 8*USE_BEGIN_SYNC   # NOPS for RQ to be ready to ran
RT_BEGIN_SYNC_NOP                = 15*USE_BEGIN_SYNC  # NOPS for RT to be ready to ran
HARDWARE_SYNC_CMD_COMPLATE       = 4                  # This NOPS have to be after last AMM write and the CMD_COMPLATE=0 (END_OF_Slit NOPs)

# CBC Creator constants for TSBP_CBC_IR
ALTERNATIV_STALL_SHIFT            = 7                  # Alternative position offset for STALL insertion
MIN_NUM_CMDS_IN_FIRST_CHANNEL     = 2                  # Minimum number of commands in first channel
MIN_NUM_CMDS_IN_ONE_CHANNEL       = 9                  # Minimum number of commands in one channel
CLK_FROM_EOC_TO_RQ_PARAM          = 7                  # Clocks from end of channel to RQ param command

START_RQ_NOPS = 9
START_RT_NOPS = 14
CLK_FROM_EOC_TO_AMM_WRITE        = 26 

SPLIT_TALE_SIZE_IN_BYTES = (HARDWARE_SYNC_CMD_COMPLATE*WLOC_SINGLE_SHORT_SIZE + 
                                                      1*WLOC_SINGLE_LONG_SIZE +
                                                      2*WLOC_SINGLE_SHORT_SIZE)