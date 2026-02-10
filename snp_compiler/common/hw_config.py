from common.enums import GridConfig
from collections import OrderedDict
import math

SNP_GRID_HEIGHT = 14
ROUGH_SHIFT_BITS = 10 # MAC activation shift allows shift of 10 bits left. weights fine shift allows up to 9 bits shift left
BIAS_FRACTIONAL_BITS = 2
BIAS_MULITIPLIER = math.pow(2,BIAS_FRACTIONAL_BITS)
BIAS_ROUNDING_ADD = 2.5
MAX_MAC_SHIFT = 19
FRACTIONAL_BITS = 8
MAX_BIAS_BITS = 12 #  This excludes the sign bit. so 12 means INT13 bias

MAX_MAC_BITS = 26 # Origianlly was 26
REDUCED_MAC_RESCALE_BUS_WIDTH = False # This flag enables reduction of bus width between MAC OUTPUT to RQ. When enabled some of the shifting is done in mac and is setup by special wloc entries of activation and weight shifts
MAX_REDUCE_BUS_WIDTH = 15 # (This is X bits including sign bit) This takes effect only if REDUCED_MAC_RESCALE_BUS_WIDTH is True. it limits mac result after dynamic reduce to MAX_DYNAMIC_REDUCE_WIDTH bits including sign
MCHP_NUMERICS = True
TFLITE_REQUANT = True # Requantization like TFLite, currently the 2 math block version

TFLITE_MODEL = True

if TFLITE_REQUANT:
    ROUGH_SHIFT_BITS = 2 # 4 possible values, {8, 16, 24, 32}
    # TODO: May need to update these
    BIAS_FRACTIONAL_BITS = 2
    BIAS_MULITIPLIER = math.pow(2,BIAS_FRACTIONAL_BITS)
    BIAS_ROUNDING_ADD = 2.5
    MAX_MAC_SHIFT = 19
    FRACTIONAL_BITS = 8
    MAX_BIAS_BITS = 12 #  This excludes the sign bit. so 12 means INT13 bias

MCHP_MAC_TO_RQ_BUS_WIDTH = 18 # This is used if MCHP_NUMERICS is True
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
GRID_OPS = ['Conv','Add','MaxPool','Resize','Identity','Sync', 'AveragePool']
MULTIPLE_INPUT_OPS = ['Add','Concat']
SINGLE_QUANT_PARAMS_OPS = ['Add','MaxPool','Resize','Identity','Sync'] # Unlike conv which has per channel quanization params (scale,zp etc.)
REQUANTING_OPS = ['Add','Conv','Identity']
MULTIPLE_INPUT_NON_REQUANTING_OPS = ['Concat']
NON_REQUANTING_OPS = ['Concat','Resize','MaxPool','Sync'] # Resize and MaxPool can be made requanting ops as they are mapped to conv op that includes rq block. Just need to update rq params to support this
SINGLE_GRID_OPS = ['MaxPool'] # Currently unused. Can add 'Sync' here too.
LIMITED_GRIDS_OPS = {
    'MaxPool':2
}

DEBUG_2MAXPOOL_GRIDS = True # If True, 2 maxpool grids are supported
DUAL_CONTIGUOUS_ALLOCATION_OPS = ['Concat']
DUAL_NONCONTIGUOUS_ALLOCATION_OPS = ['Add']
FORCED_OUTPUT_FOLDING_OPS = ['Conv']
WLOC_SYMETRIC_OPS = ['Add'] # These ops must have symetric wloc which means nop insertion should be done in parallel to all grids wloc. This is to keep channels output order same as input order
INLINE_OPS = {'Add':[0]} # OP NAME, LIST OF INLINE INPUT INDEX
PER_TILE_CBC_OPS = ['Add'] # Ops that need per tile cbc since index in cbc in influenced by mem allocation
GRID_CONFIGS = OrderedDict ([
    #(8, GridConfig.H14xW8),
    (16, GridConfig.H14xW16),
    ])
SMALLEST_X_RESIZABLE_GRID_SIZE = list(GRID_CONFIGS.keys())[0]
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
WLOC_READ_AXI1_GRIDS=[]
AMM_COUNT = 4
FULL_AMM_WRITE_MASK = 2 ** AMM_COUNT -1 # This is write mask if we want to write to all AMMs
AMMS_INPUT_CHANNELS_SPLITS = { # This defines per number of ic splits which amms will have hold same data
    1: [[0,1]], # All AMMs will hold all input channels
    2: [[0],[1]], # AMMS 0 will hold split 0 of input channels AMMS 1 will hold split 1
}
AMM_WIDTH = 8
AMM_HEIGHT = SNP_GRID_HEIGHT

NUM_OF_BRAM_AMMS = 0
BRAM_BLOCK_SIZE = 128
BRAM_NUM_BLOCKS = 8
BRAM_FIRST_AMM_INDEX = 0

#URAM_BLOCK_SIZE = 32
#URAM_NUM_BLOCKS = 64
URAM_BLOCK_SIZE = 16
URAM_NUM_BLOCKS = 256
URAM_DEPTH = URAM_BLOCK_SIZE * URAM_NUM_BLOCKS
URAM_SIZE = URAM_BLOCK_SIZE * URAM_NUM_BLOCKS
URAM_FIRST_AMM_INDEX = NUM_OF_BRAM_AMMS # AMM indexes are starting from BRAM AMMs and after that URAM amms. The 1st index of URAM amm will be the number of BRAM AMMs
NO_NEIGHBOR_SLICE = 0xFFF # This value means that there is no neighbor slice in that direction
MODEL_BIN_FILE_GRANULARITY = 4096 # In Lattice design model bin file must be aligned to 4096

DDR_ADDRESS_GRANULARITY = 4096
DDR_BOX_DEPTH = 1
DDR_BOX_HEIGHT = 16
DDR_BOX_WIDTH = 16
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

# This mask to grid allows having 2 grids work on same amm, in that case when we write to this amm both grids should not read from that amm.
# format is amm_mask_bit: [grids using that AMM]
AMM_MASK_TO_GRIDS = { 
    1: [0],
    2: [1],
}

def get_grid_config(current_op_width=1):
    current_grid_mode = list(GRID_CONFIGS.items())[-1][1]
    for grid_config in GRID_CONFIGS.items():
        if current_op_width<=grid_config[0]:
            current_grid_mode = grid_config[1]
            break
    return current_grid_mode

def get_grids_per_line(gridconfig: GridConfig):
    if gridconfig==GridConfig.H14xW8:
        return 1
    elif gridconfig==GridConfig.H14xW16:
        return 1
    elif gridconfig==GridConfig.H14xW32:
        return 4
    else:
        raise ValueError ('hw_config.py/get_grids_per_line, grid config not supported')

def get_num_virtual_grids(gridconfig: GridConfig):
    if gridconfig==GridConfig.H14xW8:
        return 2
    elif gridconfig==GridConfig.H14xW16:
        return 1
    elif gridconfig==GridConfig.H14xW32:
        return 0
    else:
        raise ValueError ('hw_config.py/get_grids_per_line, grid config not supported')
    
def get_virtual_grids(node,num_grids): # Virtual grids are grids that support higher resulution and actually composed of some basic grids (e.g. H14W16 is 2 grids of H14W8)
    gridconfig = node['backend']['gridmode']
    if gridconfig==GridConfig.H14xW8:
        if node['op_type'] in LIMITED_GRIDS_OPS:
            if node['op_type']!='MaxPool':
                raise ValueError ('Currently only limited grid op maxpool is supported')
            virtual_grids = [[0],[1]]
        else:
            if num_grids == MAX_GRID_COUNT:
                virtual_grids = [[0],[1]]
            elif num_grids == 1:
                virtual_grids = [[0]]
            else:
                raise ValueError ('This gridmode and grid count is not supported.')
        return virtual_grids
    elif gridconfig==GridConfig.H14xW16:
        if node['op_type'] in LIMITED_GRIDS_OPS:
            if node['op_type']!='MaxPool':
                raise ValueError ('Currently only limited grid op maxpool is supported')
            virtual_grids = [[0,1]]
        else:
            virtual_grids = [[0,1]]
        return virtual_grids
    elif gridconfig==GridConfig.H14xW32:
        if node['op_type'] in LIMITED_GRIDS_OPS:
            raise ValueError ('Currently only limited grid op maxpool in H14xW16 is supported')
        return [[]]
    else:
        raise ValueError ('hw_config.py/get_virtual_grids, grid config not supported')
    
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
WLOC_NOPS_BEFORE_NEW_OC = 3
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
ADDITIONAL_CLOCKS_PER_OC_CALC = WLOC_NOPS_BEFORE_NEW_OC+1 # The +1 is for the FirstWLOCEntry
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

# Alex for Phase 2
WLOC_CMD_RESOLUTION       = 14
WLOC_INPUT_MEM_WIDTH      = 256
WLOC_INPUT_MEM_DEPTH      = 1024
WLOC_OUTPUT_MEM_WIDTH     = 128
WLOC_OUTPUT_MEM_DEPTH     =  (WLOC_INPUT_MEM_WIDTH*WLOC_INPUT_MEM_DEPTH)//WLOC_OUTPUT_MEM_WIDTH

RQPARAMS_CMD_RESOLUTION   = 16
RQPARAMS_INPUT_MEM_WIDTH  = 256
RQPARAMS_INPUT_MEM_DEPTH  = 1024
RQPARAMS_OUTPUT_MEM_WIDTH = 64 #128
RQPARAMS_OUTPUT_MEM_DEPTH =  (RQPARAMS_INPUT_MEM_WIDTH*RQPARAMS_INPUT_MEM_DEPTH)//RQPARAMS_OUTPUT_MEM_WIDTH

RTABLE_CMD_RESOLUTION     = 16
RTABLE_INPUT_MEM_WIDTH    = 256
RTABLE_INPUT_MEM_DEPTH    = 1024
RTABLE_OUTPUT_MEM_WIDTH   = 32
RTABLE_OUTPUT_MEM_DEPTH   =  (RTABLE_INPUT_MEM_WIDTH*RTABLE_INPUT_MEM_DEPTH)//RTABLE_OUTPUT_MEM_WIDTH
# In SequencerEngineOperationEntry, there are not enough bits in wloc/rq/rt_start_address to use all the addresses,  
# we need to reduce the address resolution to get access to the whole memory.
RTABLE_START_ADD_RESOLUTION      = 2

# for NLT we need only the address for pin-pong buffer
NLTABLE_HALF_LENGTH = 512

#CONST FOR WLOC CMD
WLOC_SINGLE_LONG_SIZE = 28
WLOC_SINGLE_SHORT_SIZE = 14
WLOC_PAIR_LONG_SIZE = 42
WLOC_PAIR_SHORT_SIZE = 28

# CONSTATNS FOR generate_grids_rq_cbc
USE_BEGIN_SYNC    = True               # use ore not the NOPS in the beginning of RQ and RT
RQ_BEGIN_SYNC_NOP = 8*USE_BEGIN_SYNC   # NOPS for RQ to be ready to ran
RT_BEGIN_SYNC_NOP = 15*USE_BEGIN_SYNC  # NOPS for RT to be ready to ran
HARDWARE_SYNC_CMD_COMPLATE = 4         # This NOPS have to be after last AMM write and the CMD_COMPLATE=0 (END_OF_Slit NOPs)