################# DEBUG FLAGS to find the compiler problems in CBC, Sequencer etc.
DEBUG_OPTIMIZE_DMA_TRANSACTIONS       = True
DEBUG_OPTIMIZE_FIRST_LAYER_DDR_ACCESS = True
DEBUG_PAIRING_USED                    = True
DEBUG_CHECK_PAIRING_AFTER_ALGO        = True # This is to check if the pairing is correct after the algo, (for debug purposes, makes the compiler slower)

DEBUG_PER_NODE_BLOB                   = [] # [15,17,19]#[222, 236,237,239, 240, 276,278,279,281] # If True, each blob will contain only 1 node
DEBUG_SKIP_BLOB_LIST                  = [] # range(0,5) # List of blob indexes to skip during compilation (for debug purposes)
DEBUG_LARGE_FIRST_BLOB                = True  # Allow first blob to be larger than the rest (for better performance)
MAX_FOLDED_K3_NODES_IN_BLOB           = 1.5 # Max number of folded K3 nodes in blob (to avoid too much ourlap blobs)

################## DEBUG FLAGS for reserch and optimization  ###############################
DEBUG_AVOID_DDR_READ_WHILE_CONV      = False # if true no DMA READ  in paralel to conv calulations the next part
DEBUG_AVOID_DDR_WRITE_WHILE_CONV     = False # if true no DMA WRITE in paralel to conv calulations the next part

DEBUG_FORCE_FOLDCONV                 = False # This forces the initial conv to be a foldin one

DEBUG_REMOVE_ENGINE_COMMAND          = False

DEBUG_PUSH_TABLES_READ_BEFORE_DDR_RW = True  # Will read tables before reading missing tensors from DDR
DEBUG_CLIP_BIAS_TO_MAX_BIAS_BITS     = True  # This is to simulate what actually happens in FPGA so bias is limited to MAX_BIAS_BITS
AMM_ALLOCATION_OPTIMIZE_LEFTOUT_SIZE = True  # Newest AMM allocation algo
DEBUG_SIMULATE_CONCAT_REQUANT        = False


NODES_LIST = {
    'Start': [], #Layer to Start the sequencer program
    'End'  : []  #Layer to End the sequencer program
}

################## DEBUG FLAGS for reports and outputs in command line ###############################
DEBUG_CREATE_HEX_FILES               = True
DEBUG_GENERATE_DDR_HEX               = False
DEBUG_PRINT_CBC                      = False  
DEBUG_PRINT_AMM_ALLOCATION           = False
DEBUG_GENERATE_PERFORMANCE_REPORT    = True
DEBUG_GENERATE_AMM_ALLOCATION_REPORT = True

################## Stefan work ###############################
DEBUG_2_TILE_RW_LINES                     = True 

# Workarounds for the bug of k3 node at the start of 1-Tile blob
DEBUG_IDENTITY_CONV_BEFORE_K3_1T          = True
DEBUG_K3_1T_BOUNDARY_EXTRA_LINE           = True
DEBUG_NO_K3_LIMIT_FOR_1T                  = True

DEBUG_TRY_TO_FIX_AUTO_SPLIT               = True  # Fix split errors encountered with empty forced fold/unfold/split lists
DEBUG_MINIMIZE_Y_FOLDING                  = False    # Do not do unnecessary Y folding, e.g. for Resize. Didn't get this to work.

################## DEBUG FLAGS for NumSim (probably have to be put in othere place) ###############################

# Flags relevant for MAC debug
DEBUG_MAC_MULTIPLICATIONS                  = False #Alex
DEBUG_MAC_MULTIPLICATIONS_LAYERS           = ['CONV2D_0']# ['Conv_6']#['Add_10'] # List of layer names, If empyt, all layers will generate mac debug file
DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC      = [0] # List of channels, if empty all channels will be in debug file
CHECK_MAC_PAIRING_ON_FLY                   = True # This is to check if the pairing is correct during the simulation

DEBUG_TFLITE_REQUANT_SATURATE              = False #Yaron's additional flags, this is for simulator only to check tflite requantization if it saturated
SIMULATE_IC_SPLIT                          = True # IC splitting has effect on accuracy (+/- 1 lsb). Keep this true to have bit exact simulation
DEBUG_UNFOLD_INPUTS_BEFORE_CONCAT          = True # Must be true so that concat works

SIM_OUTPUT_MATRIX_CBC_FORMAT               = False #True

DEBUG_MAX_OC_FOR_MULTIPLICATIONS_DEBUG     = 2048
DEBUG_MAX_IC_FOR_MULTIPLICATIONS_DEBUG     = 2048

DEBUG_NX_NUMERICS                          = True # Alex + Yaron
DEBUG_ALLOW_14BIT_BIAS                     = False
DEBUG_SIMULATOR_CLIP_ADD_BIAS_TO_INT12     = False
DEBUG_TIMING                               = False
DEBUG_SIMULATE_FOLDING                     = True
DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS  = True # Active only if DEBUG_MAC_MULTIPLICATIONS is True



# those flags will be delited after CBC 1x1 and 3x3 fulion algo will be stable
OPTIMIZE_1x1_CONV  = True # New algo for 1x1 convs
SHORT_PAIRING_USED = True
CBC_OBJECT_GENERATION_METHOD = False

################## DEBUG FLAGS for incremental memory creation (Alex 2026-08-03) ###############################
DEBUG_OPTIMIZED_MEMORY_CREATION = True  # Use incremental memory creation for WLOC split check (faster, O(N) vs O(N²))