# Compiler related flags

DEBUG_CREATE_PER_NODE_DEBUG_FILES = True # Create debug files per each node (wloc csv files, rqloc, amt, rqparams etc.)
DEBUG_CREATE_HEX_FILES = True
DEBUG_SPREADSHEET_FORMAT_XLSX = False # sets file format of wloc cbc debug files xlsx/csv
DEBUG_GENERATE_DDR_HEX = False

# Below flags are used for testing single node workloads with different configs (ic_split, deepconv and folding_conv)
DEBUG_FORCE_IC_SPLIT = False # This can be False or the number of ic split to force (Supported values False,2,4 for AMD and False,2 for MCHP - this depends on number of grids), use this for small tests such as Conv14x8_k1
DEBUG_FORCE_DEEP_CONV = False # This forces all 1x1 convs to use deep conv ()
DEBUG_FORCE_FOLDCONV = False # This forces the initial conv to be a foldin one

DEBUG_AVOID_DDR_READ_WHILE_CONV  = False    # changed with Yaron and Itamar
DEBUG_AVOID_DDR_WRITE_WHILE_CONV = False    # change  with Yaron and Itamar
DEBUG_ADD_WAIT_FOR_WMT_READ = False # Add flags to condition read from DDR to finish of its wmt read. This condition is not needed if order of commands of all reads is keptd
DEBUG_OPTIMIZE_DMA_TRANSACTIONS = True
DEBUG_OPTIMIZE_FIRST_LAYER_DDR_ACCESS = True
DEBUG_AVOID_OFFLOAD_TENSOR_DDR_WRITE_WHILE_CONV = False
DEBUG_SKIP_CBC_GENERATION = False # This is for debug puposes since CBC generation is relativly small we might want to skip it to debug sequencer part which follows cbc generation

DEBUG_AUTO_Y_FOLDING = True # If true compiler attemts to automatically set y folding and unfolding
DEBUG_AUTO_BLOB_SPLITTING = True # If False, need to specify manually where blobs are splitted
DEBUG_PER_NODE_BLOB = [] # [15,17,19]#[222, 236,237,239, 240, 276,278,279,281] # If True, each blob will contain only 1 node
DEBUG_SKIP_BLOB_LIST = [] # range(0,5) # List of blob indexes to skip during compilation (for debug purposes)
LARGE_FIRST_BLOB = True # Allow first blob to be larger than the rest (for better performance)
MAX_FOLDED_K3_NODES_IN_BLOB = 1.5

DEBUG_SPLIT_LARGE_CONV = True # Split large convolutions (in + out channels > half amm size) into 2 convolutions

# Workarounds for the bug of k3 node at the start of 1-Tile blob
DEBUG_IDENTITY_CONV_BEFORE_K3_1T = True
DEBUG_NO_K3_LIMIT_FOR_1T = True

DEBUG_TRY_TO_FIX_AUTO_FOLD = True   # Fix fold errors encountered with empty forced fold/unfold/split lists
DEBUG_TRY_TO_FIX_AUTO_SPLIT = True  # Fix split errors encountered with empty forced fold/unfold/split lists
DEBUG_MINIMIZE_Y_FOLDING = False    # Do not do unnecessary Y folding, e.g. for Resize. Didn't get this to work.

#Yaron's additional flags
DEBUG_TFLITE_REQUANT_SATURATE = False


DEBUG_ADD_UNFOLDING_X = True
#DEBUG_ADD_UINT8_INT8_CONVERSION = True # ONLY for LIVE demo, or tests with real image test

# All auto
DEBUG_FORCE_Y_FOLDING_N_256 = []
DEBUG_FORCE_Y_UNFOLDING_N_256 =  []
DEBUG_FORCE_BLOB_LIST_N_256 = []

DEBUG_PAIRING_USED = True
SHORT_PAIRING_USED = False
CHECK_MAC_PAIRING_ON_FLY = True

DEBUG_X_SLICING = True

NODES_LIST = {
    'Start': [], #Layer to Start the sequencer program
    'End': [] #Layer to End the sequencer program
}

#YOLOv5 Nano 256
'''
DEBUG_FORCE_Y_FOLDING_N_256 = ['Concat_32'] 
DEBUG_FORCE_Y_UNFOLDING_N_256 =  ['Conv_139'] #'onnxConcat_209_requantnode'
DEBUG_FORCE_BLOB_LIST_N_256 = ['Conv_2','Add_10','Conv_16','Add_24','Conv_30','Conv_33','Conv_37','Add_43','Add_48','Conv_54','Conv_61','Conv_68','Conv_81','Conv_91','Conv_96','Conv_106','Conv_109','Conv_123','Conv_125','Conv_134','Conv_137']
'''

# All auto
DEBUG_FORCE_Y_FOLDING_M_256 = []
DEBUG_FORCE_Y_UNFOLDING_M_256 = []
DEBUG_FORCE_BLOB_LIST_M_256 = []

#YOLOv5 Medium 256
'''
DEBUG_FORCE_Y_FOLDING_M_256 = ['Conv_4','Conv_16','Conv_50','Conv_52','Conv_84','Conv_154','Conv_172'] 
DEBUG_FORCE_Y_UNFOLDING_M_256 =  ['Conv_122','Conv_130','onnxConcat_311_requantnode'] 
DEBUG_FORCE_BLOB_LIST_M_256 = ['Conv_2','Add_15','Conv_21','Add_34','Add_44','Conv_48','Conv_50','Add_83','Conv_87',\
                               'Add_102','Conv_116','Conv_120','Conv_135','Resize_137','Conv_147','Conv_154','Conv_190','onnxResize_283_requantnode','Conv_234','Conv_278']
'''

'''
#testing new criteria for auto folding and blobs
#Fold only when really beneficial or before stride=2, try to keep Y folding at 0
#Each concat starts a new BLOB (or node before it temporarily for folding)
#Need to add support for folding/unfolding on concat
DEBUG_FORCE_Y_FOLDING_M_256 = ['Conv_21','Conv_50','Conv_89','Conv_154','Conv_172'] 
DEBUG_FORCE_Y_UNFOLDING_M_256 =  ['onnxConcat_288_requantnode','onnxConcat_311_requantnode'] 
DEBUG_FORCE_BLOB_LIST_M_256 = ['Conv_2','Add_15','Conv_19','Add_29','Add_34','Conv_45','Conv_48','Add_58','Add_68','Add_78','Conv_84','Conv_87',\
                               'Add_102','MaxPool_112_dupk3','Conv_114','Resize_118','Conv_130','Conv_135','Resize_137',\
                                'Conv_147','Conv_190','Conv_154','onnxResize_283_requantnode','Conv_157','Conv_234','Conv_278']
#conv147 should actually be 149, but bypassing a bug where concat result is not written to DDR
'''

#YOLOv5 Medium 416
#DEBUG_FORCE_Y_FOLDING_M_416 = ['Conv_4','Conv_16','Conv_50','Conv_52','Conv_84','Conv_154','Conv_172'] 
#DEBUG_FORCE_Y_UNFOLDING_M_416 =  ['Conv_122','Conv_130','onnxConcat_311_requantnode'] 
#DEBUG_FORCE_BLOB_LIST_M_416 = ['Conv_2','Add_15','Conv_21','Add_34','Add_44','Conv_48','Conv_50','Add_83','Conv_87',\
#                               'Add_102','Conv_116','Conv_120','Conv_135','Resize_137','Conv_147','Conv_154','Conv_190','onnxResize_283_requantnode','Conv_234','Conv_278']
DEBUG_FORCE_Y_FOLDING_M_416 = ['Conv_21','Conv_50','Conv_89','Conv_154','Conv_172'] 
DEBUG_FORCE_Y_UNFOLDING_M_416 =  ['onnxConcat_288_requantnode','onnxConcat_311_requantnode'] 
DEBUG_FORCE_BLOB_LIST_M_416 = ['Conv_2','Add_15','Conv_19','Add_29','Add_34','Conv_45','Conv_48','Add_58','Add_68','Add_78','Conv_84','Conv_87',\
                               'Add_102','MaxPool_112_dupk3','Conv_114','Resize_118','Conv_124','Conv_130','Conv_135','Resize_137',\
                                'Conv_143','Conv_147','Conv_163','Conv_190','Conv_154','onnxResize_283_requantnode','Conv_157','Conv_234','Conv_278']
#YOLOv5 Small 256
DEBUG_FORCE_Y_FOLDING_S_256 = ['Concat_32','Conv_35']
DEBUG_FORCE_Y_UNFOLDING_S_256 = ['Conv_87','Conv_91','Conv_100','Conv_106']
DEBUG_FORCE_BLOB_LIST_S_256 = ['Conv_2','Add_10','Conv_18','Conv_30','Conv_33','Conv_54','Conv_68','Resize_83','Conv_85','Conv_96','Concat_99','Conv_104','Concat_108','Conv_139','Conv_183']


# All auto
DEBUG_FORCE_Y_FOLDING_N_512 = []
DEBUG_FORCE_Y_UNFOLDING_N_512 = []
DEBUG_FORCE_BLOB_LIST_N_512 = []

#YOLOv5 Nano 512
'''
DEBUG_FORCE_Y_FOLDING_N_512 = ['Concat_32','Concat_56']
DEBUG_FORCE_Y_UNFOLDING_N_512 = ['Conv_87','Conv_91','Conv_109']
DEBUG_FORCE_BLOB_LIST_N_512 = ['Conv_2','Add_10','Conv_18','Add_24','Conv_30','Add_43','Conv_54','Conv_68','MaxPool_76','MaxPool_77_dupk3','Resize_83','Conv_85','Conv_91','Conv_96','Concat_108','Conv_123']
'''

# All auto
DEBUG_FORCE_Y_FOLDING_N_416 = []
DEBUG_FORCE_Y_UNFOLDING_N_416 = []
DEBUG_FORCE_BLOB_LIST_N_416 = []

#YOLOv5 Nano 416
'''
DEBUG_FORCE_Y_FOLDING_N_416 = ['Concat_13','Conv_18','Conv_30','Conv_111']
DEBUG_FORCE_Y_UNFOLDING_N_416 = ['Conv_100','Conv_106','Conv_139']
DEBUG_FORCE_BLOB_LIST_N_416 = ['Conv_2','Add_10','Conv_16','Add_29','Conv_35','Add_53','Add_67','Concat_70','Resize_83','Conv_85','Conv_91','Concat_99','Conv_102','Conv_104','Conv_109','Conv_139','Conv_116','Concat_122','Conv_183']
'''

# Old compiler debug flags (which means they were used in development time, usually to enforce some state but we didnt use them in long time)
# These flags are still used in code
DEBUG_CREATE_ORDERING_CONV = False # If true, an ordering conv is inserted after each output so channels order is 0,1,2,3,4.....

DEBUG_REMOVE_ENGINE_COMMAND = False

DEBUG_PUSH_TABLES_READ_BEFORE_DDR_RW = True # Will read tables before reading missing tensors from DDR
DEBUG_MERGE_IDENTICAL_WLOC_READS = True # Identical wloc tables will be read in single ddr read command
DEBUG_ADD_AXI1_WAIT_FLAG  = False # reads from ddr to tables split to 2 axi busses so we need to have wait bit for each of them
DEBUG_CLIP_BIAS_TO_MAX_BIAS_BITS = True # This is to simulate what actually happens in FPGA so bias is limited to MAX_BIAS_BITS
SUPPORT_Z_AXIS_TILING = False # This effort was not fully implemented. Keep this False or remove relevant code
AMM_ALLOCATION_OPTIMIZE_LEFTOUT_SIZE = True # Newest AMM allocation algo.
OPTIMIZE_IDENTITY = False # Identity op does not need any CBC syntesis, but need to generate wloc/rqloc/rtloc tables als templates
OPTIMIZE_1x1_CONV = True # New algo for 1x1 convs

#Yaron
DEBUG_FORCE_FOLDED_INPUT = False # This will force input to be folded, by relative factors for X, Y
DEBUG_FORCE_FOLDED_FACTOR_X = 0
DEBUG_FORCE_FOLDED_FACTOR_Y = 1
DEBUG_FIX_CONCAT_OUTPUT_PROCESSING_ORDER = True # If true, concat processing order is fixed in case of inputs are folded

DEBUG_SIMULATE_CONCAT_REQUANT = False

DEBUG_DEEPCONV_SUPPORTED = True
DEBUG_PRINT_CBC = False
DEBUG_PRINT_CHANNELS_BALANCING = False
DEBUG_PRINT_AMM_ALLOCATION = False
DEBUG_PRINT_LAYER_EFFICIENCY = False

DEBUG_SAVE_IR_AFTER_CBC_CREATION = False # This will save preliminary IR after CBC creation. It will allow to skip cbc creating in next run to reduce compile time
DEBUG_LOAD_CBC_FROM_IR = False # See DEBUG_SAVE_IR_AFTER_CBC_CREATION flag
# End of old compiler flags

# Reports
DEBUG_GENERATE_PERFORMANCE_REPORT = True
DEBUG_GENERATE_AMM_ALLOCATION_REPORT = True

# Numeric Simulator related debug flags
SIMULATE_IC_SPLIT = True # IC splitting has effect on accuracy (+/- 1 lsb). Keep this true to have bit exact simulation
DEBUG_USE_REAL_DATASET = False # Use coco/imagenet as input to numeric simulator
DEBUG_UNFOLD_INPUTS_BEFORE_CONCAT = True # Must be true so that concat works

if DEBUG_USE_REAL_DATASET: # In case of real dataset we usually dont do deep debug so some debug flags are False (e.g. DEBUG_MAC_MULTIPLICATIONS,DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS etc.)
    DEBUG_MAC_MULTIPLICATIONS = False
    DEBUG_MAC_MULTIPLICATIONS_LAYERS = [] # List of layer names, If empyt, all layers will generate mac debug file
    DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC = [] # List of channels, if empty all channels will be in debug file
    #DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC = []
    DEBUG_MAX_OC_FOR_MULTIPLICATIONS_DEBUG = 2048
    DEBUG_MAX_IC_FOR_MULTIPLICATIONS_DEBUG = 1024

    DEBUG_NX_NUMERICS = True # Checks if there are any violations of different bus width (This will slow down simulation)
    DEBUG_ALLOW_14BIT_BIAS = False
    DEBUG_SIMULATOR_CLIP_ADD_BIAS_TO_INT12 = False
    DEBUG_ALLOW_ADDITIONAL_1BIT_TO_MACTORQ_BUS = False # This allows to simulate additional single bit in mac to rq bus

    DEBUG_TIMING = False # This was used to debug numeric simulation speed so it dumps various steps timing
    DEBUG_SIMULATE_FOLDING = True
    DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS = False # Active only if DEBUG_MAC_MULTIPLICATIONS is True
    NUMSIM_COMPARE_INTERMEDIATE_TENSORS = False # Compare also intermediate tensors
    GENERATE_SYNTHETIC_INPUTS = False # If True simulator is using synthetic input data
    GET_INPUTS_FROM_DIR = False # Read inputs of simulation form specific nxi files
    DEBUG_PRINT_IMAGENET_COMPARES = False
    DEBUG_INPUT_NXI_FILENAMES = ['imagenet_input0.nxi'] # If GET_INPUTS_FROM_DIR is False these files will be used as input
    #DEBUG_INPUT_NXI_FILENAMES = ['imagenet_input0.nxi','imagenet_input1.nxi','imagenet_input2.nxi','imagenet_input3.nxi']
    DEBUG_CREATE_SAMPLE_IMAGE_PICKLE_FILE = False # Dumps inputs to a pickle file

else:
    # Flags relevant for MAC debug
    DEBUG_MAC_MULTIPLICATIONS = False #Alex
    DEBUG_MAC_MULTIPLICATIONS_LAYERS =['CONV2D_0']# ['Conv_6']#['Add_10']
    SIM_OUTPUT_MATRIX_CBC_FORMAT = False #True
    #DEBUG_MAC_MULTIPLICATIONS_LAYERS = ['Conv_124'] # List of layer names, If empyt, all layers will generate mac debug file
    #DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC = [159] # List of channels, if empty all channels will be in debug file
    DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC = [0]
    DEBUG_MAX_OC_FOR_MULTIPLICATIONS_DEBUG = 2048
    DEBUG_MAX_IC_FOR_MULTIPLICATIONS_DEBUG = 2048

    DEBUG_NX_NUMERICS = True # Alex + Yaron
    DEBUG_ALLOW_14BIT_BIAS = False
    DEBUG_ALLOW_ADDITIONAL_1BIT_TO_MACTORQ_BUS = False # This allows to simulate additional single bit in mac to rq bus
    DEBUG_SIMULATOR_CLIP_ADD_BIAS_TO_INT12 = False

    DEBUG_TIMING = False
    DEBUG_SIMULATE_FOLDING = True
    DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS = True # Active only if DEBUG_MAC_MULTIPLICATIONS is True
    NUMSIM_COMPARE_INTERMEDIATE_TENSORS = False # Compare also intermediate tensors
    GENERATE_SYNTHETIC_INPUTS = True # If True simulator is using synthetic input data
    GET_INPUTS_FROM_DIR = True
    DEBUG_PRINT_IMAGENET_COMPARES = False
    DEBUG_INPUT_NXI_FILENAMES = ['imagenet_input0.nxi'] # If GET_INPUTS_FROM_DIR is False these files will be used as input
    #DEBUG_INPUT_NXI_FILENAMES = ['imagenet_input0.nxi','imagenet_input1.nxi','imagenet_input2.nxi','imagenet_input3.nxi']
    DEBUG_CREATE_SAMPLE_IMAGE_PICKLE_FILE = False


# Samples creator util (utils\fpga\create_sample_inputs_and_outputs.py) related debug flags
DEBUG_SAMPLE_CREATOR_CREATE_MAC_DEBUG_FILES = False # This is only relevant for the sample creation utility

# Model creator flags
SYNTHETIC_CALIBRATION_SET = True # Use synthetic or dataset inputs for model calibration process during model quantization
DEBUG_ONNX_FORMAT_QDQ = True 
IDENTITY_INPUT_CONV = False # This will force input conv to be an identity conv. Should allow feeding output of test X to test X+1

DEBUG_2_TILE_RW_LINES = True
