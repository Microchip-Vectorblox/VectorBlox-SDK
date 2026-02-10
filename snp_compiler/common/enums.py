from enum import Enum
 
class GridConfig(Enum):
    H56xW28 = 1
    H28xW28 = 2
    H14xW14 = 3
    H7xW7 = 4
    H14xW8 = 5
    H14xW16 = 6
    H14xW32 = 7
    NONE = 8

class RqDSPCommand(Enum):
    NOP = 1
    MULTIPLY = 2
    MULTIPLY_AND_ADD_PREVIOUS = 3
    
class RqOP(Enum):
    MUX_IN = 1
    NOP = 2
    END = 3

class DDREntryType(Enum):
    INPUT_TENSOR = 1
    OUTPUT_TENSOR = 2
    INTERMEDIATE_TENSOR = 3
    WLOC = 4
    RQPARAMS = 5
    RESULTTABLE = 6
    NONLINEARFUNCTION = 7
    PROGRAM = 8
    REGISTER_PAGE = 9

class SequencerOPCode(Enum):
    RD_FROM_DDR = 0
    WR_TO_DDR = 1
    DMR_AMM = 2
    DMW_AMM = 3
    DMR_TABLES = 4
    ENGINE_OPERATION = 5
    CUSTOM_OPERATION = 6

    def __str__(self):
        return(str(self.name))
    
    def __repr__(self):
        if self.value == 0:
            return ('old_Read,')
        elif self.value == 1:
            return ('old_Write,')
        elif self.value == 2:
            return ('DMR_AMM,')
        elif self.value == 3:
            return ('DMW_AMM,')
        elif self.value == 4:
            return ('DMR_TABLES,')
        elif self.value == 5:
            return ('Engine,')
        else:
            return ('Custom,')

class EngineOpCode(Enum):
    CONV = 0
    MAXPOOL = 1
#    DEEPCONV = 2
    DEEPCONV = 0
    
class PadMode(Enum):
    GRIDH14XW32 = 0
    GRIDH14XW16 = 1
    GRIDH14XW8 = 2

class ScalingMode(Enum):
    RQDIRECT = 0
    FOLDING2_1 = 1
    RESIZE1_2 = 2
    UNFOLDING = 3

class CustomeOpCode(Enum):
    NOP = 0

class DDRCommandDstSrcType(Enum):
    PROGRAM = 0
    AMM = 1
    WLOC = 2
    RESULTTABLE  = 3
    RQPARAMS = 4
    NONLINEARFUNCTION = 5

    #AMM_WRT_MASK = 5

    def __str__(self):
        return('DstSrcType:'+str(self.name))
    
    def __repr__(self):
        return(str(self.name)+',')
        if self.value == AMM_WRT_MASK:
            return('WR Mask,')
        else:
            return(str(self.name))+','

class DDRPriority(Enum):
    LOW = 0
    HIGH = 1

class AMMType(Enum):
    BRAM_BASED_AMM = 0
    URAM_BASED_AMM = 1

class DDRReadOffsetType(Enum):
    INPUT_TENSOR = 0
    MODEL_MEM = 1
    MXP_BASE = 2
    def __str__(self):
        return('DDROffset:'+str(self.name))
    def __repr__(self):
        return(str(self.name)+',')

class DDRWriteOffsetType(Enum):
    OUTPUT_TENSOR = 0
    MODEL_MEM = 1
    MXP_BASE = 2
    def __str__(self):
        return('DDROffset:'+str(self.name))
    def __repr__(self):
        return(str(self.name)+',')
    
class DebugFilesFormat(Enum):
    XLSX = 0
    CSV = 1

