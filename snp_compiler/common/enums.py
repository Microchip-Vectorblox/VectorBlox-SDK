from enum import Enum

    

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

class AMMType(Enum):
    BRAM_BASED_AMM = 0
    URAM_BASED_AMM = 1

class DDRReadOffsetType(Enum):
    INPUT_TENSOR = 0
    MODEL_MEM = 2
    MXP_BASE = 3
    def __str__(self):
        return('DDROffset:'+str(self.name))
    def __repr__(self):
        return(str(self.name)+',')

class DDRWriteOffsetType(Enum):
    OUTPUT_TENSOR = 0
    MODEL_MEM = 2
    MXP_BASE = 3
    def __str__(self):
        return('DDROffset:'+str(self.name))
    def __repr__(self):
        return(str(self.name)+',')
    
class DebugFilesFormat(Enum):
    XLSX = 0
    CSV = 1

