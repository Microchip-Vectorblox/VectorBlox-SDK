import ctypes
import enum
import numpy as np
from ctypes import c_void_p,c_char_p,c_int,c_size_t,c_uint64
from ctypes import c_int16,c_int8,c_uint8,c_int32,c_float
import os.path
#_so = ctypes.cdll.LoadLibrary('libvbx_cnn_sim.so')
_so = np.ctypeslib.load_library('libvbx_cnn_sim.so', os.path.dirname(__file__))
#vbx_cnn_t vbx_cnn_init(volatile void* ctrl_reg_addr,void* firmware_blob);

_so.vbx_cnn_init.arg_types = [c_void_p,c_void_p]
_so.vbx_cnn_init.restype = c_void_p

def ctype_from_enum(e):
    if e == 0:
        return c_uint8
    if e == 1:
        return c_int8
    if e == 2:
        return c_int16
    if e == 3:
        return c_int32
    raise NotImplementedError("Unknown type enum ({})".format(e))

class size_conf(enum.Enum):
	SUPER_SMALL= 0
	SMALL= 1
	MEDIUM= 2
	LARGE = 3
	EXTRA_LARGE =  4

class vbx_cnn_err(enum.Enum):
    INVALID_FIRMWARE_ADDRESS        = 1
    FIRMWARE_ADDRESS_NOT_READY      = 2
    START_NOT_CLEAR                 = 3
    OUTPUT_VALID_NOT_SET            = 4
    FIRMWARE_BLOB_VERSION_MISMATCH  = 5
    INVALID_NETWORK_ADDRESS         = 6
    MODEL_BLOB_INVALID              = 7
    MODEL_BLOB_VERSION_MISMATCH     = 8
    MODEL_BLOB_SIZE_CONFIGURATION_MISMATCH      = 9
    FIRMWARE_BLOB_STALE            = 10

class vbx_cnn:
    def __init__(self,p):
        self.p = p
def vbx_cnn_init():
    return vbx_cnn(_so.vbx_cnn_init(None,None))

#vbx_cnn_err_e vbx_cnn_get_error_val(vbx_cnn_t* vbx_cnn);
_so.vbx_cnn_get_error_val.arg_types = [c_void_p]
_so.vbx_cnn_get_error_val.restype = c_int
def vbx_cnn_get_error_val(vbx_cnn):
    return _so.vbx_cnn_get_error_val(vbx_cnn.p).value

#int vbx_cnn_model_start(vbx_cnn_t* vbx_cnn,model_t* model,vbx_cnn_io_ptr_t io_buffers[]);
#_so.vbx_cnn_model_start.arg_types=[c_void_p,c_char_p,c_void_p)

def vbx_cnn_model_start(vbx_cnn,model,io_buffers):
    #io buffers are numpy array of int8 or int16
    num_inputs = model_get_num_inputs(model)
    num_outputs = model_get_num_outputs(model)
    for i in range(num_inputs):
        expected_type = model_get_input_datatype(model,i)
        if expected_type != io_buffers[i].dtype:
            raise TypeError("Input {} expected {} got {}".
                            format(i,expected_type,io_buffers[i].dtype))
    for o in range(num_outputs):
        i = num_inputs + o
        expected_type = model_get_output_datatype(model,o)
        if expected_type != io_buffers[i].dtype:
            raise TypeError("Output {} expected {} got {}".
                            format(o,expected_type,io_buffers[i].dtype))

    io_buffer_pointers = [ctypes.cast(np.ctypeslib.as_ctypes(b),c_void_p) for b in io_buffers]
    array_type = c_void_p*len(io_buffers)

    iob_array = array_type(*io_buffer_pointers)
    return _so.vbx_cnn_model_start(vbx_cnn.p,model,iob_array)

#vbx_cnn_state_e vbx_cnn_get_state(vbx_cnn_t* vbx_cnn);
def vbx_cnn_get_state(vbx_cnn):
    return _so.vbx_cnn_get_state(vbx_cnn.p)
#int vbx_cnn_model_poll(vbx_cnn_t* vbx_cnn);
def vbx_cnn_model_poll(vbx_cnn):
    ret = _so.vbx_cnn_model_poll(vbx_cnn.p)
    if ret < 0:
        err = vbx_cnn_err(vbx_cnn_get_err_val)
        raise RuntimeError(err.name)
    return ret
#
#
#size_t model_get_output_length(const model_t* model,int output_index);
_so.model_get_output_length.restype = c_size_t
def model_get_output_length(model,output_index):
    return _so.model_get_output_length(c_char_p(model),c_int(output_index))
#
#size_t model_get_input_length(const model_t* model,int input_index);
_so.model_get_input_length.restype = c_size_t
def model_get_input_length(model,input_index):
    return _so.model_get_input_length(c_char_p(model),c_int(input_index))
#size_t* model_get_output_dims(const model_t* model,int output_index);
_so.model_get_output_dims.restype = ctypes.POINTER(c_int)
def model_get_output_dims(model,output_index):
    ret = _so.model_get_output_dims(c_char_p(model),c_int(output_index))
    return [ret[_] for _ in range(3)]
#
#size_t* model_get_input_dims(const model_t* model,int input_index);
_so.model_get_input_dims.restype = ctypes.POINTER(c_int)
def model_get_input_dims(model,input_index):
    ret = _so.model_get_input_dims(c_char_p(model),c_int(input_index))
    return [ret[_] for _ in range(3)]
#
#vbx_cnn_calc_type_e model_get_output_datatype(const model_t* model,int output_index);
def model_get_output_datatype(model,output_index):
    ret = _so.model_get_output_datatype(c_char_p(model),c_int(output_index))
    return np.dtype(ctype_from_enum(ret))
#vbx_cnn_calc_type_e model_get_input_datatype(const model_t* model,int input_index);
def model_get_input_datatype(model,input_index):
    ret = _so.model_get_input_datatype(c_char_p(model),c_int(input_index))
    return np.dtype(ctype_from_enum(ret))


#size_t model_get_num_inputs(const model_t* model);
_so.model_get_num_inputs.restype = c_size_t
def model_get_num_inputs(model):
    ret = _so.model_get_num_inputs(c_char_p(model))
    return ret
_so.model_get_data_bytes.restype = c_size_t
def model_get_data_bytes(model):
    ret = _so.model_get_data_bytes(c_char_p(model))
    return ret
_so.model_get_allocate_bytes.restype = c_size_t
def model_get_allocate_bytes(model):
    ret = _so.model_get_allocate_bytes(c_char_p(model))
    return ret

#
#size_t model_get_num_outputs(const model_t* model);
_so.model_get_num_outputs.restype = c_size_t
def model_get_num_outputs(model):
    ret = _so.model_get_num_outputs(c_char_p(model))
    return ret


#vbx_cnn_size_conf_e model_get_size_conf(const model_t* model);
def model_get_size_conf(model):
    size = _so.model_get_size_conf(c_char_p(model))
    return size
#void* model_get_test_input(const model_t* model,int input_index);
_so.model_get_test_input.restype= c_void_p
def model_get_test_input(model,input_index):
    ti = _so.model_get_test_input(c_char_p(model),c_int(input_index))
    datatype = model_get_input_datatype(model,input_index)
    ctypes_datatype = np.ctypeslib.as_ctypes_type(datatype)
    ptr = ctypes.cast(ti,ctypes.POINTER(ctypes_datatype))
    length = model_get_input_length(model,input_index)
    return np.ctypeslib.as_array(ptr,shape=(length,))

_so.model_get_output_scale_value.restype = c_float
def model_get_output_scale_value(model,output_index):
    return _so.model_get_output_scale_value(model,output_index)
def model_check_sanity(model):
    return _so.model_check_sanity(model)

class simulator_stats(ctypes.Structure):
    MAX_INSTR_VAL = 41;
    NUM_OP_SIZE = 3;
    LOG2_MAX_DMA_LANES =9
    _fields_ = [
        ("instruction_cycles",ctypes.c_uint64*NUM_OP_SIZE*(MAX_INSTR_VAL+1)),
        ("instruction_count",ctypes.c_uint64*(MAX_INSTR_VAL+1)),
        ("accumulate_cycles",ctypes.c_uint64*NUM_OP_SIZE),
        ("set_vl",ctypes.c_uint64),
        ("set_2D",ctypes.c_uint64),
        ("set_3D",ctypes.c_uint64),
        ("dma_bytes",ctypes.c_uint64),
        ("dma_calls",ctypes.c_uint64),
        ("dma_cycles",ctypes.c_uint64*LOG2_MAX_DMA_LANES)]
_so.vbxsim_get_instructions.restype = c_uint64
def vbxsim_get_instructions():
    return _so.vbxsim_get_instructions()
def vbxsim_reset_stats():

    _so.vbxsim_reset_stats()
_so.vbxsim_get_stats.restype = simulator_stats
def vbxsim_get_stats():
    return _so.vbxsim_get_stats()


#float model_get_output_scale_value(const model_t* model,int index);
