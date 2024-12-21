from . import c as c_shim
import numpy as np
from enum import  IntEnum
import time
import os


def Fletcher32(data):
    data_bytes = data.tobytes()
    if len(data_bytes) % 2:
        data_bytes = data_bytes[:-1]
    data = np.frombuffer(data_bytes,dtype=np.uint16)
    datalen = len(data)
    c0 = 0
    c1 = 0
    leftover = datalen % 360
    for i in range(0,datalen,360):
        for j in range(360):
            if i+j < datalen:
                c0 += data[i+j]
                c1 += c0
        c0 %= 65535
        c1 %= 65535
    return (c1 << 16) | c0


class Model:
    vbx_cnn = c_shim.vbx_cnn_init()
    cnn_pid = os.getpid()
    class SizeConf(IntEnum):
        V250 = 0
        V500 = 1
        V1000 = 2
    def __init__(self,model_bytes):
        if os.getpid() != Model.cnn_pid:
            Model.vbx_cnn = c_shim.vbx_cnn_init()
            Model.cnn_pid = os.getpid()
        if type(model_bytes) is not bytes:
            raise TypeError("Model parameter must by bytes")

        if c_shim.model_check_sanity(model_bytes) != 0:
            raise ValueError("Does not appear to be a valid model")
        allocate_bytes = c_shim.model_get_allocate_bytes(model_bytes)
        if allocate_bytes > len(model_bytes): #TODO should be >
            model_bytes = model_bytes + bytes(allocate_bytes - len(model_bytes))


        self.model_bytes = model_bytes

        self.model_size_conf = \
            Model.SizeConf(c_shim.model_get_size_conf(model_bytes))

        self.num_outputs = c_shim.model_get_num_outputs(model_bytes)
        self.num_inputs = c_shim.model_get_num_inputs(model_bytes)

        self.output_lengths = [
            c_shim.model_get_output_length(self.model_bytes,i)
            for i in range(self.num_outputs)]
        self.input_lengths = [
            c_shim.model_get_input_length(self.model_bytes,i)
            for i in range(self.num_inputs)]
        self.output_dims = [
            c_shim.model_get_output_dims(self.model_bytes,i)
            for i in range(self.num_outputs)]
        self.input_dims = [
            c_shim.model_get_input_dims(self.model_bytes,i)
            for i in range(self.num_inputs)]
        self.output_shape = [
            c_shim.model_get_output_shape(self.model_bytes,i)
            for i in range(self.num_outputs)]
        self.input_shape = [
            c_shim.model_get_input_shape(self.model_bytes,i)
            for i in range(self.num_inputs)]
        self.output_dtypes = [
            c_shim.model_get_output_datatype(self.model_bytes,i)
            for i in range(self.num_outputs)]
        self.input_scale_factor = [
            c_shim.model_get_input_scale_value(self.model_bytes,i)
            for i in range(self.num_inputs)]
        self.output_scale_factor = [
            c_shim.model_get_output_scale_value(self.model_bytes,i)
            for i in range(self.num_outputs)]
        
        self.output_zeropoint = [
            c_shim.model_get_output_zeropoint(self.model_bytes,i)
            for i in range(self.num_outputs)]

        self.input_zeropoint = [
            c_shim.model_get_input_zeropoint(self.model_bytes,i)
            for i in range(self.num_inputs)]

        self.input_dtypes = [
            c_shim.model_get_input_datatype(self.model_bytes,i)
            for i in range(self.num_inputs)]

        self.test_input = [
            c_shim.model_get_test_input(self.model_bytes,i)
            for i in range(self.num_inputs)]

        self.test_output = [
            c_shim.model_get_test_output(self.model_bytes,i)
            for i in range(self.num_outputs)]

    def run(self,inputs):
        outputs = [ np.zeros(l,dtype=t) for l,t in
                    zip(self.output_lengths,self.output_dtypes)]
        io_buffers = list(inputs+outputs)


        c_shim.vbx_cnn_model_start(Model.vbx_cnn,self.model_bytes,io_buffers)
        while(c_shim.vbx_cnn_model_poll(Model.vbx_cnn) > 0):
            time.sleep(0.1)
        self.stats = c_shim.vbxsim_get_stats()
        return outputs
    def get_bandwidth_per_run(self):
        if not getattr(self,'stats'):
            self.run(self.test_input)
        return self.stats.dma_bytes
    def get_estimated_runtime(self,Hz=None):
        "Returns runtime in seconds if Hz is specified, otherwise cycles"
        if not getattr(self,'stats'):
            self.run(self.test_input)
        instr_cycles = self.stats.instruction_cycles
        instr_cycles = sum([ sum(c) for c in instr_cycles])
        log_lanes = (2,3,3,4,4)[self.model_size_conf]

        dma_cycles = self.stats.dma_cycles[log_lanes]
        instr_runtime = 0
        cycles = max(dma_cycles,instr_cycles)
        if Hz is not None:
            return cycles/Hz
        else :
            return cycles
def main(model_bytes,expected_checksum,verbose=True):


    m = Model(model_bytes)
    # c.vbxsim_reset_stats()
    odata = m.run(m.test_input)
    checksum = Fletcher32(odata[0])
    for od in odata[1:]:
        checksum = checksum ^ Fletcher32(od)
    if verbose:
        print("DMA_BYTES = {}".format(m.get_bandwidth_per_run()))
        print("INSTR_CYCLES = {}".format(m.get_estimated_runtime()))
        print("CHECKSUM = {:08x}".format(checksum))

    if expected_checksum is not None:
        if checksum != expected_checksum:
            return checksum
    return 0
