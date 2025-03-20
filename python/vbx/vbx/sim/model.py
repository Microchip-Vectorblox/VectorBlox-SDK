from . import c as c_shim
import numpy as np
from enum import  IntEnum
import time
import os
import json


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
def main(model_file,expected_checksum,debug=False,verbose=True):


    model_bytes = open(model_file, 'rb').read()
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

    if debug:
        data = {'inputs': [], 'outputs': [], 'test_outputs': []}
        for i,(arr,shape,zero,scale,dtype) in enumerate(zip(m.test_input,m.input_shape,m.input_zeropoint,m.input_scale_factor,m.input_dtypes)):
            np.save(os.path.join(os.path.dirname(model_file),'vnnx.input.{}.npy'.format(i)), arr.reshape(shape))
            data['inputs'].append({'data': arr.tolist(), 'shape': shape, 'zero': zero, 'scale': scale, 'dtype': dtype.name.upper()})
        for o,(arr,test_arr,shape,zero,scale,dtype) in enumerate(zip(odata,m.test_output,m.output_shape,m.output_zeropoint,m.output_scale_factor,m.output_dtypes)):
            np.save(os.path.join(os.path.dirname(model_file),'vnnx.output.{}.npy'.format(o)), arr.reshape(shape))
            data['outputs'].append({'data': arr.tolist(), 'shape': shape, 'zero': zero, 'scale': scale, 'dtype': dtype.name.upper()})
            np.save(os.path.join(os.path.dirname(model_file),'test.output.{}.npy'.format(o)), test_arr.reshape(shape))
            data['test_outputs'].append({'data': test_arr.tolist(), 'shape': shape, 'zero': zero, 'scale': scale, 'dtype': dtype.name.upper()})
            heat = arr - test_arr
            while len(heat.shape) < 3:
                heat = np.expand_dims(heat, axis=0)
            np.save("heatmap.{}.npy".format(o), heat)

            print('\nTotal absdiff between VNNX and test outputs', np.sum(np.abs(heat)))
            for c,channel in enumerate(np.squeeze(heat, axis=0)):
                absdiff = np.sum(np.abs(channel))
                if absdiff != 0:
                    print('\tChannel', c, absdiff)

        with open('io.json', 'w') as f:
            json.dump(data, f)

    if expected_checksum is not None:
        if checksum != expected_checksum:
            return checksum
    return 0
