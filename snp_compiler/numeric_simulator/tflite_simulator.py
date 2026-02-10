import sys
sys.path.append('.')
import common.internal_representation as internal_representation
from common.hw_config import GRID_CONFIGS, AMM_HEIGHT, TFLITE_REQUANT
import math
import numpy as np
import numeric_simulator.neuronix_ops as neuronix_ops
import pickle
from numeric_simulator.defs import qTensor
import os
from common.debug_flags import DEBUG_CREATE_ORDERING_CONV, DEBUG_SIMULATE_FOLDING, DEBUG_SIMULATE_CONCAT_REQUANT, DEBUG_OPTIMIZE_DMA_TRANSACTIONS, NODES_LIST
from common.ddr_ir import create_tsnp_tensor_byte_array
import compiler.folding_algo as folding_algo
from typing import Tuple
from tqdm import tqdm
import json
import argparse
import time
import struct
import mmap
import posix_ipc
from numpy.random import RandomState
import re
import tempfile

DEBUG_PRINT = False
sync_dir = None

sync_name = "/sync_shared_memory"
sync_semaphore = "/sync_semaphore"

try:
    # Attempt to create the semaphore, raise exception if it exists
    semaphore = posix_ipc.Semaphore(sync_semaphore, posix_ipc.O_CREX, initial_value=1)
except posix_ipc.ExistentialError:
    # semaphore exists, get a handle to it
    semaphore = posix_ipc.Semaphore(sync_semaphore)
    
def read_array_from_csv(input_file: str) -> np.array:
    with open(input_file, 'r') as file:
        content = file.read().strip()
    if content[-1] == ',':
        content = content[:-1]
    return np.array(content.split(','), dtype=int)

# Read inputs from .csv file or sent from MXP
# If from .csv file, it is the same format as VNNX unit tests, or generated from
# the flattened tflite/vnnx input in yoloInfer.py
def get_model_inputs(ir: internal_representation.IR, read_from_mxp, input_file, model_bin, pad_to_gridsize = False):
    graph_inputs = ir.inputs
    inputs = {}
    padded_inputs = {}
    for graph_input in graph_inputs:

        input_shape = ir.tensors[graph_input].shape
        input_scale = ir.tensors[graph_input].scale
        zero_point = ir.tensors[graph_input].zero_point

        if read_from_mxp:
            input = wait_for_sync(return_tensor = True, tensor_shape=input_shape).reshape(input_shape)
        else:
            if input_file != None:
                input = read_array_from_csv(input_file).reshape(input_shape)
                if not TFLITE_REQUANT:
                    # Add 128 because TFLite is int8, ONNX is uint8
                    input = input + 128
            else:
                if os.path.exists(model_bin):
                    with open(model_bin, 'rb') as mf:
                        header_size = struct.unpack('i', mf.read(4))[0]
                        input_address_offset = 24
                        mf.seek(input_address_offset, 0)
                        input_address = header_size + struct.unpack('i', mf.read(4))[0]
                        input_size_offset = 20
                        mf.seek(input_size_offset, 0)
                        input_size = struct.unpack('i', mf.read(4))[0]
                        mf.seek(input_address, 0)
                        input_data = mf.read(input_size)
                        input = np.frombuffer(input_data, dtype=np.int8)
                        input = input.reshape(input_shape)
                    
                    if (input.shape[3] % 16 != 0):
                        with open(model_bin, 'r+b') as mf:
                            header_size = struct.unpack('i', mf.read(4))[0]         
                            padding_width = math.ceil(input.shape[3]/16)*16 - input.shape[3]
                            input = np.pad(input, ((0, 0), (0, 0), (0, 0), (0, padding_width)), mode="constant", constant_values=0)
                            input_size_offset = 20
                            input_size = int(np.prod(input.shape))
                            mf.seek(input_size_offset, 0)
                            mf.write(input_size.to_bytes(4, 'little'))    
                            input_bytearray = bytearray(input.astype(np.int8))
                            input_address_offset = 24
                            mf.seek(input_address_offset, 0)
                            input_address = header_size + struct.unpack('i', mf.read(4))[0]
                            mf.seek(input_address, 0)
                            mf.write(input_bytearray)
                            print(f"Successfully modified {len(input_bytearray)} bytes at {hex(input_address)} offset in {model_bin}")
                else:
                    prng = RandomState(5095)
                    if not TFLITE_REQUANT:
                        float_input = prng.uniform(size=input_shape,low=0,high=1).astype(np.float32)
                        input = np.round(float_input/input_scale)+zero_point
                        input = np.clip(input,0,255)
                    else:
                        input = prng.randint(size=input_shape,low=-128,high=127)
        input = input.astype(np.float32)
        input_tensor = qTensor(input,input_scale,zero_point)
        inputs[graph_input] = input_tensor
        # Pad to nearest grid size
        if pad_to_gridsize and len(input_shape)==4:
            # The below handles case where we need to pad x axis original tensor to fit grid width once folded.
            # This is not needed in y axis?
            current_input_width=input_shape[3]
            current_input_height = input_shape[2]
            for minimal_grid_width,gridmode in GRID_CONFIGS.items():
                if current_input_width<=minimal_grid_width:
                    break
            minimal_grid_width_mult_factor = math.ceil(max(0,math.log2(current_input_width/minimal_grid_width)))
            padded_width = int(math.pow(2,minimal_grid_width_mult_factor)*minimal_grid_width)
            x_padding = padded_width-current_input_width
            minimal_grid_height = AMM_HEIGHT
            minimal_grid_height_mult_factor = math.ceil(max(0,math.log2(current_input_height/minimal_grid_height)))
            padded_height = int(math.pow(2,minimal_grid_height_mult_factor)*minimal_grid_height)
            y_padding = padded_height-current_input_height
            padded_input=np.pad(input,[(0,0),(0,0),(0,y_padding),(0,x_padding)], mode='constant', constant_values=0)
            padded_input_tensor = qTensor(padded_input,input_scale,zero_point)
            padded_inputs[graph_input] = padded_input_tensor
        else:
            padded_inputs = inputs
    return padded_inputs, input

# Compare to an expected output tensor
def compare_to_expected(nx_output, expected_output, output_num, zero_point, output_dir):
    # Comparison threshold (for int8 quantized outputs)
    THRESHOLD = 1

    if not TFLITE_REQUANT:
        # Add 128 because TFLite is int8, ONNX is uint8
        expected_output = expected_output + 128
    # this is Alex patch to have the same shape for my test case    
    if nx_output.shape[2] > expected_output.shape[2] or nx_output.shape[3] > expected_output.shape[3]:
        # padding_height = nx_output.shape[2] - expected_output.shape[2]
        # padding_width = nx_output.shape[3] - expected_output.shape[3]
        # expected_output = np.pad(expected_output, ((0, 0), (0, 0), (0, padding_height), (0, padding_width)), mode="constant", constant_values=zero_point)
        nx_output = nx_output[:,:,:expected_output.shape[2],:expected_output.shape[3]]

    # Find where the arrays differ
    mismatches = np.where(abs(nx_output - expected_output) > THRESHOLD)
    '''
    if len(mismatches[0]) > 0:
        print("Mismatched indices:")
        for mismatch in zip(*mismatches):
            diff = nx_output[mismatch] - expected_output[mismatch]
            print(f"{mismatch}\tNX: {nx_output[mismatch]}, TFLite: {expected_output[mismatch]}, Diff: {diff}")
        print("\n")
    '''

    # Find where the arrays match
    matches = np.where(abs(nx_output - expected_output) <= THRESHOLD)
    '''
    print("Matched indices:")
    for match in zip(*matches):
        print(f"{match}\tValue: {nx_output[match]}")
    '''
    
    # Summary: Total number of mismatches and matches
    total_mismatches = len(mismatches[0])
    print(f"Total mismatches: {total_mismatches}")
    total_matches = len(matches[0])
    print(f"Total matches: {total_matches}")
    max_diff = int(np.abs(nx_output - expected_output).max())
    print(f"Max Difference: {max_diff}")
    
    # with open(os.path.join(output_dir, "results_"+str(output_num)+".txt"), "w") as file:
    #     file.write("Total mismatches: " + str(total_mismatches) + "\n")
    #     file.write("Total matches: " + str(total_matches) + "\n")
    #     file.write("max_diff: " + str(max_diff) + "\n")

# Send sync to MXP
def send_sync(tensor_data):
    data = tensor_data.flatten()
    if not TFLITE_REQUANT:
        # Add 128 because TFLite is int8, ONNX is uint8
        data = data - 128
    data = data.astype(np.int8)
    data_bytes = data.tobytes()
    size_in_bytes = len(data_bytes)
    print('send_sync: Acquire Semaphore')
    while True:
        try:
            shm = posix_ipc.SharedMemory(sync_name, flags=posix_ipc.O_CREX, size=size_in_bytes)
            semaphore.acquire()
        except posix_ipc.ExistentialError:
            continue
        else:
            break
            
    try:
        # Map the shared memory block into process's address space
        mapfile = mmap.mmap(shm.fd, shm.size)
        # Write data to shared memory
        mapfile.seek(0)
        mapfile.write(data_bytes)
        mapfile.flush()
    finally:
        semaphore.release()
        print('send_sync: Release Semaphore')
    
# Wait for sync from MXP
def wait_for_sync(return_tensor = False, tensor_shape = None):
    tensor_size = tensor_shape[0] * tensor_shape[1] * tensor_shape[2] * tensor_shape[3]
    print("Wating for shared memory creation ......")
    while True:
        try:
            memory = posix_ipc.SharedMemory(sync_name)
            if (tensor_size == memory.size):
                break
            else:
                os.close(memory.fd)
        except:
            continue
    
    print('wait_for_sync: Acquire Semaphore')
    semaphore.acquire()
    
    try:
        # Map the shared memory block into process's address space
        memory_map = mmap.mmap(memory.fd, memory.size)
        os.close(memory.fd)    
        # Read data from shared memory (in another process)
        data = memory_map.read()
        tensor = np.frombuffer(data, dtype=np.int8)
        if not TFLITE_REQUANT:
            # Add 128 because TFLite is int8, ONNX is uint8
            tensor = (tensor + 128).astype(np.uint8)
        # Clean up
        memory_map.close()
        posix_ipc.unlink_shared_memory(sync_name)
    finally:
        semaphore.release()
        print('wait_for_sync: Release Semaphore')
    
    return tensor

# Function which simulates the Sync node
def nx_sync(node_name,model_inputs:dict,intermediate_tensors: dict,node, is_intermediate_node, debug_dir = None):

    assert len(node['inputs']) == 1
    assert len(node['outputs']) == 1


    # Send the sync by creating a file
    # Also write node['inputs'] to a file
    sync_input = intermediate_tensors[node['inputs'][0]]
    send_sync(sync_input.data)

    # Now, wait for the response
    output_data = wait_for_sync(return_tensor = True, tensor_shape=node['frontend']['output_tensor'].shape)

    # Read the mxp output to the tensor
    output_data = output_data.reshape(node['frontend']['output_tensor'].shape)
    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output_folding_factor_x = node['frontend']['output_tensor'].folding_factor_x
    output_folding_factor_y = node['frontend']['output_tensor'].folding_factor_y
    output_x_slices = node['frontend']['output_tensor'].x_slices

    #output_tensor_bytearray = create_tsnp_tensor_byte_array(output_data, is_intermediate_node=is_intermediate_node, num_xslices=output_x_slices, pad_value=output_zero_point)
    output_tensor_bytearray = create_tsnp_tensor_byte_array(output_data)
    output_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
    with open(output_filename,'wb') as bin_file:
        bin_file.write(output_tensor_bytearray)

    output_tensor = qTensor(output_data, scale=output_scale, zero_point=output_zero_point, folding_factor_x=output_folding_factor_x, folding_factor_y=output_folding_factor_y, x_slices=output_x_slices)
    output_name = node['outputs'][0]
    intermediate_tensors[output_name] = output_tensor

    return [output_tensor]

def get_nx_sync_function(node_name, execution_order_sorted_graph):
    if node_name == execution_order_sorted_graph[0]:
        # If at the start, no need to write, just wait
        assert False, "Need to add function for Sync at start"
    if node_name == execution_order_sorted_graph[-1]:
        # If at the end, no need to wait, just write
        assert False, "Need to add function for Sync at end"

    return nx_sync # Defined in this file currently

# This is based on simulate from simulator.py.
def simulate(ir: internal_representation.IR, model_inputs: dict, intermediate_nodes:list, debug_dir:str, sync_with_mxp: bool) -> Tuple[list,dict]:
    intermediate_tensors = {}
    current_op_outputs = []
    model_outputs = []
    execution_order_sorted_graph = ir.lexicographical_topological_sorted_graph

    # Simulate each node
    for node_name in execution_order_sorted_graph:
        node = ir.graph.nodes[node_name]
        current_op_type = node['op_type']
        
        is_intermediate_node = False
        if (intermediate_nodes != None):
            if (node_name in intermediate_nodes):
                is_intermediate_node = True
        
        if 'output_reorder_node' in node:
            continue
        if current_op_type in neuronix_ops.ops_directory:
            neuronix_op = neuronix_ops.ops_directory[current_op_type]
        elif current_op_type == "Sync":
            neuronix_op = get_nx_sync_function(node_name, execution_order_sorted_graph)
        else:
            raise ValueError('Op type: %s not supported by nx compiler' % current_op_type)
        current_op_outputs = neuronix_op(node_name,model_inputs,intermediate_tensors,node,is_intermediate_node=is_intermediate_node,debug_dir=debug_dir)
        
        if 'split' in node_name and current_op_type != "Resize":
            output_name = node['outputs'][0]
            prev_split_output_name = output_name[:-1] + str(int(output_name[-1]) - 1)
            original_output_name = output_name.split('_split')[0]
            if original_output_name not in intermediate_tensors:
                output_tensor = intermediate_tensors[prev_split_output_name]
                original_tensor = qTensor(output_tensor.data, scale = output_tensor.scale, zero_point = output_tensor.zero_point, \
                                        folding_factor_x = output_tensor.folding_factor_x,folding_factor_y=output_tensor.folding_factor_y, x_slices=output_tensor.x_slices)
                intermediate_tensors[original_output_name] = original_tensor
            intermediate_tensors[original_output_name].data = \
                np.concatenate((intermediate_tensors[original_output_name].data, intermediate_tensors[output_name].data), axis = 1)
            
    for output_name in ir.outputs:
        intermediate_tensor_name = 'pre_ordering_'+output_name
        if intermediate_tensor_name not in intermediate_tensors:
            intermediate_tensor_name = output_name
            if intermediate_tensor_name not in intermediate_tensors:
                raise ValueError ('Didnt find output name in workloads output tensors. Please check...')
        model_outputs.append(intermediate_tensors[intermediate_tensor_name])

    # Sync always at the end of the network.
    # Note: This is done after so it includes the final reordering

    return model_outputs,intermediate_tensors

def main():
    parser = argparse.ArgumentParser(description='VBX 3.0 numeric simulator')

    # Required
    parser.add_argument('-m','--model', default=None, help='Quantized TFLite model file', required=True)
    parser.add_argument('-o','--output_dir', default=None, help='Output directory for compiler results.', required=True)

    # Optional
    parser.add_argument('-i','--input_file', default=None, help='Input csv file with tensor data (int8).', required=False)
    parser.add_argument('-t','--tflite', default=None, help='Path to .tflite file to calculate expected (int8) output.', required=False)
    parser.add_argument('--no_mxp', action="store_true", help='Simulate TFLite standalone (no syncing with MXP).')
    parser.add_argument('-r','--output_order', default=None, help='Specify the output order for vectorblox outputs.', required=False)
    
    global args
    args = parser.parse_args()

    sync_with_mxp = True
    if args.no_mxp:
        sync_with_mxp = False

    model_name = args.model
    output_dir = args.output_dir

    print('Running numeric simulator for model: %s...' % args.model)

    ir = internal_representation.IR('')
    ir_filename = os.path.join(output_dir,(model_name+'_numsim.nxir'))
    ir = ir.load(ir_filename)

    if ir.debug:
        debug_dir = os.path.join(output_dir, 'temp', 'debug')
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
    else:
        tmp_dir_obj = tempfile.TemporaryDirectory()
        debug_dir = tmp_dir_obj.name
    

    model_bin = os.path.join(output_dir, model_name+'.ucomp')
    if not os.path.exists(model_bin):
        print("Please generate combined binary file .ucomp before running the numeric simulator")
        sys.exit(0)
    
    # List to store extracted data
    intermediate_nodes = []    
    if DEBUG_OPTIMIZE_DMA_TRANSACTIONS:
        ddr_file_path = os.path.join(output_dir, model_name + '_ddr_info.txt')
        # Pattern to extract relevant data
        pattern = re.compile(
            r"Address: .*?Description:.*?Blob.*?producer node: (\S+), folded shape: \[(\d+), (\d+), (\d+), (\d+)\]"
        )
        with open(ddr_file_path, "r") as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    producer_node = match.groups()[0]

                    # MXP/TSNP Sync nodes aren't intermediate nodes
                    if 'Sync_' in producer_node:
                        continue

                    x = int(match.groups()[4])
                    intermediate_nodes.append(producer_node)
    
    tests_to_run=1
    #tqdm_iterator = tqdm(range(tests_to_run))
    #for image_idx in tqdm_iterator:
    for image_idx in range(tests_to_run):
        #padded_inputs are used for neuronix model
        # e.g. width7 is padded to 14(minimal width) 27 will be padded to 28 etc
        # we always pad to 14*2^X

        # If simulating with MXP, sync always at the start of the network (get a start from MXP).
        # Input is currently read from MXP.
        # Otherwise, will read from file.
        padded_inputs, input_data = get_model_inputs(ir, sync_with_mxp, args.input_file, model_bin, pad_to_gridsize=False)

        nx_outputs, _ = simulate(ir, padded_inputs, intermediate_nodes, debug_dir, sync_with_mxp)

        input_name = ir.inputs[0]
        input_tensor = ir.tensors[input_name]
        input_tensor = list(padded_inputs.values())[0].data
        input_tensor_bytearray = bytearray(input_tensor.astype(np.int8))
        input_filename = os.path.join(debug_dir,(model_name+'_input.nxi'))
        with open(input_filename,'wb') as bin_file:
            bin_file.write(input_tensor_bytearray)

        nx_ordered_outputs = []
        nx_folded_ordered_outputs = []
        nx_ordered_output_zps = []
        for nx_output in nx_outputs:
            nx_folded_ordered_outputs.append(nx_output.data.astype(np.int8))
            if (nx_output.folding_factor_x!=0 or nx_output.folding_factor_y!=0) and DEBUG_SIMULATE_FOLDING: # We first get all tensors to be unfolded
                nx_output.data = folding_algo.get_asym_unfolded_tensor(nx_output.data,folding_factor_x=nx_output.folding_factor_x,folding_factor_y=nx_output.folding_factor_y)
            nx_ordered_outputs.append(nx_output.data)
            nx_ordered_output_zps.append(nx_output.zero_point)

        # Sync always at the end of the network.
        # Note: This is done here to include the final reordering
        if sync_with_mxp:
            if (len(nx_ordered_outputs) > 1):
                vnnx_order = json.loads(args.output_order)
                vnnx_ordered_outputs = []
                vnnx_ordered_output_zps = []
                vnnx_folded_ordered_outputs = []
                used_idx = []
                for i in range(len(nx_ordered_outputs)):
                    for j in range(len(nx_ordered_outputs)):
                        channels = len(nx_ordered_outputs[i][0])
                        rows = len(nx_ordered_outputs[i][0][0])
                        cols = len(nx_ordered_outputs[i][0][0][0])
                        if (channels == int(vnnx_order[j][1])) and (rows == int(vnnx_order[j][2])) and (cols == int(vnnx_order[j][3])) and \
                            (j not in used_idx):
                            used_idx.append(j)
                            break
                    vnnx_ordered_outputs.append(nx_ordered_outputs[j])
                    vnnx_ordered_output_zps.append(nx_ordered_output_zps[j])
                    vnnx_folded_ordered_outputs.append(nx_folded_ordered_outputs[j])
            else:
                vnnx_ordered_outputs = nx_ordered_outputs
            for output in vnnx_ordered_outputs:
                send_sync(output)
        else:
            vnnx_folded_ordered_outputs = nx_folded_ordered_outputs
            vnnx_ordered_outputs = nx_ordered_outputs
                
            input_tensor_bytearray = None
            output_tensor_bytearray = None
            if NODES_LIST['Start']:
                node_name = NODES_LIST['Start'][0]
                node = ir.graph.nodes[node_name]
                input_node_name = node['frontend']['input_tensor'].producer
                input_filename = os.path.join(debug_dir,(input_node_name+'_output_tensor.nxo'))
                input_size = os.path.getsize(input_filename)
                with open(input_filename,'rb') as bin_file:
                    input_tensor_bytearray = bin_file.read()
            if NODES_LIST['End']:
                node_name = NODES_LIST['End'][0]
                output_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
                output_size = os.path.getsize(output_filename)
                with open(output_filename,'rb') as bin_file:
                    output_tensor_bytearray = bin_file.read()

            if (input_tensor_bytearray != None) or (output_tensor_bytearray != None):
                with open(model_bin, 'r+b') as mf:
                    header_size = struct.unpack('i', mf.read(4))[0]
                    num_inputs_offset = 12
                    num_outputs_offset = 16
                    input_size_offset = 20
                    input_address_offset = 24
                    output_size_offset = 28
                    output_address_offset = 32
                    
                    if input_tensor_bytearray != None:
                        # Overwrite number of inputs
                        mf.seek(num_inputs_offset, 0)
                        num_inputs = int(1)
                        mf.write(num_inputs.to_bytes(4, 'little'))
                        # Overwrite input size
                        mf.seek(input_size_offset, 0)
                        mf.write(input_size.to_bytes(4, 'little'))
                        # Overwrite input address
                        mf.seek(input_address_offset, 0)
                        input_address = list(ir.input_ddr_offset)[0]
                        input_address = math.ceil(input_address/16)*16
                        mf.write(input_address.to_bytes(4, 'little'))
                        # Overwrite input data
                        mf.seek(header_size+input_address, 0)
                        mf.write(input_tensor_bytearray)
                        print(f"Successfully written Output {len(input_tensor_bytearray)} bytes at {hex(input_address)} offset in {model_bin}")

                    if input_tensor_bytearray != None:
                        # Overwrite number of outputs
                        mf.seek(num_outputs_offset, 0)
                        num_outputs = int(1)
                        mf.write(num_outputs.to_bytes(4, 'little'))
                        # Overwrite output size
                        mf.seek(output_size_offset, 0)
                        mf.write(output_size.to_bytes(4, 'little'))
                        # Overwrite output address
                        mf.seek(output_address_offset, 0)
                        output_address = list(ir.output_ddr_offset)[0]
                        output_address = math.ceil(output_address/16)*16
                        mf.write(output_address.to_bytes(4, 'little'))
                        # Overwrite output data
                        mf.seek(header_size+output_address, 0)
                        mf.write(output_tensor_bytearray)
                        print(f"Successfully written Output {len(output_tensor_bytearray)} bytes at {hex(output_address)} offset in {model_bin}")
            else:           
                # writing folded_output of VBX3.0 simulator to the .ucomp and .nxo file
                with open(model_bin, 'r+b') as mf:
                    header_size = struct.unpack('i', mf.read(4))[0]
                    num_output_offset = 16
                    mf.seek(num_output_offset, 0)
                    num_outputs = struct.unpack('i', mf.read(4))[0]
                    used_idx = []
                    for o, output in enumerate(vnnx_ordered_outputs):
                        output_bytearray = bytearray(output.astype(np.int8))
                        output_size = len(output_bytearray)
                        for out_idx in range(num_outputs):
                            output_size_offset = 28 + (8 * out_idx)
                            mf.seek(output_size_offset, 0)
                            out_size = struct.unpack('i', mf.read(4))[0]
                            if (output_size == out_size) and (out_idx not in used_idx):
                                used_idx.append(out_idx)
                                break
                        # Writing the test output to .ucomp file
                        output_address_offset = 32 + (8 * out_idx)
                        mf.seek(output_address_offset, 0)
                        output_address = header_size + struct.unpack('i', mf.read(4))[0]
                        mf.seek(output_address, 0)
                        mf.write(output_bytearray)
                        print(f"Successfully written Output #{o} {len(output_bytearray)} bytes at {hex(output_address)} offset in {model_bin}")
        
        # writing the .nxo file
        for o, output in enumerate(vnnx_ordered_outputs):
            output_bytearray = bytearray(output.astype(np.int8))
            output_filename = os.path.join(debug_dir,(model_name+'_output_'+str(o)+'.nxo'))
            with open(output_filename,'wb') as bin_file:
                bin_file.write(output_bytearray)

        # Compare to either the result from running TFLite, or tflite/vnnx output files
        expected_outputs = []
        if (args.tflite != None):
            # Calculate expected outputs from .tflite file
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=args.tflite)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            input = input_data.astype(np.int8)
            if ir.uint8_int8_lut is not None:
                input = np.where(input >= 0, input-128, np.where((input < 0), input+128, input))
                input = input.astype(np.int8)
            batch_size, height, width, channels = input_details[0]['shape']
            new_width = math.ceil(width/16)*16
            input = input.reshape(channels, height, new_width).transpose(1, 2, 0)
            input = input[np.newaxis, :, :width, :]

            interpreter.set_tensor(input_details[0]['index'], input)    # Keep as int8
            interpreter.invoke()
            outputs = []
            for i in range(len(output_details)):
                output_data = interpreter.get_tensor(output_details[i]['index'])
                if len(output_data.shape) == 4:
                    output_data = output_data.transpose((0,3,1,2))
                outputs.append(output_data)    # Keep as int8
            
            used_idx = []
            for i in range(len(vnnx_ordered_outputs)):
                channels = len(vnnx_ordered_outputs[i][0])
                rows = len(vnnx_ordered_outputs[i][0][0])
                cols = len(vnnx_ordered_outputs[i][0][0][0])
                for j in range(len(outputs)):                    
                    ref_ch = len(outputs[j][0])
                    ref_r = len(outputs[j][0][0])
                    ref_c = len(outputs[j][0][0][0])    
                    if (j not in used_idx) and (channels == ref_ch) and (rows == ref_r) and ((cols == ref_c) or (cols == math.ceil(ref_c/16)*16)):
                        used_idx.append(j)
                        break
                expected_outputs.append(outputs[j])
            
        if expected_outputs:
            assert len(vnnx_ordered_outputs) == len(expected_outputs)
            expected_tensor_bytearray = bytearray(expected_outputs[0].astype(np.int8))
            filename = os.path.join(debug_dir,(model_name+'_expected.nxo'))
            with open(filename,'wb') as bin_file:
                bin_file.write(expected_tensor_bytearray)
            for i in range(len(vnnx_ordered_outputs)):
                if sync_with_mxp:
                    compare_to_expected(vnnx_ordered_outputs[i], expected_outputs[i], i, vnnx_ordered_output_zps[i], output_dir)
                else:
                    compare_to_expected(vnnx_ordered_outputs[i], expected_outputs[i], i, nx_ordered_output_zps[i], output_dir)
        
        time.sleep(10)

        try:
            # Attempt to unlink (delete) the semaphore by its name
            posix_ipc.unlink_semaphore(sync_semaphore)
            print(f"Semaphore '{sync_semaphore}' unlinked successfully.")
        except posix_ipc.ExistentialError:
            # This error occurs if the semaphore does not exist
            print(f"Semaphore '{sync_semaphore}' does not exist, cannot unlink.")
        except Exception as e:
            print(f"An error occurred: {e}")

    if not ir.debug:
        tmp_dir_obj.cleanup() 

if __name__ == "__main__":
    main()
