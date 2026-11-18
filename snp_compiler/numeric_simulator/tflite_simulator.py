import sys
sys.path.append('.')
import common.internal_representation as internal_representation
from common.hw_config import MAX_GRID_WIDTH, MAX_GRID_HEIGHT, OpType
import math
import numpy as np
import numeric_simulator.neuronix_ops as neuronix_ops
from numeric_simulator.defs import qTensor
import os
from common.debug_flags import DEBUG_SIMULATE_FOLDING, DEBUG_OPTIMIZE_DMA_TRANSACTIONS, NODES_LIST
from common.ddr_ir import create_tsnp_tensor_byte_array
import compiler.folding_algo as folding_algo
from typing import Tuple
import argparse
import struct
import mmap
import posix_ipc
from numpy.random import RandomState
import re
import tempfile
import contextlib

# Two independent channels, one per direction: sharing one full/empty pair would let
# NX's own send_sync() immediately followed by wait_for_sync() consume the token it
# just posted itself, instead of MXP's reply.
SHM_NAME_H2D = "/shm_tsnp_mxp_h2d"  # NX -> MXP (send_sync)
SEM_EMPTY_NAME_H2D = "/sem_empty_h2d"
SEM_FULL_NAME_H2D = "/sem_full_h2d"
SHM_NAME_D2H = "/shm_tsnp_mxp_d2h"  # MXP -> NX (wait_for_sync)
SEM_EMPTY_NAME_D2H = "/sem_empty_d2h"
SEM_FULL_NAME_D2H = "/sem_full_d2h"


def _set_ipc_names(run_id):
    """Append a per-run suffix to the POSIX shared-memory / semaphore names.

    The shared memory segment and the sem_empty/sem_full semaphores live in a
    system-wide namespace, so running several simulations concurrently with the
    fixed default names makes them clobber each other's data and mispair their
    syncs. Giving each run a unique id isolates them. The MXP peer that creates
    these objects must use the same names. A falsy run_id keeps the original
    names for backward compatibility."""
    global SHM_NAME_H2D, SEM_EMPTY_NAME_H2D, SEM_FULL_NAME_H2D
    global SHM_NAME_D2H, SEM_EMPTY_NAME_D2H, SEM_FULL_NAME_D2H
    if not run_id:
        return
    SHM_NAME_H2D = f"/shm_tsnp_mxp_h2d_{run_id}"
    SEM_EMPTY_NAME_H2D = f"/sem_empty_h2d_{run_id}"
    SEM_FULL_NAME_H2D = f"/sem_full_h2d_{run_id}"
    SHM_NAME_D2H = f"/shm_tsnp_mxp_d2h_{run_id}"
    SEM_EMPTY_NAME_D2H = f"/sem_empty_d2h_{run_id}"
    SEM_FULL_NAME_D2H = f"/sem_full_d2h_{run_id}"


def _round_up_to_pow2_mult(value, base):
    if value <= base:
        return base
    ratio = value / base
    exp = math.ceil(math.log2(ratio))
    return base << exp


def _write_nxo(debug_dir, name, data):
    output_filename = os.path.join(debug_dir, name + '.nxo')
    with open(output_filename, 'wb') as bin_file:
        bin_file.write(data)


def _unlink_sem(name):
    try:
        posix_ipc.unlink_semaphore(name)
        print(f"Semaphore '{name}' unlinked successfully.")
    except posix_ipc.ExistentialError:
        print(f"Semaphore '{name}' does not exist, cannot unlink.")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_array_from_csv(input_file: str) -> np.array:
    with open(input_file, 'r') as file:
        content = file.read()
    content = content.strip().rstrip(',')
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
                else:
                    prng = RandomState(5095)
                    input = prng.randint(size=input_shape,low=-128,high=127)
        input_tensor = qTensor(input,input_scale,zero_point)
        inputs[graph_input] = input_tensor
        # Pad to nearest grid size
        if pad_to_gridsize and len(input_shape)==4:
            # The below handles case where we need to pad x axis original tensor to fit grid width once folded.
            # This is not needed in y axis?
            current_input_width = input_shape[3]
            current_input_height = input_shape[2]
            padded_width = _round_up_to_pow2_mult(current_input_width, MAX_GRID_WIDTH)
            x_padding = padded_width - current_input_width
            padded_height = _round_up_to_pow2_mult(current_input_height, ir.amm_height)
            y_padding = padded_height - current_input_height
            padded_input=np.pad(input,[(0,0),(0,0),(0,y_padding),(0,x_padding)], mode='constant', constant_values=0)
            padded_input_tensor = qTensor(padded_input,input_scale,zero_point)
            padded_inputs[graph_input] = padded_input_tensor
        else:
            padded_inputs = inputs
    return padded_inputs, input

# Compare to an expected output tensor. Returns True on pass, False on fail.
def compare_to_expected(nx_output, expected_output, zero_point):
    # Comparison threshold (for int8 quantized outputs)
    THRESHOLD = 1

    if nx_output.shape[2] > expected_output.shape[2] or nx_output.shape[3] > expected_output.shape[3]:
        nx_output = nx_output[:,:,:expected_output.shape[2],:expected_output.shape[3]]

    mismatches = np.where(abs(nx_output - expected_output) > THRESHOLD)
    matches = np.where(abs(nx_output - expected_output) <= THRESHOLD)
    max_diff = int(np.abs(nx_output - expected_output).max())
    print(f"Max Difference: {max_diff}")

    if (len(mismatches[0]) > 0):
        total_mismatches = len(mismatches[0])
        print(f"Total mismatches: {total_mismatches}")
        total_matches = len(matches[0])
        print(f"Total matches: {total_matches}")
        return False
    else:
        return True

# Send sync to MXP
def send_sync(tensor_data):
    data = tensor_data.flatten()
    data = data.astype(np.int8)
    data_bytes = data.tobytes()
    size_in_bytes = len(data_bytes)
    sem_shm_empty = posix_ipc.Semaphore(SEM_EMPTY_NAME_H2D)
    sem_shm_full = posix_ipc.Semaphore(SEM_FULL_NAME_H2D)

    print('send_sync: Acquire Semaphore')
    sem_shm_empty.acquire()
    try:
        # Unlink any existing shared memory (from prior syncs in multi-split scenarios)
        try:
            posix_ipc.unlink_shared_memory(SHM_NAME_H2D)
        except:
            pass
        shm = posix_ipc.SharedMemory(SHM_NAME_H2D, flags=posix_ipc.O_CREX, size=size_in_bytes)
        # Map the shared memory block into process's address space
        mapfile = mmap.mmap(shm.fd, shm.size)
        # Write data to shared memory
        mapfile.seek(0)
        mapfile.write(data_bytes)
        mapfile.flush()
    finally:
        sem_shm_full.release()
        print('send_sync: Release Semaphore')

# Wait for sync from MXP
def wait_for_sync(return_tensor = False, tensor_shape = None, sync_count = 1):
    tensor_size = tensor_shape[0] * tensor_shape[1] * tensor_shape[2] * tensor_shape[3]
    sem_shm_empty = posix_ipc.Semaphore(SEM_EMPTY_NAME_D2H)
    sem_shm_full = posix_ipc.Semaphore(SEM_FULL_NAME_D2H)

    # MXP receives N syncs (one per input). Only read data once (on first sync).
    # If the C side was compiled expecting only 1 sync but we're sending N, it will
    # call wait_for_sync once and then the extra syncs hang. This fallback detects
    # that and absorbs the extra syncs on the Python side.
    tensor = None
    syncs_received = 0
    while syncs_received < sync_count:
        print('wait_for_sync: Acquire Semaphore (sync %d/%d)' % (syncs_received + 1, sync_count))
        sem_shm_full.acquire()
        try:
            memory = posix_ipc.SharedMemory(SHM_NAME_D2H)
            if (tensor_size != memory.size):
                raise ValueError('TENSOR from MXP: SIZE MISMATCHED')
            # Map the shared memory block into process's address space
            memory_map = mmap.mmap(memory.fd, memory.size)
            os.close(memory.fd)
            # Read data from shared memory (in another process) - only on first sync
            if syncs_received == 0:
                data = memory_map.read()
                tensor = np.frombuffer(data, dtype=np.int8)
            # Clean up
            memory_map.close()
        finally:
            sem_shm_empty.release()
            print('wait_for_sync: Release Semaphore (sync %d/%d)' % (syncs_received + 1, sync_count))
        syncs_received += 1

    # Unlink shared memory once after all syncs are received
    posix_ipc.unlink_shared_memory(SHM_NAME_D2H)

    return tensor

# Function which simulates the Sync node
def nx_sync(node_name,model_inputs:dict,intermediate_tensors: dict,node, is_intermediate_node, debug_dir = None,amm_height=MAX_GRID_HEIGHT,tensors_to_mxp=None,split_tensor_to_original_tensor_map=None,original_tensor_to_split_tensor_map=None):

    assert len(node['outputs']) == 1
    # Sync nodes can have multiple inputs when they coordinate between channel-split pieces.
    # Use the first input as the representative for send_sync.
    assert len(node['inputs']) >= 1


    # Send one sync for each input tensor. MXP receives N syncs before proceeding.
    for input_name in node['inputs']:
        sync_input_name = input_name
        while split_tensor_to_original_tensor_map and sync_input_name in split_tensor_to_original_tensor_map:
            sync_input_name = split_tensor_to_original_tensor_map[sync_input_name]
        if sync_input_name not in intermediate_tensors and original_tensor_to_split_tensor_map:
            neuronix_ops._reassemble_split_tensor(sync_input_name, intermediate_tensors, original_tensor_to_split_tensor_map)
        sync_input = intermediate_tensors[sync_input_name]
        send_sync(sync_input.data)

    # Now, wait for the response. MXP processes all N inputs and sends back 1 output.
    # Always wait for 1 sync (the output), regardless of number of inputs.
    output_data = wait_for_sync(return_tensor = True, tensor_shape=node['frontend']['output_tensor'].shape, sync_count=1)

    # Read the mxp output to the tensor
    output_data = output_data.reshape(node['frontend']['output_tensor'].shape)
    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output_folding_factor_x = node['frontend']['output_tensor'].folding_factor_x
    output_folding_factor_y = node['frontend']['output_tensor'].folding_factor_y
    output_x_slices = node['frontend']['output_tensor'].x_slices

    output_tensor_bytearray = create_tsnp_tensor_byte_array(output_data)
    _write_nxo(debug_dir, node_name + '_output_tensor', output_tensor_bytearray)

    output_tensor = qTensor(output_data, scale=output_scale, zero_point=output_zero_point, folding_factor_x=output_folding_factor_x, folding_factor_y=output_folding_factor_y, x_slices=output_x_slices)
    output_name = node['outputs'][0]
    intermediate_tensors[output_name] = output_tensor

    return [output_tensor]

# This is based on simulate from simulator.py.
def simulate(ir: internal_representation.IR, model_inputs: dict, intermediate_nodes:list, debug_dir:str, sync_with_mxp: bool) -> Tuple[list,dict]:
    intermediate_tensors = {}
    output_split_reconstruction_buffers = {}
    current_op_outputs = []
    model_outputs = []
    execution_order_sorted_graph = ir.lexicographical_topological_sorted_graph
    neuronix_ops.set_current_graph_nodes(ir.graph.nodes)

    # Simulate each node
    for node_name in execution_order_sorted_graph:
        node = ir.graph.nodes[node_name]
        current_op_type = node['op_type']

        is_intermediate_node = False
        if (intermediate_nodes != None):
            if (node_name in intermediate_nodes):
                is_intermediate_node = True
            # Resize splits into '_x'/'_y' nodes; '_y' does the AMM->DDR write and lands in
            # intermediate_nodes, but '_x' is the one that exports the .nxo/.nxd (nx_resize returns
            # early for '_y'). Let '_x' inherit its '_y' sibling's intermediate status.
            elif current_op_type == OpType.RESIZE and node_name.endswith('_x') \
                    and (node_name[:-2] + '_y') in intermediate_nodes:
                is_intermediate_node = True

        output_tensors = ir.tensors_to_mxp.copy()
        output_tensors.update(ir.outputs)

        if 'output_reorder_node' in node:
            continue
        if current_op_type in neuronix_ops.ops_directory:
            neuronix_op = neuronix_ops.ops_directory[current_op_type]
        elif current_op_type == OpType.SYNC:
            assert node_name != execution_order_sorted_graph[0], "Need to add function for Sync at start"
            assert node_name != execution_order_sorted_graph[-1], "Need to add function for Sync at end"
            neuronix_op = nx_sync
        else:
            raise ValueError('Op type: %s not supported by nx compiler' % current_op_type)
        current_op_outputs = neuronix_op(node_name,model_inputs,intermediate_tensors,node,is_intermediate_node=is_intermediate_node,debug_dir=debug_dir,amm_height=ir.amm_height,tensors_to_mxp=output_tensors,split_tensor_to_original_tensor_map=ir.split_tensor_to_original_tensor_map,original_tensor_to_split_tensor_map=ir.original_tensor_to_split_tensor_map)

        # OUTPUT-channel split reconstruction: pieces '<orig>_split0'..'_splitN' (registered by
        # split_node_out_channels_in_n) are concatenated back into <orig> along channels as each
        # piece finishes. Use split_tensor_to_original_tensor_map as the authoritative signal rather
        # than name-substring heuristics: a node can be both an input-channel-split recombine Add
        # (not concatenated) and an outer output-channel-split piece producer (must be concatenated),
        # and no name-based rule distinguishes the two.
        _split_recombine_output_name = node['outputs'][0]
        original_output_name = ir.split_tensor_to_original_tensor_map.get(_split_recombine_output_name)
        if (original_output_name is not None and original_output_name != _split_recombine_output_name
                and current_op_type != OpType.RESIZE):
            output_name = _split_recombine_output_name
            # Pieces don't necessarily finish in channel order, so buffer by split_channel_lo and
            # only concatenate once every piece covering the full channel range has arrived.
            lo = ir.tensors[output_name].split_channel_lo
            output_split_reconstruction_buffers.setdefault(original_output_name, {})[lo] = intermediate_tensors[output_name]
            buffered = output_split_reconstruction_buffers[original_output_name]
            full_channels = ir.tensors[original_output_name].shape[1]
            # Compare REAL (unfolded) channel counts, not raw .data.shape[1]: a folded piece's raw
            # count is real_count * 2**(folding_factor_x+folding_factor_y), so summing raw counts can
            # spuriously match full_channels early, firing the merge before all pieces arrive.
            buffered_channels = sum(
                t.data.shape[1] // (2 ** (t.folding_factor_x + t.folding_factor_y)) for t in buffered.values())
            if buffered_channels == full_channels:
                ordered_pieces = [buffered[lo] for lo in sorted(buffered.keys())]
                first_piece = ordered_pieces[0]
                merged_data = neuronix_ops._merge_split_pieces_data(ordered_pieces)
                intermediate_tensors[original_output_name] = qTensor(merged_data, scale = first_piece.scale, zero_point = first_piece.zero_point, \
                                        folding_factor_x = first_piece.folding_factor_x, folding_factor_y = first_piece.folding_factor_y, x_slices = first_piece.x_slices)
                del output_split_reconstruction_buffers[original_output_name]

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


# Return the MXP's NX->MXP sync wait order (ordered list of NX tensor ids, duplicates included),
# or None when absent so the caller falls back to legacy behaviour. Persisted into the numsim
# .nxir at compile time (write_mxp_sync_order / parse_mxp_sync_order).
def read_mxp_sync_order(ir):
    return getattr(ir, 'mxp_sync_order', None)


# Map an NX IR output name ('T<tflite_tensor_id>') to its integer tensor id.
def nx_ir_output_id(name):
    m = re.match(r'T(\d+)', str(name))
    return int(m.group(1)) if m is not None else None

def main():
    parser = argparse.ArgumentParser(description='VBX 3.0 numeric simulator')

    # Required
    parser.add_argument('-m','--model', default=None, help='Quantized TFLite model file', required=True)
    parser.add_argument('-o','--output_dir', default=None, help='Output directory for compiler results.', required=True)

    # Optional
    parser.add_argument('-i','--input_file', default=None, help='Input csv file with tensor data (int8).', required=False)
    parser.add_argument('-t','--tflite', default=None, help='Path to .tflite file to calculate expected (int8) output.', required=False)
    parser.add_argument('--no_mxp', action="store_true", help='Simulate TFLite standalone (no syncing with MXP).')
    parser.add_argument('--run_id', default=os.environ.get('SIM_RUN_ID'),
                        help='Unique suffix for the POSIX shared-memory/semaphore names so multiple '
                             'simulations can sync with the MXP in parallel without colliding. The MXP '
                             'peer must create objects with matching names. Defaults to the SIM_RUN_ID '
                             'environment variable when set, otherwise the original fixed names are used.')

    args = parser.parse_args()

    _set_ipc_names(args.run_id)

    sync_with_mxp = not args.no_mxp

    model_name = args.model
    output_dir = args.output_dir

    print('Running numeric simulator for model: %s...' % args.model)

    ir = internal_representation.IR('')
    ir_filename = os.path.join(output_dir,(model_name+'_numsim.nxir'))
    ir = ir.load(ir_filename)

    with contextlib.ExitStack() as stack:
        if ir.debug:
            debug_dir = os.path.join(output_dir, 'temp', 'debug')
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
        else:
            debug_dir = stack.enter_context(tempfile.TemporaryDirectory())


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

        #padded_inputs are used for neuronix model
        # e.g. width7 is padded to 14(minimal width) 27 will be padded to 28 etc
        # we always pad to 14*2^X

        # If simulating with MXP, sync always at the start of the network (get a start from MXP).
        # Input is currently read from MXP.
        # Otherwise, will read from file.
        padded_inputs, input_data = get_model_inputs(ir, sync_with_mxp, args.input_file, model_bin, pad_to_gridsize=False)

        nx_outputs, _ = simulate(ir, padded_inputs, intermediate_nodes, debug_dir, sync_with_mxp)

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
                # The NX->MXP sync channel is an ordered, depth-1 FIFO with no tensor keying, so
                # sends must arrive in exactly the order MXP waits - generally NOT the model's
                # declared output order (mid-graph tensors like score maps are waited on before
                # final outputs; MXP-produced outputs are never synced). ir.mxp_sync_order carries
                # that exact order from the compiler.
                sync_order_ids = read_mxp_sync_order(ir)
                assert sync_order_ids is not None, (
                    "ir.mxp_sync_order not found in the numsim .nxir; recompile the model with a "
                    "current vnnx_compile so the MXP<->NX sync order is emitted.")
                # nx_ordered_outputs[i] corresponds to ir.outputs[i] (see simulate()); map each
                # NX output's tensor id to its index so we can emit the MXP wait order.
                nx_id_to_idx = {}
                output_tensor_aliases = getattr(ir, 'output_tensor_aliases', {})
                for i, oname in enumerate(ir.outputs):
                    tid = nx_ir_output_id(oname)
                    if tid is not None:
                        nx_id_to_idx[tid] = i
                    # oname may be renamed from the true tflite output (e.g. fused SiLU/Requant);
                    # mxp_sync_order still refers to the original id, so map that too.
                    alias_name = output_tensor_aliases.get(oname)
                    if alias_name is not None:
                        alias_tid = nx_ir_output_id(alias_name)
                        if alias_tid is not None:
                            nx_id_to_idx[alias_tid] = i
                missing = sorted({tid for tid in sync_order_ids if tid not in nx_id_to_idx})
                assert not missing, (
                    f"MXP sync order references tensor id(s) {missing} not produced by the "
                    f"NX engine (ir.outputs={list(ir.outputs)})")
                order = [nx_id_to_idx[tid] for tid in sync_order_ids]
                vnnx_ordered_outputs = [nx_ordered_outputs[i] for i in order]
                vnnx_ordered_output_zps = [nx_ordered_output_zps[i] for i in order]
                vnnx_folded_ordered_outputs = [nx_folded_ordered_outputs[i] for i in order]
            else:
                vnnx_ordered_outputs = nx_ordered_outputs
                vnnx_ordered_output_zps = nx_ordered_output_zps
                vnnx_folded_ordered_outputs = nx_folded_ordered_outputs
            for output in vnnx_ordered_outputs:
                send_sync(output)
        else:
            vnnx_folded_ordered_outputs = nx_folded_ordered_outputs
            vnnx_ordered_outputs = nx_ordered_outputs
            vnnx_ordered_output_zps = nx_ordered_output_zps

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
                        mf.write(input_address.to_bytes(4, 'little'))
                        # Overwrite input data
                        mf.seek(header_size+input_address, 0)
                        mf.write(input_tensor_bytearray)
                        print(f"Successfully written Output {len(input_tensor_bytearray)} bytes at {hex(input_address)} offset in {model_bin}")

                    if output_tensor_bytearray != None:
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
                    used_idx = set()
                    for o, output in enumerate(vnnx_ordered_outputs):
                        output_bytearray = bytearray(output.astype(np.int8))
                        output_size = len(output_bytearray)
                        for out_idx in range(num_outputs):
                            output_size_offset = 28 + (8 * out_idx)
                            mf.seek(output_size_offset, 0)
                            out_size = struct.unpack('i', mf.read(4))[0]
                            if (output_size == out_size) and (out_idx not in used_idx):
                                used_idx.add(out_idx)
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
            _write_nxo(debug_dir, model_name + '_output_' + str(o), output_bytearray)

        # Compare to either the result from running TFLite, or tflite/vnnx output files
        expected_outputs = []
        if (args.tflite != None):
            # Calculate expected outputs from .tflite file
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=args.tflite)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Feed the reference tflite in ITS OWN input dtype: uint8-input references need the raw
            # uint8 sample, int8-input ones need the sign-flip. input_data (int8) holds the uint8
            # sample shifted by -128 arithmetically; verified against QUANTIZECONV_0 (+128 matches
            # exactly, a plain astype(uint8) byte-reinterpret is off by 128).
            if input_details[0]['dtype'] == np.uint8:
                input = (input_data.astype(np.int64) + 128).astype(np.uint8)
            else:
                input = input_data.astype(np.int8)
            batch_size, height, width, channels = input_details[0]['shape']
            input = input.reshape(channels, height, width).transpose(1, 2, 0)
            input = input[np.newaxis, :, :, :]

            interpreter.set_tensor(input_details[0]['index'], input)    # dtype matches the reference tflite
            interpreter.invoke()
            outputs = []
            for i in range(len(output_details)):
                output_data = interpreter.get_tensor(output_details[i]['index'])
                if len(output_data.shape) == 4:
                    output_data = output_data.transpose((0,3,1,2))
                outputs.append(output_data)    # Keep as int8

            used_idx = set()
            for i in range(len(vnnx_ordered_outputs)):
                channels, rows, cols = vnnx_ordered_outputs[i].shape[1:4]
                matched = None
                for j in range(len(outputs)):
                    if j in used_idx:
                        continue
                    ref_ch, ref_r, ref_c = outputs[j].shape[1:4]
                    # cols may be padded up to a multiple of 16 (grid width); accept either exact or padded match.
                    if (channels == ref_ch) and (rows == ref_r) and ((cols == ref_c) or (cols == math.ceil(ref_c/16)*16)):
                        matched = j
                        used_idx.add(j)
                        break
                if matched is None:
                    raise ValueError(f"No tflite reference output matches vnnx output {i} shape {(channels, rows, cols)}")
                expected_outputs.append(outputs[matched])

        all_passed = True
        if expected_outputs:
            assert len(vnnx_ordered_outputs) == len(expected_outputs)
            expected_tensor_bytearray = bytearray(expected_outputs[0].astype(np.int8))
            _write_nxo(debug_dir, model_name + '_expected', expected_tensor_bytearray)
            for i in range(len(vnnx_ordered_outputs)):
                passed = compare_to_expected(vnnx_ordered_outputs[i], expected_outputs[i], vnnx_ordered_output_zps[i])
                if not passed:
                    all_passed = False

        _unlink_sem(SEM_EMPTY_NAME_H2D)
        _unlink_sem(SEM_FULL_NAME_H2D)
        _unlink_sem(SEM_EMPTY_NAME_D2H)
        _unlink_sem(SEM_FULL_NAME_D2H)

    if all_passed:
        print("!!! TEST PASSED !!!")
    else:
        print("!!! TEST FAILED !!!")
        

if __name__ == "__main__":
    main()
