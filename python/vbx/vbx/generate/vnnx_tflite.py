import json
import struct
import sys
import os.path
from math import floor, ceil, log2, frexp, copysign, exp, tanh, pow, log, sqrt
from contextlib import contextmanager
import numpy as np
import os
import glob
import silence_tensorflow.auto
import tensorflow as tf

import vbx.sim
from .vnnx_types import *
from .vnnx_allocation import *
from .utils import compress1, compress2, compress4, json_load
from .split_tflite import is_split_idx_in_engine_graphs, channels_first_shape
from .transform_tflite import graph_pattern, lut_i8_to_u8
from scipy.linalg import toeplitz


VERBOSE = 0
USE_CONV_VECTOR = 1
PRECALC_OUTPUT = 0
USE_CONV_FIA = 1
USE_FIA_DB = 1

USE_PRECALC = 1
QMULT = 31
QTRUNCATE = 0

OPTIMIZED_MUL = 1
OPTIMIZED_ADD = 1
OPTIMIZED_DEPTHWISE = 0

ALL_WEIGHTS = 0
SPLIT_WEIGHT_SHAPER_BUFFERS = 1

CONV_NLF    = 1

QUANTIZATION_RECORD_WIDTH = 64
C_INPUT_L_WIDTH   = 10
C_INPUT_H_WIDTH   = 32
MULTIPLIER_WIDTH  = 16
O_SHIFT_WIDTH     = 2
C_INPUT_L_LSB   = 0
C_INPUT_H_LSB   = C_INPUT_L_WIDTH + C_INPUT_L_LSB
MULTIPLIER_LSB  = C_INPUT_H_WIDTH + C_INPUT_H_LSB
O_SHIFT_LSB     = MULTIPLIER_WIDTH + MULTIPLIER_LSB
QUANTIZATION_RECORD_WIDTH_BYTES = QUANTIZATION_RECORD_WIDTH // 8



def compress_weights(weights, verbose=False):
    # weights should be of shape [k,h,w,c]
    k,h,w,c = weights.shape

    weights = weights.flatten()

    UNCOMPRESSED_BLOCK_SIZE = 8
    num_blocks = ceil(np.prod(weights.shape)/UNCOMPRESSED_BLOCK_SIZE)
    weights_comp = np.zeros(0,dtype=int)
    ctrl_comp = np.zeros(0,dtype=int)

    ctrl_comp_arr = []
    weights_comp_arr = []

    R = 1
    while True:
        for block in range(num_blocks):
            weight_block = weights[block*UNCOMPRESSED_BLOCK_SIZE : (block+1)*UNCOMPRESSED_BLOCK_SIZE]

            # compress weights during compilation
            #   R               1       number of times the same block of activations is used
            #   ctrlComp        [R]     compressed mux control; each element is an int with the lowest 7 bits used
            #   weightsComp     [R,2]   compressed block weights
            if R == 1:
                success,ctrl,sign,ctrlComp,weightsComp = compress1(weight_block)
                if not success:
                    R = 2
                    weights_comp_arr = []
                    ctrl_comp_arr = []
                    break
                weights_comp_arr.append(weightsComp.flatten())
                ctrl_comp_arr.append(ctrlComp)
            elif R == 2:
                success,ctrl,sign,ctrlComp,weightsComp = compress2(weight_block)
                if not success:
                    R = 4
                    weights_comp_arr = []
                    ctrl_comp_arr = []
                    break
                weights_comp_arr.append(weightsComp.flatten())
                ctrl_comp_arr.append(ctrlComp)
            else:
                success,ctrl,sign,ctrlComp,weightsComp = compress4(weight_block)
                weights_comp_arr.append(weightsComp.flatten())
                ctrl_comp_arr.append(ctrlComp)

        if success:
            break
    
    compressed_c = c
    if R == 1:
        compressed_c = c // 4
    elif R == 2:
        compressed_c = c // 2

    if verbose:
        print("COMPRESSION/REPEAT LEVEL", R)
        print("channels before / after", c, "/", compressed_c)

    weights_comp = np.concatenate(weights_comp_arr)
    ctrl_comp = np.concatenate(ctrl_comp_arr)

    weights_comp_khwc = weights_comp.reshape((k,h,w,compressed_c))
    ctrl_comp_khwc   = ctrl_comp.reshape((k,h,w,compressed_c//2))

    return weights_comp_khwc, ctrl_comp_khwc, R


def pack_compressed_weights(weights_comp, ctrl_comp, R, tile_channels, tile_maps, fit_weights=0):
    """
    Packs compressed weights and control data for tiled neural network layers.
    This function reorganizes and packs compressed weights and control data for efficient processing
    in hardware accelerators. It handles channel padding, chunking, and bank separation, then
    concatenates and flattens the data into a single array suitable for hardware consumption.
    Args:
        weights_comp (np.ndarray): Compressed weights array of shape (k, h, w, compressed_c).
        ctrl_comp (np.ndarray): Control data array associated with the weights.
        R (int): Compression ratio (determines how channels are expanded).
        tile_channels (int): Number of channels per tile (will be padded to a multiple of 8).
        tile_maps (int): Number of output maps per tile.
    Returns:
        flat (np.ndarray): Flattened packed weights and control data.
        total_shape (tuple): Shape of the packed data before flattening.
    Notes:
        - The function pads tile_channels to a multiple of 8 for compression.
        - Channels are reorganized into banks and chunks for hardware compatibility.
        - The main loop iterates over kernel and channel chunks, transposes and concatenates
          the weights and control data, and flattens them into the output array.
    """
    k,h,w,compressed_c = weights_comp.shape
    c = compressed_c
    if R == 1:
        c = compressed_c * 4
    elif R == 2:
        c = compressed_c * 2

    # weights channels are padded tile-wise up to a multiple of 8 for compression
    tile_channels += (8 - (tile_channels % 8)) if (tile_channels % 8 != 0) else 0
    num_paired_channels = R * (tile_channels//8)
    # total_num_paired_channels == R * (c // 8)
    # total_num_paired_channels == compressed_c / 2
    # compressed_c              == total_num_paired_channels * 2

    # re-organize the pairs into their respective banks per channel group
    # k,h,w,c => k,c,h,w
    channel_chunks = ceil(c / tile_channels)
    for chunk in range(channel_chunks):
        bank0 = weights_comp[:,:,:,(chunk*num_paired_channels*2):(chunk+1)*num_paired_channels*2:2].transpose((0,3,1,2))
        bank1 = weights_comp[:,:,:,(chunk*num_paired_channels*2)+1:(chunk+1)*num_paired_channels*2:2].transpose((0,3,1,2))
        chunk_ctrl = ctrl_comp[:,:,:,(chunk*num_paired_channels):(chunk+1)*num_paired_channels].transpose((0,3,1,2))

        chunk_weights = np.concatenate((bank0,bank1), axis=1)
        
        if chunk == 0:
            rechunked_weights = chunk_weights
            rechunked_ctrl = chunk_ctrl
        else:
            rechunked_weights = np.concatenate((rechunked_weights,chunk_weights),axis=1)
            rechunked_ctrl = np.concatenate((rechunked_ctrl,chunk_ctrl), axis=1)

    weights_comp = rechunked_weights # [k,compressed_c,h,w]
    ctrl_comp = rechunked_ctrl # [k,compressed_c/2,h,w]

    # k,c,h,w -> c,h,w,k
    # chunks of weights are transposed (c,h,w,k) according to tile omaps then concatenated
    total_shape = (k, weights_comp.shape[1] + ctrl_comp.shape[1], h, w)
    flat = np.zeros(np.prod(total_shape), dtype=np.int8)

    bank_chunk_size = num_paired_channels * tile_maps * h * w # expected size for each bank per (kernel,channel) chunk
    weights_chunk_size = bank_chunk_size * 2
    kernel_chunks = ceil(k/tile_maps)
    
    if fit_weights == 1:
        chunk_start = 0
        chunk_end = 0
        bank0_chunk = np.zeros(np.prod((k, compressed_c//2, h, w)), dtype=np.int8)
        bank1_chunk = np.zeros(np.prod((k, compressed_c//2, h, w)), dtype=np.int8)
        ctrl_chunk  = np.zeros(np.prod((k, compressed_c//2, h, w)), dtype=np.int8)
        bank_channel_chunks = channel_chunks * 2
        for ke in range(kernel_chunks):
            bank_chunk_size = num_paired_channels * tile_maps * h * w # expected size for each bank per (kernel,channel) chunk
            bank_channel_chunk_offset = 0
            final_tile_maps = tile_maps
            if ke == kernel_chunks - 1: # adjust for final tile maps which may be different
                final_tile_maps = (k % tile_maps) if (k % tile_maps) else tile_maps
                bank_chunk_size = num_paired_channels * final_tile_maps * h * w
            for ch in range(channel_chunks):
                if ch == channel_chunks - 1:  # adjust for final tile channels which may be different
                    final_tile_channels = (c % tile_channels) if (c % tile_channels) else tile_channels
                    final_tile_num_paired_channels = R * (final_tile_channels//8)
                    bank_chunk_size = final_tile_num_paired_channels * final_tile_maps * h * w
                    
                    chunk_end += bank_chunk_size

                    bank0_chunk[chunk_start:chunk_end] = \
                        weights_comp[ke*tile_maps:(ke+1)*tile_maps,(bank_channel_chunk_offset+0)*num_paired_channels:((bank_channel_chunk_offset+0)*num_paired_channels) + final_tile_num_paired_channels,:,:].transpose((1,2,3,0)).flatten()
                    bank1_chunk[chunk_start:chunk_end] = \
                        weights_comp[ke*tile_maps:(ke+1)*tile_maps,((bank_channel_chunk_offset+0)*num_paired_channels) + final_tile_num_paired_channels:((bank_channel_chunk_offset+0)*num_paired_channels) + (2*final_tile_num_paired_channels),:,:].transpose((1,2,3,0)).flatten()
                else:

                    chunk_end += bank_chunk_size

                    bank0_chunk[chunk_start:chunk_end] = \
                        weights_comp[ke*tile_maps:(ke+1)*tile_maps,(bank_channel_chunk_offset+0)*num_paired_channels:(bank_channel_chunk_offset+1)*num_paired_channels,:,:].transpose((1,2,3,0)).flatten()
                    bank1_chunk[chunk_start:chunk_end] = \
                        weights_comp[ke*tile_maps:(ke+1)*tile_maps,(bank_channel_chunk_offset+1)*num_paired_channels:(bank_channel_chunk_offset+2)*num_paired_channels,:,:].transpose((1,2,3,0)).flatten()                
                
                ctrl_chunk[chunk_start:chunk_end] = \
                    ctrl_comp[ke*tile_maps:(ke+1)*tile_maps,ch*num_paired_channels:(ch+1)*num_paired_channels,:,:].transpose((1,2,3,0)).flatten()
                chunk_start = chunk_end
                bank_channel_chunk_offset += 2

        flat = np.concatenate((bank0_chunk, bank1_chunk, ctrl_chunk), axis=-1)
    else:
        flat_start = 0
        flat_end = 0
        for ke in range(kernel_chunks):
            total_chunk_size = weights_chunk_size + bank_chunk_size
            final_tile_maps = tile_maps
            if ke == kernel_chunks - 1: # adjust for final tile maps which may be different
                final_tile_maps = (k % tile_maps) if (k % tile_maps) else tile_maps
                total_chunk_size = (num_paired_channels * final_tile_maps * h * w) * 3
            for ch in range(channel_chunks):
                if ch == channel_chunks - 1:  # adjust for final tile channels which may be different
                    final_tile_channels = (c % tile_channels) if (c % tile_channels) else tile_channels
                    final_tile_num_paired_channels = R * (final_tile_channels//8)
                    total_chunk_size = (final_tile_num_paired_channels * final_tile_maps * h * w) * 3
                data_chunk = weights_comp[ke*tile_maps:(ke+1)*tile_maps,ch*num_paired_channels*2:(ch+1)*num_paired_channels*2,:,:].transpose((1,2,3,0))
                ctrl_chunk = ctrl_comp[ke*tile_maps:(ke+1)*tile_maps,ch*num_paired_channels:(ch+1)*num_paired_channels,:,:].transpose((1,2,3,0))
                flat_end  += total_chunk_size
                flat[flat_start:flat_end] = np.concatenate((data_chunk.flatten(), ctrl_chunk.flatten()), axis=-1)
                flat_start = flat_end

    return flat


def pad_weights_and_pack_2_1(filter_data, node, parallel_output_maps, tile, preset, opcode, weight_pad=0, is_transpose=False, \
                             sparse=None, tmp_dir=None, graph_idx=None, tmp_dir_obj=None):
    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    repack_maps = node.maps

    if node.type == BuiltinOperator.FULLY_CONNECTED:
        o,a = filter_data.shape
        o_ = (o + node.cols-1) // node.cols
        pad_value = 64 if (sparse==1) else 0 # control value 64 for only using bank1
        padded_data = np.full(filter_data.shape, fill_value=pad_value)

        flat = np.zeros(2*(o_ * node.cols * a), dtype=filter_data.dtype)
        
        for i in range(o_):
            f = filter_data[i*node.cols: (i+1)*node.cols]
            f_ = f.transpose((1,0)).flatten()
            pad_chunk = padded_data[i*node.cols: (i+1)*node.cols].transpose((1,0)).flatten()
            flat[2*i*node.cols*a:2*(i*node.cols+f.shape[0])*a] = np.concatenate((f_, pad_chunk), axis=-1)
        filter_data = flat
    elif node.type == BuiltinOperator.CONV_2D and node.Conv2DOptions.use_depthwise:
        # should only enter this block if depthwise and non-compressed
        tile = allocation(node, preset, opcode, sparse=sparse, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj) # re-tile in case of dilation (weights are padded, and tile will affect this packing)

        conv8 = node.Conv2DOptions
        _, kmaps, kh, kw = conv8.filter_shape_dims
        filter_data = filter_data.squeeze(axis=0)

        # Pad the kernel for the Toeplitz construction, which is how we will DMA / shape weights in practice
        # Given a weight map of size kh*kw, for each column in the weight map (size kh), we will pad each column to be HxW == (kh+PARALLEL_MAPS-1)*(PARALLEL_MAPS).
        # For a 3x3 kernel map, this means it will be padded to 288x3 (assuming PARALLEL_MAPS==16)
        # e.g. starting map = [[1 2 3]]
        #                     [[4 5 6]]
        #                     [[7 8 9]]

        col_h_out = kh + parallel_output_maps - 1
        col_w_out = parallel_output_maps

        # initialize the row of zeros and the padded_chunk
        # only the first two chunks of weights will need to do this extra DMA
        # for non-sparse, the extra DMA targets the 2nd weight bank and will fill with 0
        # for sparse, the extra DMA will additionally target the 3rd weight bank (control bank) and will fill with 64
        zero = np.zeros(col_w_out)
        filler_chunk = np.zeros(tile[IMAPS] * col_h_out * col_w_out * kw)
        padded_kmaps = kmaps + (2*tile[IMAPS])
        if tile[IMAPS] == kmaps: # the tile fits all imaps/maps, so only one extra chunk is needed (additional DMAs will be dummies)
            padded_kmaps = kmaps + tile[IMAPS]
        if sparse==1:
            chunk_of_zeros = np.zeros(tile[IMAPS] * col_h_out * col_w_out * kw)
            filler_chunk = np.concatenate((chunk_of_zeros, np.full((tile[IMAPS] * col_h_out * col_w_out * kw), 64)))
            padded_kmaps = kmaps + (4*tile[IMAPS])
            if tile[IMAPS] == kmaps: # the tile fits all imaps/maps, so only one extra chunk is needed (additional DMAs will be dummies)
                padded_kmaps = kmaps + (2*tile[IMAPS])
        padded_weights = np.zeros(np.prod((padded_kmaps, col_h_out*col_w_out, kw)), dtype=np.int8)
        col_chunk_size = col_h_out*col_w_out

        kmap = 0
        filter_data_kmap = 0
        chunks_padded = 0
        while kmap < padded_kmaps:
            # create a Toeplitz matrix for each column of the map
            for col in range(kw):
                # e.g. first column == [1, 4, 7, 0, 0, ..., 0]
                c = np.zeros(col_h_out)
                c[:kh] = filter_data[filter_data_kmap, :, col]

                # Toeplitz matrix (essentially our padded column), should be of shape (kh+PARALLEL_MAPS-1)*(PARALLEL_MAPS)
                # e.g. for first column, before flattening:
                #       [[1 0 0 0 ... 0 0]]
                #       [[4 1 0 0 ... 0 0]]
                #       [[7 4 1 0 ... 0 0]]
                #       [[0 7 4 1 ... 0 0]]
                #                ...
                #       [[0 0 0 0 ... 7 4]]
                #       [[0 0 0 0 ... 0 7]]
                padded_weights[((kmap*kw)+col)*col_chunk_size:((kmap*kw)+col+1)*col_chunk_size] = toeplitz(c, zero).flatten()
            kmap += 1
            filter_data_kmap += 1

            # pad the weights with the filler chunk for the first two chunks (buffers, tiles etc)
            if (filter_data_kmap % tile[IMAPS] == 0 or filter_data_kmap % tile[MAPS] == 0) and chunks_padded < 2:
                if sparse==1:
                    padded_weights[(kmap*kw*col_chunk_size):((kmap+(2*tile[IMAPS]))*kw*col_chunk_size)] = filler_chunk
                    kmap += (2*tile[IMAPS])
                else:
                    padded_weights[(kmap*kw*col_chunk_size):((kmap+tile[IMAPS])*kw*col_chunk_size)] = filler_chunk
                    kmap += tile[IMAPS]
                chunks_padded += 1

        filter_data = padded_weights

    else:
        conv8 = node.Conv2DOptions
        kmaps, kc, kh, kw = conv8.filter_shape_dims
        if not conv8.use_depthwise:
            if (sparse==1) and kc % 8 != 0:
                # pad channels up to multiple of 8 for compression
                channel_pad = (8-(kc % 8))
                filter_data = np.pad(filter_data, pad_width=((0,0), (0,channel_pad), (0,0), (0,0)), mode='constant', constant_values=0)                    
            elif kc % 2 != 0: 
                # only need one channel padded if uneven total # channels 
                # (allocation should guarantee even channels per general tile)
                filter_data = np.pad(filter_data, pad_width=((0,0), (0,1), (0,0), (0,0)), mode='constant', constant_values=0)

            # pad final kernel tile if applicable, must be parallel_kernels or a multiple of it, allocation should already guarantee this per general tile
            if (kmaps % parallel_output_maps) != 0:
                padding = parallel_output_maps - (kmaps % parallel_output_maps)
                filter_data = np.pad(filter_data, pad_width=((0,padding), (0,0), (0,0), (0,0)), mode='constant', constant_values=0)
                if node.maps == kmaps: # tile is doing all omaps
                    repack_maps = filter_data.shape[0]

            # k,c,h,w -> k,h,w,c -> compress -> re-tile if needed -> k,c,h,w -> c,h,w,k
            if (sparse == 1):
                retiled = False                    
                filter_data_khwc = filter_data.transpose((0,2,3,1))
                weights_comp, ctrl_comp, R = compress_weights(filter_data_khwc)
                conv8.repeat = R

                # re-tile if conv8.repeat is not 4 (weights became compressed)
                if conv8.repeat != 4:
                    tile = allocation(node, preset, opcode, sparse=sparse, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)
                    retiled = True

                if conv8.fit_weights == 1 and ALL_WEIGHTS and (np.prod(weights_comp.shape) + np.prod(ctrl_comp.shape)) > (fia_weight_shaper_size_kb(parallel_output_maps)*1024):
                    conv8.fit_weights = 0
                    conv8.split_weight_shaper_buffers = 1
                    tile = allocation(node, preset, opcode, sparse=sparse, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)
                    retiled = True

                if retiled:
                    repack_maps = node.maps
                    if node.maps == kmaps: # tile is doing all omaps
                        repack_maps = filter_data.shape[0] # this is the padded kernel value

                filter_data = pack_compressed_weights(weights_comp, ctrl_comp, R, tile[IMAPS], repack_maps, conv8.fit_weights)
            else:
                if conv8.fit_weights == 1 and ALL_WEIGHTS and np.prod(filter_data.shape) > (fia_weight_shaper_size_kb(parallel_output_maps)*1024):
                    conv8.fit_weights = 0
                    conv8.split_weight_shaper_buffers = 1
                    tile = allocation(node, preset, opcode, sparse=sparse, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)
                    repack_maps = node.maps
                    if node.maps == kmaps: # tile is doing all omaps
                        repack_maps = filter_data.shape[0] # this is the padded kernel value

                if is_transpose:
                    filter_data = conv_pack_weights(filter_data, repack_maps, tile[IMAPS], is_transpose=True)
                else:
                    filter_data = conv_pack_weights(filter_data, repack_maps, tile[IMAPS])

    return filter_data

@contextmanager
def exception_catcher( node, node_index, tensor_index=None, subnode=None, subnode_index=None):
    try:
        yield
    except AssertionError as e:
        print(e)
        print("Error in VNNX node index {} type {}".format(node_index, node.name))
        if tensor_index is not None:
            print("Tensor index {}".format(tensor_index))
        if subnode_index is not None and subnode is not None:
            print("Subnode index {} type {}".format(subnode_index, subnode))
        sys.exit(1)

def resize_nearest_indices(in_size, out_size):
    in_rows, in_cols = in_size
    out_rows, out_cols = out_size
    c_idx, r_idx = [],[]

    for c in range(out_cols):
        srcIndex = c*in_cols // out_cols
        if srcIndex >= in_cols:
            srcIndex = in_cols - 1
        c_idx.append(srcIndex)

    c_inc = [c_idx.index(_) for _ in set(c_idx)]
    assert(len(c_inc) == in_cols)
    c_incr = [c_idx[c_inc[i+1]] - c_idx[c_inc[i]] for i in range(len(c_inc)-1)]
    c_num_copies = [c_idx.count(_) for _ in set(c_idx)]
    c_all_same_incr = all([c_incr[0] == d for d in c_incr])

    # print('NEAREST NEIGHBOR')
    # print('cols', c_incr[0], (c_num_copies[0], c_num_copies[-1]), c_all_same_incr)
    return c_num_copies, c_inc   

#rows and columns indices
def resize_bilinear_indices(in_size, out_size):
    in_rows, in_cols = in_size
    out_rows, out_cols = out_size
    c_idx, steps, steps1 = [],[], []
    c1_idx = []
    ratio_cols = int(((1 << 10) * in_cols + out_cols / 2) / out_cols)
    for c in range(out_cols):
        step = ratio_cols * c
        row0_frac = step & ((1 << 10) - 1)
        row1_frac = (1 << 10) - row0_frac
        steps.append(row0_frac)
        steps1.append(row1_frac)
        pos =  max(step / (1 << 10), 0)
        pos1 = min((step + (1 << 10) - 1) / (1 << 10), in_cols - 1)

        if pos >= in_cols-1:
           pos = in_cols - 1
        c_idx.append(int(pos))
        c1_idx.append(int(pos1))

    c_inc = [c_idx.index(_) for _ in set(c_idx)]
    step_inc = [c_idx.index(_) for _ in set(c_idx)]
    c1_inc = [c1_idx.index(_) for _ in set(c1_idx)]
    
    if in_cols < out_cols:
        assert(len(c_inc) == in_cols)
    c_incr = [c_idx[c_inc[i+1]] - c_idx[c_inc[i]] for i in range(len(c_inc)-1)]
    c_num_copies = [c_idx.count(_) for _ in set(c_idx)]
    steps_num_copies = [steps.count(_) for _ in set(steps)]
    c1_num_copies = [c1_idx.count(_) for _ in set(c1_idx)]
  
    # print("steps = ", steps)   
    # print("steps1 = ", steps1)    
    # print("c_idx = ", c_idx) 
    # print("c1_idx = ", c1_idx) 
    # print("c_inc = ", c_inc) 
    # print("c1_inc = ", c1_inc) 
    # print("c_num_copies = ", c_num_copies)
    # print("num steps = ", len(steps))

    # print('BILINEAR ')
    # print('cols', c_incr[0], (c_num_copies[0], c_num_copies[-1]), c_all_same_incr)
    return c_num_copies, c_inc, steps    


def sigmoid(x):
  if x > 0:   
    z = np.exp(-x)
    return 1/(1+z)
  else:
    z = np.exp(x)
    return z/(1+z)


def allocation(node, preset, opcode, debug=False, sparse=None, tmp_dir=None, graph_idx=None, tmp_dir_obj=None):
    tile = None
    return tile_subgraph(node, preset, opcode, sparse, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)


def get_subop_parameters(subop, tensors, buffers):
    i_tensor = tensors[subop['inputs'][0]]
    f_tensor = tensors[subop['inputs'][1]]
    if 'buffer' in i_tensor and 'data' in buffers[i_tensor['buffer']]:
        i_tensor = tensors[subop['inputs'][1]]
        f_tensor = tensors[subop['inputs'][0]]
    filter_data = get_numpy_data(f_tensor, buffers)
    
    filter_offset = f_tensor['quantization']['zero_point'][0]
    input_offset = i_tensor['quantization']['zero_point'][0]

    input_scale = i_tensor['quantization']['scale']
    filter_scale = f_tensor['quantization']['scale']
            
    k, h, w, c = 1, 1, 1, 1
    if len(f_tensor['shape']) == 1:
        c = tuple(f_tensor['shape'])[0]
    elif len(f_tensor['shape']) == 2:
        k, c = tuple(f_tensor['shape'])
    elif len(f_tensor['shape']) == 3:
        h, w, c = tuple(f_tensor['shape'])
        filter_data = filter_data.transpose((2, 0, 1))
    elif len(f_tensor['shape']) == 4:
        k, h, w, c = tuple(f_tensor['shape'])
        filter_data = filter_data.transpose((0, 3, 1, 2))
    elif len(f_tensor['shape']) == 5:
        assert (f_tensor['shape'][0] == 1)
        k, h, w, c = f_tensor['shape'][1:]
        filter_data = filter_data.transpose((0, 1, 4, 2, 3))
    filter_shape_dims = [k, c, h, w]

    filter_data = filter_data.flatten()
    
    return i_tensor, f_tensor, input_scale[0], filter_scale, input_offset, filter_offset, filter_shape_dims, filter_data


# Use the same LUT generation code for both uint8_t and int8_t. Int8_t indexes
# will be directly casted to uint8_t, the int8 LUT will thus be ordered as [0,
# 1, ..., 127, -128, ..., -2, -1] instead of [-128, -127, ..., -1, 0, 1, ...,
# 126, 127].
def LUTPopulateInt8(input_scale, input_zero_point, output_scale, output_zero_point, transform, transform_params=None, itype=calc_type.INT8, otype=calc_type.INT8):
    lut_uint8 = np.zeros((256,), dtype=np.uint8)
    inverse_scale = 1. / output_scale

    max_idx, min_idx = 127, -128
    if itype==calc_type.UINT8:
        max_idx, min_idx = 255, 0

    max_out, min_out = 127, -128
    if otype==calc_type.UINT8:
        max_out, min_out = 255, 0
    
        
    for idx in range(min_idx, max_idx+1):
        dequantized = input_scale * (idx - input_zero_point)
        transformed = LUTTransform(transform, dequantized, transform_params)
        quantized = -128
        
        if otype==calc_type.UINT8:
            quantized = 0

        if not tf.math.is_nan(transformed) and not tf.math.is_inf(transformed):
            rescaled = TfLiteRound(transformed * inverse_scale)
            quantized = (rescaled + output_zero_point)
        lut_uint8[np.int32(idx).astype(np.uint8)] = np.int32(max(min(max_out, quantized), min_out)).astype(np.uint8)

    vals = {}
    step_vals = [] 
    step_indices = [] 
    prev_val = None
    for idx in range(min_idx,max_idx+1):
        val = lut_uint8[np.int32(idx).astype(np.uint8)]
        if prev_val is None or val != prev_val:
            step_vals.append(np.int8(val))
            step_indices.append(idx)
        if val in vals:
            vals[val].append(idx)
        else:
            vals[val] = [idx]
        prev_val = val

    min_val = [vals[k] for k in vals if min_idx in vals[k]][0]
    if len(min_val) == 1:
        first_unique_idx = min_idx
    else:
        for i in range(min_idx+1, max_idx):
            if not i in min_val:
                first_unique_idx = i
                break

    max_val = [vals[k] for k in vals if max_idx in vals[k]][0]
    if len(max_val) == 1:
        last_unique_idx = max_idx
    else:
        for i in range(max_idx-1, min_idx-1, -1):
            if not i in max_val:
                last_unique_idx = i
                break

    first_unique_idx = min_idx
    last_unique_idx = max_idx

    return lut_uint8, first_unique_idx, last_unique_idx, step_vals, step_indices


def LUTPopulate(input_scale, input_zero_point, output_scale, output_zero_point, transform, transform_params=None, bytes=4, itype=calc_type.INT8):
    lut_dtype = np.uint8
    max_val, min_val = 127, -128

    if bytes == 4:
        lut_dtype = np.uint32
        max_val, min_val = 2**31-1, -2**31

    lut_uint = np.zeros((256,), dtype=lut_dtype)
    inverse_scale = 1. / output_scale

    max_idx, min_idx = 127, -128
    if itype==calc_type.UINT8:
        max_idx, min_idx = 255, 0

    for idx in range(min_idx, max_idx+1):
        dequantized = input_scale * (idx - input_zero_point)
        transformed = LUTTransform(transform, dequantized, transform_params)
        quantized = min_val
        if not tf.math.is_nan(transformed) and not tf.math.is_inf(transformed):
            rescaled = TfLiteRound(transformed * inverse_scale)
            quantized = (rescaled + output_zero_point)
        lut_uint[np.int32(idx).astype(np.uint8)] = np.int32(max(min(max_val, quantized), min_val)).astype(lut_dtype)

    vals = {}
    step_vals = [] 
    step_indices = [] 
    prev_val = None
    for idx in range(min_idx,max_idx+1):
        val = lut_uint[np.int32(idx).astype(np.uint8)]
        if prev_val is None or val != prev_val:
            if bytes == 4:
                step_vals.append(np.int32(val))
            else:
                step_vals.append(np.int8(val))
            step_indices.append(idx)
        if val in vals:
            vals[val].append(idx)
        else:
            vals[val] = [idx]
        prev_val = val

    min_val = [vals[k] for k in vals if min_idx in vals[k]][0]
    if len(min_val) == 1:
        first_unique_idx = min_idx
    else:
        for i in range(min_idx+1, max_idx):
            if not i in min_val:
                first_unique_idx = i
                break

    max_val = [vals[k] for k in vals if max_idx in vals[k]][0]
    if len(max_val) == 1:
        last_unique_idx = max_idx
    else:
        for i in range(max_idx-1, min_idx-1, -1):
            if not i in max_val:
                last_unique_idx = i
                break

    first_unique_idx = min_idx
    last_unique_idx = max_idx

    return lut_uint, first_unique_idx, last_unique_idx, step_vals, step_indices




def LUTTransform(transform, dequantized, transform_params=None):
    return transform(dequantized)


def pad_list(x, len_pad, value=1):
    l = list(x)
    if len(l) < len_pad:
        l = [value for _ in range(len_pad-len(l))] + l
    return l


def conv_pack_shift(data, maps):
    kernels = len(data)
    flat = np.asarray(data, dtype=np.int32)
    packed = np.zeros(flat.shape, dtype=flat.dtype)

    for k in range(ceil(kernels/maps)):
        bytes = flat[k*maps:(k+1)*maps].astype(np.int8)
        zeros = np.zeros(bytes.shape, dtype=bytes.dtype) 
        words = np.concatenate((bytes,zeros,zeros,zeros), dtype=np.int8)
        packed[k*maps:(k+1)*maps] = np.frombuffer(words.tobytes(), dtype=np.int32)

    return packed.tolist()


def conv_pack_biases(data, maps):
    kernels = len(data)
    flat = np.asarray(data, dtype=np.int32)
    packed = np.zeros(flat.shape, dtype=flat.dtype)

    for k in range(ceil(kernels/maps)):
        bytes = np.frombuffer(flat[k*maps:(k+1)*maps].flatten().tobytes(), dtype=np.uint8)
        shuffled = bytes.reshape((-1,4)).transpose((1,0)).flatten()
        packed[k*maps:(k+1)*maps] = np.frombuffer(shuffled.tobytes(), dtype=np.int32)

    return packed.tolist()


def conv_pack_weights(data, maps, tile_channels, use_depthwise=False, is_transpose=False, weight_pad=0):
    # k,c,h,w -> c,h,w,k
    s = data.shape
    t = (1,2,3,0)
    kernels = s[0]
    channels = s[1]
    kernel_height = s[2]
    kernel_width = s[3]
    chunk_size = s[1]*s[2]*s[3]

    if s[0] == 1 and not is_transpose:
        kernels = s[1]
        channels = 1
        chunk_size = s[2]*s[3]
        data = data.squeeze(axis=0)
        t = (1,2,0)
        if use_depthwise:
            data = np.flip(data, axis=1) # along height axis
            t = (0,2,1)
            new_data_shape = (1, kernels, kernel_width, kernel_height+(2*weight_pad))
            chunk_size = new_data_shape[2] * new_data_shape[3]
            data = np.pad(data, ((0,0), (weight_pad, weight_pad), (0,0)), mode='constant', constant_values=0)

    # kernel_chunks = ceil(kernels/maps)
    # channel_chunks = ceil(channels / tile_channels)
    # tile_chunk_size = maps * tile_channels * kernel_height * kernel_width
    # flat = np.zeros(data.flatten().shape, dtype=data.dtype)
    # flat_start = 0
    # flat_end = 0
    # for k in range(kernel_chunks):
    #     final_tile_maps = maps
    #     if k == kernel_chunks - 1: # adjust for final tile maps which may be different
    #         final_tile_maps = (k % maps) if (k % maps) else maps
    #         tile_chunk_size = (final_tile_maps * tile_channels * kernel_height * kernel_width)
    #     for ch in range(channel_chunks): # adjust for final tile channels which may be different
    #         if ch == channel_chunks - 1:
    #             final_tile_channels = (channels % tile_channels) if (channels % tile_channels) else tile_channels
    #             tile_chunk_size = (final_tile_maps * final_tile_channels * kernel_height * kernel_width)
    #         chunk = data[k*maps:(k+1)*maps, ch*tile_channels:(ch+1)*tile_channels,:,:].transpose(t)
    #         flat_end += tile_chunk_size
    #         flat[flat_start:flat_end] = chunk.flatten()
    #         flat_start = flat_end
            
    flat = np.zeros(data.flatten().shape, dtype=data.dtype)
    for k in range(ceil(kernels/maps)):
        chunk = data[k*maps:(k+1)*maps].transpose(t)
        flat[k*chunk_size*maps:(k+1)*chunk_size*maps] = chunk.flatten()

    return flat.reshape(s) if not use_depthwise else flat.reshape(new_data_shape)

def quantize_two_math_block(output_offset, output_activation_min, output_activation_max, op=None, tensors=None, i_tensor={}, o_tensor={}, f_tensor={}, scale=None, bias_data=None):

    assert((op is not None and tensors is not None) or \
           (i_tensor is not None and o_tensor is not None and f_tensor is not None))

    if op is not None and tensors is not None:
        i_tensor = tensors[op['inputs'][0]]
        o_tensor = tensors[op['outputs'][0]]
        if scale is None:
            f_tensor = tensors[op['inputs'][1]]

    input_scale = i_tensor['quantization']['scale']
    output_scale = o_tensor['quantization']['scale']
    if scale is None:
        scale = f_tensor['quantization']['scale']

    output_multiplier = []
    output_shift = []
    c_input_L = []
    c_input_H = []
    effective_output_scale = [input_scale[0] * f / output_scale[0] for f in scale]
    
    OM_BITS = 16     # compressed output multiplier number of bits (unsigned)
    O_SHIFT_RANGE = [8,16,24,32]     # possible output shift

    int30min = -2**29
    int30max = 2**29-1
    int48min = -2**47
    int48max = 2**47-1

    def saturate(x,minRange,maxRange):
        sat = x<minRange or x>maxRange
        x = max(min(x,maxRange),minRange)
        return x,sat

    # TODO this assertion run at runtime instead, but should be doable at compile tile (but only after we have the tile)
    # assert(acc <= int30max and acc >= int30min) 

    if bias_data is None:
        bias_data = np.zeros((len(scale),), dtype=np.int64)

    if len(effective_output_scale) < len(bias_data) and len(effective_output_scale) == 1: # FullyConnected
        effective_output_scale = effective_output_scale * len(bias_data)

    for bias, scale in zip(bias_data, effective_output_scale):
    # COMPILER ##################
        max_val = (int30max+bias)*scale+output_offset
        min_val = (int30min+bias)*scale+output_offset
        expected_saturation = (max_val<output_activation_min and min_val<output_activation_min) or (max_val>output_activation_max and min_val>output_activation_max)
        if expected_saturation:
            scale = 1   # handles cases with large bias that will always cause overflow
        if scale==0:
            o_shift = O_SHIFT_RANGE[0]
            m_shift = -o_shift
            m_shift_min = -o_shift
        else:
            om = scale
            m_shift_min = 0
            while round(2.0*om) <= 2**OM_BITS-1:
                om *= 2.0
                m_shift_min -= 1
            while round(om) > 2**OM_BITS-1:
                om /= 2.0
                m_shift_min += 1
        while 1:
            indices = np.where(np.array(O_SHIFT_RANGE) <= -m_shift_min)[0]  # o_shift = -m_shift
            ind = 0
            if len(indices)>0:
                ind = indices[-1]
            o_shift = O_SHIFT_RANGE[ind]
            m_shift = -o_shift

            om, om_sat = saturate(int(round(scale * 2.0**-m_shift)),0,2**(OM_BITS)-1)
            if om_sat: assert(m_shift<m_shift_min)
            else: assert(m_shift>=m_shift_min)
            
            bias_premult = bias*om # using the floating-point scale here actually has a less accurate result in many cases
            offset = output_offset<<o_shift
            round_bit = 1<<(o_shift-1)
            _,m_sat_high = saturate(bias_premult+offset+round_bit+int30max*om, int48min, int48max)
            _,m_sat_low = saturate(bias_premult+offset+round_bit+int30min*om, int48min, int48max)
            cInput,c_sat = saturate(bias_premult+offset+round_bit, int48min, int48max)
            cInputL = cInput%65536              # 16 bits
            cInputH = (cInput-cInputL)>>16      # 32 bits
            cInputL = 64*floor(cInputL/64.0)    # round to zero lowest 6 bits, which has a negligible effect on precision
            if m_sat_high or m_sat_low or c_sat:
                m_shift_min+=1
            else:
                break

        output_multiplier.append(om)
        output_shift.append(ind)
        c_input_L.append(cInputL)
        c_input_H.append(cInputH)

    return effective_output_scale, output_multiplier, output_shift, c_input_L, c_input_H


def quantize_multiplier(double_multiplier, TFLITE_SINGLE_ROUNDING=False):

    if TFLITE_SINGLE_ROUNDING:
        pass
    assert(double_multiplier >= 0), "Double multiplier of quantize_multiplier is not greater or equal to 0"
    if double_multiplier == 0.:
        return 0, 0 
    q, shift = frexp(double_multiplier)
    q_fixed = np.int64(TfLiteRound(q * (2**QMULT))) # QMULT == 31
    if QTRUNCATE > 0:
        q_fixed = np.int64(TfLiteRound(q * (2**(QMULT-QTRUNCATE))))

    assert(q_fixed <= (2**QMULT)), "q_fixed in quantize_multiplier is not less or equal to 2^QMULT"
    if q_fixed == (2**QMULT):
        q_fixed /= 2
        shift += 1
    assert(q_fixed <= np.iinfo(np.int32).max), "q_fized in quantize_multiplier is not less or equal to numpy int32 maximum"

    if shift < -QMULT:
        shift = 0
        q_fixed = 0
    if TFLITE_SINGLE_ROUNDING:
        if shift > QMULT-1:
            shift = QMULT-1 
            q_fixed = (2**QMULT) - 1

    quantized_multiplier = np.int32(q_fixed)
    return quantized_multiplier, shift 



def get_quantized_multiplier(scale):
    multiplier = []
    shift = []
    for s in scale:
        significand, channel_shift = quantize_multiplier(s)
        multiplier.append(significand)
        shift.append(channel_shift)
        
    return multiplier, shift


def get_quantized_multiplier_from_tensor(tensor):
    scale = tensor['quantization']['scale']
    multiplier, shift = get_quantized_multiplier(scale)
    return scale, multiplier, shift



def get_effective_quantized_multiplier_from_tensors(i_tensor, o_tensor, f_tensor, filter_scale=None):

    input_scale = i_tensor['quantization']['scale']
    output_scale = o_tensor['quantization']['scale']
    if filter_scale is None:
        filter_scale = f_tensor['quantization']['scale']

    output_multiplier = []
    output_shift = []
    effective_output_scale = [input_scale[0] * f / output_scale[0] for f in filter_scale]
    output_multiplier, output_shift = get_quantized_multiplier(effective_output_scale)

    return effective_output_scale, output_multiplier, output_shift


def get_effective_quantized_multiplier_from_op(op, tensors, filter_scale=None):

    i_tensor = tensors[op['inputs'][0]]
    o_tensor = tensors[op['outputs'][0]]
    if filter_scale is None:
        f_tensor = tensors[op['inputs'][1]]
    else:
        f_tensor = None
    return get_effective_quantized_multiplier_from_tensors(i_tensor, o_tensor, f_tensor, filter_scale)


# def MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift):
#     total_shift = QMULT - shift
#     round = 2 ** (total_shift - 1)
#     result = int(x * quantized_multiplier + round)
#     result = result // (2**total_shift)

#     return int(result)


# def DivideByQuantizedMultiplier(x, quantized_multiplier, shift):
#     total_shift = QMULT - shift
#     round = 2 ** (total_shift - 1)

#     result = int(x * (2**total_shift))
#     result = (result + round) // quantized_multiplier

#     return int(result)


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return floor(n*multiplier + 0.5) / multiplier


def TfLiteRound(n, decimals=0):
    rounded_abs = round_half_up(abs(n), decimals)
    return copysign(rounded_abs, n)


def CalculateInputRadius(input_integer_bits, input_left_shift, total_signed_bits):
    return [((2 ** input_integer_bits) - 1) * (2**(total_signed_bits - input_integer_bits)) // (2**i) for i in input_left_shift]
    # else:
    #     max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) * (1ll << (total_signed_bits - input_integer_bits)) / (1ll << input_left_shift)
    #     return floor(max_input_rescaled)


# def get_quantized_multiplier_gemmlowp(scale):

#     assert(scale > 0)
#     assert(scale < 1)
#     s = 0
#     while(scale < 0.5):
#         scale *= 2.
#         s += 1
#     q = TfLiteRound(scale * (2**31))
#     assert(q <= (2**31))
#     if q == (2**31):
#         q /= 2
#         s -= 1
#     assert(s >= 0)
#     assert(q <= np.iinfo(np.int32).max)
#     return q,s


def adjust_transform(transform, dims):
    s = list(transform)
    while 1:
        if s[0] == 0:
            s = s[1:] - 1
        else:
            break

    if dims > len(s):
        outer = dims - len(s)
        transform = tuple([_ for _ in range(outer)] + [outer + _ for _ in s])

    return transform


def flatten_n(ishape, oshape):
    idim = len(ishape)
    odim = len(oshape)

    if idim - odim > 0:
        flatn = 1 + idim - odim
        if oshape[-1] == np.prod(ishape[-flatn:]):
            return flatn
    return 0


def check_unpack(ishape, oshape, axis, c_axis, count):

    iarr = np.random.randint(0, 10, ishape)
    c_iarr = channels_first_array(iarr)

    oarr = tf.unstack(iarr, axis=axis)[0]
    c_oarr = channels_first_array(oarr)

    if c_axis == -1:
        if len(ishape) >= 4: #axis = -1    NHWC -> NHC   NCHW->CNH  single col then 1,0,2
            tmp = tf.unstack(c_iarr, axis=-1)[0]
            tmp = np.transpose(tmp, adjust_transform([1,0,2], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 3: #axis = -1 HWC -> HC   CHW->HC  single col then 0,2,1
            tmp = tf.unstack(c_iarr, axis=-1)[0]
            tmp = np.transpose(tmp, adjust_transform([1,0], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 2: #axis = -1 HW -> H   HW->H  single col
            tmp = tf.unstack(c_iarr, axis=-1)[0]
            assert(np.all(tmp == c_oarr))
            return True
    elif c_axis == -2:
        if len(ishape) >= 4: #axis = -2    NHWC -> NWC   NCHW->CNW  single row, then 1,0,2
            tmp = tf.unstack(c_iarr, axis=-2)[0]
            tmp = np.transpose(tmp, adjust_transform([1,0,2], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 3: #axis = -2  HWC -> WC   CHW->WC  single row, then 0,2,1
            tmp = tf.unstack(c_iarr, axis=-2)[0]
            tmp = np.transpose(tmp, adjust_transform([1,0], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 2: #axis = -2 HW -> W   HW->W  single row
            tmp = tf.unstack(c_iarr, axis=-2)[0]
            assert(np.all(tmp == c_oarr))
            return True
    elif c_axis == -3: #axis = -3    NHWC -> NHW   NCHW->WNH single channel then 2,0,1
        if len(ishape) >= 4:
            tmp = tf.unstack(c_iarr, axis=-3)[0]
            tmp = np.transpose(tmp, adjust_transform([2,0,1], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 3: #axis = -3  HWC -> HW   CHW->HW  single channel
            tmp = tf.unstack(c_iarr, axis=-3)[0]
            assert(np.all(tmp == c_oarr))
            return True
    elif c_axis == -4: #axis = -4    NHWC -> HWC   NCHW->CHW  single batch
        tmp = tf.unstack(c_iarr, axis=-4)[0]
        assert(np.all(tmp == c_oarr))
        return True

    return False

def check_pack(ishape, oshape, axis, c_axis, count):

    iarrs = [np.random.randint(0, 10, ishape) for _ in range(count)]
    oarr = np.stack(iarrs, axis=axis)
    assert(tuple(oshape) == oarr.shape)

    c_iarrs = [channels_first_array(_) for _ in iarrs]
    c_oarr = channels_first_array(oarr)

    if c_axis == -1:
        if len(ishape) >= 3: #axis = -1    HWC -> HWxC   CHW->HCWx  1,0,2 then multi-col
            tmp = [np.transpose(_, adjust_transform([1,0,2], _.ndim)) for _ in c_iarrs]
            tmp = np.stack(tmp, axis=-1)
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 2: #axis = -1    HW -> HWx   HW->xHW  multi-maps
            pass
        elif len(ishape) == 1: #axis = -1    W -> Wx   W->Wx  multi-col?
            pass
    elif c_axis == -2:
        if len(ishape) >= 3: #axis = -2    HWC -> HxWC   CHW->HCxW  1,0,2 then multi-row
            tmp = [np.transpose(_, adjust_transform([1,0,2], _.ndim)) for _ in c_iarrs]
            tmp = np.stack(tmp, axis=-2)
            assert(np.all(tmp == c_oarr))
            return True
        elif len(ishape) == 2: #axis = -2    HW -> HxW   HW->WxH  1,0 then multi-row
            pass
    elif c_axis == -3: #axis = -3    HWC -> HWCx   CHW->HxWC 1,2,0 then multi-map
        tmp = [np.transpose(_, adjust_transform([1,2,0], _.ndim)) for _ in c_iarrs]
        tmp = np.stack(tmp, axis=-3)
        assert(np.all(tmp == c_oarr))
        return True
    elif c_axis == -4: #axis = -4    NHWC -> XNHWC   NCHW->NxCHW  multi-batch
        tmp = np.stack(c_iarrs, axis=-4)
        assert(np.all(tmp == c_oarr))
        return True

    return False


def channels_first_array_reshape(ishape, transform):
    if tuple(ishape) ==  transform:
        return 0

    iarr = np.random.randint(0, 10, ishape)
    oarr = np.reshape(iarr, transform)
    oshape = oarr.shape

    c_iarr = channels_first_array(iarr)
    c_oarr = channels_first_array(oarr)
    cishape = c_iarr.shape
    coshape = c_oarr.shape

    dims = len(transform)
    idims = len(ishape)
    odims = len(oshape)

    mode = -1
    ones = lambda x: len([_ for _ in x if _ == 1])
    squeezed_shape = lambda sh, ni: tuple([_ for i,_ in enumerate(sh) if i != len(sh) + ni])


    squeeze = False
    if ones(ishape) - 1 == ones(oshape):
        squeeze = True

        squeeze_axis = None
        for i in range(-1, -(len(ishape)+1), -1):
            if squeezed_shape(ishape, i) == tuple(oshape):
                squeeze_axis = i
                break

    if squeeze:
        assert(squeeze_axis)
        if squeeze_axis < -4:
            print('squeeze axis < -4 not supported')
            mode = -1
        elif squeeze_axis == -1: # n,h,w aka w,n,h  2,0,1
            tmp = np.squeeze(c_iarr, axis=-3)
            tmp = np.transpose(tmp, adjust_transform([2,0,1], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            mode = 4 + 1
        elif squeeze_axis == -2: # n,h,c aka c,n,h  1,0,2
            tmp = np.squeeze(c_iarr, axis=-1)
            tmp = np.transpose(tmp, adjust_transform([1,0,2], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            mode = 4 + 2
        elif squeeze_axis == -3: # n,w,c aka c,n,w  1,0,2
            tmp = np.squeeze(c_iarr, axis=-2)
            tmp = np.transpose(tmp, adjust_transform([1,0,2], tmp.ndim))
            assert(np.all(tmp == c_oarr))
            mode = 4 + 3
        elif squeeze_axis == -4:  #1,h,w,c -> c,h,w NOP
            mode = 0
        else:
            print('squeeze unknown axis ', squeeze_axis) 
        return mode

    expand = False
    if ones(ishape) + 1 == ones(oshape):
        expand = True
        expand_axis = None

        for i in range(-1, -(len(oshape)+1), -1):
            if squeezed_shape(oshape, i) == tuple(ishape):
                expand_axis = i
                break
    if expand:
        assert(expand_axis)
        if expand_axis < -3:
            print('expand axis < -3 not supported')
            mode = -1
        elif expand_axis == -1 and len(ishape) == 3: # n,h,w aka n,h,w,1  
            mode = 0
        elif expand_axis == -1: # n,h,w,c,1 aka n,1,h,w,c  
            tmp = np.transpose(c_iarr, adjust_transform([1,2,0], c_iarr.ndim))
            tmp = np.expand_dims(tmp, axis=-3)
            assert(np.all(tmp == c_oarr))
            mode = 8 + 1
        elif expand_axis == -2: # n,h,w,1,c aka n,h,c,w,1
            tmp = np.transpose(c_iarr, adjust_transform([1,0,2], c_iarr.ndim))
            tmp = np.expand_dims(tmp, axis=-1)
            assert(np.all(tmp == c_oarr))
            mode = 8 + 2
        elif expand_axis == -3: # n,h,1,w,c aka n,h,c,1,w
            tmp = np.transpose(c_iarr, adjust_transform([1,0,2], c_iarr.ndim))
            tmp = np.expand_dims(tmp, axis=-2)
            assert(np.all(tmp == c_oarr))
            mode = 8 + 3
        else:
            print('expand unknown axis ', expand_axis) 
        return mode

    join = False
    if len(ishape) - 1 == len(oshape):
        join = True
        join_axis = None
        for i in range(1, len(ishape)):
            if ishape[-i]*ishape[-i-1] == oshape[-i]:
                join_axis = -i
                break

    if join:
        assert(join_axis)
        if join_axis < -3:
            print('join axis < -3 not supported')
            mode = -1
        elif join_axis == -1: # n,h,w*c aka w*c,n,h
            mode = 12 + 1
            tmp = np.transpose(c_iarr, adjust_transform([2,0,1], c_iarr.ndim)) #2,0,1   n,c,h,w -> n,w,c,h
            tshape = list(tmp.shape)[:-3] + [tmp.shape[-3]*tmp.shape[-2]] + [tmp.shape[-1]]
            tmp = np.reshape(tmp, tshape) # n,w*c,h
            tmp = np.transpose(tmp, adjust_transform([1,0,2], tmp.ndim)) #1,0,2
            assert(np.all(tmp == c_oarr))
        elif join_axis == -2: # n,h*w,c aka c,n,h*w
            mode = 12 + 2
        elif join_axis == -3: # n*h,w,c aka c,n*h,w
            mode = 12 + 3
        else:
            print('join unknown axis ', join_axis) 
        return mode

    seperate = False
    if len(ishape) + 1 == len(oshape):
        seperate = True
        seperate_axis = 0
        for i in range(1, len(oshape)):
            if oshape[-i]*oshape[-i-1] == ishape[-i]:
                seperate_axis = -i
                break

    if seperate:
        if seperate_axis < -3:
            print('seperate axis < -3 not supported')
            mode = -1
        elif seperate_axis == -1: # n,h,w*c aka
            mode = 16 + 1
        elif seperate_axis == -2: # n,h*w,c aka
            mode = 16 + 2
        elif seperate_axis == -3: # n*h,w,c aka
            mode = 16 + 3
        else:
            print('seperate unknown axis ', seperate_axis) 
        return mode

    if idims == odims:
        if oshape[-3] == ishape[-1] and oshape[-2] == ishape[-3] and oshape[-1] == ishape[-2]: # swap channels first to last aka 2,0,1
            return 2


    flatten = flatten_n(ishape, oshape) # flatten to channels 
    if flatten:
        return 0

    inv_flatten = flatten_n(oshape, ishape) # expand w/ ones, keeping channels
    if inv_flatten:
        return 0

    if len(ishape) - len(oshape) > 1:
        print(tuple(ishape), "->", tuple(oshape), "cannot currently handle multi-axis squeeze")
    if len(oshape) - len(ishape) > 1:
        print(tuple(ishape), "->", tuple(oshape), "cannot currently handle multi-axis expand")

    return mode


def channels_first_array(arr):
    s = list(arr.shape)
    if len(s) >= 3:
        carr = np.transpose(arr, adjust_transform([2,0,1], arr.ndim))
    else:
        carr = np.copy(arr)
    return carr


def channels_first_axis(axis, dims):
    if axis < 0:
        axis = dims + axis

    if dims >= 3:
        if axis == dims - 1:
            axis = dims - 3
        elif axis >= dims - 3:
            axis += 1

    return axis


def get_numpy_data_from_index(index, tensors, buffers):
    tensor = tensors[index]
    # shape = tensor['shape']
    shape = tensor.get('shape', (1))
    dtype = np.dtype(tensor['type'].lower())
    raw_data = bytearray(buffers[tensor['buffer']]['data'])
    data = np.frombuffer(raw_data, dtype=dtype).reshape(shape)

    return data


def get_numpy_data(tensor, buffers):
    # shape = tensor['shape']
    shape = tensor.get('shape', (1))
    dtype = np.dtype(tensor['type'].lower())
    raw_data = bytearray(buffers[tensor['buffer']]['data'])
    data = np.frombuffer(raw_data, dtype=dtype).reshape(shape)

    return data


def op_activations(op, activations):
    input_activations =  [_ for _ in op['inputs'] if _ in activations] 
    output_activations =  [_ for _ in op['outputs'] if _ in activations] 
    return input_activations, output_activations


def op_params(op, activations):
    input_params =  [_ for _ in op['inputs'] if not (_ in activations)] 
    output_params =  [_ for _ in op['outputs'] if not (_ in activations)] 
    return input_params, output_params


def get_graph_activations(json_subgraphs):
    activations = {}
    subgraph_inputs = []
    subgraph_outputs = []

    for g, graph in enumerate(json_subgraphs):
        buffers = graph['buffers']
        subgraph = graph['subgraphs'][0]
        tensors = subgraph['tensors']
        ops = subgraph['operators']

        subgraph_inputs += subgraph['inputs']
        subgraph_outputs += subgraph['outputs']

        subgraph_activations = []
        for op in ops:
            for _ in op['inputs']:
                if 'buffer' in tensors[_]:
                    if not 'data' in buffers[tensors[_]['buffer']]:
                        subgraph_activations += [_]   

        for op in ops:
            for _ in op['outputs']:
                if 'buffer' in tensors[_]:
                    if not 'data' in buffers[tensors[_]['buffer']]:
                        subgraph_activations += [_]
                
        for a in subgraph_activations:
            if a not in activations:
                activations[a] = tensors[a]

    inputs = [_ for _ in subgraph_inputs if _ not in subgraph_outputs]
    outputs = [_ for _ in subgraph_outputs if _ not in subgraph_inputs]

    return inputs, outputs, activations

 
def precalculate_filter_input_bias(filters, input_offset, filter_offset=0, reduce=True):
    if filters.shape == (): # numpy array 0-dimensional, single element filter
        return np.array( [(np.sum(filters) * -input_offset) + (-filter_offset * -input_offset) ])
    if filters.shape[0] == 1 and reduce and len(filters.shape)!=1: #TODO cover when group > 1, but not depthwise
        filters = filters[0]

    precalc = np.zeros((filters.shape[0],),dtype=np.int32)
    for i,f in enumerate(filters):
        precalc[i] += (np.sum(f) * -input_offset) + (-filter_offset * -input_offset)
    return precalc


def precalculate_output_offset_bias(scale, offset):
    precalc = []
    for s in scale:
        if s == 0:
            x = 0
        else:
            x = 1.0 * offset / s
            round = 0.5
            if x < 0:
                round = -0.5
            x = int(x + round)
        precalc.append(x)
    return precalc


def find_tensors_by_id(tensors, id):
    x = []
    for t in tensors:
        if t.id == id:
            x.append(t)
    return x


def find_tensors_by_name(tensors, name):
    x = []
    for t in tensors:
        if t.name == name:
            x.append(t)
    return x


def get_node_and_offset(Nodes, id):
    for i,n in enumerate(Nodes):
        for j,t in enumerate(n.tensor_array):
            # if t.id == id:
            if t.name == id:
                return i, j
    print(id, 'not found')
    return None


def mod_shape(shape, x):
    if len(shape) > 1:
        w = (shape[-1] + (x-1)) // x * x
        h = (shape[-2] + (x-1)) // x * x
        return tuple(list(shape)[:-2] + [h, w])
    return shape

def update_tensor_shapes(Nodes):
    all_tensor_array = [_ for n in Nodes for _ in n.tensor_array]

    for t in all_tensor_array:
        while len(t.shape) < SHAPE_DIMS: 
            t.shape = list(t.shape) + [0]

# create a mapping of nodes, inputs, outputs to be used for pad sublayer injection
# key: input_id, value: list of output_ids that are created from the input_id
# if input_id has more than one output_id, means the input_id is taken as input to different Nodes
def tensor_in_id_to_out_id_mapping(json_subgraphs):
    in_id_to_out_ids = {}
    for graph in json_subgraphs:
        for op in graph['subgraphs'][0]['operators']:
            g_inputs = op['inputs']
            g_outputs = op['outputs']
            for in_id in g_inputs:
                current_value = in_id_to_out_ids.get(in_id, [])
                current_value += g_outputs
                in_id_to_out_ids[in_id] = current_value
    in_id_to_out_ids = {key: value for key, value in in_id_to_out_ids.items() if len(value) > 1}
    return in_id_to_out_ids

# this function is to be called after populate_nodes and used for pad sublayer injection
def final_check_pad_sublayer_injection(in_id_to_out_ids, Nodes, ids_with_dummies):
    # using the in_id_to_out_ids dict, go through Nodes again, and find the node with output id matching the input id key (source node), 
    # as well as the nodes with output id matching with the output id values (destination nodes)
    for in_id, out_ids in in_id_to_out_ids.items():
        source_node = None
        dest_nodes = []
        for node in Nodes:
            if node.tensor_array[node.num_inputs].id == in_id:
                source_node = node
            elif node.tensor_array[0].id == in_id and node.tensor_array[node.num_inputs].id in out_ids:
                dest_nodes.append(node)

        # Must confirm that the nodes that share the input tensor (destination nodes) are:
            # a) Conv nodes
            # b) check their conv8 pad attributes (shows that they use SAME padding) and check that they're exact
            # c) have the exact same pad value (input_offset)
        # if all conditions are satisfied, then the pad sublayer can be injected directly to the source node
        # else, identity+pad must be injected for each of the applicable destination nodes (conv w/ SAME padding)
        if len(dest_nodes) > 0:
            # verify all dest nodes have the same attributes
            all_same_type = all(dnode.type == BuiltinOperator.CONV_2D for dnode in dest_nodes)
            all_same_input_offset = all(dnode.input_offset == dest_nodes[0].input_offset for dnode in dest_nodes)
            pad_attributes = ['padding_width', 'padding_height']
            all_same_pad_hw = all_same_type and all(all(getattr(dnode.Conv2DOptions, attr) == getattr(dest_nodes[0].Conv2DOptions, attr) for attr in pad_attributes) for dnode in dest_nodes)
            all_matching_attr = all_same_input_offset and all_same_pad_hw

            if all_matching_attr:
            # if 0:
                # inject pad sublayer to source node
                opcode=''
                # TODO update with correct arguments when calling function is fixed
                # inject_pad_subnode_to_previous_node(source_node, dest_nodes, ids_with_dummies, preset, opcode)
            else:
                # go through dest nodes and inject identity+pad where applicable
                for dnode in dest_nodes:
                    pad_hw = [dnode.Conv2DOptions.padding_height, dnode.Conv2DOptions.padding_width]
                    if dnode.type == BuiltinOperator.CONV_2D and (dnode.Conv2DOptions.padding_height > 0 or dnode.Conv2DOptions.padding_width > 0):
                        # print('2 ', dnode.tensor_array[0].id)
                        Nodes = inject_dummy_identity(Nodes, ids_with_dummies, 
                                                        dnode,
                                                        preset, 
                                                        pad_hw,
                                                        inject_strided=False)

                        # reset dest_nodes conv padding attributes as node is injected explicitly
                        dnode.Conv2DOptions.padding_height = 0
                        dnode.Conv2DOptions.padding_width = 0

    # for g, node in enumerate(Nodes):
    #     if node.type == BuiltinOperator.CONV_2D:
    #         if node.Conv2DOptions.padding_height != 0 or node.Conv2DOptions.padding_width != 0:
    #             import pdb; pdb.set_trace()

    # print(ids_with_dummies)

# Write all the tensors to a file with their offset
def write_tensor_offset_map(tmp_dir, Nodes, activations_offset, activations_size):
    # Note: Some tensors have multiple offsets now (bug)
    nx_dirname = os.path.join(tmp_dir, "nx_engine")
    offset_map_fname = os.path.join(nx_dirname, "mxp_tensor_offset_map.txt")
    with open(offset_map_fname, 'w') as table:
        table.write(f"Intermediates base address: {activations_offset}\n")
        table.write(f"Intermediates size: {activations_size}\n\n")
        table.write("\t".join(["id", "offset", "name"]) + "\n")
        printed = []
        tensor_ids = []
        for n in Nodes:
            for t in n.tensor_array:
                # Skip duplicate tensors from identity nodes
                # For example, Conv -> Pool may be 1 subgraph in 2.0, but 2 subgraphs in 3.0,
                # where Conv is on NX but Pool on MXP. In the 3.0 case, the Pool is a subnode
                # of an Identity. This means a new tensor with ".id" at the end of the name
                # will be made, but currently its tensor.id is not updated so it looks like a
                # conflicting id / offset mapping. Since this identity tensor is not seen by
                # NX can just omit it for now.
                if t.name.endswith('.id'):
                    continue
                assert t.buffer[0] == 0
                assert t.buffer[1] == t.direct
                entry = (t.id, t.name, t.direct)
                
                if (t.external_producer or t.external_consumer):
                    try:
                        idx = tensor_ids.index(t.id)
                        printed.pop(idx)
                    except ValueError:
                        pass
                
                if entry not in printed:
                    printed.append(entry)
                    tensor_ids.append(t.id)

        for _, entry in enumerate(printed):
            table.write("\t".join([str(entry[0]), str(entry[2]), entry[1]]) + "\n")

# Write a table of each input/output tensor and its size/offset in the vnnx file
# Similar to set_io_buffers
def write_io_offset_map(tmp_dir, Nodes, weights, test_inputs, test_outputs, io_vnnx_offset):
    all_tensor_array = []
    for n in Nodes:
        all_tensor_array += n.tensor_array

    unique_names = []
    for t in all_tensor_array:
        if t.name not in unique_names:
            unique_names.append(t.name)

    offset = io_vnnx_offset
    nx_dirname = os.path.join(tmp_dir, "nx_engine")
    offset_map_fname = os.path.join(nx_dirname, "vnnx_io_offsets.txt")
    table_entries = set()
    with open(offset_map_fname, 'w') as table:
        table.write("\t".join(["type", "name", "id", "size", "offset"]) + "\n")
        for name in unique_names:
            ts = find_tensors_by_name(all_tensor_array, name)
            ts_io = [t for t in ts if (t.name in test_inputs.keys() or t.name in test_outputs.keys())]
            for idx, t in enumerate(ts_io):
                if t.name in test_inputs.keys():
                    data = test_inputs[t.name].astype(np.int8).tobytes()
                    io_type = "input"
                elif t.name in test_outputs.keys():
                    data = test_outputs[t.name].astype(np.int8).tobytes()
                    io_type = "output"
                table_entry = (io_type, t.name, str(t.id), str(len(data)))
                if table_entry in table_entries:
                    continue
                table_entries.add(table_entry)
                table.write("\t".join([io_type, t.name, str(t.id), str(len(data)), str(offset)]) + "\n")
                offset += np.prod(mod_shape(t.shape[:4], 4))*2 # See set_tensor_buffer


def set_fia_preload(Nodes):
    prev_fia_node = None
    for n,node in enumerate(Nodes):
        if node.type in [BuiltinOperator.CONV_2D, BuiltinOperator.DEPTHWISE_CONV_2D, BuiltinOperator.TRANSPOSE_CONV]:
            node.Conv2DOptions.fia_preloaded = 0
            node.Conv2DOptions.next_fia_preload = -1

            if prev_fia_node:
                prev_fia_node.Conv2DOptions.next_fia_preload = n
            prev_fia_node = node


    valid_count = 0
    fia_count = 0
    for n, node in enumerate(Nodes):
        if node.type in [BuiltinOperator.CONV_2D, BuiltinOperator.DEPTHWISE_CONV_2D, BuiltinOperator.TRANSPOSE_CONV]:
            fia_count += 1
            if node.Conv2DOptions.next_fia_preload != -1:
                nn, next_node = node.Conv2DOptions.next_fia_preload, Nodes[node.Conv2DOptions.next_fia_preload]

                nn_prev = -1
                for p,_ in prev_nodes(next_node, Nodes):
                    if p > nn_prev:
                        nn_prev = p
                if nn_prev != -1:
                    if nn_prev < n:
                        next_node.Conv2DOptions.fia_preloaded = 1
                    elif nn_prev == n:
                        last_maps = node.Conv2DOptions.kernels
                        while last_maps - node.maps > 0:
                            last_maps -= node.maps

                        if not next_node.Conv2DOptions.use_depthwise:
                            if node.Conv2DOptions.kernels - last_maps > next_node.Conv2DOptions.imaps:
                                next_node.Conv2DOptions.fia_preloaded = 1
                        # elif not next_node.Conv2DOptions.use_strided and node.orow_last > (next_node.row_start + next_node.rows_0):
                        #     next_node.Conv2DOptions.fia_preloaded = 1

    if 0:
        for n, node in enumerate(Nodes):
            if node.type in [BuiltinOperator.CONV_2D, BuiltinOperator.DEPTHWISE_CONV_2D, BuiltinOperator.TRANSPOSE_CONV]:
                if node.Conv2DOptions.fia_preloaded:
                    valid_count += 1
        print('inputs preloadable', valid_count, '/', fia_count)

def get_graph_sparsity(compression_vbx):

    match compression_vbx:
        case 'ncomp': #no sparsity, original config
            return 0
        case 'comp': #sparsity enabled
            return 1
        case 'ucomp': #unstructured sparsity 3.0
            return 2
        case _:  # The wildcard `_` acts as the default case
            print("ERROR: Unknown Sparisity Type for compression_vbx:", compression_vbx)
            exit(1)

def generate_vnnx_from_json_subgraphs(json_subgraphs, preset, test_inputs, test_outputs, include_io_data=0, tmp_dir=None,\
                                      engine_graphs_nx=None, debug=False, compression_vbx=None, tmp_dir_obj=None):
    graph_inputs, graph_outputs, graph_activations = get_graph_activations(json_subgraphs)
    graph_inputs = list(test_inputs.keys())
    graph_outputs = list(test_outputs.keys())

    aliased_ids = {}
    ids_with_dummies = dict()

    weights = weight_array()

    # Get a list of tensors that are inputs and outputs to external (NX) ops.
    external_inputs = None
    external_outputs = None
    if engine_graphs_nx:
        external_inputs, external_outputs = get_external_io(json_subgraphs, engine_graphs_nx)

    in_id_to_out_ids = tensor_in_id_to_out_id_mapping(json_subgraphs)
    # if while populating nodes we see an input id in this dict, we will inject identity+pad. otherwise, inject pad sublayer to previous node.
    # TODO it will need to change such that we don't inject identity+pad in populate_nodes. it should be done after and if we can't get away with pad sublayer injection.
    sparsity = get_graph_sparsity(compression_vbx)

    Nodes = populate_nodes(json_subgraphs, preset, graph_activations, weights, aliased_ids, ids_with_dummies, tmp_dir,\
        in_id_to_out_ids, engine_graphs_nx, external_inputs, external_outputs, sparsity, tmp_dir_obj=tmp_dir_obj)

    set_fia_preload(Nodes)
    # TODO
    # final_check_pad_sublayer_injection(in_id_to_out_ids, Nodes, ids_with_dummies)

    # setup graph object
    vnnx_graph = Graph()
    vnnx_graph.num_inputs = len(graph_inputs)
    vnnx_graph.num_outputs = len(graph_outputs)

    vnnx_graph.version = graph_version()
    vnnx_graph.vbx_nn_preset = preset_select['PRESET'][preset]
    vnnx_graph.num_layers = len(Nodes)
    vnnx_graph.sparsity = sparsity

    while len(vnnx_graph.description) < DESCRIPTION_CHARS:
        vnnx_graph.description += b"\0"

    io_nodes, io_offsets = set_io_nodes(vnnx_graph, Nodes, graph_inputs, graph_outputs, weights)
    set_io_nodes(vnnx_graph, Nodes, graph_inputs, graph_outputs, weights)
    set_skip_channel_split(Nodes, test_inputs, test_outputs, external_inputs)
    set_skip_concat(Nodes, test_inputs, test_outputs, external_inputs)
    if compression_vbx == 'ucomp':
        act_buffer_size, io_buffer_size = set_io_buffers(Nodes, weights, test_inputs, test_outputs)
    else:
        act_buffer_size, io_buffer_size = set_io_buffers_reused(Nodes, weights, test_inputs, test_outputs)

    update_tensor_shapes(Nodes)

    vnnx_graph.replay_buffer = len(weights)
    vnnx_graph.replay_buffer_size = 1024*1024*512
    replay = bytearray(vnnx_graph.replay_buffer_size)

    node_data, subnode_data, tensor_data, align1 = update_offsets(vnnx_graph, Nodes)

    # do twice, once to find allocate_length, once for real
    vnnx_graph.include_io_data = True
    # vnnx_graph.include_io_data = include_io_data
    vnnx_graph.allocate_bytes = 0
    vnnx_graph.data_bytes = 0
    graph_data = [vnnx_graph.get_structured_data()]
    data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[weights]+[replay])

    replay_start = len(b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[weights]))
    vnnx_graph.replay_buffer = replay_start
    graph_data = [vnnx_graph.get_structured_data()]
    data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[weights]+[replay])

    node_start = len(b"".join(graph_data))
    node_stop = len(b"".join(graph_data+node_data))

    tensor_start = len(b"".join(graph_data+node_data+subnode_data))
    tensor_stop = len(b"".join(graph_data+node_data+subnode_data+tensor_data))

    vnnx_graph.data_bytes = len(data) # - act_buffer_size
    vnnx_graph.allocate_bytes = len(data)

    graph_data = [vnnx_graph.get_structured_data()]
    data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[weights]+[replay])

    replay_first = 1
    trim_act_buffer = 1

    data = data[:vnnx_graph.data_bytes]
    m = vbx.sim.Model(data)
    try:
        sim_outputs = m.run(m.test_input)

        node_data = [m.model_bytes[node_start : node_stop]]
        replay_size = int.from_bytes(m.model_bytes[4*(14+4):4*(14+1+4)], byteorder='little')
        vnnx_graph.replay_buffer_size = replay_size

        if not replay_first:
            replay = m.model_bytes[replay_start : replay_start + vnnx_graph.replay_buffer_size]
            vnnx_graph.fixed_replay_buffer0 = int.from_bytes(m.model_bytes[4*(4):4*(4+1)], byteorder='little')
            vnnx_graph.fixed_replay_buffer1 = int.from_bytes(m.model_bytes[4*(5):4*(5+1)], byteorder='little')
            vnnx_graph.fixed_replay_buffer2 = int.from_bytes(m.model_bytes[4*(6):4*(6+1)], byteorder='little')
            vnnx_graph.fixed_replay_buffer3 = int.from_bytes(m.model_bytes[4*(7):4*(7+1)], byteorder='little')
        else:
            vnnx_graph.replay_buffer_size += 128
            replay = bytearray(vnnx_graph.replay_buffer_size)
            replay_start = len(b"".join(graph_data+node_data+subnode_data+tensor_data+align1))
            vnnx_graph.replay_buffer = replay_start

            node_data, subnode_data, tensor_data, align2 = update_offsets(vnnx_graph, Nodes, vnnx_graph.replay_buffer_size, vnnx_graph.replay_buffer)
            vnnx_graph.replay_buffer = replay_start
            graph_data = [vnnx_graph.get_structured_data()]
            data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[replay]+align2+[weights])

            vnnx_graph.data_bytes = len(data)
            if trim_act_buffer:
                vnnx_graph.data_bytes = len(data) - act_buffer_size
                if not vnnx_graph.include_io_data:
                    vnnx_graph.data_bytes -= io_buffer_size
            vnnx_graph.allocate_bytes = len(data)
            graph_data = [vnnx_graph.get_structured_data()]
            data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[replay]+align2+[weights])

            data = data[:vnnx_graph.data_bytes]
            m = vbx.sim.Model(data)
            sim_outputs = m.run(m.test_input)

            node_data = [m.model_bytes[node_start : node_stop]]
            replay_size = int.from_bytes(m.model_bytes[4*(14+4):4*(14+1+4)], byteorder='little')
            replay = m.model_bytes[replay_start : replay_start + vnnx_graph.replay_buffer_size]
            vnnx_graph.fixed_replay_buffer0 = int.from_bytes(m.model_bytes[4*(4):4*(4+1)], byteorder='little')
            vnnx_graph.fixed_replay_buffer1 = int.from_bytes(m.model_bytes[4*(5):4*(5+1)], byteorder='little')
            vnnx_graph.fixed_replay_buffer2 = int.from_bytes(m.model_bytes[4*(6):4*(6+1)], byteorder='little')
            vnnx_graph.fixed_replay_buffer3 = int.from_bytes(m.model_bytes[4*(7):4*(7+1)], byteorder='little')
    except:
        sys.stderr.write('ERROR: model failed to run\n')
        sys.exit(1)

    if debug and (compression_vbx != 'ucomp'):
        print()
        print('NODE {:3.2f}'.format(len(b"".join(node_data)) / 2**20))
        print('SUBNODE {:3.2f}'.format(len(b"".join(subnode_data)) / 2**20))
        print('TENSOR {:3.2f}'.format(len(b"".join(tensor_data)) / 2**20))

        print('WEIGHTS {:3.2f}'.format(len(b"".join([weights])) / 2**20))
        print('ACT BUFFERS {:3.2f}'.format(act_buffer_size / 2**20))
        print('IO BUFFERS {:3.2f}'.format(io_buffer_size / 2**20))
        print('WEIGHTS - BUFFERS {:3.2f}'.format((len(b"".join([weights]))-(act_buffer_size+io_buffer_size)) / 2**20))
        print('REPLAY {:3.2f}'.format(vnnx_graph.replay_buffer_size / 2**20))
        print()
        print('BINARY {:3.2f}'.format(vnnx_graph.data_bytes / 2**20))
        print('RUNTIME {:3.2f}'.format(vnnx_graph.allocate_bytes / 2**20))
        print()

    if not replay_first:
        data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[weights]+[replay])
        vnnx_graph.data_bytes = len(data)
        vnnx_graph.allocate_bytes = len(data)
        graph_data = [vnnx_graph.get_structured_data()]
        data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[weights]+[replay])
    else:
        data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[replay]+align2+[weights])
        vnnx_graph.data_bytes = len(data)
        if trim_act_buffer:
            vnnx_graph.data_bytes = len(data) - act_buffer_size
            if not vnnx_graph.include_io_data:
                vnnx_graph.data_bytes -= io_buffer_size
        vnnx_graph.allocate_bytes = len(data)
        graph_data = [vnnx_graph.get_structured_data()]
        data = b"".join(graph_data+node_data+subnode_data+tensor_data+align1+[replay]+align2+[weights])
        
        if debug and (compression_vbx != 'ucomp'): #TODO enable for now
            print_io_buffers(Nodes, test_inputs, test_outputs, io_nodes, io_offsets)
            print('NODES OFFSET', node_start)
            print('TENSORS OFFSET', tensor_start)

            print()
            Graph().print_structure()
            Tensor().print_structure()

    # For 3.0, when graphs are being run in another engine, need also to write the
    # physical memory address of each tensor. Currently, for each (relevant) tensor,
    # write the address offset from the graph. Later, the address of the graph will also
    # be known from the reference design (e.g., 0x30_0010_0000).
    if engine_graphs_nx is not None:
        write_tensor_offset_map(tmp_dir, Nodes, vnnx_graph.data_bytes, act_buffer_size)
        # Also write offset of inputs and outputs in .vnnx file, and their sizes
        write_io_offset_map(tmp_dir, Nodes, weights, test_inputs, test_outputs, vnnx_graph.data_bytes - io_buffer_size)

    return data[:vnnx_graph.data_bytes]


def update_offsets(vnnx_graph, Nodes, weights_offset=None, min_addr=None):
    # All buffers allocated, now adjust offsets, and dump to bytearray
    node_size = Node().get_structure_size()
    subnode_size = Subnode().get_structure_size()
    tensor_size = Tensor().get_structure_size()
    num_subnodes = sum([len(n.subnode_array) for n in Nodes])
    num_tensors = sum([len(n.tensor_array) for n in Nodes])
    node_offset = vnnx_graph.get_structure_size()
    subnode_offset = vnnx_graph.get_structure_size() + node_size * len(Nodes)
    tensor_offset = subnode_offset + num_subnodes*subnode_size
        
    if weights_offset is None:
        weights_offset = tensor_offset + num_tensors*tensor_size
    
    #align weights_offset
    align = [bytearray(0)]
    if weights_offset % 16 != 0:
        align_len = 16 - (weights_offset % 16)
        align = [bytearray(align_len)]
        weights_offset += align_len

    vnnx_graph.update_offsets(weights_offset, min_addr)
    for n in Nodes:
        n.update_offsets(weights_offset, min_addr)

    all_subnode_array = []
    for n in Nodes:
        all_subnode_array += n.subnode_array
        n.sublayers = subnode_offset
        n.num_sublayers = len(n.subnode_array)
        subnode_offset += len(n.subnode_array)*subnode_size

    all_tensor_array = []
    for n in Nodes:
        all_tensor_array += n.tensor_array
        n.tensors = tensor_offset
        tensor_offset += len(n.tensor_array)*tensor_size

    for sn in all_subnode_array:
        sn.update_offsets(weights_offset, min_addr)

    for tn in all_tensor_array:
        tn.update_offsets(weights_offset, min_addr)
        tn.buffer[1] += weights_offset #TODO fix data structure


    tensor_data = [t.get_structured_data() for t in all_tensor_array]
    subnode_data = [sn.get_structured_data() for sn in all_subnode_array]
    node_data = [n.get_structured_data() for n in Nodes]

    return node_data, subnode_data, tensor_data, align


def set_concat_tensor_buffer(tname, all_tensor_array, weights, concat_map, concat_offset):
    t = find_tensors_by_name(all_tensor_array, tname)[0]
    if t.buffer[1] == -1:
        if t.name in concat_map.keys():
            t_ = find_tensors_by_name(all_tensor_array, concat_map[t.name])[0]
            if t_.buffer[1] == -1:
                size = np.prod(mod_shape(t_.shape, 4))*2 #TODO fix for proper sized maps (w/ output_strides)
                t_.buffer[1], t_.direct = len(weights), len(weights)
                weights += bytearray(size)

            offset = concat_offset[t.name]
            t.buffer[1], t.direct = t_.buffer[1] + offset, t_.direct + offset


def set_tensor_buffer(idx, t, ts, weights, test_inputs, test_outputs):
    if t.buffer[1] == -1:
        for prev in ts[:idx]:
            if prev.buffer[1] and prev.shape == t.shape:
                t.buffer[1], t.direct = prev.buffer[1], prev.direct
                break
        if t.buffer[1] == -1:
            size = np.prod(mod_shape(t.shape, 4))*2*sizeof_calc_type(t.type) #TODO fix for proper sized maps (w/ output_strides)
            t.buffer[1] = len(weights)
            t.direct = len(weights)
            weights += bytearray(size)


def prev_nodes(node, Nodes):
    input_names = [_.name for _ in node.tensor_array[:node.num_inputs]]
    _nodes = []
    idx = []
    for n,n_ in enumerate(Nodes):
        for output_tensor in n_.tensor_array[n_.num_inputs:]:
            if output_tensor.name in input_names:
                _nodes.append(n_)
                idx.append(n)

    return zip(idx, _nodes)


def next_nodes(node, Nodes):
    output_names = [_.name for _ in node.tensor_array[node.num_inputs:]]
    _nodes = []
    idx = []
    for n in Nodes:
        for input_tensor in n.tensor_array[:n.num_inputs]:
            if input_tensor.name in output_names:
                _nodes.append(n)
                idx.append(n)

    return zip(idx, _nodes)


def set_skip_concat(Nodes, test_inputs, test_outputs, external_inputs):
    for n,node in enumerate(Nodes):
        if node.type == BuiltinOperator.CONCATENATION:
            concat_io = [t for t in node.tensor_array if (t.name in test_inputs.keys() or t.name in test_outputs.keys())]
            idims = len(node.tensor_array[0].shape)

            if (node.ConcatOptions.axis - idims == -3) and len(node.subnode_array) == 0 and len(concat_io) == 0:
                node.skip = 1
                # as we set concat skip last, skip if another node sharing an input is already skipped
                for _, prev in prev_nodes(node, Nodes):
                    shared_nodes = [n_ for _,n_ in next_nodes(prev, Nodes) if n_ != node]
                    if any([_.skip == 1 for _ in shared_nodes]):
                        node.skip = 0
                        break

            # If this concat is output to NX then need to keep it
            if external_inputs:
                assert node.num_outputs == 1
                output_tensor = node.tensor_array[node.num_inputs]
                if output_tensor.id in external_inputs:
                    node.skip = 0


def set_skip_channel_split(Nodes, test_inputs, test_outputs, external_inputs):
    for n,node in enumerate(Nodes):
        if node.type == VNNXOperator.IDENTITY:
            if len(node.subnode_array) > 0 and node.subnode_array[0].type == BuiltinOperator.SLICE:
                split_io = [t for t in node.tensor_array if (t.name in test_outputs.keys())]
                sn = node.subnode_array[0]
                if len(sn.tensor_array[0].shape) >= 3 and len(sn.tensor_array[1].shape) >= 3:
                    width_matches = sn.tensor_array[0].shape[-1] == sn.tensor_array[1].shape[-1]
                    height_matches = sn.tensor_array[0].shape[-2] == sn.tensor_array[1].shape[-2]
                    channel_matches = sn.tensor_array[0].shape[-3] == sn.tensor_array[1].shape[-3]
                    is_channel_slice = width_matches and height_matches and not channel_matches

                    if is_channel_slice and len(split_io) == 0:
                        if not any([n_.type == BuiltinOperator.CONCATENATION for _,n_ in next_nodes(node, Nodes)]):
                            node.skip = 1

        if node.type == BuiltinOperator.SPLIT:
            split_io = [t for t in node.tensor_array if (t.name in test_outputs.keys())]
            idims = len(node.tensor_array[0].shape)

            if (node.SplitOptions.axis == -3) and len(node.subnode_array) == 0 and len(split_io) == 0:
                if node.output_strides[0] == 1 and node.output_strides[1] == 1:
                    if not any([n_.type == BuiltinOperator.CONCATENATION for _,n_ in next_nodes(node, Nodes)]):
                        node.skip = 1


def shared_buffers(nodes, inputs, outputs, concat_map):
    alloc = dict()
    dealloc = dict()
    io_size = dict()

    all_tensor_array = []
    for n in nodes:
        all_tensor_array += n.tensor_array

    io_names = []
    for n, node in enumerate(nodes):
        for t in nodes[n].tensor_array:
            if t.name in concat_map:
                t = find_tensors_by_name(all_tensor_array, concat_map[t.name])[0]
            if t.buffer[1] != -1:
                io_names.append(t.name)
            else:
                io_size[t.name] = np.prod(mod_shape(t.shape,4))*sizeof_calc_type(t.type)
            
    for n, node in enumerate(nodes):
        for t in nodes[n].tensor_array:
            if t.name in concat_map:
                t = find_tensors_by_name(all_tensor_array, concat_map[t.name])[0]
            if not t.name in io_names:
                if not t.name in alloc:
                    alloc[t.name] = n         # allocate only on the first use
                dealloc[t.name] = n       # deallocate on the last use; don't deallocate inputs/outputs

    heap = [(0, 0)]  # occupied memory; each list element is the range of an allocation; start with dummy node
    heap_ptr = dict()   # pointers to heap, including deallocated; heap_ptr[io_id] = memory location
    heap_max = heap[0][1]   # keep track of the largest amount of memory used
    for n in range(len(nodes)):
        for u, alloc_ind in alloc.items():
            if n == alloc_ind:  # find all allocations during this layer
                for h in range(1, len(heap)):
                    if heap[h][0] - heap[h-1][1] >= io_size[u]: # see if new buffer can fit between allocations
                        heap_ptr[u] = heap[h-1][1]
                        heap.insert(h,(heap_ptr[u], heap_ptr[u]+io_size[u]))
                        break
                if not u in heap_ptr:  # put new buffer at the end of the heap
                    heap_ptr[u] = heap[-1][1]
                    heap.append((heap_ptr[u], heap_ptr[u] + io_size[u]))
                heap_max = max(heap_max, heap[-1][1])
        for u, dealloc_ind in dealloc.items():
            if n == dealloc_ind:    # find all deallocations during this layer
                for h in range(1, len(heap)):
                    if heap_ptr[u] == heap[h][0]:  # find the allocation in the heap
                        heap.pop(h)
                        break

    return heap_ptr, heap_max


def set_all_tensor_buffers(tarray, weights, map):
    for name in map.keys():
        ts_io = find_tensors_by_name(tarray, name)
        for idx, t in enumerate(ts_io):
            data = map[t.name].astype(np_type(t.type)).tobytes()
            weights[t.direct:t.direct+len(data)] = data


def remap_all_tensor_buffers(tarray, weights, remap, offset, filter=None):
    for name in remap.keys():
        if filter is None or name in filter:
            ts = find_tensors_by_name(tarray, name)
            for t in ts:
                t_ = find_tensors_by_name(tarray, remap[t.name])[0]
                if t_.buffer[1] != -1:
                    map_all_tensor_buffers(tarray, weights, [t_.name])
                t.buffer[1], t.direct = t_.buffer[1] + offset[t.name], t_.direct + offset[t.name]


def map_all_tensor_buffers(tarray, weights, names):
    for name in names:
        ts = find_tensors_by_name(tarray, name)
        if any(_.buffer[1] == -1 for _ in ts):
            size = np.prod(mod_shape(ts[0].shape, 4))*sizeof_calc_type(ts[0].type) #TODO fix for proper sized maps (w/ output_strides)
            for t in ts:
                t.buffer[1], t.direct = len(weights), len(weights)
            weights += bytearray(size)


def set_io_buffers_reused(Nodes, weights, test_inputs, test_outputs):
    '''
    allocate buffers
    '''
    init_len = len(weights)

    all_tensor_array = []
    for n in Nodes:
        all_tensor_array += n.tensor_array

    unique_names = []
    for t in all_tensor_array:
        if t.name not in unique_names: 
            unique_names.append(t.name)

    concat_offset = {}
    concat_map = {}

    for n, node in enumerate(Nodes):
        if node.type == BuiltinOperator.CONCATENATION:
            if node.skip:
                names = [_.name for _ in node.tensor_array]
                shapes = [_.shape for _ in node.tensor_array]
                offset = 0
                for name, shape in zip(names[:-1],shapes[:-1]):
                    concat_map[name] = names[-1]
                    concat_offset[name] = offset
                    offset += np.prod(shape)
        if node.type == BuiltinOperator.SPLIT:
            if node.skip:
                names = [_.name for _ in node.tensor_array]
                shapes = [_.shape for _ in node.tensor_array]
                offset = 0
                for name, shape in zip(names[1:],shapes[1:]):
                    concat_map[name] = names[0]
                    concat_offset[name] = offset
                    offset += np.prod(shape)

    io_names = list(test_inputs.keys())+list(test_outputs.keys())
    map_all_tensor_buffers(all_tensor_array, weights, io_names)
    remap_all_tensor_buffers(all_tensor_array, weights, concat_map, concat_offset, io_names)
    set_all_tensor_buffers(all_tensor_array, weights, test_inputs)
    set_all_tensor_buffers(all_tensor_array, weights, test_outputs)

    io_len = len(weights)
    io_buffer_size = io_len - init_len

    heap_ptr, heap_max = shared_buffers(Nodes, test_inputs, test_outputs, concat_map)
    for t in all_tensor_array:
        if t.buffer[1] == -1:
            if t.name in heap_ptr:
                t.buffer[1] = len(weights) + heap_ptr[t.name]
                t.direct = len(weights) + heap_ptr[t.name]
            elif t.name not in concat_map:
                print('WARNING not in heap_ptr or remapping', t.name)
    weights += bytearray(heap_max)
    remap_all_tensor_buffers(all_tensor_array, weights, concat_map, concat_offset)

    act_buffer_size = len(weights) - io_len

    return act_buffer_size, io_buffer_size


def set_io_buffers(Nodes, weights, test_inputs, test_outputs):
    '''
    allocate buffers
    '''
    init_len = len(weights)
    # Align inputs/outputs address offsets to 16 bytes for TSNP
    if init_len % 16 != 0:
        align_len = 16 - (init_len % 16)
        init_len += align_len
                    
    all_tensor_array = []
    for n in Nodes:
        all_tensor_array += n.tensor_array

    unique_names = []
    for t in all_tensor_array:
        if t.name not in unique_names: 
            unique_names.append(t.name)

    concat_offset = {}
    concat_map = {}

    for n, node in enumerate(Nodes):
        if node.type == BuiltinOperator.CONCATENATION and node.skip:
            names = [_.name for _ in node.tensor_array]
            shapes = [_.shape for _ in node.tensor_array]

            offset = 0
            for name, shape in zip(names[:-1],shapes[:-1]):
                concat_map[name] = names[-1]
                concat_offset[name] = offset
                offset += np.prod(shape)

    for name in unique_names:
        ts = find_tensors_by_name(all_tensor_array, name)
        ts_io = [t for t in ts if (t.name in test_inputs.keys() or t.name in test_outputs.keys())]
        for idx, t in enumerate(ts_io):
            set_concat_tensor_buffer(t.name, all_tensor_array, weights, concat_map, concat_offset)
            set_tensor_buffer(idx, t, ts_io, weights, test_inputs, test_outputs)

            if t.name in test_inputs.keys():
                # Align inputs/outputs address offsets to 16 bytes for TSNP
                if t.direct % 16 != 0:
                    align_len = 16 - (t.direct % 16)
                    t.direct += align_len
                    t.buffer[1] = t.direct
                data = test_inputs[t.name].astype(np_type(t.type)).tobytes()
                weights[t.direct:t.direct+len(data)] = data
            elif t.name in test_outputs.keys():
                # Align inputs/outputs address offsets to 16 bytes for TSNP
                if t.direct % 16 != 0:
                    align_len = 16 - (t.direct % 16)
                    t.direct += align_len
                    t.buffer[1] = t.direct
                data = test_outputs[t.name].astype(np_type(t.type)).tobytes()
                weights[t.direct:t.direct+len(data)] = data

    io_len = len(weights)
    io_buffer_size = io_len - init_len

    for name in unique_names:
        ts = find_tensors_by_name(all_tensor_array, name)
        for idx, t in enumerate(ts):
            set_concat_tensor_buffer(t.name, all_tensor_array, weights, concat_map, concat_offset)
            set_tensor_buffer(idx, t, ts, weights, test_inputs, test_outputs)


    act_buffer_size = len(weights) - io_len

    return act_buffer_size, io_buffer_size


def print_io_buffers(Nodes, test_inputs, test_outputs, io_nodes, io_offsets, offset=0):
    all_tensor_array = []
    for n in Nodes:
        all_tensor_array += n.tensor_array

    unique_names = []
    for t in all_tensor_array:
        if t.name not in unique_names: 
            unique_names.append(t.name)

    io_idx = 0
    for name in unique_names:
        ts = find_tensors_by_name(all_tensor_array, name)
        ts_in = [t for t in ts if t.name in test_inputs.keys()][:1]
        for idx, t in enumerate(ts_in):
             print('INPUT {}\n\tNAME {}\n\tSHAPE {}\n\tLENGTH {}\n\tDIRECT {} ({})\n\tIO NODE {}\n\tIO OFFSET {}'.format(io_idx, t.name, t.shape, np.prod([_ for _ in t.shape if _ > 0]), t.buffer[1], hex(t.buffer[1]), io_nodes[io_idx], io_offsets[io_idx]))
             io_idx += 1

    input_io_idx = io_idx
    for name in unique_names:
        ts = find_tensors_by_name(all_tensor_array, name)
        ts_out = [t for t in ts if t.name in test_outputs.keys()][:1]
        for idx, t in enumerate(ts_out):
             print('OUTPUT {}\n\tNAME {}\n\tSHAPE {}\n\tLENGTH {}\n\tDIRECT {} ({})\n\tIO NODE {}\n\tIO OFFSET {}'.format(io_idx - input_io_idx, t.name, t.shape, np.prod([_ for _ in t.shape if _ > 0]), t.buffer[1], hex(t.buffer[1]), io_nodes[io_idx], io_offsets[io_idx]))
             io_idx += 1


def set_io_nodes(vnnx_graph, Nodes, graph_inputs, graph_outputs, weights):
    node_inputs = []
    offset_inputs = []
    node_outputs = []
    offset_outputs = []

    for i in graph_inputs + graph_outputs:
        n, off = get_node_and_offset(Nodes, i)
        node_inputs.append(n)
        offset_inputs.append(off)
    
        if type(Nodes[n].tensor_array[off].scale) == type([]):
            Nodes[n].tensor_array[off].scale_f16 = [int(_ * 2**16) for _ in Nodes[n].tensor_array[off].scale]
        else:
            Nodes[n].tensor_array[off].scale_f16 = int(Nodes[n].tensor_array[off].scale * 2**16)

    io_nodes = np.array(node_inputs + node_outputs, dtype=np.int32)
    io_offsets = np.array(offset_inputs + offset_outputs, dtype=np.int32)
    vnnx_graph.io_nodes = len(weights)
    weights += io_nodes.tobytes()
    vnnx_graph.io_offsets = len(weights)
    weights += io_offsets.tobytes()

    return io_nodes, io_offsets


def inject_dummy_identity(Nodes, ids_with_dummies, reference_node, preset, opcode, pad_hw=None, inject_strided=0, transpose_dilate=[1,1], 
                          sparse=None, tmp_dir=None, graph_idx=None, tmp_dir_obj=None):

    dummy_nodes = []
    
    for i in range(reference_node.num_inputs):
        input_shape = reference_node.tensor_array[i].shape
        reference_shape = input_shape
        output_shape = reference_shape

        node = Node()

        node.type = VNNXOperator.IDENTITY

        node.input_data_type = reference_node.input_data_type
        node.output_data_type = reference_node.input_data_type

        if inject_strided == 1:
            node.output_strides = reference_node.input_strides

        if pad_hw is not None:
            sn = Subnode()
            sn.num_inputs = 0 #TODO proper fix
            sn.num_outputs = 0

            pad_h, pad_w = pad_hw[0], pad_hw[1]

            new_h = reference_shape[-2] + pad_h + ((transpose_dilate[0]-1) * (reference_shape[-2] - 1))
            new_w = reference_shape[-1] + pad_w + ((transpose_dilate[1]-1) * (reference_shape[-1] - 1))
            reference_shape = tuple(list(reference_shape)[:-2] + [new_h, new_w])
            output_shape = reference_shape
            reference_node.m, reference_node.n = reference_shape[-2], reference_shape[-1]

            sn.input_data_type = node.input_data_type
            sn.output_data_type = node.output_data_type

            sn.type = BuiltinOperator.PAD
            sn.pads = [0, floor(pad_h/2), floor(pad_w/2), 
                       0, ceil(pad_h/2), ceil(pad_w/2)]
            if reference_node.type == BuiltinOperator.TRANSPOSE_CONV:
                sn.pads = [0, ceil(pad_h/2), ceil(pad_w/2), 
                           0, floor(pad_h/2), floor(pad_w/2)]
            sn.PadOptions.value = reference_node.input_offset
            sn.PadOptions.transpose_dilate_h = transpose_dilate[0]
            sn.PadOptions.transpose_dilate_w = transpose_dilate[1]

            if inject_strided == 1:
                sn.strides = node.output_strides
                osh, osw = node.output_strides[0], node.output_strides[1]

                strided_h = int(ceil(output_shape[-2] / osh)) * osh
                strided_w = int(ceil(output_shape[-1] / osw)) * osw
                # output_shape = tuple(list(output_shape)[:-2] + [strided_h, strided_w])

            # update padding to 0 because padding node will be injected explicitly with identity
            reference_node.Conv2DOptions.padding_width = 0
            reference_node.Conv2DOptions.padding_height = 0
            node.subnode_array.append(sn)

        # if strides == 2,2 then identity+pad/pad were injected to do transpose dilation
        # so set the stride to 1
        if transpose_dilate == [2,2]:
            reference_node.Conv2DOptions.stride_height = 1
            reference_node.Conv2DOptions.stride_width = 1

        node.channels, node.m, node.n = 1, 1, 1
        if len(input_shape) >= 3:
            node.channels = input_shape[-3]
        if len(input_shape) >= 2:
            node.m = input_shape[-2]
        node.n = input_shape[-1]


        node.activation_max = -1
        node.activation_min = -1
        node.input_multiplier = -1
        node.input_shift = -1
        node.output_multiplier = -1
        node.output_shift = -1
        node.input_offset = reference_node.input_offset
        node.output_offset = reference_node.input_offset        

        ref_id = reference_node.tensor_array[i].id
        dummy_id_list = ids_with_dummies.get(ref_id, [])
        dummy_id_name = str(ref_id) + '.' + str(len(dummy_id_list)+1)
        dummy_id_list.append(dummy_id_name)
        ids_with_dummies[ref_id] = dummy_id_list

        # new input tensor for Identity node that is a copy of the reference node input tensor
        ref_tensor = reference_node.tensor_array[i]
        t = Tensor()
        t.type = ref_tensor.type
        t.shape = ref_tensor.shape
        t.dims = ref_tensor.dims
        t.scale = ref_tensor.scale
        t.zero = ref_tensor.zero
        t.id = ref_tensor.id
        t.name = ref_tensor.name
        node.tensor_array.append(t)
        if len(node.subnode_array) == 1: # pad subnode injected
            sn.tensor_array.append(t)

        # new output tensor for Identity node that is what would now be fed into the reference node
        out_t = Tensor()
        out_t.type = ref_tensor.type
        out_t.scale = ref_tensor.scale
        out_t.zero = ref_tensor.zero
        out_t.shape = output_shape
        out_t.dims = ref_tensor.dims
        out_t.id = float(dummy_id_name)
        out_t.name = dummy_id_name
        node.tensor_array.append(out_t)
        if len(node.subnode_array) == 1: # pad subnode injected
            sn.tensor_array.append(out_t)

        if len(node.subnode_array) == 1: # pad subnode injected
            sn.num_tensors = len(sn.tensor_array)

        # node.tensor_array += sn.tensor_array #TODO proper fix.
        node.num_tensors = len(node.tensor_array)

        ref_tensor.shape = reference_shape
        ref_tensor.id = float(dummy_id_name)
        ref_tensor.name = dummy_id_name
        reference_node.tensor_array[i] = ref_tensor

        tile = allocation(node, preset, opcode, sparse=sparse, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)

        # keep description scheme here just for referencing injected nodes
        node.input_description = reference_node.input_description
        while len(node.input_description) < DESCRIPTION_CHARS:
            node.input_description += b"\0"
        node.output_description = dummy_id_name.encode()[:DESCRIPTION_CHARS]
        while len(node.output_description) < DESCRIPTION_CHARS:
            node.output_description += b"\0"

        t = ''
        if ref_tensor.type == calc_type.INT8:
            t = 'INT8'
        elif ref_tensor.type == calc_type.UINT8:
            t = 'UINT8'
        else:
            print('warning: not 8bit')

        dummy_nodes.append(node)
    Nodes = Nodes + dummy_nodes
    return Nodes


def inject_output_shaper(conv_node, conv_o_tensor, subnode_tensor_array, ids_with_dummies, debug=0):
    sn = Subnode()

    sn.input_offset = conv_o_tensor.zero
    sn.scale = conv_o_tensor.scale
    sn.output_offset = conv_o_tensor.zero
    sn.output_scale = conv_o_tensor.scale
 
    sn.input_data_type = conv_node.output_data_type
    sn.output_data_type = conv_node.output_data_type

    sn.activation_min = -128
    sn.activation_max = 127

    sn.type = VNNXOperator.OUTPUT_SHAPER

    oh, ow, osh, osw = get_output_shapes(conv_node.Conv2DOptions, conv_node.rows_0, conv_node.cols_0)
    ohf, owf, oshf, oswf = get_output_shapes(conv_node.Conv2DOptions, conv_node.rows_final, conv_node.cols_final)
    if debug:
        print((oh,ow,osh,osw), (ohf,owf,oshf,oswf), conv_node.Conv2DOptions.use_strided, conv_node.Conv2DOptions.stride_height)
    sn.ShaperOptions.height = oh
    sn.ShaperOptions.height_final = ohf
    sn.ShaperOptions.shaper_height = osh
    sn.ShaperOptions.shaper_height_final = oshf
    sn.ShaperOptions.width = ow
    sn.ShaperOptions.shaper_width = osw
    sn.ShaperOptions.stride_height = conv_node.Conv2DOptions.stride_height
    sn.ShaperOptions.use_strided = conv_node.Conv2DOptions.use_strided

    _, _, osh, osw = get_output_shapes(conv_node.Conv2DOptions, conv_node.m, conv_node.n)
    adjusted_conv_o_tensor_shape = (conv_o_tensor.shape[0], conv_o_tensor.shape[1], osh, osw)

    # pre-sliced tensor ID and name init
    ref_id = conv_o_tensor.id
    dummy_id_list = ids_with_dummies.get(ref_id, [])
    dummy_id_name = str(ref_id) + '.' + str(len(dummy_id_list)+1)
    dummy_id_list.append(dummy_id_name)
    ids_with_dummies[ref_id] = dummy_id_list

    # create new tensor for unshaped input / conv new output
    tn = Tensor()
    tn.type = sn.input_data_type
    tn.shape = adjusted_conv_o_tensor_shape
    # tn.shape = conv_o_tensor.shape
    tn.dims = conv_o_tensor.dims
    tn.scale = sn.output_scale
    tn.zero = sn.output_offset
    tn.id = float(dummy_id_name)
    tn.name = dummy_id_name
    sn.tensor_array.append(tn)
    subnode_tensor_array.insert(0, tn)

    # post-shaped tensor (conv node former output, now is output shaper output)
    tn = Tensor()
    tn.type = sn.output_data_type
    tn.shape = conv_o_tensor.shape
    tn.dims = conv_o_tensor.dims
    tn.scale = sn.output_scale
    tn.zero = sn.output_offset
    tn.id = conv_o_tensor.id
    tn.name = conv_o_tensor.name
    sn.tensor_array.append(tn)
    subnode_tensor_array.insert(1, tn)

    sn.num_tensors = len(sn.tensor_array)
    conv_o_tensor.id = float(dummy_id_name)
    conv_o_tensor.name = dummy_id_name
    conv_o_tensor.shape = adjusted_conv_o_tensor_shape
    conv_node.subnode_array.insert(0, sn)    


def inject_strided_slice(conv_node, conv_o_tensor, subnode_tensor_array, ids_with_dummies):
    sn = Subnode()

    sn.input_offset = conv_o_tensor.zero
    sn.scale = conv_o_tensor.scale
    sn.output_offset = conv_o_tensor.zero
    sn.output_scale = conv_o_tensor.scale
 
    sn.input_data_type = conv_node.output_data_type
    sn.output_data_type = conv_node.output_data_type

    sn.activation_min = -128
    sn.activation_max = 127

    # assuming NCHW format, adjust width of conv node output tensor
    # ignore stride_width in computation as we stride with StridedSlice
    conv_in_w = conv_node.tensor_array[0].shape[-1]
    pad_w = conv_node.Conv2DOptions.padding_width
    filter_w = conv_node.Conv2DOptions.filter_shape_dims[-1]
    dilation_w = conv_node.Conv2DOptions.dilation_width_factor
    adjusted_conv_o_tensor_width = conv_in_w + pad_w - (((filter_w-1)*dilation_w)+1) + 1 
    adjusted_conv_o_tensor_shape = (conv_o_tensor.shape[0], conv_o_tensor.shape[1], conv_o_tensor.shape[2], adjusted_conv_o_tensor_width)

    sn.type = BuiltinOperator.SLICE

    sn.SliceOptions.begin = (0, 0, 0, 0)
    sn.SliceOptions.end = adjusted_conv_o_tensor_shape
    sn.SliceOptions.stride = (1, 1, 1, conv_node.Conv2DOptions.stride_width)

    # pre-sliced tensor ID and name init
    ref_id = conv_o_tensor.id
    dummy_id_list = ids_with_dummies.get(ref_id, [])
    dummy_id_name = str(ref_id) + '.' + str(len(dummy_id_list)+1)
    dummy_id_list.append(dummy_id_name)
    ids_with_dummies[ref_id] = dummy_id_list

    # create new tensor (pre-sliced) for strided slice input / conv new output
    tn = Tensor()
    tn.type = sn.input_data_type
    tn.shape = adjusted_conv_o_tensor_shape
    tn.dims = conv_o_tensor.dims
    tn.scale = sn.output_scale
    tn.zero = sn.output_offset
    tn.id = float(dummy_id_name)
    tn.name = dummy_id_name
    sn.tensor_array.append(tn)
    subnode_tensor_array.insert(0, tn)

    # post-sliced tensor (conv node former output, now is strided slice output)
    tn = Tensor()
    tn.type = sn.output_data_type
    tn.shape = conv_o_tensor.shape
    tn.dims = conv_o_tensor.dims
    tn.scale = sn.output_scale
    tn.zero = sn.output_offset
    tn.id = conv_o_tensor.id
    tn.name = conv_o_tensor.name
    sn.tensor_array.append(tn)
    subnode_tensor_array.insert(1, tn)

    sn.num_tensors = len(sn.tensor_array)
    conv_node.Conv2DOptions.stride_width = 1
    conv_o_tensor.id = float(dummy_id_name)
    conv_o_tensor.name = dummy_id_name
    conv_o_tensor.shape = adjusted_conv_o_tensor_shape
    conv_node.subnode_array.insert(0, sn)    


def inject_pad_subnode_to_previous_node(prev_node, nodes, ids_with_dummies, preset, opcode, weights, prev_node_graph, transpose_dilate=[1,1], \
                                        sparse=None, tmp_dir=None, graph_idx=None, tmp_dir_obj=None):
    sn = Subnode()

    if isinstance(nodes, Node):
        node = nodes
    elif isinstance(nodes, list):
        node = nodes[0]

    # pad_h, pad_w = pad_hw[0], pad_hw[1]
    pad_h = node.Conv2DOptions.padding_height
    pad_w = node.Conv2DOptions.padding_width

    input_id = node.tensor_array[0].id
    input_shape = node.tensor_array[0].shape

    new_h = input_shape[-2] + pad_h + ((transpose_dilate[0]-1) * (input_shape[-2] - 1))
    new_w = input_shape[-1] + pad_w + ((transpose_dilate[1]-1) * (input_shape[-1] - 1))
    input_shape = tuple(list(input_shape)[:-2] + [new_h, new_w])
    prev_node_output_shape = input_shape
    node.m, node.n = input_shape[-2], input_shape[-1]

    sn.input_data_type = prev_node.input_data_type
    sn.output_data_type = prev_node.output_data_type

    sn.type = BuiltinOperator.PAD
    sn.pads = [0, floor(pad_h/2), floor(pad_w/2), 
                0, ceil(pad_h/2), ceil(pad_w/2)]
    if node.type == BuiltinOperator.TRANSPOSE_CONV:
        sn.pads = [0, ceil(pad_h/2), ceil(pad_w/2), 
                    0, floor(pad_h/2), floor(pad_w/2)]
    sn.PadOptions.value = node.input_offset
    sn.PadOptions.transpose_dilate_h = transpose_dilate[0]
    sn.PadOptions.transpose_dilate_w = transpose_dilate[1]

    dummy_id_list = ids_with_dummies.get(input_id, [])
    dummy_id_name = str(input_id) + '.' + str(len(dummy_id_list)+1)
    dummy_id_list.append(dummy_id_name)
    ids_with_dummies[input_id] = dummy_id_list

    # creating new input tensor for pad subnode, based off current node's input
    ref_tensor = node.tensor_array[0]
    t = Tensor()
    t.type = ref_tensor.type
    t.shape = ref_tensor.shape
    t.dims = ref_tensor.dims
    t.scale = ref_tensor.scale
    t.zero = ref_tensor.zero
    t.id = ref_tensor.id
    t.name = ref_tensor.name
    sn.tensor_array.append(t)

    # creating new output tensor for pad subnode, based off current node's padded input
    out_t = Tensor()
    out_t.type = ref_tensor.type
    out_t.scale = ref_tensor.scale
    out_t.zero = ref_tensor.zero
    out_t.shape = prev_node_output_shape
    out_t.dims = ref_tensor.dims
    out_t.id = float(dummy_id_name)
    out_t.name = dummy_id_name
    sn.tensor_array.append(out_t)

    sn.num_tensors = len(sn.tensor_array)

    # if strides == 2,2 then identity+pad/pad were injected to do transpose dilation
    # so set the stride to 1
    if transpose_dilate == [2,2]:
        node.Conv2DOptions.stride_height = 1
        node.Conv2DOptions.stride_width = 1

    # update current node's input to be padded
    ref_tensor.shape = prev_node_output_shape
    ref_tensor.id = float(dummy_id_name)
    ref_tensor.name = dummy_id_name
    node.tensor_array[0] = ref_tensor

    # keep description scheme here just for referencing injected nodes
    prev_node.output_description = dummy_id_name.encode()[:DESCRIPTION_CHARS]
    while len(prev_node.output_description) < DESCRIPTION_CHARS:
        prev_node.output_description += b"\0"
    node.input_description = dummy_id_name.encode()[:DESCRIPTION_CHARS]
    while len(node.input_description) < DESCRIPTION_CHARS:
        node.input_description += b"\0"

    # padding set to 0 because padding subnode is injected explicitly into prev node
    node.Conv2DOptions.padding_width = 0
    node.Conv2DOptions.padding_height = 0

    # if nodes was a list, go through it and update all dest nodes input tensor and other attributes
    # this is for when the previous node's output branches, but all receiving nodes want the same pad
    if isinstance(nodes, list):
        for dnode in nodes[1:]:
            dnode.m, dnode.n = input_shape[-2], input_shape[-1]
            
            in_tensor = dnode.tensor_array[0]
            in_tensor.shape = prev_node_output_shape
            in_tensor.id = float(dummy_id_name)
            in_tensor.name = dummy_id_name
            dnode.tensor_array[0] = in_tensor
            
            dnode.input_description = dummy_id_name.encode()[:DESCRIPTION_CHARS]
            while len(dnode.input_description) < DESCRIPTION_CHARS:
                dnode.input_description += b"\0"
            
            dnode.Conv2DOptions.padding_width = 0
            dnode.Conv2DOptions.padding_height = 0

    # append pad subnode and update prev node # of tensors
    prev_node.tensor_array += sn.tensor_array
    prev_node.subnode_array.append(sn)
    prev_node.num_tensors = len(prev_node.tensor_array)

    # re-allocate previous node due to pad subnode
    codes = [_['builtin_code'] for _ in prev_node_graph['operator_codes']]
    subgraph = prev_node_graph['subgraphs'][0]
    ops = subgraph['operators']
    opcodes = [str(codes[_['opcode_index']]) for _ in ops]
    prev_opcode = opcodes[0]
    tile = allocation(prev_node, preset, prev_opcode, sparse=sparse, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)
    assert(tile is not None)

    if prev_node.type == BuiltinOperator.CONV_2D:
        set_conv_attributes(prev_node, tile, preset, prev_opcode, weights, prev_node_graph, sparse=sparse, tmp_dir=tmp_dir)


# helper function to run code that is tile-dependent for previous Conv2D layers that were re-allocated
def set_conv_attributes(node, tile, preset, opcode, weights, prev_node_graph, sparse=None, tmp_dir=None, graph_idx=None, tmp_dir_obj=None):
    subgraph = prev_node_graph['subgraphs'][0]
    ops = subgraph['operators']
    tensors = subgraph['tensors']
    op = ops[0]
    f_tensor = tensors[op['inputs'][1]]
    buffers = prev_node_graph['buffers']
    filter_data = get_numpy_data(f_tensor, buffers).transpose((0,3,1,2))

    conv8 = node.Conv2DOptions

    # if not full rows, then don't double buffer
    # previously disabling double buffer for k > 3 because the reduced scratchpad size causes
    # # the first k=6 layer of yolov5 to not fit all columns, and that gives incorrect results
    if node.n != tile[-1]:
        conv8.mxp_double_buffer = 0
        conv8.split_weight_shaper_buffers = 0
        tile = allocation(node, preset, opcode, sparse=sparse, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)

    assert not (tile is None)

    # TODO would need to re-determine prev_fia_node whenever collision is fixed
    # as check_node_for_collision only forces collisions, which would've been done for a previous node already,
    # no need to run it here again
    # if conv8.use_fia:
    #     check_node_for_collision(node, Nodes, prev_fia_node)

    #     prev_fia_node = node

    weight_pad = 0
    _, _, _, _, _, rows, columns = tile
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]    
    if conv8.use_fia and conv8.use_depthwise and node.m == rows and node.n == columns: # check if full maps, if so pad weights here for 1D DMA
        dilated_filter_height = ((h-1)*conv8.dilation_height_factor) + 1
        padded_input_height = node.m + conv8.padding_height
        output_shaper_height = (padded_input_height - dilated_filter_height) + 1                
        conv8.fit_weights = 0
        weight_pad = min(parallel_output_maps - 1, output_shaper_height)
    
    # if not full input maps, then set weight pad to be parallel_output_maps-1. We want to avoid splat in FIA. Offset appropriately when DMAing in FIA code
    # TODO should fit all weights if possible (same for above case)
    elif conv8.use_fia and conv8.use_depthwise and (node.m != rows or node.n != columns): 
        conv8.fit_weights = 0
        weight_pad = parallel_output_maps-1

    # force byte alignment for weights
    if len(weights) % parallel_output_maps != 0:
        filler = parallel_output_maps - (len(weights) % parallel_output_maps)
        fmt = "{}x".format(filler)
        weights += struct.pack(fmt)
    conv8.filter_data = len(weights)
    conv8.repeat = -1
    filter_data = pad_weights_and_pack_2_1(filter_data, node, parallel_output_maps, tile, preset, opcode, weight_pad, \
                                            sparse=sparse, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)
    if node.tensor_array[-1].type==calc_type.UINT8:
        fmt = "{}B".format(len(filter_data.flatten()))
    else:
        fmt = "{}b".format(len(filter_data.flatten()))
    weights += struct.pack(fmt, *filter_data.flatten())    


def update_nodes_with_dequant_params(Nodes, dequantize_op_params):
    for idx,node in enumerate(Nodes):
        for t_idx, tensor in enumerate(node.tensor_array):
            '''
            if tensor.name in dequantize_op_params.keys():

                offset = dequantize_op_params[tensor.name][1]
                scale = dequantize_op_params[tensor.name][0]

                if dequantize_op_params[tensor.name][3] == 'input':
                    node.input_offset = offset
            '''
            if tensor.id in dequantize_op_params.keys():

                offset = dequantize_op_params[tensor.id][1]
                scale = dequantize_op_params[tensor.id][0]

                if dequantize_op_params[tensor.id][3] == 'input':
                    node.input_offset = offset
                else:
                    node.output_offset = offset
                tensor.zero = offset
                tensor.scale = scale
                node.tensor_array[t_idx] = tensor
                
                if node.type == BuiltinOperator.SPLIT:
                    for t_idx2, tensor2 in enumerate(node.tensor_array):
                        tensor2.zero = offset
                        tensor2.scale = scale
                        node.tensor_array[t_idx2] = tensor2
                Nodes[idx] = node
    return Nodes


# Get a set of all tensors which are inputs and outputs of NX (external to MXP) ops
def get_external_io(json_subgraphs: list, engine_graphs_nx: list) -> tuple[set, set]:
    external_inputs = set()
    external_outputs = set()
    if not engine_graphs_nx:
        return external_inputs, external_outputs

    # Collect all the inputs and outputs of NX ops
    # Note: can use name instead of ID, but ID might be faster
    for eg in engine_graphs_nx:
        for split in eg:
            g = json_subgraphs[split]
            subgraph = g['subgraphs'][0]
            for op in subgraph['operators']:
                for i in op['inputs']:
                    external_inputs.add(i)
                for o in op['outputs']:
                    external_outputs.add(o)

    return external_inputs, external_outputs


# Add external_producer and external_consumer attributes to a tensor
def add_external_producer_consumer_info(t, external_inputs, external_outputs,\
    already_external_producer, already_external_consumer, subgraph_idx):

    has_external_producer = (t.id in external_outputs)
    has_external_consumer = (t.id in external_inputs)

    already_marked_external_producer = (t.id in already_external_producer and already_external_producer[t.id] != subgraph_idx)
    already_marked_external_consumer = (t.id in already_external_consumer and already_external_consumer[t.id] != subgraph_idx)

    if has_external_producer and not already_marked_external_producer:
        t.external_producer = True
        already_external_producer[t.id] = subgraph_idx

    if has_external_consumer and not already_marked_external_consumer:
        t.external_consumer = True
        already_external_consumer[t.id] = subgraph_idx


def check_node_for_collision(node, Nodes, prev_fia_node):
    node.Conv2DOptions.fia_collision = 1
    node.FullyConnectedOptions.fia_collision = 1
    return
    # check if prev node(s) that feeds into this node, is non-FIA (current node must wait for signal from MXP), count it as collision
    if len(Nodes) != 0:
        for idx in range(node.num_inputs): 
            input_id = node.tensor_array[idx].id

            # check tensors of every node for a match for input, if so check if the node uses MXP (i.e. is not FIA)
            for prev_node in Nodes:
                for tensor in prev_node.tensor_array:
                    if tensor.id == input_id: 
                        if prev_node.Conv2DOptions.use_fia == 1 or prev_node.FullyConnectedOptions.use_fia == 1:
                            # FIA tile collision is handled separately below
                            pass 
                        else:
                            node.Conv2DOptions.fia_collision = 1
                            node.FullyConnectedOptions.fia_collision = 1
                            return

    # check FIA tile collision
    if prev_fia_node:

        prev_fia_o_tensor_id = prev_fia_node.tensor_array[-1].id
        node_i_tensor_id = node.tensor_array[0].id
        node_i_tensor_ids = [node.tensor_array[_].id for _ in range(node.num_inputs)]

        # if prev_fia_node feeds into current node, then it is collision
        # TODO determine if we can get no collisions on other layer specs / tilings
        if prev_fia_o_tensor_id in node_i_tensor_ids:
            # if prev_fia_node.type == BuiltinOperator.CONV_2D or prev_fia_node.type == BuiltinOperator.TRANSPOSE_CONV:
            #     prev_fia_node_omaps = prev_fia_node.Conv2DOptions.kernels
            # else:
            #     prev_fia_node_omaps = 1

            # last_tile_maps = prev_fia_node_omaps % prev_fia_node.maps
            # if last_tile_maps == 0: # if 0, then prev_fia_node_omaps == prev_fia_node.maps
            #     last_tile_maps = prev_fia_node.maps
            # prev_fia_node_completed_maps = prev_fia_node_omaps - last_tile_maps

            # row_collision_threshold = node.row_start + node.rows_0
            # if node.Conv2DOptions.use_db and node.Conv2DOptions.conv_rows:
            #     row_collision_threshold = node.Conv2DOptions.conv_rows

            # col_collision_threshold = node.col_start + node.cols_0

            # channel_collision_threshold = node.channels
            # if node.type == BuiltinOperator.CONV_2D:
            #     channel_collision_threshold = node.Conv2DOptions.imaps
            # if node.type == BuiltinOperator.FULLY_CONNECTED:
            #     channel_collision_threshold = 1

            # if channel_collision_threshold < prev_fia_node_completed_maps:
            #     return

            # if row_collision_threshold <= prev_fia_node.row_last // prev_fia_node.Conv2DOptions.stride_height:
            #     return

            # if col_collision_threshold <= prev_fia_node.col_last // prev_fia_node.Conv2DOptions.stride_width:
            #     return

            node.Conv2DOptions.fia_collision = 1
            node.FullyConnectedOptions.fia_collision = 1


def populate_nodes(json_subgraphs, preset, graph_activations, weights, aliased_ids, ids_with_dummies, tmp_dir,\
    in_id_to_out_ids, engine_graphs_nx=None, external_inputs=None, external_outputs=None, compression_vbx=None, tmp_dir_obj=None):
    dequantize_op_params = dict() # old_id : (scale, offset, new_id, "input"/"output")
    Nodes = []
    node_ids = []
    Nodes_to_graph_dict = {} # can be used for referencing past Nodes while populating current/future nodes

    valid_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'supported_ops.json') 
    valid = json_load(valid_path)
    is_vectorized = lambda x: x in valid and valid[x]['vectorized'][0] >= 1

    # This dict is needed instead of a set because e.g. max pool is a subnode to an identity.
    # So the same input tensor is first checked on the pool and then on the identity.
    # But that means on the identity (the node, not subnode) the input sync will be false.
    # The solution below is to set the sync for all tensors in the node. If this causes any
    # issues, another solution is to ignore subnodes, and just do this for the main nodes.
    # Another possibility is to keep a data structure in the C/C++ like a cache of what tensor
    # is loaded / saved (e.g., by indirect / direct address) to avoid duplicates.
    already_external_producer = {}
    already_external_consumer = {}

    first_fia = False
    prev_fia_node = None

    for g, graph in enumerate(json_subgraphs):
        codes = [_['builtin_code'] for _ in graph['operator_codes']]
        subgraph = graph['subgraphs'][0]
        buffers = graph['buffers']
        sig = graph['signature_defs'][0]


        node = Node()
        Nodes_to_graph_dict[node] = graph
        ops = subgraph['operators']
        opcodes = [str(codes[_['opcode_index']]) for _ in ops]
        tensors = subgraph['tensors']
        opcode = opcodes[0]
        layer_codes = []

        if engine_graphs_nx:
            # Set whether or not this node is going to be offloaded to NX
            node.offloaded = is_split_idx_in_engine_graphs(g, engine_graphs_nx)
            # Most of the below can be skipped, but still need to add input and output tensors so
            # set_io_nodes can find the nodes for these tensors

        node.use_replay = all([is_vectorized(_) for _ in opcodes])
        if not node.use_replay:
            print('WARNING: replay cannot be used on subgraph {}'.format(g))
            for c, code in enumerate(opcodes):
                if not is_vectorized(code):
                    print('\t{} {} is not vectorized'.format(c, code))

        for sig_input in sig['inputs']:
            if sig_input['tensor_index'] in graph_activations:
                input_type = calc_type.from_str(graph_activations[sig_input['tensor_index']]['type'])
                break
        
        for sig_output in sig['outputs']:
            if sig_output['tensor_index'] in graph_activations:
                output_type = calc_type.from_str(graph_activations[sig_output['tensor_index']]['type'])
                break
        
        node.input_description = sig['inputs'][0]['name'].encode()[:DESCRIPTION_CHARS]
        while len(node.input_description) < DESCRIPTION_CHARS:
            node.input_description += b"\0"
        node.output_description = sig['outputs'][0]['name'].encode()[:DESCRIPTION_CHARS]
        while len(node.output_description) < DESCRIPTION_CHARS:
            node.output_description += b"\0"

        input_shapes = []
        for _ in sig['inputs']:
            if 'shape' in tensors[_['tensor_index']]:
                input_shapes += [tensors[_['tensor_index']]['shape']]
                break
        
        output_shapes = [tensors[_['tensor_index']]['shape'] for _ in sig['outputs']]

        input_buffers = []
        for _ in ops[0]['inputs']:
            if 'buffer' in tensors[_]:
                input_buffers += [buffers[tensors[_]['buffer']]]
        multi_input = len(input_buffers) > 1 and not any(['data' in _ for _ in input_buffers])

        REDUCTION = ["ARG_MIN", "ARG_MAX", "SUM", "REDUCE_PROD", "REDUCE_MAX", "REDUCE_MIN", "MEAN"]
        COMPARISON = ["GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL", "EQUAL", "NOT_EQUAL"]
        ACTIVATION = ["LEAKY_RELU", "SOFTMAX", "LOG_SOFTMAX", "RELU", "PRELU", "RELU6", "RELU_N1_TO_1", "RELU_0_TO_1"]
        LUT = ["TANH", "HARD_SWISH", "SILU", "LOGISTIC", "GELU", "ELU", "LOG", "POW", "EXP"]
        POINT = ['ABS', 'NEG', 'RSQRT', 'SQUARED_DIFFERENCE']
        DATA_MOVEMENT = ["SLICE", "STRIDED_SLICE", "GATHER", "TRANSPOSE"]
        DATA_MOVEMENT += ["UNPACK"]
        DATA_MOVEMENT += ["RESHAPE", "EXPAND_DIMS", "SQUEEZE"]
        DATA_MOVEMENT += ["DEPTH_TO_SPACE", "SPACE_TO_DEPTH", 'BATCH_TO_SPACE_ND', 'SPACE_TO_BATCH_ND']
        PADDING = ['PAD', 'PADV2', 'MIRROR_PAD', 'DILATE']
        POOLING = ['MAX_POOL_2D', 'AVERAGE_POOL_2D']
        KERNEL_REGULARIZER = ['L2_NORMALIZATION']
        POST_PROCESSING = ['EMBEDDING_LOOKUP']
        BINARY = ['ADD', 'SUB', 'MUL', 'SUB', 'DIV', "SQUARED_DIFFERENCE"]
        TYPE_CAST = ['CAST']
        if g == 0: #TODO if input use eltwise, or take in input 1
            BINARY += ['ADD']
        VALID_SUBNODE = BINARY + PADDING + POOLING + REDUCTION + COMPARISON + ACTIVATION + LUT + POINT + DATA_MOVEMENT + KERNEL_REGULARIZER + POST_PROCESSING + TYPE_CAST
        if g != 0: #TODO
            VALID_SUBNODE += ['ADD']
        VALID_SUBNODE += ['DEPTHWISE_CONV_2D']
        VALID_SUBNODE += ['RESIZE_BILINEAR']
        VALID_SUBNODE += ['RESIZE_NEAREST_NEIGHBOR']
        VALID_SUBNODE += ['MINIMUM', 'MAXIMUM']
        VALID_SUBNODE += ['CAST']

        if opcode in ['CONV_2D', 'DEPTHWISE_CONV_2D']:
            node.type = BuiltinOperator.CONV_2D
            subops = ops[1:]
            subcodes = opcodes[1:]
        elif opcode in ['FULLY_CONNECTED']:
            node.type = BuiltinOperator.FULLY_CONNECTED
            subops = ops[1:]
            subcodes = opcodes[1:]
        elif opcode in ['BATCH_MATMUL']:
            node.type = BuiltinOperator.BATCH_MATMUL
            subops = ops[1:]
            subcodes = opcodes[1:]
        elif opcode in ["CONCATENATION", "PACK", "SPLIT", "SPLIT_V", "TILE"]:
            node.type = getattr(BuiltinOperator, f"{opcode}")
            subops = ops[1:]
            subcodes = opcodes[1:]
        elif opcode in ['UNIDIRECTIONAL_SEQUENCE_LSTM']:
            node.type = BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM
            subops = ops[1:]
            subcodes = opcodes[1:]
        elif opcode in ['TRANSPOSE_CONV']:
            node.type = BuiltinOperator.TRANSPOSE_CONV
            subops = ops[1:]
            subcodes = opcodes[1:]
        elif opcode in BINARY + COMPARISON and multi_input:
            node.type = VNNXOperator.IDENTITY
            subops = ops
            subcodes = opcodes
        elif opcode in ["QUANTIZE", "DEQUANTIZE"] or opcode in VALID_SUBNODE or opcode in POST_PROCESSING:
            node.type = VNNXOperator.IDENTITY
            subops = ops
            subcodes = opcodes
        else:
            sys.stderr.write('ERROR: {} not implemented\n'.format(opcode))
            sys.exit(1)

        if node.type == VNNXOperator.IDENTITY:
            layer_codes.append('IDENTITY')
        else:
            layer_codes.append(opcode)

        prev_subop = None
        subnode_tensor_array = []
        
        i = 0
        lut_ops = None
        reshape_ops = None
        nlf_data = None
        while i < len(subops):
            subcode, pattern = graph_pattern(subops[i:], codes, tensors, buffers)
            pattern = [_+i for _ in pattern]

            if len(pattern):
                subops_ = [subops[_] for _ in pattern]
                subcodes_ = [subcodes[_] for _ in pattern]
                if subcode == "LUT":
                    lut_ops = list(zip(subops_, subcodes_, pattern))
                    if node.type in [BuiltinOperator.CONV_2D, BuiltinOperator.TRANSPOSE_CONV, BuiltinOperator.FULLY_CONNECTED] and i == 0 and len(pattern) == 3 and CONV_NLF and tensors[subops_[2]['outputs'][0]]['type'] in ['INT8']: #TODO add UINT8 support
                        nlf_data = get_numpy_data_from_index(subops_[2]['inputs'][0], tensors, buffers)
                        nlf_data = lut_i8_to_u8(nlf_data)
                        i += len(pattern)
                        continue
                elif subcode == "TRANSFORM":
                    reshape_ops = list(zip(subops_, subcodes_, pattern))
            else:
                subops_ = [subops[i]]
                subcode = subcodes[i]
            layer_codes.append(subcode)
            subnode_array, subnode_tensors, prev_subop = populate_subnodes(subcode, ops, lut_ops, reshape_ops, subops_, prev_subop, graph_activations, tensors, buffers, weights, dequantize_op_params, aliased_ids, tmp_dir,\
                engine_graphs_nx, external_inputs, external_outputs, already_external_producer, already_external_consumer, node.offloaded, g)
           
            node.subnode_array += subnode_array
            subnode_tensor_array += subnode_tensors
            i += len(subops_)
        # print op and sub_ops as they are actually ran
        # print(g, layer_codes)

        op = ops[0]
        opts = None
        if 'builtin_options' in op:
            opts = op['builtin_options']

        if node.type == BuiltinOperator.SPLIT:
            i_tensor = tensors[op['inputs'][1]]
            
        elif node.type == BuiltinOperator.TRANSPOSE_CONV:
            i_tensor = tensors[op['inputs'][2]]
        else:
            i_tensor = tensors[op['inputs'][0]]
            for input in op['inputs']:
                tensor = tensors[input]
                if 'buffer' in tensor and 'data' in buffers[tensor['buffer']]:
                    pass
                else:
                    i_tensor = tensor
                    break

        o_tensor = tensors[op['outputs'][0]]
        input_offset = 0
        input_scale = 1.0
        if 'quantization' in i_tensor.keys() and 'zero_point' in i_tensor['quantization'].keys():
            input_offset = i_tensor['quantization']['zero_point'][0]
        if 'quantization' in i_tensor.keys() and 'scale' in i_tensor['quantization'].keys():
            input_scale = i_tensor['quantization']['scale'][0]

        output_offset = 0
        output_scale = 1.
        if 'quantization' in o_tensor.keys() and 'zero_point' in o_tensor['quantization'].keys():
            output_offset = o_tensor['quantization']['zero_point'][0]
        if 'quantization' in o_tensor.keys() and 'scale' in o_tensor['quantization'].keys():
            output_scale = o_tensor['quantization']['scale'][0]

        # force undefined batch to 1 TODO move to preprocess
        if i_tensor['shape'][0] == -1:
            i_tensor['shape'][0] = 1
        if o_tensor['shape'][0] == -1:
            o_tensor['shape'][0] = 1

        node.input_offset = input_offset
        node.output_offset = output_offset

        node.input_data_type = calc_type.from_str(i_tensor['type'])
        node.output_data_type = calc_type.from_str(o_tensor['type'])

        input_shape, idims = channels_first_shape(i_tensor['shape'])

        t = Tensor()
        t.type = calc_type.from_str(i_tensor['type'])
        t.shape = input_shape
        t.dims = idims
        t.scale = input_scale
        t.zero = input_offset
        t.id = i_tensor['buffer'] - 1
        t.name = i_tensor['name']
        if not node.offloaded and engine_graphs_nx:
            add_external_producer_consumer_info(t, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)
        node.tensor_array.append(t)

        node.channels, node.m, node.n = 1, 1, 1
        if idims >= 3:
            node.channels = t.shape[-3]
        if idims >= 2:
            node.m = t.shape[-2]
        node.n = t.shape[-1]

        if node.type == BuiltinOperator.CONV_2D:
            conv8 = node.Conv2DOptions
            conv8.direct_dma = 0

            conv8.in_used = -1
            conv8.w_used = -1
            conv8.qt_used = -1
            conv8.out_used = -1
            conv8.sp_used = -1

            f_tensor = tensors[op['inputs'][1]]
            b_tensor = None
            if len(op['inputs']) > 2 and op['inputs'][2] != -1:
                b_tensor = tensors[op['inputs'][2]]

            k, h, w, c = tuple(f_tensor['shape'])

            filter_shape_dims = [k, c, h, w]
            conv8.filter_shape_dims = filter_shape_dims

            conv8.kernels = k
            conv8.fit_weights = 0
            conv8.use_depthwise = 0
            conv8.group = 1
            if k == 1 and c > 1 and opcode == 'DEPTHWISE_CONV_2D':
                conv8.kernels = c
                conv8.group = c
                conv8.use_depthwise = 1
            # else:
            parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]
            if k*h*w*c <= ((fia_weight_shaper_size_per_bank_kb(parallel_output_maps)*1024)*WEIGHT_SHAPER_DATA_BANKS):
                conv8.fit_weights = ALL_WEIGHTS

            node.input_strides = [opts['stride_h'], opts['stride_w']]
            conv8.stride_width = opts['stride_w']
            conv8.stride_height = opts['stride_h']
            conv8.dilation_height_factor = opts['dilation_h_factor']
            conv8.dilation_width_factor = opts['dilation_w_factor']
            conv8.fused_activation = 0
            node.activation_max = 127
            node.activation_min = -128
            if opts['fused_activation_function'] == 'RELU':
                node.activation_min = output_offset
            elif opts['fused_activation_function'] == 'RELU6':
                node.activation_min = output_offset
                node.activation_max = round((6 / output_scale) + output_offset)
                node.activation_max = min(node.activation_max, 127)

            conv8.use_strided = 0
            conv8.use_vector = USE_CONV_VECTOR
            conv8.use_fia = conv8.use_vector and USE_CONV_FIA
            conv8.use_db = USE_FIA_DB
            conv8.conv_rows = USE_FIA_DB

            if conv8.use_fia and conv8.stride_width != 1 and not conv8.use_depthwise:
                conv8.use_strided = 1

            pad_hw = None
            pad_h, pad_w = 0, 0

            if opts['padding'] == 'VALID':
                conv8.padding_width = 0
                conv8.padding_height = 0

            elif opts['padding'] == 'SAME':
                # inject identity node with padding before this conv node
                stride_h, stride_w = conv8.stride_height, conv8.stride_width
                dilation_h, dilation_w = conv8.dilation_height_factor, conv8.dilation_width_factor
                kernel_h, kernel_w = h, w

                i_h, i_w = node.m, node.n
                o_h, o_w = tensors[op['outputs'][0]]['shape'][-3], tensors[op['outputs'][0]]['shape'][-2]

                pad_h = max(stride_h * (o_h - 1) - i_h + kernel_h + (kernel_h-1)*(dilation_h-1), 0)
                pad_w = max(stride_w * (o_w - 1) - i_w + kernel_w + (kernel_w-1)*(dilation_w-1), 0)

                conv8.padding_width = 0
                conv8.padding_height = 0

                if pad_w > 0 or pad_h > 0:
                    # these values will be updated to 0 later if an identity+pad or just pad are injected explicitly
                    conv8.padding_width = pad_w
                    conv8.padding_height = pad_h

                    # for FIA, doing padding in injected layer is faster than doing padding in FIA which would force 2D or 3D DMA
                    pad_hw = [pad_h, pad_w]

            # identity should only be injected if any of these are true:
            # 1) pad_hw is set, and there is no previous subgraph, i.e. this Conv node is the first subgraph of the network
            # 2) it is strided FIA and there's no previous subgraph to re-map the inputs, or 
            #    if the prev subgraph's output is mapped to different subgraphs that do not want re-mapped inputs            
            inject_identity = False

            # find previous node
            input_id = node.tensor_array[0].id
            _,prev_node = get_prev_node(Nodes, node)

            if pad_hw is not None:
                if prev_node is None:
                    inject_identity = True
                elif op['inputs'][0] in in_id_to_out_ids.keys():
                    inject_identity = True
                elif prev_node.num_outputs > 1:
                    inject_identity = True
                elif input_id in in_id_to_out_ids.keys():
                    # TODO the input to this node is shared with another node, should skip and determine after populate_nodes if pad can be injected or identity must be injected
                    # until implemented, just continue inject_identity for these cases
                    inject_identity = True
                else: #input id not in in_id_to_out_ids.keys                    
                    inject_pad_subnode_to_previous_node(prev_node, node, ids_with_dummies, preset, opcode, weights, Nodes_to_graph_dict[prev_node], sparse=compression_vbx, tmp_dir=tmp_dir)
                # elif len(prev_node.subnode_array) == 0:
                #     inject_identity = True
                # elif prev_node.subnode_array[-1].type != BuiltinOperator.PAD:
                #     inject_identity = True

            if conv8.use_strided:
                if prev_node is None:
                    inject_identity = True
                else:
                    # get prev_node output buffer idand search for it among all other subgraphs inputs
                    # it should not occur more than once, otherwise inject identity
                    # TODO if other node it outputs to is strided as well, then an identity need not be injected (same idea as pad optimization)
                    output_id = prev_node.tensor_array[-1].id
                    found = False
                    for temp_g in json_subgraphs:
                        temp_g_inputs = temp_g['subgraphs'][0]['operators'][0]['inputs']
                        if output_id in temp_g_inputs:
                            if found is False:
                                found = True
                            else:
                                inject_identity = True
                                break
                        
                if inject_identity is False:
                    prev_node.output_strides = [conv8.stride_height, conv8.stride_width]

            # If offloading this node to another processor anyway, no need to add an identity
            if node.offloaded:
                inject_identity = False

            if inject_identity:
                with exception_catcher( node.type, len(Nodes)):
                    prev_len = len(Nodes)
                    Nodes = inject_dummy_identity(Nodes, ids_with_dummies, 
                                                            node,
                                                            # op,
                                                            preset, 
                                                            # graph, 
                                                            opcode,
                                                            pad_hw,
                                                            inject_strided=conv8.use_strided, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)
                    for i in range(len(Nodes)-prev_len):
                        node_ids.append([-1])
                    


            # Currently disabling weight shaper double buffer for FC
            is_fc = (node.n == 1 and node.m == 1)
            conv8.mxp_double_buffer = 1
            conv8.split_weight_shaper_buffers = 0
            #split_weight_shaper_buffers only if not FC, not fit_weights, double_buffer, not depthwise
            if (not is_fc) and (not conv8.fit_weights) and conv8.mxp_double_buffer and (not conv8.use_depthwise):
                conv8.split_weight_shaper_buffers = SPLIT_WEIGHT_SHAPER_BUFFERS
            tile = allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)

            # if not full rows, then don't double buffer
            # previously disabling double buffer for k > 3 because the reduced scratchpad size causes
            # # the first k=6 layer of yolov5 to not fit all columns, and that gives incorrect results
            if node.n != tile[-1]:
                conv8.mxp_double_buffer = 0
                conv8.split_weight_shaper_buffers = 0
                tile = allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)

            assert not (tile is None)

            conv8.first_fia = 0
            conv8.last_fia = 0
            conv8.fia_collision = 0
            if conv8.use_fia:
                if first_fia is False:
                    conv8.first_fia = 1
                    first_fia = True

                conv8.last_fia = 1
                
                if prev_fia_node:
                    prev_fia_node.Conv2DOptions.last_fia = 0
                    prev_fia_node.FullyConnectedOptions.last_fia = 0

                check_node_for_collision(node, Nodes, prev_fia_node)

                prev_fia_node = node

            node.input_offset = input_offset
            node.output_offset = output_offset

            filter_data = get_numpy_data(f_tensor, buffers).transpose((0,3,1,2))

            # TODO dilated weights for depthwise for 2.1?
            if conv8.use_fia and conv8.use_depthwise and conv8.dilation_height_factor != 1 and h != 1:
                dilated_height = ((conv8.dilation_height_factor-1) * (h-1)) + h
                h = dilated_height
                filter_shape_dims = [k, c, h, w]
                conv8.filter_shape_dims = filter_shape_dims
                dilated_data = np.zeros(filter_shape_dims, dtype=np.int8)
                dilated_data[:, :, ::conv8.dilation_height_factor, :] = filter_data
                filter_data = dilated_data
                conv8.dilation_height_factor = 1

            bias_data = np.zeros((k,), dtype=np.int64)
            if not (b_tensor is None):
                bias_data = get_numpy_data(b_tensor, buffers).astype(np.int64)

            if conv8.use_fia or USE_PRECALC:
                if opcode == 'CONV_2D':
                    bias_data += precalculate_filter_input_bias(filter_data, input_offset, reduce=False)
                else:
                    bias_data += precalculate_filter_input_bias(filter_data, input_offset, reduce=True)
            # if PRECALC_OUTPUT: # 0
            #     bias_data += precalculate_output_offset_bias(effective_scale, output_offset)
            #     if conv8.use_fia:
            #         node.output_offset = 0

            bias_data = bias_data.clip(np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            bias_data = bias_data.astype(np.int32)

            # effective_scale unused (would be used in PRECALC_OUTPUT)
            if conv8.use_fia:
                _, output_multiplier, output_shift, c_input_L, c_input_H = quantize_two_math_block(output_offset, node.activation_min, node.activation_max, op, tensors, bias_data=bias_data)
            else:
                with exception_catcher( node.type, len(Nodes), op['inputs'][0]):
                    _, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(op, tensors)

            if len(output_multiplier) == 1 and conv8.kernels > 1:
                output_multiplier = [output_multiplier[0] for _ in range(conv8.kernels)]
            if len(output_shift) == 1 and conv8.kernels > 1:
                output_shift = [output_shift[0] for _ in range(conv8.kernels)]

            node.output_multiplier = len(weights)
            fmt = "{}i".format(len(output_multiplier))
            weights += struct.pack(fmt, *output_multiplier)

            node.output_shift = len(weights)
            fmt = "{}i".format(len(output_shift))
            weights += struct.pack(fmt, *output_shift)

            conv8.bias_data = len(weights)
            fmt = "{}i".format(len(bias_data))
            weights += struct.pack(fmt, *bias_data)

            _, _, _, _, _, rows, columns = tile
            weight_pad = 0

            # force byte alignment for weights
            if len(weights) % parallel_output_maps != 0:
                filler = parallel_output_maps - (len(weights) % parallel_output_maps)
                fmt = "{}x".format(filler)
                weights += struct.pack(fmt)
            conv8.filter_data = len(weights)
            conv8.repeat = -1
            filter_data = pad_weights_and_pack_2_1(filter_data, node, parallel_output_maps, tile, preset, opcode, weight_pad, \
                                                   sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)
            if t.type==calc_type.UINT8:
                fmt = "{}B".format(len(filter_data.flatten()))
            else:
                fmt = "{}b".format(len(filter_data.flatten()))
            weights += struct.pack(fmt, *filter_data.flatten())

            # pack quant records
            if len(weights) % QUANTIZATION_RECORD_WIDTH_BYTES != 0:
                filler = QUANTIZATION_RECORD_WIDTH_BYTES - (len(weights) % QUANTIZATION_RECORD_WIDTH_BYTES)
                fmt = "{}x".format(filler)
                weights += struct.pack(fmt)
            conv8.quantization_records = len(weights)

            if conv8.use_fia:
                for kernel in range(conv8.kernels):
                    c_input_L_mask = ((c_input_L[kernel] >> 6) & ((2**C_INPUT_L_WIDTH)-1)) << C_INPUT_L_LSB
                    c_input_H_mask = (c_input_H[kernel] & ((2**C_INPUT_H_WIDTH)-1)) << C_INPUT_H_LSB
                    mult_mask = (output_multiplier[kernel] & ((2**MULTIPLIER_WIDTH)-1)) << MULTIPLIER_LSB
                    shift_mask = (output_shift[kernel] & ((2**O_SHIFT_WIDTH)-1)) << O_SHIFT_LSB

                    quant_record = shift_mask | mult_mask | c_input_H_mask | c_input_L_mask
                    
                    weights += struct.pack("<Q", quant_record)

            conv8.nlf_data = -1
            if nlf_data is not None:
                conv8.nlf_data = len(weights)
                fmt = "{}b".format(len(nlf_data))
                weights += struct.pack(fmt, *nlf_data.flatten())
                op['outputs'] = lut_ops[-1][0]['outputs']
            elif 0: #force passthru NLF
                nlf_data = np.asarray([_ for _ in range(0,256)], dtype=np.uint8).astype(np.int8)
                conv8.nlf_data = len(weights)
                fmt = "{}b".format(len(nlf_data))
                weights += struct.pack(fmt, *nlf_data.flatten())


        elif node.type ==  BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM:

            allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)

        elif node.type ==  BuiltinOperator.TRANSPOSE_CONV:

            conv8 = node.Conv2DOptions

            conv8.in_used = -1
            conv8.w_used = -1
            conv8.qt_used = -1
            conv8.out_used = -1
            conv8.sp_used = -1

            conv8.direct_dma = 0
            conv8.fit_weights = 0
            conv8.use_depthwise = 0
            conv8.group = 1
            conv8.dilation_height_factor = 1
            conv8.dilation_width_factor = 1

            f_tensor = tensors[op['inputs'][1]]
            b_tensor = None
            if len(op['inputs']) > 3 and op['inputs'][3] != -1:
                b_tensor = tensors[op['inputs'][3]]

            k, h, w, c = tuple(f_tensor['shape'])
            filter_shape_dims = [k, c, h, w]
            conv8.filter_shape_dims = filter_shape_dims
            parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]
            if k*h*w*c <= (fia_weight_shaper_size_kb(parallel_output_maps)*1024):
                conv8.fit_weights = ALL_WEIGHTS

            conv8.kernels = k
            conv8.use_vector = USE_CONV_VECTOR
            conv8.use_fia = USE_CONV_VECTOR and USE_CONV_FIA
            conv8.use_db = USE_FIA_DB
            conv8.conv_rows = USE_FIA_DB

            conv8.use_strided = 0

            node.input_strides = [opts['stride_h'], opts['stride_w']]
            conv8.stride_width = opts['stride_w']
            conv8.stride_height = opts['stride_h']
            conv8.fused_activation = 0
            node.activation_max = 127
            node.activation_min = -128
            if opts['fused_activation_function'] == 'RELU':
                node.activation_min = output_offset
            elif opts['fused_activation_function'] == 'RELU6':
                node.activation_min = output_offset
                node.activation_max = round((6 / output_scale) + output_offset)
                node.activation_max = min(node.activation_max, 127)

            # pad (or modified pad) should only be injected if stride_check is True
            stride_check = (conv8.stride_height==conv8.stride_width and conv8.stride_width <= 2)
            pad_hw = None
            pad_h, pad_w = 0, 0
            if opts['padding'] == 'VALID':
                conv8.padding_width = 0
                conv8.padding_height = 0

                if conv8.use_fia and stride_check:
                    _, _, kh, kw = conv8.filter_shape_dims
                    pl = kw - 1
                    pr = kw - 1
                    pu = kh - 1
                    pd = kh - 1
                    pad_w = pl + pr
                    pad_h = pu + pd

                    # these values will be updated to 0 later if an identity+pad or just pad are injected explicitly
                    conv8.padding_width = pad_w
                    conv8.padding_height = pad_h

                    if pad_w != 0 or pad_h != 0:
                        pad_hw = [pad_h, pad_w]

            elif opts['padding'] == 'SAME':
                stride_h, stride_w = conv8.stride_height, conv8.stride_width
                kernel_h, kernel_w = conv8.filter_shape_dims[2], conv8.filter_shape_dims[3]

                i_h, i_w = i_tensor['shape'][-3], i_tensor['shape'][-2]
                o_h, o_w = o_tensor['shape'][-3], o_tensor['shape'][-2]

                if conv8.use_fia:
                    pad_h = ((i_h * stride_h) + (kernel_h-stride_h)) - o_h
                    pad_w = ((i_w * stride_w) + (kernel_w-stride_w)) - o_w
                    conv8.padding_width  = pad_w
                    conv8.padding_height = pad_h

                    if (pad_w > 0 or pad_h > 0) and stride_check:
                        pl = (kernel_w - (pad_w//2) - 1)
                        pr = (kernel_w - ((pad_w//2) + (pad_w%2)) - 1)
                        pu = (kernel_h - (pad_h//2) - 1)
                        pd = (kernel_h - ((pad_h//2) + (pad_h%2)) - 1)
                        pad_w = pl + pr
                        pad_h = pu + pd

                        # these values will be updated to 0 later if an identity+pad or just pad are injected explicitly
                        conv8.padding_width = pad_w
                        conv8.padding_height = pad_h

                        # for FIA, doing padding in injected layer is faster than doing padding in FIA which would force 2D or 3D DMA
                        if pad_w != 0 or pad_h != 0:
                            pad_hw = [pad_h, pad_w]

                else: # not FIA
                    pad_h = ((i_h * stride_h) + (kernel_h-1)) - o_h
                    pad_w = ((i_w * stride_w) + (kernel_w-1)) - o_w

                    conv8.padding_width = pad_w // 2
                    conv8.padding_height = pad_h // 2

            # identity+pad or just pad should only be injected if either:
            # 1) pad_hw is set and stride_check is True, and there is no previous subgraph, i.e. this Conv node is the first subgraph of the network
            # 2) strides are 2,2 and so the input must be dilated with a modified Pad
            inject_identity = False

            # find previous node
            input_id = node.tensor_array[0].id
            _,prev_node = get_prev_node(Nodes, node)

            if pad_hw is not None or (stride_check and conv8.stride_width == 2):
                if prev_node is None:
                    inject_identity = True
                elif input_id in in_id_to_out_ids.keys():
                    # TODO the input to this node is shared with another node, should skip and determine after populate_nodes if pad can be injected or identity must be injected
                    # until implemented, just continue inject_identity for these cases
                    inject_identity = True
                else: #input id not in in_id_to_out_ids.keys                    
                    inject_pad_subnode_to_previous_node(prev_node, node, ids_with_dummies, preset, opcode, weights, Nodes_to_graph_dict[prev_node], transpose_dilate=[conv8.stride_height, conv8.stride_width], sparse=compression_vbx, tmp_dir=tmp_dir)
                # elif len(prev_node.subnode_array) == 0:
                #     inject_identity = True
                # elif prev_node.subnode_array[-1].type != BuiltinOperator.PAD:
                #     inject_identity = True

            if inject_identity:
                with exception_catcher( node.type, len(Nodes)):
                    prev_len = len(Nodes)
                    Nodes = inject_dummy_identity(Nodes, ids_with_dummies, 
                                                            node,
                                                            # op,
                                                            preset, 
                                                            # graph, 
                                                            opcode,
                                                            pad_hw,
                                                            inject_strided=0,
                                                            transpose_dilate=[conv8.stride_height, conv8.stride_width],
                                                            sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)
                    for i in range(len(Nodes)-prev_len):
                        node_ids.append([-1])

            # if doing FIA and stride_check is True, then TransposeConv can be run as regular Conv, and use Conv2D FIA C code
            if conv8.use_fia and stride_check:
                node.type = BuiltinOperator.CONV_2D

            node.input_offset = input_offset
            node.output_offset = output_offset

            conv8.imaps = -1
            conv8.mxp_double_buffer = 1
            conv8.split_weight_shaper_buffers = 0
            tile = allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)
            assert not (tile is None)

            conv8.first_fia = 0
            conv8.last_fia = 0
            conv8.fia_collision = 0
            if conv8.use_fia:
                if first_fia is False:
                    conv8.first_fia = 1
                    first_fia = True

                conv8.last_fia = 1
                
                if prev_fia_node:
                    prev_fia_node.Conv2DOptions.last_fia = 0
                    prev_fia_node.FullyConnectedOptions.last_fia = 0

                check_node_for_collision(node, Nodes, prev_fia_node)

                prev_fia_node = node

            filter_data = get_numpy_data(f_tensor, buffers).transpose((0,3,1,2))
            if conv8.use_fia:
                # rotate 2d weights 90 degrees clockwise
                filter_data = filter_data.transpose((0,1,3,2))
                filter_data = np.flip(filter_data, axis=3)
                # rotate 90 more degrees
                filter_data = filter_data.transpose((0,1,3,2))
                filter_data = np.flip(filter_data, axis=3)

            bias_data = np.zeros((k,), dtype=np.int64)
            if not (b_tensor is None):
                bias_data = get_numpy_data(b_tensor, buffers)

            if USE_PRECALC:
                # bias_data += precalculate_filter_input_bias(filter_data, input_offset, reduce=True)
                bias_data += precalculate_filter_input_bias(filter_data, input_offset, reduce=False)
            # if PRECALC_OUTPUT: # 0
            #     bias_data += precalculate_output_offset_bias(effective_scale, output_offset)
            #     if conv8.use_fia:
            #         node.output_offset = 0
                
            # effective_scale unused (would be used in PRECALC_OUTPUT)
            if conv8.use_fia:
                _, output_multiplier, output_shift, c_input_L, c_input_H = quantize_two_math_block(output_offset, node.activation_min, node.activation_max, \
                                                                                                  i_tensor=i_tensor, o_tensor=o_tensor, f_tensor=f_tensor, bias_data=bias_data)
            else:
                _, output_multiplier, output_shift = get_effective_quantized_multiplier_from_tensors(i_tensor, o_tensor, f_tensor)

            if len(output_multiplier) == 1 and conv8.kernels > 1:
                output_multiplier = [output_multiplier[0] for _ in range(conv8.kernels)]
            if len(output_shift) == 1 and conv8.kernels > 1:
                output_shift = [output_shift[0] for _ in range(conv8.kernels)]

            node.output_multiplier = len(weights)
            fmt = "{}i".format(len(output_multiplier))
            weights += struct.pack(fmt, *output_multiplier)

            node.output_shift = len(weights)
            fmt = "{}i".format(len(output_shift))
            weights += struct.pack(fmt, *output_shift)

            conv8.bias_data = len(weights)
            fmt = "{}i".format(len(bias_data))
            weights += struct.pack(fmt, *bias_data)

            # force byte alignment for weights
            if len(weights) % parallel_output_maps != 0:
                filler = parallel_output_maps - (len(weights) % parallel_output_maps)
                fmt = "{}x".format(filler)
                weights += struct.pack(fmt)
            conv8.filter_data = len(weights)
            conv8.repeat = -1
            filter_data = pad_weights_and_pack_2_1(filter_data, node, parallel_output_maps, tile, preset, opcode, is_transpose=True, \
                                                   sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)
            if t.type==calc_type.UINT8:
                fmt = "{}B".format(len(filter_data.flatten()))
            else:
                fmt = "{}b".format(len(filter_data.flatten()))
            weights += struct.pack(fmt, *filter_data.flatten())

            # used for FIA
            # TODO maybe redundant as we pack bias, output multiplier, and output shift earlier
            # TODO future optimization if VNNX is too large then clean up redundancy
            # force byte alignment for records
            if len(weights) % QUANTIZATION_RECORD_WIDTH_BYTES != 0:
                filler = QUANTIZATION_RECORD_WIDTH_BYTES - (len(weights) % QUANTIZATION_RECORD_WIDTH_BYTES)
                fmt = "{}x".format(filler)
                weights += struct.pack(fmt)
            conv8.quantization_records = len(weights)

            if conv8.use_fia:
                for kernel in range(conv8.kernels):
                    c_input_L_mask = ((c_input_L[kernel] >> 6) & ((2**C_INPUT_L_WIDTH)-1)) << C_INPUT_L_LSB
                    c_input_H_mask = (c_input_H[kernel] & ((2**C_INPUT_H_WIDTH)-1)) << C_INPUT_H_LSB
                    mult_mask = (output_multiplier[kernel] & ((2**MULTIPLIER_WIDTH)-1)) << MULTIPLIER_LSB
                    shift_mask = (output_shift[kernel] & ((2**O_SHIFT_WIDTH)-1)) << O_SHIFT_LSB

                    quant_record = shift_mask | mult_mask | c_input_H_mask | c_input_L_mask
                    
                    weights += struct.pack("<Q", quant_record)

            conv8.nlf_data = -1
            if nlf_data is not None:
                conv8.nlf_data = len(weights)
                fmt = "{}b".format(len(nlf_data))
                weights += struct.pack(fmt, *nlf_data.flatten())
                op['outputs'] = lut_ops[-1][0]['outputs']

        elif node.type == BuiltinOperator.FULLY_CONNECTED:

            if idims >= 3:
                node.channels = t.shape[-2]
                node.m = t.shape[-1]
                node.n = t.shape[-3]
            elif idims >= 2:
                node.channels = 1
                node.m = t.shape[-2]
                node.n = t.shape[-1]
            elif idims >= 1:
                node.channels = 1
                node.m = 1
                node.n = t.shape[-1]

            f_tensor = tensors[op['inputs'][1]]
            b_tensor = None
            if len(op['inputs']) > 2 and op['inputs'][2] != -1:
                b_tensor = tensors[op['inputs'][2]]

            fc8 = node.FullyConnectedOptions

            output_depth, accum_depth = tuple(f_tensor['shape'])
            fc8.filter_shape_dims = [output_depth, accum_depth]
            if engine_graphs_nx != None:
                fc8.use_fia = 0
            else:
                fc8.use_fia = USE_CONV_VECTOR and USE_CONV_FIA
            fc8.mxp_double_buffer = 1

            node.activation_max = 127
            node.activation_min = -128
            if opts['fused_activation_function'] == 'RELU':
                node.activation_min = output_offset

            node.input_offset = input_offset
            node.output_offset = output_offset

            tile = allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)
            assert not (tile is None)

            fc8.first_fia = 0
            fc8.last_fia = 0
            fc8.fia_collision = 0
            if fc8.use_fia:
                if first_fia is False:
                    fc8.first_fia = 1
                    first_fia = True

                fc8.last_fia = 1
                
                if prev_fia_node:
                    prev_fia_node.Conv2DOptions.last_fia = 0
                    prev_fia_node.FullyConnectedOptions.last_fia = 0

                check_node_for_collision(node, Nodes, prev_fia_node)

                prev_fia_node = node


            filter_data = get_numpy_data(f_tensor, buffers)

            bias_data = np.zeros((output_depth,), dtype=np.int64)
            if not (b_tensor is None):
                bias_data = get_numpy_data(b_tensor, buffers)

            if USE_PRECALC:
                res = precalculate_filter_input_bias(filter_data, input_offset)
                bias_data = bias_data + res
            # if PRECALC_OUTPUT: # 0
            #     bias_data += precalculate_output_offset_bias(effective_scale, output_offset)
            #     if fc8.use_fia:
            #         node.output_offset = 0

            # effective_scale unused (would be used in PRECALC_OUTPUT)
            if fc8.use_fia:
                _, output_multiplier, output_shift, c_input_L, c_input_H = quantize_two_math_block(output_offset, node.activation_min, node.activation_max, op, tensors, bias_data=bias_data)
            else:
                with exception_catcher( node.type, len(Nodes), op['inputs'][0]):
                    _, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(op, tensors)

            node.output_multiplier = len(weights)
            fmt = "{}i".format(len(output_multiplier))
            weights += struct.pack(fmt, *output_multiplier)

            node.output_shift = len(weights)
            fmt = "{}i".format(len(output_shift))
            weights += struct.pack(fmt, *output_shift)

            fc8.bias_data = len(weights)
            fmt = "{}i".format(len(bias_data))
            weights += struct.pack(fmt, *bias_data)

            fc8.filter_data = len(weights)
            if fc8.use_fia:
                filter_data = pad_weights_and_pack_2_1(filter_data, node, preset_select['PARALLEL_OUTPUT_MAPS'][preset], tile, preset, opcode, \
                                                       sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)

            fmt = "{}b".format(len(filter_data.flatten()))
            weights += struct.pack(fmt, *filter_data.flatten())

            # used for FIA
            # TODO maybe redundant as we pack bias, output multiplier, and output shift earlier
            # TODO future optimization if VNNX is too large then clean up redundancy
            # TODO for fully connected, clean up redundancy of storing same multiplier+shift for each record
            # force byte alignment for records
            if len(weights) % QUANTIZATION_RECORD_WIDTH_BYTES != 0:
                filler = QUANTIZATION_RECORD_WIDTH_BYTES - (len(weights) % QUANTIZATION_RECORD_WIDTH_BYTES)
                fmt = "{}x".format(filler)
                weights += struct.pack(fmt)
            fc8.quantization_records = len(weights)

            if fc8.use_fia:
                for out in range(output_depth):
                    c_input_L_mask = ((c_input_L[out] >> 6) & ((2**C_INPUT_L_WIDTH)-1)) << C_INPUT_L_LSB
                    c_input_H_mask = (c_input_H[out] & ((2**C_INPUT_H_WIDTH)-1)) << C_INPUT_H_LSB
                    mult_mask = (output_multiplier[0] & ((2**MULTIPLIER_WIDTH)-1)) << MULTIPLIER_LSB
                    shift_mask = (output_shift[0] & ((2**O_SHIFT_WIDTH)-1)) << O_SHIFT_LSB

                    quant_record = shift_mask | mult_mask | c_input_H_mask | c_input_L_mask
                    
                    weights += struct.pack("<Q", quant_record)

            fc8.nlf_data = -1
            if nlf_data is not None:
                fc8.nlf_data = len(weights)
                fmt = "{}b".format(len(nlf_data))
                weights += struct.pack(fmt, *nlf_data.flatten())
                op['outputs'] = lut_ops[-1][0]['outputs']              

        elif node.type == BuiltinOperator.BATCH_MATMUL: #TODO not adding explicit tensor
            for t in op['inputs'][1:]:
                tensor = tensors[t]
                offset = tensor['quantization']['zero_point'][0]
                scale = tensor['quantization']['scale'][0]
                shape, dims = channels_first_shape(tensor['shape'])

                tn = Tensor()
                tn.type = calc_type.from_str(tensor['type'])
                tn.shape = shape
                tn.dims = dims
                tn.scale = scale
                tn.zero = offset
                tn.id = tensor['buffer'] - 1
                tn.name = tensor['name']
                if not node.offloaded and engine_graphs_nx:
                    add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)
                node.tensor_array.append(tn)

            node.m, node.n = max(node.m, shape[-2]), max(node.n, shape[-1]) #increase size (to handle channelwise broadcast) 

            node.num_inputs = len(node.tensor_array)

            node.BatchMatMulOptions.adj_x = 1
            node.BatchMatMulOptions.adj_y = 1
            node.BatchMatMulOptions.asym = 1

            allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)

        elif node.type == VNNXOperator.IDENTITY: #TODO not adding explicit tensor
            if multi_input:
                if node.subnode_array[0].type == VNNXOperator.ELTWISE:
                    node.subnode_array[0].eltwise8.swap = False

                for t in op['inputs'][1:]:
                    tensor = tensors[t]
                    offset = tensor['quantization']['zero_point'][0]
                    scale = tensor['quantization']['scale'][0]
                    shape, dims = channels_first_shape(tensor['shape'])

                    tn = Tensor()
                    tn.type = calc_type.from_str(tensor['type'])
                    tn.shape = shape
                    tn.dims = dims
                    tn.scale = scale
                    tn.zero = offset
                    tn.id = tensor['buffer'] - 1
                    tn.name = tensor['name']
                    if not node.offloaded and engine_graphs_nx:
                        add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)
                    node.tensor_array.append(tn)

                node.m, node.n = max(node.m, shape[-2]), max(node.n, shape[-1]) #increase size (to handle channelwise broadcast) 

            node.num_inputs = len(node.tensor_array)
            allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)

        elif node.type == BuiltinOperator.CONCATENATION:
            o_tensor = tensors[op['outputs'][0]]
            axis = op['builtin_options']['axis']
            node.ConcatOptions.axis = channels_first_axis(axis, len(o_tensor['shape']))

            for t in op['inputs'][1:]:
                tensor = tensors[t]
                offset = tensor['quantization']['zero_point'][0]
                scale = tensor['quantization']['scale'][0]
                shape, dims = channels_first_shape(tensor['shape'])

                tn = Tensor()
                tn.type = calc_type.from_str(tensor['type'])
                tn.shape = shape
                tn.dims = dims
                tn.scale = scale
                tn.zero = offset
                tn.id = tensor['buffer'] - 1
                tn.name = tensor['name']
                if not node.offloaded and engine_graphs_nx:
                    add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)
                node.tensor_array.append(tn)

            output_shape, odims = channels_first_shape(o_tensor['shape'])

            node.channels = 1
            if odims > 2:
                node.channels = output_shape[-3]
            node.m = 1
            if odims > 1:
                node.m = output_shape[-2]
            node.n = output_shape[-1]
            node.num_inputs = len(node.tensor_array)
            allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)

        elif node.type == BuiltinOperator.PACK:
            idims = len(i_tensor['shape'])
            odims = len(o_tensor['shape'])
            node.PackOptions.axis = channels_first_axis(opts['axis'], odims) - odims
            node.PackOptions.count = opts['values_count']
            node.PackOptions.dims = idims

            assert(check_pack(i_tensor['shape'], o_tensor['shape'], opts['axis'], node.PackOptions.axis, node.PackOptions.count))

            for t in op['inputs'][1:]:
                tensor = tensors[t]
                offset = tensor['quantization']['zero_point'][0]
                scale = tensor['quantization']['scale'][0]
                shape, dims = channels_first_shape(tensor['shape'])

                tn = Tensor()
                tn.type = calc_type.from_str(tensor['type'])
                tn.shape = shape
                tn.dims = dims
                tn.scale = scale
                tn.zero = offset
                tn.id = tensor['buffer'] - 1
                tn.name = tensor['name']
                if not node.offloaded and engine_graphs_nx:
                    add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)
                node.tensor_array.append(tn)

            node.num_inputs = len(node.tensor_array)

            allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)

        elif node.type == BuiltinOperator.UNPACK:
            input_shape, idims = channels_first_shape(i_tensor['shape'])
            node.PackOptions.axis = channels_first_axis(opts['axis'], idims) - idims
            node.PackOptions.count = opts['num']
            node.PackOptions.dims = idims

            assert(check_unpack(i_tensor['shape'], o_tensor['shape'],
                                opts['axis'], node.PackOptions.axis, opts['num']))
            allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)
       
        elif node.type in [BuiltinOperator.SPLIT, BuiltinOperator.SPLIT_V]:
            if node.type == BuiltinOperator.SPLIT_V:
                axis_tensor = tensors[op['inputs'][2]]
            else:
                axis_tensor = tensors[op['inputs'][0]]
            dtype = np.dtype(axis_tensor['type'].lower())
            raw_data = bytearray(buffers[axis_tensor['buffer']]['data'])
            axis = np.frombuffer(raw_data, dtype=dtype)[0]

            num_splits = len(op['outputs'])
            if node.type == BuiltinOperator.SPLIT_V:
                split_tensor = tensors[op['inputs'][1]]
                dtype = np.dtype(split_tensor['type'].lower())
                raw_data = bytearray(buffers[split_tensor['buffer']]['data'])
                splits = np.frombuffer(raw_data, dtype=dtype)
            else:
                num_splits = len(op['outputs'])
                splits = [i_tensor['shape'][axis] // num_splits for _ in range(num_splits)]

            idims = len(i_tensor['shape'])
            node.SplitOptions.axis = channels_first_axis(axis, idims) - idims
            node.num_outputs = num_splits

            node.SplitOptions.splits = len(weights)
            node.SplitOptions.max_split = max(splits)

            fmt = "{}i".format(len(splits))
            weights += struct.pack(fmt, *splits)

            allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)

        elif node.type == BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
            i_tensor = tensors[op['inputs'][0]]
            o_tensor = tensors[op['outputs'][0]]
            s_tensor = tensors[op['inputs'][1]]

            resize = node.ResizeOptions
            resize.mode = resize_mode.NEAREST 
            resize.b_postproc_tiling = 0

            new_size_data = get_numpy_data(s_tensor, buffers)
            resize.scale = (new_size_data / i_tensor['shape'][1:3]).tolist() # height and width scales
            # resize.scale = ( i_tensor['shape'][1:3] /new_size_data).tolist()
            # print("resize.scale == ", resize.scale)

            allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)
        elif node.type == BuiltinOperator.RESIZE_BILINEAR:
            i_tensor = tensors[op['inputs'][0]]
            o_tensor = tensors[op['outputs'][0]]
            s_tensor = tensors[op['inputs'][1]]

            resize = node.ResizeOptions
            resize.mode = resize_mode.LINEAR
            resize.b_postproc_tiling = 0
            if 'postprocessing' in s_tensor['name']:
                resize.b_postproc_tiling = 1
            new_size_data = get_numpy_data(s_tensor, buffers)
            resize.scale = (new_size_data / i_tensor['shape'][1:3]).tolist() # height and width scales
            # resize.scale = ( i_tensor['shape'][1:3] /new_size_data).tolist()


            # TODO m and n are currently expected to be shapes of the outputs in the c code
            node.m = output_shapes[0][-3]
            node.n = output_shapes[0][-2]
            allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)

        elif node.type == BuiltinOperator.TILE:
            i_tensor = tensors[op['inputs'][0]]
            o_tensor = tensors[op['outputs'][0]]
            m_tensor = tensors[op['inputs'][1]]
            multipliers = get_numpy_data(m_tensor, buffers) 
            if idims == 1:
                multipliers =  [1, 1, 1, multipliers[0]]
            elif idims == 2: 
                multipliers =  [1, 1, multipliers[0], multipliers[1]]
            elif idims == 3: #NHWC -> NCHW
                multipliers =  [1, multipliers[-1], multipliers[-3], multipliers[-2]]
            else: #NHWC -> NCHW
                multipliers =  [multipliers[-4], multipliers[-1], multipliers[-3], multipliers[-2]]
            node.TileOptions.tile = multipliers

            for t in op['inputs'][1:]:
                tensor = tensors[t]
                offset = 0
                scale = 0
                shape, dims = channels_first_shape(tensor['shape'])

                tn = Tensor()
                tn.type = calc_type.from_str(tensor['type'])
                tn.shape = shape
                tn.dims = dims
                tn.scale = 0
                tn.zero = 0
                tn.id = tensor['buffer'] - 1
                tn.name = tensor['name']
                if not node.offloaded and engine_graphs_nx:
                    add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)
                node.tensor_array.append(tn)

            node.num_inputs = len(node.tensor_array)

            allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj)
        
        else:
            sys.stderr.write("ERROR: no parsing for {}\n".format(node.type))
            sys.exit(1)
                    
        node.num_outputs = len(op['outputs'])
        for t in op['outputs']:
            tensor = tensors[t]
            offset = tensor['quantization'].get('zero_point', [0])[0]
            scale = tensor['quantization'].get('scale', [0.0])[0]
            shape, dims = channels_first_shape(tensor['shape'])

            tn = Tensor()
            tn.type = calc_type.from_str(tensor['type'])
            tn.shape = shape
            tn.dims = dims
            tn.scale = scale
            tn.zero = offset
            tn.id = tensor['buffer'] - 1
            tn.name = tensor['name']
            if not node.offloaded and engine_graphs_nx:
                add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, g)

            if node.type == BuiltinOperator.CONV_2D and node.Conv2DOptions.use_fia:
                if node.Conv2DOptions.use_depthwise and node.Conv2DOptions.stride_width != 1:
                    inject_strided_slice(node, tn, subnode_tensor_array, ids_with_dummies)
                    
                    tile = allocation(node, preset, opcode, sparse=compression_vbx, tmp_dir=tmp_dir, graph_idx=g, tmp_dir_obj=tmp_dir_obj) # re-allocate as stride_width forced to 1
                    assert not (tile is None)

                oh, ow, osh, osw = get_output_shapes(node.Conv2DOptions, node.rows_0, node.cols_0)
                if (oh*ow != osh*osw) and (not node.offloaded):
                    inject_output_shaper(node, tn, subnode_tensor_array, ids_with_dummies)

            node.tensor_array.append(tn)

        assert(node.num_inputs + node.num_outputs == len(node.tensor_array))

        # TODO does this need to apply to injected nodes?
        if node.type == VNNXOperator.IDENTITY and node.num_inputs == 1: #injected identity, rename tensors TODO fix for multi-input
            node_input_name = node.tensor_array[0].name
            node.tensor_array[1].name = node_input_name + '.id'
            node.tensor_array[1].shape = node.tensor_array[0].shape
            for s, t in enumerate(subnode_tensor_array):
                if t.name == node_input_name:
                    subnode_tensor_array[s].name = node.tensor_array[1].name

        node.tensor_array += subnode_tensor_array
        node.num_tensors = len(node.tensor_array)
        
        Nodes.append(node)

        if (not node.offloaded):
            for idx in range(node.num_tensors):
                t = node.tensor_array[idx]
                if (t.external_producer or t.external_consumer):
                    shape = list(t.shape)
                    if node.type == BuiltinOperator.FULLY_CONNECTED:
                        width = node.m
                    else:
                        width = node.n
                    if (shape[-1] != width):
                        shape[-1] = node.n
                        t.shape = tuple(shape)
                        
    Nodes = update_nodes_with_dequant_params(Nodes, dequantize_op_params)
    return Nodes


def get_prev_node(Nodes, node):
    input_id = node.tensor_array[0].id
    for p,pnode in enumerate(Nodes):
        for o in range(pnode.num_outputs):
            out_id = pnode.tensor_array[pnode.num_inputs + o].id
            if input_id == out_id:
                return p,pnode
    return -1,None


def hard_swish(x):
    if x < -3:
        return 0
    elif x > 3:
        return x
    else:
        return x * (x + np.float32(3.)) * np.float32(1./ 6.)


def lut_func(code, scale=None, param=None):
    if code == "SILU":
        fn = lambda s: s*sigmoid(s)
    elif code == "QUANTIZE":
        fn = lambda s: s
    elif code == "LOGISTIC":
        fn = lambda s: sigmoid(s)
    elif code == "HARD_SWISH":
        fn = lambda s: s * tf.nn.relu6(s + np.float32(3.)) * np.float32(1./ 6.)
        # fn = hard_swish
    elif code == 'LEAKY_RELU':
        fn = lambda s: tf.nn.leaky_relu(s, scale)
    elif code == 'RELU':
        fn = lambda s: tf.nn.relu(s)
    elif code == 'RELU6':
        fn = lambda s: tf.nn.relu6(s)
    elif code == 'RELU_0_TO_1':
        fn = lambda s: min(1.0, tf.nn.relu(s))
    elif code == "MUL":
        fn = lambda s: tf.multiply(s, scale)
    elif code == "ADD":
        fn = lambda s: tf.add(s, scale)
    elif code == "SUB":
        fn = lambda s: tf.subtract(s, scale)
    elif code == "POST_PROCESSING":
        fn = lambda s: tf.gather(s, param, axis=1)
    elif code == "SQUARED_DIFFERENCE":
        print("lut_func")
        fn = lambda s: tf.math.squared_difference(s, scale)
    return fn


def populate_subnodes(subcode, ops, lut_ops, reshape_ops, subops, prev_subop, graph_activations, tensors, buffers, weights, dequantize_op_params, aliased_ids, tmp_dir,\
engine_graphs_nx, external_inputs, external_outputs, already_external_producer, already_external_consumer, node_offloaded, subgraph_idx):
    subnode_array = []
    subnode_tensor_array = []
    sn = Subnode()
    subop = subops[0]
    input_buffers = []
    for _ in subop['inputs']:
        if 'buffer' in tensors[_]:
            input_buffers += [buffers[tensors[_]['buffer']]]
    multi_input = len(input_buffers) > 1 and not any(['data' in _ for _ in input_buffers])

    i_acts, _ = op_activations(subop, graph_activations)
    _, o_acts = op_activations(subops[-1], graph_activations)

    i_tensor = graph_activations[i_acts[0]]
    o_tensor = graph_activations[o_acts[0]]

    # force undefined batch to 1 TODO move to preprocess
    if i_tensor['shape'][0] == -1:
        i_tensor['shape'][0] = 1
    if o_tensor['shape'][0] == -1:
        o_tensor['shape'][0] = 1
    
    input_offset = 0
    input_scale = 1.
    if 'quantization' in i_tensor.keys() and 'zero_point' in i_tensor['quantization'].keys():
        input_offset = i_tensor['quantization']['zero_point'][0]
    if 'quantization' in i_tensor.keys() and 'scale' in i_tensor['quantization'].keys():
        input_scale = i_tensor['quantization']['scale'][0]

    output_offset = 0
    output_scale = 1.
    if 'quantization' in o_tensor.keys() and 'zero_point' in o_tensor['quantization'].keys():
        output_offset = o_tensor['quantization']['zero_point'][0]
    if 'quantization' in o_tensor.keys() and 'scale' in o_tensor['quantization'].keys():
        output_scale = o_tensor['quantization']['scale'][0]

    output_type = o_tensor['type']
    sn.output_data_type = calc_type.from_str(output_type)

    sn.output_offset = output_offset

    input_shape, idims = channels_first_shape(i_tensor['shape'])
    output_shape, odims = channels_first_shape(o_tensor['shape'])

    input_type = i_tensor['type']
    sn.input_data_type = calc_type.from_str(input_type)

    if not subcode in ['LUT', 'QUANTIZE', 'GATHER', 'CAST', 'TOPK_V2', 'RESHAPE', 'NEG']:
        assert(sn.input_data_type == calc_type.INT8 or sn.input_data_type == calc_type.UINT8)

    sn.activation_min = -128
    sn.activation_max = 127

    tn = Tensor()
    tn.type = sn.input_data_type
    tn.shape = input_shape
    tn.dims = idims
    tn.scale = input_scale
    tn.zero = input_offset
    tn.id = i_tensor['buffer'] - 1
    tn.name = i_tensor['name']
    if not node_offloaded and engine_graphs_nx:
        add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, subgraph_idx)
    sn.tensor_array.append(tn)

    opts = None
    if 'builtin_options' in subop:
        opts = subop['builtin_options']

    if subcode == 'AVERAGE_POOL_2D': # functions as MEAN operator
        if output_shape[-2] == 1 and output_shape[-1] == 1:
            subcode = 'MEAN'


    if subcode == 'DEPTHWISE_CONV_2D':
        sn.type = BuiltinOperator.DEPTHWISE_CONV_2D

        conv8 = sn.Conv2DOptions
        conv8.direct_dma = 0

        f_tensor = tensors[subop['inputs'][1]]
        b_tensor = None
        if len(op['inputs']) > 2 and op['inputs'][2] != -1:
            b_tensor = tensors[op['inputs'][2]]

        k, h, w, c = tuple(f_tensor['shape'])
        filter_shape_dims = [k, c, h, w]
        conv8.filter_shape_dims = filter_shape_dims

        conv8.kernels = k
        conv8.fit_weights = 0
        conv8.use_depthwise = 0
        conv8.group = 1
        if k == 1 and c > 1 and subcode == 'DEPTHWISE_CONV_2D':
            conv8.kernels = c
            conv8.group = c
            conv8.use_depthwise = 1


            sn.input_strides = [opts['stride_h'], opts['stride_w']]
            conv8.stride_width = opts['stride_w']
            conv8.stride_height = opts['stride_h']
            conv8.dilation_height_factor = opts['dilation_h_factor']
            conv8.dilation_width_factor = opts['dilation_w_factor']
            conv8.fused_activation = 0
            sn.activation_max = 127
            sn.activation_min = -128
            if opts['fused_activation_function'] == 'RELU':
                sn.activation_min = output_offset
            elif opts['fused_activation_function'] == 'RELU6':
                sn.activation_min = output_offset
                sn.activation_max = round((6 / output_scale) + output_offset)
                sn.activation_max = min(sn.activation_max, 127)

            conv8.use_strided = 0
            if OPTIMIZED_DEPTHWISE:
                conv8.use_vector = 2 #OPTIMIZED
            else:
                conv8.use_vector = 1
            conv8.use_fia = False
            conv8.use_db = 0
            conv8.conv_rows = 0

            assert(opts['padding'] == 'VALID')

            if opts['padding'] == 'VALID':
                conv8.padding_width = 0
                conv8.padding_height = 0
            

            conv8.imaps = -1

            sn.input_offset = input_offset
            sn.output_offset = output_offset

            filter_data = get_numpy_data(f_tensor, buffers).transpose((0,3,1,2))

            bias_data = np.zeros((k,), dtype=np.int64)
            if not (b_tensor is None):
                bias_data = get_numpy_data(b_tensor, buffers).astype(np.int64)

            if USE_PRECALC:
                if subcode == 'CONV_2D':
                    bias_data += precalculate_filter_input_bias(filter_data, input_offset, reduce=False)
                else:
                    bias_data += precalculate_filter_input_bias(filter_data, input_offset, reduce=True)
            # if PRECALC_OUTPUT: # 0
            #     bias_data += precalculate_output_offset_bias(effective_scale, output_offset)

            bias_data = bias_data.clip(np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            bias_data = bias_data.astype(np.int32)

            # effective_scale unused (would be used in PRECALC_OUTPUT)
            if conv8.use_fia:
                _, output_multiplier, output_shift, c_input_L, c_input_H = quantize_two_math_block(output_offset, node.activation_min, node.activation_max, subop, tensors, bias_data=bias_data)
            else:
                _, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(subop, tensors)

            if len(output_multiplier) == 1 and conv8.kernels > 1:
                output_multiplier = [output_multiplier[0] for _ in range(conv8.kernels)]
            if len(output_shift) == 1 and k > 1:
                output_shift = [output_shift[0] for _ in range(conv8.kernels)]

            sn.output_multiplier = len(weights)
            fmt = "{}i".format(len(output_multiplier))
            weights += struct.pack(fmt, *output_multiplier)

            sn.output_shift = len(weights)
            fmt = "{}i".format(len(output_shift))
            weights += struct.pack(fmt, *output_shift)

            conv8.bias_data = len(weights)
            fmt = "{}i".format(len(bias_data))
            weights += struct.pack(fmt, *bias_data)

            conv8.filter_data = len(weights)
            fmt = "{}b".format(len(filter_data.flatten()))
            weights += struct.pack(fmt, *filter_data.flatten())

            conv8.quantization_records = -1
            conv8.nlf_data = -1

        subnode_array.append(sn)

    elif subcode in ['ADD', 'SUB', 'MUL', 'DIV', "GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL", "EQUAL", "NOT_EQUAL", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM"] and multi_input:  # >>>>>>>>> Main ELTWISE BLOCK with multi_inputs
       
        sn.type = VNNXOperator.ELTWISE
        eltwise8 = sn.eltwise8
        eltwise8.type = getattr(eltwise_type, f"ELTWISE_{subcode}")
        if eltwise8.type is None:
            raise ValueError(f"Invalid input string: {subcode}")

        i_tensor = tensors[subop['inputs'][0]] # left
        i2_tensor = tensors[subop['inputs'][1]] # right
        # Node brings in input2, assumed to be later 
        # eltwise8.swap = subop['inputs'][1] < subop['inputs'][0]

        eltwise8.swap = False
        op_outputs = []
        for op in ops:
            if op == subop:
                break
            else:
                op_outputs += op['outputs']

        if subop['inputs'][0] in op_outputs:
            eltwise8.swap = True

        o_tensor = tensors[subop['outputs'][0]] 
        if subcode in ['MUL', 'ADD', 'SUB', 'DIV', 'MINIMUM', 'MAXIMUM', 'SQUARED_DIFFERENCE']:
            output_offset = o_tensor['quantization']['zero_point'][0] 
            output_scale = o_tensor['quantization']['scale']
        else:
            output_offset = 0
            output_scale = [1.]

        input_scale = i_tensor['quantization']['scale']
        input_offset = i_tensor['quantization']['zero_point'][0]
        input2_scale = i2_tensor['quantization']['scale']
        input2_offset = i2_tensor['quantization']['zero_point'][0]

        eltwise8.optimized = 0
        if subcode == 'ADD' and (2*input_scale[0]/output_scale[0]) < 2**(15-Q16) and (2*input2_scale[0]/output_scale[0]) < 2**(15-Q16):
            if len(i_tensor['shape']) > 3: # TODO fix post-procesing accuracy drop
                eltwise8.optimized = OPTIMIZED_ADD
        else:
            if OPTIMIZED_ADD and VERBOSE:
                print(subcode, input_scale[0]/output_scale[0], input2_scale[0]/output_scale[0], "WARNING can't optimize multi")

        max_input_scale = input_scale[0]
        for t in subop['inputs'][1:]:
            tensor = tensors[t]

            ishape, idims = channels_first_shape(tensor['shape'])
            tn = Tensor()
            tn.type = calc_type.from_str(tensor['type'])
            tn.shape = ishape
            tn.dims = idims
            tn.scale = tensor['quantization']['scale'][0]
            if tn.scale > max_input_scale:
                max_input_scale = tn.scale
            tn.zero = tensor['quantization']['zero_point'][0]
            tn.id = tensor['buffer'] - 1
            tn.name = tensor['name']
            if not node_offloaded and engine_graphs_nx:
                add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, subgraph_idx)
            sn.tensor_array.append(tn)

        twice_max_input_scale = 2 * max_input_scale

        if subcode in ['ADD', 'SUB']:
            left_shift = 15
            if 'pot_scale_int16' in opts and opts['pot_scale_int16']:
                left_shift = 20
            twice_max_input_scale = 2 * max(input_scale[0], input2_scale[0])
            real_input_multiplier = (np.asarray(input_scale) / twice_max_input_scale).tolist()
            real_input2_multiplier = (np.asarray(input2_scale) / twice_max_input_scale).tolist()
            real_output_multiplier = (twice_max_input_scale / (2**left_shift * np.asarray(output_scale))).tolist()

            input_multiplier, input_shift = get_quantized_multiplier(real_input_multiplier)
            input2_multiplier, input2_shift = get_quantized_multiplier(real_input2_multiplier)
            output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)

            sn.activation_max = 127
            sn.activation_min = -128

            if opts['fused_activation_function'] == 'RELU':
                sn.activation_min = output_offset

        elif subcode == 'SQUARED_DIFFERENCE':
            left_shift = 7
            twice_max_input_scale = 2 * max(input_scale[0], input2_scale[0])
            real_input_multiplier = (np.asarray(input_scale) / twice_max_input_scale).tolist()
            real_input2_multiplier = (np.asarray(input2_scale) / twice_max_input_scale).tolist()
            # real_output_multiplier = ((twice_max_input_scale * twice_max_input_scale) /
            #                             (2** left_shift*2 * np.asarray(output_scale))).tolist()
            real_output_multiplier = ((twice_max_input_scale*twice_max_input_scale) / ((1<<left_shift * 2) * np.asarray(output_scale))).tolist()
            # real_output_multiplier = (twice_max_input_scale / (2**left_shift * np.asarray(output_scale))).tolist()

            input_multiplier, input_shift = get_quantized_multiplier(real_input_multiplier)
            input2_multiplier, input2_shift = get_quantized_multiplier(real_input2_multiplier)
            output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)

            sn.activation_max = 127
            sn.activation_min = -128

        elif subcode in ['MUL', 'DIV']:
            left_shift = -1
            input_multiplier, input_shift = [], []
            input2_multiplier, input2_shift = [], []
            with exception_catcher( sn.type, len(subnode_array), subop['inputs'][0]):
                _, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(subop, tensors)

            sn.activation_max = 127
            sn.activation_min = -128

            if opts['fused_activation_function'] == 'RELU':
                sn.activation_min = output_offset

        elif subcode in ['MINIMUM', 'MAXIMUM']:
            input1_shape, idims = channels_first_shape(i_tensor['shape'])
            input2_shape, i2dims = channels_first_shape(i2_tensor['shape'])
            left_shift = 8
            if subcode == 'MAXIMUM':
                sn.MinMaxOptions.max = 1
            else:
                sn.MinMaxOptions.max = 0
    
            if input1_shape != input2_shape:
                if len(input2_shape) > 1:
                    sys.stderr.write("ERROR: SUBOP{} doesn't support difference tensor shapes: {}, {}.\n".format(subcode, input1_shape, len(input2_shape)))
                    sys.exit(1)

            left_shift = 0
            input_multiplier, input_shift = get_quantized_multiplier(input_scale)
            input2_multiplier, input2_shift = get_quantized_multiplier(input2_scale)
            output_multiplier, output_shift = get_quantized_multiplier(output_scale)

            sn.activation_max = 127
            sn.activation_min = -128
        else:
            left_shift = 8
            output_offset = 0
            output_scale = [1.]
            input_multiplier, input_shift = get_quantized_multiplier(input_scale)
            input2_multiplier, input2_shift = get_quantized_multiplier(input2_scale)
            output_multiplier, output_shift = get_quantized_multiplier(output_scale)

            sn.activation_max = 1
            sn.activation_min = 0
        
        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.input_multiplier = len(weights)
        fmt = "{}i".format(len(input_multiplier))
        weights += struct.pack(fmt, *input_multiplier)

        sn.input_shift = len(weights)
        fmt = "{}i".format(len(input_shift))
        weights += struct.pack(fmt, *input_shift)

        eltwise8.input2_multiplier = len(weights)
        fmt = "{}i".format(len(input2_multiplier))
        weights += struct.pack(fmt, *input2_multiplier)

        eltwise8.input2_shift = len(weights)
        fmt = "{}i".format(len(input2_shift))
        weights += struct.pack(fmt, *input2_shift)
    
        eltwise8.bias_data = -1
        sn.input_offset = input_offset
        sn.output_offset = output_offset
        eltwise8.input2_offset = input2_offset
        eltwise8.left_shift = left_shift

        subnode_array.append(sn)

    elif subcode == 'TRANSFORM':
        # transform = None
        # sn.ActivationOptions.lut_count = 1

        l = 0
        bytes = 1
        final_output_scale = None
        final_output_offset = None
        while l < len(reshape_ops):
            op, code ,_ = reshape_ops[l]
            o_tensor = tensors[op['outputs'][0]]

            next_op, next_code = None, ''
            if l < len(reshape_ops)-1:
                next_op, next_code, _ = reshape_ops[l+1]
            next_next_op, next_next_code = None, ''
            if l < len(reshape_ops)-2:
                next_next_op, next_next_code, _ = reshape_ops[l+2]

            if code == 'RESHAPE' and next_code == 'TRANSPOSE' and next_next_code == 'RESHAPE':
                l += 2
                bytes = 1

                o_tensor = tensors[next_next_op['outputs'][0]]
                output_type = o_tensor['type']
                sn.output_data_type = calc_type.from_str(output_type)
                final_output_offset = o_tensor['quantization'].get('zero_point', [0])[0]
                final_output_scale = o_tensor['quantization'].get('scale', [1.0])[0]
                sn.type = VNNXOperator.PIXEL_SHUFFLE
                r = o_tensor['shape'][-2]//i_tensor['shape'][-2]
                assert(r == o_tensor['shape'][-3]//i_tensor['shape'][-3])
                sn.PixelShuffleOptions.r = r
            l += 1
        sn.input_offset = input_offset
        sn.output_offset = final_output_offset

        sn.input_shift = -1
        sn.input_multiplier = 1 
        sn.output_multiplier = len(weights)
        sn.output_shift = len(weights)

        subnode_array.append(sn)

    elif subcode == 'LUT':
        sn.type = VNNXOperator.LUT
        transform = None
        sn.ActivationOptions.lut_count = 0

        bytes = sizeof_calc_type(sn.output_data_type)

        lut = []

        for lop, code, _ in lut_ops:
            if code == "GATHER":
                l = get_numpy_data_from_index(lop['inputs'][0], tensors, buffers)
                if sn.input_data_type in [calc_type.INT8, calc_type.INT32]:
                    l = lut_i8_to_u8(l)

                if bytes == 4:
                    lut_repacked = [0xff & _ for _ in l]
                    lut_repacked += [(0xff * (2**8) & _) // (2**8) for _ in l]
                    lut_repacked += [(0xff * (2**16) & _) // (2**16) for _ in l]
                    lut_repacked += [(0xff * (2**24) & _) // (2**24) for _ in l]
                    lut += lut_repacked
                else:
                    lut += [0xff & _ for _ in l]
                sn.ActivationOptions.lut_count += 1

        # sn.input_offset = input_offset
        sn.ActivationOptions.input_range_radius = -1
        sn.ActivationOptions.count = -1
        sn.ActivationOptions.lut_int8 = -1
        sn.ActivationOptions.idx_int8 = -1

        sn.ActivationOptions.vci_int8 = len(weights)
        fmt = "{}B".format(len(lut))
        weights += struct.pack(fmt, *lut)

        sn.input_shift = -1
        sn.input_multiplier = 1 
        sn.output_multiplier = len(weights)
        sn.output_shift = len(weights)

        subnode_array.append(sn)

    elif subcode in ['PAD', 'PADV2']:
        sn.type = BuiltinOperator.PAD

        pads = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        c, h, w = [0, 0], [0, 0], [0, 0]
        if len(pads) == 4:
            c, h, w = pads[3], pads[1], pads[2]
        elif len(pads) == 3:
            c, h, w = pads[2], pads[0], pads[1]
        elif len(pads) == 2:
            h, w = pads[0], pads[1]
        elif len(pads) == 1:
            w = pads[0]
        sn.pads = [c[0], h[0], w[0], c[1], h[1], w[1]]
        assert len(sn.pads) == 6
        sn.PadOptions.value = input_offset
        sn.PadOptions.transpose_dilate_w = 1
        sn.PadOptions.transpose_dilate_h = 1
        if subcode == 'PADV2':
            value = get_numpy_data_from_index(subop['inputs'][2], tensors, buffers)
            sn.PadOptions.value = value

        subnode_array.append(sn)

    elif subcode in ['DILATE']:
        sn.type = BuiltinOperator.DILATE

        dilations = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        sn.pads = [0, 0, 0, 0, 0, 0]
        assert len(sn.pads) == 6
        sn.PadOptions.value = input_offset
        sn.PadOptions.transpose_dilate_w = dilations[-2]
        sn.PadOptions.transpose_dilate_h = dilations[-3]

        subnode_array.append(sn)

    elif subcode == "RESIZE_NEAREST_NEIGHBOR":
        sn.type = BuiltinOperator.RESIZE_NEAREST_NEIGHBOR
        s_tensor = tensors[subop['inputs'][1]]

        resize = sn.ResizeOptions
        resize.mode = resize_mode.NEAREST
        
        resize.b_postproc_tiling = 0 

        new_size_data = get_numpy_data(s_tensor, buffers)
        resize.scale = (new_size_data / i_tensor['shape'][1:3]).tolist() # height and width scales

        c_num_copies = []
        steps=[0] #Only needed for bilinear --> so empty for NN
        if all([sc > 1 for sc in resize.scale]):
            c_num_copies, c_out = resize_nearest_indices(i_tensor['shape'][1:3], new_size_data)

        sn.ResizeOptions.num_c_inc = len(c_num_copies)

        resize.c_inc = len(weights)
        fmt = "{}B".format(len(c_num_copies))
        weights += struct.pack(fmt, *c_num_copies)
        

        subnode_array.append(sn)

    elif subcode == "RESIZE_BILINEAR":
        sn.type = BuiltinOperator.RESIZE_BILINEAR
        s_tensor = tensors[subop['inputs'][1]]

        resize = sn.ResizeOptions
        resize.mode = resize_mode.LINEAR 
        resize.ratio =  ((1 << 10) * i_tensor['shape'][1] + o_tensor['shape'][1] / 2) / o_tensor['shape'][1]
        
        resize.b_postproc_tiling = 0
        if 'postprocessing' in s_tensor['name']:
            resize.b_postproc_tiling = 1

        new_size_data = get_numpy_data(s_tensor, buffers)
        resize.scale = (new_size_data / i_tensor['shape'][1:3]).tolist() # height and width scales

        c_num_copies, steps = [], []
        c_num_copies, c_out, steps = resize_bilinear_indices(i_tensor['shape'][1:3], new_size_data)
        # print(c_num_copies)
        sn.ResizeOptions.num_c_inc = len(c_num_copies)

        resize.c_inc = len(weights)
        fmt = "{}B".format(len(c_num_copies))
        weights += struct.pack(fmt, *c_num_copies)

        if i_tensor['shape'][1:3] == [1,1]:
            sn.type = BuiltinOperator.RESIZE_NEAREST_NEIGHBOR
            resize.mode = resize_mode.NEAREST 

        subnode_array.append(sn)

    elif subcode == 'MIRROR_PAD':
        sn.type = BuiltinOperator.MIRROR_PAD

        pads = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        c, h, w = [0, 0], [0, 0], [0, 0]
        if len(pads) == 4:
            c, h, w = pads[3], pads[1], pads[2]
        elif len(pads) == 3:
            c, h, w = pads[2], pads[0], pads[1]
        elif len(pads) == 2:
            h, w = pads[0], pads[1]
        elif len(pads) == 1:
            w = pads[0]
        sn.pads = [c[0], h[0], w[0], c[1], h[1], w[1]]
        assert len(sn.pads) == 6

        # 0 = SYMMETRIC, 1 = REFLECT
        pad_mode = opts['mode']
        sn.MirrorPadOptions.mode = 1 if pad_mode=='REFLECT' else 0

        subnode_array.append(sn)

    elif subcode == 'SOFTMAX':
        sn.type = BuiltinOperator.SOFTMAX

        assert (i_tensor['shape'] == o_tensor['shape'])
        beta = opts['beta']
        input_scale = i_tensor['quantization']['scale']

        sn.SoftmaxOptions.diff_min = -1
        output_scale = o_tensor['quantization']['scale']
        output_multiplier = 1. / (output_scale[0] * 2**16)
        quantized_multiplier, left_shift = get_quantized_multiplier([output_multiplier])

        sn.input_multiplier = len(weights)
        fmt = "{}i".format(len(quantized_multiplier))
        weights += struct.pack(fmt, *quantized_multiplier)

        sn.input_shift = len(weights)
        fmt = "{}i".format(len(left_shift))
        weights += struct.pack(fmt, *left_shift)

        axis = -1
        if 'axis' in opts:
            axis = opts['axis'] 
        axis = channels_first_axis(axis, len(input_shape)) - idims
        sn.SoftmaxOptions.axis = axis
        sn.SoftmaxOptions.depth = output_shape[axis]


        lut_scale = min(2**24, 2**31 / sn.SoftmaxOptions.depth)
        transform = lambda s: np.exp((s-127)*input_scale[0]) * lut_scale
        lut, first, last, step_vals, step_indices = LUTPopulate(1., 0., 1., 0, transform, bytes=4)

        sn.SoftmaxOptions.vci_int8 = len(weights)
        #repack
        lut_repacked = [0xff & _ for _ in lut]
        lut_repacked += [(0xff * (2**8) & _) // (2**8) for _ in lut]
        lut_repacked += [(0xff * (2**16) & _) // (2**16) for _ in lut]
        lut_repacked += [(0xff * (2**24) & _) // (2**24) for _ in lut]
        fmt = "{}B".format(len(lut_repacked))
        weights += struct.pack(fmt, *lut_repacked)

        sn.SoftmaxOptions.lut_int32 = len(weights)
        fmt = "{}I".format(len(step_vals))
        weights += struct.pack(fmt, *step_vals)

        sn.SoftmaxOptions.idx_int8 = len(weights)
        fmt = "{}b".format(len(step_indices))
        weights += struct.pack(fmt, *step_indices)
        sn.SoftmaxOptions.count = len(step_vals)

        subnode_array.append(sn)

    elif subcode == 'LOG_SOFTMAX':
        sn.type = BuiltinOperator.LOG_SOFTMAX

        kOutputZeroPoint = 127
        kOutputScale = 16.0 / 256
        kBeta = 1.0
        input_integer_bits = 5

        input_scale = i_tensor['quantization']['scale']
        input_offset = i_tensor['quantization']['zero_point'][0]

        max_real_multiplier = (1 << 31) - 1.0
        input_beta_real_multiplier = min(kBeta * input_scale[0] * (1 << (31 - input_integer_bits)), max_real_multiplier)
        assert (input_beta_real_multiplier > 1.0)
        quantized_multiplier, left_shift = get_quantized_multiplier([input_beta_real_multiplier])
        assert (all(x >= 0 for x in left_shift))
        assert (i_tensor['shape'] == o_tensor['shape'])

        real_reverse_scaling_divisor = (1 << (31 - left_shift[0])) / quantized_multiplier[0]
        assert (real_reverse_scaling_divisor < 1.0 and real_reverse_scaling_divisor > 0.0)
        reverse_scaling_divisor, reverse_scaling_left_shift = get_quantized_multiplier([real_reverse_scaling_divisor])

        reverse_scaling_right_shift = -1 * reverse_scaling_left_shift[0]
        assert (reverse_scaling_right_shift >= 0)

        output_multiplier, output_shift = get_quantized_multiplier([kOutputScale])

        max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) * (1 << (31 - input_integer_bits)) / (1 << left_shift[0])
        diff_min = -1 * floor(max_input_rescaled)

        sn.input_multiplier = len(weights)
        fmt = "{}i".format(len(quantized_multiplier))
        weights += struct.pack(fmt, *quantized_multiplier)

        sn.input_shift = len(weights)
        fmt = "{}i".format(len(left_shift))
        weights += struct.pack(fmt, *left_shift)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.output_offset = kOutputZeroPoint

        axis = -1
        if 'axis' in opts:
            axis = opts['axis']
        axis = channels_first_axis(axis, len(input_shape)) - idims

        sn.LogSoftmaxOptions.reverse_scaling_divisor = reverse_scaling_divisor[0]
        sn.LogSoftmaxOptions.reverse_scaling_right_shift = reverse_scaling_right_shift
        sn.LogSoftmaxOptions.diff_min = diff_min
        sn.LogSoftmaxOptions.axis = axis
        sn.LogSoftmaxOptions.depth = output_shape[axis]

        # assuming only one vector
        if len(o_tensor['shape']) > 1:
            sn.LogSoftmaxOptions.outer_size = o_tensor['shape'][0]
        else:
            sn.LogSoftmaxOptions.outer_size = 1

        sn.LogSoftmaxOptions.depth = o_tensor['shape'][-1]

        subnode_array.append(sn)

    elif subcode == 'MAX_POOL_2D':
        sn.type = BuiltinOperator.MAX_POOL_2D

        sn.kernel_shape = [opts['filter_height'], opts['filter_width']]
        sn.strides = [opts['stride_h'], opts['stride_w']]
        sn.pads = [0, 0, 0, 0, 0, 0]

        # inject Pad subnode before if necessary
        if opts['padding'] == "SAME":
            i_h, i_w = input_shape[-2:]
            o_h, o_w = output_shape[-2:]
            stride_h, stride_w = sn.strides
            kernel_h, kernel_w = sn.kernel_shape
            pad_h = max(0, stride_h * (o_h - 1) - i_h + kernel_h)
            pad_w = max(0, stride_w * (o_w - 1) - i_w + kernel_w)

            if pad_h > 0 or pad_w > 0:
                pad_sn = Subnode()
                pad_sn.type = BuiltinOperator.PAD

                pad_sn.input_data_type = calc_type.from_str(input_type)
                pad_sn.output_data_type = calc_type.from_str(output_type)


                pad_sn.pads = [0, floor(pad_h/2), floor(pad_w/2), 0, ceil(pad_h/2), ceil(pad_w/2)]
                pad_sn.PadOptions.value = -128
                pad_sn.PadOptions.transpose_dilate_w = 1
                pad_sn.PadOptions.transpose_dilate_h = 1

                pad_sn.num_inputs = 0
                pad_sn.num_outputs = 0

                subnode_array.append(pad_sn)

        subnode_array.append(sn)

    elif subcode == 'AVERAGE_POOL_2D':
        assert(sn.input_data_type == calc_type.INT8)
        i_h, i_w = input_shape[-2:]
        o_h, o_w = output_shape[-2:]

        sn.type = BuiltinOperator.AVERAGE_POOL_2D

        sn.kernel_shape = [opts['filter_height'], opts['filter_width']]
        if sn.kernel_shape[0] > i_h:
            sn.kernel_shape[0] = i_h
        if sn.kernel_shape[1] > i_w:
            sn.kernel_shape[1] = i_w

        sn.strides = [opts['stride_h'], opts['stride_w']]
        sn.pads = [0, 0, 0, 0, 0, 0]

        if opts['padding'] == "SAME":
            stride_h, stride_w = sn.strides
            kernel_h, kernel_w = sn.kernel_shape

            pad_h = max(0, stride_h * (o_h - 1) - i_h + kernel_h)
            pad_w = max(0, stride_w * (o_w - 1) - i_w + kernel_w)

            sn.pads = [0, floor(pad_h/2), floor(pad_w/2), 0, ceil(pad_h/2), ceil(pad_w/2)]

        subnode_array.append(sn)

    elif subcode == 'MUL': 
        sn.type = BuiltinOperator.MUL

        i_tensor = tensors[subop['inputs'][0]]
        f_tensor = tensors[subop['inputs'][1]]

        swap_input_order = 0
        if 'buffer' in i_tensor and 'data' in buffers[i_tensor['buffer']]:
            swap_input_order = 1
            i_tensor = tensors[subop['inputs'][1]]
            f_tensor = tensors[subop['inputs'][0]]

        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]
        filter_offset = f_tensor['quantization']['zero_point'][0]

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']
        filter_scale = f_tensor['quantization']['scale']


        effective_scale, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(subop, tensors)

        broad8 = sn.broadcast8
        broad8.iscale = input_scale[0]
        broad8.fscale = filter_scale[0]
        broad8.oscale = output_scale[0]

        broad8.iscale = sqrt(broad8.iscale*broad8.fscale)
        broad8.fscale = broad8.iscale

        broad8.swap_inputs = swap_input_order

        filter_data = get_numpy_data(f_tensor, buffers)
        if len(f_tensor['shape']) >= 3:
            if len(f_tensor['shape']) == len(i_tensor['shape']):
                pass

        k, h, w, c = 1, 1, 1, 1
        if len(f_tensor['shape']) == 1:
            c = tuple(f_tensor['shape'])[0]
        elif len(f_tensor['shape']) == 2:
            k, c = tuple(f_tensor['shape'])
        elif len(f_tensor['shape']) == 3:
            h, w, c = tuple(f_tensor['shape'])
            filter_data = filter_data.transpose((2,0,1))
        elif len(f_tensor['shape']) == 4:
            k, h, w, c = tuple(f_tensor['shape'])
            filter_data = filter_data.transpose((0,3,1,2))
        elif len(f_tensor['shape']) == 5:
            assert(f_tensor['shape'][0] == 1)
            k, h, w, c = f_tensor['shape'][1:]
            filter_data = filter_data.transpose((0,1,4,2,3))

        broad8.filter_shape_dims = [k, c, h, w]

        broad8.broadcast = 0
        ishape, _ = channels_first_shape(i_tensor['shape'])
        if broad8.filter_shape_dims[-2] == 1 and broad8.filter_shape_dims[-1] == 1: # channels
            broad8.broadcast = 1
        elif broad8.filter_shape_dims[-2] == 1 and broad8.filter_shape_dims[-1] == ishape[-1]: #row
            broad8.broadcast = 2
        elif broad8.filter_shape_dims[-2] == ishape[-2] and broad8.filter_shape_dims[-1] == ishape[-1]: #maps
            broad8.broadcast = 3

        broad8.optimized = 0
        sf = input_scale[0] * filter_scale[0] / output_scale[0] 
        scaled_filters = [sf * (_ - filter_offset) for _ in filter_data.flatten()]
        valid_filters = [_ < 2**(15-Q16) for _ in scaled_filters]
        if (np.all(valid_filters)):
            if broad8.broadcast == 1:
                broad8.optimized = OPTIMIZED_MUL
        else:
            if OPTIMIZED_MUL and VERBOSE:
                print(subcode, input_scale[0], filter_scale[0], output_scale[0], "WARNING can't optimize")


        sn.input_multiplier = -1
        sn.input_shift = -1

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        broad8.filter_multiplier = -1
        broad8.filter_shift = -1

        broad8.filter_data = len(weights)
        fmt = "{}b".format(len(filter_data.flatten()))
        weights += struct.pack(fmt, *filter_data.flatten())

        if USE_PRECALC:
            bias_data = precalculate_filter_input_bias(filter_data, input_offset, filter_offset)
            bias_data += precalculate_output_offset_bias(effective_scale, output_offset)
            broad8.bias_data = len(weights)
            fmt = "{}i".format(len(bias_data))
            weights += struct.pack(fmt, *bias_data)
        else:
            broad8.bias_data = -1

        sn.input_offset = input_offset
        sn.output_offset = output_offset
        broad8.filter_offset = filter_offset
        broad8.left_shift = -1
        broad8.sub = 0

        subnode_array.append(sn)

    elif subcode in ['ADD', 'SUB', 'SQUARED_DIFFERENCE', 'SQUARED_ROOT']:
        if subcode == 'ADD':
            sn.type = BuiltinOperator.ADD
        else:
            sn.type = BuiltinOperator.SUB
            
        i_tensor = tensors[subop['inputs'][0]]
        f_tensor = tensors[subop['inputs'][1]]

        swap_input_order = 0
        if 'buffer' in i_tensor and 'data' in buffers[i_tensor['buffer']]:
            swap_input_order = 1
            i_tensor = tensors[subop['inputs'][1]]
            f_tensor = tensors[subop['inputs'][0]]

        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]
        filter_offset = f_tensor['quantization']['zero_point'][0]

        filter_data = get_numpy_data(f_tensor, buffers)

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']
        filter_scale = f_tensor['quantization']['scale']

        left_shift = 15
        if subcode != 'SQUARED_DIFFERENCE':
            if 'pot_scale_int16' in opts and opts['pot_scale_int16']:
                left_shift = 20
        
        twice_max_input_scale = 2 * max(input_scale[0], filter_scale[0])
        real_input_multiplier = (np.asarray(input_scale) / twice_max_input_scale).tolist()
        real_filter_multiplier = (np.asarray(filter_scale) / twice_max_input_scale).tolist()
        real_output_multiplier = (twice_max_input_scale / (2**left_shift * np.asarray(output_scale))).tolist()

        input_multiplier, input_shift = get_quantized_multiplier(real_input_multiplier)
        filter_multiplier, filter_shift = get_quantized_multiplier(real_filter_multiplier)
        output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)


        broad8 = sn.broadcast8
        broad8.iscale = input_scale
        broad8.fscale = filter_scale
        broad8.oscale = output_scale

        broad8.sub = 0
        broad8.swap_inputs = swap_input_order
        if subcode == 'SUB':
            broad8.sub = 1
        
        if subcode == 'SQUARED_DIFFERENCE':
            sn.type = BuiltinOperator.SQUARED_DIFFERENCE
            left_shift = 7
            sn.activation_max = 127
            sn.activation_min = -128
            twice_max_input_scale = 2 * max(input_scale[0], filter_scale[0])
            real_input_multiplier = (np.asarray(input_scale) / twice_max_input_scale).tolist()
            real_filter_multiplier = (np.asarray(filter_scale) / twice_max_input_scale).tolist()
            real_output_multiplier = ((twice_max_input_scale*twice_max_input_scale) / ((1<<left_shift * 2) * np.asarray(output_scale))).tolist()

            input_multiplier, input_shift = get_quantized_multiplier(real_input_multiplier)
            filter_multiplier, filter_shift = get_quantized_multiplier(real_filter_multiplier)
            output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)
            
        k, h, w, c = 1, 1, 1, 1
        if len(f_tensor['shape']) == 1:
            c = tuple(f_tensor['shape'])[0]
        elif len(f_tensor['shape']) == 2:
            k, c = tuple(f_tensor['shape'])
        elif len(f_tensor['shape']) == 3:
            h, w, c = tuple(f_tensor['shape'])
            filter_data = filter_data.transpose((2,0,1))
        elif len(f_tensor['shape']) == 4:
            k, h, w, c = tuple(f_tensor['shape'])
            filter_data = filter_data.transpose((0,3,1,2))
        elif len(f_tensor['shape']) == 5:
            assert(f_tensor['shape'][0] == 1)
            k, h, w, c = f_tensor['shape'][1:]
            filter_data = filter_data.transpose((0,1,4,2,3))
        broad8.filter_shape_dims = [k, c, h, w]


        broad8.broadcast = 0
        ishape, _ = channels_first_shape(i_tensor['shape'])
        if broad8.filter_shape_dims[-2] == 1 and broad8.filter_shape_dims[-1] == 1: # channels
            broad8.broadcast = 1
        elif broad8.filter_shape_dims[-2] == 1 and broad8.filter_shape_dims[-1] == ishape[-1]: #row
            broad8.broadcast = 2
        elif broad8.filter_shape_dims[-2] == ishape[-2] and broad8.filter_shape_dims[-1] == ishape[-1]: #maps
            broad8.broadcast = 3

        broad8.optimized = 0
        if broad8.broadcast == 1 and (2*input_scale[0]/output_scale[0]) < 2**(15-Q16) and (2*filter_scale[0]/output_scale[0]) < 2**(15-Q16):
            broad8.optimized = OPTIMIZED_ADD
        else:
            if OPTIMIZED_ADD and VERBOSE:
                print(subcode, input_scale[0]/output_scale[0], filter_scale[0]/output_scale[0], "WARNING can't optimize")

        if subcode != 'SQUARED_DIFFERENCE':
            if opts['fused_activation_function'] == 'RELU':
                sn.activation_min = output_offset

        sn.input_multiplier = len(weights)
        fmt = "{}i".format(len(input_multiplier))
        weights += struct.pack(fmt, *input_multiplier)

        sn.input_shift = len(weights)
        fmt = "{}i".format(len(input_shift))
        weights += struct.pack(fmt, *input_shift)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        broad8.filter_multiplier = len(weights)
        fmt = "{}i".format(len(filter_multiplier))
        weights += struct.pack(fmt, *filter_multiplier)

        broad8.filter_shift = len(weights)
        fmt = "{}i".format(len(filter_shift))
        weights += struct.pack(fmt, *filter_shift)

        broad8.filter_data = len(weights)
        fmt = "{}b".format(len(filter_data.flatten()))
        weights += struct.pack(fmt, *filter_data.flatten())

        broad8.bias_data = -1
        sn.input_offset = input_offset
        sn.output_offset = output_offset
        broad8.filter_offset = filter_offset
        broad8.left_shift = left_shift

        subnode_array.append(sn)
    elif subcode in ["GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL", "EQUAL", "NOT_EQUAL"]: # > to remove
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        f_tensor = tensors[subop['inputs'][1]]
        if np.squeeze(np.array(f_tensor['shape'])) != ():
            sys.stderr.write('ERROR: {} comparison not on scalar\n'.format(subcode))
            sys.exit(1)
        filter_offset = f_tensor['quantization']['zero_point'][0]
        filter_data = get_numpy_data(f_tensor, buffers)
        filter_scale = f_tensor['quantization']['scale']
        output_offset = 0
        output_scale = [1.]

        left_shift = 8

        input_multiplier, input_shift = get_quantized_multiplier(input_scale)
        filter_multiplier, filter_shift = get_quantized_multiplier(filter_scale)
        output_multiplier, output_shift = get_quantized_multiplier(output_scale)

        broad8 = sn.broadcast8
        broad8.iscale = input_scale
        broad8.fscale = filter_scale
        broad8.oscale = output_scale

        sn.activation_max = 1
        sn.activation_min = 0

        sn.input_multiplier = len(weights)
        fmt = "{}i".format(len(input_multiplier))
        weights += struct.pack(fmt, *input_multiplier)

        sn.input_shift = len(weights)
        fmt = "{}i".format(len(input_shift))
        weights += struct.pack(fmt, *input_shift)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        broad8.filter_multiplier = len(weights)
        fmt = "{}i".format(len(filter_multiplier))
        weights += struct.pack(fmt, *filter_multiplier)

        broad8.filter_shift = len(weights)
        fmt = "{}i".format(len(filter_shift))
        weights += struct.pack(fmt, *filter_shift)

        broad8.filter_data = len(weights)
        fmt = "{}b".format(len(filter_data.flatten()))
        weights += struct.pack(fmt, *filter_data.flatten())

        broad8.bias_data = 0
        sn.input_offset = input_offset
        sn.output_offset = output_offset
        broad8.filter_offset = filter_offset
        broad8.left_shift = left_shift
        broad8.sub = 0
        broad8.swap_inputs = 0

        subnode_array.append(sn)

    elif subcode in ['MEAN', 'SUM', 'REDUCE_PROD', 'REDUCE_MAX', 'REDUCE_MIN']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']

        num_elements_in_axis = input_shape[-2]*input_shape[-1]

        if subcode == 'REDUCE_PROD':
            _, output_multiplier, output_shift = get_quantized_multiplier_from_tensor(o_tensor)
        elif subcode == 'MEAN':
            _, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(subop, tensors, [1.0/num_elements_in_axis,])
        else:
            _, output_multiplier, output_shift = get_effective_quantized_multiplier_from_op(subop, tensors, [1.0,])
        
        sn.input_multiplier = -1
        sn.input_shift = -1

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.input_offset = input_offset
        sn.output_offset = output_offset

        subnode_array.append(sn)

    elif subcode == 'TRANSPOSE':
        sn.type = BuiltinOperator.TRANSPOSE

        transform = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        transform_size = len(np.trim_zeros([_ if i == _ else 0 for i,_ in enumerate(transform)], 'f'))
        transform_size = max(3, transform_size)

        assert(np.all([_ == i for i,_ in enumerate(transform[:-3])]))
        transform = [t - (len(transform) - 3) for t in transform[-3:]]

        assert(len(transform) == 3)

        transform = [(transform[2] + 1) % 3, (transform[0] + 1) % 3, (transform[1] + 1) % 3]

        sn.TransposeOptions.permutation = transform
        sn.TransposeOptions.out_rows_at_once = 1
        sn.TransposeOptions.out_maps_at_once = 1

        subnode_array.append(sn)

    elif subcode in ['UNPACK']:
        sn.type = BuiltinOperator.UNPACK

        input_shape, idims = channels_first_shape(i_tensor['shape'])
        sn.PackOptions.axis = channels_first_axis(opts['axis'], idims) - idims
        sn.PackOptions.count = opts['num']
        sn.PackOptions.dims = idims

        assert(check_unpack(i_tensor['shape'], o_tensor['shape'],
                            opts['axis'], sn.PackOptions.axis, opts['num']))
        subnode_array.append(sn)

    elif subcode in ['SQUEEZE', 'EXPAND_DIMS', 'RESHAPE']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        transform = tuple(get_numpy_data_from_index(subop['inputs'][1], tensors, buffers))

        ishape = i_tensor['shape']
        mode = channels_first_array_reshape(ishape, transform)

        sn.nop = 0
        if mode == 0:
            sn.nop = 1

        if mode == -1:
            print('RESHAPE: not implemented')
            assert(0)

        sn.ReshapeOptions.mode = mode
        subnode_array.append(sn)
    elif subcode == 'QUANTIZE':
        if len(dequantize_op_params) % 2 != 0:
            input_id = subop['inputs'][0]
            output_id = subop['outputs'][0]
            o_tensor = tensors[subop['outputs'][0]]
            output_offset = o_tensor['quantization']['zero_point'][0]
            output_scale = o_tensor['quantization']['scale']
            dequantize_op_params[input_id] = (output_scale, output_offset, output_id, 'output')
        else:
            sn.type = BuiltinOperator.DEQUANTIZE

            i_tensor = tensors[subop['inputs'][0]]
            o_tensor = tensors[subop['outputs'][0]]

            input_offset = 0
            input_scale = 1.0
            if 'quantization' in i_tensor.keys():
                input_offset = i_tensor['quantization'].get('zero_point', [0])[0]
                input_scale = i_tensor['quantization'].get('scale', 1.0)
            output_offset = o_tensor['quantization']['zero_point'][0]
            output_scale = o_tensor['quantization']['scale']

            real_input_multiplier = [1.0]
            real_output_multiplier = (np.asarray(input_scale) / np.asarray(output_scale) ).tolist()

            output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)

            i_type = calc_type.from_str(i_tensor['type'])
            o_type = calc_type.from_str(o_tensor['type'])
            mixed_type_int8_uint8 = i_type == calc_type.INT8 and o_type == calc_type.UINT8
            mixed_type_uint8_int8 = i_type == calc_type.UINT8 and o_type == calc_type.INT8
            zero_point_diff = input_offset - output_offset
            if output_multiplier[0] == 2**30 and output_shift[0] == 1: 
                if mixed_type_int8_uint8 and zero_point_diff == -128: 
                    sn.activation_min = 0
                    sn.activation_max = 0
                elif mixed_type_uint8_int8 and zero_point_diff == 128:
                    sn.activation_min = 0
                    sn.activation_max = 0

            sn.output_multiplier = len(weights)
            fmt = "{}i".format(len(output_multiplier))
            weights += struct.pack(fmt, *output_multiplier)

            sn.output_shift = len(weights)
            fmt = "{}i".format(len(output_shift))
            weights += struct.pack(fmt, *output_shift)

            sn.input_offset = input_offset
            sn.output_offset = output_offset

            subnode_array.append(sn)

    elif subcode == 'DEQUANTIZE':
        # case where Dequantize followed by Split followed by Quantize (yolo-v4-tiny-tf)
        # skip Dequantize and Quantize ops
        # TODO raise error for other cases, as they will not be handled correctly
        input_id = subop['inputs'][0]
        output_id = subop['outputs'][0]
        i_tensor = tensors[subop['inputs'][0]]
        scale = i_tensor['quantization']['scale'][0]
        zero_point = i_tensor['quantization']['zero_point'][0]
        dequantize_op_params[output_id] = (scale, zero_point, input_id, 'input')

                
    elif subcode == 'PRELU':
        sn.type = BuiltinOperator.PRELU

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        alpha_tensor = tensors[subop['inputs'][1]]
        alpha_data = get_numpy_data(alpha_tensor, buffers)

        b, h, w, c = 1, 1, 1, 1
        if len(alpha_tensor['shape']) == 3:
            h, w, c = tuple(alpha_tensor['shape'])
            alpha_data = alpha_data.transpose((2,0,1))
        else:
            sys.stderr.write('ERROR: broadcast on 2D or 3D input is not supported yet in {}\n'.format(subcode))
            sys.exit(1)
        sn.prelu.alpha_shape = [b, c, h, w]

        input_offset = i_tensor['quantization']['zero_point'][0]
        alpha_offset = alpha_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]

        input_scale = i_tensor['quantization']['scale']
        alpha_scale = alpha_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']
        sn.prelu.iscale = input_scale[0]
        sn.prelu.ascale = alpha_scale[0]
        sn.prelu.oscale = output_scale[0]

        sn.prelu.optimized = 0
        sn.prelu.vci_int8 = -1
        sn.prelu.maps_at_once = 1

        sf = input_scale[0] * alpha_scale[0] / output_scale[0] 
        scaled_filters = [sf * (_ - alpha_offset) for _ in alpha_data.flatten()]
        valid_filters = [_ < 2**(15-Q16) for _ in scaled_filters]
        if (np.all(valid_filters)):
            sn.prelu.optimized = 1
            sn.prelu.maps_at_once = 8

        real_input_multiplier = [1.0]
        real_output_multiplier = (np.asarray(input_scale) / np.asarray(output_scale) ).tolist()
        real_alpha_multiplier = (np.asarray(input_scale) * np.asarray(alpha_scale) / np.asarray(output_scale)).tolist()

        output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)
        alpha_multiplier, alpha_shift = get_quantized_multiplier(real_alpha_multiplier)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.prelu.alpha_multiplier = len(weights)
        fmt = "{}i".format(len(alpha_multiplier))
        weights += struct.pack(fmt, *alpha_multiplier)

        sn.prelu.alpha_shift = len(weights)
        fmt = "{}i".format(len(alpha_shift))
        weights += struct.pack(fmt, *alpha_shift)

        sn.prelu.alpha_data = len(weights)
        fmt = "{}b".format(len(alpha_data.flatten()))
        weights += struct.pack(fmt, *alpha_data.flatten())

        sn.input_offset = input_offset
        sn.output_offset = output_offset
        sn.prelu.alpha_offset = alpha_offset

        subnode_array.append(sn)

    elif subcode == 'LEAKY_RELU':
        sn.type = BuiltinOperator.LEAKY_RELU

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']

        alpha = opts['alpha']
        real_input_multiplier = [1.0]
        real_output_multiplier = (np.asarray(input_scale) / np.asarray(output_scale) ).tolist()
        real_alpha_multiplier = [o*alpha for o in real_output_multiplier]

        output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)
        alpha_multiplier, alpha_shift = get_quantized_multiplier(real_alpha_multiplier)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.leakyrelu.alpha_multiplier = len(weights)
        fmt = "{}i".format(len(alpha_multiplier))
        weights += struct.pack(fmt, *alpha_multiplier)

        sn.leakyrelu.alpha_shift = len(weights)
        fmt = "{}i".format(len(alpha_shift))
        weights += struct.pack(fmt, *alpha_shift)

        sn.input_offset = input_offset
        sn.output_offset = output_offset

        subnode_array.append(sn)

    elif subcode in ['RELU', 'RELU6', 'RELU_N1_TO_1', 'RELU_0_TO_1']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']


        # TODO
        # quantized_activation_min = max(-127,output_offset + int(roundf(act_min / output_scale)))
        # quantized_activation_max = min(128, int(roundf(act_max / output_scale)))
        sn.activation_min = max(-128, output_offset)
        if subcode == 'RELU6':
            sn.activation_max = min(127, output_offset + int(round(6.0 / output_scale[0])))
        elif subcode == 'RELU_0_TO_1':
            # sn.activation_min = max(-128, output_offset + int(round(0.0 / output_scale[0])))
            sn.activation_max = min(127, output_offset + int(round(1.0 / output_scale[0])))
        elif subcode == 'RELU_N1_TO_1':
            sn.activation_min = max(-128, output_offset + int(round(-1.0 / output_scale[0])))
            sn.activation_max = min(127, output_offset + int(round(1.0 / output_scale[0])))

        real_output_multiplier = (np.asarray(input_scale) / np.asarray(output_scale) ).tolist()
        output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.input_offset = input_offset
        sn.output_offset = output_offset

        subnode_array.append(sn)

    elif subcode in ['ARG_MIN', 'ARG_MAX', 'TOPK_V2']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")
        # sys.stderr.write('\nsubop=={}\n'.format(subop))
        axis = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        # tflite: n,y,x,c {0, -3, -2, -1}
        # tflite: n,y,x,c {0, 1, 2, 3}
        # vnnx: 1,c,y,x {0, 3, 1, 2}
        # neg vnnx: 1,c,y,x {-4, -3, -2, -1}
        if axis == -1 or axis == 3:
            sn.reduce8.axis =  -3 #vnnx channels first
            axis_list = np.array(range(input_shape[1]))
            sn.reduce8.axis_list = len(weights)
            fmt = "{}b".format(len(axis_list.flatten()))
            weights += struct.pack(fmt, *axis_list.flatten())
        elif axis == 0:
            sys.stderr.write('ERROR: reduction on axis {} is not supported in {}\n'.format(axis,subcode))
            sys.exit(1)
        elif axis == -2 or axis == 2:
            sn.reduce8.axis =  -2 #vnnx channels first
            sys.stderr.write('ERROR: reduction on axis {} is not supported in {}\n'.format(axis,subcode))
            sys.exit(1)
        elif axis == -3 or axis == 1:
            sn.reduce8.axis =  -1 #vnnx channels first
            sys.stderr.write('ERROR: reduction on axis {} is not supported in {}\n'.format(axis,subcode))
            sys.exit(1)
        else:
            sys.stderr.write('ERROR: reduction on axis {} is not supported in {}\n'.format(axis,subcode))
            sys.exit(1)

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        input_offset = 0
        output_offset = 0

        input_scale = 1.
        output_scale = 1.

        assert(len(o_tensor['shape']) == len(i_tensor['shape']) - 1)
        
        sn.reduce8.arg_max = 0
        if subcode == 'ARG_MAX':
            sn.reduce8.arg_max = 1
        
        subnode_array.append(sn)

    elif subcode == "SLICE":
        sn.type = BuiltinOperator.SLICE

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        begin = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        size = get_numpy_data_from_index(subop['inputs'][2], tensors, buffers)

        assert(len(begin) <= 4)
        assert(len(size) <= 4)

        begin, _ = channels_first_shape(begin)
        size, _ = channels_first_shape(size)
        ishape, _ = channels_first_shape(i_tensor['shape'])

        begin = pad_list(begin, 4, 0)
        size = pad_list(size, 4)
        ishape = pad_list(ishape, 4)
        for i,d in enumerate(size):
            if d == -1:
                size[i] = ishape[i]

        sn.SliceOptions.begin = begin
        sn.SliceOptions.end = [sum(x) for x in zip(sn.SliceOptions.begin, size)]
        sn.SliceOptions.stride = [1,1,1,1]
        subnode_array.append(sn)

    elif subcode == "STRIDED_SLICE":
        sn.type = BuiltinOperator.STRIDED_SLICE

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        begin_tmp = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        end_tmp = get_numpy_data_from_index(subop['inputs'][2], tensors, buffers)
        stride = get_numpy_data_from_index(subop['inputs'][3], tensors, buffers)
       
        oshape, _ = channels_first_shape(o_tensor['shape'])
        ishape, _ = channels_first_shape(i_tensor['shape'])
        begin_tmp, _ = channels_first_shape(begin_tmp)
        end_tmp, _ = channels_first_shape(end_tmp)
        stride, _ = channels_first_shape(stride)
        
        begin_mask = opts['begin_mask']
        end_mask = opts['end_mask']        

        ishape = pad_list(ishape, 4)
        oshape = pad_list(oshape, 4)
        begin = pad_list(begin_tmp, 4, 0)
        end = pad_list(end_tmp, 4)
        stride = pad_list(stride, 4)

        sn.SliceOptions.begin = begin
        sn.SliceOptions.stride = stride
        sn.SliceOptions.end = [min(_,ishape[idx]) if _ != 0 else ishape[idx] for idx, _ in enumerate(end)]

        subnode_array.append(sn)

    elif subcode == "GATHER":
        sn.type = BuiltinOperator.GATHER
        axis = opts['axis']
        batch_dims = opts['batch_dims']
        # TODO: generalize for batch_dims > 0
        if batch_dims != 0:
            sys.stderr.write('ERROR: batch_dim = {} for GATHER is not implemented\n'.format(batch_dims))

        
        coord_data_tensor = tensors[subop['inputs'][1]]
        coord_data = []
        swap_input_order = subop['inputs'][1] < subop['inputs'][0]
        # if 'buffer' in coord_data_tensor and 'params' not in coord_data_tensor:
        #     swap_input_order = 1
        #     coord_data_tensor = tensors[subop['inputs'][1]]
        #     coord_data = np.random.randint(0, 20 + 1, coord_data_tensor['shape'])#get_numpy_data_from_index(subop['inputs'][0], tensors, buffers)
        #     # sys.stderr.write("\n------------------- coord_data{} {}".format(coord_data, coord_data_tensor['shape']))
        #     coord_data = coord_data.flatten()
        # if swap_input_order:
            #Error here because coord_data are not const, so there is nothing inside the buffer
        coord_data = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)  
        # assert (len(coord_data_tensor['shape']) == 1)

        input_shape = tensors[subop['inputs'][0]]['shape']
        output_shape = tensors[subop['outputs'][0]]['shape']

        pos_axis = axis
        if axis < 0:
            pos_axis = len(input_shape) + axis
        # axis_size = input_shape[axis]
        # batch_size = int(np.prod(input_shape[:batch_dims]))
        # outer_size = int(np.prod(input_shape[batch_dims:axis]))
        # inner_size = int(np.prod(input_shape[pos_axis+1:]))
        # coord_size = coord_data_tensor['shape'][0]
        # print(batch_dims, pos_axis)
        # print('ax', axis_size, 'bat', batch_size, 'out', outer_size, 'in', inner_size, 'cord', coord_size)

        # # o(1, 3, 1, 2), i(1, 20, 1, 2)
        # print('batch gap', outer_size*coord_size*inner_size, 'map gap', coord_size*inner_size, 'elem gap', inner_size)
        # print('batch gap', outer_size*axis_size*inner_size, 'map gap', axis_size*inner_size, 'elem gap', inner_size)
    
        # sys.stderr.write("\n-------------------output_shape {}".format(output_shape))
        cinput_shape, _ = channels_first_shape(input_shape)
        coutput_shape, _ = channels_first_shape(output_shape)
        caxis = channels_first_axis(axis, len(input_shape))
        sn.GatherOptions.swap_input_order =swap_input_order
        sn.GatherOptions.axis_size = cinput_shape[caxis]
        sn.GatherOptions.batch_size = int(np.prod(cinput_shape[:batch_dims]))
        sn.GatherOptions.outer_size = int(np.prod(cinput_shape[batch_dims:caxis]))
        sn.GatherOptions.inner_size = int(np.prod(cinput_shape[caxis+1:]))
        sn.GatherOptions.coord_size = coord_data_tensor['shape'][0]

        sn.GatherOptions.batch_dims = batch_dims
        sn.GatherOptions.axis = caxis
        # print(cinput_shape, coutput_shape)
        # print(batch_dims, pos_axis)
        # print('ax', sn.GatherOptions.axis_size, 'bat', sn.GatherOptions.batch_size, 'out', sn.GatherOptions.outer_size, 'in', sn.GatherOptions.inner_size, 'cord', sn.GatherOptions.coord_size)

        # print('batch gap', sn.GatherOptions.outer_size*sn.GatherOptions.coord_size*sn.GatherOptions.inner_size, 'map gap', sn.GatherOptions.coord_size*sn.GatherOptions.inner_size, 'elem gap', sn.GatherOptions.inner_size)
        # print('batch gap', sn.GatherOptions.outer_size*sn.GatherOptions.axis_size*sn.GatherOptions.inner_size, 'map gap', sn.GatherOptions.axis_size*sn.GatherOptions.inner_size, 'elem gap', sn.GatherOptions.inner_size)

        sn.GatherOptions.coord_data = len(weights)
        fmt = "{}i".format(len(coord_data))
        weights += struct.pack(fmt, *coord_data)
        
        subnode_array.append(sn)

    elif subcode in ['DEPTH_TO_SPACE', 'SPACE_TO_DEPTH']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")
        subnode_array.append(sn)

    elif subcode in ['BATCH_TO_SPACE_ND', 'SPACE_TO_BATCH_ND']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        block_shape_data = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
        paddings_data = get_numpy_data_from_index(subop['inputs'][2], tensors, buffers)
        
        if subcode == "SPACE_TO_BATCH_ND":
            sn.SpaceToBatchNDOptions.block_shape_data = len(weights)
            fmt = "{}i".format(len(block_shape_data))
            weights += struct.pack(fmt, *block_shape_data)

            sn.SpaceToBatchNDOptions.paddings_data = len(weights)
            fmt = "{}i".format(len(paddings_data.flatten()))
            weights += struct.pack(fmt, *paddings_data.flatten())

        elif subcode == "BATCH_TO_SPACE_ND":
            sn.BatchToSpaceNDOptions.block_shape_data = len(weights)
            fmt = "{}i".format(len(block_shape_data))
            weights += struct.pack(fmt, *block_shape_data)

            sn.BatchToSpaceNDOptions.crop_data = len(weights)
            fmt = "{}i".format(len(paddings_data.flatten()))
            weights += struct.pack(fmt, *paddings_data.flatten())
            # sys.stderr.write("\n-------------------BATCH_TO_SPACE_ND Test1")

        subnode_array.append(sn)

    elif subcode in ['ABS']:
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]
        input_offset = i_tensor['quantization']['zero_point'][0]
        output_offset = o_tensor['quantization']['zero_point'][0]

        input_scale = i_tensor['quantization']['scale']
        output_scale = o_tensor['quantization']['scale']


        real_output_multiplier = (np.asarray(input_scale) / np.asarray(output_scale) ).tolist()

        output_multiplier, output_shift = get_quantized_multiplier(real_output_multiplier)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.input_offset = input_offset
        sn.output_offset = output_offset

        subnode_array.append(sn)

    elif subcode == "L2_NORMALIZATION":
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        i_tensor = tensors[subop['inputs'][0]]
        o_tensor = tensors[subop['outputs'][0]]

        input_zero_point = i_tensor['quantization']['zero_point'][0]

        sn.input_offset = input_zero_point

        subnode_array.append(sn)
    
    elif subcode == 'NEG':
        sn.type = getattr(BuiltinOperator, f"{subcode}")

        # i_tensor = tensors[subop['inputs'][0]]
        # o_tensor = tensors[subop['outputs'][0]]

        # input_zero_point = i_tensor['quantization']['zero_point'][0]

        # sn.input_offset = input_zero_point

        subnode_array.append(sn)
    
    elif subcode == "EMBEDDING_LOOKUP":
        sn.type = getattr(BuiltinOperator, f"{subcode}")
        embedding = sn.embedding
        i_tensor = tensors[subop['inputs'][0]]
        i_tensor_value = tensors[subop['inputs'][1]]

        color_data = get_numpy_data(i_tensor_value, buffers)
        h, c = tuple(i_tensor_value['shape'])
        color_dims = [h, c]

        embedding.colar_map_data = len(weights)
        fmt = "{}i".format(len(color_data))
        weights += struct.pack(fmt, *color_data.flatten())
        
        embedding.colar_map_dims = len(weights)
        fmt = "{}i".format(len(color_dims))
        weights += struct.pack(fmt, *color_dims)

        sn.input_offset = input_zero_point

        subnode_array.append(sn)

    elif subcode in ['MAXIMUM', 'MINIMUM']:
        if subcode == 'MAXIMUM':
            sn.type = BuiltinOperator.MAXIMUM
            sn.MinMaxOptions.max = 1
        else:
            sn.type = BuiltinOperator.MINIMUM
            sn.MinMaxOptions.max = 0

        i_tensor = tensors[subop['inputs'][0]]
        f_tensor = tensors[subop['inputs'][1]]
        filter_data = get_numpy_data(f_tensor, buffers)
        
        k, h, w, c = 1, 1, 1, 1
        if len(f_tensor['shape']) == 1:
            c = tuple(f_tensor['shape'])[0]
        elif len(f_tensor['shape']) == 2:
            k, c = tuple(f_tensor['shape'])
        elif len(f_tensor['shape']) == 3:
            h, w, c = tuple(f_tensor['shape'])
            filter_data = filter_data.transpose((2,0,1))
        elif len(f_tensor['shape']) == 4:
            k, h, w, c = tuple(f_tensor['shape'])
            filter_data = filter_data.transpose((0,3,1,2))
        elif len(f_tensor['shape']) == 5:
            assert(f_tensor['shape'][0] == 1)
            k, h, w, c = f_tensor['shape'][1:]
            filter_data = filter_data.transpose((0,1,4,2,3))

        sn.MinMaxOptions.filter_shape_dims = [k, c, h, w]    

        input_offset = i_tensor['quantization']['zero_point'][0]
        input_scale = i_tensor['quantization']['scale']

        output_offset = o_tensor['quantization']['zero_point'][0] 
        output_scale = o_tensor['quantization']['scale']
        
        filter_offset = f_tensor['quantization']['zero_point'][0]
        filter_scale = f_tensor['quantization']['scale']
        
        input_multiplier, input_shift = get_quantized_multiplier(input_scale)
        filter_multiplier, filter_shift = get_quantized_multiplier(filter_scale)
        output_multiplier, output_shift = get_quantized_multiplier(output_scale)

        sn.activation_max = 127
        sn.activation_min = -128

        sn.input_multiplier = len(weights)
        fmt = "{}i".format(len(input_multiplier))
        weights += struct.pack(fmt, *input_multiplier)

        sn.input_shift = len(weights)
        fmt = "{}i".format(len(input_shift))
        weights += struct.pack(fmt, *input_shift)

        sn.output_multiplier = len(weights)
        fmt = "{}i".format(len(output_multiplier))
        weights += struct.pack(fmt, *output_multiplier)

        sn.output_shift = len(weights)
        fmt = "{}i".format(len(output_shift))
        weights += struct.pack(fmt, *output_shift)

        sn.MinMaxOptions.filter_multiplier = len(weights)
        fmt = "{}i".format(len(filter_multiplier))
        weights += struct.pack(fmt, *filter_multiplier)

        sn.MinMaxOptions.filter_shift = len(weights)
        fmt = "{}i".format(len(filter_shift))
        weights += struct.pack(fmt, *filter_shift)
        
        sn.MinMaxOptions.filter_offset = filter_offset
        sn.MinMaxOptions.left_shift = 8

        sn.MinMaxOptions.filter_data = len(weights)
        fmt = "{}b".format(len(filter_data.flatten()))
        weights += struct.pack(fmt, *filter_data.flatten())

        sn.input_offset = input_offset
        sn.output_offset = output_offset

        subnode_array.append(sn)

    elif subcode == "CAST":

        sn.type = getattr(BuiltinOperator, f"{subcode}")

        subnode_array.append(sn)
    else:
        sys.stderr.write('ERROR: Subnode {} not implemented\n'.format(subcode))
        sys.exit(1)

    sn.num_inputs = len(sn.tensor_array)

    tn = Tensor()
    tn.type = sn.output_data_type
    tn.shape = output_shape
    tn.dims = odims
    tn.scale = output_scale
    tn.zero = output_offset
    tn.id = o_tensor['buffer'] - 1
    tn.name = o_tensor['name']
    if not node_offloaded and engine_graphs_nx:
        add_external_producer_consumer_info(tn, external_inputs, external_outputs, already_external_producer, already_external_consumer, subgraph_idx)
    sn.tensor_array.append(tn)
    sn.num_outputs = 1 #TODO currently must have 1 output

    sn.num_tensors = len(sn.tensor_array)

    assert(sn.num_outputs + sn.num_inputs == sn.num_tensors)

    return subnode_array, sn.tensor_array, subop
