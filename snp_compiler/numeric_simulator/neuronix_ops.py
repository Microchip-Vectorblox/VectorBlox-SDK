import sys

from compiler.cbc_creator import calculate_pairing
sys.path.append('.')
import math
import numpy as np
import torch
from numeric_simulator.defs import qTensor
from common.hw_config import BIAS_FRACTIONAL_BITS, MAX_GRID_HEIGHT, MAX_GRID_WIDTH, MAX_BIAS_BITS, MAX_MAC_BITS, MAX_REDUCE_BUS_WIDTH, FRACTIONAL_BITS,\
                MCHP_MAC_TO_RQ_BUS_WIDTH, FINAL_RESULTS_BITS, OVERFLOW_EXTRA_BITS, INT_SCALE_BITS, MAC_ROUGH_SHIFT_GRANULARITY, TFLITE_REQUANT
from common.debug_flags import CHECK_MAC_PAIRING_ON_FLY, DEBUG_MAC_MULTIPLICATIONS,DEBUG_MAX_OC_FOR_MULTIPLICATIONS_DEBUG, DEBUG_MAX_IC_FOR_MULTIPLICATIONS_DEBUG, DEBUG_PAIRING_USED, DEBUG_TIMING,\
                DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS, DEBUG_MAC_MULTIPLICATIONS_LAYERS, DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC,\
                DEBUG_SIMULATE_FOLDING,DEBUG_NX_NUMERICS,DEBUG_ALLOW_14BIT_BIAS,DEBUG_ALLOW_ADDITIONAL_1BIT_TO_MACTORQ_BUS,\
                DEBUG_SIMULATOR_CLIP_ADD_BIAS_TO_INT12, DEBUG_SIMULATE_CONCAT_REQUANT, SIMULATE_IC_SPLIT, DEBUG_UNFOLD_INPUTS_BEFORE_CONCAT, \
                SIM_OUTPUT_MATRIX_CBC_FORMAT, DEBUG_TFLITE_REQUANT_SATURATE

import compiler.folding_algo as folding_algo
import datetime
from common.ddr_ir import create_tsnp_tensor_byte_array, create_tsnp_tensor_xwrap_array, create_nxd_byte_array
import copy
import os

int8min = np.iinfo(np.int8).min
int8max = np.iinfo(np.int8).max

def sign_extention(input_array,bits): # bits number means INTbits so if bits == 14 this is converted to INT14 (bit14 is sign bit)
    sign_bit = 1 << (bits - 1) # This will be 8192 in case of bits == 14
    output = (input_array & (sign_bit-1)) - (input_array & sign_bit)
    return output

def pad_small_output(output,output_tensor):
    if DEBUG_SIMULATE_FOLDING:
        real_output_shape = output_tensor.get_folded_shape()
    else:
        real_output_shape = output_tensor.get_original_shape()

    original_output_shape = output.shape
    if real_output_shape[3]!=original_output_shape[3]:
        #raise ValueError ('Didnt expect that. Please check...')
        output=output[:,:,0:real_output_shape[2],0:real_output_shape[3]]
        padding = original_output_shape[3]-real_output_shape[3]
        padding_val = output_tensor.zero_point
        output=np.pad(output,[(0,0),(0,0),(0,padding),(0,padding)], mode='constant', constant_values=padding_val)
    return output
        
def export_mac_output_to_debug_file(filename,mac_int,oc_processing_order):
    c = mac_int.shape[1]
    h = mac_int.shape[2]
    w = mac_int.shape[3]
    with open(filename,'w') as filehandler:
        for oc_idx in range(min(c,DEBUG_MAX_OC_FOR_MULTIPLICATIONS_DEBUG)):
            channel = oc_processing_order[oc_idx]
            for y in range(h):
                for x in range(w):
                    filehandler.write ('MAC value in (c=%d,h=%d,w=%d) = %d \n' % (channel,y,x,mac_int[0,channel,y,x]))

def export_mac_calcs_to_debug_file(filename,node,mac_int,q_input_padded,q_w_int8_np,requant_scale_uint14,requant_bias_int12,
                mac_shifted,rescaled_mac,rescaled_mac_biased,rescaled_mac_shifted,output,oc_processing_order):
    if 'input_channels_reorder_dict' in node['backend']:
        input_channels_reorder_dict = node['backend']['input_channels_reorder_dict']
        reordered_channels = [item[0] for item in input_channels_reorder_dict.items()]
        q_w_int8_np = q_w_int8_np[:,reordered_channels,:,:]
    c = mac_int.shape[1]
    h = mac_int.shape[2]
    w = mac_int.shape[3]
    k_x=q_w_int8_np.shape[3]
    k_y=q_w_int8_np.shape[2]
    input_channels = q_w_int8_np.shape[1]
    output_channels = q_w_int8_np.shape[0]
    per_ic_group_sorted_weight_activation_pairs = node['backend']['per_ic_group_sorted_weight_activation_pairs']
    
    #alex debug file  matrix without RQ
                
    with open(filename,'w') as filehandler:        
        for current_oc_idx in range(min(DEBUG_MAX_OC_FOR_MULTIPLICATIONS_DEBUG,output_channels)):
            if current_oc_idx>=len(oc_processing_order):
                continue
            current_oc = oc_processing_order[current_oc_idx]
            if len(DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC)>0 and current_oc not in DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC:
                continue
            mtx_before_rq = np.zeros((w,h))
            for current_h in range(h):
                for current_w in range(w):
                    current_mac = mac_int[0,current_oc,current_h,current_w]
                    my_str = 'MAC Calcs for pixel (c=%d,h=%d,w=%d): %d\n' % (current_oc,current_h,current_w, current_mac)
                    filehandler.write(my_str)
                    mac=0
                    for current_k_x in range(k_x):
                        for current_k_y in range(k_y):
                            mblock_str = 'mblock (%d,%d)\n' % (((current_h+current_k_y-1+MAX_GRID_HEIGHT) % MAX_GRID_HEIGHT), ((current_w+current_k_x-1+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                            filehandler.write(mblock_str)
                            #for current_group_pairs in per_ic_group_sorted_weight_activation_pairs:
                            #    for current_ic,activation_index in current_group_pairs:
                            for current_ic in range(min(DEBUG_MAX_IC_FOR_MULTIPLICATIONS_DEBUG, input_channels)):
                                current_weight = q_w_int8_np[current_oc,current_ic,current_k_y,current_k_x]
                                current_input = q_input_padded[0,current_ic,current_h+current_k_y,current_w+current_k_x]
                                if current_weight != 0:
                                    mac=mac+current_weight*current_input
                                    my_str = 'Input (%d) x Weight (%d) = %d, weight/input location [ic=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (current_input,current_weight,\
                                        current_weight*current_input,current_ic,current_ic,current_k_y,current_k_x, mac) 
                                    filehandler.write(my_str)
                    if (k_x==3):
                        file_str = 'mblock (%d,%d)\n' % (((current_h-1+MAX_GRID_HEIGHT) % MAX_GRID_HEIGHT), ((current_w+2+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                        filehandler.write(file_str)
                        file_str = 'mblock (%d,%d)\n' % (((current_h+MAX_GRID_HEIGHT) % MAX_GRID_HEIGHT), ((current_w+2+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                        filehandler.write(file_str)                    
                                        
                    my_str = 'Sum = %d\n' % mac
                    mtx_before_rq[current_w,current_h] = mac
                    filehandler.write(my_str)
                    if type(requant_scale_uint14) == int: # Conv of Add op has scale and bias of type int and not list
                        rqparams_str = 'scale_14bit = 0x%X, bias_13bits = 0x%X\n' % (requant_scale_uint14,requant_bias_int12)
                    else:
                        rqparams_str = 'scale_14bit = 0x%X, bias_13bits = 0x%X\n' % (requant_scale_uint14[current_oc],requant_bias_int12[current_oc])
                    filehandler.write(rqparams_str)
                    rq_calcs_str = 'mac_shifted (mac output after rough shift)= 0x%X, rescaled_mac (after mul with uint14 scale)= 0x%X, rescaled_mac_shifted (after cut to INT14)= 0x%X, +bias = 0x%X, output(shifted+clipped) = 0x%X\n' %\
                            (mac_shifted[0,current_oc,current_h,current_w],rescaled_mac[0,current_oc,current_h,current_w],\
                            rescaled_mac_shifted[0,current_oc,current_h,current_w],rescaled_mac_biased[0,current_oc,current_h,current_w],\
                            output[0,current_oc,current_h,current_w])
                    filehandler.write(rq_calcs_str)

def export_mac_calcs_to_debug_file_tflite(filename,node,mac_int,q_input_padded,q_w_int8_np, multiplier, shift, biasH, biasL, output, oc_processing_order):
    if 'input_channels_reorder_dict' in node['backend']:
        input_channels_reorder_dict = node['backend']['input_channels_reorder_dict']
        reordered_channels = [item[0] for item in input_channels_reorder_dict.items()]
        q_w_int8_np = q_w_int8_np[:,reordered_channels,:,:]
    c = mac_int.shape[1]
    h = mac_int.shape[2]
    w = mac_int.shape[3]
    k_x=q_w_int8_np.shape[3]
    k_y=q_w_int8_np.shape[2]
    input_channels = q_w_int8_np.shape[1]
    output_channels = q_w_int8_np.shape[0]

    with open(filename,'w') as filehandler:        
        for current_oc_idx in range(min(DEBUG_MAX_OC_FOR_MULTIPLICATIONS_DEBUG,output_channels)):
            if current_oc_idx>=len(oc_processing_order):
                continue
            current_oc = oc_processing_order[current_oc_idx]
            if len(DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC)>0 and current_oc not in DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC:
                continue
            mtx_before_rq = np.zeros((w,h))
            for current_h in range(h):
                for current_w in range(w):
                    current_mac = mac_int[0,current_oc,current_h,current_w]
                    my_str = 'MAC Calcs for pixel (c=%d,h=%d,w=%d): %d\n' % (current_oc,current_h,current_w, current_mac)
                    filehandler.write(my_str)
                    mac=0
                    mac_pair = 0
                    for current_k_x in range(k_x):
                        for current_k_y in range(k_y):
                            one_by_one_conv_acc = 0
                            mblock_str = 'mblock (%d,%d)\n' % (((current_h+current_k_y-1+MAX_GRID_HEIGHT) % MAX_GRID_HEIGHT), ((current_w+current_k_x-1+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                            filehandler.write(mblock_str)
                            for current_ic in range(min(DEBUG_MAX_IC_FOR_MULTIPLICATIONS_DEBUG, input_channels)):
                                current_weight = q_w_int8_np[current_oc,current_ic,current_k_y,current_k_x]
                                current_input = q_input_padded[0,current_ic,current_h+current_k_y,current_w+current_k_x]
                                if current_weight != 0:
                                    mac                 += current_weight*current_input
                                    one_by_one_conv_acc += current_weight*current_input
                                    my_str = 'Input (%d) x Weight (%d) = %d, weight/input location [ic=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (current_input,current_weight,\
                                        current_weight*current_input,current_ic,current_ic,current_k_y,current_k_x, mac) 
                                    filehandler.write(my_str)

                            if CHECK_MAC_PAIRING_ON_FLY == True:
                                vct_current_weight = q_w_int8_np[current_oc,:,current_k_y,current_k_x]
                                vct_current_input  = q_input_padded[0,:,current_h+current_k_y,current_w+current_k_x]
                                if sum(abs(vct_current_weight)) == 0: # kernrl is 0- kernel
                                    continue
                                
                                one_by_one_conv_acc_pair = 0
                                if DEBUG_PAIRING_USED:
                                    pos_pairs, neg_pairs, non_pairs_inx = calculate_pairing(kernel_1x1=vct_current_weight)
                                    
                                else:
                                    non_pairs_inx  = np.array([i for i, num in enumerate(vct_current_weight) if num != 0])
                                    pos_pairs = []
                                    neg_pairs = []
                                    

                                my_str = 'Pairing version\n'

                                filehandler.write(my_str)
                                #pairs pos                                
                                for i_calc in pos_pairs:
                                    in_0 = vct_current_input[i_calc[0]]
                                    in_1 = vct_current_input[i_calc[1]]
                                    w_all= vct_current_weight[i_calc[0]]
                                    ic0  = i_calc[0]
                                    ic1  = i_calc[1]
                                    mac_pair+= (in_0+in_1)*w_all
                                    my_str = '[Input0 (%d)+ Input1(%d)]  x Weight (%d) = %d, weight/input location [ic0=%d(@amm=%d),ic1=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (in_0,in_1, w_all, (in_0+in_1)*w_all,ic0, ic0, ic1,ic1,current_k_y,current_k_x, mac_pair) 
                                    filehandler.write(my_str)
                                #pairs_neg
                                for i_calc in neg_pairs:
                                    in_0 = vct_current_input[i_calc[0]]
                                    in_1 = vct_current_input[i_calc[1]]
                                    w_all= vct_current_weight[i_calc[0]]
                                    ic0  = i_calc[0]
                                    ic1  = i_calc[1]
                                    mac_pair+= (in_0-in_1)*w_all
                                    my_str = '[Input0 (%d)- Input1(%d)]  x Weight (%d) = %d, weight/input location [ic0=%d(@amm=%d),ic1=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (in_0,in_1, w_all, (in_0-in_1)*w_all,ic0, ic0, ic1,ic1,current_k_y,current_k_x, mac_pair) 
                                    filehandler.write(my_str)
                                #no pairs
                                half_len = len(non_pairs_inx)//2
                                delta = len(non_pairs_inx)%2
                                for i_c in range(half_len):
                                    i_calc_w0 = non_pairs_inx[i_c]
                                    i_calc_w1 = non_pairs_inx[i_c+half_len+delta]

                                    in_0_w0 = vct_current_input[ i_calc_w0]
                                    in_0_w1 = vct_current_input[ i_calc_w1]

                                    w_w0= vct_current_weight[i_calc_w0]
                                    w_w1= vct_current_weight[i_calc_w1]

                                    ic0_w0  =  i_calc_w0
                                    ic0_w1  =  i_calc_w1

                                    mac_pair+= in_0_w0*w_w0+in_0_w1*w_w1
                                    my_str = '[Input_w0 (%d)]  x Weight_w0 (%d) + [Input_w1 (%d)]  x Weight_w1 (%d)= %d, weight/input location [ic0_w0=%d(@amm=%d),ic0_w1=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (in_0_w0, w_w0, in_0_w1, w_w1, (in_0_w0*w_w0 + in_0_w1*w_w1 ),ic0_w0, ic0_w0,ic0_w1,ic0_w1, current_k_y,current_k_x, mac_pair) 
                                    filehandler.write(my_str)
                                ##### add the odd wights if exixsts    
                                if len(non_pairs_inx)%2:
                                    i_calc_w0 = non_pairs_inx[half_len]
                                    i_calc_w1 = -1

                                    in_0_w0 = vct_current_input[i_calc_w0]
                                    in_0_w1 = 0

                                    w_w0= vct_current_weight[i_calc_w0]
                                    w_w1= 0

                                    ic0_w0  =  i_calc_w0
                                    ic0_w1  =  i_calc_w1
                                    mac_pair+= in_0_w0*w_w0+in_0_w1*w_w1
                                    my_str = '[Input_w0 (%d)]  x Weight_w0 (%d) + [Input_w1 (%d)]  x Weight_w1 (%d)= %d, weight/input location [ic0_w0=%d(@amm=%d),ic0_w1=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (in_0_w0, w_w0, in_0_w1, w_w1, (in_0_w0*w_w0 + in_0_w1*w_w1 ),ic0_w0, ic0_w0,ic0_w1,ic0_w1, current_k_y,current_k_x, mac_pair) 
                                    filehandler.write(my_str)

                                    
                                        

                                if len(neg_pairs)>0:
                                    one_by_one_conv_acc_pair += np.dot(vct_current_weight[neg_pairs[:,0]],(vct_current_input[neg_pairs[:,0]]-vct_current_input[neg_pairs[:,1]]))
                                if len(pos_pairs)>0:
                                    one_by_one_conv_acc_pair += np.dot(vct_current_weight[pos_pairs[:,0]],(vct_current_input[pos_pairs[:,0]]+vct_current_input[pos_pairs[:,1]]))
                                if len(non_pairs_inx)>0:        
                                    one_by_one_conv_acc_pair += np.dot(vct_current_weight[non_pairs_inx],vct_current_input[non_pairs_inx])
                                if (one_by_one_conv_acc!=one_by_one_conv_acc_pair):
                                    raise ValueError ('Pairing cacculation is not correct. Please check it') 


                    if (k_x==3):
                        file_str = 'mblock (%d,%d)\n' % (((current_h-1+MAX_GRID_HEIGHT) % MAX_GRID_HEIGHT), ((current_w+2+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                        filehandler.write(file_str)
                        file_str = 'mblock (%d,%d)\n' % (((current_h+MAX_GRID_HEIGHT) % MAX_GRID_HEIGHT), ((current_w+2+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                        filehandler.write(file_str)                    
                                        
                    my_str = 'Sum = %d\n' % mac
                    mtx_before_rq[current_w,current_h] = mac
                    filehandler.write(my_str)
                    if type(multiplier) == int: # Conv of Add op has scale and bias of type int and not list
                        rqparams_str = 'multiplier = %d, shift = %d, bias_H = %d, bias_L = %d\n' % (multiplier, shift, biasH, biasL)
                    else:
                        rqparams_str = 'multiplier = %d, shift = %d, bias_H = %d, bias_L = %d\n' % (multiplier[current_oc], shift[current_oc], biasH[current_oc], biasL[current_oc])
                    filehandler.write(rqparams_str)
                    in_hex = hex(output[0,current_oc,current_h,current_w] & 0xFF)[2:].upper().zfill(2)
                    rq_calcs_str = 'output (mac output after tflite requant)= %d, (%s)\n' % (output[0,current_oc,current_h,current_w], in_hex)
                    filehandler.write(rq_calcs_str)

def export_mac_calcs_to_debug_file_maxpool(filename, node, q_input_padded, q_w_int8_np, output, oc_processing_order):
    if 'input_channels_reorder_dict' in node['backend']:
        input_channels_reorder_dict = node['backend']['input_channels_reorder_dict']
        reordered_channels = [item[0] for item in input_channels_reorder_dict.items()]
        q_w_int8_np = q_w_int8_np[:,reordered_channels,:,:]
    c = output.shape[1]
    h = output.shape[2]
    w = output.shape[3]
    k_x=q_w_int8_np.shape[3]
    k_y=q_w_int8_np.shape[2]
    input_channels = q_w_int8_np.shape[1]
    output_channels = q_w_int8_np.shape[0]
    print(c, h, w, k_x, k_y, input_channels, output_channels)

    with open(filename,'w') as filehandler:        
        for current_oc_idx in range(min(DEBUG_MAX_OC_FOR_MULTIPLICATIONS_DEBUG,output_channels)):
            if current_oc_idx>=len(oc_processing_order):
                continue
            current_oc = oc_processing_order[current_oc_idx]
            if len(DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC)>0 and current_oc not in DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC:
                continue
            mtx_before_rq = np.zeros((w,h))
            for current_h in range(h):
                for current_w in range(w):
                    my_str = 'MAC Calcs for pixel (c=%d,h=%d,w=%d): \n' % (current_oc,current_h,current_w)
                    filehandler.write(my_str)
                    mac=-128
                    for current_k_x in range(k_x):
                        for current_k_y in range(k_y):
                            mblock_str = 'mblock (%d,%d)\n' % (((current_h+current_k_y-1+MAX_GRID_HEIGHT) % MAX_GRID_HEIGHT), ((current_w+current_k_x-1+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                            filehandler.write(mblock_str)
                            for current_ic in range(min(DEBUG_MAX_IC_FOR_MULTIPLICATIONS_DEBUG, input_channels)):
                                current_weight = q_w_int8_np[current_oc,current_ic,current_k_y,current_k_x]
                                current_input = q_input_padded[0,current_ic,current_h+current_k_y,current_w+current_k_x]
                                if current_weight != 0:
                                    mac=max(mac, current_input)
                                    my_str = 'Input (%d) , Weight (%d) , weight/input location [ic=%d(@amm=%d),ky=%d,kx=%d, acc=%d]\n' % (current_input,current_weight,\
                                        current_ic,current_ic,current_k_y,current_k_x, mac) 
                                    filehandler.write(my_str)
                    if (k_x==3):
                        file_str = 'mblock (%d,%d)\n' % (((current_h-1+MAX_GRID_HEIGHT) % MAX_GRID_HEIGHT), ((current_w+2+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                        filehandler.write(file_str)
                        file_str = 'mblock (%d,%d)\n' % (((current_h+MAX_GRID_HEIGHT) % MAX_GRID_HEIGHT), ((current_w+2+MAX_GRID_WIDTH) % MAX_GRID_WIDTH))
                        filehandler.write(file_str)                    
                                        
                    my_str = 'Max = %d\n' % mac
                    mtx_before_rq[current_w,current_h] = mac
                    filehandler.write(my_str)
                    rq_calcs_str = 'output (mac output after tflite requant)= %d\n' % (output[0,current_oc,current_h,current_w])
                    filehandler.write(rq_calcs_str)

###################################
def export_mac_calcs_to_debug_file_alex(filename,node,mac_int,q_input_padded,q_w_int8_np,requant_scale_uint14,requant_scale_uint10,requant_scale_uint17,requant_scale_f_uint2,requant_bias_int12,mac_shifted,rescaled_mac,rescaled_mac_biased,rescaled_mac_shifted,output,oc_processing_order):
    if 'input_channels_reorder_dict' in node['backend']:
        input_channels_reorder_dict = node['backend']['input_channels_reorder_dict']
        reordered_channels = [item[0] for item in input_channels_reorder_dict.items()]
        q_w_int8_np = q_w_int8_np[:,reordered_channels,:,:]
    c = mac_int.shape[1]
    h = mac_int.shape[2]
    w = mac_int.shape[3]
    k_x=q_w_int8_np.shape[3]
    k_y=q_w_int8_np.shape[2]
    input_channels = q_w_int8_np.shape[1]
    output_channels = q_w_int8_np.shape[0]
    per_ic_group_sorted_weight_activation_pairs = node['backend']['per_ic_group_sorted_weight_activation_pairs']
    
    #alex debug file  matrix 
    name_f, ext = os.path.splitext(filename)
    new_filename = f"{name_f+'_ALEX_MTRX'}{ext}"

    with open(new_filename,'w') as filehandler:
        
        for current_oc_idx in range(min(DEBUG_MAX_OC_FOR_MULTIPLICATIONS_DEBUG,output_channels)):
            if current_oc_idx>=len(oc_processing_order):
                continue
            current_oc = oc_processing_order[current_oc_idx]
            if len(DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC)>0 and current_oc not in DEBUG_MAC_MULTIPLICATIONS_SPECIFIC_OC:
                continue
            mtx_before_rq = np.zeros((w,h))
            inx_of_h_y = np.arange(h)
            inx_of_w_x = np.arange(w)
            cnv_mtx = np.zeros((14,16))
            my_str = '===> Channal %d \n' % (current_oc)
            filehandler.write(my_str)
            
            #current_mac = mac_int[0,current_oc,current_h,current_w]
            mac=0
            for current_k_x in range(k_x):
                for current_k_y in range(k_y):            
        
                    current_weight_vct = q_w_int8_np[current_oc,:,current_k_y,current_k_x]
                    h_indices, w_indices = np.meshgrid(inx_of_h_y+current_k_y, inx_of_w_x+current_k_x, indexing="ij")
                    current_input_vct  = q_input_padded[0,:,h_indices,w_indices]


                    inx_not_zero      = (current_weight_vct!=0)
                    weights_no_zeros  = current_weight_vct [inx_not_zero]                            
                    list_half_len = -(-len(weights_no_zeros)//2)

                    ## padding to 14, 16 matrix
                    pad_y  = (0, max(0,14 - current_input_vct.shape[1]))  
                    pad_x  = (0, max(0,16 - current_input_vct.shape[1]))   
                    pad_0 = (0, 0)       
                    current_input_vct = np.pad(current_input_vct, (pad_y, pad_x, pad_0), mode='constant', constant_values=0)
                    current_input_vct =current_input_vct[:14,:16,inx_not_zero]
                    


                    #for the odd number of calculation, add one zero calculation, to save the simetry of wloc
                    if len(weights_no_zeros) % 2 ==True:
                        weights_no_zeros = np.concatenate((weights_no_zeros, [0]))
                        current_input_vct =np.concatenate((current_input_vct,  current_input_vct[:,:,1:2]),axis=2)
                        
                    
                    
                    if len(weights_no_zeros) < 2:
                        wgh_mtx  = [0, 0]
                        amm_mtrx = [current_input_vct[:, :,1], current_input_vct[:, :,1]]
                    else:
                        kernel_1x1_no_zeros_inx_half_0 = weights_no_zeros[:list_half_len]
                        kernel_1x1_no_zeros_inx_half_1 = weights_no_zeros[list_half_len:]
                        
                        for i_deepth in range (list_half_len): 
                            wgh_mtx  = [weights_no_zeros[i_deepth], weights_no_zeros[i_deepth+list_half_len]]
                            amm_mtrx = [current_input_vct[ :, :, i_deepth], current_input_vct[ :, :, i_deepth+list_half_len]]
                            cnv_mtx+= wgh_mtx[0]*amm_mtrx[0]+wgh_mtx[0]*amm_mtrx[0]

                            filehandler.write("weight" + "\n")
                            filehandler.write(str(wgh_mtx)  + "\n")
                            
                            filehandler.write("amm" + "\n")
                            for mtx in amm_mtrx:
                                for row in mtx:
                                    filehandler.write(" ".join(f"{x:+08}" for x in row) + "\n")
                                filehandler.write("\n")

                            
                            filehandler.write("cnv" + "\n")                                                                   
                            for row in cnv_mtx:
                                filehandler.write(" ".join(f"{x:+08}" for x in row) + "\n")
                            filehandler.write("\n")

                            # mark shifts
                    if (current_k_x !=2):
                        filehandler.write(" SHD" + "\n")
                if (current_k_y !=2):
                        filehandler.write(" SHR" + "\n")    
                                
            filehandler.write(" SHD" + "\n")                              
            filehandler.write(" SHR" + "\n") 

###################################
                        
def export_tensor_to_nxo(filename,output,oc_processing_order, x_slices = 1, x_folding = 0, y_folding = 0, is_intermediate_node = False, is_split_node = False, is_resize = False):
    oc_processing_order = list(range(len(oc_processing_order))) ## alex changed the order of OC to be in right order
    
    if not DEBUG_SIMULATE_FOLDING:
        raise ValueError ('Currently, export to nxo only supported if simulate folding is enabled.')
    shuffled_tensor = output[0,oc_processing_order,:]
    shuffled_tensor = np.expand_dims(shuffled_tensor,axis=0)
    tensor_byte_array = create_tsnp_tensor_byte_array(shuffled_tensor)
    # Creating .nxd files for comparison with hardware
    if is_resize:
        # Unfolding for hardware output comparison, since AMM to DDR write happens
        if (y_folding > 0):
            shuffled_tensor = folding_algo.get_asym_unfolded_tensor(shuffled_tensor,folding_factor_x=0,folding_factor_y=y_folding)
            y_folding = 0
    
    
    with open(filename,'wb') as bin_file:
        bin_file.write(tensor_byte_array)
    
    create_nxd_byte_array(shuffled_tensor, is_intermediate_node = is_intermediate_node, num_xslices = x_slices, y_folding = y_folding, is_split_node = is_split_node, filename = filename)


def compute_mac_shift(node_name, mac_int, requant_scale_float, requant_scale_shift, mac_rough_shift_mux, is_conv_add = False):
    leftover_shift = 0
    if isinstance(requant_scale_float,float): #In case of non per channel data we convert to array with shape (1,) so that later broadcast of shift will be done on right axis
        requant_scale_float = np.array([requant_scale_float])
    needed_shift_right = np.minimum(FRACTIONAL_BITS - np.trunc(np.log2(np.array(requant_scale_float)))+BIAS_FRACTIONAL_BITS,31) # This is how much we want to shift right after multiplication in
    # integer scale in order to get to INT8. we split it to 2 parts: nedded_shift_right - BIAS_FRACTIONAL_BITS is done right after integer scale multiplication
    # and additional BIAS_FRACTIONAL_BITS is needed after bias add
    # FINAL_RESULTS_BITS+OVERFLOW_EXTRA_BITS = MAC_BITS+INT_SCALE_BITS-needed_shift_right =>
    # MAC_BITS = FINAL_RESULTS_BITS+OVERFLOW_EXTRA_BITS-INT_SCALE_BITS+needed_shift_right =>
    # MAC_BITS = 8+4-10+needed_shift_right
    # So in order to get to bus of size MCHP_MAC_TO_RQ_BUS_WIDTH we need to shift MAC result right in MAC_BITS - MCHP_MAC_TO_RQ_BUS_WIDTH
    # so per_och_mac_shift is 8+4-10+needed_shift_right - MCHP_MAC_TO_RQ_BUS_WIDTH = needed_shift_right+2 - MCHP_MAC_TO_RQ_BUS_WIDTH
    # leftover shift (shift needed afterint scale mult) = needed_shift_right - per_och_mac_shift - BIAS_FRACTIONAL_BITS
    # And additional BIAS_FRACTIONAL_BITS shift right is done after bias add
    expected_mac_bits = needed_shift_right + FINAL_RESULTS_BITS + OVERFLOW_EXTRA_BITS - INT_SCALE_BITS # This is the expected number of bits in the mac result
    #selected_rough_shift_msb,actual_shift = get_rough_shift_msb(expected_mac_bits)
    # We split the shift right to 2 parts. We shift as minimal as possible in the mac in order reduce the bus size between MAC and RQ
    # The rest will be shifted in RQ after we multiply in integer scale.
    per_och_mac_shift = np.array(expected_mac_bits - MCHP_MAC_TO_RQ_BUS_WIDTH) # We shift right to limit MAC to MCHP_MAC_TO_RQ_BUS_WIDTH bits
    # NOTE: Negative shift is possible if its -1 or -2 since rough shift will be 0 and fine shift left will be done by integer scale.
    # We achieve the per och mac shift by rough shift right between MAC and RQ (multiplexer) and fine shift left by adding bits to the int scale (UINT13 instead of UINT10)
    rough_shift_right = np.array(mac_rough_shift_mux)
    rough_shift_right_bits = rough_shift_right * MAC_ROUGH_SHIFT_GRANULARITY
    if DEBUG_NX_NUMERICS and np.any(rough_shift_right_bits>3*MAC_ROUGH_SHIFT_GRANULARITY):
        raise ValueError ('Not enough rough shift bits. please check')
    leftover_shift = (np.array(requant_scale_shift,dtype=np.int64) - BIAS_FRACTIONAL_BITS) - per_och_mac_shift
    leftover_shift = np.array(np.expand_dims(leftover_shift,(1,2)),dtype=np.int64) # Need to expand dims so when we shift numpy knows which axis to broadcast
    if DEBUG_NX_NUMERICS:
        if np.any(leftover_shift!=14):
            raise ValueError ('leftover shift !=14, please check calc')
    if is_conv_add:
        rough_shift_right_bits = np.array(np.expand_dims(rough_shift_right_bits,(1,2)),dtype=np.int64) # Need to expand dims so when we shift numpy knows which axis to broadcast               
    mac_shifted = mac_int >> rough_shift_right_bits

    if DEBUG_NX_NUMERICS:
        if math.log(abs(mac_shifted).max()+sys.float_info.min,2)>MCHP_MAC_TO_RQ_BUS_WIDTH-1:
            # Dans: note to self: in the expected_mac_num_bits calc (few lines above), if we change the +2 to +1 we sometimes exceed the expected max mac bits. This could be logical if using histogram calibration but we got it in min/max calibration. need to check why...
            problematic_channels = np.where(np.log2(np.amax(abs(mac_shifted),axis=(2,3)))>MCHP_MAC_TO_RQ_BUS_WIDTH-1)[1]
            print('At layer: %s , %d channels exceded MCHP_MAC_TO_RQ_BUS_WIDTH' % (node_name,len(problematic_channels)))
            if DEBUG_ALLOW_ADDITIONAL_1BIT_TO_MACTORQ_BUS:
                mac_shifted = sign_extention(mac_shifted,MCHP_MAC_TO_RQ_BUS_WIDTH+1)
            else:
                mac_shifted = sign_extention(mac_shifted,MCHP_MAC_TO_RQ_BUS_WIDTH)    
    
    return mac_shifted, leftover_shift

def saturate(x, minRange, maxRange):
    sat = x<minRange or x>maxRange
    x = max(min(x, maxRange), minRange)
    return x, sat

def compute_tflite_mac_shift(node_name, mac_requant, output_channels, output_multiplier, cInputH, cInputL, o_shift):
    output = np.zeros(mac_requant.shape, dtype=np.int64)
    for och in range(output_channels):
        for r in range(mac_requant.shape[2]):
            for c in range(mac_requant.shape[3]):
                acc = mac_requant[0, och, r, c]
                accL = acc % 65536            # lower 16 bits
                accH = (acc - accL) >> 16     # upper bits
                if isinstance(output_multiplier, int):
                    p0 = accH * output_multiplier + cInputH
                    c1 = cInputL + (p0 << 16)
                    p1 = accL * output_multiplier + c1
                    res = int(p1 >> o_shift)
                else:
                    p0 = accH * output_multiplier[och] + cInputH[och]
                    c1 = cInputL[och] + (p0 << 16)
                    p1 = accL * output_multiplier[och] + c1
                    res = int(p1 >> o_shift[och])
                output[0, och, r, c], sat = saturate(res, int8min, int8max)
                if DEBUG_TFLITE_REQUANT_SATURATE:
                    if sat:
                        _, sat11 = saturate(res, -2**10, 2**10-1)
                        if sat11:
                            raise ValueError ('11-bit Saturated @ Node: ' + node_name + ', Channel: ' + str(och) + ', row: ' + str(r) + ', col: ' + str(c))
    return output

def nx_conv(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None):
    start_time = datetime.datetime.now()
    folding_conv_x = 'force_folding_x' in node['frontend']
    folding_conv_y = 'force_folding_y' in node['frontend']
    unfolding_conv_y = 'force_unfolding_y' in node['frontend']
    
    if DEBUG_SIMULATE_FOLDING:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_folded_shape(folding_conv_y=folding_conv_y,unfolding_conv_y=unfolding_conv_y)
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape(folding_conv_x=folding_conv_x,
                                                                                        producing_node_stride=node['frontend']['stride']) # In case of folding conv we want to get the shape before output folding) 
    else:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape()

    input_channels = folded_input_tensor_shape[1]
    output_channels = folded_output_tensor_shape[1]
    if 'force_unfolding_x' in node['frontend']:
        output_channels = output_channels * 2
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']
    if (input_folding_factor_x>0 or input_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING:
        kernel_size = node['frontend']['folded_kernel_size']
        stride = 1
        q_w_int8_np = node['frontend']['folded_weights_tensor'].data
        padding = node['frontend']['folded_padding']
        if not TFLITE_REQUANT:
            requant_bias_int12 = node['frontend']['folded_requant_bias_int12']
            # New MAC shift logic
            requant_scale_uint14 = node['frontend']['folded_requant_scale_uint14']
            mac_rough_shift_mux = node['frontend']['folded_mac_rough_shift_mux']
            requant_scale_shift = node['frontend']['folded_requant_scale_shift']
            requant_scale_float = node['frontend']['folded_requant_scale_float']
    else:
        kernel_size = node['frontend']['kernel_size']
        stride = node['frontend']['stride']
        q_w_int8_np = node['frontend']['weights_tensor'].data
        padding = node['frontend']['padding']

        if not TFLITE_REQUANT:
            requant_bias_int12 = node['frontend']['requant_bias_int12']
            # New MAC shift logic
            requant_scale_uint14 = node['frontend']['requant_scale_uint14']
            mac_rough_shift_mux = node['frontend']['mac_rough_shift_mux']
            requant_scale_shift = node['frontend']['requant_scale_shift']
            requant_scale_float = node['frontend']['requant_scale_float']
    
    mac_output = np.zeros(folded_output_tensor_shape,dtype=np.int64) # This is the shape before the hw output folding
    ic_splits = node['backend']['ic_splits']
    ic_groups = node['backend']['ic_groups']
    
    #load quantized input with 0 padding
    input_name = node['inputs'][0]
    if input_name in model_inputs:
        input = copy.deepcopy(model_inputs[input_name]) # We copy the input tensor sice we might fold it and dont want to change the inputs dict
        if (input_folding_factor_x>0 or input_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING:
            input.data = create_tsnp_tensor_xwrap_array(input.data, input_folding_factor_x=input_folding_factor_x)
            input.data = folding_algo.get_asym_folded_input(input.data,input_folding_factor_x=0,input_folding_factor_y=input_folding_factor_y)
            input.folding_factor_x = input_folding_factor_x
            input.folding_factor_y = input_folding_factor_y
    elif input_name in intermediate_tensors:
        input = intermediate_tensors[input_name]
    else:
        raise ValueError ('Op %s input: %s not found!!' % (node_name,input_name))

    if folding_conv_y and DEBUG_SIMULATE_FOLDING:
        y_folded_input = folding_algo.get_asym_folded_input(input.data,input_folding_factor_x=0,input_folding_factor_y=1)
    elif unfolding_conv_y and DEBUG_SIMULATE_FOLDING:
        y_folded_input = folding_algo.get_asym_unfolded_tensor(input.data,folding_factor_x=0,folding_factor_y=1)
    else:
        y_folded_input = input.data
    npd=np.array(y_folded_input,dtype=np.int64)
    if np.any(input.real_tensor_data!=None):
        real_input_tensor = input.real_tensor_data
        if folding_conv_y and DEBUG_SIMULATE_FOLDING:
            real_input_tensor = folding_algo.get_asym_folded_input(real_input_tensor,input_folding_factor_x=0,input_folding_factor_y=1)
        if unfolding_conv_y and DEBUG_SIMULATE_FOLDING:
            real_input_tensor = folding_algo.get_asym_unfolded_tensor(real_input_tensor,folding_factor_x=0,folding_factor_y=1)
    else:
        real_input_tensor = input.data
    real_input_tensor_int = np.array(real_input_tensor,dtype=np.int64)
    pad_val = node['frontend']['input_tensor_zp']

    if ('_fold_x' not in node_name):
        if TFLITE_REQUANT:
            # if (npd.shape[2] % 2 == 0) and (node['frontend']['output_tensor'].get_original_shape()[2] % 2 == 1):
            #     output_dims_diff = npd.shape[2] * node['frontend']['stride'] - node['frontend']['output_tensor'].get_original_shape()[2]
            #     if (output_dims_diff == 1) and (padding == [0,0,0,0]):
            #         padding = [1, 1, 1, 1]
            q_input_padded=torch.nn.functional.pad(torch.tensor(npd), padding, 'constant', float(pad_val)).numpy()
            real_input_tensor_padded = torch.nn.functional.pad(torch.tensor(real_input_tensor_int), padding, 'constant', float(pad_val)).numpy()
        else:
            q_input_padded=np.pad(npd,[(0,0),(0,0),(padding,padding),(padding,padding)], mode='constant', constant_values=pad_val)
            real_input_tensor_padded = np.pad(real_input_tensor_int,[(0,0),(0,0),(padding,padding),(padding,padding)], mode='constant', constant_values=pad_val)
    else:
        q_input_padded = npd
        real_input_tensor_padded = real_input_tensor_int
    
    per_ic_split_output = []
    for current_ic_split in range(ic_splits):        
        if SIMULATE_IC_SPLIT:
            weights = q_w_int8_np[:,ic_groups[current_ic_split],:,:]
            input_fp32 = q_input_padded.astype(np.float32)[:,ic_groups[current_ic_split],:,:]
            actual_input_channels = input_channels // ic_splits
        else:
            weights=q_w_int8_np
            input_fp32 = q_input_padded.astype(np.float32)
            actual_input_channels = input_channels
        
        m = torch.nn.Conv2d(actual_input_channels, output_channels, kernel_size, stride, padding = 0, bias=False)
        md = dict()
        md["weight"] = torch.tensor(weights) 
        m.load_state_dict(md)
        mac_output=m(torch.tensor(input_fp32)).detach().numpy()
        mac_int = mac_output.astype(np.int64)
        if TFLITE_REQUANT:
            mac_int = np.where(mac_int<0, mac_int | ~((1 << 29) - 1), mac_int & ((1 << 29) - 1))
            output_multiplier = node['frontend']['output_multiplier'] 
            cInputH = node['frontend']['cInputH']
            cInputL = node['frontend']['cInputL']
            o_shift = node['frontend']['o_shift']
            mac_shifted = compute_tflite_mac_shift(node_name, mac_int, output_channels, output_multiplier, cInputH, cInputL, o_shift)
        else:
            mac_int =np.where(mac_int<0,mac_int | ~((1 << MAX_MAC_BITS) - 1), mac_int & ((1 << MAX_MAC_BITS) - 1))
            mac_shifted, leftover_shift = compute_mac_shift(node_name, mac_int, requant_scale_float, requant_scale_shift, mac_rough_shift_mux, is_conv_add=True)
        if SIMULATE_IC_SPLIT:
            per_ic_split_output.append(mac_shifted)
    
    if SIMULATE_IC_SPLIT:
        mac_shifted=np.zeros(per_ic_split_output[0].shape,dtype=np.int64)
        for current_split_output in per_ic_split_output:
            mac_shifted += current_split_output
        
    if TFLITE_REQUANT:
        output = mac_shifted
    else:
        per_och_requant_scale = np.array(requant_scale_uint14,dtype=np.int64)
        if isinstance(per_och_requant_scale,np.ndarray) and len(per_och_requant_scale.shape)!=0:
            per_och_requant_scale = np.expand_dims(per_och_requant_scale,(1,2)) # Need to expand dims so when we multiply numpy knows which axis to broadcast
        rescaled_mac = mac_shifted*per_och_requant_scale
        per_och_add = np.array(requant_bias_int12,dtype=np.int64)
        if DEBUG_NX_NUMERICS and np.log(np.abs(per_och_add).max())>=MAX_BIAS_BITS:            
            print('At layer: %s , bias exceeds MAX_BIAS_BITS (INT%d)' % (node_name,MAX_BIAS_BITS+1))
            if DEBUG_ALLOW_14BIT_BIAS:
                per_och_add = np.where(per_och_add<0,per_och_add | ~((1 << MAX_BIAS_BITS+1) - 1), per_och_add & ((1 << MAX_BIAS_BITS+1) - 1))
            else:
                per_och_add = np.where(per_och_add<0,per_och_add | ~((1 << MAX_BIAS_BITS) - 1), per_och_add & ((1 << MAX_BIAS_BITS) - 1))
        if isinstance(per_och_add,np.ndarray) and len(per_och_add.shape)!=0:
            per_och_add = np.expand_dims(per_och_add,(1,2)) # Need to expand dims so when we add numpy knows which axis to broadcast

        rescaled_mac_shifted = rescaled_mac >> leftover_shift
        # After the shift we should have left with MAX_REDUCE_BUS_WIDTH (14bits including sign bit)
        if DEBUG_NX_NUMERICS:
            rescaled_mac_shifted = np.where(rescaled_mac_shifted<0,rescaled_mac_shifted | ~((1 << MAX_REDUCE_BUS_WIDTH-1) - 1), rescaled_mac_shifted & ((1 << MAX_REDUCE_BUS_WIDTH-1) - 1))
        rescaled_mac_biased = rescaled_mac_shifted + per_och_add    
        rescaled_mac_clipped = rescaled_mac_biased >> BIAS_FRACTIONAL_BITS
        rescaled_mac_clipped = np.clip(rescaled_mac_clipped,0,255)
        output = rescaled_mac_clipped

    if 'backend' in node and 'oc_order' in node['backend'] and DEBUG_SIMULATE_FOLDING:
        oc_processing_order = node['backend']['oc_order']
    else:
        oc_processing_order = [oc for oc in range(output_channels)]
    if (folding_conv_x and output_folding_factor_x>0) or (folding_conv_y and output_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING: # output_folding_factor>0 means its a 28x28 folding conv which doesnt drop 3/4 of output channels. ==0 means its a 14x14 folding conv which drops 3/4 of output channels (to execute stride=2)
        channels_multiplier = 1
        pre_folding_oc_processing_order = oc_processing_order
        if folding_conv_x:
            pre_folding_oc_processing_order = list((np.array(oc_processing_order)[::2]/2).astype(int))
            channels_multiplier*=2
        if folding_conv_y>0:
            pass # Nothing to do here since y folding is done in input
    else:
        pre_folding_oc_processing_order = oc_processing_order
    if DEBUG_MAC_MULTIPLICATIONS and debug_dir:
        if (len(DEBUG_MAC_MULTIPLICATIONS_LAYERS)>0 and node_name in DEBUG_MAC_MULTIPLICATIONS_LAYERS) or len(DEBUG_MAC_MULTIPLICATIONS_LAYERS)==0:
            mac_calcs_filename = os.path.join(debug_dir,(node_name+'_MAC_CALCS_DEBUG.txt'))
            mac_debug_filename = os.path.join(debug_dir,(node_name+'_MAC_DEBUG.txt'))

            if SIM_OUTPUT_MATRIX_CBC_FORMAT == True:
                export_mac_calcs_to_debug_file_alex(mac_calcs_filename,node,mac_int,real_input_tensor_padded,q_w_int8_np,requant_scale_uint14,requant_bias_int12,mac_shifted,\
                                        rescaled_mac,rescaled_mac_biased,rescaled_mac_shifted,output,pre_folding_oc_processing_order)
            else:
                if TFLITE_REQUANT:
                    export_mac_calcs_to_debug_file_tflite(mac_calcs_filename,node,mac_int,q_input_padded,q_w_int8_np, \
                            output_multiplier, o_shift, cInputH, cInputL, output, pre_folding_oc_processing_order)
                else:
                    export_mac_calcs_to_debug_file(mac_calcs_filename,node,mac_int,real_input_tensor_padded,q_w_int8_np,requant_scale_uint14, requant_bias_int12, \
                            mac_shifted, rescaled_mac,rescaled_mac_biased,rescaled_mac_shifted,output,pre_folding_oc_processing_order)
                
    if ('lut_silu' in node['frontend']):
        output_uint8 = output.astype(np.uint8)
        nlf = node['backend']['nlf']
        cmd_mem = np.array(nlf[0].cmd_mem, dtype=np.int8)
        output = cmd_mem[output_uint8]

    if ('activation_silu' in node['attributes']) and (node['attributes']['activation_silu'] != None):
        output_pad_value = node['attributes']['activation_silu']['output_zp'][0]
    else:
        output_pad_value = node['frontend']['output_tensor_zp']

    if (output.shape[3] != math.ceil(output.shape[3]/16)*16):
        padding_width = math.ceil(output.shape[3]/16)*16 - output.shape[3]
        padding_height = 0
        if ('pad_extra_line' in node['frontend']) and node['frontend']['pad_extra_line']:
            padding_height += 1
        output = np.pad(output, ((0, 0), (0, 0), (0, padding_height), (0, padding_width)), mode="constant", constant_values=output_pad_value)      
    
    if (node['backend']['output_padding_start_x'] < 15):
        output_pad_idxs = 15 - node['backend']['output_padding_start_x']
        output[:,:,:,-output_pad_idxs:] = output_pad_value
    if (node['backend']['output_padding_start_y'] < 14) and (output.shape[2] > 14):
        output_pad_idxs = 14 - node['backend']['output_padding_start_y']
        output[:,:,-output_pad_idxs:,:] = output_pad_value

    if folding_conv_x and DEBUG_SIMULATE_FOLDING: # In case of hardware folding we do same (fold at output and drop 1/2 of output channels if stride==2)
        if (input_folding_factor_y > 0):
            output = folding_algo.get_asym_unfolded_tensor(output,folding_factor_x=0,folding_factor_y=input_folding_factor_y)
        output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=1,input_folding_factor_y=input_folding_factor_y)
        oc_processing_order = [oc for oc in range(output_channels*(2**(output_folding_factor_x-input_folding_factor_x)))]
        if (len(node['frontend']['preceding_nodes_params']) == 0):
            output_uint8 = output.astype(np.uint8)
            nlf = node['backend']['nlf']
            cmd_mem = np.array(nlf[0].cmd_mem, dtype=np.int8)
            output = cmd_mem[output_uint8]
        if node['frontend']['stride']==2:
            output=output[:,::2,:,:]
    
    if 'force_unfolding_x' in node['frontend']:
        oc_processing_order = list((np.array(oc_processing_order)[::2]/2).astype(int))
        output = folding_algo.get_asym_unfolded_tensor(output,folding_factor_x=1,folding_factor_y=input_folding_factor_y)
        if (node['frontend']['output_tensor'].x_slices != 2*node['frontend']['x_slices']):
            output_width = 16*node['frontend']['output_tensor'].x_slices
            output = output[:,:,:,:output_width]
        if (input_folding_factor_y > 0):
            output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=0,input_folding_factor_y=input_folding_factor_y)

    if DEBUG_SIMULATE_FOLDING and folding_conv_y and output_folding_factor_y==0: # This means its a folding conv x for stride implementation
        if node['frontend']['stride']!=2:
            raise ValueError ('Expected this to be a stided conv. Please check...')
        pass # Nothing to do here as y folding is done on input the stride will be implemented by setting of weights to drop 1/2 of the output channels
    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
        x_slices = node['frontend']['output_tensor'].x_slices
        is_split_node = False
        if '_split' in node_name:
            is_split_node = True
        export_tensor_to_nxo(debug_tensor_filename,output,oc_processing_order, is_intermediate_node = is_intermediate_node, x_slices = x_slices, x_folding = output_folding_factor_x, y_folding = output_folding_factor_y, is_split_node=is_split_node)

    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output_x_slices = node['frontend']['output_tensor'].x_slices
    output_tensor = qTensor(output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
    output_name = node['outputs'][0]
    intermediate_tensors[output_name] = output_tensor
    if DEBUG_TIMING>0:
        if DEBUG_MAC_MULTIPLICATIONS:
            print('Warning: DEBUG_MAC_MULTIPLICATIONS is on so timing includes debug file creation (which is significant)')
        print('At layer: %s, total time: %s' % (node_name,str(datetime.datetime.now()-start_time)))

    return [output_tensor]

def nx_identity(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None):
    start_time = datetime.datetime.now()
    folding_conv_x = 'force_folding_x' in node['frontend']
    folding_conv_y = 'force_folding_y' in node['frontend']
    unfolding_conv_y = 'force_unfolding_y' in node['frontend']
    if DEBUG_SIMULATE_FOLDING:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_folded_shape(folding_conv_y=folding_conv_y,unfolding_conv_y=unfolding_conv_y)
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape(folding_conv_x=folding_conv_x,
                                                                                        producing_node_stride=node['frontend']['stride']) # In case of folding conv we want to get the shape before output folding)
    else:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape()
    
    input_channels = folded_input_tensor_shape[1]
    output_channels = folded_output_tensor_shape[1]
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']
    if (input_folding_factor_x>0 or input_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING:
        kernel_size = node['frontend']['folded_kernel_size']
        stride = 1
        q_w_int8_np = node['frontend']['folded_weights_tensor'].data
        padding = node['frontend']['folded_padding']
        if not TFLITE_REQUANT:
            requant_bias_int12 = node['frontend']['folded_requant_bias_int12']
            # New MAC shift logic
            requant_scale_uint14 = node['frontend']['folded_requant_scale_uint14']
            mac_rough_shift_mux = node['frontend']['folded_mac_rough_shift_mux']
            requant_scale_shift = node['frontend']['folded_requant_scale_shift']
            requant_scale_float = node['frontend']['folded_requant_scale_float']
        
    else:
        kernel_size = node['frontend']['kernel_size']
        stride = node['frontend']['stride']
        q_w_int8_np = node['frontend']['weights_tensor'].data
        padding = node['frontend']['padding']
        if not TFLITE_REQUANT:
            requant_bias_int12 = node['frontend']['requant_bias_int12']
            # New MAC shift logic
            requant_scale_uint14 = node['frontend']['requant_scale_uint14']
            mac_rough_shift_mux = node['frontend']['mac_rough_shift_mux']
            requant_scale_shift = node['frontend']['requant_scale_shift']
            requant_scale_float = node['frontend']['requant_scale_float']
        
    mac_output = np.zeros(folded_output_tensor_shape,dtype=np.int64) # This is the shape before the hw output folding
    ic_splits = node['backend']['ic_splits']
    ic_groups = node['backend']['ic_groups']
   
    #load quantized input with 0 padding
    input_name = node['inputs'][0]
    if input_name in model_inputs:
        input = copy.deepcopy(model_inputs[input_name]) # We copy the input tensor sice we might fold it and dont want to change the inputs dict
        if (input_folding_factor_x>0 or input_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING:
            input.data = folding_algo.get_asym_folded_input(input.data,input_folding_factor_x=input_folding_factor_x,input_folding_factor_y=input_folding_factor_y)
            input.folding_factor_x = input_folding_factor_x
            input.folding_factor_y = input_folding_factor_y
    elif input_name in intermediate_tensors:
        input = intermediate_tensors[input_name]
    else:
        raise ValueError ('Op %s input:%s not found!!' % (node_name,input_name))

    if folding_conv_y and DEBUG_SIMULATE_FOLDING:
        y_folded_input = folding_algo.get_asym_folded_input(input.data,input_folding_factor_x=0,input_folding_factor_y=1)
    elif unfolding_conv_y and DEBUG_SIMULATE_FOLDING:
        y_folded_input = folding_algo.get_asym_unfolded_tensor(input.data,folding_factor_x=0,folding_factor_y=1)
    else:
        y_folded_input = input.data
    npd=np.array(y_folded_input,dtype=np.int64)
    if np.any(input.real_tensor_data!=None):
        real_input_tensor = input.real_tensor_data
        if folding_conv_y and DEBUG_SIMULATE_FOLDING:
            real_input_tensor = folding_algo.get_asym_folded_input(real_input_tensor,input_folding_factor_x=0,input_folding_factor_y=1)
        if unfolding_conv_y and DEBUG_SIMULATE_FOLDING:
            real_input_tensor = folding_algo.get_asym_unfolded_tensor(real_input_tensor,folding_factor_x=0,folding_factor_y=1)
    else:
        real_input_tensor = input.data
    real_input_tensor_int = np.array(real_input_tensor,dtype=np.int64)
    pad_val = node['frontend']['input_tensor_zp']
    q_input_padded=np.pad(npd,[(0,0),(0,0),(padding,padding),(padding,padding)], mode='constant', constant_values=pad_val)
    real_input_tensor_padded = np.pad(real_input_tensor_int,[(0,0),(0,0),(padding,padding),(padding,padding)], mode='constant', constant_values=pad_val)
    per_ic_split_output = []
    for current_ic_split in range(ic_splits):    
        if SIMULATE_IC_SPLIT:
            weights = q_w_int8_np[:,ic_groups[current_ic_split],:,:]
            input_fp32 = q_input_padded.astype(np.float32)[:,ic_groups[current_ic_split],:,:]
            actual_input_channels = input_channels // ic_splits
        else:
            weights=q_w_int8_np
            input_fp32 = q_input_padded.astype(np.float32)
            actual_input_channels = input_channels
        m = torch.nn.Conv2d(actual_input_channels, output_channels, kernel_size, stride,bias=False)
        md = dict()
        md["weight"] = torch.tensor(weights) 
        m.load_state_dict(md)
        mac_output=m(torch.tensor(input_fp32)).detach().numpy()
        mac_int = mac_output.astype(np.int64)
        if TFLITE_REQUANT:
            mac_int = np.where(mac_int<0, mac_int | ~((1 << 29) - 1), mac_int & ((1 << 29) - 1))
            output_multiplier = node['frontend']['output_multiplier'] 
            cInputH = node['frontend']['cInputH']
            cInputL = node['frontend']['cInputL']
            o_shift = node['frontend']['o_shift']
            mac_shifted = compute_tflite_mac_shift(node_name, mac_int, output_channels, output_multiplier, cInputH, cInputL, o_shift)
        else:
            mac_int =np.where(mac_int<0,mac_int | ~((1 << MAX_MAC_BITS) - 1), mac_int & ((1 << MAX_MAC_BITS) - 1))
            mac_shifted, leftover_shift = compute_mac_shift(node_name, mac_int, requant_scale_float, requant_scale_shift, mac_rough_shift_mux, is_conv_add=False)
        if SIMULATE_IC_SPLIT:
            per_ic_split_output.append(mac_shifted)
    
    if SIMULATE_IC_SPLIT:
        mac_shifted=np.zeros(per_ic_split_output[0].shape,dtype=np.int64)
        for current_split_output in per_ic_split_output:
            mac_shifted+= current_split_output
    if TFLITE_REQUANT:
        output = mac_shifted
    else:
        per_och_requant_scale = np.array(requant_scale_uint14,dtype=np.int64)
        if isinstance(per_och_requant_scale,np.ndarray) and len(per_och_requant_scale.shape)!=0:
            per_och_requant_scale = np.expand_dims(per_och_requant_scale,(1,2)) # Need to expand dims so when we multiply numpy knows which axis to broadcast
        rescaled_mac = mac_shifted*per_och_requant_scale
        per_och_add = np.array(requant_bias_int12,dtype=np.int64)
        if DEBUG_NX_NUMERICS and np.log(np.abs(per_och_add).max())>=MAX_BIAS_BITS:        
            print('At layer: %s , bias exceeds MAX_BIAS_BITS (INT%d)' % (node_name,MAX_BIAS_BITS+1))
            if DEBUG_ALLOW_14BIT_BIAS:
                per_och_add = np.where(per_och_add<0,per_och_add | ~((1 << MAX_BIAS_BITS+1) - 1), per_och_add & ((1 << MAX_BIAS_BITS+1) - 1))
            else:
                per_och_add = np.where(per_och_add<0,per_och_add | ~((1 << MAX_BIAS_BITS) - 1), per_och_add & ((1 << MAX_BIAS_BITS) - 1))
        if isinstance(per_och_add,np.ndarray) and len(per_och_add.shape)!=0:
            per_och_add = np.expand_dims(per_och_add,(1,2)) # Need to expand dims so when we add numpy knows which axis to broadcast
        rescaled_mac_shifted = rescaled_mac >> leftover_shift
        # After the shift we should have left with MAX_REDUCE_BUS_WIDTH (14bits including sign bit)
        if DEBUG_NX_NUMERICS:
            rescaled_mac_shifted = np.where(rescaled_mac_shifted<0,rescaled_mac_shifted | ~((1 << MAX_REDUCE_BUS_WIDTH-1) - 1), rescaled_mac_shifted & ((1 << MAX_REDUCE_BUS_WIDTH-1) - 1))
        rescaled_mac_biased = rescaled_mac_shifted + per_och_add
        rescaled_mac_clipped = rescaled_mac_biased >> BIAS_FRACTIONAL_BITS
        rescaled_mac_clipped = np.clip(rescaled_mac_clipped,0,255)
        output = rescaled_mac_clipped

    if 'backend' in node and 'oc_order' in node['backend'] and DEBUG_SIMULATE_FOLDING:
        oc_processing_order = node['backend']['oc_order']
    else:
        oc_processing_order = [oc for oc in range(output_channels)]
    if (folding_conv_x and output_folding_factor_x>0) or (folding_conv_y and output_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING: # output_folding_factor>0 means its a 28x28 folding conv which doesnt drop 3/4 of output channels. ==0 means its a 14x14 folding conv which drops 3/4 of output channels (to execute stride=2)
        channels_multiplier = 1
        pre_folding_oc_processing_order = oc_processing_order
        if folding_conv_x:
            pre_folding_oc_processing_order = list((np.array(oc_processing_order)[::2]/2).astype(int))
            channels_multiplier*=2
        if folding_conv_y>0:
            pass # Nothing to do here since y folding is done in input
    else:
        pre_folding_oc_processing_order = oc_processing_order
    if DEBUG_MAC_MULTIPLICATIONS and debug_dir:
        if (len(DEBUG_MAC_MULTIPLICATIONS_LAYERS)>0 and node_name in DEBUG_MAC_MULTIPLICATIONS_LAYERS) or len(DEBUG_MAC_MULTIPLICATIONS_LAYERS)==0:
            mac_calcs_filename = os.path.join(debug_dir,(node_name+'_MAC_CALCS_DEBUG.txt'))
            mac_debug_filename = os.path.join(debug_dir,(node_name+'_MAC_DEBUG.txt'))
            if TFLITE_REQUANT:
                export_mac_calcs_to_debug_file_tflite(mac_calcs_filename,node,mac_int,real_input_tensor_padded,q_w_int8_np, \
                        output_multiplier, o_shift, cInputH, cInputL, output, oc_processing_order)
            else:
                export_mac_calcs_to_debug_file(mac_calcs_filename,node,mac_int,real_input_tensor_padded,q_w_int8_np,requant_scale_uint14,requant_bias_int12,\
                            mac_shifted,rescaled_mac,rescaled_mac_biased,rescaled_mac_shifted,output,pre_folding_oc_processing_order)
            #export_mac_output_to_debug_file(mac_debug_filename,mac_int,oc_processing_order)
    
    if ('activation_silu' in node['attributes']) and (node['attributes']['activation_silu'] != None):
        output_pad_value = node['attributes']['activation_silu']['output_zp'][0]
    else:
        output_pad_value = node['frontend']['output_tensor_zp']
    
    if (output.shape[3] != math.ceil(output.shape[3]/16)*16):
        padding_width = math.ceil(output.shape[3]/16)*16 - output.shape[3]
        padding_height = 0
        if ('pad_extra_line' in node['frontend']) and node['frontend']['pad_extra_line']:
            padding_height += 1
        output = np.pad(output, ((0, 0), (0, 0), (0, padding_height), (0, padding_width)), mode="constant", constant_values=output_pad_value)

    if (node['backend']['output_padding_start_x'] < 15):
        output_pad_idxs = 15 - node['backend']['output_padding_start_x']
        output[:,:,:,-output_pad_idxs:] = output_pad_value
    if (node['backend']['output_padding_start_y'] < 14) and (output.shape[2] > 14):
        output_pad_idxs = 14 - node['backend']['output_padding_start_y']
        output[:,:,-output_pad_idxs:,:] = output_pad_value
    
    if folding_conv_x: # In case of hardware folding we do same (fold at output and drop 1/2 of output channels if stride==2)
        output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=1,input_folding_factor_y=0)
        if node['frontend']['stride']==2:
            output=output[:,::2,:,:]

    if DEBUG_SIMULATE_FOLDING and folding_conv_y and output_folding_factor_y==0: # This means its a folding conv x for stride implementation
        if node['frontend']['stride']!=2:
            raise ValueError ('Expected this to be a stided conv. Please check...')
        pass # Nothing to do here as y folding is done on input the stride will be implemented by setting of weights to drop 1/2 of the output channels
    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        export_tensor_to_nxo(debug_tensor_filename,output,oc_processing_order, is_intermediate_node = is_intermediate_node, x_slices = x_slices, x_folding = output_folding_factor_x, y_folding = output_folding_factor_y)

    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output_x_slices = node['frontend']['output_tensor'].x_slices
    output_tensor = qTensor(output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
    output_name = node['frontend']['output_tensor'].name
    intermediate_tensors[output_name] = output_tensor
    if DEBUG_TIMING>0:
        if DEBUG_MAC_MULTIPLICATIONS:
            print('Warning: DEBUG_MAC_MULTIPLICATIONS is on so timing includes debug file creation (which is significant)')
        print('At layer: %s, total time: %s' % (node_name,str(datetime.datetime.now()-start_time)))

    return [output_tensor]

def nx_gemm(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None):
    start_time = datetime.datetime.now()
    if DEBUG_SIMULATE_FOLDING:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_folded_shape()
    else:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()

    folding_conv_x = 'force_folding_x' in node['frontend']
    folding_conv_y = 'force_folding_y' in node['frontend']
    if DEBUG_SIMULATE_FOLDING:
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape(folding_conv_x=folding_conv_x,
                                                                                        producing_node_stride=node['frontend']['stride']) # In case of folding conv we want to get the shape before output folding
    else:
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape() # In case of folding conv we want to get the shape before output folding
    input_channels = folded_input_tensor_shape[1]
    output_channels = folded_output_tensor_shape[1]
    input_folding_factor = node['frontend']['input_folding_factor']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']
    q_w_int8_np = node['frontend']['weights_tensor'].data
    if not TFLITE_REQUANT:
        requant_bias_int12 = node['frontend']['requant_bias_int12']
        # New MAC shift logic
        requant_scale_uint14 = node['frontend']['requant_scale_uint14']
        requant_scale_shift = node['frontend']['requant_scale_shift']
        requant_scale_float = node['frontend']['requant_scale_float']
    mac_output = np.zeros(folded_output_tensor_shape,dtype=np.int64)
    
    #load quantized input with 0 padding
    input_name = node['inputs'][0]
    if input_name in model_inputs:
        input = copy.deepcopy(model_inputs[input_name]) # We copy the input tensor sice we might fold it and dont want to change the inputs dict
        if input_folding_factor>0 and DEBUG_SIMULATE_FOLDING:
            input.data = folding_algo.get_folded_input(input.data,input_folding_factor)
            input.folding_factor = input_folding_factor
    elif input_name in intermediate_tensors:
        input = intermediate_tensors[input_name]
    else:
        raise ValueError ('Op %s input:%s not found!!' % (node_name,input_name))

    npd=np.array(input.data,dtype=np.int64)
    kernel_size = 1
    stride = 1
    npd = np.expand_dims(npd,(2,3))
    m = torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride,bias=False)
    md = dict()
    converted_weights_tensor = np.expand_dims(q_w_int8_np.transpose(),(2,3))
    md["weight"] = torch.tensor(converted_weights_tensor) 
    inp = npd.astype(np.float32)
    m.load_state_dict(md)
    mac_output=m(torch.tensor(inp)).detach().numpy()
    mac_int = mac_output.astype(np.int64)
    if TFLITE_REQUANT:
        mac_int = np.where(mac_int<0, mac_int | ~((1 << 29) - 1), mac_int & ((1 << 29) - 1))
        output_multiplier = node['frontend']['output_multiplier'] 
        cInputH = node['frontend']['cInputH']
        cInputL = node['frontend']['cInputL']
        o_shift = node['frontend']['o_shift']
        output = compute_tflite_mac_shift(node_name, mac_int, output_channels, output_multiplier, cInputH, cInputL, o_shift)
    else:
        mac_int =np.where(mac_int<0,mac_int | ~((1 << MAX_MAC_BITS) - 1), mac_int & ((1 << MAX_MAC_BITS) - 1))
        leftover_shift = 0
        mac_shifted = mac_int
        per_och_requant_scale = np.array(requant_scale_uint14,dtype=np.int64)
        per_och_requant_scale = np.expand_dims(per_och_requant_scale,(1,2)) # Need to expand dims so when we multiply numpy knows which axis to broadcast
        rescaled_mac = mac_shifted*per_och_requant_scale

        per_och_add = np.array(requant_bias_int12,dtype=np.int64)
        if DEBUG_NX_NUMERICS and math.log(abs(per_och_add).max(),2)>=MAX_BIAS_BITS:
            print('At layer: %s , bias exceeds MAX_BIAS_BITS (INT%d)' % (node_name,MAX_BIAS_BITS+1))
            if DEBUG_ALLOW_14BIT_BIAS:
                per_och_add = np.where(per_och_add<0,per_och_add | ~((1 << MAX_BIAS_BITS+1) - 1), per_och_add & ((1 << MAX_BIAS_BITS+1) - 1))
            else:
                per_och_add = np.where(per_och_add<0,per_och_add | ~((1 << MAX_BIAS_BITS) - 1), per_och_add & ((1 << MAX_BIAS_BITS) - 1))

        per_och_add = np.expand_dims(per_och_add,(1,2)) # Need to expand dims so when we add numpy knows which axis to broadcast
        requant_scale_shift = np.expand_dims(requant_scale_shift,(1,2))
        rescaled_mac_shifted = rescaled_mac >> requant_scale_shift
        rescaled_mac_biased = rescaled_mac_shifted + per_och_add
        rescaled_mac_clipped = rescaled_mac_biased >> BIAS_FRACTIONAL_BITS
        rescaled_mac_clipped = np.clip(rescaled_mac_clipped,0,255)
        output = rescaled_mac_clipped
        oc_processing_order = [oc for oc in range(output_channels)]
        pre_folding_oc_processing_order = oc_processing_order

        if DEBUG_MAC_MULTIPLICATIONS and debug_dir:
            if (len(DEBUG_MAC_MULTIPLICATIONS_LAYERS)>0 and node_name in DEBUG_MAC_MULTIPLICATIONS_LAYERS) or len(DEBUG_MAC_MULTIPLICATIONS_LAYERS)==0:
                mac_calcs_filename = os.path.join(debug_dir,(node_name+'_MAC_CALCS_DEBUG.txt'))
                mac_debug_filename = os.path.join(debug_dir,(node_name+'_MAC_DEBUG.txt'))
                export_mac_calcs_to_debug_file(mac_calcs_filename,node,mac_int,npd,converted_weights_tensor,requant_scale_uint14,requant_bias_int12,\
                                            mac_shifted,rescaled_mac,rescaled_mac_biased,rescaled_mac_shifted,output,pre_folding_oc_processing_order)
                #export_mac_output_to_debug_file(mac_debug_filename,mac_int,oc_processing_order)

    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        export_tensor_to_nxo(debug_tensor_filename,output,oc_processing_order, is_intermediate_node = is_intermediate_node, x_slices = x_slices, x_folding = output_folding_factor_x, y_folding = output_folding_factor_y)
    
    output=np.squeeze(output,(2,3))
    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output_x_slices = node['frontend']['output_tensor'].x_slices
    output_tensor = qTensor(output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
    output_name = node['outputs'][0]
    intermediate_tensors[output_name] = output_tensor
    if DEBUG_TIMING>0:
        if DEBUG_MAC_MULTIPLICATIONS:
            print('Warning: DEBUG_MAC_MULTIPLICATIONS is on so timing includes debug file creation (which is significant)')
        print('At layer: %s, total time: %s' % (node_name,str(datetime.datetime.now()-start_time)))

    return [output_tensor]

def nx_add(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None):
    # Neuronix Add is executed by a convolution with pre-defined weights. The 2 inputs are concatenated in the cin axis to form the convolution's input
    start_time = datetime.datetime.now()
    folding_conv_x = 'force_folding_x' in node['frontend']
    folding_conv_y = 'force_folding_y' in node['frontend']
    unfolding_conv_y = 'force_unfolding_y' in node['frontend']
    if DEBUG_SIMULATE_FOLDING:
        folded_input_tensor_shape = node['frontend']['input_tensors'][0].get_folded_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape(folding_conv_x=folding_conv_x,
                                            producing_node_stride=node['frontend']['stride']) # In case of folding conv we want to get the shape before output folding
    else:
        folded_input_tensor_shape = node['frontend']['input_tensors'][0].get_original_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape() # In case of folding conv we want to get the shape before output folding

    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']

    input_channels = folded_input_tensor_shape[1]
    output_channels = folded_output_tensor_shape[1]
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    if (input_folding_factor_x>0 or input_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING:
        q_w_int8_np = node['frontend']['folded_weights_tensor'].data # This is calculated in the frontend hardware agnostic part
    else:
        q_w_int8_np = node['frontend']['weights_tensor'].data # This is calculated in the frontend hardware agnostic part

    kernel_size = 1
    stride = 1
    mac_output = np.zeros(folded_output_tensor_shape,dtype=np.int64)

    if not TFLITE_REQUANT:
        requant_bias_int12 = node['frontend']['requant_bias_int12']
        # New MAC shift logic
        requant_scale_shift = node['frontend']['requant_scale_shift']
        requant_scale_float = node['frontend']['requant_scale_float']
        # MCHP numerics
        requant_scale_uint14 = node['frontend']['requant_scale_uint14']
        mac_rough_shift_mux = node['frontend']['mac_rough_shift_mux']

    #load quantized input with 0 padding
    add_inputs=[]
    input0_name = node['inputs'][0]
    for input_name in node['inputs']:
        if input_name in model_inputs:
            input = model_inputs[input_name]
        elif input_name in intermediate_tensors:
            input = intermediate_tensors[input_name]
        else:
            raise ValueError ('Op %s input:%s not found!!' % (node_name,input_name))
        add_inputs.append(input)
    input = np.concatenate([add_inputs[0].data,add_inputs[1].data],axis=1)
    npd=np.array(input,dtype=np.int64)
    m = torch.nn.Conv2d(input_channels*2, output_channels, kernel_size, stride,bias=False)
    md = dict()
    md["weight"] = torch.tensor(q_w_int8_np) 
    inp = input.astype(np.float32)
    m.load_state_dict(md)
    mac_output=m(torch.tensor(inp)).detach().numpy()
    mac_int = mac_output.astype(np.int64)
    if TFLITE_REQUANT:
        mac_int = np.where(mac_int<0, mac_int | ~((1 << 29) - 1), mac_int & ((1 << 29) - 1))
        output_multiplier = node['frontend']['output_multiplier'] 
        cInputH = node['frontend']['cInputH']
        cInputL = node['frontend']['cInputL']
        o_shift = node['frontend']['o_shift']
        output = compute_tflite_mac_shift(node_name, mac_int, output_channels, output_multiplier, cInputH, cInputL, o_shift)
    else:
        mac_int =np.where(mac_int<0,mac_int | ~((1 << MAX_MAC_BITS) - 1), mac_int & ((1 << MAX_MAC_BITS) - 1))
        mac_shifted, leftover_shift = compute_mac_shift(node_name, mac_int, requant_scale_float, requant_scale_shift, mac_rough_shift_mux, is_conv_add=False)
        per_och_requant_scale = np.array(requant_scale_uint14,dtype=np.int64)
        if isinstance(per_och_requant_scale,np.ndarray) and len(per_och_requant_scale.shape)!=0:
            per_och_requant_scale = np.expand_dims(per_och_requant_scale,(1,2)) # Need to expand dims so when we multiply numpy knows which axis to broadcast
        rescaled_mac = mac_shifted*per_och_requant_scale
        per_och_add = np.array(requant_bias_int12,dtype=np.int64)
        if isinstance(per_och_add,np.ndarray) and len(per_och_add.shape)!=0:
            per_och_add = np.expand_dims(per_och_add,(1,2)) # Need to expand dims so when we add numpy knows which axis to broadcast
        if DEBUG_NX_NUMERICS and math.log(abs(per_och_add).max(),2)>=MAX_BIAS_BITS:
            print('At layer: %s , bias exceeds MAX_BIAS_BITS (INT%d)' % (node_name,MAX_BIAS_BITS+1))
            if DEBUG_SIMULATOR_CLIP_ADD_BIAS_TO_INT12:
                max_bias_int12 = 2 ** MAX_BIAS_BITS-1
                min_bias_int12 = -1*max_bias_int12
                requant_bias_int12 = np.minimum(max_bias_int12,np.maximum(min_bias_int12,requant_bias_int12)) # Clip bias to MAX_BIAS_BITS
                print('At layer %s, bias exceeded INT%d , clipping to (%d,%d)' % (node_name,MAX_BIAS_BITS+1,min_bias_int12,max_bias_int12))
            if DEBUG_ALLOW_14BIT_BIAS:
                requant_bias_int12 = np.where(requant_bias_int12<0,requant_bias_int12 | ~((1 << MAX_BIAS_BITS+1) - 1), requant_bias_int12 & ((1 << MAX_BIAS_BITS+1) - 1))
            else:
                requant_bias_int12 = np.where(requant_bias_int12<0,requant_bias_int12 | ~((1 << MAX_BIAS_BITS) - 1), requant_bias_int12 & ((1 << MAX_BIAS_BITS) - 1))
        rescaled_mac_shifted = rescaled_mac >> leftover_shift
        # After the shift we should have left with MAX_REDUCE_BUS_WIDTH (14bits including sign bit)
        if DEBUG_NX_NUMERICS:
            rescaled_mac_shifted = np.where(rescaled_mac_shifted<0,rescaled_mac_shifted | ~((1 << MAX_REDUCE_BUS_WIDTH-1) - 1), rescaled_mac_shifted & ((1 << MAX_REDUCE_BUS_WIDTH-1) - 1))
        rescaled_mac_biased = rescaled_mac_shifted + per_och_add   
        rescaled_mac_clipped = rescaled_mac_biased >> BIAS_FRACTIONAL_BITS
        rescaled_mac_clipped = np.clip(rescaled_mac_clipped,0,255)
        output = rescaled_mac_clipped

    if 'backend' in node and 'oc_order' in node['backend'] and DEBUG_SIMULATE_FOLDING:
        oc_processing_order = node['backend']['oc_order']
    else:
        oc_processing_order = [oc for oc in range(output_channels)]
    if (folding_conv_x and output_folding_factor_x>0) or (folding_conv_y and output_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING: # output_folding_factor>0 means its a 28x28 folding conv which doesnt drop 3/4 of output channels. ==0 means its a 14x14 folding conv which drops 3/4 of output channels (to execute stride=2):
        channels_multiplier = 1
        if output_folding_factor_x>0:
            channels_multiplier*=2
        if output_folding_factor_y>0:
            channels_multiplier*=2
        pre_folding_oc_processing_order = list((np.array(oc_processing_order)[::channels_multiplier]/channels_multiplier).astype(int))
    else:
        pre_folding_oc_processing_order = oc_processing_order

    if DEBUG_MAC_MULTIPLICATIONS and debug_dir:
        if (len(DEBUG_MAC_MULTIPLICATIONS_LAYERS)>0 and node_name in DEBUG_MAC_MULTIPLICATIONS_LAYERS) or len(DEBUG_MAC_MULTIPLICATIONS_LAYERS)==0:
            mac_calcs_filename = os.path.join(debug_dir,(node_name+'_MAC_CALCS_DEBUG.txt'))
            mac_debug_filename = os.path.join(debug_dir,(node_name+'_MAC_DEBUG.txt'))

            if SIM_OUTPUT_MATRIX_CBC_FORMAT:
                export_mac_calcs_to_debug_file_alex(mac_calcs_filename,node,mac_int,npd,q_w_int8_np,requant_scale_uint14, requant_bias_int12,mac_shifted,\
                                           rescaled_mac,rescaled_mac_biased,rescaled_mac_shifted,output,pre_folding_oc_processing_order)
            else:
                if TFLITE_REQUANT:
                    export_mac_calcs_to_debug_file_tflite(mac_calcs_filename, node, mac_int, npd, q_w_int8_np, \
                            output_multiplier, o_shift, cInputH, cInputL, output, pre_folding_oc_processing_order)
                else:
                    export_mac_calcs_to_debug_file(mac_calcs_filename, node, mac_int, npd, q_w_int8_np, requant_scale_uint14, requant_bias_int12,\
                                    mac_shifted, rescaled_mac, rescaled_mac_biased, rescaled_mac_shifted, output, pre_folding_oc_processing_order)
    

            #export_mac_output_to_debug_file(mac_debug_filename,mac_int,oc_processing_order)
    
    if ('activation_silu' in node['attributes']) and (node['attributes']['activation_silu'] != None):
        output_pad_value = node['attributes']['activation_silu']['output_zp'][0]
    else:
        output_pad_value = node['frontend']['output_tensor_zp']

    if (output.shape[3] != math.ceil(output.shape[3]/16)*16):
        padding_width = math.ceil(output.shape[3]/16)*16 - output.shape[3]
        padding_height = 0
        if ('pad_extra_line' in node['frontend']) and node['frontend']['pad_extra_line']:
            padding_height += 1
        output = np.pad(output, ((0, 0), (0, 0), (0, padding_height), (0, padding_width)), mode="constant", constant_values=node['frontend']['output_tensor'].zero_point)

    if (node['backend']['output_padding_start_x'] < 15):
        output_pad_idxs = 15 - node['backend']['output_padding_start_x']
        output[:,:,:,-output_pad_idxs:] = output_pad_value
    if (node['backend']['output_padding_start_y'] < 14) and (output.shape[2] > 14):
        output_pad_idxs = 14 - node['backend']['output_padding_start_y']
        output[:,:,-output_pad_idxs:,:] = output_pad_value
    
    if (folding_conv_x and output_folding_factor_x>0) or (folding_conv_y and output_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING: # output_folding_factor>0 means its a 28x28 folding conv which doesnt drop 3/4 of output channels. ==0 means its a 14x14 folding conv which drops 3/4 of output channels (to execute stride=2):
        output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=1,input_folding_factor_y=1)
    if DEBUG_SIMULATE_FOLDING and folding_conv_x and output_folding_factor_x==0: # This means its a folding conv x for stride implementation
        if node['frontend']['stride']!=2:
            raise ValueError ('Expected this to be a stided conv. Please check...')
        output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=1,input_folding_factor_y=0)
        output=output[:,::2,:,:]
    if DEBUG_SIMULATE_FOLDING and folding_conv_y and output_folding_factor_y==0: # This means its a folding conv x for stride implementation
        if node['frontend']['stride']!=2:
            raise ValueError ('Expected this to be a stided conv. Please check...')
        output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=0,input_folding_factor_y=1)
        output=output[:,::2,:,:]

    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        export_tensor_to_nxo(debug_tensor_filename,output,oc_processing_order, is_intermediate_node = is_intermediate_node, x_slices = x_slices, x_folding = output_folding_factor_x, y_folding = output_folding_factor_y)

    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output_x_slices = node['frontend']['output_tensor'].x_slices
    output_tensor = qTensor(output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
    output_name = node['outputs'][0]
    intermediate_tensors[output_name] = output_tensor
    if DEBUG_TIMING>0:
        if DEBUG_MAC_MULTIPLICATIONS:
            print('Warning: DEBUG_MAC_MULTIPLICATIONS is on so timing includes debug file creation (which is significant)')
        print('At layer: %s, total time: %s' % (node_name,str(datetime.datetime.now()-start_time)))

    return [output_tensor]

def nx_maxpool(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None):
    start_time = datetime.datetime.now()
    folding_conv_x = 'force_folding_x' in node['frontend']
    folding_conv_y = 'force_folding_y' in node['frontend']
    unfolding_conv_y = 'force_unfolding_y' in node['frontend']
    if DEBUG_SIMULATE_FOLDING:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_folded_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape(folding_conv_x=folding_conv_x,
                                            producing_node_stride=node['frontend']['stride']) # In case of folding conv we want to get the shape before output folding
    else:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape() # In case of folding conv we want to get the shape before output folding
    
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']

    kernel_size = node['frontend']['kernel_size']
    stride = node['frontend']['stride']
    padding = node['frontend']['padding']
    output = np.zeros(folded_output_tensor_shape,dtype=np.int64)
    mac_output = np.zeros(folded_output_tensor_shape,dtype=np.int64)
    
    if (input_folding_factor_x>0 or input_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING:
        q_w_int8_np = node['frontend']['folded_weights_tensor'].data # This is calculated in the frontend hardware agnostic part
    else:
        q_w_int8_np = node['frontend']['weights_tensor'].data # This is calculated in the frontend hardware agnostic part
    
    #load quantized input with 0 padding
    input_name = node['inputs'][0]
    if input_name in model_inputs:
        input = copy.deepcopy(model_inputs[input_name]) # We copy the input tensor sice we might fold it and dont want to change the inputs dict
        if (input_folding_factor_x>0 or input_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING:
            input.data = folding_algo.get_asym_folded_input(input.data,input_folding_factor_x=input_folding_factor_x,input_folding_factor_y=input_folding_factor_y)
            input.folding_factor_x = input_folding_factor_x
            input.folding_factor_y = input_folding_factor_y
    elif input_name in intermediate_tensors:
        input = intermediate_tensors[input_name]
    else:
        raise ValueError ('Op %s input:%s not found!!' % (node_name,input_name))

    if folding_conv_y and DEBUG_SIMULATE_FOLDING:
        y_folded_input = folding_algo.get_asym_folded_input(input.data,input_folding_factor_x=0,input_folding_factor_y=1)
    elif unfolding_conv_y and DEBUG_SIMULATE_FOLDING:
        y_folded_input = folding_algo.get_asym_unfolded_tensor(input.data,folding_factor_x=0,folding_factor_y=1)
    else:
        y_folded_input = input.data

    # In order to maxpool using pytorch maxpool op we need to unfold the input tensor if needed
    if DEBUG_SIMULATE_FOLDING:
        unfolded_input = folding_algo.get_asym_unfolded_tensor(y_folded_input,folding_factor_x=input_folding_factor_x,folding_factor_y=input_folding_factor_y)
    else:
        unfolded_input = y_folded_input

    npd=np.array(unfolded_input,dtype=np.int64)
    pad_val = node['frontend']['input_tensor_zp']
    if isinstance(padding, int):
        q_input_padded=np.pad(npd,[(0,0),(0,0),(padding,padding),(padding,padding)], mode='constant', constant_values=pad_val)
    else:
        q_input_padded=torch.nn.functional.pad(torch.tensor(npd), padding, 'constant', float(pad_val)).numpy()
    input_channels = node['frontend']['input_tensor'].get_original_shape()[1]
    m = torch.nn.MaxPool2d(kernel_size, stride=stride)
    inp = q_input_padded.astype(np.float32)
    mac_output=m(torch.tensor(inp)).detach().numpy()

    output = mac_output.astype(np.int64)
    output_channels = folded_output_tensor_shape[1]
    if 'backend' in node and 'oc_order' in node['backend'] and DEBUG_SIMULATE_FOLDING:
        oc_processing_order = node['backend']['oc_order']
    else:
        oc_processing_order = [oc for oc in range(output_channels)]
        
    if DEBUG_MAC_MULTIPLICATIONS and debug_dir:
        if (len(DEBUG_MAC_MULTIPLICATIONS_LAYERS)>0 and node_name in DEBUG_MAC_MULTIPLICATIONS_LAYERS) or len(DEBUG_MAC_MULTIPLICATIONS_LAYERS)==0:
            mac_calcs_filename = os.path.join(debug_dir,(node_name+'_MAC_CALCS_DEBUG.txt'))
            export_mac_calcs_to_debug_file_maxpool(mac_calcs_filename, node, q_input_padded, q_w_int8_np, output, oc_processing_order)

    if (folding_conv_x and output_folding_factor_x>0) or (folding_conv_y and output_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING: # output_folding_factor>0 means its a 28x28 folding conv which doesnt drop 3/4 of output channels. ==0 means its a 14x14 folding conv which drops 3/4 of output channels (to execute stride=2)
        pre_folding_oc_processing_order = list((np.array(oc_processing_order)[::4]/4).astype(int))
    else:
        pre_folding_oc_processing_order = oc_processing_order

    if ('activation_silu' in node['attributes']) and (node['attributes']['activation_silu'] != None):
        output_pad_value = node['attributes']['activation_silu']['output_zp'][0]
    else:
        output_pad_value = node['frontend']['output_tensor_zp']

    if (output.shape[3] != math.ceil(output.shape[3]/16)*16):
        padding_width = math.ceil(output.shape[3]/16)*16 - output.shape[3]
        padding_height = 0
        if ('pad_extra_line' in node['frontend']) and node['frontend']['pad_extra_line']:
            padding_height += 1
        output = np.pad(output, ((0, 0), (0, 0), (0, padding_height), (0, padding_width)), mode="constant", constant_values=output_pad_value)      
    
    if (node['backend']['output_padding_start_x'] < 15):
        output_pad_idxs = 15 - node['backend']['output_padding_start_x']
        output[:,:,:,-output_pad_idxs:] = output_pad_value
    if (node['backend']['output_padding_start_y'] < 14) and (output.shape[2] > 14):
        output_pad_idxs = 14 - node['backend']['output_padding_start_y']
        output[:,:,-output_pad_idxs:,:] = output_pad_value
    
    # fold output to match actual output on nx (nx uses conv mechanism to do maxpooling)
    if DEBUG_SIMULATE_FOLDING:
        output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=output_folding_factor_x,input_folding_factor_y=output_folding_factor_y) # If its a folding conv it will add this folding as well
    #output = pad_small_output(output,node['frontend']['output_tensor'])
    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        export_tensor_to_nxo(debug_tensor_filename,output,oc_processing_order, is_intermediate_node = is_intermediate_node, x_slices = x_slices, x_folding = output_folding_factor_x, y_folding = output_folding_factor_y)

    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output_x_slices = node['frontend']['output_tensor'].x_slices
    output_tensor = qTensor(output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
    output_name = node['outputs'][0]
    intermediate_tensors[output_name] = output_tensor
    if DEBUG_TIMING>0:
        if DEBUG_MAC_MULTIPLICATIONS:
            print('Warning: DEBUG_MAC_MULTIPLICATIONS is on so timing includes debug file creation (which is significant)')
        print('At layer: %s, total time: %s' % (node_name,str(datetime.datetime.now()-start_time)))

    return [output_tensor]

def nx_globalavgpool(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None):
    start_time = datetime.datetime.now()
    if DEBUG_SIMULATE_FOLDING:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_folded_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape() # In case of folding conv we want to get the shape before output folding
    else:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape() # In case of folding conv we want to get the shape before output folding

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']
    output = np.zeros(folded_output_tensor_shape,dtype=np.int64)
    mac_output = np.zeros(folded_output_tensor_shape,dtype=np.int64)

    #load quantized input with 0 padding
    input_name = node['inputs'][0]
    if input_name in model_inputs:
        input = copy.deepcopy(model_inputs[input_name]) # We copy the input tensor sice we might fold it and dont want to change the inputs dict
        if (input_folding_factor_x>0) or (input_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING:
            input.data = folding_algo.get_asym_folded_input(output,input_folding_factor_x=input_folding_factor_x,input_folding_factor_y=input_folding_factor_y)
            input.folding_factor_x = input_folding_factor_x
            input.folding_factor_y = input_folding_factor_y
    elif input_name in intermediate_tensors:
        input = intermediate_tensors[input_name]
    else:
        raise ValueError ('Op %s input:%s not found!!' % (node_name,input_name))

    # In order to pool using pytorch maxpool op we need to unfold the input tensor if needed
    if DEBUG_SIMULATE_FOLDING:
        unfolded_input = folding_algo.get_asym_unfolded_tensor(input.data,folding_factor_x=input_folding_factor_x,folding_factor_y=input_folding_factor_y)

    else:
        unfolded_input = input.data

    npd=np.array(unfolded_input,dtype=np.int64)
    original_height = node['frontend']['input_tensor'].get_original_shape()[2]
    original_width = node['frontend']['input_tensor'].get_original_shape()[3]
    npd=npd[:,:,0:original_height,0:original_width]
    input_scale = node['frontend']['input_tensor'].scale
    input_zero_point = node['frontend']['input_tensor'].zero_point

    dequantized_input = (npd-input_zero_point)* input_scale
    float_output = torch.mean(torch.tensor(dequantized_input),(2,3), True).detach().numpy()

    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output = (np.trunc(float_output/output_scale)+output_zero_point).astype(np.int64)
    output_channels = folded_output_tensor_shape[1]
    oc_processing_order = [oc for oc in range(output_channels)]
    pre_folding_oc_processing_order = oc_processing_order
    # fold output to match actual output on nx (nx uses conv mechanism to do maxpooling)
    output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=output_folding_factor_x,input_folding_factor_y=output_folding_factor_y) # If its a folding conv it will add this folding as well
    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        if (output.shape[3] != math.ceil(output.shape[3]/16)*16):
            padding_width = math.ceil(output.shape[3]/16)*16 - output.shape[3]
            if ('activation_silu' in node['attributes']) and (node['attributes']['activation_silu'] != None):
                output_pad_value = node['attributes']['activation_silu']['output_zp'][0]
            else:
                output_pad_value = node['frontend']['output_tensor_zp']
            output_padded = np.pad(output, ((0, 0), (0, 0), (0, 0), (0, padding_width)), mode="constant", constant_values=output_pad_value)
        else:
            output_padded = output 
        export_tensor_to_nxo(debug_tensor_filename,output_padded,oc_processing_order, is_intermediate_node = is_intermediate_node, x_slices = x_slices, x_folding = output_folding_factor_x, y_folding = output_folding_factor_y)

    output_x_slices = node['frontend']['output_tensor'].x_slices
    output_tensor = qTensor(output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
    output_name = node['outputs'][0]
    intermediate_tensors[output_name] = output_tensor
    if DEBUG_TIMING>0:
        if DEBUG_MAC_MULTIPLICATIONS:
            print('Warning: DEBUG_MAC_MULTIPLICATIONS is on so timing includes debug file creation (which is significant)')
        print('At layer: %s, total time: %s' % (node_name,str(datetime.datetime.now()-start_time)))

    return [output_tensor]

def nx_resize(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None):
    start_time = datetime.datetime.now()

    if '_split' in node_name:
        output_folding_factor_x = node['frontend']['output_folding_factor_x']
        output_folding_factor_y = node['frontend']['output_folding_factor_y']
        output_scale = node['frontend']['output_tensor'].scale
        output_zero_point = node['frontend']['output_tensor'].zero_point
        output_x_slices = node['frontend']['output_tensor'].x_slices

        output_name = node['outputs'][0].split('_split')[0]
        output_tensor = intermediate_tensors[output_name]
        output = output_tensor.data
        channels_selected = node['frontend']['output_channels'] * (2 ** output_folding_factor_x) * (2 ** output_folding_factor_y)
        split_output = output[:, -channels_selected:, :, :]
        output_tensor = qTensor(split_output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
        intermediate_tensors[node['outputs'][0]] = output_tensor
        oc_processing_order = [oc for oc in range(channels_selected)]

        if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
            debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
            x_slices = node['frontend']['x_slices']
            is_split_node = True
            if (split_output.shape[3] != math.ceil(split_output.shape[3]/16)*16):
                padding_width = math.ceil(split_output.shape[3]/16)*16 - split_output.shape[3]
                if ('activation_silu' in node['attributes']) and (node['attributes']['activation_silu'] != None):
                    output_pad_value = node['attributes']['activation_silu']['output_zp'][0]
                else:
                    output_pad_value = node['frontend']['output_tensor_zp']
                output_padded = np.pad(split_output, ((0, 0), (0, 0), (0, 0), (0, padding_width)), mode="constant", constant_values=output_pad_value)
            else:
                output_padded = split_output
            export_tensor_to_nxo(debug_tensor_filename,output_padded,oc_processing_order, is_intermediate_node = True, x_slices = x_slices, x_folding = output_folding_factor_x, y_folding = output_folding_factor_y, is_split_node=is_split_node, is_resize = True)
        
        return [output_tensor]

    if DEBUG_SIMULATE_FOLDING:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_folded_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape() # In case of folding conv we want to get the shape before output folding
    else:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape() # In case of folding conv we want to get the shape before output folding

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']
    output = np.zeros(folded_output_tensor_shape,dtype=np.int64)
    mac_output = np.zeros(folded_output_tensor_shape,dtype=np.int64)
    
    #load quantized input with 0 padding
    input_name = node['inputs'][0]
    if input_name in model_inputs:
        input = copy.deepcopy(model_inputs[input_name]) # We copy the input tensor sice we might fold it and dont want to change the inputs dict
        if (input_folding_factor_x>0 or input_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING:
            input.data = folding_algo.get_asym_folded_input(input.data,input_folding_factor_x=input_folding_factor_x,input_folding_factor_y=input_folding_factor_y)
            input.folding_factor_x = input_folding_factor_x
            input.folding_factor_y = input_folding_factor_y
    elif input_name in intermediate_tensors:
        input = intermediate_tensors[input_name]
    else:
        raise ValueError ('Op %s input:%s not found!!' % (node_name,input_name))

    folding_conv_y = 'force_folding_y' in node['frontend']
    unfolding_conv_y = 'force_unfolding_y' in node['frontend']
    if (folding_conv_y and output_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING: # y axis folding/unfolding is done on input read from DDR
        y_folded_input = folding_algo.get_asym_folded_input(input.data,input_folding_factor_x=0,input_folding_factor_y=1)
    elif unfolding_conv_y and DEBUG_SIMULATE_FOLDING: # y axis folding/unfolding is done on input read from DDR
        y_folded_input = folding_algo.get_asym_unfolded_tensor(input.data,folding_factor_x=0,folding_factor_y=1)
    else:
        y_folded_input = input.data

    # In order to resize using numpy repeat op we need to unfold the input tensor if needed
    if DEBUG_SIMULATE_FOLDING:
        unfolded_input = folding_algo.get_asym_unfolded_tensor(y_folded_input,folding_factor_x=input_folding_factor_x,folding_factor_y=input_folding_factor_y)
    else:
        unfolded_input = y_folded_input

    npd=np.array(unfolded_input,dtype=np.int64)
    original_height = node['frontend']['input_tensor'].get_original_shape()[2]
    original_width = node['frontend']['input_tensor'].get_original_shape()[3]
    original_channels = node['frontend']['input_tensor'].get_original_shape()[1]
    npd=npd[:,:,0:original_height,0:original_width]
    input_scale = node['frontend']['input_tensor'].scale
    input_zero_point = node['frontend']['input_tensor'].zero_point
    
    output = npd.repeat(2,axis=2).repeat(2,axis=3)

    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output_channels = folded_output_tensor_shape[1]
    if 'backend' in node and 'oc_order' in node['backend'] and DEBUG_SIMULATE_FOLDING:
        oc_processing_order = node['backend']['oc_order']
    else:
        oc_processing_order = [oc for oc in range(output_channels)]
    pre_folding_oc_processing_order = oc_processing_order
    # fold output to match actual output on nx (nx uses conv mechanism to do maxpooling)
    if DEBUG_SIMULATE_FOLDING:
        output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=output_folding_factor_x,input_folding_factor_y=output_folding_factor_y) # If its a folding conv it will add this folding as well
        
    if ('_split' in node['outputs'][0]):
        channels_selected = node['frontend']['output_channels'] * (2 ** output_folding_factor_x) * (2 ** output_folding_factor_y)
        split_output = output[:, 0:channels_selected, :, :]
        oc_processing_order = [oc for oc in range(channels_selected)]

    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        is_split_node = False
        if ('_split' in node['outputs'][0]):
            output_before_pad = split_output
        else:
            output_before_pad = output

        if (output_before_pad.shape[3] != math.ceil(output_before_pad.shape[3]/16)*16):
            padding_width = math.ceil(output_before_pad.shape[3]/16)*16 - output_before_pad.shape[3]
            if ('activation_silu' in node['attributes']) and (node['attributes']['activation_silu'] != None):
                output_pad_value = node['attributes']['activation_silu']['output_zp'][0]
            else:
                output_pad_value = node['frontend']['output_tensor_zp']
            output_padded = np.pad(output_before_pad, ((0, 0), (0, 0), (0, 0), (0, padding_width)), mode="constant", constant_values=output_pad_value)
        else:
            output_padded = output_before_pad
        export_tensor_to_nxo(debug_tensor_filename,output_padded,oc_processing_order, is_intermediate_node = is_intermediate_node, x_slices = x_slices, x_folding = output_folding_factor_x, y_folding = output_folding_factor_y, is_split_node=is_split_node, is_resize = True)

    output_x_slices = node['frontend']['output_tensor'].x_slices
    output_tensor = qTensor(output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
    output_name = node['outputs'][0].split('_split')[0]
    intermediate_tensors[output_name] = output_tensor
    if DEBUG_TIMING>0:
        if DEBUG_MAC_MULTIPLICATIONS:
            print('Warning: DEBUG_MAC_MULTIPLICATIONS is on so timing includes debug file creation (which is significant)')
        print('At layer: %s, total time: %s' % (node_name,str(datetime.datetime.now()-start_time)))
    
    if ('_split' in node['outputs'][0]):
        output_tensor = qTensor(split_output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
        intermediate_tensors[node['outputs'][0]] = output_tensor
        
    return [output_tensor]

def nx_concat(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None):
    start_time = datetime.datetime.now()
    if DEBUG_SIMULATE_FOLDING:
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape() # In case of folding conv we want to get the shape before output folding
    else:
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape() # In case of folding conv we want to get the shape before output folding

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']
    output = np.zeros(folded_output_tensor_shape,dtype=np.int64)
    mac_output = np.zeros(folded_output_tensor_shape,dtype=np.int64)
    
    #load quantized input with 0 padding
    concat_inputs=[]
    for input_idx,input_name in enumerate(node['inputs']):
        if input_name in model_inputs:
            input = model_inputs[input_name]
        elif input_name in intermediate_tensors:
            input = intermediate_tensors[input_name]
        else:
            raise ValueError ('Op %s input:%s not found!!' % (node_name,input_name))
        if DEBUG_UNFOLD_INPUTS_BEFORE_CONCAT:
            input_folding_factor_x = node['frontend']['input_tensors'][input_idx].folding_factor_x
            input_folding_factor_y = node['frontend']['input_tensors'][input_idx].folding_factor_y
            unfolded_input = copy.deepcopy(input)
            unfolded_input.data = folding_algo.get_asym_unfolded_tensor(unfolded_input.data,folding_factor_x=input_folding_factor_x,folding_factor_y=input_folding_factor_y)
            unfolded_input.scale = node['frontend']['input_tensors_scale'][input_idx]
            unfolded_input.zero_point = node['frontend']['input_tensors_zp'][input_idx]
            concat_inputs.append(unfolded_input)
        else:
            folded_input = copy.deepcopy(input)
            folded_input.scale = node['frontend']['input_tensors_scale'][input_idx]
            folded_input.zero_point = node['frontend']['input_tensors_zp'][input_idx]
            concat_inputs.append(folded_input)

    output_tensor = node['frontend']['output_tensor']
    output_tensor_scale = output_tensor.scale
    output_tensor_zp = output_tensor.zero_point
    concat_inputs_data = []
    # The "real" inputs is how inputs actually look in FPGA. 
    # We need this in order to be able to export intermediate nxo that will match FPGA.
    # In FPGA the inputs are first y folded and then concatenated while in numeric simulator they are first concatenated and the y folded.
    # The following Conv will expect concat op to be first concatenated and then folded. So in FPGA we over come this by re-arranging the oc_processing order of the inputs to concat
    # This allows us to re-arrange channels order so that this conv will "see" the "first concat then fold" tensor
    real_concat_inputs_data = [] 
    for input_tensor_index,q_tensor in enumerate(concat_inputs):

        current_input_scale = q_tensor.scale
        current_input_zp = q_tensor.zero_point
        real_q_tensor_data = q_tensor.data
        # In real FPGA the concatenated tensors are folded and hence they violate the rules of folding (e.g. instead of all even channels then all odd channels
        # we will have even channels of input 0 then odd channels of input 0 then even channels of input 1 then odd). We fix this by shuffeling the channels back
        # to their place by manipulating the oc processing order of the concat output
        # In the below we use output folding factor to take into account the node'ss input folding/unfolding operation done by DMA
        if not DEBUG_UNFOLD_INPUTS_BEFORE_CONCAT: # If we dont simulate with unfold before concat we just need the additional fold/unfold of the input done already on the input
            force_y_folding = 'force_folding_y' in node['frontend']
            force_y_unfolding = 'force_unfolding_y' in node['frontend']
            if force_y_folding:
                real_q_tensor_data = folding_algo.get_asym_folded_input(q_tensor.data,input_folding_factor_x=0,input_folding_factor_y=1)
            if force_y_unfolding:
                real_q_tensor_data = folding_algo.get_asym_unfolded_tensor(q_tensor.data,folding_factor_x=0,folding_factor_y=1)

        else: 
            real_q_tensor_data = folding_algo.get_asym_folded_input(q_tensor.data,input_folding_factor_x=output_folding_factor_x,input_folding_factor_y=output_folding_factor_y)


        if DEBUG_SIMULATE_CONCAT_REQUANT:
            if current_input_scale!=output_tensor_scale or current_input_zp!=output_tensor_zp:
                tensor_data_float = ((q_tensor.data-current_input_zp)*current_input_scale)
                real_tensor_data_float = ((real_q_tensor_data-current_input_zp)*current_input_scale)
                rescaled_tensor_data = np.trunc(((tensor_data_float/output_tensor_scale)+output_tensor_zp)+0.5).astype(np.int64)
                real_rescaled_tensor_data = np.trunc(((real_tensor_data_float/output_tensor_scale)+output_tensor_zp)+0.5).astype(np.int64)
                concat_inputs_data.append(rescaled_tensor_data)
                real_concat_inputs_data.append(real_rescaled_tensor_data)
            else:
                concat_inputs_data.append(q_tensor.data)
                real_concat_inputs_data.append(real_q_tensor_data)
        else:
            if current_input_scale!=output_tensor_scale or current_input_zp!=output_tensor_zp:
                raise ValueError ('input q params are different from output qparams. Input name: %s' % (node['inputs'][input_tensor_index]))
            
            concat_inputs_data.append(q_tensor.data)
            real_concat_inputs_data.append(real_q_tensor_data) # In case of concat which is also y_folding the actual tensor is first folded and then concatenated while the next conv expects it to be first concatenated and then folded
    output = np.concatenate(concat_inputs_data,axis=1)
    output = np.array(output,dtype=np.int64)
    real_output = np.concatenate(real_concat_inputs_data,axis=1)
    real_output = np.array(real_output,dtype=np.int64)
    output_channels = folded_output_tensor_shape[1]
    # In concat ops we create 'simulator_oc_order' which holds the real processing order of concat inputs. this will be used to generate its nxo
    # then we use 'oc_order' to tell following layers to look on converted order.
    # The conversion is needed since if the inputs to concat are y folded the cocatenated tensor should keep the folding rules in which each y fold
    # contains data of all the inputs and only then comes the next y fold. while actually in fpga we have all y folds of input 1 then y folds of input 2 etc.
    if 'backend' in node and 'simulator_oc_order' in node['backend'] and DEBUG_SIMULATE_FOLDING:
        oc_processing_order = node['backend']['simulator_oc_order']
    elif 'backend' in node and 'oc_order' in node['backend'] and DEBUG_SIMULATE_FOLDING:
        oc_processing_order = node['backend']['oc_order']
    else:
        oc_processing_order = [oc for oc in range(output_channels)]
    if DEBUG_UNFOLD_INPUTS_BEFORE_CONCAT: #We fold back after concat. This will include input folding/unfolding conv
        output_folding_factor_x = node['frontend']['output_tensor'].folding_factor_x
        output_folding_factor_y = node['frontend']['output_tensor'].folding_factor_y
        output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=output_folding_factor_x,input_folding_factor_y=output_folding_factor_y)
    else:
        if DEBUG_SIMULATE_FOLDING and output_folding_factor_y==input_folding_factor_y+1:
            output = folding_algo.get_asym_folded_input(output,input_folding_factor_x=0,input_folding_factor_y=1)
        elif DEBUG_SIMULATE_FOLDING and output_folding_factor_y!=input_folding_factor_y:
            raise ValueError ('This type of folding is not supported yet')
  

    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
        x_slices = node['frontend']['x_slices']
        export_tensor_to_nxo(debug_tensor_filename,output,oc_processing_order, is_intermediate_node = is_intermediate_node, x_slices = x_slices, x_folding = output_folding_factor_x, y_folding = output_folding_factor_y)

    output_x_slices = node['frontend']['output_tensor'].x_slices
    output_tensor = qTensor(output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
    output_tensor.real_tensor_data = real_output
    output_name = node['outputs'][0]
    intermediate_tensors[output_name] = output_tensor
    if DEBUG_TIMING>0:
        if DEBUG_MAC_MULTIPLICATIONS:
            print('Warning: DEBUG_MAC_MULTIPLICATIONS is on so timing includes debug file creation (which is significant)')
        print('At layer: %s, total time: %s' % (node_name,str(datetime.datetime.now()-start_time)))

    return [output_tensor]

def nx_flatten(node_name,model_inputs:dict,intermediate_tensors: dict,node,is_intermediate_node = False,debug_dir = None):
    start_time = datetime.datetime.now()
    if DEBUG_SIMULATE_FOLDING:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_folded_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape() # In case of folding conv we want to get the shape before output folding
    else:
        folded_input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
        folded_output_tensor_shape = node['frontend']['output_tensor'].get_original_shape() # In case of folding conv we want to get the shape before output folding

    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    output_folding_factor_x = node['frontend']['output_folding_factor_x']
    output_folding_factor_y = node['frontend']['output_folding_factor_y']
    output = np.zeros(folded_output_tensor_shape,dtype=np.int64)
    mac_output = np.zeros(folded_output_tensor_shape,dtype=np.int64)
    
    #load quantized input with 0 padding
    input_name = node['inputs'][0]
    if input_name in model_inputs:
        input = copy.deepcopy(model_inputs[input_name]) # We copy the input tensor sice we might fold it and dont want to change the inputs dict
        if (input_folding_factor_x>0 or input_folding_factor_y>0) and DEBUG_SIMULATE_FOLDING:
            input.data = folding_algo.get_asym_folded_input(input.data,input_folding_factor_x=input_folding_factor_x,input_folding_factor_y=input_folding_factor_y)
            input.folding_factor_x = input_folding_factor_x
            input.folding_factor_y = input_folding_factor_y
    elif input_name in intermediate_tensors:
        input = intermediate_tensors[input_name]
    else:
        raise ValueError ('Op %s input:%s not found!!' % (node_name,input_name))

    # In order to pool using pytorch maxpool op we need to unfold the input tensor if needed
    if DEBUG_SIMULATE_FOLDING:
        unfolded_input = folding_algo.get_asym_unfolded_tensor(input.data,folding_factor_x=input_folding_factor_x,folding_factor_y=input_folding_factor_y)
    else:
        unfolded_input = input.data

    npd=np.array(unfolded_input,dtype=np.int64)
    input_scale = node['frontend']['input_tensor'].scale
    input_zero_point = node['frontend']['input_tensor'].zero_point

    output = torch.flatten(torch.tensor(npd)).detach().numpy()

    output_scale = node['frontend']['output_tensor'].scale
    output_zero_point = node['frontend']['output_tensor'].zero_point
    output = np.expand_dims(output,0)
    output = (output).astype(np.int64)
    output_channels = folded_output_tensor_shape[1]
    oc_processing_order = [oc for oc in range(output_channels)]
    pre_folding_oc_processing_order = oc_processing_order
    if DEBUG_SIMULATOR_SAVE_INTERMEDIATE_TENSORS and debug_dir:
        debug_tensor_filename = os.path.join(debug_dir,(node_name+'_output_tensor.nxo'))
        expanded_output=np.expand_dims(output,(2,3))
        x_slices = node['frontend']['x_slices']
        export_tensor_to_nxo(debug_tensor_filename,output,oc_processing_order, is_intermediate_node = is_intermediate_node, x_slices = x_slices, x_folding = output_folding_factor_x, y_folding = output_folding_factor_y)

    output_x_slices = node['frontend']['output_tensor'].x_slices
    output_tensor = qTensor(output,scale = output_scale, zero_point = output_zero_point,folding_factor_x = output_folding_factor_x,folding_factor_y=output_folding_factor_y,x_slices=output_x_slices)
    output_name = node['outputs'][0]
    intermediate_tensors[output_name] = output_tensor
    if DEBUG_TIMING>0:
        if DEBUG_MAC_MULTIPLICATIONS:
            print('Warning: DEBUG_MAC_MULTIPLICATIONS is on so timing includes debug file creation (which is significant)')
        print('At layer: %s, total time: %s' % (node_name,str(datetime.datetime.now()-start_time)))

    return [output_tensor]

ops_directory = {
    'Conv': nx_conv,
    'Identity': nx_identity,
    'Add': nx_add,
    'MaxPool': nx_maxpool,
    'GlobalAveragePool': nx_globalavgpool,
    'Flatten': nx_flatten,
    'Gemm': nx_gemm,
    'Resize': nx_resize,
    'Concat': nx_concat,
}
