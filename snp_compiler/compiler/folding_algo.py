# Dans: This test is how to convert a kernel_size=1, stride=1 conv h=56,w=56 to kernel_size=1, stride=1 h=28,w=28 conv

import numpy as np
import torch
import math


def asym_fold_input_tensor_continousychannels(input_matrix,fold_factor_y,fold_factor_x):
    if len(input_matrix.shape)!=3:
        raise ValueError('weights rank expected to be 3')
    width = input_matrix.shape[2]
    height = input_matrix.shape[1]
    input_channels = input_matrix.shape[0]
    if (width % fold_factor_x) > 0:
        needed_padding_x = fold_factor_x - (width % fold_factor_x)
    else:
        needed_padding_x = 0
    if (height % fold_factor_y) > 0:
        needed_padding_y = fold_factor_y - (height % fold_factor_y)
    else:
        needed_padding_y = 0
    input_matrix = np.pad(input_matrix,((0,0),(0,needed_padding_y),(0,needed_padding_x)))
    output_width = input_matrix.shape[2] // fold_factor_x
    output_height = input_matrix.shape[1] // fold_factor_y
    # Equivalent to the original triple loop building output[y*fx*C + x*C + ic] = input[ic, y::fy, x::fx]
    output_matrix = (input_matrix
                     .reshape(input_channels, output_height, fold_factor_y, output_width, fold_factor_x)
                     .transpose(2, 4, 0, 1, 3)
                     .reshape(fold_factor_y * fold_factor_x * input_channels, output_height, output_width))
    return output_matrix


def asym_get_pixel_folded_pos_continousychannels(original_c,original_h,original_w,folding_factor_y,folding_factor_x,original_input_channels=0, use_old_format=False):
    if (not use_old_format): 
        folded_c=original_c+(original_h % folding_factor_y)*folding_factor_x*original_input_channels+(original_w % folding_factor_x)*original_input_channels
    else:
        folded_c=original_c*folding_factor_x+(original_h % folding_factor_y)*folding_factor_x*original_input_channels+original_w % folding_factor_x
    folded_h= original_h // folding_factor_y
    folded_w= original_w // folding_factor_x
    return folded_c,folded_h,folded_w

def asym_fold_weights_continousychannels(original_weights,resolution_fold_y,resolution_fold_x,stride_x=1,stride_y=1,asymmetric_padding=False,use_old_format=False):
    if len(original_weights.shape)!=4:
        raise ValueError('weights rank expected to be 4')
    original_output_channels = original_weights.shape[0]
    original_input_channels = original_weights.shape[1]
    original_height = original_weights.shape[2]
    original_width = original_weights.shape[3]
    if stride_x not in [1,2]:
        raise ValueError ('Only stride = 1 or 2 is supported')
    if stride_y not in [1,2]:
        raise ValueError ('Only stride = 1 or 2 is supported')
    #folded_output_channels = int(original_output_channels * resolution_fold_y * resolution_fold_x / (stride_x * stride_y))
    folded_output_channels = original_output_channels * resolution_fold_y * resolution_fold_x
    folded_input_channels = original_input_channels * resolution_fold_y * resolution_fold_x

    min_x_pos = -1*(original_width-1) // 2
    max_x_pos = original_width // 2
    if min_x_pos==0:
        folded_x_start = 0
    else:
        folded_x_start = int((min_x_pos+1) / resolution_fold_x) - 1
    folded_x_end = max_x_pos // resolution_fold_x # =QUOTIENT(X29,$V$27)
    if abs(folded_x_start)>folded_x_end:

        center_fixup_x = 1#=IF(ABS(Y29)>Z29,1,0)
    else:
        center_fixup_x = 0#=IF(ABS(Y29)>Z29,1,0)
    folded_width = folded_x_end - folded_x_start + center_fixup_x +1

    min_y_pos = -1*(original_height-1) // 2
    max_y_pos = original_height // 2
    if min_y_pos==0:
        folded_y_start = 0
    else:
        folded_y_start = int((min_y_pos+1) / resolution_fold_y) - 1
    folded_y_end = max_y_pos // resolution_fold_y # =QUOTIENT(X29,$V$27)
    if abs(folded_y_start)>folded_y_end:

        center_fixup_y = 1#=IF(ABS(Y29)>Z29,1,0)
    else:
        center_fixup_y = 0#=IF(ABS(Y29)>Z29,1,0)
    folded_height = folded_y_end - folded_y_start + center_fixup_y +1

    if not asymmetric_padding:
        folded_x_start_pos = folded_x_start
        folded_y_start_pos = folded_y_start
        original_x_start_pos = -1*((original_width -1)// 2) # This is because kernel is symetric around pixel so if width = 3 we start at position -1.
        original_y_start_pos = -1*((original_height -1)// 2) # This is because kernel is symetric around pixel so if width = 3 we start at position -1.
    else :
        original_x_start_pos = 0*((original_width -1)// 2) # This is because kernel is asymmetric around pixel so if width = 3 we start at position 0.
        original_y_start_pos = 0*((original_height -1)// 2) # This is because kernel is asymmetric around pixel so if width = 3 we start at position 0.
    folded_weights = np.zeros((folded_output_channels,folded_input_channels,folded_height,folded_width))
    if resolution_fold_x % stride_x !=0 or resolution_fold_y % stride_y !=0:
        raise ValueError ('Cant apply stride to unfolded axis.')
    #actual_resolution_fold_x = int(resolution_fold_x / stride_x)
    #actual_resolution_fold_y = int(resolution_fold_y / stride_y)
    for current_original_oc in range(original_output_channels):
        for current_original_ic in range(original_input_channels):
            for current_original_y in range(original_height):
                for current_original_x in range(original_width):
                    actual_x_pos = original_x_start_pos + current_original_x
                    actual_y_pos = original_y_start_pos + current_original_y
                    for oc_y_fold in range(resolution_fold_y):
                        for oc_x_fold in range(resolution_fold_x):
                            folded_input_channel,actual_folded_y,actual_folded_x = asym_get_pixel_folded_pos_continousychannels(current_original_ic,actual_y_pos+oc_y_fold,actual_x_pos+oc_x_fold,resolution_fold_y,resolution_fold_x,original_input_channels=original_input_channels, use_old_format=use_old_format)
                            if not asymmetric_padding:
                                folded_y = actual_folded_y - folded_y_start_pos
                                folded_x = actual_folded_x - folded_x_start_pos
                            else:
                                folded_y = actual_folded_y
                                folded_x = actual_folded_x
                            if (not use_old_format): 
                                folded_oc = int(current_original_oc + oc_y_fold*original_output_channels*resolution_fold_x + oc_x_fold*original_output_channels)
                            else:
                                folded_oc = int(current_original_oc*(resolution_fold_x / stride_x) + oc_y_fold*original_output_channels*(resolution_fold_x / stride_y) + oc_x_fold)
                            folded_weights[folded_oc,folded_input_channel,folded_y,folded_x] = \
                                    original_weights[current_original_oc,current_original_ic,current_original_y,current_original_x]
    
    strided_indexes = np.ones(folded_output_channels, dtype=bool)
    if stride_x == 2:
        for i in range(1, resolution_fold_x*resolution_fold_y,2):
            strided_indexes[i*original_output_channels:(i+1)*original_output_channels] = False
    if stride_y == 2:
        for i in range(1, resolution_fold_x*resolution_fold_y,2):
            strided_indexes[i*resolution_fold_x*original_output_channels:(i+1)*resolution_fold_x*original_output_channels] = False
    strided_folded_weights = folded_weights[strided_indexes,:,:,:]
    return strided_folded_weights


def asym_fold_biases_continousychannels(original_biases,resolution_fold_y,resolution_fold_x,stride_x=1,stride_y=1):
    folded_biases = original_biases
    folded_biases = np.tile(folded_biases,resolution_fold_x)
    folded_biases = np.tile(folded_biases,resolution_fold_y)
    return folded_biases


def asym_unfold_resfold_conv_output_continousychannels(folded_tensor,resolution_fold_y,resolution_fold_x):
    if len(folded_tensor.shape)!=4:
        raise ValueError('weights rank expected to be 4')
    folded_tensor = folded_tensor.squeeze(0)
    folded_input_channels = folded_tensor.shape[0]
    folded_height = folded_tensor.shape[1]
    folded_width = folded_tensor.shape[2]
    original_input_channels = folded_input_channels // (resolution_fold_y * resolution_fold_x)
    original_height = folded_height * resolution_fold_y
    original_width = folded_width * resolution_fold_x
    original_tensor = np.zeros((original_input_channels,original_height,original_width))
    for ic in range(original_input_channels):
        for y in range (resolution_fold_y):
            for x in range (resolution_fold_x):
                current_folded_ic = ic+x*original_input_channels+y*original_input_channels*resolution_fold_x
                folded_tensor_slice = folded_tensor[current_folded_ic,:,:]
                original_tensor[ic,y::resolution_fold_y,x::resolution_fold_x] = folded_tensor_slice
    return original_tensor


def get_asym_folded_weights(original_weights,input_folding_factor_x,input_folding_factor_y,stride_x=1,stride_y=1,asymmetric_padding = False,use_old_format=False):
    folded_weights = original_weights
    folded_weights = asym_fold_weights_continousychannels(folded_weights,resolution_fold_y=pow(2,input_folding_factor_y),resolution_fold_x=pow(2,input_folding_factor_x), \
                                            stride_x=stride_x,stride_y=stride_y, asymmetric_padding = asymmetric_padding, use_old_format=use_old_format)
    folded_weights=folded_weights.astype(np.int8)
    return folded_weights


def get_asym_folded_per_oc_params(original_per_oc_params,input_folding_factor_x=0,input_folding_factor_y=0,stride_x=1,stride_y=1):
    folded_per_oc_params = original_per_oc_params
    actual_resolution_fold_x = int(pow(2, input_folding_factor_x) / stride_x)
    actual_resolution_fold_y = int(pow(2, input_folding_factor_y) / stride_y)
    folded_per_oc_params = asym_fold_biases_continousychannels(folded_per_oc_params,actual_resolution_fold_y,actual_resolution_fold_x,stride_x=stride_x,stride_y=stride_y)
    return folded_per_oc_params


def get_asym_folded_input(original_input,input_folding_factor_x=0,input_folding_factor_y=0):
    folded_input = original_input[0,:]
    folded_input = asym_fold_input_tensor_continousychannels(folded_input,fold_factor_x=pow(2, input_folding_factor_x),fold_factor_y=pow(2,input_folding_factor_y))
    folded_input = np.expand_dims(folded_input,axis=0)
    return folded_input


def get_asym_unfolded_tensor(folded_tensor,folding_factor_x=0,folding_factor_y=0):
    unfolded_output = folded_tensor
    unfolded_output = np.expand_dims(asym_unfold_resfold_conv_output_continousychannels(unfolded_output, resolution_fold_x=pow(2,folding_factor_x),resolution_fold_y=pow(2,folding_factor_y)),0)
    return unfolded_output

