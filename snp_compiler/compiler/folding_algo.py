# Dans: This test is how to convert a kernel_size=1, stride=1 conv h=56,w=56 to kernel_size=1, stride=1 h=28,w=28 conv

import numpy as np
import torch
import math
from common.debug_flags import DEBUG_X_SLICING

'''
def fold_input_tensor(input_matrix,fold_factor):
    if len(input_matrix.shape)!=3:
        raise ValueError('weights rank expected to be 3')
    width = input_matrix.shape[2]
    input_channels = input_matrix.shape[0]
    if (width % fold_factor) > 0:
        needed_padding =fold_factor - (width % fold_factor)
        input_matrix = np.pad(input_matrix,((0,0),(0,needed_padding),(0,needed_padding)))
    output_width = input_matrix.shape[2] // fold_factor
    output_height = input_matrix.shape[1] // fold_factor
    output_matrix = np.zeros((0,output_height,output_width))
    for ic in range (input_channels):
        for y in range (fold_factor):
            for x in range (fold_factor):
                current_tensor = np.expand_dims(input_matrix[ic,y::fold_factor,x::fold_factor],0)
                output_matrix=np.concatenate((output_matrix,current_tensor),axis=0)
    return output_matrix

def asym_fold_input_tensor(input_matrix,fold_factor_y,fold_factor_x):
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
    output_matrix = np.zeros((0,output_height,output_width))
    for ic in range (input_channels):
        for y in range (fold_factor_y):
            for x in range (fold_factor_x):
                current_tensor = np.expand_dims(input_matrix[ic,y::fold_factor_y,x::fold_factor_x],0)
                output_matrix=np.concatenate((output_matrix,current_tensor),axis=0)
    return output_matrix
'''

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
    output_matrix = np.zeros((0,output_height,output_width))
    for y in range (fold_factor_y):
        for x in range (fold_factor_x):
            for ic in range (input_channels):        
                current_tensor = np.expand_dims(input_matrix[ic,y::fold_factor_y,x::fold_factor_x],0)
                output_matrix=np.concatenate((output_matrix,current_tensor),axis=0)
    return output_matrix

'''
def fold_weights_for_stride(weights_tensor,stride):
    if len(weights_tensor.shape)!=4:
        raise ValueError('weights rank expected to be 4')
    width = weights_tensor.shape[3]
    needed_padding = 0
    if (width % stride) > 0:
        needed_padding =stride - (width % stride)
    folded_width = (width+needed_padding) // stride
    output_channels = weights_tensor.shape[0]
    input_channels = weights_tensor.shape[1]
    folded_filters=[]
    for current_output_channel in range(output_channels):
        current_filter = weights_tensor[current_output_channel,:,:,:]
        folded_filters.append(fold_input_tensor(current_filter,stride))
    output_matrix = np.stack(folded_filters)
    return output_matrix

def fold_weights_for_resolution_folding(weights_tensor,resolution_fold):
    if len(weights_tensor.shape)!=4:
        raise ValueError('weights rank expected to be 4')
    output_channels = weights_tensor.shape[0]
    input_channels = weights_tensor.shape[1]
    height = weights_tensor.shape[2]
    width = weights_tensor.shape[3]
    folded_output_filters=[]
    zeros_input_filter_slice = np.zeros((output_channels,input_channels,height,width))
    fold_factor = resolution_fold*resolution_fold
    for output_filter_slice in range(fold_factor):
        folded_input_filters=[]
        for input_filter_slice in range(fold_factor):
            if input_filter_slice==output_filter_slice:
                folded_input_filters.append(weights_tensor)
            else:
                folded_input_filters.append(zeros_input_filter_slice)
        output_filter_slice = np.concatenate(folded_input_filters,axis=1)
        folded_output_filters.append(output_filter_slice)
    folded_weights = np.concatenate(folded_output_filters,axis=0)
    return folded_weights

def get_pixel_folded_pos(original_c,original_h,original_w,folding_factor):
    folded_c=original_c*folding_factor*folding_factor+(original_h % folding_factor)*folding_factor+original_w % folding_factor
    folded_h= original_h // folding_factor
    folded_w= original_w // folding_factor
    return folded_c,folded_h,folded_w

def asym_get_pixel_folded_pos(original_c,original_h,original_w,folding_factor_y,folding_factor_x):
    folded_c=original_c*folding_factor_y*folding_factor_x+(original_h % folding_factor_y)*folding_factor_x+original_w % folding_factor_x
    folded_h= original_h // folding_factor_y
    folded_w= original_w // folding_factor_x
    return folded_c,folded_h,folded_w

def fold_weights(original_weights,resolution_fold,stride):
    if len(original_weights.shape)!=4:
        raise ValueError('weights rank expected to be 4')
    original_output_channels = original_weights.shape[0]
    original_input_channels = original_weights.shape[1]
    original_height = original_weights.shape[2]
    original_width = original_weights.shape[3]
    if stride not in [1,2]:
        raise ValueError ('Only stride = 1 or 2 is supported')
    folded_output_channels = int(original_output_channels * resolution_fold * resolution_fold / (stride * stride))
    folded_input_channels = original_input_channels * resolution_fold * resolution_fold
    folded_width = math.ceil(original_width / resolution_fold)
    folded_height = math.ceil(original_height / resolution_fold)
    # Because kernel is always symetric around center pixel and In case of even kernel size it has +1 pixels on the right side
    # We need to all +1 to folded kernel size to be able to reach all original kernel positions
    folded_x_start_pos = -1*(folded_width // resolution_fold)
    folded_y_start_pos = -1*(folded_width // resolution_fold)
    if original_width % 2 == 0:
        folded_width+=1
    else:
        if folded_width % 2 == 0:
            folded_width+=1

    if original_height % 2 == 0:
        folded_height+=1
    else:
        if folded_height % 2 == 0:
            folded_height+=1
    original_x_start_pos = -1*((original_width-1) // 2) # This is because kernel is symetric around pixel so if width = 3 we start at position -1.
    original_y_start_pos = -1*((original_height-1) // 2) # This is because kernel is symetric around pixel so if width = 3 we start at position -1.
    folded_weights = np.zeros((folded_output_channels,folded_input_channels,folded_height,folded_width),dtype=np.int8)
    actual_resolution_fold = int(resolution_fold / stride)
    for current_original_oc in range(original_output_channels):
        for current_original_ic in range(original_input_channels):
            for current_original_y in range(original_height):
                for current_original_x in range(original_width):
                    actual_x_pos = original_x_start_pos + current_original_x
                    actual_y_pos = original_y_start_pos + current_original_y
                    for oc_y_fold in range(actual_resolution_fold):
                        for oc_x_fold in range(actual_resolution_fold):
                            folded_input_channel,actual_folded_y,actual_folded_x = get_pixel_folded_pos(current_original_ic,actual_y_pos+oc_y_fold,actual_x_pos+oc_x_fold,resolution_fold)
                            folded_y = actual_folded_y - folded_y_start_pos
                            folded_x = actual_folded_x - folded_x_start_pos
                            folded_oc = int(current_original_oc*(resolution_fold*resolution_fold / (stride * stride)) + oc_y_fold*resolution_fold + oc_x_fold)
                            folded_weights[folded_oc,folded_input_channel,folded_y,folded_x] = \
                                    original_weights[current_original_oc,current_original_ic,current_original_y,current_original_x]
    return folded_weights

def asym_fold_weights(original_weights,resolution_fold_y=0,resolution_fold_x=0,stride_x=1,stride_y=1):
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
    folded_output_channels = int(original_output_channels * resolution_fold_y * resolution_fold_x / (stride_x * stride_y))
    folded_input_channels = original_input_channels * resolution_fold_y * resolution_fold_x

    min_x_pos = -1*(original_width-1) // 2
    max_x_pos = original_width // 2
    if min_x_pos==0: #=IF(W33(min_x_pos)=0,0,QUOTIENT(W33+1,$V$27)-1). Note that in python a // b if a<0 doesnt give integer part of division. this is why we use int(float(a/b))
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

    folded_x_start_pos = folded_x_start
    folded_y_start_pos = folded_y_start

    original_x_start_pos = -1*((original_width -1)// 2) # This is because kernel is symetric around pixel so if width = 3 we start at position -1.
    original_y_start_pos = -1*((original_height -1)// 2) # This is because kernel is symetric around pixel so if width = 3 we start at position -1.
    folded_weights = np.zeros((folded_output_channels,folded_input_channels,folded_height,folded_width))
    if resolution_fold_x % stride_x !=0 or resolution_fold_y % stride_y !=0:
        raise ValueError ('Cant apply stride to unfolded axis.')
    actual_resolution_fold_x = int(resolution_fold_x / stride_x)
    actual_resolution_fold_y = int(resolution_fold_y / stride_y)
    for current_original_oc in range(original_output_channels):
        for current_original_ic in range(original_input_channels):
            for current_original_y in range(original_height):
                for current_original_x in range(original_width):
                    actual_x_pos = original_x_start_pos + current_original_x
                    actual_y_pos = original_y_start_pos + current_original_y
                    for oc_y_fold in range(actual_resolution_fold_y):
                        for oc_x_fold in range(actual_resolution_fold_x):
                            folded_input_channel,actual_folded_y,actual_folded_x = asym_get_pixel_folded_pos(current_original_ic,actual_y_pos+oc_y_fold,actual_x_pos+oc_x_fold,resolution_fold_y,resolution_fold_x)
                            folded_y = actual_folded_y - folded_y_start_pos
                            folded_x = actual_folded_x - folded_x_start_pos
                            folded_oc = int(current_original_oc*(resolution_fold_y*resolution_fold_x / (stride_x * stride_y)) + oc_y_fold*resolution_fold_x + oc_x_fold)
                            folded_weights[folded_oc,folded_input_channel,folded_y,folded_x] = \
                                    original_weights[current_original_oc,current_original_ic,current_original_y,current_original_x]
    return folded_weights
'''

def asym_get_pixel_folded_pos_continousychannels(original_c,original_h,original_w,folding_factor_y,folding_factor_x,original_input_channels=0, use_old_format=False):
    if DEBUG_X_SLICING and (not use_old_format): 
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
                            if DEBUG_X_SLICING and (not use_old_format): 
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

'''
def fold_biases(original_biases,resolution_fold,stride):
    folded_biases = original_biases
    if stride==1:
        pixels_per_fold = resolution_fold*resolution_fold
        #folded_biases = np.zeros(original_biases.shape[0]*pixels_per_fold)
        #folded_biases[::pixels_per_fold]= original_biases
        folded_biases = np.repeat(folded_biases,4,axis=0)
    return folded_biases

def asym_fold_biases(original_biases,resolution_fold_y=0,resolution_fold_x=0,stride_x=1,stride_y=1):
    folded_biases = original_biases
    pixels_per_fold = (resolution_fold_y // stride_y)*(resolution_fold_x // stride_x)
    if pixels_per_fold>1:
        folded_biases = np.repeat(folded_biases,pixels_per_fold,axis=0)
    return folded_biases
'''

def asym_fold_biases_continousychannels(original_biases,resolution_fold_y,resolution_fold_x,stride_x=1,stride_y=1):
    folded_biases = original_biases
    if DEBUG_X_SLICING:
        folded_biases = np.tile(folded_biases,resolution_fold_x)
        folded_biases = np.tile(folded_biases,resolution_fold_y)
    else:
        if stride_x==1 or stride_y==1:
            if stride_x==1:
                folded_biases = np.tile(folded_biases,resolution_fold_x)
            if stride_y==1:
                folded_biases = np.repeat(np.expand_dims(folded_biases,0),resolution_fold_y,axis=0).flatten()
    return folded_biases

'''
def unfold_resfold_conv_output(folded_tensor,resolution_fold):
    if len(folded_tensor.shape)!=4:
        raise ValueError('weights rank expected to be 4')
    folded_tensor = folded_tensor.squeeze(0)
    folded_input_channels = folded_tensor.shape[0]
    folded_height = folded_tensor.shape[1]
    folded_width = folded_tensor.shape[2]
    original_input_channels = folded_input_channels // (resolution_fold * resolution_fold)
    original_width = folded_height * resolution_fold
    original_height = folded_width * resolution_fold
    original_tensor = np.zeros((original_input_channels,original_height,original_width))
    for ic in range(original_input_channels):
        for y in range (resolution_fold):
                for x in range (resolution_fold):
                    current_folded_ic = ic*resolution_fold*resolution_fold+x+y*resolution_fold
                    folded_tensor_slice = folded_tensor[ic*resolution_fold*resolution_fold+x+y*resolution_fold,:,:]
                    original_tensor[ic,y::resolution_fold,x::resolution_fold] = folded_tensor_slice
    return original_tensor

def asym_unfold_resfold_conv_output(folded_tensor,resolution_fold_y,resolution_fold_x):
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
                    current_folded_ic = ic*resolution_fold_y*resolution_fold_x+x+y*resolution_fold_x
                    folded_tensor_slice = folded_tensor[ic*resolution_fold_y*resolution_fold_x+x+y*resolution_fold_x,:,:]
                    original_tensor[ic,y::resolution_fold_y,x::resolution_fold_x] = folded_tensor_slice
    return original_tensor
'''

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

'''
def get_folded_weights(original_weights,input_folding_factor,stride):
    resolution_fold = 2
    folded_weights = original_weights
    for folding_iteration in range(int(input_folding_factor)):
        if folding_iteration == 0: # We implement the stride only on the 1st folding
            actual_stride = stride
        else:
            actual_stride = 1
        folded_weights = fold_weights(folded_weights,resolution_fold,actual_stride)
    return folded_weights
'''

def get_asym_folded_weights(original_weights,input_folding_factor_x,input_folding_factor_y,stride_x=1,stride_y=1,asymmetric_padding = False,use_old_format=False):
    folded_weights = original_weights
    if DEBUG_X_SLICING:
        folded_weights = asym_fold_weights_continousychannels(folded_weights,resolution_fold_y=pow(2,input_folding_factor_y),resolution_fold_x=pow(2,input_folding_factor_x), \
                                                stride_x=stride_x,stride_y=stride_y, asymmetric_padding = asymmetric_padding, use_old_format=use_old_format)
    else:
        max_folding_factor=max(input_folding_factor_x,input_folding_factor_y)
        for folding_iteration in range(int(max_folding_factor)):
            if folding_iteration == 0: # We implement the stride only on the 1st folding
                actual_stride_x = stride_x
                actual_stride_y = stride_y
            else:
                actual_stride_x = 1
                actual_stride_y = 1
            resolution_fold_x=1
            resolution_fold_y=1
            if folding_iteration<input_folding_factor_x:
                resolution_fold_x=2
            if folding_iteration<input_folding_factor_y:
                resolution_fold_y=2
            folded_weights = asym_fold_weights_continousychannels(folded_weights,resolution_fold_y=resolution_fold_y,resolution_fold_x=resolution_fold_x, \
                                                   stride_x=actual_stride_x,stride_y=actual_stride_y, asymmetric_padding = asymmetric_padding, use_old_format=use_old_format)
    folded_weights=folded_weights.astype(np.int8)
    return folded_weights

'''
def get_folded_per_oc_params(original_per_oc_params,input_folding_factor,stride):
    resolution_fold = 2
    folded_per_oc_params = original_per_oc_params
    for folding_iteration in range(int(input_folding_factor)):
        if folding_iteration == 0: # We implement the stride only on the 1st folding
            actual_stride = stride
        else:
            actual_stride = 1
        folded_per_oc_params = fold_biases(folded_per_oc_params,resolution_fold,actual_stride)
    return folded_per_oc_params
'''

def get_asym_folded_per_oc_params(original_per_oc_params,input_folding_factor_x=0,input_folding_factor_y=0,stride_x=1,stride_y=1):
    folded_per_oc_params = original_per_oc_params
    if DEBUG_X_SLICING:
        actual_resolution_fold_x = int(pow(2, input_folding_factor_x) / stride_x)
        actual_resolution_fold_y = int(pow(2, input_folding_factor_y) / stride_y)
        folded_per_oc_params = asym_fold_biases_continousychannels(folded_per_oc_params,actual_resolution_fold_y,actual_resolution_fold_x,stride_x=stride_x,stride_y=stride_y)
    else:
        max_folding_factor=max(input_folding_factor_x,input_folding_factor_y)
        for folding_iteration in range(int(max_folding_factor)):
            if folding_iteration == 0: # We implement the stride only on the 1st folding
                actual_stride_x = stride_x
                actual_stride_y = stride_y
            else:
                actual_stride_x = 1
                actual_stride_y = 1
            resolution_fold_x=1
            resolution_fold_y=1
            if folding_iteration<input_folding_factor_x:
                resolution_fold_x=2
            if folding_iteration<input_folding_factor_y:
                resolution_fold_y=2
            folded_per_oc_params = asym_fold_biases_continousychannels(folded_per_oc_params,resolution_fold_y,resolution_fold_x,stride_x=actual_stride_x,stride_y=actual_stride_y)
    return folded_per_oc_params

'''
def get_folded_input(original_input,input_folding_factor):
    resolution_fold = 2
    folded_input = original_input[0,:]
    for folding_iteration in range(int(input_folding_factor)):
        folded_input = fold_input_tensor(folded_input,resolution_fold)
    folded_input = np.expand_dims(folded_input,axis=0)
    return folded_input
'''

def get_asym_folded_input(original_input,input_folding_factor_x=0,input_folding_factor_y=0):
    folded_input = original_input[0,:]
    if DEBUG_X_SLICING:
        folded_input = asym_fold_input_tensor_continousychannels(folded_input,fold_factor_x=pow(2, input_folding_factor_x),fold_factor_y=pow(2,input_folding_factor_y))
    else:
        max_folding_factor=max(input_folding_factor_x,input_folding_factor_y)
        for folding_iteration in range(int(max_folding_factor)):
            resolution_fold_x=1
            resolution_fold_y=1
            if folding_iteration<input_folding_factor_x:
                resolution_fold_x=2
            if folding_iteration<input_folding_factor_y:
                resolution_fold_y=2
            folded_input = asym_fold_input_tensor_continousychannels(folded_input,fold_factor_x=resolution_fold_x,fold_factor_y=resolution_fold_y)
    folded_input = np.expand_dims(folded_input,axis=0)
    return folded_input

'''
def get_unfolded_tensor(folded_tensor,folding_factor):
    resolution_fold = 2
    unfolded_output = folded_tensor
    for i in range(int(folding_factor)):
        unfolded_output = np.expand_dims(unfold_resfold_conv_output(unfolded_output, resolution_fold),0)
    return unfolded_output
'''

def get_asym_unfolded_tensor(folded_tensor,folding_factor_x=0,folding_factor_y=0):
    unfolded_output = folded_tensor
    if DEBUG_X_SLICING:
        unfolded_output = np.expand_dims(asym_unfold_resfold_conv_output_continousychannels(unfolded_output, resolution_fold_x=pow(2,folding_factor_x),resolution_fold_y=pow(2,folding_factor_y)),0)
    else:
        max_folding_factor=max(folding_factor_x,folding_factor_y)
        for folding_iteration in range(int(max_folding_factor)):
            resolution_fold_x=1
            resolution_fold_y=1
            if folding_iteration<folding_factor_x:
                resolution_fold_x=2
            if folding_iteration<folding_factor_y:
                resolution_fold_y=2
            unfolded_output = np.expand_dims(asym_unfold_resfold_conv_output_continousychannels(unfolded_output, resolution_fold_x=resolution_fold_x,resolution_fold_y=resolution_fold_y),0)
    return unfolded_output

'''
def main():
    # Original Input dims are :3x224x224 Output dims are: 64x112x112
    # Converted Input dims are: (3x64=192)x28x28 Output dims are (64x16=1024)x28x28
    kernel_size=7
    padding = kernel_size // 2
    original_stride=2
    original_input_channels = 3
    original_output_channels = 64
    original_input_height = 224
    original_input_width = 224
    total_resolution_fold = 8

    input_tensor = np.random.uniform(low=-10, high=10, size=(original_input_channels,original_input_height,original_input_width))
    #input_tensor_size = original_input_channels*original_input_height*original_input_width
    #input_tensor = np.arange(0,input_tensor_size).reshape((original_input_channels,original_input_height,original_input_width))
    original_weights = np.random.uniform(low=-10, high=10, size=(original_output_channels,original_input_channels,kernel_size,kernel_size))
    original_biases = np.random.uniform(low=-10, high=10, size=(original_output_channels))
    #weights_tensor_size = original_output_channels*original_input_channels*kernel_size*kernel_size
    #original_weights = np.arange(0,weights_tensor_size).reshape((original_output_channels,original_input_channels,kernel_size,kernel_size))

    previous_fold_input_channels = original_input_channels
    previous_fold_output_channels = original_output_channels
    previous_fold_input_tensor = input_tensor
    previous_fold_weights = original_weights
    previous_fold_biases = original_biases
    previous_fold_padding = padding
    previous_fold_kernel_size = kernel_size
    resolution_fold = 2
    num_folding_iterations = math.log(total_resolution_fold,2)
    for folding_iteration in range(int(num_folding_iterations)):
        folded_input_channels = previous_fold_input_channels * resolution_fold * resolution_fold
        if folding_iteration == 0:
            actual_stride = original_stride
        else:
            actual_stride = 1
        folded_output_channels = int(previous_fold_output_channels * resolution_fold * resolution_fold / (actual_stride * actual_stride))
        print('Fold#%d: Input tensor shape: %s, Weights shape: %s' % (folding_iteration,str(previous_fold_input_tensor.shape),str(previous_fold_weights.shape)))
        print('Fold#%d: Input tensor padding: %d, stride=%d' % (folding_iteration,previous_fold_padding,actual_stride))

        original_conv_op = torch.nn.Conv2d(previous_fold_input_channels, previous_fold_output_channels, kernel_size=previous_fold_kernel_size, stride=actual_stride,bias=True)
        md = dict()
        md["weight"] = torch.tensor(previous_fold_weights.astype(np.float32))
        md["bias"] = torch.tensor(previous_fold_biases.astype(np.float32))
        original_conv_op.load_state_dict(md)
        padded_input_tensor = np.pad(previous_fold_input_tensor,[(0,0),(previous_fold_padding,previous_fold_padding),(previous_fold_padding,previous_fold_padding)], mode='constant', constant_values=0)
        conv_input = torch.tensor(np.expand_dims(padded_input_tensor,axis=0).astype(np.float32))
        original_conv_output=original_conv_op(conv_input).detach().numpy()
        if folding_iteration==0:
            keep_original_output = original_conv_output.copy()
        print('Fold#%d: Output tensor shape: %s' % (folding_iteration,str(original_conv_output.shape)))

        folded_input_tensor = fold_input_tensor(previous_fold_input_tensor,resolution_fold) # Each resolution_fold x resolution_fold patch is folded to z axis

        folded_weights = fold_weights(previous_fold_weights,resolution_fold,actual_stride)
        folded_biases = fold_biases(previous_fold_biases,resolution_fold,actual_stride)
        folded_kernel_size = folded_weights.shape[3]
        folded_padding = folded_kernel_size // 2
        padded_folded_input_tensor = np.pad(folded_input_tensor,[(0,0),(folded_padding,folded_padding),(folded_padding,folded_padding)], mode='constant', constant_values=0)

        print('Fold#%d: Folded input tensor shape: %s, Folded conv Weights shape: %s' % (folding_iteration,str(folded_input_tensor.shape),str(folded_weights.shape)))
        print('Fold#%d: Folded Input tensor padding: %d' % (folding_iteration,folded_padding))

        folded_weights_kernel_size = folded_weights.shape[3]

        converted_conv_op = torch.nn.Conv2d(folded_input_channels, folded_output_channels, kernel_size=folded_weights_kernel_size, stride=1,bias=True)
        md = dict()
        md["weight"] = torch.tensor(folded_weights.astype(np.float32))
        md["bias"] = torch.tensor(folded_biases.astype(np.float32))
        converted_conv_op.load_state_dict(md)
        conv_input = torch.tensor(np.expand_dims(padded_folded_input_tensor,axis=0).astype(np.float32))
        folded_conv_output=converted_conv_op(conv_input).detach().numpy()

        print('Fold#%d: Folded conv output tensor shape: %s' % (folding_iteration,str(folded_conv_output.shape)))
        previous_fold_input_channels = folded_input_channels
        previous_fold_output_channels = folded_output_channels
        previous_fold_input_tensor = folded_input_tensor
        previous_fold_weights = folded_weights
        previous_fold_biases = folded_biases
        previous_fold_padding = folded_padding
        previous_fold_kernel_size = folded_weights_kernel_size
        if actual_stride == 1:
            unfolded_conv_output = unfold_resfold_conv_output(folded_conv_output, resolution_fold)
        else: # In case of stride=2 conv the unfolding is done by the folded conv itself
            unfolded_conv_output = folded_conv_output
        print('Fold#%d: Results equal: %s' % (folding_iteration,str(np.allclose(original_conv_output.squeeze(0),unfolded_conv_output,atol=0.01))))

    print('Comparing folding end to end, number of total folds: %d' % (num_folding_iterations))
    unfolded_output = folded_conv_output
    for i in range(int(num_folding_iterations)):
        if i==0 and original_stride==2:
            unfolded_output = unfolded_output
        else:
            unfolded_output = np.expand_dims(unfold_resfold_conv_output(unfolded_output, resolution_fold),0)
    print('Total folding compare results equal: %s' % (str(np.allclose(keep_original_output.squeeze(0),unfolded_output,atol=0.01))))
    print('End!')

if __name__ == "__main__":

    main()

'''