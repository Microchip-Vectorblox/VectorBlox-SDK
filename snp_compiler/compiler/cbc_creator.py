import numpy as np
from common.enums import  DebugFilesFormat
from common.program_ir import WLOCEntry, WLOCList
from common.debug_flags import DEBUG_CHECK_PAIRING_AFTER_ALGO, DEBUG_PAIRING_USED
from collections import defaultdict
import copy


def find_pairs_and_nonpairs(vec, check_for_negativ_pair=False, elem_even_info=None):
    """
    Finds matching pairs and non-paired elements in a list.

    A pair is only valid if the first element has an even index and the second has an odd index,
    according to the provided elem_odd_info list.

    Author: Alexander Logvinenko

    Parameters:
    - vec (list): List of elements.
    - check_for_negativ_pair (bool): If True, elements A and B are considered a pair if A == -B.
    - elem_odd_info (list of bool): List indicating if the element at a given index is odd-indexed (True) or even-indexed (False).

    Returns:
    - pairs (list of [int, int]): List of index pairs.
    - non_pairs (list of int): List of indices with no matching pair.
    """
    elem_odd_info = np.logical_not(elem_even_info.astype(bool))
    if elem_odd_info is None or len(elem_odd_info) != len(vec):
        raise ValueError("elem_odd_info must be provided and match the length of vec")

    pairs = []
    all_inx = np.arange(len(vec))
    evn_arr = vec[              (elem_even_info).astype(bool)]
    env_inx = all_inx[          (elem_even_info).astype(bool)]

    odd_arr = vec    [np.logical_not(elem_even_info)]
    odd_inx = all_inx[np.logical_not(elem_even_info)]
    
    odd_arr = -odd_arr if check_for_negativ_pair else odd_arr
    # Build a dictionary where each value maps to a list of indices
    value_to_indices = defaultdict(list)
    for idx, val in enumerate(odd_arr):
        value_to_indices[val].append(odd_inx[idx])

    for idx, val in enumerate(evn_arr):
        # If value exists in arr2 and there is at least one unused index
        if value_to_indices[val]:
            inx_cand = env_inx[idx]
            idx_pair = value_to_indices[val].pop(0)  # Take and remove the first available index
            pairs.append([inx_cand, idx_pair])


    all_pairs_in_one  = [x for pair in pairs for x in pair] # as a flat list
    vec_inx           = np.arange(len(vec))
    non_pairs         = [x for x in vec_inx if x not in all_pairs_in_one] # del all pairs from the given list vec

    return pairs, non_pairs


def convert_kernel_to_wlocs(kernel_1x1, kernel_1x1_no_zeros_inx,wloc_entrys, oc, node):
        list_half_len = -(-len(kernel_1x1_no_zeros_inx)//2)
            #normal case       
        for ihalf in range(2):
            kernel_1x1_no_zeros_inx_half = kernel_1x1_no_zeros_inx[ihalf * list_half_len : (ihalf + 1) * list_half_len]
            previous_weight_index = 1e57 # big number
            for weight_index_tuple in kernel_1x1_no_zeros_inx_half:
                weight_index = weight_index_tuple[0]
                # Apply aligned_input_ch transformation if it exists
                if 'aligned_input_ch' in node.get('frontend', {}):
                    weight_index = node['frontend']['aligned_input_ch'][weight_index_tuple[0]]
                current_weight_value = kernel_1x1[weight_index_tuple[0]]
                distance_from_previous_activation = max(weight_index - previous_weight_index,0)
                if len(weight_index_tuple)==1:
                    #no pairing    
                    wloc_entry = (WLOCEntry(weight_value = current_weight_value, weight_index = weight_index, weight_offset = distance_from_previous_activation,
                                            long_entry=True, ic=weight_index, oc = oc))
                else:
                    #pairing
                    pair_add = weight_index_tuple[1]
                    # Apply aligned_input_ch transformation for pair_add if it exists
                    if 'aligned_input_ch' in node.get('frontend', {}):
                        pair_add = node['frontend']['aligned_input_ch'][weight_index_tuple[1]]
                    sub_sel_preadder_minus_mode = ((int)(kernel_1x1[weight_index_tuple[0]])==-(int)(kernel_1x1[weight_index_tuple[1]]))
                    wloc_entry = (WLOCEntry(weight_value = current_weight_value, weight_index = weight_index, weight_offset = distance_from_previous_activation, is_pair = True,  pair_add = pair_add, sub_sel_preadder_minus_mode = sub_sel_preadder_minus_mode,
                                        long_entry=True, ic=weight_index, oc = oc))

                wloc_entrys[ihalf].add_entry(wloc_entry)
                previous_weight_index = weight_index

        #if len(wloc0) != len(wloc1) write one additional wloc_zero_mul_entry
        if len(kernel_1x1_no_zeros_inx) % 2 != 0:
            wloc_entrys[1].add_entry(WLOCEntry.get_zero_mul_entry())

def convert_maxpool_to_wloc(kernel_1x1, kernel_1x1_no_zeros_inx,wloc_entrys, oc, node, first_calc=False):
    previous_weight_index = 1e57
    for i, weight_index_tuple in enumerate (kernel_1x1_no_zeros_inx):
        weight_index = weight_index_tuple[0]
        # Apply aligned_input_ch transformation if it exists
        if 'aligned_input_ch' in node.get('frontend', {}):
            weight_index = node['frontend']['aligned_input_ch'][weight_index_tuple[0]]
        distance_from_previous_activation = max(weight_index - previous_weight_index,0)

        wloc_entry_p1 = (WLOCEntry(weight_value =  1, weight_index = weight_index, weight_offset = distance_from_previous_activation,
                                long_entry=True, ic=weight_index, oc = oc))
        wloc_entry_m1 = (WLOCEntry(weight_value = -1, weight_index = weight_index, weight_offset = 0,
                                long_entry=True, ic=weight_index, oc = oc))

        if (i==0 and first_calc):
            #the first entery is only one line in WLOC with wight 1
            wloc_entrys[0].add_entry(wloc_entry_p1)
            wloc_entrys[1].add_entry(WLOCEntry.get_zero_mul_entry())


        else:
            # the all other entry have to be two lines with -1 and +1
            wloc_entrys[0].add_entry(wloc_entry_m1)
            wloc_entrys[0].add_entry(WLOCEntry.get_zero_mul_entry())
            wloc_entrys[0].add_entry(wloc_entry_p1)
            wloc_entrys[0].add_entry(WLOCEntry.get_zero_mul_entry())

            wloc_entrys[1].add_entry(WLOCEntry.get_zero_mul_entry())
            wloc_entrys[1].add_entry(WLOCEntry.get_zero_mul_entry())
            wloc_entrys[1].add_entry(WLOCEntry.get_zero_mul_entry())
            wloc_entrys[1].add_entry(WLOCEntry.get_zero_mul_entry())

        #### if the WLOC have only one channel, we need to insert NOP (HW requirements)
        if len(wloc_entrys)==2 and not first_calc:
            wloc_entrys[0].insert_entry(0,WLOCEntry.get_zero_mul_entry())
            wloc_entrys[1].insert_entry(0,WLOCEntry.get_zero_mul_entry())
            wloc_entrys[0].insert_entry(0,WLOCEntry.get_zero_mul_entry())
            wloc_entrys[1].insert_entry(0,WLOCEntry.get_zero_mul_entry())





def calculate_pairing(kernel_1x1):
    kernel_1x1_no_zeros_inx  = [i for i, num in enumerate(kernel_1x1) if num != 0]
    kernel_1x1_no_zeros_inx  = np.array(kernel_1x1_no_zeros_inx) 

    pos_pairs_sub, pos_non_pairs_sub = find_pairs_and_nonpairs(kernel_1x1[kernel_1x1_no_zeros_inx], elem_even_info = (kernel_1x1_no_zeros_inx%2), check_for_negativ_pair = False)
    pos_pairs     = kernel_1x1_no_zeros_inx[pos_pairs_sub]
    pos_non_pairs = kernel_1x1_no_zeros_inx[pos_non_pairs_sub]

    neg_pairs_sub, non_pairs_inx_sub = find_pairs_and_nonpairs(kernel_1x1[pos_non_pairs],           elem_even_info = (pos_non_pairs%2),check_for_negativ_pair = True)
    neg_pairs       = pos_non_pairs[neg_pairs_sub]
    non_pairs_inx   = pos_non_pairs[non_pairs_inx_sub]


    
    #check if there are some pairs that was not found by algo
    #this check will do the compiler slower, but this is needed to be sure, the pairing is correct
    if DEBUG_CHECK_PAIRING_AFTER_ALGO:
        vec = kernel_1x1
        odd_inx = np.where(              (np.arange(len(vec))%2))[0]
        evn_inx = np.where(np.logical_not(np.arange(len(vec))%2))[0]
        e_no_pair_inx = np.intersect1d(non_pairs_inx, evn_inx)
        o_no_pair_inx = np.intersect1d(non_pairs_inx, odd_inx) 
        if np.intersect1d(abs(vec[o_no_pair_inx]), abs(vec[e_no_pair_inx])):
            raise ValueError("There are more pairs to find here, check the pairing")
    #################################################################################################
    return pos_pairs, neg_pairs, non_pairs_inx

def generate_1x1_conv_alex(kernel_1x1, oc, node, first_calc=False):

    

    kernel_1x1_no_zeros_inx  = [i for i, num in enumerate(kernel_1x1) if num != 0]
    kernel_1x1_no_zeros_inx  = np.array(kernel_1x1_no_zeros_inx) 
    
    # the kernel looks like kernel[list_oc, list_ic, y, x ]
    # the kernel_1x1 has only one oc, size(x) = seze (y) = 1 
    # kernel_1x1[input_channel_reorder]

    wloc_entrys = [WLOCList() ,WLOCList() ]
    
    if len(kernel_1x1_no_zeros_inx) == 0:
        #the kernel is "zero", return two wlocs with 0
        wloc_entrys[0].add_entry(WLOCEntry.get_zero_mul_entry())
        wloc_entrys[1].add_entry(WLOCEntry.get_zero_mul_entry())
    else:
        # check pairing
        non_pairs_tupl = [(item,) for item in kernel_1x1_no_zeros_inx]
        
        if (node['op_type'] != "MaxPool"):
            # this is Conv od Add
            if (DEBUG_PAIRING_USED == True):
                pos_pairs, neg_pairs, non_pairs_inx = calculate_pairing(kernel_1x1)

                
                convert_kernel_to_wlocs(kernel_1x1, pos_pairs, wloc_entrys, oc, node)
                convert_kernel_to_wlocs(kernel_1x1, neg_pairs, wloc_entrys, oc, node)
                non_pairs_tupl = [(item,) for item in non_pairs_inx] 

                #update the pairing statistics
                node['backend']['statistics']['wloc_pairs'] +=(len(pos_pairs)+len(neg_pairs)) 

            convert_kernel_to_wlocs(kernel_1x1, non_pairs_tupl,wloc_entrys,oc, node)

        elif (node['op_type'] == "MaxPool"):
            #this is MAX_Pool
            convert_maxpool_to_wloc(kernel_1x1, non_pairs_tupl,wloc_entrys,oc, node, first_calc=first_calc)            

    # if the wloc is too short, shorter than 2, add one more entry at beginning. This needs for delay in HW (SHD, SHR, EOC)   
    if len(wloc_entrys[1].cmd_list)==1:
        wloc_entrys[0].insert_entry(0,WLOCEntry.get_zero_mul_entry())
        wloc_entrys[1].insert_entry(0,WLOCEntry.get_zero_mul_entry())
        

    return wloc_entrys        

 
def set_padding_flags (k_x, k_y, wloc_cmd_0, wloc_cmd_1):
    # do the down shift
    # wloc_cmd_1.shift_down = True
    
    if k_x==0:
        #padding last line
        wloc_cmd_1.pad_line_last    = True
        wloc_cmd_1.long_entry       = True
    elif k_x==2:
        #padding first line
        wloc_cmd_0.pad_line_first   = True
        wloc_cmd_0.long_entry       = True       

    if k_y==0:
        #padding last colomn
        wloc_cmd_1.pad_column_last  = True
        wloc_cmd_1.long_entry       = True
    elif k_y==2:
        #padding first colomn
        wloc_cmd_0.pad_column_first = True
        wloc_cmd_0.long_entry       = True 
    
def set_shift_flags (k_x, k_y, wloc_cmd_0, wloc_cmd_1):
    match (k_y, k_x):
        case (0, 0):
            pass
        case (0, 1):
            wloc_cmd_1.shift_down = True
            wloc_cmd_1.long_entry = True
        case (0, 2):
            wloc_cmd_1.shift_down = True
            wloc_cmd_1.long_entry = True
        case (1, 0):
            wloc_cmd_0.shift_right = True
            wloc_cmd_0.long_entry  = True
        case (1, 1):
            wloc_cmd_1.shift_down = True
            wloc_cmd_1.long_entry = True
        case (1, 2):
            wloc_cmd_1.shift_down = True
            wloc_cmd_1.long_entry = True
        case (2, 0):
            wloc_cmd_0.shift_right = True
            wloc_cmd_0.long_entry  = True
        case (2, 1):
            wloc_cmd_1.shift_down = True
            wloc_cmd_1.long_entry = True
        case (2, 2):
            wloc_cmd_1.shift_down = True
            wloc_cmd_1.long_entry = True
        case _:
            return "Invalid case"
    
      
 
def generate_grids_wloc_cbc(current_op_weights_tensor_list, node, oc_ch_compute_sequence):
    

    wlocs = [WLOCList() ,WLOCList()]
    wloc_shift_down_entry  = WLOCEntry(weight_value=0, weight_index=1, nop=False, shift_down=True,  long_entry=True)
    wloc_shift_right_entry = WLOCEntry(weight_value=0, weight_index=1, nop=False, shift_right=True, long_entry=True)
    wloc_nop_entry         = WLOCEntry.get_zero_mul_entry()

    wloc_nop_entry_always_long                = WLOCEntry.get_zero_mul_entry()
    wloc_nop_entry_always_long.is_always_long = True   


   
    kernel_size = current_op_weights_tensor_list.shape[2]
    
    is_3x3_conv = (kernel_size==3)
    delete_count_max_pool = 0
    for current_oc in oc_ch_compute_sequence:
        wloc_zero_mul_entry = (WLOCEntry(weight_value = 0, weight_index = 1, weight_offset = 1, long_entry=True, oc =current_oc))
        current_oc_first_cmd = len(wlocs[0].cmd_list)
        for k_y in range(kernel_size):                               
            for k_x in range(kernel_size):
                
                # we get ALWAYS TWO inputs: for ADD, they are used, for Conv the second is empty for ADD two are full
                ## the kernel looks like kernel[list_oc, list_ic, y, x ]

                wight_current_tensor    = current_op_weights_tensor_list[current_oc,:,k_x,k_y]                                                       
                new_generated_wloc_part =  generate_1x1_conv_alex(kernel_1x1 = wight_current_tensor, oc=current_oc, node=node, first_calc=not bool(k_x+k_y))

                #set shift flags
                if is_3x3_conv:
                    set_shift_flags(      k_x,k_y,new_generated_wloc_part[0].cmd_list[1], new_generated_wloc_part[1].cmd_list[1])
                    new_generated_wloc_part[0].cmd_list[0].is_always_long = True
                    new_generated_wloc_part[1].cmd_list[0].is_always_long = True
                    
                #set shift padding
                for i in range(len(new_generated_wloc_part[0].cmd_list)):
                    if is_3x3_conv:
                        set_padding_flags(k_x,k_y,new_generated_wloc_part[0].cmd_list[i], new_generated_wloc_part[1].cmd_list[i])

                    wlocs[0].add_entry(new_generated_wloc_part[0].cmd_list[i])
                    wlocs[1].add_entry(new_generated_wloc_part[1].cmd_list[i])
 
                         
        ###### this is the TALE of 3x3 Conv ############################
        if is_3x3_conv: #add shift for the 3x3 do 1 additional shifts
            #SHR
            wlocs[0].add_entry(copy.deepcopy(wloc_nop_entry))
            wlocs[1].add_entry(copy.deepcopy(wloc_nop_entry))
            wlocs[0].add_entry(copy.deepcopy(wloc_shift_right_entry))
            wlocs[1].add_entry(copy.deepcopy(wloc_zero_mul_entry))
            #SHD
            wlocs[0].add_entry(copy.deepcopy(wloc_nop_entry))
            wlocs[1].add_entry(copy.deepcopy(wloc_nop_entry))
            wlocs[1].add_entry(copy.deepcopy(wloc_shift_down_entry))
            wlocs[0].add_entry(copy.deepcopy(wloc_zero_mul_entry))

        #EOC
        wlocs[0].add_entry(copy.deepcopy(wloc_nop_entry_always_long)) 
        wlocs[0].add_entry(copy.deepcopy(wloc_nop_entry_always_long))
        wlocs[0].cmd_list[-1].oc= current_oc
        wlocs[0].set_oc_end()

        wlocs[1].add_entry(copy.deepcopy(wloc_nop_entry_always_long))
        wlocs[1].add_entry(copy.deepcopy(wloc_nop_entry_always_long))
        wlocs[1].cmd_list[-1].oc = current_oc
        wlocs[1].set_oc_end()

        #end of channel nop     
        #if the first command is W0, delete it
        if wlocs[0].cmd_list[current_oc_first_cmd].weight_value==0 and  wlocs[1].cmd_list[current_oc_first_cmd].weight_value==0:
            del wlocs[0].cmd_list[current_oc_first_cmd]
            del wlocs[1].cmd_list[current_oc_first_cmd] 
            current_oc_first_cmd-=1             

        # ################ BEGIN fix of Max Pool first command###########
        if (node['op_type'] == "MaxPool"):
            # Find first element with weight == 1 in wlocs[0], starting from delete_count_max_pool
            for idx, cmd in enumerate(wlocs[0].cmd_list[delete_count_max_pool:], start=delete_count_max_pool):
                if cmd.weight_value == 1:
                    delete_count_max_pool = idx
                    break

            # Delete elements from current_oc_first_cmd to delete_count_max_pool (range)
            if delete_count_max_pool > (current_oc_first_cmd+1):
                del wlocs[0].cmd_list[current_oc_first_cmd+1:delete_count_max_pool]
                if len(wlocs) > 1:
                    del wlocs[1].cmd_list[current_oc_first_cmd+1:delete_count_max_pool]

                delete_count_max_pool=len(wlocs[0].cmd_list)-1                 
        # ################ END fix of Max Pool first command###########

    return wlocs


def generate_oc_ch_compute_sequence( node):      
       
    #compute each channel one after another.
    oc_ch_compute_sequence = np.arange(node['backend']['output_channels'])

    if ('force_unfolding_x' in node['frontend']) and (node['frontend']['force_unfolding_x'] == True):
        # unfolding, do each output channel x2 times
        channel_in_slice = node['frontend']['output_channels']
        base_channels = node['frontend']['output_channels']
        y_num_of_folding_groups = 2**node['frontend']['output_folding_factor_y']
        jump_to_next_folded_neighbour = channel_in_slice // y_num_of_folding_groups

        # for each folding group we have to do two times the same channel
        oc_ch_compute_sequence = []
        for i_fold_group in range(y_num_of_folding_groups):
            channels_to_calculate_in_this_folded_group = np.arange(2*base_channels)[2*i_fold_group*(base_channels//y_num_of_folding_groups):(2*i_fold_group+1)*(base_channels//y_num_of_folding_groups)]    
            new_oc_sequence = []
            for x in channels_to_calculate_in_this_folded_group:
                new_oc_sequence.extend([x, x + jump_to_next_folded_neighbour])
            oc_ch_compute_sequence.extend(new_oc_sequence)

    return oc_ch_compute_sequence

 

        




# export in the Iramar Format
def export_cbc_to_xls_itamar_format(node,filename, current_tile_num=0):

    wloc_tile = node['backend']['grids_cbc'].alex_wlocs[current_tile_num]
    total_splits = len(wloc_tile)
    
    for i_splits in range(total_splits): 

        csv_filename = filename+'_tile_'+str(current_tile_num)+'_split'+str(i_splits)+'.csv'
        csv_file = open(csv_filename,'w')
        csv_file.write('CLC, WLOC_0, WLOC_1, RQ_Param_scale, RQ_bias, RQ_rough_shift_sel, RQ_Param_shift, RQ_Param_NOP,' +
                    'RT_nop_count, AMM_write_address, AMM_write_mask, scale_grid_write_sel, resize_grid_write_sel,  pipeline_reset, CMD_complete \n') 
        
        rt_attributes = [ "nop_count", "AMM_write_add", "AMM_write_mask", "scale", "resize_grid_sel", "result_pipeline_reset", "CMD_complete"]
        current_nops_rq = current_nops_rt = 0 
        current_inx_rq  = current_inx_rt = 0


        
        for  clc_idx in range(len(wloc_tile[i_splits][0].cmd_list)):
            ##------ clc -----
            clc_str = str(clc_idx)
            ##----- wloc
            str_w0 = str(wloc_tile[i_splits][0].cmd_list[clc_idx])
            str_w1 = str(wloc_tile[i_splits][1].cmd_list[clc_idx])
        
            ##-------------RQ -------------------
            if current_nops_rq==0 and current_inx_rq<len(node["backend"]['grids_cbc'].alex_rqParam[i_splits][0].cmd_list): 
                
                cmd_rq = node["backend"]['grids_cbc'].alex_rqParam[i_splits][0].cmd_list[current_inx_rq]
                current_nops_rq = cmd_rq.nop_count
                if cmd_rq.shift_count==0 and cmd_rq.is_config_cmd== False:
                    #this is nop cmd
                    rq_scale = rq_shift = rq_bias = rq_rough= 0
                    rq_nop = cmd_rq.nop_count
                else:
                    #this is not nop cmd 
                    rq_scale = cmd_rq.scale*(cmd_rq.shift_count==0)
                    rq_bias = cmd_rq.bias*(cmd_rq.shift_count==0)
                    rq_rough = cmd_rq.rough_shift_sel*(cmd_rq.shift_count==0)
                    rq_shift = cmd_rq.shift_count
                    rq_nop = cmd_rq.nop_count

                current_inx_rq+=1
            else:
                # the "intern state"
                current_nops_rq -=1
                rq_scale=rq_shift=rq_bias=rq_rough = 0
                rq_nop = current_nops_rq
                rq_shift = max(0, rq_shift-1)         
                

            str_RQ = ','+str(rq_scale)+',' +str(rq_bias)+','+ str(rq_rough)+','+str(rq_shift)+","+str(rq_nop)
            ##-------------RT -------------------
            if current_nops_rt==0 and current_inx_rt<len(node["backend"]['grids_cbc'].RTable[i_splits].cmd_list):
                RT_cmd = node["backend"]['grids_cbc'].RTable[i_splits].cmd_list[current_inx_rt]
                current_nops_rt = RT_cmd.nop_count
                
                str_RT = ''
                for attr in rt_attributes:
                    str_RT = str_RT + ',' + str(int(getattr(RT_cmd, attr)))
                
                current_inx_rt +=1
            else:   
                current_nops_rt -=1   
                nops_rt_write = max(0,current_nops_rt)  
                str_RT = ','+ str(nops_rt_write) 
                for attr in rt_attributes[1:]:
                    str_RT = str_RT +','+ str(getattr(RT_cmd, attr)*0)
                
            
            

            csv_file.write((clc_str+','+str_w0+','+str_w1+str_RQ+str_RT+"\n")) 

        
                
        csv_file.close()


def export_cbc_to_xls_alex(node,filename,format = DebugFilesFormat.XLSX, current_tile_num=0):
    # export_cbc_to_xls_itamar_format already loops internally over every split for
    # current_tile_num, so it must be called once here, not once per split (the previous
    # `for i_splits in range(total_splits)` loop called it total_splits times, redoing all
    # splits each time - O(n^2) redundant CSV writes).
    export_cbc_to_xls_itamar_format(node,filename, current_tile_num)



