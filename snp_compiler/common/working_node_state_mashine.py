from abc import ABC, abstractmethod
from copy import deepcopy
import math
import numpy as np
from common.enums import GridConfig
from common.hw_config import HARDWARE_SYNC_CMD_COMPLATE, RQ_BEGIN_SYNC_NOP, RT_BEGIN_SYNC_NOP, STALL_NOPS_FREQ, STALLS_CLOCKS, TFLITE_REQUANT, USE_BEGIN_SYNC, WLOC_PAIR_LONG_SIZE, WLOC_SINGLE_LONG_SIZE, WLOC_SINGLE_SHORT_SIZE
from common.program_ir import NonLinearFunctionList, RQParamEntry, RQParamList, RTEntry, RTableList, LinearFunctionList, WLOCEntry, WLOCList
from common.utils import get_rq_params
import common.internal_representation as internal_representation

SHIFT_BUFFER_LENGTH = 8
RT_WRITE_BUFFER_CLC = 3
EXTRA_TFLITE_SHIFTS = 3*TFLITE_REQUANT # those shifts are needed, because TFLITE needs 2 more clock to rqant the results

def calculate_AMM_write_add_for_folding_case(current_oc, real_chanals):

    y_fold_group_num = current_oc//real_chanals
    num_in_group     = current_oc%real_chanals

    #target_oc_only for 2   = current_oc + (current_oc//real_chanals)*real_chanals
    target_oc   = y_fold_group_num*real_chanals*2+num_in_group
    jump_to_next_part = real_chanals

    return  target_oc, target_oc+jump_to_next_part

def generate_grids_1x1_cbc(ir: internal_representation.IR, node, grids_cbc, generated_wlocs):
    NUM_LONG_CMDS_IN_IDENTITY = 1+1+2+2 #one calculation + one EOC + 2 AMM_SYNC + 2 STALL
    NUM_SHOR_CMDS_IN_IDENTITY = 3 # 9 - 6 long cmds
    NUM_CMDS_IN_IDENTITY      = NUM_LONG_CMDS_IN_IDENTITY + NUM_SHOR_CMDS_IN_IDENTITY
    MIN_LEN_PRO_CHANNAL       = (NUM_LONG_CMDS_IN_IDENTITY*WLOC_SINGLE_LONG_SIZE + NUM_SHOR_CMDS_IN_IDENTITY*WLOC_SINGLE_SHORT_SIZE)
    CLK_FROM_EOC_TO_AMM_WRITE = 28-1
    CLK_CLC_FROM_EOC_TO_RQ_PARAM = 7
    START_RT_NOPS             = 14
    START_RQ_NOPS             = 9

    
    wloc_lists     = [WLOCList()    for _ in range(2)]                # create of array of WLOCList
    generated_wlocs_no_nops = [WLOCList()    for _ in range(2)]
    rq_param_lists = [RQParamList()] 
    rtable_list    = RTableList() 

    # create new tile in wloc (I think this logic with 0 have to be deleted, it was old consept to support the ADD tiles )
    grids_cbc.alex_wlocs.append([])

         
    #TODO have to write it more accurate
    #delete all nops from generated wlocs and set EOC in the right place 
    #dont touch it, because this gives the num of channal for resize and requantnode
    len_list = len(generated_wlocs[0].cmd_list)
    for num in range(len_list):
        c0 = generated_wlocs[0].cmd_list[num]
        c1 = generated_wlocs[1].cmd_list[num]
        if c0.weight_value!=0:
            generated_wlocs_no_nops[0].add_entry(deepcopy(c0))
            generated_wlocs_no_nops[1].add_entry(deepcopy(c1))


        if c0.end_of_oc:
            if generated_wlocs_no_nops[0].cmd_list[-1].end_of_oc  == True:
                #this is if the OC is empty, and we have to put the NOP+EOC
                generated_wlocs_no_nops[0].add_entry(WLOCEntry.get_zero_mul_entry())
                generated_wlocs_no_nops[1].add_entry(WLOCEntry.get_zero_mul_entry())
                generated_wlocs_no_nops[0].cmd_list[-1].oc  = c0.oc
                generated_wlocs_no_nops[1].cmd_list[-1].oc  = c0.oc

            #normal case
            generated_wlocs_no_nops[0].cmd_list[-1].end_of_oc  = True
            generated_wlocs_no_nops[1].cmd_list[-1].end_of_oc  = True
            
    #the first channal have to be minimum 2 commands, if not the RQ will be not ready to get results after START_RQ_NOPS
    # so the first channal will be minimum 3 cmds -> 2 cmd and one NOP-before-EOC, that will be added later
        
    #calculate the real channals
    output_channels = node['frontend']['output_channels']
    if (node['op_type'] == "Resize"):
        output_channels = 4*node['frontend']['input_channels'] # Resize have 4x channals
    elif ("requantnode" in node['name']):
        output_channels = len(generated_wlocs_no_nops[0].cmd_list)
    elif ("unfold" in node['name']):
        output_channels = len(generated_wlocs_no_nops[0].cmd_list)

    HEAD_SIZE         = 4*WLOC_SINGLE_LONG_SIZE #first channal have 4 clk and EOF
    ONE_CHEN_MIN_SIZE = (3*WLOC_SINGLE_LONG_SIZE + 6*WLOC_SINGLE_SHORT_SIZE)
    AMM_SIZE          = output_channels*2*(WLOC_SINGLE_LONG_SIZE-WLOC_SINGLE_SHORT_SIZE)
    TAIL_SIZE         = (CLK_FROM_EOC_TO_AMM_WRITE+HARDWARE_SYNC_CMD_COMPLATE+1)*WLOC_SINGLE_SHORT_SIZE 
    if node['frontend']['output_channels'] == len(generated_wlocs_no_nops[0].cmd_list):
        #this is Identity or Add
        # this calculation do not take the fact that EOC can be with AMM_NOP, so this is biger a bit 
        wloc_lists_total_langth = HEAD_SIZE+(output_channels-1)*ONE_CHEN_MIN_SIZE+AMM_SIZE+TAIL_SIZE
        
    elif  (node['op_type'] == "Resize") or ("requantnode" in node['name']):
        #this is Resize or requantnode
        wloc_lists_total_langth = HEAD_SIZE+(output_channels-1)*ONE_CHEN_MIN_SIZE+AMM_SIZE+TAIL_SIZE
                  
    else:
        wloc_lists_total_langth = 0
        wloc_lists_total_langth_bite = 0
        channal_len    = 0
        channel_pairs  = 0
        size_real_wloc = 0
        size_real_wloc_in_channal = 0
        for num, cmd in enumerate (generated_wlocs_no_nops[0].cmd_list):
            size_real_wloc_in_channal +=1 
            if cmd.is_pair:
                channal_len += WLOC_PAIR_LONG_SIZE
                channel_pairs += 1
            else: # long    
                channal_len += WLOC_SINGLE_LONG_SIZE
            if  cmd.end_of_oc:
                channal_len+=WLOC_SINGLE_LONG_SIZE #NOP before EOF, 
                wloc_lists_minimum= MIN_LEN_PRO_CHANNAL+channel_pairs*(WLOC_PAIR_LONG_SIZE-WLOC_SINGLE_LONG_SIZE)
                wloc_lists_total_langth += max(wloc_lists_minimum, channal_len)
                if size_real_wloc_in_channal<NUM_CMDS_IN_IDENTITY:
                    size_real_wloc_in_channal=NUM_CMDS_IN_IDENTITY+1 #this is the NOP before EOF
                size_real_wloc+=size_real_wloc_in_channal

                size_real_wloc_in_channal = 0
                channal_len               = 0
                channel_pairs             = 0

        #ADD the AMM NOPs
        size_real_wloc +=output_channels*2
        wloc_lists_total_langth +=output_channels*2*WLOC_SINGLE_LONG_SIZE # we do not calculate the replasement
        #ADD the TAIL
        size_real_wloc +=(CLK_FROM_EOC_TO_AMM_WRITE+HARDWARE_SYNC_CMD_COMPLATE+3)
        wloc_lists_total_langth +=(WLOC_SINGLE_SHORT_SIZE*(CLK_FROM_EOC_TO_AMM_WRITE+HARDWARE_SYNC_CMD_COMPLATE)+
                                   3*WLOC_SINGLE_LONG_SIZE) #this are 3 LONGS at the End
        #ADD the DRR STALLS
        num_stalls_cmd = math.ceil(size_real_wloc/STALL_NOPS_FREQ)*2
        wloc_lists_total_langth +=num_stalls_cmd*WLOC_SINGLE_SHORT_SIZE
        

    wloc_lists_total_langth_bite     = math.ceil(wloc_lists_total_langth/8)
    # Problem with this algo aproach:
    # 1) After Split, we have additonal length because of taile in each split. This have to case overflow
    # 2) The channals are not simatrical, so if in one split will be the long channels -> overflow
    # all those problem have be solved with another aproach 

    cmd_depth_cells   =  math.ceil(wloc_lists_total_langth/(wloc_lists[0].mem_output_width)) # How many depth we need for each output channel
    amm_extra_depth_cells =  math.ceil(output_channels*WLOC_SINGLE_LONG_SIZE/wloc_lists[0].mem_output_width) 
    min_stall_depth_cells =  math.ceil((output_channels*(NUM_LONG_CMDS_IN_IDENTITY+NUM_SHOR_CMDS_IN_IDENTITY)/STALLS_CLOCKS*WLOC_SINGLE_LONG_SIZE)/wloc_lists[0].mem_output_width)
    stall_extra_depth_cells =  math.ceil((len(generated_wlocs_no_nops[0].cmd_list)/STALL_NOPS_FREQ*WLOC_SINGLE_LONG_SIZE)/wloc_lists[0].mem_output_width)          
    stall_extra_depth_cells = max(stall_extra_depth_cells, min_stall_depth_cells)
    wloc_lists_total_langth_cells = cmd_depth_cells + amm_extra_depth_cells + stall_extra_depth_cells
    max_mem_for_wloc_split = wloc_lists[0].mem_output_depth//2 
    splits_needed = math.ceil(wloc_lists_total_langth_cells/max_mem_for_wloc_split)
    #this is a trik, but it works :)
    extra_splits = splits_needed//4
    splits_needed = splits_needed+extra_splits 
    num_channals_per_split = math.ceil(output_channels/splits_needed)
    act_oc_channal = 0
    cmd_in_instraction = 0
    wloc_AMM_W_nop_entry = WLOCEntry(weight_value = 0, weight_index = 1, weight_offset = 1,  long_entry=True, nop=False, nop_reason='AMM_SYNC')
    wloc_STALL_nop_entry = WLOCEntry(weight_value = 0, weight_index = 0,                     long_entry=False, nop=True, nop_reason='DDR STALL')
    wLoc_EOC_entry       = WLOCEntry(                  weight_index = 1,                     long_entry=True, nop=False, end_of_oc=True)
    wloc_nop_short_entry = WLOCEntry(weight_value = 0, weight_index = 1, weight_offset = 1,  long_entry=False, nop=False, nop_reason='EMPTY SELL NOP')

    rq_shift_command     = RQParamEntry(nop_count = 7, shift_count = 7)
    

    for i_split in range(splits_needed):
        # begin of split RQ ##################################################
        rq_param_lists[0].add_entry(RQParamEntry(nop_count = START_RQ_NOPS,                      shift_count = 0, is_config_cmd=False))             
        #rq_param_lists[0].add_entry(deepcopy(rq_set_command))
        # begin of split RT NOPS ##################################################
        rtable_list.add_entry(RTEntry(nop_count = START_RT_NOPS))  #was 15 
        last_write_cmd_index = START_RT_NOPS+1
        last_shift_cmd_index = START_RQ_NOPS+1             
        
        is_first_oc_in_split = True               
        cmd_in_wloc = 0
        stalls_cntr=0
        last_stall_clk = 0
        jump_to_next_channal = 0 

        if generated_wlocs_no_nops[0].cmd_list[cmd_in_instraction].end_of_oc  == True:
            generated_wlocs_no_nops[0].cmd_list.insert(cmd_in_instraction, WLOCEntry.get_zero_mul_entry())
            generated_wlocs_no_nops[1].cmd_list.insert(cmd_in_instraction, WLOCEntry.get_zero_mul_entry()) 


        while act_oc_channal<num_channals_per_split*(i_split+1) and act_oc_channal<output_channels:
             
            cmd_in_oc = 0
            is_chanal_finisheded = False
            
            # Itamar requested to have 1 NOP before EOC flag, so each chanal have be beginned with NOP
            if  is_first_oc_in_split==False: 
                was_nop_reason  = wloc_lists[0].cmd_list[cmd_in_wloc].nop_reason
                was_eof         = wloc_lists[0].cmd_list[cmd_in_wloc].end_of_oc                    
                wloc_lists[0].cmd_list[cmd_in_wloc] = deepcopy(WLOCEntry.get_zero_mul_entry())
                wloc_lists[1].cmd_list[cmd_in_wloc] = deepcopy(WLOCEntry.get_zero_mul_entry())
                wloc_lists[0].cmd_list[cmd_in_wloc].end_of_oc = was_eof
                wloc_lists[1].cmd_list[cmd_in_wloc].end_of_oc= was_eof
                wloc_lists[0].cmd_list[cmd_in_wloc].nop_reason = was_nop_reason
                wloc_lists[1].cmd_list[cmd_in_wloc].nop_reason = was_nop_reason


                #wloc_lists[0].add_entry(wloc_nop_short_entry) # add NOP in the end
                #wloc_lists[1].add_entry(wloc_nop_short_entry)
                cmd_in_wloc += 1
                cmd_in_oc   += 1 
            while is_chanal_finisheded==False:
                
                #write cmd for both WLOC lists
                cmd0 = generated_wlocs_no_nops[0].cmd_list[cmd_in_instraction]
                cmd1 = generated_wlocs_no_nops[1].cmd_list[cmd_in_instraction]
                oc_num = cmd0.oc



                #before adding the cmd check if we have space in WLOC
                if len(wloc_lists[0].cmd_list)<(cmd_in_wloc+1): 
                    wloc_lists[0].add_entry(wloc_nop_short_entry) 
                    wloc_lists[1].add_entry(wloc_nop_short_entry)

                #check if we can write here the cmd. This sell have to be empty
                while wloc_lists[0].cmd_list[cmd_in_wloc].nop_reason!='EMPTY SELL NOP':
                    cmd_in_wloc += 1
                    cmd_in_oc   += 1


                #now add it (if it was EOF, save the flag and put it back)
                was_eof = wloc_lists[0].cmd_list[cmd_in_wloc].end_of_oc     
                wloc_lists[0].cmd_list[cmd_in_wloc] = deepcopy(cmd0)
                wloc_lists[1].cmd_list[cmd_in_wloc] = deepcopy(cmd1)
                wloc_lists[0].cmd_list[cmd_in_wloc].end_of_oc = was_eof
                wloc_lists[1].cmd_list[cmd_in_wloc].end_of_oc = was_eof
                cmd_in_instraction += 1
                cmd_in_wloc += 1
                cmd_in_oc   += 1

                #try to add STALLS
                if ('optimal_offload_point_for_tensor' in node['backend']) or ('add_stall_nops' in node['backend'] and node['backend']['add_stall_nops'] == True): 
                    nop_is_possible_to_put = (len(wloc_lists[0].cmd_list)>cmd_in_wloc+1                      and
                                              wloc_lists[0].cmd_list[cmd_in_wloc  ].nop_reason== 'EMPTY SELL NOP' and
                                              wloc_lists[1].cmd_list[cmd_in_wloc  ].nop_reason== 'EMPTY SELL NOP' and
                                              wloc_lists[0].cmd_list[cmd_in_wloc+1].nop_reason== 'EMPTY SELL NOP' and
                                              wloc_lists[1].cmd_list[cmd_in_wloc+1].nop_reason== 'EMPTY SELL NOP' and
                                              wloc_lists[0].cmd_list[cmd_in_wloc-2].nop_reason!= 'AMM_SYNC'  and
                                              wloc_lists[0].cmd_list[cmd_in_wloc-1].nop_reason!= 'AMM_SYNC'  and 
                                              wloc_lists[0].cmd_list[cmd_in_wloc+2].nop_reason!= 'AMM_SYNC'  and
                                              wloc_lists[0].cmd_list[cmd_in_wloc+3].nop_reason!= 'AMM_SYNC'   
                                              )
                    alternativ_shift = 7
                    alternative_place = cmd_in_wloc - alternativ_shift
                    nop_for_identity_to_put = (alternative_place>0                                                 and
                                              len(wloc_lists[0].cmd_list)>alternative_place+1                      and
                                              wloc_lists[0].cmd_list[alternative_place  ].nop_reason== 'EMPTY SELL NOP' and
                                              wloc_lists[1].cmd_list[alternative_place  ].nop_reason== 'EMPTY SELL NOP' and
                                              wloc_lists[0].cmd_list[alternative_place+1].nop_reason== 'EMPTY SELL NOP' and
                                              wloc_lists[1].cmd_list[alternative_place+1].nop_reason== 'EMPTY SELL NOP' and
                                              wloc_lists[0].cmd_list[alternative_place-2].nop_reason!= 'AMM_SYNC'  and
                                              wloc_lists[0].cmd_list[alternative_place-1].nop_reason!= 'AMM_SYNC'  and 
                                              wloc_lists[0].cmd_list[alternative_place+2].nop_reason!= 'AMM_SYNC'  and
                                              wloc_lists[0].cmd_list[alternative_place+3].nop_reason!= 'AMM_SYNC'   
                                              )
                    stalls_cntr= cmd_in_wloc-last_stall_clk 
                    if stalls_cntr >= STALL_NOPS_FREQ and (nop_is_possible_to_put):
                        ## put STALL
                        for k in range(2):
                            wloc_lists[0].cmd_list[cmd_in_wloc+k] = deepcopy(wloc_STALL_nop_entry)
                            wloc_lists[1].cmd_list[cmd_in_wloc+k] = deepcopy(wloc_STALL_nop_entry)

                        for k in range(2):
                            if (wloc_lists[k].cmd_list[cmd_in_wloc+2].long_entry==False) and (wloc_lists[k].cmd_list[cmd_in_wloc+2].weight_value==0):
                                # if after stall there is EMPTY SELL NOP, change it to LONG NOP (Itamar request)
                                wloc_lists[k].cmd_list[cmd_in_wloc+2] = deepcopy(WLOCEntry.get_zero_mul_entry())
                                wloc_lists[k].cmd_list[cmd_in_wloc+2].nop_reason= 'EMPTY SELL NOP' # mark this as empty sell
                        
                        cmd_in_wloc += 2
                        cmd_in_oc   += 2
                        stalls_cntr=0
                        last_stall_clk =  cmd_in_wloc
                        node['backend']['statistics']['stall_nops'] += 2
                    elif stalls_cntr >= STALL_NOPS_FREQ+alternativ_shift and (nop_for_identity_to_put):
                        ## this is for identity, because the normal place for stalls is always imposible for identity
                        for k in range(2):
                            wloc_lists[0].cmd_list[alternative_place+k] = deepcopy(wloc_STALL_nop_entry)
                            wloc_lists[1].cmd_list[alternative_place+k] = deepcopy(wloc_STALL_nop_entry)
                        
                        for k in range(2):
                            if (wloc_lists[k].cmd_list[alternative_place+2].long_entry==False) and (wloc_lists[k].cmd_list[alternative_place+2].weight_value==0):
                                #if after stall there is EMPTY SELL NOP, change it to LONG NOP (Itamar request)
                                wloc_lists[k].cmd_list[alternative_place+2] = deepcopy(WLOCEntry.get_zero_mul_entry())
                                wloc_lists[k].cmd_list[cmd_in_wloc+2].nop_reason== 'EMPTY SELL NOP' # mark this as empty sell

                        stalls_cntr=cmd_in_wloc - alternative_place 
                        last_stall_clk =  alternative_place
                        node['backend']['statistics']['stall_nops'] += 2
                    # else:
                    #    stalls_cntr= cmd_in_wloc-last_stall_clk      

                # close the oc - add AMM NOPs
                if cmd0.end_of_oc:
                    # del  EOC flag in the end of calculation, we will set it 2 clc after it  (as Itamar requested)
                    #wloc_lists[0].cmd_list[cmd_in_wloc-1].end_of_oc = False
                    #wloc_lists[1].cmd_list[cmd_in_wloc-1].end_of_oc = False
                    
                    # finish the channal
                    jump_to_next_channal = max(0, NUM_CMDS_IN_IDENTITY - cmd_in_oc)
                    if is_first_oc_in_split:
                        jump_to_next_channal = min(0, jump_to_next_channal)
                    
                    cmd_in_wloc += jump_to_next_channal
                    node['backend']['statistics']['rq_waiting_nops']     +=jump_to_next_channal

                    # write the nops to wloc to have the opotunity oa add AMM NOP
                    if len(wloc_lists[0].cmd_list)<(cmd_in_wloc+CLK_FROM_EOC_TO_AMM_WRITE):
                        nops_to_add = cmd_in_wloc+CLK_FROM_EOC_TO_AMM_WRITE - len(wloc_lists[0].cmd_list)+3
                        for _ in range(nops_to_add):
                            wloc_lists[0].add_entry(wloc_nop_short_entry) 
                            wloc_lists[1].add_entry(wloc_nop_short_entry)                    

                    for j in range(2): #2 AMM NOPs are the last            
                        wloc_lists[0].cmd_list[cmd_in_wloc+CLK_FROM_EOC_TO_AMM_WRITE+j] = deepcopy(WLOCEntry.get_zero_mul_entry())#deepcopy(wloc_AMM_W_nop_entry)  
                        wloc_lists[0].cmd_list[cmd_in_wloc+CLK_FROM_EOC_TO_AMM_WRITE+j].nop_reason='AMM_SYNC'
                        wloc_lists[1].cmd_list[cmd_in_wloc+CLK_FROM_EOC_TO_AMM_WRITE+j] = deepcopy(WLOCEntry.get_zero_mul_entry())#deepcopy(wloc_AMM_W_nop_entry)
                        wloc_lists[1].cmd_list[cmd_in_wloc+CLK_FROM_EOC_TO_AMM_WRITE+j].nop_reason='AMM_SYNC'

                    #put EOC flag with two cmd deley (as Itamar requested)
                    wloc_lists[0].cmd_list[cmd_in_wloc+1].set_eof()
                    wloc_lists[1].cmd_list[cmd_in_wloc+1].set_eof()
         

                    #update RQ ####################################################################
                    # last_shift_cmd_index - exactly in this clc the parameters have to be set for Identity 
                    nop_to_wait_from_previous_shift = cmd_in_wloc - last_shift_cmd_index + CLK_CLC_FROM_EOC_TO_RQ_PARAM +1 
                    rq_one_clc_to_go_back = 0
                    if nop_to_wait_from_previous_shift==1:
                        # there no way to do only one Nop, so we write the Param
                        rq_set_command = RQParamEntry(           is_config_cmd=True, scale=1, bias= 1, rough_shift_sel= 0)
                        rq_param_lists[0].add_entry(deepcopy(rq_set_command))
                    elif nop_to_wait_from_previous_shift>1:       
                        rq_param_lists[0].add_entry(RQParamEntry(is_config_cmd=False, nop_count = nop_to_wait_from_previous_shift-1, scale=0))
                    elif nop_to_wait_from_previous_shift<0:
                        #if the channal is EOF in less than 7 clk between EOF and RQ_param
                        raise ('Sync Problem')                        
                    requant_scale_uint14, requant_bias_int12, rough_shift_uint2 = get_rq_params(node, channel=oc_num)
                    rq_set_command = RQParamEntry(is_config_cmd=True, nop_count = 0, scale=requant_scale_uint14, 
                                                  bias= requant_bias_int12, rough_shift_sel= rough_shift_uint2) 
                    rq_param_lists[0].add_entry(deepcopy(rq_set_command))
                    rq_param_lists[0].add_entry(deepcopy(rq_shift_command))
                    last_shift_cmd_index = cmd_in_wloc+rq_shift_command.nop_count + CLK_CLC_FROM_EOC_TO_RQ_PARAM + 3 

                    #update RT
                    rt_nops = cmd_in_wloc+CLK_FROM_EOC_TO_AMM_WRITE-last_write_cmd_index-1
                    rtable_list.add_entry(RTEntry(nop_count = rt_nops))
                    last_write_cmd_index = cmd_in_wloc+CLK_FROM_EOC_TO_AMM_WRITE+2+1-1

                    #generate RT WRITE COMMANDS ##########################
                    if not('force_folding_x' in node['frontend'])  and (not('force_unfolding_x' in node['frontend'])):
                            # no folding no unfolding
                            rtable_list.add_entry(RTEntry(nop_count = 0, AMM_write_add = oc_num, AMM_write_mask = 1, scale = 0))
                            rtable_list.add_entry(RTEntry(nop_count = 0, AMM_write_add = oc_num, AMM_write_mask = 2, scale = 1))

                    elif ('force_folding_x' in node['frontend']):
                            # folding case 
                            target_mask = 1
                            real_chanals = node['frontend']['output_channels']//(2**node['frontend']['output_folding_factor_y'])
                            amm_write_add0, amm_write_add1 = calculate_AMM_write_add_for_folding_case(current_oc=oc_num, real_chanals= real_chanals)
                            rt_write_cmd = RTEntry(AMM_write_add = amm_write_add0,   AMM_write_mask = target_mask, scale = 0)
                            rtable_list.add_entry(deepcopy(rt_write_cmd))
                            rt_write_cmd = RTEntry(AMM_write_add = amm_write_add1, AMM_write_mask = target_mask, scale = 1)
                            rtable_list.add_entry(rt_write_cmd)
                            
                    elif ('force_unfolding_x' in node['frontend']):
                            # TO DO: do as function
                            num_of_all_out_channals = node['frontend']['output_channels']
                            y_num_of_folding_groups = 2**node['frontend']['output_folding_factor_y']

                            base_channels = num_of_all_out_channals//y_num_of_folding_groups
                            folding_group_num = oc_num//(2*base_channels)                    
                            in_group_num      = oc_num%(base_channels)
                            
                            target_oc   = folding_group_num*(base_channels)+in_group_num
                            rt_write_cmd = RTEntry(AMM_write_add = target_oc, AMM_write_mask = 1, scale = 0)
                            rtable_list.add_entry(rt_write_cmd)
                            rt_write_cmd = RTEntry(AMM_write_add = target_oc, AMM_write_mask = 2, scale = 1)
                            rtable_list.add_entry(rt_write_cmd)             
                            
                    ##########################
                    is_chanal_finisheded = True
                    is_first_oc_in_split = False
                    act_oc_channal += 1                     


                #check if we need to write STALL NOPs
            # all channals are generated for this split
        # ###there can be the last channal needs a LONG NOP #####
        if wloc_lists[0].cmd_list[cmd_in_wloc].nop_reason=='EMPTY SELL NOP':
            wloc_lists[0].cmd_list[cmd_in_wloc] = deepcopy(WLOCEntry.get_zero_mul_entry())
            wloc_lists[1].cmd_list[cmd_in_wloc] = deepcopy(WLOCEntry.get_zero_mul_entry())    
        #set the teil NOPs ###############################################
        for _ in range(HARDWARE_SYNC_CMD_COMPLATE):
            wloc_lists[0].add_entry(deepcopy(wloc_nop_short_entry)) 
            wloc_lists[1].add_entry(deepcopy(wloc_nop_short_entry))
        
        # set the RQ
        rq_param_lists[0].add_entry(RQParamEntry(nop_count = len(wloc_lists[0].cmd_list)-last_shift_cmd_index-1, shift_count = 0))
        
        # set RT SYNC           
        rtable_list.add_entry(RTEntry(nop_count = HARDWARE_SYNC_CMD_COMPLATE-1))
        rtable_list.add_entry(RTEntry(CMD_complete = True))
        
        
        # create mem for each split
        wloc_lists[0].create_cmd_mem()
        wloc_lists[1].create_cmd_mem()
        rq_param_lists[0].create_cmd_mem()
        rtable_list.create_cmd_mem()
        real_len = len(wloc_lists[0].cmd_mem)
        wloc_lists_total_langth_bite
        if real_len>wloc_lists_total_langth_bite:
            pass
        if abs( wloc_lists_total_langth_bite - real_len)>200:
            pass
        ###################################################################################
        # create mem of wloc (this will be done for each tile)

        #creat 
        grids_cbc.alex_wlocs[0].append(wloc_lists)         
        grids_cbc.alex_rqParam.append( rq_param_lists )
        grids_cbc.RTable.append (      rtable_list    )

        ######### add NLF ##############
        if ('_fold_x_0' in node['name'] and (len(node['frontend']['preceding_nodes_params']) == 0)):
            if ir.uint8_int8_lut is not None:
                grids_cbc.nlf = [NonLinearFunctionList(ir.uint8_int8_lut)]
            elif ir.uint8_int8_conversion:
                lut = [(i - 128) for i in range(256)]
                ir.uint8_int8_lut = lut
                grids_cbc.nlf = [NonLinearFunctionList(lut)]
            else:
                grids_cbc.nlf = [LinearFunctionList()]
        elif ('lut_silu' in node['frontend']):
            grids_cbc.nlf = [NonLinearFunctionList(node['frontend']['lut_silu'])]
        else:
            grids_cbc.nlf = [LinearFunctionList()]
        grids_cbc.nlf[0].create_cmd_mem()
            
        # statistics
        node['backend']['statistics']['wloc_all_splits'] += len(wloc_lists[0].cmd_list)

        #reset for next split
        wloc_lists     = [WLOCList()    for _ in range(2)]                # create of array of WLOCList
        rq_param_lists = [RQParamList()]    
        rtable_list    = RTableList() 
    #################################### set statistics ##########################################
    node['backend']['statistics']['wloc_conv_cmd_count'] = act_oc_channal
    node['backend']['statistics']['nops_end_of_split']   = len(grids_cbc.alex_wlocs[0])*(CLK_FROM_EOC_TO_AMM_WRITE+HARDWARE_SYNC_CMD_COMPLATE+1) 
    node['backend']['statistics']['nops_AMM_SYNC']       = 2*len(node['backend']['oc_groups'][0]) #Nod_output * num_of_AMM_write_cmd   
    return grids_cbc

def generate_grids_identity_cbc(ir: internal_representation.IR, node, grids_cbc): 
    
    wloc_lists     = [WLOCList()    for _ in range(2)]                # create of array of WLOCList
    rq_param_lists = [RQParamList()] 
    rtable_list    = RTableList()

    # create new tile in wloc (I think this logic with 0 have to be deleted, it was old consept to support the ADD tiles )
    grids_cbc.alex_wlocs.append([])

    #Generate WLOC commands for Identity operation
    NUM_LONG_CMDS_IN_IDENTITY = 1+1+2+2 #one calculation + one EOC + 2 AMM_SYNC + 2 STALL
    NUM_SHOR_CMDS_IN_IDENTITY = 3 # 9 - 6 long cmds
    MUM_CMD_CMDS_IN_IDENTITY = NUM_LONG_CMDS_IN_IDENTITY + NUM_SHOR_CMDS_IN_IDENTITY
    CLK_FROM_EOC_TO_AMM_WRITE = 29
    ch_num_to_start_AMM_nops = CLK_FROM_EOC_TO_AMM_WRITE//MUM_CMD_CMDS_IN_IDENTITY # 29 is the langht of the pipiline between EOF and AMM WRITE

    CMD_IN_OC_IDENTITY = (NUM_LONG_CMDS_IN_IDENTITY*WLOC_SINGLE_LONG_SIZE + NUM_SHOR_CMDS_IN_IDENTITY*WLOC_SINGLE_SHORT_SIZE)
    CMD_IN_OC_IDENTITY_DEPTH_CELLS =  math.ceil(CMD_IN_OC_IDENTITY/wloc_lists[0].mem_output_width) # How many depth we need for each output channel

    WLOC_TEIL_DEPTH_CELLS =  math.ceil(CLK_FROM_EOC_TO_AMM_WRITE*WLOC_SINGLE_LONG_SIZE/wloc_lists[0].mem_output_width)  # How many depth we need for the TEIL NOPs (For simplicity all TEIL NOPs are long entries)
 
    wloc_AMM_W_nop_entry = WLOCEntry(weight_value = 0, weight_index = 1, long_entry=True, nop=True, nop_reason='AMM_SYNC')
    wloc_STALL_nop_entry = WLOCEntry(weight_value = 0, weight_index = 0, long_entry=True, nop=True, nop_reason='DDR STALL')
    wLoc_EOC_entry       = WLOCEntry(                  weight_index = 1, long_entry=True, nop=False, end_of_oc=True)
    wloc_nop_short_entry = WLOCEntry(weight_offset = 1,weight_index = 0, long_entry=False, nop=False, nop_reason='SHORT NOP') 

    rq_shift_command     = RQParamEntry(nop_count = 8, shift_count = 8)
    requant_scale_uint14, requant_bias_int12, rough_shift_uint2 = get_rq_params(node, channel=0)
    
    rq_set_command = RQParamEntry(is_config_cmd=True, nop_count = 0, 
                                  scale=requant_scale_uint14, bias= requant_bias_int12, rough_shift_sel= rough_shift_uint2) 

    #check hwo many splits we need
    output_channels = node['frontend']['output_channels']
    if (node['op_type'] == "Resize"):
        output_channels = 4*node['frontend']['input_channels'] # Resize have 4x channals

    wloc_lists_total_langth_cells = output_channels*CMD_IN_OC_IDENTITY_DEPTH_CELLS+WLOC_TEIL_DEPTH_CELLS
    max_mem_for_wloc_split = wloc_lists[0].mem_output_depth//2 
    splits_needed = math.ceil(wloc_lists_total_langth_cells/max_mem_for_wloc_split)
    num_channals_per_split = math.ceil(output_channels/splits_needed)
    act_oc_channal = 0
    # if this chanel in WLOC we start to add AMM NOPS


    # Itamar SYNC
    # wloc_lists[0].add_entry(WLOCEntry(long_entry=True,weight_value=127)) 
    # wloc_lists[1].add_entry(WLOCEntry.get_zero_mul_entry())

    #generate WLOC, RQ, RT for each split
    for i_split in range(splits_needed):
        # begin of split RQ ##################################################
        rq_param_lists[0].add_entry(RQParamEntry(nop_count = 9, shift_count = 0, is_config_cmd=False))             
        rq_param_lists[0].add_entry(deepcopy(rq_set_command))
        # begin of split RT NOPS ##################################################
        rtable_list.add_entry(RTEntry(nop_count = 14))  #was 15              
        rtable_list.add_entry(RTEntry(nop_count = 28-15)) #26 = CMD_IN_OC_IDENTITY
        is_first_oc_in_split = True
        # generate CALC + EOC for each output channel
        while act_oc_channal<num_channals_per_split*(i_split+1) and act_oc_channal<output_channels:
           for _ in range(MUM_CMD_CMDS_IN_IDENTITY):
               wloc_lists[0].add_entry(deepcopy(wloc_nop_short_entry)) 
               wloc_lists[1].add_entry(deepcopy(wloc_nop_short_entry))
           #set the end of OC     
           wloc_lists[0].cmd_list[-MUM_CMD_CMDS_IN_IDENTITY+2] = deepcopy(wLoc_EOC_entry)
           wloc_lists[1].cmd_list[-MUM_CMD_CMDS_IN_IDENTITY+2] = deepcopy(wLoc_EOC_entry)
           wloc_lists[0].cmd_list[-MUM_CMD_CMDS_IN_IDENTITY+2].oc = act_oc_channal 
           wloc_lists[1].cmd_list[-MUM_CMD_CMDS_IN_IDENTITY+2].oc = act_oc_channal

           #set the calculation find
           if (node['frontend']['input_folding_factor_x']>0 or node['frontend']['input_folding_factor_y']>0):
                weight_line = node['frontend']['folded_weights_tensor'].data[act_oc_channal,:,0,0]
           else:
                weight_line = node['frontend']['weights_tensor'].data[act_oc_channal,:,0,0]
            # find in the line the wheight that is not zero, and use it address
           wheight_address = np.where(weight_line!=0)[0]
           if len(wheight_address)!=1:
                raise ('This is no identity op, there are weights that are not zeros!')
           weight = weight_line[wheight_address[0]]  
           
           wloc_lists[0].cmd_list[-MUM_CMD_CMDS_IN_IDENTITY]= WLOCEntry.get_zero_mul_entry() #if act_oc_channal!=0 else True
           wloc_lists[0].cmd_list[-MUM_CMD_CMDS_IN_IDENTITY].weight_value = weight #if act_oc_channal!=0 else 0           
           wloc_lists[0].cmd_list[-MUM_CMD_CMDS_IN_IDENTITY].weight_index = wheight_address[0]
           wloc_lists[1].cmd_list[-MUM_CMD_CMDS_IN_IDENTITY] = WLOCEntry.get_zero_mul_entry()


           #set the AMM NOPs
           
           if act_oc_channal>=ch_num_to_start_AMM_nops:
               for i in range(2): #2 AMM NOPs are the last 2 CMDs
                   wloc_lists[0].cmd_list[-MUM_CMD_CMDS_IN_IDENTITY+i+3] = deepcopy(wloc_AMM_W_nop_entry) #wloc_lists[0].cmd_list[-1-i] = deepcopy(wloc_AMM_W_nop_entry)
                   wloc_lists[1].cmd_list[-MUM_CMD_CMDS_IN_IDENTITY+i+3] = deepcopy(wloc_AMM_W_nop_entry)
                   #wloc_lists[1].cmd_list[-1-i] = deepcopy(wloc_AMM_W_nop_entry)
           #set the STALL NOPs
           if act_oc_channal>=3:
               for i in range(2): #2 STALL NOPs are the first 2 CMDs
                   wloc_lists[0].cmd_list[-MUM_CMD_CMDS_IN_IDENTITY+3+2+2+i] = deepcopy(wloc_STALL_nop_entry)
                   wloc_lists[1].cmd_list[-MUM_CMD_CMDS_IN_IDENTITY+3+2+2+i] = deepcopy(wloc_STALL_nop_entry)


           #generate RQ and RT  ##################################################          
           if not is_first_oc_in_split: 
                # normal case - not first oc in split 
                rq_param_lists[0].add_entry(deepcopy(rq_shift_command))
                rtable_list.add_entry(RTEntry(nop_count = 8-2))
                

           #generate RT WRITE COMMANDS ##########################
           if not('force_folding_x' in node['frontend'])  and (not('force_unfolding_x' in node['frontend'])):
                # no folding no unfolding
                rtable_list.add_entry(RTEntry(nop_count = 0, AMM_write_add = act_oc_channal, AMM_write_mask = 1, scale = 0))
                rtable_list.add_entry(RTEntry(nop_count = 0, AMM_write_add = act_oc_channal, AMM_write_mask = 2, scale = 1))

           elif ('force_folding_x' in node['frontend']):
                # folding case 
                target_mask = 1
                real_chanals = node['frontend']['output_channels']//(2**node['frontend']['output_folding_factor_y'])
                amm_write_add0, amm_write_add1 = calculate_AMM_write_add_for_folding_case(current_oc=act_oc_channal, real_chanals= real_chanals)
                rt_write_cmd = RTEntry(AMM_write_add = amm_write_add0,   AMM_write_mask = target_mask, scale = 0)
                rtable_list.add_entry(deepcopy(rt_write_cmd))
                rt_write_cmd = RTEntry(AMM_write_add = amm_write_add1, AMM_write_mask = target_mask, scale = 1)
                rtable_list.add_entry(rt_write_cmd)
                
           elif ('force_unfolding_x' in node['frontend']):
                # TO DO: do as function
                num_of_all_out_channals = node['frontend']['output_channels']
                y_num_of_folding_groups = 2**node['frontend']['output_folding_factor_y']

                base_channels = num_of_all_out_channals//y_num_of_folding_groups
                folding_group_num = act_oc_channal//(2*base_channels)                    
                in_group_num      = act_oc_channal%(base_channels)
                
                target_oc   = folding_group_num*(base_channels)+in_group_num
                rt_write_cmd = RTEntry(AMM_write_add = target_oc, AMM_write_mask = 1, scale = 0)
                rtable_list.add_entry(rt_write_cmd)
                rt_write_cmd = RTEntry(AMM_write_add = target_oc, AMM_write_mask = 2, scale = 1)
                rtable_list.add_entry(rt_write_cmd)             
                 
           ##########################
           is_first_oc_in_split = False
           act_oc_channal += 1

        #set the teil NOPs ###############################################
        for _ in range(CLK_FROM_EOC_TO_AMM_WRITE+HARDWARE_SYNC_CMD_COMPLATE-5-1):
            wloc_lists[0].add_entry(deepcopy(wloc_nop_short_entry)) 
            wloc_lists[1].add_entry(deepcopy(wloc_nop_short_entry))
        # set last AMM SYNC    
        for j in range(ch_num_to_start_AMM_nops):    
            for i in range(2): #2 AMM NOPs are the last 2 CMDs
                   wloc_lists[0].cmd_list[-i-(j*MUM_CMD_CMDS_IN_IDENTITY+HARDWARE_SYNC_CMD_COMPLATE)-1] = deepcopy(wloc_AMM_W_nop_entry)
                   wloc_lists[1].cmd_list[-i-(j*MUM_CMD_CMDS_IN_IDENTITY+HARDWARE_SYNC_CMD_COMPLATE)-1] = deepcopy(wloc_AMM_W_nop_entry)
        
        # set the RQ SYNC
        rq_param_lists[0].add_entry(deepcopy(rq_shift_command))
        rq_param_lists[0].add_entry(RQParamEntry(nop_count = CLK_FROM_EOC_TO_AMM_WRITE-MUM_CMD_CMDS_IN_IDENTITY+HARDWARE_SYNC_CMD_COMPLATE-8-1, shift_count = 0))

        # set RT SYNC           
        rtable_list.add_entry(RTEntry(nop_count = HARDWARE_SYNC_CMD_COMPLATE-1))
        rtable_list.add_entry(RTEntry(CMD_complete = True))
        
        # create mem for each split
        wloc_lists[0].create_cmd_mem()
        wloc_lists[1].create_cmd_mem()
        rq_param_lists[0].create_cmd_mem()
        rtable_list.create_cmd_mem()
        ###################################################################################
        # create mem of wloc (this will be done for each tile)

        #creat 
        grids_cbc.alex_wlocs[0].append(wloc_lists)         
        grids_cbc.alex_rqParam.append( rq_param_lists )
        grids_cbc.RTable.append (      rtable_list    )

        ######### add NLF ##############
        if ('_fold_x_0' in node['name'] and (len(node['frontend']['preceding_nodes_params']) == 0)):
            if ir.uint8_int8_lut is not None:
                grids_cbc.nlf = [NonLinearFunctionList(ir.uint8_int8_lut)]
            elif ir.uint8_int8_conversion:
                lut = [(i - 128) for i in range(256)]
                ir.uint8_int8_lut = lut
                grids_cbc.nlf = [NonLinearFunctionList(lut)]
            else:
                grids_cbc.nlf = [LinearFunctionList()]
        elif ('lut_silu' in node['frontend']):
            grids_cbc.nlf = [NonLinearFunctionList(node['frontend']['lut_silu'])]
        else:
            grids_cbc.nlf = [LinearFunctionList()]
        grids_cbc.nlf[0].create_cmd_mem()
            
        # statistics
        node['backend']['statistics']['wloc_all_splits'] += len(wloc_lists[0].cmd_list)

        #reset for next split
        wloc_lists     = [WLOCList()    for _ in range(2)]                # create of array of WLOCList
        rq_param_lists = [RQParamList()]    
        rtable_list    = RTableList()        

    #################################### set statistics ##########################################
    node['backend']['statistics']['wloc_conv_cmd_count'] = act_oc_channal
    
    num_of_nops_to_finish_the_pipe = CLK_FROM_EOC_TO_AMM_WRITE-MUM_CMD_CMDS_IN_IDENTITY+3

    node['backend']['statistics']['nops_end_of_split']   = len(grids_cbc.alex_wlocs[0])*(num_of_nops_to_finish_the_pipe+HARDWARE_SYNC_CMD_COMPLATE+1) 

    if ('optimal_offload_point_for_tensor' in node['backend']) or ('add_stall_nops' in node['backend'] and node['backend']['add_stall_nops'] == True):
        #stall nops are needed
        node['backend']['statistics']['stall_nops']      = 2*act_oc_channal        
    else:
        #stall nops not needed
        node['backend']['statistics']['stall_nops']      = 0

    node['backend']['statistics']['rq_waiting_nops']     = act_oc_channal*MUM_CMD_CMDS_IN_IDENTITY - node['backend']['statistics']['stall_nops'] - 2*act_oc_channal #minus AMM 
    node['backend']['statistics']['wloc_pairs']          = 0 
    node['backend']['statistics']['nops_AMM_SYNC']       = 2*len(node['backend']['oc_groups'][0]) #Nod_output * num_of_AMM_write_cmd        
                               
    return grids_cbc


def  add_stalls(generated_wlocs, node):
    wloc_zero_mul_entry = WLOCEntry(weight_value = 0, weight_index = 1, weight_offset = 1, long_entry=True)
    wloc_stall_entry = (WLOCEntry(is_always_long = True, nop=True, nop_reason='DDR STALL'))
    stalls_cntr = 0 
    stall_pos   = 0 
    is_imposible_to_insert_add = np.zeros((len(generated_wlocs))).astype(bool)
    list_len = len(generated_wlocs[0].cmd_list)
    is_max_pool = (node['op_type'] == 'MaxPool')
    while stall_pos<list_len: 
        if stalls_cntr >= STALL_NOPS_FREQ:
            ## set the stalls in wloc
            for i in range(STALLS_CLOCKS):
                for wloc_inx,  i_wloc in enumerate(generated_wlocs):                
                   

                    is_imposible_to_insert_add[wloc_inx]  = (
                        (is_max_pool and generated_wlocs[0].cmd_list[stall_pos-1].weight_value  == -1) or
                        (stall_pos + 1 >= list_len) or
                        (stall_pos + 0 < list_len and (i_wloc.cmd_list[stall_pos + 0].shift_right or i_wloc.cmd_list[stall_pos + 0].shift_down)) or
                        (stall_pos + 1 < list_len and (i_wloc.cmd_list[stall_pos + 1].shift_right or i_wloc.cmd_list[stall_pos + 1].shift_down)) or
                        (stall_pos - 1 >= 0 and i_wloc.cmd_list[stall_pos - 1].end_of_oc) or
                        (stall_pos - 2 >= 0 and i_wloc.cmd_list[stall_pos - 2].end_of_oc) or
                        (stall_pos - 3 >= 0 and i_wloc.cmd_list[stall_pos - 3].end_of_oc)
                    )
                                   

                if all(np.logical_not( is_imposible_to_insert_add)):
                    for  i_wloc in generated_wlocs:                        
                        i_wloc.insert_entry(pos = stall_pos,entry=deepcopy(wloc_stall_entry) )
                        #check if the next cmd is SHORT
                        i_wloc.cmd_list[stall_pos+1].set_long()
                        #put the previose as long 
                        if (i==0 and i_wloc.cmd_list[stall_pos-1].nop==True):
                            i_wloc.cmd_list[stall_pos-1] = wloc_zero_mul_entry
                        #put the next as long 
                        if (i==0):
                            i_wloc.replace_posible_nop_with_zero_mul(stall_pos+1)   
                
                    # counters                
                    stalls_cntr=0
                    stall_pos +=1
                                    
                #stall_pos +=1 #we insert the enrty, before stall_pos, so we have to inc the pointer
        stall_pos +=1 #this is needed for for loop
        stalls_cntr+=1 

def check_nop_balance(generated_wlocs):
    for i_inx in range(len(generated_wlocs[0].cmd_list)):
        if (not generated_wlocs[0].cmd_list[i_inx].nop) and generated_wlocs[1].cmd_list[i_inx].nop:
            generated_wlocs[1].replace_posible_nop_with_zero_mul(i_inx)
        if (not generated_wlocs[1].cmd_list[i_inx].nop) and generated_wlocs[0].cmd_list[i_inx].nop:
            generated_wlocs[0].replace_posible_nop_with_zero_mul(i_inx)



def generate_grids_rq_cbc_alex2(grids_wloc_commands ,node, shift_reg_length = SHIFT_BUFFER_LENGTH, num_of_AMM_write_cmd = 2, cbc_mode = True):
    # 9) ADD in the begining NOPS to sync with HV
    
    AMM_DELAY_HARD_SYNC = 0 # Not needed, was used for deley btween  -> the resalt from Spatial Unit is redy, and AMM WRITE AMM NOP
    
    HARDWARE_NOP_REQ = 1+1  #HARDWARE SYNC additional time for RQ to be ready
    
    
    ic_splits = node['backend']['ic_splits']
    oc_splits = node['backend']['oc_splits']
    if (node['frontend']['input_folding_factor_x']>0 or node['frontend']['input_folding_factor_y']>0):
        kernel_size = node['frontend']['folded_kernel_size']
    else:
        kernel_size = node['frontend']['kernel_size']
    num_of_grids = node['backend']['grid_count']
    grid_mode = node['backend']['gridmode']
    current_op_output_tensor_shape = node['frontend']['output_tensor'].get_folded_shape()
    current_op_output_width = current_op_output_tensor_shape[3]
    grid_mode = GridConfig.H14xW16
    ic_splits = 1
    
    
    wloc_lists_temp= [[WLOCList()    for _ in range(2*ic_splits)]]
    wloc_lists     = [[WLOCList()    for _ in range(2*ic_splits)]] # create of array of WLOCList
    rq_param_lists = [[RQParamList() for _ in range(  ic_splits)]] # create of array of 2*split RQParamList. For example 
    rtable_list    = [RTableList()] 
    wloc_zero_mul_entry = WLOCEntry(weight_value = 0, weight_index = 1, weight_offset = 1, long_entry=True, nop=False)
    
    cbc_mode = False
    extra_3x3_shifts = 2 if kernel_size == 3 else 0
    #temp_oc = 0 # have to remove it

    #TODO this is temptemperal, fill the RQ and RT with nops
    rq_nop = RQParamEntry(nop_count = 0, shift_count = 0)
    rt_nop = RTEntry (nop_count = 0)
    if (cbc_mode==True):
        for i in range (1000):
            rq_param_lists[0][0].add_entry(rq_nop)
            rtable_list[0].add_entry(rt_nop)
    elif (cbc_mode==False):
        pass
   

    #TODO this is temperal
    #rewrite walues, use it to be the same langth
    longest_cmd_num = max(len(grids_wloc_commands[0]), len(grids_wloc_commands[1]))


    wloc_num = len(grids_wloc_commands)
    for i_cmd in range(longest_cmd_num):
        for i_wloc in range (wloc_num):
            if i_cmd>=len(grids_wloc_commands[i_wloc]):
                cmd_to_insert = WLOCEntry(nop=True)
            else:     
                cmd_to_insert = grids_wloc_commands[i_wloc][i_cmd]
            #insert
            wloc_lists_temp[0][i_wloc].add_entry(cmd_to_insert)    

        

  
     #---------------------------------------
     # 1) Insert NOPS in wloc so that the number of cmd will be larger then the number of shifts
    inx = 0  
    wlist = wloc_lists_temp[0][0].cmd_list  
    is_first_oc = True
    previous_oc_inx = 0 # previous channel oc
    rq_inx_last = 0
    additional_block_amm_clk = 0

    while inx < len(wlist):
        grids_wloc_commands_wloc_0 = wlist[inx]
        grids_wloc_commands_wloc_1 = wloc_lists_temp[0][1].cmd_list[inx]

        if (grids_wloc_commands_wloc_0.end_of_oc == True) ^ (grids_wloc_commands_wloc_1.end_of_oc == True):
            raise ValueError ('wloc is not balanced, check it')
                    
        # we have to jump over the first output channal
        if (grids_wloc_commands_wloc_0.end_of_oc == True) and (is_first_oc):
            is_first_oc = False

            wloc_nop_entry = WLOCEntry(nop=True, nop_reason = 'First section have to be more than num_of_AMM_write_cmd')
            num_of_nops = max (num_of_AMM_write_cmd+1 -inx,0)
            for i in range(num_of_nops):
                wloc_lists_temp[0][0].insert_entry(pos = inx,entry=deepcopy(wloc_zero_mul_entry) ) # shifts before eoc
                wloc_lists_temp[0][1].insert_entry(pos = inx,entry=deepcopy(wloc_zero_mul_entry) ) # shifts before eoc
                inx += 1

            #apdate statistics
            node['backend']['statistics']['rq_waiting_nops'] += num_of_nops

            previous_oc_inx = inx
            inx += 1
            continue


        if (grids_wloc_commands_wloc_0.end_of_oc == True):
            # 1) Insert NOPS in wloc so that the number of cmd will be larger than the number of shifts
            wloc_nop_entry = WLOCEntry(nop=True, nop_reason = 'WLOC is shorter than shifts of RQ')
            
            num_of_nops = max (HARDWARE_NOP_REQ + shift_reg_length+num_of_AMM_write_cmd+extra_3x3_shifts + EXTRA_TFLITE_SHIFTS -(inx - previous_oc_inx) ,0)
            for i in range(num_of_nops):
                wloc_lists_temp[0][0].insert_entry(pos = inx,entry=deepcopy(wloc_zero_mul_entry) ) # shifts before eoc
                wloc_lists_temp[0][1].insert_entry(pos = inx,entry=deepcopy(wloc_zero_mul_entry) ) # shifts before eoc
                inx += 1

            #apdate statistics
            node['backend']['statistics']['rq_waiting_nops'] += num_of_nops

            #update previous pointer   
            previous_oc_inx = inx
        inx += 1
    #---------------------------------------
    # 2) Insert 7+3 nops to the end, that the RQ and RT will be ready with calculation
    wloc_nop_entry = WLOCEntry(nop=True, nop_reason = 'WLOC give time for RQ')
    num_of_nops = shift_reg_length+AMM_DELAY_HARD_SYNC+2*num_of_AMM_write_cmd+extra_3x3_shifts+EXTRA_TFLITE_SHIFTS+1+RT_BEGIN_SYNC_NOP+HARDWARE_SYNC_CMD_COMPLATE -1
    inx = len(wloc_lists_temp[0][0].cmd_list)
    for i in range(num_of_nops):
            wloc_lists_temp[0][0].insert_entry(pos = inx,entry=deepcopy(wloc_zero_mul_entry) ) # shifts before eoc
            wloc_lists_temp[0][1].insert_entry(pos = inx,entry=deepcopy(wloc_zero_mul_entry) ) # shifts before eoc

               
    #-----------------------------------

    # Because the list of wloc changes its length, we have to use while instead of for
    inx = 0  
    wlist = wloc_lists_temp[0][0].cmd_list  
    is_first_oc = True
    previous_oc_inx = 0 # previous channel oc
    is_first_rq_entry = True

    while inx < len(wlist):
        grids_wloc_commands_wloc_0 = wlist[inx]
        grids_wloc_commands_wloc_1 = wloc_lists_temp[0][1].cmd_list[inx]

        if (grids_wloc_commands_wloc_0.end_of_oc == True) ^ (grids_wloc_commands_wloc_1.end_of_oc == True):
            raise ValueError ('wloc is not banaced, check it')
            
        
        # we have to jump over the first output channel
        if (grids_wloc_commands_wloc_0.end_of_oc == True) and (is_first_oc):
            is_first_oc = False
            previous_oc_inx = inx

            if   (cbc_mode==True):
                pass
            elif (cbc_mode==False):
                   rq_nop = RQParamEntry(nop_count = inx-1, shift_count = 0)
                   rq_param_lists[0][0].add_entry(rq_nop)
                   rt_nop = RTEntry(nop_count = inx-1)
                   rtable_list[0].add_entry(rt_nop)
                     
            inx += 1
            continue

        #now work as normal      
        if (grids_wloc_commands_wloc_0.end_of_oc) == True or inx ==len(wlist)-1:
            #--------------------------Algo---------------------------------
            # 1) Insert NOPS in wloc so that the number of cmd will be larger then the number of shifts (Done before)
            # 1a) Insert 7+3 nops to the end, that the RQ and RT will be ready with calculation
            # 2) Insert the config cmd to RQ
            # 3) insert shift cmd to RQ
            # 4) insert result_pipeline_reset cmd to RT
            # 5)Insert Write add to RT 
            # 6) Insert NOP-AMM write to WLOC (if there are not nops in this place)
            #############################################################


            # 2) Insert the config cmd to RQ (#TODO this have to be removed and done before)
            
            current_oc = wloc_lists_temp[0][1].cmd_list[previous_oc_inx].oc

            if not (node['frontend']['input_folding_factor_x']>0 or node['frontend']['input_folding_factor_y']>0):
                #no folding
                requant_scale_uint14, requant_bias_int12, rough_shift_uint2 = get_rq_params(node, channel=current_oc, folded=False)
            else:
                #with folding
                requant_scale_uint14, requant_bias_int12, rough_shift_uint2 = get_rq_params(node, channel=current_oc, folded=True)

            if (cbc_mode==True):
                # rq_set_command = RQParamEntry(nop=0, scale=requant_scale_uint17,
                #                                 bias= requant_bias_int12, rough_shift_sel= rough_shift_uint2)
                rq_param_lists[0][0].cmd_list[previous_oc_inx].is_config_cmd = True
                rq_param_lists[0][0].cmd_list[previous_oc_inx].nop_count = 0
                rq_param_lists[0][0].cmd_list[previous_oc_inx].shift_count = 0
                rq_param_lists[0][0].cmd_list[previous_oc_inx].scale = requant_scale_uint14
                rq_param_lists[0][0].cmd_list[previous_oc_inx].bias  = requant_bias_int12
                rq_param_lists[0][0].cmd_list[previous_oc_inx].rough_shift_sel = rough_shift_uint2
            elif (cbc_mode==False):
                #first insert nops, if needed
                if not (is_first_rq_entry): # the first and last case are exeptions
                    nops_ot_add = (previous_oc_inx- rq_inx_last) - 2
                    rq_param_lists[0][0].cmd_list[-1].nop_count = nops_ot_add

                if rq_param_lists[0][0].cmd_list[-1].is_config_cmd:
                    raise ValueError("cannot add nops to is_config_cmd")
                
                is_first_rq_entry = False
                rq_set_command = RQParamEntry(is_config_cmd=True, nop_count = 0, 
                                              scale=requant_scale_uint14, bias= requant_bias_int12, rough_shift_sel= rough_shift_uint2)
                
                rq_param_lists[0][0].add_entry(rq_set_command)

            # 3) insert shift cmd to RQ
            if (cbc_mode==True):
                for i in range (shift_reg_length):
                    rq_param_lists[0][0].cmd_list[previous_oc_inx+1+i].nop_count = 0
                    rq_param_lists[0][0].cmd_list[previous_oc_inx+1+i].shift_count = 1 #shift_reg_length
                    rq_param_lists[0][0].cmd_list[previous_oc_inx+1+i].scale = 1
                    rq_param_lists[0][0].cmd_list[previous_oc_inx+1+i].bias  = 0
                    rq_param_lists[0][0].cmd_list[previous_oc_inx+1+i].rough_shift_sel = 0

            elif (cbc_mode==False):
                rq_shift_command = RQParamEntry(is_config_cmd=False, nop_count = 0, shift_count = shift_reg_length+extra_3x3_shifts)
                rq_param_lists[0][0].add_entry(rq_shift_command)
                #rq_inx_last = previous_oc_inx
            
            # 4) insert conf cmd to RT
            if (cbc_mode==True):
                rtable_list[0].cmd_list[previous_oc_inx].result_pipeline_reset=1
            elif (cbc_mode==False):
                #insert nops
                
                 
                nops_ot_add = shift_reg_length-num_of_AMM_write_cmd+extra_3x3_shifts+2+EXTRA_TFLITE_SHIFTS
                 
                if (rtable_list[0].cmd_list[-1].AMM_write_mask>0) and not (is_first_rq_entry): # there was a write cmd, and we can add pipeline_reset to it
                 #normal case    
                    prev_channal_last_write_clk = rq_inx_last+shift_reg_length+num_of_AMM_write_cmd+extra_3x3_shifts+EXTRA_TFLITE_SHIFTS
                    
                    #check if to creat new cmd or put result_pipeline_reset to existing one
                    if prev_channal_last_write_clk==previous_oc_inx:
                    #add to the prev. one
                        rtable_list[0].cmd_list[-1].result_pipeline_reset=1
                        rtable_list[0].cmd_list[-1].nop_count = nops_ot_add
                    else: # create new cmd
                        rtable_list[0].cmd_list[-1].nop_count = (previous_oc_inx - prev_channal_last_write_clk-1)
                        rt_pipe_cmd = RTEntry(result_pipeline_reset=1, nop_count =nops_ot_add)
                        rtable_list[0].add_entry(rt_pipe_cmd)

                else:
                #strat shift 
                    rt_pipe_cmd = RTEntry(result_pipeline_reset=1, nop_count =nops_ot_add)
                    rtable_list[0].add_entry(rt_pipe_cmd)

                
                # the write is depends from x_folding after the calculation
                if not('force_folding_x' in node['frontend'])  and (not('force_unfolding_x' in node['frontend'])):
                    #normal case without folding
                    rt_write_cmd = RTEntry(AMM_write_add = current_oc, AMM_write_mask = 1, scale = 0)
                    rtable_list[0].add_entry(rt_write_cmd)
                    rt_write_cmd = RTEntry(AMM_write_add = current_oc, AMM_write_mask = 2, scale = 1)
                    rtable_list[0].add_entry(rt_write_cmd)

                elif ('force_folding_x' in node['frontend']):
                    ################### old Folding strategy#####################
                    # #folding case
                    # if 'folding_conv_' in node['name']:
                    #     folding_order = int(node['name'].split('folding_conv_')[-1])
                    # else: 
                    #     folding_order = 0
                    # target_factor = (2**folding_order)
                    # target_mask = (current_oc//target_factor % 2)+1
                    # gr_num = current_oc//(2*target_factor)
                    # sub_gr_num = current_oc%(2*target_factor)
                    # if (target_mask)==1:
                    #     #even case (0 and so on)
                    #     target_oc = gr_num*(2*target_factor)+sub_gr_num*2
                    # else:
                    #     #odd case  (1 and so on)
                    #     target_oc = gr_num*(2*target_factor)+(sub_gr_num-target_factor)*2

                    # rt_write_cmd = RTEntry(AMM_write_add = target_oc,   AMM_write_mask = target_mask, scale = 0)
                    # rtable_list[0].add_entry(rt_write_cmd)
                    # rt_write_cmd = RTEntry(AMM_write_add = target_oc+1, AMM_write_mask = target_mask, scale = 1)
                    # rtable_list[0].add_entry(rt_write_cmd)
                    ################### old Folding strategy END #####################

                    # ### NEW FOLDING 
                    # target_mask = 1
                    # target_oc   = current_oc
                    # num_of_all_out_channals = node['frontend']['output_channels']
                    # rt_write_cmd = RTEntry(AMM_write_add = target_oc,   AMM_write_mask = target_mask, scale = 0)
                    # rtable_list[0].add_entry(rt_write_cmd)
                    # rt_write_cmd = RTEntry(AMM_write_add = target_oc+num_of_all_out_channals, AMM_write_mask = target_mask, scale = 1)
                    # rtable_list[0].add_entry(rt_write_cmd)

                                        # rtable_list[0].add_entry(rt_write_cmd)
                    ################### old Folding strategy END #####################

                    ### NEW FOLDING 
                    ### FOLDING Xfold - first 
                    #TODO replace it with function calculate_AMM_write_add_for_folding_case
                    target_mask = 1
                    target_oc   = current_oc
                    num_of_all_out_channals = node['frontend']['output_channels']
                    real_chanals = num_of_all_out_channals//(2**node['frontend']['output_folding_factor_y'])
                    y_fold_group_num = current_oc//real_chanals
                    num_in_group     = current_oc%real_chanals
                    #target_oc_only for 2   = current_oc + (current_oc//real_chanals)*real_chanals
                    target_oc   = y_fold_group_num*real_chanals*2+num_in_group
                    jump_to_next_part = real_chanals

                    rt_write_cmd = RTEntry(AMM_write_add = target_oc,   AMM_write_mask = target_mask, scale = 0)
                    rtable_list[0].add_entry(rt_write_cmd)
                    rt_write_cmd = RTEntry(AMM_write_add = target_oc+num_of_all_out_channals, AMM_write_mask = target_mask, scale = 1)
                    rt_write_cmd = RTEntry(AMM_write_add = target_oc+jump_to_next_part, AMM_write_mask = target_mask, scale = 1)
                    rtable_list[0].add_entry(rt_write_cmd)
                
                # elif (node['op_type'] == 'Resize'):
                
                # elif (node['op_type'] == 'Resize'):
                #     last_channal = node['frontend']['output_channels']
                #     #Left x2
                #     rt_write_cmd = RTEntry(AMM_write_add = 2*current_oc+0, AMM_write_mask = 1, scale = 0)
                #     rtable_list[0].add_entry(rt_write_cmd)
                #     rt_write_cmd = RTEntry(AMM_write_add = 2*current_oc+1, AMM_write_mask = 1, scale = 0)
                #     rtable_list[0].add_entry(rt_write_cmd)
                #     #Right x2
                #     rt_write_cmd = RTEntry(AMM_write_add = 2*current_oc+0, AMM_write_mask = 2, scale = 1)
                #     rtable_list[0].add_entry(rt_write_cmd)
                #     rt_write_cmd = RTEntry(AMM_write_add = 2*current_oc+1, AMM_write_mask = 2, scale = 1)
                #     rtable_list[0].add_entry(rt_write_cmd)

                #     #X-resize in the additional channals Left
                #     rt_write_cmd = RTEntry(AMM_write_add = 2*last_channal+2*current_oc+0, AMM_write_mask = 1, scale = 0)
                #     rtable_list[0].add_entry(rt_write_cmd)
                #     rt_write_cmd = RTEntry(AMM_write_add = 2*last_channal+2*current_oc+1, AMM_write_mask = 1, scale = 0)
                #     rtable_list[0].add_entry(rt_write_cmd)

                #     #X-resize in the additional channals Right
                #     rt_write_cmd = RTEntry(AMM_write_add = 2*last_channal+2*current_oc+0, AMM_write_mask = 2, scale = 1)
                #     rtable_list[0].add_entry(rt_write_cmd)
                #     rt_write_cmd = RTEntry(AMM_write_add = 2*last_channal+2*current_oc+1, AMM_write_mask = 2, scale = 1)
                #     rtable_list[0].add_entry(rt_write_cmd)

                #temp_oc +=1   
                # unfolding
                elif ('force_unfolding_x' in node['frontend']):
                    num_of_all_out_channals = node['frontend']['output_channels']
                    y_num_of_folding_groups = 2**node['frontend']['output_folding_factor_y']

                    base_channels = num_of_all_out_channals//y_num_of_folding_groups
                    folding_group_num = current_oc//(2*base_channels)                    
                    in_group_num      = current_oc%(base_channels)
                    
                    target_oc   = folding_group_num*(base_channels)+in_group_num
                    rt_write_cmd = RTEntry(AMM_write_add = target_oc, AMM_write_mask = 1, scale = 0)
                    rtable_list[0].add_entry(rt_write_cmd)
                    rt_write_cmd = RTEntry(AMM_write_add = target_oc, AMM_write_mask = 2, scale = 1)
                    rtable_list[0].add_entry(rt_write_cmd)

                
            
            # 5)Insert Write add to RT
            sh = shift_reg_length+1
            if (cbc_mode==True):                
                rtable_list[0].cmd_list[previous_oc_inx+sh+0].AMM_write_add = current_oc
                rtable_list[0].cmd_list[previous_oc_inx+sh+0].AMM_write_mask = 1
                rtable_list[0].cmd_list[previous_oc_inx+sh+0].scale = 0
                rtable_list[0].cmd_list[previous_oc_inx+sh+0].nop_count = 0

                rtable_list[0].cmd_list[previous_oc_inx+sh+1].AMM_write_add = current_oc
                rtable_list[0].cmd_list[previous_oc_inx+sh+1].AMM_write_mask = 2
                rtable_list[0].cmd_list[previous_oc_inx+sh+1].scale = 1
                rtable_list[0].cmd_list[previous_oc_inx+sh+1].nop_count = 0
            
            elif (cbc_mode==False):
                if (is_first_rq_entry==False):
                    pass

            
            # 6) Insert NOP-AMM write to WLOC
            
            amm_nops_pos = previous_oc_inx+sh+AMM_DELAY_HARD_SYNC+extra_3x3_shifts  +1 + RT_BEGIN_SYNC_NOP+EXTRA_TFLITE_SHIFTS
            # wloc_nop_AMM_entry  = WLOCEntry(nop=True, nop_reason = 'AMM_SYNC') 
            wloc_nop_AMM_entry  = WLOCEntry.get_zero_mul_entry()
            wloc_nop_AMM_entry.nop_reason = 'AMM_SYNC'
            
            is_amm_inserted_count = False
            
            for i in range(num_of_AMM_write_cmd):
               #if amm_nops_pos<len(wloc_lists_temp[0][1].cmd_list): 
                if wloc_lists_temp[0][1].cmd_list[amm_nops_pos].nop == True:
                    #there is a nop here, nothing to do
                    #wloc_lists_temp[0][0].cmd_list[amm_nops_pos].nop_reason = 'AMM_SYNC and WLOC_RQ'
                    #wloc_lists_temp[0][1].cmd_list[amm_nops_pos].nop_reason = 'AMM_SYNC and WLOC_RQ'
                    wloc_lists_temp[0][0].cmd_list[amm_nops_pos] = deepcopy(wloc_zero_mul_entry)
                    wloc_lists_temp[0][1].cmd_list[amm_nops_pos] = deepcopy(wloc_zero_mul_entry)
                    
                else:
                    #insert nop
                    if (wloc_lists_temp[0][0].cmd_list[amm_nops_pos+0].shift_right or wloc_lists_temp[0][1].cmd_list[amm_nops_pos+0].shift_down or 
                        wloc_lists_temp[0][0].cmd_list[amm_nops_pos+1].shift_right or wloc_lists_temp[0][1].cmd_list[amm_nops_pos+1].shift_down #or

                        ):
                        # take the shift flags from the cmd, and move it to the new ZERO command (For sync with HW needed that shift will come always after 2 CLC)
                        
                        wloc_lists_temp[0][0].insert_entry(pos = amm_nops_pos,entry=deepcopy(wloc_zero_mul_entry) ) 
                        wloc_lists_temp[0][1].insert_entry(pos = amm_nops_pos,entry=deepcopy(wloc_zero_mul_entry) )
                        
                        wloc_lists_temp[0][1].cmd_list[amm_nops_pos].shift_down  = wloc_lists_temp[0][1].cmd_list[amm_nops_pos+1].shift_down
                        wloc_lists_temp[0][0].cmd_list[amm_nops_pos].shift_right = wloc_lists_temp[0][0].cmd_list[amm_nops_pos+1].shift_right
                        
                        wloc_lists_temp[0][1].cmd_list[amm_nops_pos].oc          = wloc_lists_temp[0][1].cmd_list[amm_nops_pos+1].oc
                        wloc_lists_temp[0][0].cmd_list[amm_nops_pos].oc          = wloc_lists_temp[0][0].cmd_list[amm_nops_pos+1].oc

                        wloc_lists_temp[0][1].cmd_list[amm_nops_pos+1].shift_down  = wloc_lists_temp[0][1].cmd_list[amm_nops_pos+2].shift_down
                        wloc_lists_temp[0][0].cmd_list[amm_nops_pos+1].shift_right = wloc_lists_temp[0][0].cmd_list[amm_nops_pos+2].shift_right
                        
                        wloc_lists_temp[0][1].cmd_list[amm_nops_pos+1].oc          = wloc_lists_temp[0][1].cmd_list[amm_nops_pos+2].oc
                        wloc_lists_temp[0][0].cmd_list[amm_nops_pos+1].oc          = wloc_lists_temp[0][0].cmd_list[amm_nops_pos+2].oc


                        wloc_lists_temp[0][1].cmd_list[amm_nops_pos+2].shift_down    = False
                        wloc_lists_temp[0][0].cmd_list[amm_nops_pos+2].shift_right   = False
    
                    else:                        
                        wloc_lists_temp[0][0].insert_entry(pos = amm_nops_pos,entry=deepcopy(wloc_nop_AMM_entry) ) 
                        wloc_lists_temp[0][1].insert_entry(pos = amm_nops_pos,entry=deepcopy(wloc_nop_AMM_entry) )
                        # move the shift flags to new entry

                    ####
    

                    # wloc_lists_temp[0][0].cmd_list[amm_nops_pos+1].set_long()
                    # wloc_lists_temp[0][1].cmd_list[amm_nops_pos+1].set_long()
                    

                    if (amm_nops_pos <= inx): 
                         inx += 1
                # if prev command is NOP, change it to zero mul
                for i_wloc in [wloc_lists_temp[0][0], wloc_lists_temp[0][1]]:
                    if (i==0) and (i_wloc.cmd_list[amm_nops_pos-1].nop):
                        i_wloc.cmd_list[amm_nops_pos-1] = wloc_zero_mul_entry
                        

                #check if the next cmd is SHORT, (jump on NOPS)
                for i_wloc in [wloc_lists_temp[0][0], wloc_lists_temp[0][1]]:
                    find_next_valid_cmd = False
                    i_pos = amm_nops_pos+1
                    while i_pos< len (i_wloc.cmd_list) and find_next_valid_cmd == False:
                        if i_wloc.cmd_list[i_pos].nop:
                            i_pos+=1
                        else:
                            i_wloc.cmd_list[i_pos].set_long()
                            find_next_valid_cmd = True    
                    
                    #amm_nops_pos +=1 #we insert the enrty, before amm_nops_pos, so we have to inc the pointer
                amm_nops_pos +=1 #this is needed for for loop

            #########pathch to fix all plases where AMM NOPS are with distance +- 1 from AMM WRITE
            for i_wloc in [wloc_lists_temp[0][0], wloc_lists_temp[0][1]]:
                #check -2 and -1
                for i_check in range(2):
                    if (i_wloc.cmd_list[amm_nops_pos-3 -i_check].long_entry == False) and  (i_wloc.cmd_list[amm_nops_pos-3 -i_check].weight_value==0) and (i_wloc.cmd_list[amm_nops_pos-3 -i_check].weight_offset==0):
                         #this is STALL NOP, have to change it to ZERO MUL
                        i_wloc.cmd_list[amm_nops_pos-3 -i_check] = deepcopy(wloc_zero_mul_entry)
                #check +1 and +2
                for i_check in range(2):
                    if (i_wloc.cmd_list[amm_nops_pos +i_check].long_entry == False) and  (i_wloc.cmd_list[amm_nops_pos +i_check].weight_value==0) and (i_wloc.cmd_list[amm_nops_pos +i_check].weight_offset==0):
                         #this is STALL NOP, have to change it to ZERO MUL
                        i_wloc.cmd_list[amm_nops_pos +i_check] = deepcopy(wloc_zero_mul_entry)
            ##########################################################################         
            
            rq_inx_last = previous_oc_inx          
            previous_oc_inx = inx 
            #### new pointers ######
            #pr_new = inx
            #pr_pr_new = pr_new      
    
        inx += 1
    ###########################################
    if (cbc_mode==True):
            pass
    elif (cbc_mode==False):
        nops_ot_add = len(wloc_lists_temp[0][1].cmd_list) -rq_inx_last -2
        rq_param_lists[0][0].cmd_list[-1].nop_count = nops_ot_add

        ### for rt
        #nops_ot_add = len(wloc_lists_temp[0][1].cmd_list) -rq_inx_last -2 - (shift_reg_length+num_of_AMM_write_cmd)+3
        
    ###########################################
    # put CMD_complete
    
    rtable_list[0].cmd_list[-1].nop_count       = HARDWARE_SYNC_CMD_COMPLATE
    cmd_com_command = RTEntry(CMD_complete = True)
    rtable_list[0].add_entry(cmd_com_command) #max (HARDWARE_SYNC_CMD_COMPLATE, RQ_BEGIN_SYNC_NOP)
    
    rq_sum_nop = RQ_BEGIN_SYNC_NOP+1
    for rq_ent in rq_param_lists[0][0].cmd_list:
        rq_sum_nop += (rq_ent.nop_count+1)
    rt_sum_nop = RT_BEGIN_SYNC_NOP+1
    for rt_ent in rtable_list[0].cmd_list:
        rt_sum_nop += (rt_ent.nop_count+1)

    ##### cut after the rest of the NOPS that are after HARDWARE_SYNC_CMD_COMPLATE
    #rq_param_lists[0][0].cmd_list[-1].nop_count -= (RQ_BEGIN_SYNC_NOP +1)
    rq_param_lists[0][0].cmd_list[-1].nop_count -= (rq_sum_nop-rt_sum_nop)
    n_last_wloc_entry_to_del = len(wloc_lists_temp[0][0].cmd_list) - rt_sum_nop
    if n_last_wloc_entry_to_del>0: # this if is the defence
        del wloc_lists_temp[0][0].cmd_list[-n_last_wloc_entry_to_del:]
        del wloc_lists_temp[0][1].cmd_list[-n_last_wloc_entry_to_del:]
    
    

    # cut the tables
    rq= [[RQParamList() for _ in range(  ic_splits)]] # create of array of 2*split RQParamList. For example 
    rt= [RTableList()]      
    for i in range(len(wloc_lists_temp[0][1].cmd_list)):
        if i<len(rq_param_lists[0][0].cmd_list):
            entry = rq_param_lists[0][0].cmd_list[i]
            if entry.nop_count < 1000:
                rq[0][0].add_entry(entry)
            else:
                rq[0][0].add_entry(entry)
                rq[0][0].cmd_list[-1].nop_count -=1000+1
                rq[0][0].add_entry(RQParamEntry(is_config_cmd=False, nop_count = 1000)) 

        if i<len(rtable_list[0].cmd_list):
            entry = rtable_list[0].cmd_list[i] 
            if entry.nop_count < 1000:
                rt[0].add_entry(entry)
            else:
                rt[0].add_entry(entry)
                rt[0].cmd_list[-1].nop_count -=1000+1
                rt[0].add_entry(RTEntry(                         nop_count = 1000))     

    
    


    # HARD WARE SYNC ------------------------------------------------------------------------
    # 9) ADD in the begining NOPS to sync with HV
    if USE_BEGIN_SYNC:
        rq[0][0].cmd_list.insert(0, RQParamEntry(is_config_cmd=False, nop_count = RQ_BEGIN_SYNC_NOP))
        rt[0].cmd_list.insert(   0, RTEntry(                          nop_count = RT_BEGIN_SYNC_NOP))

    return   wloc_lists_temp, rq, rt                           # only on RT

        
# if __name__ == "__main__":

#     generate_grids_rq_cbc_alex(1,1,1)



 
