import copy
import sys
from common.program_ir import (
    LinearFunctionList,
    NonLinearFunctionList,
    WLOCList,
    RQParamList,
    RTableList,
    WLOCEntry,
    RQParamEntry,
    RTEntry,
    create_mem_updated,
)
from common.hw_config import (
    HARDWARE_SYNC_CMD_COMPLATE,
    STALL_NOPS_FREQ,
    ALTERNATIV_STALL_SHIFT,
    MIN_NUM_CMDS_IN_FIRST_CHANNEL,
    MIN_NUM_CMDS_IN_ONE_CHANNEL,
    CLK_FROM_EOC_TO_RQ_PARAM,
    SPLIT_TALE_SIZE_IN_BYTES,
    CLK_FROM_EOC_TO_AMM_WRITE,
    START_RQ_NOPS,
    START_RT_NOPS

)
from common.debug_flags import DEBUG_OPTIMIZED_MEMORY_CREATION


class TSNP_CBC_IR:
    def __init__(self):     
        self.alex_wlocs     = [[]] # create  array of WLOCList there are two wlocs      
        self.alex_rqParam   = []   # create of array of 2*split RQParamList. For example 
        self.RTable         = []

        # atributes for cbc generation
         
        self.wlocs_one_split         = [WLOCList()    for _ in range(2)]              # create of array of WLOCList
        self.rq_one_split            = RQParamList()
        self.rt_one_split            = RTableList()
        self.wlocs_no_last_channel  = [WLOCList()    for _ in range(2)]

        # Memory state tracking for incremental split check
        self.wloc_mem_state_no_last_channel = [None, None]  # State BEFORE last channel
        self.wloc_mem_state_current = [None, None]          # State WITH last channel

        self.is_3x3_kernel            = False

        #set the possition pointers are used for the cbc generation
        self.last_write_cmd_index = START_RT_NOPS+1
        self.last_shift_cmd_index = START_RQ_NOPS+1             
        
                      
        self.cmd_in_wloc          = 0
        self.cmd_in_instraction   = 0
        self.cmd_in_oc            = 0
        self.stalls_cntr          = 0
        self.last_stall_clk       = 0
        self.act_oc_channel       = 0
    
    def tsnp_generate_nlf(self, node, ir):
        if ('lut_silu' in node['frontend']):
            nlf = [NonLinearFunctionList(node['frontend']['lut_silu'])]
        else:
            nlf = [LinearFunctionList()]
        nlf[0].create_cmd_mem()

        return nlf
        
    def tsnp_generate_grids_wloc_cbc(self, current_op_weights_tensor_list, node, oc_ch_compute_sequence):
        

        wlocs = [WLOCList() ,WLOCList()]
        wloc_shift_down_entry  = WLOCEntry(weight_value=0, weight_index=1, nop=False, shift_down=True,  long_entry=True)
        wloc_shift_right_entry = WLOCEntry(weight_value=0, weight_index=1, nop=False, shift_right=True, long_entry=True)
        wloc_nop_entry         = WLOCEntry.get_zero_mul_entry()

        wloc_nop_entry_always_long                = WLOCEntry.get_zero_mul_entry()
        wloc_nop_entry_always_long.is_always_long = True   


    
        kernel_size = current_op_weights_tensor_list.shape[2]
        
        is_3x3_conv = (kernel_size==3)
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
                wlocs[0].add_entry(wloc_nop_entry.clone())
                wlocs[1].add_entry(wloc_nop_entry.clone())
                wlocs[0].add_entry(wloc_shift_right_entry.clone())
                wlocs[1].add_entry(wloc_zero_mul_entry.clone())
                #SHD
                wlocs[0].add_entry(wloc_nop_entry.clone())
                wlocs[1].add_entry(wloc_nop_entry.clone())
                wlocs[1].add_entry(wloc_shift_down_entry.clone())
                wlocs[0].add_entry(wloc_zero_mul_entry.clone())

            #EOC
            for wloc_id in (0, 1):            
                wlocs[wloc_id].cmd_list[-1].end_of_oc = True

            #end of channel nop     
            #if the first command is W0, delete it
            if wlocs[0].cmd_list[current_oc_first_cmd].weight_value==0 and  wlocs[1].cmd_list[current_oc_first_cmd].weight_value==0:
                del wlocs[0].cmd_list[current_oc_first_cmd]
                del wlocs[1].cmd_list[current_oc_first_cmd]           

      
        return wlocs

    def generate_cbc (self, node, generated_wlocs):

        while self.cmd_in_instraction < min(len(generated_wlocs[0].cmd_list), len(generated_wlocs[1].cmd_list)):
        #this while is for the WLOC splits
        #prepare to work 
            eoc  = False
            self.is_3x3_kernel = not(node['frontend']['kernel_size']==1)


            self.add_first_cmd_to_cbc(node, generated_wlocs)   
            
            cmd_list0 = generated_wlocs[0].cmd_list
            cmd_list1 = generated_wlocs[1].cmd_list

            while self.cmd_in_instraction < min(len(cmd_list0), len(cmd_list1)):
                #this while is for the WLOC channels

                wloc0 = cmd_list0[self.cmd_in_instraction].clone()
                wloc1 = cmd_list1[self.cmd_in_instraction].clone()

                # update the current channel number
                self.act_oc_channel = wloc0.oc
                
                #set eoc flag and remove it from generated wlocs, to avoid problem of eoc in false place. We need delay at 2 clc (Itamar request)
                eoc = wloc0.end_of_oc or wloc1.end_of_oc
                wloc0.end_of_oc = False
                wloc1.end_of_oc = False
                #add the cmd to the CBC
                self.add_cmd_to_cbc([wloc0,wloc1], node)

                # add the stalls
                if ('optimal_offload_point_for_tensor' in node['backend']) or ('add_stall_nops' in node['backend'] and node['backend']['add_stall_nops'] == True): 
                    self.add_stall_nops(node)

                if eoc:
                    self.close_channel_in_cbc(node)

                    # run the SHORT optimization
                    for wloc_id in (0, 1):
                        self.wlocs_one_split[wloc_id].optimise_wloc_with_shorts()

                    # Check split using configured method (old vs optimized)
                    if DEBUG_OPTIMIZED_MEMORY_CREATION:
                        # Update memory states for incremental split check
                        self.update_memory_states_before_split_check()
                        # Use VALIDATED incremental version (O(N) complexity)
                        #is_to_split = self.is_wloc_split_needed_with_validation(node)
                        is_to_split = self.is_wloc_split_needed_incremental(node)
                    else:
                        # Use OLD method (full regeneration, O(N²) complexity)
                        is_to_split = self.is_wloc_split_needed(node)  # <- if we split WLOC the i_gen jump back to the previous channel start

                    ####################
                    if is_to_split:
                        self.act_oc_channel = wloc0.oc -1 
                        self.wlocs_one_split = copy.deepcopy(self.wlocs_no_last_channel)
                        
                        self.wlocs_no_last_channel  = [WLOCList()    for _ in range(2)]
                        
                        self.cmd_in_instraction = last_cmd_instruction_index
                        self.cmd_in_wloc = last_cmd_in_wloc
                        #rq
                        self.last_shift_cmd_index = last_last_shift_cmd_index
                        #rt
                        self.last_write_cmd_index = last_last_write_cmd_index

                        #del the last channels
                        del self.rq_one_split.cmd_list[-1]
                        del self.rq_one_split.cmd_list[-1]


                        self.close_canal_in_rq   (node)

                        ### now rt
                        del self.rt_one_split.cmd_list[-1]
                        del self.rt_one_split.cmd_list[-1]
                        del self.rt_one_split.cmd_list[-1]
                        self.close_canal_in_rt   (node)

                        # self.close_split_in_cbc  (node) this willbe done after break
                        # self.add_split_to_cbc(node) 
                        break
                    else: # no split is needed
                        self.wlocs_no_last_channel = copy.deepcopy(self.wlocs_one_split)
                        last_cmd_instruction_index = self.cmd_in_instraction+1
                        last_cmd_in_wloc = self.cmd_in_wloc
                        #rq
                        last_last_shift_cmd_index = self.last_shift_cmd_index
                        #rt
                        last_last_write_cmd_index = self.last_write_cmd_index

                        self.close_canal_in_rq   (node)
                        self.close_canal_in_rt   (node)

                ####################



                self.cmd_in_instraction += 1 
            #close last channel and return the results
            self.close_split_in_cbc(node)
            self.add_split_to_cbc(node)  

        return self



    def insert_stall_if_possible(self, pos, node):
        """
        Check whether STALL can be inserted at position and insert it.
        Returns True if insertion was performed.
        """

        # --- bounds check ---
        if pos < 2 or len(self.wlocs_one_split[0].cmd_list) <= pos + 3:
            return False

        # --- check EMPTY SELL NOP window ---
        nop_is_possible_to_put = (len(self.wlocs_one_split[0].cmd_list)                  > (pos+1)          and
                                      self.wlocs_one_split[0].cmd_list[pos  ].nop_reason== 'EMPTY SELL NOP' and
                                      self.wlocs_one_split[1].cmd_list[pos  ].nop_reason== 'EMPTY SELL NOP' and
                                      self.wlocs_one_split[0].cmd_list[pos+1].nop_reason== 'EMPTY SELL NOP' and
                                      self.wlocs_one_split[1].cmd_list[pos+1].nop_reason== 'EMPTY SELL NOP' and
                                      self.wlocs_one_split[0].cmd_list[pos-2].nop_reason!= 'AMM_SYNC'       and
                                      self.wlocs_one_split[0].cmd_list[pos-1].nop_reason!= 'AMM_SYNC'       and 
                                      self.wlocs_one_split[0].cmd_list[pos+2].nop_reason!= 'AMM_SYNC'       and
                                      self.wlocs_one_split[0].cmd_list[pos+3].nop_reason!= 'AMM_SYNC'   
                                    )
        if nop_is_possible_to_put:

            # --- insert STALL entries ---
            for k in range(2):
                self.wlocs_one_split[0].cmd_list[pos + k] = WLOCEntry.get_stall_nop_entry()
                self.wlocs_one_split[1].cmd_list[pos + k] = WLOCEntry.get_stall_nop_entry()

            # --- fix EMPTY SELL NOP after stall (Itamar request) ---
            for wloc_id in (0, 1):
                entry = self.wlocs_one_split[wloc_id].cmd_list[pos + 2]
                if (not entry.long_entry) and (entry.weight_value == 0):
                    new_entry = WLOCEntry.get_zero_mul_entry().clone()
                    new_entry.nop_reason = 'EMPTY SELL NOP'
                    self.wlocs_one_split[wloc_id].cmd_list[pos + 2] = new_entry

            node['backend']['statistics']['stall_nops'] += 2
            return True
        else: # not possible to put STALL here            
            return False

    def add_stall_nops(self, node):
        # add the NOPs to the CBC if possible
        self.stalls_cntr = self.cmd_in_wloc - self.last_stall_clk
        alternativ_shift = ALTERNATIV_STALL_SHIFT
        alternative_place = self.cmd_in_wloc - alternativ_shift

        # cannot happen, if the STALL_NOPS_FREQ is correct, but just for case someone will play with it
        if alternative_place < 0:
            return
        
        #we are in the end of the list, put nops as as new cmds
        if (self.stalls_cntr >= STALL_NOPS_FREQ) and (len(self.wlocs_one_split[0].cmd_list) <= self.cmd_in_wloc + 1):
            if (self.wlocs_one_split[0].cmd_list[-1].nop_reason != 'AMM_SYNC' and 
                self.wlocs_one_split[0].cmd_list[-2].nop_reason != 'AMM_SYNC' ):
                #Add the STALL NOPs to the end of the list            
                for i_wloc in range(2):
                    for _ in range(2):
                        self.wlocs_one_split[i_wloc].add_entry(WLOCEntry.get_stall_nop_entry())

                self.cmd_in_wloc += 2
                self.cmd_in_oc   += 2
                self.last_stall_clk = self.cmd_in_wloc
                self.stalls_cntr = 0
                node['backend']['statistics']['stall_nops'] += 2

        # --- try normal position ---
        if self.stalls_cntr >= STALL_NOPS_FREQ:
            if self.insert_stall_if_possible(self.cmd_in_wloc, node):
                self.cmd_in_wloc += 2
                self.cmd_in_oc   += 2
                self.last_stall_clk = self.cmd_in_wloc
                

        # --- try alternative position (good for identity, add, reqant and other similar nodes) ---   
        if self.stalls_cntr >= STALL_NOPS_FREQ + alternativ_shift:
            if self.insert_stall_if_possible(alternative_place, node):
                self.last_stall_clk = alternative_place
 
           

        #set the teil NOPs ###############################################
        for _ in range(HARDWARE_SYNC_CMD_COMPLATE+1):
            self.wlocs_one_split[0].add_entry(WLOCEntry.get_empty_cell_entry().clone()) 
            self.wlocs_one_split[1].add_entry(WLOCEntry.get_empty_cell_entry().clone())
        
        # set the RQ
        self.rq_one_split.add_entry(RQParamEntry(nop_count = len(self.wlocs_one_split[0].cmd_list)-self.last_shift_cmd_index-1, shift_count = 0))
        
        # set RT SYNC           
        self.rt_one_split.add_entry(RTEntry(nop_count = HARDWARE_SYNC_CMD_COMPLATE-1))
        self.rt_one_split.add_entry(RTEntry(CMD_complete = True))
        
        #additional 2 NOPS for Itamar in the end of WLOC ###################
        wloc_alined_com = WLOCEntry(weight_value = 0, weight_index = 1, weight_offset = 1, long_entry=False, nop=False)
        for _ in range(2):
            self.wlocs_one_split[0].add_entry(wloc_alined_com.clone())
            self.wlocs_one_split[1].add_entry(wloc_alined_com.clone())

        self.rq_one_split.add_entry(RQParamEntry(is_config_cmd=False, nop_count = 2))
        self.rt_one_split.add_entry(RTEntry(                          nop_count = 2))
              
    def close_channel_in_cbc(self, node, is_first_channel = False):

        wloc_nop_short_entry = WLOCEntry.get_zero_mul_entry()
        wloc_nop_short_entry.nop_reason = 'EMPTY SELL NOP'
        wloc_nop_short_entry.long_entry = False


        if is_first_channel:
            # if this is the first channel the min nop number is  
            jump_to_next_channel = max(0, MIN_NUM_CMDS_IN_FIRST_CHANNEL - self.cmd_in_oc)      
            self.cmd_in_wloc += jump_to_next_channel
            node['backend']['statistics']['rq_waiting_nops']     +=jump_to_next_channel
        else:
            # if the channel is too small we need to add NOPs to the end of it, and update the statistics
            jump_to_next_channel = max(0, MIN_NUM_CMDS_IN_ONE_CHANNEL - self.cmd_in_oc)      
            self.cmd_in_wloc += jump_to_next_channel
            node['backend']['statistics']['rq_waiting_nops']     +=jump_to_next_channel

 
        # write the nops to wloc to have the opotunity to add AMM NOP
        idx_to_add_AMM_nop = self.cmd_in_wloc + CLK_FROM_EOC_TO_AMM_WRITE + 1+ 4*self.is_3x3_kernel

        if len(self.wlocs_one_split[0].cmd_list)<(self.cmd_in_wloc+CLK_FROM_EOC_TO_AMM_WRITE):
            nops_to_add = idx_to_add_AMM_nop - len(self.wlocs_one_split[0].cmd_list)+2
            for _ in range(nops_to_add):
                self.wlocs_one_split[0].add_entry(WLOCEntry.get_empty_cell_entry()) 
                self.wlocs_one_split[1].add_entry(WLOCEntry.get_empty_cell_entry())                    

        for j in range(2): #2 AMM NOPs are the last
            for wloc_id in (0, 1):            
                self.wlocs_one_split[wloc_id].cmd_list[idx_to_add_AMM_nop+j] = WLOCEntry.get_zero_mul_entry()  
                self.wlocs_one_split[wloc_id].cmd_list[idx_to_add_AMM_nop+j].nop_reason='AMM_SYNC'
                self.wlocs_one_split[wloc_id].cmd_list[idx_to_add_AMM_nop+j].is_always_long = True
            

        #put EOC flag with two cmd delay (as Itamar requested) and put a LONG NOP before it
        for wloc_id in (0, 1):
            self.wlocs_one_split[wloc_id].cmd_list[self.cmd_in_wloc+1].set_eof()

        #set the counters for the next channel
        self.cmd_in_oc = 0    

    def is_wloc_split_needed(self, node):

        # run memory generation
        # TODO run only one channel
        for wloc_id in (0, 1):
            self.wlocs_one_split[wloc_id].create_cmd_mem(is_clal_only=True)
        # check if the channel fits to the memory (with the tile, if not generated)
        mem_to_fit = self.wlocs_one_split[0].mem_output_depth*self.wlocs_one_split[0].mem_output_width//8//2 - SPLIT_TALE_SIZE_IN_BYTES - 200
        max_len_wloc = max(len(self.wlocs_one_split[0].cmd_mem), len(self.wlocs_one_split[1].cmd_mem))

        if max_len_wloc > mem_to_fit:
            return True
        else:
            return False


    def is_wloc_split_needed_incremental(self, node):
        """
        INCREMENTAL version: adds only new channel commands to existing state

        Instead of regenerating memory from scratch, this function:
        1. Uses previous memory state (before last channel)
        2. Adds only NEW commands (last channel) incrementally
        3. Checks if size exceeds limit

        MUST return same result as is_wloc_split_needed()
        """

        # Get memory configuration
        mem_config = {
            'mem_cmd_resolution': self.wlocs_one_split[0].mem_cmd_resolution,
            'mem_input_width': self.wlocs_one_split[0].mem_input_width,
            'mem_output_width': self.wlocs_one_split[0].mem_output_width
        }

        max_len_wloc = 0

        for wloc_id in (0, 1):
            wloc = self.wlocs_one_split[wloc_id]
            wloc_no_last = self.wlocs_no_last_channel[wloc_id]

            # Find commands added in LAST channel
            num_commands_before = len(wloc_no_last.cmd_list)
            new_commands = wloc.cmd_list[num_commands_before:]

            # Get previous state (BEFORE last channel)
            previous_state = self.wloc_mem_state_no_last_channel[wloc_id]

            # Add new commands incrementally
            new_state = create_mem_updated(
                list_str_numbers=new_commands,
                previous_state=previous_state,
                finalize=True,  # Finalize to get accurate size
                **mem_config
            )

            # Get total size
            total_size = len(new_state.mem_created)
            max_len_wloc = max(max_len_wloc, total_size)

        # Same threshold check as original
        mem_to_fit = (self.wlocs_one_split[0].mem_output_depth *
                      self.wlocs_one_split[0].mem_output_width // 8 // 2 -
                      SPLIT_TALE_SIZE_IN_BYTES - 200)

        if max_len_wloc > mem_to_fit:
            return True
        else:
            return False



    def update_memory_states_before_split_check(self):
        """
        Update memory states BEFORE calling is_wloc_split_needed

        Called after optimise_wloc_with_shorts() to prepare for incremental check
        """

        mem_config = {
            'mem_cmd_resolution': self.wlocs_one_split[0].mem_cmd_resolution,
            'mem_input_width': self.wlocs_one_split[0].mem_input_width,
            'mem_output_width': self.wlocs_one_split[0].mem_output_width
        }

        # Save state BEFORE last channel (for incremental addition)
        for wloc_id in (0, 1):
            wloc_no_last = self.wlocs_no_last_channel[wloc_id]

            if len(wloc_no_last.cmd_list) > 0:
                # Generate memory state up to (but not including) last channel
                state = create_mem_updated(
                    list_str_numbers=wloc_no_last.cmd_list,
                    finalize=False,
                    **mem_config
                )
                self.wloc_mem_state_no_last_channel[wloc_id] = state
            else:
                # No previous channels yet
                self.wloc_mem_state_no_last_channel[wloc_id] = None


    def add_split_to_cbc(self, node):
        #generate memory for wlocs rq and rt
        for wloc_id in (0, 1):
            self.wlocs_one_split[wloc_id].create_cmd_mem()
        self.rq_one_split.create_cmd_mem()
        self.rt_one_split.create_cmd_mem()

        # add the split channel to the CBC IR
        self.alex_wlocs[0].append(self.wlocs_one_split)
        self.alex_rqParam.append([self.rq_one_split])
        self.RTable.append(self.rt_one_split)

        # reset the split tables for the next channel
        self.wlocs_one_split         = [WLOCList()    for _ in range(2)] 
        self.rq_one_split            = RQParamList()
        self.rt_one_split            = RTableList()
        self.wlocs_no_last_channel  = [WLOCList()    for _ in range(2)]

        # reset the counters for the next split
        self.last_write_cmd_index = START_RT_NOPS+1
        self.last_shift_cmd_index = START_RQ_NOPS+1
        self.cmd_in_wloc          = 0
        
        self.cmd_in_oc            = 0
        self.stalls_cntr          = 0
        self.last_stall_clk       = 0 

        


    def close_canal_in_rq(self, node):
        
        nop_to_wait_from_previous_shift = self.cmd_in_wloc - self.last_shift_cmd_index +1 + CLK_FROM_EOC_TO_RQ_PARAM +2*self.is_3x3_kernel 
        rq_shift_command = RQParamEntry(nop_count = 7, shift_count = 7)

        if self.is_3x3_kernel:
            #this is 3x3 kernel
            rq_shift_command = RQParamEntry(nop_count = 11, shift_count = 10)

        else:
            #this is 1x1 kernel
            rq_shift_command = RQParamEntry(nop_count = 7, shift_count = 7) 
        
        if nop_to_wait_from_previous_shift==1:
            # there no way to do only one Nop, so we write the Param
            rq_set_command   = RQParamEntry(is_config_cmd=True, scale=1, bias= 1, rough_shift_sel= 0)
            self.rq_one_split.add_entry(rq_set_command.clone())

        elif nop_to_wait_from_previous_shift>1:       
            self.rq_one_split.add_entry(RQParamEntry(is_config_cmd=False, nop_count = nop_to_wait_from_previous_shift-1, scale=0))
        elif nop_to_wait_from_previous_shift<0:
            #if the channel is EOF in less than 7 clk between EOF and RQ_param
            raise RuntimeError('Sync Problem')
        ###########DEBUGGING 3X3 KERNELS - set the first cmd in channel as NOP t

        requant_scale, requant_bias, rough_shift = get_rq_params(node, channel=self.act_oc_channel )
        rq_set_command = RQParamEntry(is_config_cmd=True, nop_count = 0, scale=requant_scale, 
                                        bias= requant_bias, rough_shift_sel= rough_shift) 
        self.rq_one_split.add_entry(rq_set_command.clone())
        self.rq_one_split.add_entry(rq_shift_command.clone())
        self.last_shift_cmd_index = self.cmd_in_wloc+rq_shift_command.nop_count + CLK_FROM_EOC_TO_RQ_PARAM + 3 +2*self.is_3x3_kernel
        
    def close_canal_in_rt(self, node):
        
        rt_nops = self.cmd_in_wloc+CLK_FROM_EOC_TO_AMM_WRITE-self.last_write_cmd_index +4*self.is_3x3_kernel 
        self.rt_one_split.add_entry(RTEntry(nop_count = rt_nops))
        ##### reset the pipe
        
        ###########DEBUGGING 3X3 KERNELS - set the first cmd in channel as NOP to check if the problem is in the first cmd of channel###########
        if self.is_3x3_kernel:
            self.rt_one_split.cmd_list[-1].nop_count = rt_nops - 14
            
            self.rt_one_split.add_entry(RTEntry(nop_count = 14-1))
            self.rt_one_split.cmd_list[-1].result_pipeline_reset = True  ##### reset the pipe
        ###########DEBUGGING 3X3 KERNELS - set the first cmd in channel as NOP to check if the problem is in the first cmd of channel###########    
            
        self.last_write_cmd_index = self.cmd_in_wloc+CLK_FROM_EOC_TO_AMM_WRITE+2+1 +4*self.is_3x3_kernel

        #generate RT WRITE COMMANDS ##########################
        if not('force_folding_x' in node['frontend'])  and (not('force_unfolding_x' in node['frontend'])):
                # no folding no unfolding
                self.rt_one_split.add_entry(RTEntry(nop_count = 0, AMM_write_add = self.act_oc_channel, AMM_write_mask = 1, scale = 0))
                self.rt_one_split.add_entry(RTEntry(nop_count = 0, AMM_write_add = self.act_oc_channel, AMM_write_mask = 2, scale = 1))

        elif ('force_folding_x' in node['frontend']):
                # folding case 
                target_mask = 1
                real_channels = node['frontend']['output_channels']//(2**node['frontend']['output_folding_factor_y'])
                y_fold_group_num = self.act_oc_channel // real_channels
                num_in_group     = self.act_oc_channel % real_channels

                # target_oc_only for 2 = current_oc + (current_oc//real_channels)*real_channels
                target_oc = y_fold_group_num * real_channels * 2 + num_in_group
                jump_to_next_part = real_channels

                amm_write_add0 = target_oc
                amm_write_add1 = target_oc + jump_to_next_part

                rt_write_cmd = RTEntry(AMM_write_add = amm_write_add0,   AMM_write_mask = target_mask, scale = 0)
                self.rt_one_split.add_entry(rt_write_cmd)
                rt_write_cmd = RTEntry(AMM_write_add = amm_write_add1, AMM_write_mask = target_mask, scale = 1)
                self.rt_one_split.add_entry(rt_write_cmd)
                
        elif ('force_unfolding_x' in node['frontend']):
                # TO DO: do as function
                num_of_all_out_channels = node['frontend']['output_channels']
                y_num_of_folding_groups = 2**node['frontend']['output_folding_factor_y']

                base_channels = num_of_all_out_channels//y_num_of_folding_groups
                folding_group_num = self.act_oc_channel//(2*base_channels)                    
                in_group_num      = self.act_oc_channel%(base_channels)
                
                target_oc   = folding_group_num*(base_channels)+in_group_num
                rt_write_cmd = RTEntry(AMM_write_add = target_oc, AMM_write_mask = 1, scale = 0)
                self.rt_one_split.add_entry(rt_write_cmd)
                rt_write_cmd = RTEntry(AMM_write_add = target_oc, AMM_write_mask = 2, scale = 1)
                self.rt_one_split.add_entry(rt_write_cmd) 

    def find_empty_cell(self):
        #check if we can write here the cmd. This sell have to be empty
        while (self.cmd_in_wloc<len(self.wlocs_one_split[0].cmd_list) and self.wlocs_one_split[0].cmd_list[self.cmd_in_wloc].nop_reason!='EMPTY SELL NOP'): 
                self.cmd_in_wloc += 1
                self.cmd_in_oc   += 1
        
        #before adding the cmd check if we have space in WLOC
        if len(self.wlocs_one_split[0].cmd_list)<(self.cmd_in_wloc+1): 
            self.wlocs_one_split[0].add_entry(WLOCEntry.get_empty_cell_entry()) 
            self.wlocs_one_split[1].add_entry(WLOCEntry.get_empty_cell_entry())

    def add_cmd_to_cbc(self, cmd_act, node):        
        
        #if this is the first cmd in channel, it has to be long nop (requirement of Itamar)
        self.find_empty_cell()
        if self.cmd_in_oc == 0:
            for wloc_id in (0, 1):
                was_eof = self.wlocs_one_split[wloc_id].cmd_list[self.cmd_in_wloc].end_of_oc
                self.wlocs_one_split[wloc_id].cmd_list[self.cmd_in_wloc] = WLOCEntry.get_zero_mul_entry()
                self.wlocs_one_split[wloc_id].cmd_list[self.cmd_in_wloc].end_of_oc = was_eof
                self.wlocs_one_split[wloc_id].cmd_list[self.cmd_in_wloc].is_always_long = True
                self.wlocs_one_split[wloc_id].cmd_list[self.cmd_in_wloc].nop_reason = 'FIRST_CMD_IN_CHANNEL'
            self.cmd_in_wloc += 1
            self.cmd_in_oc   += 1

        
        self.find_empty_cell()
        # this if is a patch to solve the case, if AMM is between the shift flags
        if (self.wlocs_one_split[0].cmd_list[self.cmd_in_wloc-1].nop_reason == 'AMM_SYNC') and (cmd_act[0].shift_right  or cmd_act[1].shift_down):
            
            for wloc_id in (0, 1):
                old_cmd = self.wlocs_one_split[wloc_id].cmd_list[self.cmd_in_wloc-3]
                self.wlocs_one_split[wloc_id].cmd_list[self.cmd_in_wloc-3] = WLOCEntry.get_zero_mul_entry()
                self.wlocs_one_split[wloc_id].cmd_list[self.cmd_in_wloc-3].nop_reason = 'NOP_BEFORE_SHIFT'
                self.wlocs_one_split[wloc_id].cmd_list[self.cmd_in_wloc] = old_cmd.clone()
            self.find_empty_cell()
        # # add the cmd to the CBC IR
        was_eoc = self.wlocs_one_split[0].cmd_list[self.cmd_in_wloc].end_of_oc         
        for wloc_id in (0, 1):
            self.wlocs_one_split[wloc_id].cmd_list[self.cmd_in_wloc] = cmd_act[wloc_id].clone()
            self.wlocs_one_split[wloc_id].cmd_list[self.cmd_in_wloc].end_of_oc = was_eoc

        # set the counters
        self.cmd_in_wloc += 1
        self.cmd_in_oc   += 1


            
    def add_first_cmd_to_cbc(self, node, generated_wlocs):

        # begin of split RQ 
        self.rq_one_split.add_entry(RQParamEntry(nop_count = (START_RQ_NOPS-self.is_3x3_kernel), shift_count = 0, is_config_cmd=False))             
        # begin of split RT NOPS 
        self.rt_one_split.add_entry(RTEntry(nop_count = (START_RT_NOPS+self.is_3x3_kernel)))  
        self.last_write_cmd_index = START_RT_NOPS+1
        self.last_shift_cmd_index = START_RQ_NOPS+1

        #for 3x3 do the jump in the pointers
        if self.is_3x3_kernel:
            self.last_write_cmd_index +=1 
            self.last_shift_cmd_index -=1 

        #write the first channel to the CBC IR
        eoc = False
        
        self.act_oc_channel = generated_wlocs[0].cmd_list[self.cmd_in_instraction].oc
        while not eoc:
            eoc = generated_wlocs[0].cmd_list[self.cmd_in_instraction].end_of_oc


            self.wlocs_one_split[0].add_entry(generated_wlocs[0].cmd_list[self.cmd_in_instraction])
            self.wlocs_one_split[1].add_entry(generated_wlocs[1].cmd_list[self.cmd_in_instraction])
            

            # del the eoc flag to avoid problem of eoc in false place. We need delay at 2 clc (Itamar request)
            for wloc_id in (0, 1):
                self.wlocs_one_split[wloc_id].cmd_list[-1].end_of_oc = False

            #set the counters for the next channel    
            self.cmd_in_instraction += 1
            self.cmd_in_wloc += 1

            # add the stalls
            if ('optimal_offload_point_for_tensor' in node['backend']) or ('add_stall_nops' in node['backend'] and node['backend']['add_stall_nops'] == True):
                self.add_stall_nops(node)


        # if the channel is too small we need to add NOPs to the end of it
        if self.cmd_in_wloc < MIN_NUM_CMDS_IN_FIRST_CHANNEL+1:
            for i in range(MIN_NUM_CMDS_IN_FIRST_CHANNEL+1 - self.cmd_in_wloc):
                self.wlocs_one_split[0].add_entry(WLOCEntry.get_empty_cell_entry())
                self.wlocs_one_split[1].add_entry(WLOCEntry.get_empty_cell_entry())
                
        # set the counters     
        self.cmd_in_oc   = self.cmd_in_wloc
       
        # close the channel
        self.close_channel_in_cbc(node, is_first_channel = True)
        self.close_canal_in_rq(node)
        self.close_canal_in_rt(node)

        
       

    def create_wloc_mem(self):
        for split_idx,wloc_split in enumerate(self.wlocs):
            for wloc_idx,wloc in enumerate(wloc_split):
                wloc.create_wloc_mem(split_idx,wloc_idx)

    def create_rqloc_mem(self):
        for split_idx,rqloc_split in enumerate(self.rqlocs):
            rqloc_split.create_cmd_mem()

    def create_rqparams_mem(self):
        self.rqparams.create_cmd_mem()
    
    def print_wloc(self, nr_wloc, until_line=None):
        if until_line==None:
            until_line = len(self.wlocs_one_split[nr_wloc].cmd_list)

        for i in range(until_line):
            print(self.wlocs_one_split[nr_wloc].cmd_list[i])
        print("-----------------------------")
        print("cmd_in_wloc: %d" % self.cmd_in_wloc)   




