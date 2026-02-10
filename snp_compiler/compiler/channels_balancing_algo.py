import common.internal_representation
from common.enums import GridConfig
from collections import OrderedDict
import math
import numpy as np
import prtpy
from operator import itemgetter
import time
from prtpy import outputtypes as out
from common.debug_flags import DEBUG_PRINT_CHANNELS_BALANCING
from common.hw_config import ADDITIONAL_CLOCKS_PER_OC_CALC, ADDITIONAL_CLOCKS_PER_OC_CALC_IN_CASE_OF_INPUT_SPLIT

def split_noncontigues_input_channels(input_channels, output_channels, w, input_filter_split):
    macs_per_input_channel = np.zeros((input_channels), dtype=np.int32)

    # prepare array which includes per input channel number of macs
    for input_channel in range(input_channels):
        macs_per_input_channel[input_channel] = np.count_nonzero(w[:,input_channel])

    #print('Splitting input channels by greedy algo,input channels size per split is equal between all splits')
    balancing = np.zeros((output_channels,input_filter_split), dtype=np.int32)

    algos = [prtpy.partitioning.greedy,prtpy.partitioning.karmarkar_karp_sy,prtpy.partitioning.multifit]
    best_sorted_ic_split_indexes = []
    eff=[]
    best_eff = 0
    # First we create input_filter_split groups of input filters that are balanced.
    # We do this with different algorithms (algos)
    # and then select the most effective splitting
    for algo in algos:
        items_dict = {}
        for idx,item in enumerate(macs_per_input_channel):
            items_dict[idx] = item
        start_time = time.time()
        sorted_split_indexes = prtpy.partition(algorithm=algo, numbins=input_filter_split, items=items_dict,outputtype=out.Partition)
        end_time = time.time()
        elapsed_time = end_time - start_time
        s=[sum(macs_per_input_channel[sorted_split_indexes[i]]) for i in range(input_filter_split)]
        approximation_ratio = max(s)/(sum(s)/input_filter_split)
        idle_clocks = sum([(max(s)-s[i]) for i in range(input_filter_split)])
        efficiency = (1-(idle_clocks/sum(macs_per_input_channel)))*100
        eff.append(efficiency)
        if efficiency>best_eff:
            best_sorted_ic_split_indexes=sorted_split_indexes.copy()
            best_eff = efficiency

    if DEBUG_PRINT_CHANNELS_BALANCING:
        print('Input channels split efficiency: %s' % (str(eff)))

    # Since the traditional algos dont force split sizes to be the same, we need to move tasks between splits until the number of tasks in all splits is equal
    macs_per_split=[]

    needed_moves = 0
    entries_per_split = input_channels // input_filter_split
    if (input_channels % input_filter_split)!=0:
        raise ValueError('number of input channels:%d cant be devided by ic split:%d' % (input_channels,input_filter_split))
    moved_entries = []
    
    for partition in best_sorted_ic_split_indexes:
        if len(partition) > entries_per_split:
            needed_moves = needed_moves + (len(partition)-entries_per_split)
            for idx in range(needed_moves):
                moved_entry_index = partition[-1]
                partition.remove(moved_entry_index)
                moved_entry_macs = macs_per_input_channel[moved_entry_index]
                moved_entries.append((moved_entry_index,moved_entry_macs))
    sorted_moved_entries = sorted(moved_entries,key=itemgetter(1))
    small_partitions = []
    for partition in range(input_filter_split):
        if len(best_sorted_ic_split_indexes[partition]) < entries_per_split:
            small_partitions.append(partition)
    current_small_partition_idx = 0
    for entry in sorted_moved_entries:
        current_queue_for_fill = small_partitions[current_small_partition_idx]
        if len(best_sorted_ic_split_indexes[current_queue_for_fill]) >= entries_per_split:
            current_small_partition_idx+=1

        best_sorted_ic_split_indexes[current_queue_for_fill].append(entry[0])
        if len(best_sorted_ic_split_indexes[current_queue_for_fill]) == entries_per_split:
            small_partitions.remove(small_partitions[current_small_partition_idx])
        else:
            current_small_partition_idx+=1
        if current_small_partition_idx >= len(small_partitions):
            current_small_partition_idx=0

    #print('After split balancing:')
    #for idx,partition in enumerate(best_sorted_ic_split_indexes):
    #    print('Partition #%d, has %d elements' % (idx,len(partition)))
    if len(small_partitions)>0:
        raise ValueError ('Error, expected no small partitions after equalization')


    for current_split in range(input_filter_split):
        total_macs_in_split = sum(macs_per_input_channel[best_sorted_ic_split_indexes[current_split]])
        macs_per_split.append(total_macs_in_split)
    #print('Macs gap between splits: %d' % (max(macs_per_split)-min(macs_per_split)))

    # We create an array called "balancing" which contains per each ouput channel and input channel split the number of clocks
    if input_filter_split>1: #We add overhead clocks which are "paid" per each oc calc
        additional_clocks_per_oc = ADDITIONAL_CLOCKS_PER_OC_CALC_IN_CASE_OF_INPUT_SPLIT
    else:
        additional_clocks_per_oc = ADDITIONAL_CLOCKS_PER_OC_CALC
        
    for current_output_channel in  range(output_channels):
        for current_split_idx in range(input_filter_split):
            if len(w.shape)==2: # In case of MATMUL
                current_split=w[current_output_channel,best_sorted_ic_split_indexes[current_split_idx]]
            else: # In case of convolution
                current_split=w[current_output_channel,best_sorted_ic_split_indexes[current_split_idx],:,:]
            clocks_per_current_oc_ic_group_calc = math.ceil(np.count_nonzero(current_split) / 2) # This is because we use 2 MACS in parallel so in each clock we calc 2 macs
            current_split_nonzero_weights = int(clocks_per_current_oc_ic_group_calc + additional_clocks_per_oc)
            balancing[current_output_channel,current_split_idx] = current_split_nonzero_weights
    
    return balancing, best_sorted_ic_split_indexes

def get_optimal_oc_split_and_order(per_oc_ic_group_macs, output_channels_splits, optimize_oc_order = True):
    macs_per_output_channel = np.sum(per_oc_ic_group_macs,axis=1)

    algos = [prtpy.partitioning.greedy,prtpy.partitioning.karmarkar_karp_sy,prtpy.partitioning.multifit]
    eff=[]
    best_eff = 0
    from prtpy import outputtypes as out
    for algo in algos:
        items_dict = {}
        for idx,item in enumerate(macs_per_output_channel):
            items_dict[idx] = item
        dp_start_time = time.time()
        dp_sorted_oc_split_indexes = prtpy.partition(algorithm=algo, numbins=output_channels_splits, items=items_dict,outputtype=out.Partition)
        if len(dp_sorted_oc_split_indexes)<output_channels_splits: # In some corner cases the job can be devided to smaller # of bins then specified (e.g. 2,1,1,1,1,1,1 can be splitted to [[2],[1,1],[1,1],[1,1],[1]] so we have 5 splits instead of 8)
            for i in range(output_channels_splits-len(dp_sorted_oc_split_indexes)):
                dp_sorted_oc_split_indexes.append([])
        dp_end_time = time.time()
        dp_elapsed_time = dp_end_time - dp_start_time
        dp_s=[sum(macs_per_output_channel[dp_sorted_oc_split_indexes[i]]) for i in range(output_channels_splits)]
        dp_approximation_ratio = max(dp_s)/(sum(dp_s)/output_channels_splits)
        dp_idle_clocks = sum([(max(dp_s)-dp_s[i]) for i in range(output_channels_splits)])
        dp_efficiency = (1-(dp_idle_clocks/sum(macs_per_output_channel)))*100
        eff.append(dp_efficiency)
        if dp_efficiency>best_eff:
            best_sorted_oc_split_indexes=dp_sorted_oc_split_indexes.copy()
    if DEBUG_PRINT_CHANNELS_BALANCING:
        print(str(eff))
    
    if not optimize_oc_order:
        per_ocsplit_oc_order = []
        for oc_split_idx in range(output_channels_splits):
            #current_output_channel_order = sorted_oc_split_indexes.partition[oc_split_idx] 
            current_output_channel_order = best_sorted_oc_split_indexes[oc_split_idx] 
            per_ocsplit_oc_order.append(current_output_channel_order)
    else:
        per_ocsplit_oc_order = []
        for oc_split_idx in range(output_channels_splits):
            #current_oc_partition =sorted_oc_split_indexes.partition[oc_split_idx] 
            current_oc_partition =best_sorted_oc_split_indexes[oc_split_idx] 
            if len(current_oc_partition) == 0:
                per_ocsplit_oc_order.append([])
                continue
            first_oc_to_process = current_oc_partition[0]
            current_output_channel_order=[first_oc_to_process]
            start_queue = per_oc_ic_group_macs[first_oc_to_process,:].copy()
            channels=[]
            for current_oc_idx in range(1,len(current_oc_partition)):
                current_oc=current_oc_partition[current_oc_idx]
                channels.append((current_oc,per_oc_ic_group_macs[current_oc,:]))
            while len(channels)>0:
                best_score=-1
                for idx,current_output_channel in enumerate(channels):
                    current_queue=start_queue+current_output_channel[1]
                    score = max(current_queue)-min(current_queue)
                    if best_score == -1:
                        best_score = score
                        best_output_channel = current_output_channel [0]
                        best_channel_index=idx
                    elif score<best_score:
                        best_score = score
                        best_output_channel = current_output_channel [0]
                        best_channel_index=idx
                #print(best_score)
                start_queue += channels[best_channel_index][1]
                current_output_channel_order.append(best_output_channel)
                del channels[best_channel_index]
            per_ocsplit_oc_order.append(current_output_channel_order)
    return per_ocsplit_oc_order

def simulate_mac_processing_shiftprocessing(node_name,node,per_ocsplit_oc_order,per_oc_ic_group_macs):
    per_layer_max_fifo_depth = 0
    output_filter_split = len(per_ocsplit_oc_order)
    input_filter_split = per_oc_ic_group_macs.shape[1]
    output_channels = per_oc_ic_group_macs.shape[0]
    total_layer_idle_clocks=0
    max_per_ocsplit_clocks = np.full(output_filter_split,0)
    total_per_ocsplit_idle_clocks = np.full(output_filter_split,0)
    for oc_split_idx in range(output_filter_split):
        per_engine_max_fifo_depth = 0
        current_split_oc_order = per_ocsplit_oc_order[oc_split_idx]
        if len(current_split_oc_order) > 0:
            first_oc_to_process = current_split_oc_order[0]
            current_split_next_finish_clock = per_oc_ic_group_macs[first_oc_to_process,:].copy()

            current_split_output_channel=np.zeros(input_filter_split, dtype=np.int32)
            current_ocsplit_finished_icgroups = np.full(input_filter_split,False)
            current_ocsplit_icgroup_idle_clocks =np.full(input_filter_split,0)
            # Assuming we have N PEs
            # PEn is on wait state if it finished processing current OC and PE(n+1) is still working on last oc (Excluding PEn=N)
            # PEn is on wait state if it finishd processing current OC and PE(n-1) still working on current channel (Excluding PE0)
            finished_last_icgroup=False
            while not finished_last_icgroup:
                masked_current_split_next_finish_clock = np.ma.array(current_split_next_finish_clock, mask = current_ocsplit_finished_icgroups)
                next_finishing_splits = np.where(masked_current_split_next_finish_clock == np.min(masked_current_split_next_finish_clock))
                current_clock = np.min(masked_current_split_next_finish_clock)
                next_clock_icgroup_oc = current_split_output_channel.copy()
                for finished_split in next_finishing_splits[0]:
                    if input_filter_split == 1: # This is special case where there is no input channel split
                        next_clock_icgroup_oc[0]+=1 # Move to next output channel
                        if next_clock_icgroup_oc[0] == len(current_split_oc_order):
                            finished_last_icgroup = True
                            max_per_ocsplit_clocks[oc_split_idx] = current_split_next_finish_clock[-1]
                            total_per_ocsplit_idle_clocks[oc_split_idx] = sum(current_ocsplit_icgroup_idle_clocks)
                            break # This is the last oc of the last PE
                        else:
                            current_split_actual_output_channel = current_split_oc_order[next_clock_icgroup_oc[0]]
                            current_split_next_finish_clock[0]+=per_oc_ic_group_macs[current_split_actual_output_channel,0]

                    elif finished_split==0: # This is the 1st PE
                        if current_split_output_channel[0]>current_split_output_channel[1]:
                            current_split_next_finish_clock[0]+=1 # Since next PE is still working on last channel we need to delay shift by additional clock
                            current_ocsplit_icgroup_idle_clocks[0]+=1
                        else:
                            next_clock_icgroup_oc[0]+=1 # Move to next output channel
                            if next_clock_icgroup_oc[0] == len(current_split_oc_order):
                                current_ocsplit_finished_icgroups[0] = True
                            else:
                                current_split_actual_output_channel = current_split_oc_order[next_clock_icgroup_oc[0]]
                                current_split_next_finish_clock[0]+=per_oc_ic_group_macs[current_split_actual_output_channel,0]
                    elif finished_split == input_filter_split-1: # This is the last PE
                        if current_split_output_channel[-1]>=current_split_output_channel[-2]:
                            current_split_next_finish_clock[-1]+=1 # Since previous PE is still working on current channel we need to delay shift by additional clock
                            current_ocsplit_icgroup_idle_clocks[-1]+=1
                        else:
                            next_clock_icgroup_oc[-1]+=1 # Move to next output channel
                            if next_clock_icgroup_oc[-1] == len(current_split_oc_order):
                                current_ocsplit_finished_icgroups[-1] = True
                                for icgroup in range(input_filter_split-1): # Add to each icgroup the time it was idle waiting for the last PE to finish
                                    current_ocsplit_icgroup_idle_clocks[icgroup]+=current_split_next_finish_clock[-1]-current_split_next_finish_clock[icgroup]
                                finished_last_icgroup = True
                                max_per_ocsplit_clocks[oc_split_idx] = current_split_next_finish_clock[-1]
                                total_per_ocsplit_idle_clocks[oc_split_idx] = sum(current_ocsplit_icgroup_idle_clocks)
                                break # This is the last oc of the last PE
                            else:
                                current_split_actual_output_channel = current_split_oc_order[next_clock_icgroup_oc[-1]]
                                current_split_next_finish_clock[-1]+=per_oc_ic_group_macs[current_split_actual_output_channel,-1]
                    else: # Any other PE
                        if current_split_output_channel[finished_split]>current_split_output_channel[finished_split+1]:
                            current_split_next_finish_clock[finished_split]+=1 # Since next PE is still working on last channel we need to delay shift by additional clock
                            current_ocsplit_icgroup_idle_clocks[finished_split]+=1
                        elif current_split_output_channel[finished_split]>=current_split_output_channel[finished_split-1]:
                            current_split_next_finish_clock[finished_split]+=1 # Since previous PE is still working on current channel we need to delay shift by additional clock
                            current_ocsplit_icgroup_idle_clocks[finished_split]+=1
                        else:
                            next_clock_icgroup_oc[finished_split]+=1 # Move to next output channel
                            if next_clock_icgroup_oc[finished_split] == len(current_split_oc_order):
                                current_ocsplit_finished_icgroups[finished_split] = True
                            else:
                                current_split_actual_output_channel = current_split_oc_order[next_clock_icgroup_oc[finished_split]]
                                current_split_next_finish_clock[finished_split]+=per_oc_ic_group_macs[current_split_actual_output_channel,finished_split]
                current_split_output_channel = next_clock_icgroup_oc
        
    max_layer_clocks = max_per_ocsplit_clocks.max()
    total_clocks = max_layer_clocks * input_filter_split * output_filter_split
    per_layer_idle_clocks = 0
    for oc_split_idx,per_ocsplit_idle_clocks in enumerate(total_per_ocsplit_idle_clocks):
        per_layer_idle_clocks += per_ocsplit_idle_clocks + (max_layer_clocks-max_per_ocsplit_clocks[oc_split_idx]) * input_filter_split
    node['backend']['total_actual_MACS'] = total_clocks
    node['backend']['idle_MACS'] = per_layer_idle_clocks
    per_layer_efficiency = 100 * (total_clocks - per_layer_idle_clocks) / total_clocks
    node['backend']['efficiency'] = per_layer_efficiency
    return total_clocks, per_layer_idle_clocks, per_layer_efficiency
