import numpy as np
from common import internal_representation
from common.enums import RqDSPCommand, GridConfig, DebugFilesFormat
from common.program_ir import NonLinearFunctionList, WLOCEntry,RTEntry,RQParamEntry
from common.hw_config import AMM_MASK_TO_GRIDS, ROUGH_SHIFT_BITS, MAX_REDUCE_BUS_WIDTH, BIAS_FRACTIONAL_BITS,\
     MAX_MAC_SHIFT, FULL_AMM_WRITE_MASK, RQ_NOPS_AT_START, RQ_NOPS_AT_START_MAXPOOL, FROM_DSP_OUT_TO_DEST_WRITE, WLOC_NOPS_BEFORE_NEW_OC,\
     MAX_GRID_COUNT, MULTIPLE_INPUT_OPS, STALL_NOPS_FREQ,\
     MULTIPLY_COMMAND_CYCLES, MULTIPLY_ADD_COMMAND_CYCLES, MAX_WLOC_128BIT_ENTRIES, SPARE_72BIT_ENTRIES_FOR_SPLIT,\
     RQ_ADDITIONAL_CLOCKS_FOR_FOLDING_CONV_1CHANNEL,RQ_ADDITIONAL_CLOCKS_FOR_FOLDING_CONV_4CHANNELS, RQ_ADDITIONAL_CLOCKS_FOR_FOLDING_CONV_2CHANNELS,\
     DUAL_NONCONTIGUOUS_ALLOCATION_OPS, URAM_BLOCK_SIZE, SINGLE_QUANT_PARAMS_OPS,BALANCED_WLOC_ENTRIES_LIMIT, MINIMAL_EOC_TO_FREEZE_DISTANCE,\
     WLOC_NOPS_BEFORE_NEW_OC_IN_CASE_OF_INPUT_SPLIT, RQ_BUSY_CLOCKS_HW_X_RESIZE, PER_TILE_CBC_OPS, WLOC_SYMETRIC_OPS, REDUCED_MAC_RESCALE_BUS_WIDTH,\
     SINGLE_GRID_OPS,LIMITED_GRIDS_OPS,MCHP_NUMERICS,WLOCS_PER_GRID,get_grids_per_line, get_virtual_grids, TFLITE_REQUANT
from common.program_ir import CBC_IR,WLOCList,RQParamList, RTableList
from common.debug_flags import DEBUG_DEEPCONV_SUPPORTED, DEBUG_FORCE_DEEP_CONV, DEBUG_PAIRING_USED, DEBUG_PRINT_CBC, DEBUG_PRINT_LAYER_EFFICIENCY, DEBUG_SKIP_CBC_GENERATION
import xlsxwriter
from collections import OrderedDict, defaultdict
import copy
import math

# CBC example can be found here: https://docs.google.com/spreadsheets/d/1v1R42zBXJAJZmyvd7aRrApjVCSgiI7c4ZvirY3nIIUE/edit?usp=sharing

from common.utils import get_tflite_rq_params
from common.hw_config import FROM_LAST_MUL_TO_DSP_OUT, FROM_LAST_MUL_TO_DSP_OUT_MAXPOOL, GRID_OC_END_TO_RQ_READ, FROM_RQ_READ_TO_RQ_START_MUL, RQ_BUSY_CLOCKS

def get_current_clock_wloc_commands(grids_wloc_commands,current_clock_idx):
    current_wloc_commmands = [] # For each clock we create a list that has current clock wloc op for all grids
    for wloc_idx,grid_wloc_command in enumerate(grids_wloc_commands):
        if wloc_idx % 2 == 1: # When generating RQLOC, we look only on even wlocs as the odd wlocs are of same grid and will have same functionality
            continue
        if current_clock_idx<len(grid_wloc_command.wloc_list):
            current_wloc_commmands.append(grid_wloc_command.wloc_list[current_clock_idx])
        else: # If specific grid has less commands we insert nop commands
            nop_wloc_command = WLOCEntry(nop = True)
            current_wloc_commmands.append(nop_wloc_command)
    return current_wloc_commmands

def insert_nop_in_grid_if_needed(grids_wloc_commands,grids_in_group,current_clock_idx,end_of_oc = False,used_for_dma = False):
    for grid in grids_in_group:
        if current_clock_idx>(len(grids_wloc_commands[grid*2].wloc_list)-1):
            continue
#        if used_for_dma:
#            #avoid putting a DMA NOP 0-1 clocks away from an existing NOP
#            if current_clock_idx>(len(grids_wloc_commands[grid*2].wloc_list)-3):
#                continue
#            if grids_wloc_commands[grid*2].wloc_list[current_clock_idx-1].nop or \
#                grids_wloc_commands[grid*2].wloc_list[current_clock_idx].nop or \
#                grids_wloc_commands[grid*2].wloc_list[current_clock_idx+1].nop or \
#                grids_wloc_commands[grid*2].wloc_list[current_clock_idx+2].nop:
#                continue
        current_even_wloc_command = grids_wloc_commands[grid*2].wloc_list[current_clock_idx]
        if not(isinstance(current_even_wloc_command,FirstWLOCEntry) or current_even_wloc_command.nop):
            nop_wloc_command = WLOCEntry(nop = True, end_of_oc= end_of_oc)
            grids_wloc_commands[grid*2].insert_entry(current_clock_idx,nop_wloc_command)
            grids_wloc_commands[grid*2+1].insert_entry(current_clock_idx,nop_wloc_command)

def all_nops(current_wloc_commmands):
    for command in current_wloc_commmands:
        if isinstance(command,FirstWLOCEntry) or not command.nop:
            return False
    return True

def get_per_grid_ic_split(grid_mode,ic_splits,num_grids):
    per_grid_ic_split = []
    for grid_idx in range(num_grids):
        if grid_mode==GridConfig.H14xW32:
            current_split = 0
        elif grid_mode==GridConfig.H14xW16:
            if ic_splits==1:
                current_split = 0
            elif ic_splits==2:
                current_split = (grid_idx & 4)>>2 
            else:
                raise ValueError ('Grid mode H12x16 doesnt support ic split>2')
        elif grid_mode==GridConfig.H14xW8:
            if ic_splits==1:
                current_split = 0
            elif ic_splits==2:
                current_split = (grid_idx & 2)>>1
            elif ic_splits==4:
                current_split = (grid_idx & 6)>>1
            else:
                raise ValueError ('Grid mode H12x16 doesnt support ic split>2')
        per_grid_ic_split.append(current_split)
    return per_grid_ic_split

def get_ic_split_grid_groups(grid_mode,ic_splits,num_grids):
    ic_split_grid_groups = []
    if ic_splits==1:
        for i in range(num_grids):
            ic_split_grid_groups.append([i])
    elif ic_splits==2:
        if grid_mode == GridConfig.H14xW8:
            for i in range(num_grids // 2):
                current_group = [0,1]
                ic_split_grid_groups.append(current_group)
        else:
            raise ValueError('Not supported')
    else:
        raise ValueError('Not supported')
    return ic_split_grid_groups

def get_grids_in_virtual_grid(node,num_grids):
    grids_in_virtual_grid = []
    gridmode_virtual_grids = get_virtual_grids(node,num_grids)
    for grid_idx in range(num_grids):
        grids_in_group = []
        for virtual_grid in gridmode_virtual_grids:
            if grid_idx in virtual_grid:
                grids_in_group = virtual_grid
                break
        grids_in_virtual_grid.append(grids_in_group)
    return grids_in_virtual_grid

class GridsStateMachine:
    # Grids state is according to following details:
    # 0) grid is calculating/idle. Its result register is empty
    # 1) result register is full
    # 2) result register started handling by rq can be free after RQ_READ_CLOCKS+ic_group_size-1
    def __init__(self,num_grids,node,ic_splits,oc_splits,per_oc_non_empty_ic_groups,node_op_type):
        grid_mode = node['backend']['gridmode']
        self.grids_state = [0 for i in range(num_grids)]
        self.registered_oc = [-1 for i in range(num_grids)]
        self.per_grid_ic_split = get_per_grid_ic_split(grid_mode,ic_splits,num_grids)
        self.ic_split_grid_groups = get_ic_split_grid_groups(grid_mode,ic_splits,num_grids)
        # self.non_empty_ic_groups holds for each of the current grids its group of grids that need to eoc for completing the entire oc
        self.non_empty_ic_groups = [[] for i in range(num_grids)] # TODO Dans, currently not used. check if can be removed
        self.ic_splits = ic_splits
        self.oc_splits = oc_splits
        self.per_oc_non_empty_ic_groups = per_oc_non_empty_ic_groups
        self.num_grids = num_grids
        self.grid_mode = grid_mode
        self.grids_in_virtual_grid = get_grids_in_virtual_grid(node,num_grids)
        self.gridmode_virtual_grids_groups = get_virtual_grids(node,num_grids)
        self.node_op_type = node_op_type
        self.all_grids_mask = [i for i in range(MAX_GRID_COUNT)]

    def update(self,current_wloc_commands, grids_wloc_commands,current_clock_idx):
        inserted_nops=0
        for gridmode_virtual_grids_group in self.gridmode_virtual_grids_groups:
            current_grids_group_is_oc_end = [current_wloc_commands[grid_idx].is_oc_end() for grid_idx in gridmode_virtual_grids_group] # We expect all grids in group to have oc end at same clock
            if all(current_grids_group_is_oc_end):
                if all([self.grids_state[grid_idx]==0 for grid_idx in gridmode_virtual_grids_group]):
                    output_channel = None

                    for grid_idx in gridmode_virtual_grids_group:
                        current_wloc = current_wloc_commands[grid_idx]
                        self.grids_state[grid_idx] = 1
                        if output_channel==None:
                            output_channel = current_wloc.oc
                        else:
                            if current_wloc.oc!=output_channel:
                                raise ValueError ('Expected all grids in virtual grid to handle same output channel. Please check...')

                        self.registered_oc[grid_idx] = current_wloc.oc
                        self.non_empty_ic_groups[grid_idx] = self.per_oc_non_empty_ic_groups[current_wloc.oc]
                else: # If not all grids in virtual grids reached EOC we delay the ones in the group that did so they all finish together
                    insert_nop_in_grid_if_needed(grids_wloc_commands,gridmode_virtual_grids_group,current_clock_idx) # We insert NOP in specific grid if we finish new channel calcs its output register wasnt read yet by RQ
                    inserted_nops+=len(gridmode_virtual_grids_group)
            #elif any(current_grids_group_is_oc_end):
            #    raise ValueError ('We expect all grids in the virtual grid group to have oc end at same clock. something went wrong...')
            for grid_idx in gridmode_virtual_grids_group:
                ic_group_size = len(set(self.non_empty_ic_groups[grid_idx])) # The non_empty_ic_groups includes per each grid that handles this oc which ic_group it belongs to
                ic_group_size = len(self.non_empty_ic_groups[grid_idx]) # The non_empty_ic_groups includes per each grid that handles this oc which ic_group it belongs to
                if ic_group_size>0:
                    if self.grids_state[grid_idx] == FROM_RQ_READ_TO_RQ_START_MUL+(ic_group_size-1)*MULTIPLY_ADD_COMMAND_CYCLES+MULTIPLY_COMMAND_CYCLES: # Added the (self.ic_splits-1) since when ic_splits==2 we needed extra clock to hold grids machine in busy state
                        self.grids_state[grid_idx] = 0
                    if self.grids_state[grid_idx] > 1: # This means that this specific grid started handling by rq machine
                        self.grids_state[grid_idx] += 1
        return inserted_nops
    def get_tsnp_current_clock_finished_grids(self,current_wloc_commands):
        current_finished_oc_grids = []
        for ic_split_grid_group in self.ic_split_grid_groups:
            oc_handled_by_ic_group = []
            ic_group_per_oc_dict = {}
            for grid_idx in ic_split_grid_group:
                handled_oc = self.registered_oc[grid_idx]
                if handled_oc!=-1: # -1 means this grid is still not handling any oc
                    oc_handled_by_ic_group.append(handled_oc)
                    if handled_oc in ic_group_per_oc_dict:
                        ic_group_per_oc_dict[handled_oc].append(grid_idx)
                    else:
                        ic_group_per_oc_dict[handled_oc] = [grid_idx]
            oc_handled_by_ic_group = set(oc_handled_by_ic_group)
            for current_oc in oc_handled_by_ic_group:
                expected_grids_for_current_oc = len(set(self.per_oc_non_empty_ic_groups[current_oc])) # This is the expected grids that will work on that oc. It takes into account that an ic split might be empty and hence we dont need to wait for it
                grids_handling_current_oc = ic_group_per_oc_dict[current_oc]
                if len(grids_handling_current_oc)!=expected_grids_for_current_oc:
                    continue
                current_split_finished = True
                current_split_finished_this_clock = False
                # Notes:
                # 1) It is not guranteed that ic_split_grid_group is working on same OC (if some of ic groups are empty,no weights,)
                # 2) In some cases, a specific ic split is all zeroes weights. in such case this split will not be calculated and we shouldnt wait for it
                num_finished_grids = 0
                finished_grids_list = []
                for current_grid_idx in grids_handling_current_oc:
                    if (self.grids_state[current_grid_idx] == 1 or current_wloc_commands[current_grid_idx].is_oc_end()):
                        num_finished_grids+=1
                        finished_grids_list.append(current_grid_idx)
                    if current_wloc_commands[current_grid_idx].is_oc_end():
                        current_split_finished_this_clock = True
                current_split_finished = (num_finished_grids == expected_grids_for_current_oc)
                if current_split_finished and current_split_finished_this_clock:
                    current_finished_oc_grids.append([finished_grids_list,current_oc])
        return current_finished_oc_grids

class RqStateMachine:
    def __init__(self, grids = [], write_mask = FULL_AMM_WRITE_MASK,is_folding_conv_last_write = False,grids_to_fold = 0, is_hw_x_resize = False):
        self.RQ_BUSY = True
        self.current_clock = 0
        self.grids=grids
        self.write_mask = write_mask
        self.grids_to_fold = grids_to_fold
        if is_folding_conv_last_write:
            if self.grids_to_fold == 4:
                self.rq_busy_clocks = RQ_BUSY_CLOCKS + RQ_ADDITIONAL_CLOCKS_FOR_FOLDING_CONV_4CHANNELS
            elif self.grids_to_fold == 2:
                self.rq_busy_clocks = RQ_BUSY_CLOCKS + RQ_ADDITIONAL_CLOCKS_FOR_FOLDING_CONV_2CHANNELS
            elif self.grids_to_fold == 1:
                self.rq_busy_clocks = RQ_BUSY_CLOCKS + RQ_ADDITIONAL_CLOCKS_FOR_FOLDING_CONV_1CHANNEL
            else:
                raise ValueError('at folding conv. illegal channels to fold number (should be 1 or 4): %d' % self.grids_to_fold)
        else:
            self.rq_busy_clocks = RQ_BUSY_CLOCKS
            if len(self.grids)==1:
                self.rq_busy_clocks-=1 # When handling only single grid (no ic_split) multiply commands can come back to back
        if is_hw_x_resize:
            self.rq_busy_clocks = RQ_BUSY_CLOCKS_HW_X_RESIZE
    
    def clock_tick(self):
        
        if self.current_clock == self.rq_busy_clocks+(len(self.grids)-1)*MULTIPLY_ADD_COMMAND_CYCLES: # This is dependant on len(self.grids)
            self.RQ_BUSY=False
            self.current_clock = 0
        else:
            self.current_clock+=1

    def is_busy(self):
        return self.RQ_BUSY
    def start(self):
        self.RQ_BUSY = True

class ChannelsHandlingStateMachine:
    def __init__(self,grids_per_channel=4):
        self.handled_grids = 0
        self.handled_channel = 0
        self.grids_per_channel = grids_per_channel
    
    def additional_grid_handled(self):
        self.handled_grids+=1
        if self.handled_grids==self.grids_per_channel:
            self.handled_grids = 0

    def is_channel_in_process(self):
        if self.handled_grids == 0:
            return False
        else:
            return True

    def reset(self,handled_channel = 0):
        self.handled_grids = 0
        self.handled_channel = handled_channel

class FoldingStateMachine:
    def __init__(self,grids_to_fold=4):
        self.state_clock = 0
        self.per_grid_amm_write_mask = []
        self.grids_to_fold = grids_to_fold
    
    def clock_tick(self,amm_write_mask):
        self.per_grid_amm_write_mask.append(amm_write_mask)
        self.state_clock+=1
        if self.state_clock>self.grids_to_fold:
            raise ValueError ('Folding State Machine in unknown state')

    def all_buffers_loaded(self):
        if self.state_clock == self.grids_to_fold:
            return True
        else:
            return False

    def reset(self):
        self.state_clock = 0
        self.per_grid_amm_write_mask = []
def get_folding_write_mask_from_grid_idx(current_grid_idx):
     # In AMD We want to write to buffer 0 in case of grids 0,1,4,5 and to buffer 1 if grid is 2,3,6,7 In MCHP grid 0 write to buffer 0 and grid 1 writes to buffer 1
     # In AMD buffer_id = (current_grid_idx&0x2)>>1
    buffer_id = current_grid_idx
    return buffer_id

def get_finished_oc_splits(current_wloc_commmands, ic_splits = 0, oc_splits = 0):
    current_finished_oc_grids = []
    for split in range(oc_splits):
        current_split_finished = True
        for i in range(ic_splits):
            current_grid = i+split*ic_splits
            if not current_wloc_commmands[current_grid].is_oc_end():
                current_split_finished = False
        current_finished_oc_grids.append(current_split_finished)
    return current_finished_oc_grids

def get_mac_shifts_from_frontend(scale_shift):
    # scale shift is the number of bits we shift left the float scale in order to get int scale (The uint10 scale)
    # So after we multiply mac output by integer scale we need to shift it back (right) to get to the actual result
    # Assuming its X bits, we split this shift between 2 places:
    # 1) 1st shift is done to the MAC output so that it will not exceed 14 bits.
    # 2) the rest is shifted after the requant.
    # Here we handle the 1st shift and we do it by shifting right all weight inputs to the MAC and rough shifting right activation inputs to the MAC
    # In order to turn left shift (which we do by weights and activations left shift) to right shift we actually take higher bits from the output.
    # We know the result should be 14 bits. maximum left shift is 19 bits. so we always take result between bits 32-19. If we want 0 shift right we shift left 19 bits.
    # if we want 1 shift right we shift left 18 bits and so on
    mac_shift_right = scale_shift - MAX_REDUCE_BUS_WIDTH + 3 # This is amount of bits we need to shift right in mac (This is the same 3 from neuronix_ops.py we added for 1 sign bit and 2 bits to allow extra bits for out of scale numerics)
    shift_left = MAX_MAC_SHIFT - mac_shift_right # Since we always take from mac bits 32-19 we achive shift right by shifting input to the left
    weight_shift = shift_left % ROUGH_SHIFT_BITS
    activation_shift = 2 if (shift_left // ROUGH_SHIFT_BITS == 0) else 1
    return weight_shift,activation_shift

def unify_splitted_dual_input_ic_groups(ic_groups,second_input_channel_offset):
    unified_ic_groups = copy.deepcopy(ic_groups[0])
    for group_idx,ic_group in enumerate(unified_ic_groups):
        ic_group.extend((np.array(ic_groups[1][group_idx])+second_input_channel_offset).tolist())
    return unified_ic_groups
def get_nodes_real_ic_groups(node):
    ic_groups = node['backend']['ic_groups']
    unified_ic_groups = copy.deepcopy(ic_groups)
    if node['op_type'] in MULTIPLE_INPUT_OPS:
        original_node_input_channels = node['frontend']['input_tensors'][0].get_folded_shape()[1]
        unified_ic_groups = unify_splitted_dual_input_ic_groups(ic_groups,original_node_input_channels)
    return unified_ic_groups

def unify_splitted_dual_input_noncontiguous_ic_lookup_dicts(ic_lookup_dicts,second_input_channel_offset):
    unified_ic_groups = ic_lookup_dicts[0]
    for group_idx,ic_group in enumerate(unified_ic_groups):
        for key,value in ic_lookup_dicts[1][group_idx].items():
            unified_ic_groups[group_idx][key+second_input_channel_offset] = value+second_input_channel_offset
    return unified_ic_groups

def get_nodes_real_ic_lookup_dicts(node):
    ic_lookup_dicts = copy.deepcopy(node['backend']['ic_lookup_dicts'])
    if node['op_type'] in DUAL_NONCONTIGUOUS_ALLOCATION_OPS:
        original_node_input_channels = node['frontend']['input_tensors'][0].get_folded_shape()[1]
        ic_lookup_dicts = unify_splitted_dual_input_noncontiguous_ic_lookup_dicts(ic_lookup_dicts,original_node_input_channels)
    return ic_lookup_dicts


def get_wloc_128bitentries_stats(per_grid_wloc_list):
    max_wloc_size = 0
    min_wloc_size = 1000000
    for wloc in per_grid_wloc_list:
        if wloc.get_size()>max_wloc_size:
            max_wloc_size = wloc.get_size()
        if wloc.get_size()<min_wloc_size:
            min_wloc_size = wloc.get_size()
    wloc_size_diff = max_wloc_size-min_wloc_size
    return max_wloc_size,wloc_size_diff

def get_current_clock_max_wloc_size(per_grid_wloc_list,clock):
    max_wloc_size = 0
    min_wloc_size = 1000000
    for wloc in per_grid_wloc_list:
        if wloc.get_size_at_clock(clock)>max_wloc_size:
            max_wloc_size = wloc.get_size_at_clock(clock)
        if wloc.get_size_at_clock(clock)<min_wloc_size:
            min_wloc_size = wloc.get_size_at_clock(clock)
    wloc_size_diff = max_wloc_size-min_wloc_size
    return max_wloc_size,wloc_size_diff
    
def get_wloc_clocks_stats(current_wlocs):
    max_wloc_size = 0
    min_wloc_size = 1000000
    for wloc in current_wlocs:
        if len(wloc.wloc_list)>max_wloc_size:
            max_wloc_size = len(wloc.wloc_list)
        if len(wloc.wloc_list)<min_wloc_size:
            min_wloc_size = len(wloc.wloc_list)
    wloc_size_diff = max_wloc_size-min_wloc_size
    return max_wloc_size,wloc_size_diff
def get_max_list_length(listoflists):
    max_length=0
    for current_list in listoflists:
        if len(current_list)>max_length:
            max_length = len(current_list)
    return max_length
def get_per_ic_group_sorted_weight_activation_pairs(ic_lookup_dicts):
    sorted_pairs_per_group = []
    for dict in ic_lookup_dicts:
        sorted_pairs = sorted(dict.items(),key=lambda x:x[1])
        sorted_pairs_per_group.append(sorted_pairs)
    return sorted_pairs_per_group
def get_current_requant_scale_shift(node,real_oc):
    if (node['frontend']['input_folding_factor_x']>0 or node['frontend']['input_folding_factor_y']>0):
        requant_scale_shift = node['frontend']['folded_requant_scale_shift']
    else:
        requant_scale_shift = node['frontend']['requant_scale_shift']
    if node['op_type'] in SINGLE_QUANT_PARAMS_OPS:
        current_requant_scale_shift = requant_scale_shift
    else:
        current_requant_scale_shift = requant_scale_shift[real_oc]
    return current_requant_scale_shift

def create_oc_groups_for_add_inline_command(input0_lookup_dict,oc_splits):
    oc_groups=[[] for i in range(oc_splits)]
    if len(input0_lookup_dict)>1:
        raise ValueError ('IC split input to add op not supported yet.')
    for idx,oc in enumerate(input0_lookup_dict[0].keys()):
        oc_groups[idx % oc_splits].append(oc)
    return oc_groups
def insert_wloc_initial_nops(grids_cbc,current_grid_idx,deepconv,is_ic_split):
    if is_ic_split:
        nops_to_add = WLOC_NOPS_BEFORE_NEW_OC_IN_CASE_OF_INPUT_SPLIT
    else:
        nops_to_add = WLOC_NOPS_BEFORE_NEW_OC

    for i in range(nops_to_add):
        # TODO Alex
        # del all deepconv, it's out of usage
        #grids_cbc.wlocs[0][current_grid_idx].add_entry(WLOCEntry(nop = True,deep_conv=deepconv))
        grids_cbc.wlocs[0][current_grid_idx].add_entry(WLOCEntry(nop = True))
    return nops_to_add

def generate_empty_grids_wloc_cbc_for_debug(node):
    num_of_grids = node['backend']['grid_count']
    grid_mode = node['backend']['gridmode']
    grids_cbc = CBC_IR(num_of_grids)
    ic_splits = node['backend']['ic_splits']
    oc_splits = node['backend']['oc_splits']
    per_grid_macs = [[[] for i in range(ic_splits)] for j in range(oc_splits)]
    node['backend']['per_grid_macs'] = per_grid_macs
    ic_groups = get_nodes_real_ic_groups(node) # This creates a unified ic group in cases of 2 input nodes (e.g. Add) which are converted to conv
    if len(ic_groups) != ic_splits:
        raise ValueError('Something went wrong, ic groups number != ic splits')
    op_type = node['op_type']
    if op_type in DUAL_NONCONTIGUOUS_ALLOCATION_OPS:
        oc_groups = create_oc_groups_for_add_inline_command(node['backend']['ic_lookup_dicts'][0],node['backend']['oc_splits'])
    else:
        oc_groups = node['backend']['oc_groups']
    if len(oc_groups) != oc_splits:
        raise ValueError('Something went wrong, oc groups number != oc splits')
    if ic_splits>1:
        is_ic_split=True
    else:
        is_ic_split=False
    current_op_input_channels = node['backend']['input_channels']
    input_channels_per_grid = current_op_input_channels // ic_splits
    current_op_output_channels = node['backend']['output_channels']
    output_channels_per_grid = current_op_output_channels // oc_splits
    non_empty_output_channels = {}
    grids_first_wloc_in_layer = [True for i in range(num_of_grids)]
    grids_cbc.wlocs = [[WLOCList() for i in range(num_of_grids*2)]] # We actually have 2 WLOCS per each grid
    deepconv = node['backend']['deepconv']
    for current_grid_idx in range(num_of_grids):
        if grid_mode == GridConfig.H14xW32:
            current_oc=current_grid_idx & 0x1
        elif grid_mode == GridConfig.H14xW16:
            current_oc=(current_grid_idx & 0x1) | ((current_grid_idx & 0x4)>>1)
        elif grid_mode == GridConfig.H14xW8:
            current_oc = current_grid_idx
        else:
            raise ValueError ('Grid mode not supported: %s' % (str(grid_mode)))
        # grids_cbc.wlocs[0][current_grid_idx*2].add_entry(FirstWLOCEntry(weight_shift=0,activation_shift=0, oc = 0 ))
        # grids_cbc.wlocs[0][current_grid_idx*2+1].add_entry(FirstWLOCEntry(weight_shift=0,activation_shift=0, oc = 0 ))
        insert_wloc_initial_nops(grids_cbc,current_grid_idx,deepconv,is_ic_split)
        wloc_entry = (WLOCEntry(weight_value = 1, weight_index = 0, weight_offset = 0,
                        use_left_pixel=False, use_right_pixel=False,use_top_pixel=False,use_bottom_pixel=False,
                        end_of_oc=True,long_entry=True, deep_conv = deepconv, ic=0,oc = current_oc))
        grids_cbc.wlocs[0][current_grid_idx*2].add_entry(wloc_entry)
        grids_cbc.wlocs[0][current_grid_idx*2+1].add_entry(wloc_entry)
        grids_cbc.per_oc_non_empty_ic_groups[current_oc] = [0]
        non_empty_output_channels[current_oc] = True
    node['backend']['non_empty_output_channels'] = non_empty_output_channels
    node['backend']['total_minimal_oc_clocks_inserted_nops'] = 12
    node['backend']['total_first_wloc_entries'] = 4

    return grids_cbc

def find_pairs_and_nonpairs(vec, check_for_negativ_pair=False, elem_even_info=None):
    """
    Finds matching pairs and non-paired elements in a list.

    A pair is only valid if the first element has an even index and the second has an odd index,
    according to the provided elem_odd_info list.

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


def convert_kernel_to_wlocs(kernel_1x1, kernel_1x1_no_zeros_inx,wloc_entrys, oc):
        list_half_len = -(-len(kernel_1x1_no_zeros_inx)//2)
            #normal case       
        for ihalf in range(2):
            kernel_1x1_no_zeros_inx_half = kernel_1x1_no_zeros_inx[ihalf * list_half_len : (ihalf + 1) * list_half_len]
            previous_weight_index = 1e57 # big number
            for weight_index_tuple in kernel_1x1_no_zeros_inx_half:
                weight_index = weight_index_tuple[0]
                current_weight_value = kernel_1x1[weight_index]
                distance_from_previous_activation = max(weight_index - previous_weight_index,0)
                if len(weight_index_tuple)==1:
                    #no pairing    
                    wloc_entry = (WLOCEntry(weight_value = current_weight_value, weight_index = weight_index, weight_offset = distance_from_previous_activation,                                            
                                        long_entry=True, ic=weight_index, oc = oc))
                else:
                    #pairing
                    pair_add = weight_index_tuple[1]
                    sub_sel_preadder_minus_mode = (kernel_1x1[weight_index]==-kernel_1x1[pair_add])
                    wloc_entry = (WLOCEntry(weight_value = current_weight_value, weight_index = weight_index, weight_offset = distance_from_previous_activation, is_pair = True,  pair_add = pair_add, sub_sel_preadder_minus_mode = sub_sel_preadder_minus_mode,                                           
                                        long_entry=True, ic=weight_index, oc = oc))

                wloc_entrys[ihalf].add_entry(wloc_entry)
                previous_weight_index = weight_index

        #if len(wloc0) != len(wloc1) write one additional wloc_zero_mul_entry 
        if len(kernel_1x1_no_zeros_inx) % 2 != 0:
            wloc_entrys[1].add_entry(WLOCEntry.get_zero_mul_entry())

def convert_maxpool_to_wloc(kernel_1x1, kernel_1x1_no_zeros_inx,wloc_entrys, oc, first_calc=False):
    previous_weight_index = 1e57
    for i, weight_index_tuple in enumerate (kernel_1x1_no_zeros_inx): 
        weight_index = weight_index_tuple[0]
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
        
        #### if the WLOC have only one channal, we need to insert NOP (HW requirements)
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


    #this check will do the compiler slower, but this is needed to be sure, the pairing is correct
    #check if there are some pairs that was not found by algo
    vec = kernel_1x1
    odd_inx = np.where(              (np.arange(len(vec))%2))[0]
    evn_inx = np.where(np.logical_not(np.arange(len(vec))%2))[0]
    e_no_pair_inx = np.intersect1d(non_pairs_inx, evn_inx)
    o_no_pair_inx = np.intersect1d(non_pairs_inx, odd_inx) 
    if np.intersect1d(abs(vec[o_no_pair_inx]), abs(vec[e_no_pair_inx])):
        raise ValueError("There are more pairs to find here, check the pairing")

    return pos_pairs, neg_pairs, non_pairs_inx

def generate_1x1_conv_alex(kernel_1x1, oc, node, first_calc=False):

    # If we have odd number of macs we need to add a zero mult to the odd grid so that even and odd grids have same number of clocks
    # In this case we take weight=0 and index=1 since a_shift and w_shift already set and we cant use them to zero result. 
    # We dont want to take nop since we cant mark EOC only on long commands and NOP is a short one
       

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

                
                convert_kernel_to_wlocs(kernel_1x1, pos_pairs, wloc_entrys, oc) 
                convert_kernel_to_wlocs(kernel_1x1, neg_pairs, wloc_entrys, oc)
                non_pairs_tupl = [(item,) for item in non_pairs_inx] 

                #update the pairing statistics
                node['backend']['statistics']['wloc_pairs'] +=(len(pos_pairs)+len(neg_pairs)) 

            convert_kernel_to_wlocs(kernel_1x1, non_pairs_tupl,wloc_entrys,oc)

        elif (node['op_type'] == "MaxPool"):
            #this is MAX_Pool 
            convert_maxpool_to_wloc(kernel_1x1, non_pairs_tupl,wloc_entrys,oc, first_calc=first_calc)
            
    # if the wloc is too short, shorter than 2, add one more entry at begining This needs for deley in HW (SHD, SHR, EOC)   
    if len(wloc_entrys[1].cmd_list)==1:
        wloc_entrys[0].insert_entry(0,WLOCEntry.get_zero_mul_entry())
        wloc_entrys[1].insert_entry(0,WLOCEntry.get_zero_mul_entry())
        

    return wloc_entrys        

 
def set_padding_flags (k_x, k_y, wloc_cmd_0, wloc_cmd_1):
    # do the down shift
    # wloc_cmd_1.shift_down = True
    
    if k_x==0:
        #padding last line
        wloc_cmd_1.pad_line_last = True
        wloc_cmd_1.long_entry = True
    elif k_x==2:
        #padding first line
        wloc_cmd_0.pad_line_first = True
        wloc_cmd_0.long_entry = True       

    if k_y==0:
        #padding last colomn
        wloc_cmd_1.pad_column_last = True
        wloc_cmd_1.long_entry = True
    elif k_y==2:
        #padding first colomn
        wloc_cmd_0.pad_column_first = True
        wloc_cmd_0.long_entry = True 
    
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
    
      
 
def generate_grids_wloc_cbc_alex(oc_groups, current_op_weights_tensor_list, mem_offset_add_list, node):
    wlocs = [WLOCList() ,WLOCList()]
    wloc_shift_down_entry = WLOCEntry(weight_value=0, weight_index=1, nop=False, shift_down=True, long_entry=True)
    wloc_shift_right_entry = WLOCEntry(weight_value=0, weight_index=1, nop=False, shift_right=True, long_entry=True)
    #wloc_nop_entry        = WLOCEntry(nop = True, nop_reason='WLOC SYNC eoc tale')
    wloc_nop_entry         = WLOCEntry.get_zero_mul_entry()

    wloc_nop_entry_always_long        = WLOCEntry.get_zero_mul_entry()
    wloc_nop_entry_always_long.is_always_long = True

    
  
    # those lines have to be removed together with reordering unit
    # input_channels_reorder_dict = node['backend']['ic_lookup_dicts']
    # input_channel_reorder = list(input_channels_reorder_dict[0].keys())

    # there is a differe if we write the results to the lowest addres, or to the highst addres 
    
    #we write to the highst addr
    inx_of_lower_add = (mem_offset_add_list[1] < mem_offset_add_list[0])
    mem_lower_addr         = mem_offset_add_list[    inx_of_lower_add]
    mem_high_addr          = mem_offset_add_list[not inx_of_lower_add]
    
    len_of_lower_addr_vec  = current_op_weights_tensor_list[inx_of_lower_add].shape[1] 

    gap_between_two_cores     = max(0, mem_high_addr - (mem_lower_addr+len_of_lower_addr_vec))


    gep_zeros_filling_beg     = np.zeros((mem_lower_addr)).astype(np.int8)
    gep_zeros_filling_mid     = np.zeros((gap_between_two_cores)).astype(np.int8)
    
    kernel_size = current_op_weights_tensor_list[0].shape[2]
    
    is_3x3_conv = (kernel_size==3)
    for current_oc in oc_groups[0]:
        wloc_zero_mul_entry = (WLOCEntry(weight_value = 0, weight_index = 1, weight_offset = 1, long_entry=True, oc =current_oc))
        current_oc_first_cmd = len(wlocs[0].cmd_list)
        for k_y in range(kernel_size):                               
            for k_x in range(kernel_size):
                
                # we get ALWAYS TWO inputs: for ADD, they are used, for Conv the second is empty for ADD two are full
                ## the kernel looks like kernel[list_oc, list_ic, y, x ]

                united_tensor = np.concatenate(( gep_zeros_filling_beg,
                                                 current_op_weights_tensor_list[    inx_of_lower_add][current_oc,:,k_x,k_y], 
                                                 gep_zeros_filling_mid, 
                                                 current_op_weights_tensor_list[not inx_of_lower_add][current_oc,:,k_x,k_y]))
                  
                new_generated_wloc_part =  generate_1x1_conv_alex(kernel_1x1 = united_tensor, oc=current_oc, node=node, first_calc=not bool(k_x+k_y))

                #set shifts 
                if is_3x3_conv:
                    set_shift_flags(      k_x,k_y,new_generated_wloc_part[0].cmd_list[1], new_generated_wloc_part[1].cmd_list[1])

                for i in range(len(new_generated_wloc_part[0].cmd_list)):
                    if is_3x3_conv:
                        set_padding_flags(k_x,k_y,new_generated_wloc_part[0].cmd_list[i], new_generated_wloc_part[1].cmd_list[i])

                    wlocs[0].add_entry(new_generated_wloc_part[0].cmd_list[i])
                    wlocs[1].add_entry(new_generated_wloc_part[1].cmd_list[i])
 
                         
    
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

        #end of chanal nop     
        #if the first command is W0, delete it
        if wlocs[0].cmd_list[current_oc_first_cmd].weight_value==0 and  wlocs[1].cmd_list[current_oc_first_cmd].weight_value==0:
            del wlocs[0].cmd_list[current_oc_first_cmd]
            del wlocs[1].cmd_list[current_oc_first_cmd]               

    #do the same format for compatibility
    # num_of_grids = 2
    # grids_cbc = CBC_IR(num_of_grids)
    # grids_cbc.wlocs = [[WLOCList() for i in range(2*num_of_grids)]] # We actually have 2 WLOCS per each grid
    # grids_cbc.wlocs[0][0] = wlocs[0]
    # grids_cbc.wlocs[0][1] = wlocs[1]
    # grids_cbc.wlocs[0][2] = wlocs[0]
    # grids_cbc.wlocs[0][3] = wlocs[1]

    return wlocs


def generate_grids_wloc_cbc( node, in_current_op_weights_tensor):
      
    # The indexes order are tile_num, input_num,amm_num, block_num
    # this have to be calculated in frontend
    # TODO
    if node['op_type'] == 'Add': # add
        #offset0 = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][current_xslice_num][0][0][0] * URAM_BLOCK_SIZE #dest amm
        #offset1 = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][current_xslice_num+1][0][0][0] * URAM_BLOCK_SIZE
        
        offset0 = 0
        offset1 = in_current_op_weights_tensor.shape[0]

        #this is in Seq 
        #amm_start_address_for_output_even_grid = node['backend']['allocated_amm_blocks_for_output_even_grid'][current_tile_num][0][0]*URAM_BLOCK_SIZE

        input_0_mem_offset  = offset0
        input_0_mem_offset2 = offset1

        #for ADD node the format of tensor is tens[oc, 2*input_chanal, k0, k1] so split the tensor in two
        half_in = in_current_op_weights_tensor.shape[1] // 2        
        current_op_weights_tensor = in_current_op_weights_tensor[:,:half_in,:,:]
        current_op_weights_tensor2  = in_current_op_weights_tensor[:,half_in:,:,:]

        if not TFLITE_REQUANT:     
            scale = node['frontend']['requant_scale_uint14']
            bias = node['frontend']['requant_bias_int12']
            rough_shift  = node['frontend']['mac_rough_shift_mux']
            number_oc_add = current_op_weights_tensor.shape[0]

        if (not TFLITE_REQUANT and node['frontend']['input_folding_factor_x']>0 or node['frontend']['input_folding_factor_y']>0):                            
            #folded
            scale =        node['frontend']['folded_requant_scale_uint14']
            bias =         node['frontend']['folded_requant_bias_int12']
            rough_shift  = node['frontend']['folded_mac_rough_shift_mux']
            number_oc_add = current_op_weights_tensor.shape[0]

            node['frontend']['folded_requant_scale_uint14'] = [scale]*number_oc_add
            node['frontend']['folded_requant_bias_int12']   = [bias]*number_oc_add
            node['frontend']['folded_mac_rough_shift_mux']  = [rough_shift]*number_oc_add
    else:
        ### normal CONV
        current_op_weights_tensor_T,_ = internal_representation.get_node_weights_tensor(node)                
        current_op_weights_tensor = current_op_weights_tensor_T.data             

        # implement the rest as zeros, as for fold as for the unfold. This is the same                    
        input_1_mem_offset = 0
        current_op_weights_tensor2 = np.zeros((current_op_weights_tensor.shape[0], 
                                            0, current_op_weights_tensor.shape[2], 
                                            current_op_weights_tensor.shape[3])).astype(np.int8)
        input_0_mem_offset = 0
        input_0_mem_offset2 = 0            
    
    #####    
    oc_groups = node['backend']['oc_groups'] # Alex: this is a sort of atavism, I inharited from Dan

    if ('force_unfolding_x' in node['frontend']) and (node['frontend']['force_unfolding_x'] == True):
        # unfolding, do each output channal x2 times
        channal_in_slice = node['frontend']['output_channels']
        base_channels = node['frontend']['output_channels']
        y_num_of_folding_groups = 2**node['frontend']['output_folding_factor_y']
        jump_to_next_folded_neighbour = channal_in_slice // y_num_of_folding_groups

        # for each folding group we have to do two times the same channal
        oc_groups[0] = []
        for i_fold_group in range(y_num_of_folding_groups):
            channels_to_calculate_in_this_folded_group = np.arange(2*base_channels)[2*i_fold_group*(base_channels//y_num_of_folding_groups):(2*i_fold_group+1)*(base_channels//y_num_of_folding_groups)]    
            new_oc_group = []
            for x in channels_to_calculate_in_this_folded_group:
                new_oc_group.extend([x, x + jump_to_next_folded_neighbour])
            oc_groups[0].extend(new_oc_group)


    generated_wlocs =  generate_grids_wloc_cbc_alex(oc_groups, [current_op_weights_tensor, current_op_weights_tensor2], [input_0_mem_offset, input_0_mem_offset2], node)
                  

    return generated_wlocs

    # Get all relevant data from previous compiler passes
    ic_groups = get_nodes_real_ic_groups(node) # This creates a unified ic group in cases of 2 input nodes (e.g. Add) which are converted to conv
    ic_lookup_dicts = get_nodes_real_ic_lookup_dicts(node) # ic_lookup_dict is realic(index): actual ic(value)
    op_type = node['op_type']
    if op_type in DUAL_NONCONTIGUOUS_ALLOCATION_OPS:
        original_input_channels = node['frontend']['input_tensors'][0].get_folded_shape()[1]
        input_0_mem_offset = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][0][0][0] * URAM_BLOCK_SIZE # The indexes order are tile_num, input_num,amm_num, blobk_num
        input_1_mem_offset = node['backend']['allocated_amm_blocks_for_input_even_grid'][current_tile_num][1][0][0] * URAM_BLOCK_SIZE
        oc_groups = create_oc_groups_for_add_inline_command(node['backend']['ic_lookup_dicts'][0],node['backend']['oc_splits'])
    else:
        oc_groups = node['backend']['oc_groups']

    per_ic_group_sorted_weight_activation_pairs = get_per_ic_group_sorted_weight_activation_pairs(ic_lookup_dicts) # We sort the weight/activation pairs according to activation index to maximize number of short entries
    node['backend']['per_ic_group_sorted_weight_activation_pairs'] = per_ic_group_sorted_weight_activation_pairs
    if 'input_channels_reorder_dict' in node['backend']:
        convert_expected_input_channel = True
        input_channels_reorder_dict = node['backend']['input_channels_reorder_dict']
        input_channel_reorder = [item[0] for item in input_channels_reorder_dict.items()]
        current_op_weights_tensor=current_op_weights_tensor[:,input_channel_reorder,:,:]
    else:
        convert_expected_input_channel = False
    grid_mode = node['backend']['gridmode']
    num_of_grids = node['backend']['grid_count']
    grids_cbc = CBC_IR(num_of_grids)
    ic_splits = node['backend']['ic_splits']
    oc_splits = node['backend']['oc_splits']
    per_grid_macs = [[[] for i in range(ic_splits)] for j in range(oc_splits)]
    node['backend']['per_grid_macs'] = per_grid_macs
    if (node['frontend']['input_folding_factor_x']>0 or node['frontend']['input_folding_factor_y']>0):
        kernel_size = node['frontend']['folded_kernel_size'] # We assume symetric kernel size
        requant_bias_int12 = node['frontend']['folded_requant_bias_int12']
    else:
        kernel_size = node['frontend']['kernel_size'] # We assume symetric kernel size
        requant_bias_int12 = node['frontend']['requant_bias_int12']
    if DEBUG_DEEPCONV_SUPPORTED:
        if DEBUG_FORCE_DEEP_CONV and kernel_size==1:
            deepconv = True
        else:
            deepconv = node['backend']['deepconv']
    else:
        deepconv = False
    if deepconv and kernel_size>1:
        raise ValueError ('Deepconv with kernel_size>1 not supported')
    if len(ic_groups) != ic_splits:
        raise ValueError('Something went wrong, ic groups number != ic splits')
    if len(oc_groups) != oc_splits:
        raise ValueError('Something went wrong, oc groups number != oc splits')
    if ic_splits>1:
        is_ic_split=True
    else:
        is_ic_split=False
    current_op_input_channels = node['backend']['input_channels']
    input_channels_per_grid = current_op_input_channels // ic_splits
    current_op_output_channels = node['backend']['output_channels']
    output_channels_per_grid = current_op_output_channels // oc_splits
    non_empty_output_channels = {}
    grids_first_wloc_in_layer = [True for i in range(num_of_grids)]
    grids_cbc.wlocs = [[WLOCList() for i in range(num_of_grids*2)]] # We actually have 2 WLOCS per each grid
    max_output_channels_in_grids = get_max_list_length(oc_groups)
    total_minimal_oc_clocks_inserted_nops = 0
    total_first_wloc_entries = 0
    for current_output_channel_idx in range(max_output_channels_in_grids):
        for current_grid_idx in range(num_of_grids):
            if grid_mode == GridConfig.H14xW16:
                if op_type in LIMITED_GRIDS_OPS: # This part was prepared for 8 grids config where in maxpool grids 0,2 where actually working together and not 0,1 (See AMD Code)
                    if op_type=='MaxPool' and ic_splits == 1:
                        oc_group_idx = 0
                        ic_group_idx = 0
                    else:
                        raise ValueError ('Currently only maxpool, in H14xW16 is supported')
                elif ic_splits==1:
                    oc_group_idx = 0
                    ic_group_idx = 0
                else:
                    raise ValueError ('In mode 14x16 max ic split is 1')
            elif grid_mode == GridConfig.H14xW8:
                if op_type in LIMITED_GRIDS_OPS: # This part was prepared for 8 grids config where in maxpool grids 0,2 where actually working together and not 0,1 (See AMD Code)
                    if op_type=='MaxPool' and ic_splits == 1:
                        oc_group_idx = current_grid_idx
                        ic_group_idx = 0
                    else:
                        raise ValueError ('Currently only maxpool, in H14xW8 is supported')
                elif ic_splits==1:
                    oc_group_idx = current_grid_idx
                    ic_group_idx = 0
                elif ic_splits==2:
                    oc_group_idx = 0
                    ic_group_idx = current_grid_idx
                else:
                    raise ValueError ('In mode 12x8 max ic split is 2')
            else:
                raise ValueError ('At CBC creation, Gridmode %s not supported' % grid_mode)
            oc_group = oc_groups[oc_group_idx]
            if current_output_channel_idx>=len(oc_groups[oc_group_idx]): # if the current grid has less output channels than 
                continue
            current_requant_scale_shift = get_current_requant_scale_shift(node,oc_group[current_output_channel_idx])
            current_oc = oc_group[current_output_channel_idx]
            if current_oc in grids_cbc.per_oc_non_empty_ic_groups:
                current_oc_non_empty_ic_groups = grids_cbc.per_oc_non_empty_ic_groups[current_oc]
            else:
                current_oc_non_empty_ic_groups = []
            non_zero_ic = list(ic_lookup_dicts[ic_group_idx].keys())
            non_zero_ic_weights = current_op_weights_tensor[current_oc,non_zero_ic,:,:]
            non_zero_weights = 0
            reorder_node = 'reorder_node' in node
            if reorder_node: # In reorder node we also generate all zeros channels which were not calculated
                if len(non_zero_ic) < input_channels_per_grid:
                    if current_oc not in non_zero_ic: # This is a channel that is all zeros. We need to generate it manually since it was not calculated
                        weight_shift, activation_shift = 0,0
                        # grids_cbc.wlocs[0][current_grid_idx*2].add_entry(FirstWLOCEntry(weight_shift=weight_shift,activation_shift=activation_shift, oc = current_oc ))
                        # grids_cbc.wlocs[0][current_grid_idx*2+1].add_entry(FirstWLOCEntry(weight_shift=weight_shift,activation_shift=activation_shift, oc = current_oc ))
                        # total_first_wloc_entries+=1
                        if not grids_first_wloc_in_layer[current_grid_idx]: # Insert 3 NOPs before new oc (not relevant for 1st oc)
                            total_minimal_oc_clocks_inserted_nops+= insert_wloc_initial_nops(grids_cbc,current_grid_idx*2,deepconv,is_ic_split)
                            total_minimal_oc_clocks_inserted_nops+= insert_wloc_initial_nops(grids_cbc,current_grid_idx*2+1,deepconv,is_ic_split)
                        grids_first_wloc_in_layer[current_grid_idx] = False
                        wloc_entry = (WLOCEntry(weight_value = 1, weight_index = 0, weight_offset = 0,
                                        end_of_oc=True,long_entry=True, ic=0,oc = current_oc))
                        grids_cbc.wlocs[0][current_grid_idx*2].add_entry(wloc_entry)
                        grids_cbc.wlocs[0][current_grid_idx*2+1].add_entry(wloc_entry)
                        per_grid_macs[oc_group_idx][ic_group_idx].append(non_zero_weights)
                        current_oc_non_empty_ic_groups.append(ic_group_idx)
                        non_empty_output_channels[current_oc] = True
                        grids_cbc.per_oc_non_empty_ic_groups[current_oc] = current_oc_non_empty_ic_groups
                        continue
                elif len(non_zero_ic) > input_channels_per_grid:
                    raise ValueError ('Non zero input channels > input_channels_per_grid, this doesnt make sense')
            if np.all(non_zero_ic_weights==0): # If all the weights in current grid are zero we move to next grid if bias is zero and if not we add a dummy weight so that bias will be added
                if reorder_node:
                    raise ValueError ('Didnt expect empty channel in reordering node')
                # We need to keep track of all zero groups so that when we multiply and add in rq we dont wait for them
                if ic_splits == 1: # If there are no ic splits and bias=0 we can skip this channel, if bias!=0 we need to create a dummy mac so that bias will be used
                    if op_type in SINGLE_QUANT_PARAMS_OPS:
                        current_requant_bias_int12 = requant_bias_int12
                    else:
                        current_requant_bias_int12 = requant_bias_int12[current_oc]
                    if current_requant_bias_int12 == 0:
                        continue
                    else:
                        weight_shift, activation_shift = 0,0
                        # grids_cbc.wlocs[0][current_grid_idx*2].add_entry(FirstWLOCEntry(weight_shift=weight_shift,activation_shift=activation_shift, oc = current_oc ))
                        # grids_cbc.wlocs[0][current_grid_idx*2+1].add_entry(FirstWLOCEntry(weight_shift=weight_shift,activation_shift=activation_shift, oc = current_oc ))
                        # total_first_wloc_entries+=1
                        if not grids_first_wloc_in_layer[current_grid_idx]: # Insert 3 NOPs before new oc (not relevant for 1st oc)
                            total_minimal_oc_clocks_inserted_nops+= insert_wloc_initial_nops(grids_cbc,current_grid_idx*2,deepconv,is_ic_split)
                            total_minimal_oc_clocks_inserted_nops+= insert_wloc_initial_nops(grids_cbc,current_grid_idx*2+1,deepconv,is_ic_split)
                        grids_first_wloc_in_layer[current_grid_idx] = False
                        wloc_entry = (WLOCEntry(weight_value = 1, weight_index = 0, weight_offset = 0,
                                        use_left_pixel=False, use_right_pixel=False,use_top_pixel=False,use_bottom_pixel=False,
                                        end_of_oc=True,long_entry=True, deep_conv = deepconv, ic=0,oc = current_oc))
                        grids_cbc.wlocs[0][current_grid_idx*2].add_entry(wloc_entry)
                        grids_cbc.wlocs[0][current_grid_idx*2+1].add_entry(wloc_entry)
                        per_grid_macs[oc_group_idx][ic_group_idx].append(non_zero_weights)
                        current_oc_non_empty_ic_groups.append(ic_group_idx)
                        non_empty_output_channels[current_oc] = True
                        grids_cbc.per_oc_non_empty_ic_groups[current_oc] = current_oc_non_empty_ic_groups
                        continue
                else:
                    if op_type in SINGLE_QUANT_PARAMS_OPS:
                        current_requant_bias_int12 = requant_bias_int12
                    else:
                        current_requant_bias_int12 = requant_bias_int12[current_oc]
                    if current_requant_bias_int12 == 0:
                        raise ValueError ('Empty oc with bias=0 and ic_split>1, is currently not supported')
                        continue
                    elif len(current_oc_non_empty_ic_groups) !=0: # If at least 1 ic group was already not empty we can continue since bias will be added in that group
                        continue
                    else:
                        weight_shift, activation_shift = 0,0
                        # grids_cbc.wlocs[0][current_grid_idx*2].add_entry(FirstWLOCEntry(weight_shift=weight_shift,activation_shift=activation_shift, oc = current_oc ))
                        # grids_cbc.wlocs[0][current_grid_idx*2+1].add_entry(FirstWLOCEntry(weight_shift=weight_shift,activation_shift=activation_shift, oc = current_oc ))
                        # total_first_wloc_entries+=1
                        if not grids_first_wloc_in_layer[current_grid_idx]: # Insert 3 NOPs before new oc (not relevant for 1st oc)
                            total_minimal_oc_clocks_inserted_nops+= insert_wloc_initial_nops(grids_cbc,current_grid_idx*2,deepconv,is_ic_split)
                            total_minimal_oc_clocks_inserted_nops+= insert_wloc_initial_nops(grids_cbc,current_grid_idx*2+1,deepconv,is_ic_split)
                        grids_first_wloc_in_layer[current_grid_idx] = False
                    
                        wloc_entry = (WLOCEntry(weight_value = 1, weight_index = 0, weight_offset = 0,
                                        use_left_pixel=False, use_right_pixel=False,use_top_pixel=False,use_bottom_pixel=False,
                                        end_of_oc=True,long_entry=True, deep_conv = deepconv, ic=0,oc = current_oc))
                        grids_cbc.wlocs[0][current_grid_idx*2].add_entry(wloc_entry)
                        grids_cbc.wlocs[0][current_grid_idx*2+1].add_entry(wloc_entry)
                        per_grid_macs[oc_group_idx][ic_group_idx].append(non_zero_weights)
                        current_oc_non_empty_ic_groups.append(ic_group_idx)
                        non_empty_output_channels[current_oc] = True
                        grids_cbc.per_oc_non_empty_ic_groups[current_oc] = current_oc_non_empty_ic_groups
                        continue

            else:
                current_oc_non_empty_ic_groups.append(ic_group_idx)
                non_empty_output_channels[current_oc] = True
            previous_nonzero_activation_index = 0
            # The 1st WLOC entry for each output channnel calcs contains the channels weight shift and activation shift
            weight_shift, activation_shift = get_mac_shifts_from_frontend(current_requant_scale_shift)
            # grids_cbc.wlocs[0][current_grid_idx*2].add_entry(FirstWLOCEntry(weight_shift=weight_shift,activation_shift=activation_shift, oc = current_oc ))
            # grids_cbc.wlocs[0][current_grid_idx*2+1].add_entry(FirstWLOCEntry(weight_shift=weight_shift,activation_shift=activation_shift, oc = current_oc ))
            # total_first_wloc_entries+=1
            if not grids_first_wloc_in_layer[current_grid_idx]: # Insert 3 NOPs before new oc (not relevant for 1st oc)
                total_minimal_oc_clocks_inserted_nops+= insert_wloc_initial_nops(grids_cbc,current_grid_idx*2,deepconv,is_ic_split)
                total_minimal_oc_clocks_inserted_nops+= insert_wloc_initial_nops(grids_cbc,current_grid_idx*2+1,deepconv,is_ic_split)
            grids_first_wloc_in_layer[current_grid_idx] = False
            current_filter_macs = 0
            current_ic_split_non_zero_ic_weights_count = np.count_nonzero(non_zero_ic_weights)
            double_mac_clocks = int(math.ceil(current_ic_split_non_zero_ic_weights_count/2))
                        
            for k_y in range(kernel_size):                               
                for k_x in range(kernel_size):
                    long_entry = True # When we switch pixel we need to use long entry                    
                    working_on_odd_wloc=False # We need this to spot 1st entry of odd wloc and make it of LONG type
                    for current_ic,activation_index in per_ic_group_sorted_weight_activation_pairs[ic_group_idx]:
                        current_weight_value = current_op_weights_tensor[current_oc,current_ic,k_y,k_x]
                        if activation_index!=ic_lookup_dicts[ic_group_idx][current_ic]:
                            raise ValueError ('here')
                        if current_weight_value !=0:
                            real_x_position = k_x-(kernel_size-1)//2
                            real_y_position = k_y-(kernel_size-1)//2
                            non_zero_weights+=1
                            use_left_pixel = real_x_position==-1
                            use_right_pixel = real_x_position==1
                            use_top_pixel = real_y_position==-1
                            use_bottom_pixel = real_y_position==1
                            add_op_source_buffer_idx =0
                            if current_filter_macs==double_mac_clocks: # This means we are at start of odd WLOC
                                previous_nonzero_activation_index = 0
                            if op_type in DUAL_NONCONTIGUOUS_ALLOCATION_OPS:
                                # if activation_index>=original_input_channels it means this input is from input#1 of the add op. In such case we subtract original_input_channels to know the real input channel of input#1 and add the mem offset of input 1
                                if activation_index>=original_input_channels:
                                    activation_index+=input_1_mem_offset-original_input_channels
                                    add_op_source_buffer_idx = 1
                                else:
                                    activation_index+=input_0_mem_offset
                                    add_op_source_buffer_idx = 0
                            force_long_command = False
                            if activation_index>=previous_nonzero_activation_index: 
                                distance_from_previous_activation = activation_index-previous_nonzero_activation_index
                            else:
                                force_long_command = True
                                distance_from_previous_activation = current_op_input_channels-previous_nonzero_activation_index+activation_index
                            if force_long_command:
                                long_entry = True
                            if op_type!='MaxPool': #In Maxpool op only the even wloc is functioning
                                if current_filter_macs==double_mac_clocks:
                                    if not working_on_odd_wloc: # We need to force long entry on first entry of odd WLOC
                                        working_on_odd_wloc = True 
                                        long_entry = True
                            previous_nonzero_activation_index = activation_index
                            # TODO Alex  
                            # out of usage 
                            # use_left_pixel, use_right_pixel, use_top_pixel, use_bottom_pixel, deepconv
                            # for add for 3x3 conv (now pluged)
                            # shift_down, shift_right, is_pair, pair_add
                            #             (weight_value=0, weight_index=0, weight_offset=0, 
                            #              shift_down=False, shift_right=False,
                            #               end_of_oc=False, long_entry=False, is_pair = False, nop=False,
                            #               freeze_wloc=False, ic=0, oc=0, add_op_source_buffer = 0, pair_add = 0)

                            #Alex for phase 2
                            is_x_calculation_done = True 
                            wloc_entry = (WLOCEntry(weight_value = current_weight_value, weight_index = activation_index, weight_offset = distance_from_previous_activation,
                                                    shift_down=False, shift_right=False, is_pair = False, pair_add = 0,
                                                    end_of_oc=False, long_entry=long_entry,  
                                                    ic=current_ic,oc = current_oc,add_op_source_buffer=add_op_source_buffer_idx))


                            if op_type=='MaxPool': #In Maxpool op only the even wloc is functioning
                                grids_cbc.wlocs[0][current_grid_idx*2].add_entry(wloc_entry)
                                grids_cbc.wlocs[0][current_grid_idx*2+1].add_entry(WLOCEntry(nop = True))
                            else:
                                if current_filter_macs<double_mac_clocks:
                                    grids_cbc.wlocs[0][current_grid_idx*2].add_entry(wloc_entry)
                                else:
                                    grids_cbc.wlocs[0][current_grid_idx*2+1].add_entry(wloc_entry)
                            current_filter_macs+=1
                            long_entry = False

    
                
                             
            if current_ic_split_non_zero_ic_weights_count != current_filter_macs:
                raise ValueError ('current_ic_split_non_zero_ic_weights_count != current_filter_macs. This doesnt make sense, please check')
            if op_type!='MaxPool':
                if ((current_ic_split_non_zero_ic_weights_count % 2) == 1): # If we have odd number of macs we need to add a NOP to the odd grid so that even and odd grids have same number of clocks
                    wloc_entry = (WLOCEntry(weight_value = 0, weight_index = 1, weight_offset = 1, # In this case we take weight=0 and index=1 since a_shift and w_shift already set and we cant use them to zero result. We dont want to take nop since we cant mark EOC only on long commands and NOP is a short one
                                            end_of_oc=False,long_entry=True, ic=0,oc = current_oc))
                    grids_cbc.wlocs[0][current_grid_idx*2+1].add_entry(wloc_entry)
            if op_type == 'MaxPool' and current_filter_macs!=node['frontend']['kernel_size']*node['frontend']['kernel_size']:
                raise ValueError('At %s: MaxPool OP expects to have kernel_size*kernel_size(%d) macs per filter. Got: %d' % (node['name'],kernel_size*kernel_size,current_filter_macs))
            if op_type == 'Resize' and current_filter_macs!=1:
                raise ValueError('At %s: Resize OP expects to have 1 mac per filter. Got: %d' % (node['name'],current_filter_macs))
            if op_type == 'Add' and current_filter_macs!=2: # In add command each filter must include exactly 2 macs so there is balance between the grids and we can set the exact order of oc_processing order so it matches the input0 channels order to allow overwrite to same mem
                if current_filter_macs==1:
                    pass # We no longer need this adition since in double mac per clock config we would already add a nop to odd grid and always have 2 entries
                    wloc_entry = (WLOCEntry(weight_value = 0, weight_index = 1, weight_offset = 1, # In this case we take weight=0 and index=1 since a_shift and w_shift already set and we cant use them to zero result
                                    use_left_pixel=False, use_right_pixel=False,use_top_pixel=False,use_bottom_pixel=False,
                                    end_of_oc=False,long_entry=True, deep_conv = deepconv, ic=0,oc = current_oc))
                    grids_cbc.wlocs[0][current_grid_idx].add_entry(wloc_entry)
                else:
                    raise ValueError ('Didnt expect more than 2 macs per filter in Add command. got %d macs' % current_filter_macs)


            grids_cbc.wlocs[0][current_grid_idx*2].set_oc_end()
            grids_cbc.wlocs[0][current_grid_idx*2+1].set_oc_end()
            per_grid_macs[oc_group_idx][ic_group_idx].append(non_zero_weights)
            grids_cbc.per_oc_non_empty_ic_groups[current_oc] = current_oc_non_empty_ic_groups

    node['backend']['non_empty_output_channels'] = non_empty_output_channels
    node['backend']['total_minimal_oc_clocks_inserted_nops'] = total_minimal_oc_clocks_inserted_nops
    node['backend']['total_first_wloc_entries'] = total_first_wloc_entries

    return grids_cbc2
        

    
def rq_busy(rq_state_machines):
    busy = False
    for state_machine in rq_state_machines:
        if state_machine.is_busy():
            busy = True
    return busy

def get_length_of_longest_list(lst):
    return max(len(x.wloc_list) for x in lst)

def rqloc_contains_rq_ops(grids_rq_commands):
    ret_val = False
    for rq_entry in grids_rq_commands:
        if rq_entry.dest_write:
            ret_val = True
            break
    return ret_val
def get_latest_kernel_pos(wloc_list):
    pos = len(wloc_list)-1
    while not isinstance(wloc_list[pos],FirstWLOCEntry) and not wloc_list[pos].long_entry and pos>0:
        pos=pos-1
    if isinstance(wloc_list[pos],FirstWLOCEntry) or not wloc_list[pos].long_entry:# In case wloc_list doesnt contain any long entry (possible only in empty list) or first entry before split is of type FirstWLOCEntry
        use_right_pixel = False
        use_left_pixel = False
        use_bottom_pixel = False
        use_top_pixel = False
    else:
        use_right_pixel = wloc_list[pos].use_right_pixel
        use_left_pixel = wloc_list[pos].use_left_pixel
        use_bottom_pixel = wloc_list[pos].use_bottom_pixel
        use_top_pixel = wloc_list[pos].use_top_pixel
    return use_right_pixel,use_left_pixel,use_bottom_pixel,use_top_pixel

def enforce_nops_from_eoc_to_freeze(current_wloc,freeze_pos):
    distance_from_freeze=1
    current_entry = current_wloc.wloc_list[freeze_pos-distance_from_freeze]
    while isinstance(current_entry,FirstWLOCEntry) or (not current_entry.end_of_oc) and distance_from_freeze<MINIMAL_EOC_TO_FREEZE_DISTANCE:
        distance_from_freeze+=1
        current_entry = current_wloc.wloc_list[freeze_pos-distance_from_freeze]
    if distance_from_freeze<MINIMAL_EOC_TO_FREEZE_DISTANCE:
        nops_to_add = MINIMAL_EOC_TO_FREEZE_DISTANCE-distance_from_freeze
        for i in range(nops_to_add):
            nop_wloc_command = WLOCEntry(nop = True)
            current_wloc.insert_entry(freeze_pos,nop_wloc_command)

def check_length_of_wloc_rqloc(wlocs_split,grids_rq_commands_split):
    rqloc_size = len(grids_rq_commands_split)
    for current_wloc in wlocs_split:
        current_wloc_size = len(current_wloc.wloc_list)
        if current_wloc_size<rqloc_size:
            raise ValueError ('wloc size < rqloc size, Didnt expect that!')
        while current_wloc_size> rqloc_size:
            if not current_wloc.wloc_list[-1].nop:
                raise ValueError ('Cant remove none nop entries. Shouldnt have happen here. Check integrity')
            current_wloc.remove_last_entry()
            current_wloc_size = len(current_wloc.wloc_list)



def handle_end_of_wlocs_and_rqloc(wlocs_split,grids_rq_commands_split,current_split_clock_idx,deepconv,add_freeze=False):
    # Remove redundent entries from current rqloc split
    current_rq_entry_idx = len(grids_rq_commands_split)-1
    while ((grids_rq_commands_split[current_rq_entry_idx].dest_write == False) and (grids_rq_commands_split[current_rq_entry_idx].folding_buf_write == False) and (current_rq_entry_idx>current_split_clock_idx+1)):
        del(grids_rq_commands_split[current_rq_entry_idx])
        current_rq_entry_idx=current_rq_entry_idx-1

    # Set last_rq_command flag in last rq command
    grids_rq_commands_split[-1].set_last_command()
    # Fill splitted wloc with nops until rqloc end clock and add EOL at last non-nop entry
    for current_wloc in wlocs_split:
        if add_freeze:
            target_wloc_size = len(grids_rq_commands_split)-1 # If we are going to insert a Freeze entry we fill until rqloc size-1
        else:
            target_wloc_size = len(grids_rq_commands_split)

        while len(current_wloc.wloc_list)<target_wloc_size:
            nop_wloc_command = WLOCEntry(nop = True)
            current_wloc.add_entry(nop_wloc_command)
        if add_freeze:
            use_right_pixel,use_left_pixel,use_bottom_pixel,use_top_pixel = get_latest_kernel_pos(current_wloc.wloc_list)
            freeze_command = WLOCEntry(weight_value=0,weight_index=0,long_entry=True,deep_conv=deepconv,freeze_wloc=True,
                                    use_right_pixel=use_right_pixel,use_left_pixel=use_left_pixel,use_bottom_pixel=use_bottom_pixel,use_top_pixel=use_top_pixel)
            freeze_pos = len(current_wloc.wloc_list) # We need to insert the freeze command before all nop commands at end of wloc list
            while freeze_pos>0 and current_wloc.wloc_list[freeze_pos-1].nop:
                freeze_pos=freeze_pos-1
            current_wloc.insert_entry(freeze_pos,freeze_command)
            enforce_nops_from_eoc_to_freeze(current_wloc,freeze_pos)
    check_length_of_wloc_rqloc(wlocs_split,grids_rq_commands_split)

def add_initial_entries_to_rq_commands(grids_rq_commands,ic_splits,is_folding_conv_x,grids_to_fold=0,rq_nops_at_start=RQ_NOPS_AT_START,from_last_mul_to_dsp_out=FROM_LAST_MUL_TO_DSP_OUT):
    if is_folding_conv_x:
        if grids_to_fold==2:
            folding_additional_clocks=RQ_ADDITIONAL_CLOCKS_FOR_FOLDING_CONV_2CHANNELS
        else:
            raise ValueError('At folding conv, illegal grids_to_fold number')

    else:
        folding_additional_clocks=0
    
    nops_to_add = rq_nops_at_start+ic_splits + (ic_splits-1)*(MULTIPLY_ADD_COMMAND_CYCLES)+MULTIPLY_COMMAND_CYCLES+folding_additional_clocks
    nops_to_add += from_last_mul_to_dsp_out

    #befor Ph 2 Alex 
    # for i in range(nops_to_add):
    #     nop_rq_entry = RQEntry()
    #     grids_rq_commands.append(nop_rq_entry)
    
    # for Ph 2 Alex Logv
    nop_rq_entry = RQParamEntry(nop_count=nops_to_add)
    grids_rq_commands.append(nop_rq_entry)     

def get_balanced_wloc_entries_limit(node,grids_wloc_commands):
    if node['backend']['past_compile_wloc_sizes']:
        past_compile_wloc_sizes = node['backend']['past_compile_wloc_sizes']
        if len(past_compile_wloc_sizes) == 1:
            max_wloc_length = MAX_WLOC_128BIT_ENTRIES-SPARE_72BIT_ENTRIES_FOR_SPLIT
        else:
            if sum(past_compile_wloc_sizes)<=(MAX_WLOC_128BIT_ENTRIES-SPARE_72BIT_ENTRIES_FOR_SPLIT):
                max_wloc_length = MAX_WLOC_128BIT_ENTRIES-SPARE_72BIT_ENTRIES_FOR_SPLIT
            else:
                if max(past_compile_wloc_sizes)-min(past_compile_wloc_sizes) > 20: # If actual wloc sizes are too variant, recalc optimal wloc size
                    max_wloc_length = sum(past_compile_wloc_sizes)
                else:
                    return node['backend']['balanced_wloc_entries_limit']
        wloc_inflation_rate=0 # This is not needed since its actual sizes
    else:
        max_wloc_length, max_diff = get_wloc_128bitentries_stats(grids_wloc_commands) #Dans start from here
        wloc_inflation_rate=0.1 # This is an estimation of how much wloc size would increse during rqloc creation and nop insertion to wloc
    divider = 1
    added_wloc_entries_during_rqloc_creation = int(math.ceil(MAX_WLOC_128BIT_ENTRIES*wloc_inflation_rate))

    while max_wloc_length/divider>(MAX_WLOC_128BIT_ENTRIES-added_wloc_entries_during_rqloc_creation-SPARE_72BIT_ENTRIES_FOR_SPLIT):
        divider=divider+1
    balanced_wloc_entries_limit = int(math.ceil(max_wloc_length/divider))+added_wloc_entries_during_rqloc_creation
    if DEBUG_PRINT_CBC:
        print('max balanced wloc size: %d' % balanced_wloc_entries_limit)
    if balanced_wloc_entries_limit>(MAX_WLOC_128BIT_ENTRIES-SPARE_72BIT_ENTRIES_FOR_SPLIT):
        raise ValueError ('Overflow in wloc split size')
    node['backend']['balanced_wloc_entries_limit'] = balanced_wloc_entries_limit
    return balanced_wloc_entries_limit
def check_for_eoc_entry(current_wloc_commmands):
    for command in current_wloc_commmands:
        if not isinstance(command,FirstWLOCEntry) and command.end_of_oc:
            return True
    return False

def create_x_padding_enabled_per_grid_array(current_op_folded_width:int,grid_mode:GridConfig):
    x_padding_enabled_per_grid_array = []

    for current_grid in range(MAX_GRID_COUNT):
        is_x_padding_enabled = False
        if current_op_folded_width<8:
            is_x_padding_enabled = True # In this mode x padding is always enabled
        elif current_op_folded_width>8 and current_op_folded_width<16:
            if current_grid in [2,6,3,7]:
                is_x_padding_enabled = True # In this mode, grids 2,3,6,7 are padded.
        elif current_op_folded_width>16 and current_op_folded_width<24:
            if current_grid in [4,5]:
                is_x_padding_enabled = True # In this mode, and width<25 (we use only 3 out of 4 grids in group) grids 4,5 are padded. 
        elif current_op_folded_width>23 and current_op_folded_width<32: # If W=24 we completely pad grids 6,7
            if current_grid in [6,7]:
                is_x_padding_enabled = True # In this mode, and width<25 (we use only 3 out of 4 grids in group) grids 4,5 are padded. 
        x_padding_enabled_per_grid_array.append(is_x_padding_enabled)                
    return x_padding_enabled_per_grid_array
        
def get_hw_resize_write_masks(grid_mode,grid_id):
    if grid_mode == GridConfig.H14xW8:
        # In case of hw resize x from 8 to 16 all grids write first part to AMM0 and 2nd part to AMM1
        write_masks = [0x1,0x2]
    else:
        raise ValueError ('Currently x hw resize is only supported for mode GridConfig.H14xW8')
    return write_masks

def verify_oc_ordering(oc_processing_order):
    for idx,oc in enumerate(oc_processing_order):
        if oc!=idx:
            raise ValueError ('Ordering conv is not in order. Something went wrong!')

def get_folded_write_mask(grid_mode,current_amm_write_mask):
    if grid_mode == GridConfig.H14xW16:
        if current_amm_write_mask in [0x1,0x2]:
            folded_write_mask = current_amm_write_mask
        else:
            raise ValueError ('Unexpected write mask for gridmode: %s' % grid_mode)
    else:
        raise ValueError ('hw folding for Gridmode: %s not supported' % grid_mode)
    return folded_write_mask

def get_tsnp_amm_write_mask(op_type,grid_mode,current_grid,ic,input_channels_splits,ic_groups,amm_table_idx=0,is_folding_conv_x=False,is_hw_x_resize = False):
    # mode	ic split	masks			
    # 14x8	1		    0xF			
    # 14x16	1		    0x5	0xA		
    # not 14x32	1		    0x1	0x2	0x4	0x8
    # 14x8	2		    0x5/0xA			
    # not 14x16	2		    0x1/4	0x2/8		
    # not 14x8	4		    0x1/2/4/8			
    
    # The get_tsnp_amm_write_mask serves both building amm write mask table for ddr read and amm mask for rqparams table.
    # In amm mask for rqparams table case, we get a grid number (and not amm_table_idx) and map the situation to a case of getting amm_table_idx
    
    if not current_grid==None: # rqparams table case
        amm_table_idx=0
        if grid_mode == GridConfig.H14xW16:
            if current_grid in [0]:
                amm_table_idx=0
            elif current_grid in [1]:
                amm_table_idx=1
            else:
                raise ValueError ('grid number out of range')
    amm_write_mask = 0 #Dans TODO: make this based on actual number of AMMS!!!
    if is_hw_x_resize:
        amm_write_mask = get_hw_resize_write_masks(grid_mode,current_grid)
        return amm_write_mask
    if grid_mode == GridConfig.H14xW8:
        if input_channels_splits == 1:
            amm_write_mask = 0x3
        elif input_channels_splits == 2:
            if ic in ic_groups[0]:
                amm_write_mask = 0x1
            elif ic in ic_groups[1]:
                amm_write_mask = 0x2
            else:
                raise ValueError ('ic not found in ic groups')
        else:
            raise ValueError ('in GridConfig.H12xW8 only 1,2,4 input split is allowed')
    elif grid_mode == GridConfig.H14xW16:
        if input_channels_splits == 1:
            if amm_table_idx == 0:
                amm_write_mask = 0x1
            elif amm_table_idx == 1:
                amm_write_mask = 0x2
            else:
                raise ValueError ('amm table idx out of range')
        else:
            raise ValueError ('in GridConfig.H12xW16 only 1,2 input split is allowed')
    else:
        raise ValueError ('Current grid mode not supported')

    return amm_write_mask

def generate_rqparams_ir(grids_cbc,node):
    # alex for phase 2
    return

    # if (node['frontend']['input_folding_factor_x']>0 or node['frontend']['input_folding_factor_y']>0):
    #     requant_scale_uint10 = node['frontend']['folded_requant_scale_uint10']
    #     requant_scale_uint14 = node['frontend']['folded_requant_scale_uint14']
    #     requant_scale_uint17 = node['frontend']['folded_requant_scale_uint17']
    #     requant_bias_int12 = node['frontend']['folded_requant_bias_int12']
    #     rough_shift_uint2 = node['frontend']['folded_requant_scale_f_uint2']
    #     mac_rough_shift_mux = node['frontend']['folded_mac_rough_shift_mux']
        
    # else:
    #     requant_scale_uint10 = node['frontend']['requant_scale_uint10']
    #     requant_scale_uint14 = node['frontend']['requant_scale_uint14']
    #     requant_scale_uint17 = node['frontend']['requant_scale_uint17']
    #     requant_bias_int12 = node['frontend']['requant_bias_int12']
    #     rough_shift_uint2 = node['frontend']['requant_scale_f_uint2']
    #     mac_rough_shift_mux = node['frontend']['mac_rough_shift_mux']
    # rq_params_table_helper = node['backend']['rq_params_table_helper'] # This table includes (oc,amm_mask) pairs collected during cbc creation. It tells per each oc what should be its write mask. It is ordered by oc creation order

    # rqparams = grids_cbc.rqparams.cmd_list
    # for idx,current_oc_data in enumerate(rq_params_table_helper):
    #     current_oc = current_oc_data[0]
    #     current_mask = current_oc_data[1]
    #     if node['op_type'] in SINGLE_QUANT_PARAMS_OPS:
    #         if REDUCED_MAC_RESCALE_BUS_WIDTH:
    #             current_rqparam = RQParamEntry(requant_scale_uint10, requant_bias_int12, write_mask=current_mask)
    #         elif MCHP_NUMERICS:
    #             current_rqparam = RQParamEntry(requant_scale_uint14, requant_bias_int12, write_mask=current_mask,rough_shift=mac_rough_shift_mux)
    #         else:
    #             current_rqparam = RQParamEntry(requant_scale_uint17, requant_bias_int12, write_mask=current_mask,rough_shift=rough_shift_uint2)

    #     else:
    #         if REDUCED_MAC_RESCALE_BUS_WIDTH:
    #             current_rqparam = RQParamEntry(requant_scale_uint10[current_oc], requant_bias_int12[current_oc], write_mask=current_mask)
    #         elif MCHP_NUMERICS:
    #             current_rqparam = RQParamEntry(nop = 1)
    #         else:
    #             current_rqparam = RQParamEntry(requant_scale_uint17[current_oc], requant_bias_int12[current_oc], write_mask=current_mask,rough_shift=rough_shift_uint2[current_oc])
    #     rqparams.append(current_rqparam)

def print_cbc(node):
    grids_rq_commands_splits = node['backend']['grids_cbc'].rqlocs
    for split_idx,rqloc_split in enumerate(grids_rq_commands_splits):
        grids_rq_commands = rqloc_split.rqloc_list
        grids_wloc_commands = node['backend']['grids_cbc'].wlocs[split_idx]
        num_wlocs = len(grids_wloc_commands)
        total_clocks = len(grids_wloc_commands[0].wloc_list)
        print ('Total clocks in cbc: %d' % total_clocks)
        for clock_idx in range(len(grids_rq_commands)):
            if clock_idx<len(grids_wloc_commands[0].wloc_list):
                current_wloc_entries = [grid_wlocs.wloc_list[clock_idx] for grid_wlocs in grids_wloc_commands]
                for i in range(num_wlocs):
                    print('%15s' % current_wloc_entries[i], end = ' ')
            else:
                for i in range(num_wlocs):
                    print('%15s' % ' ', end = ' ')

            print(grids_rq_commands[clock_idx], end = '')
            print('')

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
                rq_scale=rq_bias=rq_rough = 0
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
        pass
        


def export_cbc_to_xls_alex(node,filename,format = DebugFilesFormat.XLSX, current_tile_num=0):
    # calculate the wloc split number

    wloc_tile = node['backend']['grids_cbc'].alex_wlocs[current_tile_num]
    total_splits = len(wloc_tile)
    
    for i_splits in range(total_splits): 

        export_cbc_to_xls_itamar_format(node,filename, current_tile_num)

        #this export in the Alex Format (no zeros in RQ and RT)
        '''
        csv_filename = filename+'_tile_'+str(current_tile_num)+'_split'+str(i_splits)+'_alex.csv'
        csv_file = open(csv_filename,'w')
        str_RQ = ' '
        srt_RT = ' '
        current_nops_rq = current_nops_rt = 0 
        current_inx_rq  = current_inx_rt = 0
        

        
        for  clc_idx in range(len(wloc_tile[i_splits][0].cmd_list)):
            
            str_w0 = str(wloc_tile[i_splits][0].cmd_list[clc_idx])
            str_w1 = str(wloc_tile[i_splits][1].cmd_list[clc_idx])
        
            ##-------------RQ -------------------
            if current_nops_rq==0 and current_inx_rq<len(node["backend"]['grids_cbc'].alex_rqParam[i_splits][0].cmd_list): 
                str_RQ = str(node["backend"]['grids_cbc'].alex_rqParam[i_splits][0].cmd_list[current_inx_rq])
                current_nops_rq = node["backend"]['grids_cbc'].alex_rqParam[i_splits][0].cmd_list[current_inx_rq].nop_count
                current_inx_rq+=1
            else:        
                str_RQ = ''
                current_nops_rq -=1
            ##-------------RT -------------------
            if current_nops_rt==0 and current_inx_rt<len(node["backend"]['grids_cbc'].RTable[i_splits].cmd_list):
                str_RT = str(node["backend"]['grids_cbc'].RTable[i_splits].cmd_list[current_inx_rt])
                current_nops_rt = node["backend"]['grids_cbc'].RTable[i_splits].cmd_list[current_inx_rt].nop_count
                current_inx_rt +=1
            else:        
                str_RT = ''
                current_nops_rt -=1
            
            

            csv_file.write((str_w0+','+str_w1+','+str_RQ+','+str_RT+"\n")) 
        
                
        csv_file.close()
    '''


def export_cbc_to_xls(node,filename,format = DebugFilesFormat.XLSX):
    actual_filename = filename
    grids_rq_commands_splits = node['backend']['grids_cbc'].rqlocs
    for split_idx,rqloc_split in enumerate(grids_rq_commands_splits):
        if format == DebugFilesFormat.XLSX:
            xls_filename = actual_filename+'_split'+str(split_idx)+'.xlsx'
            workbook = xlsxwriter.Workbook(xls_filename)
            worksheet = workbook.add_worksheet('CBC')
        else:
            csv_filename = actual_filename+'_split'+str(split_idx)+'.csv'
            csv_file = open(csv_filename,'w')

        grids_rq_commands = rqloc_split.cmd_list
        grids_wloc_commands = node['backend']['grids_cbc'].wlocs[split_idx]

        num_of_grids = node['backend']['grid_count']
        grids_header = ['WLOC#'+str(i)+'Grid#'+str(i//2) for i in range(num_of_grids*2)] # We have 2 WLOCS per each grid
        xls_header0 = ['clock'] + grids_header + ['grid mux-in','last','dsp op','dsp_result_out','rq data read','dest write','rqparams_wr_mask','folding_buf_write','fld_wr_mask','scale_grid_sel','x_output_padding_enable']
        for col,header in enumerate(xls_header0):
            if format == DebugFilesFormat.XLSX:
                worksheet.write(0,col,xls_header0[col])
            else:
                csv_file.write(xls_header0[col]+',')
        if not format == DebugFilesFormat.XLSX:
            csv_file.write('\n')


        total_clocks = len(grids_rq_commands)
        if DEBUG_PRINT_CBC:
            print ('Total clocks in cbc: %d' % total_clocks)
        for clock_idx in range(len(grids_rq_commands)):
            current_wloc_commmands = [] # For each clock we create a list that has current clock wloc op for all grids
            for grid_wloc_command in grids_wloc_commands:
                if clock_idx<len(grid_wloc_command.wloc_list):
                    current_wloc_commmands.append(grid_wloc_command.wloc_list[clock_idx])
                else: # If specific grid has less commands we insert nop commands
                    nop_wloc_command = WLOCEntry(nop = True)
                    current_wloc_commmands.append(nop_wloc_command)
            if format == DebugFilesFormat.XLSX:
                worksheet.write(clock_idx+1,0,clock_idx)
            else:
                csv_file.write(str(clock_idx)+',')
            for i in range(num_of_grids*WLOCS_PER_GRID): # We have 2 WLOCS per each grid
                if format == DebugFilesFormat.XLSX:
                    worksheet.write(clock_idx+1,i+1,str(current_wloc_commmands[i]))
                else:
                    csv_file.write(str(current_wloc_commmands[i])+',')


            current_rq_entry = grids_rq_commands[clock_idx]
            if format == DebugFilesFormat.XLSX:
                worksheet.write(clock_idx+1,num_of_grids+1,current_rq_entry.grid_mux_in)
                worksheet.write(clock_idx+1,num_of_grids+2,str(current_rq_entry.last_rq_command))
                worksheet.write(clock_idx+1,num_of_grids+3,str(current_rq_entry.dsp_command))
                worksheet.write(clock_idx+1,num_of_grids+4,current_rq_entry.dsp_result_out)
                worksheet.write(clock_idx+1,num_of_grids+5,current_rq_entry.rq_data_read)
                worksheet.write(clock_idx+1,num_of_grids+6,current_rq_entry.dest_write)
                worksheet.write(clock_idx+1,num_of_grids+7,current_rq_entry.write_mask)
                worksheet.write(clock_idx+1,num_of_grids+8,current_rq_entry.folding_buf_write)
                worksheet.write(clock_idx+1,num_of_grids+9,current_rq_entry.folding_wr_mask)
                worksheet.write(clock_idx+1,num_of_grids+10,current_rq_entry.scale_grid_sel)
                worksheet.write(clock_idx+1,num_of_grids+11,current_rq_entry.x_output_padding_enable)
            else:
                #Alex removed for Ph2 
                # csv_file.write(str(current_rq_entry.grid_mux_in)+',')
                # csv_file.write(str(current_rq_entry.last_rq_command)+',')
                # csv_file.write(str(current_rq_entry.dsp_command)+',')
                # csv_file.write(str(current_rq_entry.dsp_result_out)+',')
                # csv_file.write(str(current_rq_entry.rq_data_read)+',')
                # csv_file.write(str(current_rq_entry.dest_write)+',')
                # csv_file.write(str(current_rq_entry.write_mask)+',')
                # csv_file.write(str(current_rq_entry.folding_buf_write)+',')
                # csv_file.write(str(current_rq_entry.folding_wr_mask)+',')
                # csv_file.write(str(current_rq_entry.scale_grid_sel)+',')
                # csv_file.write(str(current_rq_entry.x_output_padding_enable)+',')
                csv_file.write('to be done')

                
            if not format == DebugFilesFormat.XLSX:
                csv_file.write('\n')


        if format == DebugFilesFormat.XLSX:
            workbook.close()
        else:
            csv_file.close()
