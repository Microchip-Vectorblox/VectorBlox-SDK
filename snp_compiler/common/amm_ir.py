import numpy as np
import math
from common.enums import AMMType
from common.hw_config import AMM_COUNT

from common.hw_config import AMM_HEIGHT,AMM_WIDTH,BRAM_BLOCK_SIZE,BRAM_NUM_BLOCKS,URAM_BLOCK_SIZE,URAM_NUM_BLOCKS,NUM_OF_BRAM_AMMS
from common.debug_flags import AMM_ALLOCATION_OPTIMIZE_LEFTOUT_SIZE
from common.tensor_ir import TensorDeAllocationList,get_tensor_name_from_tiled_tensor
from common.utils import list_of_lists_split_at_pos
import copy

def get_size_sorted_free_mem_blobs(allocated_blocks):
    found_free_blob=False
    blob_size=0
    blob_start_idx=0
    free_blobs = []
    for idx,block in enumerate(allocated_blocks):
        if block == False: # current block not allocated
            if found_free_blob == True: # Already inside a free blob
                blob_size+=1
            else:
                found_free_blob = True # A start of new free mem blob
                blob_size=1
                blob_start_idx=idx
        else: # end of free blob or continue of no blob found
            if found_free_blob == True: # Already inside a free blob
                free_blobs.append((blob_size,blob_start_idx))
                found_free_blob = False
                blob_size=0
                blob_start_idx=0
    if found_free_blob == True: #We are in middle of free blob and reached end of mem
        free_blobs.append((blob_size,blob_start_idx))
    free_blobs.sort()
    return free_blobs

class AMMAllocator:
    def __init__(self,block_size,num_blocks,amm_idx):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.allocated_blocks = [False for i in range(self.num_blocks)]
        self.amm_idx = amm_idx
    def reset_mem(self):
        self.allocated_blocks = [False for i in range(self.num_blocks)]
    def allocate_mem(self,size,double = False):
        # Find minimal number of contiguous blocks that are free and satisfy allocation size
        if double:
            minimal_blocks = math.ceil(size/self.block_size)*2
        else:
            minimal_blocks = math.ceil(size/self.block_size)
        newly_allocated_blocks = []
        possible_solutions = []
        for i in range(0,self.num_blocks-minimal_blocks+1,minimal_blocks): # First, trying to find all possible solutions which are aligned to minimal_blocks
            found = True
            for j in range(minimal_blocks):
                if self.allocated_blocks[i+j] == True:
                    found=False
                    break
            if found:
                possible_solutions.append(i)
        if len(possible_solutions)==0: # If we didnt find any aligned solution try to find one which is not aligned.
            # TODO: Dans, need to check if since we added below code which picks the solution which leaves max area free maybe we can mix between all aligned and non-aligned solutions
            for i in range(0,self.num_blocks-minimal_blocks+1):
                found = True
                for j in range(minimal_blocks):
                    if self.allocated_blocks[i+j] == True:
                        found=False
                        break
                if found:
                    possible_solutions.append(i)

        if len(possible_solutions)>0: # If we found a solution
            if AMM_ALLOCATION_OPTIMIZE_LEFTOUT_SIZE: # Take the solution which will leave maximal free mem blob
                free_mem_blobs = get_size_sorted_free_mem_blobs(self.allocated_blocks)
                selected_blob_start_block = None
                selected_solution_idx = None
                for free_mem_blob in free_mem_blobs:
                    if free_mem_blob[0]>=minimal_blocks:
                        selected_blob_start_block = free_mem_blob[1]
                        for idx,current_solution_start_block in enumerate(possible_solutions):
                            if current_solution_start_block>=selected_blob_start_block: # We select the first solution which is within the selected free mem blob there could be better solution if we optimize all solutions that fit smallest blob
                                selected_solution_idx = idx
                                break
                        if selected_solution_idx != None:
                            break
                if selected_blob_start_block == None:
                    raise ValueError ('Something went wrong, didnt find suitable free mem blob')
                if selected_solution_idx == None:
                    raise ValueError ('Something went wrong, didnt find best solution among possible solutions')
            else:
                selected_solution_idx = 0
            allocation_start_block = possible_solutions[selected_solution_idx]
            for j in range(minimal_blocks):
                self.allocated_blocks[allocation_start_block+j] = True
                newly_allocated_blocks.append(allocation_start_block+j)
        return newly_allocated_blocks
    def deallocate_block(self,idx:int):
        if self.allocated_blocks[idx] == False:
            raise ValueError ('Possible error: deallocating AMM block which was not allocated')
        self.allocated_blocks[idx] = False
    def deallocate_blocks(self,indexes_list:list):
        for idx in indexes_list:
            if self.allocated_blocks[idx] == False:
                # This used to be an error, but in cases where two tensors are in the same location
                # (e.g., Add nodes use the same location for the output and 2nd input), this may have
                # already been de-allocated.
                continue
                #raise ValueError ('Possible error: deallocating AMM block which was not allocated')
            self.allocated_blocks[idx] = False
    def allocate_blocks(self,indexes_list:list):
        for idx in indexes_list:
            assert idx >= 0 and idx < self.num_blocks
            if self.allocated_blocks[idx]:
                raise ValueError ('Error: AMM %s cannot allocate block %s, already allocated', self.amm_idx, idx)
            self.allocated_blocks[idx] = True

class BRAMAMMAllocator (AMMAllocator):
    def __init__(self,amm_idx):
        super().__init__(BRAM_BLOCK_SIZE,BRAM_NUM_BLOCKS,amm_idx)

class URAMAMMAllocator (AMMAllocator):
    def __init__(self,amm_idx):
        super().__init__(URAM_BLOCK_SIZE,URAM_NUM_BLOCKS,amm_idx)

class AMM:
    def __init__(self, amm_type:AMMType = AMMType.BRAM_BASED_AMM, amm_idx=0):
        if amm_type == AMMType.BRAM_BASED_AMM:
             self.amm_allocator = BRAMAMMAllocator(amm_idx)
        elif amm_type == AMMType.URAM_BASED_AMM:
             self.amm_allocator = URAMAMMAllocator(amm_idx)
        self.amm_depth = self.amm_allocator.block_size * self.amm_allocator.num_blocks
        self.mem = np.zeros((self.amm_depth,AMM_HEIGHT,AMM_WIDTH))
        self.amm_type = amm_type
    
    def save(self,filename):
        with open(filename,'w') as f:
                for channel in range(self.amm_depth):
                    for h in range(AMM_HEIGHT):
                        line = ''
                        for w in range(AMM_WIDTH):
                            current_byte = int(self.mem[channel,h,w])
                            line=hex(current_byte)[2:].zfill(2)+line # We need to put the 1st byte on rightmost side
                        f.write(line+'\n')
class AMMTensor:
    def __init__(self,tensor,even_grids_amm_blocks,odd_grids_amm_blocks,tile_num,xslice_num,is_valid=False,allocated_at=('',0,0)):
        self.name = tensor.name
        self.tensor = tensor
        self.even_grids_amm_blocks = even_grids_amm_blocks
        self.odd_grids_amm_blocks = odd_grids_amm_blocks
        self.is_28x28 = even_grids_amm_blocks != odd_grids_amm_blocks
        self.deallocating_node = None
        self.tile_num = tile_num
        self.xslice_num = xslice_num
        self.is_valid = is_valid # If false the tensor is allocated in AMM but it is still doesnt have its data set
        self.allocation_timestamp = allocated_at # This is debug helper to see "when" the tensor was allocated. It will hold a tupple of (node_name,tile_num)



class DeviceAMMs:
    def __init__(self):
        self.amms = []
        for i in range(AMM_COUNT):
            if i<NUM_OF_BRAM_AMMS:    
                amm_type = AMMType.BRAM_BASED_AMM
            else:
                amm_type = AMMType.URAM_BASED_AMM
            self.amms.append(AMM(amm_type = amm_type,amm_idx=i))
        self.tensors_in_amm = {}

    def reset_mem(self):
        for amm in self.amms:
            amm.amm_allocator.reset_mem()
        self.tensors_in_amm.clear()            

    def allocate_blocks_in_all_amms(self, blocks):
        for amm in self.amms:
            amm.amm_allocator.allocate_blocks(blocks)

    def allocate_mem(self,node_name,tensor,mem_size,tensors_blob_idx=0,tensors_tile_num=0,current_tile_num=0,current_xslice_num=0,is_valid=True):
        amm_tensor_name = tensor.get_amm_tensor_name(tensors_blob_idx,tensors_tile_num,current_xslice_num)
        if amm_tensor_name in self.tensors_in_amm:
            all_amms_allocated_blocks = self.tensors_in_amm[amm_tensor_name].even_grids_amm_blocks
        else:
            all_amms_allocated_blocks = []
            for amm in self.amms:
                allocated_blocks = amm.amm_allocator.allocate_mem(mem_size)
                all_amms_allocated_blocks.append(allocated_blocks)
                if len(allocated_blocks) == 0:
                    print('Not enough memory in AMM to make allocation. At layer %s' % node_name)
                    return all_amms_allocated_blocks
            self.tensors_in_amm[amm_tensor_name] = AMMTensor(tensor,all_amms_allocated_blocks,all_amms_allocated_blocks,tensors_tile_num,current_xslice_num,is_valid=is_valid,allocated_at=(node_name,current_tile_num,current_xslice_num))
        return all_amms_allocated_blocks
    
    def allocate_contiguous_mem(self,node_name,tensors,mem_sizes,tensors_blob_idx=0,tensors_tile_num=0,current_tile_num=0,num_slices=1,is_valid=True): # This will allocate 14x14 mem for ops like concat that need to allocate mem for 2 tensors
        mem_size = sum(mem_sizes)
        per_input_start_block_index = [0]
        for input_index,input_mem_size in enumerate(mem_sizes):
            if input_index>0:
                last_input_size_in_blocks = int(math.ceil(mem_sizes[input_index-1] / URAM_BLOCK_SIZE))
                per_input_start_block_index.append(last_input_size_in_blocks+per_input_start_block_index[-1])
            if input_index<(len(mem_sizes)-1) and input_mem_size % URAM_BLOCK_SIZE !=0:
                raise ValueError ('Cant allocate contigious mem if 1st tensor size (%d) is not a multiplie of block size (%d)' % (mem_sizes[0],URAM_BLOCK_SIZE))
        
        all_amms_allocated_blocks = []
        for current_xslice_num in range(num_slices):
            xslice_amms_allocated_blocks = []
            for amm in self.amms:
                allocated_blocks = amm.amm_allocator.allocate_mem(mem_size)
                xslice_amms_allocated_blocks.append(allocated_blocks)
                if len(allocated_blocks) == 0:
                    print('Not enough memory in AMM to make allocation. At layer %s' % node_name)
                    return xslice_amms_allocated_blocks, None
            for input_index,input_tensor in enumerate(tensors):
                amm_tensor_name = input_tensor.get_amm_tensor_name(tensors_blob_idx,tensors_tile_num,current_xslice_num)
                blocks_split=list_of_lists_split_at_pos(xslice_amms_allocated_blocks,input_index,per_index_start_pos=per_input_start_block_index)
                if amm_tensor_name not in self.tensors_in_amm:
                    self.tensors_in_amm[amm_tensor_name] = AMMTensor(input_tensor,blocks_split,blocks_split,tensors_tile_num,current_xslice_num,is_valid=is_valid,allocated_at=(node_name,current_tile_num,current_xslice_num))
                else:
                    assigned_blocks = self.tensors_in_amm[amm_tensor_name].even_grids_amm_blocks[0]
                    for amm in self.amms:
                        amm.amm_allocator.deallocate_blocks(assigned_blocks)
                    self.tensors_in_amm[amm_tensor_name].even_grids_amm_blocks = blocks_split
                    self.tensors_in_amm[amm_tensor_name].odd_grids_amm_blocks = blocks_split
            all_amms_allocated_blocks.extend(xslice_amms_allocated_blocks)
        return all_amms_allocated_blocks, per_input_start_block_index

    def deallocate_mem(self,node_name,tensor_name,blocks_to_deallocate:list):
        for amm in self.amms:
            amm.amm_allocator.deallocate_blocks(blocks_to_deallocate)
        if tensor_name in self.tensors_in_amm:
            del self.tensors_in_amm[tensor_name]
        else:
            raise ValueError('Tensor for deallocation not found in amm tensors list. At layer %s' % node_name)

    def deallocate_amm_tensors(self,node,tensors_deallocation_list:TensorDeAllocationList,force_free_mem = False,current_tile_num=0):
        node_name = node['name']
        for (tile_num, xslice_num) in tensors_deallocation_list.tensors_list:
            if (tile_num == current_tile_num): 
                tensors_list_copy = copy.deepcopy(tensors_deallocation_list.tensors_list[(tile_num, xslice_num)]) # We copy the list as in the code we might remove some tensors from it
                for dellocated_tensor in tensors_list_copy:
                    tiled_tensor_name = dellocated_tensor.name
                    ammtensor = self.tensors_in_amm[tiled_tensor_name]
                    free_mem=True
                    if dellocated_tensor.inline_tensor:
                        free_mem = False
                    
                    if force_free_mem or free_mem:
                        for amm in self.amms:
                            amm.amm_allocator.deallocate_blocks(ammtensor.even_grids_amm_blocks[0])

                    if tiled_tensor_name in self.tensors_in_amm:
                        deallocating_node = self.tensors_in_amm[tiled_tensor_name].deallocating_node
                        if deallocating_node:
                            if ('tensors_for_deallocation_after_output_allocation' in deallocating_node['backend']) and \
                                (tiled_tensor_name in tensors_deallocation_list.get_tensor_names_dict()[(tile_num, xslice_num)]):
                                pass
                                #tensors_deallocation_list.tensors_list[current_tile_num].remove(tiled_tensor_name)
                            elif ('tensors_for_deallocation_after_ddr_read_allocation' in deallocating_node['backend']) and \
                                (tiled_tensor_name in tensors_deallocation_list.get_tensor_names_dict()[(tile_num, xslice_num)]):
                                #tensors_deallocation_list.tensors_list[current_tile_num].remove(tiled_tensor_name)
                                pass
                            else:
                                raise ValueError ('Tried to remove amm tensor. but its deallocating node isnt set to deallocate it. Please kindly check this anomality!')
                            del self.tensors_in_amm[tiled_tensor_name]
                        else: # Inline tensors (output of op is written on input of ops) dont have deallocating nodes. They will be allocated by the output tensor
                            ammtensor_to_remove = self.tensors_in_amm[tiled_tensor_name]
                            tensor_consumers = ammtensor_to_remove.tensor.consumers
                            del self.tensors_in_amm[tiled_tensor_name]
                            print('Warning: tried to deallocate tensor without deallocating node. This is possible if its input tensor#0 to add op which is inline so no need to deallocate it.')
                            print('Tensor consumers: %s should be add op' % tensor_consumers)
                    else:
                        raise ValueError('Tensor for deallocation not found in amm tensors list. At layer %s' % node_name)

    def get_amm_tensors_except_current_node_tensors(self,current_node_tensors): # In case we have allocation problems we write all "skip" tensors to DDR and will read them when needed
        skip_tensors = [] # Tensors which are not used by current node
        skip_tensor_names = []
        for tensor_name,tensor in self.tensors_in_amm.items():
            if tensor_name not in current_node_tensors:
                skip_tensors.append(tensor)
                skip_tensor_names.append(tensor_name)
        return skip_tensor_names,skip_tensors

    def add_tile_num_to_tensors(self,current_node_tensors,current_y_tile):
        updated_tensors=[]
        for tensor_name in current_node_tensors:
            updated_tensors.append(tensor_name+'_tile'+str(current_y_tile))
        return updated_tensors

    def add_blob_and_tile_num_to_tensors(self,current_node_tensors,current_blob_num,current_y_tile,current_x_slice):
        updated_tensors=[]
        idx = 0
        for tensor_name in current_node_tensors:
            updated_tensors.append(tensor_name+'_blob'+str(current_blob_num)+'_tile'+str(current_y_tile)+'_xslice'+str(current_x_slice))
            idx += 1
        return updated_tensors

    def print_amm_tensors(self):
        for tensor_name in self.tensors_in_amm:
            amm_tensor = self.tensors_in_amm[tensor_name]
            if amm_tensor.deallocating_node:
                deallocating_node_name = amm_tensor.deallocating_node['name']
            else:
                deallocating_node_name = 'None'
            print('%s: e:%s o:%s de-alloc_node:%s' % (tensor_name,str(amm_tensor.even_grids_amm_blocks),str(amm_tensor.odd_grids_amm_blocks),deallocating_node_name))
    
    def check_mem_integrity(self):
        allocated_blocks=set()
        for tensor_name in self.tensors_in_amm:
            amm_tensor = self.tensors_in_amm[tensor_name]
            allocated_blocks|=set(amm_tensor.even_grids_amm_blocks[0])
        found_issue = False
        allocated_but_not_found_in_amm_tensors = []
        found_in_amm_tensors_but_not_allocated = []
        for block_idx,is_allocated in enumerate(self.amms[0].amm_allocator.allocated_blocks):
            if is_allocated and block_idx not in allocated_blocks:
                allocated_but_not_found_in_amm_tensors.append(block_idx)
                found_issue = True
            if not is_allocated and block_idx in allocated_blocks:
                found_in_amm_tensors_but_not_allocated.append(block_idx)
                found_issue = True
        if found_issue:
            print('allocated_but_not_found_in_amm_tensors: %s' % str(allocated_but_not_found_in_amm_tensors))
            print('found_in_amm_tensors_but_not_allocated: %s' % str(found_in_amm_tensors_but_not_allocated))
            raise ValueError('Found issue in amms allocation')

    def get_current_amms_utilization(self):
        amms_blocks = None
        amms_block_size = None
        for amm in self.amms:
            current_amm_blocks = amm.amm_allocator.allocated_blocks
            current_amm_block_size = amm.amm_allocator.block_size
            if amms_blocks:
                if amms_blocks !=current_amm_blocks:
                    raise ValueError ('AMM blocks allocation is not same for all AMMs, this is currently not allowed. Please check...')
                if amms_block_size !=current_amm_block_size:
                    raise ValueError ('AMM block size is not same for all AMMs, this is currently not allowed. Please check...')
            else:
                amms_blocks = current_amm_blocks
                amms_block_size = current_amm_block_size
        total_allocated_blocks = sum(amms_blocks)
        total_mem_utilization = total_allocated_blocks*amms_block_size
        return total_mem_utilization


class InputAMMconfig:
    def __init__(self,address = 0, amm_mask = 0):
        self.address = address # 12 bits
        self.amm_mask = amm_mask # 4 bits mask to allow simultneous read to up to 4 amms
    def get_info(self):
        int('0b'+ bin(self.amm_mask)[2:].zfill(4) + bin(self.address)[2:].zfill(12),2)


