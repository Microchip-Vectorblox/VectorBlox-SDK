import numpy as np
import math
from common.enums import AMMType
from common.hw_config import AMM_COUNT

from common.hw_config import AMM_WIDTH,BRAM_BLOCK_SIZE,BRAM_NUM_BLOCKS,NUM_OF_BRAM_AMMS
from common.debug_flags import AMM_ALLOCATION_OPTIMIZE_LEFTOUT_SIZE
from common.tensor_ir import TensorDeAllocationList,get_tensor_name_from_tiled_tensor
from common.utils import list_of_lists_split_at_pos

def get_size_sorted_free_mem_blobs(allocated_blocks):
    found_free_blob=False
    blob_size=0
    blob_start_idx=0
    free_blobs = []
    for idx,block in enumerate(allocated_blocks):
        if not block: # current block not allocated
            if found_free_blob: # Already inside a free blob
                blob_size+=1
            else:
                found_free_blob = True # A start of new free mem blob
                blob_size=1
                blob_start_idx=idx
        else: # end of free blob or continue of no blob found
            if found_free_blob: # Already inside a free blob
                free_blobs.append((blob_size,blob_start_idx))
                found_free_blob = False
                blob_size=0
                blob_start_idx=0
    if found_free_blob: #We are in middle of free blob and reached end of mem
        free_blobs.append((blob_size,blob_start_idx))
    free_blobs.sort()
    return free_blobs

class AMMAllocator:
    def __init__(self,block_size,num_blocks,amm_idx):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.allocated_blocks = [False] * self.num_blocks
        self.amm_idx = amm_idx
    def reset_mem(self):
        self.allocated_blocks = [False] * self.num_blocks
    def allocate_mem(self, size, double=False, reserved_blocks=None):
        # Single-pass O(num_blocks) scan for a free run. AMM_ALLOCATION_OPTIMIZE_LEFTOUT_SIZE
        # picks best-fit (smallest fitting run), else first-fit; ties go to leftmost start.
        # reserved_blocks: block indices to treat as occupied for this call only (not actually
        # marked allocated) - keeps output off blocks still holding a sibling's live input.
        minimal_blocks = math.ceil(size / self.block_size)
        if double:
            minimal_blocks *= 2
        blocks = self.allocated_blocks
        n = self.num_blocks
        reserved = reserved_blocks if reserved_blocks is not None else ()

        if AMM_ALLOCATION_OPTIMIZE_LEFTOUT_SIZE:
            best_size = None
            best_start = None
            run_start = None
            for i in range(n):
                if not blocks[i] and i not in reserved:
                    if run_start is None:
                        run_start = i
                else:
                    if run_start is not None:
                        run_len = i - run_start
                        if run_len >= minimal_blocks and (best_size is None or run_len < best_size):
                            best_size = run_len
                            best_start = run_start
                        run_start = None
            if run_start is not None:
                run_len = n - run_start
                if run_len >= minimal_blocks and (best_size is None or run_len < best_size):
                    best_start = run_start
            if best_start is None:
                return []
            chosen_start = best_start
        else:
            chosen_start = None
            run_start = None
            for i in range(n):
                if not blocks[i] and i not in reserved:
                    if run_start is None:
                        run_start = i
                    if i - run_start + 1 >= minimal_blocks:
                        chosen_start = run_start
                        break
                else:
                    run_start = None
            if chosen_start is None:
                return []

        end = chosen_start + minimal_blocks
        for j in range(chosen_start, end):
            blocks[j] = True
        return list(range(chosen_start, end))
    def deallocate_block(self,idx:int):
        if not self.allocated_blocks[idx]:
            raise ValueError ('Possible error: deallocating AMM block which was not allocated')
        self.allocated_blocks[idx] = False
    def deallocate_blocks(self,indexes_list:list):
        for idx in indexes_list:
            if not self.allocated_blocks[idx]:
                # This used to be an error, but in cases where two tensors are in the same location
                # (e.g., Add nodes use the same location for the output and 2nd input), this may have
                # already been de-allocated.
                continue
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
    def __init__(self,amm_idx,amm_block_size,amm_num_blocks):
        super().__init__(amm_block_size,amm_num_blocks,amm_idx)

class AMM:
    def __init__(self, amm_type:AMMType = AMMType.BRAM_BASED_AMM, amm_idx=0,amm_height=16,amm_block_size=16,amm_num_blocks=128):
        if amm_type == AMMType.BRAM_BASED_AMM:
             self.amm_allocator = BRAMAMMAllocator(amm_idx)
        elif amm_type == AMMType.URAM_BASED_AMM:
             self.amm_allocator = URAMAMMAllocator(amm_idx,amm_block_size,amm_num_blocks)
        self.amm_depth = self.amm_allocator.block_size * self.amm_allocator.num_blocks
        self.amm_height = amm_height
        self.mem = np.zeros((self.amm_depth,amm_height,AMM_WIDTH))
        self.amm_type = amm_type
    
    def save(self,filename):
        with open(filename,'w') as f:
                for channel in range(self.amm_depth):
                    for h in range(self.amm_height):
                        line = ''
                        for w in range(AMM_WIDTH):
                            current_byte = int(self.mem[channel,h,w])
                            line=hex(current_byte)[2:].zfill(2)+line # We need to put the 1st byte on rightmost side
                        f.write(line+'\n')
class AMMTensor:
    def __init__(self,tensor,even_grids_amm_blocks,odd_grids_amm_blocks,tile_num,xslice_num,htile_num,is_valid=False,allocated_at=('',0,0,0)):
        self.name = tensor.name
        self.tensor = tensor
        self.even_grids_amm_blocks = even_grids_amm_blocks
        self.odd_grids_amm_blocks = odd_grids_amm_blocks
        self.is_28x28 = even_grids_amm_blocks != odd_grids_amm_blocks
        self.deallocating_node = None
        self.tile_num = tile_num
        self.htile_num = htile_num
        self.xslice_num = xslice_num
        self.is_valid = is_valid # If false the tensor is allocated in AMM but it is still doesnt have its data set
        self.allocation_timestamp = allocated_at # This is debug helper to see "when" the tensor was allocated. It will hold a tupple of (node_name,tile_num,htile_num,xslices_num)



class DeviceAMMs:
    def __init__(self, amm_height, amm_block_size, amm_num_blocks):
        self.amms = []
        self.amm_block_size = amm_block_size
        for i in range(AMM_COUNT):
            if i<NUM_OF_BRAM_AMMS:    
                amm_type = AMMType.BRAM_BASED_AMM
            else:
                amm_type = AMMType.URAM_BASED_AMM
            self.amms.append(AMM(amm_type = amm_type,amm_idx=i,amm_height=amm_height,amm_block_size=amm_block_size,amm_num_blocks=amm_num_blocks))
        self.tensors_in_amm = {}

    def reset_mem(self):
        for amm in self.amms:
            amm.amm_allocator.reset_mem()
        self.tensors_in_amm.clear()

    def allocate_blocks_in_all_amms(self, blocks):
        for amm in self.amms:
            amm.amm_allocator.allocate_blocks(blocks)

    def allocate_mem(self,node_name,tensor,mem_size,tensors_blob_idx=0,tensors_tile_num=0,tensors_htile_num=0,current_tile_num=0,current_htile_num=0,current_xslice_num=0,is_valid=True):
        amm_tensor_name = tensor.get_amm_tensor_name(tensors_blob_idx,tensors_tile_num,tensors_htile_num,current_xslice_num)
        if amm_tensor_name in self.tensors_in_amm:
            all_amms_allocated_blocks = self.tensors_in_amm[amm_tensor_name].even_grids_amm_blocks
        else:
            all_amms_allocated_blocks = []
            for amm in self.amms:
                allocated_blocks = amm.amm_allocator.allocate_mem(mem_size)
                if len(allocated_blocks) == 0:
                    print('Not enough memory in AMM to make allocation. At layer %s' % node_name)
                    for prior_amm, prior_blocks in zip(self.amms, all_amms_allocated_blocks):
                        prior_amm.amm_allocator.deallocate_blocks(prior_blocks)
                    all_amms_allocated_blocks.append(allocated_blocks)
                    return all_amms_allocated_blocks
                all_amms_allocated_blocks.append(allocated_blocks)
            self.tensors_in_amm[amm_tensor_name] = AMMTensor(tensor,all_amms_allocated_blocks,all_amms_allocated_blocks,tensors_tile_num,tensors_htile_num,current_xslice_num,is_valid=is_valid,allocated_at=(node_name,current_tile_num,current_htile_num,current_xslice_num))
        return all_amms_allocated_blocks
    
    def allocate_contiguous_mem(self,node_name,tensors,mem_sizes,tensors_blob_idx=0,tensors_tile_num=0,tensors_htile_num=0,current_tile_num=0,current_htile_num=0,num_slices=1,is_valid=True,reserved_blocks=None): # This will allocate 14x14 mem for ops like concat that need to allocate mem for 2 tensors
        # Each input starts on a block boundary, so the run must reserve ceil(size/block) blocks
        # PER input, not ceil(sum(sizes)/block) - the raw sum under-allocates when a non-last
        # input isn't block-aligned, overrunning the run.
        mem_size = sum(int(math.ceil(s / self.amm_block_size)) for s in mem_sizes) * self.amm_block_size
        per_input_start_block_index = [0]
        for input_index,input_mem_size in enumerate(mem_sizes):
            if input_index>0:
                last_input_size_in_blocks = int(math.ceil(mem_sizes[input_index-1] / self.amm_block_size))
                per_input_start_block_index.append(last_input_size_in_blocks+per_input_start_block_index[-1])
        # Integrity guard: the per-input block slices must exactly fill the reserved run.
        required_blocks = per_input_start_block_index[-1] + int(math.ceil(mem_sizes[-1] / self.amm_block_size))
        if required_blocks != mem_size // self.amm_block_size:
            raise ValueError('Contiguous AMM allocation is inconsistent: inputs %s need %d blocks but %d were reserved (block size %d)'
                             % (mem_sizes, required_blocks, mem_size // self.amm_block_size, self.amm_block_size))
        
        all_amms_allocated_blocks = []
        registered_amm_tensor_names = []
        for current_xslice_num in range(num_slices):
            xslice_amms_allocated_blocks = []
            for amm in self.amms:
                allocated_blocks = amm.amm_allocator.allocate_mem(mem_size, reserved_blocks=reserved_blocks)
                if len(allocated_blocks) == 0:
                    print('Not enough memory in AMM to make allocation. At layer %s' % node_name)
                    for prior_amm, prior_blocks in zip(self.amms, xslice_amms_allocated_blocks):
                        prior_amm.amm_allocator.deallocate_blocks(prior_blocks)
                    for prior_amm_blocks in all_amms_allocated_blocks:
                        for prior_amm, prior_blocks in zip(self.amms, prior_amm_blocks):
                            prior_amm.amm_allocator.deallocate_blocks(prior_blocks)
                    for prior_name in registered_amm_tensor_names:
                        del self.tensors_in_amm[prior_name]
                    xslice_amms_allocated_blocks.append(allocated_blocks)
                    return xslice_amms_allocated_blocks, None
                xslice_amms_allocated_blocks.append(allocated_blocks)
            for input_index,input_tensor in enumerate(tensors):
                amm_tensor_name = input_tensor.get_amm_tensor_name(tensors_blob_idx,tensors_tile_num,tensors_htile_num,current_xslice_num)
                blocks_split=list_of_lists_split_at_pos(xslice_amms_allocated_blocks,input_index,per_index_start_pos=per_input_start_block_index)
                if amm_tensor_name not in self.tensors_in_amm:
                    self.tensors_in_amm[amm_tensor_name] = AMMTensor(input_tensor,blocks_split,blocks_split,tensors_tile_num,tensors_htile_num,current_xslice_num,is_valid=is_valid,allocated_at=(node_name,current_tile_num,current_htile_num,current_xslice_num))
                    registered_amm_tensor_names.append(amm_tensor_name)
                else:
                    assigned_blocks = self.tensors_in_amm[amm_tensor_name].even_grids_amm_blocks[0]
                    for amm in self.amms:
                        amm.amm_allocator.deallocate_blocks(assigned_blocks)
                    self.tensors_in_amm[amm_tensor_name].even_grids_amm_blocks = blocks_split
                    self.tensors_in_amm[amm_tensor_name].odd_grids_amm_blocks = blocks_split
            all_amms_allocated_blocks.append(xslice_amms_allocated_blocks)
        flattened = []
        for xslice_blocks in all_amms_allocated_blocks:
            flattened.extend(xslice_blocks)
        return flattened, per_input_start_block_index

    def evict_all_except_system_tensors(self, keep_name='SYSTEM_zero_tensor'):
        # Frees every resident AMM tensor except the system-reserved zero block. Used by the
        # single-node-blob pre-split fallback: gives the lone node the whole AMM when h-tiling
        # is exhausted; the node re-reads its input from DDR normally.
        for tensor_name in [n for n in self.tensors_in_amm if n != keep_name]:
            ammtensor = self.tensors_in_amm[tensor_name]
            for amm_idx, amm in enumerate(self.amms):
                amm.amm_allocator.deallocate_blocks(ammtensor.even_grids_amm_blocks[amm_idx])
            del self.tensors_in_amm[tensor_name]

    def deallocate_mem(self,node_name,tensor_name,blocks_to_deallocate:list):
        for amm in self.amms:
            amm.amm_allocator.deallocate_blocks(blocks_to_deallocate)
        if tensor_name in self.tensors_in_amm:
            del self.tensors_in_amm[tensor_name]
        else:
            raise ValueError('Tensor for deallocation not found in amm tensors list. At layer %s' % node_name)

    def deallocate_amm_tensors(self,node,tensors_deallocation_list:TensorDeAllocationList,force_free_mem=False,current_tile_num=0,current_htile_num=0):
        node_name = node['name']
        for (tile_num, h_tile_num, xslice_num) in tensors_deallocation_list.tensors_list:
            if (tile_num == current_tile_num) and (h_tile_num == current_htile_num):
                # Shallow copy is enough: only the list itself may be mutated by the loop body,
                # not the TensorDeAllocationInfo elements (their fields are read-only here).
                tensors_list_copy = list(tensors_deallocation_list.tensors_list[(tile_num, h_tile_num, xslice_num)])
                for deallocated_tensor in tensors_list_copy:
                    tiled_tensor_name = deallocated_tensor.name
                    ammtensor = self.tensors_in_amm.get(tiled_tensor_name)
                    if ammtensor is None:
                        continue
                    free_mem=True
                    if deallocated_tensor.inline_tensor:
                        free_mem = False

                    if force_free_mem or free_mem:
                        for amm in self.amms:
                            amm.amm_allocator.deallocate_blocks(ammtensor.even_grids_amm_blocks[0])

                    if tiled_tensor_name in self.tensors_in_amm:
                        deallocating_node = self.tensors_in_amm[tiled_tensor_name].deallocating_node
                        if deallocating_node:
                            if ('tensors_for_deallocation_after_output_allocation' in deallocating_node['backend']) and \
                                (tiled_tensor_name in tensors_deallocation_list.get_tensor_names_dict()[(tile_num, h_tile_num, xslice_num)]):
                                pass
                            elif ('tensors_for_deallocation_after_ddr_read_allocation' in deallocating_node['backend']) and \
                                (tiled_tensor_name in tensors_deallocation_list.get_tensor_names_dict()[(tile_num, h_tile_num, xslice_num)]):
                                pass
                            else:
                                raise ValueError ('Tried to remove amm tensor. but its deallocating node isnt set to deallocate it. Please kindly check this anomality!')
                            del self.tensors_in_amm[tiled_tensor_name]
                        else: # Inline tensors (output of op is written on input of ops) dont have deallocating nodes. They will be allocated by the output tensor
                            ammtensor_to_remove = self.tensors_in_amm[tiled_tensor_name]
                            tensor_consumers = ammtensor_to_remove.tensor.consumers
                            del self.tensors_in_amm[tiled_tensor_name]
                    else:
                        raise ValueError('Tensor for deallocation not found in amm tensors list. At layer %s' % node_name)

    def get_amm_tensors_except_current_node_tensors(self,current_node_tensors): # In case we have allocation problems we write all "skip" tensors to DDR and will read them when needed
        # Set lookup so the "not in" check is O(1) when the caller passes a list.
        current_set = current_node_tensors if isinstance(current_node_tensors, (set, frozenset)) else set(current_node_tensors)
        skip_tensor_names = []
        skip_tensors = []
        for tensor_name, tensor in self.tensors_in_amm.items():
            if tensor_name not in current_set:
                skip_tensor_names.append(tensor_name)
                skip_tensors.append(tensor)
        return skip_tensor_names, skip_tensors

    def _suffix_tensors(self,current_node_tensors,suffix):
        return [name + suffix for name in current_node_tensors]

    def add_tile_num_to_tensors(self,current_node_tensors,current_y_tile):
        return self._suffix_tensors(current_node_tensors, f'_tile{current_y_tile}')

    def add_blob_and_tile_num_to_tensors(self,current_node_tensors,current_blob_num,current_y_tile,current_h_tile,current_x_slice):
        return self._suffix_tensors(current_node_tensors, f'_blob{current_blob_num}_tile{current_y_tile}_htile{current_h_tile}_xslice{current_x_slice}')

    def print_amm_tensors(self):
        for tensor_name in self.tensors_in_amm:
            amm_tensor = self.tensors_in_amm[tensor_name]
            if amm_tensor.deallocating_node:
                deallocating_node_name = amm_tensor.deallocating_node['name']
            else:
                deallocating_node_name = 'None'
            print('%s: e:%s o:%s de-alloc_node:%s' % (tensor_name,str(amm_tensor.even_grids_amm_blocks),str(amm_tensor.odd_grids_amm_blocks),deallocating_node_name))
    
    def check_mem_integrity(self):
        allocated_blocks = set()
        for amm_tensor in self.tensors_in_amm.values():
            allocated_blocks.update(amm_tensor.even_grids_amm_blocks[0])
        allocated_but_not_found_in_amm_tensors = []
        found_in_amm_tensors_but_not_allocated = []
        for block_idx, is_allocated in enumerate(self.amms[0].amm_allocator.allocated_blocks):
            in_set = block_idx in allocated_blocks
            if is_allocated and not in_set:
                allocated_but_not_found_in_amm_tensors.append(block_idx)
            elif not is_allocated and in_set:
                found_in_amm_tensors_but_not_allocated.append(block_idx)
        if allocated_but_not_found_in_amm_tensors or found_in_amm_tensors_but_not_allocated:
            print('allocated_but_not_found_in_amm_tensors: %s' % str(allocated_but_not_found_in_amm_tensors), flush=True)
            print('found_in_amm_tensors_but_not_allocated: %s' % str(found_in_amm_tensors_but_not_allocated), flush=True)
            for tensor_name, amm_tensor in self.tensors_in_amm.items():
                if any(b in found_in_amm_tensors_but_not_allocated for b in amm_tensor.even_grids_amm_blocks[0]):
                    print('DEBUG culprit tensor=%s blocks=%s allocation_timestamp=%s deallocating_node=%s' % (
                        tensor_name, amm_tensor.even_grids_amm_blocks[0], amm_tensor.allocation_timestamp,
                        amm_tensor.deallocating_node['name'] if amm_tensor.deallocating_node else None), flush=True)
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
        return int('0b'+ bin(self.amm_mask)[2:].zfill(4) + bin(self.address)[2:].zfill(12),2)


