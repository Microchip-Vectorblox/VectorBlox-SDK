import networkx as nx
from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle
from common.ddr_ir import DDR, DDREntry
from common.sequencer_ir import SequencerProgram
from common.enums import DDREntryType,GridConfig
from common.hw_config import NUM_MUTEX_FLAGS, NUM_TABLE_BUFFERS, MULTIPLE_INPUT_OPS, DEBUG_2MAXPOOL_GRIDS, DUAL_CONTIGUOUS_ALLOCATION_OPS
from common.debug_flags import DEBUG_AUTO_Y_FOLDING, DEBUG_FORCE_Y_FOLDING_N_512, DEBUG_FORCE_Y_UNFOLDING_N_512, DEBUG_FORCE_Y_FOLDING_N_256,\
                             DEBUG_FORCE_Y_UNFOLDING_N_256, DEBUG_FORCE_Y_FOLDING_N_416, DEBUG_FORCE_Y_UNFOLDING_N_416, DEBUG_FORCE_Y_FOLDING_M_256,\
                             DEBUG_FORCE_Y_UNFOLDING_M_256, DEBUG_FORCE_Y_UNFOLDING_S_256, DEBUG_FORCE_Y_FOLDING_S_256, DEBUG_FORCE_Y_UNFOLDING_M_416,\
                                  DEBUG_FORCE_Y_FOLDING_M_416,\
                                  DEBUG_2_TILE_RW_LINES
from common.amm_ir import DeviceAMMs
from common.tensor_ir import Tensor
from common.utils import get_y_tile_sizes, get_y_tile_sizes_2_tile_case
import copy

class WaitFlagsAllocator:
    def __init__(self,num_flags):
        self.num_flags = num_flags
        self.allocated_flags = [False for i in range(self.num_flags)]
        self.allocation_history = []
    def allocate_flag(self,description=''):
        allocated_flag=0
        found = False
        for j in range(self.num_flags):
            if self.allocated_flags[j] == False:
                found=True
                allocated_flag = j
                self.allocated_flags[j] = True
                break
        if not found:
            print(*self.allocation_history,sep='\n')
            raise ValueError ('Error trying to allocate mutex flag. all flags are used!. See allocation history above')
        allocation_string = 'Flag %d allocated:' % allocated_flag + description
        self.allocation_history.append(allocation_string)
        return allocated_flag
    def deallocate_flag(self,idx):
        self.allocated_flags[idx] = False
    def deallocate_flags(self,flags_list):
        for idx in flags_list:
            self.allocated_flags[idx] = False
    def deallocate_flags_from_mask(self,flags_mask):
        for idx in range(self.num_flags):
            current_mask = 2 ** idx
            if (flags_mask & current_mask):
                self.allocated_flags[idx] = False
    def get_mask_from_flags(self,flags_list):
        mask = 0
        for flag in flags_list:
            mask = mask | (2 ** flag)
        return mask
class TablesBufferIDAllocator:
    def __init__(self,num_buffers):
        self.num_buffers = num_buffers
        self.allocated_buffers = [False for i in range(self.num_buffers)]
    def allocate_buffer(self):
        allocated_buffer=0
        found = False
        for j in range(self.num_buffers):
            if self.allocated_buffers[j] == False:
                found=True
                allocated_buffer = j
                self.allocated_buffers[j] = True
                break
        if not found:
            raise ValueError ('Error trying to allocate tables buffer. all flags are used!!!!')
        return allocated_buffer
    def deallocate_buffer(self,idx):
        self.allocated_buffers[idx] = False
    def deallocate_buffers(self,buffers_list):
        for idx in buffers_list:
            self.allocated_buffers[idx] = False
    def get_allocated_buffers_indexes(self):
        ret_val = [i for i,buffer_allocated in enumerate(self.allocated_buffers) if buffer_allocated==True]
        return ret_val

class IntermediateDDRTensor:
    def __init__(self,skip_tensor_name,offloading_node_name=''):
        self.skip_tensor_name = skip_tensor_name
        self.offloading_node_name = offloading_node_name
        self.write_wait_flag = None
        self.read_wait_flag = None

class TilingBlob:
    def __init__(self,nodes_in_blob,num_y_tiles=1,num_x_slices=1,read_next_tile_node_idx=0,write_finished_tile_node_idx=0,k3_nodes=0):
        self.nodes_in_blob=nodes_in_blob
        self.num_of_nodes_in_blob=len(nodes_in_blob)
        self.y_tiles = num_y_tiles
        self.x_slices = num_x_slices
        self.read_next_tile_node_idx = read_next_tile_node_idx
        self.write_last_tile_node_idx = write_finished_tile_node_idx
        self.k3_nodes=k3_nodes # We need to know how many k3 nodes are in blob to know how many lines are reduced until we write it blob output to ddr
        self.is_y_folding_blob = False
        self.next_tile_read_node_idx = 0
        self.inputs = []
        self.outputs = []
        self.num_lines_written_to_ddr_before_next_blob_read = 0
    def get_blob_outputs_names(self):
        names = []
        for output in self.outputs:
            names.append(output.name)
        return names

class IR:
    def __init__(self,model_name,compiler_output_dir='',uint8_flag=False,debug=False,mean=None,scale=None):
        self.model_name = model_name
        self.compiler_output_dir = compiler_output_dir
        self.graph = nx.DiGraph() # The graph is represented using networkx pacakge.
                                # Each node in the graph will have the following attributes
                                # op_type - String, This is the op type (e.g. Conv) as defined in onnx operators (https://github.com/onnx/onnx/blob/main/docs/Operators.md)
                                # attributes - Dictionary, This is a dictionary containing the attribures of each node
                                # backend - a dictionary containing all backend related metadata per each op:
                                #   gridmode
                                #   input_channels_split
                                #   output_channels_split
        self.lexicographical_topological_sorted_graph = []
        self.tensors = {} # A list of tensors in the graph. Each tensor is of class Tensor
        self.inputs = [] # A list of tensor names which are inputs to the graph (This does not include initializers (e.g. weights))
        self.outputs = [] # A list of tensor names of the graphs output
        self.ddr = DDR('PROGRAM_BASE_ADDRESS')
        self.inputs_ddr = DDR('INPUTS_BASE_ADDRESS')
        self.outputs_ddr = DDR('OUTPUTS_BASE_ADDRESS')
        self.amms = DeviceAMMs()
        ddr_entry_description = "First page of DDR is used for various registers such as the program start address"
        registers_page = bytearray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        registers_ddr_entry = DDREntry(registers_page,type=DDREntryType.REGISTER_PAGE,description=ddr_entry_description) # Create first page (4096 bytes at start of ddr). This will contain registers (e.g. program start address)
        self.ddr.add_entry(registers_ddr_entry)
        self.wait_flags_allocator = WaitFlagsAllocator(NUM_MUTEX_FLAGS)
        self.single_split_tables_buffer_allocator = TablesBufferIDAllocator(NUM_TABLE_BUFFERS) # This is used for rqparams and write mask tables as they dont have splits
        self.splitted_tables_buffer_allocator = TablesBufferIDAllocator(NUM_TABLE_BUFFERS) # This allocator is used for wlocs and rqlocs tables as they have splits (always same number of splits to both)
        self.intermediate_ddr_tensors = {}
        self.past_wloc_data = None
        self.marked_nodes_for_folding_x = [] # These will be automatically set by "calc_concat_folding_factor"
        self.marked_nodes_for_folding_y = [] # These will be automatically set by "calc_concat_folding_factor"
        self.marked_nodes_for_output_split = [] # This will be set by "program_compiler"
        if DEBUG_AUTO_Y_FOLDING:
            self.force_y_folding = []
            self.force_y_unfolding = []
        elif 'yolov5n6' in model_name:
            self.force_y_folding = ['Concat_32','Concat_70','Conv_168']
            self.force_y_unfolding = ['Conv_110','Conv_182']
        elif 'MaxPool16x16' in model_name:
            self.force_y_folding = ['MaxPool1']
            self.force_y_unfolding = []
        elif 'yolov5n' in model_name and '256' in model_name:
            self.force_y_folding = DEBUG_FORCE_Y_FOLDING_N_256
            self.force_y_unfolding = DEBUG_FORCE_Y_UNFOLDING_N_256
        elif 'yolov5n' in model_name and '512' in model_name:
            self.force_y_folding = DEBUG_FORCE_Y_FOLDING_N_512 # Yaron
            self.force_y_unfolding = DEBUG_FORCE_Y_UNFOLDING_N_512 # Yaron
        elif 'yolov5n' in model_name and '416' in model_name:
            self.force_y_folding = DEBUG_FORCE_Y_FOLDING_N_416 # Yaron
            self.force_y_unfolding = DEBUG_FORCE_Y_UNFOLDING_N_416 # Yaron
        elif 'yolov5m' in model_name and '256' in model_name:
            self.force_y_folding = DEBUG_FORCE_Y_FOLDING_M_256 # Yaron
            self.force_y_unfolding = DEBUG_FORCE_Y_UNFOLDING_M_256 # Yaron
        elif 'yolov5m' in model_name and '416' in model_name:
            self.force_y_folding = DEBUG_FORCE_Y_FOLDING_M_416 # Yaron
            self.force_y_unfolding = DEBUG_FORCE_Y_UNFOLDING_M_416 # Yaron
        elif 'yolov5s' in model_name and '256' in model_name:
            self.force_y_folding = DEBUG_FORCE_Y_FOLDING_S_256 # Yaron
            self.force_y_unfolding = DEBUG_FORCE_Y_UNFOLDING_S_256 # Yaron
        elif 'two_tile_test' in model_name:
            self.force_y_folding = []
            self.force_y_unfolding = []
        elif 'two_tile_alternate_test' in model_name:
            # 2 tiles for first/third blob, 1 tile for second/fourth blob
            self.force_y_folding = ['BLOBNp1_conv1', 'BLOBNp3_conv1',]
            self.force_y_unfolding = ['BLOBNp2_conv1']
        if DEBUG_2MAXPOOL_GRIDS:
            self.force_x_folding = ['']
        if 'y_folding_test' in model_name:
            self.force_y_folding.append('conv2')
        if 'y_unfolding_test' in model_name:
            self.force_y_unfolding.append('conv2')

        self.offloaded_concat_ops = []
        self.tiling_blobs = OrderedDict()
        self.axi0_tables_load_flag = 0
        self.axi1_tables_load_flag = 0
        self.local_ddr_transaction_flag = 0
        self.global_ddr_transaction_flag = 0
       
        self.sequencer_program = SequencerProgram()
        self.nodes_to_split_failing_blobs = set()

        # MXP related
        self.sync_with_MXP = False
        self.tensors_from_mxp = set()
        self.tensors_to_mxp = set()
        self.mxp_tensor_to_offset = {}

        # Data structures for splitting large convolutions
        self.original_tensor_to_split_tensor_map = {}
        self.split_tensor_to_original_tensor_map = {}
        self.uint8_int8_conversion = (uint8_flag.strip().lower() == "true") if isinstance(uint8_flag, str) else uint8_flag
        self.uint8_int8_lut = None
        self.mean = mean
        self.scale = scale
        self.debug = (debug.strip().lower() == "true") if isinstance(debug, str) else debug
        self.input_ddr_offset = set()
        self.output_ddr_offset = set()

    def save(self,filename: str):
        with open(filename, 'wb') as filehandler:
            pickle.dump(self, filehandler)

    def save_numsim_ir(self,filename: str):
        numsim_ir = IR(self.model_name)
        numsim_ir.graph = copy.deepcopy(self.graph)
        numsim_ir.lexicographical_topological_sorted_graph = copy.deepcopy(self.lexicographical_topological_sorted_graph)
        numsim_ir.uint8_int8_lut = copy.deepcopy(self.uint8_int8_lut)
        numsim_ir.tensors = copy.deepcopy(self.tensors)
        numsim_ir.inputs = copy.deepcopy(self.inputs)
        numsim_ir.outputs = copy.deepcopy(self.outputs)
        numsim_ir.debug = copy.deepcopy(self.debug)
        numsim_ir.input_ddr_offset = copy.deepcopy(self.input_ddr_offset)
        numsim_ir.output_ddr_offset = copy.deepcopy(self.output_ddr_offset)
        for node_name in self.graph.nodes:
            current_node = self.graph.nodes[node_name]
            for backend_field_name in current_node['backend']:
                if backend_field_name not in ['oc_order','simulator_oc_order','per_ic_group_sorted_weight_activation_pairs','ic_splits','ic_groups','input_channels_reorder_dict','nlf','output_padding_start_y','output_padding_start_x']:
                    del(numsim_ir.graph.nodes[node_name]['backend'][backend_field_name])
        with open(filename, 'wb') as filehandler:
            pickle.dump(numsim_ir, filehandler)

    def load(self,filename: str):
        filehandler = open(filename, 'rb') 
        ir = pickle.load(filehandler)
        return ir
    def is_k3_node(self,node):
        if 'kernel_size' in node['frontend']:
            is_k3_node_val = node['frontend']['kernel_size']>1
        else:
            is_k3_node_val = False
        return is_k3_node_val

    def get_non_constant_inputs(self,node):
        non_constant_inputs=[]
        for node_input_name in node['inputs']:
            if not self.tensors[node_input_name].is_constant:
                non_constant_inputs.append(node_input_name)
        return non_constant_inputs
    
    def get_nodes_output_qparams(self,node):
        node_output_tensor_name = node['outputs'][0]
        output_tensor = self.tensors[node_output_tensor_name]
        scale = output_tensor.scale
        zp = output_tensor.zero_point
        return (scale,zp)
    
    def switch_input_name(self,node,original_input_name = '',new_input_name=''):
        original_input_name_found = False
        for idx,input_name in enumerate(node['inputs']):
            if input_name==original_input_name:
                original_input_name_found = True
                break
        if not original_input_name_found:
            raise ValueError ('Couldnt find original input name. Please check model integrity.')
        node['inputs'][idx] = new_input_name

    def switch_input_tensor(self,node,original_input_tensor = None,new_input_tensor=None):
        original_input_tensor_found = False
        if node['op_type'] in MULTIPLE_INPUT_OPS:
            for idx,input_tensor in enumerate(node['frontend']['input_tensors']):
                if input_tensor==original_input_tensor:
                    original_input_tensor_found = True
                    break
            if not original_input_tensor_found:
                raise ValueError ('Couldnt find original input tensor. Please check model integrity.')
            node['frontend']['input_tensors'][idx] = new_input_tensor
        else:
            if node['frontend']['input_tensor'] == original_input_tensor:
                node['frontend']['input_tensor'] = new_input_tensor
            else:
                raise ValueError ('Couldnt find original input tensor. Please check model integrity.')

    def switch_tensor_consumer(self,tensor:Tensor,original_node_name='',new_node_name=''):
        original_node_found=False
        for idx,consumer_node_name in enumerate(tensor.consumers):
            if consumer_node_name == original_node_name:
                original_node_found = True
                break
        if not original_node_found:
            raise ValueError ('Couldnt find original consumer node. Please check model integrity')
        tensor.consumers[idx] = new_node_name


    # Since count on following_nodes_params to be sorted by execution order the below will update it accordingly if graph is updated
    def get_updated_following_nodes(self,node):
        sorted_graph = list(nx.lexicographical_topological_sort(self.graph))
        following_nodes_names = list(self.graph.successors(node['name']))
        if len(following_nodes_names)>1: # We want the following nodes struct to be lexicographical topological ordered (so that we can easily know which is last/first following node)
            sorted_following_nodes_names = sorted(following_nodes_names, key=lambda x: sorted_graph.index(x))
        else:
            sorted_following_nodes_names = following_nodes_names

        following_nodes_params = []
        for following_node_name in sorted_following_nodes_names:
            following_node = self.graph.nodes[following_node_name]
            if (len(node['outputs']) > 0):
                for output_idx in range(len(node['outputs'])):
                    output_tensor_name = node['outputs'][output_idx].split('_split')[0]
                    if (output_tensor_name in following_node['inputs']):
                        following_node_input_index = following_node['inputs'].index(output_tensor_name)
            following_nodes_params.append((following_node_name,following_node_input_index))
        return following_nodes_params

    # Since count on preceding_nodes_params to be sorted by execution order the below will update it accordingly if graph is updated
    def get_updated_preceding_nodes(self,node):
        sorted_graph = list(nx.lexicographical_topological_sort(self.graph))
        preceding_nodes_names = list(self.graph.predecessors(node['name']))
        if len(preceding_nodes_names)>1: # We want the preceding nodes struct to be lexicographical topological ordered (so that we can easily know which is last/first following node)
            sorted_preceding_nodes_names = sorted(preceding_nodes_names, key=lambda x: sorted_graph.index(x))
        else:
            sorted_preceding_nodes_names = preceding_nodes_names

        preceding_nodes_params = []
        for preceding_node_name in sorted_preceding_nodes_names:
            preceding_node = self.graph.nodes[preceding_node_name]
            preceding_node_input_index = node['inputs'].index(preceding_node['outputs'][0])
            preceding_nodes_params.append((preceding_node_name,preceding_node_input_index))
        return preceding_nodes_params

    def get_next_executed_node(self,node,error_on_last_node=True,current_tile_num=0,current_xslice_num=0):
        execution_sorted_graph = self.lexicographical_topological_sorted_graph
        current_node_execution_index = execution_sorted_graph.index(node['name'])
        current_node_blob_idx = node['frontend']['tiling_blob_idx']
        current_blob = self.tiling_blobs[current_node_blob_idx]
        tiles_in_current_blob = current_blob.y_tiles
        ret_val = True
        last_node_in_blob = node['name']==current_blob.nodes_in_blob[-1]
        num_xslices = node['frontend']['x_slices']
        if last_node_in_blob:
            if current_tile_num==tiles_in_current_blob-1: # If we are in last tile of last node in blob
                if (current_node_execution_index+1)>=len(execution_sorted_graph): # If its also the last node in graph
                    following_executed_node_index = current_node_execution_index # We return current node
                    next_executed_tile = current_tile_num
                    if error_on_last_node:
                        ret_val = False
                else: #its not the last blob in graph
                    following_executed_node_index = current_node_execution_index+1 # We return next executed node and restart from tile 0
                    next_executed_tile = 0
            else: # Its last node in blob but not last tile in that blob
                following_executed_node_name = current_blob.nodes_in_blob[0] # We return first node in blob and tile_num+1
                following_executed_node_index = execution_sorted_graph.index(following_executed_node_name)
                next_executed_tile = current_tile_num+1
        else: # Its not last node in blob
            following_executed_node_index = current_node_execution_index+1 # We return next executed node
            next_executed_tile = current_tile_num

        if (current_xslice_num == num_xslices-1):
            next_executed_xslice = 0
        else:
            next_executed_xslice = current_xslice_num + 1

        following_executed_node_name = execution_sorted_graph[following_executed_node_index]
        following_executed_node = self.graph.nodes[following_executed_node_name]
        return ret_val,following_executed_node,next_executed_tile,next_executed_xslice

    def get_blob_output_dealocating_node(self,current_node_blob_idx,current_tile_num, current_xslice_num, num_xslices):
        current_blob = self.tiling_blobs[current_node_blob_idx]
        current_blob_num_tiles = current_blob.y_tiles
        nodes_in_current_blob = len(current_blob.nodes_in_blob)
        if current_tile_num != current_blob_num_tiles-1: # If we are NOT at last tile the deallocating node will be in current blob, next tile
            deallocating_node_idx = (nodes_in_current_blob-1) // 2
            if (deallocating_node_idx > 0):
                deallocating_node_idx -= 1
                if (self.graph.nodes[current_blob.nodes_in_blob[deallocating_node_idx]]['op_type'] == 'Concat'):
                    deallocating_node_idx += 1
            deallocating_node_name = current_blob.nodes_in_blob[deallocating_node_idx]
            deallocating_node = self.graph.nodes()[deallocating_node_name]
            deallocation_tile_idx = current_tile_num + 1
        else:# If we are at last tile the deallocating node will be in next blob, tile 0
            if current_node_blob_idx!=len(self.tiling_blobs)-1: # We are not in last blob
                next_blob = self.tiling_blobs[current_node_blob_idx+1]
                nodes_in_next_blob = len(next_blob.nodes_in_blob)
                deallocating_node_idx = (nodes_in_next_blob - 1) // 2
                if (deallocating_node_idx > 0):
                    deallocating_node_idx -= 1
            
                # If the first node of the next blob is a Concat, delay the de-allocation.
                # For now this is done for size 2 blobs.
                if self.graph.nodes[next_blob.nodes_in_blob[0]]['op_type'] == 'Concat':
                    if nodes_in_next_blob == 2:
                        deallocating_node_idx = nodes_in_next_blob // 2
                
                if (self.graph.nodes[next_blob.nodes_in_blob[deallocating_node_idx]]['op_type'] == 'Concat') or \
                    (self.graph.nodes[next_blob.nodes_in_blob[deallocating_node_idx]]['op_type'] == 'Add'):
                    deallocating_node_idx = deallocating_node_idx + 1
                    if (deallocating_node_idx >= nodes_in_next_blob):
                        deallocating_node_idx = 0
                        next_blob = self.tiling_blobs[current_node_blob_idx+2]
                deallocating_node_name = next_blob.nodes_in_blob[deallocating_node_idx]
                deallocating_node = self.graph.nodes()[deallocating_node_name]
                deallocation_tile_idx = 0
            else:
                deallocating_node_idx = nodes_in_current_blob - 1
                deallocating_node_name = current_blob.nodes_in_blob[deallocating_node_idx]
                deallocating_node = self.graph.nodes()[deallocating_node_name]
                deallocation_tile_idx = current_tile_num
        if (current_xslice_num == num_xslices-1):
            deallocation_xslice_idx = 0
        else:    
            deallocation_xslice_idx = current_xslice_num + 1

        return deallocating_node, deallocation_tile_idx, deallocation_xslice_idx

    def get_dominant_tensor_consumer_in_blob(self,next_tile_blob,input_tensor):
        # This will return the dominant consumer of the tensor in the specified blob
        # By dominant we choose concat over other consumers since in case of concat we need to allocate all its inputs
        # at once
        consumer_index=0
        consumer_node_names = input_tensor.consumers
        found_consumer_in_blob = False
        selected_consumer_node_name = ''
        consumer_is_concat = False
        while consumer_index<len(consumer_node_names):
            current_consumer_node_name = consumer_node_names[consumer_index]
            current_consumer_node = self.graph.nodes()[current_consumer_node_name]
            current_consumer_op_type = current_consumer_node['op_type']
            if current_consumer_node_name in next_tile_blob.nodes_in_blob:
                if found_consumer_in_blob:
                    if current_consumer_op_type in MULTIPLE_INPUT_OPS:
                        if consumer_is_concat: # We already found a consumer which was a concat
                            raise ValueError ('Cant have 2 concats consume same input in same blob')
                        else: # prefer the concat consumer over the already found consumer
                            consumer_is_concat = True
                            selected_consumer_node_name = current_consumer_node_name
                    else:
                        pass
                else:
                    found_consumer_in_blob = True
                    if current_consumer_op_type in MULTIPLE_INPUT_OPS:
                        consumer_is_concat = True
                    selected_consumer_node_name = current_consumer_node_name
            consumer_index=consumer_index+1
        return found_consumer_in_blob,selected_consumer_node_name

    def get_last_tensor_consumer_in_blob(self,next_tile_blob,input_tensor):
        found_consumer_in_blob = False
        consumer_node_name = ''
        consumer_node_names = input_tensor.consumers

        blob_node_idx = []
        for consumer_idx in range(len(consumer_node_names)):
            try:
                blob_node_idx.append(next_tile_blob.nodes_in_blob.index(consumer_node_names[consumer_idx]))
            except ValueError:
                found_consumer_in_blob = False
        
        found_consumer_in_blob = False
        if (len(blob_node_idx)>0):
            found_consumer_in_blob = True
            consumer_node_name = next_tile_blob.nodes_in_blob[max(blob_node_idx)]
        
        # if len(consumer_node_names)>0:
        #     consumer_index=len(consumer_node_names)-1
        #     found_consumer_in_blob = False
        #     consumer_node_name = ''
        #     while consumer_index>=0:
        #         if consumer_node_names[consumer_index] in next_tile_blob.nodes_in_blob:
        #             found_consumer_in_blob = True
        #             consumer_node_name = consumer_node_names[consumer_index]
        #             break
        #         consumer_index=consumer_index-1
        return found_consumer_in_blob,consumer_node_name

    def get_first_tensor_consumer_in_blob(self,next_tile_blob,input_tensor):
        found_consumer_in_blob = False
        consumer_node_name = ''
        consumer_node_names = input_tensor.consumers
        if len(consumer_node_names)>0:
            consumer_index=0
            found_consumer_in_blob = False
            consumer_node_name = ''
            while consumer_index<len(consumer_node_names):
                if consumer_node_names[consumer_index] in next_tile_blob.nodes_in_blob:
                    found_consumer_in_blob = True
                    consumer_node_name = consumer_node_names[consumer_index]
                    break
                consumer_index=consumer_index+1
        return found_consumer_in_blob,consumer_node_name

    def get_consumer_input_index(self,consumer_node,target_input_tensor):
        if consumer_node['op_type'] in MULTIPLE_INPUT_OPS:
            input_tensors = consumer_node['frontend']['input_tensors']
        else:
            input_tensors = [consumer_node['frontend']['input_tensor']]
        found_tensor = False
        tensor_index = 0 
        for input_index,input_tensor in enumerate(input_tensors):
            if input_tensor.name == target_input_tensor.name:
                found_tensor = True
                tensor_index = input_index
                break
        return found_tensor,tensor_index

    def is_blob_a_sync(self, blob) -> bool:
        if len(blob.nodes_in_blob) == 1:
            node_name = blob.nodes_in_blob[0]
            if self.graph.nodes[node_name]['op_type'] == 'Sync':
                return True
        return False

    def is_next_tile_written_to_ddr(self,current_blob_idx,consecutive_2_tile_blobs,prev_blob_lines_in_ddr):

        # This checks if in last tile of current blob (before write of last tile result) there are enough lines for read of next blob 1st tile
        # This check is a bit naive as we assume blob X output goes to blob X+1 input (which is usually the case)
        current_blob = self.tiling_blobs[current_blob_idx]
        current_blob_k3_nodes = current_blob.k3_nodes
        current_blob_output_node_height = current_blob.outputs[0].get_folded_shape()[2] # We assume all blob's outputs will have same height since it can change only by read from ddr
        is_current_blob_folding = 'force_folding_y' in self.graph.nodes()[current_blob.nodes_in_blob[0]]['frontend']
        next_blob = self.tiling_blobs[current_blob_idx+1]
        next_blob_k3_nodes = next_blob.k3_nodes
        next_blob_actual_input_height = next_blob.outputs[0].get_folded_shape()[2] # We assume that actual input height will be same as output height as this can be changed only by ddr read. It will take into account if input is folded at input
        is_next_blob_folding = 'force_folding_y' in self.graph.nodes()[next_blob.nodes_in_blob[0]]['frontend']

        # If the current or next blob is a Sync node, set the number of lines written to DDR in current tile to 0,
        # because none of those lines are valid towards pre-loading the first tile of the next blob.
        is_current_blob_sync = self.is_blob_a_sync(current_blob)
        is_next_blob_sync = self.is_blob_a_sync(next_blob)
        if is_current_blob_sync or is_next_blob_sync:
            lines_written_to_ddr_in_current_tile = 0
        elif DEBUG_2_TILE_RW_LINES and current_blob.y_tiles == 2:
            # The sum skips the final tile
            lines_written_to_ddr_in_current_tile = sum(
                get_y_tile_sizes_2_tile_case(
                    current_blob_output_node_height,
                    current_blob_k3_nodes,
                    consecutive_2_tile_blobs=consecutive_2_tile_blobs,
                    prev_blob_lines_in_ddr=prev_blob_lines_in_ddr,
                    is_first_node_folding=is_current_blob_folding,
                )[0][:-1]
            )
        else:
            lines_written_to_ddr_in_current_tile = sum(get_y_tile_sizes(current_blob_output_node_height,current_blob_k3_nodes,add_padding_line=False)[0][:-1])

        # Store for later
        self.tiling_blobs[current_blob_idx].num_lines_written_to_ddr_before_next_blob_read = lines_written_to_ddr_in_current_tile

        if DEBUG_2_TILE_RW_LINES and next_blob.y_tiles == 2:
            # The next blob also has 2 tiles, so can pass + 1 to the consecutive
            # Note also we add + next_blob_k3_nodes after so this should be write lines still
            lines_to_read_in_next_blob_first_tile = get_y_tile_sizes_2_tile_case(
                next_blob_actual_input_height,
                next_blob_k3_nodes,
                consecutive_2_tile_blobs=consecutive_2_tile_blobs + 1,
                prev_blob_lines_in_ddr=lines_written_to_ddr_in_current_tile,
                is_first_node_folding=is_next_blob_folding,
            )[0][0]+next_blob_k3_nodes
        else:
            lines_to_read_in_next_blob_first_tile = get_y_tile_sizes(next_blob_actual_input_height,next_blob_k3_nodes,add_padding_line=False)[0][0]+next_blob_k3_nodes

        if is_next_blob_folding:
            lines_to_read_in_next_blob_first_tile = lines_to_read_in_next_blob_first_tile * 2
        if lines_written_to_ddr_in_current_tile>=lines_to_read_in_next_blob_first_tile:
            next_tile_written_to_ddr = True
        else:
            next_tile_written_to_ddr = False
        return next_tile_written_to_ddr
    
    def update_marked_nodes_for_folding_y(self,node_name):
        # When adding a new node that will force fold, we must remove all nodes the follow this node in execution order as it might be that the added node
        # made followig nodes folding un-necesary

        execution_order_sorted_graph = list(nx.lexicographical_topological_sort(self.graph))
        updated_forced_y_folding_nodes = copy.deepcopy(self.marked_nodes_for_folding_y)
        added_node_execution_order = execution_order_sorted_graph.index(node_name)
        for current_node_name in self.marked_nodes_for_folding_y:
            current_node_execution_index = execution_order_sorted_graph.index(current_node_name)
            if current_node_execution_index>added_node_execution_order:
                updated_forced_y_folding_nodes.remove(current_node_name)
        updated_forced_y_folding_nodes.append(node_name)
        self.marked_nodes_for_folding_y =updated_forced_y_folding_nodes

def get_node_weights_tensor(node):
    """Get the appropriate weights tensor (folded or unfolded) based on node's folding factors.
    
    Args:
        node: Node from the graph containing frontend attributes
            
    Returns:
        The appropriate weights tensor based on folding factors
    """
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']
    
    if (input_folding_factor_x > 0 or input_folding_factor_y > 0):
        weights_tensor = node['frontend']['folded_weights_tensor'] 
        kernel_size = node['frontend']['folded_kernel_size']
    else:
        weights_tensor = node['frontend']['weights_tensor'] 
        kernel_size = node['frontend']['kernel_size']
    return weights_tensor, kernel_size             

def draw_graph_from_ir(ir: IR, title: str):
    val_map = {'DequantizeLinear': 1.0,
            'QuantizeLinear': 0.5714285714285714,
            'H': 0.0}
    values = []
    for node_name in ir.graph.nodes():
        values.append(val_map.get(ir.graph.nodes()[node_name]['op_type'], 0.25))

    # Specify the edges you want here
    red_edges = [('A', 'C'), ('E', 'C')]
    edge_colours = ['black' if not edge in red_edges else 'red'
                    for edge in ir.graph.edges()]
    black_edges = [edge for edge in ir.graph.edges() if edge not in red_edges]

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(ir.graph)
    nx.draw_networkx_nodes(ir.graph, pos, cmap=plt.get_cmap('jet'), 
                        node_color = values, node_size = 500)
    nx.draw_networkx_labels(ir.graph, pos)
    nx.draw_networkx_edges(ir.graph, pos, edgelist=ir.graph.edges, arrows=False)
    plt.title(title)
    plt.show()
