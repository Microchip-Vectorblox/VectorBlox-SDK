import networkx as nx
import numpy as np
from collections import OrderedDict
import pickle
from common.ddr_ir import DDR, DDREntry
from common.sequencer_ir import SequencerProgram
from common.enums import DDREntryType
from common.hw_config import NUM_MUTEX_FLAGS, NUM_TABLE_BUFFERS, MULTIPLE_INPUT_OPS, OpType, MAX_GRID_HEIGHT
from common.debug_flags import DEBUG_2_TILE_RW_LINES
from common.amm_ir import DeviceAMMs
from common.tensor_ir import Tensor
from common.utils import get_y_tile_sizes, get_y_tile_sizes_2_tile_case, packed_num_lanes
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
            if not self.allocated_flags[j]:
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
            if not self.allocated_buffers[j]:
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
        ret_val = [i for i,buffer_allocated in enumerate(self.allocated_buffers) if buffer_allocated]
        return ret_val

class IntermediateDDRTensor:
    def __init__(self,skip_tensor_name,offloading_node_name=''):
        self.skip_tensor_name = skip_tensor_name
        self.offloading_node_name = offloading_node_name
        self.write_wait_flag = None
        self.read_wait_flag = None

class TilingBlob:
    def __init__(self,nodes_in_blob,num_y_tiles=1,num_h_tiles=1,num_x_slices=1,xslices_per_htile=None,read_next_tile_node_idx=0,write_finished_tile_node_idx=0,k3_nodes=0):
        self.nodes_in_blob=nodes_in_blob
        self.num_of_nodes_in_blob=len(nodes_in_blob)
        self.y_tiles = num_y_tiles
        self.x_slices = num_x_slices
        self.h_tiles = num_h_tiles
        self.xslices_per_htile = xslices_per_htile
        self.read_next_tile_node_idx = read_next_tile_node_idx
        self.write_last_tile_node_idx = write_finished_tile_node_idx
        self.k3_nodes=k3_nodes # We need to know how many k3 nodes are in blob to know how many lines are reduced until we write it blob output to ddr
        self.is_y_folding_blob = False
        self.next_tile_read_node_idx = 0
        # Index of the earliest next-tile node that reuses an offloaded blob-output AMM lane (the
        # residual ADD/CONCAT the next_tile_read_node ADD/CONCAT-skip jumps over). -1 means none, so
        # no separate early offload-complete (F3) wait is emitted. See offload_wait_node handling in
        # program_compiler/sequencer.
        self.offload_wait_node_idx = -1
        self.inputs = []
        self.outputs = []
        self.num_lines_written_to_ddr_before_next_blob_read = 0
    def get_blob_outputs_names(self):
        names = []
        for output in self.outputs:
            names.append(output.name)
        return names

class IR:
    def __init__(self,model_name,size_config='V1000',compiler_output_dir='',debug=False):
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
        if (size_config == 'V1000'):
            self.amm_height = 16
            self.amm_block_size = 16
            self.amm_num_blocks = 128
        else:
            self.amm_height = 8
            self.amm_block_size = 16
            self.amm_num_blocks = 128
        self.amms = DeviceAMMs(self.amm_height, self.amm_block_size, self.amm_num_blocks)
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
        self.force_y_folding = []
        self.force_y_unfolding = []
        self.force_x_folding = ['']

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
        # Ordered list of NX tensor ids the MXP waits for (NX->MXP sync wait order emitted by
        # the vnnx generator as nx_engine/mxp_sync_order.txt). Populated during compilation and
        # persisted into the numsim IR so the numeric simulator can send in this exact order
        # without needing the separate .txt file. None if the model has no MXP syncing.
        self.mxp_sync_order = None

        # When a graph output is produced through a fused LUT chain (SiLU/Requant via
        # QUANTIZE->CAST->GATHER), replace_ir_output_tensor renames that ir.outputs entry
        # from the chain's true tflite output tensor (e.g. T430, "PartitionedCall:1") to the
        # pre-activation tensor that actually holds the computed value (e.g. T428, the fused
        # conv's own output). mxp_sync_order is written from the ORIGINAL tflite graph and
        # still expects the true tensor id (430), so the numeric simulator needs this map
        # (new name -> old/true name) to resolve that id back to ir.outputs' position.
        self.output_tensor_aliases = {}

        # Data structures for splitting large convolutions
        self.original_tensor_to_split_tensor_map = {}
        self.split_tensor_to_original_tensor_map = {}
        self.debug = (debug.strip().lower() == "true") if isinstance(debug, str) else debug
        self.input_ddr_offset = set()
        self.output_ddr_offset = set()

    def save(self,filename: str):
        with open(filename, 'wb') as filehandler:
            pickle.dump(self, filehandler)

    def save_numsim_ir(self,filename: str):
        # Constant weight/bias Tensor.data/.scale/.zero_point arrays never get mutated in
        # place anywhere in the compiler (splits always build fresh Tensor objects under new
        # keys). Share them by reference across every deepcopy below via one memo, instead of
        # re-copying (and duplicating in self.graph's node['frontend'] AND self.tensors) the same
        # large arrays - this call happens once RSS is already at its peak from compilation, so
        # a full uncached copy of every weight array here was pushing this over MemoryError.
        _memo = {}
        for tensor in self.tensors.values():
            if getattr(tensor, 'is_constant', False):
                for arr in (tensor.data, getattr(tensor, 'scale', None), getattr(tensor, 'zero_point', None)):
                    if isinstance(arr, np.ndarray):
                        _memo[id(arr)] = arr
        for n in self.graph.nodes:
            frontend = self.graph.nodes[n].get('frontend', {})
            for value in frontend.values():
                if isinstance(value, Tensor) and getattr(value, 'is_constant', False):
                    for arr in (value.data, getattr(value, 'scale', None), getattr(value, 'zero_point', None)):
                        if isinstance(arr, np.ndarray):
                            _memo[id(arr)] = arr
        # numsim_ir only ever keeps these backend fields per node (see the deletion loop that used
        # to run after the deepcopy below) - deepcopying the rest (in particular the AMM allocation
        # grids, which are sized off every node's tiling and are the biggest thing backend holds for
        # a large model) just to immediately discard them was doubling this call's peak RSS at the
        # point compilation is already at its highest memory watermark. Strip them from self.graph
        # before the copy and restore them on self right after, instead of copying-then-deleting.
        _NUMSIM_BACKEND_KEEP = {'oc_order','simulator_oc_order','ic_splits','ic_groups',
                                 'input_channels_reorder_dict','nlf','output_padding_start_y',
                                 'output_padding_start_x'}
        _stripped_backend = {}
        for node_name in self.graph.nodes:
            backend = self.graph.nodes[node_name].get('backend')
            if backend:
                dropped = {k: backend.pop(k) for k in list(backend.keys()) if k not in _NUMSIM_BACKEND_KEEP}
                if dropped:
                    _stripped_backend[node_name] = dropped
        numsim_ir = IR(self.model_name)
        try:
            numsim_ir.graph = copy.deepcopy(self.graph, memo=dict(_memo))
        finally:
            for node_name, dropped in _stripped_backend.items():
                self.graph.nodes[node_name]['backend'].update(dropped)
        numsim_ir.lexicographical_topological_sorted_graph = copy.deepcopy(self.lexicographical_topological_sorted_graph, memo=dict(_memo))
        numsim_ir.tensors = copy.deepcopy(self.tensors, memo=dict(_memo))
        numsim_ir.inputs = copy.deepcopy(self.inputs)
        numsim_ir.outputs = copy.deepcopy(self.outputs)
        numsim_ir.debug = copy.deepcopy(self.debug)
        numsim_ir.amm_height = copy.deepcopy(self.amm_height)
        numsim_ir.input_ddr_offset = copy.deepcopy(self.input_ddr_offset)
        numsim_ir.output_ddr_offset = copy.deepcopy(self.output_ddr_offset)
        numsim_ir.split_tensor_to_original_tensor_map = copy.deepcopy(self.split_tensor_to_original_tensor_map, memo=dict(_memo))
        numsim_ir.original_tensor_to_split_tensor_map = copy.deepcopy(self.original_tensor_to_split_tensor_map, memo=dict(_memo))
        numsim_ir.mxp_sync_order = copy.deepcopy(self.mxp_sync_order)
        numsim_ir.output_tensor_aliases = copy.deepcopy(self.output_tensor_aliases)
        numsim_ir.tensors_to_mxp = copy.deepcopy(self.tensors_to_mxp)
        numsim_ir.tensors_from_mxp = copy.deepcopy(self.tensors_from_mxp)
        numsim_ir.mxp_tensor_to_offset = copy.deepcopy(self.mxp_tensor_to_offset)
        numsim_ir.sync_with_MXP = copy.deepcopy(self.sync_with_MXP)
        with open(filename, 'wb') as filehandler:
            pickle.dump(numsim_ir, filehandler)

    def load(self,filename: str):
        with open(filename, 'rb') as filehandler:
            return pickle.load(filehandler)
    def is_k3_node(self,node):
        if 'kernel_size' in node['frontend']:
            is_k3_node_val = node['frontend']['kernel_size']>1
        else:
            is_k3_node_val = False
        return is_k3_node_val

    def node_needs_k3_1T_identity(self,node,amm_height):
        # True when this node, as the first node of a 1-tile blob, would get a 1x1 identity prepended by
        # add_identity_before_k3_starting_1T_blob (k3 reading its input from DDR at a 1T blob start). The
        # prepended identity is an input_channels -> input_channels pass-through, so its AMM peak is driven
        # by the INPUT channels; mirror that pass's guards exactly so callers stay in sync.
        if node['frontend'].get('y_tiles') != 1:
            return False
        if node['frontend'].get('input_folding_factor_x', 0) != 0:
            return False
        if not self.is_k3_node(node):
            return False
        if node['frontend']['input_tensor'].shape[2] >= amm_height:
            return False
        return True

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
        # Match by tensor name, not object identity ('==' with no Tensor.__eq__ falls back to
        # 'is') - a deepcopy/pickle round-trip (e.g. compile()'s escalation-loop _snapshot)
        # recreates every Tensor object with the same name but a different identity, so an
        # object captured before such a snapshot no longer matches the (content-identical)
        # object now stored on the node.
        original_input_tensor_found = False
        if node['op_type'] in MULTIPLE_INPUT_OPS:
            for idx,input_tensor in enumerate(node['frontend']['input_tensors']):
                if input_tensor.name==original_input_tensor.name:
                    original_input_tensor_found = True
                    break
            if not original_input_tensor_found:
                raise ValueError ('Couldnt find original input tensor. Please check model integrity.')
            node['frontend']['input_tensors'][idx] = new_input_tensor
        else:
            if node['frontend']['input_tensor'].name == original_input_tensor.name:
                node['frontend']['input_tensor'] = new_input_tensor
            elif new_input_tensor.name in node['inputs']:
                # Not every listed input is tracked as frontend['input_tensor'] - a Sync node
                # wrapping an MXP op can list a secondary activation (or a small param tensor)
                # alongside the one _set_input_tensor_params picked as the "real" activation
                # (see frontend_hardware_agnostic.py's highest-rank-shape heuristic). switch_input_name
                # (called before this, at every call site) already retargeted node['inputs'] from
                # original_input_tensor's name to new_input_tensor's name, so its presence there now
                # confirms this was a genuine (if untracked-by-frontend) input; nothing further to update.
                pass
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


    def _get_split_lineage(self, tensor_name):
        # All tensor names reachable from tensor_name via split lineage: ancestors through
        # split_tensor_to_original_tensor_map, descendants through original_tensor_to_split_tensor_map.
        lineage = {tensor_name}
        ancestor = tensor_name
        while ancestor in self.split_tensor_to_original_tensor_map:
            ancestor = self.split_tensor_to_original_tensor_map[ancestor]
            lineage.add(ancestor)
        frontier = list(self.original_tensor_to_split_tensor_map.get(tensor_name, []))
        while frontier:
            child = frontier.pop()
            if child in lineage:
                continue
            lineage.add(child)
            frontier.extend(self.original_tensor_to_split_tensor_map.get(child, []))
        return lineage

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
            following_node_input_index = None
            if (len(node['outputs']) > 0):
                for output_idx in range(len(node['outputs'])):
                    # Try the exact output tensor name first - a following node may read it verbatim
                    # (e.g. an identity conv's output, switched onto the consumer's inputs by exact
                    # name via switch_input_name). Otherwise walk this output tensor's split lineage
                    # (ancestors via split_tensor_to_original_tensor_map, descendants via
                    # original_tensor_to_split_tensor_map - same technique as get_consumer_input_index
                    # and mark_blob_outputs_deallocations) and match against that instead. This covers
                    # both a size-split producer whose consumer still reads the unsplit root name, and
                    # nested split names at any depth - unlike a blind output_tensor_name.split('_split')[0],
                    # which over-strips names that merely CONTAIN "_split" earlier in their base
                    # identifier (e.g. "..._split1_split_ic1_identity_output").
                    output_tensor_name = node['outputs'][output_idx]
                    if (output_tensor_name in following_node['inputs']):
                        following_node_input_index = following_node['inputs'].index(output_tensor_name)
                    else:
                        lineage = self._get_split_lineage(output_tensor_name)
                        for input_idx, input_name in enumerate(following_node['inputs']):
                            if input_name in lineage:
                                following_node_input_index = input_idx
                                break
            if (following_node_input_index != None):
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
            preceding_output_name = preceding_node['outputs'][0]
            if preceding_output_name in node['inputs']:
                preceding_node_input_index = node['inputs'].index(preceding_output_name)
            else:
                # Producer may be a split piece (e.g. "<name>_split1", possibly nested) while the
                # consumer's inputs still reference the unsplit root name, or vice versa - walk the
                # split lineage (ancestors via split_tensor_to_original_tensor_map, descendants via
                # original_tensor_to_split_tensor_map) instead of blindly stripping on '_split', which
                # over-strips names that merely CONTAIN "_split" earlier in their base identifier.
                # Mirrors the matching in get_updated_following_nodes/get_consumer_input_index.
                lineage = self._get_split_lineage(preceding_output_name)
                preceding_node_input_index = next(idx for idx, inp in enumerate(node['inputs']) if inp in lineage)
            preceding_nodes_params.append((preceding_node_name,preceding_node_input_index))
        return preceding_nodes_params

    def get_next_executed_node(self,node,error_on_last_node=True,current_tile_num=0,current_htile_num=0,current_xslice_num=0):
        execution_sorted_graph = self.lexicographical_topological_sorted_graph
        current_node_execution_index = execution_sorted_graph.index(node['name'])
        current_node_blob_idx = node['frontend']['tiling_blob_idx']
        current_blob = self.tiling_blobs[current_node_blob_idx]
        ytiles_in_current_blob = current_blob.y_tiles
        htiles_in_current_blob = current_blob.h_tiles
        ret_val = True
        last_node_in_blob = node['name']==current_blob.nodes_in_blob[-1]
        num_xslices = node['frontend']['xslices_per_htile'][current_htile_num]
        if node['op_type'] in MULTIPLE_INPUT_OPS:
            input_tensors = node['frontend']['input_tensors']
        else:
            input_tensors = [node['frontend']['input_tensor']]
        num_xslices = packed_num_lanes(input_tensors[0].num_packed_xslices, num_xslices)
        if last_node_in_blob:
            if current_tile_num==ytiles_in_current_blob-1: # If we are in last tile of last node in blob
                if (current_node_execution_index+1)>=len(execution_sorted_graph): # If its also the last node in graph
                    following_executed_node_index = current_node_execution_index # We return current node
                    next_executed_ytile = current_tile_num
                    next_executed_htile = current_htile_num
                    if error_on_last_node:
                        ret_val = False
                else: #its not the last blob in graph
                    if (current_htile_num == htiles_in_current_blob-1):
                        following_executed_node_index = current_node_execution_index+1 # We return next executed node and restart from tile 0
                        next_executed_ytile = 0
                        next_executed_htile = 0
                    else:
                        following_executed_node_name = current_blob.nodes_in_blob[0] # We return first node in blob and htile_num+1
                        following_executed_node_index = execution_sorted_graph.index(following_executed_node_name)
                        next_executed_ytile = current_tile_num
                        next_executed_htile = current_htile_num+1
            else: # Its last node in blob but not last tile in that blob
                following_executed_node_name = current_blob.nodes_in_blob[0] # We return first node in blob and tile_num+1
                following_executed_node_index = execution_sorted_graph.index(following_executed_node_name)
                if (current_htile_num == htiles_in_current_blob-1):
                    next_executed_htile = 0
                    next_executed_ytile = current_tile_num + 1
                else:
                    next_executed_htile = current_htile_num + 1
                    next_executed_ytile = current_tile_num
        else: # Its not last node in blob
            following_executed_node_index = current_node_execution_index+1 # We return next executed node
            next_executed_ytile = current_tile_num
            next_executed_htile = current_htile_num
        
        if (current_xslice_num == num_xslices-1):
            next_executed_xslice = 0
        else:
            next_executed_xslice = current_xslice_num + 1
        
        following_executed_node_name = execution_sorted_graph[following_executed_node_index]
        following_executed_node = self.graph.nodes[following_executed_node_name]
        return ret_val,following_executed_node,next_executed_ytile,next_executed_htile,next_executed_xslice

    def get_blob_output_deallocating_node(self,current_node_blob_idx,current_tile_num,current_htile_num,current_xslice_num,num_xslices):
        current_blob = self.tiling_blobs[current_node_blob_idx]
        current_blob_num_ytiles = current_blob.y_tiles
        current_blob_num_htiles = current_blob.h_tiles
        nodes_in_current_blob = len(current_blob.nodes_in_blob)
        if (current_tile_num == current_blob_num_ytiles-1) and (current_htile_num == current_blob_num_htiles-1): # If we are at last y-tile and h-tile the deallocating node will be in next blob, tile 0
            deallocation_htile_idx = 0
            if current_node_blob_idx!=len(self.tiling_blobs)-1: # We are not in last blob
                next_blob = self.tiling_blobs[current_node_blob_idx+1]
                nodes_in_next_blob = len(next_blob.nodes_in_blob)
                deallocating_node_idx = (nodes_in_next_blob - 1) // 2
                if (deallocating_node_idx > 0):
                    deallocating_node_idx -= 1
            
                # If the first node of the next blob is a Concat, delay the de-allocation.
                # For now this is done for size 2 blobs.
                if self.graph.nodes[next_blob.nodes_in_blob[0]]['op_type'] == OpType.CONCAT:
                    if nodes_in_next_blob == 2:
                        deallocating_node_idx = nodes_in_next_blob // 2
                
                if self.graph.nodes[next_blob.nodes_in_blob[deallocating_node_idx]]['op_type'] in MULTIPLE_INPUT_OPS:
                    deallocating_node_idx = deallocating_node_idx + 1
                    if (deallocating_node_idx >= nodes_in_next_blob):
                        # Normally skip a MULTIPLE_INPUT_OP first node by deallocating in the blob after
                        # next. Near the end of the graph there may be no blob-after-next; fall back to
                        # the last node of the next blob (no further blob to defer into).
                        if current_node_blob_idx+2 in self.tiling_blobs:
                            deallocating_node_idx = 0
                            next_blob = self.tiling_blobs[current_node_blob_idx+2]
                        else:
                            deallocating_node_idx = nodes_in_next_blob - 1
                deallocating_node_name = next_blob.nodes_in_blob[deallocating_node_idx]
                deallocating_node = self.graph.nodes()[deallocating_node_name]
                deallocation_ytile_idx = 0
            else:
                deallocating_node_idx = nodes_in_current_blob - 1
                deallocating_node_name = current_blob.nodes_in_blob[deallocating_node_idx]
                deallocating_node = self.graph.nodes()[deallocating_node_name]
                deallocation_ytile_idx = current_tile_num
        else: # If we are NOT at last y-tile and h-tile the deallocating node will be in current blob, next tile
            deallocating_node_idx = (nodes_in_current_blob-1) // 2
            if (deallocating_node_idx > 0):
                deallocating_node_idx -= 1
                if self.graph.nodes[current_blob.nodes_in_blob[deallocating_node_idx]]['op_type'] in MULTIPLE_INPUT_OPS:
                    deallocating_node_idx += 1
            deallocating_node_name = current_blob.nodes_in_blob[deallocating_node_idx]
            deallocating_node = self.graph.nodes()[deallocating_node_name]
            if current_htile_num != current_blob_num_htiles-1: # If we are NOT at last HZ tile the deallocating node will be in current blob, next HZ tile
                deallocation_htile_idx = current_htile_num + 1
                deallocation_ytile_idx = current_tile_num
            else:
                deallocation_htile_idx = 0
                deallocation_ytile_idx = current_tile_num + 1
        
        if (current_xslice_num == num_xslices-1):
            deallocation_xslice_idx = 0
        else:    
            deallocation_xslice_idx = current_xslice_num + 1
        
        return deallocating_node, deallocation_ytile_idx, deallocation_htile_idx, deallocation_xslice_idx

    def get_dominant_tensor_consumer_in_blob(self,next_tile_blob,input_tensor):
        # This will return the dominant consumer of the tensor in the specified blob
        # By dominant we choose concat over other consumers since in case of concat we need to allocate all its inputs
        # at once
        def _multi_input_candidate_genuinely_resolves(current_consumer_node, current_consumer_op_type):
            # For a MULTIPLE_INPUT_OPS candidate, mirror exactly what
            # allocate_and_mark_for_read_input_tensor's dual_alloc block uses to resolve each operand:
            # its own cached input_tensors[idx] when that entry is flagged is_read_channel_slice,
            # otherwise the LIVE producer's frontend['output_tensor'] (remapped through
            # split_tensor_to_original_tensor_map for a Concat). A mismatch here means this candidate's
            # own live producer chain was split/mutated (e.g. split_node_out_channels_in_n reusing chunk
            # 0's node object) and no longer actually resolves back to input_tensor.
            preceding = current_consumer_node['frontend'].get('preceding_nodes_params', [])
            cached_inputs = current_consumer_node['frontend'].get('input_tensors', [])
            for prev_name, consumer_input_idx in preceding:
                cached_input = cached_inputs[consumer_input_idx] if consumer_input_idx < len(cached_inputs) else None
                if cached_input is not None and getattr(cached_input, 'is_read_channel_slice', False):
                    if cached_input.name == input_tensor.name:
                        return True
                    continue
                prev_node = self.graph.nodes.get(prev_name)
                prod_out = prev_node['frontend'].get('output_tensor') if prev_node else None
                prod_out_name = prod_out.name if prod_out is not None else None
                if (current_consumer_op_type == OpType.CONCAT and prod_out_name is not None
                        and prod_out_name in self.split_tensor_to_original_tensor_map):
                    prod_out_name = self.split_tensor_to_original_tensor_map[prod_out_name]
                if prod_out_name is not None and prod_out_name == input_tensor.name:
                    return True
            return False

        # Pass 1: collect every consumer in this blob whose own cached input_tensors genuinely still
        # lists input_tensor (guards against a stale consumers-list entry left over from rewiring).
        candidates = []  # (name, op_type, is_multi_input, genuinely_resolves)
        for current_consumer_node_name in input_tensor.consumers:
            if current_consumer_node_name not in next_tile_blob.nodes_in_blob:
                continue
            current_consumer_node = self.graph.nodes()[current_consumer_node_name]
            current_consumer_op_type = current_consumer_node['op_type']
            genuinely_reads, _ = self.get_consumer_input_index(current_consumer_node, input_tensor)
            if not genuinely_reads:
                continue
            is_multi_input = current_consumer_op_type in MULTIPLE_INPUT_OPS
            resolves = (not is_multi_input) or _multi_input_candidate_genuinely_resolves(current_consumer_node, current_consumer_op_type)
            candidates.append((current_consumer_node_name, current_consumer_op_type, is_multi_input, resolves))

        # Pass 2: pick by priority - a multi-input consumer whose live producer chain genuinely
        # resolves to input_tensor (needed for its all-inputs-at-once contiguous allocation) beats a
        # plain single-input consumer, which beats a multi-input consumer that no longer genuinely
        # resolves (kept only as a last-resort fallback so a lone, isolated multi-input node - e.g. an
        # output-split Add chunk alone in its own blob - still gets selected when it's the only option).
        resolving_multi = [c for c in candidates if c[2] and c[3]]
        if len(resolving_multi) > 1:
            raise ValueError('Cant have 2 concats consume same input in same blob')
        if resolving_multi:
            return True, resolving_multi[0][0]
        single_input = [c for c in candidates if not c[2]]
        if single_input:
            return True, single_input[0][0]
        if candidates:
            return True, candidates[0][0]
        return False, ''

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
        if not found_tensor:
            # target_input_tensor's name itself may not be what the consumer's frontend input
            # actually holds - e.g. an input-channel split (split_conv_in_channels_in_m) still
            # lists the ORIGINAL node name in target_input_tensor.consumers (so
            # get_first_tensor_consumer_in_blob can pick it), but that node's own frontend
            # input_tensor was rewired to a split descendant (e.g. '<name>_split_ic0'), not
            # target_input_tensor itself. Walk target_input_tensor's split lineage (ancestors via
            # split_tensor_to_original_tensor_map, descendants via
            # original_tensor_to_split_tensor_map) and match against that instead - same technique
            # used in mark_blob_outputs_deallocations for the equivalent name-drift problem.
            lineage = self._get_split_lineage(target_input_tensor.name)
            for input_index,input_tensor in enumerate(input_tensors):
                if input_tensor.name in lineage:
                    found_tensor = True
                    tensor_index = input_index
                    break
        return found_tensor,tensor_index

    def is_blob_a_sync(self, blob) -> bool:
        if len(blob.nodes_in_blob) == 1:
            node_name = blob.nodes_in_blob[0]
            if self.graph.nodes[node_name]['op_type'] == OpType.SYNC:
                return True
        return False

    def is_next_tile_written_to_ddr(self,current_blob_idx,consecutive_2_tile_blobs,prev_blob_lines_in_ddr):

        # This checks if in last tile of current blob (before write of last tile result) there are enough lines for read of next blob 1st tile
        # This check is a bit naive as we assume blob X output goes to blob X+1 input (which is usually the case)
        current_blob = self.tiling_blobs[current_blob_idx]
        current_blob_k3_nodes = current_blob.k3_nodes
        current_blob_last_node = self.graph.nodes()[current_blob.nodes_in_blob[-1]]
        # A blob ending in a resize_y node doubles its AMM-resident line count on the way out to DDR via
        # a DMW instruction: the AMM (and therefore the tile-size math below) only ever holds the
        # pre-resize line count, while current_blob.outputs[0]'s folded shape reflects the post-resize
        # (already-doubled) height. Use the pre-resize height for tile sizing and double the resulting
        # DDR-line count afterward, instead of feeding the doubled height into a tile split that's bounded
        # by y_tiles*amm_height and can never reach it.
        is_current_blob_dmw_resize_y = current_blob_last_node['op_type'] == OpType.RESIZE and '_y' in current_blob.nodes_in_blob[-1]
        if is_current_blob_dmw_resize_y:
            current_blob_output_node_height = current_blob_last_node['frontend']['input_tensor'].get_folded_shape()[2]
        else:
            current_blob_output_node_height = current_blob.outputs[0].get_folded_shape()[2] # We assume all blob's outputs will have same height since it can change only by read from ddr
        is_current_blob_folding = 'force_folding_y' in self.graph.nodes()[current_blob.nodes_in_blob[0]]['frontend']
        next_blob = self.tiling_blobs[current_blob_idx+1]
        next_blob_k3_nodes = next_blob.k3_nodes
        next_blob_last_node = self.graph.nodes()[next_blob.nodes_in_blob[-1]]
        # Same DMW-doubling issue as current_blob_output_node_height above: if next_blob itself ends in a
        # resize_y, its outputs[0] folded height is post-resize (doubled) and is a poor proxy for the
        # actual input height it needs on hand to start its own first tile - use its pre-resize height.
        is_next_blob_dmw_resize_y = next_blob_last_node['op_type'] == OpType.RESIZE and '_y' in next_blob.nodes_in_blob[-1]
        if is_next_blob_dmw_resize_y:
            next_blob_actual_input_height = next_blob_last_node['frontend']['input_tensor'].get_folded_shape()[2]
        else:
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
                    amm_height=self.amm_height,
                )[0][:-1]
            )
        else:
            lines_written_to_ddr_in_current_tile = sum(get_y_tile_sizes(current_blob_output_node_height,current_blob_k3_nodes,amm_height=self.amm_height)[0][:-1])

        if is_current_blob_dmw_resize_y:
            lines_written_to_ddr_in_current_tile = lines_written_to_ddr_in_current_tile * 2

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
                amm_height=self.amm_height,
            )[0][0]+next_blob_k3_nodes
        else:
            lines_to_read_in_next_blob_first_tile = get_y_tile_sizes(next_blob_actual_input_height,next_blob_k3_nodes,amm_height=self.amm_height)[0][0]+next_blob_k3_nodes

        if is_next_blob_folding:
            lines_to_read_in_next_blob_first_tile = lines_to_read_in_next_blob_first_tile * 2
        if lines_written_to_ddr_in_current_tile>=lines_to_read_in_next_blob_first_tile:
            next_tile_written_to_ddr = True
        else:
            next_tile_written_to_ddr = False
        return next_tile_written_to_ddr
    
    def update_marked_nodes_for_folding_y(self,node_name):
        # When adding a new node that will force fold, we must remove all nodes the follow this node in execution order as it might be that the added node
        # made following nodes folding unnecessary

        execution_order_sorted_graph = list(nx.lexicographical_topological_sort(self.graph))
        updated_forced_y_folding_nodes = copy.deepcopy(self.marked_nodes_for_folding_y)
        added_node_execution_order = execution_order_sorted_graph.index(node_name)
        for current_node_name in self.marked_nodes_for_folding_y:
            current_node_execution_index = execution_order_sorted_graph.index(current_node_name)
            if current_node_execution_index>added_node_execution_order:
                updated_forced_y_folding_nodes.remove(current_node_name)
        updated_forced_y_folding_nodes.append(node_name)
        self.marked_nodes_for_folding_y = updated_forced_y_folding_nodes

def get_node_weights_tensor(node):
    """Get the appropriate weights tensor (folded or unfolded) based on node's folding factors.
    For ADD operations, returns URAM-aligned version for CBC generation.

    Args:
        node: Node from the graph containing frontend attributes

    Returns:
        The appropriate weights tensor based on folding factors
    """
    input_folding_factor_x = node['frontend']['input_folding_factor_x']
    input_folding_factor_y = node['frontend']['input_folding_factor_y']

    # For ADD operations, use URAM-aligned weights for CBC generation
    is_add_op = node['op_type'] == OpType.ADD

    if (input_folding_factor_x > 0 or input_folding_factor_y > 0):
        if is_add_op and 'folded_weights_tensor_aligned_uram' in node['frontend']:
            weights_tensor = node['frontend']['folded_weights_tensor_aligned_uram']
        else:
            weights_tensor = node['frontend']['folded_weights_tensor']
        kernel_size = node['frontend']['folded_kernel_size']
    else:
        if is_add_op and 'weights_tensor_aligned_uram' in node['frontend']:
            weights_tensor = node['frontend']['weights_tensor_aligned_uram']
        else:
            weights_tensor = node['frontend']['weights_tensor']
        kernel_size = node['frontend']['kernel_size']
    return weights_tensor, kernel_size             
