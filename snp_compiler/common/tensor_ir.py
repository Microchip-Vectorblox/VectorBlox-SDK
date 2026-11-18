from collections import OrderedDict
import copy
import math

from common.hw_config import MAX_GRID_WIDTH


class Tensor:
    def __init__(self,name, data, is_constant = True, is_inline_tensor = False, shape = None, producer = None, consumers = None,scale = 1,zero_point =0,folding_factor_x = 0,folding_factor_y = 0, x_slices = 0, force_x_folding = False):
        self.name = name
        self.data = data
        self.scale = scale
        self.zero_point = zero_point
        self.producer = producer
        self.consumers = consumers.copy() if consumers else []
        self.is_constant = is_constant
        self.is_inline_tensor = is_inline_tensor
        self.shape = shape
        # calculate real x size rounded to next MAX_GRID_WIDTH (this is DMR/W requirement)
        self.shape_real_x16 = copy.deepcopy(shape) 
        if len(shape)>3 and (not is_constant):
            self.shape_real_x16[3] = ((shape[3] + (MAX_GRID_WIDTH-1)) // MAX_GRID_WIDTH) * MAX_GRID_WIDTH if shape else None
        self.ddr_entry = None
        self.folding_factor_x = folding_factor_x # This means how much resolution folding is done in nx . value of 0 means no folding, 1 means x2 folding, 2 means x4 folding etc.
        self.folding_factor_y = folding_factor_y # This means how much resolution folding is done in nx . value of 0 means no folding, 1 means x2 folding, 2 means x4 folding etc.
        self.x_slices = x_slices # number of x_slices
        self.force_x_folding = force_x_folding
        self.is_avgPool_output = False
        self.num_packed_xslices = 1
        self.xslices_per_htile = None
        
    def get_float(self):
        float_tensor = (self.data-self.zero_point)*self.scale
        return float_tensor
    def get_original_shape(self):
        if self.shape is None:
            raise ValueError('Tensors shape is not defined: %s' % (self.name))
        return self.shape
    def get_amm_tensor_name(self,tensors_blob_idx,current_tile_num,current_h_tile_num,current_x_slice):
        amm_tensor_name = self.name+'_blob'+str(tensors_blob_idx)+'_ytile'+str(current_tile_num)+'_htile'+str(current_h_tile_num)+'_xslice'+str(current_x_slice)
        return amm_tensor_name
    
    def get_folded_shape(self,folding_conv_x=False,producing_node_stride=1,folding_conv_y=False,unfolding_conv_y=False, padded =0): # If you want real output shape dont use folding conv=True. use it only if you want to know internal shape before rq for weights/generation
        # Note that folding_conv_x and stride are used for output tensors to know the folding factor before the hw folding
        # folding_conv_y, unfolding_conv_y are used for input tensors to know the actual folding factors after input folding/unfolding
        # Consider to provide 2 separate calls for input tensors calc and output tensor calc to make this clear
        output_tensor_shape = list(self.shape)  # shape is a flat list of ints; deepcopy is unnecessary overhead here
        folding_factor_x = self.folding_factor_x
        folding_factor_y = self.folding_factor_y
        force_x_folding = self.force_x_folding
        
        # folding convolutions are marked with folding_conv_x, but should not reduce the folding conv factor
        producer_node = self.producer
        # TODO: Not sure what to do with input tensor (no producer). Currently it is being treated like the default case.
        if producer_node and 'folding_conv' not in producer_node:
            if folding_conv_x: # In case of folding conv we want to get the shape before output folding for correct conv calculations
                if producing_node_stride==1: #In case of stride=2 the actual output fold is same as 1/2 of the output channels are dropped by hw
                    folding_factor_x -=1
            elif force_x_folding:
                folding_factor_x -=1

        if folding_conv_y: # In case of folding conv y we want to get the folded shape at conv input as y folding is done in read from ddr
            folding_factor_y +=1
        if unfolding_conv_y: # In case of unfolding conv y we want to get the folded shape at conv input as y unfolding is done in read from ddr before the conv
            if folding_factor_y>0:
                folding_factor_y -=1

        output_tensor_shape_real_x =   output_tensor_shape[3] + padded  # This is the real x size of the tensor
        output_tensor_shape_real_y =   output_tensor_shape[2] 
        tensor_folding_multiplier_x = int(2 ** folding_factor_x)
        tensor_folding_multiplier_y = int(2 ** folding_factor_y)
        if tensor_folding_multiplier_x!=1:
            output_tensor_shape[1] = output_tensor_shape[1] * tensor_folding_multiplier_x
            output_tensor_shape[3] = output_tensor_shape_real_x // tensor_folding_multiplier_x
            output_tensor_shape[3] = math.ceil(output_tensor_shape[3]/MAX_GRID_WIDTH)*MAX_GRID_WIDTH
        if tensor_folding_multiplier_y!=1:
            output_tensor_shape[1] = output_tensor_shape[1] * tensor_folding_multiplier_y
            output_tensor_shape[2] = math.ceil(output_tensor_shape_real_y / tensor_folding_multiplier_y)
        
        if self.is_avgPool_output:
            output_tensor_shape[3] = (self.shape[3] + 1) // tensor_folding_multiplier_x
            output_tensor_shape[2] = (self.shape[2] + 1) // tensor_folding_multiplier_y

        return output_tensor_shape
    
    def get_size_on_amm(self, amm_block_size):
        folded_shape=self.get_folded_shape()
        size_on_amm = int(math.ceil(folded_shape[1] / amm_block_size) * amm_block_size)
        return size_on_amm

    def get_output_tensor_folded_shape(self,folding_conv_x=False,original_stride = 1): # In case of folding conv we want to get the shape before output folding
            output_tensor_shape = list(self.shape)  # shape is a flat list of ints; deepcopy is unnecessary overhead here
            folding_factor_x = self.folding_factor_x
            folding_factor_y = self.folding_factor_y
            if folding_conv_x: # In case of folding conv we want to get the shape before output folding for correct conv calculations
                if original_stride==1: # This is not true for a folding conv with stride=2 where we drop 3/4 of output channels
                    folding_factor_x -=1

            tensor_folding_multiplier_x = int(2 ** folding_factor_x)
            tensor_folding_multiplier_y = int(2 ** folding_factor_y)
            if tensor_folding_multiplier_x!=1:
                output_tensor_shape[1] = output_tensor_shape[1] * tensor_folding_multiplier_x
                if output_tensor_shape[3] % tensor_folding_multiplier_x!=0:
                    raise ValueError('tensor width (%d) is not a multiple of folding factor (%d). Currently not supported' % (output_tensor_shape[3],tensor_folding_multiplier_x))
                output_tensor_shape[3] = output_tensor_shape[3] // tensor_folding_multiplier_x
            if tensor_folding_multiplier_y!=1:
                output_tensor_shape[1] = output_tensor_shape[1] * tensor_folding_multiplier_y
                if output_tensor_shape[2] % tensor_folding_multiplier_y!=0:
                    raise ValueError('tensor height (%d) is not a multiple of folding factor (%d). Currently not supported' % (output_tensor_shape[2],tensor_folding_multiplier_y))
                output_tensor_shape[2] = output_tensor_shape[2] // tensor_folding_multiplier_y
            return output_tensor_shape


    def get_dtype(self):
        return self.data.dtype

class InputTensorInfo:
    def __init__(self,name,input_index,tensor_tile_num,tensor_h_tile_num,tensor_xslice_num,load_at_execution_of_tile_idx,load_at_execution_of_h_tile_idx,load_at_execution_of_xslice_idx,consumer_node):
        self.name = name
        self.input_index = input_index
        self.tensor_tile_num = tensor_tile_num
        self.tensor_h_tile_num = tensor_h_tile_num
        self.tensor_xslice_num = tensor_xslice_num
        self.load_at_execution_of_tile_idx = load_at_execution_of_tile_idx
        self.load_at_execution_of_h_tile_idx = load_at_execution_of_h_tile_idx
        self.load_at_execution_of_xslice_idx = load_at_execution_of_xslice_idx
        self.consumer_node = consumer_node

class TensorDeAllocationInfo:
    def __init__(self,name,input_index,tensor_tile_num,tensor_h_tile_num,tensor_xslice_num,deallocate_at_execution_of_tile_idx,deallocate_at_execution_of_h_tile_idx,deallocate_at_execution_of_xslice_idx,inline_tensor=False):
        self.name = name
        self.input_index = input_index
        self.tensor_tile_num = tensor_tile_num
        self.tensor_h_tile_num = tensor_h_tile_num
        self.tensor_xslice_num = tensor_xslice_num
        self.deallocate_at_execution_of_tile_idx = deallocate_at_execution_of_tile_idx
        self.deallocate_at_execution_of_h_tile_idx = deallocate_at_execution_of_h_tile_idx
        self.deallocate_at_execution_of_xslice_idx = deallocate_at_execution_of_xslice_idx
        self.inline_tensor = inline_tensor # Inline tensor is an input tensor that is also used for writing the output. Hence when deallocated it is only removed from amm tensors list but the mem is not actually deallocated

class TensorDeAllocationList:
    def __init__(self):
        self.tensors_list = OrderedDict()
    def __str__(self):
        tensor_strings=[]
        for deallocating_tile,tensors_list in self.tensors_list.items():
            tensor_strings.append('@'+str(deallocating_tile)+':'+';'.join(t.name for t in tensors_list))
        return_str = ';'.join(tensor_strings)
        return return_str
    def add_tensor(self,deallocated_tensor:TensorDeAllocationInfo):
        deallocation_tile_idx = deallocated_tensor.deallocate_at_execution_of_tile_idx
        deallocation_h_tile_idx = deallocated_tensor.deallocate_at_execution_of_h_tile_idx
        deallocation_xslice_idx = deallocated_tensor.deallocate_at_execution_of_xslice_idx
        if (deallocation_tile_idx, deallocation_h_tile_idx, deallocation_xslice_idx) in self.tensors_list:
            self.tensors_list[(deallocation_tile_idx, deallocation_h_tile_idx, deallocation_xslice_idx)].append(deallocated_tensor)
        else:
            self.tensors_list[(deallocation_tile_idx, deallocation_h_tile_idx, deallocation_xslice_idx)] = [deallocated_tensor]
    def get_tensor_names_dict(self):
        tensor_names_dict=OrderedDict()
        for (tensor_dealloc_tile_idx, tensor_dealloc_h_tile_idx, tensor_dealloc_xslice_idx)  in self.tensors_list:
            tensor_dealloc_info_list = self.tensors_list[(tensor_dealloc_tile_idx, tensor_dealloc_h_tile_idx, tensor_dealloc_xslice_idx)]
            for tensor_dealloc_info in tensor_dealloc_info_list:
                if (tensor_dealloc_tile_idx, tensor_dealloc_h_tile_idx, tensor_dealloc_xslice_idx) in tensor_names_dict:
                    tensor_names_dict[(tensor_dealloc_tile_idx, tensor_dealloc_h_tile_idx, tensor_dealloc_xslice_idx)].append(tensor_dealloc_info.name)
                else:
                    tensor_names_dict[(tensor_dealloc_tile_idx, tensor_dealloc_h_tile_idx, tensor_dealloc_xslice_idx)] = [tensor_dealloc_info.name]
        return tensor_names_dict

def get_tensor_name_from_tiled_tensor(tiled_tensor_name:str)->str:
    tensor_name = tiled_tensor_name.split('_tile')[0]
    return tensor_name