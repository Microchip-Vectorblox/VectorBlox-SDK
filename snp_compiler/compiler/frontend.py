import onnx
import onnx_parser
import tflite_parser
import frontend_hardware_agnostic
import frontend_hardware_dependant
import common.internal_representation as internal_representation
import reports
import os

from common.debug_flags import DEBUG_CREATE_ORDERING_CONV,DEBUG_ONNX_FORMAT_QDQ,DEBUG_SIMULATE_CONCAT_REQUANT,DEBUG_SPLIT_LARGE_CONV
from common.tensor_ir import Tensor
from common.hw_config import MAX_GRID_WIDTH
from sys import platform
import math 
import numpy as np

onnx_models_dict = {
    'resnet50_deci': 'resnet50PrunedQuantized_deci.onnx'
}

'''
from frontend_hardware_dependant import update_x_fold_slice

def get_input_folding_factors(original_input_width):
    input_folding_factor_x = math.ceil(math.log(original_input_width / MAX_GRID_WIDTH,2))
    if input_folding_factor_x<0: # If input resolution is less than MAX_GRID_WIDTH, folding is not needed
        input_folding_factor_x=0
    return input_folding_factor_x

def add_foldingConv_nodes(ir):
    input_shape = ir.tensors[ir.inputs[0]].shape
    org_first_nodes = ir.tensors[ir.inputs[0]].consumers[:]
    first_node_attributes = ir.graph.nodes[org_first_nodes[0]]['attributes']
    stride = first_node_attributes['stride_w']
    kernel_size = first_node_attributes['kernel_shape'][0]
    folding_factor_x = get_input_folding_factors(input_shape[3])
    folding_factor_x, _ = update_x_fold_slice(folding_factor_x, input_shape[3], kernel_size, stride)
    if (folding_factor_x > 0):
        channels = input_shape[1]
        conv_W = np.zeros((channels,channels,1,1), dtype=np.int8)
        conv_B = np.zeros((channels), dtype=np.int32)
        conv_W_scale = np.zeros(channels, dtype=np.float32)
        conv_B_scale = np.ones(channels, dtype=np.float32) * 0.000030878
        conv_W_zp = np.zeros(channels, dtype=np.int8)
        conv_B_zp = np.zeros(channels, dtype=np.float32)
        for current_cout in range(channels):
            conv_W[current_cout,current_cout,0,0] = 8 # value in int8
            conv_W_scale[current_cout] = 1/8
        op_input_name = ir.inputs[0]
        folding_conv_attr = {}
        folding_conv_attr['kernel_shape'] = [1,1]
        folding_conv_attr['strides'] = [1,1]
        folding_conv_attr['pads'] = [0,0,0,0]
        for i in range(folding_factor_x):
            op_name = 'folding_conv_' + str(i)
            op_output_name = op_name+'_out'
            conv_W_initializer_tensor_name = op_name+"_W"
            conv_B_initializer_tensor_name = op_name+"_B"
            if conv_W_initializer_tensor_name not in ir.tensors:
                ir.tensors[conv_W_initializer_tensor_name] = Tensor(conv_W_initializer_tensor_name,conv_W,is_constant=True, shape = conv_W.shape)
                ir.tensors[conv_W_initializer_tensor_name].consumers = op_name
                ir.tensors[conv_W_initializer_tensor_name].scale = conv_W_scale
                ir.tensors[conv_W_initializer_tensor_name].zero_point = conv_W_zp
            if conv_B_initializer_tensor_name not in ir.tensors:
                ir.tensors[conv_B_initializer_tensor_name] = Tensor(conv_B_initializer_tensor_name,conv_B,is_constant=True, shape = conv_B.shape)
                ir.tensors[conv_B_initializer_tensor_name].consumers = op_name
                ir.tensors[conv_B_initializer_tensor_name].scale = conv_B_scale
                ir.tensors[conv_B_initializer_tensor_name].zero_point = conv_B_zp
            ir.tensors[op_output_name] = Tensor(op_output_name,np.zeros(input_shape),is_constant=False, shape=input_shape)
            ir.tensors[op_output_name].scale = ir.tensors[ir.inputs[0]].scale
            ir.tensors[op_output_name].zero_point = ir.tensors[ir.inputs[0]].zero_point
            attributes_dict = {}
            attributes_dict['name'] = op_name
            attributes_dict['attributes'] = folding_conv_attr
            attributes_dict['op_type'] = 'Conv'
            ir.graph.add_nodes_from([(op_name, attributes_dict)])

            inputs = []
            inputs.append(op_input_name)
            inputs.append(conv_W_initializer_tensor_name)
            inputs.append(conv_B_initializer_tensor_name)
            outputs = []
            outputs.append(op_output_name)

            ir.graph.nodes[op_name]['inputs'] = inputs
            ir.graph.nodes[op_name]['outputs'] = outputs

            ir.tensors[op_input_name].consumers = [op_name]
            ir.tensors[op_output_name].producer = op_name
            if ir.tensors[op_input_name].producer:
                ir.graph.add_edge(ir.tensors[op_input_name].producer, op_name)
            
            op_input_name = op_output_name
        ir.tensors[op_output_name].consumers = org_first_nodes
        for i in range(len(org_first_nodes)):
            ir.graph.add_edge(ir.tensors[op_input_name].producer, org_first_nodes[i])
            index = ir.graph.nodes[org_first_nodes[i]]['inputs'].index(ir.inputs[0])
            ir.graph.nodes[org_first_nodes[i]]['inputs'][index] = op_output_name
    return ir
'''

# Common to ONNX and TFLite
def compile_frontend_common(ir: internal_representation.IR) -> internal_representation.IR:
    #Adding Folding convolutions at the input
    # ir = add_foldingConv_nodes(ir)
    ir = frontend_hardware_agnostic.parse_nodes_params(ir)
    if not DEBUG_SIMULATE_CONCAT_REQUANT:
        ir = frontend_hardware_agnostic.update_nodes_qparams(ir)
    ir = frontend_hardware_dependant.calc_nodes_folding_factor(ir)
    folding_report_filename = os.path.join(ir.compiler_output_dir, ir.model_name+'_folding_pretiling')
    reports.generate_folding_report(folding_report_filename,ir)
    ir = frontend_hardware_dependant.duplicate_maxpool_nodes(ir)

    if DEBUG_SPLIT_LARGE_CONV:
        ir = frontend_hardware_dependant.split_large_conv_nodes(ir)
    ir = frontend_hardware_dependant.calc_nodes_quantization_params(ir)
    return ir

# Default ONNX compile_frontend
def compile_frontend(ir: internal_representation.IR,onnx_graph : onnx.onnx_ml_pb2.GraphProto) -> internal_representation.IR:
    print('At Frontend:')
    ir = onnx_parser.onnx_to_ir(ir,onnx_graph)
    ir = onnx_parser.remove_quantization_nodes(ir)
    return compile_frontend_common(ir)

# TFLite compile_frontend
def compile_frontend_tflite(ir: internal_representation.IR, tflite_graph : dict,
                            mxp_ops: list, mxp_tensor_to_offset: dict) -> internal_representation.IR:
    print('At TFLite Frontend:')
    ir = tflite_parser.tflite_to_ir(ir,tflite_graph,mxp_ops,mxp_tensor_to_offset)
    return compile_frontend_common(ir)

def main ():
    #model_def_resnet50 = onnx.load('resnet50.onnx')
    #model_def_mobilenetv2 = onnx.load('mobilenetv2.onnx')
    model_name = 'stage1_conv1'
    #model_name = 'Conv14x14_k1'
    model_name = 'resnet50_deci'
    #model_name = 'resnet50_nonuniform'
    model_name = 'resnet50_imagenetcal_nonuniform'
    #model_name = 'stage2_stage4'
    #model_name = 'stage4'
    #model_name = 'stage6'
    #model_name = 'globalavgpool'
    #model_name = 'fc2048x1000'
    model_name = 'concat'
    model_name = 'yolov5l_nx_s94sub'
    model_name =  'MaxPool7x7_k5'
    model_name = 'yolov5l_nx_s94'
    model_name = 'Concat14x14'
    #model_name = 'MultiInputConcat14x14'
    #model_name = 'MultiOutputModel'
    model_name = 'foldingfactorupdate'
    model_name = 'Conv12x8_k1'

    if platform == 'linux':
        output_dir = '/home/neuronix/dan/tsnp_output/'
        onnx_dir = '/home/neuronix/dan/snp_output/'
    else:
        output_dir = 'C:/Users/dshir/Documents/tsnp_output/'
        onnx_dir = 'C:/Users/dshir/Documents/snp_output/'
    if model_name in onnx_models_dict:
        onnx_model_name = onnx_dir+'onnx/'+onnx_models_dict[model_name]

    else:
        if DEBUG_ONNX_FORMAT_QDQ:
            onnx_model_name = onnx_dir+'onnx/'+model_name+'_pcq_qdq.onnx'
        else:
            onnx_model_name = onnx_dir+'onnx/'+model_name+'_pcq_qop.onnx'
    if DEBUG_CREATE_ORDERING_CONV:
        model_name=model_name+'_with_reordering'
    model_def = onnx.load(onnx_model_name)
    model_def = onnx.shape_inference.infer_shapes(model_def)
    onnx.checker.check_model(model_def)
    onnx_withshapes_model_name = os.path.splitext(onnx_model_name)[0]+'_withshapes.onnx'

    onnx.save(model_def, onnx_withshapes_model_name)

    compiler_output_dir = output_dir+'/'+model_name+'/'
    ir = internal_representation.IR(model_name,compiler_output_dir=compiler_output_dir)
    ir = compile_frontend(ir,model_def.graph)
    internal_representation.draw_graph_from_ir(ir,'Imported from onnx')
    if not os.path.exists(compiler_output_dir):
        os.makedirs(compiler_output_dir)
    compiler_output_ir_dir = output_dir+'/nx_ir/'
    if not os.path.exists(compiler_output_ir_dir):
        os.makedirs(compiler_output_ir_dir)
    ir_filename = compiler_output_ir_dir+model_name+'_frontend.nxir'
    ir.save(ir_filename)

if __name__ == "__main__":

    main()    