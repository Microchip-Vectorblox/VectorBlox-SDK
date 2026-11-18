import argparse
from pathlib import Path
import numpy as np
import onnx

# get onnx model ready for quantization
def get_shape(tensor_shape):
    shape = []
    for dim in tensor_shape.dim:
        shape.append(dim.dim_value)
    return shape

# model_inputs = ['x']
# model_outputs = ['conv2d_214', 'conv2d_217', 'conv2d_211',
#                 'conv2d_223', 'conv2d_226', 'conv2d_220',
#                 'conv2d_232', 'conv2d_235', 'conv2d_229']
# onnx.utils.extract_model(tutorial_path/'yolov9s-spn_768x512.13.onnx', tutorial_path/'yolov9s-spn_768x512.cut.onnx', model_inputs, model_outputs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    args = parser.parse_args()

    model = onnx.load(args.model)
    input_shape = get_shape(model.graph.input[0].type.tensor_type.shape)
    
    initializer_append = []
    initializer_remove = []
    node_append = []
    node_remove = []
    output_append = []
    output_remove = []

    for node in model.graph.node:
        if node.op_type == 'Conv':
            for output in model.graph.output:
                if output.name == node.output[0]:
                    output_shape = get_shape(output.type.tensor_type.shape)
                    stride = input_shape[2]//output_shape[2]
                    if output_shape[1]==64:
                        # raw bounding boxes; just label the output
                        output_name = f'box{stride}'
                        node.output[0] = output_name
                        output.name = output_name
                    elif output_shape[1]==1:
                        # detection scores; add sigmoid node to output to improve quantization
                        output_name = f'det{stride}'
                        sigmoid_node = onnx.helper.make_node(
                            op_type="Sigmoid",
                            inputs=node.output,
                            outputs=[output_name],
                            name=f"sigmoid_str{stride}",
                        )
                        output_sigmoid = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)
                        output_remove.append(output)
                        output_append.append(output_sigmoid)
                        node_append.append(sigmoid_node)
                    elif output_shape[1]==9:
                        # raw pose; separate the outputs to improve quantization
                        name = f'pose{stride}'
                        for init in model.graph.initializer:
                            if init.name == node.input[1]:  # kernel
                                k_orig = init
                                k = onnx.numpy_helper.to_array(init)
                            if init.name == node.input[2]:  # bias
                                b_orig = init
                                b = onnx.numpy_helper.to_array(init)
                        ind_xy = np.arange(0,2)
                        ind_z = np.arange(2,3)
                        ind_rot = np.arange(3,9)
                        kxy = onnx.numpy_helper.from_array(k[ind_xy,:], name=name+'_kernel_xy')
                        kz = onnx.numpy_helper.from_array(k[ind_z,:], name=name+'_kernel_z')
                        krot = onnx.numpy_helper.from_array(k[ind_rot,:], name=name+'_kernel_rot')
                        bxy = onnx.numpy_helper.from_array(b[ind_xy], name=name+'_bias_xy')
                        bz = onnx.numpy_helper.from_array(b[ind_z], name=name+'_bias_z')
                        brot = onnx.numpy_helper.from_array(b[ind_rot], name=name+'_bias_rot')
                        output_xy = onnx.helper.make_tensor_value_info(name+"_xy", onnx.helper.TensorProto.FLOAT, [1,2]+output_shape[2:])
                        output_z = onnx.helper.make_tensor_value_info(name+"_z", onnx.helper.TensorProto.FLOAT, [1,1]+output_shape[2:])
                        output_rot = onnx.helper.make_tensor_value_info(name+"_rot", onnx.helper.TensorProto.FLOAT, [1,6]+output_shape[2:])
                        conv_xy = onnx.helper.make_node(op_type=node.op_type, inputs=[node.input[0],kxy.name,bxy.name], outputs=[output_xy.name], name='conv_'+name+'_xy')
                        conv_xy.attribute.MergeFrom(node.attribute)
                        conv_z = onnx.helper.make_node(op_type=node.op_type, inputs=[node.input[0],kz.name,bz.name], outputs=[output_z.name], name='conv_'+name+'_s')
                        conv_z.attribute.MergeFrom(node.attribute)
                        conv_rot = onnx.helper.make_node(op_type=node.op_type, inputs=[node.input[0],krot.name,brot.name], outputs=[output_rot.name], name='conv_'+name+'_s')
                        conv_rot.attribute.MergeFrom(node.attribute)
                        initializer_append += [kxy,kz,krot,bxy,bz,brot]
                        initializer_remove += [k_orig,b_orig]
                        node_append += [conv_xy,conv_z,conv_rot]
                        node_remove += [node]
                        output_append += [output_xy,output_z,output_rot]
                        output_remove += [output]
                    else:
                        print('WARNING: onnx model output with enexpected shape')

    for init in initializer_append:
        model.graph.initializer.append(init)
    for init in initializer_remove:
        model.graph.initializer.remove(init)
    for node in node_append:
        model.graph.node.append(node)
    for node in node_remove:
        model.graph.node.remove(node)
    for output in output_append:
        model.graph.output.append(output)
    for output in output_remove:
        model.graph.output.remove(output)

    onnx.checker.check_model(model)
    onnx.save(model, args.model.replace('.onnx','.prep.onnx'))

if __name__ == "__main__":
    main()