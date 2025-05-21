import onnx
import numpy as np

def getShape(tensor_shape):
    shape = []
    for dim in tensor_shape.dim:
        shape.append(dim.dim_value)
    return shape


if __name__ == "__main__":
    file_in = 'yolov8n-pose.onnx'
    file_cut = 'yolov8n-pose_cut.onnx'
    file_out = 'yolov8n-pose_edit.onnx'

    onnx.utils.extract_model(file_in,file_cut,['images'],['/model.22/cv4.2/cv4.2.2/Conv_output_0','/model.22/cv2.2/cv2.2.2/Conv_output_0','/model.22/cv3.2/cv3.2.2/Conv_output_0','/model.22/cv4.1/cv4.1.2/Conv_output_0','/model.22/cv2.1/cv2.1.2/Conv_output_0','/model.22/cv3.1/cv3.1.2/Conv_output_0','/model.22/cv4.0/cv4.0.2/Conv_output_0','/model.22/cv2.0/cv2.0.2/Conv_output_0','/model.22/cv3.0/cv3.0.2/Conv_output_0'])

    model = onnx.load(file_cut)

    input_shape = getShape(model.graph.input[0].type.tensor_type.shape)
    
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
                    output_shape = getShape(output.type.tensor_type.shape)
                    stride = input_shape[2]//output_shape[2]
                    if output_shape[1]==1:     # detection scores
                        name = 'det'+str(stride)
                        node.output[0] = name
                        output.name = name
                    if output_shape[1]==64:    # bounding boxes
                        name = 'box'+str(stride)
                        node.output[0] = name
                        output.name = name
                    if output_shape[1]==51:    # keypoints
                        name = 'kp'+str(stride)
                        for init in model.graph.initializer:
                            if init.name == node.input[1]:  # kernel
                                k_orig = init
                                k = onnx.numpy_helper.to_array(init)
                            if init.name == node.input[2]:  # bias
                                b_orig = init
                                b = onnx.numpy_helper.to_array(init)
                        ind_s = np.arange(51)[2::3]
                        ind_xy = np.setdiff1d(np.arange(51),ind_s)
                        kxy = onnx.numpy_helper.from_array(k[ind_xy,:], name=name+'_kernel_xy')
                        ks = onnx.numpy_helper.from_array(k[ind_s,:], name=name+'_kernel_s')
                        bxy = onnx.numpy_helper.from_array(b[ind_xy], name=name+'_bias_xy')
                        bs = onnx.numpy_helper.from_array(b[ind_s], name=name+'_bias_s')
                        output_xy = onnx.helper.make_tensor_value_info(name+"_xy", onnx.helper.TensorProto.FLOAT, [1,17*2]+output_shape[2:])
                        output_s = onnx.helper.make_tensor_value_info(name+"_s", onnx.helper.TensorProto.FLOAT, [1,17]+output_shape[2:])
                        conv_xy = onnx.helper.make_node(op_type=node.op_type, inputs=[node.input[0],kxy.name,bxy.name], outputs=[output_xy.name], name='conv_'+name+'_xy')
                        conv_xy.attribute.MergeFrom(node.attribute)
                        conv_s = onnx.helper.make_node(op_type=node.op_type, inputs=[node.input[0],ks.name,bs.name], outputs=[output_s.name], name='conv_'+name+'_s')
                        conv_s.attribute.MergeFrom(node.attribute)
                        initializer_append += [kxy,ks,bxy,bs]
                        initializer_remove += [k_orig,b_orig]
                        node_append += [conv_xy,conv_s]
                        node_remove += [node]
                        output_append += [output_xy,output_s]
                        output_remove += [output]
    
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
    onnx.save(model, file_out)
