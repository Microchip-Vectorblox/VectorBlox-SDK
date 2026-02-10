import sys
sys.path.append('.')
import base64
import math
import numpy as np
import networkx as nx
import common.internal_representation as internal_representation
from common.tensor_ir import Tensor
from common.hw_config import NON_REQUANTING_OPS, MULTIPLE_INPUT_NON_REQUANTING_OPS, TFLITE_REQUANT
import copy

def get_producer_op(tflite_graph, tensor_index) -> int:
    operators = tflite_graph['subgraphs'][0]['operators']
    producer_ops = []
    for i, op in enumerate(operators):
        if tensor_index in op['outputs']:
            producer_ops.append(i)
    # No producer op if all tensors are inputs
    if len(producer_ops) == 0:
        return None
    assert len(producer_ops) == 1
    return producer_ops[0]

def get_consumer_ops(tflite_graph, tensor_index) -> list:
    operators = tflite_graph['subgraphs'][0]['operators']
    consumer_ops = []
    for i, op in enumerate(operators):
        if tensor_index in op['inputs']:
            consumer_ops.append(i)
    return consumer_ops

def get_op_type(tflite_graph, op_idx):
    operators = tflite_graph['subgraphs'][0]['operators']
    opcode_index = operators[op_idx]['opcode_index']
    operator_codes = tflite_graph['operator_codes']
    return operator_codes[opcode_index]['builtin_code']

# Whether this PAD op can be skipped
def skip_pad_op(tflite_graph, operator_idx) -> bool:

    # Get the output tensor of the PAD operator
    operators = tflite_graph['subgraphs'][0]['operators']
    pad_outputs = operators[operator_idx]['outputs']
    assert len(pad_outputs) == 1

    # Get the consumers of this output tensor
    # If there is more than 1, don't skip the pad
    consumer_ops = get_consumer_ops(tflite_graph, pad_outputs[0])
    for idx in range(len(consumer_ops)):
        # Get the consumer node type
        # If it's not a CONV_2D, don't skip
        consumer_type = get_op_type(tflite_graph, consumer_ops[idx])
        if consumer_type != "CONV_2D" and consumer_type != "MAX_POOL_2D":
            return False

    # Here can also check that the CONV has VALID padding, but even if
    # it's SAME, can just add this PAD's values to that padding
    return True

def skip_logistic_mul_op(tflite_graph, operator_idx) -> bool:
    
    # Get the input tensor of the Logistic operator
    operators = tflite_graph['subgraphs'][0]['operators']
    logistic_outputs = operators[operator_idx]['outputs']
    assert len(logistic_outputs) == 1

    # Get the consumers of this output tensor
    # If there is more than 1, don't skip the Logistic
    consumer_ops = get_consumer_ops(tflite_graph, logistic_outputs[0])
    if len(consumer_ops) != 1:
        return False

    # Get the consumer node type
    # If it's not a MUL, don't skip
    consumer_type = get_op_type(tflite_graph, consumer_ops[0])
    if consumer_type != "MUL":
        return False

    # Here can also check that the CONV has VALID padding, but even if
    # it's SAME, can just add this PAD's values to that padding
    return True

def skip_cast_gather_op(tflite_graph, operator_idx) -> bool:
    
    # Get the input tensor of the Logistic operator
    operators = tflite_graph['subgraphs'][0]['operators']
    cast_outputs = operators[operator_idx]['outputs']
    assert len(cast_outputs) == 1

    # Get the consumers of this output tensor
    # If there is more than 1, don't skip the Logistic
    consumer_ops = get_consumer_ops(tflite_graph, cast_outputs[0])
    if len(consumer_ops) != 1:
        return False

    # Get the consumer node type
    # If it's not a MUL, don't skip
    consumer_type = get_op_type(tflite_graph, consumer_ops[0])
    if consumer_type != "GATHER":
        return False

    # Here can also check that the CONV has VALID padding, but even if
    # it's SAME, can just add this PAD's values to that padding
    return True

# Whether this op should be skipped or not when creating IR
def skip_op(tflite_graph, operator_idx) -> bool:

    # Get the operator type
    op_type = get_op_type(tflite_graph, operator_idx)

    # Check each type

    # Pad can be skipped if the next op is a Conv, since it
    # will be merged into the padding of that Conv
    if op_type == "PAD":
        return skip_pad_op(tflite_graph, operator_idx)

    # Quantize are currently handled by
    # frontend_hardware_agnostic.update_nodes_qparams
    # which checks if qparams of 2 nodes are different and calls
    # insert_requant_node_between
    elif op_type == "QUANTIZE":
        return True
    
    elif op_type == "LOGISTIC":
        return skip_logistic_mul_op(tflite_graph, operator_idx)

    elif op_type == "MUL":
        return True

    elif op_type == "CAST":
        return skip_cast_gather_op(tflite_graph, operator_idx)
    
    elif op_type == "GATHER":
        return True
    
    elif op_type == "RESHAPE":
        return True
    
    return False

# Given tensor, check if its producer is a Pad, and if so, return the pad values
def get_input_pad_values(tflite_graph, tensor_index):
    subgraph = tflite_graph['subgraphs'][0]

    # Check if the input is a PAD
    producer_idx = get_producer_op(tflite_graph, tensor_index)
    if producer_idx is None:    # This Conv has no input op
        return None
    producer_type = get_op_type(tflite_graph, producer_idx)
    if producer_type != "PAD":
        return None

    # Extract padding values from the inputs of the PAD operator
    producer = subgraph['operators'][producer_idx]
    pad_inputs = producer['inputs']
    # Input 0 is the input tensor, Input 1 is the pad values
    assert len(pad_inputs) >= 2
    padding_tensor_index = pad_inputs[1]
    padding_tensor = subgraph['tensors'][padding_tensor_index]
    # Get the buffer for the padding tensor
    padding_buffer_index = padding_tensor['buffer']
    buffer_data = tflite_graph['buffers'][padding_buffer_index]['data']
    assert buffer_data

    # Decode the buffer into padding values (assumes INT32 encoding)
    assert isinstance(buffer_data, list)
    raw_data = bytes(buffer_data)
    tensor_data = np.frombuffer(raw_data, dtype=np.int32).copy()
    assert len(tensor_data) == 8    # 4 dimensions * 2 for before/after

    # input_tensor_shape = subgraph['tensors'][pad_inputs[0]]['shape']
    # if ((input_tensor_shape[2] % 16) != 0):
        # input_padding = 16 - (input_tensor_shape[2] % 16)
        # if (input_padding == 1):
            # tensor_data[5] = 0
    
    # Return before/after for height and width
    return tensor_data[2], tensor_data[3], tensor_data[4], tensor_data[5]

# Get the tensor data from a tensor index in the graph
def get_tensor_data_from_tensor_index(tflite_graph, tensor_index, dtype):
    subgraph = tflite_graph['subgraphs'][0]
    tensor = subgraph['tensors'][tensor_index]
    buffer_index = tensor['buffer']
    buffer_data = tflite_graph['buffers'][buffer_index]['data']
    assert buffer_data
    # Decode the buffer into values
    assert isinstance(buffer_data, list)
    raw_data = bytes(buffer_data)
    tensor_data = np.frombuffer(raw_data, dtype=dtype)
    return tensor_data

# Given a strided slice operator, check its strides are currently handled
def check_strided_slice_attributes(operator, tflite_graph):
    # Inputs are input data, begin, end, strides
    begin_tensor_data   = get_tensor_data_from_tensor_index(tflite_graph, operator['inputs'][1], np.int32)
    end_tensor_data     = get_tensor_data_from_tensor_index(tflite_graph, operator['inputs'][2], np.int32)
    strides_tensor_data = get_tensor_data_from_tensor_index(tflite_graph, operator['inputs'][3], np.int32)

    assert len(begin_tensor_data) == 4
    assert len(end_tensor_data) == 4
    assert len(strides_tensor_data) == 4

    # For now, handle strides of 1
    assert (strides_tensor_data[0], strides_tensor_data[1], strides_tensor_data[2], strides_tensor_data[3]) == (1, 1, 1, 1)

    # Assert the strided slice is channels only
    assert (begin_tensor_data[0], begin_tensor_data[1], begin_tensor_data[2]) == (0, 0, 0)
    assert (end_tensor_data[0], end_tensor_data[1], end_tensor_data[2]) == (0, 0, 0)

    # Return the begin and end channel
    return begin_tensor_data[3], end_tensor_data[3]

def get_numpy_dtype(tflite_type):
    if tflite_type == 'INT8':
        return np.int8
    elif tflite_type == 'INT32':
        return np.int32
    elif tflite_type == 'UINT8':
        return np.uint8
    raise ValueError(f"Unsupported TFLite tensor type: {tflite_type}")

def get_tflite_attr_type(attr_name, attr_value):
    attr_type = "FLOAT" if isinstance(attr_value, float) else \
                "INT" if isinstance(attr_value, int) else \
                "STRING" if isinstance(attr_value, str) else \
                "TENSOR" if attr_name == "t" else \
                "FLOATS" if isinstance(attr_value, list) and all(isinstance(v, float) for v in attr_value) else \
                "INTS" if isinstance(attr_value, list) and all(isinstance(v, int) for v in attr_value) else \
                "STRINGS" if isinstance(attr_value, list) and all(isinstance(v, str) for v in attr_value) else None
    return attr_type

def tflite_tensor_to_onnx_tensor(tensor_data):
    # TFLite is [output_channels, kernel_height, kernel_width, input_channels]
    # ONNX is [output_channels, input_channels, kernel_height, kernel_width]
    if tensor_data.ndim == 4:
        tensor_data = np.transpose(tensor_data, (0, 3, 1, 2))
    return tensor_data

def tflite_tensor_shape_to_onnx_tensor_shape(tensor_shape):
    # Weights
    #   TFLite is [output_channels, kernel_height, kernel_width, input_channels]
    #   ONNX is [output_channels, input_channels, kernel_height, kernel_width]
    # Activations
    #   TFLite is [batch, height, width, channels]
    #   ONNX is [batch, channels, height, width]
    if len(tensor_shape) == 4:
        return [tensor_shape[0], tensor_shape[3], tensor_shape[1], tensor_shape[2]]
    return tensor_shape

def get_conv_kernel_shape_attribute(operator, tflite_graph) -> list:
    # Get the weights tensor shape
    weights_index = operator['inputs'][1]
    weights_tensor_info = tflite_graph['subgraphs'][0]['tensors'][weights_index]
    # Convert to ONNX format
    # TFLite is [output_channels, kernel_height, kernel_width, input_channels]
    # ONNX is [output_channels, input_channels, kernel_height, kernel_width]
    # But the IR only stores the [kernel_height, kernel_width] part
    w_shape = weights_tensor_info['shape']
    return [w_shape[1], w_shape[2]]

def calculate_same_padding(input_dim, kernel_size, stride):
    # Calculate the output dimension for SAME padding
    output_dim = math.ceil(input_dim / stride)

    # Calculate total padding required for this dimension
    padding_total = max((output_dim - 1) * stride + kernel_size - input_dim, 0)

    # Split the total padding into two parts: before and after
    pad_before = padding_total // 2
    pad_after = padding_total - pad_before
    return pad_before, pad_after

def get_conv_padding_attribute(operator, tflite_graph, attr_dict) -> list:
    """
    Convert TFLite padding to ONNX `pads`.

    Returns:
    - list of four padding values [pad_top, pad_bottom, pad_left, pad_right]
    """

    # input shape is e.g. [1, 16, 16, 16], which is [batch, height, width, channels]
    input_tensor_index = operator['inputs'][0]
    input_tensor = tflite_graph['subgraphs'][0]['tensors'][input_tensor_index]
    input_height, input_width = input_tensor['shape'][1], input_tensor['shape'][2]

    # kernel shape attribute is only [kernel_height, kernel_width]
    kernel_height = attr_dict['kernel_shape'][0]
    kernel_width = attr_dict['kernel_shape'][1]

    stride_height, stride_width = attr_dict['stride_h'], attr_dict['stride_w']
    padding_type = attr_dict['padding'] # "SAME" or "VALID"

    if padding_type == "SAME":
        pad_top, pad_bottom = calculate_same_padding(input_height, kernel_height, stride_height)
        pad_left, pad_right = calculate_same_padding(input_width, kernel_width, stride_width)
    elif padding_type == "VALID":
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    else:
        raise ValueError("Invalid padding type. Must be 'SAME' or 'VALID'.")

    # If the input to this Conv was a PAD, add its PAD values to this Conv
    prev_pad_values = get_input_pad_values(tflite_graph, input_tensor_index)
    if prev_pad_values:
        pad_top    += prev_pad_values[0]
        pad_bottom += prev_pad_values[1]
        pad_left   += prev_pad_values[2]
        pad_right  += prev_pad_values[3]

    return [pad_top, pad_bottom, pad_left, pad_right]

def get_logistic_mul_params(operator, tflite_graph) -> list:
    output_tensor_index = operator['outputs'][0]
    silu_input_tensor = tflite_graph['subgraphs'][0]['tensors'][output_tensor_index]

    consumer_ops = get_consumer_ops(tflite_graph, output_tensor_index)
    if (len(consumer_ops) != 0):
        consumer_type = get_op_type(tflite_graph, consumer_ops[0])
    else:
        return False, None, None

    if (consumer_type != "LOGISTIC"):
        return False, None, None
    
    operators = tflite_graph['subgraphs'][0]['operators']
    consumer_type = get_op_type(tflite_graph, consumer_ops[1])
    if (consumer_type == "MUL"):
        out_idx = operators[consumer_ops[1]]['outputs'][0]
        silu_output_tensor = tflite_graph['subgraphs'][0]['tensors'][out_idx]
        return True, silu_input_tensor, silu_output_tensor
    
    return False, silu_input_tensor, None

def get_quant_cast_gather_params(operator, tflite_graph) -> list:
    output_tensor_index = operator['outputs'][0]
    silu_input_tensor = tflite_graph['subgraphs'][0]['tensors'][output_tensor_index]

    consumer_ops = get_consumer_ops(tflite_graph, output_tensor_index)
    if (len(consumer_ops) != 0):
        consumer_type = get_op_type(tflite_graph, consumer_ops[0])
    else:
        return False, None, None, None

    if (consumer_type != "QUANTIZE"):
        return False, None, None, None
    
    operators = tflite_graph['subgraphs'][0]['operators']
    quant_out_idx = operators[consumer_ops[0]]['outputs'][0]
    quant_consumer_ops = get_consumer_ops(tflite_graph, quant_out_idx)
    quant_consumer_type = get_op_type(tflite_graph, quant_consumer_ops[0])
    if (quant_consumer_type == "CAST"):
        cast_out_idx = operators[quant_consumer_ops[0]]['outputs'][0]
        cast_consumer_ops = get_consumer_ops(tflite_graph, cast_out_idx)
        cast_consumer_type = get_op_type(tflite_graph, cast_consumer_ops[0])
        if (cast_consumer_type == "GATHER"):
            gather_out_idx = operators[cast_consumer_ops[0]]['outputs'][0]
            gather_consumer_ops = get_consumer_ops(tflite_graph, gather_out_idx)
            if (len(gather_consumer_ops) > 0):
                gather_consumer_type = get_op_type(tflite_graph, gather_consumer_ops[0])
            else:
                gather_consumer_type = ''
            if (gather_consumer_type == 'RESIZE_NEAREST_NEIGHBOR') and (len(gather_consumer_ops) == 1):
                return False, None, None, None
            else:
                lut_info = operators[cast_consumer_ops[0]]['inputs'][0]
                lut = get_tensor_data_from_tensor_index(tflite_graph, lut_info, np.int8)
                lut_int8 = np.zeros((256,), dtype=np.int8)
                for idx in range(256):
                    if (idx < 128):
                        lut_int8[idx] = lut[128+idx]
                    else:
                        lut_int8[idx] = lut[idx-128]
                silu_output_tensor = tflite_graph['subgraphs'][0]['tensors'][gather_out_idx] 
                return True, silu_input_tensor, silu_output_tensor, lut_int8

    return False, None, None, None

def get_silu_attribute(operator, tflite_graph) -> list:
    
    is_valid_pattern, input_tensor, output_tensor = get_logistic_mul_params(operator, tflite_graph)
    if not is_valid_pattern:
        is_valid_pattern, input_tensor, output_tensor, lut = get_quant_cast_gather_params(operator, tflite_graph)
        if is_valid_pattern:
            silu_attributes = {}
            silu_attributes['input_scale'] = input_tensor['quantization']['scale']
            silu_attributes['input_zp'] = input_tensor['quantization']['zero_point']
            silu_attributes['output_scale'] = output_tensor['quantization']['scale']
            silu_attributes['output_zp'] = output_tensor['quantization']['zero_point']
            silu_attributes['lut'] = lut
            return silu_attributes
    else:
        silu_attributes = {}
        silu_attributes['input_scale'] = input_tensor['quantization']['scale']
        silu_attributes['input_zp'] = input_tensor['quantization']['zero_point']
        silu_attributes['output_scale'] = output_tensor['quantization']['scale']
        silu_attributes['output_zp'] = output_tensor['quantization']['zero_point']
        return silu_attributes
    
    return None

def get_pool_padding_attribute(operator, tflite_graph, attr_dict) -> list:
    input_tensor_index = operator['inputs'][0]
    input_tensor = tflite_graph['subgraphs'][0]['tensors'][input_tensor_index]
    input_height, input_width = input_tensor['shape'][1], input_tensor['shape'][2]
    if attr_dict["padding"] == "SAME":
        pad_top, pad_bottom = calculate_same_padding(input_height, attr_dict['filter_height'], attr_dict['stride_h'])
        pad_left, pad_right = calculate_same_padding(input_width, attr_dict['filter_width'], attr_dict['stride_w'])
    else:
        assert attr_dict["padding"] == "VALID"
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    
    # If the input to this Conv was a PAD, add its PAD values to this Conv
    prev_pad_values = get_input_pad_values(tflite_graph, input_tensor_index)
    if prev_pad_values:
        pad_top    += prev_pad_values[0]
        pad_bottom += prev_pad_values[1]
        pad_left   += prev_pad_values[2]
        pad_right  += prev_pad_values[3]
        
    return [pad_top, pad_bottom, pad_left, pad_right]

def tflite_attrs_to_dict(builtin_options, opcode, operator, tflite_graph):

    # Helper function to process attributes based on TFLite type
    def process_attr(value, attr_type):
        if attr_type == "STRING":
            return value.decode() if isinstance(value, bytes) else value
        elif attr_type == "TENSOR":
            assert False
            #return import_tensor(value) # See onnx parser
        elif attr_type in ["FLOATS", "INTS"]:
            return list(value) if isinstance(value, list) else [value]
        elif attr_type == "STRINGS":
            return [v.decode() if isinstance(v, bytes) else v for v in value]
        else:
            return value

    # Process each attribute in builtin_options based on its type
    attr_dict = {}
    for attr_name, attr_value in builtin_options.items():
        # Process attribute and add to attr_dict
        attr_type = get_tflite_attr_type(attr_name, attr_value)
        if attr_type:
            attr_dict[attr_name] = process_attr(attr_value, attr_type)

    # If this is a Conv2D, add some other attributes that are in ONNX
    if (opcode == 'CONV_2D'):
        # 'kernel_shape' attribute -- not in TFLite because it can be inferred
        # from the weight tensor.
        attr_dict['kernel_shape'] = get_conv_kernel_shape_attribute(operator, tflite_graph)

        # 'pads' attribute -- not in TFLite because it has 'padding' which is
        # 'SAME' or 'VALID', but ONNX has the actual pads
        attr_dict['pads'] = get_conv_padding_attribute(operator, tflite_graph, attr_dict)

        # 'strides' attribute
        attr_dict['strides'] = [attr_dict['stride_h'], attr_dict['stride_w']]

        # other (unused)
        #attr_dict['dilations'] = [attr_dict['dilation_h_factor'], attr_dict['dilation_w_factor']]
        #attr_dict['group'] = 1

        # 'silu' attribute if output of CONV_2D is logistic and Mul
        attr_dict['activation_silu'] = get_silu_attribute(operator, tflite_graph)

    # Temporary: Also add Max Pool attributes
    # (for debugging only, will be on MXP later)
    elif opcode == 'MAX_POOL_2D':
        attr_dict['kernel_shape'] = [attr_dict['filter_height'], attr_dict['filter_width']]
        attr_dict['strides'] = [attr_dict['stride_h'], attr_dict['stride_w']]
        attr_dict['pads'] = get_pool_padding_attribute(operator, tflite_graph, attr_dict)

        # other (unused)
        #attr_dict['ceil_mode'] = 0
    elif opcode == 'AVERAGE_POOL_2D':
        attr_dict['kernel_shape'] = [attr_dict['filter_height'], attr_dict['filter_width']]
        if attr_dict['kernel_shape']==[2,2]:
            attr_dict['kernel_shape'] = [3,3]
            attr_dict['strides'] = [attr_dict['stride_h'], attr_dict['stride_w']]
            attr_dict['pads'] = [1,1,1,1]
        else:
            raise ValueError("Only 2x2 Average Pool supported")

        # other (unused)
        #attr_dict['ceil_mode'] = 0
        #attr_dict['count_include_pad'] = 0

    return attr_dict

def tflite_to_onnx_op_type(tflite_op_type: str) -> str:
    op_map = {
        'ADD': 'Add',
        'SUB': 'Sub',
        'MUL': 'Mul',
        'DIV': 'Div',
        'MATRIX_MULTIPLICATION': 'MatMul',
        'CONV_2D': 'Conv',
        'RELU': 'Relu',
        'SIGMOID': 'Sigmoid',
        'LOGISTIC': 'Logistic',
        'TANH': 'Tanh',
        'FLATTEN': 'Flatten',
        'RESHAPE': 'Reshape',
        'TRANSPOSE': 'Transpose',
        'CONCATENATION': 'Concat',
        'MAX_POOL_2D': 'MaxPool',
        'AVERAGE_POOL_2D': 'AveragePool',
        'BATCH_NORM': 'BatchNormalization',
        'LEAKY_RELU': 'LeakyRelu',
        'SOFTMAX': 'Softmax',
        'DROPOUT': 'Dropout',
        'FULLY_CONNECTED': 'Gemm',
        'IDENTITY': 'Identity',
        'STRIDED_SLICE': 'Identity',
        'SPLIT': 'Identity',
        'RESIZE_NEAREST_NEIGHBOR': 'Resize',
        'GATHER': 'Gather',
        'CAST': 'Cast',
    }

    if tflite_op_type in op_map:
        return op_map[tflite_op_type]
    assert False, "Unsupported TFLite Op: " + tflite_op_type

# Avoid long names. Maybe not needed if using .tflite instead of .json, but
# the .json contains names like:
#"model/tf.nn.relu_38/Relu;model/tf.math.add_52/Add;model/tf.nn.convolution_55/
# convolution;model/tf.nn.convolution_38/convolution;Const_118"
def get_short_tensor_name(idx: int) -> str:
    return "T" + str(idx)

# Get an op which consumes the output tensor of the input op.
# For now, assumes 1 output tensor of input op and 1 consumer.
# Can extend to more general cases later.
def get_next_op_idx(tflite_graph, op_idx) -> int:
    operators = tflite_graph['subgraphs'][0]['operators']
    outputs = operators[op_idx]['outputs']
    assert len(outputs) == 1
    consumer_ops = get_consumer_ops(tflite_graph, outputs[0])
    assert len(consumer_ops) == 1
    return consumer_ops[0]

# Check that a Quantize op will be handled by update_nodes_qparams and
# update the qparams to account for the skipped Quantize op
def update_qparams_for_skipped_quantize(ir, tflite_graph, quantize_idx, quantize_input_idx):
    # Quantize operations are inserted before Concat, to make sure all the
    # inputs have the same scale and zero point. This is handled automatically
    # by frontend_hardware_agnostic.update_nodes_qparams, so check that this
    # Quantize operation will be handled.

    # Get the next node and check if it is a Concat. If not, check if it is
    # an op which does not requantize, since that might also be an input to
    # a Concat. This is commonly the case with Resize.
    intermediate_ops = []
    next_op_idx = get_next_op_idx(tflite_graph, quantize_idx)
    next_op_type = tflite_to_onnx_op_type(get_op_type(tflite_graph, next_op_idx))
    found_concat = False
    while next_op_type in NON_REQUANTING_OPS:
        # Check if handled by frontend_hardware_agnostic.update_nodes_qparams
        if next_op_type in MULTIPLE_INPUT_NON_REQUANTING_OPS:
            found_concat = True
            break
        intermediate_ops.append(next_op_idx)
        next_op_idx = get_next_op_idx(tflite_graph, next_op_idx)
        next_op_type = tflite_to_onnx_op_type(get_op_type(tflite_graph, next_op_idx))

    # If the Quantize op is not an input to a Concat, or to a node which does
    # not change the quantization and itself goes to a Concat, then
    # frontend_hardware_agnostic.update_nodes_qparams will not handle this case.
    #
    # Such a case has not been encountered yet, but some possible solutions are:
    # - Update frontend_hardware_agnostic.update_nodes_qparams to look for all
    #   cases where requantize is needed, not just before a Concat
    # - Add code to call insert_requant_node_between for each Quantize op rather
    #   than rely on update_nodes_qparams, and either don't call update_nodes_qparams
    #   or make sure it does not add redundant requantization
    # - Resize and MaxPool can also be made requanting ops as they are mapped to
    #   conv op that includes rq block (need to update rq params to support this)
    if not found_concat:
        assert False, "Found Quantize Op not handled by update_nodes_qparams"

    # Get the quantize input tensor
    operators = tflite_graph['subgraphs'][0]['operators']
    curr_op_type = get_op_type(tflite_graph, quantize_idx)
    if (curr_op_type == 'GATHER'):
        assert len(operators[quantize_idx]['inputs']) == 2
    else:
        assert len(operators[quantize_idx]['inputs']) == 1
    
    consumer_type = get_op_type(tflite_graph, quantize_idx - 2)
    quantize_input = ir.tensors[get_short_tensor_name(quantize_input_idx)]

    # Update the intermediate_ops to have the qparams of the quantize input.
    # These are all in NON_REQUANTING_OPS so they don't change the qparams.
    # The Concat output will still have the qparams of the quantize output.
    for op_idx in intermediate_ops:
        assert len(operators[op_idx]['outputs']) == 1
        output_idx = operators[op_idx]['outputs'][0]
        output = ir.tensors[get_short_tensor_name(output_idx)]
        if (consumer_type == 'LOGISTIC'):
            idx = operators[quantize_idx]['inputs'][0]
            output.scale = ir.tensors[get_short_tensor_name(idx)].scale
            output.zero_point = ir.tensors[get_short_tensor_name(idx)].zero_point
        else:
            output.scale = quantize_input.scale
            output.zero_point = quantize_input.zero_point
    return ir

# Import ir constants (with data) and activations (just shape)
def add_tflite_tensors(ir, tflite_graph):
    # Iterate through each tensor in the TFLite JSON
    for tensor_idx, tensor in enumerate(tflite_graph['subgraphs'][0]['tensors']):
        if tensor == {}: # Happens if graph is cut, e.g., for unit tests
            continue
        if tensor == {'type': 'FLOAT32', 'buffer': 0, 'is_variable': False, 'has_rank': False}:
            continue     # Happens for unused tensors in cut graphs
        #tensor_name = tensor['name']
        tensor_name = get_short_tensor_name(tensor_idx)
        buffer_index = tensor['buffer']
        tflite_tensor_shape = tensor.get('shape', [])
        tensor_shape = tflite_tensor_shape_to_onnx_tensor_shape(tflite_tensor_shape)
        tensor_type = tensor['type']
        assert tensor['is_variable'] is False
        dtype = get_numpy_dtype(tensor_type)

        # Some tensors are not quantized. For example, PAD has as input for
        # the paddings, which are a [4, 2] tensor containing the before and
        # after padding for all 4 of the data dimensions.
        scale_list = tensor['quantization'].get('scale')
        zp_list = tensor['quantization'].get('zero_point')

        # Get the buffer data if it's present
        buffer_data = tflite_graph['buffers'][buffer_index].get('data')

        if scale_list is None:
            assert zp_list is None
            assert (dtype is np.int32) or (dtype is np.int8)

            # Check the node types that use this tensor
            consumers = get_consumer_ops(tflite_graph, tensor_idx)
            assert len(consumers) > 0
            consumer_op_types = [get_op_type(tflite_graph, c) for c in consumers]
            # For now, assert all op types are the same
            assert all(op == consumer_op_types[0] for op in consumer_op_types)
            consumer_op_type = consumer_op_types[0]

            # Check if no need to add tensor, because it will not be used. This is done for:
            # - Pad (the tensor with the paddings. Unused since it is fused with the Conv)
            if consumer_op_type in ["PAD"]:
                continue

            # Other tensors have no quantization parameters but we do
            # need to add them. For example:
            # - Resize (the tensor with the resize dimensions)
            # - Strided Slice (begin, end, strides)
            if consumer_op_type not in ["RESIZE_NEAREST_NEIGHBOR", "STRIDED_SLICE", "SPLIT", "GATHER", "RESHAPE"]:
                raise ValueError ("Found new op without quantization: " + consumer_op_type)

            ir.tensors[tensor_name] = Tensor(tensor_name, data=None, is_constant=True, shape=list(tensor_shape))

        elif buffer_data is None:
            # Only add shape for activations
            # TODO: Check this for a graph with intermediate activations
            # (currently this adds inputs/outputs but those are added later as well)
            assert len(scale_list) == 1 and len(zp_list) == 1
            # Note: Activation tensors also have min/max (in addition to scale, zp)
            scale = np.array(scale_list[0])
            if TFLITE_REQUANT:
                zp = np.array(zp_list[0])
            else:
                zp = np.array(zp_list[0] + 128) # TFLite is int8, ONNX is uint8
            ir.tensors[tensor_name] = Tensor(tensor_name, None, is_constant=False, shape=list(tensor_shape),\
                                             scale=scale, zero_point=zp)
        else:
            # If not a list, decode base64 data to a numpy array
            #raw_data = base64.b64decode(buffer_data)
            assert isinstance(buffer_data, list)
            raw_data = bytes(buffer_data)
            # Convert list of ints to bytes
            # If uint8, numpy automatically will convert it to int8
            tensor_data = np.frombuffer(raw_data, dtype=dtype)
            # Reshape if shape is provided
            tensor_data = tensor_data.reshape(tflite_tensor_shape)
            # Transpose to ONNX shape
            tensor_data = tflite_tensor_to_onnx_tensor(tensor_data)
            scale = np.array(scale_list)
            zp = np.array(zp_list) # This is not used since assumed to be 0
            ir.tensors[tensor_name] = Tensor(tensor_name, tensor_data, is_constant=True, shape=list(tensor_data.shape),\
                                             scale=scale, zero_point=zp)

# Import graph inputs
def add_tflite_inputs(ir, tflite_graph):
    # Iterate through input tensors by their indices in the "inputs" list
    for input_index in tflite_graph['subgraphs'][0]['inputs']:
        tensor = tflite_graph['subgraphs'][0]['tensors'][input_index]
        #tensor_name = tensor['name']
        tensor_name = get_short_tensor_name(input_index)
        tflite_tensor_shape = tensor.get('shape', [])
        tensor_shape = tflite_tensor_shape_to_onnx_tensor_shape(tflite_tensor_shape)
        tensor_type = tensor.get('type')
        dtype = get_numpy_dtype(tensor_type)

        scale_list = tensor['quantization']['scale']
        zp_list = tensor['quantization']['zero_point']
        assert len(scale_list) == 1 and len(zp_list) == 1
        scale = np.array(scale_list[0])
        if TFLITE_REQUANT:
            zp = np.array(zp_list[0])
        else:
            zp = np.array(zp_list[0] + 128) # TFLite is int8, ONNX is uint8

        ir.inputs.append(tensor_name)
        ir.tensors[tensor_name] = Tensor(tensor_name,np.zeros(tensor_shape,dtype=dtype),is_constant=False, shape=tensor_shape,\
                                         scale=scale, zero_point=zp)
        # Remember for the sequencer
        ir.tensors_from_mxp.add(tensor_name)

# Import graph outputs
def add_tflite_outputs(ir, tflite_graph):
    # Iterate through output tensors by their indices in the "output" list
    for output_index in tflite_graph['subgraphs'][0]['outputs']:
        tensor = tflite_graph['subgraphs'][0]['tensors'][output_index]
        #tensor_name = tensor['name']
        tensor_name = get_short_tensor_name(output_index)
        tflite_tensor_shape = tensor.get('shape', [])
        tensor_shape = tflite_tensor_shape_to_onnx_tensor_shape(tflite_tensor_shape)
        tensor_type = tensor.get('type')
        dtype = get_numpy_dtype(tensor_type)

        scale_list = tensor['quantization']['scale']
        zp_list = tensor['quantization']['zero_point']
        assert len(scale_list) == 1 and len(zp_list) == 1
        scale = np.array(scale_list[0])
        if TFLITE_REQUANT:
            zp = np.array(zp_list[0])
        else:
            zp = np.array(zp_list[0]) + 128 # TFLite is int8, ONNX is uint8

        ir.outputs.append(tensor_name)
        ir.tensors[tensor_name] = Tensor(tensor_name,np.zeros(tensor_shape,dtype=dtype),is_constant=False, shape=tensor_shape,\
                                         scale=scale, zero_point=zp)
        # Remember for the sequencer
        ir.tensors_to_mxp.add(tensor_name)

# Perform a DFS of the tflite graph from operator_idx
# to get all the ops in mxp_ops
def get_ops_in_sync_node(tflite_graph: dict, operator_idx: int, mxp_ops: list) -> list:
    operators = tflite_graph['subgraphs'][0]['operators']
    visited = set()
    sync_node_ops = []
    stack = [operator_idx]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        # Add to sync_node_ops if it's in mxp_ops, otherwise
        # stop searching in this direction
        if current in mxp_ops:
            sync_node_ops.append(current)
        else:
            continue

        # Get input and output tensors of the current operator
        inputs = operators[current].get('inputs', [])
        outputs = operators[current].get('outputs', [])

        # Traverse downwards: Find operators consuming these outputs
        for i, op in enumerate(operators):
            # Append any op which has an input in the current op's output list
            # (as long as it's not been visited already)
            if any(output in op.get('inputs', []) for output in outputs) and i not in visited:
                stack.append(i)

        # Traverse upwards: Find operators producing these inputs
        for i, op in enumerate(operators):
            # Append any op which has an output in the current op's input list
            # (as long as it's not been visited already)
            if any(input_ in op.get('outputs', []) for input_ in inputs) and i not in visited:
                stack.append(i)

    return sync_node_ops

# Get the input and output tensors of a Sync node which contains multiple ops
def get_sync_node_io(tflite_graph: dict, current_sync_ops: int) -> tuple:
    operators = tflite_graph['subgraphs'][0]['operators']
    external_inputs = set()
    external_outputs = set()

    # Collect all tensors used in the sync node
    all_input_tensors = set()
    all_output_tensors = set()
    for idx in current_sync_ops:
        all_input_tensors.update(operators[idx].get('inputs', []))
        all_output_tensors.update(operators[idx].get('outputs', []))

    # Identify external inputs (produced outside the sync node)
    for input_ in all_input_tensors:
        # Check if input_ is the output of any op in the node.
        # If not, it's the output of some external op.
        if not any(input_ in operators[idx].get('outputs', []) for idx in current_sync_ops):
            external_inputs.add(input_)

    # Identify external outputs (consumed outside the sync node)
    for output in all_output_tensors:
        # Check if output is the input of any op in the node.
        # If not, it's the input to some external op.
        if not any(output in operators[idx].get('inputs', []) for idx in current_sync_ops):
            external_outputs.add(output)

    return external_inputs, external_outputs

# Add TFLite ops to IR
def add_tflite_ops(ir, tflite_graph, mxp_ops):
    # Some ops are skipped, e.g., Pad is combined with the next Conv.
    # For some ops, e.g., Pad, an alternative is to modify the tflite graph in
    # advance to merge the nodes, but this might not work in all cases. For now,
    # the .tflite graph stays the same and the IR is reflected to skip the node.
    # Make a dict mapping from the output tensor of the skipped node to the
    # input tensor to replace it with.
    skip_inputs = {}
    skip_outputs = {}
    skipped_quantize = []
    skipped_quantize_inputs = []
    
    # Keep track of ops in the current Sync node
    current_sync_ops = []

    # Import ir graph
    # Iterate through each operator in the subgraph
    for operator_idx, operator in enumerate(tflite_graph['subgraphs'][0]['operators']):

        opcode_index = operator['opcode_index']
        opcode = tflite_graph["operator_codes"][opcode_index]['builtin_code']

        # Skip certain ops, such as PAD, which is merged with the next CONV_2D
        if skip_op(tflite_graph, operator_idx):
            skip_lut = False
            back_to_back_luts = False # Happens when SILU activation follows RE-QUANT operation
            # Keep track of the skipped tensor
            operators = tflite_graph['subgraphs'][0]['operators']
            skipped_input  = operators[operator_idx]['inputs'][0]
            assert len(operators[operator_idx]['outputs']) == 1
            skipped_output = operators[operator_idx]['outputs'][0]
            if (operator_idx > 2) and ((opcode == 'PAD') or (opcode == 'QUANTIZE')):
                producer_op = get_producer_op(tflite_graph, skipped_input)
                producer_type = get_op_type(tflite_graph, producer_op)
                if (producer_type == 'MUL'):
                    input = operators[producer_op]['inputs'][1]
                    prev_producer_op = get_producer_op(tflite_graph, input)
                    prev_producer_type = get_op_type(tflite_graph, prev_producer_op)
                    if (prev_producer_type == 'LOGISTIC'):
                        skipped_input = operators[prev_producer_op]['inputs'][0]
                elif (producer_type == 'GATHER'):
                    input_op = operators[producer_op]['inputs'][1]
                    producer_op = get_producer_op(tflite_graph, input_op)
                    producer_type = get_op_type(tflite_graph, producer_op)
                    if (producer_type == 'CAST'):
                        input = operators[producer_op]['inputs'][0]
                        prev_producer_op = get_producer_op(tflite_graph, input)
                        prev_producer_type = get_op_type(tflite_graph, prev_producer_op)
                        if (prev_producer_type == 'QUANTIZE'):
                            if (opcode == 'QUANTIZE'):
                                back_to_back_luts = True
                            if (prev_producer_op != 0):
                                skipped_input = operators[prev_producer_op]['inputs'][0]
                        else:
                            skipped_input = input
            elif (opcode == 'CAST'):
                producer_op = get_producer_op(tflite_graph, skipped_input)
                producer_type = get_op_type(tflite_graph, producer_op)
            elif (opcode == 'GATHER'):
                producer_op = get_producer_op(tflite_graph, operators[operator_idx]['inputs'][1])
                producer_type = get_op_type(tflite_graph, producer_op)
                if (producer_type == 'CAST'):
                    input = operators[producer_op]['inputs'][0]
                    prev_producer_op = get_producer_op(tflite_graph, input)
                    prev_producer_type = get_op_type(tflite_graph, prev_producer_op)
                    if (prev_producer_type == 'QUANTIZE'):
                        # If this LUT is for UINT8 to INT8 conversion
                        if (prev_producer_op != 0):
                            prev_input = operators[prev_producer_op]['inputs'][0]
                            prev_op = get_producer_op(tflite_graph, prev_input)
                            prev_op_type = get_op_type(tflite_graph, prev_op)
                            if (prev_op_type == "GATHER"):
                                skipped_input = skip_inputs[prev_input]
                            else:
                                skipped_input = operators[prev_producer_op]['inputs'][0]
                        else:
                            skip_lut = True
                            orig_input = 'T'+str(operators[prev_producer_op]['inputs'][0])
                            new_input = 'T'+str(skipped_output)
                            orig_in_idx = ir.inputs.index(orig_input)
                            ir.inputs[orig_in_idx] = new_input
                            orig_in_idx = ir.io_tensor_names.index(orig_input)
                            ir.io_tensor_names[orig_in_idx] = new_input
                            ir.tensors_from_mxp.discard(orig_input)
                            ir.tensors_from_mxp.add(new_input)
                    else:
                        skipped_input = input
            if not skip_lut:
                skip_inputs[skipped_output] = skipped_input

            # If Quantize is skipped, store it to update the qparams
            if opcode == 'QUANTIZE':
                pattern_quant_cast_gather = True
                quant_consumer_ops = get_consumer_ops(tflite_graph, skipped_output)
                quant_consumer_type = get_op_type(tflite_graph, quant_consumer_ops[0])
                skip_quant_op_idx = operator_idx
                if (quant_consumer_type == 'CAST'):
                    quant_out_idx = operators[quant_consumer_ops[0]]['outputs'][0]
                    cast_consumer_ops = get_consumer_ops(tflite_graph, quant_out_idx)
                    cast_consumer_type = get_op_type(tflite_graph, cast_consumer_ops[0])
                    if (cast_consumer_type == 'GATHER'):
                        gather_out_idx = operators[cast_consumer_ops[0]]['outputs'][0]
                        gather_consumer_ops = get_consumer_ops(tflite_graph, gather_out_idx)
                        if (len(gather_consumer_ops) > 0):
                            gather_consumer_type = get_op_type(tflite_graph, gather_consumer_ops[0])
                        else:
                            gather_consumer_type = ''
                        if (operator_idx == 0):
                            lut_info = operators[cast_consumer_ops[0]]['inputs'][0]
                            lut = get_tensor_data_from_tensor_index(tflite_graph, lut_info, np.int8)
                            ir.uint8_int8_lut = lut
                        else:
                            if back_to_back_luts or ((gather_consumer_type == 'RESIZE_NEAREST_NEIGHBOR') and (len(gather_consumer_ops) == 1)):
                                pattern_quant_cast_gather = False
                                skip_quant_op_idx = cast_consumer_ops[0]

                if not pattern_quant_cast_gather:
                    skipped_quantize.append(skip_quant_op_idx)
                    skipped_quantize_inputs.append(operators[operator_idx]['inputs'][0])

            if opcode == 'LOGISTIC':
                consumer_ops = get_consumer_ops(tflite_graph, skipped_output)
                out_idx = operators[consumer_ops[0]]['outputs'][0]
                if (f"T{out_idx}" in ir.outputs):
                    skip_outputs[out_idx] = skipped_input

            if opcode == 'GATHER':
                if (f"T{skipped_output}" in ir.outputs):
                    skip_outputs[skipped_output] = skipped_input
        
            continue

        # If this is an MXP op, use Sync for the op type
        if operator_idx in mxp_ops:
            # If this is already in the current_sync_ops, then skip it
            if operator_idx in current_sync_ops:
                continue

            # Otherwise, this is a new sync node
            current_sync_ops = get_ops_in_sync_node(tflite_graph, operator_idx, mxp_ops)
            # Get the tensors which are inputs/outputs to this sync node
            input_list, output_list = get_sync_node_io(tflite_graph, current_sync_ops)
            # Remember these for the sequencer
            ir.tensors_to_mxp.update([get_short_tensor_name(x) for x in input_list]) # sync inputs are sent to MXP
            ir.tensors_from_mxp.update([get_short_tensor_name(x) for x in output_list]) # sync outputs are read from MXP

            # Add node to IR
            attributes_dict = {}
            attributes_dict['op_type'] = 'Sync'
            node_name = "Sync_" + str(operator_idx)
            attributes_dict['name'] = node_name
            ir.graph.add_nodes_from([(node_name, attributes_dict)])

        # Otherwise process normally
        else:
            # Retrieve opcode and built-in options (attributes) for this node
            attributes_dict = {}
            builtin_options = operator.get("builtin_options", {})
            attributes_dict['attributes'] = tflite_attrs_to_dict(builtin_options, opcode, operator, tflite_graph)
            attributes_dict['op_type'] = tflite_to_onnx_op_type(opcode)

            # For Strided Slice, currently support the case of strides == 1
            if opcode == 'STRIDED_SLICE':
                begin_input_channel, end_input_channel = check_strided_slice_attributes(operator, tflite_graph)
                attributes_dict['attributes']['begin_input_channel'] = begin_input_channel
                attributes_dict['attributes']['end_input_channel'] = end_input_channel
            elif opcode == 'SPLIT':
                split_axis = get_tensor_data_from_tensor_index(tflite_graph, operator['inputs'][0], np.int32)[0]
                subgraph = tflite_graph['subgraphs'][0]
                tensor = subgraph['tensors'][operator['inputs'][1]]
                channels = tensor['shape'][split_axis]
                num_splits = attributes_dict['attributes']['num_splits']
                if (num_splits > 2):
                    raise ValueError('Currently we only support 2 splits ')
                attributes_dict['attributes']['begin_input_channel'] = channels // num_splits
                attributes_dict['attributes']['end_input_channel'] = channels

                attributes_dict0 = copy.deepcopy(attributes_dict)
                attributes_dict0['attributes']['begin_input_channel'] = 0
                attributes_dict0['attributes']['end_input_channel'] = channels // num_splits

            # Unlike ONNX, there is no name like Conv_1, Conv_2 for these, so name them.
            # Will get rid of _ from the opcode name because python simulator assumes very specific parsing of node names.
            if opcode == 'SPLIT':
                input_list = [operator['inputs'][1]]
                node_name = opcode.replace("_", "") + "_" + str(operator_idx) + "_1"
                attributes_dict['name'] = node_name
                ir.graph.add_nodes_from([(node_name, attributes_dict)])
                output_list = [operator['outputs'][1]]

                node_name0 = opcode.replace("_", "") + "_" + str(operator_idx) + "_0"
                attributes_dict0['name'] = node_name0
                ir.graph.add_nodes_from([(node_name0, attributes_dict0)])
                output_list0 = [operator['outputs'][0]]
            else:
                input_list = operator['inputs']
                node_name = opcode.replace("_", "") + "_" + str(operator_idx)
                attributes_dict['name'] = node_name
                ir.graph.add_nodes_from([(node_name, attributes_dict)])
                output_list = operator['outputs']

        # Update inputs
        inputs = []
        for idx, input_index in enumerate(input_list):
            # Skip if input index is -1 (null input)
            if input_index == -1:
                assert False    # TODO: Continue instead

            # If the input to this node is skipped, update the input tensor
            if input_index in skip_inputs:
                input_index = skip_inputs[input_index]

            #input_tensor_info = tflite_graph['subgraphs'][0]['tensors'][input_index]
            #input_name = input_tensor_info['name']
            input_name = get_short_tensor_name(input_index)
            if input_name not in ir.tensors:
                # In ONNX Resize op, the 2nd input is NULL.
                #if opcode == "RESIZE" and idx == 1:
                #    continue
                raise ValueError('Input tensor not found, this is a bug since it should have already been allocated')

            inputs.append(input_name)
            input_tensor = ir.tensors[input_name]
            input_tensor.consumers.append(node_name)
            # If the input comes from a node, add a graph edge (connection between 2 nodes)
            # Input which doesnt have a producer is an input tensor to the graph or constant
            if input_tensor.producer:
                ir.graph.add_edge(input_tensor.producer, node_name)
                if opcode == 'SPLIT':
                    ir.graph.add_edge(input_tensor.producer, node_name0)
        ir.graph.nodes[node_name]['inputs'] = inputs

        # Update outputs
        outputs = []
        for idx, output_index in enumerate(output_list):
            # Skip if output index is -1 (null output)
            if output_index == -1:
                assert False    # TODO: Continue instead
            
            #output_tensor_info = tflite_graph['subgraphs'][0]['tensors'][output_index]
            #output_name = output_tensor_info['name']
            output_name = get_short_tensor_name(output_index)
            if output_name not in ir.tensors:
                raise ValueError ('Found tensor without shape. Name: %s. Make sure the supplied graph includes shape inference.' % output_name)
            outputs.append(output_name)
            output_tensor = ir.tensors[output_name]
            # Update the tensors dict. the output tensor to the node should have it as consumer
            output_tensor.producer = node_name
        ir.graph.nodes[node_name]['outputs'] = outputs
            
        if opcode == 'SPLIT':
            # Update outputs for the first output of split
            outputs = []
            for idx, output_index in enumerate(output_list0):
                # Skip if output index is -1 (null output)
                if output_index == -1:
                    assert False    # TODO: Continue instead
                
                #output_tensor_info = tflite_graph['subgraphs'][0]['tensors'][output_index]
                #output_name = output_tensor_info['name']
                output_name = get_short_tensor_name(output_index)
                if output_name not in ir.tensors:
                    raise ValueError ('Found tensor without shape. Name: %s. Make sure the supplied graph includes shape inference.' % output_name)
                outputs.append(output_name)
                output_tensor = ir.tensors[output_name]
                # Update the tensors dict. the output tensor to the node should have it as consumer
                output_tensor.producer = node_name0
            ir.graph.nodes[node_name0]['outputs'] = outputs
            ir.graph.nodes[node_name0]['inputs'] = inputs

    # Update qparams if any quantize ops were skipped
    for idx in range(len(skipped_quantize)):
        ir = update_qparams_for_skipped_quantize(ir, tflite_graph, skipped_quantize[idx], skipped_quantize_inputs[idx])

    for out_idx in skip_outputs:
        old_output = f"T{out_idx}"
        new_output = f"T{skip_outputs[out_idx]}"
        if (old_output in ir.outputs):
            idx = ir.outputs.index(old_output)
            ir.outputs[idx] = new_output
            ir.tensors_to_mxp.remove(old_output)
            ir.tensors_to_mxp.add(new_output)
            idx = ir.io_tensor_names.index(old_output)
            ir.io_tensor_names[idx] = new_output
    
# Generate IR from TFLite graph
def tflite_to_ir(ir,tflite_graph,mxp_ops,mxp_tensor_to_offset):
    # Process JSON for entire graph, including MXP subgraphs.
    # MXP subgraphs are replaced with a Sync node.
    # Can also avoid adding tensors if every input and output is an MXP node.
    add_tflite_tensors(ir, tflite_graph)
    add_tflite_inputs(ir, tflite_graph)
    add_tflite_outputs(ir, tflite_graph)
    add_tflite_ops(ir, tflite_graph, mxp_ops)
    
    if ir.sync_with_MXP:
        # TODO: Each tensor to/from MXP is stored below, and F4/F5 flags will
        # set/wait before these are written/read. However, if a tensor needs to be
        # read multiple times, e.g., by 2 separate blobs, the current implementation
        # will wait both times. This can cause a deadlock if the MXP only sets the
        # flag once.
        print("From MXP: " + str(ir.tensors_from_mxp))
        print("To MXP:   " + str(ir.tensors_to_mxp))

        # Also add an offset from each tensor to its location in DDR
        for tensor_id in mxp_tensor_to_offset:
            tensor_name = f"T{tensor_id}"
            ir.mxp_tensor_to_offset[tensor_name] = mxp_tensor_to_offset[tensor_id]

    return ir

def main():
    pass

if __name__ == "__main__":
    main()
