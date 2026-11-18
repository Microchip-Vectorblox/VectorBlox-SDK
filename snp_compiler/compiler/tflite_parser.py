import sys
sys.path.append('.')
import math
import numpy as np
from common.tensor_ir import Tensor
from common.hw_config import NON_REQUANTING_OPS, MULTIPLE_INPUT_NON_REQUANTING_OPS, OpType
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

def _mul_has_two_dynamic_inputs(tflite_graph, operator_idx) -> bool:
    operators = tflite_graph['subgraphs'][0]['operators']
    tensors = tflite_graph['subgraphs'][0]['tensors']
    buffers = tflite_graph['buffers']
    inputs = [i for i in operators[operator_idx]['inputs'] if i != -1]
    dynamic_inputs = [i for i in inputs if 'data' not in buffers[tensors[i]['buffer']]]
    return len(dynamic_inputs) > 1

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
        # A MUL with a single dynamic input (the other operand a constant, or a
        # self-gate like x*sigmoid(x) fused via the preceding LOGISTIC/requant
        # handling) is absorbed elsewhere and should keep being skipped. A MUL
        # with two genuinely dynamic inputs (e.g. an SE-block excite: full-res
        # feature map * a per-channel gate) is a real elementwise op that
        # cannot be fused that way - it must be emitted as its own node (a
        # Sync node, if it's MXP-routed via mxp_op_idx.txt).
        return not _mul_has_two_dynamic_inputs(tflite_graph, operator_idx)

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
def check_strided_slice_attributes(operator, tflite_graph, node_name=None):
    # Inputs are input data, begin, end, strides
    begin_tensor_data   = get_tensor_data_from_tensor_index(tflite_graph, operator['inputs'][1], np.int32)
    end_tensor_data     = get_tensor_data_from_tensor_index(tflite_graph, operator['inputs'][2], np.int32)
    strides_tensor_data = get_tensor_data_from_tensor_index(tflite_graph, operator['inputs'][3], np.int32)

    if len(begin_tensor_data) != 4:
        raise ValueError(f"Strided_Slice expects a 4-element begin tensor. This model has a node {node_name} with begin {tuple(begin_tensor_data)}")
    if len(end_tensor_data) != 4:
        raise ValueError(f"Strided_Slice expects a 4-element end tensor. This model has a node {node_name} with end {tuple(end_tensor_data)}")
    if len(strides_tensor_data) != 4:
        raise ValueError(f"Strided_Slice expects a 4-element strides tensor. This model has a node {node_name} with strides {tuple(strides_tensor_data)}")

    # For now, handle strides of 1
    if (strides_tensor_data[0], strides_tensor_data[1], strides_tensor_data[2], strides_tensor_data[3]) != (1, 1, 1, 1):
        raise ValueError(
            f"Strided_Slice only supports strides of 1 on all dimensions. This model has a node {node_name} with strides {tuple(strides_tensor_data)}")

    # Strided slice must be channels only
    if (begin_tensor_data[0], begin_tensor_data[1], begin_tensor_data[2]) != (0, 0, 0) or \
       (end_tensor_data[0], end_tensor_data[1], end_tensor_data[2]) != (0, 0, 0):
        raise ValueError(
            f"Unstructured sparsity only supports Strided_Slice on the channels dimension. This model has a node {node_name} that needs un-supported Strided_Slice op")

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

def get_resize_step_sizes(v_in, v_out):
    """Break a resize from v_in to v_out into a chain of near-2x steps: [v_in, ..., v_out].

    Builds the chain backwards from v_out by repeated ceil-halving until reaching (or dropping
    below) v_in, so every step scales by at most ~2x - e.g. (5, 36) -> [5, 9, 18, 36]. If v_out
    doesn't shrink below v_in (no upscale), returns the trivial 1-step chain [v_in, v_out].
    """
    if v_out <= v_in:
        return [v_in, v_out]
    sizes = [v_out]
    while sizes[0] > v_in:
        nxt = (sizes[0] + 1) // 2
        if nxt <= v_in:
            break
        sizes.insert(0, nxt)
    if sizes[0] != v_in:
        sizes.insert(0, v_in)
    return sizes

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

def get_quant_cast_gather_params(operator, tflite_graph) -> list:
    output_tensor_index = operator['outputs'][0]
    silu_input_tensor = tflite_graph['subgraphs'][0]['tensors'][output_tensor_index]

    consumer_ops = get_consumer_ops(tflite_graph, output_tensor_index)
    if (len(consumer_ops) != 0):
        consumer_type = get_op_type(tflite_graph, consumer_ops[0])
    else:
        return None, None, None, None

    if (consumer_type != "QUANTIZE"):
        return None, None, None, None
    
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
                lut_info = operators[cast_consumer_ops[0]]['inputs'][0]
                lut = get_tensor_data_from_tensor_index(tflite_graph, lut_info, np.int8)
                lut_int8 = np.zeros((256,), dtype=np.int8)
                for idx in range(256):
                    if (idx < 128):
                        lut_int8[idx] = lut[128+idx]
                    else:
                        lut_int8[idx] = lut[idx-128]
                silu_output_tensor = tflite_graph['subgraphs'][0]['tensors'][gather_out_idx] 
                return 'Requant', silu_input_tensor, silu_output_tensor, lut_int8
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
                return 'Silu', silu_input_tensor, silu_output_tensor, lut_int8

    return False, None, None, None

def get_lut_info(operator, tflite_graph) -> list:
    
    is_valid_pattern, input_tensor, output_tensor, lut = get_quant_cast_gather_params(operator, tflite_graph)
    if is_valid_pattern == 'Silu':
        silu_attributes = {}
        silu_attributes['input_scale'] = input_tensor['quantization']['scale']
        silu_attributes['input_zp'] = input_tensor['quantization']['zero_point']
        silu_attributes['output_scale'] = output_tensor['quantization']['scale']
        silu_attributes['output_zp'] = output_tensor['quantization']['zero_point']
        silu_attributes['lut'] = lut
        return silu_attributes
    elif is_valid_pattern == 'Requant':
        requant_attributes = {}
        requant_attributes['input_scale'] = input_tensor['quantization']['scale']
        requant_attributes['input_zp'] = input_tensor['quantization']['zero_point']
        requant_attributes['output_scale'] = output_tensor['quantization']['scale']
        requant_attributes['output_zp'] = output_tensor['quantization']['zero_point']
        requant_attributes['lut'] = lut
        return requant_attributes
    else:
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

def check_small_input_kernel(operator, tflite_graph, kernel_shape, stride_h, stride_w, node_name=None):
    """Reject conv/pool configurations whose input is too small for the kernel and stride.

    - kernel 3 with stride 2, or kernel in (3, 5]: input height and width must be >= 8.
    - kernel > 5: input height and width must be >= 16.
    """
    input_tensor = tflite_graph['subgraphs'][0]['tensors'][operator['inputs'][0]]
    input_shape = input_tensor.get('shape', [])
    if len(input_shape) != 4:
        # Only an NHWC activation has a spatial size to check; anything else is left to the other checks.
        return
    input_height, input_width = input_shape[1], input_shape[2]
    kernel_max = max(kernel_shape)
    if kernel_max > 5:
        min_size = 16
    elif (kernel_max == 3 and stride_h == 2 and stride_w == 2) or (3 < kernel_max <= 5):
        min_size = 8
    else:
        return
    if input_height < min_size or input_width < min_size:
        raise ValueError(
            f"Unstructured sparsity does not support kernel {kernel_shape} with stride ({stride_h},{stride_w}) on an input "
            f"smaller than {min_size}x{min_size}; this model needs input height and width >= {min_size}, "
            f"but this node has input {input_height}x{input_width} at node: {node_name}")


def tflite_attrs_to_dict(builtin_options, opcode, operator, tflite_graph, node_name=None):

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
        if attr_dict['kernel_shape'][0] > 9 or attr_dict['kernel_shape'][1] > 9:
            raise ValueError(
                f"Unstructured sparsity only supports symmetric kernels upto 9x9. This model needs a Kernel size of {attr_dict['kernel_shape']} at node: {node_name}")

        # 'pads' attribute -- not in TFLite because it has 'padding' which is
        # 'SAME' or 'VALID', but ONNX has the actual pads
        attr_dict['pads'] = get_conv_padding_attribute(operator, tflite_graph, attr_dict)

        # 'strides' attribute
        attr_dict['strides'] = [attr_dict['stride_h'], attr_dict['stride_w']]
        check_small_input_kernel(operator, tflite_graph, attr_dict['kernel_shape'], attr_dict['stride_h'], attr_dict['stride_w'], node_name)

        # other (unused)
        #attr_dict['dilations'] = [attr_dict['dilation_h_factor'], attr_dict['dilation_w_factor']]
        #attr_dict['group'] = 1

        # Convert 2x2 Conv to 3x3 with padding
        # Original padding was for 2x2 kernel, now need to adjust for 3x3
        # To keep same output size: pad_total = (kernel_new - kernel_old) = (3 - 2) = 1
        # Distribute: pad bottom=1, right=1 (keep original weights in top-left of 3x3)
        if attr_dict['kernel_shape'] == [2, 2]:
            original_pads = attr_dict['pads']
            attr_dict['kernel_shape'] = [3, 3]
            # Add 1 pixel padding on bottom and right to compensate for larger kernel
            # pads = [top, bottom, left, right]
            attr_dict['pads'] = [original_pads[0], original_pads[1] + 1, original_pads[2], original_pads[3] + 1]

    # TRANSPOSE_CONV: its own op type (OpType.TRANSPOSE_CONV). The is_transpose_conv flag is kept
    # as a harmless marker for the deferred hardware math. Weights (inputs[1]) share the CONV_2D
    # [out_ch, kh, kw, in_ch] layout, so kernel_shape reuses get_conv_kernel_shape_attribute.
    # NOTE: transpose-conv pad semantics differ from CONV_2D (the effective conv is the "full"
    # convolution over the dilated input), and stride>1 requires input dilation. Those are NOT
    # computed here - pads are a placeholder until the transpose-specific math is added.
    # TODO(user): configure real pad/dilation values here when implementing the NX transpose conv.
    elif opcode == 'TRANSPOSE_CONV':
        attr_dict['is_transpose_conv'] = True
        attr_dict['kernel_shape'] = get_conv_kernel_shape_attribute(operator, tflite_graph)
        if attr_dict['kernel_shape'][0] > 9 or attr_dict['kernel_shape'][1] > 9:
            raise ValueError(
                f"Unstructured sparsity only supports symmetric kernels upto 9x9. This model needs a Kernel size of {attr_dict['kernel_shape']} at node: {node_name}")

        attr_dict['strides'] = [attr_dict['stride_h'], attr_dict['stride_w']]
        check_small_input_kernel(operator, tflite_graph, attr_dict['kernel_shape'], attr_dict['stride_h'], attr_dict['stride_w'], node_name)
        # Placeholder pads; real transpose-conv padding to be configured later.
        attr_dict['pads'] = [0, 0, 0, 0]

    # Temporary: Also add Max Pool attributes
    # (for debugging only, will be on MXP later)
    elif opcode == 'MAX_POOL_2D':
        attr_dict['kernel_shape'] = [attr_dict['filter_height'], attr_dict['filter_width']]
        if attr_dict['kernel_shape'][0] > 9 or attr_dict['kernel_shape'][1] > 9:
            raise ValueError(
                f"Unstructured sparsity only supports symmetric kernels upto 9x9. This model needs a Kernel size of {attr_dict['kernel_shape']} at node: {node_name}")

        attr_dict['strides'] = [attr_dict['stride_h'], attr_dict['stride_w']]
        check_small_input_kernel(operator, tflite_graph, attr_dict['kernel_shape'], attr_dict['stride_h'], attr_dict['stride_w'], node_name)
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

    # 'activation_silu' attribute: SiLU (Logistic followed by Mul) or a Requant pattern following
    # this op. Computed for every op whose type is represented in OpType (previously only CONV_2D),
    # except Concat and Sync which never carry a fused non-linearity. Skipped when the op carries a
    # fused activation function other than NONE, since SiLU cannot be fused on top of an existing one.
    if TFLITE_TO_ONNX_OP_MAP.get(opcode) in (OpType.values() - {OpType.CONCAT, OpType.SYNC}):
        if attr_dict.get('fused_activation_function', 'NONE') == 'NONE':
            attr_dict['activation_silu'] = get_lut_info(operator, tflite_graph)

    return attr_dict

TFLITE_TO_ONNX_OP_MAP = {
    'ADD': 'Add',
    'SUB': 'Sub',
    'MUL': 'Mul',
    'DIV': 'Div',
    'MATRIX_MULTIPLICATION': 'MatMul',
    'CONV_2D': 'Conv',
    # TRANSPOSE_CONV is its own first-class op type (OpType.TRANSPOSE_CONV). It gets a dedicated
    # frontend param function (frontend_hardware_agnostic.get_transpose_conv_params) that wires
    # inputs/outputs/weights/bias, stays distinct through folding + qparams, then is flipped to
    # OpType.CONV at the end of the qparams stage (AveragePool pattern) so it reuses the conv
    # backend. Its inputs are reordered into conv order at node creation (see below). The
    # transpose-specific hardware math (180-degree kernel flip, stride>1 dilation) is not yet
    # configured and is deferred to a later change.
    'TRANSPOSE_CONV': OpType.TRANSPOSE_CONV,
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

def tflite_to_onnx_op_type(tflite_op_type: str) -> str:
    if tflite_op_type in TFLITE_TO_ONNX_OP_MAP:
        return TFLITE_TO_ONNX_OP_MAP[tflite_op_type]
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
            # if consumer_op_type not in ["RESIZE_NEAREST_NEIGHBOR", "STRIDED_SLICE", "SPLIT", "GATHER", "RESHAPE"]:
            #     raise ValueError ("Found new op without quantization: " + consumer_op_type)

            ir.tensors[tensor_name] = Tensor(tensor_name, data=None, is_constant=True, shape=list(tensor_shape))

        elif buffer_data is None:
            # Only add shape for activations
            # TODO: Check this for a graph with intermediate activations
            # (currently this adds inputs/outputs but those are added later as well)
            assert len(scale_list) == 1 and len(zp_list) == 1
            # Note: Activation tensors also have min/max (in addition to scale, zp)
            scale = np.array(scale_list[0])
            zp = np.array(zp_list[0])
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
        zp = np.array(zp_list[0])

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
        zp = np.array(zp_list[0])

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
    tensors = tflite_graph['subgraphs'][0]['tensors']
    buffers = tflite_graph['buffers']
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
        if input_ == -1:
            continue
        # Constant tensors (weights, biases, param vectors baked into the tflite buffer) have no
        # producer op, so the "not produced inside this sync node" check below would otherwise
        # misclassify them as an external NX->MXP sync input. They're already resident on the MXP
        # side, not something NX needs to send - exclude them here.
        if 'data' in buffers[tensors[input_]['buffer']]:
            continue
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

    # external_inputs/outputs are sets (unordered), so inputs[0] of the resulting Sync node was
    # arbitrary. MXP ops can list a small param tensor (e.g. a reshape shape vector) alongside the
    # real activation; downstream conv-engine passes assume inputs[0] is the 4-D activation. Order
    # inputs by tensor rank (activation first), with tensor index as a deterministic tie-break.
    tensors = tflite_graph['subgraphs'][0]['tensors']
    def _rank(t_idx):
        return len(tensors[t_idx].get('shape', [])) if t_idx < len(tensors) else 0
    ordered_inputs = sorted(external_inputs, key=lambda t: (-_rank(t), t))
    # Outputs only need a deterministic order (was an unordered set); keep index order to avoid
    # disturbing output->consumer wiring / MXP offset mapping.
    ordered_outputs = sorted(external_outputs)
    return ordered_inputs, ordered_outputs

def get_single_output(operator):
    outputs = operator.get("outputs", [])
    assert len(outputs) == 1
    return outputs[0]


def get_single_input(operator):
    inputs = operator.get("inputs", [])
    assert len(inputs) >= 1
    return inputs[0]


def get_only_consumer_op(tflite_graph, tensor_idx):
    consumers = get_consumer_ops(tflite_graph, tensor_idx)
    if len(consumers) != 1:
        return None
    return consumers[0]


def match_quantize_cast_gather_pattern(tflite_graph, quantize_op_idx):
    """
    Match the pattern:

        QUANTIZE -> CAST -> GATHER

    Returns a dictionary with all relevant op and tensor indices if matched.
    Returns None otherwise.
    """

    operators = tflite_graph["subgraphs"][0]["operators"]

    if get_op_type(tflite_graph, quantize_op_idx) != "QUANTIZE":
        return None

    quantize_op = operators[quantize_op_idx]
    quantize_input = get_single_input(quantize_op)
    quantize_output = get_single_output(quantize_op)

    cast_op_idx = get_only_consumer_op(tflite_graph, quantize_output)
    if cast_op_idx is None:
        return None

    if get_op_type(tflite_graph, cast_op_idx) != "CAST":
        return None

    cast_op = operators[cast_op_idx]
    cast_output = get_single_output(cast_op)

    gather_op_idx = get_only_consumer_op(tflite_graph, cast_output)
    if gather_op_idx is None:
        return None

    if get_op_type(tflite_graph, gather_op_idx) != "GATHER":
        return None

    gather_op = operators[gather_op_idx]
    gather_output = get_single_output(gather_op)

    return {
        "quantize_op_idx": quantize_op_idx,
        "cast_op_idx": cast_op_idx,
        "gather_op_idx": gather_op_idx,

        "quantize_input": quantize_input,
        "quantize_output": quantize_output,

        "cast_output": cast_output,
        "gather_output": gather_output,

        "gather_lut_input": gather_op["inputs"][0],
        "gather_indices_input": gather_op["inputs"][1],
    }


def replace_ir_input_tensor(ir, old_tensor_name, new_tensor_name):
    """
    Replace an external IR input tensor name safely.
    """

    if old_tensor_name in ir.inputs:
        idx = ir.inputs.index(old_tensor_name)
        ir.inputs[idx] = new_tensor_name

    if hasattr(ir, "io_tensor_names") and old_tensor_name in ir.io_tensor_names:
        idx = ir.io_tensor_names.index(old_tensor_name)
        ir.io_tensor_names[idx] = new_tensor_name

    # Only transfer MXP-set membership if the old tensor was actually a member;
    # a plain graph input must not be spuriously marked as MXP-sourced.
    if old_tensor_name in ir.tensors_from_mxp:
        ir.tensors_from_mxp.discard(old_tensor_name)
        ir.tensors_from_mxp.add(new_tensor_name)


def replace_ir_output_tensor(ir, old_tensor_name, new_tensor_name):
    """
    Replace an external IR output tensor name safely.
    """

    if old_tensor_name in ir.outputs:
        idx = ir.outputs.index(old_tensor_name)
        ir.outputs[idx] = new_tensor_name
        # mxp_sync_order (built from the original tflite graph_outputs list) still expects
        # old_tensor_name's tflite id; remember the alias so the numeric simulator can match it.
        if hasattr(ir, "output_tensor_aliases"):
            ir.output_tensor_aliases[new_tensor_name] = old_tensor_name

    # Only transfer MXP-set membership if the old tensor was actually a member,
    # so a non-MXP output is not spuriously marked as sent to MXP.
    if old_tensor_name in ir.tensors_to_mxp:
        ir.tensors_to_mxp.remove(old_tensor_name)
        ir.tensors_to_mxp.add(new_tensor_name)

    if hasattr(ir, "io_tensor_names") and old_tensor_name in ir.io_tensor_names:
        idx = ir.io_tensor_names.index(old_tensor_name)
        ir.io_tensor_names[idx] = new_tensor_name

    # If old tensor has MXP offset, copy it to the new tensor
    if hasattr(ir, "mxp_tensor_to_offset") and old_tensor_name in ir.mxp_tensor_to_offset:
        ir.mxp_tensor_to_offset[new_tensor_name] = ir.mxp_tensor_to_offset[old_tensor_name]


def build_quantize_cast_gather_skip_plan(ir, tflite_graph, skip_inputs, skip_outputs):
    """
    Build a skip plan for all QUANTIZE -> CAST -> GATHER patterns.

    For regular internal patterns, the full chain is bypassed:

        gather_output -> quantize_input

    For graph-input conversion patterns, usually op index 0, the LUT is saved
    and the graph input is replaced by the GATHER output tensor.
    """

    skipped_ops = set()
    operators = tflite_graph["subgraphs"][0]["operators"]

    for operator_idx, _ in enumerate(operators):
        pattern = match_quantize_cast_gather_pattern(tflite_graph, operator_idx)
        if pattern is None:
            continue

        quantize_op_idx = pattern["quantize_op_idx"]
        cast_op_idx = pattern["cast_op_idx"]
        gather_op_idx = pattern["gather_op_idx"]

        quantize_input = pattern["quantize_input"]
        quantize_output = pattern["quantize_output"]
        cast_output = pattern["cast_output"]
        gather_output = pattern["gather_output"]

        skipped_ops.update([
            quantize_op_idx,
            cast_op_idx,
            gather_op_idx,
        ])

        # Special case: UINT8 -> INT8 conversion at graph input.
        # In this case we do not map gather_output back to quantize_input.
        # Instead, we make gather_output the new external graph input.
        if quantize_op_idx == 0:
            old_input_name = get_short_tensor_name(quantize_input)
            new_input_name = get_short_tensor_name(gather_output)
            replace_ir_input_tensor(ir, old_input_name, new_input_name)

            # Internal tensors of the skipped chain should resolve to the new input.
            skip_inputs[quantize_output] = gather_output
            skip_inputs[cast_output] = gather_output

            continue

        # Regular case: bypass the whole LUT chain.
        skip_inputs[quantize_output] = quantize_input
        skip_inputs[cast_output] = quantize_input
        skip_inputs[gather_output] = quantize_input

        old_output_name = get_short_tensor_name(gather_output)
        if old_output_name in ir.outputs:
            skip_outputs[gather_output] = quantize_input

    return skipped_ops



def _scalarize(v):
    # scale/zp may be a python scalar, numpy scalar/0-d/array, or length-1 list/tuple; or None/empty.
    if v is None:
        return None
    if hasattr(v, 'ndim'):  # numpy array / scalar
        if v.ndim == 0:
            return float(v)
        if v.size == 0:
            return None
        v = v.reshape(-1)[0]
    elif isinstance(v, (list, tuple)):
        if len(v) == 0:
            return None
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return v

def _qparams_equal(a, b):
    sa, sb = _scalarize(a), _scalarize(b)
    # If either side is unquantized (no scale), treat as "equal" so we don't try to requant it.
    if sa is None or sb is None:
        return True
    return sa == sb


def insert_concat_input_requant_nodes(ir, bypassed_quantize_tensor_names):
    """Insert an Identity (requant) conv before any Concat input whose quantization differs from
    the Concat's output quantization.

    A QUANTIZE that requantizes a producer (e.g. LeakyRelu/Resize/another LUT) to the Concat's scale
    is skipped by skip_op() and bypassed, so the Concat input keeps the pre-requant scale and
    calc_concat_qparams later raises ("input scale is different from output scale"). Re-materialize
    that requant as an Identity conv between the producer and the Concat: an Identity conv with
    input scale A and output scale B already requantizes A->B through its standard rq block
    (calc_identity_qparams), so the Concat now sees a matching scale/zp. This replaces the bypassed
    second LUT feeding the Concat (the upstream/first LUT is left untouched).

    bypassed_quantize_tensor_names is the exact set of (short) tensor names that a skipped/bypassed
    QUANTIZE op's INPUT tensor resolved to (see skipped_quantize_inputs in add_tflite_ops) - i.e. the
    real "second LUT feeding a Concat" signature: producer LUT -> [bypassed 2nd-LUT QUANTIZE] -> Concat.
    Gating on this exact set (rather than a heuristic like "producer has any LUT activation") avoids
    firing on every ordinary LeakyRelu-fed Concat input that never had a bypassed quantize at all."""
    graph = ir.graph
    for concat_name in list(graph.nodes):
        node = graph.nodes[concat_name]
        if node.get('op_type') != OpType.CONCAT:
            continue
        out_tensor = ir.tensors[node['outputs'][0]]
        out_scale, out_zp = out_tensor.scale, out_tensor.zero_point
        new_inputs = list(node['inputs'])
        changed = False
        for idx, in_name in enumerate(node['inputs']):
            in_tensor = ir.tensors[in_name]
            if getattr(in_tensor, 'is_constant', False):
                continue
            if _qparams_equal(in_tensor.scale, out_scale) and _qparams_equal(in_tensor.zero_point, out_zp):
                continue

            # Only the back-to-back-LUT case (a 2nd LUT feeding this Concat was bypassed by
            # skip_op, leaving this scale mismatch) needs a requant identity here. A plain scale
            # mismatch with no bypassed-quantize chain is a different situation and should not be
            # silently patched over.
            if in_name not in bypassed_quantize_tensor_names:
                continue

            # Resize already has its own dedicated qparams/folding handling downstream; instead of
            # inserting a separate requant Identity conv after it (which is not valid there), fuse
            # the requant directly onto the Resize node's own output via requant_info - the same
            # attribute/mechanism already used for the "LUT feeding a Resize" case (calc_resize_qparams).
            producer_name = in_tensor.producer
            if producer_name is None or not graph.has_node(producer_name):
                continue
            producer_node = graph.nodes[producer_name]
            if producer_node.get('op_type') == OpType.RESIZE:
                producer_node.setdefault('attributes', {})['requant_info'] = {
                    'input_scale': [in_tensor.scale], 'input_zp': [in_tensor.zero_point],
                    'output_scale': [out_scale], 'output_zp': [out_zp],
                }
                continue

            rq_node_name = '%s_in%d_requant' % (concat_name, idx)
            rq_tensor_name = '%s_rq_%s' % (in_name, concat_name)
            if rq_tensor_name in ir.tensors:  # already inserted (idempotent)
                continue

            # New tensor carries the Concat's qparams; same shape as the bypassed input.
            rq_tensor = Tensor(rq_tensor_name, None, is_constant=False, shape=list(in_tensor.shape),
                               scale=out_scale, zero_point=out_zp)
            rq_tensor.producer = rq_node_name
            rq_tensor.consumers = [concat_name]
            ir.tensors[rq_tensor_name] = rq_tensor

            attrs = {'op_type': OpType.IDENTITY, 'name': rq_node_name, 'attributes': {'concat_requant': True}}
            graph.add_nodes_from([(rq_node_name, attrs)])
            graph.nodes[rq_node_name]['inputs'] = [in_name]
            graph.nodes[rq_node_name]['outputs'] = [rq_tensor_name]

            # Rewire: producer -> requant -> concat ; concat input idx now reads the requant output.
            producer = in_tensor.producer
            if producer is not None and graph.has_node(producer):
                graph.add_edge(producer, rq_node_name)
            graph.add_edge(rq_node_name, concat_name)
            if concat_name in in_tensor.consumers:
                in_tensor.consumers = [c for c in in_tensor.consumers if c != concat_name]
            in_tensor.consumers.append(rq_node_name)
            new_inputs[idx] = rq_tensor_name
            changed = True

        if changed:
            node['inputs'] = new_inputs
            # Drop now-stale direct producer->concat edges (those producers feed the concat only
            # through the new requant nodes); keep edges for inputs still consumed directly.
            direct_producers = {ir.tensors[t].producer for t in new_inputs
                                if t in ir.tensors and ir.tensors[t].producer is not None}
            for pred in list(graph.predecessors(concat_name)):
                if pred not in direct_producers and pred != concat_name:
                    graph.remove_edge(pred, concat_name)


IDENTITY_WEIGHT_VALUE = 8

# Replace a QUANTIZE op whose input is UINT8 (e.g. the raw-image dtype cast
# ahead of a conv_uint8 identity conv) with a real 1x1 identity CONV_2D node in
# the IR, instead of skipping it. Its weight/bias tensors don't exist in the
# .tflite file (QUANTIZE has no weight input), so they're synthesized here the
# same way program_compiler.py's insert_identity_conv_before_node does for its
# own injected identity convs. Returns (node_name, attributes_dict, input_list,
# output_list) in the same shape the main add_tflite_ops loop uses for every
# other op, so the shared input/output-wiring code (edges, tensor.producer/
# consumers) still runs uniformly for this node.
def build_identity_conv_for_quantize(ir, tflite_graph, operator_idx):
    operators = tflite_graph['subgraphs'][0]['operators']
    operator = operators[operator_idx]
    input_idx = operator['inputs'][0]
    output_idx = operator['outputs'][0]
    input_name = get_short_tensor_name(input_idx)

    input_tensor = ir.tensors[input_name]
    c = input_tensor.get_original_shape()[1]

    # Give the synthesized weight/bias tensors fake negative tensor indices (no
    # real .tflite tensor uses these) so they flow through the same
    # get_short_tensor_name()-keyed input-wiring loop as every other op's inputs.
    weight_idx = -(1_000_000 + 2 * operator_idx)
    bias_idx = -(1_000_000 + 2 * operator_idx + 1)
    weight_name = get_short_tensor_name(weight_idx)
    bias_name = get_short_tensor_name(bias_idx)

    weight_data = np.zeros((c, c, 1, 1), dtype=np.int8)
    for ch in range(c):
        weight_data[ch, ch, 0, 0] = IDENTITY_WEIGHT_VALUE
    weight_scale = np.full((c,), 1.0 / IDENTITY_WEIGHT_VALUE, dtype=np.float64)
    weight_zp = np.zeros((c,), dtype=np.int64)
    ir.tensors[weight_name] = Tensor(weight_name, weight_data, is_constant=True, shape=list(weight_data.shape),
                                      scale=weight_scale, zero_point=weight_zp)

    # Bake the UINT8->INT8 byte-reinterpretation shift directly into the bias,
    # instead of relying on a separate NLF/LUT stage. The numeric simulator (and
    # hardware) always reads activation data as INT8 bytes regardless of a
    # tensor's declared dtype, so a UINT8 input's raw byte pattern is
    # (declared_value - 128) in the domain the MAC actually operates in. The
    # per-channel requant math computes `bias - w_sum*input_zero_point` using
    # the tensor's *declared* zero_point; adding w_sum*128 to the raw bias here
    # is algebraically equivalent to using the effective zero_point
    # (declared_zero_point - 128) that actually matches the raw byte domain:
    #   (bias + w_sum*128) - w_sum*declared_zp == bias - w_sum*(declared_zp-128)
    # w_sum per channel is just the diagonal weight (IDENTITY_WEIGHT_VALUE),
    # since every other weight in the row is 0.
    bias_data = np.full((c,), IDENTITY_WEIGHT_VALUE * 128, dtype=np.int32)
    bias_scale = np.ones((c,), dtype=np.float64)
    bias_zp = np.zeros((c,), dtype=np.int64)
    ir.tensors[bias_name] = Tensor(bias_name, bias_data, is_constant=True, shape=list(bias_data.shape),
                                    scale=bias_scale, zero_point=bias_zp)

    node_name = "QUANTIZECONV_" + str(operator_idx)
    attributes_dict = {
        'op_type': OpType.CONV,
        'attributes': {
            'kernel_shape': [1, 1],
            'pads': [0, 0, 0, 0],
            'strides': [1, 1],
            'stride_h': 1,
            'stride_w': 1,
            'fused_activation_function': 'NONE',
            'activation_silu': None,
        },
        'name': node_name,
    }
    input_list = [input_idx, weight_idx, bias_idx]
    output_list = operator['outputs']
    return node_name, attributes_dict, input_list, output_list

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
    # Tensor indices of the ultimate (fully-resolved) root feeding a genuine back-to-back-LUT chain:
    # a full QUANTIZE->CAST->GATHER pattern whose own input is itself the output of another such
    # pattern (detected below via prev_op_type == 'GATHER'). Both patterns are proper LUTs (not a
    # plain bypassed QUANTIZE), so neither is captured by skipped_quantize_inputs - this is the exact
    # "2 LUT patterns together" signature insert_concat_input_requant_nodes needs to re-materialize
    # the dropped 2nd LUT as a requant Identity conv.
    back_to_back_lut_inputs = []

    # Precompute clean skip plan for QUANTIZE -> CAST -> GATHER.
    skipped_ops = build_quantize_cast_gather_skip_plan(
        ir,
        tflite_graph,
        skip_inputs,
        skip_outputs,
    )
    
    # Keep track of ops in the current Sync node
    current_sync_ops = []

    # Import ir graph
    # Iterate through each operator in the subgraph
    for operator_idx, operator in enumerate(tflite_graph["subgraphs"][0]["operators"]):

        opcode_index = operator["opcode_index"]
        opcode = tflite_graph["operator_codes"][opcode_index]["builtin_code"]

        # A QUANTIZE whose input is UINT8 (e.g. the dtype cast ahead of a
        # conv_uint8 identity conv) is replaced with a real identity CONV_2D
        # node instead of being skipped, unless it's genuinely part of the
        # QUANTIZE->CAST->GATHER LUT pattern (already handled by skipped_ops).
        is_uint8_quantize_conv = False
        if opcode == 'QUANTIZE' and operator_idx not in skipped_ops:
            quant_input_tensor = tflite_graph['subgraphs'][0]['tensors'][operator['inputs'][0]]
            is_uint8_quantize_conv = quant_input_tensor['type'] == 'UINT8'

        if is_uint8_quantize_conv:
            node_name, attributes_dict, input_list, output_list = build_identity_conv_for_quantize(
                ir, tflite_graph, operator_idx)
            ir.graph.add_nodes_from([(node_name, attributes_dict)])

        # Skip ops that were already handled by a higher-level pattern
        # (e.g. QUANTIZE -> CAST -> GATHER handled by the precomputed skip plan).
        # This must stay enabled so the legacy inline skip logic below does not
        # re-process those ops and overwrite the skip plan's tensor mappings.
        # if operator_idx in skipped_ops:
        #     continue

        # Skip certain ops, such as PAD, which is merged with the next CONV_2D
        elif skip_op(tflite_graph, operator_idx):
            skip_lut = False
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
                                back_to_back_lut_inputs.append(skipped_input)
                            else:
                                skipped_input = operators[prev_producer_op]['inputs'][0]
                        else:
                            skipped_input = operators[prev_producer_op]['inputs'][0]
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
                    if (cast_consumer_type != 'GATHER'):
                        pattern_quant_cast_gather = False
                else:
                    pattern_quant_cast_gather = False
                    
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
        elif operator_idx in mxp_ops:
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
            attributes_dict['op_type'] = OpType.SYNC
            # Regular ops carry an 'attributes' dict; Sync nodes (MXP-handled groups) had none, so the
            # many downstream `node['attributes']` lookups (grid config, activation_silu, folding, ...)
            # raised KeyError. Give Sync nodes an empty attributes dict so those `'x' in attrs`/`.get`
            # checks resolve to "no attribute" instead of crashing.
            attributes_dict['attributes'] = {}
            node_name = "Sync_" + str(operator_idx)
            attributes_dict['name'] = node_name
            ir.graph.add_nodes_from([(node_name, attributes_dict)])

        # Otherwise process normally
        else:
            # Retrieve opcode and built-in options (attributes) for this node
            attributes_dict = {}
            builtin_options = operator.get("builtin_options", {})
            error_node_name = opcode.replace("_", "") + "_" + str(operator_idx)
            attributes_dict['attributes'] = tflite_attrs_to_dict(builtin_options, opcode, operator, tflite_graph, error_node_name)
            attributes_dict['op_type'] = tflite_to_onnx_op_type(opcode)

            # For Strided Slice, currently support the case of strides == 1
            if opcode == 'STRIDED_SLICE':
                begin_input_channel, end_input_channel = check_strided_slice_attributes(operator, tflite_graph, error_node_name)
                attributes_dict['attributes']['begin_input_channel'] = begin_input_channel
                attributes_dict['attributes']['end_input_channel'] = end_input_channel
            elif opcode == 'SPLIT':
                split_axis = get_tensor_data_from_tensor_index(tflite_graph, operator['inputs'][0], np.int32)[0]
                subgraph = tflite_graph['subgraphs'][0]
                tensor = subgraph['tensors'][operator['inputs'][1]]
                channels = tensor['shape'][split_axis]
                num_splits = attributes_dict['attributes']['num_splits']
                split_size = channels // num_splits
                # The last split output is handled by the primary node_name below; the
                # remaining num_splits-1 outputs each get their own node so that one
                # convolution is created per split.
                attributes_dict['attributes']['begin_input_channel'] = (num_splits - 1) * split_size
                attributes_dict['attributes']['end_input_channel'] = channels

                split_attributes = []
                for i in range(num_splits - 1):
                    ad = copy.deepcopy(attributes_dict)
                    ad['attributes']['begin_input_channel'] = i * split_size
                    ad['attributes']['end_input_channel'] = (i + 1) * split_size
                    split_attributes.append(ad)
            elif opcode == 'RESIZE_NEAREST_NEIGHBOR':
                if ((operator_idx - 4) < 0):
                    pass
                else:
                    preceding_op = operators[operator_idx-4]
                    preceding_opcode_index = preceding_op['opcode_index']
                    preceding_opcode = tflite_graph["operator_codes"][preceding_opcode_index]['builtin_code']
                    preceding_node_name = preceding_opcode.replace("_", "") + "_" + str(operator_idx-4)
                    if (preceding_opcode != 'CONV_2D') or \
                        ((preceding_opcode == 'CONV_2D') and (ir.graph.nodes[preceding_node_name]['attributes']['fused_activation_function'] != 'NONE')):
                        requant_info = get_lut_info(preceding_op, tflite_graph)
                        if (requant_info != None):
                            attributes_dict['attributes']['requant_info'] = requant_info
                
            # Unlike ONNX, there is no name like Conv_1, Conv_2 for these, so name them.
            # Will get rid of _ from the opcode name because python simulator assumes very specific parsing of node names.
            if opcode == 'SPLIT':
                input_list = [operator['inputs'][1]]
                node_name = opcode.replace("_", "") + "_" + str(operator_idx) + "_" + str(num_splits - 1)
                attributes_dict['name'] = node_name
                ir.graph.add_nodes_from([(node_name, attributes_dict)])
                output_list = [operator['outputs'][num_splits - 1]]

                split_node_names = []
                for i in range(num_splits - 1):
                    node_name_i = opcode.replace("_", "") + "_" + str(operator_idx) + "_" + str(i)
                    split_attributes[i]['name'] = node_name_i
                    ir.graph.add_nodes_from([(node_name_i, split_attributes[i])])
                    split_node_names.append(node_name_i)
            elif attributes_dict['op_type'] == OpType.RESIZE:
                input_list = operator['inputs']
                subgraph = tflite_graph['subgraphs'][0]
                tensor_in = subgraph['tensors'][operator['inputs'][0]]
                tensor_out = subgraph['tensors'][operator['outputs'][0]]
                # A single resize op can scale by more than 2x in a dimension (e.g. 5->36 on height).
                # Hardware-facing tiling logic (blob h-tiling, get_y_tile_sizes/get_num_y_tiles) is only
                # sized for ~2x height growth per blob boundary, so a resize that grows height by more
                # than that in one shot desyncs the blob's y_tiles from its real output height (see
                # get_node_tiling_info). Decompose into a chain of near-2x resize (x+y) steps instead -
                # each step's own blob (should_start_new_blob limits a blob to one resize pair) then only
                # ever sees ~2x growth, matching what the tiling math already assumes.
                h_sizes = get_resize_step_sizes(tensor_in['shape'][1], tensor_out['shape'][1])
                w_sizes = get_resize_step_sizes(tensor_in['shape'][2], tensor_out['shape'][2])
                num_steps = max(len(h_sizes), len(w_sizes)) - 1
                if len(h_sizes) - 1 < num_steps:
                    h_sizes = [h_sizes[0]] * (num_steps - (len(h_sizes) - 1)) + h_sizes
                if len(w_sizes) - 1 < num_steps:
                    w_sizes = [w_sizes[0]] * (num_steps - (len(w_sizes) - 1)) + w_sizes

                base_name = opcode.replace("_", "") + "_" + str(operator_idx)
                resize_step_node_names = []
                resize_step_h_sizes = []
                resize_step_w_sizes = []
                for step in range(num_steps):
                    suffix = '' if num_steps == 1 else ('_%d' % (step + 1))
                    node_name_x_i = base_name + suffix + "_x"
                    node_name_y_i = base_name + suffix + "_y"
                    attrs_x = copy.deepcopy(attributes_dict)
                    attrs_x['name'] = node_name_x_i
                    attrs_x['y_in'] = h_sizes[step]
                    attrs_x['y_out'] = h_sizes[step]
                    ir.graph.add_nodes_from([(node_name_x_i, attrs_x)])
                    attrs_y = copy.deepcopy(attributes_dict)
                    attrs_y['name'] = node_name_y_i
                    attrs_y['y_in'] = h_sizes[step]
                    attrs_y['y_out'] = h_sizes[step + 1]
                    ir.graph.add_nodes_from([(node_name_y_i, attrs_y)])
                    resize_step_node_names.append((node_name_x_i, node_name_y_i))
                    resize_step_h_sizes.append((h_sizes[step], h_sizes[step + 1]))
                    resize_step_w_sizes.append((w_sizes[step], w_sizes[step + 1]))

                node_name = resize_step_node_names[0][0]
                node_name_y = resize_step_node_names[-1][1]
                output_list = [operator['outputs'][0]]
                tensor_out_x_shape = copy.deepcopy(tensor_out['shape'])
                tensor_out_x_shape[1] = resize_step_h_sizes[0][0]
                tensor_out_x_shape[2] = resize_step_w_sizes[0][1]
                tensor_out_x_shape = tflite_tensor_shape_to_onnx_tensor_shape(tensor_out_x_shape)
            elif opcode == 'TRANSPOSE_CONV':
                # TFLite TRANSPOSE_CONV inputs are [output_shape(const), weights, input, (bias)].
                # Reorder into conv order [input, weights, bias] (dropping the constant
                # output_shape) so the aliased-to-Conv frontend resolves inputs[0]=activation,
                # inputs[1]=weights, inputs[2]=bias.
                op_ins = operator['inputs']
                input_list = [op_ins[2], op_ins[1]]
                if len(op_ins) > 3 and op_ins[3] != -1:
                    input_list.append(op_ins[3])
                node_name = opcode.replace("_", "") + "_" + str(operator_idx)
                attributes_dict['name'] = node_name
                ir.graph.add_nodes_from([(node_name, attributes_dict)])
                output_list = operator['outputs']
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
            # A SPLIT is modeled as num_splits separate nodes that all consume the SAME input tensor
            # (node_name is the last split; split_node_names holds the rest). Register every split as a
            # consumer, not just the primary, so downstream blob-consumer lookups
            # (get_dominant_tensor_consumer_in_blob) resolve for whichever split lands in a given blob.
            # This is independent of input_tensor.producer (graph-input splits have producer=None).
            if opcode == 'SPLIT':
                for nn in split_node_names:
                    input_tensor.consumers.append(nn)
            # If the input comes from a node, add a graph edge (connection between 2 nodes)
            # Input which doesnt have a producer is an input tensor to the graph or constant
            if input_tensor.producer:
                ir.graph.add_edge(input_tensor.producer, node_name)
                if opcode == 'SPLIT':
                    for nn in split_node_names:
                        ir.graph.add_edge(input_tensor.producer, nn)
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
            # Update the tensors dict. the output tensor to the node should have it as consumer
            output_tensor = ir.tensors[output_name]
            if attributes_dict['op_type'] == OpType.RESIZE:
                output_name_x = output_name + '_x'
                # The resize's real input tensor's own .scale/.zero_point are the tflite-declared
                # (pre-fusion) quant params - if the producing node has a fused activation_silu/
                # requant LUT, the values it ACTUALLY emits are requantized to that LUT's output
                # scale/zp instead (see update_input_scale_zp, frontend_hardware_agnostic.py, for
                # the same override applied later for other op types). Use the effective post-LUT
                # scale/zp here too, so the resize chain's intermediate tensors carry the same
                # quantization the real data will have at runtime.
                resize_input_tensor = ir.tensors[inputs[0]]
                resize_input_scale = resize_input_tensor.scale
                resize_input_zp = resize_input_tensor.zero_point
                resize_producer_name = resize_input_tensor.producer
                if resize_producer_name and ir.graph.has_node(resize_producer_name):
                    resize_producer_attrs = ir.graph.nodes[resize_producer_name].get('attributes', {}) or {}
                    resize_act_silu = resize_producer_attrs.get('activation_silu')
                    resize_requant_info = resize_producer_attrs.get('requant_info')
                    if resize_act_silu is not None:
                        resize_input_scale = resize_act_silu['output_scale'][0]
                        resize_input_zp = resize_act_silu['output_zp'][0]
                    elif resize_requant_info is not None:
                        resize_input_scale = resize_requant_info['output_scale'][0]
                        resize_input_zp = resize_requant_info['output_zp'][0]
                tensor_out_x = Tensor(output_name_x,np.zeros(tensor_out_x_shape),is_constant=False, shape=tensor_out_x_shape,
                                       scale=resize_input_scale, zero_point=resize_input_zp)
                tensor_out_x.producer = node_name
                ir.tensors[output_name_x] = tensor_out_x
                outputs.append(output_name_x)
                first_y_node_name = resize_step_node_names[0][1]
                output_tensor.producer = node_name_y
                ir.graph.add_edge(node_name, first_y_node_name)
            else:
                outputs.append(output_name)
                output_tensor.producer = node_name

        ir.graph.nodes[node_name]['outputs'] = outputs
        if attributes_dict['op_type'] == OpType.RESIZE:
            # Wire the rest of the chained resize (x,y) step pairs. Step 0's x-node input/output was
            # already wired above (real input tensors -> output_name_x); from here we connect each
            # step's y-node to its x-node's output, and (for step>0) each step's x-node to the previous
            # step's y-node output, ending at the real output tensor on the final step's y-node.
            prev_y_out_name = output_name_x
            prev_y_node_name = None
            for step_idx, (x_name, y_name) in enumerate(resize_step_node_names):
                if step_idx > 0:
                    x_in_name = prev_y_out_name
                    ir.tensors[x_in_name].consumers.append(x_name)
                    ir.graph.nodes[x_name]['inputs'] = [x_in_name, inputs[1]]
                    x_out_shape = copy.deepcopy(tensor_out['shape'])
                    x_out_shape[1] = resize_step_h_sizes[step_idx][0]
                    x_out_shape[2] = resize_step_w_sizes[step_idx][1]
                    x_out_shape = tflite_tensor_shape_to_onnx_tensor_shape(x_out_shape)
                    x_out_name = output_name + ('_%d_x' % step_idx)
                    tensor_x_out = Tensor(x_out_name, np.zeros(x_out_shape), is_constant=False, shape=x_out_shape,
                                           scale=resize_input_scale, zero_point=resize_input_zp)
                    tensor_x_out.producer = x_name
                    ir.tensors[x_out_name] = tensor_x_out
                    ir.graph.nodes[x_name]['outputs'] = [x_out_name]
                    ir.graph.add_edge(prev_y_node_name, x_name)
                    y_in_name = x_out_name
                else:
                    y_in_name = output_name_x

                if step_idx == len(resize_step_node_names) - 1:
                    y_out_name = output_name
                else:
                    y_out_shape = copy.deepcopy(tensor_out['shape'])
                    y_out_shape[1] = resize_step_h_sizes[step_idx][1]
                    y_out_shape[2] = resize_step_w_sizes[step_idx][1]
                    y_out_shape = tflite_tensor_shape_to_onnx_tensor_shape(y_out_shape)
                    y_out_name = output_name + ('_%d_y' % step_idx)
                    tensor_y_out = Tensor(y_out_name, np.zeros(y_out_shape), is_constant=False, shape=y_out_shape,
                                           scale=resize_input_scale, zero_point=resize_input_zp)
                    tensor_y_out.producer = y_name
                    ir.tensors[y_out_name] = tensor_y_out

                ir.tensors[y_in_name].consumers.append(y_name)
                ir.graph.nodes[y_name]['outputs'] = [y_out_name]
                ir.graph.nodes[y_name]['inputs'] = [y_in_name, inputs[1]]
                ir.graph.add_edge(x_name, y_name)

                prev_y_out_name = y_out_name
                prev_y_node_name = y_name

        if opcode == 'SPLIT':
            # Update outputs for each of the remaining split outputs (one node per split)
            for i, node_name_i in enumerate(split_node_names):
                outputs = []
                output_index = operator['outputs'][i]
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
                output_tensor.producer = node_name_i
                ir.graph.nodes[node_name_i]['outputs'] = outputs
                ir.graph.nodes[node_name_i]['inputs'] = inputs

    # Update qparams if any quantize ops were skipped
    for idx in range(len(skipped_quantize)):
        ir = update_qparams_for_skipped_quantize(ir, tflite_graph, skipped_quantize[idx], skipped_quantize_inputs[idx])

    for out_idx in skip_outputs:
        old_output = get_short_tensor_name(out_idx)
        new_output = get_short_tensor_name(skip_outputs[out_idx])
        replace_ir_output_tensor(ir, old_output, new_output)

    # After all skip/bypass resolution, any Concat input whose quantization no longer matches the
    # Concat output (because a requantizing QUANTIZE before it was bypassed) gets an Identity requant
    # conv inserted, so calc_concat_qparams sees matching scale/zp. Restrict to inputs that actually
    # trace back through one of the bypassed QUANTIZE ops recorded above, or through a genuine
    # back-to-back-LUT chain (two full QUANTIZE->CAST->GATHER patterns stacked, resolved to their
    # ultimate root tensor above) - both are real signatures of "this input lost a requant on the
    # way to the Concat", as opposed to an ordinary single-LUT producer whose reported scale already
    # matches the Concat (which must NOT get a requant node).
    bypassed_quantize_tensor_names = {get_short_tensor_name(t) for t in skipped_quantize_inputs}
    bypassed_quantize_tensor_names |= {get_short_tensor_name(t) for t in back_to_back_lut_inputs}
    insert_concat_input_requant_nodes(ir, bypassed_quantize_tensor_names)

# Generate IR from TFLite graph
def tflite_to_ir(ir,tflite_graph,mxp_ops,mxp_tensor_to_offset):
    # Process JSON for entire graph, including MXP subgraphs.
    # MXP subgraphs are replaced with a Sync node.
    # Can also avoid adding tensors if every input and output is an MXP node.
    add_tflite_tensors(ir, tflite_graph)
    add_tflite_inputs(ir, tflite_graph)
    add_tflite_outputs(ir, tflite_graph)

    if ir.sync_with_MXP:
        # Populate ir.mxp_tensor_to_offset BEFORE add_tflite_ops so that
        # replace_ir_output_tensor can copy offsets when renaming tensors
        for tensor_id in mxp_tensor_to_offset:
            tensor_name = f"T{tensor_id}"
            ir.mxp_tensor_to_offset[tensor_name] = mxp_tensor_to_offset[tensor_id]

    add_tflite_ops(ir, tflite_graph, mxp_ops)

    # After IR construction, some Sync nodes may have had their inputs rewired (e.g., GATHER skipped/fused),
    # but the tensors_to_mxp set was not updated. Scan all Sync nodes and ensure their input tensors
    # are marked as going to MXP so the simulator clips them before transfer.
    if ir.sync_with_MXP:
        for node_name, node in ir.graph.nodes.items():
            if node.get('op_type') == OpType.SYNC:
                for input_tensor_name in node.get('inputs', []):
                    # Only add if it has an offset entry (means it was originally an MXP tensor)
                    if input_tensor_name in ir.mxp_tensor_to_offset:
                        ir.tensors_to_mxp.add(input_tensor_name)
                    else:
                        # For rewired inputs (e.g., producer output substituted for GATHER output),
                        # use the producer's offset if available
                        for prod_node_name, prod_node in ir.graph.nodes.items():
                            if input_tensor_name in prod_node.get('outputs', []):
                                # Producer node found; check if it has offset info we can reuse
                                # For now, create a dummy offset (will be fixed during sequencing)
                                if input_tensor_name not in ir.mxp_tensor_to_offset:
                                    ir.mxp_tensor_to_offset[input_tensor_name] = 0
                                ir.tensors_to_mxp.add(input_tensor_name)
                                break

    if ir.sync_with_MXP:
        # TODO: Each tensor to/from MXP is stored below, and F4/F5 flags will
        # set/wait before these are written/read. However, if a tensor needs to be
        # read multiple times, e.g., by 2 separate blobs, the current implementation
        # will wait both times. This can cause a deadlock if the MXP only sets the
        # flag once.
        print("From MXP: " + str(ir.tensors_from_mxp))
        print("To MXP:   " + str(ir.tensors_to_mxp))

    return ir

def main():
    pass

if __name__ == "__main__":
    main()
