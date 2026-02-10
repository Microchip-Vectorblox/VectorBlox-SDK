import argparse
import os
import json
import sys
from contextlib import contextmanager

from .utils import existing_file, existing_dir, generate_inputs_outputs
from .transform_tflite import load_graph, save_graph, transform_graph, is_multi_input


@contextmanager
def exception_catcher(unsupported_ops):
    try:
        yield
    except AssertionError as e:
        error_msg = (
                "\n" + "="*75 + "\n"
                "\033[31m" + "CRITICAL ERROR: Unsupported Layer Parameters".center(75) + "\033[0m\n"
                "\n\033[31mIssue:\033[0m Some layers have parameters/arguments that are not currently supported.\n"
                "Those layers are saved in \033[31munsupported_ops\033[0m folder in your working directory.\n"
                "\n\033[31mUnsupported Operations:\033[0m\n"
            )
        sys.stderr.write(error_msg)

        for i, op, opcode, error_param, param_value, input_param in unsupported_ops:
            op_msg = f"\n  \033[31m[Op {i}]\033[0m {opcode}\n"
            
            if len(param_value) == 0:
                op_msg += f"    Status: Not currently supported\n"
            else:
                op_msg += (
                    f"    Parameter: \033[31m{param_value[0]}\033[0m, Value: {input_param[0]}\n"
                    f"    Supported: {error_param[0]}\n"
                )
            
            sys.stderr.write(op_msg)
        
        footer_msg = (
            "\n" + "="*75 + "\n"
            "We are continuously working to improve the SDK."
            "\n\033[36mFor further assistance:\033[0m\n"
            "  Email: vectorblox@microchip.com\n"
            "  Repository: https://github.com/Microchip-Vectorblox\n"
            + "="*75 + "\n"
        )
        
        sys.stderr.write(footer_msg)
        sys.exit(1)


def get_unique_values(values):
    uniq_values = []

    for v in values:
        if type(v) == str:
            uniq_values.append(v.upper())
        else:
            uniq_values.append(v)

    return list(set(uniq_values))


def get_negative_axis(axis, dims=4):
    if(axis>=0):
        return axis-dims

    return axis


def check_valid(valid, op_code, op, tensors):
    input_param = []
    error_param = []
    param_value = []

    if not op_code in valid:
        return False, error_param, param_value, input_param 

    if op_code == "SPLIT":
        i_tensor = tensors[op['inputs'][1]]
    elif op_code =="TRANSPOSE_CONV":
        i_tensor = tensors[op['inputs'][2]]
    else:
        i_tensor = tensors[op['inputs'][0]]
    idims = len(i_tensor['shape'])
    opts = None
    if 'builtin_options' in op:
        opts = op['builtin_options']
    
    if i_tensor is not None: 
        if "type" in i_tensor:
            values = valid[op_code]["type"] 
            dtype = i_tensor["type"]  
            if dtype.upper() not in values:
                error_param.append(get_unique_values(values))
                param_value.append("Input data type")
                input_param.append(dtype)
                return False, error_param, param_value, input_param
        
    if opts is not None:    
        if "axis" in opts:
            values = valid[op_code]["axis"]
            axis = get_negative_axis(opts["axis"], idims)
            if axis not in values:
                error_param.append(get_unique_values(values))
                param_value.append("axis")
                input_param.append(axis)
                return False, error_param, param_value, input_param
                
        if "fused_activation_function" in opts: 
            values = valid[op_code]["fused_activation_function"]
            faf = opts["fused_activation_function"]
            if type(faf) == str:
                faf = faf.upper()
            if faf not in values:
                error_param.append(get_unique_values(values))
                param_value.append("fused_activation_function")
                input_param.append(faf)
                return False, error_param, param_value, input_param
                
        if "mode" in opts:  
            values = valid[op_code]["mode"]
            mode = opts["mode"]

            if type(mode) == str:
                mode = mode.upper()

            if mode not in values:
                error_param.append(get_unique_values(values))
                param_value.append("mode")
                input_param.append(mode)
                return False, error_param, param_value, input_param
                
        if "dim" in opts: 
            dims = valid[op_code]
            values = dims["dim"]
            dim = get_negative_axis(opts["mode"], idims)
            if dim not in values:
                error_param.append(get_unique_values(values))
                param_value.append("dim")
                input_param.append(dim)
                return False, error_param, param_value, input_param
        
        if "padding" in opts: 
            values = valid[op_code]["padding"]
            pad = opts["padding"]

            if type(pad) == str:
                pad = pad.upper()
                
            if pad not in values:
                error_param.append(get_unique_values(values))
                param_value.append("padding")
                input_param.append(pad)
                return False, error_param, param_value, input_param

    return True, error_param, param_value, input_param


def verify_graph(graph, core, accel, debug, transformed=False):
    subgraph = graph['subgraphs'][0]
    operators, tensors = subgraph['operators'], subgraph['tensors']
    buffers, opcodes = graph['buffers'], graph['operator_codes']
    builtin_codes = [_['builtin_code'] for _ in opcodes]

    if not transformed:
        id_count = 0
        pattern_count = 0
        i = 0
        while i < len(operators): 
            op = operators[i]
            opts = None
            if 'builtin_options' in op:
                opts = op['builtin_options']
            opcode = builtin_codes[op['opcode_index']]

            if opcode in ['CONV_2D']:
                if 'FLOAT32' in [tensors[_]['type'] for _ in op['inputs']]:
                    print('FLOAT32 inputs not supported (for operator {} {})'.format(i, opcode))
                    return False

            if opcode in ['CONV_2D','SUM']:
                if 'FLOAT32' in [tensors[_]['type'] for _ in op['outputs']]:
                    print('FLOAT32 outputs not supported (operator {} {})'.format(i, opcode))
                    return False

            if opcode in ['DIV']:
                if is_multi_input(op, tensors, buffers):
                    if 'FLOAT32' in [tensors[_]['type'] for _ in op['outputs']]:
                        print('FLOAT32 DIV w/ multiple inputs is not supported (operator {} {})'.format(i, opcode))
                        return False

            i += 1
    else:
        errors = []
        
        codes = [_['builtin_code'] for _ in graph['operator_codes']]
        valid_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'supported_ops.json') 
        with open(valid_path) as f:
            valid = json.load(f)

        i = 0
        while i < len(operators): 
            op = operators[i]
            if 'builtin_options' in op:
                opts = op['builtin_options']
            opcode = codes[op['opcode_index']]

            valid_op, error_param, param_value, input_param = check_valid(valid, opcode, op, tensors)
            if not valid_op:
                errors.append([i,op,opcode, error_param, param_value, input_param])

            i += 1

        if len(errors) >0:
            with exception_catcher(errors):
                assert(0)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite", type=existing_file)
    parser.add_argument("-c", "--core", choices=['MXP', 'RISCV'], default='MXP')
    parser.add_argument("-a", "--accel", choices=['FIA', 'TSNP'], default='FIA')
    parser.add_argument("-t", "--transformed", action='store_true')
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()

    graph, dir_obj = load_graph(args.tflite)

    verify_graph(graph, args.core, args.accel, args.verbose, args.transformed)

    if not dir_obj is None:
        dir_obj.cleanup()


if __name__ == "__main__":
    main()
