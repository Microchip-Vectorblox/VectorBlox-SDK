import argparse
import json
import copy
import subprocess
import shlex
import os.path
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import sys, shutil, glob, tempfile
from tqdm import tqdm 
import numpy as np

#LUT related optimizations options
OPTIMIZED_WITH_LUT = 1
VCI_LUT = 1
MAX_LUTs = 4

@contextmanager
def exception_catcher(unsupported_ops):
    try:
        yield
    except AssertionError as e:
        sys.stderr.write("\n\033[31m######################## VECTORBLOX ERROR! #############################\033[0m\n")
        sys.stderr.write("\033[31mSome layers have parameters/arguments that we do not currently support.\033[0m")
        sys.stderr.write("\nPlease check the folder \033[31m unsupported_ops \033[0m in your working directory.")
        sys.stderr.write("\nFor futher assistance, please send the layer(s) to the vectorblox team at:\n\033[31mvectorblox@microchip.com\033[0m\n")

        sys.stderr.write("\nError unsupported ops:")
        for i,op,opcode, error_param, param_value, input_param in unsupported_ops:
            if len(param_value) == 0:
                print("\t {} at location {} is not currently supported.\n".format(opcode, i))   
            else:
                print("\t {} at location {}, \033[31m {}={} \033[0m is not currently supported.".format(opcode, i, param_value[0], input_param[0]))     
                print("\t\t-->Supported values are = {}\n".format(error_param[0]))
        # print('\nThese operators have been extracted and saved to ./unsupported_ops')
        # print('Please send to VectorBlox team for support')
        print()
        sys.exit(1)


def forks(op0, op0_inputs, op0_outputs, operators, tensors):
    if len(op0_outputs) == 2:
        return True

    count = 0
    tensor = op0_outputs[0]
    for i, op in enumerate(operators):
        if op != op0:
            op_inputs = [_ for _ in op['inputs'] if _ != -1]
            for t in op_inputs:
                if t == tensor:
                    count += 1

    return count > 1


def in_inputs(op0, op):

    idx_input_tensors = [_ for _ in op0['inputs'] if _ != -1]
    op_outputs = [_ for _ in op['outputs'] if _ != -1]
    for o in op_outputs:
        if o in idx_input_tensors:
            return True

    return False


def is_input(idx, operators, cuts, prev_nodes=[]):
    nodes = [idx]

    if idx != -1:
        iop = operators[idx]

        for i, op in enumerate(operators):
            if i not in prev_nodes:
                if in_inputs(iop, op) and i not in cuts:
                    nodes.append(i)
                    nodes += is_input(i, operators, cuts, nodes+prev_nodes)

    return nodes


def get_cuts(jname, cuts, include_outputs=True):
    with open(jname) as f:
        graph = json.load(f)
    # assert(len(graph['subgraphs']) == 1)

    buffers = graph['buffers']
    subgraph = graph['subgraphs'][0]
    outputs = subgraph['outputs']
    operators = subgraph['operators']
    if include_outputs:
        for i, op in enumerate(operators):
            for o in op['outputs']:
                if o in outputs:
                    cuts.append(i)
    tensors = subgraph['tensors']
    codes = [_['builtin_code'] for _ in graph['operator_codes']]
    splits = []
    current = []
    prev_op = None
    prev_opcode = None
    prev_inputs = None
    prev_outputs = None

    cut_groups = {}
    for c in cuts:
        cut_groups[c] = set(is_input(c, operators, cuts))

    while len(cut_groups.keys()):
        modified = False
        keys = list(cut_groups.keys())
        k0 = keys[0]

        for k in keys:
            if k != k0:
                s0 = cut_groups[k0]
                s = cut_groups[k]

                if len(s) + len(s0) > len(s0.union(s)): #shared nodes
                    cut_groups[k0] = s0.union(s)
                    cut_groups.pop(k)
                    modified = True

        if not modified:
            splits.append(list(cut_groups.pop(k0)))

    return splits

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


def lut_pattern(operators, codes, tensors, buffers, idx, opcode):
    if not OPTIMIZED_WITH_LUT:
        return []

    patterns = []

    prev_op = None
    prev_inputs = None
    prev_outputs = None

    max_idx = len(operators)
    while idx < max_idx:
        op = operators[idx]
        opcode = codes[op['opcode_index']]
        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]

        ibuffers = []
        for _ in op_inputs:
            if 'buffer' in tensors[_]:
                ibuffers += [buffers[tensors[_]['buffer']]]
        multi_input = len(ibuffers) > 1 and not any(['data' in _ for _ in ibuffers])
        filters = [_ for _ in op_inputs if 'data' in buffers[tensors[_]['buffer']]]

        next_op, next_opcode = None, ''
        if idx < max_idx -1:
            next_op = operators[idx+1]
            next_opcode = codes[next_op['opcode_index']]

        connected = True
        if prev_op != None:
            connected = any([_ in prev_outputs for _ in op_inputs])

        forked = False
        if prev_op != None:
            forked = forks(prev_op, prev_inputs, prev_outputs, operators, tensors)


        if not connected:
            break
        
        #SILU pattern
        elif opcode == "LOGISTIC" and next_opcode == "MUL": 
            patterns.append(idx)
            patterns.append(idx+1)
            idx += 1

        elif forked:
            break
        
        elif opcode in ["HARD_SWISH", "LOGISTIC", "QUANTIZE"]:
            patterns.append(idx)

        elif opcode in ['LEAKY_RELU', 'RELU', 'RELU6', 'RELU_N1_TO_1', 'RELU_0_TO_1']:
            patterns.append(idx)
          
        elif opcode in ["MUL", "ADD", "SUB"] and len(filters) == 1:
            weight_tensor = tensors[filters[0]]
            if 'shape' in weight_tensor and len(weight_tensor['shape'])> 0 :
                weight_shape = weight_tensor['shape']
                lut_count = weight_shape[-1] #channels last
            else:
                lut_count = 1

            if lut_count <= MAX_LUTs:
                patterns.append(idx)
            else:
                break
        else:
            break

        idx += 1

        prev_op = op
        prev_inputs = op_inputs
        prev_outputs = op_outputs
        
    return patterns

def get_splits(jname, split_every_op=False):
    valid_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'supported_ops.json') 
    with open(valid_path) as f:
        valid = json.load(f)

    with open(jname) as f:
        graph = json.load(f)
    assert(len(graph['subgraphs']) == 1)

    buffers = graph['buffers']
    subgraph = graph['subgraphs'][0]
    outputs = subgraph['outputs']
    operators = subgraph['operators'] # ops = subgraph['operators']
    tensors = subgraph['tensors']
    codes = [_['builtin_code'] for _ in graph['operator_codes']]
    splits = []
    errors = []
    specials = []
    current = []
    prev_op = None
    prev_opcode = None
    prev_inputs = None
    prev_outputs = None

    i = 0
    while i < len(operators):
        op = operators[i]
        if 'builtin_options' in op:
            opts = op['builtin_options']
        opcode = codes[op['opcode_index']]

        next_op, next_opcode = None, ''
        if i < len(operators)-1:
            next_op = operators[i+1]
            next_opcode = codes[next_op['opcode_index']]

        op_inputs = [_ for _ in op['inputs'] if _ != -1]
        op_outputs = [_ for _ in op['outputs'] if _ != -1]
        
        input_buffers = [buffers[tensors[_]['buffer']] for _ in op_inputs]
        multi_input = len(input_buffers) > 1 and not any(['data' in _ for _ in input_buffers]) 
        output_buffers = [buffers[tensors[_]['buffer']] for _ in op_outputs]

        connected = True
        if prev_op != None:
            connected = any([_ in prev_outputs for _ in op_inputs])

        forked = False
        if prev_op != None:
            forked = forks(prev_op, prev_inputs, prev_outputs, operators, tensors)
            
        valid_op, error_param, param_value, input_param = check_valid(valid, opcode, op, tensors)

        if not valid_op:
            errors.append([i,op,opcode, error_param, param_value, input_param]) 

        # used for debugging, add ops to capture and save as solo graphs
        if opcode in ['QUANTIZE']:
            specials.append([i])

        if split_every_op:
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
        elif prev_opcode in ['UNPACK', 'RESHAPE','TRANSPOSE']: #TODO start a new graph after OP
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
              
        # elif not connected or forked:
        elif not connected:
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
        elif opcode == 'LOGISTIC' and next_opcode == 'MUL' and len(lut_pattern(operators, codes, tensors, buffers, i, opcode)):
            patterns = lut_pattern(operators, codes, tensors, buffers, i, opcode) 

            current += patterns
            i += len(patterns) - 1
            # update outputs to be from last op in pattern
            op = operators[i]
            op_outputs = [_ for _ in op['outputs'] if _ != -1]
        elif forked:
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
        elif any([_ for _ in op_inputs if _ in outputs]):
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
            # @TODO: Maybe inject an identity if we can fit full map  --> suppose to be only FIA layers
        elif opcode in ['CONV_2D', 'TRANSPOSE_CONV', 'FULLY_CONNECTED', 'RESIZE_BILINEAR', ' UNIDIRECTIONAL_SEQUENCE_LSTM', 'SOFTMAX', 'ARG_MAX', 'TILE', 'SPLIT', 'SPLIT_V', 'PACK', 'UNPACK', 'RESHAPE','TRANSPOSE', 'AVERAGE_POOL_2D', 'MEAN']: # start a new graph before key subgraph OP
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
        elif opcode in ['CONCATENATION']: # start a new graph before key subgraph OP
            if len(current):
                splits.append(current)
            current = []
            current.append(i)
        elif len(lut_pattern(operators, codes, tensors, buffers, i, opcode)):
            patterns = lut_pattern(operators, codes, tensors, buffers, i, opcode)

            current += patterns
            i += len(patterns) - 1
            # update outputs to be from last op in pattern
            op = operators[i]
            op_outputs = [_ for _ in op['outputs'] if _ != -1]
            
        # elif opcode in ['DEPTHWISE_CONV_2D'] and 'padding' in opts and opts['padding'] != 'VALID': # start a new graph before key subgraph OP
        elif opcode in ['DEPTHWISE_CONV_2D']: # start a new graph before key subgraph OP
            if len(current):
                splits.append(current)
            current = []
            current.append(i)

        elif opcode in ['MUL', 'SUB', 'DIV', 'GREATER', 'GREATER_EQUAL', 'LESS', 'LESS_EQUAL', 'EQUAL', 'NOT_EQUAL', 'MINIMUM', 'MAXIMUM'] and multi_input: # start a new graph before multi input subgraph OPS
            if len(current):
                splits.append(current)
            current = []
            current.append(i)

        else:
            current.append(i)

        prev_op = op
        prev_opcode = opcode

        prev_inputs = op_inputs
        prev_outputs = op_outputs
        i = i + 1

    splits.append(current)
    return splits, errors, specials


def gen_subgraph(idir, odir, jname, splits, vnnx, forced_shape=None):

    with open(jname) as f:
        g = json.load(f)

    buffers = g['buffers']
    subgraph = g['subgraphs'][0]
    operators = subgraph['operators']
    tensors = subgraph['tensors']

    show_progress = False
    if len(splits) > 100:
        show_progress = True

    json_subgraphs = []
    for s, split_indices in tqdm(enumerate(splits), total=len(splits), disable=(not show_progress)):
        subgraph_jname = os.path.join(odir, os.path.basename(jname).replace('.json', '.{}.json'.format(s)))
        json_subgraphs.append(subgraph_jname)

        sg = {}
        sg['subgraphs'] = [{}]
        sg['signature_defs'] = [{'inputs':None, 'outputs':None, 'signature_key':'serving_default', 'subgraph_index':0}]
        ssubgraph = sg['subgraphs'][0]

        if 'name' in subgraph:
            ssubgraph['name'] = subgraph['name']

        for key in ['version', 'description', 'metadata', 'operator_codes']:
            try:
                sg[key] = g[key]
            except:                
                sg[key] = []

        selected_operators = []
        for i in split_indices:
            selected_operators.append(operators[i])

        input_ops = selected_operators[0]['inputs']
        output_ops = selected_operators[-1]['outputs']

        
        input_ops = selected_operators[0]['inputs']
        output_ops = selected_operators[-1]['outputs']

        all_input_ops = []
        all_output_ops = []

        for op in selected_operators:
            all_input_ops += op['inputs']
            all_output_ops += op['outputs']

        input_ops = [_ for _ in all_input_ops if _ not in all_output_ops]
        output_ops = [_ for _ in all_output_ops if _ not in all_input_ops]

        input_names = []
        input_tensors = []
        for io in input_ops:
            if io != -1:
                t = tensors[io]
                buf = buffers[t['buffer']]
                if 'data' not in buf:
                    input_names.append(t['name'])
                    input_tensors.append(io)

        output_names = []
        output_tensors = []
        for io in output_ops:
            if io != -1:
                t = tensors[io]
                buf = buffers[t['buffer']]
                if 'data' not in buf:
                    output_names.append(t['name'])
                    output_tensors.append(io)

        # remove unused buffers/tensors
        
        used_buffers = []
        used_tensors = []
        for op in selected_operators:
            
            op_inputs = [_ for _ in op['inputs'] if _ != -1]
            op_outputs = [_ for _ in op['outputs'] if _ != -1]
            op_intermediates = []

            if 'intermediates' in op:
                op_intermediates = [_ for _ in op['intermediates'] if _ != -1]

            for io in op_inputs + op_outputs + op_intermediates:
                used_buffers.append(tensors[io]['buffer'])
                used_tensors.append(io)

        sbuffers = []
        for b in range(len(buffers)):
            if b not in used_buffers:
                sbuffers.append({})
            else:
                sbuffers.append(buffers[b])
        sg['buffers'] = sbuffers

        stensors = []
        for t in range(len(tensors)):
            if t not in used_tensors:
                stensors.append({})
            else:
                if not forced_shape is None:
                    buf = buffers[tensors[t]['buffer']]
                    if 'data' not in buf:
                        tensors[t]['shape'] = forced_shape
                stensors.append(tensors[t])
        ssubgraph['tensors'] = stensors

        ssubgraph['operators'] = selected_operators
        ssubgraph['inputs'] = input_tensors
        ssubgraph['outputs'] = output_tensors

        graph_inputs = [{'name': n, 'tensor_index': i} for n, i in zip(input_names, input_tensors)]
        graph_outputs = [{'name': n, 'tensor_index': i} for n, i in zip(output_names, output_tensors)]

        sg['signature_defs'][0]['inputs'] = graph_inputs
        sg['signature_defs'][0]['outputs'] = graph_outputs

        schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
        
        with open(subgraph_jname, 'w+') as f:
            json.dump(sg, f)
        cmd = 'flatc -b --strict-json --defaults-json -o {} {} {}'.format(odir, schema_path, subgraph_jname)
        subprocess.run(shlex.split(cmd))

        if vnnx:
            subgraph_tflite = subgraph_jname.replace('.json', '.tflite')
            subgraph_vnnx = subgraph_jname.replace('.json', '.vnnx')
            cmd = 'vnnx_compile -c {} -t {} -o {}'.format(os.environ.get('HW_CONFIG'),subgraph_tflite, subgraph_vnnx)
            subprocess.run(cmd, shell=True, check=True)

    return json_subgraphs


def generate_split_graphs(tflite_model, dir, split_every_op=False, cuts=None, vnnx=False):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    jname = os.path.join(dir, os.path.basename(tflite_model).replace('.tflite','.json'))

    cmd = 'flatc -t --strict-json --defaults-json -o {} {} -- {}'.format(dir, schema_path, tflite_model)
    subprocess.run(shlex.split(cmd))

    #update operator builtin_code
    with open(schema_path.replace('.fbs', '.json')) as f:
        schema = json.load(f)
        operators = schema["definitions"]["tflite_BuiltinOperator"]["enum"]
        with open(jname) as f:
            graph = json.load(f)
            for i,g in enumerate(graph['operator_codes']):
                if 'deprecated_builtin_code' in g and not g['deprecated_builtin_code'] == 127:
                    graph['operator_codes'][i]['builtin_code'] = operators[g['deprecated_builtin_code']]
        with open(jname, 'w') as f:
            json.dump(graph, f)


    subdir = os.path.join(dir, 'subgraphs') 
    if not os.path.exists(subdir):
        os.mkdir(subdir)

    # errors.append([i,op,opcode])
    splits, error_ops, special_ops = get_splits(jname, split_every_op)
    # print(splits)
    # sys.stderr.write('splits: {} not implemented\n'.format(splits))
    if len(error_ops)>0: 
        errors_dir = os.path.join(os.path.join(os.getcwd(), 'unsupported_ops'))
        if os.path.exists(errors_dir):
            shutil.rmtree(errors_dir)
            os.mkdir(errors_dir)
        else:
            os.mkdir(errors_dir)
        error_locations = [[_[0]] for _ in error_ops]
        
        gen_subgraph(subdir, errors_dir, jname, error_locations, False)
        for i, (error_op, fname) in enumerate(zip(error_ops, sorted(glob.glob(os.path.join(errors_dir, '*.tflite'))))):
            id, op, op_code = error_op[:3]
            dst = fname.replace('.{}.tflite'.format(i), '.{}.{}.tflite'.format(op_code,id))
            shutil.move(fname, dst)
        for tmp in glob.glob(os.path.join(errors_dir, '*.json')):
            os.remove(tmp)
        with exception_catcher(error_ops):
            assert(0)

    if len(special_ops):
        special_dir = os.path.join(subdir, 'special')
        if not os.path.exists(special_dir):
            os.makedirs(special_dir)
        gen_subgraph(subdir, special_dir, jname, special_ops, False) #, [256,])
    return gen_subgraph(subdir, subdir, jname, splits, vnnx)


def generate_cut_graphs(tflite_model, cuts=None):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    dir = os.path.dirname(tflite_model)
    if dir == '':
        dir = './'
    jname = os.path.join(dir, os.path.basename(tflite_model).replace('.tflite','.json'))

    cmd = 'flatc -t --strict-json --defaults-json -o {} {} -- {}'.format(dir, schema_path, tflite_model)
    subprocess.run(shlex.split(cmd))

    #update operator builtin_code
    with open(schema_path.replace('.fbs', '.json')) as f:
        schema = json.load(f)
        operators = schema["definitions"]["tflite_BuiltinOperator"]["enum"]
        with open(jname) as f:
            graph = json.load(f)
            for i,g in enumerate(graph['operator_codes']):
                if 'deprecated_builtin_code' in g and not g['deprecated_builtin_code'] == 127:
                    graph['operator_codes'][i]['builtin_code'] = operators[g['deprecated_builtin_code']]
        with open(jname, 'w') as f:
            json.dump(graph, f)

    cmd = 'flatc -b --strict-json --defaults-json -o {} {} {}'.format(dir, schema_path, jname)
    subprocess.run(shlex.split(cmd))

    cuts = get_cuts(jname, cuts)
    return gen_subgraph(dir, dir, jname, cuts, False)


def join_graph(graphs):
    graph = copy.deepcopy(graphs[0])

    subgraph_inputs = []
    subgraph_outputs = []
    for g in graphs:
        subgraph_inputs += g['signature_defs'][0]['inputs']
        subgraph_outputs += g['signature_defs'][0]['outputs']
    graph_inputs = [_ for _ in subgraph_inputs if _ not in subgraph_outputs]
    graph_outputs = [_ for _ in subgraph_outputs if _ not in subgraph_inputs]

    operators = []
    for g in graphs:
        operators += g['subgraphs'][0]['operators']

    graph['subgraphs'][0]['operators'] = operators
    graph['signature_defs'][0]['inputs'] = graph_inputs
    graph['signature_defs'][0]['outputs'] = graph_outputs
    graph['subgraphs'][0]['inputs'] = [_['tensor_index'] for _ in graph_inputs]
    graph['subgraphs'][0]['outputs'] = [_['tensor_index'] for _ in graph_outputs]

    return graph


def generate_join_graphs(tflite_model, dir, debug=False):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    json_subgraphs = []
    for subgraph_jname in sorted([_ for _ in os.listdir(dir) if '.json' in _]):
        with open(os.path.join(dir, subgraph_jname)) as f:
            json_subgraphs.append(json.load(f))

    graph = join_graph(json_subgraphs)
    jname = os.path.join(os.path.dirname(tflite_model), os.path.basename(tflite_model).replace('.tflite','.json'))
    with open(jname, 'w') as f:
        json.dump(graph, f)

    if debug:
        cmd = 'flatc -b --strict-json --defaults-json -o {} {} {}'.format(os.path.dirname(tflite_model), schema_path, jname)
        subprocess.run(shlex.split(cmd))

    return graph

def get_output_scale(scale_factor):
    q_min = -128
    q_max = 127

    min_output_fp = (q_min +128)*(1/scale_factor)
    max_output_fp = (q_max +128)*(1/scale_factor)

    output_scale = (max_output_fp-min_output_fp)/255

    output_zero_point = q_min - (min_output_fp/output_scale)

    return output_scale, output_zero_point


def get_details_from_constants(var_factor):
    
    max_value = max(var_factor)
    min_value = min(var_factor)
    if max_value > 0 :
        zero_point =  -128
        q_value = 127
        fp_value = max_value
    else:
        zero_point = 127
        q_value = -128
        fp_value = min_value

    scale = fp_value/(q_value-zero_point)

    q_value = [round((i/scale) + zero_point) for i in var_factor]
    
    return scale, zero_point, q_value


def get_quantize_values(scale, mean):

    if isinstance(mean, list) and isinstance(scale, list):
        #scaling mean and scale inputs
        scale_factor = [1 / j for j in scale]

        mean_factor = [i / j for i, j in zip(mean, scale)]
        mean_factor = [-1 * j for j in mean_factor]

        mean_scale, mean_zero_point, mean_q_value = get_details_from_constants(mean_factor)
        scale, scale_zero_point, scale_q_value = get_details_from_constants(scale_factor)

        return [scale, scale_zero_point, scale_q_value], [mean_scale, mean_zero_point, mean_q_value]

    elif isinstance(scale, list) and mean ==0.0:
        scale_q_value = 127
        scale = [1 / j for j in scale]
        scale_zero_point = 126

        return [scale, scale_zero_point, scale_q_value], []


def inject_preprocess(i, o0, t0, opcodes, tensors, buffers, graph_inputs, inputs, operators, scale, mean):

    # add QUANTIZE
    buffers.append({'offset': 0, 'size': 0})
    tensors.append({'shape': t0['shape'],
        'type': 'UINT8',
        'buffer': len(buffers)-1,
        'name': 'preprocess_data:{}'.format(i),
        'quantization': {'scale': [1.0], 'zero_point': [0], 'details_type': 'NONE', 'quantized_dimension': 0},
        'is_variable': False,
        'has_rank': True})
    graph_inputs[i]['tensor_index'] = len(tensors)-1
    inputs[i] = len(tensors)-1 

    inject_op = {'opcode_index': opcodes.index('QUANTIZE'),
                 'inputs': [len(tensors)-1],
                 'outputs': [o0]}
    operators = operators[:i] + [inject_op] + operators[i:]

    do_mul = isinstance(scale, list) or (scale != 1.0)
    do_add = isinstance(mean, list) or (mean != 0.0)
    output_scale = 1.0
    output_zeropoint = -128
    channel = 1
    mul_offset = 0

    if do_mul:
        mul_offset = 1
        scale_details, shift_details = get_quantize_values(scale, mean)
        scale_factor = max(scale)
        scale = scale_details[0]
        scale_zero_point = scale_details[1]
        scale_q_value = scale_details[2]
        
        if isinstance(scale_q_value, list):
            channel = len(scale_q_value)
            scale_q_value = np.array(scale_q_value).astype(np.uint8) 
        else:
            scale_q_value = np.array([scale_q_value]).astype(np.uint8) 

        scale_q_value = scale_q_value.tolist()

        output_scale, output_zeropoint = get_output_scale(scale_factor)
        output_zeropoint = round(output_zeropoint)

        if isinstance(scale, list):
            scale = scale[0]

        buffers.append({'offset': 0, 'size': 0})
        tensors.append({'shape': t0['shape'],
            'type': 'INT8',
            'buffer': len(buffers)-1,
            'name': 'scale_data:{}'.format(i),
            'quantization': {'scale': [1.0], 'zero_point': [-128], 'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

        operators[i]['outputs'] = [len(tensors)-1]
        buffers.append({'data': scale_q_value, 'offset': 0, 'size': 0})
        tensors.append({'shape': [1,1,1,channel],
            'type': 'INT8',
            'buffer': len(buffers)-1,
            'name': 'mul_data:{}'.format(i),
            'quantization': {'scale': [scale], 'zero_point': [scale_zero_point], 'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})
        inject_op = {'opcode_index': opcodes.index('MUL'),
                    'builtin_options_type': 'MulOptions',
                    'builtin_options': {'fused_activation_function': 'NONE'},
                    'custom_options_format': 'FLEXBUFFERS',
                    'large_custom_options_offset': 0,
                    'large_custom_options_size': 0,
                    'inputs': [len(tensors)-2, len(tensors)-1], 
                    'outputs': [o0]}

        operators = operators[:i+1] + [inject_op] + operators[i+1:]

    if do_add:
        # add ADD
        buffers.append({'offset': 0, 'size': 0})
        tensors.append({'shape': t0['shape'],
            'type': 'INT8',
            'buffer': len(buffers)-1,
            'name': 'shift_data:{}'.format(i),
            'quantization': {'scale': [output_scale], 'zero_point': [output_zeropoint], 'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})

        operators[i+mul_offset]['outputs'] = [len(tensors)-1]
        
        channel = 1
        scale = shift_details[0]
        mean_zero_point = shift_details[1]
        mean_q_value = shift_details[2]

        if isinstance(mean_q_value, list):
            channel = len(mean_q_value)
            mean_q_value = np.array(mean_q_value).astype(np.uint8) 
        else:
            mean_q_value = np.array([mean_q_value]).astype(np.uint8) 

        mean_q_value = mean_q_value.tolist()

        if isinstance(scale, list):
            scale = scale[0]
        
        buffers.append({'data': mean_q_value, 'offset': 0, 'size': 0})
        tensors.append({'shape': [channel],
            'type': 'INT8',
            'buffer': len(buffers)-1,
            'name': 'add_data:{}'.format(i),
            # 'quantization': {'scale': [1/255*(1/scale * 128/127)], 'zero_point': [-128], 'details_type': 'NONE', 'quantized_dimension': 0},
            'quantization': {'scale': [scale], 'zero_point': [mean_zero_point], 'details_type': 'NONE', 'quantized_dimension': 0},
            'is_variable': False,
            'has_rank': True})
        inject_op = {'opcode_index': opcodes.index('ADD'),
                    'builtin_options_type': 'AddOptions',
                    'builtin_options': {'fused_activation_function': 'NONE'},
                    'custom_options_format': 'FLEXBUFFERS',
                    'large_custom_options_offset': 0,
                    'large_custom_options_size': 0,
                    'inputs': [len(tensors)-2, len(tensors)-1],
                    'outputs': [o0]}

        operators = operators[:i+mul_offset+1] + [inject_op] + operators[i+mul_offset+1:]

    return tensors, buffers, graph_inputs, inputs, operators


def preprocess_graphs(tflite_model, scale=1.0, mean=0):
    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'schema.fbs') 
    dir = os.path.dirname(tflite_model)
    if dir == '':
        dir = './'
    jname = os.path.join(dir, os.path.basename(tflite_model).replace('.tflite','.json'))
    cmd = 'flatc -t --strict-json --defaults-json -o {} {} -- {}'.format(dir, schema_path, tflite_model)
    subprocess.run(shlex.split(cmd))
    jname = os.path.join(os.path.dirname(tflite_model), os.path.basename(tflite_model).replace('.tflite','.json'))

    with open(jname) as f:
        graph = json.load(f)

        assert(len(graph['subgraphs']) == 1)
        subgraph = graph['subgraphs'][0]
        operators = subgraph['operators']
        tensors = subgraph['tensors']

        opcodes = [_['builtin_code'] for _ in graph['operator_codes']]
        if not 'QUANTIZE' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 114, 'version': 1, 'builtin_code': 'QUANTIZE'})
        if not 'MUL' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 18, 'version': 2, 'builtin_code': 'MUL'})
        if not 'ADD' in opcodes:
            graph['operator_codes'].append({'deprecated_builtin_code': 0, 'version': 2, 'builtin_code': 'ADD'})

        opcodes = [_['builtin_code'] for _ in graph['operator_codes']]
        inputs = subgraph['inputs']
        
        buffers = graph['buffers']
        if graph['signature_defs']:
            graph_inputs = graph['signature_defs'][0]['inputs'] 
        else:
            input_ops = operators[0]['inputs']
            output_ops = operators[-1]['outputs']
            all_input_ops = []
            all_output_ops = []

            for op in operators:
                all_input_ops += op['inputs']
                all_output_ops += op['outputs']

            input_ops = [_ for _ in all_input_ops if _ not in all_output_ops]
            output_ops = [_ for _ in all_output_ops if _ not in all_input_ops]
            input_names = []
            input_tensors = []
            for io in input_ops:
                if io != -1:
                    t = tensors[io]
                    buf = buffers[t['buffer']]
                    if 'data' not in buf:
                        input_names.append(t['name'])
                        input_tensors.append(io)
            output_names = []
            output_tensors = []
            for io in output_ops:
                if io != -1:
                    t = tensors[io]
                    buf = buffers[t['buffer']]
                    if 'data' not in buf:
                        output_names.append(t['name'])
                        output_tensors.append(io)
            graph_inputs = [{'name': n, 'tensor_index': i} for n, i in zip(input_names, input_tensors)]
            graph_outputs = [{'name': n, 'tensor_index': i} for n, i in zip(output_names, output_tensors)]
            graph['signature_defs'] = [{'inputs':None, 'outputs':None, 'signature_key':'serving_default', 'subgraph_index':0}]
            graph['signature_defs'][0]['inputs'] = graph_inputs
            graph['signature_defs'][0]['outputs'] = graph_outputs

        for i, idx in enumerate(inputs[:1]):
            op = subgraph['operators'][idx]

            for o, o0 in enumerate(op['inputs']):
                t0 = subgraph['tensors'][o0]
                if 'buffer' in t0 and not ('data' in buffers[t0['buffer']]):
                    b0 = buffers[t0['buffer']]
                    q0 = t0['quantization']

                    tensors, buffers, graph_inputs, inputs, operators = inject_preprocess(i, o0, t0, opcodes, tensors, buffers, graph_inputs, inputs, operators, scale, mean)

        graph['signature_defs'][0]['inputs'] = graph_inputs
        graph['buffers'] = buffers
        subgraph['inputs'] = inputs
        subgraph['operators'] = operators
        subgraph['tensors'] = tensors

        jname = os.path.join(os.path.dirname(tflite_model), os.path.basename(tflite_model).replace('.tflite','.pre.json'))
        with open(jname, 'w') as f:
            json.dump(graph, f)
        cmd = 'flatc -b --strict-json --defaults-json -o {} {} {}'.format('./', schema_path, jname)
        subprocess.run(shlex.split(cmd))

def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite")
    parser.add_argument("-s", "--scale", type=float, nargs='+', default=1.0)
    parser.add_argument("-m", "--mean", type=float, nargs='+', default=0.)
    args = parser.parse_args()

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    tmp_tflite = os.path.join(tmp_dir, os.path.basename(args.tflite))
    shutil.copyfile(args.tflite, tmp_tflite)

    preprocess_graphs(tmp_tflite, args.scale, args.mean)
    graphs = glob.glob(os.path.join(tmp_dir, "*.tflite"))
    for src in graphs:
        if src != tmp_tflite:
            dst = os.path.join(os.path.dirname(args.tflite),os.path.basename(src))
            shutil.copyfile(src, dst)
    tmp_dir_obj.cleanup()


def cut():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite")
    parser.add_argument("-c", "--cuts", type=int, nargs='*')
    args = parser.parse_args()

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    tmp_tflite = os.path.join(tmp_dir, os.path.basename(args.tflite))
    shutil.copyfile(args.tflite, tmp_tflite)

    generate_cut_graphs(tmp_tflite, args.cuts)
    graphs = glob.glob(os.path.join(tmp_dir, "*.tflite"))
    for src in graphs:
        if src != tmp_tflite:
            dst = os.path.join(os.path.dirname(args.tflite),os.path.basename(src))
            shutil.copyfile(src, dst)
    tmp_dir_obj.cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite")
    parser.add_argument("dir")
    parser.add_argument("-j", "--join", action='store_true')
    parser.add_argument("-d", "--debug", action='store_true')
    parser.add_argument("-e", "--split-every-op", action='store_true')
    parser.add_argument("-c", "--cuts", type=int, nargs='*')
    parser.add_argument("--vnnx", action='store_true')
    args = parser.parse_args()

    if not args.join:
        generate_split_graphs(args.tflite, args.dir, args.split_every_op, args.cuts, args.vnnx)
    else:
        generate_join_graphs(args.tflite, args.dir, args.debug)

if __name__ == "__main__":
    main()
