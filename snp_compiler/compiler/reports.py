import sys
sys.path.append('.')
import common.internal_representation as internal_representation
from common.hw_config import GRID_OPS, MULTIPLE_INPUT_OPS, GRID_CONFIGS, MAX_GRID_COUNT, FRACTIONAL_BITS, MAX_REDUCE_BUS_WIDTH,\
                             BIAS_FRACTIONAL_BITS, LONG_ENTRY_BITS, AMM_HEIGHT,AMM_WIDTH
from common.debug_flags import DEBUG_CREATE_ORDERING_CONV, DEBUG_GENERATE_PERFORMANCE_REPORT, DEBUG_GENERATE_AMM_ALLOCATION_REPORT, DEBUG_SPREADSHEET_FORMAT_XLSX
from common.enums import DebugFilesFormat
from common.utils import allocated_blocks_to_string
import xlsxwriter
import math
import os
import networkx as nx
import sys
import numpy as np
import pandas as pd

def spreadsheet_write(worksheet,csv_file,line,col,data,format = DebugFilesFormat.XLSX):
    if format == DebugFilesFormat.XLSX:
        worksheet.write(line,col,data)
    else:
        csv_file.write(str(data)+',')
def generate_performance_report_old(performance_report_filename,ir):
    if DEBUG_SPREADSHEET_FORMAT_XLSX:
        performance_report_filename = performance_report_filename+'.xlsx'
        workbook = xlsxwriter.Workbook(performance_report_filename)
        worksheet = workbook.add_worksheet('Performance')
        spreadsheet_format = DebugFilesFormat.XLSX
        csv_file = None
    else:
        performance_report_filename = performance_report_filename+'.csv'
        csv_file = open(performance_report_filename,'w')
        spreadsheet_format = DebugFilesFormat.CSV
        worksheet = None
    # xls_header0 = ['#','name','op_type','input shape','', '', 'output shape','', '', 'kernel size','stride','expected','actual','efficiency',' wloc entries',''   ,'nops'   ,'devide by 8 grids for clocks',''        ,'']

    xls_header0 = ['#','blob', 'name','op_type', 'c in', 'h in','w in', 'c out', 'h out','w out',\
                'kernel','stride','group', 'in_fold_factor_x', 'in_fold_factor_y','out_fold_factor_x',\
                'out_fold_factor_y','y_tiles', 'slices','theor clk','sparsity','sparse clk','pairs','conv cmd only',\
                'rq_waiting_nops', 'nops_end_of_split', 'stall nop', 'AMM nops', 'wloc_all_splits','clks per layer', 'Efficiency']
  
    for col,header in enumerate(xls_header0):
        spreadsheet_write(worksheet,csv_file,0,col,xls_header0[col],format=spreadsheet_format)
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.write('\n')
    current_node_num=0
    current_line = 2
    if len(ir.lexicographical_topological_sorted_graph)==0:
        ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    total_dense_macs = 0
    total_sparse_macs = 0
    total_dense_weights = 0
    total_sparse_weights = 0
    macs_per_clock = 2*MAX_GRID_COUNT*AMM_HEIGHT*AMM_WIDTH

    for node_name in ir.lexicographical_topological_sorted_graph:
        node = ir.graph.nodes()[node_name]
        current_node_num+=1
        current_op_type = node['op_type']
        if (node['op_type']=="Concat") or (node['op_type']=="Max_Pool") :
            continue

        output_tensor      = node['frontend']['output_tensor'] 
        current_op_blob    = node['frontend']['tiling_blob_idx']
        op_type            = node['op_type']
        kernel_size        = node['frontend']['kernel_size']
        stride             = node['frontend']['stride']  
        in_fold_factor_x   = node['frontend']['input_folding_factor_x'] 
        in_fold_factor_y   = node['frontend']['input_folding_factor_y']           
        out_fold_factor_x  = output_tensor.folding_factor_x
        out_fold_factor_y  = output_tensor.folding_factor_y
        y_tiles            = node['frontend']['y_tiles']
        slices             = node['frontend']['x_slices']
        if ('macs_sparsity' in node['frontend']):
            sparsity       = node['frontend']['macs_sparsity']
        else:
            sparsity        = 0
        output_tensor_shape = node['frontend']['output_tensor'].get_original_shape()

        ##### NOP STATISTICS##############################
        wloc_conv_cmd_count = node['backend']['statistics']['wloc_conv_cmd_count'] 
        rq_waiting_nops     = node['backend']['statistics']['rq_waiting_nops']     
        nops_end_of_split   = node['backend']['statistics']['nops_end_of_split']   
        stall_nops          = node['backend']['statistics']['stall_nops']         
        nops_AMM_SYNC       = node['backend']['statistics']['nops_AMM_SYNC']
        wloc_all_splits     = node['backend']['statistics']['wloc_all_splits']
        wloc_pairs          = node['backend']['statistics']['wloc_pairs']
        clks_per_layer      = wloc_all_splits*y_tiles*slices
             


        if op_type=="Add":
                group = output_tensor_shape[1]/2
        elif ("folding_conv" in node_name) or (node['op_type'] == 'Resize') or ("requant" in node_name):
                group = output_tensor_shape[1]
        else:
                group = 1
        if op_type in MULTIPLE_INPUT_OPS:
            input_tensor_shape = node['frontend']['input_tensors'][0].get_original_shape()
        else:
            input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()

        

        theor_clk = int(input_tensor_shape[1]*output_tensor_shape[1]*output_tensor_shape[2]*output_tensor_shape[3]*(kernel_size**2)/(macs_per_clock*group))
        sparse_clk = int(theor_clk*(1-sparsity))
        efficiency = int(100*sparse_clk/clks_per_layer) if clks_per_layer>0 else 0
        
        xls_line = []
        xls_line.append(current_node_num)        
        xls_line.append(current_op_blob)        
        xls_line.append(node_name)
        xls_line.append(op_type)
        xls_line.append(input_tensor_shape[1])
        xls_line.append(input_tensor_shape[2])
        xls_line.append(input_tensor_shape[3])
        xls_line.append(output_tensor_shape[1])
        xls_line.append(output_tensor_shape[2])
        xls_line.append(output_tensor_shape[3])
        xls_line.append(kernel_size)
        xls_line.append(stride)
        xls_line.append(group)
        xls_line.append(in_fold_factor_x)
        xls_line.append(in_fold_factor_y)
        xls_line.append(out_fold_factor_x)
        xls_line.append(out_fold_factor_y)
        xls_line.append(y_tiles)
        xls_line.append(slices)
        xls_line.append(theor_clk)
        xls_line.append(f"{int(sparsity*100)}%")
        xls_line.append(sparse_clk)
        #### NOPS Statistic        
        xls_line.append(wloc_pairs)
        xls_line.append(wloc_conv_cmd_count)
        xls_line.append(rq_waiting_nops)
        xls_line.append(nops_end_of_split)
        xls_line.append(stall_nops)
        xls_line.append(nops_AMM_SYNC)
        xls_line.append(wloc_all_splits)
        xls_line.append(clks_per_layer)
        xls_line.append(f"{int(efficiency)}%")

    # OLD DAN STATISTICS
    #     expected_clocks = int(math.ceil(node['frontend']['sparse_macs']/macs_per_clock))
    #     if 'dense_weights' in node['frontend']:
    #         total_dense_weights+=node['frontend']['dense_weights']
    #         total_sparse_weights+=node['frontend']['sparse_weights']
    #         total_dense_macs+=node['frontend']['dense_macs']
    #     total_sparse_macs+=node['frontend']['sparse_macs']
    #     xls_line.append(expected_clocks)
    #     actual_clocks = node['backend']['total_rqloc_clocks']
    #     xls_line.append(actual_clocks)
    #     efficiency = expected_clocks/actual_clocks*100
    #     eff_str = ('%3.4f%%' % efficiency)
    #     xls_line.append(eff_str)
    #     total_first_wloc_entries = node['backend']['total_first_wloc_entries']
    #     xls_line.append(total_first_wloc_entries)
    #     total_minimal_oc_clocks_inserted_nops = node['backend']['total_minimal_oc_clocks_inserted_nops']
    #     xls_line.append(total_minimal_oc_clocks_inserted_nops)
    #     total_rq_busy_inserted_nops = node['backend']['total_rq_busy_inserted_nops']
    #     xls_line.append(total_rq_busy_inserted_nops)
    #     total_write_state_inserted_nops = node['backend']['total_write_state_inserted_nops']
    #     xls_line.append(total_write_state_inserted_nops)
    #     total_ic_group_wait_inserted_nops = node['backend']['total_ic_group_wait_inserted_nops']
    #     xls_line.append(total_ic_group_wait_inserted_nops)
    #     total_ddr_write_inserted_nops = node['backend']['total_ddr_write_inserted_nops']
    #     xls_line.append(total_ddr_write_inserted_nops)


        for col,val in enumerate(xls_line):
            spreadsheet_write(worksheet,csv_file,current_line,col,val,format=spreadsheet_format)
        if spreadsheet_format==DebugFilesFormat.CSV:
            csv_file.write('\n')
        current_line+=1
    
    # #weights_sparsity=100*(1-total_sparse_weights/total_dense_weights)
    # if (total_dense_weights == 0):
    #     weights_sparsity = 0
    # else :
    #     weights_sparsity=100*(1-total_sparse_weights/total_dense_weights)

    # #macs_sparsity=100*(1-total_sparse_macs/total_dense_macs)
    # if (total_dense_macs == 0):
    #     macs_sparsity = 0
    # else :
    #     macs_sparsity=100*(1-total_sparse_macs/total_dense_macs)

    # spreadsheet_write(worksheet,csv_file,current_line+1,1,'Weights Sparsity:',format=spreadsheet_format)
    # spreadsheet_write(worksheet,csv_file,current_line+1,2,weights_sparsity,format=spreadsheet_format)
    # if spreadsheet_format==DebugFilesFormat.CSV:
    #     csv_file.write('\n')
    # spreadsheet_write(worksheet,csv_file,current_line+2,1,'MACS Sparsity:',format=spreadsheet_format)
    # spreadsheet_write(worksheet,csv_file,current_line+2,2,macs_sparsity,format=spreadsheet_format)
    # if spreadsheet_format==DebugFilesFormat.CSV:
    #     csv_file.write('\n')
    # spreadsheet_write(worksheet,csv_file,current_line+3,1,'Total sparse MACS:',format=spreadsheet_format)
    # spreadsheet_write(worksheet,csv_file,current_line+3,2,total_sparse_macs,format=spreadsheet_format)
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.write('\n')
        csv_file.close()
    else:
        workbook.close()

def generate_performance_report(performance_report_filename, ir):
    """
    Generates the same performance report as generate_performance_report,
    but uses a pandas DataFrame for data handling and output.
    Adds a summary line at the end for theor_clk to clks per layer and overall efficiency.
    """
    xls_header0 = ['#','blob', 'name','op_type', 'c in', 'h in','w in', 'c out', 'h out','w out',
                   'kernel','stride','group', 'in_fold_factor_x', 'in_fold_factor_y','out_fold_factor_x',
                   'out_fold_factor_y','y_tiles', 'slices','theor clk','sparsity','sparse clk','theor_weights', 'sparse_weights','pairs','pairing savings','conv cmd only',
                   'rq_waiting_nops', 'nops_end_of_split', 'stall nop', 'AMM nops', 'wloc_all_splits','clks per layer', 'Efficiency']

    data_rows = []
    current_node_num = 0
    if len(ir.lexicographical_topological_sorted_graph) == 0:
        ir.lexicographical_topological_sorted_graph = list(nx.lexicographical_topological_sort(ir.graph))
    macs_per_clock = 2 * MAX_GRID_COUNT * AMM_HEIGHT * AMM_WIDTH

    for node_name in ir.lexicographical_topological_sorted_graph:
        node = ir.graph.nodes()[node_name]
        current_node_num += 1
        current_op_type = node['op_type']
        if current_op_type in ["Concat", "Max_Pool"]:
            continue

        output_tensor = node['frontend']['output_tensor']
        current_op_blob = node['frontend']['tiling_blob_idx']
        op_type = node['op_type']
        kernel_size = node['frontend']['kernel_size']
        stride = node['frontend']['stride']
        in_fold_factor_x = node['frontend']['input_folding_factor_x']
        in_fold_factor_y = node['frontend']['input_folding_factor_y']
        out_fold_factor_x = output_tensor.folding_factor_x
        out_fold_factor_y = output_tensor.folding_factor_y
        y_tiles = node['frontend']['y_tiles']
        slices = node['frontend']['x_slices']
        #if this is a folding_x node then number of slices needs to be even
        if ('force_folding_x' in node['frontend']) and (slices % 2 != 0):
            slices += 1
        sparsity = node['frontend'].get('macs_sparsity', 0)
        output_tensor_shape = output_tensor.get_original_shape()

        wloc_conv_cmd_count = node['backend']['statistics']['wloc_conv_cmd_count']
        rq_waiting_nops = node['backend']['statistics']['rq_waiting_nops']
        nops_end_of_split = node['backend']['statistics']['nops_end_of_split']
        stall_nops = node['backend']['statistics']['stall_nops']
        nops_AMM_SYNC = node['backend']['statistics']['nops_AMM_SYNC']
        wloc_all_splits = node['backend']['statistics']['wloc_all_splits']
        wloc_pairs = node['backend']['statistics']['wloc_pairs']
        clks_per_layer = wloc_all_splits * y_tiles * slices

        input_tensor_shape = node['frontend']['input_tensors'][0].get_original_shape() if op_type in MULTIPLE_INPUT_OPS \
            else node['frontend']['input_tensor'].get_original_shape()

        if op_type == "Add":
            group = input_tensor_shape[1] // 2
        elif (op_type == "Identity") or (op_type == "MaxPool") or (op_type == "Resize"):
            group = input_tensor_shape[1]
        elif (op_type == "Identity") or ("fold_x" in node_name) or ("requant" in node_name) or ("SPLIT" in node_name) or ("identity" in node_name):
            group = input_tensor_shape[1]
            op_type = "Identity"
        else:
            group = 1
        
        theor_clk = int(input_tensor_shape[1] * output_tensor_shape[1] * output_tensor_shape[2] * output_tensor_shape[3] * (kernel_size ** 2) / (macs_per_clock * group))
        sparse_clk = int(theor_clk * (1 - sparsity))
        theor_weights = int(input_tensor_shape[1] * output_tensor_shape[1] * (kernel_size ** 2) / group)
        sparse_weights = int(theor_weights * (1 - sparsity))
        pairing_savings = wloc_pairs / sparse_weights if sparse_weights > 0 else 0
        efficiency = sparse_clk * (1 - pairing_savings) / clks_per_layer if clks_per_layer > 0 else 0

        row = [
            current_node_num,
            current_op_blob,
            node_name,
            op_type,
            input_tensor_shape[1],
            input_tensor_shape[2],
            input_tensor_shape[3],
            output_tensor_shape[1],
            output_tensor_shape[2],
            output_tensor_shape[3],
            kernel_size,
            stride,
            group,
            in_fold_factor_x,
            in_fold_factor_y,
            out_fold_factor_x,
            out_fold_factor_y,
            y_tiles,
            slices,
            theor_clk,
            f"{int(sparsity * 100)}%",
            sparse_clk,
            theor_weights,
            sparse_weights,
            wloc_pairs,
            f"{int(pairing_savings * 100)}%",
            wloc_conv_cmd_count,
            rq_waiting_nops,
            nops_end_of_split,
            stall_nops,
            nops_AMM_SYNC,
            wloc_all_splits,
            clks_per_layer,
            f"{int(efficiency * 100)}%"
        ]
        data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=xls_header0)

    # Summary calculation for theor_clk to clks per layer
    theor_clk_sum = df['theor clk'].sum()
    sparse_clk_sum = df['sparse clk'].sum()
    total_sparsity = int(100 * (1 - sparse_clk_sum / theor_clk_sum)) if theor_clk_sum > 0 else 0
    theor_weights_sum = df['theor_weights'].sum()
    sparse_weights_sum = df['sparse_weights'].sum()
    pairs_sum = df['pairs'].sum()
    pairing_savings_total = pairs_sum / sparse_weights_sum if sparse_weights_sum > 0 else 0
    conv_cmd_sum = df['conv cmd only'].sum()
    rq_waiting_nops_sum = df['rq_waiting_nops'].sum()
    nops_end_of_split_sum = df['nops_end_of_split'].sum()
    stall_nop_sum = df['stall nop'].sum()
    amm_nops_sum = df['AMM nops'].sum()
    wloc_all_splits_sum = df['wloc_all_splits'].sum()
    clks_per_layer_sum = df['clks per layer'].sum()
    # Overall efficiency
    overall_efficiency = int(100 * sparse_clk_sum * (1-pairing_savings_total) / clks_per_layer_sum) if clks_per_layer_sum > 0 else 0

    # Insert blank line before summary
    blank_row = [''] * len(xls_header0)
    df.loc[len(df)] = blank_row

    summary_row = ['Total'] + ([''] * 18) + [
        theor_clk_sum,
        f"{total_sparsity}%",
        sparse_clk_sum,
        theor_weights_sum,
        sparse_weights_sum,
        pairs_sum,
        f"{int(pairing_savings_total*100)}%",
        conv_cmd_sum,
        rq_waiting_nops_sum,
        nops_end_of_split_sum,
        stall_nop_sum,
        amm_nops_sum,
        wloc_all_splits_sum,
        clks_per_layer_sum,
        f"{overall_efficiency}%"
    ]
    df.loc[len(df)] = summary_row

    if DEBUG_SPREADSHEET_FORMAT_XLSX:
        output_file = performance_report_filename + '.xlsx'
        df.to_excel(output_file, index=False)
    else:
        output_file = performance_report_filename + '.csv'
        df.to_csv(output_file, index=False, header=True)

def generate_qparams_report(qparams_report_filename:str ,ir: internal_representation.IR):
    if DEBUG_SPREADSHEET_FORMAT_XLSX:
        qparams_report_filename = qparams_report_filename+'.xlsx'
        workbook = xlsxwriter.Workbook(qparams_report_filename)
        worksheet = workbook.add_worksheet('QParams')
        spreadsheet_format = DebugFilesFormat.XLSX
        csv_file = None
    else:
        qparams_report_filename = qparams_report_filename+'.csv'
        csv_file = open(qparams_report_filename,'w')
        spreadsheet_format = DebugFilesFormat.CSV
        worksheet = None
    xls_header0 = ['#','blob','name','op_type','input shape','', '', ''      ,''      ,''        ,'output shape','', '', ''      ,''      ,''        ,'x_wrapping','kernel size','stride','y_tiles','ic_splits','oc_splits','folding_x','folding_y','unfolding_x','unfolding_y','Inputs',''  ,'output',''   ,'weights sparsity']
    xls_header1 = ['', '',    '',    '',       'channels',   'h','w','x_fold','y_fold','x_slices','channels',    'h','w','x_fold','y_fold','x_slices','',          '',           '',      '',       '',         '',         '',         '',         '',           '',           'scale', 'zp','scale' ,'zp', '']
    for col,header in enumerate(xls_header0):
        spreadsheet_write(worksheet,csv_file,0,col,xls_header0[col],format=spreadsheet_format)
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.write('\n')

    for col,header in enumerate(xls_header0):
        spreadsheet_write(worksheet,csv_file,1,col,xls_header1[col],format=spreadsheet_format)
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.write('\n')
    current_node_num=0
    current_line = 2
    sorted_graph = ir.lexicographical_topological_sorted_graph
    for node_name in sorted_graph:
        node = ir.graph.nodes()[node_name]
        current_node_num+=1
        current_op_type = node['op_type']
        xls_line = []
        xls_line.append(current_node_num)
        if 'tiling_blob_idx' in node['frontend']:
            xls_line.append(node['frontend']['tiling_blob_idx'])
        else:
            xls_line.append('')
        xls_line.append(node_name)
        op_type = node['op_type']
        xls_line.append(op_type)
        if op_type in MULTIPLE_INPUT_OPS:
            input_tensor = node['frontend']['input_tensors'][0]
            input_tensor_shape = input_tensor.get_original_shape()
        else:
            input_tensor = node['frontend']['input_tensor']
            input_tensor_shape = input_tensor.get_original_shape()
        output_tensor = node['frontend']['output_tensor']
        output_tensor_shape = output_tensor.get_original_shape()
        xls_line.append(input_tensor_shape[1])
        xls_line.append(input_tensor_shape[2])
        xls_line.append(input_tensor_shape[3])
        # We take the nodes folding factor since its the actual folding factor
        xls_line.append(node['frontend']['input_folding_factor_x'])
        xls_line.append(node['frontend']['input_folding_factor_y'])
        xls_line.append(input_tensor.x_slices)
        xls_line.append(output_tensor_shape[1])
        xls_line.append(output_tensor_shape[2])
        xls_line.append(output_tensor_shape[3])
        xls_line.append(output_tensor.folding_factor_x)
        xls_line.append(output_tensor.folding_factor_y)
        xls_line.append(output_tensor.x_slices)
        if 'x_wrapping' in node['frontend']:
            xls_line.append(node['frontend']['x_wrapping'])
        else:
            xls_line.append('')
        if op_type in GRID_OPS:
            xls_line.append(node['frontend']['kernel_size'])
            xls_line.append(node['frontend']['stride'])
            xls_line.append(node['frontend']['y_tiles'])
            xls_line.append(node['backend']['ic_splits'])
            xls_line.append(node['backend']['oc_splits'])
            xls_line.append('force_folding_x' in node['frontend'])
            xls_line.append('force_folding_y' in node['frontend'])
            xls_line.append('force_unfolding_x' in node['frontend'])
            xls_line.append('force_unfolding_y' in node['frontend'])
        else:
            xls_line.extend(['',''])
            xls_line.append(node['frontend']['y_tiles'])
            xls_line.append(node['backend']['ic_splits'])
            xls_line.append(node['backend']['oc_splits'])
            xls_line.append('force_folding_x' in node['frontend'])
            xls_line.append('force_folding_y' in node['frontend'])
            xls_line.append('force_unfolding_x' in node['frontend'])
            xls_line.append('force_unfolding_y' in node['frontend'])

        input_tensors=[]
        inputs_scales=[]
        inputs_zps=[]
        for input_name in node['inputs']:
            input_tensors.append(ir.tensors[input_name])
        if op_type == 'Conv': #Remove weights and biases from report
            #if this was an Average Pool it will have fake weights and biases
            if len(input_tensors)==1:
                pass
            else:    

                del input_tensors[2]
                del input_tensors[1]
        elif op_type == 'Resize':
            del input_tensors[1]
        elif ('STRIDEDSLICE' in node['name']):
            del input_tensors[3]
            del input_tensors[2]
            del input_tensors[1]
        idx = 0

        ###### Alex patch, to compile 3x3 ###############
        #if node['name'] == 'CONV2D_0_identity': 
        #    node['frontend']['input_tensor_scale'] = node['frontend']['input_tensor'].scale

        for input_tensor in input_tensors:
            if (len(input_tensors) == 1):
                inputs_scales.append(float(node['frontend']['input_tensor_scale']))
                inputs_zps.append(int(node['frontend']['input_tensor_zp']))
            else:
                inputs_scales.append(float(node['frontend']['input_tensors_scale'][idx]))
                inputs_zps.append(int(node['frontend']['input_tensors_zp'][idx]))
            idx += 1
        xls_line.append(str(inputs_scales).replace(',',';'))
        xls_line.append(str(inputs_zps).replace(',',';'))
        output_name = node['outputs'][0]
        output_tensor = ir.tensors[output_name]
        if ('activation_silu' in node['attributes']) and (node['attributes']['activation_silu'] != None):
            output_scale = node['attributes']['activation_silu']['output_scale'][0]
            output_zp = node['attributes']['activation_silu']['output_zp'][0]
        else:
            output_scale = output_tensor.scale
            output_zp = output_tensor.zero_point
        xls_line.append(str(output_scale))
        xls_line.append(str(output_zp))
        if 'weights_sparsity' in node['frontend']:
            xls_line.append(node['frontend']['weights_sparsity'])
        else:
            xls_line.append('')

        for col,val in enumerate(xls_line):
            spreadsheet_write(worksheet,csv_file,current_line,col,val,format=spreadsheet_format)
        if spreadsheet_format==DebugFilesFormat.CSV:
            csv_file.write('\n')
        current_line+=1
            
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.write('\n')
        csv_file.close()
    else:
        workbook.close()
def generate_folding_report(folding_report_filename:str ,ir: internal_representation.IR):
    if DEBUG_SPREADSHEET_FORMAT_XLSX:
        folding_report_filename = folding_report_filename+'.xlsx'
        workbook = xlsxwriter.Workbook(folding_report_filename)
        worksheet = workbook.add_worksheet('Folding')
        spreadsheet_format = DebugFilesFormat.XLSX
        csv_file = None
    else:
        folding_report_filename = folding_report_filename+'.csv'
        csv_file = open(folding_report_filename,'w')
        spreadsheet_format = DebugFilesFormat.CSV
        worksheet = None
    xls_header0 = ['#','blob','name','op_type','channels',   'h','w','x_fold','y_fold','channels',    'h','w','x_fold','y_fold','kernel size','stride','y_tiles','folding_x','folding_y','unfolding_y']
    for col,header in enumerate(xls_header0):
        spreadsheet_write(worksheet,csv_file,0,col,xls_header0[col],format=spreadsheet_format)
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.write('\n')

    current_node_num=0
    current_line = 1
    sorted_graph = ir.graph.nodes
    for node_name in sorted_graph:
        node = ir.graph.nodes()[node_name]
        current_node_num+=1
        current_op_type = node['op_type']
        xls_line = []
        xls_line.append(current_node_num)
        if 'tiling_blob_idx' in node['frontend']:
            xls_line.append(node['frontend']['tiling_blob_idx'])
        else:
            xls_line.append('')
        xls_line.append(node_name)
        op_type = node['op_type']
        xls_line.append(op_type)
        if op_type in MULTIPLE_INPUT_OPS:
            input_tensor = node['frontend']['input_tensors'][0]
            input_tensor_shape = input_tensor.get_original_shape()
        else:
            input_tensor = node['frontend']['input_tensor']
            input_tensor_shape = input_tensor.get_original_shape()
        output_tensor = node['frontend']['output_tensor']
        output_tensor_shape = output_tensor.get_original_shape()
        xls_line.append(input_tensor_shape[1])
        xls_line.append(input_tensor_shape[2])
        xls_line.append(input_tensor_shape[3])
        xls_line.append(input_tensor.folding_factor_x)
        xls_line.append(node['frontend']['input_folding_factor_y']) # We take the nodes folding factor as if its a y folding/unfolding the not will contain the actual folding factor
        xls_line.append(output_tensor_shape[1])
        xls_line.append(output_tensor_shape[2])
        xls_line.append(output_tensor_shape[3])
        xls_line.append(output_tensor.folding_factor_x)
        xls_line.append(output_tensor.folding_factor_y)
        if op_type == 'Conv':
            xls_line.append(node['frontend']['kernel_size'])
            xls_line.append(node['frontend']['stride'])
            xls_line.append(node['frontend']['y_tiles'])
            xls_line.append('force_folding_x' in node['frontend'])
            xls_line.append('force_folding_y' in node['frontend'])
            xls_line.append('force_unfolding_y' in node['frontend'])
        else:
            xls_line.extend(['',''])
            xls_line.append(node['frontend']['y_tiles'])
            xls_line.append('force_folding_x' in node['frontend'])
            xls_line.append('force_folding_y' in node['frontend'])
            xls_line.append('force_unfolding_y' in node['frontend'])

        for col,val in enumerate(xls_line):
            spreadsheet_write(worksheet,csv_file,current_line,col,val,format=spreadsheet_format)
        if spreadsheet_format==DebugFilesFormat.CSV:
            csv_file.write('\n')
        current_line+=1
            
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.write('\n')
        csv_file.close()
    else:
        workbook.close()
    
def generate_ammalloc_report(ammalloc_report_filename,ir):
    if DEBUG_SPREADSHEET_FORMAT_XLSX:
        ammalloc_report_filename = ammalloc_report_filename+'.xlsx'
        workbook = xlsxwriter.Workbook(ammalloc_report_filename)
        worksheet = workbook.add_worksheet('AMM Allocation')
        spreadsheet_format = DebugFilesFormat.XLSX
        csv_file = None
    else:
        ammalloc_report_filename = ammalloc_report_filename+'.csv'
        csv_file = open(ammalloc_report_filename,'w')
        spreadsheet_format = DebugFilesFormat.CSV
        worksheet = None

    xls_header0 = ['#',    '',    '',       'name','op_type','input shape','', '', 'output shape','', '', 'kernel size','stride','folding_conv','allocated blocks','',          'deallocated tensors','',                'amm_utilization','tables read(bytes)',''     ,''        ,'tensor read','tensor write']
    xls_header1 = ['blob', 'tile','x_slice','',    '',       'channels',   'h','w','channels',    'h','w','',           '',      '',            'for input',       'for output','after output alloc' ,'after input read',''               ,'wloc'              ,'rqloc','rqtables','bytes'      ,'bytes']
    for col,header in enumerate(xls_header0):
        spreadsheet_write(worksheet,csv_file,0,col,xls_header0[col],format=spreadsheet_format)
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.write('\n')
    for col,header in enumerate(xls_header0):
        spreadsheet_write(worksheet,csv_file,1,col,xls_header1[col],format=spreadsheet_format)
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.write('\n')

    current_node_num=0
    current_line = 2
    tiling_blobs=ir.tiling_blobs
    for blob_idx,tiling_blob in enumerate(tiling_blobs.values()):
        for current_y_tile in range(tiling_blob.y_tiles):
            for node_idx,current_node_name in enumerate(tiling_blob.nodes_in_blob):
                node = ir.graph.nodes()[current_node_name]
                current_node_num+=1
                current_op_type = node['op_type']
                num_slices = max(node['frontend']['x_slices'], node['frontend']['output_tensor'].x_slices)
                for current_x_slice in range(num_slices):
                    xls_line = []
                    if current_y_tile==0 and node_idx==0:
                        xls_line.append(blob_idx)
                    else:
                        xls_line.append('')
                    if node_idx==0 and current_x_slice==0:
                        xls_line.append(current_y_tile)
                    else:
                        xls_line.append('')
                    xls_line.append(current_x_slice)
                    xls_line.append(current_node_name)
                    op_type = node['op_type']
                    xls_line.append(op_type)
                    if op_type in MULTIPLE_INPUT_OPS:
                        input_tensor_shape = node['frontend']['input_tensors'][0].get_original_shape()
                    else:
                        input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
                    output_tensor_shape = node['frontend']['output_tensor'].get_original_shape()
                    xls_line.append(input_tensor_shape[1])
                    xls_line.append(input_tensor_shape[2])
                    xls_line.append(input_tensor_shape[3])
                    xls_line.append(output_tensor_shape[1])
                    xls_line.append(output_tensor_shape[2])
                    xls_line.append(output_tensor_shape[3])
                    if current_op_type in GRID_OPS:
                        xls_line.append(node['frontend']['kernel_size'])
                        xls_line.append(node['frontend']['stride'])
                    else:
                        xls_line.append('')
                        xls_line.append('')

                    xls_line.append('force_folding_x' in node['frontend'])
                    op_type = node['op_type']
                    if ('force_unfolding_x' in node['frontend']):
                        input_x_slice = int(current_x_slice / 2)
                    else:
                        if ('input_tensors' in node['frontend']):
                            input_tensor = node['frontend']['input_tensors'][0]
                        else:
                            input_tensor = node['frontend']['input_tensor']
                        input_x_slice = current_x_slice // input_tensor.num_packed_xslices
                    if op_type in MULTIPLE_INPUT_OPS:
                        blocks_allocated_for_input=''
                        for input_idx,input_tensor in enumerate(node['frontend']['input_tensors']):
                            blocks_allocated_for_input += 'i'+str(input_idx)+'e:'+allocated_blocks_to_string(node['backend']['allocated_amm_blocks_for_input_even_grid'][current_y_tile][input_x_slice][input_idx][0]) # indexs are tile#,Input#,amm#
                            blocks_allocated_for_input += ',o:'+allocated_blocks_to_string(node['backend']['allocated_amm_blocks_for_input_odd_grid'][current_y_tile][input_x_slice][input_idx][0])
                    else:
                        blocks_allocated_for_input = 'i0e:'+allocated_blocks_to_string(node['backend']['allocated_amm_blocks_for_input_even_grid'][current_y_tile][input_x_slice][0][0])
                        blocks_allocated_for_input += ',o:'+allocated_blocks_to_string(node['backend']['allocated_amm_blocks_for_input_odd_grid'][current_y_tile][input_x_slice][0][0])
                    blocks_allocated_for_input=blocks_allocated_for_input.replace(',',';')
                    xls_line.append(blocks_allocated_for_input)
                    if ('force_folding_x' in node['frontend']):
                        output_x_slice = int(current_x_slice / 2)
                    else:
                        output_x_slice = current_x_slice
                    blocks_allocated_for_output = 'e:'+allocated_blocks_to_string(node['backend']['allocated_amm_blocks_for_output_even_grid'][current_y_tile][output_x_slice][0]) # indexs are tile#,amm#
                    blocks_allocated_for_output += ',o:'+allocated_blocks_to_string(node['backend']['allocated_amm_blocks_for_output_odd_grid'][current_y_tile][output_x_slice][0])
                    blocks_allocated_for_output=blocks_allocated_for_output.replace(',',';')
                    xls_line.append(blocks_allocated_for_output)
                    if 'tensors_for_deallocation_after_output_allocation_report' in node['backend']:
                        tensors_deallocated_struct = node['backend']['tensors_for_deallocation_after_output_allocation_report']
                        if (current_y_tile, current_x_slice) in tensors_deallocated_struct.tensors_list:
                            current_tile_dealocated_tensors = tensors_deallocated_struct.get_tensor_names_dict()[(current_y_tile, current_x_slice)]
                            tensors_deallocated_after_output_allocation = ';'.join(current_tile_dealocated_tensors)
                        else: 
                            tensors_deallocated_after_output_allocation = ''
                    else:
                        tensors_deallocated_after_output_allocation = ''
                    tensors_deallocated_after_output_allocation = tensors_deallocated_after_output_allocation.replace(',',';')
                    xls_line.append(tensors_deallocated_after_output_allocation)
                    if 'tensors_for_deallocation_after_ddr_read_allocation_report' in node['backend']:
                        tensors_deallocated_struct = node['backend']['tensors_for_deallocation_after_ddr_read_allocation_report']
                        if (current_y_tile, current_x_slice) in tensors_deallocated_struct.tensors_list:
                            current_tile_dealocated_tensors = tensors_deallocated_struct.get_tensor_names_dict()[(current_y_tile,current_x_slice)]
                            tensors_deallocated_after_ddr_read = ';'.join(current_tile_dealocated_tensors)
                        else: 
                            tensors_deallocated_after_ddr_read = ''
                    else:
                        tensors_deallocated_after_ddr_read = ''
                    tensors_deallocated_after_ddr_read = tensors_deallocated_after_ddr_read.replace(',',';')
                    xls_line.append(tensors_deallocated_after_ddr_read)

                    if 'post_op_mem_utilization' in node['backend']:
                        xls_line.append(node['backend']['post_op_mem_utilization'])
                    else:
                        xls_line.append('')

                    if 'total_wloc_tables_bytes' in node['backend']:
                        if current_y_tile in node['backend']['total_wloc_tables_bytes']:
                            xls_line.append(node['backend']['total_wloc_tables_bytes'][current_y_tile])
                    else:
                        xls_line.append('')
                    if 'total_rqloc_table_bytes' in node['backend']:
                        if current_y_tile in node['backend']['total_rqloc_table_bytes']:
                            xls_line.append(node['backend']['total_rqloc_table_bytes'][current_y_tile])
                    else:
                        xls_line.append('')
                    if 'total_rqparams_table_bytes' in node['backend']:
                        if current_y_tile in node['backend']['total_rqparams_table_bytes']:
                            xls_line.append(node['backend']['total_rqparams_table_bytes'][current_y_tile])
                    else:
                        xls_line.append('')
                    if 'total_tensor_read_bytes' in node['backend']:
                        if (current_y_tile, current_x_slice) in node['backend']['total_tensor_read_bytes']:
                            xls_line.append(node['backend']['total_tensor_read_bytes'][(current_y_tile, current_x_slice)])
                    else:
                        xls_line.append('')

                    if 'total_tensor_write_bytes' in node['backend']:
                        if (current_y_tile, current_x_slice) in node['backend']['total_tensor_write_bytes']:
                            xls_line.append(node['backend']['total_tensor_write_bytes'][(current_y_tile, current_x_slice)])
                    else:
                        xls_line.append('')


                    for col,val in enumerate(xls_line):
                        spreadsheet_write(worksheet,csv_file,current_line,col,val,format=spreadsheet_format)
                    current_line+=1
                    if spreadsheet_format==DebugFilesFormat.CSV:
                        csv_file.write('\n')

            
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.close()
    else:
        workbook.close()

def generate_wloc_balancing_report(wloc_balancing_report_filename,ir):
    if DEBUG_SPREADSHEET_FORMAT_XLSX:
        wloc_balancing_report_filename = wloc_balancing_report_filename+'.xlsx'
        workbook = xlsxwriter.Workbook(wloc_balancing_report_filename)
        worksheet = workbook.add_worksheet('Wloc balancing')
        spreadsheet_format = DebugFilesFormat.XLSX
        csv_file = None
    else:
        wloc_balancing_report_filename = wloc_balancing_report_filename+'.csv'
        csv_file = open(wloc_balancing_report_filename,'w')
        spreadsheet_format = DebugFilesFormat.CSV
        worksheet = None

    xls_header0 = ['#','name','op_type','input shape','', '', 'output shape','', '', 'kernel size','stride','initial']
    xls_header1 = ['', '',    '',       'channels',   'h','w','channels',    'h','w','',           '',      'wloc']
    for col,header in enumerate(xls_header0):
        spreadsheet_write(worksheet,csv_file,0,col,xls_header0[col],format=spreadsheet_format)
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.write('\n')
    for col,header in enumerate(xls_header0):
        spreadsheet_write(worksheet,csv_file,1,col,xls_header1[col],format=spreadsheet_format)
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.write('\n')
    current_node_num=0
    current_line = 2
    for node_name in ir.lexicographical_topological_sorted_graph:
        node = ir.graph.nodes()[node_name]
        current_node_num+=1
        current_op_type = node['op_type']
        if current_op_type in GRID_OPS:
            xls_line = []
            xls_line.append(current_node_num)
            xls_line.append(node_name)
            op_type = node['op_type']
            xls_line.append(op_type)
            if op_type in MULTIPLE_INPUT_OPS:
                input_tensor_shape = node['frontend']['input_tensors'][0].get_original_shape()
            else:
                input_tensor_shape = node['frontend']['input_tensor'].get_original_shape()
            output_tensor_shape = node['frontend']['output_tensor'].get_original_shape()
            xls_line.append(input_tensor_shape[1])
            xls_line.append(input_tensor_shape[2])
            xls_line.append(input_tensor_shape[3])
            xls_line.append(output_tensor_shape[1])
            xls_line.append(output_tensor_shape[2])
            xls_line.append(output_tensor_shape[3])
            xls_line.append(node['frontend']['kernel_size'])
            xls_line.append(node['frontend']['stride'])
            # initial_max_wloc_128bit_entries = node['backend']['initial_max_wloc_128bit_entries']
            # xls_line.append(initial_max_wloc_128bit_entries)
            # total_actual_wloc_entries = node['backend']['total_actual_wloc_entries']
            # xls_line.append(total_actual_wloc_entries)
            # wloc_inflation_factor = node['backend']['wloc_inflation_factor']
            # xls_line.append(wloc_inflation_factor)
            # balanced_max_wloc_128bit_entries = node['backend']['balanced_max_wloc_128bit_entries']
            # xls_line.append(balanced_max_wloc_128bit_entries)
            # per_split_wloc_size = node['backend']['per_split_wloc_size']
            # xls_line.append(str(per_split_wloc_size))
            for col,val in enumerate(xls_line):
                spreadsheet_write(worksheet,csv_file,current_line,col,val,format=spreadsheet_format)
            if spreadsheet_format==DebugFilesFormat.CSV:
                csv_file.write('\n')
            current_line+=1
    if spreadsheet_format==DebugFilesFormat.CSV:
        csv_file.close()
    else:
        workbook.close()

def main ():
    if sys.platform == 'linux':
        output_dir = '/home/neuronix/dan/tsnp_output/'
    else:
        output_dir = 'C:/Users/dshir/Documents/tsnp_output/'
    models = ['yolov5l_nx_s94']

    for model_name in models:
        if DEBUG_CREATE_ORDERING_CONV:
            model_name=model_name+'_with_reordering'
        compiler_output_dir = output_dir+'/'+model_name+'/'
        compiler_output_ir_dir = output_dir+'/nx_ir/'
        ir = internal_representation.IR('')
        ir_filename = compiler_output_ir_dir+model_name+'_numsim.nxir'
        ir = ir.load(ir_filename)
        if DEBUG_GENERATE_PERFORMANCE_REPORT:
            performance_report_filename = compiler_output_dir+model_name+'_perf'
            generate_performance_report(performance_report_filename,ir)
            wloc_balancing_report_filename = compiler_output_dir+model_name+'_wloc_balancing'
            generate_wloc_balancing_report(wloc_balancing_report_filename,ir)
        if DEBUG_GENERATE_AMM_ALLOCATION_REPORT:
            amm_allocation_report_filename = compiler_output_dir+model_name+'_ammalloc'
            generate_ammalloc_report(amm_allocation_report_filename,ir)

if __name__ == "__main__":

    main()