import sys
sys.path.append('.')
import common.internal_representation as internal_representation
from common.hw_config import GRID_OPS, MULTIPLE_INPUT_OPS, MAX_GRID_COUNT, AMM_WIDTH, OpType
from common.debug_flags import DEBUG_GENERATE_PERFORMANCE_REPORT, DEBUG_GENERATE_AMM_ALLOCATION_REPORT
from common.enums import DebugFilesFormat
from common.utils import allocated_blocks_to_string, packed_lane_of_logical, is_packed_xslices
import networkx as nx
import sys
import pandas as pd

def spreadsheet_write(worksheet,csv_file,line,col,data,format = DebugFilesFormat.XLSX):
    if format == DebugFilesFormat.XLSX:
        worksheet.write(line,col,data)
    else:
        csv_file.write(str(data)+',')


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
    macs_per_clock = 2 * MAX_GRID_COUNT * ir.amm_height * AMM_WIDTH

    for node_name in ir.lexicographical_topological_sorted_graph:
        node = ir.graph.nodes()[node_name]
        current_node_num += 1
        current_op_type = node['op_type']
        if current_op_type in [OpType.CONCAT, OpType.MAXPOOL, OpType.SYNC]:
            continue
        if (node['op_type'] == OpType.RESIZE) and ('_y' in node['name']):
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

        if op_type == OpType.ADD:
            group = input_tensor_shape[1] // 2
        elif (op_type == OpType.IDENTITY) or ("fold_x" in node_name) or ("requant" in node_name) or ("SPLIT" in node_name) or ("identity" in node_name):
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


    output_file = performance_report_filename + '.csv'
    df.to_csv(output_file, index=False, header=True)

def generate_qparams_report(qparams_report_filename:str ,ir: internal_representation.IR):

    qparams_report_filename = qparams_report_filename+'.csv'
    csv_file = open(qparams_report_filename,'w')
    spreadsheet_format = DebugFilesFormat.CSV
    worksheet = None

    xls_header0 = ['#','blob','name','op_type','input shape','', '', ''      ,''      ,''        ,'output shape','', '', ''      ,''      ,''        ,'x_wrapping','kernel size','stride','y_tiles','ic_splits','oc_splits','folding_x','folding_y','unfolding_x','unfolding_y','Inputs',''  ,'output',''   ,'weights sparsity','write_align','h_tiles','xslices_per_htile','output_xslices_per_htile','padding']
    xls_header1 = ['', '',    '',    '',       'channels',   'h','w','x_fold','y_fold','x_slices','channels',    'h','w','x_fold','y_fold','x_slices','',          '',           '',      '',       '',         '',         '',         '',         '',           '',           'scale', 'zp','scale' ,'zp', ''                ,''           ,''       ,''                 ,''                        ,'']
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
            xls_line.append(1)
            xls_line.append(1)
            xls_line.append('force_folding_x' in node['frontend'])
            xls_line.append('force_folding_y' in node['frontend'])
            xls_line.append('force_unfolding_x' in node['frontend'])
            xls_line.append('force_unfolding_y' in node['frontend'])
        else:
            xls_line.extend(['',''])
            xls_line.append(node['frontend']['y_tiles'])
            xls_line.append(1)
            xls_line.append(1)
            xls_line.append('force_folding_x' in node['frontend'])
            xls_line.append('force_folding_y' in node['frontend'])
            xls_line.append('force_unfolding_x' in node['frontend'])
            xls_line.append('force_unfolding_y' in node['frontend'])

        input_tensors=[]
        inputs_scales=[]
        inputs_zps=[]
        for input_name in node['inputs']:
            input_tensors.append(ir.tensors[input_name])
        if op_type == OpType.CONV: #Remove weights and biases from report
            #if this was an Average Pool it will have fake weights and biases
            if len(input_tensors)==1:
                pass
            else:    

                del input_tensors[2]
                del input_tensors[1]
        elif op_type == OpType.RESIZE:
            del input_tensors[1]
        elif ('STRIDEDSLICE' in node['name']):
            del input_tensors[3]
            del input_tensors[2]
            del input_tensors[1]
        idx = 0


        for input_tensor in input_tensors:
            if (len(input_tensors) == 1):
                inputs_scales.append(float(node['frontend']['input_tensor_scale']))
                inputs_zps.append(int(node['frontend']['input_tensor_zp']))
            else:
                # input_tensors_scale/zp are normally per-input arrays, but some nodes store a
                # single (0-d numpy / scalar) value. Coerce so the qparams report never crashes
                # (a report writer must not fail the whole compile).
                scale_v = node['frontend']['input_tensors_scale']
                zp_v = node['frontend']['input_tensors_zp']
                try:
                    s = scale_v[idx]
                except (TypeError, IndexError):
                    s = scale_v.item() if hasattr(scale_v, 'item') else scale_v
                try:
                    z = zp_v[idx]
                except (TypeError, IndexError):
                    z = zp_v.item() if hasattr(zp_v, 'item') else zp_v
                inputs_scales.append(float(s))
                inputs_zps.append(int(z))
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

        if ('ddr_write_align' in node['frontend']):
            xls_line.append(node['frontend']['ddr_write_align'])
        else:
            xls_line.append('')
        
        if ('h_tiles' in node['frontend']):
            xls_line.append(node['frontend']['h_tiles'])
        else:
            xls_line.append('')
        
        if ('xslices_per_htile' in node['frontend']):
            xls_line.append(str(node['frontend']['xslices_per_htile']).replace(',',';'))
        else:
            xls_line.append('')

        if ('out_xslices_per_htile' in node['frontend']):
            xls_line.append(str(node['frontend']['out_xslices_per_htile']).replace(',',';'))
        else:
            xls_line.append('')

        if ('padding' in node['frontend']):
            xls_line.append(str(node['frontend']['padding']).replace(',',';'))
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
        if op_type == OpType.CONV:
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


    ammalloc_report_filename = ammalloc_report_filename+'.csv'
    csv_file = open(ammalloc_report_filename,'w')
    spreadsheet_format = DebugFilesFormat.CSV
    worksheet = None

    xls_header0 = ['#',    '',      '',      '',       'name','op_type','input shape','', '', 'output shape','', '', 'kernel size','stride','folding_conv','allocated blocks','',          'deallocated tensors','',                'amm_utilization','tables read(bytes)',''     ,''        ,'tensor read','tensor write']
    xls_header1 = ['blob', 'y_tile','h_tile','x_slice','',    '',       'channels',   'h','w','channels',    'h','w','',           '',      '',            'for input',       'for output','after output alloc' ,'after input read',''               ,'wloc'              ,'rqloc','rqtables','bytes'      ,'bytes']
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
            for current_h_tile in range(tiling_blob.h_tiles):
                for node_idx,current_node_name in enumerate(tiling_blob.nodes_in_blob):
                    node = ir.graph.nodes()[current_node_name]
                    current_node_num+=1
                    current_op_type = node['op_type']
                    # A scatter-concat (see scatter_concat_inputs) is a no-op in mem_allocation_per_htile -
                    # it never allocates AMM blocks, so none of the 'allocated_amm_blocks_for_*' backend
                    # fields below exist for it. Skip it in this report; it has no allocation to show.
                    if node['frontend'].get('is_scatter_concat'):
                        continue
                    num_slices = max(node['frontend']['xslices_per_htile'][current_h_tile], node['frontend']['out_xslices_per_htile'][current_h_tile])
                    for current_x_slice in range(num_slices):
                        xls_line = []
                        if current_y_tile==0 and current_h_tile==0 and node_idx==0:
                            xls_line.append(blob_idx)
                        else:
                            xls_line.append('')
                        if node_idx==0 and current_x_slice==0:
                            xls_line.append(current_y_tile)
                        else:
                            xls_line.append('')
                        if node_idx==0 and current_x_slice==0:
                            xls_line.append(current_h_tile)
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
                        if ('force_unfolding_x' in node['frontend']) or (node['op_type'] == OpType.RESIZE):
                            input_x_slice = int(current_x_slice / 2)
                        else:
                            if ('input_tensors' in node['frontend']):
                                input_tensor = node['frontend']['input_tensors'][0]
                            else:
                                input_tensor = node['frontend']['input_tensor']
                            if is_packed_xslices(input_tensor.num_packed_xslices):
                                input_x_slice = packed_lane_of_logical(input_tensor.num_packed_xslices, input_tensor.xslices_per_htile[current_h_tile], current_x_slice)[0]
                            else:
                                input_x_slice = current_x_slice // input_tensor.num_packed_xslices
                        # When a node is channel-split, input_x_slice may exceed the allocated array bounds.
                        # Skip the allocation report for out-of-bounds x-slices (they belong to split pieces).
                        even_input_grid = node['backend'].get('allocated_amm_blocks_for_input_even_grid', [[[]]])
                        odd_input_grid = node['backend'].get('allocated_amm_blocks_for_input_odd_grid', [[[]]])
                        if op_type in MULTIPLE_INPUT_OPS:
                            blocks_allocated_for_input=''
                            for input_idx,input_tensor in enumerate(node['frontend']['input_tensors']):
                                if (current_y_tile < len(even_input_grid) and current_h_tile < len(even_input_grid[current_y_tile])
                                    and input_x_slice < len(even_input_grid[current_y_tile][current_h_tile])):
                                    blocks_allocated_for_input += 'i'+str(input_idx)+'e:'+allocated_blocks_to_string(even_input_grid[current_y_tile][current_h_tile][input_x_slice][input_idx][0]) # indexs are y_tile#,h_tile#,x_slice#,Input#,amm#
                                    blocks_allocated_for_input += ',o:'+allocated_blocks_to_string(odd_input_grid[current_y_tile][current_h_tile][input_x_slice][input_idx][0])
                                else:
                                    blocks_allocated_for_input += 'i'+str(input_idx)+'e:N/A,o:N/A'
                        else:
                            if (current_y_tile < len(even_input_grid) and current_h_tile < len(even_input_grid[current_y_tile])
                                and input_x_slice < len(even_input_grid[current_y_tile][current_h_tile])):
                                blocks_allocated_for_input = 'i0e:'+allocated_blocks_to_string(even_input_grid[current_y_tile][current_h_tile][input_x_slice][0][0])
                                blocks_allocated_for_input += ',o:'+allocated_blocks_to_string(odd_input_grid[current_y_tile][current_h_tile][input_x_slice][0][0])
                            else:
                                blocks_allocated_for_input = 'i0e:N/A,o:N/A'
                        blocks_allocated_for_input=blocks_allocated_for_input.replace(',',';')
                        xls_line.append(blocks_allocated_for_input)
                        if ('force_folding_x' in node['frontend']):
                            output_x_slice = int(current_x_slice / 2)
                        else:
                            output_x_slice = current_x_slice
                        # When a node is channel-split, output_x_slice may exceed the allocated array bounds.
                        # Skip the allocation report for out-of-bounds x-slices (they belong to split pieces).
                        even_grid = node['backend'].get('allocated_amm_blocks_for_output_even_grid', [[[]]])
                        odd_grid = node['backend'].get('allocated_amm_blocks_for_output_odd_grid', [[[]]])
                        if (current_y_tile < len(even_grid) and current_h_tile < len(even_grid[current_y_tile])
                            and output_x_slice < len(even_grid[current_y_tile][current_h_tile])):
                            blocks_allocated_for_output = 'e:'+allocated_blocks_to_string(even_grid[current_y_tile][current_h_tile][output_x_slice][0]) # indexs are tile#,h_tile#,amm#
                            blocks_allocated_for_output += ',o:'+allocated_blocks_to_string(odd_grid[current_y_tile][current_h_tile][output_x_slice][0])
                        else:
                            blocks_allocated_for_output = 'e:N/A,o:N/A'  # Out-of-bounds for split nodes
                        blocks_allocated_for_output=blocks_allocated_for_output.replace(',',';')
                        xls_line.append(blocks_allocated_for_output)
                        if 'tensors_for_deallocation_after_output_allocation_report' in node['backend']:
                            tensors_deallocated_struct = node['backend']['tensors_for_deallocation_after_output_allocation_report']
                            if (current_y_tile, current_h_tile, current_x_slice) in tensors_deallocated_struct.tensors_list:
                                current_tile_dealocated_tensors = tensors_deallocated_struct.get_tensor_names_dict()[(current_y_tile, current_h_tile, current_x_slice)]
                                tensors_deallocated_after_output_allocation = ';'.join(current_tile_dealocated_tensors)
                            else:
                                tensors_deallocated_after_output_allocation = ''
                        else:
                            tensors_deallocated_after_output_allocation = ''
                        tensors_deallocated_after_output_allocation = tensors_deallocated_after_output_allocation.replace(',',';')
                        xls_line.append(tensors_deallocated_after_output_allocation)
                        if 'tensors_for_deallocation_after_ddr_read_allocation_report' in node['backend']:
                            tensors_deallocated_struct = node['backend']['tensors_for_deallocation_after_ddr_read_allocation_report']
                            if (current_y_tile, current_h_tile, current_x_slice) in tensors_deallocated_struct.tensors_list:
                                current_tile_dealocated_tensors = tensors_deallocated_struct.get_tensor_names_dict()[(current_y_tile, current_h_tile, current_x_slice)]
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