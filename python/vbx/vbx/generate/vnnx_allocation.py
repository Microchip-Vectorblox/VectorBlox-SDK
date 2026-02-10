import os
from .vnnx_types import *
from functools import reduce
from math import floor, ceil, log2, sqrt
from .constrain_fia_tile import allocate_fia_tile
from .utils import sp_invalid_exception_catcher

# NEW_FIA_TILING = 1  # Will remove later
MXP_DOUBLE_BUFFER = 1   # Will remove later
FIA_WRITE_DB = 0
OPTIMIZED_AVERAGE_POOL = 0


FORCE_SB = 0

WEIGHT_SHAPER_DATA_BANKS = 2
INPUT_SHAPER_BANKS = 2
INPUT_SHAPER_BANKS_SPARSE = 8
WEIGHT_SHAPER_BANKS = 2
WEIGHT_SHAPER_BANKS_SPARSE = 3
INPUT_SHAPER_BANK_SIZE_KB = 2
INPUT_SHAPER_BANK_SIZE_KB_SPARSE = 1

DEBUG_TILE_MAPPING = 0

WEIGHT_SHAPER_BANK_SIZE_KB = 1
MEMORY_MAPPED_WIDTH = 256
QUANTIZATION_RECORD_WIDTH = 64
QUANTIZATION_RECORD_WIDTH_BYTES = QUANTIZATION_RECORD_WIDTH // 8

##################################################################
### The below 6 functions must match fia.h equivalent functions; update both at the same time.
##################################################################
def get_input_shaper_banks_kb(sparse):
    if (sparse == 1):
        return INPUT_SHAPER_BANK_SIZE_KB_SPARSE
    else:
        return INPUT_SHAPER_BANK_SIZE_KB
    
def get_input_shaper_banks(sparse):
    if (sparse == 1):
        return INPUT_SHAPER_BANKS_SPARSE
    else:
        return INPUT_SHAPER_BANKS

def get_weight_shaper_ctrl_banks(sparse):
    if sparse != None:
        if (sparse == 2):
            return 0
        else:
            return sparse
    else:
        return 0

def get_weight_shaper_banks(sparse):
    return WEIGHT_SHAPER_DATA_BANKS + get_weight_shaper_ctrl_banks(sparse)

def fia_input_shaper_size_kb(filter_copies, sparse):
    return filter_copies * get_input_shaper_banks(sparse) * get_input_shaper_banks_kb(sparse)

def fia_input_shaper_size_per_bank_kb(filter_copies, sparse):
    return filter_copies * get_input_shaper_banks_kb(sparse)

def fia_weight_shaper_size_kb(parallel_kernels, sparse):
    return parallel_kernels * get_weight_shaper_banks(sparse) * WEIGHT_SHAPER_BANK_SIZE_KB

def fia_weight_shaper_size_per_bank_kb(parallel_kernels):
    return parallel_kernels * WEIGHT_SHAPER_BANK_SIZE_KB

def fia_quantization_shaper_size_kb():
    return 64//8

def sp_output_shaper_size_b(sp, no_sublayers=True): # returns the size of a single buffer in SP
    return sp // 2 if no_sublayers else sp // 4

# Get the input and weight shaper sizes
def _get_input_and_weight_shaper_sizes(filter_copies, parallel_output_maps, sparse):

    input_shaper_size = fia_input_shaper_size_kb(filter_copies, sparse) * 1024
    weight_shaper_size = fia_weight_shaper_size_kb(parallel_output_maps, sparse) * 1024

    return input_shaper_size, weight_shaper_size

def _get_input_and_weight_shaper_bank_sizes(filter_copies, parallel_output_maps, sparse):

    input_shaper_size = fia_input_shaper_size_kb(filter_copies, sparse) * 1024
    weight_shaper_size = fia_weight_shaper_size_kb(parallel_output_maps, sparse) * 1024

    input_shaper_bank_size  = input_shaper_size // get_input_shaper_banks(sparse)
    weight_shaper_bank_size = weight_shaper_size // get_weight_shaper_banks(sparse)

    return input_shaper_bank_size, weight_shaper_bank_size

def sp_used_fn(node):
    if MXP_DOUBLE_BUFFER and node.Conv2DOptions.mxp_double_buffer:
        return lambda o,t0,t: o*2+t0 if (len(node.subnode_array) == 0 and node.output_strides == [1,1]) else max(o*3 + t, o*2+t0+t)
    else:
        return lambda o,t0,t: o+t0 if (len(node.subnode_array) == 0 and node.output_strides == [1,1]) else max(o*2 + t, o+t0+t)


def aligned_size(size, vector_lanes):
    aligned = vector_lanes * 4
    if (size % aligned):
        return (size // aligned + 1) * aligned
    return size


def get_output_shapes(conv8, input_height, input_width):
    padded_input_width  = input_width + conv8.padding_width
    padded_input_height = input_height + conv8.padding_height

    k, c, kernel_height, kernel_width = conv8.filter_shape_dims
    dilation_height_factor, dilation_width_factor = conv8.dilation_height_factor, conv8.dilation_width_factor

    stride_height = conv8.stride_height
    stride_width = conv8.stride_width
    use_strided_input_maps = conv8.use_strided

    filter_height = kernel_height
    filter_width = kernel_width
    dilated_filter_height = ((filter_height-1)*dilation_height_factor) + 1
    dilated_filter_width  = ((filter_width-1)*dilation_width_factor) + 1
    output_height = (padded_input_height - dilated_filter_height) // stride_height + 1
    output_width =  (padded_input_width - dilated_filter_width) // stride_width + 1

    if conv8.use_depthwise:
        output_shaper_height = (padded_input_height - dilated_filter_height) + 1
        output_height = (output_shaper_height - 1) // stride_height + 1

        output_shaper_width = (padded_input_width - dilated_filter_width) + 1
        output_width = output_shaper_width
    else:
        if not use_strided_input_maps:
            output_shaper_height = padded_input_height - dilated_filter_height + 1
            output_shaper_width = padded_input_width
            if padded_input_width < dilated_filter_width:
                output_shaper_width = dilated_filter_width
        else:
            output_shaper_height = (input_height // stride_height)
            if input_height % stride_height:
                output_shaper_height += 1 
            output_shaper_width = (input_width // stride_width)
            if input_width % stride_width:
                output_shaper_width += 1

    return output_height, output_width, output_shaper_height, output_shaper_width


def composite_function(*func): 
    def compose(f, g): 
        return lambda x : f(g(x)) 
              
    return reduce(compose, func, lambda x : x)


def is_sp_invalid(node, vl, sp, tile, sparse=None):
    v_out, v_tmp0, v_tmp = max_scratchpad_required(node, vl, tile, sparse=sparse)
    sp_invalid = sp_used_fn(node)(v_out,v_tmp0, v_tmp) > sp

    # if v_out > 32*1024:
    #     return True
    # if max(v_tmp0,v_tmp) > (sp - 3*32*1024):
    #     return True

    return sp_invalid



def maximize_tile(node, vector_lanes, sp, fx, tile, idx, limit, scale=1, offset=0, sparse=None):
    tile = tile.copy()

    if tile[idx] < limit:
        tmp = tile[idx]
        sp_invalid = is_sp_invalid(node, vector_lanes, sp, tile, sparse=sparse)
        while(not sp_invalid):
            tmp = tile[idx]
            if tmp == limit:
                break
            tile[idx] = min(tmp * scale + offset , limit)
            sp_invalid = is_sp_invalid(node, vector_lanes, sp, tile, sparse=sparse)
        else:
            tile[idx] = tmp

    # we reduce the tile to a valid tile
    while(not np.all(fx(tile) >= 1)):
        tile[idx]-=1

    return tile


def multiply_tile(node, vector_lanes, sp, fx, tile, idx, limit, scale, sparse=None):
    return maximize_tile(node, vector_lanes, sp, fx, tile, idx, limit, scale=scale, offset=0, sparse=sparse)

def increment_tile(node, vector_lanes, sp, fx, tile, idx, limit, offset=1, sparse=None):
    return maximize_tile(node, vector_lanes, sp, fx, tile, idx, limit, scale=1, offset=offset, sparse=sparse)


def compose_subgraph(node, gx):
    fn = [gx(node)]
    for sn in node.subnode_array:
        fn.append(gx(sn))
    fn.reverse()
    return composite_function(*fn)


def minimum_tile_subgraph(node, fx):
    tile = np.array([1,1,1,1,1,1,1])
    j=0
    while(not np.all(fx(tile) >= 1)):
        for i,t in enumerate(fx(tile)):
            if t < 1:
                tile[i] += 1
        j = j + 1
    return tile

def minimum_valid_tile_subgraph(node, vl, sp, fx, opcode, tmp_dir=None, graph_idx=None, sparse=None, tmp_dir_obj=None):
    
    tile = minimum_tile_subgraph(node, fx)  
    sp_invalid = is_sp_invalid(node, vl, sp, tile, sparse=sparse)

    if sp_invalid:
        with sp_invalid_exception_catcher(sp, opcode, tmp_dir, graph_idx, tmp_dir_obj):
            assert(0)
        
    return tile

def max_scratchpad_required(node, vector_lanes, tile, sp=None, sparse=None):
    tile = tile.copy()
    v_out, v_tmp0 = max_sp_tile(node, vector_lanes, sp)(tile)
    tile = max_output_tile(node, sparse=sparse)(tile)

    v_tmp = 0
    for sn in node.subnode_array:
        v_0, v_1 = max_sp_tile(sn, vector_lanes, sp)(tile)
        v_out, v_tmp = max(v_out, v_0), max(v_tmp, v_1)

        # tile = min_output_tile(sn)(tile)
        tile = max_output_tile(sn, sparse=sparse)(tile)

    return v_out, v_tmp0, v_tmp


'''
per layer:
    have a function that when given the dimensions of an input tile
    returns the minimum dimensions of an output tile
'''
def min_output_tile(node):
    default = lambda s: s

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6

    # input dims == output dims
    default_types = [BuiltinOperator.RELU,
                     BuiltinOperator.RELU6,
                     BuiltinOperator.RELU_N1_TO_1,
                     BuiltinOperator.RELU_0_TO_1,
                     BuiltinOperator.PRELU,
                     BuiltinOperator.LEAKY_RELU,
                     BuiltinOperator.ABS,
                     BuiltinOperator.RSQRT,
                     BuiltinOperator.NEG,
                    #  BuiltinOperator.MUL,
                     BuiltinOperator.ADD,
                     BuiltinOperator.SUB,
                     BuiltinOperator.DEQUANTIZE,
                     BuiltinOperator.TANH,
                     BuiltinOperator.HARD_SWISH,
                     BuiltinOperator.ELU,
                     BuiltinOperator.GELU,
                     BuiltinOperator.EXP,
                     BuiltinOperator.LOG,
                     BuiltinOperator.LOGISTIC,
                     BuiltinOperator.GREATER,
                     BuiltinOperator.GREATER_EQUAL,
                     BuiltinOperator.SQUARED_DIFFERENCE,
                     BuiltinOperator.LESS,
                     BuiltinOperator.LESS_EQUAL,
                     BuiltinOperator.EQUAL,
                     BuiltinOperator.NOT_EQUAL,
                     BuiltinOperator.CONCATENATION,
                     VNNXOperator.IDENTITY,
                     BuiltinOperator.MINIMUM,
                     BuiltinOperator.MAXIMUM,
                     BuiltinOperator.CAST,
                     VNNXOperator.ELTWISE,
                     VNNXOperator.LUT,
                     BuiltinOperator.SLICE,
                     BuiltinOperator.STRIDED_SLICE,
                     BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,
                     BuiltinOperator.PACK,
                     BuiltinOperator.LOG_SOFTMAX,
                     BuiltinOperator.SPLIT,
                     BuiltinOperator.SPLIT_V,
                     ]

    # requires full channels, rows, columns, ...  for development only
    full_types = [BuiltinOperator.GATHER,
                  BuiltinOperator.DEPTH_TO_SPACE,
                  BuiltinOperator.SPACE_TO_DEPTH,
                  BuiltinOperator.BATCH_TO_SPACE_ND,
                  BuiltinOperator.SPACE_TO_BATCH_ND,
                  BuiltinOperator.L2_NORMALIZATION,
                  ]

    e = node.type
    if e in default_types:
        return default
    elif e in full_types: 
        def x(tile):
            d = -len(node.tensor_array[0].shape)
            while d < 0:
                if tile[d] < node.tensor_array[0].shape[d]:
                    t = np.copy(tile)
                    t[d] = 0
                    return t
                d += 1
            return tile
        return x
    elif e in [BuiltinOperator.CONV_2D, BuiltinOperator.DEPTHWISE_CONV_2D]:
        _, _, kh, kw = node.Conv2DOptions.filter_shape_dims
        sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
        dhf, dwf = node.Conv2DOptions.dilation_height_factor, node.Conv2DOptions.dilation_width_factor
        ph, pw = node.Conv2DOptions.padding_height, node.Conv2DOptions.padding_width

        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            orows = (rows - (1 + (kh-1)*dhf)) // sh + 1
            if rows == node.m:
                orows = (rows + ph - (1 + (kh-1)*dhf)) // sh + 1
            ocols = (cols - (1 + (kw-1)*dwf)) // sw + 1
            if cols == node.n:
                ocols = (cols + pw - (1 + (kw-1)*dwf)) // sw + 1
            return np.asarray([batch, imaps, irows, icols, maps, orows, ocols])

        return x 

    elif e == BuiltinOperator.TRANSPOSE_CONV:
        _, _, kh, kw = node.Conv2DOptions.filter_shape_dims
        sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
        dhf, dwf = node.Conv2DOptions.dilation_height_factor, node.Conv2DOptions.dilation_width_factor
        ph, pw = node.Conv2DOptions.padding_height, node.Conv2DOptions.padding_width

        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            transformed_rows = rows + (rows*(sh-1))
            transformed_rows += (kh - (ph//2) - 1) # pad along top of input
            # if full rows, then adjust dilated input and add the pad along bottom of input
            if rows == node.m:
                transformed_rows -= (sh-1)
                transformed_rows += (kh - ((ph//2) + (ph%2)) - 1)

            # TODO remove width adjustment when PACK sublayer injection working
            transformed_cols = cols + (cols*(sw-1))
            transformed_cols += (kw - (pw//2) - 1) # pad along left of input
            # if full cols, then adjust dilated input and add the pad along right of input
            if cols == node.n:
                transformed_cols -= (sw-1)
                transformed_cols += (kw - ((pw//2) + (pw%2)) - 1)

            orows = transformed_rows - (1 + (kh-1)*dhf) + 1
            ocols = transformed_cols - (1 + (kw-1)*dwf) + 1
            return np.asarray([batch, imaps, irows, icols, maps, orows, ocols])
            
        return x

    elif e == BuiltinOperator.FULLY_CONNECTED:
        output_depth, accum_depth = node.FullyConnectedOptions.filter_shape_dims
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            return np.asarray([batch, imaps, irows, icols, maps, rows, irows])
        return x 
    elif e == BuiltinOperator.BATCH_MATMUL:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            return np.asarray([batch, imaps, irows, icols, maps, rows, cols])
        return x   
    elif e in [BuiltinOperator.MEAN, BuiltinOperator.SUM, BuiltinOperator.REDUCE_PROD, BuiltinOperator.REDUCE_MAX, BuiltinOperator.REDUCE_MIN]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if rows < node.tensor_array[0].shape[-2]: # if not full rows, return 0 in rows
                return np.asarray([batch, imaps, irows, icols, maps, 0, 1])
            if cols < node.tensor_array[0].shape[-1]: #if not full cols, return 0 in cols
                return np.asarray([batch, imaps, irows, icols, maps, 1, 0])
            return np.asarray([batch, imaps, irows, icols, maps, 1, 1])
        return x   
    elif e in [BuiltinOperator.ARG_MAX, BuiltinOperator.ARG_MIN, BuiltinOperator.TOPK_V2]:
        axis = node.reduce8.axis
        assert(axis == -3)
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if axis == -1 and cols != node.tensor_array[0].shape[-1]:
                return np.asarray([batch, imaps, irows, icols, maps, rows, 0])
            if axis == -2 and rows != node.tensor_array[0].shape[-2]:
                return np.asarray([batch, imaps, irows, icols, maps, 0, cols])
            if axis == -3 and maps != node.tensor_array[0].shape[-3]:
                return np.asarray([batch, imaps, irows, icols, 0, rows, cols])
            return np.asarray([batch, imaps, irows, icols, maps, rows, cols])
        return x 
       
    elif e in [BuiltinOperator.AVERAGE_POOL_2D]:
        kh, kw = node.kernel_shape
        sh, sw = node.strides
        pads = node.pads.copy()

        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if cols != node.tensor_array[0].shape[-1]:
                return np.asarray([batch, imaps, irows, icols, maps, (rows-kh) // sh + 1, 0])

            if OPTIMIZED_AVERAGE_POOL:
                if node.tensor_array[0].shape[-3] % 2 == 0 and kh == 2 and kw == 2 and sh == 1 and sw == 1 and np.sum(pads) == 0:
                    if maps % 2 != 0:
                        return np.asarray([batch, imaps, irows, icols, 0, (rows-kh) // sh + 1, (cols-kw) // sw + 1])

                if node.tensor_array[0].shape[-2] % 2 == 0 and node.tensor_array[0].shape[-1] % 2 == 0 and kh == 2 and kw == 2 and sh == 2 and sw == 2 and np.sum(pads) == 0:
                    if cols % 2 != 0:
                        return np.asarray([batch, imaps, irows, icols, maps, (rows-kh) // sh + 1, 0])
                    if rows % 2 != 0:
                        return np.asarray([batch, imaps, irows, icols, maps, 0, (cols-kw) // sw + 1])
            return np.asarray([batch, imaps, irows, icols, maps, (rows-kh) // sh + 1, (cols-kw) // sw + 1])
        return x 

    elif e in [BuiltinOperator.MAX_POOL_2D]:
        kh, kw = node.kernel_shape
        sh, sw = node.strides

        return lambda s: np.concatenate((s[:ROWS],[(s[ROWS] - kh) // sh + 1],[(s[COLUMNS] - kw) // sw + 1]))

    elif e == BuiltinOperator.TRANSPOSE:
        permutation = node.TransposeOptions.permutation
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if permutation == [0,2,1]:
                return np.asarray([batch, imaps, irows, icols, maps, cols, rows])
            elif permutation == [1,0,2]:
                return np.asarray([batch, imaps, irows, icols, rows, maps, cols])
            elif permutation == [1,2,0]:
                return np.asarray([batch, imaps, irows, icols, rows, cols, maps])
            elif permutation == [2,0,1]:
                return np.asarray([batch, imaps, irows, icols, cols, maps, rows])
            elif permutation == [2,1,0]:
                return np.asarray([batch, imaps, irows, icols, cols, rows, maps])
            else:
                assert(permutation == [0,1,2])
                return np.asarray([batch, imaps, irows, icols, maps, rows, cols])
        return x

    elif e == VNNXOperator.PIXEL_SHUFFLE:
        r = node.PixelShuffleOptions.r
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if maps % r**2:
                return np.asarray([batch, imaps, irows, icols, 0, rows*r, cols*r])
            return np.asarray([batch, imaps, irows, icols, maps // (r**2), rows*r, cols*r])
        return x

    elif e == BuiltinOperator.TILE:
        tile_n, tile_c, tile_h, tile_w = node.TileOptions.tile
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if tile_n > 1:
                if cols != node.n:
                    return np.asarray([batch, imaps, irows, icols, maps, rows, 0])
                if rows != node.m:
                    return np.asarray([batch, imaps, irows, icols, maps, 0, cols])
                if maps != node.channels:
                    return np.asarray([batch, imaps, irows, icols, 0, rows, cols])
            elif tile_c > 1:
                if cols != node.n:
                    return np.asarray([batch, imaps, irows, icols, maps, rows, 0])
                if rows != node.m:
                    return np.asarray([batch, imaps, irows, icols, maps, 0, cols])
                if maps != node.channels:
                    return np.asarray([batch, imaps, irows, icols, 0, rows, cols])
            elif tile_h > 1:
                if cols != node.n:
                    return np.asarray([batch, imaps, irows, icols, maps, rows, 0])
                if rows != node.m:
                    return np.asarray([batch, imaps, irows, icols, maps, 0, cols])
            elif tile_w > 1:
                if cols != node.n:
                    return np.asarray([batch, imaps, irows, icols, maps, rows, 0])
            return np.asarray([batch, imaps, irows, icols, maps, rows, cols])
        return x 
    elif e in [BuiltinOperator.PAD, BuiltinOperator.DILATE, BuiltinOperator.MIRROR_PAD]:
        pads = node.pads.copy()

        if e in [BuiltinOperator.PAD, BuiltinOperator.DILATE]:
            dh, dw = node.PadOptions.transpose_dilate_h, node.PadOptions.transpose_dilate_w
            dhf = dh - 1
            dwf = dw - 1

            if dhf == 0 and dwf == 0:
                return default

            m = node.tensor_array[0].shape[-2]
            n = node.tensor_array[0].shape[-1]

            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile

                omaps = maps + pads[0] + pads[3]
                orows = rows + (dhf*rows)
                ocols = cols + (dwf*cols)
                if rows == m:
                    orows = rows + pads[1] + pads[4] + (dhf*rows)
                    orows -= dhf
                if cols == n:
                    ocols = cols + pads[2] + pads[5] + (dwf*cols)
                    ocols -= dwf

                return np.asarray([batch, imaps, irows, icols, omaps, orows, ocols])
            return x
        else:
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                omaps = maps + pads[0] + pads[3]
                orows =  pads[1] + pads[4]
                ocols =  pads[2] + pads[5]
                mode = node.MirrorPadOptions.mode
                if rows < orows*2+1 and orows > 0:
                    return np.asarray([batch, imaps, irows, icols, maps, 0, ocols])
                
                if cols < ocols*2+1 and ocols > 0:
                    return np.asarray([batch, imaps, irows, icols, maps, orows, 0])
                # omaps = maps + pads[0] + pads[3]
                orows += rows 
                ocols += cols
                return np.asarray([batch, imaps, irows, icols, omaps, orows, ocols])
            return x

    elif e == BuiltinOperator.UNPACK:
        shape = node.tensor_array[0].shape
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile

            if cols < shape[-1]:
                return np.asarray([batch, imaps, irows, icols, maps, rows, 0])
            elif len(shape) > 1 and rows < shape[-2]:
                return np.asarray([batch, imaps, irows, icols, maps, 0, cols])
            elif len(shape) > 2 and maps < shape[-3]:
                return np.asarray([batch, imaps, irows, icols, 0, rows, cols])
            return np.asarray([batch, imaps, irows, icols, maps, rows, cols])
        return x 
    elif e == BuiltinOperator.RESHAPE:
        mode = node.ReshapeOptions.mode
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if mode == 0:
                pass
            elif mode == 1:
                if cols < node.tensor_array[0].shape[-1]:
                    return np.asarray([batch, imaps, irows, icols, maps, rows, 0])
                elif maps < node.tensor_array[0].shape[-3]:
                    return np.asarray([batch, imaps, irows, icols, 0, rows, cols])
            elif mode == 2:
                pass
            elif mode == 3:
                pass
            elif mode == 12+1:
                if cols < node.tensor_array[0].shape[-1]:
                    return np.asarray([batch, imaps, irows, icols, maps, rows, 0])
                elif maps < node.tensor_array[0].shape[-3]:
                    return np.asarray([batch, imaps, irows, icols, 0, rows, cols])
            return np.asarray([batch, imaps, irows, icols, maps, rows, cols])
        return x 
    
    elif e == BuiltinOperator.PACK:
        axis = node.PackOptions.axis
        count = node.PackOptions.count
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if axis == -4:
                return np.asarray([count*batch, imaps, irows, icols, maps, rows, cols])
            elif axis == -3:
                return np.asarray([batch, imaps, irows, icols, count*maps, rows, cols])
            elif axis == -2:
                return np.asarray([batch, imaps, irows, icols, maps, count*rows, cols])
            elif axis == -1:
                return np.asarray([batch, imaps, irows, icols, maps, rows, count*cols])
            
        return x

    elif e == BuiltinOperator.SOFTMAX:
        depth = node.SoftmaxOptions.depth # requires full dimension
        axis = node.SoftmaxOptions.axis

        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if axis ==-3:
                if maps < node.tensor_array[0].shape[-3]:
                    return np.asarray([batch, imaps, irows, icols, 0, rows, cols]) # need full maps
            if axis == -1:
                if cols < node.tensor_array[0].shape[-1]:
                    return np.asarray([batch, imaps, irows, icols, maps, rows, 0]) # need full cols
                
            if axis == -2:
                if rows < node.tensor_array[0].shape[-2]:
                    return np.asarray([batch, imaps, irows, icols, maps, 0, cols]) # full rows
            
            return np.asarray([batch, imaps, irows, icols, maps, rows, cols])

        return x
    elif e in [BuiltinOperator.RESIZE_BILINEAR, BuiltinOperator.RESIZE_NEAREST_NEIGHBOR]:
        sh, sw = node.ResizeOptions.scale   
        enable_postprocessing_tiling = node.ResizeOptions.b_postproc_tiling
        align_corner=0
        
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            temp_row = (rows)*sh
            temp_col = (cols)*sw 
            in_rows, in_cols = node.tensor_array[0].shape[-2], node.tensor_array[0].shape[-1]
            out_rows, out_cols = in_rows*sh, in_cols*sw

            if temp_row > out_rows:
                temp_row = out_rows
            if temp_col > out_cols:
                temp_col = out_cols

            if e == BuiltinOperator.RESIZE_BILINEAR:
                # temp_row = (rows-1)*sh
                # temp_col = (cols-1)*sw 

                if (sh==2.0 and sw ==2.0) or (sh==4.0 and sw ==4.0):
                    temp_row = (rows)*sh
                    temp_col = (cols)*sw 
                    if temp_col != int(temp_col) or cols<2:
                        return np.asarray([batch, imaps, irows, icols, maps, temp_row , 0])
                
                    if temp_row != int(temp_row) or rows<2:
                        return np.asarray([batch, imaps, irows, icols, maps, 0 , temp_col])
                
                else:
                    if temp_col != int(temp_col) and temp_col < out_cols:
                        return np.asarray([batch, imaps, irows, icols, maps, int(temp_row) , 0])

                    if (temp_row != int(temp_row) and temp_row < out_rows):
                        return np.asarray([batch, imaps, irows, icols, maps, 0 , int(temp_col)])
            else:
                    
                if temp_col != int(temp_col):
                    return np.asarray([batch, imaps, irows, icols, maps, temp_row , 0])
                    
                if temp_row != int(temp_row) and temp_row <= out_rows:
                    return np.asarray([batch, imaps, irows, icols, maps, 0 , temp_col])
            
            return np.asarray([batch, imaps, irows, icols, maps, int(temp_row) , int(temp_col)])

        return x
    elif e == BuiltinOperator.MUL:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            filter_dim = node.broadcast8.filter_shape_dims
            filter_rows = filter_dim[2]
            filter_columns = filter_dim[3]
            if e == BuiltinOperator.MUL:
                if filter_rows == node.tensor_array[0].shape[-2] and filter_columns == node.tensor_array[0].shape[-1]:
                    if cols != node.tensor_array[0].shape[-1]:
                        return np.asarray([batch, imaps, irows, icols, maps, rows, 0])
            
            # if e in [BuiltinOperator.MUL, BuiltinOperator.ADD]:
            #     if filter_rows == node.tensor_array[0].shape[-2] and filter_columns == node.tensor_array[0].shape[-1]:
            #         if cols != node.tensor_array[0].shape[-1]:
            #             # print('min_output_tile', e, cols, node.tensor_array[0].shape[-1])
            #             return np.asarray([batch, imaps, irows, icols, maps, rows, 0])
                    
            return np.asarray([batch, imaps, irows, icols, maps, rows, cols])
        return x
        
    elif e == VNNXOperator.UNKNOWN:
        pass
    print(e, 'min_output_tile')
    return None

'''
per layer:
    have a function that when given the dimensions of an input tile
    returns the maximum dimensions of an output tile
'''
def max_output_tile(node, sparse=None):
    default = lambda s: s

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6

    e = node.type
    if e in [BuiltinOperator.CONV_2D, BuiltinOperator.DEPTHWISE_CONV_2D]:
        _, _, kh, kw = node.Conv2DOptions.filter_shape_dims
        sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
        dhf, dwf = node.Conv2DOptions.dilation_height_factor, node.Conv2DOptions.dilation_width_factor
        ph, pw = node.Conv2DOptions.padding_height, node.Conv2DOptions.padding_width

        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            orows = (rows + (ph+1)//2 - (1 + (kh-1)*dhf)) // sh + 1
            if rows == node.m:
                orows = (rows + ph - (1 + (kh-1)*dhf)) // sh + 1
            ocols = (cols + (pw+1)//2 - (1 + (kw-1)*dwf)) // sw + 1
            if cols == node.n:
                ocols = (cols + pw - (1 + (kw-1)*dwf)) // sw + 1
            return np.asarray([batch, imaps, irows, icols, maps, orows, ocols])

        return x 

    elif e == BuiltinOperator.TRANSPOSE_CONV:
        _, _, kh, kw = node.Conv2DOptions.filter_shape_dims
        sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
        dhf, dwf = node.Conv2DOptions.dilation_height_factor, node.Conv2DOptions.dilation_width_factor
        ph, pw = node.Conv2DOptions.padding_height, node.Conv2DOptions.padding_width

        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            transformed_rows = rows + (rows*(sh-1))
            transformed_rows += (kh - (ph//2) - 1) # pad along top of input
            # if full rows, then adjust dilated input and add the pad along bottom of input
            if rows == node.m:
                transformed_rows -= (sh-1)
                transformed_rows += (kh - ((ph//2) + (ph%2)) - 1)

            # TODO remove width adjustment when PACK sublayer injection working
            transformed_cols = cols + (cols*(sw-1))
            transformed_cols += (kw - (pw//2) - 1) # pad along left of input
            # if full cols, then adjust dilated input and add the pad along right of input
            if cols == node.n:
                transformed_cols -= (sw-1)
                transformed_cols += (kw - ((pw//2) + (pw%2)) - 1)

            orows = transformed_rows - (1 + (kh-1)*dhf) + 1
            ocols = transformed_cols - (1 + (kw-1)*dwf) + 1
            return np.asarray([batch, imaps, irows, icols, maps, orows, ocols])
            
        return x

    elif e in [BuiltinOperator.PAD, BuiltinOperator.DILATE, BuiltinOperator.MIRROR_PAD]:
        pads = node.pads.copy()
        if e in [BuiltinOperator.PAD, BuiltinOperator.DILATE]:
            dh, dw = node.PadOptions.transpose_dilate_h, node.PadOptions.transpose_dilate_w
            dhf = dh - 1
            dwf = dw - 1

            if (sparse == 2):
                return default

            m = node.tensor_array[0].shape[2]
            n = node.tensor_array[0].shape[3]

            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile

                omaps = maps + pads[0] + pads[3]
                orows = rows + pads[1] + pads[4] + (dhf*rows)
                ocols = cols + pads[2] + pads[5] + (dwf*cols)
                if rows == m:
                    orows -= dhf
                if cols == n:
                    ocols -= dwf

                return np.asarray([batch, imaps, irows, icols, omaps, orows, ocols])
            return x
        else:
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                omaps = maps + pads[0] + pads[3]
                orows = rows + pads[1] + pads[4]
                ocols = cols + pads[2] + pads[5]
                return np.asarray([batch, imaps, irows, icols, omaps, orows, ocols])
            return x

    return min_output_tile(node)
    

'''
per layer:
    have a function that when given the dimensions of an input tile
    returns the scratchpad usage  (output_size, temporary_size)
'''
def max_sp_tile(node, vector_lanes, sp=None):
    aligned = lambda sz : aligned_size(sz, vector_lanes)
    default = lambda s: [aligned(s[0]*s[-3]*s[-2]*s[-1]), 0]

    # only needs output allocated, no temporary (sp_malloc) needed
    default_types = [BuiltinOperator.MAX_POOL_2D,
                    #  BuiltinOperator.NEG,
                     BuiltinOperator.LOG_SOFTMAX,
                     BuiltinOperator.RESHAPE,
                     BuiltinOperator.SLICE,
                     BuiltinOperator.STRIDED_SLICE,
                     # placeholder below
                     BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,
                     BuiltinOperator.L2_NORMALIZATION,
                     BuiltinOperator.GATHER,
                     BuiltinOperator.DEPTH_TO_SPACE,
                     BuiltinOperator.SPACE_TO_DEPTH,
                     BuiltinOperator.BATCH_TO_SPACE_ND,
                     BuiltinOperator.SPACE_TO_BATCH_ND,
                    #  BuiltinOperator.RESIZE_NEAREST_NEIGHBOR,
                     ]

    e = node.type
    if e in default_types:
        return default
    elif e in [BuiltinOperator.CONV_2D, BuiltinOperator.DEPTHWISE_CONV_2D]:
        _, _, kh, kw = node.Conv2DOptions.filter_shape_dims
        sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
        dhf, dwf = node.Conv2DOptions.dilation_height_factor, node.Conv2DOptions.dilation_width_factor
        ph, pw = node.Conv2DOptions.padding_height, node.Conv2DOptions.padding_width
        use_strided_input_maps = node.Conv2DOptions.use_strided

        if not node.Conv2DOptions.use_fia: # vector
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                rows += ph
                cols += pw

                v_tmp = 0
                if type(node) != Subnode:
                    v_tmp += aligned(batch * imaps * rows * cols) # v_in
                if node.Conv2DOptions.use_vector == 2: #optimized
                    v_tmp += aligned(batch * maps * rows * cols * 2)
                else:
                    v_tmp += aligned(batch * rows * cols * sw * 2) # v_mul
                    v_tmp += aligned(batch * maps * rows * cols * 4) # v_acc
                
                output_rows = (rows - (1 + (kh-1)*dhf)) // sh + 1;
                output_cols = (cols - (1 + (kw-1)*dwf)) // sw + 1;
                v_out = aligned(batch * maps * output_rows * output_cols)

                if node.Conv2DOptions.use_vector:
                    return [v_out, v_tmp]
                else:
                    return [v_out, 0]
            return x
        else:
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile

                orows = (rows + (ph+1)//2 - (1 + (kh-1)*dhf)) // sh + 1
                if rows == node.m:
                    orows = (rows + ph - (1 + (kh-1)*dhf)) // sh + 1
                ocols = (cols + (pw+1)//2 - (1 + (kw-1)*dwf)) // sw + 1
                if cols == node.n:
                    ocols = (cols + pw - (1 + (kw-1)*dwf)) // sw + 1

                if not node.offloaded:
                    _, _, orows, ocols = get_output_shapes(node.Conv2DOptions, rows, cols)
                
                v_out = aligned(batch * maps * orows * ocols)

                return [v_out, 0]
            return x

    elif e == BuiltinOperator.TRANSPOSE_CONV:
        _, _, kh, kw = node.Conv2DOptions.filter_shape_dims
        sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
        dhf, dwf = node.Conv2DOptions.dilation_height_factor, node.Conv2DOptions.dilation_width_factor
        ph, pw = node.Conv2DOptions.padding_height, node.Conv2DOptions.padding_width
        use_strided_input_maps = node.Conv2DOptions.use_strided

        # TODO sp usage for vector code
        if not node.Conv2DOptions.use_fia:
            pass

        else: # accelerator

            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                rows += ph
                cols += pw

                transformed_rows = rows + (rows*(sh-1))
                transformed_rows += (kh - (ph//2) - 1) # pad along top of input
                # if full rows, then adjust dilated input and add the pad along bottom of input
                if rows == node.m:
                    transformed_rows -= (sh-1)
                    transformed_rows += (kh - ((ph//2) + (ph%2)) - 1)

                # TODO remove width adjustment when PACK sublayer injection working
                transformed_cols = cols + (cols*(sw-1))
                transformed_cols += (kw - (pw//2) - 1) # pad along left of input
                # if full cols, then adjust dilated input and add the pad along right of input
                if cols == node.n:
                    transformed_cols -= (sw-1)
                    transformed_cols += (kw - ((pw//2) + (pw%2)) - 1)

                # TODO output offset formula mimics formula for regular conv, but how was it formulated?
                out_offset = (kh//2)*dhf*transformed_cols+(kw//2)*dwf

                v_out = aligned(batch * maps * (transformed_rows * transformed_cols + out_offset))

                return [v_out, 0]
            return x

            pass
    elif e == BuiltinOperator.FULLY_CONNECTED:
        output_depth, accum_depth = node.FullyConnectedOptions.filter_shape_dims
        if not node.FullyConnectedOptions.use_fia: # vector
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile

                v_tmp = aligned(batch*imaps*(icols + icols + irows*4 + irows*4))
                v_out = aligned(batch*maps*irows)

                return [v_out, v_tmp]
            return x
        else: # fia
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                v_out = aligned(batch*maps*icols*cols*rows)

                return [v_out, 0]
            return x
    elif e == BuiltinOperator.BATCH_MATMUL:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile

            v_tmp = aligned(batch*maps*rows*cols)
            v_tmp += aligned(batch*maps*rows*cols)
            v_out = aligned(batch*maps*rows*cols)

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.REDUCE_MAX, BuiltinOperator.REDUCE_MIN]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_out = aligned(maps) #assumes reducing to maps - axis == [-3,-2]
            v_tmp = 0

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.MEAN, BuiltinOperator.SUM]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_out = aligned(maps)
            v_tmp = aligned(maps*4)

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.REDUCE_PROD]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_out = aligned(maps)
            v_tmp = aligned(rows*cols*4) # single map @ word
            v_tmp += aligned(maps*4) # acc @ word

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.ARG_MAX, BuiltinOperator.ARG_MIN, BuiltinOperator.TOPK_V2]:
        axis = node.reduce8.axis
        assert(axis == -3)
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_out = 0
            v_tmp = 0
            if axis == -3:
                v_tmp = aligned(batch * rows * cols * batch*3) #@1 word + 3 bytes + v_in
                # v_tmp += aligned(batch * rows * cols * 4)
                v_out = aligned(batch * 1 * rows * cols * 4)

            return [v_out, v_tmp]
        return x
    elif e == VNNXOperator.PIXEL_SHUFFLE:
        r = node.PixelShuffleOptions.r
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_out = aligned(batch*(maps//(r**2))*rows*r*cols*r) # v_in == v_out
            v_tmp = 0

            return [v_out, v_tmp]
        return x
    elif e == BuiltinOperator.AVERAGE_POOL_2D:
        kh, kw = node.kernel_shape
        sh, sw = node.strides
        pads = node.pads.copy()
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile

            padded_height = rows + pads[1] + pads[4]
            padded_width = cols + pads[2] + pads[5]
            out_rows = (padded_height - kh)//sh + 1 
            out_cols = (padded_width - kw)//sw + 1 
            strided_out_cols  = (padded_width - kw)//1 + 1

            v_tmp = aligned(batch*maps*out_rows*out_cols*2)
            v_tmp += aligned(batch*maps*out_rows*strided_out_cols*2)

            if OPTIMIZED_AVERAGE_POOL:
                if node.tensor_array[0].shape[-2] % 2 == 0 and node.tensor_array[0].shape[-1] % 2 == 0 and kh == 2 and kw == 2 and sh == 2 and sw == 2 and np.sum(pads) == 0:
                    v_tmp = 0

                if node.tensor_array[0].shape[-3] % 2 == 0 and kh == 2 and kw == 2 and sh == 1 and sw == 1 and np.sum(pads) == 0:
                    v_tmp = 0

            v_out = aligned(batch*maps*out_rows*out_cols)

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.RELU, BuiltinOperator.RELU_N1_TO_1, BuiltinOperator.RELU_0_TO_1, BuiltinOperator.LEAKY_RELU, BuiltinOperator.ABS]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_tmp = aligned(batch * maps * rows * cols * 4) * 2 # 2 word size vectors (of input tile)
            v_out = aligned(batch * maps * rows * cols) # byte size vector( same as input tile size)

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.NEG]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_tmp = 0
            v_out = aligned(batch * maps * rows * cols) # byte size vector( same as input tile size)

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.PRELU]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if node.prelu.optimized:
                # node.prelu.maps_at_once = maps
                if node.prelu.vci_int8 == -1:
                    v_tmp = aligned(batch * min(maps,node.prelu.maps_at_once) * rows * cols * 2) * 2 # 2 half size vectors (of input tile)
                else:
                    v_tmp = aligned(batch * min(maps,node.prelu.maps_at_once) * rows * cols * 2) * 1 # 1 half size vectors (of input tile)
            else:
                v_tmp = aligned(batch * min(maps,node.prelu.maps_at_once) * rows * cols * 4) * 2 # 2 word size vectors (of input tile)
            v_out = aligned(batch * maps * rows * cols) # byte size vector( same as input tile size)

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.RELU6]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_tmp = aligned(batch * maps * rows * cols * 4 * 2) # 2 word size vectors (of input tile)
            v_tmp = aligned(batch * maps * rows * cols * 4 * 8) # vbx_MultiplyByQuantizedMultiplier_shift_less_than_1 tmp
            v_tmp = aligned(batch * maps * rows * cols * 2)
            v_out = aligned(batch * maps * rows * cols) # byte size vector( same as input tile size)

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.SOFTMAX]:
        axis = node.SoftmaxOptions.axis
        depth = node.SoftmaxOptions.depth
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile 
            if axis == -3:
                inner_size = rows * cols
                outer_size = batch 
            elif  axis  == -2:
                inner_size = cols
                outer_size = maps*batch
            elif axis ==  -1: 
                inner_size = 1
                outer_size = rows*maps*batch # #[1, 1, 400, 1]

            v_tmp = aligned(inner_size*depth*4)  # v_exp
            v_tmp += aligned(inner_size*4) # v_sum 
            v_tmp += aligned(inner_size*4) # v_inv_sum 
            v_tmp += aligned(inner_size*4) # v_tmp 
            v_out = aligned(batch * maps * rows * cols)  # byte size vector( same as input tile size)

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.SPLIT, BuiltinOperator.SPLIT_V]: #currently vnnx_subgraph, allocating v_in
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_tmp = aligned(batch * maps * rows * cols) #v_in
            v_out = aligned(batch * maps * rows * cols)

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.PAD, BuiltinOperator.MIRROR_PAD, BuiltinOperator.DILATE]:
        pads = node.pads.copy()
        dhf = 0
        dwf = 0
        if e in [BuiltinOperator.PAD, BuiltinOperator.DILATE]:
            dh, dw = node.PadOptions.transpose_dilate_h, node.PadOptions.transpose_dilate_w
            dhf = dh - 1
            dwf = dw - 1

        if dhf > 0 or dwf > 0: # transpose dilated pad
            m = node.tensor_array[0].shape[2]
            n = node.tensor_array[0].shape[3]

            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile

                padded_maps = maps + pads[0] + pads[3]
                padded_rows = rows + pads[1] + pads[4] + (dhf*rows)
                padded_cols = cols + pads[2] + pads[5] + (dwf*cols)
                if rows == m:
                    padded_rows -= dhf
                if cols == n:
                    padded_cols -= dwf
                
                v_out = aligned(batch * padded_maps * padded_rows * padded_cols)
                v_tmp = aligned(batch * padded_rows * padded_cols) # v_map_flag

                return [v_out, v_tmp]
        else:
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile

                padded_maps = maps + pads[0] + pads[3]
                padded_rows = rows + pads[1] + pads[4]
                padded_cols = cols + pads[2] + pads[5]
                v_out = aligned(batch * padded_maps * padded_rows * padded_cols)
                v_tmp = 0
                return [v_out, v_tmp]            
        return x
    elif e == BuiltinOperator.DEQUANTIZE:
        activation_max = node.activation_max
        activation_min = node.activation_min
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if activation_min == 0 and activation_max == 0:
                v_tmp = 0
            else:
                v_tmp = aligned(batch * maps * rows * cols * 4)
            v_out = aligned(batch * maps * rows * cols)
            
            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.MUL, BuiltinOperator.ADD, BuiltinOperator.SUB, BuiltinOperator.SQUARED_DIFFERENCE, BuiltinOperator.MAXIMUM, BuiltinOperator.MINIMUM]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if node.broadcast8.optimized:
                if e == BuiltinOperator.MUL:
                    v_tmp = aligned(batch * maps * rows * cols * 2)
                    if node.broadcast8.broadcast != 1:
                        v_tmp += aligned(batch * maps * rows * cols * 2)
                    node.broadcast8.isize = 0
                else:
                    node.broadcast8.isize = rows*cols
                    v_tmp = aligned(rows * cols * 2)
                    if node.broadcast8.broadcast != 1:
                        v_tmp += aligned(rows * cols * 2)
            else:
                node.broadcast8.isize = 0
                v_tmp = aligned(batch * maps * rows * cols * 4)
                if node.broadcast8.broadcast != 1:
                    v_tmp += aligned(batch * maps * rows * cols * 4)
            v_out = aligned(batch * maps * rows * cols)
           
            return [v_out, v_tmp]
        return x
    elif e == BuiltinOperator.TRANSPOSE:
        permutation = node.TransposeOptions.permutation
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_tmp = aligned(batch * maps * rows * cols)
            v_out = aligned(batch * maps * rows * cols)

            return [v_out, v_tmp]
        return x
    elif e == BuiltinOperator.TILE:
        tile_n, tile_c, tile_h, tile_w = node.TileOptions.tile
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_tmp = aligned(tile_n * batch * tile_c * maps * tile_h * rows * tile_w * cols)
            v_out = aligned(tile_n * batch * tile_c * maps * tile_h * rows * tile_w * cols)

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.TANH, BuiltinOperator.HARD_SWISH, BuiltinOperator.ELU, BuiltinOperator.GELU, BuiltinOperator.EXP, BuiltinOperator.LOG, BuiltinOperator.RSQRT, BuiltinOperator.LOGISTIC, VNNXOperator.LUT]:
        vci_lut = True
        if node.ActivationOptions.vci_int8 == -1:
            vci_lut = False
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            bytes = sizeof_calc_type(node.output_data_type)
            v_out = aligned(batch*maps*rows*cols*bytes)
            if vci_lut:
                v_tmp = 0 #we preload at beginning of vnnx-subgraph, so no temporaries when calling TILES
            else:
                v_tmp = aligned(batch*maps*rows*cols) # assumes LUT
            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.PACK, BuiltinOperator.UNPACK]:
        count = node.PackOptions.count
        axis = node.PackOptions.axis
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_out = aligned(count*batch*maps*rows*cols)
            if axis != -4:
                v_tmp = aligned(batch*maps*rows*cols) # v_tran
            v_tmp = aligned(batch*maps*rows*cols) # v_in

            return [v_out, v_tmp]
        return x
    elif e in [VNNXOperator.IDENTITY, BuiltinOperator.CONCATENATION]:
        num_inputs = node.num_inputs 
        bytes = sizeof_calc_type(node.input_data_type)
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_out = aligned(batch * maps * rows * cols * bytes)
            v_tmp = 0

            return [v_out, v_tmp]
        return x
    elif e == BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
        sh, sw = node.ResizeOptions.scale
        if sh == 2. and sw == 2.:
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                size = batch * maps * rows * cols
                v_tmp = 0 #aligned(size*2) 
                v_out = aligned(size*2*2)

                return [v_out, v_tmp]
            return x
        elif sh == 0.5 and sw == 0.5:
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                size = batch * maps * rows * cols
                v_tmp = 0 #aligned(size)
                v_out = aligned(int(size*0.5*0.5))

                return [v_out, v_tmp]
            return x
        elif sh == 4.0 and sw == 4.0:
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                size = batch * maps * rows * cols
                v_tmp = 0 #aligned(size)
                v_out = aligned(size*4*4)

                return [v_out, v_tmp]
            return x
        else:
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                size = batch * maps * rows * cols
                # v_tmp = aligned(size)
                if  sw <1 :
                    v_tmp = aligned(int(rows * cols))
                else:
                    v_tmp = aligned(int(rows * sw *cols))
                # v_tmp += aligned(int(rows*sw*cols +0.5))
                v_out = aligned(int((size*sh)*sw))

                return [v_out, v_tmp]
            return x
    elif e == BuiltinOperator.RESIZE_BILINEAR:
        sh, sw = node.ResizeOptions.scale
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            size = batch * maps * rows * cols      
            v_tmp_size = max(rows, int(cols*sw))    
            v_tmp = aligned(v_tmp_size*4)
            v_tmp += aligned(cols*sw*rows*sh)

            if sh==4.0 or sh==2.0:
                v_tmp+=aligned(cols*4)
                v_tmp+=aligned(cols*4)
                v_tmp+=aligned(cols*sw)
            elif sw>1 :    #upscale    
                v_tmp_size2 = rows*int(sw*cols)
                v_tmp += aligned(v_tmp_size2) #xp_temp
                v_tmp += aligned(4*2) #xp_temp
            elif sw<1:   #downscale
                v_tmp += aligned(rows * cols)
                v_tmp += aligned(rows * cols*sw)  #xp_temp2
            
            v_out = aligned(int((size*sh)*sw))

            return [v_out, v_tmp]
        return x
    elif e == VNNXOperator.ELTWISE:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile

            v_tmp = 0
            if node.eltwise8.optimized:
                isize = batch*maps*rows*cols
                if node.eltwise8.type in [eltwise_type.ELTWISE_ADD, eltwise_type.ELTWISE_SUB]:
                    sf = 8
                    if maps*rows > sf:
                        isize = (batch*maps*rows*cols+(sf-1)) // sf
                        isize = min(8*cols, isize) 
                node.eltwise8.isize = isize

                v_tmp += aligned(isize * 2) # v_tmp
                v_tmp += aligned(isize * 2) # v_add
            elif node.eltwise8.type in [eltwise_type.ELTWISE_MAXIMUM, eltwise_type.ELTWISE_MINIMUM]:
                v_tmp = aligned(batch * maps * rows * cols) #v_in
                node.eltwise8.isize = 0
            else:
                node.eltwise8.isize = 0
                v_tmp += aligned(batch * maps * rows * cols * 4) # v_tmp
                v_tmp += aligned(batch * maps * rows * cols * 4) # v_add

            if isinstance(node, Node):
                v_tmp += aligned(batch * maps * rows * cols) # v_in

            # TODO bring in isize chunks if optimized, not full
            v_tmp += aligned(batch * maps * rows * cols) # v_in2

            v_out = aligned(batch * maps * rows * cols)

            return [v_out, v_tmp]
        return x
    
    elif e in [BuiltinOperator.MAXIMUM, BuiltinOperator.MINIMUM]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            size = batch  * rows * cols * maps
            filter_shape = node.MinMaxOptions.filter_shape_dims
            filter_size = np.prod(filter_shape)

            v_tmp = 0
            if filter_size > 1:
                v_tmp = aligned(size)
            
            v_out = aligned(batch * maps * rows * cols)

            return [v_out, v_tmp]
        return x
      
    elif e == BuiltinOperator.CAST:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            size = batch * maps * rows * cols

            v_tmp = 0
            v_out = aligned(size) 

            return [v_out, v_tmp]
        return x
    elif e == VNNXOperator.UNKNOWN:
        pass
    print('no set', e)

    return None


def get_tile_steps(fx, idx, len_idx, limit_idx, start, stop, max_input_len):
    if start + max_input_len == stop:
        return [0], [max_input_len], [0]
    layer = [1 for _ in range(15)]
    layer[limit_idx] = stop

    layer[idx], layer[len_idx] = start, stop
    total_output_len = max(1, fx(layer)[len_idx])

    steps = []
    strides = []
    osteps = []

    i = start
    while 1:
        # find maximum output length possible
        input_len = max_input_len
        layer[idx], layer[len_idx] = i, input_len
        max_output_len = fx(layer)[len_idx] 
        assert(max_output_len > 0)

        # shrink if maximum output length exceeds total output length
        if (fx(layer)[idx] + max_output_len) > total_output_len:
            max_output_len = total_output_len - fx(layer)[idx]
            input_len = 0
            layer[len_idx] = input_len
            while (fx(layer)[len_idx] != max_output_len):
                if((fx(layer)[len_idx])>max_output_len):
                    break
                input_len += 1
                layer[len_idx] = input_len

        # find smallest input length to produce output length
        while (fx(layer)[len_idx] == max_output_len):
            input_len -= 1
            layer[len_idx] = input_len
        else:
            input_len += 1
            layer[len_idx] = input_len

        steps.append(i)
        strides.append(input_len)
        osteps.append(fx(layer)[idx])

        if (fx(layer)[idx] + max_output_len) == total_output_len:
            break

        # find next step
        next_output_idx = fx(layer)[idx] + fx(layer)[len_idx]
        step = 0
        while((fx(layer)[idx]) != next_output_idx):
            if((fx(layer)[idx])>next_output_idx):
                break
            step += 1
            layer[idx] = i + step
        i += step

    return steps, strides, osteps


def valid_conv_rows(tile, conv_rows, node, preset, sparse=None):
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    batch, imaps, irows, icols, maps, rows, cols = tile
    input_shaper_size, _ = _get_input_and_weight_shaper_sizes(filter_copies, parallel_output_maps, sparse)

    if node.Conv2DOptions.use_strided:
        sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
        srows = conv_rows // sh
        if conv_rows % sh:
            srows += 1
        scols = cols // sw
        if cols % sw:
            scols += 1
        elements = srows * scols

        if elements*sh*sw*imaps > (input_shaper_size / 2):
            return False
    else:
        pw = node.Conv2DOptions.padding_width
        pcols  = cols  + pw

        if imaps*pcols*conv_rows > (input_shaper_size / 2):
            return False

    return True

'''
batch   b_start  b_stop  b_inc
input   i_start  i_stop  i_inc
row     y_start  y_stop  y_inc y_rows0 y_inc0 y_rows_final
col     x_start  x_stop  x_inc x_cols0 x_inc0 x_cols_final
map     m_start  m_stop  m_inc
'''
def set_tile_attr(node, vector_lanes, sp, tile, cores=1, sparse=None):
    v_out, v_tmp0, v_tmp  = max_scratchpad_required(node, vector_lanes, tile, sparse=sparse)
    node.scratchpad_bytes = v_out

    # if v_out > 32*1024 or v_tmp0 > 32*1024 or v_tmp > 32*1024:
    #     print(v_out, v_tmp0, v_tmp, tile, node.type)
    # node.scratchpad_bytes = 32*1024

    fx = compose_subgraph(node, adjust_tile)
    batch, imaps, irows, icols, maps, rows, cols = tile

    shape = node.tensor_array[0].shape
    if len(shape) < 5:
        shape = [1 for _ in range(5 - len(shape))] + list(shape)
    o_start, b_start, m_start, y_start, x_start = [0,0,0,0,0]
    o_stop, b_stop, m_stop, y_stop, x_stop = shape[-5:]

    y_stop = node.m
    x_stop = node.n
    m_stop = node.channels


    node.maps = maps


    e = node.type
    if e in [BuiltinOperator.CONV_2D, BuiltinOperator.DEPTHWISE_CONV_2D, BuiltinOperator.TRANSPOSE_CONV]:
        m_stop = node.Conv2DOptions.kernels
        _, _, kh, kw = node.Conv2DOptions.filter_shape_dims
        sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
        node.Conv2DOptions.imaps = imaps
        use_strided = node.Conv2DOptions.use_strided
    elif e == BuiltinOperator.FULLY_CONNECTED:
        node.FullyConnectedOptions.input_stride = icols
        output_depth, accum_depth = node.FullyConnectedOptions.filter_shape_dims
        x_stop = output_depth
    if e in [BuiltinOperator.SPLIT_V, BuiltinOperator.SPLIT]:
        if node.SplitOptions.axis == -3:
            m_stop = node.SplitOptions.max_split
            if node.maps > m_stop:
                node.maps = m_stop
        if node.SplitOptions.axis == -2:
            y_stop = node.SplitOptions.max_split
        if node.SplitOptions.axis == -1:
            x_stop = node.SplitOptions.max_split
    # elif e in [BuiltinOperator.MEAN, BuiltinOperator.SUM, BuiltinOperator.REDUCE_PROD, BuiltinOperator.REDUCE_MAX, BuiltinOperator.REDUCE_MIN]:
    #     x_stop = 1
    #     y_stop = 1

    o_idx, b_idx, c_idx, m_idx, n_idx = 0, 1, 2, 3, 4
    o_step_idx, b_step_idx, maps_idx, rows_idx, cols_idx = 5, 6, 7, 8, 9
    outer_idx, batches_idx, channels_idx, y_idx, x_idx = 10, 11, 12, 13, 14
    
    y, r, oy = get_tile_steps(fx, y_idx, rows_idx, m_idx, y_start, y_stop, rows)
    x, c, ox = get_tile_steps(fx, x_idx, cols_idx, n_idx, x_start, x_stop, cols)

     #print match input tiles to output size
    if DEBUG_TILE_MAPPING:
        try:
            adjust_fn = fx
            print('TILE MAPPING (input -> output)')
            for yi, ystart in enumerate(y):
                ylen = r[yi]
                for xi, xstart in enumerate(x):
                    xlen = c[xi]
                    # build layer in the expected 15-field format
                    layer = [1, 1, 1, node.m, node.n, 1, 1, maps, ylen, xlen, 1, 1, node.channels, ystart, xstart]
                    out = adjust_fn(layer)
                    out_y = out[13]; out_x = out[14]; out_r = out[8]; out_col = out[9]
                    print(f'in y=[{ystart},{ystart+ylen-1}] x=[{xstart},{xstart+xlen-1}] -> out y=[{out_y},{out_y+out_r-1}] x=[{out_x},{out_x+out_col-1}]')
        except Exception:
            pass

    y_diff = [a-b for a,b in zip(y[1:], y[:-1])]
    x_diff = [a-b for a,b in zip(x[1:], x[:-1])]
    node.row_start = y_start
    node.row_inc = node.row_inc0 = y_stop
    if len(y_diff):
        node.row_inc = y_diff[-1]
        node.row_inc0 = y_diff[0]

    node.rows_0 = node.rows = node.rows_final = r[0]
    if len(r) > 1:
        node.rows = r[1]
        node.rows_final = r[-1]
    node.row_last = y[-1]
    node.orow_last = oy[-1]


    node.col_start = x_start
    node.col_inc = node.col_inc0 = x_stop
    if len(x_diff):
        node.col_inc = x_diff[-1]
        node.col_inc0 = x_diff[0]

    node.cols_0 = node.cols = node.cols_final = c[0]
    if len(c) > 1:
        node.cols = c[1]
        node.cols_final = c[-1]
    node.col_last = x[-1]
    node.ocol_last = ox[-1]

'''
per layer:
    have a function that when given the dimensions of an input tile
    returns the dimensions of an output tile
'''
def adjust_tile(node):
    default = lambda l: l

    # for a given input tile size, output tile size will be same
    default_types = [
                     BuiltinOperator.ABS,
                     BuiltinOperator.RSQRT,
                     BuiltinOperator.NEG,
                     BuiltinOperator.RELU,
                     BuiltinOperator.RELU6,
                     BuiltinOperator.RELU_N1_TO_1,
                     BuiltinOperator.RELU_0_TO_1,
                     BuiltinOperator.PRELU,
                     BuiltinOperator.LEAKY_RELU,
                     BuiltinOperator.CONCATENATION,
                     BuiltinOperator.MUL,
                     BuiltinOperator.ADD,
                     BuiltinOperator.SUB,
                     BuiltinOperator.SQUARED_DIFFERENCE,
                     BuiltinOperator.DEQUANTIZE,
                     BuiltinOperator.SPLIT,
                     BuiltinOperator.SPLIT_V,
                     BuiltinOperator.SLICE,
                    #  BuiltinOperator.STRIDED_SLICE,
                     BuiltinOperator.SOFTMAX,
                     BuiltinOperator.TANH,
                     BuiltinOperator.HARD_SWISH,
                     BuiltinOperator.ELU,
                     BuiltinOperator.GELU,
                     BuiltinOperator.EXP,
                     BuiltinOperator.LOG,
                     BuiltinOperator.LOGISTIC,
                     BuiltinOperator.GREATER,
                     BuiltinOperator.GREATER_EQUAL,
                     BuiltinOperator.LESS,
                     BuiltinOperator.LESS_EQUAL,
                     BuiltinOperator.EQUAL,
                     BuiltinOperator.NOT_EQUAL,
                     BuiltinOperator.MINIMUM,
                     BuiltinOperator.MAXIMUM,
                     BuiltinOperator.CAST,
                     VNNXOperator.IDENTITY,
                     VNNXOperator.ELTWISE,
                     VNNXOperator.LUT,
                     # placeholders below
                     BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,
                     BuiltinOperator.UNPACK,
                     BuiltinOperator.GATHER,
                     BuiltinOperator.DEPTH_TO_SPACE,
                     BuiltinOperator.SPACE_TO_DEPTH,
                     BuiltinOperator.BATCH_TO_SPACE_ND,
                     BuiltinOperator.SPACE_TO_BATCH_ND,
                     ]

    e = node.type
    if e in default_types:
        return default
    elif e in [BuiltinOperator.CONV_2D, BuiltinOperator.DEPTHWISE_CONV_2D]:
        _, _, kh, kw = node.Conv2DOptions.filter_shape_dims
        sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
        dhf, dwf = node.Conv2DOptions.dilation_height_factor, node.Conv2DOptions.dilation_width_factor
        ph, pw = node.Conv2DOptions.padding_height, node.Conv2DOptions.padding_width
        pl, pr, pu, pd = pw // 2, (pw // 2) + (pw % 2), ph // 2, (ph // 2) + (ph % 2)


        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer

            last_x = x + col >= n
            last_y = y + r >= m
            pl_, pr_, pu_, pd_ = pl, pr, pu, pd
            if x != 0:
                pl_ = 0
                x += pl
            if not last_x:
                pr_ = 0
            if y != 0:
                pu_ = 0
                y += pu
            if not last_y:
                pd_ = 0
                
            n = (n + pw - (1 + (kw-1)*dwf)) // sw + 1;
            col = (col + (pl_+pr_) - (1 + (kw-1)*dwf)) // sw + 1;
            m = (m + ph - (1 + (kh-1)*dhf)) // sh + 1;
            r = (r + (pu_+pd_) - (1 + (kh-1)*dhf)) // sh + 1;
            y = y // sh
            x = x // sw

            return o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x


        return adjust
    elif e == BuiltinOperator.FULLY_CONNECTED:
        output_depth, accum_depth = node.FullyConnectedOptions.filter_shape_dims

        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer

            n = output_depth

            return o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x
        return adjust
    elif e == BuiltinOperator.BATCH_MATMUL:
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer

            return o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x
        return adjust
    elif e == BuiltinOperator.TRANSPOSE_CONV:
        
        _, _, kh, kw = node.Conv2DOptions.filter_shape_dims
        sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
        dhf, dwf = node.Conv2DOptions.dilation_height_factor, node.Conv2DOptions.dilation_width_factor
        ph, pw = node.Conv2DOptions.padding_height, node.Conv2DOptions.padding_width

        # TODO for vector code
        if not node.Conv2DOptions.use_fia:
            pass

        else: # accelerator
            pl = kw - (pw // 2) - 1
            pr = kw - ((pw // 2) + (pw % 2)) - 1
            pu = kh - (ph // 2) - 1
            pd = kh - ((ph // 2) + (ph % 2)) - 1


            def adjust(layer):
                o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer

                last_x = x + col >= n
                last_y = y + r >= m
                pl_, pr_, pu_, pd_ = pl, pr, pu, pd
                if x != 0:
                    pl_ = 0
                if not last_x:
                    pr_ = 0
                if y != 0:
                    pu_ = 0
                if not last_y:
                    pd_ = 0

                n = n + pl + pr - kw + 1 + ((sw-1) * (n-1))
                col = col + pl_ + pr_ - kw + 1 + ((sw-1) * col)
                if last_x: 
                    col -= (sw - 1)
                x *= sw
                if x != 0:
                    x += pl

                m = m + pu + pd - kh + 1 + ((sh-1) * (m-1))
                r = r + pu_ + pd_ - kh + 1 + ((sh-1) * r)
                if last_y:
                    r -= (sh - 1)
                y *= sh
                if y != 0:
                    y += pu

                return o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x

        return adjust
    elif e in [BuiltinOperator.MEAN, BuiltinOperator.SUM, BuiltinOperator.REDUCE_PROD, BuiltinOperator.REDUCE_MAX, BuiltinOperator.REDUCE_MIN]:
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer
            return o, b, c, 1, 1, o_step, b_step, maps, 1, 1, outer, batches, channels, y, x
        return adjust
    elif e in [BuiltinOperator.ARG_MAX, BuiltinOperator.ARG_MIN, BuiltinOperator.TOPK_V2]:
        axis = node.reduce8.axis
        assert(axis == -3)
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer
            if axis == -3:
                return o, b, 1, m, n, o_step, b_step, 1, r, col, outer, batches, channels, y, x
            else:
                return layer
        return adjust
    elif e in [BuiltinOperator.MAX_POOL_2D, BuiltinOperator.AVERAGE_POOL_2D]:
        kh, kw = node.kernel_shape
        sh, sw = node.strides

        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer

            pads = node.pads.copy()
            last_x = x + col >= n
            last_y = y + r >= m
            last_set_of_maps = c + maps >= channels
            if y != 0:
                pads[1] = 0
                y += node.pads[1]
            if not last_y:
                pads[4] = 0
            if x != 0:
                pads[2] = 0
                x += node.pads[2]
            if not last_x:
                pads[5] = 0
            if not last_set_of_maps:
                pads[3] = 0
                
            m += node.pads[1] + node.pads[4]
            n += node.pads[2] + node.pads[5]
            r += pads[1] + pads[4]
            col += pads[2] + pads[5]
            maps += node.pads[0] + node.pads[3]


            m = (m - kh) // sh + 1
            r = (r - kh) // sh + 1
            n = (n - kw) // sw + 1
            col = (col - kw) // sw + 1
            y = y // sh
            x = x // sw

            return o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x
        return adjust
    elif e in [BuiltinOperator.RESIZE_NEAREST_NEIGHBOR, BuiltinOperator.RESIZE_BILINEAR]:
        sh, sw = node.ResizeOptions.scale
        
        # Manage overlaping between tiles
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer

            if e == BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
                y= floor(y*sh)
                x=floor(x*sw)
                r=floor(r*sh)
                col=floor(col*sw)
            # for each min input tile, we return th output tile size
            elif e == BuiltinOperator.RESIZE_BILINEAR:
                if (sh==2.0 and sw==2.0) or (sh==4.0 and sw==4.0):
                    y= floor(y*sh)
                    x=floor(x*sw)
                    r=floor((r-1)*sh)
                    col=floor((col-1)*sw)
                else:
                    y= floor(y*sh)
                    x=floor(x*sw)                    
                    r=floor((r)*sh)
                    col=floor((col)*sw)

            return o, b, c, m*sh, n*sw, o_step, b_step, maps, r, col, outer, batches, channels, y, x
        return adjust
    elif e in [BuiltinOperator.PAD, BuiltinOperator.MIRROR_PAD, BuiltinOperator.DILATE]:
        dhf = 0
        dwf = 0
        if e in [BuiltinOperator.PAD, BuiltinOperator.DILATE]:
            dh, dw = node.PadOptions.transpose_dilate_h, node.PadOptions.transpose_dilate_w
            dhf = dh - 1
            dwf = dw - 1

        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer

            pads = node.pads.copy()
            last_x = x + col >= n
            last_y = y + r >= m
            last_set_of_maps = c + maps >= channels
            if y != 0:
                pads[1] = 0
            if not last_y:
                pads[4] = dhf
            if x != 0:
                pads[2] = 0
            if not last_x:
                pads[5] = dwf
            if not last_set_of_maps:
                pads[3] = 0
            
            # Only Transpose dilated pad uses the below adjusments (dhf==dwf==0 for other pads)
            m += (dhf * (m-1))
            n += (dwf * (n-1))
            if y != 0:
                y += (dhf * y)
            if x != 0:
                x += (dwf * x)
            r += (dhf * (r-1))
            col += (dwf * (col-1))

            # all pads use the below adjustments
            if y != 0:
                y += node.pads[1]
            if x != 0:
                x += node.pads[2]
            m += node.pads[1] + node.pads[4]
            n += node.pads[2] + node.pads[5]
            r += pads[1] + pads[4]
            col += pads[2] + pads[5]
            maps += node.pads[0] + node.pads[3]

            return o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x
        return adjust
    elif e == BuiltinOperator.PACK:
        axis = node.PackOptions.axis
        count = node.PackOptions.count
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer

            if axis == -4:
                batches *= count
                b_step *= count
            elif axis == -3:
                channels *= count
                maps *= count
            elif axis == -2:
                y *= count
                r *= count
            elif axis == -2:
                x *= count
                col *= count

            return o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x
        return adjust
    elif e == BuiltinOperator.TRANSPOSE:
        permutation = node.TransposeOptions.permutation
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer
            if permutation == [0,2,1]: # maps, cols, rows
                return o, b, c, n, m, o_step, b_step, maps, col, r, outer, batches, channels, x, y
            elif permutation == [1,0,2]: # rows, maps, cols
                return o, b, m, c, n, o_step, b_step, r, maps, col, outer, batches, y, channels, x
            elif permutation == [1,2,0]: # rows, cols, maps
                return o, b, m, n, c, o_step, b_step, r, col, maps, outer, batches, y, x, channels 
            elif permutation == [2,0,1]: # cols, maps, rows
                return o, b, n, c, m, o_step, b_step, col, maps, r, outer, batches, x, channels, y
            elif permutation == [2,1,0]: # cols, rows, maps
                return o, b, n, m, c, o_step, b_step, col, r, maps, outer, batches, x, y, channels
            else:
                assert(permutation == [0,1,2])
                return o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x
        return adjust
    elif e == BuiltinOperator.RESHAPE:
        mode = node.ReshapeOptions.mode
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer
            if mode == 0: # flatten
                return 1, 1, o, b, c*m*n, 1, 1, o_step, b_step, maps*r*col, 0, 0, outer, batches, channels*m*n+y*n+x
            elif mode == 1: # n,h,w,c-> n,h,w*c or n,c,h,w -> c*w,n,h
                pass
            elif mode == 2: # n,1,w,c -> n,w,c,1 or n,c,1,w -> n,1,w,c
                pass
            elif mode == 3: # n,c,h*w -> n,c,h,w
                pass

            return o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x
        return adjust
    elif e == VNNXOperator.PIXEL_SHUFFLE:
        r = node.PixelShuffleOptions.r
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, row, col, outer, batches, channels, y, x = layer
            return o, b, c // (r**2), m*r, n*r, o_step, b_step, maps // (r**2), row*r, col*r, outer, batches, channels // (r**2), y*r, x*r
        return adjust
    elif e == BuiltinOperator.TILE:
        tile_n, tile_c, tile_h, tile_w = node.TileOptions.tile
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer
            return o, b*tile_n, c*tile_c, m*tile_h, n*tile_w, o_step, b_step*tile_n, maps*tile_c, r*tile_h, col*tile_w, outer, batches*tile_n, channels*tile_c, y*tile_h, x*tile_w
        return adjust
    
    elif e == BuiltinOperator.STRIDED_SLICE:
        stride = node.SliceOptions.stride
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer
            return o, b, c, m, n, o_step, b_step, maps, int(r/stride[-2]), int(col/stride[-1]), outer, batches, channels, int(y/stride[-2]), int(x/stride[-1])
        return adjust

    elif e == VNNXOperator.UNKNOWN:
        pass

    return None


def valid_sp(maps, imaps, rows, cols, node, preset, sparse=None):
    sp = preset_select['SCRATCHPAD_KB'][preset]*1024 - 256 #TODO get exact indirect
    vector_lanes = preset_select['VECTOR_LANES'][preset]

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    tile = np.array([1,1,1,1,1,1,1])
    tile[MAPS] = maps
    tile[IMAPS] = imaps
    tile[ROWS] = rows
    tile[COLUMNS] = cols

    return not is_sp_invalid(node, vector_lanes, sp, tile, sparse=sparse)


def valid_fia_fc(accum_depth, rows, cols, preset, sparse=None, no_sublayers=True):
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]
    sp = preset_select['SCRATCHPAD_KB'][preset]*1024 - 256 #TODO get exact indirect

    osh = rows
    osw = cols # output_depth
    iw = accum_depth

    # Get the input and weight shaper sizes
    # using a single bank for FC, so measure against bank sizes
    input_shaper_size, weight_shaper_size = _get_input_and_weight_shaper_bank_sizes(filter_copies, parallel_output_maps, sparse)
    input_shaper_buffer_size = iw*rows
    weight_shaper_buffer_size = accum_depth*cols
    # account for padded weights for alignment
    weight_shaper_buffer_size += (parallel_output_maps - (weight_shaper_buffer_size % parallel_output_maps)) if (weight_shaper_buffer_size % parallel_output_maps) else 0

    if input_shaper_buffer_size > input_shaper_size:
        return False
    if weight_shaper_buffer_size > weight_shaper_size:
        return False
    if cols*QUANTIZATION_RECORD_WIDTH_BYTES > (fia_quantization_shaper_size_kb()*1024): # cols*sizeof(quantization_record)
        return False
    if osh*osw > sp_output_shaper_size_b(sp, no_sublayers): # output_shaper_size
        return False

    return True


def valid_fia(maps, imaps, rows, cols, node, preset, use_db=None, db_rows=None, split_weights=None, sparse=None):
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]
    sp = preset_select['SCRATCHPAD_KB'][preset]*1024 - 256 #TODO get exact indirect
    no_sublayers = (len(node.subnode_array) == 0 and node.output_strides == [1,1])

    k, c, kh, kw = node.Conv2DOptions.filter_shape_dims
    sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
    dhf, dwf = node.Conv2DOptions.dilation_height_factor, node.Conv2DOptions.dilation_width_factor
    ph, pw = node.Conv2DOptions.padding_height, node.Conv2DOptions.padding_width
    if split_weights is None:
        split_weights = node.Conv2DOptions.split_weight_shaper_buffers
    if use_db is None:
        use_db = node.Conv2DOptions.use_db
    if db_rows is None:
        db_rows = node.Conv2DOptions.conv_rows

    pl = pw // 2
    pr = (pw // 2) + (pw % 2)
    pu = ph // 2
    pd = (ph // 2) + (ph % 2)
    pcols = cols + pl
    if cols == node.n:
        pcols += pr
    prows = rows + pu
    if rows == node.m:
        prows += pd

    dh = ((kh-1)*dhf) + 1
    dw = ((kw-1)*dwf) + 1
    orows = (prows - dh) // sh + 1
    ocols = (pcols - dw) // sw + 1

    input_shaper_banks = get_input_shaper_banks(sparse)

    if node.type == BuiltinOperator.TRANSPOSE_CONV:
        pl = kw - (pw // 2) - 1
        pr = kw - ((pw // 2) + (pw % 2)) - 1
        pu = kh - (ph // 2) - 1
        pd = kh - ((ph // 2) + (ph % 2)) - 1

        # TODO remove width adjustment when PACK sublayer injection working
        pcols = cols + pl
        pcols += (cols * (sw-1))
        # if full cols, then adjust dilated input and add the pad along right of input
        if cols == node.n:
            pcols -= (sw-1)
            pcols += pr

        prows = rows + pu
        prows += (rows * (sh-1))
        # if full rows, then adjust dilated input and add the pad along bottom of input
        if rows == node.m:
            prows -= (sh-1)
            prows += pd

        orows = prows - dh + 1
        ocols = pcols - dw + 1

    srows = prows-(dh-1)
    scols = pcols
    if scols < dw :
        scols = dw
    elements = ((prows-(dh-1))*pcols)-(dw-1)

    weight_pad = srows
    if parallel_output_maps - 1 <= srows:
        weight_pad = parallel_output_maps - 1

    # Get the input and weight shaper sizes
    input_shaper_size, weight_shaper_size = _get_input_and_weight_shaper_sizes(filter_copies, parallel_output_maps, sparse)

    if node.Conv2DOptions.use_depthwise:
        # using a single bank for depthwise, so measure against bank sizes
        input_shaper_size, weight_shaper_size = _get_input_and_weight_shaper_bank_sizes(filter_copies, parallel_output_maps, sparse)
        weight_shaper_buffer_size = imaps * kw * ((kh+parallel_output_maps-1)*parallel_output_maps)
        if not use_db:
            if imaps*pcols*prows > input_shaper_size: #input_shaper_size
                return False
            if split_weights:
                if weight_shaper_buffer_size > (weight_shaper_size / 2): #weight_shaper_size
                    return False
            else:
                if weight_shaper_buffer_size > weight_shaper_size: #weight_shaper_size
                    return False
        elif use_db:
            if imaps*pcols*prows > (input_shaper_size / 2): #input_shaper_size
                return False
            if weight_shaper_buffer_size > (weight_shaper_size / 2): #weight_shaper_size
                return False
        if maps * QUANTIZATION_RECORD_WIDTH_BYTES > (fia_quantization_shaper_size_kb() * 1024): # quantization_shaper_size maps*sizeof(quantization_record))
            return False
        if maps*scols*srows > sp_output_shaper_size_b(sp, no_sublayers): #output_shaper_size
            return False
    else:
        if node.Conv2DOptions.use_strided:
            srows = rows // sh
            if rows % sh:
                srows += 1
            scols = cols // sw
            if cols % sw:
                scols += 1
            elements = srows * scols
    
        input_imaps = imaps
        # input shaper check needs to account for potential additional channel per bank
        # this line is rounding it up to even if non-sparse, or to multiple of 8 if sparse
        input_imaps = imaps + (input_shaper_banks - ((imaps % input_shaper_banks) if (imaps % input_shaper_banks) else input_shaper_banks))
        if not db_rows:
            if not node.Conv2DOptions.use_strided: # input_shaper_size
                if use_db:
                    if input_imaps*pcols*prows > (input_shaper_size / 2):
                        return False
                else:
                    if input_imaps*pcols*prows > input_shaper_size:
                        return False
            else:
                if use_db:
                    if elements*sh*sw*input_imaps > (input_shaper_size / 2):
                        return False
                else:
                    if elements*sh*sw*input_imaps > input_shaper_size:
                        return False

        # weight shaper check needs to account for weights channel padding and kernel padding and compression
        weight_maps = maps
        weight_imaps = imaps
        if (sparse == 1):
            weight_imaps = imaps + (8 - ((imaps % 8) if (imaps % 8) else 8))   
            if node.Conv2DOptions.repeat == 2:
                weight_imaps //= 2
            elif node.Conv2DOptions.repeat == 1:
                weight_imaps //= 4
        else:
            weight_imaps = imaps + (imaps%2)

        weight_maps += (parallel_output_maps - ((maps % parallel_output_maps) if (maps % parallel_output_maps) else parallel_output_maps))
        if use_db:
            if weight_maps*kh*kw*weight_imaps > (weight_shaper_size / 2): # weight_shaper_size
                return False
        elif split_weights:
            if weight_maps*kh*kw*weight_imaps > (weight_shaper_size / 2): # weight_shaper_size
                return False
        else:
            if weight_maps*kh*kw*weight_imaps > weight_shaper_size: # weight_shaper_size
                return False
        if maps * QUANTIZATION_RECORD_WIDTH_BYTES > (fia_quantization_shaper_size_kb() * 1024): # quantization_shaper_size maps*sizeof(quantization_record))            
            return False
        if not FIA_WRITE_DB:
            if maps*scols*srows > sp_output_shaper_size_b(sp, no_sublayers): # output_shaper_size
                return False
        elif FIA_WRITE_DB:
            if 2*maps_per_pass*scols*srows > sp_output_shaper_size_b(sp, no_sublayers): # output_shaper_size
                return False

    # needs to check against individual bank sizes as well for ishaper and wshaper for non-depthwise
    if not node.Conv2DOptions.use_depthwise:
        input_shaper_bank_size, weight_shaper_bank_size = _get_input_and_weight_shaper_bank_sizes(filter_copies, parallel_output_maps, sparse)
        
        assert(input_imaps % input_shaper_banks == 0)
        imaps_per_in_bank = (input_imaps//input_shaper_banks)
        if not db_rows:
            if not node.Conv2DOptions.use_strided:
                if use_db:
                    if imaps_per_in_bank*pcols*prows > (input_shaper_bank_size / 2):
                        return False
                else:
                    if imaps_per_in_bank*pcols*prows > input_shaper_bank_size:
                        return False
            else:
                if use_db:
                    if elements*sh*sw*imaps_per_in_bank > (input_shaper_bank_size / 2):
                        return False
                else:
                    if elements*sh*sw*imaps_per_in_bank > input_shaper_bank_size:
                        return False

        assert(weight_imaps % WEIGHT_SHAPER_DATA_BANKS == 0)
        imaps_per_weight_bank 	= weight_imaps // WEIGHT_SHAPER_DATA_BANKS
        if use_db:
            if weight_maps*kh*kw*imaps_per_weight_bank > (weight_shaper_bank_size / 2):
                return False
        elif split_weights:
            if weight_maps*kh*kw*imaps_per_weight_bank > (weight_shaper_bank_size / 2):
                return False
        else:
            if weight_maps*kh*kw*imaps_per_weight_bank > weight_shaper_bank_size:
                return False

    return True


def fit_full_rows(tile, node, vl, sp, fx, preset, sparse=None):
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    target = tile.copy()
    t = tile.copy()

    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, sparse=sparse)
    if not v(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
        return False

    max_maps = node.Conv2DOptions.filter_shape_dims[0]
    if max_maps > parallel_output_maps:
        max_maps = parallel_output_maps

    target[MAPS] = max_maps
    target[IMAPS] = -1
    target[ROWS] = -1
    target[COLUMNS] = node.n

    return fit_target(target, t, v, node, vl, sp, fx, sparse=sparse)


def fit_all_minus_rows(tile, node, vl, sp, fx, preset, sparse=None):
    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    target = tile.copy()
    t = tile.copy()

    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, sparse=sparse)
    if not v(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
        return False, t

    max_maps = node.Conv2DOptions.filter_shape_dims[0]

    target[MAPS] = max_maps
    target[IMAPS] = node.Conv2DOptions.filter_shape_dims[1]
    target[ROWS] = -1
    target[COLUMNS] = node.n

    return fit_target(target, t, v, node, vl, sp, fx, sparse=sparse)


def fit_all_omaps(tile, node, vl, sp, fx, preset, use_db=None, db_rows=None, split_weights=None, sparse=None):
    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    target = tile.copy()
    t = tile.copy()

    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, use_db, db_rows, split_weights, sparse=sparse)
    if not v(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
        return False, t

    max_maps = node.Conv2DOptions.filter_shape_dims[0]

    target[MAPS] = max_maps
    target[IMAPS] = -1
    target[ROWS] = -1
    target[COLUMNS] = -1

    return fit_target(target, t, v, node, vl, sp, fx, sparse=sparse)


def fit_all_imaps(tile, node, vl, sp, fx, preset, use_db=None, db_rows=None, split_weights=None, sparse=None):
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    target = tile.copy()
    t = tile.copy()

    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, use_db, db_rows, split_weights, sparse=sparse)
    if not v(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
        return False, t

    max_maps = node.Conv2DOptions.filter_shape_dims[0]
    if max_maps > parallel_output_maps:
        max_maps = parallel_output_maps

    target[MAPS] = max_maps
    target[IMAPS] = node.Conv2DOptions.filter_shape_dims[1]
    target[ROWS] = -1
    target[COLUMNS] = node.n 

    return fit_target(target, t, v, node, vl, sp, fx, sparse=sparse)


def fit_full_maps(tile, node, vl, sp, fx, preset, use_db=None, db_rows=None, split_weights=None, sparse=None):
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    target = tile.copy()
    t = tile.copy()

    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, use_db, db_rows, split_weights, sparse=sparse)
    if not v(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
        return False, t

    max_maps = node.Conv2DOptions.filter_shape_dims[0]
    if max_maps > parallel_output_maps:
        max_maps = parallel_output_maps

    target[MAPS] = max_maps
    target[IMAPS] = -1
    target[ROWS] = node.m
    target[COLUMNS] = node.n


    return fit_target(target, t, v, node, vl, sp, fx, sparse=sparse)


def _decrement_imaps(imaps, channels, sparse=None):
    if (sparse == 1): # need multiple of 8 channels for sparsity (unless there are less than 8 channels)
        imaps -= (imaps % 8) if (imaps % 8) else 8
        if channels < 8:
            return channels
        elif imaps > 8:
            return imaps
        else:
            return 8
    
    else: # multiple of 2 channels for non-sparse
        imaps -= (imaps % 2) if (imaps % 2) else 2
        return imaps if imaps > 0 else 1


def fit_target(target, t, v, node, vl, sp, fx, sparse=None):
    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6

    if target[MAPS] != -1:
        limit_maps = target[MAPS]
        while limit_maps > t[MAPS] and not v(limit_maps,t[IMAPS],t[ROWS],t[COLUMNS]):
            limit_maps -= 1
        if limit_maps != target[MAPS]:
            return False, t

        t = increment_tile(node, vl, sp, fx, t, MAPS, limit=limit_maps, sparse=sparse)
        if t[MAPS] != target[MAPS]:
            return False, t

    if target[COLUMNS] != -1:
        limit_cols = target[COLUMNS]
        while limit_cols > t[COLUMNS] and not v(t[MAPS],t[IMAPS],t[ROWS],limit_cols):
            limit_cols -= 1
        if limit_cols != target[COLUMNS]:
            return False, t

        t = increment_tile(node, vl, sp, fx, t, COLUMNS, limit=limit_cols, sparse=sparse)
        if t[COLUMNS] != target[COLUMNS]:
            return False, t

    if target[ROWS] != -1:
        limit_rows = target[ROWS]
        while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
            limit_rows -= 1
        if limit_rows != target[ROWS]:
            return False, t

        t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows, sparse=sparse)
        if t[ROWS] != target[ROWS]:
            return False, t

    if target[IMAPS] != -1:
        limit_imaps = target[IMAPS]
        while limit_imaps > t[IMAPS] and not v(t[MAPS],limit_imaps,t[ROWS],t[COLUMNS]):
            limit_imaps = _decrement_imaps(limit_imaps, node.channels)
        if limit_imaps != target[IMAPS]:
            return False, t

        t = increment_tile(node, vl, sp, fx, t, IMAPS, limit=limit_imaps, sparse=sparse)
        if t[IMAPS] != target[IMAPS]:
            return False, t

    return True, t


def fit_depthwise(tile, node, vl, sp, fx, preset, use_db=None, row_db=None, sparse=None):
    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, use_db, row_db, sparse=sparse)
    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    t = tile.copy()
    

    limit_cols = node.n # adjust COLUMN limit to be maximum valid fia COLUMN limit
    while limit_cols > t[COLUMNS] and not v(t[MAPS],t[IMAPS],t[ROWS],limit_cols):
        limit_cols -= 1
    t = increment_tile(node, vl, sp, fx, t, COLUMNS, limit=limit_cols, sparse=sparse)

    # max rows
    limit_rows = node.m
    while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
        limit_rows -= 1
    t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows, sparse=sparse)

    # max maps
    while True:
        next_maps = t[MAPS] + 1
        if not v(next_maps,t[IMAPS],t[ROWS],t[COLUMNS]):
            break
        next_tile = increment_tile(node, vl, sp, fx, t, MAPS, limit=next_maps, sparse=sparse)
        if next_tile[MAPS] == next_maps:
            t = next_tile
        else:
            break

        if t[MAPS] == node.Conv2DOptions.filter_shape_dims[1]:
            break

    # max imaps per buffer
    while True:
        next_imaps = t[IMAPS] + 1
        if not v(t[MAPS], next_imaps, t[ROWS], t[COLUMNS]):
            break
        next_tile = increment_tile(node, vl, sp, fx, t, IMAPS, limit=next_imaps, sparse=sparse)
        if next_tile[IMAPS] == next_imaps:
            t = next_tile
        else:
            break

        if t[IMAPS] == node.Conv2DOptions.filter_shape_dims[1]:
            break        

    # get an even split of imaps for DMAs if possible
    iterations = ceil(t[MAPS] / t[IMAPS])
    if t[MAPS] % iterations == 0:
        t[IMAPS] = t[MAPS] // iterations

    return True, t


def fit_conv(tile, node, vl, sp, fx, preset, use_db=None, row_db=None, split_weights=None, sparse=None):
    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, use_db, row_db, split_weights, sparse=sparse)
    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    t = tile.copy()

    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    # for conv rows DB, min chunk of rows has to be a factor of the dilated filter height
    # when strided, possible chunk of rows has to be a factor of the stride_height with dilated filter height added
    # e.g. dkh, dkh + (1*sh), dkh + (2*sh), etc.
    prows = node.m + node.Conv2DOptions.padding_height
    _, _, kh, _ = node.Conv2DOptions.filter_shape_dims
    dh = node.Conv2DOptions.dilation_height_factor
    dkh = ((kh-1)*dh) + 1

    is_all_imaps_omaps = False
    is_all_imaps_omaps_cols = False

    is_all_imaps, tile_all_imaps =  fit_all_imaps(t, node, vl, sp, fx, preset, use_db, row_db, split_weights, sparse=sparse)
    is_all_omaps, tile_all_omaps =  fit_all_omaps(t, node, vl, sp, fx, preset, use_db, row_db, split_weights, sparse=sparse)
    is_full_maps, tile_full_maps =  fit_full_maps(t, node, vl, sp, fx, preset, use_db, row_db, split_weights, sparse=sparse)

    if is_all_imaps:
        is_all_imaps_omaps, tile_all_imaps_omaps =  fit_all_omaps(tile_all_imaps, node, vl, sp, fx, preset, use_db, row_db, split_weights, sparse=sparse)

    if is_all_imaps_omaps:
        limit_cols = node.n
        is_all_imaps_omaps_cols = v(tile_all_imaps_omaps[MAPS],tile_all_imaps_omaps[IMAPS],tile_all_imaps_omaps[ROWS],limit_cols)

    is_all_omaps_full_maps = False
    if is_all_omaps:
        limit_cols = node.n
        limit_rows = node.m
        is_all_omaps_full_maps = v(tile_all_omaps[MAPS],tile_all_omaps[IMAPS],limit_rows,limit_cols)
        if not valid_sp(tile_all_omaps[MAPS],tile_all_omaps[IMAPS],limit_rows,limit_cols, node,preset):
            is_all_omaps_full_maps = False

    # row DB # 
    # TODO shouldnt need to be constrained like this, need to generalize/determine better rules
    if is_all_imaps_omaps_cols and (use_db and row_db):
        t = tile_all_imaps_omaps

        # max columns
        limit_cols = node.n
        t = increment_tile(node, vl, sp, fx, t, COLUMNS, limit=limit_cols, sparse=sparse)

        # max rows and conv_rows
        limit_rows = node.m
        if node.Conv2DOptions.stride_height != 1:
            sh = node.Conv2DOptions.stride_height
            limit_rows = prows + dkh
            while limit_rows > prows: # get the max possible limit_rows, that is less than prows, that follows dkh + (factor of sh)
                limit_rows -= sh

            while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
                limit_rows -= sh
            t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows, offset=sh, sparse=sparse)

            # max conv_rows
            conv_rows = t[ROWS]
            while conv_rows > dkh and not valid_conv_rows(t, conv_rows, node, preset, sparse=sparse):
                conv_rows -= sh

        else:
            while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
                limit_rows -= 1
            t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows, sparse=sparse)

            # max conv_rows
            conv_rows = t[ROWS]
            while conv_rows > dkh and not valid_conv_rows(t, conv_rows, node, preset, sparse=sparse):
                conv_rows -= 1

        return True, t, conv_rows

    # if row_db and wasn't able to find a tile (i.e. is_all_imaps_omaps_cols wasn't true), exit early instead of proceeding. 
    # This is because row_db allows i_shaper size check to be skipped in valid_fia, which can cause problems for conv_rows tiling if allowed to proceed.
    if row_db:
        return False, t, 0

    elif is_full_maps and use_db:
        conv_rows = 0

        t = tile_full_maps
        t[ROWS] = node.m
        t[COLUMNS] = node.n

        # max DB imaps
        limit_imaps = (node.Conv2DOptions.filter_shape_dims[1] + 1) // 2
        # get limit_imaps to start as a multiple of 8 (sparse) or even (non_sparse), since it is not starting as whole channels
        if ((sparse==1) and limit_imaps % 8 != 0) or ((sparse!=1) and limit_imaps % 2 != 0):
            limit_imaps = _decrement_imaps(limit_imaps, node.channels)
        while not v(t[MAPS],limit_imaps,t[ROWS],t[COLUMNS]):
            limit_imaps = _decrement_imaps(limit_imaps, node.channels)
        t = increment_tile(node, vl, sp, fx, t, IMAPS, limit=limit_imaps, sparse=sparse)

        limit_maps = node.Conv2DOptions.filter_shape_dims[0]
        while limit_maps > t[MAPS] and not v(limit_maps,t[IMAPS],t[ROWS],t[COLUMNS]):
            limit_maps -= 1
        if limit_maps != node.Conv2DOptions.filter_shape_dims[0]:
            limit_maps = limit_maps // parallel_output_maps * parallel_output_maps
        t = increment_tile(node, vl, sp, fx, t, MAPS, limit=limit_maps, sparse=sparse)

        if t[MAPS] != node.Conv2DOptions.filter_shape_dims[0]:
            if t[MAPS] % parallel_output_maps and t[MAPS] > parallel_output_maps:
                t[MAPS] = t[MAPS] // parallel_output_maps * parallel_output_maps


        return True, t, conv_rows

    else: # not all imaps/omaps/cols so disable conv_rows and check tiles again
        conv_rows = 0

        # full maps aviods weights reuse 
        is_full_maps, tile_full_maps =  fit_full_maps(t, node, vl, sp, fx, preset, use_db, row_db, split_weights, sparse=sparse)

        # all input maps avoids input maps reuse
        # also checks if all columns fit
        is_all_imaps, tile_all_imaps =  fit_all_imaps(t, node, vl, sp, fx, preset, use_db, row_db, split_weights, sparse=sparse)

        # all output maps avoid input map reuse TODO
        is_all_omaps, tile_all_omaps =  fit_all_omaps(t, node, vl, sp, fx, preset, use_db, row_db, split_weights, sparse=sparse)
    

        if is_all_imaps:
            # set all_imaps
            t = tile_all_imaps

            # max columns
            limit_cols = node.n # adjust COLUMN limit to be maximum valid fia COLUMN limit
            while limit_cols > t[COLUMNS] and not v(t[MAPS],t[IMAPS],t[ROWS],limit_cols):
                limit_cols -= 1
            t = increment_tile(node, vl, sp, fx, t, COLUMNS, limit=limit_cols, sparse=sparse)

            # max rows
            limit_rows = node.m
            while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
                limit_rows -= 1
            t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows, sparse=sparse)

            # max maps
            limit_maps = node.Conv2DOptions.filter_shape_dims[0]
            while limit_maps > t[MAPS] and not v(limit_maps,t[IMAPS],t[ROWS],t[COLUMNS]):
                limit_maps -= 1
            if limit_maps != node.Conv2DOptions.filter_shape_dims[0]:
                limit_maps = limit_maps // parallel_output_maps * parallel_output_maps
            t = increment_tile(node, vl, sp, fx, t, MAPS, limit=limit_maps, sparse=sparse)

            if t[MAPS] != node.Conv2DOptions.filter_shape_dims[0]:
                if t[MAPS] % parallel_output_maps and t[MAPS] > parallel_output_maps:
                    t[MAPS] = t[MAPS] // parallel_output_maps * parallel_output_maps

            return True, t, conv_rows

        elif is_full_maps:
            # set full_maps
            t = tile_full_maps

            # max imaps
            limit_imaps = node.Conv2DOptions.filter_shape_dims[1]
            while not v(t[MAPS],limit_imaps,t[ROWS],t[COLUMNS]):
                limit_imaps = _decrement_imaps(limit_imaps, node.channels)
            t = increment_tile(node, vl, sp, fx, t, IMAPS, limit=limit_imaps, sparse=sparse)

            # max maps
            limit_maps = node.Conv2DOptions.filter_shape_dims[0]
            while limit_maps > t[MAPS] and not v(limit_maps,t[IMAPS],t[ROWS],t[COLUMNS]):
                limit_maps -= 1
            if limit_maps != node.Conv2DOptions.filter_shape_dims[0]:
                limit_maps = limit_maps // parallel_output_maps * parallel_output_maps
            t = increment_tile(node, vl, sp, fx, t, MAPS, limit=limit_maps, sparse=sparse)

            if t[MAPS] != node.Conv2DOptions.filter_shape_dims[0]:
                if t[MAPS] % parallel_output_maps and t[MAPS] > parallel_output_maps:
                    t[MAPS] = t[MAPS] // parallel_output_maps * parallel_output_maps

            return True, t, conv_rows

        elif is_all_omaps:
            # set all_omaps
            t = tile_all_omaps

            # max columns
            limit_cols = node.n # adjust COLUMN limit to be maximum valid fia COLUMN limit
            while limit_cols > t[COLUMNS] and not v(t[MAPS],t[IMAPS],t[ROWS],limit_cols):
                limit_cols -= 1
            t = increment_tile(node, vl, sp, fx, t, COLUMNS, limit=limit_cols, sparse=sparse)

            # max rows
            limit_rows = node.m
            while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
                limit_rows -= 1
            t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows, sparse=sparse)

            # max imaps
            limit_imaps = node.Conv2DOptions.filter_shape_dims[1]
            while not v(t[MAPS],limit_imaps,t[ROWS],t[COLUMNS]):
                limit_imaps = _decrement_imaps(limit_imaps, node.channels)
            t = increment_tile(node, vl, sp, fx, t, IMAPS, limit=limit_imaps, sparse=sparse)

            # remove overlap from ROWS for accurate product calc and comparison
            def custom_prod(arr, kh):
                arr_copy = arr.copy()
                arr_copy[ROWS] -= (kh-1)
                return np.prod(arr_copy)

            # minimize maps while adding to rows/imaps for growing/maintaining performance efficiency
            parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]
            best_prod = custom_prod(t, kh)
            next_t = t.copy()
            if t[MAPS] > parallel_output_maps:
                if t[MAPS] % parallel_output_maps:
                    next_t[MAPS] -= (t[MAPS] % parallel_output_maps)
                else:
                    next_t[MAPS] -= parallel_output_maps

            # decrement maps while growing rows/imaps, and see if the product becomes larger (replace tile)
            while next_t[MAPS] >= parallel_output_maps:

                # max columns now if unable to do so before
                limit_cols = node.n # adjust COLUMN limit to be maximum valid fia COLUMN limit
                while limit_cols > next_t[COLUMNS] and not v(next_t[MAPS],next_t[IMAPS],next_t[ROWS],limit_cols):
                    limit_cols -= 1
                next_t = increment_tile(node, vl, sp, fx, next_t, COLUMNS, limit=limit_cols, sparse=sparse)

                # grow rows for next_t
                limit_rows = node.m
                while limit_rows > next_t[ROWS] and not v(next_t[MAPS],next_t[IMAPS],limit_rows, next_t[COLUMNS]):
                    limit_rows -= 1
                next_t = increment_tile(node, vl, sp, fx, next_t, ROWS, limit=limit_rows, sparse=sparse)

                # grow imaps for next_t
                limit_imaps = node.Conv2DOptions.filter_shape_dims[1]
                while not v(next_t[MAPS],limit_imaps,next_t[ROWS],next_t[COLUMNS]):
                    limit_imaps = _decrement_imaps(limit_imaps, node.channels)
                next_t = increment_tile(node, vl, sp, fx, next_t, IMAPS, limit=limit_imaps, sparse=sparse)

                if custom_prod(next_t, kh) > best_prod:
                    t = next_t.copy()
                    best_prod = custom_prod(next_t, kh)

                next_t[MAPS] -= parallel_output_maps

            return True, t, conv_rows

        else:
            # can't fit full anything, start with setting tile omaps to parallel maps
            tile[MAPS] = min(node.Conv2DOptions.filter_shape_dims[0], parallel_output_maps)
            t = tile

            # max columns
            limit_cols = node.n # adjust COLUMN limit to be maximum valid fia COLUMN limit
            while limit_cols > t[COLUMNS] and not v(t[MAPS],t[IMAPS],t[ROWS],limit_cols):
                limit_cols -= 1
            t = increment_tile(node, vl, sp, fx, t, COLUMNS, limit=limit_cols, sparse=sparse)

            # max rows
            limit_rows = node.m
            while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
                limit_rows -= 1
            t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows, sparse=sparse)

            # max imaps
            limit_imaps = node.Conv2DOptions.filter_shape_dims[1]
            while not v(t[MAPS],limit_imaps,t[ROWS],t[COLUMNS]):
                limit_imaps = _decrement_imaps(limit_imaps, node.channels)
            t = increment_tile(node, vl, sp, fx, t, IMAPS, limit=limit_imaps, sparse=sparse)

            # max maps
            limit_maps = node.Conv2DOptions.filter_shape_dims[0]
            while limit_maps > t[MAPS] and not v(limit_maps,t[IMAPS],t[ROWS],t[COLUMNS]):
                limit_maps -= 1
            if limit_maps != node.Conv2DOptions.filter_shape_dims[0]:
                limit_maps = limit_maps // parallel_output_maps * parallel_output_maps
            t = increment_tile(node, vl, sp, fx, t, MAPS, limit=limit_maps, sparse=sparse)

            if t[MAPS] != node.Conv2DOptions.filter_shape_dims[0]:
                if t[MAPS] % parallel_output_maps and t[MAPS] > parallel_output_maps:
                    t[MAPS] = t[MAPS] // parallel_output_maps * parallel_output_maps  

            return True, t, conv_rows          

def usage(node, tile, preset, verbose=False, sparse=None):

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6

    sp_size = preset_select['SCRATCHPAD_KB'][preset]*1024 - 256 #TODO get exact indirect
    vl = preset_select['VECTOR_LANES'][preset]
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    in_size = fia_input_shaper_size_kb(filter_copies, sparse) * 1024
    w_size = fia_weight_shaper_size_kb(parallel_output_maps, sparse) * 1024
    q_size = fia_quantization_shaper_size_kb() * 1024
    out_size = sp_output_shaper_size_b(sp_size, (len(node.subnode_array) == 0 and node.output_strides == [1,1]))

    v_out, v_tmp0, v_tmp = max_scratchpad_required(node, vl, tile, sparse=sparse)
    sp_used = sp_used_fn(node)(v_out,v_tmp0, v_tmp)

    k, c, kh, kw = node.Conv2DOptions.filter_shape_dims
    sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width

    if node.Conv2DOptions.use_depthwise:
        w_used = tile[IMAPS]*kw*(kh+parallel_output_maps-1)*parallel_output_maps
        # if node.Conv2DOptions.use_db:
        #     w_used *= 2
    else:
        if node.Conv2DOptions.fit_weights:
            c_used_in_w = c
            k_used_in_w = k
        # elif (not node.Conv2DOptions.use_db):
        #     c_used_in_w = node.channels
        #     k_used_in_w = tile[MAPS]
        else:
            c_used_in_w = tile[IMAPS]
            k_used_in_w = tile[MAPS]
        if (sparse == 1):
            c_used_in_w = c_used_in_w + (8 - ((c_used_in_w % 8) if (c_used_in_w % 8) else 8))   
            if node.Conv2DOptions.repeat == 2:
                c_used_in_w //= 2
            elif node.Conv2DOptions.repeat == 1:
                c_used_in_w //= 4
        else:
            c_used_in_w = c_used_in_w + (c_used_in_w%2)

        k_used_in_w += (parallel_output_maps - ((k_used_in_w % parallel_output_maps) if (k_used_in_w % parallel_output_maps) else parallel_output_maps))
        w_used = kh*kw*c_used_in_w*k_used_in_w

    # if node.Conv2DOptions.split_weight_shaper_buffers and not node.Conv2DOptions.use_db:
    #     w_used *= 2
    # if not node.Conv2DOptions.fit_weights and node.Conv2DOptions.use_db and not node.Conv2DOptions.conv_rows:
    #     w_used *= 2

    total_c = c
    total_k = k
    if (sparse == 1):
        total_c = total_c + (8 - ((total_c % 8) if (total_c % 8) else 8))   
        if node.Conv2DOptions.repeat == 2:
            total_c //= 2
        elif node.Conv2DOptions.repeat == 1:
            total_c //= 4
    else:
        total_c = total_c + (total_c%2)

    total_k += (parallel_output_maps - ((k % parallel_output_maps) if (k % parallel_output_maps) else parallel_output_maps))
    w_total = (kh*kw*total_c*total_k)

    if node.Conv2DOptions.use_depthwise:
        q_total = c*QUANTIZATION_RECORD_WIDTH_BYTES
    else:
        q_total = k*QUANTIZATION_RECORD_WIDTH_BYTES
    q_used = tile[MAPS]*QUANTIZATION_RECORD_WIDTH_BYTES

    if node.Conv2DOptions.use_db and node.Conv2DOptions.conv_rows:
        # in_used = tile[IMAPS]*node.Conv2DOptions.conv_rows*tile[COLUMNS]*2
        in_used = tile[IMAPS]*node.Conv2DOptions.conv_rows*tile[COLUMNS]
    # elif node.Conv2DOptions.use_db:
    #     in_used = tile[IMAPS]*tile[ROWS]*tile[COLUMNS]*2
    else:
        in_used = tile[IMAPS]*tile[ROWS]*tile[COLUMNS]

    out_used = tile[MAPS]*((tile[ROWS]-(kh-1))+(sh-1))//sh*((tile[COLUMNS]-(kw-1))+(sw-1))//sw

    if verbose:
        print('in  {}% (used {}, size {})'.format( int(100. * in_used/ in_size), in_used, in_size))
        print('wt  {}% (used {}, size {}, total {})'.format( int(100. * w_used/ w_size), w_used, w_size, w_total))
        print('qt  {}% (used {}, size {}, total {})'.format( int(100. * q_used/ q_size), q_used, q_size, q_total))
        print('out {}% (used {}, size {})'.format( int(100. * out_used/ out_size), out_used, out_size))
        print('sp  {}% (used {}, size {}, {})'.format( int(100. * sp_used/ sp_size), sp_used, sp_size, (v_out, v_tmp0, v_tmp)))

    node.Conv2DOptions.in_used = in_used
    node.Conv2DOptions.w_used = w_used
    node.Conv2DOptions.qt_used = q_used
    node.Conv2DOptions.out_used = out_used
    node.Conv2DOptions.sp_used = sp_used


'''
start w/ minimum viable tile size (produces some valid output)
maximize (up to a limit), as specific tile dimension
'''
def tile_subgraph(node, preset, opcode, sparse, node_idx=None, tmp_dir=None, graph_idx=None, tmp_dir_obj=None):
    sp = preset_select['SCRATCHPAD_KB'][preset]*1024 - 256 #TODO get exact indirect
    vl = preset_select['VECTOR_LANES'][preset]
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6

    if (not node.offloaded):
        for idx in range(node.num_tensors):
            t = node.tensor_array[idx]
            if (t.external_producer or t.external_consumer):
                shape = list(t.shape)
                if (shape[-1] % 16 != 0):
                    node.sync_offset = int(shape[-1] % 16)
                    shape[-1] = ceil(shape[-1]/16)*16
                    t.shape = tuple(shape)
                    if node.type == BuiltinOperator.FULLY_CONNECTED:
                        node.m = shape[-1]    
                    else:
                        node.n = shape[-1]
        for i in range(len(node.subnode_array)):
            sn = node.subnode_array[i]
            for idx in range(sn.num_tensors):
                t = sn.tensor_array[idx]
                if (t.external_producer or t.external_consumer):
                    shape = list(t.shape)
                    if (shape[-1] % 16 != 0):
                        node.sync_offset = int(shape[-1] % 16)
                        shape[-1] = ceil(shape[-1]/16)*16
                        t.shape = tuple(shape)
                        node.n = shape[-1]

    # STEP 1: get minimum tile (that produces a viable output)
    fx = compose_subgraph(node, min_output_tile)
    # tile = minimum_tile_subgraph(node, fx)
    tile = minimum_valid_tile_subgraph(node, vl, sp, fx, opcode, tmp_dir=tmp_dir, graph_idx=graph_idx, sparse=sparse, tmp_dir_obj=tmp_dir_obj)
    min_tile = tile.copy()

    # STEP 2: grow tile dimensions (within scratchpad capacity) according to per-operator rules
    e = node.type
    if e in [BuiltinOperator.CONV_2D, BuiltinOperator.TRANSPOSE_CONV]:
        v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, sparse=sparse)
        is_allocated = False

        node.Conv2DOptions.conv_rows    = 0
        node.Conv2DOptions.use_db       = (not FORCE_SB)
        # print('Tiling Conv2D: use_db {}, conv_rows {}, split_weights {}'.format(node.Conv2DOptions.use_db, node.Conv2DOptions.conv_rows, node.Conv2DOptions.split_weight_shaper_buffers))

        if not node.Conv2DOptions.use_fia: # scalar + vector
            tile = increment_tile(node, vl, sp, fx, tile, COLUMNS, limit=node.n, sparse=sparse)
            tile = increment_tile(node, vl, sp, fx, tile, ROWS, limit=node.m, sparse=sparse)
            tile = increment_tile(node, vl, sp, fx, tile, MAPS, limit=node.Conv2DOptions.kernels, sparse=sparse)
            is_allocated = True

        elif node.Conv2DOptions.use_depthwise: # scalar + vector
            # row_db unused for depthwise
            node.Conv2DOptions.conv_rows = 0
            
            depthwise_db_maps, tile_depthwise_db_maps =  fit_depthwise(tile, node, vl, sp, fx, preset, use_db=1, row_db=0, sparse=sparse) 
            depthwise_single_buffer, tile_depthwise_single_buffer =  fit_depthwise(tile, node, vl, sp, fx, preset, use_db=0, row_db=0, sparse=sparse)

            # TODO fix depthwise 5x5 kernel
            # if node.Conv2DOptions.use_db and \
            #     not (node.Conv2DOptions.filter_shape_dims[-2] == 5 and node.Conv2DOptions.filter_shape_dims[-1] == 5 and tile_depthwise_db_maps[COLUMNS] > 1000):
            if 1:
                tile = tile_depthwise_db_maps
                is_allocated = depthwise_db_maps
                node.Conv2DOptions.split_weight_shaper_buffers = 0
            else:
                node.Conv2DOptions.use_db = 0
                tile = tile_depthwise_single_buffer
                is_allocated = depthwise_single_buffer

        else: # accelerator

            # conv_db_rows, tile_conv_db_rows, rows_db_rows =  fit_conv(tile, node, vl, sp, fx, preset, use_db=1, row_db=1, sparse=sparse)
            conv_db_maps, tile_conv_db_maps, rows_db_maps =  fit_conv(tile, node, vl, sp, fx, preset, use_db=1, row_db=0, sparse=sparse)
            # conv_single_buffer, tile_conv_single_buffer, rows_single_buffer =  fit_conv(tile, node, vl, sp, fx, preset, use_db=0, row_db=0, sparse=sparse)

            # conv_single_buffer2, tile_conv_single_buffer2, rows_single_buffer2 =  fit_conv(tile, node, vl, sp, fx, preset, use_db=0, row_db=0, split_weights=0, sparse=sparse)
            # is_fc = (node.n == 1 and node.m == 1)

            # DB kernel tiling
            tile = tile_conv_db_maps
            is_allocated = conv_db_maps
            node.Conv2DOptions.conv_rows = 0
            node.Conv2DOptions.split_weight_shaper_buffers = 0

            # old tiling scheme and priority for old FIA
            # kh = node.Conv2DOptions.filter_shape_dims[-2]
            # if kh == 1 and node.Conv2DOptions.use_db and node.Conv2DOptions.conv_rows and (5 < rows_db_rows < tile_conv_db_rows[ROWS] < node.m):
            #     # print("DB ROWS to 1 for 1x1 kernel")
            #     tile = tile_conv_db_rows
            #     is_allocated = conv_db_rows
            #     node.Conv2DOptions.conv_rows = rows_db_rows
            #     node.Conv2DOptions.split_weight_shaper_buffers = 0

            # elif tile_conv_single_buffer[IMAPS] == node.channels and not is_fc: #prefer all imaps (DMA input/weights once)
            #     # print("FAVOR single buffer all imaps")
            #     node.Conv2DOptions.use_db = 0
            #     node.Conv2DOptions.conv_rows = 0
            #     tile = tile_conv_single_buffer
            #     is_allocated = conv_single_buffer

            # elif tile_conv_single_buffer2[IMAPS] == node.channels and not is_fc: #same but w/o split weight buffers
            #     # print("FAVOR single buffer all imaps no split weights")
            #     node.Conv2DOptions.use_db = 0
            #     node.Conv2DOptions.conv_rows = 0
            #     node.Conv2DOptions.split_weight_shaper_buffers = 0
            #     tile = tile_conv_single_buffer2
            #     is_allocated = conv_single_buffer2

            # elif node.Conv2DOptions.use_db and tile_conv_db_maps[IMAPS] < node.channels: #DB maps if not all channels
            #     # print("FAVOR DB maps i_DB=", node.Conv2DOptions.use_db)
            #     tile = tile_conv_db_maps
            #     is_allocated = conv_db_maps
            #     node.Conv2DOptions.conv_rows = 0
            #     node.Conv2DOptions.split_weight_shaper_buffers = 0

            # else: #fall through to single buffer
            #     # print("FALLBACK single buffer")
            #     node.Conv2DOptions.use_db = 0
            #     node.Conv2DOptions.conv_rows = 0
            #     tile = tile_conv_single_buffer
            #     is_allocated = conv_single_buffer

            if (node.Conv2DOptions.kernels == 1): #TODO fix tiling
                if tile[-1] == 40 and tile[-2] > 20 and tile[-2] < 40:
                    tile[-2] = 20

            if not is_allocated:
                print('not allocated')
                if not v(tile[MAPS],tile[IMAPS],tile[ROWS],tile[COLUMNS]):
                    return None

                limit_cols = node.n # adjust COLUMN limit to be maximum valid fia COLUMN limit
                while limit_cols > tile[COLUMNS] and not v(tile[MAPS],tile[IMAPS],tile[ROWS],limit_cols):
                    limit_cols -= 1
                tile = increment_tile(node, vl, sp, fx, tile, COLUMNS, limit=limit_cols, sparse=sparse)

                
                limit_imaps = node.Conv2DOptions.filter_shape_dims[1]
                while not v(tile[MAPS],limit_imaps,tile[ROWS],tile[COLUMNS]):
                    limit_imaps = _decrement_imaps(limit_imaps, node.channels)
                tile = increment_tile(node, vl, sp, fx, tile, IMAPS, limit=limit_imaps, sparse=sparse)

                limit_maps = node.Conv2DOptions.filter_shape_dims[0]
                while limit_maps > tile[MAPS] and not v(limit_maps,tile[IMAPS],tile[ROWS],tile[COLUMNS]):
                    limit_maps -= 1
                limit_maps = limit_maps // parallel_output_maps * parallel_output_maps
                tile = increment_tile(node, vl, sp, fx, tile, MAPS, limit=limit_maps, sparse=sparse)

                limit_rows = node.m
                while limit_rows > tile[ROWS] and not v(tile[MAPS],tile[IMAPS],limit_rows, tile[COLUMNS]):
                    limit_rows -= 1
                tile = increment_tile(node, vl, sp, fx, tile, ROWS, limit=limit_rows, sparse=sparse)

            if (sparse != 2):
                isize, wsize = _get_input_and_weight_shaper_sizes(filter_copies, parallel_output_maps, sparse=sparse)
                if sparse:
                    wsize = wsize // WEIGHT_SHAPER_BANKS_SPARSE * WEIGHT_SHAPER_DATA_BANKS # only use DATA_BANKS
                osize = sp_output_shaper_size_b(sp, (len(node.subnode_array) == 0 and node.output_strides == [1,1]))
                osize *= 2

                t = allocate_fia_tile(node, preset, vl, sp, sparse, isize, wsize, osize, tile, min_tile, is_sp_invalid, valid_fia,\
                                       k_parallel=parallel_output_maps, opcode=opcode, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)
                if not t is None:
                    node.Conv2DOptions.use_db, node.Conv2DOptions.conv_rows, node.Conv2DOptions.split_weight_shaper_buffers = 1, 0, 0
                    tile = t

                else: # if failed to find tile in allocate_fia_tile, then fall-back to other tile if it does full columns else exit
                    if tile[COLUMNS] != node.n:
                        with sp_invalid_exception_catcher(sp, opcode, tmp_dir, graph_idx, tmp_dir_obj):
                            assert(0)


    elif e == BuiltinOperator.FULLY_CONNECTED:
        output_depth, accum_depth = node.FullyConnectedOptions.filter_shape_dims

        if node.FullyConnectedOptions.use_fia: #
            v = lambda a, r, c: valid_fia_fc(a, r, c, preset, (len(node.subnode_array) == 0 and node.output_strides == [1,1]))

            # maximize accum depth (input width, filter rows)
            limit_accum_depth = accum_depth
            while limit_accum_depth > tile[ICOLUMNS] and not v(limit_accum_depth, tile[ROWS], tile[COLUMNS]):
                limit_accum_depth -= 1
            tile = increment_tile(node, vl, sp, fx, tile, ICOLUMNS, limit=limit_accum_depth, sparse=sparse)

            # maximize output depth (filter columns, output columns)
            limit_output_depth = output_depth
            while limit_output_depth > tile[COLUMNS] and not v(tile[ICOLUMNS], tile[ROWS], limit_output_depth):
                limit_output_depth -= 1
            tile = increment_tile(node, vl, sp, fx, tile, COLUMNS, limit=limit_output_depth, sparse=sparse)

            # maximize input rows (output rows)
            limit_rows = node.m
            while limit_rows > tile[ROWS] and not v(tile[ICOLUMNS], limit_rows, tile[COLUMNS]):
                limit_rows -= 1
            tile = increment_tile(node, vl, sp, fx, tile, ROWS, limit=limit_rows, sparse=sparse)
        else:
            tile = increment_tile(node, vl, sp, fx, tile, ICOLUMNS, limit=accum_depth, sparse=sparse)
            tile = increment_tile(node, vl, sp, fx, tile, COLUMNS, limit=output_depth, sparse=sparse)
            tile = increment_tile(node, vl, sp, fx, tile, IROWS, limit=output_depth, sparse=sparse)


    else: # common case
        tile = increment_tile(node, vl, sp, fx, tile, COLUMNS, limit=node.n, sparse=sparse)
        tile = increment_tile(node, vl, sp, fx, tile, ROWS, limit=node.m, sparse=sparse)
        tile = increment_tile(node, vl, sp, fx, tile, MAPS, limit=node.channels, sparse=sparse)

    # verify that final tile fits in sp, and if it is Conv additionally verify that valid_fia still true else exit
    if not is_sp_invalid(node, vl, sp, tile, sparse):
        if e in [BuiltinOperator.CONV_2D, BuiltinOperator.TRANSPOSE_CONV]:
            if not valid_fia(tile[MAPS], tile[IMAPS], tile[ROWS], tile[COLUMNS], node, preset, use_db=1, db_rows=0, split_weights=0, sparse=sparse):
                with sp_invalid_exception_catcher(sp, opcode, tmp_dir, graph_idx, tmp_dir_obj):
                    assert(0)
    else:
        with sp_invalid_exception_catcher(sp, opcode, tmp_dir, graph_idx, tmp_dir_obj):
            assert(0)                    

    # STEP 3: for a given tile, determine how to walk vnnx_graph
    # print("tile:", tile)
    set_tile_attr(node, vl, sp, tile, sparse=sparse)
    
    # FIA shaper and sp utilization
    if os.getenv('VECTORBLOX_TIME_LAYERS') and node.Conv2DOptions.use_fia:
        usage(node, tile, preset, verbose=False, sparse=sparse)

    return tile
