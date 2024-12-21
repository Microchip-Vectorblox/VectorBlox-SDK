from .vnnx_types import *
from functools import reduce
from math import floor, ceil, log2

MXP_DOUBLE_BUFFER = 1   # Will remove later
FIA_WRITE_DB = 0
FORCE_SB = 0

# Split shaper buffers to use 2 buffers
SPLIT_WEIGHT_SHAPER_BUFFERS = 1 # Will remove later

##################################################################
### The below 4 functions must match fia.h equivalent functions; update both at the same time.
##################################################################
def fia_input_shaper_size_kb(filter_copies):
	return filter_copies*2

def fia_weight_shaper_size_kb(parallel_kernels):
	return parallel_kernels*2

def fia_quantization_shaper_size_kb(filter_copies):
    return filter_copies // 4

def fia_output_shaper_size_kb(filter_copies, parallel_kernels):
    return max(parallel_kernels*2 , filter_copies*2)


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


def composite_function(*func): 
    def compose(f, g): 
        return lambda x : f(g(x)) 
              
    return reduce(compose, func, lambda x : x)

def maximize_tile(node, vector_lanes, sp, fx, tile, idx, limit, scale=1, offset=0):
    tile = tile.copy()
    sp_used = sp_used_fn(node)

    v_out, v_tmp0, v_tmp = scratchpad_required(node, vector_lanes, tile)
    if tile[idx] < limit:
        tmp = tile[idx]
        while(sp_used(v_out,v_tmp0, v_tmp) <= sp):
            tmp = tile[idx]
            if tmp == limit:
                break
            tile[idx] = min(tmp * scale + offset , limit)
            v_out, v_tmp0, v_tmp = scratchpad_required(node, vector_lanes, tile)
        else:
            tile[idx] = tmp

    return tile


def multiply_tile(node, vector_lanes, sp, fx, tile, idx, limit, scale):
    return maximize_tile(node, vector_lanes, sp, fx, tile, idx, limit, scale=scale, offset=0)

def increment_tile(node, vector_lanes, sp, fx, tile, idx, limit, offset=1):
    return maximize_tile(node, vector_lanes, sp, fx, tile, idx, limit, scale=1, offset=offset)


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

def minimum_valid_tile_subgraph(node, vl, sp, fx):
    
    tile = minimum_tile_subgraph(node, fx)

    sp_used = sp_used_fn(node)

    v_out, v_tmp0, v_tmp = scratchpad_required(node, vl, tile) 
    used = sp_used(v_out,v_tmp0, v_tmp)
    if used > sp:
        sys.stderr.write("\n\033[31m######################## VECTORBLOX ERROR! #############################\033[0m\n")
        sys.stderr.write("\033[31mThe layer {} cannot currently fit within the memory scratchpad!\033[0m").format(node.type)
        sys.stderr.write("\nWe are continously working to improve the SDK.")
        sys.stderr.write("\nFor futher assistance, please contact the vectorblox team at:\n\033[31mvectorblox@microchip.com\033[0m\n\n")
        sys.exit(1)

    return tile

def scratchpad_required(node, vector_lanes, tile):
    tile = tile.copy()
    v_out, v_tmp0 = max_sp_tile(node, vector_lanes)(tile)
    # tile = min_output_tile(node)(tile)
    tile = max_output_tile(node)(tile)

    v_tmp = 0
    for sn in node.subnode_array:
        v_0, v_1 = max_sp_tile(sn, vector_lanes)(tile)
        v_out, v_tmp = max(v_out, v_0), max(v_tmp, v_1)

        # tile = min_output_tile(sn)(tile)
        tile = max_output_tile(sn)(tile)

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
                     BuiltinOperator.MUL,
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
                     BuiltinOperator.LESS,
                     BuiltinOperator.LESS_EQUAL,
                     BuiltinOperator.EQUAL,
                     BuiltinOperator.NOT_EQUAL,
                     BuiltinOperator.PAD,
                     BuiltinOperator.MIRROR_PAD,
                     BuiltinOperator.CONCATENATION,
                     VNNXOperator.IDENTITY,
                     VNNXOperator.ELTWISE,
                     VNNXOperator.LUT,
                     BuiltinOperator.SLICE,
                     BuiltinOperator.STRIDED_SLICE,
                     BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,
                     BuiltinOperator.PACK,
                     BuiltinOperator.LOG_SOFTMAX,
                     ]

    # requires full channels, rows, columns, ...  for development only
    full_types = [BuiltinOperator.GATHER,
                  BuiltinOperator.DEPTH_TO_SPACE,
                  BuiltinOperator.SPACE_TO_DEPTH,
                  BuiltinOperator.BATCH_TO_SPACE_ND,
                  BuiltinOperator.SPACE_TO_BATCH_ND,
                  BuiltinOperator.TRANSPOSE_CONV,
                  BuiltinOperator.L2_NORMALIZATION,
                  #BuiltinOperator.ARG_MAX,
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
    
    elif e == BuiltinOperator.FULLY_CONNECTED:
        output_depth, accum_depth = node.FullyConnectedOptions.filter_shape_dims
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            return np.asarray([batch, imaps, irows, icols, maps, rows, irows])
        return x 
    elif e in [BuiltinOperator.MEAN, BuiltinOperator.SUM, BuiltinOperator.REDUCE_PROD, BuiltinOperator.REDUCE_MAX, BuiltinOperator.REDUCE_MIN, BuiltinOperator.ARG_MAX, BuiltinOperator.ARG_MIN]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if rows < node.tensor_array[0].shape[-2]: # if not full rows, return 0 in rows
                return np.asarray([batch, imaps, irows, icols, maps, 0, 1])
            if cols < node.tensor_array[0].shape[-1]: #if not full cols, return 0 in cols
                return np.asarray([batch, imaps, irows, icols, maps, 1, 0])
            return np.asarray([batch, imaps, irows, icols, maps, 1, 1])
        return x 
    elif e in [BuiltinOperator.MAX_POOL_2D, BuiltinOperator.AVERAGE_POOL_2D]:
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
    elif e in [BuiltinOperator.SPLIT, BuiltinOperator.SPLIT_V]:
        axis = node.SplitOptions.axis
        if axis > 0:
            axis -= node.tensor_array[0].dims
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            if axis == -1 and cols != node.n:
                return np.asarray([batch, imaps, irows, icols, maps, rows, 0])
            if axis == -2 and rows != node.m:
                return np.asarray([batch, imaps, irows, icols, maps, 0, cols])
            if axis == -3 and maps != node.channels:
                return np.asarray([batch, imaps, irows, icols, 0, rows, cols])
            return np.asarray([batch, imaps, irows, icols, maps, rows, cols])
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
    # elif e  == BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
    #     sh, sw = node.ResizeOptions.scale            
    #     def x(tile):
    #         batch, imaps, irows, icols, maps, rows, cols = tile
    #         if maps < node.tensor_array[0].shape[-3]:
    #             return np.asarray([batch, imaps, irows, icols, 0, rows* sh, cols* sw]) 
                
    #         return np.asarray([batch, imaps, irows, icols, maps, rows* sh, cols* sw])

    #     return x
    
    elif e in [BuiltinOperator.RESIZE_BILINEAR, BuiltinOperator.RESIZE_NEAREST_NEIGHBOR]:
        sh, sw = node.ResizeOptions.scale
        return lambda s: np.concatenate((s[:ROWS],[s[ROWS] * sh],[s[COLUMNS] * sw]))
    
    elif e == VNNXOperator.UNKNOWN:
        pass
    print(e, 'min_output_tile')
    return None

'''
per layer:
    have a function that when given the dimensions of an input tile
    returns the maximum dimensions of an output tile
'''
def max_output_tile(node):
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

    return min_output_tile(node)
    

'''
per layer:
    have a function that when given the dimensions of an input tile
    returns the scratchpad usage  (output_size, temporary_size)
'''
def max_sp_tile(node, vector_lanes):
    aligned = lambda sz : aligned_size(sz, vector_lanes)
    default = lambda s: [aligned(s[0]*s[-3]*s[-2]*s[-1]), 0]

    # only needs output allocated, no temporary (sp_malloc) needed
    default_types = [BuiltinOperator.MAX_POOL_2D,
                     BuiltinOperator.NEG,
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
                     BuiltinOperator.TRANSPOSE_CONV,
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
        elif node.Conv2DOptions.use_depthwise: # FIA depthwise
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                rows += ph
                cols += pw

                weight_stride = maps
                stride_factor = 2
                strided_m = (rows + (sh-1)) // sh
                strided_n = (cols + (sw-1)) // sw

                out_offset = (kh//2)*dhf*cols+(kw//2)*dwf
                if use_strided_input_maps:
                    out_offset = (kh//2)*dhf*strided_n+(kw//2)*dwf

                v_out = aligned(batch * maps * (rows * cols + out_offset))
                return [v_out, 0]
            return x

        else: # accelerator
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                rows += ph
                cols += pw

                weight_stride = maps
                strided_m = (rows + (sh-1)) // sh
                strided_n = (cols + (sw-1)) // sw

                out_offset = (kh//2)*dhf*cols+(kw//2)*dwf
                if use_strided_input_maps:
                    out_offset = (kh//2)*dhf*strided_n+(kw//2)*dwf

                v_out = aligned(batch * maps * (rows * cols + out_offset))

                return [v_out, 0]
            return x
    elif e == BuiltinOperator.FULLY_CONNECTED:
        output_depth, accum_depth = node.FullyConnectedOptions.filter_shape_dims
        if not node.FullyConnectedOptions.use_fia: # vector
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile

                v_tmp = aligned(batch*imaps*(icols + icols + irows*4 + irows*4))
                v_out = aligned(batch*maps*irows)

                return [v_out, v_tmp]
            return x
        else:
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                v_out = aligned(batch*maps*irows)

                return [v_out, 0]
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
    elif e in [BuiltinOperator.ARG_MAX, BuiltinOperator.ARG_MIN]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_out = aligned(batch * rows* cols)
            v_tmp = aligned(batch * rows*cols*(4+3 + 1)) #@1 word + 3 bytes + v_in

            return [v_out, v_tmp]
        return x
    elif e == BuiltinOperator.AVERAGE_POOL_2D:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_out = aligned(batch*maps*rows*cols)
            v_tmp = aligned(batch*maps*rows*cols*2*2) #TODO

            return [v_out, v_tmp]
        return x
    elif e in [BuiltinOperator.RELU, BuiltinOperator.RELU_N1_TO_1, BuiltinOperator.RELU_0_TO_1, BuiltinOperator.LEAKY_RELU, BuiltinOperator.ABS]:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_tmp = aligned(batch * maps * rows * cols * 4) * 2 # 2 word size vectors (of input tile)
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
    elif e in [BuiltinOperator.PAD, BuiltinOperator.MIRROR_PAD]:
        pads = node.pads.copy()
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
    elif e in [BuiltinOperator.MUL, BuiltinOperator.ADD, BuiltinOperator.SUB]:
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
            v_out = aligned(batch*maps*rows*cols)
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
    elif e == BuiltinOperator.SPLIT:
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_out = aligned(batch * maps * rows * cols)
            v_tmp = 0
            return [v_out, v_tmp]
        return x
    elif e in [VNNXOperator.IDENTITY, BuiltinOperator.CONCATENATION]:
        num_inputs = node.num_inputs 
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            v_out = aligned(batch * maps * rows * cols)
            v_tmp = 0

            return [v_out, v_tmp]
        return x
    elif e == BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
        sh, sw = node.ResizeOptions.scale
        if sh == 2. and sw == 2.:
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                size = batch * maps * rows * cols
                v_tmp = aligned(size*2)
                v_out = aligned(size*2*2)

                return [v_out, v_tmp]
            return x
        else:
            def x(tile):
                batch, imaps, irows, icols, maps, rows, cols = tile
                size = batch * maps * rows * cols
                # v_tmp = aligned(size)
                v_tmp = aligned(int(rows * sw *cols +0.5))
                v_tmp += aligned(int(rows*sw*cols +0.5))
                v_out = aligned(int((size*sh +0.5)*sw+0.5))

                return [v_out, v_tmp]
            return x
        # else:
            # return None
    elif e == BuiltinOperator.RESIZE_BILINEAR:
        sh, sw = node.ResizeOptions.scale
        def x(tile):
            batch, imaps, irows, icols, maps, rows, cols = tile
            size = batch * maps * rows * cols
            col_size = max(rows, cols)
            v_tmp = aligned(rows*cols*sw +0.5) #xp_temp
            v_tmp += aligned(col_size*4) #v_tmp0
            v_tmp += aligned(col_size*4) #v_tmp1
            v_out = aligned(int((size*sh +0.5)*sw+0.5))

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
                node.eltwise8.isize = isize

                v_tmp += aligned(isize * 2) # v_tmp
                v_tmp += aligned(isize * 2) # v_add
            else:
                node.eltwise8.isize = 0
                v_tmp += aligned(batch * maps * rows * cols * 4) # v_tmp
                v_tmp += aligned(batch * maps * rows * cols * 4) # v_add

            if isinstance(node, Node):
                v_tmp += aligned(batch * maps * rows * cols) # v_in
            v_tmp += aligned(batch * maps * rows * cols) # v_in2

            v_out = aligned(batch * maps * rows * cols)

            return [v_out, v_tmp]
        return x
    elif e == VNNXOperator.UNKNOWN:
        pass
    print('no set', e)

    return None


def get_tile_steps(fx, idx, len_idx, limit_idx, start, stop, max_input_len):
    if start + max_input_len == stop:
        return [0], [max_input_len]

    layer = [0 for _ in range(15)]
    layer[limit_idx] = stop

    layer[idx], layer[len_idx] = start, stop
    total_output_len = max(1, fx(layer)[len_idx])

    steps = []
    strides = []

    i = start
    while 1:
        # find maximum output length possible
        input_len = max_input_len
        layer[idx], layer[len_idx] = i, input_len
        max_output_len = fx(layer)[len_idx] 

        # shrink if maximum output length exceeds total output length
        if (fx(layer)[idx] + max_output_len) > total_output_len:
            max_output_len = total_output_len - fx(layer)[idx]
            input_len = 0
            layer[len_idx] = input_len
            while (fx(layer)[len_idx] != max_output_len):
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

        if (fx(layer)[idx] + max_output_len) == total_output_len:
            break

        # find next step
        next_output_idx = fx(layer)[idx] + fx(layer)[len_idx]
        step = 0
        while((fx(layer)[idx]) != next_output_idx):
            step += 1
            layer[idx] = i + step
        i += step

    return steps, strides


def valid_conv_rows(tile, conv_rows, node, preset):
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    batch, imaps, irows, icols, maps, rows, cols = tile
    input_shaper_size, _ = _get_input_and_weight_shaper_sizes(filter_copies, parallel_output_maps)

    if node.Conv2DOptions.use_strided:
        sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
        srows = conv_rows // sh
        if conv_rows % sh:
            srows += 1
        scols = cols // sw
        if cols % sw:
            scols += 1
        elements = srows * scols

        if elements*sh*sw*imaps > input_shaper_size /2:
            return False
    else:
        pw = node.Conv2DOptions.padding_width
        pcols  = cols  + pw

        if imaps*pcols*conv_rows > input_shaper_size / 2:
            return False

    return True

'''
batch   b_start  b_stop  b_inc
input   i_start  i_stop  i_inc
row     y_start  y_stop  y_inc y_rows0 y_inc0 y_rows_final
col     x_start  x_stop  x_inc x_cols0 x_inc0 x_cols_final
map     m_start  m_stop  m_inc
'''
def set_tile_attr(node, vector_lanes, sp, tile, cores=1):
    v_out, v_tmp0, v_tmp  = scratchpad_required(node, vector_lanes, tile)
    node.scratchpad_bytes = v_out

    sp_used = sp_used_fn(node)
    used = sp_used(v_out,v_tmp0, v_tmp)

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
    # elif e in [BuiltinOperator.MEAN, BuiltinOperator.SUM, BuiltinOperator.REDUCE_PROD, BuiltinOperator.REDUCE_MAX, BuiltinOperator.REDUCE_MIN]:
    #     x_stop = 1
    #     y_stop = 1

    o_idx, b_idx, c_idx, m_idx, n_idx = 0, 1, 2, 3, 4
    o_step_idx, b_step_idx, maps_idx, rows_idx, cols_idx = 5, 6, 7, 8, 9
    outer_idx, batches_idx, channels_idx, y_idx, x_idx = 10, 11, 12, 13, 14
    

    y, r = get_tile_steps(fx, y_idx, rows_idx, m_idx, y_start, y_stop, rows)
    x, c = get_tile_steps(fx, x_idx, cols_idx, n_idx, x_start, x_stop, cols)
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

'''
per layer:
    have a function that when given the dimensions of an input tile
    returns the dimensions of an output tile
'''
def adjust_tile(node):
    default = lambda l: l

    # for a given input tile size, output tile size will be same
    default_types = [BuiltinOperator.FULLY_CONNECTED,
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
                     BuiltinOperator.DEQUANTIZE,
                     BuiltinOperator.SPLIT,
                     BuiltinOperator.SPLIT_V,
                     BuiltinOperator.SLICE,
                     BuiltinOperator.STRIDED_SLICE,
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
                     BuiltinOperator.TRANSPOSE_CONV,
                     ]

    e = node.type
    if e in default_types:
        return default
    elif e in [BuiltinOperator.CONV_2D, BuiltinOperator.DEPTHWISE_CONV_2D]:
        _, _, kh, kw = node.Conv2DOptions.filter_shape_dims
        sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
        dhf, dwf = node.Conv2DOptions.dilation_height_factor, node.Conv2DOptions.dilation_width_factor
        ph, pw = node.Conv2DOptions.padding_height, node.Conv2DOptions.padding_width
        pl, pr, pu, pd = pw // 2, pw // 2 + (pw % 2), ph // 2, ph // 2 + (ph % 2)


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
    elif e in [BuiltinOperator.MEAN, BuiltinOperator.SUM, BuiltinOperator.REDUCE_PROD, BuiltinOperator.REDUCE_MAX, BuiltinOperator.REDUCE_MIN, BuiltinOperator.ARG_MAX, BuiltinOperator.ARG_MIN]:
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer
            return o, b, c, 1, 1, o_step, b_step, maps, 1, 1, outer, batches, channels, y, x
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
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer
            return o, b, c, m*sh, n*sw, o_step, b_step, maps, r*sh, col*sw, outer, batches, channels, y*sh, x*sw
        return adjust
    elif e in [BuiltinOperator.PAD, BuiltinOperator.MIRROR_PAD]:
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
    elif e == BuiltinOperator.TILE:
        tile_n, tile_c, tile_h, tile_w = node.TileOptions.tile
        def adjust(layer):
            o, b, c, m, n, o_step, b_step, maps, r, col, outer, batches, channels, y, x = layer
            return o, b*tile_n, c*tile_c, m*tile_h, n*tile_w, o_step, b_step*tile_n, maps*tile_c, r*tile_h, col*tile_w, outer, batches*tile_n, channels*tile_c, y*tile_h, x*tile_w
        return adjust
    elif e == VNNXOperator.UNKNOWN:
        pass

    return None


def valid_sp(maps, imaps, rows, cols, node, preset):
    sp = preset_select['SCRATCHPAD_KB'][preset]*1024 - 256 #TODO get exact indirect
    vector_lanes = preset_select['VECTOR_LANES'][preset]

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    tile = np.array([1,1,1,1,1,1,1])
    tile[MAPS] = maps
    tile[IMAPS] = imaps
    tile[ROWS] = rows
    tile[COLUMNS] = cols
    v_out, v_tmp0, v_tmp = scratchpad_required(node, vector_lanes, tile)

    sp_used = sp_used_fn(node)
    if sp_used(v_out,v_tmp0, v_tmp) > sp:
        return False

    return True



def valid_fia_fc(accum_depth, rows, cols, preset):
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    osh = rows
    osw = cols
    iw = accum_depth

    if iw*rows > (fia_input_shaper_size_kb(filter_copies)*1024): # input_shaper_size
        return False
    if accum_depth*cols > (fia_weight_shaper_size_kb(parallel_output_maps)*1024): # weigth_shaper_size
        return False
    if cols*16 > (fia_quantization_shaper_size_kb(filter_copies)*1024): # cols*sizeof(quantization_record)
        return False
    if osh*osw > (fia_output_shaper_size_kb(filter_copies, parallel_output_maps)*1024): # output_shaper_size
        return False

    return True


# Get the input and weight shaper sizes
def _get_input_and_weight_shaper_sizes(filter_copies, parallel_output_maps):

    input_shaper_size = fia_input_shaper_size_kb(filter_copies) * 1024
    weight_shaper_size = fia_weight_shaper_size_kb(parallel_output_maps) * 1024

    return input_shaper_size, weight_shaper_size


def valid_fia(maps, imaps, rows, cols, node, preset, use_db=None, db_rows=None, split_weights=None):
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

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

    pcols  = cols  + pw
    prows = rows + ph

    dh = ((kh-1)*dhf) + 1
    dw  = ((kw-1)*dwf)	 + 1
    orows = (prows - dh) // sh + 1
    ocols = (pcols - dw) // sw + 1
    srows = prows-(dh-1)
    scols = pcols
    if scols < dw :
        scols = dw
    elements = ((prows-(dh-1))*pcols)-(dw-1)

    weight_pad = srows
    if parallel_output_maps - 1 <= srows:
        weight_pad = parallel_output_maps - 1

    # Get the input and weight shaper sizes
    input_shaper_size, weight_shaper_size = _get_input_and_weight_shaper_sizes(filter_copies, parallel_output_maps)

    if node.Conv2DOptions.use_depthwise:
        if not use_db:
            if imaps*pcols*prows > input_shaper_size: #input_shaper_size
                return False
            if split_weights:
                if (imaps * kw * (kh + (2*weight_pad)) + weight_pad) > weight_shaper_size / 2: #weight_shaper_size
                    return False
            else:
                if (imaps * kw * (kh + (2*weight_pad)) + weight_pad) > weight_shaper_size: #weight_shaper_size
                    return False
        elif use_db:
            if imaps*pcols*prows > input_shaper_size / 2: #input_shaper_size
                return False
            if (imaps * kw * (kh + (2*weight_pad)) + weight_pad) > weight_shaper_size / 2: #weight_shaper_size
                return False
        if maps * 16 > (fia_quantization_shaper_size_kb(filter_copies) * 1024): # quantization_shaper_size maps*sizeof(quantization_record))
            return False
        if maps*scols*srows > (fia_output_shaper_size_kb(filter_copies, parallel_output_maps)*1024): #output_shaper_size
            return False
    else:
        if node.channels % imaps != 0:
            return False

        if node.Conv2DOptions.use_strided:
            srows = rows // sh
            if rows % sh:
                srows += 1
            scols = cols // sw
            if cols % sw:
                scols += 1
            elements = srows * scols
    
        if not db_rows:
            if not node.Conv2DOptions.use_strided: # input_shaper_size
                if use_db:
                    if imaps*pcols*prows > input_shaper_size / 2:
                        return False
                else:
                    if imaps*pcols*prows > input_shaper_size:
                        return False
            else:
                if use_db:
                    if elements*sh*sw*imaps > input_shaper_size / 2:
                        return False
                else:
                    if elements*sh*sw*imaps > input_shaper_size:
                        return False
        if use_db:
            if maps*kh*kw*imaps > weight_shaper_size / 2: # weight_shaper_size
                return False
        elif split_weights:
            if maps*kh*kw*imaps > weight_shaper_size / 2: # weight_shaper_size
                return False
        else:
            if maps*kh*kw*imaps > weight_shaper_size: # weight_shaper_size
                return False
        if maps * 16 > (fia_quantization_shaper_size_kb(filter_copies) * 1024): # quantization_shaper_size maps*sizeof(quantization_record))            
            return False
        if not FIA_WRITE_DB:
            if maps*scols*srows > (fia_output_shaper_size_kb(filter_copies, parallel_output_maps)*1024): # output_shaper_size
                return False
        elif FIA_WRITE_DB:
            if 2*maps_per_pass*scols*srows > (fia_output_shaper_size_kb(filter_copies, parallel_output_maps)*1024): # output_shaper_size
                return False

    return True


def fit_full_rows(tile, node, vl, sp, fx, preset):
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    target = tile.copy()
    t = tile.copy()

    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset)
    if not v(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
        return False

    max_maps = node.Conv2DOptions.filter_shape_dims[0]
    if max_maps > parallel_output_maps:
        max_maps = parallel_output_maps

    target[MAPS] = max_maps
    target[IMAPS] = -1
    if node.Conv2DOptions.use_depthwise:
        target[IMAPS] = max_maps 
    target[ROWS] = -1
    target[COLUMNS] = node.n

    return fit_target(target, t, v, node, vl, sp, fx)


def fit_all_minus_rows(tile, node, vl, sp, fx, preset):
    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    target = tile.copy()
    t = tile.copy()

    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset)
    if not v(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
        return False, t

    max_maps = node.Conv2DOptions.filter_shape_dims[0]
    if node.Conv2DOptions.use_depthwise:
        max_maps = node.Conv2DOptions.filter_shape_dims[1]

    target[MAPS] = max_maps
    target[IMAPS] = node.Conv2DOptions.filter_shape_dims[1]
    target[ROWS] = -1
    target[COLUMNS] = node.n

    return fit_target(target, t, v, node, vl, sp, fx)


def fit_all_omaps(tile, node, vl, sp, fx, preset, use_db=None, db_rows=None, split_weights=None):
    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    target = tile.copy()
    t = tile.copy()

    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, use_db, db_rows, split_weights)
    if not v(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
        return False, t

    max_maps = node.Conv2DOptions.filter_shape_dims[0]
    if node.Conv2DOptions.use_depthwise:
        max_maps = node.Conv2DOptions.filter_shape_dims[1]

    target[MAPS] = max_maps
    target[IMAPS] = -1
    target[ROWS] = -1
    target[COLUMNS] = -1

    return fit_target(target, t, v, node, vl, sp, fx)


def fit_all_imaps(tile, node, vl, sp, fx, preset, use_db=None, db_rows=None, split_weights=None):
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    target = tile.copy()
    t = tile.copy()

    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, use_db, db_rows, split_weights)
    if not v(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
        return False, t

    max_maps = node.Conv2DOptions.filter_shape_dims[0]
    if node.Conv2DOptions.use_depthwise:
        max_maps = node.Conv2DOptions.filter_shape_dims[1]
    if max_maps > parallel_output_maps:
        max_maps = parallel_output_maps

    target[MAPS] = max_maps
    target[IMAPS] = node.Conv2DOptions.filter_shape_dims[1]
    target[ROWS] = -1
    target[COLUMNS] = -1
    if node.Conv2DOptions.stride_width > 1 and node.Conv2DOptions.filter_shape_dims[-1]:
        target[COLUMNS] = node.n 

    return fit_target(target, t, v, node, vl, sp, fx)


def fit_full_maps(tile, node, vl, sp, fx, preset, use_db=None, db_rows=None, split_weights=None):
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    target = tile.copy()
    t = tile.copy()

    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, use_db, db_rows, split_weights)
    if not v(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
        return False, t

    max_maps = node.Conv2DOptions.filter_shape_dims[0]
    if max_maps > parallel_output_maps:
        max_maps = parallel_output_maps

    target[MAPS] = max_maps
    target[IMAPS] = -1
    if node.Conv2DOptions.use_depthwise:
        target[IMAPS] = max_maps 
    target[ROWS] = node.m
    target[COLUMNS] = node.n


    return fit_target(target, t, v, node, vl, sp, fx)


def fit_target(target, t, v, node, vl, sp, fx):
    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6

    if target[MAPS] != -1:
        limit_maps = target[MAPS]
        while limit_maps > t[MAPS] and not v(limit_maps,t[IMAPS],t[ROWS],t[COLUMNS]):
            limit_maps -= 1
        if limit_maps != target[MAPS]:
            return False, t

        t = increment_tile(node, vl, sp, fx, t, MAPS, limit=limit_maps)
        if t[MAPS] != target[MAPS]:
            return False, t

    if target[COLUMNS] != -1:
        limit_cols = target[COLUMNS]
        while limit_cols > t[COLUMNS] and not v(t[MAPS],t[IMAPS],t[ROWS],limit_cols):
            limit_cols -= 1
        if limit_cols != target[COLUMNS]:
            return False, t

        t = increment_tile(node, vl, sp, fx, t, COLUMNS, limit=limit_cols)
        if t[COLUMNS] != target[COLUMNS]:
            return False, t

    if target[ROWS] != -1:
        limit_rows = target[ROWS]
        while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
            limit_rows -= 1
        if limit_rows != target[ROWS]:
            return False, t

        t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows)
        if t[ROWS] != target[ROWS]:
            return False, t

    if target[IMAPS] != -1:
        limit_imaps = target[IMAPS]
        while limit_imaps > t[IMAPS] and not v(t[MAPS],limit_imaps,t[ROWS],t[COLUMNS]):
            limit_imaps -= 1
        if limit_imaps != target[IMAPS]:
            return False, t

        t = increment_tile(node, vl, sp, fx, t, IMAPS, limit=limit_imaps)
        if t[IMAPS] != target[IMAPS]:
            return False, t

    return True, t


def fit_depthwise(tile, node, vl, sp, fx, preset, use_db=None, row_db=None):
    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, use_db, row_db)
    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6
    t = tile.copy()
    

    limit_cols = node.n # adjust COLUMN limit to be maximum valid fia COLUMN limit
    while limit_cols > t[COLUMNS] and not v(t[MAPS],t[IMAPS],t[ROWS],limit_cols):
        limit_cols -= 1
    t = increment_tile(node, vl, sp, fx, t, COLUMNS, limit=limit_cols)

    # max rows
    limit_rows = node.m
    while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
        limit_rows -= 1
    t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows)

    # max maps
    while True:
        next_maps = t[MAPS] + 1
        if use_db: # for DB, do not max imaps and maps at the same time, max imaps occurs later
            next_imaps = 1
        else:
            next_imaps = t[IMAPS] + 1
        if not v(next_maps,next_imaps,t[ROWS],t[COLUMNS]):
            break
        next_tile = increment_tile(node, vl, sp, fx, t, IMAPS, limit=next_imaps)
        next_tile = increment_tile(node, vl, sp, fx, next_tile, MAPS, limit=next_maps)
        if next_tile[MAPS] == next_maps and next_tile[IMAPS] == next_imaps:
            t = next_tile
        else:
            break

        if t[MAPS] == node.Conv2DOptions.filter_shape_dims[1]:
            break

    # max # imaps per buffer for DB depthwise
    if use_db:
        while t[IMAPS] != t[MAPS]:
            next_imaps = t[IMAPS] + 1
            if not v(t[MAPS], next_imaps, t[ROWS], t[COLUMNS]):
                break
            next_tile = increment_tile(node, vl, sp, fx, t, IMAPS, limit=next_imaps)
            if next_tile[IMAPS] == next_imaps:
                t = next_tile
            else:
                break

        if t[MAPS] != node.Conv2DOptions.filter_shape_dims[1]:

            iterations = ceil(node.Conv2DOptions.filter_shape_dims[1] / t[IMAPS])
            if node.Conv2DOptions.filter_shape_dims[1] % iterations == 0:
                t[IMAPS] = node.Conv2DOptions.filter_shape_dims[1] // iterations
            

            t[MAPS] -= (t[MAPS] % t[IMAPS])
    return True, t


def fit_conv(tile, node, vl, sp, fx, preset, use_db=None, row_db=None, split_weights=None):
    v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, use_db, row_db, split_weights)
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

    is_all_imaps, tile_all_imaps =  fit_all_imaps(t, node, vl, sp, fx, preset, use_db, row_db, split_weights)
    is_all_omaps, tile_all_omaps =  fit_all_omaps(t, node, vl, sp, fx, preset, use_db, row_db, split_weights)
    is_full_maps, tile_full_maps =  fit_full_maps(t, node, vl, sp, fx, preset, use_db, row_db, split_weights)

    if is_all_imaps:
        is_all_imaps_omaps, tile_all_imaps_omaps =  fit_all_omaps(tile_all_imaps, node, vl, sp, fx, preset, use_db, row_db, split_weights)

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
        t = increment_tile(node, vl, sp, fx, t, COLUMNS, limit=limit_cols)

        # max rows
        limit_rows = node.m
        if node.Conv2DOptions.use_strided:
            sh = node.Conv2DOptions.stride_height
            limit_rows = prows + dkh
            while limit_rows > prows: # get the max possible limit_rows, that is less than prows, that follows dkh + (factor of sh)
                limit_rows -= sh

            while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
                limit_rows -= sh
            t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows, offset=sh)

            is_allocated = True

            # max conv_rows
            conv_rows = t[ROWS]
            while conv_rows > dkh and not valid_conv_rows(t, conv_rows, node, preset):
                conv_rows -= sh
            node.Conv2DOptions.conv_rows = conv_rows

        else:
            limit_rows -= (prows % dkh)
            while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
                limit_rows -= dkh
            t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows, offset=dkh)

            is_allocated = True

            # max conv_rows
            conv_rows = t[ROWS]
            while conv_rows > dkh and not valid_conv_rows(t, conv_rows, node, preset):
                conv_rows -= dkh

        return True, t, conv_rows


    elif is_full_maps and use_db:
        conv_rows = 0

        t = tile_full_maps
        t[ROWS] = node.m
        t[COLUMNS] = node.n

        # max DB imaps
        limit_imaps = (node.Conv2DOptions.filter_shape_dims[1] + 1) // 2
        while not v(t[MAPS],limit_imaps,t[ROWS],t[COLUMNS]):
            limit_imaps -= 1
        t = increment_tile(node, vl, sp, fx, t, IMAPS, limit=limit_imaps)

        limit_maps = node.Conv2DOptions.filter_shape_dims[0]
        while limit_maps > t[MAPS] and not v(limit_maps,t[IMAPS],t[ROWS],t[COLUMNS]):
            limit_maps -= 1
        if limit_maps != node.Conv2DOptions.filter_shape_dims[0]:
            limit_maps = limit_maps // parallel_output_maps * parallel_output_maps
        t = increment_tile(node, vl, sp, fx, t, MAPS, limit=limit_maps)

        if t[MAPS] != node.Conv2DOptions.filter_shape_dims[0]:
            if t[MAPS] % parallel_output_maps:
                t[MAPS] = t[MAPS] // parallel_output_maps * parallel_output_maps

        limit_imaps = node.Conv2DOptions.filter_shape_dims[1]
        while not v(t[MAPS],limit_imaps,t[ROWS],t[COLUMNS]):
            limit_imaps -= 1
        t = increment_tile(node, vl, sp, fx, t, IMAPS, limit=limit_imaps)

        return True, t, conv_rows

    else: # not all imaps/omaps/cols so disable conv_rows and check tiles again
        conv_rows = 0

        # full maps aviods weights reuse 
        is_full_maps, tile_full_maps =  fit_full_maps(t, node, vl, sp, fx, preset, use_db, row_db, split_weights)

        # all input maps avoids input maps reuse
        is_all_imaps, tile_all_imaps =  fit_all_imaps(t, node, vl, sp, fx, preset, use_db, row_db, split_weights)

        # all output maps avoid input map reuse TODO
        is_all_omaps, tile_all_omaps =  fit_all_omaps(t, node, vl, sp, fx, preset, use_db, row_db, split_weights)
    

        if is_all_imaps:
            # set all_imaps
            t = tile_all_imaps

            # max columns
            limit_cols = node.n # adjust COLUMN limit to be maximum valid fia COLUMN limit
            while limit_cols > t[COLUMNS] and not v(t[MAPS],t[IMAPS],t[ROWS],limit_cols):
                limit_cols -= 1
            t = increment_tile(node, vl, sp, fx, t, COLUMNS, limit=limit_cols)

            # max rows
            limit_rows = node.m
            while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
                limit_rows -= 1
            t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows)

            # max maps
            limit_maps = node.Conv2DOptions.filter_shape_dims[0]
            while limit_maps > t[MAPS] and not v(limit_maps,t[IMAPS],t[ROWS],t[COLUMNS]):
                limit_maps -= 1
            if limit_maps != node.Conv2DOptions.filter_shape_dims[0]:
                limit_maps = limit_maps // parallel_output_maps * parallel_output_maps
            t = increment_tile(node, vl, sp, fx, t, MAPS, limit=limit_maps)

            if t[MAPS] != node.Conv2DOptions.filter_shape_dims[0]:
                if t[MAPS] % parallel_output_maps:
                    t[MAPS] = t[MAPS] // parallel_output_maps * parallel_output_maps

            return True, t, conv_rows

        elif is_full_maps:
            # set full_maps
            t = tile_full_maps

            # max imaps
            limit_imaps = node.Conv2DOptions.filter_shape_dims[1]
            while not v(t[MAPS],limit_imaps,t[ROWS],t[COLUMNS]):
                limit_imaps -= 1
            t = increment_tile(node, vl, sp, fx, t, IMAPS, limit=limit_imaps)

            # max maps
            limit_maps = node.Conv2DOptions.filter_shape_dims[0]
            while limit_maps > t[MAPS] and not v(limit_maps,t[IMAPS],t[ROWS],t[COLUMNS]):
                limit_maps -= 1
            if limit_maps != node.Conv2DOptions.filter_shape_dims[0]:
                limit_maps = limit_maps // parallel_output_maps * parallel_output_maps
            t = increment_tile(node, vl, sp, fx, t, MAPS, limit=limit_maps)

            if t[MAPS] != node.Conv2DOptions.filter_shape_dims[0]:
                if t[MAPS] % parallel_output_maps:
                    t[MAPS] = t[MAPS] // parallel_output_maps * parallel_output_maps

            return True, t, conv_rows

        elif is_all_omaps:
            # set all_omaps
            t = tile_all_omaps

            # max columns
            limit_cols = node.n # adjust COLUMN limit to be maximum valid fia COLUMN limit
            while limit_cols > t[COLUMNS] and not v(t[MAPS],t[IMAPS],t[ROWS],limit_cols):
                limit_cols -= 1
            t = increment_tile(node, vl, sp, fx, t, COLUMNS, limit=limit_cols)

            # max rows
            limit_rows = node.m
            while limit_rows > t[ROWS] and not v(t[MAPS],t[IMAPS],limit_rows, t[COLUMNS]):
                limit_rows -= 1
            t = increment_tile(node, vl, sp, fx, t, ROWS, limit=limit_rows)

            # max imaps
            limit_imaps = node.Conv2DOptions.filter_shape_dims[1]
            while not v(t[MAPS],limit_imaps,t[ROWS],t[COLUMNS]):
                limit_imaps -= 1
            t = increment_tile(node, vl, sp, fx, t, IMAPS, limit=limit_imaps)

            return True, t, conv_rows


def usage(node, tile, preset):

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6

    sp_size = preset_select['SCRATCHPAD_KB'][preset]*1024 - 256 #TODO get exact indirect
    vl = preset_select['VECTOR_LANES'][preset]
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    in_size = fia_input_shaper_size_kb(filter_copies) * 1024
    w_size = fia_weight_shaper_size_kb(parallel_output_maps) * 1024
    q_size = fia_quantization_shaper_size_kb(filter_copies) * 1024
    out_size = fia_output_shaper_size_kb(filter_copies, parallel_output_maps) * 1024

    v_out, v_tmp0, v_tmp = scratchpad_required(node, vl, tile)
    sp_used = sp_used_fn(node)(v_out,v_tmp0, v_tmp)

    k, c, kh, kw = node.Conv2DOptions.filter_shape_dims
    sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width

    if node.Conv2DOptions.use_depthwise:
        w_used = kh*kw*tile[IMAPS]
        if node.Conv2DOptions.use_db:
            w_used *= 2
    elif node.Conv2DOptions.fit_weights:
        w_used = kh*kw*c*k
    elif (not node.Conv2DOptions.use_db):
        w_used = kh*kw*node.channels*tile[MAPS]
    else:
        w_used = kh*kw*tile[IMAPS]*tile[MAPS]

    if node.Conv2DOptions.split_weight_shaper_buffers and not node.Conv2DOptions.use_db:
        w_used *= 2
    if not node.Conv2DOptions.fit_weights and node.Conv2DOptions.use_db and not node.Conv2DOptions.conv_rows:
        w_used *= 2

    w_total = (kh*kw*c*k)

    if node.Conv2DOptions.use_depthwise:
        q_total = k*c*16
    else:
        q_total = k*16
    q_used = tile[MAPS]*16

    if node.Conv2DOptions.use_db and node.Conv2DOptions.conv_rows:
        in_used = tile[IMAPS]*node.Conv2DOptions.conv_rows*tile[COLUMNS]*2
    elif node.Conv2DOptions.use_db:
        in_used = tile[IMAPS]*tile[ROWS]*tile[COLUMNS]*2
    else:
        in_used = tile[IMAPS]*tile[ROWS]*tile[COLUMNS]

    out_used = tile[MAPS]*((tile[ROWS]-(kh-1))+(sh-1))//sh*((tile[COLUMNS]-(kw-1))+(sw-1))//sw

    print('in {}% ({}, {})'.format( int(100. * in_used/ in_size), in_used, in_size))
    print('out {}% ({}, {})'.format( int(100. * out_used/ out_size), out_used, out_size))
    print('sp {}% ({}, {}, {})'.format( int(100. * sp_used/ sp_size), sp_used, sp_size, (v_out, v_tmp0, v_tmp)))
    print('wt {}% ({}, {}, {})'.format( int(100. * w_used/ w_size), w_used, w_size, w_total))
    print('qt {}% ({}, {}, {})'.format( int(100. * q_used/ q_size), q_used, q_size, q_total))



'''
start w/ minimum viable tile size (produces some valid output)
maximize (up to a limit), as specific tile dimension
'''
def tile_subgraph(node, preset):
    sp = preset_select['SCRATCHPAD_KB'][preset]*1024 - 256 #TODO get exact indirect
    vl = preset_select['VECTOR_LANES'][preset]
    filter_copies = preset_select['FILTER_COPIES'][preset]
    parallel_output_maps = preset_select['PARALLEL_OUTPUT_MAPS'][preset]

    BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6

    # STEP 1: get minimum tile (that produces a viable output)
    fx = compose_subgraph(node, min_output_tile)
    # tile = minimum_tile_subgraph(node, fx)
    tile = minimum_valid_tile_subgraph(node, vl, sp, fx)
    # print("minimum_valid_tile_subgraph", tile)

    # STEP 2: grow tile dimensions (within scratchpad capacity) according to per-operator rules
    e = node.type
    if e == BuiltinOperator.CONV_2D:
        v = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset)
        is_allocated = False

        if FORCE_SB:
            node.Conv2DOptions.conv_rows = 0
            node.Conv2DOptions.use_db = 0

        # disable conv_rows for depthwise
        if node.Conv2DOptions.use_depthwise:
            node.Conv2DOptions.conv_rows = 0

        if not node.Conv2DOptions.use_fia: # scalar + vector
            tile = increment_tile(node, vl, sp, fx, tile, COLUMNS, limit=node.n)
            tile = increment_tile(node, vl, sp, fx, tile, ROWS, limit=node.m)
            tile = increment_tile(node, vl, sp, fx, tile, MAPS, limit=node.Conv2DOptions.kernels)
            is_allocated = True

        elif node.Conv2DOptions.use_depthwise: # scalar + vector
            depthwise_db_maps, tile_depthwise_db_maps =  fit_depthwise(tile, node, vl, sp, fx, preset, use_db=1, row_db=1)
            depthwise_single_buffer, tile_depthwise_single_buffer =  fit_depthwise(tile, node, vl, sp, fx, preset, use_db=0, row_db=0)

            if node.Conv2DOptions.use_db:
                tile = tile_depthwise_db_maps
                is_allocated = depthwise_db_maps
                node.Conv2DOptions.split_weight_shaper_buffers = 0
            else:
                node.Conv2DOptions.use_db = 0
                tile = tile_depthwise_single_buffer
                is_allocated = depthwise_single_buffer

        else: # accelerator

            conv_db_rows, tile_conv_db_rows, rows_db_rows =  fit_conv(tile, node, vl, sp, fx, preset, use_db=1, row_db=1)
            conv_db_maps, tile_conv_db_maps, rows_db_maps =  fit_conv(tile, node, vl, sp, fx, preset, use_db=1, row_db=0)
            conv_single_buffer, tile_conv_single_buffer, rows_single_buffer =  fit_conv(tile, node, vl, sp, fx, preset, use_db=0, row_db=0)

            conv_single_buffer2, tile_conv_single_buffer2, rows_single_buffer2 =  fit_conv(tile, node, vl, sp, fx, preset, use_db=0, row_db=0, split_weights=0)

            is_fc = (node.n == 1 and node.m == 1)

            #TODO fix conv rows DB 
            kh = node.Conv2DOptions.filter_shape_dims[-2]
            if 0 and kh == 1 and node.Conv2DOptions.use_db and (5 < rows_db_rows < tile_conv_db_rows[ROWS] < node.m):
                tile = tile_conv_db_rows
                is_allocated = conv_db_rows
                node.Conv2DOptions.conv_rows = rows_db_rows
                node.Conv2DOptions.split_weight_shaper_buffers = 0

            elif tile_conv_single_buffer[IMAPS] == node.channels and not is_fc: #prefer all imaps (DMA input/weights once)
                node.Conv2DOptions.use_db = 0
                node.Conv2DOptions.conv_rows = 0
                tile = tile_conv_single_buffer
                is_allocated = conv_single_buffer

            elif tile_conv_single_buffer2[IMAPS] == node.channels and not is_fc: #same but w/o split weight buffers
                node.Conv2DOptions.use_db = 0
                node.Conv2DOptions.conv_rows = 0
                node.Conv2DOptions.split_weight_shaper_buffers = 0
                tile = tile_conv_single_buffer2
                is_allocated = conv_single_buffer2

            elif node.Conv2DOptions.use_db and tile_conv_db_maps[IMAPS] < node.channels: #DB maps if not all channels
                tile = tile_conv_db_maps
                is_allocated = conv_db_maps
                node.Conv2DOptions.conv_rows = 0
                node.Conv2DOptions.split_weight_shaper_buffers = 0

            else: #fall through to single buffer
                node.Conv2DOptions.use_db = 0
                node.Conv2DOptions.conv_rows = 0
                tile = tile_conv_single_buffer
                is_allocated = conv_single_buffer


            if not is_allocated:
                print('not allocated')
                if not v(tile[MAPS],tile[IMAPS],tile[ROWS],tile[COLUMNS]):
                    return None

                limit_cols = node.n # adjust COLUMN limit to be maximum valid fia COLUMN limit
                while limit_cols > tile[COLUMNS] and not v(tile[MAPS],tile[IMAPS],tile[ROWS],limit_cols):
                    limit_cols -= 1
                tile = increment_tile(node, vl, sp, fx, tile, COLUMNS, limit=limit_cols)

                
                limit_imaps = node.Conv2DOptions.filter_shape_dims[1]
                while not v(tile[MAPS],limit_imaps,tile[ROWS],tile[COLUMNS]):
                    limit_imaps -= 1
                tile = increment_tile(node, vl, sp, fx, tile, IMAPS, limit=limit_imaps)

                limit_maps = node.Conv2DOptions.filter_shape_dims[0]
                while limit_maps > tile[MAPS] and not v(limit_maps,tile[IMAPS],tile[ROWS],tile[COLUMNS]):
                    limit_maps -= 1
                limit_maps = limit_maps // parallel_output_maps * parallel_output_maps
                tile = increment_tile(node, vl, sp, fx, tile, MAPS, limit=limit_maps)

                limit_rows = node.m
                while limit_rows > tile[ROWS] and not v(tile[MAPS],tile[IMAPS],limit_rows, tile[COLUMNS]):
                    limit_rows -= 1
                tile = increment_tile(node, vl, sp, fx, tile, ROWS, limit=limit_rows)


    elif e == BuiltinOperator.FULLY_CONNECTED:
        output_depth, accum_depth = node.FullyConnectedOptions.filter_shape_dims

        tile = increment_tile(node, vl, sp, fx, tile, ICOLUMNS, limit=accum_depth)
        if node.FullyConnectedOptions.use_fia:
            v = lambda c: valid_fia_fc(accum_depth, 1, c, preset) #TODO allow partial accum
            limit_depth = output_depth
            while not v(limit_depth):
                limit_depth -= 1
            if limit_depth < 1:
                return None

            tile = increment_tile(node, vl, sp, fx, tile, COLUMNS, limit=limit_depth)
            tile = increment_tile(node, vl, sp, fx, tile, IROWS, limit=limit_depth)
        else:
            tile = increment_tile(node, vl, sp, fx, tile, COLUMNS, limit=output_depth)
            tile = increment_tile(node, vl, sp, fx, tile, IROWS, limit=output_depth)

    else: # common case
        tile = increment_tile(node, vl, sp, fx, tile, COLUMNS, limit=node.n)
        tile = increment_tile(node, vl, sp, fx, tile, ROWS, limit=node.m)
        tile = increment_tile(node, vl, sp, fx, tile, MAPS, limit=node.channels)

    # if  e == BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
    # STEP 3: for a given tile, determine how to walk vnnx_graph
    set_tile_attr(node, vl, sp, tile)
    
    return tile
