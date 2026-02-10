import numpy as np
from .vnnx_types import *
from .utils import sp_invalid_exception_catcher


BATCH, IMAPS, IROWS, ICOLUMNS, MAPS, ROWS, COLUMNS = 0, 1, 2, 3, 4, 5, 6

def constrain_channels(c_padded, c_max, k_tile, weight_mem, compress_ratio, kh, kw, c_parallel):
    c_max = min(c_max, int(np.floor(weight_mem / (k_tile / compress_ratio*kh*kw))))
    c_count = int(np.ceil(c_padded/c_max))
    c_tile  = int(np.ceil(c_padded/c_count))
    if c_tile < c_padded:
        while c_tile % c_parallel != 0:
            c_tile -= 1
    if c_padded / c_tile > c_count: # increase c_count if c_tile decremented increased the count
        c_count += 1
        new_c_tile = int(np.ceil(c_padded/c_count)) # grab new c_tile for incremented c_count
        while new_c_tile % c_parallel != 0 and new_c_tile < c_tile:
            new_c_tile += 1 # increase up to c_tile
        c_tile = new_c_tile

    return c_tile, c_count

def constrain_weight_shaper(tiling, params, compress_ratio, weight_mem, c_parallel, k_parallel):
    c_padded, k_padded, kh, kw = params['pc'], params['pk'], params['kh'], params['kw']
    k_max = int(np.floor(weight_mem / (c_padded / compress_ratio*kh*kw)))
    k_max = min(k_max, k_padded)
    k_tile = (k_max//k_parallel)*k_parallel

    if k_tile < k_parallel:
        k_tile = k_parallel
        c_tile, c_count = constrain_channels(c_padded, c_padded, k_tile, weight_mem, compress_ratio, kh, kw, c_parallel)
    else:
        c_count = 1
        c_tile = c_padded

    k_count = int(np.ceil(k_padded/k_tile))
    if k_tile > k_parallel:   # find the minimum tile size that still has the same number of tiles
        while int(np.ceil(k_padded / (k_tile - k_parallel))) == k_count:
            k_tile -= k_parallel
            if k_tile == k_parallel:
                break

    tiling['k_tile'] = k_tile
    tiling['k_count'] = k_count
    tiling['c_tile'] = c_tile
    tiling['c_count'] = c_count
    return tiling


def constrain_input_shaper(tiling, params, compress_ratio, input_mem, weight_mem, c_parallel, \
                           sp=None, opcode=None, tmp_dir=None, graph_idx=None, tmp_dir_obj=None):
    k_tile, c_tile, c_count = tiling['k_tile'], tiling['c_tile'], tiling['c_count']
    c_padded, iw, ph, pw, oh, ow = params['pc'], params['iw'], params['ph'], params['pw'], params['oh'], params['ow'] 
    min_h, min_w, sh, sw, kh, kw = params['mh'], params['mw'], params['sh'], params['sw'], params['kh'], params['kw'] 

    ih_max = int(np.floor(input_mem / (c_tile*(iw+pw)) - ph))
    ih_max -= sh-1
    iw_max = iw
    if sw == 2:
        if iw_max % 2:
            iw_max += 1

    while ih_max < min_h and c_tile // 2 >= c_parallel:
        c_tile, c_count = constrain_channels(c_padded, c_tile // 2, k_tile, weight_mem, compress_ratio, kh, kw, c_parallel)
        ih_max = int(np.floor(input_mem / (c_tile*(iw_max+pw)) - ph))
        ih_max -= sh-1

    while iw_max // 2 > min_w and ih_max < min_h:
        iw_max //= 2
        ih_max = int(np.floor(input_mem / (c_tile*(iw_max+pw)) - ph))
        ih_max -= sh-1

    oh_tile, oh_count = output_from_input(ih_max, oh, sh, sp=sp, opcode=opcode, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)
    ow_tile, ow_count = output_from_input(iw_max, ow, sw, sp=sp, opcode=opcode, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)

    tiling['c_tile'], tiling['c_count'] = c_tile, c_count
    tiling['oh_tile'], tiling['oh_count'] = oh_tile, oh_count
    tiling['ow_tile'], tiling['ow_count'] = ow_tile, ow_count
    return tiling



def output_from_input(isize, max_sz, stride, sp=None, opcode=None, tmp_dir=None, graph_idx=None, tmp_dir_obj=None):
    if stride > 1:
        sz = (isize-(stride-1))//stride+1
    else:
        sz = isize
    sz = min(sz, max_sz)
    count = int(np.ceil(max_sz/sz))
    tile = int(np.ceil(max_sz/count))

    # assert tile > 0
    if tile<=0:
        with sp_invalid_exception_catcher(sp, opcode, tmp_dir, graph_idx, tmp_dir_obj):
            assert(0)
    return tile, count


def input_from_output(sz, max_sz, stride):
    if stride > 1:
        isize = (sz-(stride-1))*stride+1
        if isize == (max_sz - 1):
            isize = max_sz
    else:
        isize = sz

    return isize


def constrain_output_shaper(tiling, params, output_mem, acc_mem, k_parallel):
    c_count, k_count, k_tile = tiling['c_count'], tiling['k_count'], tiling['k_tile']
    oh_count, oh_tile, ow_count, ow_tile = tiling['oh_count'], tiling['oh_tile'], tiling['ow_count'], tiling['ow_tile']
    k_padded, oh = params['pk'], params['oh']

    tile_size = k_tile*oh_tile*ow_tile
    if c_count > 1:
        output_sz = acc_mem
    else:
        output_sz = output_mem

    while tile_size > output_sz:   # if limited by output shaper
        if k_tile > k_parallel:
            if k_tile % k_parallel != 0:
                k_tile -= k_tile % k_parallel
            else:
                k_tile -= k_parallel
            k_count = int(np.ceil(k_padded/k_tile))

            if k_tile > k_parallel:   # find the minimum tile size that still has the same number of tiles
                while int(np.ceil(k_padded/(k_tile-k_parallel)))==k_count:
                    k_tile -= k_parallel
                    if k_tile == k_parallel:
                        break
        else:
            oh_max = int(np.floor((output_sz)/(k_tile*ow_tile)))
            if oh_max < 1:
                while np.ceil(ow_tile / 2) > 1 and oh_max < 1:
                    ow_count *= 2
                    ow_tile = int(np.ceil(ow_tile / 2))
                    oh_max = int(np.floor((output_sz)/(k_tile*ow_tile)))
            oh_count = int(np.ceil(oh/oh_max))
            oh_tile = int(np.ceil(oh/oh_count))

        tile_size = k_tile*oh_tile*ow_tile

    tiling['k_tile'], tiling['k_count'] = k_tile, k_count
    tiling['oh_tile'], tiling['oh_count'] = oh_tile, oh_count
    tiling['ow_tile'], tiling['ow_count'] = ow_tile, ow_count
    return tiling


def constrain_fia_tile(input, kernel, output, min_input, compress_ratio, stride, dilation, c_parallel, k_parallel, input_mem, weight_mem, output_mem, \
                       sp=None, opcode=None, tmp_dir=None, graph_idx=None, tmp_dir_obj=None):
    k,c,ih,iw = output[-3], input[-3], input[-2], input[-1]
    ph,pw = (kernel[-2] - 1)*dilation[-2], (kernel[-1] - 1)*dilation[-1]
    kh,kw = kernel[-2], kernel[-1]
    sh,sw = stride[-2], stride[-1]

    c_padded = int(np.ceil(c / c_parallel)*c_parallel)
    k_padded = int(np.ceil(k / k_parallel)*k_parallel)
    weights = np.prod(kernel) / compress_ratio

    buffers = 2
    wmem = weight_mem / buffers
    imem = input_mem / buffers
    omem = output_mem / buffers
    accmem = 32*256

    params = {}
    params['c'], params['ih'], params['iw'] = input[-3], input[-2], input[-1]
    params['k'], params['oh'], params['ow'] = output[-3], output[-2], output[-1]
    params['pk'], params['pc'] = k_padded, c_padded
    params['ph'], params['pw'] = (kernel[-2] - 1)*dilation[-2], (kernel[-1] - 1)*dilation[-1]
    params['kh'], params['kw'] = kernel[-2], kernel[-1]
    params['sh'], params['sw'] = stride[-2], stride[-1]
    params['mh'], params['mw'] = min_input[-2], min_input[-1]

    tiling = {_: -1 for _ in ['k_tile','k_count','c_tile','c_count','oh_tile','oh_count','ow_tile','ow_count']}
    # constrain weight memory
    tiling = constrain_weight_shaper(tiling, params, compress_ratio, wmem, c_parallel, k_parallel)

    # constrain input memory
    tiling = constrain_input_shaper(tiling, params, compress_ratio, imem, wmem, c_parallel, sp=sp, opcode=opcode, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)

    tiling['k_tile'],tiling['c_tile'] = min(k, tiling['k_tile']), min(c, tiling['c_tile'])

    # constrain output memory
    tiling = constrain_output_shaper(tiling, params, omem, accmem, k_parallel)


    w_tile = input_from_output(tiling['ow_tile'], iw, sw)
    h_tile = input_from_output(tiling['oh_tile'], ih, sh)

    out_sz = tiling['k_tile']*tiling['oh_tile']*tiling['ow_tile']
    in_sz = tiling['c_tile'] * (h_tile+ph) * (w_tile+pw)
    w_sz = out_sz*(in_sz/compress_ratio*kh*kw)

    tile_count = tiling['c_count'] * tiling['k_count'] * tiling['oh_count'] * tiling['ow_count']

    tile = [1 for _ in range(7)]
    tile[IMAPS] = tiling['c_tile']
    tile[MAPS] = tiling['k_tile']
    tile[ROWS] = h_tile + ph
    tile[COLUMNS] = w_tile + pw

    return tile, tile_count


def allocate_fia_tile(node, preset, vl, sp, sparse, input_mem, weight_mem, output_mem, tile, min_tile, is_sp_invalid, valid_fia, k_parallel, \
                      opcode, tmp_dir, graph_idx, tmp_dir_obj):
    kk, kc, kh, kw = node.Conv2DOptions.filter_shape_dims
    sh, sw = node.Conv2DOptions.stride_height, node.Conv2DOptions.stride_width
    dhf, dwf = node.Conv2DOptions.dilation_height_factor, node.Conv2DOptions.dilation_width_factor
    ph, pw = node.Conv2DOptions.padding_height, node.Conv2DOptions.padding_width

    assert(ph == 0 and pw == 0)
    ph, pw = kh - 1, kw - 1
    ph = kh - 1
    ih = node.m - ph
    oh = (ih + ph - (1 + (kh-1)*dhf)) // sh + 1
    iw = node.n - pw
    ow = (iw + pw - (1 + (kw-1)*dwf)) // sw + 1

    
    ishape = [1, node.channels, ih, iw]
    kshape = [kk,kc,kh,kw]
    oshape = [1, kk, oh, ow]
    min_input = [1, 1, min_tile[ROWS], min_tile[COLUMNS]]
    compress_ratio = 1
    if node.Conv2DOptions.repeat == 1:
        compress_ratio = 4
    elif node.Conv2DOptions.repeat == 2:
        compress_ratio = 2

    padded_channels = 2
    if (sparse == 1):
        padded_channels = 8
    
    t, count = constrain_fia_tile(ishape, kshape, oshape, min_input, compress_ratio, (sh,sw), (dhf,dwf), padded_channels, k_parallel, input_mem, weight_mem, output_mem, \
                                  sp=sp, opcode=opcode, tmp_dir=tmp_dir, graph_idx=graph_idx, tmp_dir_obj=tmp_dir_obj)

    if not all([t_ >= m for t_,m in zip(t,min_tile)]):
        return None


    vt = lambda m,im,r,c : valid_fia(m, im, r, c, node, preset, use_db=1, db_rows=0, split_weights=0, sparse=sparse)
    if vt(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
        sp_invalid = is_sp_invalid(node, vl, sp, t, sparse=sparse)
        while(sp_invalid and t[MAPS] >= 2*k_parallel):
            # ensure that general tile is a multiple of k_parallel (if it wasn't, it was likely doing all maps)
            if t[MAPS] % k_parallel != 0:
                t[MAPS] -= (t[MAPS] % k_parallel)
            else:
                t[MAPS] -= k_parallel
            if vt(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
                sp_invalid = is_sp_invalid(node, vl, sp, t, sparse=sparse)
        while(sp_invalid):
            t[ROWS] -= 1
            while t[ROWS] and not vt(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
                t[ROWS] -= 1
            if t[ROWS] and vt(t[MAPS],t[IMAPS],t[ROWS],t[COLUMNS]):
                sp_invalid = is_sp_invalid(node, vl, sp, t, sparse=sparse)
            else:
                # print('BAD', "couldn't find valid tile w/ valid scratchpad size", t, '#sn', len(node.subnode_array))
                # with sp_invalid_exception_catcher(sp, opcode, tmp_dir, graph_idx, tmp_dir_obj):
                #     assert(0)
                t = None
                break
    else:
        # print('BAD', 'invalid tile', tile, 'w/', t, '#sn', len(node.subnode_array))
        # with sp_invalid_exception_catcher(sp, opcode, tmp_dir, graph_idx, tmp_dir_obj):
        #         assert(0)
        t = None
    return t


if __name__ == "__main__":

    for l in layer:
        params, count, tile = constrain_fia_tile(l['inputShapes'][0], l['kernel'], l['outputShapes'][0], [1,1,1,1],
                                                 l['compRatio'], 1, 8, 16, 32*1024)
        print('i', l['inputShapes'][0], 'o', l['outputShapes'][0])
        print(params, 'x{}'.format(count), tile)
