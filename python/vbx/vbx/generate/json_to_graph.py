import enum
import json
import itertools
import struct
import sys
import os.path
import base64
from math import floor,ceil,log2
import argparse
import numpy as np
import hashlib


VERBOSE = 0
NEXT = True

Q32 = 16
Q16 = 13
Q8 = 7
U8 = 8
ACCUMULATORS = 16
COEFFICIENT_BITS = 8
INPUT_SIZE_BYTES = 1
OUTPUT_SIZE_BYTES = 1
DOTP_MODE = 1
MULTIPLIER_BITS = 18
ACCUMULATOR_FXP_BITS = 3
SCALE_FXP_BITS = 8
USE_CVI = 1
CORES = 1


def printv(*args):
    if VERBOSE:
        print(*args)


Graph_struct = [
                ('uint32_t', 'version'),
                ('int32_t', 'vbx_nn_preset'),
                ('int32_t', 'num_inputs'),
                ('int32_t', 'num_outputs'),
                ('int32_t', 'data_bytes'),
                ('int32_t', 'allocate_bytes'),
                ('offset', 'io_nodes'),
                ('int32_t', 'num_layers'),
                ('offset', 'replay_buffer'),
                ('int32_t', 'replay_buffer_size'),
                ('int32_t', 'magic'),
                ]

Node_struct = [('int32_t', 'type'),
               ('int32_t', 'input_data_type'),
               ('int32_t', 'output_data_type'),
               ('int32_t', 'input_unsigned'),
               ('int32_t', 'output_unsigned'),
               ('int32_t', 'input_size'),
               ('int32_t', 'output_size'),
               ('int32_t[2]', 'output_strides'),
               ('int32_t', 'scratchpad_bytes'),
               ('int32_t', 'dma_split'),
               ('int32_t', 'dma_channel_offset'),
               ('int32_t', 'dma_input_buffer_offset'),
               ('int32_t', 'dma_output_buffer_offset'),
               ('offset', 'input_data'),
               ('offset', 'output_data'),
               ('int8_t[24]', 'input_description'),
               ('int8_t[24]', 'output_description'),
               ('offset', 'test_input_data'),
               ('offset', 'test_output_data'),
               ('offset', 'sublayers'),
               ('float', 'output_scale_factor'),
               ('int32_t', 'num_sublayers'),
               ('int32_t[2]', 'sublayer_stride'),
               ('int32_t[2]', 'sublayer_shape'),
               ('int32_t[2]', 'sublayer_shape_0'),
               ('int32_t[2]', 'sublayer_shape_full'),
               ('int32_t[2]', 'sublayer_shape_last'),
               ('int32_t', 'sublayer_rows'),
               ('int32_t', 'sublayer_columns'),
               ('int32_t', 'sublayer_scratchpad_per_map'),
               # replay is an offset in vnnx-types.h,
               # but not here so it stays initialized to zero
               ('int32_t', 'use_replay'),
               ('int64_t', 'replay_buffer'),
               ('int32_t', 'replay_buffer_size'),
               ('int32_t[3]','input_shape'),
               ('int32_t[3]','output_shape'),
               ('union', {'conv': [('int32_t', 'fxp_scalar'),
                                   ('int32_t', 'bias_scalar'),
                                   ('int32_t', 'bias_lower_scalar'),
                                   ('int32_t', 'kernels'),
                                   ('int32_t', 'channels'),
                                   ('int32_t[2]', 'kernel_shape'),
                                   ('int32_t[2]', 'strides'),
                                   ('int32_t[2]', 'dilations'),
                                   ('int32_t', 'group'),
                                   ('int32_t', 'm'),
                                   ('int32_t', 'n'),
                                   ('int32_t', 'padded_kernels'),
                                   ('int32_t', 'padded_channels'),
                                   ('int32_t', 'imaps'),
                                   ('int32_t', 'maps'),
                                   ('int32_t', 'acc_maps'),
                                   ('int32_t', 'rows'),
                                   ('int32_t', 'cols'),
                                   ('int32_t', 'inc_rows'),
                                   ('int32_t', 'conv_rows'),
                                   ('int32_t', 'core_split'),
                                   ('int32_t', 'core_maps'),
                                   ('int32_t', 'core_m'),
                                   ('int32_t', 'use_weights32'),
                                   ('int32_t', 'use_cvi'),
                                   ('int32_t', 'use_depthwise'),
                                   ('int32_t', 'use_strided'),
                                   ('float', 'max_weight'),
                                   ('offset', 'weights'),
                                   ('offset', 'weights32'),
                                   ('offset', 'biases'),
                                   ('offset', 'biases_lower'),
                                   ('offset', 'scale')],
                          'sum':[('int32_t', 'channels'),
                                 ('int32_t', 'm'),
                                 ('int32_t', 'n'),
                                 ('int32_t', 'num_inputs'),
                                 ('int32_t', 'maps'),
                                 ('int32_t', 'rows')],
                          'max':[('int32_t', 'channels'),
                                    ('int32_t', 'm'),
                                    ('int32_t', 'n'),
                                    ('int32_t', 'num_inputs')],
                          'min':[('int32_t', 'channels'),
                                    ('int32_t', 'm'),
                                    ('int32_t', 'n'),
                                    ('int32_t', 'num_inputs')],
                          'argmax':[('int32_t', 'channels'),
                                    ('int32_t', 'm'),
                                    ('int32_t', 'n'),
                                    ('int32_t', 'pixels_per_loop')],
                          'identity':[('int32_t', 'channels'),
                                      ('int32_t', 'm'),
                                      ('int32_t', 'n'),
                                      ('int32_t', 'maps'),
                                      ('int32_t', 'core_split'),
                                      ('int32_t', 'core_maps'),
                                      ('int32_t', 'core_m'),
                                      ('int32_t', 'rows')],
                          'gemm':[('int32_t', 'max_input_size'),
                                  ('int32_t', 'max_output_size'),
                                  ('int32_t', 'input_size'),
                                  ('int32_t', 'output_size'),
                                  ('offset', 'weights'),
                                  ('offset', 'biases')],
                          'lrn':[('float', 'alpha'),
                                 ('float', 'beta'),
                                 ('float', 'bias'),
                                 ('float', 'scale'),
                                 ('int32_t', 'size'),
                                 ('int32_t', 'channels'),
                                 ('int32_t', 'm'),
                                 ('int32_t', 'n'),
                                 ('int32_t', 'maps'),
                                 ('int32_t', 'rows')],
                          'transpose':[('int32_t', 'channels'),
                                       ('int32_t', 'm'),
                                       ('int32_t', 'n'),
                                       ('int32_t[3]', 'permutation'),
                                       ('int32_t','out_maps_at_once'),
                                       ('int32_t','out_rows_at_once')],
                          'resize':[('float[2]', 'scale'),
                                    ('int32_t', 'mode'),
                                    ('int32_t', 'channels'),
                                    ('int32_t', 'm'),
                                    ('int32_t', 'n'),
                                    ('int32_t', 'maps'),
                                    ('int32_t', 'rows')],
                          'tile':[('int32_t[3]', 'tile'),
                                    ('int32_t', 'channels'),
                                    ('int32_t', 'm'),
                                    ('int32_t', 'n'),
                                    ('int32_t', 'maps'),
                                    ('int32_t', 'rows')],
                          'reduce':[('int32_t', 'channels'),
                                    ('int32_t', 'm'),
                                    ('int32_t', 'm0'),
                                    ('int32_t', 'n')],
                          'reorg':[('int32_t', 'stride'),
                                    ('int32_t', 'channels'),
                                    ('int32_t', 'm'),
                                    ('int32_t', 'n'),
                                    ('int32_t', 'maps'),
                                    ('int32_t', 'rows')],
                          'activation':[('offset', 'scale'),
                                     ('int32_t', 'mode'),
                                     ('int32_t', 'channels'),
                                     ('int32_t', 'm'),
                                     ('int32_t', 'n'),
                                     ('int32_t', 'maps'),
                                     ('int32_t', 'rows')]
                         })]

Subnode_struct = [('int32_t', 'type'),
                  ('int32_t', 'input_data_type'),
                  ('int32_t', 'output_data_type'),
                  ('int32_t[2]', 'strides'), 
                  ('int32_t[2]', 'kernel_shape'),
                  ('int32_t[2]', 'dilations'),
                  ('int32_t[6]', 'pads'),  
                  ('int32_t', 'maps'), 
                  ('union', {"pad_const": [('float', 'value')],
                             "clip": [('float', 'min'),
                                      ('float', 'max')],
                             "depthwise": [('int32_t', 'unsigned_input'),
                                           ('int32_t', 'unsigned_output'),
                                           ('offset', 'weights')],
                             "prelu":[('offset', 'slope')],
                             "leakyrelu":[('int32_t', 'alpha')],
                             "mul_scalar":[("int32_t", "use_xl"),
                                        ("float", "scalarf32"),
                                        ('int32_t', 'scalar32'),
                                        ('int16_t', 'scalar16'),
                                        ('int8_t', 'scalar8'),
                                        ('uint8_t', 'scalaru8'),
                                        ('pad[1]', 'padding')],
                             "add_broadcast_map": [('int32_t', 'use_xl'),
                                         ('offset', 'array'),
                                         ('offset', 'array_xl')],
                             "add_broadcast_row": [('int32_t', 'use_xl'),
                                         ('offset', 'array'),
                                         ('offset', 'array_xl')],
                             "mul_broadcast_map":[('int32_t', 'use_xl'),
                                        ('offset', 'array'),
                                        ('offset', 'array_xl')],
                             "mul_broadcast_row":[('int32_t', 'use_xl'),
                                        ('offset', 'array'),
                                        ('offset', 'array_xl')],
                             "cast":[('int32_t', 'scale')],
                             "prefetch":[('offset', 'memory_offset')]}
                   )]

def type_string_to_fmt(t):
    fmt_dic = {'pad': 'x',
               'int32_t': 'i',
               'uint32_t': 'I',
               'uint64_t': 'Q',
               'int64_t': 'q',
               'int8_t': 'b',
               'uint8_t': 'B',
               'int16_t': 'h',
               'offset': 'q',
               'float': 'f'}
    if t.find('[') >= 0:
        ty, sz = t.split('[')
        sz = sz.strip(']')
        sz = int(sz)
        return "{}{}".format(sz, fmt_dic[ty])
    else:
        return fmt_dic[t]


class Struct:
    class _union_class:
        def __init__(self):
            self.ordered_attributes = []
            self.offset_attributes = []

    def __init__(self, description):
        self.description = description.copy()
        self.subnode_array = []
        self.ordered_attributes = []
        self.union_names = []
        self.union_fmt = {"":""}
        self.fmt = "<"
        self.offset_attributes = []

        for typ, name in self.description:
            if typ == "union":
                union = name
                for union_name in union.keys():
                    self.union_names.append(union_name)
                    setattr(self, union_name, Struct._union_class())
                    self.union_fmt[union_name]= ""
                    u_obj = getattr(self, union_name)
                    for ty,nm in union[union_name]:
                        self.union_fmt[union_name] += type_string_to_fmt(ty)
                        setattr(u_obj, nm, None)
                        if nm != 'padding':
                            u_obj.ordered_attributes.append(nm)
                        if ty == 'offset':
                            u_obj.offset_attributes.append(nm)
            else:
                if typ == 'offset':
                    self.offset_attributes.append(name)
                self.fmt += type_string_to_fmt(typ)
                if name != 'padding':
                    self.ordered_attributes.append(name)
                setattr(self, name, None)

    def get_union(self):
        union_str = enum_to_union_name(self.type)
        if len(union_str):
            union = getattr(self,union_str)
        else:
            union = Struct._union_class()
        return union

    def update_offsets(self,offset):
        union = self.get_union()
        for attr in self.offset_attributes:
            orig = getattr(self,attr)
            if orig is not None:
                setattr(self,attr,orig+offset)
        for attr in union.offset_attributes:
            orig = getattr(union,attr)
            if orig is not None:
                setattr(union,attr,orig+offset)

    def get_structured_data(self):
        struct_format = self.fmt + self.union_fmt[enum_to_union_name(self.type)]
        ordered_data = [ getattr(self,nm) for nm in self.ordered_attributes]
        union = self.get_union()

        ordered_union_data = [ getattr(union,nm) for nm in union.ordered_attributes]

        pad_bytes = self.get_structure_size() - struct.calcsize(struct_format)
        struct_format += '{}x'.format(pad_bytes)

        assert struct.calcsize(struct_format) == self.get_structure_size()

        data = []

        for d in ordered_data + ordered_union_data:
            if hasattr(d,'__iter__'):
                data += d
            else:
                data.append(d)
        try:
            packed_bytes = struct.pack(struct_format,*data)
        except struct.error as e:
            print(data)
            for t,n in self.description:
                union_str = enum_to_union_name(self.type)
                if t != 'union':
                    sys.stderr.write("{}\t{}\n".format(getattr(self,n),n))
                elif len(union_str):
                    union = getattr(self,union_str)
                    for t1,n1 in n[union_str]:
                        sys.stderr.write("{}\t{}.{}\n".format(getattr(union,n1),union_str,n1))

            raise e
        return packed_bytes

    def get_structure_size(self):
        union_struct_formats = [self.fmt+ufmt for ufmt in self.union_fmt.values()]
        return max([struct.calcsize(f) for f in union_struct_formats])


class Graph(Struct):
    def __init__(self):
        super().__init__(Graph_struct)
        self.type = None
        self.magic = 0x1ABE11ED


class Node(Struct):
    def __init__(self):
        super().__init__(Node_struct)
        #these are set at runtime
        self.replay_buffer =0;
        self.replay_buffer_size = 0;
        self.output_strides = [1,1];
        self.dma_split = 1;


class Subnode(Struct):
    def __init__(self):
        super().__init__(Subnode_struct)
        self.kernel_shape=[1,1]
        self.strides = [1,1]
        self.dilations = [1,1]
        self.pads = [0,0,0,0,0,0]
        self.maps = 0

        #####
        ## TODO: I don't think these attributes are used
        self.input_data_type = 0
        self.output_data_type = 0


class resize_mode(enum.IntEnum):
    NEAREST = 0
    LINEAR = 1

class activation_mode(enum.IntEnum):
    SOFTMAX = 0
    SIGMOID = 1
    TANH = 2
    MISH = 3
    ELU = 4
    SELU = 5
    SWISH = 6
    HTANH = 7
    HSWISH = 8


class calc_type(enum.IntEnum):
    UINT8 = 0
    INT8 = 1
    INT16 = 2
    INT32 = 3
    UNKNOWN = 4


def sizeof_calc_type(t):
       return {
            calc_type.UINT8 : 1,
            calc_type.INT8  :1,
            calc_type.INT16 :2,
            calc_type.INT32 :4,
        } [t]


class subgraph_type(enum.IntEnum):
    CONV = 0
    GEMM = 1
    SUM = 2
    IDENTITY = 3
    LRN = 4
    TRANSPOSE = 5
    ACTIVATION = 6
    RESIZE = 7
    REORG = 8
    ARGMAX = 9
    REDUCEMEAN = 10
    TILE = 11
    MAX = 12
    MIN = 13
    UNKNOWN = 14

    def from_str(e):
        e = e.upper()
        if e == "CONV":
            return subgraph_type.CONV
        if e == "GEMM":
            return subgraph_type.GEMM
        if e == "SUM":
            return subgraph_type.SUM
        if e == "IDENTITY":
            return subgraph_type.IDENTITY
        if e == "LRN":
            return subgraph_type.LRN
        if e == "TRANSPOSE":
            return subgraph_type.TRANSPOSE
        if e == "SOFTMAX":
            return subgraph_type.ACTIVATION
        if e == "SIGMOID":
            return subgraph_type.ACTIVATION
        if e == "RESIZE":
            return subgraph_type.RESIZE
        if e == "REORG":
            return subgraph_type.REORG
        if e == "ARGMAX":
            return subgraph_type.ARGMAX
        if e == "REDUCEMEAN":
            return subgraph_type.REDUCEMEAN
        if e == "TILE":
            return subgraph_type.TILE
        if e == "MAX":
            return subgraph_type.MAX
        if e == "MIN":
            return subgraph_type.MIN
        return subgraph_type.UNKNOWN


class layer_type(enum.IntEnum):
    GLOBAL_AVGPOOL_I8 = 0
    GLOBAL_AVGPOOL_I16 = 1
    ABS_I8 = 2
    ABS_I16 = 3
    CLIP_I8 = 4
    CLIP_I16 = 5
    AVGPOOL_U8 = 6
    AVGPOOL_I8 = 7
    AVGPOOL_I16 = 8
    MAXPOOL_U8 = 9
    MAXPOOL_I8 = 10
    MAXPOOL_I16 = 11
    CAST_I16_I8 = 12
    CAST_I16_I32 = 13
    CAST_I32_I16 = 14
    CAST_U8_I16 = 15
    CAST_U8_I8 = 16
    CAST_U8_I32 = 17
    CAST_I8_I16 = 18
    CAST_I8_I32 = 19
    DEPTHWISE_CONV_I8 = 20
    LEAKYRELU_I8 = 21
    LEAKYRELU_I16 = 22
    RELU_I8 = 23
    RELU_I16 = 24
    PRELU_I8 = 25
    PRELU_I16 = 26
    PADCONST_U8 = 27
    PADCONST_I8 = 28
    PADCONST_I16 = 29
    MUL_SCALAR_I8 = 30
    MUL_SCALAR_I16 = 31
    MUL_SCALAR_U8 = 32
    MUL_SCALAR_U16 = 33
    MUL_BROADCAST_MAP_I8 = 34
    MUL_BROADCAST_MAP_I16 = 35
    MUL_BROADCAST_ROW_I8 = 36
    MUL_BROADCAST_ROW_I16 = 37
    ADD_BROADCAST_MAP_U8 = 38
    ADD_BROADCAST_MAP_I8 = 39
    ADD_BROADCAST_MAP_I16 = 40
    ADD_BROADCAST_ROW_I8 = 41
    ADD_BROADCAST_ROW_I16 = 42
    PREFETCH = 43
    LAYER_UNKNOWN = 44


def sublayer_bytes(l):
    if l in {layer_type.GLOBAL_AVGPOOL_I8 ,
             layer_type.ABS_I8 ,
             layer_type.CLIP_I8 ,
             layer_type.DEPTHWISE_CONV_I8,
             layer_type.AVGPOOL_U8 ,
             layer_type.AVGPOOL_I8 ,
             layer_type.MAXPOOL_U8 ,
             layer_type.MAXPOOL_I8 ,
             layer_type.CAST_I16_I8 ,
             layer_type.CAST_U8_I8 ,
             layer_type.CAST_U8_I32 ,
             layer_type.CAST_I8_I32 ,
             layer_type.LEAKYRELU_I8 ,
             layer_type.RELU_I8 ,
             layer_type.PRELU_I8 ,
             layer_type.PADCONST_U8 ,
             layer_type.PADCONST_I8 ,
             layer_type.MUL_SCALAR_I8 ,
             layer_type.MUL_SCALAR_U8 ,
             layer_type.MUL_BROADCAST_MAP_I8 ,
             layer_type.MUL_BROADCAST_ROW_I8,
             layer_type.ADD_BROADCAST_MAP_I8,
             layer_type.ADD_BROADCAST_ROW_I8,
             layer_type.PREFETCH,
             }:
        return 1

    if l in {layer_type.GLOBAL_AVGPOOL_I16 ,
             layer_type.ABS_I16 ,
             layer_type.CLIP_I16 ,
             layer_type.AVGPOOL_I16 ,
             layer_type.MAXPOOL_I16 ,
             layer_type.CAST_U8_I16 ,
             layer_type.CAST_I8_I16 ,
             layer_type.LEAKYRELU_I16 ,
             layer_type.RELU_I16 ,
             layer_type.PRELU_I16 ,
             layer_type.PADCONST_I16 ,
             layer_type.MUL_SCALAR_I16 ,
             layer_type.MUL_SCALAR_U16 ,
             layer_type.MUL_BROADCAST_MAP_I16 ,
             layer_type.MUL_BROADCAST_ROW_I16,
             layer_type.ADD_BROADCAST_MAP_I16,
             layer_type.ADD_BROADCAST_ROW_I16}:
        return 2;
    if l in {layer_type.CAST_I16_I32,
             layer_type.CAST_I32_I16}:
        return 4
    assert False,"Unknown size of layertype {}".format(l.name)


def enum_to_union_name(e):
    union_names = [(subgraph_type.CONV, "conv"),
                   (subgraph_type.GEMM, "gemm"),
                   (subgraph_type.SUM, "sum"),
                   (subgraph_type.IDENTITY, "identity"),
                   (subgraph_type.LRN, "lrn"),
                   (subgraph_type.TRANSPOSE, "transpose"),
                   (subgraph_type.ACTIVATION, "activation"),
                   (subgraph_type.RESIZE, "resize"),
                   (subgraph_type.REORG, "reorg"),
                   (subgraph_type.ARGMAX, "argmax"),
                   (subgraph_type.REDUCEMEAN, "reduce"),
                   (subgraph_type.TILE, "tile"),
                   (subgraph_type.MAX, "max"),
                   (subgraph_type.MIN, "min"),
                   (subgraph_type.UNKNOWN, "unknown"),
                   (layer_type.GLOBAL_AVGPOOL_I8, ""),
                   (layer_type.GLOBAL_AVGPOOL_I16, ""),
                   (layer_type.ABS_I8, ""),
                   (layer_type.ABS_I16, ""),
                   (layer_type.CLIP_I8, "clip"),
                   (layer_type.CLIP_I16, "clip"),
                   (layer_type.DEPTHWISE_CONV_I8, "depthwise"),
                   (layer_type.AVGPOOL_I8, ""),
                   (layer_type.AVGPOOL_I16, ""),
                   (layer_type.MAXPOOL_U8, ""),
                   (layer_type.MAXPOOL_I8, ""),
                   (layer_type.MAXPOOL_I16, ""),
                   (layer_type.CAST_I16_I8, "cast"),
                   (layer_type.CAST_I16_I32, "cast"),
                   (layer_type.CAST_I32_I16, "cast"),
                   (layer_type.CAST_U8_I16, "cast"),
                   (layer_type.CAST_U8_I8, "cast"),
                   (layer_type.CAST_I8_I16, "cast"),
                   (layer_type.CAST_I8_I32, "cast"),
                   (layer_type.CAST_U8_I32, "cast"),
                   (layer_type.LEAKYRELU_I8, "leakyrelu"),
                   (layer_type.LEAKYRELU_I16, "leakyrelu"),
                   (layer_type.RELU_I8, ""),
                   (layer_type.RELU_I16, ""),
                   (layer_type.PRELU_I8, "prelu"),
                   (layer_type.PRELU_I16, "prelu"),
                   (layer_type.PADCONST_U8, "pad_const"),
                   (layer_type.PADCONST_I8, "pad_const"),
                   (layer_type.PADCONST_I16, "pad_const"),
                   (layer_type.MUL_SCALAR_I8, "mul_scalar"),
                   (layer_type.MUL_SCALAR_I16, "mul_scalar"),
                   (layer_type.MUL_SCALAR_U8, "mul_scalar"),
                   (layer_type.MUL_SCALAR_U16, "mul_scalar"),
                   (layer_type.MUL_BROADCAST_MAP_I8, "mul_broadcast_map"),
                   (layer_type.MUL_BROADCAST_MAP_I16, "mul_broadcast_map"),
                   (layer_type.ADD_BROADCAST_MAP_I8, "add_broadcast_map"),
                   (layer_type.ADD_BROADCAST_MAP_I16, "add_broadcast_map"),
                   (layer_type.MUL_BROADCAST_ROW_I8, "mul_broadcast_row"),
                   (layer_type.MUL_BROADCAST_ROW_I16, "mul_broadcast_row"),
                   (layer_type.ADD_BROADCAST_ROW_I8, "add_broadcast_row"),
                   (layer_type.ADD_BROADCAST_ROW_I16, "add_broadcast_row"),
                   (layer_type.PREFETCH, "prefetch"),
                   (layer_type.LAYER_UNKNOWN, "")]
    for t, n in union_names:
        if type(t) == type(e) and t == e:
            return n
    return ""


def requires16(t, output_bytes):
    return {
        subgraph_type.CONV: 0,
        subgraph_type.GEMM: 1,
        subgraph_type.SUM: 0,
        subgraph_type.IDENTITY: 0,
        subgraph_type.LRN: 0,
        subgraph_type.TRANSPOSE: 0,
        subgraph_type.ACTIVATION: 1,
        subgraph_type.RESIZE: 0,
        subgraph_type.REORG: 0,
        subgraph_type.ARGMAX: 0,
        subgraph_type.REDUCEMEAN: 0,
        subgraph_type.TILE: 0,
        subgraph_type.MAX: 0,
        subgraph_type.MIN: 0,
        subgraph_type.UNKNOWN: 1 if output_bytes > 1 else 0
    }[t]


def pointwise_weight_index(group,
                           packed_weight_stride,
                           input_maps_per_group,
                           intra_group_input_map,
                           filter_rows,
                           filter_row,
                           filter_columns,
                           filter_column):
    return ((((((group*input_maps_per_group)+intra_group_input_map)*filter_rows)+filter_row)*filter_columns)+filter_column)*packed_weight_stride


def scalar_weight_index(group,
                        output_maps_per_group,
                        intra_group_output_map,
                        input_maps_per_group,
                        intra_group_input_map,
                        filter_rows,
                        filter_row,
                        filter_columns,
                        filter_column):
    return (((((((group*output_maps_per_group)+intra_group_output_map)*input_maps_per_group)+intra_group_input_map)*filter_rows)+filter_row)*filter_columns)+filter_column


DEPTHWISE_BIAS_SCALE_SIZE = 5
DEPTHWISE_WEIGHT_STRIDE = 3

def depthwise_weight_index(group,
                           output_maps_per_group,
                           intra_group_output_map,
                           input_maps_per_group,
                           intra_group_input_map,
                           filter_rows,
                           filter_row,
                           filter_column_passes,
                           filter_column_pass):
    idx = DEPTHWISE_BIAS_SCALE_SIZE
    idx += (((group * output_maps_per_group) + intra_group_output_map) * DEPTHWISE_BIAS_SCALE_SIZE)
    idx += (((((((((group*output_maps_per_group)+intra_group_output_map)*input_maps_per_group)+intra_group_input_map)*filter_column_passes)+filter_column_pass)*filter_rows)+filter_row)*DEPTHWISE_WEIGHT_STRIDE)
    return idx


def pack_weights(weight, packed_weight, filter_columns, filter_rows, input_maps, output_maps, groups, packed_weight_stride):
    input_maps_per_group = input_maps // groups
    output_maps_per_group = output_maps // groups

    for group in range(groups):
        for intra_group_input_map in range(input_maps_per_group):
            for filter_row in range(filter_rows):
                for filter_column in range(filter_columns):
                    for intra_group_output_map in range(output_maps_per_group):
                        src = scalar_weight_index(group,
                                                  output_maps_per_group,
                                                  intra_group_output_map,
                                                  input_maps_per_group,
                                                  intra_group_input_map,
                                                  filter_rows,
                                                  filter_row,
                                                  filter_columns,
                                                  filter_column)
                        dst = pointwise_weight_index(group,
                                                     packed_weight_stride,
                                                     input_maps_per_group,
                                                     intra_group_input_map,
                                                     filter_rows,
                                                     filter_row,
                                                     filter_columns,
                                                     filter_column)

                        dst += intra_group_output_map
                        packed_weight[dst] = weight[src]


def depthwise_pack_weights(weight, packed_weight,
                           scale_minus1,
                           bias_upper_activation, bias_upper,
                           bias_lower_activation, bias_lower,
                           filter_columns, filter_rows,
                           input_maps, output_maps,
                           groups):
    input_maps_per_group = input_maps//groups
    output_maps_per_group = output_maps//groups

    filter_column_passes = (filter_columns+2)//3

    for group in range(groups):
        for intra_group_output_map in range(output_maps_per_group):
            output_map = (group*output_maps_per_group)+intra_group_output_map

            # Pack scale and biases
            dst = depthwise_weight_index(group,
                                         output_maps_per_group,
                                         intra_group_output_map,
                                         input_maps_per_group,
                                         0,
                                         filter_rows,
                                         0,
                                         filter_column_passes,
                                         0) - 5
            packed_weight[dst] = scale_minus1[output_map]
            dst += 1
            packed_weight[dst] = bias_upper_activation
            dst += 1
            packed_weight[dst] = bias_upper[output_map]
            dst += 1
            packed_weight[dst] = bias_lower_activation
            dst += 1
            packed_weight[dst] = bias_lower[output_map]

            for intra_group_input_map in range(input_maps_per_group):
                for filter_column_pass in range(filter_column_passes):
                    for filter_row in range(filter_rows):
                        for intra_pass_filter_column in range(3):
                            filter_column = (filter_column_pass*3)+intra_pass_filter_column
                            this_weight = 0
                            if (filter_column < filter_columns):
                                src = scalar_weight_index(group,
                                                          output_maps_per_group,
                                                          intra_group_output_map,
                                                          input_maps_per_group,
                                                          intra_group_input_map,
                                                          filter_rows,
                                                          filter_row,
                                                          filter_columns,
                                                          filter_column)
                                this_weight = weight[src]
                            dst = depthwise_weight_index(group,
                                                         output_maps_per_group,
                                                         intra_group_output_map,
                                                         input_maps_per_group,
                                                         intra_group_input_map,
                                                         filter_rows, filter_row,
                                                         filter_column_passes,
                                                         filter_column_pass)
                            dst += intra_pass_filter_column
                            packed_weight[dst] = this_weight


def round_up_pwr_2(v):
    v -= 1
    v |= (v >> 1)
    v |= (v >> 2)
    v |= (v >> 4)
    v |= (v >> 8)
    v |= (v >> 16)
    v += 1
    return v


def round_down_pwr_2(v):
    v |= (v >> 1)
    v |= (v >> 2)
    v |= (v >> 4)
    v |= (v >> 8)
    v |= (v >> 16)
    return v - (v >> 1)


def set_sublayer_attributes(node, sl_array):
    shape = [0,0]
    shape_0 = [0,0]
    shape_last = [0,0]
    shape_full = [0,0]

    max_rows = 0
    scratchpad_per_map = 0
    max_columns = 0
    max_maps = 0
    maps = 0
    rows = 0
    columns = 0
    stride = [1,1]

    # shape states how many additional inputs are required
    # pad at end does not change this
    # pad before activation doesn't
    for sl in sl_array:
        shape[0] += (sl.kernel_shape[0] - 1)*stride[0]
        shape_0[0] += (sl.kernel_shape[0] - 1 - sl.pads[1])*stride[0]
        shape_last[0] += (sl.kernel_shape[0] - 1 - sl.pads[4])*stride[0]
        shape_full[0] += (sl.kernel_shape[0] - 1 -
                       (sl.pads[1] + sl.pads[4]))*stride[0]
        stride[0] *= sl.strides[0]


        shape[1] += (sl.kernel_shape[1] - 1)*stride[1]
        shape_0[1] += (sl.kernel_shape[1] - 1 - sl.pads[2])*stride[1]
        shape_last[1] += (sl.kernel_shape[1] - 1 - sl.pads[5])*stride[1]
        shape_full[1] += (sl.kernel_shape[1] - 1 -
                       (sl.pads[2] + sl.pads[5]))*stride[1]
        stride[1] *= sl.strides[1]

        columns += ((sl.pads[2] + sl.pads[5] - (sl.kernel_shape[1]-1))
                    // sl.strides[1])
        if (columns > max_columns):
            max_columns = columns
        rows += ((sl.pads[1] + sl.pads[4] - (sl.kernel_shape[0]-1)) //
                 sl.strides[0])
        if (rows > max_rows):
            max_rows = rows
        maps += sl.pads[0] + sl.pads[3]
        if (maps > max_maps):
            max_maps = maps

        if sl.type == layer_type.DEPTHWISE_CONV_I8:
            scratchpad_per_map += ((sl.kernel_shape[0]+2)//3*3*sl.kernel_shape[1]+5)

    node.sublayer_shape = shape
    node.sublayer_shape_full = shape_full
    node.sublayer_shape_last = shape_last
    node.sublayer_shape_0 = shape_0
    node.sublayer_stride = stride
    node.sublayer_columns = max_columns
    node.sublayer_rows = max_rows
    node.sublayer_maps = max_maps
    node.sublayer_scratchpad_per_map = scratchpad_per_map


def elements(m, n, shape, dilations, strides, use_strided_maps):
    if use_strided_maps:
        m0 = (m + (strides[0]-1)) // strides[0]
        n0 = (n + (strides[1]-1)) // strides[1]
        start_output_element = ((shape[0] // 2 // strides[0]) * dilations[0] * n0) + (shape[1] // 2 // strides[1] * dilations[1])
        end_output_element = (m0*n0) - ((((shape[0]-1) // 2 // strides[0]) * dilations[0] * n0) + ((shape[1]-1) // 2 // strides[1]) * dilations[1])
        return end_output_element-start_output_element
    else:
        start_output_element = (
            (shape[0] // 2) * dilations[0] * n) + (shape[1] // 2 * dilations[1])
        end_output_element = (m * n) - ((((shape[0] - 1) // 2) * dilations[0] * n) +
                                        ((shape[1] - 1) // 2) * dilations[1])
        return end_output_element - start_output_element


def pad_weights16(src, kernels, channels, group, kernel_height, kernel_width,
                  padded_kernels, padded_channels):
    assert padded_channels == channels
    if padded_kernels == kernels:
        return src
    kh = kernel_height
    kw = kernel_width

    padded_kernels_per_group = padded_kernels // group
    kernels_per_group = kernels // group
    num_padded_weights = (kw * kh * channels ) * padded_kernels
    padded = [0] * num_padded_weights

    it = itertools.product(range(group), range(kernels_per_group),
                           range(channels), range(kernel_height),
                           range(kernel_width))
    for g, k, c, y, x in it:
        padded_group_offset = (g * padded_kernels_per_group * padded_channels *
                               kh * kw)
        group_offset = g * kernels_per_group * channels * kh * kw
        w_index = (padded_group_offset + k * padded_channels * kh * kw +
                   c * kh * kw + y * kw + x)
        r_index = (group_offset + k * channels * kh * kw +
                   c * kh * kw + y * kw + x)
        padded[w_index] = src[r_index]

    return padded


def interleave_weights(src, channels, kernels, kernel_height, kernel_width):
    nweights = channels * kernel_height
    nweights = nweights * kernel_width
    nweights = nweights * kernels
    dst = [0] * nweights
    for k in range(kernels):
        for c in range(channels):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    w = (((((k * channels) + c) * kernel_height) + kh) *
                         kernel_width) + kw
                    iw = (((((c * kernel_height) + kh) * kernel_width) + kw) *
                          kernels) + k
                    dst[iw] = src[w]
    return np.asarray(dst, dtype=np.int8)


def aligned_size(size, vector_lanes):
    aligned = vector_lanes * 4
    if (size % aligned):
        return (size // aligned + 1) * aligned
    return size


def float_to_fixed(flt_in, out_type, MAX_BITS=None, FRAC_BITS=None, ROUNDING=True):
    rounding = 0.5 if ROUNDING else 0
    max_bits = 8
    frac_bits = U8
    if out_type == calc_type.INT8:
        max_bits = 7
        frac_bits = Q8
    if out_type == calc_type.INT16:
        max_bits = 15
        frac_bits = Q16
    if out_type == calc_type.INT32:
        max_bits = 31
        frac_bits = Q32
    if FRAC_BITS is not None:
        frac_bits = FRAC_BITS
    if MAX_BITS is not None:
        max_bits = MAX_BITS

    max_min = ((1 << max_bits) - 1, (1 << max_bits) * -1)
    max_min = ((1 << max_bits) - 1, -1* ((1 << max_bits)-1))
    if out_type == calc_type.UINT8:
        max_min = (max_min[0], 0)
    if flt_in >= 0:
        value = flt_in * (1 << frac_bits) + rounding
    else:
        value = flt_in * (1 << frac_bits) - rounding

    value = max(max_min[1], min(max_min[0], value))

    return int(value)


def float_to_fixed_np(flt_in, out_type, MAX_BITS=None, FRAC_BITS=None, ROUNDING=True):
    rounding = 0.5 if ROUNDING else 0
    max_bits = 8
    frac_bits = U8
    np_type = np.uint8
    if out_type == calc_type.UINT8:
        np_type = np.uint8
        max_bits = 8
        frac_bits = U8
    if out_type == calc_type.INT8:
        np_type = np.int8
        max_bits = 7
        frac_bits = Q8
    if out_type == calc_type.INT16:
        np_type = np.int16
        max_bits = 15
        frac_bits = Q16
    if out_type == calc_type.INT32:
        np_type = np.int32
        max_bits = 31
        frac_bits = Q32
    if FRAC_BITS is not None:
        frac_bits = FRAC_BITS
    if MAX_BITS is not None:
        max_bits = MAX_BITS
    max_min = ((1 << max_bits) - 1, (1 << max_bits) * -1)
    max_min = ((1 << max_bits) - 1, -1* ((1 << max_bits)-1))
    if out_type == calc_type.UINT8:
        max_min = (max_min[0], 0)
    tmp = flt_in * (1 << frac_bits)
    tmp = np.where(flt_in >= 0, tmp+rounding, tmp-rounding)
    tmp = np.clip(tmp, max_min[1], max_min[0])

    return tmp.astype(np_type)


def float_array_to_weights(flt_array, typ):
    fmt = 'B'
    if typ == calc_type.INT8:
        fmt = 'b'
    if typ == calc_type.INT16:
        fmt = 'h'
    if typ == calc_type.INT32:
        fmt = 'i'
    return b"".join([struct.pack(fmt, float_to_fixed(f, typ)) for f in flt_array])


def base64_to_float_array(b64):
    byte_stream = base64.b64decode(b64)
    flt_array = struct.unpack("f"*(len(byte_stream)//4), byte_stream)
    return flt_array


def base64_to_float_nparray(b64):
    byte_stream = base64.b64decode(b64)
    flt_array = np.frombuffer(byte_stream, dtype=np.float32)
    return flt_array


class weight_array(bytearray):
    def __iadd__(self, other):
        super().__iadd__(other)
        while(len(self) % 4):
            self.append(0)
        return self


def set_conv_vbx(node, sp_size, min_rows, inc_rows):
    offset = min_rows - inc_rows
    conv = node.conv

    m = conv.m
    n = conv.n
    kernel_height = conv.kernel_shape[0]
    kernel_width = conv.kernel_shape[1]
    n_extra = node.sublayer_columns
    m_extra = node.sublayer_rows
    conv_input_bytes = 1
    if node.input_data_type == calc_type.INT16:
        conv_input_bytes = 2
    conv_output_bytes = 1
    if node.output_data_type == calc_type.INT16:
        conv_output_bytes = 2

    conv.core_m = conv.m
    conv.core_split = 1
    conv.core_maps = ceil(conv.kernels / CORES)

    # calculte the max maps
    max_maps = -1
    coeff_width = ceil(kernel_width*COEFFICIENT_BITS*4 /8)
    m0, n0 = (m-(kernel_height-1), n-(kernel_width-1))
    if conv_input_bytes == 1:
        max_maps = (sp_size - (2*m*n + 2*m0*n0))/(2*2*m*n)
    elif conv_input_bytes == 2:
        max_maps = (sp_size - (4*m*n + 2*m0*n0))/(2*2*m*n)
    elif conv_input_bytes == 4:
        max_maps = (sp_size - (8*m*n+4*m0*n0)) / (4*m0*n0)

    if conv.group > 1 and max_maps > (conv.kernels/conv.group):
        # adjust max_maps to cap at 1 filter group at a time
        if conv.group == conv.kernels and conv.group == conv.channels:
            pass  # depthwise
        else:
            max_maps = conv.kernels/conv.group

    if conv.m == 1 and conv.n == 1:
        conv.imaps = int((sp_size - (2*(conv.channels+conv.kernels))) / (2*2*conv.channels))
        if conv.imaps == 0: # ignore conv_1x1_specialcase, as doesn't fit
            conv.imaps = -1
        else:
            while (conv.kernels % conv.imaps) > 0:
                conv.imaps -= 1

    if max_maps > 1:
        conv.maps = conv.core_maps
        while conv.maps > max_maps:
            conv.maps /= 2
        conv.maps = int(conv.maps)
        if conv.group > 1:  # TODO handle multi-core
            while (conv.kernels % conv.maps) != 0:
                conv.maps -= 1
            assert(conv.maps >= 1)
    else:
        conv.maps = 1

    # calculate max rows
    def max_conv_rows(kernels, n):
        n0 = n - (kernel_width-1)
        m = -1
        if (conv_input_bytes == 1):
            m = sp_size // (2*n + 2*n0 + 2*kernels*2*n)
        elif (conv_input_bytes == 2):
            m = sp_size // (4*n + 2*n0 + 2*kernels*2*n)
        elif (conv_input_bytes == 4):
            m = sp_size // (8*n+4*n0+kernels*4*n0)
        return m


    conv.rows = max_conv_rows(conv.maps, n+n_extra)
    if conv.m == 1 and conv.n == 1:
        conv.rows = 1

    if conv.rows >= conv.m:
        conv.rows = conv.m
    else:
        while (conv.rows - offset) % inc_rows:
            conv.rows -= 1
        assert conv.rows >= min_rows

    node.scratchpad_bytes = conv.maps*(conv.rows+m_extra)*(conv.n+n_extra)*2
    return True




def conv_populate_attributes(node, json_node, conv_cvi, sp_size, vector_lanes, filter_copies, parallel_output_maps):
    conv = node.conv
    conv.use_cvi = conv_cvi
    conv.rows = conv.m
    conv.cols = conv.n
    # TODO: These are not used as far as i can tell, they
    # should be removed from the struct
    conv.padded_channels = 0
    conv.fxp_scalar = 0
    conv.max_weight = 0

    conv.use_weights32 = 0
    conv.weights32 = 0

    m = conv.m
    n = conv.n
    kernel_height = conv.kernel_shape[0]
    kernel_width = conv.kernel_shape[1]
    n_extra = node.sublayer_columns
    m_extra = node.sublayer_rows
    conv_input_bytes = 1
    if node.input_data_type == calc_type.INT16 or conv.use_cvi and INPUT_SIZE_BYTES == 2:
        conv_input_bytes = 2

    conv_output_bytes = 1
    if node.output_data_type == calc_type.INT16 or conv.use_cvi and OUTPUT_SIZE_BYTES == 2:
        conv_output_bytes = 2

    layer_output_bytes = 1
    if node.output_data_type == calc_type.INT32:
        layer_output_bytes = 4
    elif node.output_data_type == calc_type.INT16 or conv_output_bytes == 2:
        layer_output_bytes = 2

    def vci_conv_rows(conv, imaps, maps, row_split=1):

        if conv.use_depthwise:
            channels = imaps
            if channels < maps:
                channels *= 2  # double-buffered
            coeff_stride = ceil(COEFFICIENT_BITS*maps/8)
            remaining = sp_size
            remaining -= aligned_size(((conv.kernel_shape[0]+2)//3*3*conv.kernel_shape[1]+5)*coeff_stride, vector_lanes)  # v_w0
            remaining -= vector_lanes*4  # Padding wavefront
        else:
            channels = imaps
            if channels < conv.channels:
                channels *= 2  # double-buffered
            coeff_stride = ceil(COEFFICIENT_BITS*maps/8)

            remaining = sp_size
            remaining -= vector_lanes*4  # v_zero_wavefront
            remaining -= aligned_size(coeff_stride, vector_lanes)  # v_bias
            remaining -= vector_lanes*4  # Padding wavefront
            remaining -= aligned_size(coeff_stride, vector_lanes)  # v_bias_lower
            remaining -= vector_lanes*4  # Padding wavefront
            remaining -= aligned_size(coeff_stride, vector_lanes)  # v_scale
            remaining -= vector_lanes*4  # Padding wavefront
            # v_in0 (depends on conv_rows)
            remaining -= vector_lanes*4  # Padding wavefront
            remaining -= aligned_size(imaps*conv.kernel_shape[0] *
                                      conv.kernel_shape[1]*coeff_stride, vector_lanes)  # v_w0
            remaining -= vector_lanes*4  # Padding wavefront

        if not conv.use_depthwise and imaps < conv.channels:
            # double-buffered
            # v_in1 (depends on conv_rows)
            remaining -= vector_lanes*4  # Padding wavefront
            remaining -= aligned_size(imaps*conv.kernel_shape[0] *
                                      conv.kernel_shape[1]*coeff_stride, vector_lanes)  # v_w1
            remaining -= vector_lanes*4  # Padding wavefront
        elif conv.use_depthwise and imaps < maps:
            # double-buffered
            remaining -= vector_lanes*4  # Padding wavefront
            remaining -= aligned_size(imaps*((conv.kernel_shape[0]+2)//3*3*conv.kernel_shape[1]+5)*coeff_stride, vector_lanes)  # v_w1
            remaining -= vector_lanes*4  # Padding wavefront

        strides_n = 1
        if conv.use_strided:
            strides_n = conv.strides[1]

        conv_n = conv.n
        if row_split > 1:
            conv_n = (conv.n+kernel_width-1+row_split-1)//row_split
        size_per_irow = aligned_size(conv_input_bytes*channels*conv_n, vector_lanes)
        size_per_orow = aligned_size(layer_output_bytes*maps*((conv_n+strides_n-1)//strides_n+n_extra), vector_lanes)

        rows = (remaining - m_extra * size_per_orow) // (size_per_irow + size_per_orow)

        if conv.use_strided and rows % conv.strides[0]:
            rows -= 1
        return rows


    def vci_conv_rows_db(conv, imaps, maps, irows):
        coeff_stride = ceil(COEFFICIENT_BITS*maps/8)

        remaining = sp_size
        remaining -= vector_lanes*4  # v_zero_wavefront
        remaining -= aligned_size(coeff_stride, vector_lanes)  # v_bias
        remaining -= vector_lanes*4  # Padding wavefront
        remaining -= aligned_size(coeff_stride, vector_lanes)  # v_bias_lower
        remaining -= vector_lanes*4  # Padding wavefront
        remaining -= aligned_size(coeff_stride, vector_lanes)  # v_scale
        remaining -= vector_lanes*4  # Padding wavefront
        # v_in0 (depends on conv_rows)

        remaining -= aligned_size(imaps*conv.kernel_shape[0] *
                                  conv.kernel_shape[1]*coeff_stride, vector_lanes)  # v_w0
        remaining -= vector_lanes*4  # Padding wavefront

        strides_n = 1
        if conv.use_strided:
            strides_n = conv.strides[1]

        size_per_irow = aligned_size(conv_input_bytes*imaps*n, vector_lanes)
        remaining -= vector_lanes*4  # Padding wavefront
        size_per_irow += aligned_size(conv_input_bytes*imaps*n, vector_lanes)

        remaining -= irows * size_per_irow
        size_per_orow = aligned_size(layer_output_bytes*maps*((n+strides_n-1)//strides_n+n_extra), vector_lanes)

        rows = (remaining // size_per_orow) - m_extra

        if conv.use_strided and rows % conv.strides[0]:
            rows -= 1
        return rows


    def invalid_number_of_rows(rows, conv_rows, post_rows):
        return (not (conv_rows >= rows) or not (post_rows >= rows))


    def vci_post_rows(node, maps, dma_split=1, row_split=1):
        conv = node.conv
        strides_n = 1
        if conv.use_strided:
            strides_n = conv.strides[1]

        rows =  (sp_size - ((maps+(dma_split-1))//dma_split*node.sublayer_scratchpad_per_map))//((maps+((maps+(dma_split-1))//dma_split))*layer_output_bytes*(((n+row_split-1)//row_split+strides_n-1)//strides_n+n_extra)) - m_extra

        return rows




    def set_conv_cvi_depthwise(node, sp_size, min_rows, inc_rows):
        conv = node.conv
        row_offset = min_rows - inc_rows
        # determine if full maps can be done at once
        depth_split = 1
        vci_fit_full_maps = True
        if vci_conv_rows(conv, 1, depth_split) < conv.m:
            vci_fit_full_maps = False

        # initially set maps to be kernels (pow2), currently ignoring core split
        conv.core_maps = round_up_pwr_2(conv.kernels)
        conv.core_split = 1
        conv.maps = conv.core_maps
        conv.core_m = conv.m
        conv.acc_maps = 1

        # initially set imaps to maps / split
        conv.imaps = conv.maps 
        if conv.maps % depth_split == 0:
            conv.imaps = conv.maps // depth_split

        # if full maps, set rows to m, else to min_rows
        if vci_fit_full_maps:
            rows = conv.m
        else:
            rows = min_rows

        conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps)
        post_rows = vci_post_rows(node, conv.maps)

        # reduce imaps/maps until rows are valid
        while invalid_number_of_rows(rows, conv_rows, post_rows) and conv.imaps > 1:
            conv.imaps = ceil(conv.imaps/2)
            conv.maps = conv.imaps * depth_split
            conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps)
            post_rows = vci_post_rows(node, conv.maps)

        # increase rows as much as possible
        if rows != conv.m:
            while not invalid_number_of_rows(rows+inc_rows, conv_rows, post_rows):
                rows += inc_rows
                if rows > conv.m + row_offset:
                    rows = conv.m + row_offset
                    break
            while (rows - row_offset) % inc_rows:
                rows-= 1

        conv.conv_rows = 0
        conv.rows = rows
        node.scratchpad_bytes = layer_output_bytes*conv.maps*(conv.rows+m_extra)*(conv.n+n_extra)
        return True


    conv.maps = -1
    conv.imaps = -1
    conv.rows = -1
    node.scratchpad_bytes = -1
    min_rows = (1+(conv.kernel_shape[0]-1)*conv.dilations[0] +
                node.sublayer_shape[0]*conv.strides[0])
    inc_rows = conv.strides[0] * node.sublayer_stride[0]
    conv.inc_rows = conv.strides[0]
    conv.conv_rows = 0
    row_offset = (min_rows - inc_rows)

    if node.sublayer_shape_0[0] > 0:
        if row_offset <= min_rows - (node.sublayer_shape[0] - node.sublayer_shape_0[0])*conv.strides[0]:
            min_rows += (node.sublayer_shape[0] - node.sublayer_shape_0[0])*conv.strides[0]

    min_cols = (1+(conv.kernel_shape[1]-1)*conv.dilations[1] +
                node.sublayer_shape[1]*conv.strides[1])
    inc_cols = conv.strides[1] * node.sublayer_stride[1]
    col_offset = (min_cols - inc_cols)

    is_set = False
    if not conv.use_cvi:
        is_set = set_conv_vbx(node, sp_size, min_rows, inc_rows)
    else:
        if conv.use_depthwise:
            is_set = set_conv_cvi_depthwise(node, sp_size, min_rows, inc_rows)
        else:
            max_elements = filter_copies*ACCUMULATORS
            elems_to_process = elements(m, n, conv.kernel_shape, conv.dilations, conv.strides, conv.use_strided)
            vci_fit_full_maps = max_elements > elems_to_process
            conv.core_m = conv.m
            conv.core_split = 1
            conv.maps = round_up_pwr_2(conv.kernels)
            if (conv.group > 1 and conv.maps > conv.kernels/conv.group):
                conv.maps = conv.kernels//conv.group

            if len(node.subnode_array):
                dma_split = 4
                if dma_split > conv.maps:
                    dma_split = conv.maps
            else:
                dma_split = 1

            if vci_fit_full_maps:
                rows = conv.m

                # maximize number of output maps at once
                conv.acc_maps = ACCUMULATORS // ceil(elems_to_process/filter_copies)
                max_output_maps = parallel_output_maps * conv.acc_maps
                if conv.maps > max_output_maps:
                    conv.maps = max_output_maps

                conv.imaps = conv.channels // conv.group
                # split channels to force double buffering
                if (conv.channels // conv.group) % 8 == 0:
                    if (conv.channels // conv.group) // 8 < 32:
                        conv.imaps = 32
                    else:
                        conv.imaps = (conv.channels // conv.group) // 8

                conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps)
                post_rows = vci_post_rows(node, conv.maps, dma_split)

                # reduce imaps until full maps fit
                if invalid_number_of_rows(rows, conv_rows, post_rows) and conv.imaps > 1:
                    while invalid_number_of_rows(rows, conv_rows, post_rows) and conv.imaps > 1:
                        conv.imaps = ceil(conv.imaps//2)
                        conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps)
                        post_rows = vci_post_rows(node, conv.maps, dma_split)

                if not invalid_number_of_rows(rows, conv_rows, post_rows):
                    conv.conv_rows = 0
                    conv.rows = rows
                    node.dma_split = dma_split
                    node.scratchpad_bytes = layer_output_bytes*conv.maps*(conv.rows+m_extra)*(conv.n+n_extra)
                    if conv.use_strided:
                        node.scratchpad_bytes = layer_output_bytes*conv.maps*(conv.rows+m_extra)*((conv.n+conv.strides[1]-1)//conv.strides[1]+n_extra)
                    is_set = True
            else:
                rows = max(4, min_rows)
                irows = min_rows
                while irows < 4:
                    irows += conv.inc_rows

                conv.acc_maps = 1
                max_output_maps = parallel_output_maps
                if conv.maps > max_output_maps:
                    conv.maps = max_output_maps

                # double buffer across rows, fitting all input maps
                conv.imaps = conv.channels // conv.group

                conv_rows2 = vci_conv_rows_db(conv, conv.imaps, conv.maps, irows)
                post_rows = vci_post_rows(node, conv.maps, dma_split)

                vci_fit_all_imaps = True
                if invalid_number_of_rows(rows, conv_rows2, post_rows):
                    vci_fit_all_imaps = False

                if vci_fit_all_imaps:
                    # fix full maps
                    if post_rows > 2*conv.m and conv_rows2 > 2*conv.m:
                        rows = conv.m

                    # check if possible to increase output maps in scratch:
                    while conv.maps*2 <= conv.kernels and conv.maps // dma_split >= 1: 
                        conv_rows2 = vci_conv_rows_db(conv, conv.imaps, conv.maps*2, irows)
                        if dma_split > 1:
                            post_rows = vci_post_rows(node, conv.maps*2, dma_split*2)
                        else:
                            post_rows = vci_post_rows(node, conv.maps*2, dma_split)

                        if invalid_number_of_rows(rows, conv_rows2, post_rows):
                            break
                        else:
                            conv.maps *= 2
                            if dma_split > 1:
                                dma_split *= 2

                    conv_rows2 = vci_conv_rows_db(conv, conv.imaps, conv.maps, irows)
                    post_rows = vci_post_rows(node, conv.maps, dma_split)
                    rows = min(conv_rows2, post_rows, conv.m)

                    # increase number of inputs rows, keeping full maps
                    ratio = 1.9
                    while True:
                        conv_rows2 = vci_conv_rows_db(conv, conv.imaps, conv.maps, irows+conv.inc_rows)
                        if not invalid_number_of_rows(rows, conv_rows2, post_rows) and (irows+conv.inc_rows-(conv.kernel_shape[0]-1))*ratio < min(conv_rows2, post_rows, conv.m):
                            irows += conv.inc_rows
                        else:
                            break

                    conv_rows2 = vci_conv_rows_db(conv, conv.imaps, conv.maps, irows)
                    post_rows = vci_post_rows(node, conv.maps, dma_split)
                    rows = min(conv_rows2, post_rows, conv.m)

                    # decrease rows if leaves small final row
                    if rows < conv.m:
                        split = ceil(conv.m/rows)
                        rows = (conv.m + (split-1)) // split

                    if rows < conv.m:
                        if rows > conv.core_m + row_offset:
                            rows = conv.core_m + row_offset
                        while (rows - row_offset) % inc_rows:
                            rows -= 1

                    if conv.use_strided and irows % conv.strides[0]:
                        irows -= 1

                    # minimize split once max rows determined
                    while dma_split > 4 and vci_post_rows(node, conv.maps, dma_split // 2) >= rows:
                        dma_split //= 2

                    conv.rows = rows
                    conv.conv_rows = irows
                    node.dma_split = dma_split
                    node.scratchpad_bytes = layer_output_bytes*conv.maps*(conv.rows+m_extra)*(conv.n+n_extra)
                    if conv.use_strided:
                        node.scratchpad_bytes = layer_output_bytes*conv.maps*(conv.rows+m_extra)*((conv.n+conv.strides[1]-1)//conv.strides[1]+n_extra)
                    if conv.kernel_shape == [1,1] and conv.strides == [1, 1]:
                        is_set = True
        # conv_cvi
        if not is_set:
            max_elements = filter_copies*ACCUMULATORS
            elems_to_process = elements(m, n, conv.kernel_shape, conv.dilations, conv.strides, conv.use_strided)
            if conv.use_depthwise:
                elems_to_process = 0

            vci_fit_full_maps = max_elements > elems_to_process
            if conv.use_depthwise:
                if vci_conv_rows(conv, 1, 1) < conv.m:
                    vci_fit_full_maps = False

            conv.core_maps = round_up_pwr_2(conv.kernels)
            if max_elements < elems_to_process:
                conv.acc_maps = 1
            elif conv.use_depthwise:
                conv.acc_maps = 1
            else:
                conv.acc_maps = ACCUMULATORS // ceil(elems_to_process/filter_copies)
            # divide maps if entire output map can't fit
            max_output_maps = parallel_output_maps * conv.acc_maps
            conv.core_m = conv.m
            conv.core_split = 1

            if vci_fit_full_maps:
                rows = conv.m
                conv.core_maps = ceil(conv.core_maps/CORES)
            else:
                c = CORES
                while c > 1:
                    if conv.core_maps > 2*max_output_maps:
                        conv.core_maps //= 2
                    else:
                        conv.core_split *= 2
                    c /= 2
                if conv.core_split > 1:
                    conv.core_m = ceil(m / conv.core_split)
                while conv.core_m % inc_rows:
                    conv.core_m += 1
                rows = min_rows

            # ensure only 1 filter group at a time
            conv.maps = conv.core_maps
            if not conv.use_depthwise and (conv.group > 1 and conv.maps > conv.kernels/conv.group):
                conv.maps = conv.kernels//conv.group
            if not conv.use_depthwise and conv.maps > max_output_maps:
                conv.maps = max_output_maps

            # set input maps
            if conv.use_depthwise:
                conv.imaps = conv.maps 
            else:
                conv.imaps = conv.channels // conv.group

                # empirically shown to run faster if capped as it improves double
                # buffering with large numbers of output maps
                if conv.imaps > 2*parallel_output_maps and conv.imaps > conv.kernels:
                    conv.imaps = parallel_output_maps

            if len(node.subnode_array) and not conv.use_depthwise:  
                dma_split = 4
                if dma_split > conv.maps:
                    dma_split = conv.maps
            else:
                dma_split = 1

            row_split = 1

            conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps, row_split)
            post_rows = vci_post_rows(node, conv.maps, dma_split, row_split)

            if invalid_number_of_rows(rows, conv_rows, post_rows) and conv.m == 1 and conv.n > 1:
                while invalid_number_of_rows(rows, conv_rows, post_rows) and row_split*2 < conv.n:
                    row_split *= 2
                    conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps, row_split)
                    post_rows = vci_post_rows(node, conv.maps, dma_split, row_split)

            if conv.imaps < conv.channels/conv.group:
                if invalid_number_of_rows(rows, conv_rows, post_rows) and conv.imaps > 0:
                    while invalid_number_of_rows(rows, conv_rows, post_rows) and conv.maps > 1:
                        conv.maps = ceil(conv.maps/2)
                        conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps, row_split)
                        post_rows = vci_post_rows(node, conv.maps, dma_split, row_split)

            if invalid_number_of_rows(rows, conv_rows, post_rows) and conv.imaps > 0:
                conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps, row_split)
                post_rows = vci_post_rows(node, conv.maps, dma_split, row_split)

                while conv_rows < rows and (conv.use_depthwise or conv.imaps > 2*parallel_output_maps):
                    conv.imaps = ceil(conv.imaps/2)
                    if conv.use_depthwise:
                        conv.maps = conv.imaps
                    conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps, row_split)
                    post_rows = vci_post_rows(node, conv.maps, dma_split, row_split)

                while invalid_number_of_rows(rows, conv_rows, post_rows) and conv.maps > 1:
                    conv.maps = ceil(conv.maps/2)
                    if conv.use_depthwise:
                        conv.imaps = conv.maps
                    conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps, row_split)
                    post_rows = vci_post_rows(node, conv.maps, dma_split, row_split)

            assert conv_rows >= rows
            assert post_rows >= rows
            assert conv.imaps != 0

            if not vci_fit_full_maps and rows != conv.m:
                all_imaps = (conv.imaps == conv.channels//conv.group) or conv.use_depthwise
                if not all_imaps:
                    elems_to_process = elements(rows, n, conv.kernel_shape, conv.dilations, conv.strides, conv.use_strided)
                    if not (max_elements >= elems_to_process):
                        raise NotImplementedError("Convolution CVI cannot hold minimum rows required")

                elems_to_process = elements(rows + inc_rows, n, conv.kernel_shape, conv.dilations, conv.strides, conv.use_strided)
                inc_elements = int(all_imaps or max_elements >= elems_to_process)

                while conv_rows >= (rows+inc_rows) and post_rows >= (rows+inc_rows) and inc_elements:
                    rows += inc_rows
                    elems_to_process = elements(rows + inc_rows, n, conv.kernel_shape, conv.dilations, conv.strides, conv.use_strided)
                    inc_elements = int(all_imaps or (max_elements > elems_to_process))
                    if rows > conv.core_m + row_offset:
                        break

                if rows > conv.core_m + row_offset:
                    rows = conv.core_m + row_offset
                while (rows - row_offset) % inc_rows:
                    rows -= 1

                
            node.dma_split = dma_split
            conv.rows = rows
            conv.cols = (conv.n + row_split-1) // row_split
            if conv.cols < conv.n:
                if conv.cols > conv.n + col_offset:
                    conv.cols = conv.n + col_offset
                while (conv.cols - col_offset) % inc_cols:
                    conv.cols -= 1

            conv.conv_rows = 0
            node.scratchpad_bytes = layer_output_bytes*conv.maps*(conv.rows+m_extra)*(conv.cols+n_extra)
            if conv.use_strided:
                node.scratchpad_bytes = layer_output_bytes*conv.maps*(conv.rows+m_extra)*((conv.cols+conv.strides[1]-1)//conv.strides[1]+n_extra)

    if(conv.group > 2 and conv.group < conv.kernels):
        sys.stderr.write("WARNING: quick fix. group/maps overlap\n")
        conv.maps = 1
    conv.padded_kernels = ceil(conv.kernels / conv.maps) * conv.maps

    tmp = ceil(conv.padded_kernels / (CORES / conv.core_split))
    conv.core_maps = ceil(tmp / conv.maps )* conv.maps
    assert conv.maps != 0
    assert conv.imaps != 0
    assert conv.rows != 0
    assert node.scratchpad_bytes != 0
    assert isinstance(conv.maps, int)
    assert isinstance(conv.imaps, int)
    assert isinstance(conv.rows, int)


def ignore_cvi(json_node):
    if json_node['op_type'] == 'Conv':
        if json_node['use_cvi'] == 0:
            return 1
    return 0


def max_ident_maps(m, n, sp_size, bytes, dma_split=1):
    return sp_size // ceil(m*n*(1.+1./dma_split)*bytes)


def max_gemm_columns(gemm, sp_size, bytes):
    max_input_size = gemm.input_size
    r = (sp_size - (max_input_size * 3 * bytes)) // (2*bytes)
    while r < 1:
        max_input_size = (max_input_size + 1) // 2
        r = (sp_size - (max_input_size*3*bytes)) // (2*bytes)
    return max_input_size


def conv_pack_parameters(weights, conv_cvi, padded_kernels, maps, input_unsigned, output_unsigned, conv, json_node, corrections, all_corrections, all_corrections_names):
    # COEFFICIENTS
    num_padded_weights = (
        conv.kernel_shape[0]*conv.kernel_shape[1] *
        (conv.channels//conv.group) * padded_kernels)

    num_weights = (conv.kernel_shape[0]*conv.kernel_shape[1] *
                   (conv.channels/conv.group) * conv.kernels)
    weights_f32 = base64_to_float_nparray(json_node['weights'])
    layer_max_weight = np.max(np.abs(weights_f32))
    assert num_weights == len(weights_f32)

    weights_per_kernel = ((conv.channels//conv.group) *
                          conv.kernel_shape[0]*conv.kernel_shape[1])

    num_biases = conv.kernels
    biases_f32 = base64_to_float_nparray(json_node['biases'])
    assert num_biases == len(biases_f32)

    corr = None
    if not(corrections is None):
        desc = json_node['name']
        if desc in corrections:
            corr = np.asarray(corrections[desc])
            biases_f32 = biases_f32 + corr

    if INPUT_SIZE_BYTES == 1:
        if input_unsigned:
            conv.bias_scalar = (1 << U8) - 1
            conv.bias_lower_scalar = 1
        else:
            conv.bias_scalar = -((1 << Q8) - 1)
            conv.bias_lower_scalar = -1
    else:
        conv.bias_scalar = -(1 << Q16)
        conv.bias_lower_scalar = 0

    QWEIGHT = 7
    QSCALE = 7
    QEXTRA = QWEIGHT + QSCALE - ACCUMULATOR_FXP_BITS - SCALE_FXP_BITS

    if input_unsigned and not output_unsigned:
        QEXTRA += 1
    if output_unsigned and not input_unsigned:
        QEXTRA -= 1

    # skipped = 0
    min_weight = 1.0 / (1 << 16)

    scale_f32 = [0. for k in range(conv.kernels)]
    scale_i16 = [0 for k in range(conv.kernels)]
    shift_i16 = [QEXTRA for k in range(conv.kernels)]
    weights_i16 = float_to_fixed_np(weights_f32, calc_type.INT16)
    weights_i32 = float_to_fixed_np(weights_f32, calc_type.INT32)
    biases_i16 = [float_to_fixed(f, calc_type.INT16) for f in biases_f32]
    biases_lower_i16 = [0 for _ in biases_i16]

    no_correction = corr is None and all_corrections is None
    no_potential_correction = all_corrections and not (json_node['name'] in all_corrections_names)
    use_bias_lower = False 
    USE_ROUNDING = False
    if no_correction or no_potential_correction:
        USE_ROUNDING = True
        use_bias_lower = True
        for k in range(conv.kernels):
            k_weights = weights_f32[k*weights_per_kernel:(k+1)*weights_per_kernel]
            max_weight = np.max(np.abs(k_weights))

            if abs(biases_f32[k]) > max_weight:
                use_bias_lower = False

    if VERBOSE:
        weights_big = False
        q_max = 0
        weights_small = False
        q_min = 8

    for k in range(conv.kernels):
        k_weights = weights_f32[k*weights_per_kernel:(k+1)*weights_per_kernel]
        max_weight = np.max(np.abs(k_weights))
        if abs(biases_f32[k]) > max_weight:
            max_weight = abs(biases_f32[k])


        qweight = QWEIGHT
        qin = 7
        qout = 7
        if input_unsigned:
            qin = 8
        if output_unsigned:
            qout = 8
        mweight = max_weight
        qdiff = ACCUMULATOR_FXP_BITS + SCALE_FXP_BITS + qout - qin

        if mweight == 0:
            qscale = 0
        else:
            qscale = qdiff + log2(mweight) - qweight

        if VERBOSE:
            if qscale > q_max:
                q_max = qscale
            if qscale < q_min:
                q_min = qscale

        if qscale < 0:
            if VERBOSE:
                print('small', node_n, k, qscale, max_weight, input_unsigned, output_unsigned)
                weights_small = True
            qscale = 0
        if qscale > 8:
            if VERBOSE:
                print('big', node_n, k, qscale, max_weight, input_unsigned, output_unsigned)
                weights_big = True
            qscale = 8
        if conv_cvi:
            scale_i16[k] = int(ceil(2**qscale)) - 1
            reqscale = log2(scale_i16[k]+1)
            scale_f32[k] = 2**(reqscale+qweight-qdiff)

        if conv_cvi:
            if scale_f32[k] == 0.:
                k_weights_i16 = [0 for w in k_weights]
                weights_i16[k*weights_per_kernel:(k+1)*weights_per_kernel] = k_weights_i16
                biases_i16[k] = 0
            else:
                k_weights_i16 = float_to_fixed_np(k_weights/scale_f32[k], calc_type.INT16, COEFFICIENT_BITS-1, qweight)
                weights_i16[k*weights_per_kernel:(k+1)*weights_per_kernel] = k_weights_i16

                if input_unsigned:
                    biases_i16[k] = float_to_fixed(biases_f32[k]/scale_f32[k], calc_type.INT16, COEFFICIENT_BITS-1, qweight, ROUNDING=USE_ROUNDING)
                else:
                    biases_i16[k] = float_to_fixed(-biases_f32[k]/scale_f32[k], calc_type.INT16, COEFFICIENT_BITS-1, qweight, ROUNDING=USE_ROUNDING)

                if INPUT_SIZE_BYTES == 1 and use_bias_lower:
                    if input_unsigned:
                        biases_i16[k] = float_to_fixed(biases_f32[k]/scale_f32[k], calc_type.INT16, COEFFICIENT_BITS-1, qweight)
                        biases_lower_i16[k] = float_to_fixed(1./255.*biases_f32[k]/scale_f32[k], calc_type.INT16, COEFFICIENT_BITS-1, qweight)
                    else:
                        biases_i16[k] = float_to_fixed(-biases_f32[k]/scale_f32[k], calc_type.INT16, COEFFICIENT_BITS-1, qweight)
                        biases_lower_i16[k] = float_to_fixed(-1./127.*biases_f32[k]/scale_f32[k], calc_type.INT16, COEFFICIENT_BITS-1, qweight)

    if not use_bias_lower:
        conv.bias_lower_scalar = 0

    if VERBOSE:
        print('{}\t{:3.2f}\t{:3.2f}'.format(node_n, q_max, q_min))
        if weights_big:
            print('big', node_n, input_unsigned, output_unsigned)
        if weights_small:
            print('small', node_n, node.unsigned_input, node.unsigned_output)

    if conv_cvi:
        padded_scale = pad_weights16(scale_i16, conv.kernels, 1, 1, 1, 1, padded_kernels, 1)
        packed_scales = np.zeros(padded_kernels, dtype=np.uint8)
        for k in range(0, padded_kernels, maps):
            iscales = interleave_weights(padded_scale[k:k+maps],
                                         1, maps, 1, 1)
            packed_scales[k:k+maps] = iscales
        conv.scale = len(weights)
        weights += packed_scales.tobytes()

    else:
        conv.scale = len(weights)
        padded_scales = scale_i16 + [0]*(padded_kernels-conv.kernels)
        weights += struct.pack("{}H".format(padded_kernels),
                               *padded_scales)

    if conv_cvi:
        padded_weights = pad_weights16(weights_i16.tolist(), conv.kernels,
                                       conv.channels//conv.group,
                                       conv.group,
                                       conv.kernel_shape[0],
                                       conv.kernel_shape[1],
                                       padded_kernels,
                                       conv.channels//conv.group)
        assert num_padded_weights == len(padded_weights)

        packed_weights = np.zeros(num_padded_weights, dtype=np.uint8)


        offset = ((conv.channels//conv.group) *
                  conv.kernel_shape[0]*conv.kernel_shape[1])

        for k in range(0, padded_kernels, maps):
            pack_weights(padded_weights[k*offset:], packed_weights[k*offset:], conv.kernel_shape[0], conv.kernel_shape[1], conv.channels//conv.group, maps, 1, maps)

        if conv.use_depthwise:
            packed_weights = np.zeros(((conv.kernel_shape[0]+2)//3*3*conv.kernel_shape[1] + 5)*padded_kernels, dtype=np.uint8)
            depthwise_pack_weights(padded_weights, packed_weights, scale_i16, conv.bias_scalar, biases_i16, conv.bias_lower_scalar, biases_lower_i16, conv.kernel_shape[0], conv.kernel_shape[1], conv.channels, padded_kernels, conv.group)

        conv.weights = len(weights)
        weights += packed_weights.tobytes()

    else:
        conv.use_weights32 = 1 if layer_max_weight > 1.0 * (1 << ((16-1)-Q16)) else 0
        conv.acc_maps = 0

        if conv.use_weights32:
            conv.weights32 = len(weights)
            fmt = "{}i".format(num_padded_weights)
            weights += struct.pack(fmt,
                                   *weights_i32)
            conv.weights = 0
        else:
            conv.weights = len(weights)
            fmt = "{}h".format(num_padded_weights)
            weights += struct.pack(fmt,
                                   *weights_i16)

    packed_biases = []
    packed_biases_lower = []

    if conv_cvi:
        padded_biases = pad_weights16(biases_i16,
                                      conv.kernels, 1, 1, 1, 1,
                                      padded_kernels, 1)
        packed_biases = np.zeros(padded_kernels, dtype=np.uint8)
        offset = 1
        for k in range(0, padded_kernels, maps):
            interleaved_biases = interleave_weights(padded_biases[k:k+maps],
                                                    1, maps,
                                                    1, 1)
            packed_biases[k:k + maps] = interleaved_biases
        conv.biases = len(weights)
        weights += packed_biases.tobytes()

        padded_biases_lower = pad_weights16(biases_lower_i16,
                                            conv.kernels, 1, 1, 1, 1,
                                            padded_kernels, 1)
        packed_biases_lower = np.zeros(padded_kernels, dtype=np.uint8)
        offset = 1
        for k in range(0, padded_kernels, maps):
            interleaved_biases_lower = interleave_weights(padded_biases_lower[k:k+maps],
                                                          1, maps,
                                                          1, 1)
            packed_biases_lower[k:k + maps] = interleaved_biases_lower

        conv.biases_lower = len(weights)
        weights += packed_biases_lower.tobytes()
    else:
        conv.biases = len(weights)
        weights += struct.pack("{}h".format(len(biases_i16)),
                               *biases_i16)
        conv.biases_lower = len(weights)
        weights += struct.pack("{}h".format(len(biases_i16)),
                               *biases_i16)
    return


def json_to_graph(json_string, preset, io_info=None, script_dir=None, output_bytes=4, bias_corrections=None, bias_correction_nodes=(None, None)):

    current_corrections, all_corrections = bias_correction_nodes
    corrections = None
    if bias_corrections:
        with open(bias_corrections) as f:
            corrections = json.load(f)

    if script_dir is None:
        script_path = os.path.dirname(os.path.realpath(__file__))
    else:
        script_path = script_dir

    # installed directory
    vnnx_types_path = os.path.join(script_path, "vnnx-types.h")
    if not os.path.exists(vnnx_types_path):
        vnnx_types_path = os.path.join(script_path,"../../../../libvnnx/include/vnnx-types.h")
    vnnx_type_hash = hashlib.md5(open(vnnx_types_path,"rb").read()).hexdigest()
    VNNX_GRAPH_VERSION = int(vnnx_type_hash[:8], 16)

    #Set parameters frm preset
    preset_select= {"VECTOR_LANES": { "V250" : 4,
                                      "V500"       : 8,
                                      "V1000"      : 8,
                                      "V2000"       : 16,
                                      "V4000" : 16},
                    "FILTER_COPIES": { "V250" : 16,
                                       "V500"       : 32,
                                       "V1000"      : 32,
                                       "V2000"       : 64,
                                       "V4000" : 64},
                    "PARALLEL_OUTPUT_MAPS": { "V250" : 16,
                                              "V500"       : 16,
                                              "V1000"      : 32,
                                              "V2000"       : 32,
                                              "V4000" : 64},
                    "SCRATCHPAD_KB": { "V250" : 64 ,
                                       "V500"       : 128,
                                       "V1000"      : 256,
                                       "V2000"       : 512,
                                       "V4000" : 512},
                    "PRESET" : { "V250" : 0 ,
                                 "V500"       : 1,
                                 "V1000"      : 2,
                                 "V2000"       : 3,
                                 "V4000" : 4},
    }

    VECTOR_LANES = preset_select['VECTOR_LANES'][preset]
    FILTER_COPIES = preset_select['FILTER_COPIES'][preset]
    PARALLEL_OUTPUT_MAPS = preset_select['PARALLEL_OUTPUT_MAPS'][preset]
    SCRATCHPAD_KB = preset_select['SCRATCHPAD_KB'][preset]
    VBX_NN_PRESET = preset_select['PRESET'][preset]
    sp_size = SCRATCHPAD_KB*1024

    js = json.loads(json_string)
    Nodes = []

    weights = weight_array()

    # get input and output nodes, and layers
    js_layers = js['layers']
    if current_corrections:
        js_layers = [js['layers'][_] for _ in current_corrections]
        output_id = js_layers[-1]['output_id']
        next_nodes = [l for l in js['layers'] if l['input_id'] == output_id]
        if len(next_nodes):
            next_node_type = subgraph_type.from_str(next_nodes[0]["op_type"])
            if requires16(next_node_type, 1):
                output_bytes = 2
            else:
                output_bytes = 1
        all_corrections_names = [js['layers'][_]['name'] for _ in all_corrections]
    else:
        all_corrections_names = []

    inputs = [l['input_id'] for l in js_layers]
    outputs = [l['output_id'] for l in js_layers]
    output_indices = [n for n, l in enumerate(js_layers) if l['output_id'] not in inputs]
    input_indices = [n for n, l in enumerate(js_layers) if l['input_id'] not in outputs]
    output_ids = [js_layers[i]['output_id'] for i in output_indices]
    input_ids = [js_layers[i]['input_id'] for i in input_indices]

    for node_n, _ in enumerate(js_layers):
        node = Node()
        Nodes.append(node)
        json_node = js_layers[node_n]
        node.type = subgraph_type.from_str(json_node["op_type"])
        node.input_unsigned = json_node['input_unsigned']
        node.output_unsigned = json_node['output_unsigned']

        node.input_description = js_layers[node_n]['input_description'].encode()[:24]
        while len(node.input_description) < 24:
            node.input_description += b"\0"
        node.output_description = js_layers[node_n]['output_description'].encode()[:24]
        while len(node.output_description) < 24:
            node.output_description += b"\0"

        if 'output_strides' in js_layers[node_n]:
            node.output_strides = js_layers[node_n]['output_strides']

        output_id = js_layers[node_n]['output_id']
        next_nodes = [l for l in js_layers if l['input_id'] == output_id]
        if len(next_nodes):
            next_node_type = subgraph_type.from_str(next_nodes[0]["op_type"])
        else:
            next_node_type = subgraph_type.UNKNOWN

        ignore = ignore_cvi(json_node)
        if ignore and VERBOSE:
            sys.stdout.write("Forcing vector on layer {} because {}\n"
                             .format(node_n, ignore))
        conv_cvi = USE_CVI and not ignore

        cvi8 = INPUT_SIZE_BYTES == 1
        # is_preprocess = (node_n == 0 and node.type == subgraph_type.IDENTITY and
        #                  ('Add' in [sl['op_type'] for sl in json_node['sublayers']] or
        #                   'Mul' in [sl['op_type'] for sl in json_node['sublayers']]))
        # is_preprocess = node_n == 0 and node.type == subgraph_type.IDENTITY
        is_preprocess = json_node == js['layers'][0] and node.type == subgraph_type.IDENTITY # doesn't work for multiple inputs

        current_data_type = None

        node.input_data_type = calc_type.INT16
        if cvi8 and not requires16(node.type, output_bytes):
            if node.input_unsigned:
                node.input_data_type = calc_type.UINT8
                if node.type == subgraph_type.CONV and not node.output_unsigned:
                    current_data_type = calc_type.INT8
            else:
                node.input_data_type = calc_type.INT8
                if node.type == subgraph_type.CONV and node.output_unsigned:
                    current_data_type = calc_type.UINT8

        node.output_data_type = calc_type.INT16
        if cvi8 and not requires16(next_node_type, output_bytes):
            if node.output_unsigned:
                node.output_data_type = calc_type.UINT8
            else:
                node.output_data_type = calc_type.INT8

        if json_node['output_id'] in output_ids and output_bytes == 4:
            node.output_data_type = calc_type.INT32

        if current_data_type == None:
            current_data_type = node.input_data_type
        if (node.type == subgraph_type.CONV and not conv_cvi and INPUT_SIZE_BYTES == 1):
            current_data_type = calc_type.INT16

        printv("layer_type =", json_node['op_type'], current_data_type)

        pool_shape = [1, 1]
        pool_strides = [1, ]
        pad_channels = 0

        if node.output_data_type != current_data_type:
            type_conv = (current_data_type, node.output_data_type)
            if type_conv == (calc_type.INT8, calc_type.INT16):
                sn = Subnode()
                node.subnode_array.append(sn)
                sn.type = layer_type.CAST_I8_I16
                sn.cast.scale = 1
                current_data_type = calc_type.INT16

            elif type_conv == (calc_type.UINT8, calc_type.INT16):
                sn = Subnode()
                node.subnode_array.append(sn)
                sn.type = layer_type.CAST_U8_I16
                sn.cast.scale = 1
                current_data_type = calc_type.INT16

            elif type_conv == (calc_type.UINT8, calc_type.INT8) and json_node['op_type'] == 'Identity':
                sn = Subnode()
                node.subnode_array.append(sn)
                sublayers = [_ for _ in json_node['sublayers'] if _['op_type'] in ['Add', 'Mul']]
                sn.cast.scale = 1
                if len(sublayers):
                    sn.type = layer_type.CAST_U8_I16
                    current_data_type = calc_type.INT16
                else:
                    sn.type = layer_type.CAST_U8_I8
                    current_data_type = calc_type.INT8

        for json_sl in json_node['sublayers']:
            sn = Subnode()
            node.subnode_array.append(sn)
            op_type = json_sl['op_type']
            printv(op_type)

            if current_data_type == calc_type.UINT8:
                if op_type in ['Relu', 'PRelu', 'LeakyRelu', 'Clip', 'Add', 'Abs']:
                    print('not supported')

            if op_type in ['MaxPool', 'AveragePool']:
                sn.kernel_shape = json_sl['kernel_shape']
                sn.strides = json_sl['strides']
                pool_shape = sn.kernel_shape
                pool_strides = sn.strides
                sn.pads = json_sl['pads']

                if op_type == 'MaxPool':
                    pool = (pool_shape[0], pool_strides[0])
                    if current_data_type == calc_type.INT8:
                        sn.type = layer_type.MAXPOOL_I8
                    elif current_data_type == calc_type.INT16:
                        sn.type = layer_type.MAXPOOL_I16
                    elif current_data_type == calc_type.UINT8:
                        sn.type = layer_type.MAXPOOL_U8
                    else:
                        print('error for maxpool')
                else:
                    if current_data_type == calc_type.INT8:
                        sn.type = layer_type.AVGPOOL_I8
                    elif current_data_type == calc_type.INT16:
                        sn.type = layer_type.AVGPOOL_I16
                    elif current_data_type == calc_type.UINT8:
                        sn.type = layer_type.AVGPOOL_U8
                    else:
                        print('error for averagepool')

            elif op_type == 'Abs':
                sn.type = layer_type.ABS_I16
                if current_data_type == calc_type.INT8:
                    sn.type = layer_type.ABS_I8

            elif op_type == 'Conv':
                assert(json_sl['use_depthwise'])

                attrs = ('kernels', 'channels', 'group', 'm', 'n',
                         'kernel_shape', 'strides', 'dilations',
                         'use_cvi', 'use_strided', 'use_depthwise')
                for a in attrs:
                    setattr(sn.depthwise, a, json_sl[a])

                sn.depthwise.unsigned_input = json_sl['input_unsigned']
                sn.depthwise.unsigned_output = json_sl['output_unsigned']
                sn.kernel_shape = json_sl['kernel_shape']
                sn.strides = json_sl['strides']
                sn.dilation = json_sl['dilations']
                sn.depthwise.weights = 0

                if current_data_type == calc_type.INT8:
                    assert(sn.depthwise.unsigned_input == 0)
                elif current_data_type == calc_type.UINT8:
                    assert(sn.depthwise.unsigned_input == 1)

                if current_data_type == calc_type.INT8:
                    if sn.depthwise.unsigned_output == 1:
                        current_data_type = calc_type.UINT8
                        node.output_data_type = calc_type.UINT8

                if current_data_type == calc_type.UINT8:
                    if sn.depthwise.unsigned_output == 0:
                        current_data_type = calc_type.INT8
                        node.output_data_type = calc_type.INT8

                sn.type = layer_type.DEPTHWISE_CONV_I8

            elif op_type == 'PRelu':
                sn.prelu.slope = len(weights)
                sn.maps = len(json_sl['slope'])
                weights += float_array_to_weights(json_sl['slope'], current_data_type)
                sn.type = layer_type.PRELU_I8
                if current_data_type == calc_type.INT16:
                    sn.type = layer_type.PRELU_I16

            elif op_type == 'LeakyRelu':
                sn.leakyrelu.alpha = float_to_fixed(json_sl['alpha'], current_data_type)
                sn.type = layer_type.LEAKYRELU_I8
                if current_data_type == calc_type.INT16:
                    sn.type = layer_type.LEAKYRELU_I16

            elif op_type == 'Relu':
                sn.type = layer_type.RELU_I8
                if current_data_type == calc_type.INT16:
                    sn.type = layer_type.RELU_I16

            elif op_type == 'Clip':
                sn.clip.min = json_sl['min']
                sn.clip.max = json_sl['max']
                sn.type = layer_type.CLIP_I8
                if current_data_type == calc_type.INT16:
                    sn.type = layer_type.CLIP_I16

            elif op_type == 'Add':
                arr = np.asarray(json_sl['array'])
                corr = None
                if not(corrections is None):
                    desc = json_node['name']
                    if desc in corrections:
                        corr = np.asarray(corrections[desc])
                        arr = arr + corr

                dims = json_sl['dims']
                arr = arr.tolist()
                sn.maps = len(json_sl['array'])
                if dims[-1] == 1:
                    sn.add_broadcast_map.use_xl = 0
                    if current_data_type == calc_type.INT8:
                        sn.type = layer_type.ADD_BROADCAST_MAP_I8
                        sn.add_broadcast_map.array = len(weights)
                        weights += float_array_to_weights(arr, calc_type.INT8)
                        sn.add_broadcast_map.array_xl = len(weights)
                        weights += float_array_to_weights(arr, calc_type.INT16)
                        max_possible_weight = 1.0 * (1 << ((8-1) - Q8))
                    if current_data_type == calc_type.INT16:
                        sn.type = layer_type.ADD_BROADCAST_MAP_I16
                        sn.add_broadcast_map.array = len(weights)
                        weights += float_array_to_weights(arr, calc_type.INT16)
                        sn.add_broadcast_map.array_xl = len(weights)
                        weights += float_array_to_weights(arr, calc_type.INT32)
                        max_possible_weight = 1.0 * (1 << ((16-1) - Q16))
                    if max(arr) > max_possible_weight:
                        sn.add_broadcast_map.use_xl = 1
                elif dims[-2] == 1 and dims[-1] > 1:
                    sn.add_broadcast_row.use_xl = 0
                    if current_data_type == calc_type.INT8:
                        sn.type = layer_type.ADD_BROADCAST_ROW_I8
                        sn.add_broadcast_row.array = len(weights)
                        weights += float_array_to_weights(arr, calc_type.INT8)
                        sn.add_broadcast_row.array_xl = len(weights)
                        weights += float_array_to_weights(arr, calc_type.INT16)
                        max_possible_weight = 1.0 * (1 << ((8-1) - Q8))
                    if current_data_type == calc_type.INT16:
                        sn.type = layer_type.ADD_BROADCAST_ROW_I16
                        sn.add_broadcast_row.array = len(weights)
                        weights += float_array_to_weights(arr, calc_type.INT16)
                        sn.add_broadcast_row.array_xl = len(weights)
                        weights += float_array_to_weights(arr, calc_type.INT32)
                        max_possible_weight = 1.0 * (1 << ((16-1) - Q16))
                    if max(arr) > max_possible_weight:
                        sn.add_broadcast_row.use_xl = 1
            elif op_type == 'Mul':
                dims = json_sl['dims']
                if dims == [1]:
                    sn.mul_scalar.use_xl = 0
                    sn.mul_scalar.scalarf32 = json_sl['array'][0]
                    sn.mul_scalar.scalaru8 = float_to_fixed(json_sl['array'][0], calc_type.UINT8)
                    sn.mul_scalar.scalar8 = float_to_fixed(json_sl['array'][0], calc_type.INT8)
                    sn.mul_scalar.scalar16 = float_to_fixed(json_sl['array'][0], calc_type.INT16)
                    sn.mul_scalar.scalar32 = float_to_fixed(json_sl['array'][0], calc_type.INT32)
                    if current_data_type == calc_type.UINT8:
                        assert(np.max(np.asarray(json_sl['array'][0])) > 0)
                        sn.type = layer_type.MUL_SCALAR_U8
                        max_possible_weight = 1.0 * (1 << ((8) - U8))
                    elif current_data_type == calc_type.INT8:
                        sn.type = layer_type.MUL_SCALAR_I8
                        max_possible_weight = 1.0 * (1 << ((8-1) - Q8))
                    elif current_data_type == calc_type.INT16:
                        sn.type = layer_type.MUL_SCALAR_I16
                        max_possible_weight = 1.0 * (1 << ((8-1) - Q8))

                    if max(json_sl['array']) > max_possible_weight:
                        sn.mul_scalar.use_xl = 1
                elif dims[-1] == 1:
                    sn.type = layer_type.MUL_BROADCAST_MAP_I16
                    sn.maps = len(json_sl['array'])
                    sn.mul_broadcast_map.use_xl = 0
                    sn.mul_broadcast_map.array=len(weights)
                    weights += float_array_to_weights(json_sl['array'], calc_type.INT16)
                    sn.mul_broadcast_map.array_xl = len(weights)
                    weights += float_array_to_weights(json_sl['array'], calc_type.INT32)
                    max_possible_weight = 1.0 * (1 << ((16-1) - Q16))
                    if (current_data_type == calc_type.INT8):
                        sn.type = layer_type.MUL_BROADCAST_MAP_I8
                    if max(json_sl['array']) > max_possible_weight:
                        sn.mul_broadcast_map.use_xl = 1
                elif dims[-2] == 1 and dims[-1] > 1:
                    sn.type = layer_type.MUL_BROADCAST_ROW_I16
                    sn.mul_broadcast_row.use_xl = 0
                    sn.mul_broadcast_row.array=len(weights)
                    weights += float_array_to_weights(json_sl['array'], calc_type.INT16)
                    sn.mul_broadcast_row.array_xl = len(weights)
                    weights += float_array_to_weights(json_sl['array'], calc_type.INT32)
                    max_possible_weight = 1.0 * (1 << ((16-1) - Q16))
                    if (current_data_type == calc_type.INT8):
                        sn.type = layer_type.MUL_BROADCAST_ROW_I8
                    if max(json_sl['array']) > max_possible_weight:
                        sn.mul_broadcast_row.use_xl = 1
            elif op_type == 'Pad':
                sn.pad_const.value = json_sl['value']
                if current_data_type == calc_type.UINT8:
                    if sn.pad_const.value < 0:
                        sn.pad_const.value = 0
                sn.pads = json_sl['pads']
                pad_channels += sn.pads[0] + sn.pads[3]
                assert len(sn.pads) == 6
                if current_data_type == calc_type.INT8:
                    sn.type = layer_type.PADCONST_I8
                elif current_data_type == calc_type.UINT8:
                    sn.type = layer_type.PADCONST_U8
                elif current_data_type == calc_type.INT16:
                    sn.type = layer_type.PADCONST_I16
            else:
                errmsg = "Unknown subnode type {}. skipping\n".format(op_type)
                sys.stderr.write(errmsg)

        # at this point the output type of the nodes sublayers should be
        # the same as the output of nodes declared output type
        if node.output_data_type != current_data_type:
            sn = Subnode()
            node.subnode_array.append(sn)
            type_conv = (current_data_type, node.output_data_type)
            if type_conv == (calc_type.INT16, calc_type.INT8):
                sn.type = layer_type.CAST_I16_I8
                sn.cast.scale = 1
                current_data_type = calc_type.INT8
            elif json_node['output_id'] in output_ids:
                scale_factor = io_info['output_scale_factors'][output_ids.index(json_node['output_id'])]
                if type_conv == (calc_type.INT8, calc_type.INT32):
                    sn.type = layer_type.CAST_I8_I32
                    sn.cast.scale = int(scale_factor * (1 << (Q32-Q8)))
                    current_data_type = calc_type.INT32
                elif type_conv == (calc_type.INT16, calc_type.INT32):
                    sn.type = layer_type.CAST_I16_I32
                    sn.cast.scale = int(scale_factor * (1 << (Q32-Q16)))
                    current_data_type = calc_type.INT32
                elif type_conv == (calc_type.UINT8, calc_type.INT32):
                    sn.type = layer_type.CAST_U8_I32
                    sn.cast.scale = int(scale_factor * (1 << (Q32-U8)))
                    current_data_type = calc_type.INT32
                else:
                    print(current_data_type, node.output_data_type)
                    assert False, "BAD TYPES"
            else:
                print(current_data_type, node.output_data_type)
                assert False, "BAD TYPES"

        node.input_size = json_node['input_size']
        node.output_size = json_node['output_size']

        node.input_shape = list(json_node['input_shape'])
        while len(node.input_shape) < 3:
            node.input_shape=[1]+node.input_shape
        node.output_shape = list(json_node['output_shape'])
        while len(node.output_shape) < 3:
            node.output_shape=[1]+node.output_shape

        node.dma_channel_offset = json_node['dma_offset']
        node.dma_output_buffer_offset = json_node['output_buffer_offset']*sizeof_calc_type(node.output_data_type)
        node.dma_input_buffer_offset = json_node['input_buffer_offset']*sizeof_calc_type(node.input_data_type)
        node.use_replay = json_node['use_replay']

        set_sublayer_attributes(node, node.subnode_array)

        if node.type != subgraph_type.IDENTITY and node.sublayer_maps > 1:
            raise NotImplementedError("Padding that adds channels must be in Identity layer")

        if(node.type == subgraph_type.CONV):
            attrs = ('kernels', 'channels', 'group', 'm', 'n',
                     'kernel_shape', 'strides', 'dilations',
                     'use_cvi', 'use_strided', 'use_depthwise')
            for a in attrs:
                setattr(node.conv, a, json_node[a])

            conv_populate_attributes(node, json_node, conv_cvi, sp_size, VECTOR_LANES, FILTER_COPIES, PARALLEL_OUTPUT_MAPS)

            conv = node.conv

            conv_pack_parameters(weights, conv_cvi, conv.padded_kernels, conv.maps, node.input_unsigned, node.output_unsigned, conv, json_node, corrections, all_corrections, all_corrections_names)
                
        elif node.type == subgraph_type.IDENTITY:
            identity = node.identity
            identity_bytes = 1
            if node.input_data_type == calc_type.INT16 or node.output_data_type == calc_type.INT16 or len(node.subnode_array) > 1:
                identity_bytes = 2

            identity_bytes = max(sizeof_calc_type(node.input_data_type),
                                 sizeof_calc_type(node.output_data_type),
                                 *[sublayer_bytes(sn.type) for sn in node.subnode_array])

            identity.channels = json_node['channels']
            identity.m = json_node['m']
            identity.n = json_node['n']
            identity.maps = identity.channels

            identity.core_m = identity.m
            identity.core_split = 1
            identity.core_maps = ceil(identity.maps / CORES)

            if len(node.subnode_array):
                dma_split = 4
                if dma_split > identity.maps:
                    dma_split = identity.maps
            else:
                dma_split = 1
            dma_split = 1 #TODO

            max_maps = max_ident_maps(identity.m + node.sublayer_rows, identity.n + node.sublayer_columns, sp_size, identity_bytes, dma_split)
            if max_maps - node.sublayer_maps > 1:
                identity.maps = identity.core_maps
                while identity.maps > max_maps - node.sublayer_maps:
                    identity.maps //= 2
            else:
                identity.maps = 1

            if dma_split > identity.maps:
                dma_split = identity.maps


            identity.rows = max_ident_maps(identity.maps + node.sublayer_maps, identity.n + node.sublayer_columns, sp_size, identity_bytes, dma_split) - node.sublayer_rows

            if identity.rows > identity.m:
                identity.rows = identity.m
            if identity.rows != identity.m:
                min_rows = 1 + node.sublayer_shape[0]
                inc_rows = node.sublayer_stride[0]
                while (identity.rows - min_rows) % inc_rows:
                    identity.rows -= 1
                assert identity.rows >= min_rows

            node.scratchpad_bytes = ((identity.rows + node.sublayer_rows) *
                                     (identity.n + node.sublayer_columns) *
                                     (identity.maps + node.sublayer_maps)* identity_bytes)

            node.dma_split = dma_split

        elif node.type == subgraph_type.SUM:
            # Dont use shortcut here since sum is a builtin name
            # sum = node.sum
            node.sum.channels = json_node['channels']
            node.sum.m = json_node['m']
            node.sum.n = json_node['n']
            node.sum.num_inputs = json_node['num_inputs']
            node.sum.rows = node.sum.m
            node.sum.maps = ceil(node.sum.channels / CORES)

            sum_bytes = max(sizeof_calc_type(node.input_data_type),
                                 sizeof_calc_type(node.output_data_type),
                                 *[sublayer_bytes(sn.type) for sn in node.subnode_array])
            assert node.sum.maps*CORES == node.sum.channels, "SUM Node must have channels divisible by CORES"
            map_size = (node.sum.m + node.sublayer_rows)*(node.sum.n + node.sublayer_columns)*sum_bytes
            node.sum.maps = min(sp_size// ((2+1)*ceil(map_size/32)*32),node.sum.maps)

            if node.sum.maps < 1:
                node.sum.maps =1
                partial_map_size = sp_size // (sum_bytes*(2+1))
                node.sum.rows = min(partial_map_size// (node.sum.n + node.sublayer_columns),node.sum.rows)

            node.scratchpad_bytes = node.sum.maps * node.sum.rows*node.sum.n*sum_bytes
        elif node.type == subgraph_type.MAX:
            node.max.channels = json_node['channels']
            node.max.m = json_node['m']
            node.max.n = json_node['n']
            node.max.num_inputs = json_node['num_inputs']

            calc_bytes = max(sizeof_calc_type(node.input_data_type),
                                  sizeof_calc_type(node.output_data_type),
                                  *[sublayer_bytes(sn.type) for sn in node.subnode_array])

            map_size = node.output_shape[1]*node.output_shape[2]
            node.scratchpad_bytes = node.output_shape[0]*map_size*calc_bytes
        elif node.type == subgraph_type.MIN:
            node.min.channels = json_node['channels']
            node.min.m = json_node['m']
            node.min.n = json_node['n']
            node.min.num_inputs = json_node['num_inputs']

            calc_bytes = max(sizeof_calc_type(node.input_data_type),
                                  sizeof_calc_type(node.output_data_type),
                                  *[sublayer_bytes(sn.type) for sn in node.subnode_array])

            map_size = node.output_shape[1]*node.output_shape[2]
            node.scratchpad_bytes = node.output_shape[0]*map_size*calc_bytes
        elif node.type == subgraph_type.ARGMAX:
            argmax = node.argmax
            argmax.channels = json_node['channels']
            argmax.m = json_node['m']
            argmax.n = json_node['n']

            passes = 1
            channel_size=argmax.m*argmax.n
            while ceil((channel_size*argmax.channels*4 +channel_size*4)/passes) > sp_size:
                passes+=1
            argmax.pixels_per_loop = channel_size//passes;
            node.scratchpad_bytes = 0;
        elif node.type == subgraph_type.LRN:
            node.lrn.channels = json_node["channels"]
            node.lrn.m = json_node["m"]
            node.lrn.n = json_node["n"]

            node.lrn.size = json_node["size"]
            node.lrn.alpha = json_node["alpha"]
            node.lrn.beta = json_node["beta"]
            node.lrn.scale = json_node["scale"]
            node.lrn.bias = 1.0 if json_node["bias"] is None else json_node["bias"]
            node.lrn.rows = node.lrn.m
            node.lrn.maps = ceil(node.lrn.channels/CORES)
            node.scratchpad_bytes = 0
            assert node.lrn.maps*CORES == node.lrn.channels
        elif node.type == subgraph_type.ACTIVATION:
            float_array = json_node['scale']
            node.activation.channels = json_node["channels"]
            node.activation.m = json_node["m"]
            node.activation.n = json_node["n"]
            node.activation.scale = len(weights)
            weights += struct.pack("f"*len(float_array), *float_array)

            node.activation.mode = {"Softmax":activation_mode.SOFTMAX,
                                "Sigmoid":activation_mode.SIGMOID}[json_node['op_type']]

            assert(sp_size > (4*4*node.activation.n))
            node.activation.maps = 1
            node.activation.rows = 1
            node.scratchpad_bytes = 0

        elif node.type == subgraph_type.TRANSPOSE:
            node.transpose.channels = json_node["channels"]
            node.transpose.m = json_node["m"]
            node.transpose.n = json_node["n"]
            node.transpose.permutation = json_node['permutation']

            calc_bytes = max(sizeof_calc_type(node.input_data_type),
                                  sizeof_calc_type(node.output_data_type),
                                  *[sublayer_bytes(sn.type) for sn in node.subnode_array])
            map_size = max(node.output_shape[1]*node.output_shape[2], node.transpose.m*node.transpose.n)
            row_size = max(node.output_shape[2], node.transpose.n)

            maps_at_once = sp_size//(2*map_size*calc_bytes)
            if maps_at_once > node.transpose.channels:
                maps_at_once = node.transpose.channels

            if maps_at_once > 0:
                rows_at_once = node.transpose.m
            else:
                maps_at_once = 1
                rows_at_once = sp_size//(2*2*row_size*calc_bytes) # TODO
                if rows_at_once > node.transpose.m:
                    rows_at_once = node.transpose.m

            assert(maps_at_once > 0 and rows_at_once > 0)

            node.transpose.out_maps_at_once = maps_at_once
            node.transpose.out_rows_at_once = rows_at_once
            node.scratchpad_bytes = maps_at_once*rows_at_once*row_size*calc_bytes

        elif node.type == subgraph_type.REDUCEMEAN:
            calc_bytes = max(sizeof_calc_type(node.input_data_type),
                                  sizeof_calc_type(node.output_data_type),
                                  *[sublayer_bytes(sn.type) for sn in node.subnode_array])
            map_size = node.output_shape[1]*node.output_shape[2]
            assert((sp_size-(map_size*4))//(map_size*calc_bytes) > node.output_shape[0])
            node.scratchpad_bytes = node.output_shape[0]*map_size*calc_bytes
            node.reduce.channels = json_node['channels']
            node.reduce.m = json_node['m']
            node.reduce.m0 = json_node['m0']
            node.reduce.n = json_node['n']

        elif node.type == subgraph_type.RESIZE:
            node.resize.channels = json_node["channels"]
            node.resize.mode = {"nearest":resize_mode.NEAREST,
                                "linear":resize_mode.LINEAR}[json_node['mode']]
            node.resize.m = json_node["m"]
            node.resize.n = json_node["n"]
            node.resize.scale = json_node["scale"]
            node.resize.rows = node.resize.m
            node.resize.maps = ceil(node.resize.channels/CORES)
            node.scratchpad_bytes = 0
            assert node.resize.maps*CORES == node.resize.channels
        elif node.type == subgraph_type.TILE:
            calc_bytes = max(sizeof_calc_type(node.input_data_type),
                                  sizeof_calc_type(node.output_data_type),
                                  *[sublayer_bytes(sn.type) for sn in node.subnode_array])
            map_size = node.output_shape[1]*node.output_shape[2]
            assert((sp_size-(map_size*4)*2)//(map_size*calc_bytes) > node.output_shape[0])
            node.scratchpad_bytes = node.output_shape[0]*map_size*calc_bytes
            node.tile.channels = json_node["channels"]
            node.tile.m = json_node["m"]
            node.tile.n = json_node["n"]
            node.tile.tile = json_node["tile"]
            node.tile.rows = node.tile.m
            node.tile.maps = ceil(node.tile.channels/CORES)
            assert node.tile.maps*CORES == node.tile.channels
        elif node.type == subgraph_type.REORG:
            node.reorg.channels = json_node["channels"]
            node.reorg.m = json_node["m"]
            node.reorg.n = json_node["n"]
            node.reorg.stride = json_node["stride"]
            node.reorg.rows = node.reorg.m
            node.reorg.maps = ceil(node.reorg.channels/CORES)
            node.scratchpad_bytes = 0
            assert node.reorg.maps*CORES == node.reorg.channels
        elif node.type == subgraph_type.GEMM:
            assert node.input_data_type == calc_type.INT16
            gemm = node.gemm
            gemm.input_size = json_node['gemm_input_size']
            gemm.output_size = json_node['gemm_output_size']
            gemm_bytes = 2

            gemm.max_input_size = max_gemm_columns(gemm, sp_size, gemm_bytes)
            gemm.max_output_size = (sp_size - (gemm.max_input_size*3*gemm_bytes)) / (2*gemm_bytes)

            ideal_size_per_core = ceil(gemm.max_input_size/CORES)
            while ideal_size_per_core > gemm.max_output_size:
                ideal_size_per_core = ceil(ideal_size_per_core/2)
            gemm.max_output_size = ideal_size_per_core

            nweights = gemm.output_size * gemm.input_size
            weights_f32 = base64_to_float_nparray(json_node['weights'])
            assert len(weights_f32) == nweights

            if gemm.max_output_size > gemm.output_size:
                gemm.max_output_size = gemm.output_size

            gemm.weights = len(weights)
            weights += float_to_fixed_np(weights_f32, calc_type.INT16).tobytes()

            # nbiases = gemm.output_size
            biases_f32 = base64_to_float_array(json_node['biases'])
            if not(corrections is None):
                desc = json_node['name']
                if desc in corrections:
                    corr = np.asarray(corrections[desc])
                    biases_f32 = biases_f32 + corr

            gemm.biases = len(weights)
            weights += float_array_to_weights(biases_f32, calc_type.INT16)
            node.scratchpad_bytes = 2*gemm.max_output_size

        prefetch_subnodes = []
        for sn in node.subnode_array:
            if sn.type == layer_type.DEPTHWISE_CONV_I8:
                json_nodes = [json_sl for json_sl in  json_node['sublayers'] if json_sl['op_type'] == "Conv"]
                assert(len(json_nodes) == 1)
                json_node = json_nodes[0]
                conv_pack_parameters(weights, True, node.conv.padded_kernels, node.conv.maps//node.dma_split, sn.depthwise.unsigned_input, sn.depthwise.unsigned_output, sn.depthwise, json_node, corrections, all_corrections, all_corrections_names)

                prefetch_sn = Subnode()
                prefetch_sn.type = layer_type.PREFETCH
                prefetch_sn.prefetch.memory_offset = sn.depthwise.weights
                prefetch_subnodes.append(prefetch_sn)
        node.subnode_array = prefetch_subnodes + node.subnode_array


    # setup graph object
    graph = Graph()
    graph.num_inputs = len(input_indices)
    graph.num_outputs = len(output_indices)

    io_nodes = np.array(input_indices + output_indices, dtype=np.int32)
    graph.io_nodes = len(weights)
    weights += io_nodes.tobytes()

    io_buffer_size = [sz for sz in js['buffers']]

    REUSE_IO_MEMORY = all_corrections is None
    REUSE_IO_MEMORY = False
    if REUSE_IO_MEMORY:
        # determine when each i/o buffer should be allocated and deallocated, and find its size in bytes
        alloc = dict()      # layer to allocate buffer; alloc[io_id] = layer index
        dealloc = dict()    # layer to deallocate buffer; dealloc[io_id] = layer index
        for ind in input_indices:
            io = js_layers[ind]['input_id']
            alloc[io] = 0       # allocate all inputs at the beginning
        for ind in output_indices:
            io = js_layers[ind]['output_id']
            alloc[io] = 0       # allocate all outputs at the beginning
        for ind, json_node in enumerate(js_layers):
            for io in (json_node['input_id'], json_node['output_id']):
                if not io in alloc:
                    alloc[io] = ind         # allocate only on the first use
                if not ind in input_indices+output_indices:
                    dealloc[io] = ind       # deallocate on the last use; don't deallocate inputs/outputs
            if Nodes[ind].input_data_type == calc_type.INT16:
                io_buffer_size[json_node['input_id']] *= 2
            if Nodes[ind].input_data_type == calc_type.INT32:
                io_buffer_size[json_node['input_id']] *= 4
            if Nodes[ind].output_data_type == calc_type.INT16:
                io_buffer_size[json_node['output_id']] *= 2
            if Nodes[ind].output_data_type == calc_type.INT32:
                io_buffer_size[json_node['output_id']] *= 4
        # determine the location of each i/o buffer, reusing memory when possible
        heap = [(0, 0)]  # occupied memory; each list element is the range of an allocation; start with dummy node
        heap_ptr = dict()   # pointers to heap, including deallocated; heap_ptr[io_id] = memory location
        heap_max = heap[0][1]   # keep track of the largest amount of memory used
        for ind in range(len(js_layers)):
            for io, alloc_ind in alloc.items():
                if ind==alloc_ind:  # find all allocations during this layer
                    for h in range(1, len(heap)):
                        if heap[h][0] - heap[h-1][1] >= io_buffer_size[io]: # see if new buffer can fit between allocations
                            heap_ptr[io] = heap[h-1][1]
                            heap.insert(h,(heap_ptr[io], heap_ptr[io]+io_buffer_size[io]))
                            break
                    if not io in heap_ptr:  # put new buffer at the end of the heap
                        heap_ptr[io] = heap[-1][1]
                        heap.append((heap_ptr[io], heap_ptr[io] + io_buffer_size[io]))
                    heap_max = max(heap_max, heap[-1][1])
            for io, dealloc_ind in dealloc.items():
                if ind == dealloc_ind:    # find all deallocations during this layer
                    for h in range(1, len(heap)):
                        if heap_ptr[io] == heap[h][0]:  # find the allocation in the heap
                            heap.pop(h)
                            break
        for ind, json_node in enumerate(js_layers):
            Nodes[ind].input_data = heap_ptr[json_node['input_id']] + len(weights)
            Nodes[ind].output_data = heap_ptr[json_node['output_id']] + len(weights)

        # create heap memory and write input and output test buffers
        heap_mem = bytearray(heap_max)
        for n, ind in enumerate(input_indices):
            io = js_layers[ind]['input_id']
            float_array = base64_to_float_nparray(js['test_input'][n])
            byte_array = float_to_fixed_np(float_array, Nodes[ind].input_data_type, ROUNDING=False).tobytes()
            heap_mem[heap_ptr[io] : heap_ptr[io]+io_buffer_size[io]] = byte_array
        for n,ind in enumerate(output_indices):
            io = js_layers[ind]['output_id']
            float_array = base64_to_float_nparray(js['test_output'][n])
            scale_factor = io_info['output_scale_factors'][n]
            byte_array = float_to_fixed_np(scale_factor*float_array, node.output_data_type).tobytes()
            heap_mem[heap_ptr[io] : heap_ptr[io]+io_buffer_size[io]] = byte_array
        weights += heap_mem     # append heap to weights
    else:   # previous allocation code
        buffer_ids_allocated = dict()

        #sort nodes so that input and outputs have their buffers
        #allocated first
        def node_sort(item):
            if item[0] in input_indices:
                return 0
            if item[0] in output_indices:
                return 1
            return 2

        sorted_nodes = sorted(list(enumerate(js_layers)),key = node_sort)

        for i, json_node in sorted_nodes:
            node = Nodes[i]
            input_id = json_node['input_id']
            output_id = json_node['output_id']

            if input_id not in buffer_ids_allocated:
                node.input_data = len(weights)
                buffer_ids_allocated[input_id] = (node.input_data, len(weights))
                size = io_buffer_size[input_id]
                data_type = node.input_data_type
                if data_type == calc_type.INT16:
                    size *= 2
                elif data_type == calc_type.INT32:
                    size *= 4
                if i in input_indices:
                    try:
                        float_array = base64_to_float_nparray(js['test_input'][input_indices.index(i)])
                        weights += float_to_fixed_np(float_array, node.input_data_type, ROUNDING=False).tobytes()
                    except:
                        weights += bytearray(size)
                else:
                    weights += bytearray(size)
            else:
                node.input_data = buffer_ids_allocated[input_id][0]

            if output_id not in buffer_ids_allocated:
                node.output_data = len(weights)
                buffer_ids_allocated[output_id] = (node.output_data, len(weights))
                size = io_buffer_size[output_id]
                data_type = node.output_data_type
                if data_type == calc_type.INT16:
                    size *= 2
                elif data_type == calc_type.INT32:
                    size *= 4
                if i in output_indices:
                    float_array = base64_to_float_nparray(js['test_output'][output_indices.index(i)])
                    scale_factor = io_info['output_scale_factors'][output_indices.index(i)]
                    weights += float_to_fixed_np(scale_factor*float_array, node.output_data_type).tobytes()
                else:
                    weights += bytearray(size)
            else:
                node.output_data = buffer_ids_allocated[output_id][0]

    # set output_scales
    for n in Nodes:
        n.output_scale_factor = 0.0
        n.test_input_data = n.input_data
        n.test_output_data = n.output_data


    graph.version = VNNX_GRAPH_VERSION
    graph.vbx_nn_preset = VBX_NN_PRESET

    for n, s in zip(output_indices, io_info['output_scale_factors']):
        fxp_scale_factor = 1 / (1 << Q8)
        if Nodes[n].output_data_type == calc_type.INT16:
            fxp_scale_factor = 1 / (1 << Q16)
        if Nodes[n].output_data_type == calc_type.INT32:
            fxp_scale_factor = 1 / (1 << Q32)
            s = 1.
        Nodes[n].output_scale_factor = s*fxp_scale_factor

    graph.num_layers = len(js_layers)
    graph.replay_buffer = len(weights)
    graph.replay_buffer_size = 1024
    weights += bytearray(graph.replay_buffer_size)

    while len(graph.description) < 24:
        graph.description += b"\0"

    # All buffers allocated, now adjust offsets,
    # and dump to bytearray
    node_size = Node().get_structure_size()
    subnode_size = Subnode().get_structure_size()
    num_subnodes = sum([len(n.subnode_array) for n in Nodes])
    # node_offset = graph.get_structure_size()
    subnode_offset = graph.get_structure_size() + node_size * len(Nodes)
    weights_offset = subnode_offset + num_subnodes*subnode_size
    all_subnode_array = []

    graph.update_offsets(weights_offset)
    for n in Nodes:
        n.update_offsets(weights_offset)

    for n in Nodes:
        all_subnode_array += n.subnode_array
        n.sublayers = subnode_offset
        n.num_sublayers = len(n.subnode_array)
        subnode_offset += len(n.subnode_array)*subnode_size

    for sn in all_subnode_array:
        sn.update_offsets(weights_offset)

    # do twice, once to find allocate_length, once for real
    graph.allocate_bytes = 0
    graph.data_bytes = 0
    graph_data = [graph.get_structured_data()]
    subnode_data = [sn.get_structured_data() for sn in all_subnode_array]
    node_data = [n.get_structured_data() for n in Nodes]

    data = b"".join(graph_data+node_data+subnode_data+[weights])
    graph.allocate_bytes = len(data)
    data = data.rstrip(b'\0')
    pad_zeros = (8 - (len(data) % 8))%8
    graph.data_bytes = len(data) + pad_zeros
    graph_data = [graph.get_structured_data()]
    data = b"".join(graph_data+node_data+subnode_data+[weights])

    if True:
        import vbx.sim
        data =  data[:graph.data_bytes]
        m = vbx.sim.Model(data)
        try:
            m.run(m.test_input)
        except:
            sys.stderr.write("ERROR: An unexpected error occurred, please contact Microchip for support\n")
            sys.exit(1)
        instr_count=vbx.sim.c.vbxsim_get_instructions()
        replay_size = instr_count*16
        adjustment = graph.replay_buffer_size - replay_size;

        graph.replay_buffer_size -= adjustment
        graph.allocate_bytes -=  adjustment
        graph_data = [graph.get_structured_data()]
        data = b"".join(graph_data+node_data+subnode_data+[weights])

        return data[:graph.data_bytes]
    else:
        data= data.rstrip(b'\0')
        return data + bytes(pad_zeros)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output','-o',help='Input file,defaults to stdout',required=True)
    parser.add_argument('--preset','-p',help='preset enum for parameters',
                        choices = ['V250','V500','V1000','V2000','V4000'])
    parser.add_argument('--cores',help='number of cores in system',default=1,type=int)
    parser.add_argument('json_file',help='Json file to parse, defaults to stdin')
    args = parser.parse_args()
    kwargs = dict()
    outdir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(outdir):
        sys.stderr.write("Output directory does not exist\n")
        sys.exit(1)
    if args.json_file is None:
        input_file = sys.stdin
    else:
        input_file = open(args.json_file)

    json_string = input_file.read()
    bin_output = json_to_graph(json_string, args.preset)
    with open(args.output, "wb") as output_file:
        output_file.write(bin_output)
