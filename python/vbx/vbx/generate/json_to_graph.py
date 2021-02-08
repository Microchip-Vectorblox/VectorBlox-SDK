import enum
import json
import itertools
import struct
import sys
import os.path
import base64
from math import ceil,log2
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
               ('int32_t', 'dma_channel_offset'),
               ('int32_t', 'dma_buffer_offset'),
               ('offset', 'input_data'),
               ('offset', 'output_data'),
               ('int8_t[24]', 'input_description'),
               ('int8_t[24]', 'output_description'),
               ('offset', 'test_input_data'),
               ('offset', 'test_output_data'),
               ('offset', 'sublayers'),
               ('float', 'output_scale_factor'),
               ('int32_t', 'num_sublayers'),
               ('int32_t', 'sublayer_stride'),
               ('int32_t', 'sublayer_shape'),
               ('int32_t', 'sublayer_shape_0'),
               ('int32_t', 'sublayer_shape_full'),
               ('int32_t', 'sublayer_shape_last'),
               ('int32_t', 'sublayer_rows'),
               ('int32_t', 'sublayer_columns'),
               # replay is an offset in vnnx-types.h,
               # but not here so it stays initialized to zero
               ('int32_t', 'use_replay'),
               ('int64_t', 'replay_buffer'),
               ('int32_t', 'replay_buffer_size'),
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
                          'transpose':[('int32_t[3]', 'permutation'),
                                       ('int32_t','out_maps_at_once')],
                          'resize':[('float[2]', 'scale'),
                                    ('int32_t', 'mode'),
                                    ('int32_t', 'channels'),
                                    ('int32_t', 'm'),
                                    ('int32_t', 'n'),
                                    ('int32_t', 'maps'),
                                    ('int32_t', 'rows')],
                          'reorg':[('int32_t', 'stride'),
                                    ('int32_t', 'channels'),
                                    ('int32_t', 'm'),
                                    ('int32_t', 'n'),
                                    ('int32_t', 'maps'),
                                    ('int32_t', 'rows')],
                          'softmax':[('offset', 'scale'),
                                     ('int32_t', 'size')]
                         })]

Subnode_struct = [('int32_t', 'type'),
                  ('int32_t', 'input_data_type'),
                  ('int32_t', 'output_data_type'),
                  ('int32_t[2]', 'strides'),  # 12
                  ('int32_t[2]', 'kernel_shape'),  # 20
                  ('int32_t[6]', 'pads'),  # 28
                  ('int32_t', 'maps'),  # 52
                  ('union', {"pad_const": [('float', 'value')],  # 56
                             "clip": [('float', 'min'),
                                      ('float', 'max')],
                             "prelu":[('offset', 'slope')],
                             "leakyrelu":[('int32_t', 'alpha')],
                             "mul_bc3":[("int32_t", "use_xl"),
                                        ("float", "scalarf32"),
                                        ('int32_t', 'scalar32'),
                                        ('int16_t', 'scalar16'),
                                        ('int8_t', 'scalar8'),
                                        ('uint8_t', 'scalaru8'),
                                        ('pad[1]', 'padding')],
                             "add_bc2": [('int32_t', 'use_xl'),
                                         ('offset', 'array'),
                                         ('offset', 'array_xl')],
                             "mul_bc2":[('int32_t', 'use_xl'),
                                        ('offset', 'array'),
                                        ('offset', 'array_xl')],
                             "cast":[('int32_t', 'scale')]}
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


class Subnode(Struct):
    def __init__(self):
        super().__init__(Subnode_struct)
        self.kernel_shape=[1,1]
        self.strides = [1,1]
        self.pads = [0,0,0,0,0,0]
        self.maps = 0

        #####
        ## TODO: I don't think these attributes are used
        self.input_data_type = 0
        self.output_data_type = 0


class resize_mode(enum.IntEnum):
    NEAREST = 0
    LINEAR = 1


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
    SOFTMAX = 6
    RESIZE = 7
    REORG = 8
    ARGMAX = 9
    UNKNOWN = 10

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
            return subgraph_type.SOFTMAX
        if e == "RESIZE":
            return subgraph_type.RESIZE
        if e == "REORG":
            return subgraph_type.REORG
        if e == "ARGMAX":
            return subgraph_type.ARGMAX
        return subgraph_type.UNKNOWN


class layer_type(enum.IntEnum):
    GLOBAL_AVGPOOL_I8 = 0
    GLOBAL_AVGPOOL_I16 = 1
    ABS_I8 = 2
    ABS_I16 = 3
    CLIP_I8 = 4
    CLIP_I16 = 5
    AVGPOOL_I8 = 6
    AVGPOOL_I16 = 7
    MAXPOOL_U8 = 8
    MAXPOOL_I8 = 9
    MAXPOOL_I16 = 10
    CAST_I16_I8 = 11
    CAST_I16_I32 = 12
    CAST_I32_I16 = 13
    CAST_U8_I16 = 14
    CAST_U8_I8 = 15
    CAST_I8_I16 = 16
    CAST_I8_I32 = 17
    LEAKYRELU_I8 = 18
    LEAKYRELU_I16 = 19
    RELU_I8 = 20
    RELU_I16 = 21
    PRELU_I8 = 22
    PRELU_I16 = 23
    PADCONST_U8 = 24
    PADCONST_I8 = 25
    PADCONST_I16 = 26
    MUL_BC3_I8 = 27
    MUL_BC3_I16 = 28
    MUL_BC3_U8 = 29
    MUL_BC3_U16 = 30
    MUL_BC2_I8 = 31
    MUL_BC2_I16 = 32
    ADD_I8 = 33
    ADD_I16 = 34
    LAYER_UNKNOWN = 35


def sublayer_bytes(l):
    if l in {layer_type.GLOBAL_AVGPOOL_I8 ,
             layer_type.ABS_I8 ,
             layer_type.CLIP_I8 ,
             layer_type.AVGPOOL_I8 ,
             layer_type.MAXPOOL_U8 ,
             layer_type.MAXPOOL_I8 ,
             layer_type.CAST_I16_I8 ,
             layer_type.CAST_U8_I8 ,
             layer_type.CAST_I8_I32 ,
             layer_type.LEAKYRELU_I8 ,
             layer_type.RELU_I8 ,
             layer_type.PRELU_I8 ,
             layer_type.PADCONST_U8 ,
             layer_type.PADCONST_I8 ,
             layer_type.MUL_BC3_I8 ,
             layer_type.MUL_BC3_U8 ,
             layer_type.MUL_BC2_I8 ,
             layer_type.ADD_I8 }:
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
             layer_type.MUL_BC3_I16 ,
             layer_type.MUL_BC3_U16 ,
             layer_type.MUL_BC2_I16 ,
             layer_type.ADD_I16 }:
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
                   (subgraph_type.SOFTMAX, "softmax"),
                   (subgraph_type.RESIZE, "resize"),
                   (subgraph_type.REORG, "reorg"),
                   (subgraph_type.ARGMAX, "argmax"),
                   (subgraph_type.UNKNOWN, "unknown"),
                   (layer_type.GLOBAL_AVGPOOL_I8, ""),
                   (layer_type.GLOBAL_AVGPOOL_I16, ""),
                   (layer_type.ABS_I8, ""),
                   (layer_type.ABS_I16, ""),
                   (layer_type.CLIP_I8, "clip"),
                   (layer_type.CLIP_I16, "clip"),
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
                   (layer_type.LEAKYRELU_I8, "leakyrelu"),
                   (layer_type.LEAKYRELU_I16, "leakyrelu"),
                   (layer_type.RELU_I8, ""),
                   (layer_type.RELU_I16, ""),
                   (layer_type.PRELU_I8, "prelu"),
                   (layer_type.PRELU_I16, "prelu"),
                   (layer_type.PADCONST_U8, "pad_const"),
                   (layer_type.PADCONST_I8, "pad_const"),
                   (layer_type.PADCONST_I16, "pad_const"),
                   (layer_type.MUL_BC3_I8, "mul_bc3"),
                   (layer_type.MUL_BC3_I16, "mul_bc3"),
                   (layer_type.MUL_BC3_U8, "mul_bc3"),
                   (layer_type.MUL_BC3_U16, "mul_bc3"),
                   (layer_type.MUL_BC2_I8, "mul_bc2"),
                   (layer_type.MUL_BC2_I16, "mul_bc2"),
                   (layer_type.ADD_I8, "add_bc2"),
                   (layer_type.ADD_I16, "add_bc2"),
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
        subgraph_type.SOFTMAX: 1,
        subgraph_type.RESIZE: 0,
        subgraph_type.REORG: 0,
        subgraph_type.ARGMAX: 0,
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


def set_sublayer_attributes(sg, sl_array):
    shape = 0
    shape_0 = 0
    shape_last = 0
    shape_full = 0

    max_rows = 0
    max_columns = 0
    rows = 0
    columns = 0
    stride = 1

    # shape states how many additional inputs are required
    # pad at end does not change this
    # pad before activation doesn't
    for sl in sl_array:
        shape += (sl.kernel_shape[0] - 1)*stride
        shape_0 += (sl.kernel_shape[0] - 1 - sl.pads[1])*stride
        shape_last += (sl.kernel_shape[0] - 1 - sl.pads[4])*stride
        shape_full += (sl.kernel_shape[0] - 1 -
                       (sl.pads[1] + sl.pads[4]))*stride

        stride *= sl.strides[0]

        columns += ((sl.pads[2] + sl.pads[5] - (sl.kernel_shape[1]-1))
                    // sl.strides[1])
        if (columns > max_columns):
            max_columns = columns
        rows += ((sl.pads[1] + sl.pads[4] - (sl.kernel_shape[0]-1)) //
                 sl.strides[0])
        if (rows > max_rows):
            max_rows = rows

    sg.sublayer_shape = shape
    sg.sublayer_shape_full = shape_full
    sg.sublayer_shape_last = shape_last
    sg.sublayer_shape_0 = shape_0
    sg.sublayer_stride = stride
    sg.sublayer_columns = max_columns
    sg.sublayer_rows = max_rows


def elements_to_process(m, n, shape, dilations):
    start_output_element = (
        (shape[0] // 2) * dilations[0] * n) + (shape[1] // 2 * dilations[1])
    end_output_element = (m * n) - ((((shape[0] - 1) // 2) * dilations[0] * n) +
                                    ((shape[1] - 1) // 2) * dilations[1])
    return end_output_element - start_output_element


def strided_elements_to_process(m, n, shape, dilations, strides):
    m0 = (m + (strides[0]-1)) // strides[0]
    n0 = (n + (strides[1]-1)) // strides[1]
    start_output_element = ((shape[0] // 2 // strides[0]) * dilations[0] * n0) + (shape[1] // 2 // strides[1] * dilations[1])
    end_output_element = (m0*n0) - ((((shape[0]-1) // 2 // strides[0]) * dilations[0] * n0) + ((shape[1]-1) // 2 // strides[1]) * dilations[1])
    return end_output_element-start_output_element


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


def conv_populate_attributes(node, json_node, conv_cvi, sp_size, vector_lanes, filter_copies, parallel_output_maps):
    conv = node.conv
    conv.use_cvi = conv_cvi
    conv.rows = conv.m
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
    conv_output_bytes = 1
    if node.input_data_type == calc_type.INT16 or conv_cvi and INPUT_SIZE_BYTES == 2:
        conv_input_bytes = 2
    if node.output_data_type == calc_type.INT16 or conv_cvi and OUTPUT_SIZE_BYTES == 2:
        conv_output_bytes = 2

    def vci_conv_rows(conv, imaps, maps):
        if conv.use_depthwise:
            channels = imaps
            coeff_stride = ceil(COEFFICIENT_BITS*maps/8)
            remaining = sp_size
            remaining -= aligned_size((conv.kernel_shape[0]*conv.kernel_shape[1]+5)*coeff_stride, vector_lanes)  # v_w0
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


        size_per_row = (aligned_size(conv_output_bytes*maps*(n+n_extra), vector_lanes) +
                        aligned_size(conv_input_bytes*channels*n, vector_lanes))
        rows = (remaining // size_per_row) - m_extra

        if conv.use_strided and rows % conv.strides[0]:
            rows -= 1
        return rows


    def vci_post_rows(maps):
        return sp_size//(2*conv_output_bytes*maps*(n+n_extra)) - m_extra

    conv.maps = -1
    conv.imaps = -1
    conv.rows = -1
    node.scratchpad_bytes = -1
    min_rows = (1+(conv.kernel_shape[0]-1)*conv.dilations[0] +
                node.sublayer_shape*conv.strides[0])
    inc = conv.strides[0] * node.sublayer_stride
    offset = (min_rows - inc)


    if not conv_cvi:
        conv.core_m = conv.m
        conv.core_split = 1
        conv.core_maps = ceil(conv.kernels / CORES)

        # calculte the max maps
        max_maps = -1
        coeff_width = ceil(kernel_width*COEFFICIENT_BITS*4 /8)
        m0, n0 = (m-(kernel_height-1), n-(kernel_width-1))
        if conv_input_bytes == 1 and conv_cvi:
            max_maps = (sp_size-8*m*n) / (2*m0*n0 + 2*coeff_width*kernel_height + 2*m*n0)
            max_maps = round_down_pwr_2(max_maps)
        elif conv_input_bytes == 2 and conv_cvi:
            max_maps = sp_size / (2*m0*n0 + 2*(coeff_width*kernel_height+m*n0) + 16*m*n)
        elif conv_input_bytes == 1 and not conv_cvi:
            max_maps = (sp_size - (2*m*n + 2*m0*n0))/(2*2*m*n)
        elif conv_input_bytes == 2 and not conv_cvi:
            max_maps = (sp_size - (4*m*n + 2*m0*n0))/(2*2*m*n)
        elif conv_input_bytes == 4 and not conv_cvi:
            max_maps = (sp_size - (8*m*n+4*m0*n0)) / (4*m0*n0)

        if conv.group > 1 and max_maps > (conv.kernels/conv.group):
            # adjust max_maps to cap at 1 filter group at a time
            if conv.group == conv.kernels and conv.group == conv.channels:
                pass  # depthwise
            else:
                max_maps = conv.kernels/conv.group

        if conv.m == 1 and conv.n == 1:
            conv.imaps = int((sp_size - (2*(conv.channels+conv.kernels))) / (2*2*conv.channels))
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
            while (conv.rows - offset) % inc:
                conv.rows -= 1
            assert conv.rows >= min_rows

        node.scratchpad_bytes = conv.maps*(conv.rows+m_extra)*(conv.n+n_extra)*2
    else:
        # conv_cvi
        max_elements = filter_copies*ACCUMULATORS
        elems_to_process = elements_to_process(m, n, conv.kernel_shape, conv.dilations)
        if conv.use_strided:
            elems_to_process = strided_elements_to_process(m, n, conv.kernel_shape, conv.dilations, conv.strides)
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
            while conv.core_m % inc:
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
            # buffering with latge numbers of output maps
            if conv.imaps > 2*parallel_output_maps and conv.imaps > conv.kernels:
                conv.imaps = parallel_output_maps

        def rows_check():
            # TODO name this better
            return (not (conv_rows >= rows) or not (post_rows >= rows))

        conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps)
        post_rows = vci_post_rows(conv.maps)
        if conv.imaps < conv.channels/conv.group:
            if rows_check() and conv.imaps > 0:
                while rows_check() and conv.maps > 1:
                    conv.maps = ceil(conv.maps/2)
                    conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps)
                    post_rows = vci_post_rows(conv.maps)

        if rows_check() and conv.imaps > 0:
            conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps)
            post_rows = vci_post_rows(conv.maps)

            while conv_rows < rows and (conv.use_depthwise or conv.imaps > 2*parallel_output_maps):
                conv.imaps = ceil(conv.imaps/2)
                if conv.use_depthwise:
                    conv.maps = conv.imaps
                conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps)
                post_rows = vci_post_rows(conv.maps)

            while rows_check() and conv.maps > 1:
                conv.maps = ceil(conv.maps/2)
                if conv.use_depthwise:
                    conv.imaps = conv.maps
                conv_rows = vci_conv_rows(conv, conv.imaps, conv.maps)
                post_rows = vci_post_rows(conv.maps)
        assert conv_rows >= rows
        assert post_rows >= rows
        assert conv.imaps != 0

        if not vci_fit_full_maps and rows != conv.m:
            all_imaps = (conv.imaps == conv.channels//conv.group) or conv.use_depthwise
            elems_to_process = elements_to_process(rows + inc, n, conv.kernel_shape, conv.dilations)
            if conv.use_strided:
                elems_to_process = strided_elements_to_process(rows + inc, n, conv.kernel_shape, conv.dilations, conv.strides)
            inc_elements = int(all_imaps or max_elements >= elems_to_process)

            while conv_rows >= (rows+inc) and post_rows >= (rows+inc) and inc_elements:
                rows += inc
                elems_to_process = elements_to_process(rows + inc, n, conv.kernel_shape, conv.dilations)
                if conv.use_strided:
                    elems_to_process = strided_elements_to_process(rows + inc, n, conv.kernel_shape, conv.dilations, conv.strides)
                inc_elements = int(all_imaps or (max_elements > elems_to_process))
                if rows > conv.core_m + offset:
                    break
            if rows > conv.core_m + offset:
                rows = conv.core_m + offset
            while (rows - offset) % inc:
                rows-= 1
        conv.rows = rows
        node.scratchpad_bytes = conv_output_bytes*conv.maps*(conv.rows+m_extra)*(conv.n+n_extra)

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


def max_ident_maps(m, n, sp_size, bytes):
    return sp_size // (m*n*2*bytes)


def max_gemm_columns(gemm, sp_size, bytes):
    max_input_size = gemm.input_size
    r = (sp_size - (max_input_size * 3 * bytes)) // (2*bytes)
    while r < 1:
        max_input_size = (max_input_size + 1) // 2
        r = (sp_size - (max_input_size*3*bytes)) // (2*bytes)
    return max_input_size


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
        if(node.type == subgraph_type.CONV and not conv_cvi and INPUT_SIZE_BYTES == 1):
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
                    # print('u8 to i16')
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
                    else:
                        print('error for averagepool')

            elif op_type == 'Abs':
                sn.type = layer_type.ABS_I16
                if current_data_type == calc_type.INT8:
                    sn.type = layer_type.ABS_I8

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
                assert dims == 1
                sn.add_bc2.use_xl = 0
                sn.maps = len(json_sl['array'])
                if current_data_type == calc_type.INT8:
                    sn.type = layer_type.ADD_I8
                    sn.add_bc2.array = len(weights)
                    weights += float_array_to_weights(arr, calc_type.INT8)
                    sn.add_bc2.array_xl = len(weights)
                    weights += float_array_to_weights(arr, calc_type.INT16)
                    max_possible_weight = 1.0 * (1 << ((8-1) - Q8))
                if current_data_type == calc_type.INT16:
                    sn.type = layer_type.ADD_I16
                    sn.add_bc2.array = len(weights)
                    weights += float_array_to_weights(arr, calc_type.INT16)
                    sn.add_bc2.array_xl = len(weights)
                    weights += float_array_to_weights(arr, calc_type.INT32)
                    max_possible_weight = 1.0 * (1 << ((16-1) - Q16))
                if max(arr) > max_possible_weight:
                    sn.add_bc2.use_xl = 1

            elif op_type == 'Mul':
                dims = json_sl['dims']
                assert dims < 2
                if dims == 1:
                    sn.type = layer_type.MUL_BC2_I16
                    sn.maps = len(json_sl['array'])
                    sn.mul_bc2.use_xl = 0
                    sn.mul_bc2.array=len(weights)
                    weights += float_array_to_weights(json_sl['array'], calc_type.INT16)
                    sn.mul_bc2.array_xl = len(weights)
                    weights += float_array_to_weights(json_sl['array'], calc_type.INT32)
                    max_possible_weight = 1.0 * (1 << ((16-1) - Q16))
                    if (current_data_type == calc_type.INT8):
                        sn.type = layer_type.MUL_BC2_I8
                        max_possible_weight = 1.0 * (1 << ((8-1) - Q8))
                    if max(json_sl['array']) > max_possible_weight:
                        sn.mul_bc2.use_xl = 1
                if dims == 0:
                    sn.mul_bc3.use_xl = 0
                    sn.mul_bc3.scalarf32 = json_sl['array'][0]
                    sn.mul_bc3.scalaru8 = float_to_fixed(json_sl['array'][0], calc_type.UINT8)
                    sn.mul_bc3.scalar8 = float_to_fixed(json_sl['array'][0], calc_type.INT8)
                    sn.mul_bc3.scalar16 = float_to_fixed(json_sl['array'][0], calc_type.INT16)
                    sn.mul_bc3.scalar32 = float_to_fixed(json_sl['array'][0], calc_type.INT32)
                    if current_data_type == calc_type.UINT8:
                        assert(np.max(np.asarray(json_sl['array'][0])) > 0)
                        sn.type = layer_type.MUL_BC3_U8
                        max_possible_weight = 1.0 * (1 << ((8) - U8))
                    elif current_data_type == calc_type.INT8:
                        sn.type = layer_type.MUL_BC3_I8
                        max_possible_weight = 1.0 * (1 << ((8-1) - Q8))
                    elif current_data_type == calc_type.INT16:
                        sn.type = layer_type.MUL_BC3_I16
                        max_possible_weight = 1.0 * (1 << ((8-1) - Q8))

                    if max(json_sl['array']) > max_possible_weight:
                        sn.mul_bc2.use_xl = 1
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
                else:
                    print(current_data_type, node.output_data_type)
                    assert False, "BAD TYPES"
            else:
                print(current_data_type, node.output_data_type)
                assert False, "BAD TYPES"

        node.input_size = json_node['input_size']
        node.output_size = json_node['output_size']
        node.output_shape = list(json_node['output_shape'])
        while len(node.output_shape)<3:
            node.output_shape=[1]+node.output_shape
        node.dma_channel_offset = json_node['dma_offset']
        node.dma_buffer_offset = json_node['buffer_offset']*sizeof_calc_type(node.output_data_type)
        node.use_replay = json_node['use_replay']

        set_sublayer_attributes(node, node.subnode_array)

        if(node.type == subgraph_type.CONV):

            attrs = ('kernels', 'channels', 'group', 'm', 'n',
                     'kernel_shape', 'strides', 'dilations',
                     'use_cvi', 'use_strided', 'use_depthwise')
            for a in attrs:
                setattr(node.conv, a, json_node[a])

            conv_populate_attributes(node, json_node, conv_cvi, sp_size, VECTOR_LANES, FILTER_COPIES, PARALLEL_OUTPUT_MAPS)

            conv = node.conv
            # COEFFICIENTS
            num_padded_weights = (
                conv.kernel_shape[0]*conv.kernel_shape[1] *
                (conv.channels//conv.group) * conv.padded_kernels)

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
                if node.input_unsigned:
                    conv.bias_scalar = (1 << U8) - 1
                    conv.bias_lower_scalar = 1
                else:
                    conv.bias_scalar = -((1 << Q8) - 1)
                    conv.bias_lower_scalar = -1
            else:
                conv.bias_scalar = -(1 << Q16)
                conv.bias_lower_scalar = 0

            pow2 = False
            QWEIGHT = 7
            QSCALE = 7
            QEXTRA = QWEIGHT + QSCALE - ACCUMULATOR_FXP_BITS - SCALE_FXP_BITS

            if node.input_unsigned and not node.output_unsigned:
                QEXTRA += 1
            if node.output_unsigned and not node.input_unsigned:
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
                if NEXT:
                    qin = 7
                    qout = 7
                    if node.input_unsigned:
                        qin = 8
                    if node.output_unsigned:
                        qout = 8
                    # mweight = 127./128*max_weight
                    # mweight = 128./127*max_weight
                    mweight = max_weight
                    qdiff = ACCUMULATOR_FXP_BITS + SCALE_FXP_BITS + qout - qin
                    qfactor = log2(mweight) + qweight
                    qscale = qdiff + log2(mweight) - qweight

                    if VERBOSE:
                        if qscale > q_max:
                            q_max = qscale
                        if qscale < q_min:
                            q_min = qscale

                    if qscale < 0:
                        if VERBOSE:
                            print('small', node_n, k, qscale, max_weight, node.input_unsigned, node.output_unsigned)
                            weights_small = True
                        qscale = 0
                    if qscale > 8:
                        if VERBOSE:
                            print('big', node_n, k, qscale, max_weight, node.input_unsigned, node.output_unsigned)
                            weights_big = True
                        qscale = 8
                    if conv_cvi:
                        scale_i16[k] = int(ceil(2**qscale)) - 1
                        reqscale = log2(scale_i16[k]+1)
                        scale_f32[k] = 2**(reqscale+qweight-qdiff)
                else:
                    if conv_cvi:
                        if max_weight != 0.0:
                            shift_i16 = max(QEXTRA, int(ceil(log2(max_weight * (1 << QSCALE)))))
                            factor_i16 = max(QEXTRA, log2(max_weight * ((1 << QSCALE) + (1 << (QSCALE-QEXTRA-1)))))
                        else:
                            shift_i16 = QEXTRA
                            factor_i16 = QEXTRA
                        if pow2:
                            scale_i16[k] = (1 << (shift_i16 - QEXTRA))-1
                        else:
                            scale_i16[k] = int(ceil(2**(factor_i16 - QEXTRA)))-1
                        scale_f32[k] = 1.0 * (scale_i16[k]+1) / (1 << (QSCALE-QEXTRA))

                if conv_cvi:
                    if scale_f32[k] == 0.:
                        k_weights_i16 = [0 for w in k_weights]
                        weights_i16[k*weights_per_kernel:(k+1)*weights_per_kernel] = k_weights_i16
                        biases_i16[k] = 0
                    else:
                        k_weights_i16 = float_to_fixed_np(k_weights/scale_f32[k], calc_type.INT16, COEFFICIENT_BITS-1, qweight)
                        weights_i16[k*weights_per_kernel:(k+1)*weights_per_kernel] = k_weights_i16

                        if node.input_unsigned:
                            biases_i16[k] = float_to_fixed(biases_f32[k]/scale_f32[k], calc_type.INT16, COEFFICIENT_BITS-1, qweight, ROUNDING=USE_ROUNDING)
                        else:
                            biases_i16[k] = float_to_fixed(-biases_f32[k]/scale_f32[k], calc_type.INT16, COEFFICIENT_BITS-1, qweight, ROUNDING=USE_ROUNDING)

                        if INPUT_SIZE_BYTES == 1 and use_bias_lower:
                            if node.input_unsigned:
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
                    print('big', node_n, node.input_unsigned, node.output_unsigned)
                if weights_small:
                    print('small', node_n, node.unsigned_input, node.unsigned_output)

            type_size = 2 if INPUT_SIZE_BYTES == 1 else 1
            if conv_cvi:
                padded_scale = pad_weights16(scale_i16, conv.kernels, 1, 1, 1, 1, conv.padded_kernels, 1)
                packed_scales = np.zeros(conv.padded_kernels*type_size, dtype=np.uint8)
                for k in range(0, conv.padded_kernels, conv.maps):
                    iscales = interleave_weights(padded_scale[k:k+conv.maps],
                                                 1, conv.maps, 1, 1)
                    packed_scales[k*type_size:(k*type_size)+conv.maps] = iscales
                conv.scale = len(weights)
                weights += packed_scales.tobytes()

            else:
                conv.scale = len(weights)
                padded_scales = scale_i16 + [0]*(conv.padded_kernels-conv.kernels)
                weights += struct.pack("{}H".format(conv.padded_kernels),
                                       *padded_scales)

            if conv_cvi:
                padded_weights = pad_weights16(weights_i16.tolist(), conv.kernels,
                                               conv.channels//conv.group,
                                               conv.group,
                                               conv.kernel_shape[0],
                                               conv.kernel_shape[1],
                                               conv.padded_kernels,
                                               conv.channels//conv.group)
                assert num_padded_weights == len(padded_weights)

                packed_weights = np.zeros(num_padded_weights*type_size, dtype=np.uint8)


                offset = ((conv.channels//conv.group) *
                          conv.kernel_shape[0]*conv.kernel_shape[1])

                for k in range(0, conv.padded_kernels, conv.maps):
                    pack_weights(padded_weights[k*offset:], packed_weights[k*type_size*offset:], conv.kernel_shape[0], conv.kernel_shape[1], conv.channels//conv.group, conv.maps, 1, conv.maps)

                if conv.use_depthwise:
                    packed_weights = np.zeros(num_padded_weights*type_size + conv.padded_kernels*type_size*5, dtype=np.uint8)
                    depthwise_pack_weights(padded_weights, packed_weights, scale_i16, conv.bias_scalar, biases_i16, conv.bias_lower_scalar, biases_lower_i16, conv.kernel_shape[0], conv.kernel_shape[1], conv.channels, conv.padded_kernels, conv.group)

                conv.weights = len(weights)
                weights += packed_weights.tobytes()

            else:
                conv.acc_maps = 0

                conv.weights = len(weights)
                fmt = "{}h".format(num_padded_weights)
                weights += struct.pack(fmt,
                                       *weights_i16)
                conv.weights32 = len(weights)
                fmt = "{}i".format(num_padded_weights)
                weights += struct.pack(fmt,
                                       *weights_i32)
            if not conv_cvi:
                conv.use_weights32 = 1 if layer_max_weight > 1.0 * (1 << ((16-1)-Q16)) else 0

            packed_biases = []
            packed_biases_lower = []

            if conv_cvi:
                padded_biases = pad_weights16(biases_i16,
                                              conv.kernels, 1, 1, 1, 1,
                                              conv.padded_kernels, 1)
                packed_biases = np.zeros(conv.padded_kernels*type_size, dtype=np.uint8)
                offset = 1
                for k in range(0, conv.padded_kernels, conv.maps):
                    interleaved_biases = interleave_weights(padded_biases[k:k+conv.maps],
                                                            1, conv.maps,
                                                            1, 1)
                    packed_biases[k*type_size:k*type_size + conv.maps] = interleaved_biases
                conv.biases = len(weights)
                weights += packed_biases.tobytes()

                padded_biases_lower = pad_weights16(biases_lower_i16,
                                                    conv.kernels, 1, 1, 1, 1,
                                                    conv.padded_kernels, 1)
                packed_biases_lower = np.zeros(conv.padded_kernels*type_size, dtype=np.uint8)
                offset = 1
                for k in range(0, conv.padded_kernels, conv.maps):
                    interleaved_biases_lower = interleave_weights(padded_biases_lower[k:k+conv.maps],
                                                                  1, conv.maps,
                                                                  1, 1)
                    packed_biases_lower[k*type_size:k*type_size + conv.maps] = interleaved_biases_lower

                conv.biases_lower = len(weights)
                weights += packed_biases_lower.tobytes()
            else:
                conv.biases = len(weights)
                weights += struct.pack("{}h".format(len(biases_i16)),
                                       *biases_i16)
                conv.biases_lower = len(weights)
                weights += struct.pack("{}h".format(len(biases_i16)),
                                       *biases_i16)

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

            max_maps = max_ident_maps(identity.m + node.sublayer_rows, identity.n + node.sublayer_columns, sp_size, identity_bytes)
            if max_maps > 1:
                identity.maps = identity.core_maps
                while identity.maps > max_maps:
                    identity.maps //= 2
            else:
                identity.maps = 1

            identity.rows = max_ident_maps(identity.maps, identity.n + node.sublayer_columns, sp_size, identity_bytes)-node.sublayer_rows

            if identity.rows > identity.m:
                identity.rows = identity.m
            if identity.rows != identity.m:
                min_rows = 1 + node.sublayer_shape
                inc = node.sublayer_stride
                while (identity.rows - min_rows) % inc:
                    identity.rows -= 1
                assert identity.rows >= min_rows

            node.scratchpad_bytes = ((identity.rows + node.sublayer_rows) *
                                     (identity.n + node.sublayer_columns) *
                                     identity.maps * identity_bytes)

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
        elif node.type == subgraph_type.SOFTMAX:
            float_array = json_node['scale']
            node.softmax.scale = len(weights)
            node.softmax.size = len(float_array)
            weights += struct.pack("f"*len(float_array), *float_array)
            node.scratchpad_bytes = 0
        elif node.type == subgraph_type.TRANSPOSE:
            calc_bytes = max(sizeof_calc_type(node.input_data_type),
                                  sizeof_calc_type(node.output_data_type),
                                  *[sublayer_bytes(sn.type) for sn in node.subnode_array])
            map_size = node.output_shape[1]*node.output_shape[2]
            maps_at_once= sp_size//(2*map_size*calc_bytes)
            if maps_at_once >node.output_shape[0]:
                maps_at_once = node.output_shape[0]

            node.transpose.permutation=json_node['permutation']
            node.transpose.out_maps_at_once=maps_at_once;
            node.scratchpad_bytes = maps_at_once*map_size*calc_bytes;
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


    # setup graph object
    graph = Graph()
    graph.num_inputs = len(input_indices)
    graph.num_outputs = len(output_indices)

    io_nodes = np.array(input_indices + output_indices, dtype=np.int32)
    graph.io_nodes = len(weights)
    weights += io_nodes.tobytes()

    io_buffer_size = [sz for sz in js['buffers']]
    
    REUSE_IO_MEMORY = all_corrections is None
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
                if ind==dealloc_ind:    # find all deallocations during this layer
                    for h in range(1,len(heap)):
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

    # print('l, sl:', len(Nodes), len(all_subnode_array))

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
        except Exception:
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
