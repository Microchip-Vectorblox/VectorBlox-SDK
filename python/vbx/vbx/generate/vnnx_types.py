import enum
import itertools
import struct
import sys
import os.path
import numpy as np
import hashlib
import numpy as np


Q32 = 27
Q16 = 7
Q8 = 7
U8 = 7
ACCUMULATORS = 64
COEFFICIENT_BITS = 8
INPUT_SIZE_BYTES = 1
OUTPUT_SIZE_BYTES = 1
DOTP_MODE = 1
MULTIPLIER_BITS = 18
ACCUMULATOR_FXP_BITS = 3
SCALE_FXP_BITS = 8
CORES = 1
SHAPE_DIMS = 6
DESCRIPTION_CHARS = 48

preset_select= {"VECTOR_LANES": { "V250" : 2,
                                  "V500"       : 4,
                                  "V1000"      : 8,
                                  "V2000"       : 8,
                                  "V4000" : 8},
                "FILTER_COPIES": { "V250" : 16,
                                   "V500"       : 16,
                                   "V1000"      :  32,
                                   "V2000"       : 32,
                                   "V4000" : 32},
                "PARALLEL_OUTPUT_MAPS": { "V250" : 8,
                                          "V500"       : 16,
                                          "V1000"      : 16,
                                          "V2000"       : 16,
                                          "V4000" : 32},
                "SCRATCHPAD_KB": { "V250" : 64,
                                   "V500"       : 128,
                                   "V1000"      : 256,
                                   "V2000"       : 256,
                                   "V4000" : 256},
                "PRESET" : { "V250" : 0 ,
                             "V500"       : 1,
                             "V1000"      : 2,
                             "V2000"       : 3,
                             "V4000" : 4},
}

Graph_struct = [('uint32_t', 'version'),
                ('int32_t', 'vbx_nn_preset'),
                ('int32_t', 'num_inputs'),
                ('int32_t', 'num_outputs'),
                ('uint32_t', 'fixed_replay_buffer0'),
                ('uint32_t', 'fixed_replay_buffer1'),
                ('uint32_t', 'fixed_replay_buffer2'),
                ('uint32_t', 'fixed_replay_buffer3'),
                ('int32_t', 'include_io_data'),
                ('uint32_t', 'data_bytes'),
                ('uint32_t', 'allocate_bytes'),
                ('offset', 'io_nodes'),
                ('offset', 'io_offsets'),
                ('int32_t', 'num_layers'),
                ('offset', 'replay_buffer'),
                ('int32_t', 'replay_buffer_size'),
                ('int32_t', 'magic'),
                ]

Tensor_struct = [('int32_t', 'type'),
               ('int32_t[{}]'.format(SHAPE_DIMS), 'shape'),
               ('int32_t', 'dims'),
               ('int32_t', 'external_producer'),
               ('int32_t', 'external_consumer'),
               ('float', 'scale'),
               ('int32_t', 'scale_f16'),
               ('int32_t', 'zero'),
               ('int32_t', 'multiplier'),
               ('int32_t', 'shift'),
               ('int32_t[2]', 'buffer'),
               ('offset', 'direct'),
                ]

Node_struct = [('int32_t', 'type'),
               ('int32_t', 'input_data_type'),
               ('int32_t', 'output_data_type'),
               ('int32_t', 'offloaded'),
               ('int32_t[2]', 'input_strides'),
               ('int32_t[2]', 'output_strides'),
               ('int32_t', 'channels'),
               ('int32_t', 'm'),
               ('int32_t', 'n'),
               ('int32_t', 'maps'),
               ('int32_t', 'rows'),
               ('int32_t', 'cols'),
               ('int32_t', 'skip'),
               ('int32_t', 'scratchpad_bytes'),
               ('int8_t[{}]'.format(DESCRIPTION_CHARS), 'input_description'),
               ('int8_t[{}]'.format(DESCRIPTION_CHARS), 'output_description'),
               ('offset', 'sublayers'),
               ('int32_t', 'num_sublayers'),
               ('int32_t', 'row_start'),
               ('int32_t', 'row_last'),
               ('int32_t', 'row_inc'),
               ('int32_t', 'row_inc0'),
               ('int32_t', 'rows_0'),
               ('int32_t', 'rows_final'),
               ('int32_t', 'col_start'),
               ('int32_t', 'col_last'),
               ('int32_t', 'col_inc'),
               ('int32_t', 'col_inc0'),
               ('int32_t', 'cols_0'),
               ('int32_t', 'cols_final'),
               ('int32_t', 'prefetch_bytes_per_map'),
               # replay is an offset in vnnx-types.h,
               # but not here so it stays initialized to zero
               ('int32_t', 'use_replay'),
               ('int64_t', 'replay_buffer'),
               ('int32_t', 'replay_buffer_size'),
               ('int32_t', 'num_inputs'),
               ('int32_t', 'num_outputs'),
               ('int32_t', 'num_tensors'),
               ('offset', 'tensors'),
               ('int32_t', 'activation_min'),
               ('int32_t', 'activation_max'),
               ('offset', 'input_multiplier'),
               ('offset', 'input_shift'),
               ('int32_t', 'input_offset'),
               ('offset', 'output_multiplier'),
               ('offset', 'output_shift'),
               ('int32_t', 'output_offset'),
               ('union', {'Conv2DOptions': [('int32_t', 'kernels'),
                                    ('int32_t', 'stride_width'),
                                    ('int32_t', 'stride_height'),
                                    ('int32_t', 'dilation_width_factor'),
                                    ('int32_t', 'dilation_height_factor'),
                                    ('int32_t', 'padding_width'),
                                    ('int32_t', 'padding_height'),
                                    ('int32_t[4]', 'filter_shape_dims'),
                                    ('int32_t', 'group'),
                                    ('int32_t', 'imaps'),
                                    ('int32_t', 'conv_rows'),
                                    ('int32_t', 'use_vector'),
                                    ('int32_t', 'use_fia'),
                                    ('int32_t', 'use_db'),
                                    ('int32_t', 'use_depthwise'),
                                    ('int32_t', 'use_strided'),
                                    ('int32_t', 'fit_weights'),
                                    ('int32_t', 'split_weight_shaper_buffers'),
                                    ('int32_t', 'direct_dma'),
                                    ('int32_t', 'mxp_double_buffer'),
                                    ('int32_t', 'first_fia'),
                                    ('int32_t', 'last_fia'),
                                    ('int32_t', 'fia_collision'),
                                    ('offset', 'filter_data'),
                                    ('offset', 'bias_data'),
                                    ('offset', 'quantization_records')],
                             "eltwise8": [('offset', 'input2_multiplier'),
                                          ('offset', 'input2_shift'),
                                          ('int32_t', 'input2_offset'),
                                          ('offset', 'bias_data'),
                                          ('int32_t', 'swap'),
                                          ('int32_t', 'optimized'),
                                          ('int32_t', 'isize'),
                                          ('int32_t', 'left_shift'),
                                          ('int32_t', 'type')],
                            'FullyConnectedOptions': [('int32_t[2]', 'filter_shape_dims'),
                                    ('int32_t', 'input_stride'),
                                    ('int32_t', 'use_fia'),
                                    ('int32_t', 'first_fia'),
                                    ('int32_t', 'last_fia'),
                                    ('int32_t', 'fia_collision'),
                                    ('int32_t', 'mxp_double_buffer'),
                                    ('offset', 'filter_data'),
                                    ('offset', 'bias_data'),
                                    ('offset', 'quantization_records'),],
                          'ConcatOptions':[('int32_t', 'axis')],
                          "PackOptions":[('int32_t', 'axis'),
                                       ('int32_t', 'count'),
                                       ('int32_t', 'dims')],
                          'argmax':[('int32_t', 'pixels_per_loop')],
                          'lrn':[('float', 'alpha'),
                                 ('float', 'beta'),
                                 ('float', 'bias'),
                                 ('float', 'scale'),
                                 ('int32_t', 'size')],
                          'TileOptions':[('int32_t[4]', 'tile')],
                          'reduce':[('int32_t', 'm0')],
                          'reorg':[('int32_t', 'stride')],
                          'ResizeOptions':[('float[2]', 'scale'),
                                    ('int32_t', 'mode')],
                          'TransposeOptions':[('int32_t[3]', 'permutation'),
                                       ('int32_t','out_maps_at_once'),
                                       ('int32_t','out_rows_at_once')],
                          'SplitOptions':[('int32_t', 'axis'),
                                     ('offset', 'splits')]
                         })]

Subnode_struct = [('int32_t', 'type'),
                  ('int32_t', 'input_data_type'),
                  ('int32_t', 'output_data_type'),
                  ('int32_t[2]', 'strides'), 
                  ('int32_t[2]', 'kernel_shape'),
                  ('int32_t[2]', 'dilations'),
                  ('int32_t[6]', 'pads'),  
                  ('int32_t', 'maps'), 
                  ('int32_t', 'num_inputs'),
                  ('int32_t', 'num_outputs'),
                  ('int32_t', 'num_tensors'),
                  ('int32_t', 'activation_min'),
                  ('int32_t', 'activation_max'),
                  ('offset', 'input_multiplier'),
                  ('offset', 'input_shift'),
                  ('int32_t', 'input_offset'),
                  ('offset', 'output_multiplier'),
                  ('offset', 'output_shift'),
                  ('int32_t', 'output_offset'),
                  ('int32_t', 'nop'),
                  ('offset', 'tensors'),
                  ('union', {'Conv2DOptions': [('int32_t', 'kernels'),
                                    ('int32_t', 'stride_width'),
                                    ('int32_t', 'stride_height'),
                                    ('int32_t', 'dilation_width_factor'),
                                    ('int32_t', 'dilation_height_factor'),
                                    ('int32_t', 'padding_width'),
                                    ('int32_t', 'padding_height'),
                                    ('int32_t[4]', 'filter_shape_dims'),
                                    ('int32_t', 'group'),
                                    ('int32_t', 'imaps'),
                                    ('int32_t', 'conv_rows'),
                                    ('int32_t', 'use_vector'),
                                    ('int32_t', 'use_fia'),
                                    ('int32_t', 'use_db'),
                                    ('int32_t', 'use_depthwise'),
                                    ('int32_t', 'use_strided'),
                                    ('int32_t', 'fit_weights'),
                                    ('int32_t', 'split_weight_shaper_buffers'),
                                    ('int32_t', 'direct_dma'),
                                    ('int32_t', 'mxp_double_buffer'),
                                    ('offset', 'filter_data'),
                                    ('offset', 'bias_data'),
                                    ('offset', 'quantization_records')],
                             'ConcatOptions':[('int32_t', 'axis')],
                             "eltwise8": [('offset', 'input2_multiplier'),
                                          ('offset', 'input2_shift'),
                                          ('int32_t', 'input2_offset'),
                                          ('offset', 'bias_data'),
                                          ('int32_t', 'swap'),
                                          ('int32_t', 'optimized'),
                                          ('int32_t', 'isize'),
                                          ('int32_t', 'left_shift'),
                                          ('int32_t', 'type')],
                             "broadcast8":[('int32_t[4]', 'filter_shape_dims'),
                                          ('offset', 'filter_multiplier'),
                                          ('offset', 'filter_shift'),
                                          ('int32_t', 'filter_offset'),
                                          ('offset', 'filter_data'),
                                          ('offset', 'bias_data'),
                                          ('float', 'iscale'),
                                          ('float', 'fscale'),
                                          ('float', 'oscale'),
                                          ('int32_t', 'broadcast'),
                                          ('int32_t', 'optimized'),
                                          ('int32_t', 'isize'),
                                          ('int32_t', 'left_shift'),
                                          ('int32_t', 'sub'),
                                          ('int32_t', 'swap_inputs')],
                             "reduce8":[('int32_t', 'axis'),
                                        ('int32_t', 'arg_max'),
                                        ('offset', 'axis_list')],
                             "PadOptions": [('int32_t', 'value'),
                                            ('int32_t', 'transpose_dilate_w'),
                                            ('int32_t', 'transpose_dilate_h'),
                                            ],
                             "clip": [('float', 'min'),
                                      ('float', 'max')],
                             "depthwise": [('int32_t', 'unsigned_input'),
                                           ('int32_t', 'unsigned_output'),
                                           ('offset', 'weights')],
                             "prelu":[('offset', 'alpha_multiplier'),
                                      ('offset', 'alpha_shift'),
                                      ('int32_t', 'alpha_offset'),
                                      ('int32_t', 'optimized'),
                                      ('int32_t', 'maps_at_once'),
                                      ('float', 'iscale'),
                                      ('float', 'ascale'),
                                      ('float', 'oscale'),
                                      ('offset', 'vci_int8'),
                                      ('offset', 'alpha_data'),
                                      ('int32_t[4]', 'alpha_shape')],
                             "leakyrelu":[('offset', 'alpha_multiplier'),
                                        ('offset', 'alpha_shift')],
                             "SoftmaxOptions":[('int32_t', 'diff_min'),
                                        ('int32_t', 'axis'),
                                        ('int32_t', 'depth'),
                                        ('int32_t', 'count'),
                                        ('offset', 'vci_int8'),
                                        ('offset', 'lut_int32'),
                                        ('offset', 'idx_int8')],
                             "ActivationOptions":[('int32_t', 'input_range_radius'),
                                        ('int32_t', 'count'),
                                        ('int32_t', 'lut_count'),
                                        ('offset', 'vci_int8'),
                                        ('offset', 'lut_int8'),
                                        ('offset', 'idx_int8')],
                             "LogSoftmaxOptions":[('int32_t', 'diff_min'),
                                        ('int32_t', 'outer_size'),
                                        ('int32_t', 'depth'),
                                        ('int32_t', 'reverse_scaling_divisor'),
			                            ('int32_t', 'reverse_scaling_right_shift'),
                                        ('int32_t', 'axis')],
                             "prefetch":[('offset', 'memory_offset')],
                             "SliceOptions":[('int32_t[4]', 'begin'),
                                      ('int32_t[4]', 'end'),
                                      ('int32_t[4]', 'stride')],
                             "GatherOptions":[('int32_t', 'axis'),
                                       ('int32_t', 'batch_dims'),
                                       ('offset', 'coord_data'),
                                       ('int32_t', 'batch_size'),
                                       ('int32_t', 'outer_size'),
                                       ('int32_t', 'axis_size'),
                                       ('int32_t', 'inner_size'),
                                       ('int32_t', 'coord_size'),
                                       ('int32_t', 'swap_input_order')],
                             "SpaceToBatchNDOptions":[('offset', 'block_shape_data'),
                                       ('offset', 'paddings_data')],
                             "BatchToSpaceNDOptions":[('offset', 'block_shape_data'),
                                       ('offset', 'crop_data')],
                             "MirrorPadOptions":[('int32_t', 'mode')],
                             "ReshapeOptions":[('int32_t', 'mode')],
                             "PackOptions":[('int32_t', 'axis'),
                                       ('int32_t', 'count'),
                                       ('int32_t', 'dims')],
                          'ResizeOptions':[('float[2]', 'scale'),
                                    ('int32_t', 'mode'),
                                    ('int32_t', 'mode')],
                          'TransposeOptions':[('int32_t[3]', 'permutation'),
                                       ('int32_t','out_maps_at_once'),
                                       ('int32_t','out_rows_at_once')],
                          'SplitOptions':[('int32_t', 'axis'),
                                     ('offset', 'splits')],
                            'embedding':[('int32_t[2]', 'colar_map_dims'),
                                          ('offset', 'colar_map_data')],
                            'MinMaxOptions':[('int32_t', 'max'),
                                             ('int32_t[4]', 'filter_shape_dims'),
                                            ('offset', 'filter_multiplier'),
                                            ('offset', 'filter_shift'),
                                            ('int32_t', 'filter_offset'),
                                            ('offset', 'filter_data')]
                                       }
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
        self.tensor_array = []
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

    def update_offsets(self,offset, min_orig=None):
        union = self.get_union()
        for attr in self.offset_attributes:
            orig = getattr(self,attr)
            if orig is not None and orig != -1:
                if min_orig is None or orig >= min_orig:
                    # print(attr, orig, orig+offset, offset)
                    setattr(self,attr,orig+offset)
        for attr in union.offset_attributes:
            orig = getattr(union,attr)
            if orig is not None and orig != -1:
                if min_orig is None or orig >= min_orig:
                    # print(union, attr, orig, orig+offset, offset)
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
        self.include_io_data = 1
        self.magic = 0x1ABE11ED
        self.fixed_replay_buffer0 = 0
        self.fixed_replay_buffer1 = 0
        self.fixed_replay_buffer2 = 0
        self.fixed_replay_buffer3 = 0


class Tensor(Struct):
    def __init__(self):
        super().__init__(Tensor_struct)
        self.multiplier = -1
        self.shift = -1
        self.scale_f16 = -1
        self.buffer = [0,-1]
        self.direct = -1
        self.external_producer = 0
        self.external_consumer = 0


class Node(Struct):
    def __init__(self):
        super().__init__(Node_struct)
        #these are set at runtime
        self.replay_buffer =0
        self.replay_buffer_size = 0
        self.input_strides = [1,1]
        self.output_strides = [1,1]
        self.channels = 0
        self.m = 0
        self.n = 0
        self.maps = 0
        self.rows = 0
        self.cols = 0
        self.skip = 0
        self.num_inputs = 1
        self.num_outputs = 1
        self.num_tensors = -1
        self.prefetch_bytes_per_map = 0

        self.use_replay = False

        self.activation_max = -1
        self.activation_min = -1
        self.input_offset = -1
        self.input_multiplier = -1
        self.input_shift = -1
        self.output_offset = -1
        self.output_multiplier = -1
        self.output_shift = -1

        self.offloaded = 0


class Subnode(Struct):
    def __init__(self):
        super().__init__(Subnode_struct)
        self.kernel_shape=[1,1]
        self.strides = [1,1]
        self.dilations = [1,1]
        self.pads = [0,0,0,0,0,0]
        self.maps = 0
        self.nop = 0
        self.tensor_array = []
        self.tensors = -1

        self.num_inputs = 1
        self.num_outputs = 1
        self.num_tensors = -1
        self.activation_max = -1
        self.activation_min = -1
        self.input_offset = -1
        self.input_multiplier = -1
        self.input_shift = -1
        self.output_offset = -1
        self.output_multiplier = -1
        self.output_shift = -1

        #####
        ## TODO: I don't think these attributes are used
        self.input_data_type = 0
        self.output_data_type = 0


class resize_mode(enum.IntEnum):
    NEAREST = 0
    LINEAR = 1


class eltwise_type(enum.IntEnum):
    ELTWISE_ADD = 0
    ELTWISE_MUL = 1
    ELTWISE_SUB = 2
    ELTWISE_DIV = 3
    ELTWISE_GREATER = 4
    ELTWISE_GREATER_EQUAL = 5
    ELTWISE_LESS= 6
    ELTWISE_LESS_EQUAL = 7
    ELTWISE_EQUAL = 8
    ELTWISE_NOT_EQUAL = 9
    ELTWISE_MINIMUM = 10
    ELTWISE_MAXIMUM = 11
    ELTWISE_SQUARED_DIFFERENCE = 12


class calc_type(enum.IntEnum):
    UINT8 = 0
    INT8 = 1
    INT16 = 2
    INT32 = 3
    UNKNOWN = 4

    def from_str(e):
        e = e.upper()
        if e == "INT8":
            return calc_type.INT8
        if e == "UINT8":
            return calc_type.UINT8
        if e == "INT16":
            return calc_type.INT16
        if e == "INT32":
            return calc_type.INT32
        if e == "BOOL":
            return calc_type.UINT8
        return calc_type.UNKNOWN


def np_type(t):
       return {
            calc_type.UINT8 : np.uint8,
            calc_type.INT8  : np.int8,
            calc_type.INT16 : np.int16,
            calc_type.INT32 : np.int32,
        } [t]


def sizeof_calc_type(t):
       return {
            calc_type.UINT8 : 1,
            calc_type.INT8  :1,
            calc_type.INT16 :2,
            calc_type.INT32 :4,
        } [t]

class VNNXLUTOperator(enum.IntEnum):
    SILU = 300
    MUL = 301
    ADD = 302
    SUB = 303
    MUL_ADD = 304
    UNKNOWN = 305

    def from_str(e):
        e = e.upper()
        if e == "SILU":
            return VNNXLUTOperator.SILU
        if e == "MUL":
            return VNNXLUTOperator.MUL
        if e == "ADD":
            return VNNXLUTOperator.ADD
        if e == "SUB":
            return VNNXLUTOperator.SUB
        if e == "MUL_ADD":
            return VNNXLUTOperator.MUL_ADD
        
        return VNNXLUTOperator.UNKNOWN

class VNNXOperator(enum.IntEnum):
    IDENTITY = 200
    ELTWISE = 201
    PREFETCH = 202
    LUT = 203
    UNKNOWN = 204

    def from_str(e):
        e = e.upper()
        if e == "ELTWISE":
            return VNNXOperator.ELTWISE
        if e == "IDENTITY":
            return VNNXOperator.IDENTITY
        if e == "PREFETCH":
            return VNNXOperator.PREFETCH
        if e == "LUT":
            return VNNXOperator.LUT
        return VNNXOperator.UNKNOWN


class BuiltinOperator(enum.IntEnum):
    ADD = 0
    AVERAGE_POOL_2D = 1
    CONCATENATION = 2
    CONV_2D = 3
    DEPTHWISE_CONV_2D = 4
    DEPTH_TO_SPACE = 5
    DEQUANTIZE = 6
    EMBEDDING_LOOKUP = 7
    FLOOR = 8
    FULLY_CONNECTED = 9
    HASHTABLE_LOOKUP = 10
    L2_NORMALIZATION = 11
    L2_POOL_2D = 12
    LOCAL_RESPONSE_NORMALIZATION = 13
    LOGISTIC = 14
    LSH_PROJECTION = 15
    LSTM = 16
    MAX_POOL_2D = 17
    MUL = 18
    RELU = 19
    RELU_N1_TO_1 = 20
    RELU6 = 21
    RESHAPE = 22
    RESIZE_BILINEAR = 23
    RNN = 24
    SOFTMAX = 25
    SPACE_TO_DEPTH = 26
    SVDF = 27
    TANH = 28
    CONCAT_EMBEDDINGS = 29
    SKIP_GRAM = 30
    CALL = 31
    CUSTOM = 32
    EMBEDDING_LOOKUP_SPARSE = 33
    PAD = 34
    UNIDIRECTIONAL_SEQUENCE_RNN = 35
    GATHER = 36
    BATCH_TO_SPACE_ND = 37
    SPACE_TO_BATCH_ND = 38
    TRANSPOSE = 39
    MEAN = 40
    SUB = 41
    DIV = 42
    SQUEEZE = 43
    UNIDIRECTIONAL_SEQUENCE_LSTM = 44
    STRIDED_SLICE = 45
    BIDIRECTIONAL_SEQUENCE_RNN = 46
    EXP = 47
    TOPK_V2 = 48
    SPLIT = 49
    LOG_SOFTMAX = 50
    DELEGATE = 51
    BIDIRECTIONAL_SEQUENCE_LSTM = 52
    CAST = 53
    PRELU = 54
    MAXIMUM = 55
    ARG_MAX = 56
    MINIMUM = 57
    LESS = 58
    NEG = 59
    PADV2 = 60
    GREATER = 61
    GREATER_EQUAL = 62
    LESS_EQUAL = 63
    SELECT = 64
    SLICE = 65
    SIN = 66
    TRANSPOSE_CONV = 67
    SPARSE_TO_DENSE = 68
    TILE = 69
    EXPAND_DIMS = 70
    EQUAL = 71
    NOT_EQUAL = 72
    LOG = 73
    SUM = 74
    SQRT = 75
    RSQRT = 76
    SHAPE = 77
    POW = 78
    ARG_MIN = 79
    FAKE_QUANT = 80
    REDUCE_PROD = 81
    REDUCE_MAX = 82
    PACK = 83
    LOGICAL_OR = 84
    ONE_HOT = 85
    LOGICAL_AND = 86
    LOGICAL_NOT = 87
    UNPACK = 88
    REDUCE_MIN = 89
    FLOOR_DIV = 90
    REDUCE_ANY = 91
    SQUARE = 92
    ZEROS_LIKE = 93
    FILL = 94
    FLOOR_MOD = 95
    RANGE = 96
    RESIZE_NEAREST_NEIGHBOR = 97
    LEAKY_RELU = 98
    SQUARED_DIFFERENCE = 99
    MIRROR_PAD = 100
    ABS = 101
    SPLIT_V = 102
    UNIQUE = 103
    CEIL = 104
    REVERSE_V2 = 105
    ADD_N = 106
    GATHER_ND = 107
    COS = 108
    WHERE = 109
    RANK = 110
    ELU = 111
    REVERSE_SEQUENCE = 112
    MATRIX_DIAG = 113
    QUANTIZE = 114
    MATRIX_SET_DIAG = 115
    ROUND = 116
    HARD_SWISH = 117
    IF = 118
    WHILE = 119
    NON_MAX_SUPPRESSION_V4 = 120
    NON_MAX_SUPPRESSION_V5 = 121
    SCATTER_ND = 122
    SELECT_V2 = 123
    DENSIFY = 124
    SEGMENT_SUM = 125
    BATCH_MATMUL = 126
    PLACEHOLDER_FOR_GREATER_OP_CODES = 127
    CUMSUM = 128
    CALL_ONCE = 129
    BROADCAST_TO = 130
    RFFT2D = 131
    CONV_3D = 132
    IMAG=133
    REAL=134
    COMPLEX_ABS=135
    HASHTABLE = 136
    HASHTABLE_FIND = 137
    HASHTABLE_IMPORT = 138
    HASHTABLE_SIZE = 139
    REDUCE_ALL = 140
    CONV_3D_TRANSPOSE = 141
    VAR_HANDLE = 142
    READ_VARIABLE = 143
    ASSIGN_VARIABLE = 144
    BROADCAST_ARGS = 145
    RANDOM_STANDARD_NORMAL = 146
    BUCKETIZE = 147
    RANDOM_UNIFORM = 148
    MULTINOMIAL = 149
    GELU = 150
    DYNAMIC_UPDATE_SLICE = 151
    RELU_0_TO_1 = 152
    UNSORTED_SEGMENT_PROD = 153
    UNSORTED_SEGMENT_MAX = 154
    UNSORTED_SEGMENT_SUM = 155
    ATAN2 = 156
    UNSORTED_SEGMENT_MIN = 157
    SIGN = 158
    BITCAST = 159
    BITWISE_XOR = 160
    RIGHT_SHIFT = 161


def enum_to_union_name(e):
    union_names = [(BuiltinOperator.CONV_2D, "Conv2DOptions"),
                   (BuiltinOperator.FULLY_CONNECTED, "FullyConnectedOptions"),
                   (BuiltinOperator.CONCATENATION, "ConcatOptions"), 
                   (VNNXOperator.ELTWISE, "eltwise8"),
                   (BuiltinOperator.TRANSPOSE_CONV, "Conv2DOptions"),
                   (BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM, "LSTMOptions"),
                   (BuiltinOperator.TRANSPOSE, "TransposeOptions"),
                   (BuiltinOperator.RESIZE_NEAREST_NEIGHBOR, "ResizeOptions"),
                   (BuiltinOperator.RESIZE_BILINEAR, "ResizeOptions"),
                   (BuiltinOperator.SPACE_TO_DEPTH, ""),
                   (BuiltinOperator.ARG_MAX, "reduce8"),
                   (BuiltinOperator.ARG_MIN, "reduce8"),
                   (BuiltinOperator.DEPTHWISE_CONV_2D, "Conv2DOptions"),
                   (BuiltinOperator.AVERAGE_POOL_2D, ""),
                   (BuiltinOperator.MAX_POOL_2D, ""),
                   (BuiltinOperator.LEAKY_RELU, "leakyrelu"),
                   (BuiltinOperator.PRELU, "prelu"),
                   (BuiltinOperator.PAD, "PadOptions"),
                   (BuiltinOperator.MUL, "broadcast8"),
                   (BuiltinOperator.ADD, "broadcast8"),
                   (BuiltinOperator.SUB, "broadcast8"),
                   (BuiltinOperator.SQUARED_DIFFERENCE, "broadcast8"),
                   (VNNXOperator.PREFETCH, "prefetch"),
                   (BuiltinOperator.SOFTMAX, "SoftmaxOptions"),
                   (BuiltinOperator.TANH, "ActivationOptions"),
                   (BuiltinOperator.ELU, "ActivationOptions"),
                   (BuiltinOperator.EXP, "ActivationOptions"),
                   (BuiltinOperator.LOG, "ActivationOptions"),
                   (BuiltinOperator.GELU, "ActivationOptions"),
                   (VNNXOperator.LUT, "ActivationOptions"),
                   (BuiltinOperator.LOGISTIC, "ActivationOptions"),
                   (BuiltinOperator.HARD_SWISH, "ActivationOptions"),
                   (BuiltinOperator.RSQRT, "ActivationOptions"),
                   (BuiltinOperator.LOG_SOFTMAX, "LogSoftmaxOptions"),
                   (BuiltinOperator.GREATER, "greater_broadcast"),
                   (BuiltinOperator.GREATER_EQUAL, "greater_equal_broadcast"),
                   (BuiltinOperator.LESS, "less_broadcast"),
                   (BuiltinOperator.LESS_EQUAL, "less_equal_broadcast"),
                   (BuiltinOperator.EQUAL, "equal_broadcast"),
                   (BuiltinOperator.NOT_EQUAL, "not_equal_broadcast"),
                   (BuiltinOperator.SLICE, "SliceOptions"),
                   (BuiltinOperator.STRIDED_SLICE, "SliceOptions"),
                   (BuiltinOperator.GATHER, "GatherOptions"),
                   (BuiltinOperator.MIRROR_PAD, "MirrorPadOptions"),
                   (BuiltinOperator.RESHAPE, "ReshapeOptions"),
                   (BuiltinOperator.SPACE_TO_BATCH_ND, 'SpaceToBatchNDOptions'),
                   (BuiltinOperator.BATCH_TO_SPACE_ND, 'BatchToSpaceNDOptions'),
                   (BuiltinOperator.PACK, 'PackOptions'),
                   (BuiltinOperator.UNPACK, 'PackOptions'),
                   (BuiltinOperator.TILE, 'TileOptions'),
                   (BuiltinOperator.SPLIT, "SplitOptions"),
                   (BuiltinOperator.SPLIT_V, "SplitOptions"),
                   (BuiltinOperator.EMBEDDING_LOOKUP, "embedding"),
                   (BuiltinOperator.MAXIMUM, "MinMaxOptions"),
                   (BuiltinOperator.MINIMUM, "MinMaxOptions")
                   ]
    for t, n in union_names:
        if type(t) == type(e) and t == e:
            return n
    return ""


class weight_array(bytearray):
    def __iadd__(self, other):
        super().__iadd__(other)
        while(len(self) % 4):
            self.append(0)
        return self


def graph_version(script_dir=None):
    if script_dir is None:
        script_path = os.path.dirname(os.path.realpath(__file__))
    else:
        script_path = script_dir

    # installed directory
    vnnx_types_path = os.path.join(script_path, "vnnx-types.h")
    if not os.path.exists(vnnx_types_path):
        vnnx_types_path = os.path.join(script_path,"../../../../libvnnx/include/vnnx-types.h")
    vnnx_type_hash = hashlib.md5(open(vnnx_types_path,"rb").read()).hexdigest()
    return int(vnnx_type_hash[:8], 16)
