"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import math

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import bool_to_str
from mo.graph.graph import Node, Graph
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op, PermuteAttrs


def infer_for_opset4(node: Node):
    assert len([p for p in node.in_ports().values() if not p.disconnected()]) in [3, 4], \
        "Interpolate-4 node {} must have 3 or 4 inputs".format(node.soft_get(node.name, node.id))
    assert node.has_valid('mode')
    assert node.has_valid('shape_calculation_mode')
    src_shape = node.in_port(0).data.get_shape()
    assert src_shape is not None

    input_rank = len(src_shape)

    pads_begin = correct_pad(node.soft_get('pads_begin', [0]), input_rank)
    pads_end = correct_pad(node.soft_get('pads_end', [0]), input_rank)
    node['pads_begin'] = pads_begin
    node['pads_end'] = pads_end

    if len(node.in_ports()) == 3:
        axes = list(range(0, input_rank))
    else:
        axes = node.in_port(3).get_source().data.get_value()
        assert axes is not None, \
            "Interpolate-4 node with name {} has None as 'axes' input".format(node.soft_get('name', node.id))

    axes = int64_array(axes)
    output_shape = src_shape + pads_begin + pads_end
    if node.shape_calculation_mode == 'sizes':
        dst_shape = node.in_port(1).data.get_value()
        assert dst_shape is not None
        correct_scales_using_dst_shape(node, dst_shape, src_shape, axes)
        for i, axis in enumerate(axes):
            output_shape[axis] = dst_shape[i]
    else:
        scales = node.in_port(2).data.get_value()
        assert scales is not None
        for i, axis in enumerate(axes):
            output_shape[axis] = math.floor(scales[i] * output_shape[axis] + 1.0e-5)

    if node.is_in_port_connected(3):
        PermuteInputs().set_input_permutation(node.in_node(3), node, 'input:0', 'axis')

    node.out_port(0).data.set_shape(output_shape)


def infer_for_opset1(node: Node):
    assert len([p for p in node.in_ports().values() if not p.disconnected()]) == 2
    assert node.has_valid('mode')
    assert node.has_valid('axes')

    src_shape = node.in_port(0).data.get_shape()

    assert src_shape is not None
    dst_shape = node.in_port(1).data.get_value()
    assert dst_shape is not None

    output_shape = src_shape.copy()
    for ind, axis in enumerate(node.axes):
        output_shape[axis] = dst_shape[ind]

    node.out_port(0).data.set_shape(output_shape)

    PermuteAttrs.create_permute_attrs(node, attrs=[('axes', 'input:0')])


def pad_attribute_to_str(node: Node, attr: str):
    return ','.join(map(str, node[attr])) if node.has_valid(attr) else None


def correct_pad(pad, rank):
    pad_len = len(pad)
    if pad_len < rank:
        return np.pad(pad, (0, rank - pad_len), 'constant').astype(np.int64)
    elif pad_len > rank:
        return np.array(pad[: rank]).astype(np.int64)
    else:
        return np.array(pad, dtype=np.int64)


def correct_scales_using_dst_shape(node, dst_shape, src_shape, axes):
    scales_value = node.in_port(2).data.get_value()
    if scales_value is None or len(scales_value) != len(dst_shape):
        corrected_scales = np.zeros(len(dst_shape))
        for i, axis in enumerate(list(axes)):
            corrected_scales[i] = dst_shape[i] / src_shape[axis]
        node.in_port(2).data.set_value(corrected_scales)


class Interpolate(Op):
    op = 'Interpolate'
    enabled = False
    infers = {
        'opset1': infer_for_opset1,
        'opset4': infer_for_opset4
    }

    def __init__(self, graph: Graph, attrs: dict):
        self.attributes_for_opsets = {
            'opset1': [
                ('axes', lambda node: ','.join(map(str, node.axes))),
                ('antialias', lambda node: bool_to_str(node, 'antialias')),
                ('align_corners', lambda node: bool_to_str(node, 'align_corners')),
                'mode', 'pads_begin', 'pads_end',
            ],
            'opset4': [
                'mode', 'nearest_mode', 'cube_coeff', 'coordinate_transformation_mode',
                'shape_calculation_mode',
                ('antialias', lambda node: bool_to_str(node, 'antialias')),
                ('pads_begin', lambda node: pad_attribute_to_str(node, 'pads_begin')),
                ('pads_end', lambda node: pad_attribute_to_str(node, 'pads_end')),
            ]
        }

        mandatory_props = {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'axes': None,
            'mode': None,
            'align_corners': 0,
            'antialias': 0,
            'pads_begin': 0,
            'pads_end': 0,

            'infer': self.infer,

            'force_precision_in_ports': {1: 'int64'},
            'in_ports_count': 2,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        opset = self.get_opset()
        key = opset if opset in self.attributes_for_opsets else 'opset1'
        return self.attributes_for_opsets[key]

    def infer(self, node: Node):
        opset = self.get_opset()
        key = opset if opset in self.infers else 'opset1'
        self.infers[key](node)

    @staticmethod
    def get_axes(node: Node) -> np.ndarray:
        opset = node.get_opset()
        if opset == 'opset1':
            interp_axes = node.soft_get('axes', None)
            return interp_axes if interp_axes is None else int64_array(interp_axes)

        src_shape = node.in_port(0).data.get_shape()
        assert src_shape is not None
        input_rank = len(src_shape)

        if len(node.in_ports()) == 3:
            axes = list(range(0, input_rank))
        else:
            axes = node.in_port(3).get_source().data.get_value()
        return int64_array(axes)
