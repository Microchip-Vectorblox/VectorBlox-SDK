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

import numpy as np

from mo.front.common.layout import get_width_dim, get_height_dim
from mo.front.extractor import attr_getter, bool_to_str
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class PriorBoxClusteredOp(Op):
    op = 'PriorBoxClustered'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': self.priorbox_clustered_infer,
            'type_infer': self.type_infer,
            'clip': True,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'width',
            'height',
            'flip',
            'clip',
            'variance',
            'img_size',
            'img_h',
            'img_w',
            'step',
            'step_h',
            'step_w',
            'offset'
        ]

    def backend_attrs(self):
        return [
            ('clip', lambda node: int(node.clip)),  # We need to convert this boolean attribute value to int to keep
            # forward compatibility with IE 2021.2
            'img_h',
            'img_w',
            'step',
            'step_h',
            'step_w',
            'offset',
            ('variance', lambda node: attr_getter(node, 'variance')),
            ('width', lambda node: attr_getter(node, 'width')),
            ('height', lambda node: attr_getter(node, 'height'))
        ]

    @staticmethod
    def type_infer(node):
        node.out_port(0).set_data_type(np.float32)

    @staticmethod
    def priorbox_clustered_infer(node: Node):
        layout = node.graph.graph['layout']
        data_shape = node.in_node(0).shape
        num_ratios = len(node.width)

        if node.has_and_set('V10_infer'):
            assert node.in_node(0).value is not None
            node.out_node(0).shape = np.array([2, np.prod(node.in_node(0).value) * num_ratios * 4], dtype=np.int64)
        else:
            res_prod = data_shape[get_height_dim(layout, 4)] * data_shape[get_width_dim(layout, 4)] * num_ratios * 4
            node.out_node(0).shape = np.array([1, 2, res_prod], dtype=np.int64)
