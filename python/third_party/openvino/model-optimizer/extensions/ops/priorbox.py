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


class PriorBoxOp(Op):
    op = 'PriorBox'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'flip': True,
            'clip': True,
            'scale_all_sizes': True,
            'max_size': np.array([]),
            'min_size': np.array([]),
            'aspect_ratio': np.array([]),
            'density': np.array([]),
            'fixed_size': np.array([]),
            'fixed_ratio': np.array([]),
            'in_ports_count': 2,
            'out_ports_count': 1,
            'type_infer': self.type_infer,
            'infer': self.priorbox_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'min_size',
            'max_size',
            'aspect_ratio',
            'flip',
            'clip',
            'variance',
            'img_size',
            'img_h',
            'img_w',
            'step',
            'step_h',
            'step_w',
            'offset',
            'density',
            'fixed_size',
            'fixed_ratio',
        ]

    def backend_attrs(self):
        return [
            ('flip', lambda node: int(node.flip)),  # We need to convert this boolean attribute value to int to keep
            # forward compatibility with IE 2021.2
            ('clip', lambda node: int(node.clip)),  # We need to convert this boolean attribute value to int to keep
            # forward compatibility with IE 2021.2
            'step',
            'offset',
            ('scale_all_sizes', lambda node: bool_to_str(node, 'scale_all_sizes')),
            ('min_size', lambda node: attr_getter(node, 'min_size')),
            ('max_size', lambda node: attr_getter(node, 'max_size')),
            ('aspect_ratio', lambda node: attr_getter(node, 'aspect_ratio')),
            ('variance', lambda node: attr_getter(node, 'variance')),
            ('density', lambda node: attr_getter(node, 'density')),
            ('fixed_size', lambda node: attr_getter(node, 'fixed_size')),
            ('fixed_ratio', lambda node: attr_getter(node, 'fixed_ratio')),
        ]

    @staticmethod
    def type_infer(node):
        node.out_port(0).set_data_type(np.float32)

    @staticmethod
    def priorbox_infer(node: Node):
        layout = node.graph.graph['layout']
        data_shape = node.in_node(0).shape

        # calculate all different aspect_ratios (the first one is always 1)
        # in aspect_ratio 1/x values will be added for all except 1 if flip is True
        ar_seen = [1.0]
        ar_seen.extend(node.aspect_ratio.copy())
        if node.flip:
            for s in node.aspect_ratio:
                ar_seen.append(1.0 / s)

        ar_seen = np.unique(np.array(ar_seen).round(decimals=6))

        num_ratios = 0
        if len(node.min_size) > 0:
            num_ratios = len(ar_seen) * len(node.min_size)

        if node.has_valid('fixed_size') and len(node.fixed_size) > 0:
            num_ratios = len(ar_seen) * len(node.fixed_size)

        if node.has_valid('density') and len(node.density) > 0:
            for d in node.density:
                if node.has_valid('fixed_ratio') and len(node.fixed_ratio) > 0:
                    num_ratios = num_ratios + len(node.fixed_ratio) * (pow(d, 2) - 1)
                else:
                    num_ratios = num_ratios + len(ar_seen) * (pow(d, 2) - 1)

        num_ratios = num_ratios + len(node.max_size)

        if node.has_and_set('V10_infer'):
            assert node.in_node(0).value is not None
            node.out_node(0).shape = np.array([2, np.prod(node.in_node(0).value) * num_ratios * 4], dtype=np.int64)
        else:
            res_prod = data_shape[get_height_dim(layout, 4)] * data_shape[get_width_dim(layout, 4)] * num_ratios * 4
            node.out_node(0).shape = np.array([1, 2, res_prod], dtype=np.int64)
