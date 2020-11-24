"""
 Copyright (C) 2018-2020 Intel Corporation

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

import unittest

import numpy as np

from extensions.ops.psroipooling import PSROIPoolingOp
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'node_2': {'type': 'Identity', 'kind': 'op'},
                    'psroipool': {'type': 'PSROIPooling', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': { 'kind': 'op', 'op': 'Result'}
                    }


class TestPSROIPooling(unittest.TestCase):
    def test_psroipool_infer_nchw(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'psroipool'),
                             ('node_2', 'psroipool'),
                             ('psroipool', 'node_3'),
                             ('node_3', 'op_output')
                            ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'node_2': {'shape': np.array([100, 5])},
                             'psroipool': {'output_dim': 4, 'group_size': 15}
                             })
        graph.graph['layout'] = 'NCHW'
        psroipool_node = Node(graph, 'psroipool')
        PSROIPoolingOp.psroipooling_infer(psroipool_node)
        exp_shape = np.array([100, 4, 15, 15])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_psroipool_infer_nhwc(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'psroipool'),
                             ('node_2', 'psroipool'),
                             ('psroipool', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 227, 227, 3])},
                             'node_2': {'shape': np.array([100, 5])},
                             'psroipool': {'output_dim': 4, 'group_size': 15}
                             })
        graph.graph['layout'] = 'NHWC'
        psroipool_node = Node(graph, 'psroipool')
        PSROIPoolingOp.psroipooling_infer(psroipool_node)
        exp_shape = np.array([100, 15, 15, 4])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_psroipool_infer_no_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'psroipool'),
                             ('node_2', 'psroipool'),
                             ('psroipool', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': None},
                             'node_2': {'shape': np.array([100, 5])},
                             'psroipool': {'output_dim': 4, 'group_size': 224}
                             })
        graph.graph['layout'] = 'NCHW'

        psroipool_node = Node(graph, 'psroipool')
        PSROIPoolingOp.psroipooling_infer(psroipool_node)
        res_shape = graph.node['node_3']['shape']
        self.assertIsNone(res_shape)
