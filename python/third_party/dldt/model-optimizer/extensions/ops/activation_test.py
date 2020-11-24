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

from extensions.ops.activation_ops import Elu
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph


class TestActivationOp(unittest.TestCase):
    nodes_attributes = {
        'node_1': {
            'shape': np.array([227, 227, 227, 227]),
            'value': None
        },
        'activation_node': {
            'op': 'Activation',
            'kind': 'op'
        },
        'node_3': {
            'shape': None
        }
    }

    def test_activation_elu_infer(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('node_1', 'activation_node'),
                                ('activation_node', 'node_3')
                            ],
                            {
                                'node_1': {
                                    'value': np.array([6, -4, -2, -1])
                                },
                                'activation_node': {
                                    'operation': 'elu',
                                    'alpha': 1.0,
                                },
                                'node_3': {
                                    'value': None
                                }
                            })
        graph.graph['layout'] = 'NCHW'
        activation_node = Node(graph, 'activation_node')
        Elu.infer(activation_node)
        exp_shape = np.array([227, 227, 227, 227])
        res_shape = graph.node['node_3']['shape']
        res_value = graph.node['node_3']['value']
        exp_value = np.array([6., -0.98168436, -0.86466472, -0.63212056])
        for i, value in enumerate(exp_shape):
            self.assertEqual(res_shape[i], value)
        for i, value in enumerate(exp_value):
            self.assertAlmostEqual(res_value[i], value)
