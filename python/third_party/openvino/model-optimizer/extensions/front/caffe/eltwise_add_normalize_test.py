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

from extensions.front.caffe.eltwise_add_normalize import EltwiseAddNormalize
from mo.front.common.partial_infer.utils import int64_array
from mo.middle.passes.eliminate_test import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs

input_shape = int64_array([1, 4, 10])
const_1_value = np.array([2.0])
const_2_value = np.array([3.0])
const_3_value = np.array([4.0])

nodes_attributes = {
    'placeholder_1': {'shape': input_shape, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_2': {'shape': input_shape, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_3': {'shape': input_shape, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'eltwise': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'eltwise_n': {'type': 'EltwiseN', 'kind': 'op', 'op': 'EltwiseN', 'operation': 'sum'},

    'sigmoid_1': {'type': 'Sigmoid', 'kind': 'op', 'op': 'Sigmoid'},
    'sigmoid_2': {'type': 'Sigmoid', 'kind': 'op', 'op': 'Sigmoid'},

    'mul_1': {'type': 'Multiply', 'op': 'Mul', 'kind': 'op'},
    'const_1': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'mul_2': {'type': 'Multiply', 'op': 'Mul', 'kind': 'op'},
    'const_2': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'mul_3': {'type': 'Multiply', 'op': 'Mul', 'kind': 'op'},
    'const_3': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'shape': None, 'value': None},
}


class EltwiseAddNormalizationTest(unittest.TestCase):
    def test_first_input_coeff_not_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'eltwise'),
                             ('placeholder_2', 'eltwise'),
                             ('eltwise', 'sigmoid_1'),
                             ('eltwise', 'sigmoid_2'),
                             ],
                            {'eltwise': {'coeff': np.array([2.0, 1.0])}
                             }, nodes_with_edges_only=True)
        graph.stage = 'front'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'mul_1'),
                                 ('const_1', 'mul_1'),
                                 ('mul_1', 'eltwise'),
                                 ('placeholder_2', 'eltwise'),
                                 ('eltwise', 'sigmoid_1'),
                                 ('eltwise', 'sigmoid_2'),
                                 ],
                                {'const_1': {'value': const_1_value, 'shape': const_1_value.shape},
                                 }, nodes_with_edges_only=True)

        EltwiseAddNormalize().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_second_input_coeff_not_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'eltwise'),
                             ('placeholder_2', 'eltwise'),
                             ('eltwise', 'sigmoid_1'),
                             ('eltwise', 'sigmoid_2'),
                             ],
                            {'eltwise': {'coeff': np.array([1.0, const_2_value[0]])}
                             }, nodes_with_edges_only=True)
        graph.stage = 'front'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'eltwise'),
                                 ('placeholder_2', 'mul_2'),
                                 ('const_2', 'mul_2'),
                                 ('mul_2', 'eltwise'),
                                 ('eltwise', 'sigmoid_1'),
                                 ('eltwise', 'sigmoid_2'),
                                 ],
                                {'const_2': {'value': const_2_value, 'shape': const_2_value.shape},
                                 }, nodes_with_edges_only=True)

        EltwiseAddNormalize().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_both_input_coeff_not_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'eltwise'),
                             ('placeholder_2', 'eltwise'),
                             ('eltwise', 'sigmoid_1'),
                             ('eltwise', 'sigmoid_2'),
                             ],
                            {'eltwise': {'coeff': np.array([const_1_value[0], const_2_value[0]])}
                             }, nodes_with_edges_only=True)
        graph.stage = 'front'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'mul_1'),
                                 ('const_1', 'mul_1'),
                                 ('mul_1', 'eltwise'),
                                 ('placeholder_2', 'mul_2'),
                                 ('const_2', 'mul_2'),
                                 ('mul_2', 'eltwise'),
                                 ('eltwise', 'sigmoid_1'),
                                 ('eltwise', 'sigmoid_2'),
                                 ],
                                {'const_1': {'value': const_1_value, 'shape': const_1_value.shape},
                                 'const_2': {'value': const_2_value, 'shape': const_2_value.shape},
                                 }, nodes_with_edges_only=True)

        EltwiseAddNormalize().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_both_input_coeff_equal_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'eltwise'),
                             ('placeholder_2', 'eltwise'),
                             ('eltwise', 'sigmoid_1'),
                             ('eltwise', 'sigmoid_2'),
                             ],
                            {'eltwise': {'coeff': np.array([1.0, 1.0])}
                             }, nodes_with_edges_only=True)
        graph.stage = 'front'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'eltwise'),
                                 ('placeholder_2', 'eltwise'),
                                 ('eltwise', 'sigmoid_1'),
                                 ('eltwise', 'sigmoid_2'),
                                 ],
                                {}, nodes_with_edges_only=True)

        EltwiseAddNormalize().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_both_input_coeff_not_defined(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'eltwise'),
                             ('placeholder_2', 'eltwise'),
                             ('eltwise', 'sigmoid_1'),
                             ('eltwise', 'sigmoid_2'),
                             ],
                            {'eltwise': {'coeff': None}
                             }, nodes_with_edges_only=True)
        graph.stage = 'front'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'eltwise'),
                                 ('placeholder_2', 'eltwise'),
                                 ('eltwise', 'sigmoid_1'),
                                 ('eltwise', 'sigmoid_2'),
                                 ],
                                {}, nodes_with_edges_only=True)

        EltwiseAddNormalize().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1', check_op_attrs=True)
        self.assertTrue(flag, resp)


    def test_3_inputs_with_coeffs(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'eltwise'),
                             ('placeholder_2', 'eltwise'),
                             ('placeholder_3', 'eltwise'),
                             ('eltwise', 'sigmoid_1'),
                             ],
                            {'eltwise': {'coeff': np.array([const_1_value[0], const_2_value[0], const_3_value[0]])}
                             }, nodes_with_edges_only=True)
        graph.stage = 'front'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'mul_1'),
                                 ('const_1', 'mul_1'),
                                 ('mul_1', 'eltwise_n'),
                                 ('placeholder_2', 'mul_2'),
                                 ('const_2', 'mul_2'),
                                 ('mul_2', 'eltwise_n'),
                                 ('placeholder_3', 'mul_3'),
                                 ('const_3', 'mul_3'),
                                 ('mul_3', 'eltwise_n'),
                                 ('eltwise_n', 'sigmoid_1'),
                                 ],
                                {'const_1': {'value': const_1_value, 'shape': const_1_value.shape},
                                 'const_2': {'value': const_2_value, 'shape': const_2_value.shape},
                                 'const_3': {'value': const_3_value, 'shape': const_3_value.shape},
                                 }, nodes_with_edges_only=True)

        EltwiseAddNormalize().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1', check_op_attrs=True)
        self.assertTrue(flag, resp)