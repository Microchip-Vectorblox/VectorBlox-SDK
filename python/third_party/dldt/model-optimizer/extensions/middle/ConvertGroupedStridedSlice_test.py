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
from generator import generator, generate

from extensions.middle.ConvertGroupedStridedSlice import ConvertGroupedStridedSlice
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'placeholder_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_2': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_begin_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_end_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_stride_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    # StridedSlice layers
    'sslice_1': {'type': None, 'kind': 'op', 'op': 'StridedSlice', 'slices': None,
                 'shrink_axis_mask': np.array([0, 0, 0, 0])},
    'sslice_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'sslice_2': {'type': None, 'kind': 'op', 'op': 'StridedSlice', 'slices': None,
                 'shrink_axis_mask': np.array([0, 0, 0, 0])},
    'sslice_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'sslice_3': {'type': None, 'kind': 'op', 'op': 'StridedSlice', 'slices': None,
                 'shrink_axis_mask': np.array([0, 0, 0, 0])},
    'sslice_3_data': {'value': None, 'shape': None, 'kind': 'data'},

    # Split layer
    'axis_const': {'kind': 'op'},
    'axis_const_data': {'value': None, 'shape': None, 'kind': 'data'},
    'split_dim_const': {'kind': 'op'},
    'split_dim_const_data': {'value': None, 'shape': None, 'kind': 'data'},

    'split_1': {'type': 'VariadicSplit', 'kind': 'op', 'op': 'VariadicSplit'},
    'split_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'split_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'split_3_data': {'value': None, 'shape': None, 'kind': 'data'},
    'split_4_data': {'value': None, 'shape': None, 'kind': 'data'},

    # Concat1 operation
    'concat_1': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'concat_1_data': {'value': None, 'shape': None, 'kind': 'data'},

    'op_output': {'kind': 'op', 'op': 'Result'},
    'op_output_1': {'kind': 'op', 'op': 'Result'},
    'op_output_2': {'kind': 'op', 'op': 'Result'},

    # Squeeze layers
    'sslice_1/Squeeze_shrink': {'type': None, 'value': None, 'kind': 'op', 'op': 'Squeeze'},
    'sslice_1/Squeeze_shrink_data': {'value': None, 'shape': None, 'kind': 'data'},
    'sslice_1/squeeze_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': int64_array([2])},
    'sslice_1/squeeze_const_data': {'kind': 'data', 'value': None, 'shape': None},

    'sslice_2/Squeeze_shrink': {'type': None, 'value': None, 'kind': 'op', 'op': 'Squeeze'},
    'sslice_2/Squeeze_shrink_data': {'value': None, 'shape': None, 'kind': 'data'},
    'sslice_2/squeeze_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': int64_array([2])},
    'sslice_2/squeeze_const_data': {'kind': 'data', 'value': None, 'shape': None},

    # Unsqueeze layer
    'sslice_2/Unsqueeze_new': {'type': None, 'value': None, 'kind': 'op', 'op': 'Unsqueeze'},
    'sslice_2/Unsqueeze_new_data': {'value': None, 'shape': None, 'kind': 'data'},
    'sslice_2/unsqueeze_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': int64_array([2])},
    'sslice_2/unsqueeze_const_data': {'kind': 'data', 'value': None, 'shape': None},
}

one_strided_slice_case_node_attributes = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'sslice': {'type': None, 'kind': 'op', 'op': 'StridedSlice', 'slices': None,
               'shrink_axis_mask': np.array([0, 0, 0, 0])},
    'sslice_data': {'value': None, 'shape': None, 'kind': 'data'},
    'op_output': {'kind': 'op', 'op': 'Result'},
}

one_strided_slice_case_edges = [
    ('placeholder', 'placeholder_data'),
    ('placeholder_data', 'sslice'),
    ('sslice', 'sslice_data'),
    ('sslice_data', 'op_output'),
]


@generator
class ConvertGroupedStridedSliceTests(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('placeholder_1_data', 'sslice_3'),
                             ('sslice_3', 'sslice_3_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('sslice_3_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 18, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(18, 36, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_3': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(36, 54, 1)])},
                             'sslice_3_data': {'shape': np.array([1, 227, 227, 18])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'split_1', {'in': 0}),
                                 ('split_1', 'split_1_data'),
                                 ('split_1', 'split_2_data'),
                                 ('split_1', 'split_3_data'),
                                 ('split_1_data', 'concat_1'),
                                 ('split_2_data', 'concat_1'),
                                 ('split_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output'),

                                 ('axis_const', 'axis_const_data'),
                                 ('split_dim_const', 'split_dim_const_data'),
                                 ('axis_const_data', 'split_1', {'in': 1}),
                                 ('split_dim_const_data', 'split_1', {'in': 2}),

                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'axis_const': {'value': 3},
                                 'split_1_data': {'shape': np.array([1, 227, 227, 18])},
                                 'split_2_data': {'shape': np.array([1, 227, 227, 18])},
                                 'split_3_data': {'shape': np.array([1, 227, 227, 18])},
                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 })

        ConvertGroupedStridedSlice().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('placeholder_1_data', 'sslice_3'),
                             ('sslice_3', 'sslice_3_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('sslice_3_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(19, 37, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(37, 54, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 17])},

                             'sslice_3': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 19, 1)])},
                             'sslice_3_data': {'shape': np.array([1, 227, 227, 19])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'split_1'),
                                 ('split_1', 'split_1_data'),
                                 ('split_1', 'split_2_data'),
                                 ('split_1', 'split_3_data'),
                                 ('split_1_data', 'concat_1'),
                                 ('split_2_data', 'concat_1'),
                                 ('split_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output'),

                                 ('axis_const', 'axis_const_data'),
                                 ('split_dim_const', 'split_dim_const_data'),
                                 ('axis_const_data', 'split_1', {'in': 1}),
                                 ('split_dim_const_data', 'split_1', {'in': 2}),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'axis_const': {'value': 3},
                                 'split_1_data': {'shape': np.array([1, 227, 227, 18])},
                                 'split_2_data': {'shape': np.array([1, 227, 227, 17])},
                                 'split_3_data': {'shape': np.array([1, 227, 227, 19])},
                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 })

        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Intersection of split ranges in feature dimension
    def test_3_neg(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('placeholder_1_data', 'sslice_3'),
                             ('sslice_3', 'sslice_3_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('sslice_3_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(19, 39, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 20])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(37, 54, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 17])},

                             'sslice_3': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 19, 1)])},
                             'sslice_3_data': {'shape': np.array([1, 227, 227, 19])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_1'),
                                 ('sslice_1', 'sslice_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2_data'),
                                 ('placeholder_1_data', 'sslice_3'),
                                 ('sslice_3', 'sslice_3_data'),
                                 ('sslice_1_data', 'concat_1'),
                                 ('sslice_2_data', 'concat_1'),
                                 ('sslice_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                                 'sslice_1': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(19, 39, 1)])},
                                 'sslice_1_data': {'shape': np.array([1, 227, 227, 20])},

                                 'sslice_2': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(37, 54, 1)])},
                                 'sslice_2_data': {'shape': np.array([1, 227, 227, 17])},

                                 'sslice_3': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 19, 1)])},
                                 'sslice_3_data': {'shape': np.array([1, 227, 227, 19])},

                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 })

        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Split range overflow in feature dimension
    def test_4_neg(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('placeholder_1_data', 'sslice_3'),
                             ('sslice_3', 'sslice_3_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('sslice_3_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(19, 37, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(37, 55, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_3': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 19, 1)])},
                             'sslice_3_data': {'shape': np.array([1, 227, 227, 19])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_1'),
                                 ('sslice_1', 'sslice_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2_data'),
                                 ('placeholder_1_data', 'sslice_3'),
                                 ('sslice_3', 'sslice_3_data'),
                                 ('sslice_1_data', 'concat_1'),
                                 ('sslice_2_data', 'concat_1'),
                                 ('sslice_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                                 'sslice_1': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(19, 37, 1)])},
                                 'sslice_1_data': {'shape': np.array([1, 227, 227, 18])},

                                 'sslice_2': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(37, 55, 1)])},
                                 'sslice_2_data': {'shape': np.array([1, 227, 227, 18])},

                                 'sslice_3': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 19, 1)])},
                                 'sslice_3_data': {'shape': np.array([1, 227, 227, 19])},

                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 })

        ConvertGroupedStridedSlice().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Split(1,H,W,54)--->Fake_data (1,H,W,1)
    #       |`---->Sslice1_out (1,H,W,18)
    #       |`---->Sslice2_out (1,H,W,18)
    #       `----->Sslice3_out (1,H,W,17)
    def test_5(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('placeholder_1_data', 'sslice_3'),
                             ('sslice_3', 'sslice_3_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('sslice_3_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(19, 37, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(37, 54, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 17])},

                             'sslice_3': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(1, 19, 1)])},
                             'sslice_3_data': {'shape': np.array([1, 227, 227, 18])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'split_1'),
                                 ('split_1', 'split_1_data'),
                                 ('split_1', 'split_2_data'),
                                 ('split_1', 'split_3_data'),
                                 ('split_1', 'split_4_data'),
                                 ('split_2_data', 'concat_1'),
                                 ('split_3_data', 'concat_1'),
                                 ('split_4_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output'),
                                 ('split_1_data', 'op_output_1'),

                                 ('axis_const', 'axis_const_data'),
                                 ('split_dim_const', 'split_dim_const_data'),
                                 ('axis_const_data', 'split_1', {'in': 1}),
                                 ('split_dim_const_data', 'split_1', {'in': 2}),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'axis_const': {'value': 3},
                                 'split_1_data': {'shape': np.array([1, 227, 227, 1])},
                                 'split_2_data': {'shape': np.array([1, 227, 227, 18])},
                                 'split_3_data': {'shape': np.array([1, 227, 227, 17])},
                                 'split_4_data': {'shape': np.array([1, 227, 227, 18])},
                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 })

        ConvertGroupedStridedSlice().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Split(1,H,W,54)
    #       |`---->Sslice1_out (1,H,W,(0,18))
    #       |`---->Fake_data (1,H,W,(18,27))
    #       |`---->Sslice3_out (1,H,W,(27,45))
    #       `----->Fake_data (1,H,W,(45,54))
    def test_6(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 18, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 18])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(27, 45, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 18])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'split_1'),
                                 ('split_1', 'split_1_data'),
                                 ('split_1', 'split_2_data'),
                                 ('split_1', 'split_3_data'),
                                 ('split_1', 'split_4_data'),
                                 ('split_1_data', 'concat_1'),
                                 ('split_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output'),
                                 ('split_2_data', 'op_output_1'),
                                 ('split_4_data', 'op_output_2'),

                                 ('axis_const', 'axis_const_data'),
                                 ('split_dim_const', 'split_dim_const_data'),
                                 ('axis_const_data', 'split_1', {'in': 1}),
                                 ('split_dim_const_data', 'split_1', {'in': 2}),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'axis_const': {'value': 3},
                                 'split_1_data': {'shape': np.array([1, 227, 227, 18])},
                                 'split_2_data': {'shape': np.array([1, 227, 227, 9])},
                                 'split_3_data': {'shape': np.array([1, 227, 227, 18])},
                                 'split_4_data': {'shape': np.array([1, 227, 227, 9])},
                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 })

        ConvertGroupedStridedSlice().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_7_neg(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 10, 1), slice(0, 227, 1), slice(0, 18, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 10, 227, 18])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(10, 227, 1), slice(0, 227, 1), slice(27, 45, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 217, 227, 18])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_1'),
                                 ('sslice_1', 'sslice_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2_data'),
                                 ('sslice_1_data', 'concat_1'),
                                 ('sslice_2_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                                 'sslice_1': {'slices': np.array(
                                     [slice(0, 1, 1), slice(0, 10, 1), slice(0, 227, 1), slice(0, 18, 1)])},
                                 'sslice_1_data': {'shape': np.array([1, 10, 227, 18])},

                                 'sslice_2': {'slices': np.array(
                                     [slice(0, 1, 1), slice(10, 227, 1), slice(0, 227, 1), slice(27, 45, 1)])},
                                 'sslice_2_data': {'shape': np.array([1, 217, 227, 18])},

                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 })

        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Split(1,54,W,C)
    #       |`---->Sslice1_out (1,(0,18),W,C)
    #       |`---->Sslice2_out (1,(18,36),W,C)
    #       `----->Fake_data (1,(36,54),W,C)
    def test_8(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 54, 54, 3])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 18, 1), slice(0, 54, 1), slice(0, 3, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 18, 54, 3])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(18, 36, 1), slice(0, 54, 1), slice(0, 3, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 18, 54, 3])},

                             'concat_1_data': {'shape': np.array([1, 54, 54, 3])},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'split_1'),
                                 ('split_1', 'split_1_data'),
                                 ('split_1', 'split_2_data'),
                                 ('split_1', 'split_3_data'),
                                 ('split_1_data', 'concat_1'),
                                 ('split_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output'),
                                 ('split_2_data', 'op_output_1'),

                                 ('axis_const', 'axis_const_data'),
                                 ('split_dim_const', 'split_dim_const_data'),
                                 ('axis_const_data', 'split_1', {'in': 1}),
                                 ('split_dim_const_data', 'split_1', {'in': 2}),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 54, 54, 3])},
                                 'axis_const': {'value': 1},
                                 'split_1_data': {'shape': np.array([1, 18, 54, 3])},
                                 'split_2_data': {'shape': np.array([1, 18, 54, 3])},
                                 'split_3_data': {'shape': np.array([1, 18, 54, 3])},
                                 'concat_1_data': {'shape': np.array([1, 54, 54, 3])},
                                 })

        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Test for the case when there is only 1 StridedSlice.
    @generate(*[(np.array([1, 227, 227, 54]),
                 np.array([slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 18, 1)]),
                 np.array([1, 227, 227, 18])),
                (np.array([57, 16, 100, 23]),
                 np.array([slice(3, 16, 1), slice(0, 16, 1), slice(0, 100, 1), slice(0, 23, 1)]),
                 np.array([13, 16, 100, 23])),
                (np.array([16, 800, 1024, 17]),
                 np.array([slice(0, 16, 1), slice(0, 800, 1), slice(13, 817, 1), slice(0, 17, 1)]),
                 np.array([16, 800, 804, 17]))])
    def test_9(self, input_shape, slices, output_shape):
        graph = build_graph(nodes_attrs=one_strided_slice_case_node_attributes,
                            edges=one_strided_slice_case_edges,
                            update_attributes={
                                'placeholder_data': {'shape': input_shape},
                                'sslice': {'slices': slices},
                                'sslice_data': {'shape': output_shape},
                            })
        graph.graph['layout'] = 'NHWC'
        graph_ref = build_graph(nodes_attrs=one_strided_slice_case_node_attributes,
                                edges=one_strided_slice_case_edges,
                                update_attributes={
                                    'placeholder_data': {'shape': input_shape},
                                    'sslice': {'slices': slices},
                                    'sslice_data': {'shape': output_shape},
                                })
        pattern = ConvertGroupedStridedSlice()
        pattern.find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'op_output', check_op_attrs=True)
        self.assertTrue(flag, resp)


class AddReshapeAfterStridedSliceTests(unittest.TestCase):
    def test_ss_1_shrink_last(self):
        slices = np.array([slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1), slice(0, 54, 1)])
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('placeholder_begin_data', 'sslice_1'),
                             ('placeholder_end_data', 'sslice_1'),
                             ('placeholder_stride_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('sslice_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                             'sslice_1': {'slices': slices,
                                          'shrink_axis_mask': [0, 0, 1, 0],
                                          'new_axis_mask': np.array([0, 0, 0, 0])},
                             'sslice_1_data': {'shape': np.array([1, 227, 54])},
                             }, nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_1'),
                                 ('placeholder_begin_data', 'sslice_1'),
                                 ('placeholder_end_data', 'sslice_1'),
                                 ('placeholder_stride_data', 'sslice_1'),
                                 ('sslice_1', 'sslice_1_data'),
                                 ('sslice_1_data', 'sslice_1/Squeeze_shrink'),
                                 ('sslice_1/squeeze_const', 'sslice_1/squeeze_const_data'),
                                 ('sslice_1/squeeze_const_data', 'sslice_1/Squeeze_shrink'),
                                 ('sslice_1/Squeeze_shrink', 'sslice_1/Squeeze_shrink_data'),
                                 ('sslice_1/Squeeze_shrink_data', 'op_output'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'sslice_1': {'slices': slices,
                                              'shrink_axis_mask': np.array([0, 0, 0, 0]),
                                              'new_axis_mask': np.array([0, 0, 0, 0])},
                                 'sslice_1_data': {'shape': np.array([1, 227, 1, 54])},
                                 'sslice_1/Squeeze_shrink_data': {'shape': np.array([1, 227, 54])}
                                 }, nodes_with_edges_only=True)

        ConvertGroupedStridedSlice().add_squeeze_for_shrink(graph, Node(graph, 'sslice_1'))

        (flag, resp) = compare_graphs(graph, graph_ref, 'sslice_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_ss_1_shrink(self):
        slices = np.array([slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1), slice(0, 54, 1)])

        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('placeholder_begin_data', 'sslice_2'),
                             ('placeholder_end_data', 'sslice_2'),
                             ('placeholder_stride_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_2_data', 'placeholder_2', {'out': 0}),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('sslice_2_data', 'op_output', {'out': 0})
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                             'sslice_2': {'slices': slices,
                                          'shrink_axis_mask': [0, 0, 1, 0],
                                          'new_axis_mask': np.array([0, 0, 0, 0])},
                             'sslice_2_data': {'shape': np.array([1, 227, 54])}
                             }, nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('placeholder_begin_data', 'sslice_2'),
                                 ('placeholder_end_data', 'sslice_2'),
                                 ('placeholder_stride_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2_data'),
                                 ('sslice_2_data', 'sslice_2/Squeeze_shrink'),
                                 ('sslice_2/squeeze_const', 'sslice_2/squeeze_const_data'),
                                 ('sslice_2/squeeze_const_data', 'sslice_2/Squeeze_shrink'),
                                 ('sslice_2/Squeeze_shrink', 'sslice_2/Squeeze_shrink_data'),
                                 ('sslice_2/Squeeze_shrink_data', 'placeholder_2', {'out': 0}),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('sslice_2/Squeeze_shrink_data', 'op_output', {'out': 0})
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'sslice_2': {'slices': slices,
                                              'shrink_axis_mask': np.array([0, 0, 0, 0]),
                                              'new_axis_mask': np.array([0, 0, 0, 0])},
                                 'sslice_2_data': {'shape': np.array([1, 227, 1, 54])},
                                 'sslice_2/squeeze_const': {'value': np.array([2])},
                                 'sslice_2/Squeeze_shrink_data': {'shape': np.array([1, 227, 54])},
                                 }, nodes_with_edges_only=True)

        ConvertGroupedStridedSlice().add_squeeze_for_shrink(graph, Node(graph, 'sslice_2'))

        (flag, resp) = compare_graphs(graph, graph_ref, 'op_output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_ss_2_shrink(self):
        slices = np.array([slice(0, 1, 1), slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1)])

        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('placeholder_begin_data', 'sslice_2'),
                             ('placeholder_end_data', 'sslice_2'),
                             ('placeholder_stride_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_2_data', 'placeholder_2', {'out': 0}),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('sslice_2_data', 'op_output', {'out': 0})
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                             'sslice_2': {'slices': slices,
                                          'shrink_axis_mask': np.array([0, 1, 0, 1]),
                                          'new_axis_mask': np.array([0, 0, 0, 0])},
                             'sslice_2_data': {'shape': np.array([1, 227])}
                             }, nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('placeholder_begin_data', 'sslice_2'),
                                 ('placeholder_end_data', 'sslice_2'),
                                 ('placeholder_stride_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2_data'),
                                 ('sslice_2_data', 'sslice_2/Squeeze_shrink'),
                                 ('sslice_2/squeeze_const', 'sslice_2/squeeze_const_data'),
                                 ('sslice_2/squeeze_const_data', 'sslice_2/Squeeze_shrink'),
                                 ('sslice_2/Squeeze_shrink', 'sslice_2/Squeeze_shrink_data'),
                                 ('sslice_2/Squeeze_shrink_data', 'placeholder_2', {'out': 0}),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('sslice_2/Squeeze_shrink_data', 'op_output', {'out': 0})
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'sslice_2': {'slices': slices,
                                              'shrink_axis_mask': np.array([0, 0, 0, 0]),
                                              'new_axis_mask': np.array([0, 0, 0, 0])},
                                 'sslice_2_data': {'shape': np.array([1, 1, 227, 1])},
                                 'sslice_2/squeeze_const': {'value': np.array([1, 3])},
                                 'sslice_2/Squeeze_shrink_data': {'shape': np.array([1, 227])},
                                 }, nodes_with_edges_only=True)

        ConvertGroupedStridedSlice().add_squeeze_for_shrink(graph, Node(graph, 'sslice_2'))

        (flag, resp) = compare_graphs(graph, graph_ref, 'op_output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_ss_1_new(self):
        slices = np.array([slice(0, 1, 1), slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 54, 1)])
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('placeholder_begin_data', 'sslice_2'),
                             ('placeholder_end_data', 'sslice_2'),
                             ('placeholder_stride_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_2_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'), ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                             'sslice_2': {'slices': slices,
                                          'shrink_axis_mask': np.array([0, 0, 0, 0, 0]),
                                          'new_axis_mask': np.array([0, 1, 0, 0, 0])},
                             'sslice_2_data': {'shape': np.array([1, 1, 227, 227, 54])}
                             }, nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('placeholder_begin_data', 'sslice_2'),
                                 ('placeholder_end_data', 'sslice_2'),
                                 ('placeholder_stride_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2_data'),
                                 ('sslice_2_data', 'sslice_2/Unsqueeze_new'),
                                 ('sslice_2/unsqueeze_const', 'sslice_2/unsqueeze_const_data'),
                                 ('sslice_2/unsqueeze_const_data', 'sslice_2/Unsqueeze_new'),
                                 ('sslice_2/Unsqueeze_new', 'sslice_2/Unsqueeze_new_data'),
                                 ('sslice_2/Unsqueeze_new_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data')],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'sslice_2': {'slices': slices,
                                              'shrink_axis_mask': np.array([0, 0, 0, 0, 0]),
                                              'new_axis_mask': np.array([0, 0, 0, 0, 0])},
                                 'sslice_2_data': {'shape': np.array([1, 227, 227, 54])},
                                 'sslice_2/unsqueeze_const': {'value': int64_array([1])},
                                 'sslice_2/Unsqueeze_new_data': {'shape': np.array([1, 1, 227, 227, 54])},
                                 }, nodes_with_edges_only=True)

        pattern = ConvertGroupedStridedSlice()
        pattern.add_unsqueeze_for_new(graph, Node(graph, 'sslice_2'))

        (flag, resp) = compare_graphs(graph, graph_ref, 'sslice_2_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_ss_shrink_new(self):
        slices = np.array([slice(0, 1, 1), slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1), slice(0, 54, 1)])

        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('placeholder_begin_data', 'sslice_2'),
                             ('placeholder_end_data', 'sslice_2'),
                             ('placeholder_stride_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_2_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('sslice_2_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                             'sslice_2': {'slices': slices,
                                          'shrink_axis_mask': np.array([0, 0, 0, 1, 0]),
                                          'new_axis_mask': np.array([0, 1, 0, 0, 0])},
                             'sslice_2_data': {'shape': np.array([1, 1, 227, 54])}
                             }, nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'sslice_2'),
                                 ('placeholder_begin_data', 'sslice_2'),
                                 ('placeholder_end_data', 'sslice_2'),
                                 ('placeholder_stride_data', 'sslice_2'),
                                 ('sslice_2', 'sslice_2_data'),
                                 ('sslice_2_data', 'sslice_2/Unsqueeze_new'),
                                 ('sslice_2/unsqueeze_const', 'sslice_2/unsqueeze_const_data'),
                                 ('sslice_2/unsqueeze_const_data', 'sslice_2/Unsqueeze_new'),
                                 ('sslice_2/Unsqueeze_new', 'sslice_2/Unsqueeze_new_data'),
                                 ('sslice_2/Unsqueeze_new_data', 'sslice_2/Squeeze_shrink'),
                                 ('sslice_2/squeeze_const', 'sslice_2/squeeze_const_data'),
                                 ('sslice_2/squeeze_const_data', 'sslice_2/Squeeze_shrink'),
                                 ('sslice_2/Squeeze_shrink', 'sslice_2/Squeeze_shrink_data'),
                                 ('sslice_2/Squeeze_shrink_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('sslice_2/Squeeze_shrink_data', 'op_output')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'sslice_2': {'slices': slices,
                                              'shrink_axis_mask': np.array([0, 0, 0, 0, 0]),
                                              'new_axis_mask': np.array([0, 0, 0, 0, 0])},
                                 'sslice_2_data': {'shape': np.array([1, 227, 1, 54])},
                                 'sslice_2/unsqueeze_const': {'value': int64_array([1])},
                                 'sslice_2/Unsqueeze_new_data': {'shape': np.array([1, 1, 227, 1, 54])},
                                 'sslice_2/squeeze_const': {'value': np.array([3])},
                                 'sslice_2/Squeeze_shrink_data': {'shape': np.array([1, 1, 227, 54])},
                                 }, nodes_with_edges_only=True)

        pattern = ConvertGroupedStridedSlice()
        pattern.add_squeeze_for_shrink(graph, Node(graph, 'sslice_2'))
        pattern.add_unsqueeze_for_new(graph, Node(graph, 'sslice_2'))

        (flag, resp) = compare_graphs(graph, graph_ref, 'op_output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # test case for strided slice that only shrinks dimension
    def test_ss_shrink_only(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('placeholder_begin_data', 'sslice_2'),
                             ('placeholder_end_data', 'sslice_2'),
                             ('placeholder_stride_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_2_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 1, 54])},
                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1), slice(0, 54, 1)]),
                                 'shrink_axis_mask': np.array([0, 0, 1, 0])},
                             'sslice_2_data': {'shape': np.array([1, 227, 54])}
                             }, nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'

        graph_ref = graph.copy()

        ConvertGroupedStridedSlice().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'sslice_2_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_ss_shrink_only_short(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('placeholder_begin_data', 'sslice_2'),
                             ('placeholder_end_data', 'sslice_2'),
                             ('placeholder_stride_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_2_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 1, 54])},
                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1), slice(0, 54, 1)]),
                                 'shrink_axis_mask': np.array([0, 0, 1])},
                             'sslice_2_data': {'shape': np.array([1, 227, 54])}
                             }, nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'

        graph_ref = graph.copy()

        ConvertGroupedStridedSlice().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'sslice_2_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_ss_shrink_only_long(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('placeholder_begin_data', 'sslice_2'),
                             ('placeholder_end_data', 'sslice_2'),
                             ('placeholder_stride_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('sslice_2_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 1, 54])},
                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 1, 1), slice(0, 54, 1)]),
                                 'shrink_axis_mask': np.array([0, 0, 1, 0, 0])},
                             'sslice_2_data': {'shape': np.array([1, 227, 54])}
                             }, nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'

        graph_ref = graph.copy()

        ConvertGroupedStridedSlice().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'sslice_2_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # test case with 2 strided slices with the same parameters but different outputs
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'sslice_1'),
                             ('sslice_1', 'sslice_1_data'),
                             ('placeholder_1_data', 'sslice_2'),
                             ('sslice_2', 'sslice_2_data'),
                             ('placeholder_1_data', 'sslice_3'),
                             ('sslice_3', 'sslice_3_data'),
                             ('sslice_1_data', 'concat_1'),
                             ('sslice_2_data', 'concat_1'),
                             ('sslice_3_data', 'placeholder_2'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output'),
                             ('placeholder_2_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},

                             'sslice_1': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 27, 1)])},
                             'sslice_1_data': {'shape': np.array([1, 227, 227, 27])},

                             'sslice_2': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(27, 54, 1)])},
                             'sslice_2_data': {'shape': np.array([1, 227, 227, 27])},

                             'sslice_3': {'slices': np.array(
                                 [slice(0, 1, 1), slice(0, 227, 1), slice(0, 227, 1), slice(0, 27, 1)])},
                             'sslice_3_data': {'shape': np.array([1, 227, 227, 27])},

                             'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                             }, nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'split_1'),
                                 ('split_1', 'split_1_data'),
                                 ('split_1', 'split_2_data'),
                                 ('split_1_data', 'concat_1'),
                                 ('split_2_data', 'concat_1'),
                                 ('split_1_data', 'placeholder_2'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output'),
                                 ('placeholder_2_data', 'op_output'),

                                 ('axis_const', 'axis_const_data'),
                                 ('split_dim_const', 'split_dim_const_data'),
                                 ('axis_const_data', 'split_1', {'in': 1}),
                                 ('split_dim_const_data', 'split_1', {'in': 2}),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 'axis_const': {'value': 3},
                                 'split_1_data': {'shape': np.array([1, 227, 227, 27])},
                                 'split_2_data': {'shape': np.array([1, 227, 227, 27])},
                                 'concat_1_data': {'shape': np.array([1, 227, 227, 54])},
                                 }, nodes_with_edges_only=True)

        ConvertGroupedStridedSlice().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)


if __name__ == '__main__':
    unittest.main()
