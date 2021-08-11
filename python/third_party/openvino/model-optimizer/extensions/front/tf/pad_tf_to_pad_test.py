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

import unittest

import numpy as np

from extensions.front.tf.pad_tf_to_pad import PadTFToPad
from mo.front.common.partial_infer.utils import int64_array, float_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, const

nodes_attributes = {
    'placeholder': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'tfpad': {'type': None, 'kind': 'op', 'op': 'TFPad', 'mode': 'constant', 'name': 'tfpad_name'},
    **const('paddings', int64_array([1, 2, 3, 4, 5, 6]).reshape([3, 2])),
    **const('fill', float_array(5.75)),
    'result': {'type': 'Result', 'value': None, 'kind': 'op', 'op': 'Result'},

    # new Pad layer and sub-graph
    'pad': {'type': 'Pad', 'kind': 'op', 'op': 'Pad', 'mode': 'constant'},
    'transpose': {'type': 'Transpose', 'kind': 'op', 'op': 'Transpose'},
    **const('transpose_order', int64_array([1, 0])),
    'split': {'type': 'Split', 'kind': 'op', 'op': 'Split', 'num_splits': 2},
    **const('split_axis', int64_array(0)),
    'squeeze_1': {'type': 'Squeeze', 'kind': 'op', 'op': 'Squeeze'},
    **const('squeeze_1_axis', int64_array([0])),
    'squeeze_2': {'type': 'Squeeze', 'kind': 'op', 'op': 'Squeeze'},
    **const('squeeze_2_axis', int64_array([0])),

    **const('pad_fill', np.array(0.0)),
}

common_edges = [('placeholder', 'pad', {'in': 0, 'out': 0}),

                ('paddings', 'transpose', {'in': 0, 'out': 0}),
                ('transpose_order', 'transpose', {'in': 1, 'out': 0}),

                ('transpose', 'split', {'in': 0, 'out': 0}),
                ('split_axis', 'split', {'in': 1, 'out': 0}),

                ('split', 'squeeze_1', {'in': 0, 'out': 0}),
                ('squeeze_1_axis', 'squeeze_1', {'in': 1, 'out': 0}),

                ('split', 'squeeze_2', {'in': 0, 'out': 1}),
                ('squeeze_2_axis', 'squeeze_2', {'in': 1, 'out': 0}),

                ('squeeze_1', 'pad', {'in': 1, 'out': 0}),
                ('squeeze_2', 'pad', {'in': 2, 'out': 0}),

                ('pad', 'result')
                ]


class PadTFToPadTest(unittest.TestCase):
    def _run_test(self, graph, graph_ref):
        graph.graph['layout'] = 'NHWC'
        graph.stage = 'front'

        replacer = PadTFToPad()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Pad')[0]]['name'] == 'tfpad_name')
        self.assertTrue(flag, resp)

    def test_2_inputs(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'tfpad', {'in': 0, 'out': 0}),
                             ('paddings', 'tfpad', {'in': 1, 'out': 0}),
                             ('tfpad', 'result', {'in': 0, 'out': 0}),
                             ],
                            {}, nodes_with_edges_only=True)
        graph.get_op_nodes(op='TFPad')[0].add_input_port(2)

        graph_ref = build_graph(nodes_attributes, common_edges + [('pad_fill', 'pad', {'in': 3, 'out': 0})],
                                {}, nodes_with_edges_only=True)
        self._run_test(graph, graph_ref)

    def test_3_inputs(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'tfpad', {'in': 0, 'out': 0}),
                             ('paddings', 'tfpad', {'in': 1, 'out': 0}),
                             ('fill', 'tfpad', {'in': 2, 'out': 0}),
                             ('tfpad', 'result', {'in': 0, 'out': 0}),
                             ],
                            {}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes, common_edges + [('fill', 'pad', {'in': 3, 'out': 0})],
                                {}, nodes_with_edges_only=True)

        self._run_test(graph, graph_ref)

    def test_3_inputs_with_non_constant_pad(self):
        updated_paddings_attrs = {'type': 'Parameter', 'op': 'Parameter', 'value': None}
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'tfpad', {'in': 0, 'out': 0}),
                             ('paddings', 'tfpad', {'in': 1, 'out': 0}),
                             ('fill', 'tfpad', {'in': 2, 'out': 0}),
                             ('tfpad', 'result', {'in': 0, 'out': 0}),
                             ],
                            {'paddings': updated_paddings_attrs}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes, common_edges + [('fill', 'pad', {'in': 3, 'out': 0})],
                                {'paddings': updated_paddings_attrs}, nodes_with_edges_only=True)

        self._run_test(graph, graph_ref)

    def test_2_inputs_non_constant_mode(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'tfpad', {'in': 0, 'out': 0}),
                             ('paddings', 'tfpad', {'in': 1, 'out': 0}),
                             ('tfpad', 'result', {'in': 0, 'out': 0}),
                             ],
                            {'tfpad': {'mode': 'reflect'}}, nodes_with_edges_only=True)
        graph.get_op_nodes(op='TFPad')[0].add_input_port(2)

        graph_ref = build_graph(nodes_attributes, common_edges,
                                {'pad': {'mode': 'reflect'}}, nodes_with_edges_only=True)
        self._run_test(graph, graph_ref)
