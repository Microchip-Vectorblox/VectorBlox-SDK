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

from extensions.front.AttributedPadToPad import AttributedPadToPad
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, const

nodes_attributes = {
    'placeholder': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'attr_pad': {'type': None, 'kind': 'op', 'op': 'AttributedPad', 'mode': 'constant', 'name': 'attr_pad',
                 'pads': int64_array([1, 2, 3, 4, 5, 6]).reshape([3, 2]), 'fill_value': 0.75},
    'result': {'type': 'Result', 'value': None, 'kind': 'op', 'op': 'Result'},

    # new Pad layer and inputs
    'pad': {'type': 'Pad', 'kind': 'op', 'op': 'Pad', 'mode': 'constant'},
    **const('pad_begin', int64_array([1, 3, 5])),
    **const('pad_end', int64_array([2, 4, 6])),
    **const('pad_fill', np.array(0.75)),
}


class AttributedPadToPadTest(unittest.TestCase):
    def test_mode_constant(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'attr_pad', {'in': 0, 'out': 0}),
                             ('attr_pad', 'result', {'in': 0, 'out': 0}),
                             ],
                            {}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'pad', {'in': 0, 'out': 0}),
                                 ('pad_begin', 'pad', {'in': 1, 'out': 0}),
                                 ('pad_end', 'pad', {'in': 2, 'out': 0}),
                                 ('pad_fill', 'pad', {'in': 3, 'out': 0}),
                                 ('pad', 'result')
                                 ],
                                {}, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = AttributedPadToPad()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Pad')[0]]['name'] == 'attr_pad')

    def test_mode_non_constant(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'attr_pad', {'in': 0, 'out': 0}),
                             ('attr_pad', 'result', {'in': 0, 'out': 0}),
                             ],
                            {'attr_pad': {'mode': 'reflect'}}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'pad', {'in': 0, 'out': 0}),
                                 ('pad_begin', 'pad', {'in': 1, 'out': 0}),
                                 ('pad_end', 'pad', {'in': 2, 'out': 0}),
                                 ('pad', 'result')
                                 ],
                                {'pad': {'mode': 'reflect'}}, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = AttributedPadToPad()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Pad')[0]]['name'] == 'attr_pad')
