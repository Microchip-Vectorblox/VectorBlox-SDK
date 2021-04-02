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

from extensions.front.onnx.AttributedSliceToSlice import AttributedSliceToSliceReplacer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op_with_empty_data, result, const, connect_front


@generator
class SliceReplacerTest(unittest.TestCase):
    @generate(*[
       {'op': 'AttributedSlice', 'type': None, 'starts': np.array([0, 0]), 'ends': np.array([1, -1]), 'axes': np.array([0, 1])}
    ])
    def test_attributed_slice_replacer(self, attributed_slice_attrs):
        nodes = {
            **regular_op_with_empty_data('input', {'type': 'Parameter'}),
            **regular_op_with_empty_data('attributed_slice', attributed_slice_attrs),
            **result(),

            # nodes after replacement
            **const('start', np.array([0, 0])),
            **const('end', np.array([1, -1])),
            **const('axis', np.array(np.array([0, 1]))),
            **regular_op_with_empty_data('slice', {'op': 'Slice', 'type': None}),
        }

        graph = build_graph(nodes_attrs=nodes, edges=[
            ('input', 'attributed_slice'),
            ('attributed_slice', 'output'),
        ], nodes_with_edges_only=True)
        graph.stage = 'front'

        AttributedSliceToSliceReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes, edges=[
            ('input', 'slice'),
            *connect_front('start', '1:slice'),
            *connect_front('end', '2:slice'),
            *connect_front('axis', '3:slice'),
            ('slice', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
