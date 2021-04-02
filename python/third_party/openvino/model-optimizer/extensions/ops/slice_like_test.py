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

from extensions.ops.slice_like import SliceLike
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'input': {'kind': 'op', 'op': 'Const'},
    'input_data': {'kind': 'data', 'shape': int64_array([3, 4]), 'value': np.arange(1, 13).reshape([3, 4])},
    'shape_like': {'kind': 'op', 'op': 'Const', 'shape': int64_array([2, 3]), 'value': None},
    'shape_like_data': {'kind': 'data', 'shape': int64_array([2, 3]), 'value': None},
    'slice_like': {'kind': 'op', 'op': 'slice_data'},
    'out_data': {'kind': 'data', 'shape': None, 'value': None}
}

edges = [
    ('input', 'input_data'),
    ('input_data', 'slice_like', {'in': 0}),
    ('shape_like', 'shape_like_data'),
    ('shape_like_data', 'slice_like', {'in': 1}),
    ('slice_like', 'out_data')
]


class SliceLikeTest(unittest.TestCase):

    def test_1(self):
        graph = build_graph(nodes_attributes, edges, {'slice_like': {'axes': None}})
        slice_like = Node(graph, 'slice_like')
        SliceLike.infer(slice_like)
        ref_shape = int64_array([2, 3])
        ref_value = np.array([[1, 2, 3], [5, 6, 7]])
        res_shape = graph.node['out_data']['shape']
        res_value = graph.node['out_data']['value']
        self.assertTrue(np.array_equal(res_shape, ref_shape))
        self.assertTrue(np.array_equal(res_value, ref_value))

    def test_2(self):
        graph = build_graph(nodes_attributes, edges, {'slice_like': {'axes': (0, 1)}})
        slice_like = Node(graph, 'slice_like')
        SliceLike.infer(slice_like)
        ref_shape = int64_array([2, 3])
        ref_value = np.array([[1, 2, 3], [5, 6, 7]])
        res_shape = graph.node['out_data']['shape']
        res_value = graph.node['out_data']['value']
        self.assertTrue(np.array_equal(res_shape, ref_shape))
        self.assertTrue(np.array_equal(res_value, ref_value))

    def test_3(self):
        graph = build_graph(nodes_attributes, edges, {'slice_like': {'axes': (0,)}})
        slice_like = Node(graph, 'slice_like')
        SliceLike.infer(slice_like)
        ref_shape = int64_array([2, 4])
        ref_value = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        res_shape = graph.node['out_data']['shape']
        res_value = graph.node['out_data']['value']
        self.assertTrue(np.array_equal(res_shape, ref_shape))
        self.assertTrue(np.array_equal(res_value, ref_value))

    def test_4(self):
        graph = build_graph(nodes_attributes, edges, {'slice_like': {'axes': (-1,)}})
        slice_like = Node(graph, 'slice_like')
        SliceLike.infer(slice_like)
        ref_shape = int64_array([3, 3])
        ref_value = np.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
        res_shape = graph.node['out_data']['shape']
        res_value = graph.node['out_data']['value']
        self.assertTrue(np.array_equal(res_shape, ref_shape))
        self.assertTrue(np.array_equal(res_value, ref_value))
