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

from extensions.back.SpecialNodesFinalization import CreateConstNodesReplacement, RemoveConstToResult
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph_with_attrs


class CreateConstNodesReplacementTest(unittest.TestCase):
    nodes = [
        ('data_node', {'kind': 'data', 'shape': None, 'value': None}),
        ('next_node', {'kind': 'op'}),
    ]
    edges = [
        ('data_node', 'next_node')
    ]

    new_nodes = [
        ('const', {'kind': 'op', 'op': 'Const'}),
        ('const_data', {'kind': 'data'})
    ]
    new_edges = [
        ('const', 'data_node'),
        ('const_data', 'const')
    ]

    def test_one_node(self):
        """We should add Const node and data node."""
        shape = np.array([2, 3, 4])
        data = np.zeros(shape)
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes,
            edges_with_attrs=self.edges,
            update_nodes_attributes=[('data_node', {'shape': shape, 'value': data})]
        )
        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=self.nodes + self.new_nodes,
            edges_with_attrs=self.edges + self.new_edges,
            update_nodes_attributes=[('data_node', {'shape': shape, 'value': data}),
                                     ('const_data', {'shape': shape, 'value': data})]
        )
        tested_pattern = CreateConstNodesReplacement()
        tested_pattern.find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='next_node')
        self.assertTrue(flag, resp)

    def test_one_bin_node(self):
        """Nothing should happen."""
        shape = np.array([2, 3, 4])
        data = np.zeros(shape)
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes,
            edges_with_attrs=self.edges,
            update_nodes_attributes=[('data_node', {'shape': shape, 'value': data})],
            update_edge_attrs={('data_node', 'next_node', 0): {'bin': 0}},
        )
        tested_pattern = CreateConstNodesReplacement()
        tested_pattern.find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph, last_node='next_node')
        self.assertTrue(flag, resp)

    def test_two_nodes_with_bin(self):
        """Test case for data node with 2 consumers with bin edge attr.
        Nothing should happened."""
        shape = np.array([2, 3, 4])
        data = np.zeros(shape)
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes + [('next_node_2', {'kind': 'op'})],
            edges_with_attrs=self.edges + [('data_node', 'next_node_2')],
            update_nodes_attributes=[('data_node', {'shape': shape, 'value': data})],
            update_edge_attrs={('data_node', 'next_node', 0): {'bin': 0}, ('data_node', 'next_node_2', 0): {'bin': 0}},
        )
        tested_pattern = CreateConstNodesReplacement()
        tested_pattern.find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph, last_node='next_node')
        self.assertTrue(flag, resp)

    def test_two_nodes_one_bin(self):
        """Test case for two output nodes, one with 'bin' parameter, other without."""
        shape = np.array([2, 3, 4])
        data = np.zeros(shape)
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes + [('next_node_2', {'kind': 'op'})],
            edges_with_attrs=self.edges + [('data_node', 'next_node_2')],
            update_nodes_attributes=[('data_node', {'shape': shape, 'value': data})],
            update_edge_attrs={('data_node', 'next_node', 0): {'bin': 0}},
        )
        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=self.nodes + self.new_nodes + [('next_node_2', {'kind': 'op'})],
            edges_with_attrs=self.edges + self.new_edges + [('data_node', 'next_node_2')],
            update_nodes_attributes=[('data_node', {'shape': shape, 'value': data}),
                                     ('const_data', {'shape': shape, 'value': data})]
        )
        tested_pattern = CreateConstNodesReplacement()
        tested_pattern.find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='next_node')
        self.assertTrue(flag, resp)


class RemoveConstToResultReplacementTest(unittest.TestCase):
    def test_only_consumer(self):
        """Result node is only consumer of Const data node"""
        nodes = [
            ('const_node', {'type': 'Const', 'kind': 'op'}),
            ('const_data', {'kind': 'data', 'value': np.array(5)}),
            ('result_node', {'type': 'Result', 'kind': 'op'}),

            ('placeholder_1', {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'}),
            ('placeholder_1_data', {'kind': 'data'}),
            ('relu_1', {'type': 'ReLU', 'kind': 'op', 'op': 'ReLU'}),
            ('relu_1_data', {'kind': 'data'}),
        ]
        edges = [
            ('const_node', 'const_data'),
            ('const_data', 'result_node'),

            ('placeholder_1', 'placeholder_1_data'),
            ('placeholder_1_data', 'relu_1'),
            ('relu_1', 'relu_1_data')
        ]
        new_nodes=[
            ('placeholder_1', {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'}),
            ('placeholder_1_data', {'kind': 'data'}),
            ('relu_1', {'type': 'ReLU', 'kind': 'op', 'op': 'ReLU'}),
            ('relu_1_data', {'kind': 'data'}),
        ]
        new_edges=[
            ('placeholder_1', 'placeholder_1_data'),
            ('placeholder_1_data', 'relu_1'),
            ('relu_1', 'relu_1_data')
        ]

        graph = build_graph_with_attrs(
            nodes_with_attrs=nodes,
            edges_with_attrs=edges,
        )
        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=new_nodes,
            edges_with_attrs=new_edges,
        )
        tested_pattern = RemoveConstToResult()
        tested_pattern.find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='relu_1_data')
        self.assertTrue(flag, resp)
        self.assertNotIn('const_node', graph.node)
        self.assertNotIn('const_data', graph.node)
        self.assertNotIn('result_node', graph.node)

    def test_two_consumers(self):
        """Const data node has two consumers: Result and ReLu"""
        nodes = [
            ('const_node', {'type': 'Const', 'kind': 'op'}),
            ('const_data', {'kind': 'data', 'value': np.array(5)}),
            ('result_node', {'type': 'Result', 'kind': 'op'}),
            ('relu_1', {'type': 'ReLU', 'kind': 'op', 'op': 'ReLU'}),
            ('relu_1_data', {'kind': 'data'}),
        ]
        edges = [
            ('const_node', 'const_data'),
            ('const_data', 'result_node'),
            ('const_data', 'relu_1'),
            ('relu_1', 'relu_1_data')
        ]
        new_nodes=[
            ('const_node', {'type': 'Const', 'kind': 'op'}),
            ('const_data', {'kind': 'data'}),
            ('relu_1', {'type': 'ReLU', 'kind': 'op', 'op': 'ReLU'}),
            ('relu_1_data', {'kind': 'data'}),
        ]
        new_edges=[
            ('const_node', 'const_data'),
            ('const_data', 'relu_1'),
            ('relu_1', 'relu_1_data')
        ]

        graph = build_graph_with_attrs(
            nodes_with_attrs=nodes,
            edges_with_attrs=edges,
        )
        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=new_nodes,
            edges_with_attrs=new_edges,
        )
        tested_pattern = RemoveConstToResult()
        tested_pattern.find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='relu_1_data')
        self.assertTrue(flag, resp)
        self.assertNotIn('result_node', graph.node)
