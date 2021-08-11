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

from mo.graph.graph import Graph
from mo.utils.error import Error
from mo.utils.graph import bfs_search, is_connected_component, sub_graph_between_nodes


class TestGraphUtils(unittest.TestCase):
    def test_simple_dfs(self):
        graph = Graph()
        graph.add_nodes_from(list(range(1, 5)))
        graph.add_edges_from([(1, 2), (1, 3), (3, 4)])

        visited = set()
        order = graph.dfs(1, visited)
        self.assertTrue(order == [4, 3, 2, 1] or order == [2, 4, 3, 1])

    def test_bfs_search_default_start_nodes(self):
        """
        Check that BFS automatically determines input nodes and start searching from them.
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 6)))
        graph.add_edges_from([(1, 3), (2, 3), (3, 4), (4, 5)])

        order = bfs_search(graph)
        self.assertTrue(order == [1, 2, 3, 4, 5] or order == [2, 1, 3, 4, 5])

    def test_bfs_search_specific_start_nodes(self):
        """
        Check that BFS stars from the user defined nodes and doesn't go in backward edge direction.
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 3), (2, 3), (3, 4), (4, 5), (6, 1)])

        order = bfs_search(graph, [1])
        self.assertTrue(order == [1, 3, 4, 5])

    def test_is_connected_component_two_separate_sub_graphs(self):
        """
        Check that if there are two separate sub-graphs the function returns False.
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])
        self.assertFalse(is_connected_component(graph, list(range(1, 7))))
        self.assertFalse(is_connected_component(graph, [1, 3]))
        self.assertFalse(is_connected_component(graph, [6, 4]))
        self.assertFalse(is_connected_component(graph, [2, 5]))

    def test_is_connected_component_two_separate_sub_graphs_divided_by_ignored_node(self):
        """
        Check that if there are two separate sub-graphs the function connected by an edge going through the ignored node
        then the function returns False.
        """
        graph = Graph()
        node_names = list(range(1, 8))
        graph.add_nodes_from(node_names)
        graph.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6), (1, 7), (7, 4)])
        self.assertFalse(is_connected_component(graph, list(range(1, 7))))

    def test_is_connected_component_connected(self):
        """
        Check that if the sub-graph is connected.
        """
        graph = Graph()
        node_names = list(range(1, 8))
        graph.add_nodes_from(node_names)
        graph.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6), (1, 7), (7, 4)])
        self.assertTrue(is_connected_component(graph, list(range(1, 8))))

    def test_is_connected_component_edges_direction_is_ignored(self):
        """
        Check that edges direction is ignored when checking for the connectivity.
        """
        graph = Graph()
        node_names = list(range(1, 5))
        graph.add_nodes_from(node_names)
        graph.add_edges_from([(2, 1), (2, 3), (4, 3)])
        self.assertTrue(is_connected_component(graph, node_names))
        self.assertTrue(is_connected_component(graph, [2, 1]))
        self.assertTrue(is_connected_component(graph, [4, 2, 3]))

    def test_is_connected_component_edges_direction_is_ignored_not_connected(self):
        """
        Check that edges direction is ignored when checking for the connectivity. In this case the graph is not
        connected.
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 5)))
        graph.add_edges_from([(2, 1), (2, 3), (4, 3)])
        self.assertFalse(is_connected_component(graph, [1, 2, 4]))
        self.assertFalse(is_connected_component(graph, [1, 4]))
        self.assertFalse(is_connected_component(graph, [2, 4]))
        self.assertFalse(is_connected_component(graph, [3, 4, 1]))

    def test_sub_graph_between_nodes_include_incoming_edges_for_internal_nodes(self):
        """
        Check that the function adds input nodes for the internal nodes of the graph. For example, we need to add node 5
        and 6 in the case below if we find match from node 1 till node 4.
        6 -> 5 ->
                 \
            1 -> 2 -> 3 -> 4
        :return:
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2), (6, 5)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [1], [4])
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), list(range(1, 7)))

        sub_graph_nodes = sub_graph_between_nodes(graph, [1], [2])
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), [1, 2, 5, 6])

    def test_sub_graph_between_nodes_do_not_include_incoming_edges_for_input_nodes(self):
        """
        Check that the function doesn't add input nodes for the start nodes of the sub-graph. For example, we do not
        need to add node 5 in the case below if we find match from node 1 till node 4.
          5->
             \
        1 -> 2 -> 3 -> 4
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 6)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [2], [4])
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), [2, 3, 4])

    def test_sub_graph_between_nodes_placeholder_included(self):
        """
        Check that the function doesn't allow to add Placeholders to the sub-graph. 5 is the Placeholder op.
          5->
             \
        1 -> 2 -> 3 -> 4
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 6)))
        graph.node[5]['op'] = 'Parameter'
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2)])
        self.assertRaises(Error, sub_graph_between_nodes, graph, [1], [4])

    def test_sub_graph_between_nodes_placeholder_excluded(self):
        """
        Check that the function do not check that node is Placeholders for the nodes not included into the sub-graph.
        For example, node 5 is Placeholder but it is not included into the sub-graph, so this attribute is ignored.
          5->
             \
        1 -> 2 -> 3 -> 4
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 6)))
        graph.node[5]['op'] = 'Parameter'
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [2], [4])
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), [2, 3, 4])

    def test_sub_graph_between_nodes_multiple_inputs(self):
        """
        Check that the function works correctly when multiple inputs specified.
          5->
             \
        1 -> 2 -> 3 -> 4
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 6)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [2, 5], [4])
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), sorted([2, 3, 4, 5]))

    def test_sub_graph_between_nodes_branches_included(self):
        """
        Check that the function works correctly for tree like structures.
        1 -> 2 -> 3 -> 4
             \
             5 -> 6
            / \
        9 ->   -> 7 -> 8
        """
        graph = Graph()
        node_names = list(range(1, 10))
        graph.add_nodes_from(node_names)
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (5, 7), (7, 8), (9, 5)])
        self.assertListEqual(sorted(sub_graph_between_nodes(graph, [1], [4])), node_names)
        self.assertListEqual(sorted(sub_graph_between_nodes(graph, [1], [6])), node_names)
        self.assertListEqual(sorted(sub_graph_between_nodes(graph, [1], [8])), node_names)
        # all nodes except 4 because it is a child of end node
        self.assertListEqual(sorted(sub_graph_between_nodes(graph, [1], [3])), [n for n in node_names if n != 4])
        # all nodes except 1 because it is a parent node child of start node. The nodes 3 and 4 must be added because
        # after merging node 2 into sub-graph the node 2 will be removed and it is not known how to calculate the tensor
        # between node 2 and 3.
        self.assertListEqual(sorted(sub_graph_between_nodes(graph, [2], [8])), [n for n in node_names if n != 1])

    def test_sub_graph_between_nodes_control_flow_included(self):
        """
        Check that the function works correctly for case when control flow edges must be traversed (edge 5 -> 2).
        6 -> 5->
                \
           1 -> 2 -> 3 -> 4
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2, {'control_flow_edge': True}), (6, 5)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [1], [4], include_control_flow=True)
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), sorted([1, 2, 3, 4, 5, 6]))

    def test_sub_graph_between_nodes_control_flow_not_included(self):
        """
        Check that the function works correctly for case when control flow edges should not be traversed (edge 5 -> 2).
        6 -> 5->
                \
           1 -> 2 -> 3 -> 4
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2, {'control_flow_edge': True}), (6, 5)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [1], [4], include_control_flow=False)
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), sorted([1, 2, 3, 4]))

    def test_sub_graph_between_nodes_control_flow_included_forward(self):
        """
        Check that the function works correctly for case when control flow edges should not be traversed (edge 3 -> 5).
           1 -> 2 -> 3 -> 4
                      \
                       -> 5 -> 6
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 5, {'control_flow_edge': True}), (5, 6)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [1], [4], include_control_flow=True)
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), sorted([1, 2, 3, 4, 5, 6]))

    def test_sub_graph_between_nodes_control_flow_not_included_forward(self):
        """
        Check that the function works correctly for case when control flow edges should not be traversed (edge 3 -> 5).
           1 -> 2 -> 3 -> 4
                      \
                       -> 5 -> 6
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 5, {'control_flow_edge': True}), (5, 6)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [1], [4], include_control_flow=False)
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), sorted([1, 2, 3, 4]))
