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
import numpy as np

from extensions.front.pass_separator import FrontStart
from extensions.front.restore_ports import RestorePorts
from extensions.ops.loop import Loop
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node
from mo.ops.const import Const
from mo.ops.unsqueeze import Unsqueeze


class ONNXLoopNormalize(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [FrontStart]

    def run_after(self):
        return [RestorePorts]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Loop'):
            self.normalize_body_graph(node)

    @staticmethod
    def normalize_body_graph(loop_node: Node):
        loop_name = loop_node.soft_get('name', loop_node.id)
        # connect "trip count" input if it is not connected with default value "Infinity" (-1)
        if not loop_node.is_in_port_connected(0):
            loop_node.add_input_port(0, skip_if_exist=True)
            Const(loop_node.graph, {'name': loop_name + '/trip_count', 'value': int64_array(-1)}).\
                create_node().out_port(0).connect(loop_node.in_port(0))

        # connect "execution condition" input if it is not connected with default value True
        if not loop_node.is_in_port_connected(1):
            loop_node.add_input_port(1, skip_if_exist=True)
            Const(loop_node.graph, {'name': loop_name + '/execution_cond', 'value': np.array(True, dtype=np.bool)}).\
                create_node().out_port(0).connect(loop_node.in_port(1))

        # scan output need Unsqueeze over axis 0
        for record in loop_node.output_port_map:
            body_node = Loop.get_body_node_by_internal_id(loop_node, record['internal_layer_id'])
            assert body_node is not None
            assert body_node.soft_get('type') == 'Result'

            if record['axis'] is not None:
                unsqueeze = create_op_with_const_inputs(loop_node.body, Unsqueeze, {1: int64_array([0])})
                body_node.in_port(0).get_connection().insert_node(unsqueeze)

        Loop.normalize_input_output_ports(loop_node)
