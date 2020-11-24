"""
 Copyright (c) 2020 Intel Corporation

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

from extensions.front.mxnet.mx_reshape_to_reshape import MXReshapeToReshape
from extensions.ops.Reverse import Reverse
from extensions.ops.mxreshape import MXReshape
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze


class MXReshapeReverse(FrontReplacementOp):
    """
    If reshape layer with reverse True, special values will inferred from right to left.
    The Replacer simulate the behavior. The replaced subgraph reverse input data and special dims,
    and after reshape reverse output result to backward.
    Resulting subgraph:  reshape(reverse=True) -> reverse - reshape(reverse=False) -reverse subgraph.
    """
    op = 'MXReshape'
    enabled = True

    def run_before(self):
        return [MXReshapeToReshape]

    def replace_sub_graph(self, graph: Graph, match: dict):
        mxreshape = match['op']
        if not mxreshape.reverse:
            return

        shape_node = Shape(graph, dict(name=mxreshape.id + '/Shape')).create_node()
        forward_reverse_unsqueeze_node = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0]),
                                                                          dict(name=str(mxreshape.id) + '/ForwardUnsqueeze'))
        forward_reverse_node = Reverse(graph, dict(name=mxreshape.id + '/ForwardReverse', axis=1)).create_node()

        forward_reverse_squeeze_node = create_op_node_with_second_input(graph, Squeeze, int64_array([0]),
                                                                        dict(name=str(mxreshape.id) + '/ForwardSqueeze'))
        reshape_node = Reshape(graph, dict(name=mxreshape.id + '/Reshape')).create_node()
        shape_node.in_port(0).connect(mxreshape.in_port(0).get_source())
        mxreshape.in_port(0).get_connection().set_destination(reshape_node.in_port(0))

        forward_reverse_unsqueeze_node.in_port(0).connect(shape_node.out_port(0))
        forward_reverse_node.in_port(0).connect(forward_reverse_unsqueeze_node.out_port(0))
        forward_reverse_squeeze_node.in_port(0).connect(forward_reverse_node.out_port(0))
        reshape_node.in_port(1).connect(forward_reverse_squeeze_node.out_port(0))

        reshape_shape_node = create_op_node_with_second_input(graph, Reshape, int64_array(np.flip(mxreshape.dim, 0)),
                                                              dict(name=str(mxreshape.id) + '/ReshapeShape'))
        if np.sum(np.in1d([-2, -3, -4], mxreshape.dim), axis=0):
            reshape_shape_node = MXReshape(graph, dict(name=mxreshape.id + '/Reshape',
                                     dim=int64_array(np.flip(mxreshape.dim, 0)))).create_node()

        reshape_shape_node.in_port(0).connect(reshape_node.out_port(0))

        backward_shape_node = Shape(graph, dict(name=mxreshape.id + '/BackwardShape')).create_node()
        backward_reverse_unsqueeze_node = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0]),
                                                                           dict(name=str(mxreshape.id) + '/BackwardUnsqueeze'))
        backward_reverse_node = Reverse(graph, dict(name=mxreshape.id + '/BackwardReverse', axis=1)).create_node()
        backward_reverse_squeeze_node = create_op_node_with_second_input(graph, Squeeze, int64_array([0]),
                                                                         dict(name=str(mxreshape.id) + '/BackwardSqueeze'))
        backward_reshape_node = Reshape(graph, dict(name=mxreshape.id + '/BackwardReshape')).create_node()

        backward_shape_node.in_port(0).connect(reshape_shape_node.out_port(0))
        backward_reverse_unsqueeze_node.in_port(0).connect(backward_shape_node.out_port(0))
        backward_reverse_node.in_port(0).connect(backward_reverse_unsqueeze_node.out_port(0))
        backward_reverse_squeeze_node.in_port(0).connect(backward_reverse_node.out_port(0))

        backward_reshape_node.in_port(0).connect(reshape_shape_node.out_port(0))
        backward_reshape_node.in_port(1).connect(backward_reverse_squeeze_node.out_port(0))

        mxreshape.out_port(0).get_connection().set_source(backward_reshape_node.out_port(0))
