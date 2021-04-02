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
from extensions.ops.Cast import Cast
from extensions.ops.cumsum import CumSum
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs, mxnet_str_dtype_to_np
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, rename_node, Node
from mo.ops.const import Const


class CumSumFrontReplacer(FrontReplacementOp):
    op = 'MXNetCumSum'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        name = node.soft_get('name', node.id)
        axis = node.soft_get('axis', 0)

        rename_node(node=node, name=name + '/to_be_removed')
        cumsum_node = create_op_node_with_second_input(graph, CumSum, int64_array(axis),
                                                       {'name': name, 'reverse': False, 'exclusive': False})
        rename_node(cumsum_node, name)

        node.in_port(0).get_connection().set_destination(cumsum_node.in_port(0))
        if node.has_valid('mx_out_type') and node['mx_out_type'] is not None:
            rename_node(node=cumsum_node, name=name + '/CumSum')
            convert = Cast(graph, {'name': name, 'dst_type': node['mx_out_type']}).create_node()
            rename_node(convert, name)
            cumsum_node.out_port(0).connect(convert.in_port(0))
            return [convert.id]
        else:
            return [cumsum_node.id]
