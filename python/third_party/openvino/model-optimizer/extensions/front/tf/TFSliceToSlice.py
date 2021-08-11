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

import numpy as np

from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Add, Equal
from extensions.ops.select import Select
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.ops.shape import Shape
from mo.ops.slice import Slice


class TFSliceToSliceReplacer(FrontReplacementOp):
    """
    This transformation converts TFSlice to internal Slice operation.
    TFSlice has 'size' on the second input while Slice has 'ends', therefore we insert Add(begin, size).
    size[i] == -1 is a magic number that means take the whole range along axis i up to the end.
    To process the case when size[i] == -1 we insert subgraph with ShapeOf.
    """
    op = 'TFSlice'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        tf_slice_node = match['op']
        slice_name = tf_slice_node.soft_get('name', tf_slice_node.id)
        slice_node = Slice(graph).create_node()
        rename_nodes([(tf_slice_node, slice_name + '/to_be_removed'), (slice_node, slice_name)])
        ends_node = Add(graph, {'name': slice_name + '/ends'}).create_node()

        # reconnect input, begin, and size from TFSlice to the subgraph with Slice
        tf_slice_node.in_port(0).get_connection().set_destination(slice_node.in_port(0))
        tf_slice_node.in_port(1).get_connection().set_destination(slice_node.in_port(1))
        tf_slice_node.in_port(2).get_connection().set_destination(ends_node.in_port(0))
        slice_node.in_port(1).get_connection().add_destination(ends_node.in_port(1))

        max_ends = Shape(graph, {'name': slice_name + '/ShapeOf'}).create_node()
        slice_node.in_port(0).get_connection().add_destination(max_ends.in_port(0))

        # check if size[i] == -1, will be applied elementwisely: len(size) = len(begin) = input_rank
        where_max_ends_is_needed = create_op_with_const_inputs(graph, Equal, {0: int64_array(-1)},
                                                               {'name': slice_name + '/where_max_ends_is_needed'})
        ends_node.in_port(0).get_connection().add_destination(where_max_ends_is_needed.in_port(1))
        # select requires equal dtypes, need to convert ends to I64
        ends_casted_to_i64 = Cast(graph, {'name': slice_name + '/CastToI64',
                                          'dst_type': np.int64}).create_node([ends_node])
        # if size[i] == 1 then take max_ends values
        correct_ends = Select(graph, {'name': slice_name + '/chosen_ends'}).create_node([where_max_ends_is_needed,
                                                                                        max_ends, ends_casted_to_i64])
        correct_ends.out_port(0).connect(slice_node.in_port(2))

        tf_slice_node.out_port(0).get_connection().set_source(slice_node.out_port(0))
