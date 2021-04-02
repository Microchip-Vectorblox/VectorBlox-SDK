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

from extensions.back.insert_compatibility_l2normalization import CompatibilityL2NormalizationPattern
from extensions.ops.elementwise import Mul
from extensions.ops.normalize_l2 import NormalizeL2Op
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_node


class NormalizeToNormalizeL2(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [CompatibilityL2NormalizationPattern]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('normalize', {'type': 'Normalize'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['normalize']

        # rename normalize node since it will be no longer output node after the transformation
        output_name = node.soft_get('name', node.id)
        normalizel2_name = output_name + '/normalizel2'
        rename_node(node, normalizel2_name)

        assert node.in_port(0).data.get_shape().size in [2, 3, 4]
        assert node.has_valid('across_spatial')
        assert node.has_valid('channel_shared')
        assert node.has_valid('eps')

        if 'bin' in node.in_edge(1):
            del node.in_edge(1)['bin']

        weights = node.in_port(1).data.get_value()
        assert weights is not None
        # in the code below we intentionally use get_source() to get the out port. Because updating the out port will
        # update the Const node 'value' and 'shape' attributes
        if node.channel_shared or all(weights == weights[0]):
            node.in_port(1).get_source().data.set_value(np.array([weights[0]]))
        else:
            new_shape = np.ones((len(node.in_port(0).data.get_shape())), dtype=np.int64)
            new_shape[1] = -1
            node.in_port(1).get_source().data.set_value(np.array(weights).reshape(new_shape))

        mul = Mul(graph, {'name': output_name}).create_node()
        rename_node(mul, output_name)

        if not node.across_spatial:
            axes = int64_array([1])
        else:
            axes = int64_array(np.arange(start=1, stop=node.in_port(0).data.get_shape().size))

        normalizel2 = create_op_with_const_inputs(graph, NormalizeL2Op, {1: axes}, {'eps_mode': 'add', 'eps': node.eps})

        node.out_port(0).get_connection().set_source(mul.out_port(0))
        node.in_port(1).get_connection().get_source().connect(mul.in_port(1))
        normalizel2.out_port(0).connect(mul.in_port(0))
        node.in_port(0).get_connection().set_destination(normalizel2.in_port(0))
