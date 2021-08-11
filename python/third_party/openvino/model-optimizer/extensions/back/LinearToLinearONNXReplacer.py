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

from extensions.back.InterpolateReshape import InterpolateConcat, InterpolateReshapeWA
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class LinearToLinearONNXReplacer(BackReplacementPattern):
    """
    If we don't use this transformation, then we have a performance drop, because CPU and GPU have no optimized
    version of the 'linear' mode of the operation Interpolate.
    TODO: delete this transformation, when CPU and GPU will have optimized version of the 'linear' mode.
    """
    enabled = True

    def run_after(self):
        return [InterpolateConcat, InterpolateReshapeWA]

    def find_and_replace_pattern(self, graph: Graph):
        for interpolate_node in graph.get_op_nodes(type='Interpolate', version='opset4', mode='linear'):
            input_shape = interpolate_node.in_port(0).data.get_shape()
            interpolate_name = interpolate_node.soft_get('name', interpolate_node.id)
            assert input_shape is not None, \
                'Shape of interpolated data for node {} must not be None'.format(interpolate_name)
            input_rank = len(input_shape)
            if input_rank == 4:
                interpolate_node['mode'] = 'linear_onnx'
