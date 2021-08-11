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

from mo.front.caffe.extractors.utils import get_canonical_axis_index
import numpy as np
from mo.utils.error import Error


def slice_axis_ext(attrs):
    axis = attrs.int("axis", 0)
    begin = attrs.int("begin", 0)
    end = attrs.int("end", None)

    node_attrs = {
        'type': 'Crop',
        'axis': axis,
        'offset': begin,
        'dim': end,
        'infer': mxnet_slice_axis_infer
    }
    return node_attrs


def mxnet_slice_axis_infer(node):
    in_shape = node.in_port(0).data.get_shape()
    node.axis = get_canonical_axis_index(in_shape, node.axis)
    slice_axis = node.axis

    new_shape = np.array(in_shape, dtype=np.int64)
    new_shape[slice_axis] = new_shape[slice_axis] / len(node.out_nodes())

    axis_size = in_shape[slice_axis]
    if node.offset < 0:
        node.offset += axis_size

    if not node.dim:
        node.dim = axis_size
    elif node.dim < 0:
        node.dim += axis_size

    input_dim = in_shape.size
    node.dim = (node.dim - node.offset)
    if node.dim > in_shape[slice_axis]:
        raise Error(
            '{0} node dimension value is bigger than the corresponding value in the input shape {1}. ' +
            '\nIn particular {2} is bigger than {3}. The Model Optimizer does not support this case. ' +
            '\nTo overcome, try to edit the original model "end" property of the {0} layer.',
            node.name, ','.join(str(i) for i in in_shape), str(node.dim), str(in_shape[slice_axis])
        )

    for i in range(0, input_dim):
        if i == slice_axis:
            new_shape[i] = node.dim
        else:
            new_shape[i] = in_shape[i]

    for i in range(0, len(node.out_nodes())):
        node.out_node(i)['shape'] = new_shape
