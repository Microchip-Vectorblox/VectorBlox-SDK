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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import bool_to_str
from mo.graph.graph import Node, Graph
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op

reduce_map = {
    'ReduceSum': np.sum,
    'ReduceProd': np.prod,
    'ReduceL1': lambda x, axis, keepdims: np.sum(a=np.absolute(x), axis=axis, keepdims=keepdims),
    'ReduceL2': lambda x, axis, keepdims: np.sqrt(np.sum(a=np.square(x), axis=axis, keepdims=keepdims)),
    'ReduceMax': np.max,
    'ReduceMin': np.min,
    'ReduceMean': np.mean,
    'ReduceAnd': np.all,
    'ReduceLogicalAnd': np.all,
    'ReduceLogicalOr': np.any,
}


def reduce_infer(node: Node):
    connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
    assert len(connected_in_ports) == 2, \
        "{} node `{}` should have 2 input ports, where 0-input is data input and 1-input represent " \
        "`reduction_indices`".format(node.op, node.id)

    in_data = node.in_port(0).data
    in_shape = in_data.get_shape()
    axis = node.in_port(1).data.get_value()

    # If the axis is None then reduce over all the dimensions of the input tensor
    if axis.size == 1 and axis.item() is None:
        axis = int64_array(list(range(len(in_shape))))
        node.in_port(1).data.set_value(axis)

    assert in_shape is not None, "Can not infer {} node `{}`: shape of 0-input unknown".format(node.op, node.id)

    axis = axis.copy()
    if axis.size == 1:
        axis = int64_array([axis.item()])

    in_value = in_data.get_value()

    if in_value is not None:
        value = reduce_map[node.op](in_value.copy(), axis=tuple(axis), keepdims=node.keep_dims)
        node.out_port(0).data.set_value(value)
    else:
        used_dims = np.zeros(len(in_shape), dtype=np.bool)
        output_shape = in_shape.copy()

        for dim in axis:
            used_dims[dim] = True
            output_shape[dim] = 1

        # In case if keep dims == False, we should remove all 1 dims that was used in reduction
        if not node.keep_dims:
            output_shape = output_shape[np.invert(used_dims)]

        node.out_port(0).data.set_shape(output_shape)

    # if the operation changes the rank of the output tensor then it is necessary to insert Permute if the input is 4D
    # or 5D
    if not node.keep_dims:
        node['reinterp_shape'] = True

    PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')


class ReduceOp(Op):
    enabled = False
    op = None
    op_type = None
    version = 'opset1'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op_type,
            'version': self.version,
            'infer': reduce_infer,
            'keep_dims': 0,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'force_precision_in_ports': {
                1: 'int64'},
        }, attrs)
        assert isinstance(self.attrs['keep_dims'], int) or isinstance(self.attrs['keep_dims'], bool)
        self.attrs['keep_dims'] = bool(self.attrs['keep_dims'])

    def supported_attrs(self):
        return [
            ('keep_dims', lambda node: bool_to_str(node, 'keep_dims')),
        ]


class ReduceSum(ReduceOp):
    enabled = True
    op = 'ReduceSum'
    op_type = 'ReduceSum'


class ReduceProd(ReduceOp):
    op = 'ReduceProd'
    op_type = 'ReduceProd'
    enabled = True


class ReduceMin(ReduceOp):
    op = 'ReduceMin'
    op_type = 'ReduceMin'
    enabled = True


class ReduceMax(ReduceOp):
    op = 'ReduceMax'
    op_type = 'ReduceMax'
    enabled = True


class ReduceMean(ReduceOp):
    op = 'ReduceMean'
    op_type = 'ReduceMean'
    enabled = True


class ReduceL1(ReduceOp):
    op = 'ReduceL1'
    op_type = 'ReduceL1'
    version = 'opset4'

class ReduceL2(ReduceOp):
    op = 'ReduceL2'
    op_type = 'ReduceL2'
    version = 'opset4'


class ReduceAnd(ReduceOp):
    op = 'ReduceAnd'
    op_type = 'ReduceLogicalAnd'
    enabled = True


class ReduceLogicalAnd(ReduceOp):
    op = 'ReduceLogicalAnd'
    op_type = 'ReduceLogicalAnd'
    enabled = True


class ReduceLogicalOr(ReduceOp):
    op = 'ReduceLogicalOr'
    op_type = 'ReduceLogicalOr'
    enabled = True
