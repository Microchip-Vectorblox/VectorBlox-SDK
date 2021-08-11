"""
 Copyright (C) 2017-2021 Intel Corporation

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
from mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from mo.ops.op import Op
from mo.utils.error import Error


class CTCGreedyDecoderSeqLenOp(Op):
    op = 'CTCGreedyDecoderSeqLen'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset6',

            'infer': self.infer,
            'type_infer': self.type_infer,

            'in_ports_count': 3,
            'out_ports_count': 2,

            'merge_repeated': True,
            'classes_index_type': np.int32,
            'sequence_length_type': np.int32
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        version = self.get_opset()
        if version == 'opset6':
            return [('classes_index_type', lambda node: np_data_type_to_destination_type(node.classes_index_type)),
                    ('sequence_length_type', lambda node: np_data_type_to_destination_type(node.sequence_length_type)),
                    ('merge_repeated', lambda node: bool_to_str(node, 'merge_repeated'))]
        else:
            raise Error('Unknown opset version "{}"'.format(version))

    @staticmethod
    def type_infer(node):
        opset = node.get_opset()
        if opset == 'opset6':
            node.out_port(0).set_data_type(node.classes_index_type)
            node.out_port(1).set_data_type(node.sequence_length_type)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) in [2, 3], \
            "Incorrect number of inputs for {} node".format(node_name)

        logits_shape = node.in_port(0).data.get_shape()
        sequence_len_shape = node.in_port(1).data.get_shape()
        if len(node.in_nodes()) == 3:
            blank_index_shape = node.in_port(2).data.get_shape()
            assert len(blank_index_shape) == 1, \
                'Incorrect rank of blank_index for {} node'.format(node_name)

        # check shapes of input tensors
        assert len(logits_shape) == 3, \
            'Incorrect rank of logits for {} node'.format(node_name)

        assert len(sequence_len_shape) == 1, \
            'Incorrect rank of sequence length tensor for {} node'.format(node_name)
        assert logits_shape[0] == sequence_len_shape[0], \
            'Batch dimensions of input tensors must be the same for {} node'.format(node_name)

        batch_size = logits_shape[0]
        time_size = logits_shape[1]
        if node.is_out_port_connected(0):
            node.out_port(0).data.set_shape(int64_array([batch_size, time_size]))
        if node.is_out_port_connected(1):
            node.out_port(1).data.set_shape(int64_array([batch_size]))
