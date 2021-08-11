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

from extensions.middle.ONNXRNNSequenceNormalize import ONNXRNNSequenceNormalize
from extensions.middle.permute_tensor_iterator import TransposeTensorIteratorLSTM
from mo.graph.graph import Graph, Node
from mo.middle.passes.eliminate import remove_op_node_with_data_node
from mo.middle.replacement import MiddleReplacementPattern


class ReverseTensorIteratorLSTM(MiddleReplacementPattern):
    """ Fuses Reverse operations around TI: ReverseSequence --> TI  --> ReverseSequence.

        WARNING This transformation is limited to support of very special case of TI but
        code doesn't check all the cases.
    """

    enabled = True

    def run_after(self):
        return [
            ONNXRNNSequenceNormalize,
            TransposeTensorIteratorLSTM,
        ]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    @staticmethod
    def is_fusable_reverse_sequence(node: Node):
        sequence_lengths = node.in_port(1).data.get_value()
        assert sequence_lengths is not None
        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None

        seq_len = input_shape[node.seq_axis]
        return np.all(sequence_lengths == seq_len)

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict(kind='data')),

                ('const', dict(type='Const')),
                ('const_d', dict(kind='data')),

                ('direct_reverse', dict(op='ReverseSequence')),
                ('input_reversed', dict(kind='data')),
                ('init_hidden', dict(kind='data')),

                ('ti', dict(kind='op', op='TensorIterator')),

                ('output_reversed', dict(kind='data')),

                ('const_1', dict(type='Const')),
                ('const_1_d', dict(kind='data')),

                ('inverse_reverse', dict(op='ReverseSequence')),
                ('output', dict(kind='data')),
            ],
            edges=[
                ('input', 'direct_reverse', {'in': 0}),
                ('const', 'const_d'),
                ('const_d', 'direct_reverse', {'in': 1}),
                ('direct_reverse', 'input_reversed'),

                ('input_reversed', 'ti', {'in': 0}),
                ('init_hidden', 'ti', {'in': 1}),
                ('ti', 'output_reversed', {'out': 0}),

                ('output_reversed', 'inverse_reverse', {'in': 0}),
                ('const_1', 'const_1_d'),
                ('const_1_d', 'inverse_reverse', {'in': 1}),
                ('inverse_reverse', 'output'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        ti = match['ti']
        direct_reverse = match['direct_reverse']
        inverse_reverse = match['inverse_reverse']

        assert direct_reverse.seq_axis == inverse_reverse.seq_axis
        assert direct_reverse.batch_axis is None and inverse_reverse.batch_axis is None or \
               direct_reverse.batch_axis == inverse_reverse.batch_axis

        if not self.is_fusable_reverse_sequence(direct_reverse) or \
                not self.is_fusable_reverse_sequence(inverse_reverse):
            # we can not merge ReverseSequence without equal sequences
            return

        # Modify stride in TI
        for port_map in [ti.input_port_map, ti.output_port_map]:
            for port in port_map:
                if 'axis' in port and port['axis'] is not None and 'external_port_id' in port:
                    assert port['axis'] == direct_reverse.seq_axis, \
                        'axis == {} != {} == direct_reverse.seq_dim'.format(port['axis'], direct_reverse.seq_axis)
                    if 'stride' not in port or port['stride'] is None:
                        port['stride'] = 1
                    assert port['stride'] in [-1, 1]
                    port['stride'] = -port['stride']
                    if port['stride'] == -1:
                        port['start'] = -1
                        port['end'] = 0
                    elif port['stride'] == 1:
                        port['start'] = None
                        port['end'] = None

        # Remove reverses
        remove_op_node_with_data_node(graph, direct_reverse)
        remove_op_node_with_data_node(graph, inverse_reverse)
