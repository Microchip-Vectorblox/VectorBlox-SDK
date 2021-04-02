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
from extensions.front.kaldi.replace_lstm_node_pattern import unique_id, create_zero_value_with_batch_from_input
from extensions.ops.split import VariadicSplit
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.assign import Assign
from mo.ops.concat import Concat
from mo.ops.crop import Crop
from mo.ops.read_value import ReadValue
from mo.ops.result import Result


class ReplaceSpliceNodePattern(MiddleReplacementPattern):
    """
       This pass decomposes Splice layer to the sequence Slice Concat and Memory layers
       For example:
           Let's suppose we have next graph:

           Input (N, H) -> Slice -> Next_Layer (N, k*H)

           Where (N, k*H) is is real input of subsequent topology.
           Splice is used for accumulation next (k-1)/2 and previous (k-1)/2 input data

           So this pass will convert this graph to the next one:

                                    Input [N, H]                  __
                                                /               /
                                                 Concat [N, k*H]
                                                /               \
           Memory [N, k*H] -> Slice [N, (k-1)*H]                 Memory [N, k*H]

   """
    enabled = True

    def run_after(self):
        from extensions.middle.RemoveDuplicationMemory import MergeNeighborSplicePattern, RemoveMemoryDuplicationPattern
        return [MergeNeighborSplicePattern,
                RemoveMemoryDuplicationPattern]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(op='Splice'))],
            edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        in_shape = node.in_port(0).data.get_shape().copy()
        memory_element = in_shape[1] - node.const_dim
        memory_size = memory_element * len(node.context)

        memory_pair_id = unique_id('id')
        # Memory(in)
        input_memory = ReadValue(graph, {'name': 'prev_splice_memory',
                                         'variable_id': memory_pair_id}).create_node()

        # Memory(in)  \
        #             Crop
        # Input(temp) /
        crop = Crop(graph, {'name': 'Splice_Crop',
                            'axis': int64_array([1]),
                            'offset': int64_array([memory_element]),
                            'dim': int64_array([memory_size - memory_element])}).create_node()
        crop.in_port(0).connect(input_memory.out_port(0))

        # Crop   \
        #         Concat
        # Input  /
        concat_node = Concat(graph, {'name': 'Splice_Concat',
                                     'in_ports_count': 2,
                                     'axis': 1}).create_node()
        concat_node.in_port(0).connect(crop.out_port(0))

        # Concat -> Memory(out)
        mem_out = Assign(graph, {'name': 'out_splice_memory', 'variable_id': memory_pair_id}).create_node()
        mem_out.in_port(0).connect(concat_node.out_port(0))
        Result(graph).create_node().in_port(0).connect(mem_out.out_port(0))

        if node.const_dim != 0:
            memory_element_constdim = node.const_dim
            memory_size_constdim = memory_element_constdim * len(node.context)

            split = create_op_with_const_inputs(
                graph, VariadicSplit, {1: int64_array(1), 2: int64_array([memory_element, memory_element_constdim])},
                {'name': node.id + '_split_const', 'out_ports_count': 2})

            split.out_port(0).connect(concat_node.in_port(1))

            # create separate splice construction for const_dim
            memory_pair_id = unique_id('memory_for_const_dim')
            init_value_input_memory_const_dim = create_zero_value_with_batch_from_input(split.out_port(1),
                                                                                        memory_size_constdim)
            input_memory_const_dim = ReadValue(graph, {'name': 'const_dim_in_memory',
                                                       'variable_id': memory_pair_id}).create_node()
            init_value_input_memory_const_dim.out_port(0).connect(input_memory_const_dim.in_port(0))

            crop_const_dim = Crop(graph, {'name': 'const_dim_crop',
                                          'axis': int64_array([1]),
                                          'offset': int64_array([memory_element_constdim]),
                                          'dim': int64_array(
                                              [memory_size_constdim - memory_element_constdim])}).create_node()
            crop_const_dim.in_port(0).connect(input_memory_const_dim.out_port(0))

            concat_node_const_dim = Concat(graph, {'name': 'const_dim_concat',
                                                   'in_ports_count': 2,
                                                   'axis': 1}).create_node()
            concat_node_const_dim.in_port(0).connect(crop_const_dim.out_port(0))

            mem_out_const_dim = Assign(graph, {'name': 'const_dim_out_memory',
                                               'variable_id': memory_pair_id}).create_node()
            mem_out_const_dim.in_port(0).connect(concat_node_const_dim.out_port(0))
            Result(graph).create_node().in_port(0).connect(mem_out_const_dim.out_port(0))

            # connect splice to Split as begin and Concat as the end
            split.out_port(1).connect(concat_node_const_dim.in_port(1))
            crop_first = Crop(graph, {'name': 'const_dim_crop_first',
                                      'axis': int64_array([1]),
                                      'offset': int64_array([0]),
                                      'dim': int64_array([memory_element_constdim])}).create_node()
            crop_first.in_port(0).connect(concat_node_const_dim.out_port(0))

            concat_const = Concat(graph, {'name': node.id + '_concat_const', 'axis': 1,
                                          'in_ports_count': 2}).create_node()
            concat_const.in_port(1).connect(crop_first.out_port(0))
            concat_const.in_port(0).connect(concat_node.out_port(0))

            init_value_input_memory = create_zero_value_with_batch_from_input(split.out_port(0),
                                                                              memory_size)
            init_value_input_memory.out_port(0).connect(input_memory.in_port(0))
            node.in_port(0).get_connection().set_destination(split.in_port(0))
            node.out_port(0).get_connection().set_source(concat_const.out_port(0))
        else:
            init_value_input_memory = create_zero_value_with_batch_from_input(node.in_port(0).get_source(),
                                                                              memory_size)
            init_value_input_memory.out_port(0).connect(input_memory.in_port(0))
            node.in_port(0).get_connection().set_destination(concat_node.in_port(1))
            node.out_port(0).get_connection().set_source(concat_node.out_port(0))

        # to avoid re-inference of shape and touching in next replacements
        graph.remove_node(node.id)
