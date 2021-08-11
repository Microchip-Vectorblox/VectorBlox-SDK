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

from typing import Dict

import numpy as np

from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Sub, Div, Mul, Negative
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, Node
from mo.middle.passes.convert_data_type import data_type_str_to_np, np_data_type_to_destination_type
from mo.ops.const import Const


class CompressQuantizeWeights(BackReplacementPattern):
    """
    Compress weights transformation goal is to pre-quantize data to minimize runtime calculations with constant data.
    To achieve this goal we perform FakeQuantize decomposition to separate quantization from dequantization in it.

    FakeQuantize:
        -[src_dtype]-> FakeQuantize -[src_dtype]->
    is an operation that could be represented as:
        -[src_dtype]-> Quantize -[quantized_dtype]-> Dequantize -[src_dtype]->

     Quantize and Dequantize operations are not present in OpenVINO supported opsets, but can be easily expressed
     through supported ones. Transformation algorithm doesn't contain all the steps described
     below (some of them are optimized). Steps are presented only to show the idea in details.

    Step 1: FQ decomposition
        -[src_dtype]-> Quantize -[quantized_dtype]-> Dequantize -[src_dtype]->

    Step 2: Representing Quantize and Dequantize through FakeQuantize and Convert operations
        Simplified view:
            -[src_dtype]-> FakeQuantize -[src_dtype]-> Convert -[quantized_dtype]-> Convert -[src_dtype]-> FakeQuantize -[quantized_dtype]->

        Detailed view:
            initial_input_low       initial_input_high               initial_output_low   initial_output_high
                       \                /                                 |              /
                       (in: 1)    (in: 2)                               (in: 3)     (in: 4)
                          V         V                                     V          V
            Constant -> FakeQuantize` --> Convert --> Convert --> initial FakeQuantize -->
                     ^          ^     (quant_dtype)  (src_dtype)           ^         ^
                     |          |                                       (in: 1)    (in: 2)
                (in: 3)    (in: 4)                                         |          |
                   |           \________________          _________________|          |
                   |                            \        /                            |
               new_output_low                 new_output_high                         |
               -(levels // 2)          (levels + new_output_low - 1)                  |
                   |__________________________________________________________________|

    Step 3: All inputs of initial FQ are Constants and we haven't added dynamic dependencies. Means we can const-fold
        sub-graph we already have, but as our goal is to have quantized data, we should mark nodes to be folded.

        -[src_dtype]-> FakeQuantize -[src_dtype]-> Convert -[quantized_dtype]-> Convert -[src_dtype]-> FakeQuantize -[src_dtype]->
        |-------------------------Const Folding-------------------------------|----------------------Stays----------------------------|

        Resulting graph:
            Constant -[quantized_dtype]-> Convert -[src_dtype]-> FakeQuantize -[src_dtype]->

    Step 4: We reduced heavy manipulations with constant data in runtime, but we can go even further.
        At this stage FakeQuantize node is playing dequantization role. It means it only shifts and scales the data.
        No rounding is performed by this FakeQuantize as data was fully quantized earlier.
        Also, runtime calculates this shift (zero point) and scale during low precision transformation.
        It means we can pre-calculate even this information for them by simply decomposing FakeQuantize that plays
        dequantization role to Subtract-Multiply sequence so resulting graph would be:
            Constant -[quantized_dtype]-> Convert -[src_dtype]-> Subtract (zero_point) -> Multiply (scale) -[src_dtype]->

        Where:
            scale = (output_high - output_low) / (input_high - input_low)
                WARNING: division by zero imposes restriction -- input_high can not be equal to input_low
            zero_point = input_low - output_low / scale

    TODO: steps 5 and 6 are NOT IMPLEMENTED YET
    TODO: DOES LPT NEED IT???
    Step 5: Having zero_point == 0 is really beneficial for performance, so we try to fuse Subtract up to the Constant.
        It is not always possible because of the quantized_dtype possible range of values.

    Step 6: (Optional) From the nature of Subtract and Multiply operations they may be optimized out in cases:
            zero_point == 0
            scale == 1

    BENEFITS:
        Such constant data packing reduces IR size (.bin file size)
        Also, transformation prepares quantized constant data for Low Precision pipeline.
        With that we can skip same calculations in the runtime and make loading of such sub-graphs to the plugin faster.
    """

    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].disable_weights_compression]

    force_clean_up = True

    def pattern(self):
        return dict(
            nodes=[
                ('const', dict(type='Const')),
                ('const_d', dict()),
                ('fake_quantize', dict(type='FakeQuantize', levels=lambda x: x is not None and 2 < x <= 256)),
            ],
            edges=[
                ('const', 'const_d'),
                ('const_d', 'fake_quantize', {'in': 0}),
            ]
        )

    @staticmethod
    def quantize_data(fake_quantize: Node, dst_type: type):
        graph = fake_quantize.graph
        name = fake_quantize.soft_get('name', fake_quantize.id)
        levels = fake_quantize.levels

        quantize = fake_quantize.copy_node(dict(name=name + '/Copy', stop_value_propagation=False), graph)
        fake_quantize.in_port(0).get_connection().set_destination(quantize.in_port(0))

        # inherit input limits
        fake_quantize.in_port(1).get_connection().set_destination(quantize.in_port(1))
        fake_quantize.in_port(2).get_connection().set_destination(quantize.in_port(2))

        # calculate output limits for quantized weights
        i_min = np.array([-(levels // 2)], dtype=dst_type)
        i_max = np.array(levels + i_min - 1, dtype=dst_type)
        assert i_max - i_min == levels - 1
        out_low = Const(graph, dict(name=name + '/Copy/out_low', value=i_min)).create_node()
        out_high = Const(graph, dict(name=name + '/Copy/out_high', value=i_max)).create_node()

        out_low.out_port(0).connect(quantize.in_port(3))
        out_high.out_port(0).connect(quantize.in_port(4))
        out_low.out_port(0).connect(fake_quantize.in_port(1))
        out_high.out_port(0).connect(fake_quantize.in_port(2))

        original_const = quantize.in_port(0).get_source().node
        quantized_data_name = original_const.soft_get('name', original_const.id) + '/quantized'
        cast = Cast(graph, dict(name=quantized_data_name, dst_type=np.int8, stop_value_propagation=False)).create_node()

        quantize.out_port(0).connect(cast.in_port(0))

        cast.out_port(0).connect(fake_quantize.in_port(0))

    @staticmethod
    def dequantize_data(fake_quantize: Node, dst_type: type) -> Node:
        graph = fake_quantize.graph
        quantized_data = fake_quantize.in_port(0).get_source().node
        name = fake_quantize.soft_get('name', fake_quantize.id)

        assert quantized_data.soft_get('type') == 'Convert' and quantized_data.dst_type == np.int8, \
            'Weights aren`t compressed as expected for node {}'.format(fake_quantize.soft_get('name', fake_quantize.id))

        dequantizing_cast = Cast(graph, dict(
            name=quantized_data.name + "/to_{}".format(np_data_type_to_destination_type(dst_type)),
            dst_type=dst_type, stop_value_propagation=True)).create_node()
        fake_quantize.in_port(0).get_connection().set_destination(dequantizing_cast.in_port(0))

        # limits of dequantize
        in_low = fake_quantize.in_port(1).get_source()
        in_high = fake_quantize.in_port(2).get_source()
        out_low = fake_quantize.in_port(3).get_source()
        out_high = fake_quantize.in_port(4).get_source()

        # scale calculation
        output_range = Sub(graph, {'name': name + '/output_range'}).create_node()
        output_range.in_port(0).connect(out_high)
        output_range.in_port(1).connect(out_low)

        input_range = Sub(graph, {'name': name + '/input_range'}).create_node()
        input_range.in_port(0).connect(in_high)
        input_range.in_port(1).connect(in_low)

        scale = Div(graph, {'name': name + '/scale'}).create_node()
        scale.in_port(0).connect(output_range.out_port(0))
        scale.in_port(1).connect(input_range.out_port(0))

        # shift calculation
        descaled_output_low = Div(graph, {'name': name + '/descaled_output_low'}).create_node()
        descaled_output_low.in_port(0).connect(out_low)
        descaled_output_low.in_port(1).connect(scale.out_port(0))

        shift = Sub(graph, {'name': name + '/zero_point'}).create_node()
        shift.in_port(0).connect(in_low)
        shift.in_port(1).connect(descaled_output_low.out_port(0))

        # DeQuantize(x) == Mul(Sub(x, zero_point), scale)
        sub_zp = Sub(graph, {'name': name + '/minus_zp'}).create_node()
        sub_zp.in_port(0).connect(dequantizing_cast.out_port(0))
        sub_zp.in_port(1).connect(shift.out_port(0))

        mul_scale = Mul(graph, {'name': name + '/mulpiply_by_scale'}).create_node()
        mul_scale.in_port(0).connect(sub_zp.out_port(0))
        mul_scale.in_port(1).connect(scale.out_port(0))

        fake_quantize.out_port(0).get_connection().set_source(mul_scale.out_port(0))

        graph.remove_nodes_from([fake_quantize.id, fake_quantize.out_node(0)])

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        fake_quantize = match['fake_quantize']

        dst_type = match['const'].value.dtype
        if np.issubdtype(dst_type, np.floating):
            dst_type = data_type_str_to_np(graph.graph['cmd_params'].data_type)

        self.quantize_data(fake_quantize, dst_type)
        self.dequantize_data(fake_quantize, dst_type)
