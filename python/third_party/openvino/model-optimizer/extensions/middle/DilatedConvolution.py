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
import logging as log

import numpy as np

from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const


class DilatedConvolutionConverter(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[
                ('conv', dict(kind='op', op=lambda value: value in ['Conv2D', 'DepthwiseConv2dNative', 'Conv3D'])),
                ('space_to_batch', dict(kind='op', op='SpaceToBatch')),
                ('batch_to_space', dict(kind='op', op='BatchToSpace')),
                ('input', dict(kind='data')),
                ('output', dict(kind='data')),
                ('conv_output', dict(kind='data')),
                ('stb_output', dict(kind='data')),
                ('stb_bs', dict(kind='data')),
                ('stb_pad_begin', dict(kind='data')),
                ('stb_pad_end', dict(kind='data')),
                ('bts_bs', dict(kind='data')),
                ('bts_crop_begin', dict(kind='data')),
                ('bts_crop_end', dict(kind='data'))
            ],
            edges=[
                ('input', 'space_to_batch', {'in': 0}),
                ('stb_bs', 'space_to_batch', {'in': 1}),
                ('stb_pad_begin', 'space_to_batch', {'in': 2}),
                ('stb_pad_end', 'space_to_batch', {'in': 3}),
                ('space_to_batch', 'stb_output', {'out': 0}),
                ('stb_output', 'conv', {'in': 0}),
                ('conv', 'conv_output', {'out': 0}),
                ('conv_output', 'batch_to_space', {'in': 0}),
                ('bts_bs', 'batch_to_space', {'in': 1}),
                ('bts_crop_begin', 'batch_to_space', {'in': 2}),
                ('bts_crop_end', 'batch_to_space', {'in': 3}),
                ('batch_to_space', 'output', {'out': 0}),
            ])

    def replace_pattern(self, graph: Graph, match: dict):
        conv = match['conv']
        stb = match['space_to_batch']
        bts = match['batch_to_space']

        block_size = match['stb_bs']

        input = match['input']
        output = match['output']
        stb_out = match['stb_output']
        conv_out = match['conv_output']

        in_edge_attrs = graph.get_edge_data(input.id, stb.id)[0]
        out_edge_attrs = graph.get_edge_data(bts.id, output.id)[0]

        graph.remove_edge(input.id, stb.id)
        graph.remove_edge(stb_out.id, conv.id)
        graph.remove_edge(conv.id, conv_out.id)
        graph.remove_edge(bts.id, output.id)

        conv.dilation[conv.spatial_dims] = block_size.value[conv.spatial_dims]

        pad_begin = match['stb_pad_begin'].value - match['bts_crop_begin'].value
        pad_end = match['stb_pad_end'].value - match['bts_crop_end'].value
        conv.pad[conv.spatial_dims] = [[pad_begin[x], pad_end[x]] for x in conv.spatial_dims]
        conv['auto_pad'] = None

        graph.add_edges_from([
            (input.id, conv.id, {'in': 0, **in_edge_attrs}),
            (conv.id, output.id, {'out': 0, **out_edge_attrs}),
        ])


class DilatedConvolution1DConverter(MiddleReplacementPattern):
    """
    Transformation looks for a pattern that TF generates for a 1D dilated convolution with help of SpaceToBatch (STB)
    and BatchToSpace (BTS). The transformation removes STB and BTS operations and updates the Convolution node
    attributes with a dilation values.
    """
    enabled = True
    force_clean_up = True
    force_shape_inference = True

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[
                ('conv', dict(kind='op', op=lambda value: value in ['Conv2D', 'DepthwiseConv2dNative'])),
                ('space_to_batch', dict(kind='op', op='SpaceToBatch')),
                ('unsqueeze', dict(kind='op', op='Unsqueeze')),
                ('squeeze', dict(kind='op', op='Squeeze')),
                ('batch_to_space', dict(kind='op', op='BatchToSpace')),
                ('input_data', dict(kind='data')),
                ('output', dict(kind='data')),
                ('unsqueeze_output', dict(kind='data')),
                ('squeeze_output', dict(kind='data')),
                ('conv_output', dict(kind='data')),
                ('stb_output', dict(kind='data')),
                ('stb_bs', dict(kind='data')),
                ('unsqueeze_dim', dict(kind='data')),
                ('stb_pad', dict(kind='data')),
                ('bts_bs', dict(kind='data')),
                ('bts_crop', dict(kind='data'))
            ],
            edges=[
                ('input_data', 'space_to_batch', {'in': 0}),
                ('stb_bs', 'space_to_batch', {'in': 1}),
                ('stb_pad', 'space_to_batch', {'in': 2}),
                ('space_to_batch', 'stb_output', {'out': 0}),
                ('stb_output', 'unsqueeze', {'in': 0}),
                ('unsqueeze_dim', 'unsqueeze', {'in': 1}),
                ('unsqueeze', 'unsqueeze_output', {'out': 0}),
                ('unsqueeze_output', 'conv', {'in': 0}),
                ('conv', 'conv_output', {'out': 0}),
                ('conv_output', 'squeeze', {'in': 0}),
                ('squeeze', 'squeeze_output', {'out': 0}),
                ('squeeze_output', 'batch_to_space', {'in': 0}),
                ('bts_bs', 'batch_to_space', {'in': 1}),
                ('bts_crop', 'batch_to_space', {'in': 2}),
                ('batch_to_space', 'output', {'out': 0}),
            ])

    def swap_pad_and_unsqueeze(self, pad: Node, unsqueeze: Node):
        # insert additional items to the pads in the position specified by the Unsqueeze axis
        unsqueeze_axis = unsqueeze.in_port(1).data.get_value()
        for port_id in [1, 2]:
            current_value = pad.in_port(port_id).get_connection().data.get_value()
            new_value_node = Const(pad.graph, {'name': pad.soft_get('name', pad.id) + '/value_{}'.format(port_id),
                                               'value': np.insert(current_value, unsqueeze_axis.item(), 0),
                                               'override_output_shape': True}).create_node()
            pad.in_port(port_id).disconnect()
            pad.in_port(port_id).connect(new_value_node.out_port(0))

        # swap Pad and Unsqueeze layers
        unsqueeze.in_port(0).disconnect()
        pad.in_port(0).get_connection().set_destination(unsqueeze.in_port(0))
        unsqueeze.out_port(0).get_connection().set_source(pad.out_port(0))
        unsqueeze.out_port(0).connect(pad.in_port(0))

        # output shapes of Pad and Unsqueeze changed so need to recalculate them
        pad['override_output_shape'] = True
        unsqueeze['override_output_shape'] = True

    def replace_pattern(self, graph: Graph, match: dict):
        conv = match['conv']
        stb = match['space_to_batch']
        bts = match['batch_to_space']
        unsqueeze = match['unsqueeze']
        squeeze = match['squeeze']

        if len(conv.in_port(0).data.get_shape()) != 4:
            log.debug('The convolution node "{}" input is not 4D'.format(conv.soft_get('name', conv.id)))
            return

        block_size = stb.in_port(1).data.get_value()
        if len(block_size) != 1:
            log.debug('The block size must contain 1 element')
            return

        unsqueeze_dims = np.array(unsqueeze.in_port(1).data.get_value())
        if unsqueeze_dims.size != 1 or unsqueeze_dims.item() != 1:
            log.debug('The Unsqueeze dimension is not equal to 1')
            return

        # remove SpaceToBatch and BatchToSpace operations
        unsqueeze.in_port(0).get_connection().set_source(stb.in_port(0).get_source())
        bts.out_port(0).get_connection().set_source(squeeze.out_port(0))
        stb.in_port(0).disconnect()
        bts.in_port(0).disconnect()

        conv.dilation[conv.spatial_dims] = [1, block_size.item()]

        pad = match['stb_pad'].value - match['bts_crop'].value
        # update the pad value by inserting one zero element since the STB node consumes 3D tensor and have 1D pad value
        # but the successive convolution consumes 4D tensor
        pad = np.insert(pad, 0, 0, 0)
        conv.pad[conv.spatial_dims] = [[pad[x][0], pad[x][1]] for x in range(len(pad))]
        conv['auto_pad'] = None

        # the intermediate shapes will be changed after nodes relocation so mark nodes accordingly
        input_producer = unsqueeze.in_port(0).get_source().node
        input_producer['need_shape_inference'] = True
        input_producer['override_output_shape'] = True
        unsqueeze['need_shape_inference'] = True
        unsqueeze['override_output_shape'] = True
        conv['need_shape_inference'] = True
        conv['override_output_shape'] = True

        # if the input to SpaceToBatch is a Pad layer then we can swap it with Unsqueeze so the Pad will be fused to a
        # Convolution layer further in the pipeline
        if input_producer.soft_get('type') == 'Pad':
            self.swap_pad_and_unsqueeze(input_producer, unsqueeze)
