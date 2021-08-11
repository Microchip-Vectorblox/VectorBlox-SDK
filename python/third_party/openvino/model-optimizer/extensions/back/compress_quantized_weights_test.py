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
import unittest
from argparse import Namespace

import numpy as np
from generator import generator, generate

from extensions.back.compress_quantized_weights import CompressQuantizeWeights
from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Sub, Mul
from extensions.ops.fakequantize import FakeQuantize
from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, result, connect, \
    shaped_const_with_data


def nodes_dict(original, transformed=None, levels=255, data=None, il=[-127], ih=[127], ol=[-127], oh=[127]):
    shape = [1, 2, 3, 4] if data is None else np.array(data).shape
    data = np.ones(shape, dtype=original) if data is None else np.array(data, dtype=original)
    int_data = data.astype(dtype=np.int8)
    transformed = transformed if transformed is not None else original

    return {
        **valued_const_with_data('weights', data),
        **valued_const_with_data('int_weights', int_data),

        **regular_op_with_shaped_data(
            'cast', shape, {'type': 'Convert', 'op': 'Cast', 'infer': Cast.infer, 'dst_type': transformed}),

        **valued_const_with_data('il', np.array(il)),
        **valued_const_with_data('ih', np.array(ih)),
        **valued_const_with_data('ol', np.array(ol)),
        **valued_const_with_data('oh', np.array(oh)),

        **regular_op_with_shaped_data(
            'FQ', shape, {'type': 'FakeQuantize', 'infer': FakeQuantize.infer, 'stop_value_propagation': True,
                               'levels': levels, 'op': 'FakeQuantize'}),

        **valued_const_with_data('zp', np.array([0])),
        **valued_const_with_data('scale', np.array([1])),

        **regular_op_with_shaped_data(
            'sub', shape, {'type': 'Subtract', 'op': 'Sub', 'infer': lambda node: eltwise_infer(node, Sub.operation)}),

        **regular_op_with_shaped_data(
            'mul', shape, {'type': 'Multiply', 'op': 'Mul', 'infer': lambda node: eltwise_infer(node, Mul.operation)}),

        **result()
}


class CompressionQuantizeDequantizeSeparateTest(unittest.TestCase):
    def test_quantize(self):
        original_type = np.float32
        nodes = nodes_dict(original_type)

        graph = build_graph(nodes, [
            *connect('weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)

        error_message = 'Unexpected number of FakeQuantize nodes {} CompressQuantizeWeights.quantize_data call `{}`'
        fq_nodes = graph.get_op_nodes(type='FakeQuantize')
        self.assertEqual(len(fq_nodes), 1, error_message.format('before', len(fq_nodes)))
        fake_quantize = fq_nodes[0]

        CompressQuantizeWeights.quantize_data(fake_quantize, original_type)
        graph.clean_up()

        fq_nodes = graph.get_op_nodes(type='FakeQuantize')
        self.assertEqual(len(fq_nodes), 1, error_message.format('after', len(fq_nodes)))
        self.assertEqual(fq_nodes[0].in_port(0).get_source().node.soft_get('type'), 'Const')
        self.assertEqual(fq_nodes[0].in_port(0).get_source().node.data_type, np.int8)

        graph_ref = build_graph(nodes, [
            *connect('int_weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_dequantize(self):
        original_type = np.float32
        nodes = nodes_dict(original_type, np.int8)

        graph = build_graph(nodes, [
            *connect('weights:0', '0:cast'),
            *connect('cast:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)

        error_message = 'Unexpected number of {} nodes {} CompressQuantizeWeights.dequantize_data call `{}`'
        fq_nodes = graph.get_op_nodes(type='FakeQuantize')
        cast_nodes = graph.get_op_nodes(name='cast')
        self.assertEqual(len(fq_nodes), 1, error_message.format('FakeQuantize', 'before', len(fq_nodes)))
        self.assertEqual(len(cast_nodes), 1, error_message.format('Convert', 'before', len(cast_nodes)))
        cast_nodes[0]['need_shape_inference'] = True

        CompressQuantizeWeights.dequantize_data(fq_nodes[0], original_type)
        graph.clean_up()

        fq_nodes = graph.get_op_nodes(type='FakeQuantize')
        self.assertEqual(len(fq_nodes), 0, error_message.format('FakeQuantize', 'after', len(fq_nodes)))

        graph_ref = build_graph(nodes, [
            *connect('int_weights:0', '0:cast'),
            *connect('cast:0', '0:sub'),
            *connect('zp:0', '1:sub'),
            *connect('sub:0', '0:mul'),
            *connect('scale:0', '1:mul'),
            *connect('mul:0', 'output'),
        ], {'cast': {'dst_type': original_type}}, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)


@generator
class CompressionDataTypeTest(unittest.TestCase):
    @generate(*[
        ('FP32', np.int64),
        ('FP16', np.int64),
        ('FP32', np.int32),
        ('FP16', np.int32),
        ('FP32', np.float64, np.float32),
        ('FP16', np.float64, np.float16),
        ('FP32', np.float32, np.float32),
        ('FP16', np.float32, np.float16),
        ('FP32', np.float16, np.float32),
        ('FP16', np.float16, np.float16),
    ])
    def test_data_type(self, model_dtype, original, transformed=None):
        if transformed is None:
            transformed = original
        nodes = nodes_dict(original, transformed)

        graph = build_graph(nodes, [
            *connect('weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True, cli=Namespace(data_type=model_dtype, static_shape=True))

        CompressQuantizeWeights().find_and_replace_pattern(graph)
        graph.clean_up()

        graph_ref = build_graph(nodes, [
            *connect('int_weights:0', '0:cast'),
            *connect('cast:0', '0:sub'),
            *connect('zp:0', '1:sub'),
            *connect('sub:0', '0:mul'),
            *connect('scale:0', '1:mul'),
            *connect('mul:0', 'output'),
        ], nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)


@generator
class AccuracyCheckFP32Test(unittest.TestCase):
    eps = np.finfo(np.float32).eps

    @generate(*[
        ([-2.586, -1.338, 2.773, 4.414], [-2.586], [4.414], [-2.586], [4.414], 256),
        ([-1.5, -0.32, 0.167, 2.8], [-1.5], [2.8], [-1.5], [2.8], 256),
        ([1, 1 + eps, 1 + 2 * eps, 1 + 3 * eps], [1], [1 + 3 * eps], [1], [1 + 3 * eps], 256),
        ([1.0, 2.0, 3.0, 4.0], [1], [4], [1], [4], 256),
    ])
    def test_accuracy(self, data, in_low, in_high, out_low, out_high, levels):
        nodes = nodes_dict(np.float32, None, levels, data, in_low, in_high, out_low, out_high)

        graph = build_graph(nodes, [
            *connect('weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)
        graph_ref = graph.copy()

        CompressQuantizeWeights().find_and_replace_pattern(graph)

        for node in graph.get_op_nodes() + graph_ref.get_op_nodes():
            node['stop_value_propagation'] = False
            node['need_shape_inference'] = node.soft_get('need_shape_inference', True)

        graph.clean_up()
        graph_ref.clean_up()

        const_result_graph = build_graph({**shaped_const_with_data('weights', np.array(data).shape), **result()},
                                         [*connect('weights', 'output')], nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, const_result_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

        (flag, resp) = compare_graphs(graph_ref, const_result_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

        # as this two graphs calculated the same data through different constant folding functions, they resulted in
        # constants of different data type since FakeQuantize always have f32 output dtype, but eltwises use numpy
        # for folding which doesn't have such restriction
        const_node = graph.get_op_nodes(type='Const')
        self.assertEqual(len(const_node), 1)
        if const_node[0].data_type == np.float64:
            const_node[0].data_type = np.float32

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

        # I would like to leave this commented code here to quickly check the actual output value:
        # print(result_node.in_port(0).data.get_value())  # actual calculated value


@generator
class NegativeCompressionTestLevels(unittest.TestCase):
    @generate(*[(2), (257), (None), (0), (-5)])
    def test_negative_fq_unacceptable_levels(self, levels):
        nodes = nodes_dict(np.float32, None, levels)

        graph = build_graph(nodes, [
            *connect('weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)
        graph_ref = graph.copy()
        CompressQuantizeWeights().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
