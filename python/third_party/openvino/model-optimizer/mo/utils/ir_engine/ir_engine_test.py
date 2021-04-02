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
import os
import sys
import unittest
from unittest import mock

from generator import generator, generate

from mo.graph.graph import Node
from mo.utils.ir_engine.ir_engine import IREngine

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)


@generator
class TestFunction(unittest.TestCase):
    def setUp(self):
        path, _ = os.path.split(os.path.dirname(__file__))
        self.xml = os.path.join(path,
                                "unittest", "test_data", "mxnet_synthetic_gru_bidirectional_FP16_1_v6.xml")
        self.xml_negative = os.path.join(path,
                                         "unittest", "test_data",
                                         "mxnet_synthetic_gru_bidirectional_FP16_1_v6_negative.xml")
        self.bin = os.path.splitext(self.xml)[0] + '.bin'
        self.assertTrue(os.path.exists(self.xml), 'XML file not found: {}'.format(self.xml))
        self.assertTrue(os.path.exists(self.bin), 'BIN file not found: {}'.format(self.bin))

        self.IR = IREngine(path_to_xml=str(self.xml), path_to_bin=str(self.bin))
        self.IR_ref = IREngine(path_to_xml=str(self.xml), path_to_bin=str(self.bin))
        self.IR_negative = IREngine(path_to_xml=str(self.xml_negative), path_to_bin=str(self.bin))

    @generate(*[(4.4, True), ('aaaa', False)])
    def test_is_float(self, test_data, result):
        test_data = test_data
        self.assertEqual(IREngine._IREngine__isfloat(test_data), result,
                         "Function __isfloat is not working with value: {}".format(test_data))
        log.info('Test for function __is_float passed wit value: {}, expected result: {}'.format(test_data, result))

    # TODO add comparison not for type IREngine
    def test_compare(self):
        flag, msg = self.IR.compare(self.IR_ref)
        self.assertTrue(flag, 'Comparing false, test compare function failed')
        log.info('Test for function compare passed')

    def test_comare_negative(self):
        # Reference data for test:
        reference_msg = 'Current node "2" with type Const and reference node "2" with type Input have different attr "type" : Const and Input'
        # Check function:
        flag, msg = self.IR.compare(self.IR_negative)
        self.assertFalse(flag, 'Comparing flag failed, test compare function failed')
        self.assertEqual('\n'.join(msg), reference_msg, 'Comparing message failes, test compare negative failed')

        log.info('Test for function compare passed')

    def test_find_input(self):
        # Create references for this test:
        ref_nodes = [Node(self.IR.graph, '0')]
        # Check function:
        a = IREngine._IREngine__find_input(self.IR.graph)
        self.assertTrue(a == ref_nodes, 'Error')

    def test_get_inputs(self):
        # Reference data for test:
        ref_input_dict = {'data': [1, 10, 16]}
        # Check function:
        inputs_dict = self.IR.get_inputs()
        # is_equal = compare_dictionaries(ref_input_dict, inputs_dict)
        self.assertTrue(ref_input_dict == inputs_dict, 'Test on function get_inputs failed')
        log.info('Test for function get_inputs passed')

    def test_eq_function(self):
        self.assertTrue(self.IR == self.IR_ref, 'Comparing false, test eq function failed')
        log.info('Test for function eq passed')

    @unittest.mock.patch('numpy.savez_compressed')
    def test_generate_bin_hashes_file(self, numpy_savez):
        # Generate bin_hashes file in default directory
        self.IR.generate_bin_hashes_file()
        numpy_savez.assert_called_once()
        log.info('Test for function generate_bin_hashes_file with default folder passed')

    @unittest.mock.patch('numpy.savez_compressed')
    def test_generate_bin_hashes_file_custom_directory(self, numpy_savez):
        # Generate bin_hashes file in custom directory
        directory_for_file = os.path.join(os.path.split(os.path.dirname(__file__))[0], "unittest", "test_data",
                                          "bin_hash")
        self.IR.generate_bin_hashes_file(path_for_file=directory_for_file)
        numpy_savez.assert_called_once()
        log.info('Test for function generate_bin_hashes_file with custom folder passed')

    @generate(*[({'order': '1,0,2'}, {'order': [1, 0, 2]}),
                ({'order': '1'}, {'order': 1})])
    def test_normalize_attr(self, test_data, reference):
        result_dict = IREngine._IREngine__normalize_attrs(attrs=test_data)
        self.assertTrue(reference == result_dict, 'Test on function normalize_attr failed')
        log.info('Test for function normalize_attr passed')

    def test_load_bin_hashes(self):
        path_for_file = self.IR.generate_bin_hashes_file()
        IR = IREngine(path_to_xml=str(self.xml), path_to_bin=str(path_for_file))
        is_ok = True
        # Check for constant nodes
        const_nodes = IR.graph.get_op_nodes(type='Const')
        for node in const_nodes:
            if not node.has_valid('hashes'):
                log.error('Constant node {} do not include hashes'.format(node.name))
                is_ok = False

        # Check for TensorIterator Body
        ti_nodes = IR.graph.get_op_nodes(type='TensorIterator')
        for ti in ti_nodes:
            if not ti.has_valid('body'):
                log.error('TensorIterator has not body attrubite for node: {}'.format(ti.name))
            else:
                const_ti_nodes = ti.body.graph.get_op_nodes(type='Const')
                for node in const_ti_nodes:
                    if not node.has_valid('hashes'):
                        log.error('Constant node {} do not include hashes'.format(node.name))
                        is_ok = False

        self.assertTrue(is_ok, 'Test for function load_bin_hashes failed')
        os.remove(path_for_file)

    @generate(*[
        ("0", True),
        ("1", True),
        ("-1", True),
        ("-", False),
        ("+1", True),
        ("+", False),
        ("1.0", False),
        ("-1.0", False),
        ("1.5", False),
        ("+1.5", False),
        ("abracadabra", False),
    ])
    def test_isint(self, value, result):
        self.assertEqual(IREngine._IREngine__isint(value), result)
