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

from extensions.ops.MatMul import FullyConnected
from mo.front.kaldi.extractors.affine_transform_ext import AffineTransformFrontExtractor
from mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading
from mo.ops.op import Op


class AffineComponentFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['FullyConnected'] = FullyConnected

    @classmethod
    def create_pb_for_test_node(cls):
        pb = KaldiFrontExtractorTest.generate_learn_info()
        pb += KaldiFrontExtractorTest.generate_matrix([10, 10])
        pb += KaldiFrontExtractorTest.generate_vector(10)
        cls.test_node['parameters'] = TestKaldiUtilsLoading.bytesio_from(pb)
        AffineTransformFrontExtractor.extract(cls.test_node)

    def test_assertion(self):
        self.assertRaises(AttributeError, AffineTransformFrontExtractor.extract, None)

    def test_attrs(self):
        self.assertEqual(self.test_node['out-size'], 10)

    def test_out_blobs(self):
        self.assertTrue(np.array_equal(self.test_node.weights, range(10 * 10)))
        self.assertTrue(np.array_equal(self.test_node.biases, range(10)))
