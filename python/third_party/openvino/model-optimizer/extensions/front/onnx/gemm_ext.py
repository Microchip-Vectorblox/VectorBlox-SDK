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

from extensions.ops.MatMul import GemmONNX
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class GemmFrontExtractor(FrontExtractorOp):
    op = 'Gemm'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'alpha': onnx_attr(node, 'alpha', 'f', 1),
            'beta': onnx_attr(node, 'beta', 'f', 1),
            'transpose_a': onnx_attr(node, 'transA', 'i', 0),
            'transpose_b': onnx_attr(node, 'transB', 'i', 0),
            'broadcast_c': onnx_attr(node, 'broadcast', 'i', 1),
            # TODO: there is no axis in onnx operators.md
            'axis': np.array(onnx_attr(node, 'axis', 'i', default=0), dtype=np.int64)
        }
        GemmONNX.update_node_stat(node, attrs)
        return cls.enabled
