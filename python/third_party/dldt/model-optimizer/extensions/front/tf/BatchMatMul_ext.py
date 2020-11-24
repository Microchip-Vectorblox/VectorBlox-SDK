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

from extensions.ops.MatMul import MatMul
from mo.front.extractor import FrontExtractorOp


class BatchMatMulExtractor(FrontExtractorOp):
    op = 'BatchMatMul'
    enabled = True

    @classmethod
    def extract(cls, node):
        attr = node.pb.attr
        attrs = {
            'transpose_a': int(attr['adj_x'].b),
            'transpose_b': int(attr['adj_y'].b),
        }
        MatMul.update_node_stat(node, attrs)
        return cls.enabled


class BatchMatMulV2Extractor(FrontExtractorOp):
    op = 'BatchMatMulV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        attr = node.pb.attr
        attrs = {
            'transpose_a': int(attr['adj_x'].b),
            'transpose_b': int(attr['adj_y'].b),
        }
        MatMul.update_node_stat(node, attrs)
        return cls.enabled
