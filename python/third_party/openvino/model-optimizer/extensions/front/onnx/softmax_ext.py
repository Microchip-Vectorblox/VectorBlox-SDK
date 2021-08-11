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

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.softmax import SoftmaxONNX
from mo.ops.log_softmax import LogSoftmaxONNX


class SoftmaxExtractor(FrontExtractorOp):
    op = 'Softmax'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = onnx_attr(node, 'axis', 'i', default=1)
        SoftmaxONNX.update_node_stat(node, {'axis': axis})
        return cls.enabled


class LogSoftmaxExtractor(FrontExtractorOp):
    op = 'LogSoftmax'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = onnx_attr(node, 'axis', 'i', default=1)
        LogSoftmaxONNX.update_node_stat(node, {'axis': axis})
        return cls.enabled
