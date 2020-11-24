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

from extensions.ops.topk import TopK
from mo.front.extractor import FrontExtractorOp


class TopKExtractor(FrontExtractorOp):
    op = 'TopK'
    enabled = True

    @classmethod
    def extract(cls, node):
        sort = 'value' if node.pb.attr['sorted'] else 'none'
        TopK.update_node_stat(node, {'mode': 'max', 'axis': -1, 'sort': sort, 'k': node.pb.attr['k'].i,
                                     'index_element_type': np.int32})

        return cls.enabled


class TopKV2Extractor(FrontExtractorOp):
    op = 'TopKV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        sort = 'value' if node.pb.attr['sorted'] else 'none'
        TopK.update_node_stat(node, {'mode': 'max', 'axis': -1, 'sort': sort, 'index_element_type': np.int32})
        return cls.enabled
