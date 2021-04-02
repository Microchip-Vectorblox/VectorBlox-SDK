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
from mo.ops.pad import TFPad


class PadFrontExtractor(FrontExtractorOp):
    op = 'Pad'
    enabled = True

    @classmethod
    def extract(cls, node):
        TFPad.update_node_stat(node)
        return cls.enabled


class PadV2FrontExtractor(FrontExtractorOp):
    op = 'PadV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        TFPad.update_node_stat(node)
        return cls.enabled


class MirrorPadFrontExtractor(FrontExtractorOp):
    op = 'MirrorPad'
    enabled = True

    @classmethod
    def extract(cls, node):
        TFPad.update_node_stat(node, {'mode': node.pb.attr['mode'].s.decode('utf-8').lower()})
        return cls.enabled
