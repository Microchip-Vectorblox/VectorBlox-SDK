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

from extensions.ops.mxrepeat import MXRepeat
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.graph.graph import Node


class RepeatExt(FrontExtractorOp):
    op = 'repeat'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        axis = attrs.int('axis', 0)
        repeats = attrs.int('repeats', None)
        assert repeats is not None and repeats > 0, \
            '`repeat` op requires positive `repeats` attribute, but it is {} for node {}'.format(repeats, node.name)

        MXRepeat.update_node_stat(node, {
            'axis': axis,
            'repeats': repeats,
        })
        return cls.enabled
