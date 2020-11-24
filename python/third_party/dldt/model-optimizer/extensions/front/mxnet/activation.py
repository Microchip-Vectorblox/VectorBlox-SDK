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

from extensions.ops.activation_ops import SoftPlus, Sigmoid, Tanh, ReLU, Asinh, Acosh, Atanh
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class ActivationFrontExtractor(FrontExtractorOp):
    op = 'Activation'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        act_type = attrs.str('act_type', 'relu')
        if act_type == 'sigmoid':
            act_class = Sigmoid
        elif act_type == 'tanh':
            act_class = Tanh
        elif act_type == 'relu':
            act_class = ReLU
        elif act_type == 'softrelu':
            act_class = SoftPlus
        else:
            raise Error(
                "Operation '{}' not supported. Please register it as custom op. " +
                refer_to_faq_msg(86),
                act_type)
        act_class.update_node_stat(node)
        return cls.enabled


class AsinhFrontExtractor(FrontExtractorOp):
    op = 'arcsinh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Asinh.update_node_stat(node)
        return cls.enabled


class AcoshFrontExtractor(FrontExtractorOp):
    op = 'arccosh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Acosh.update_node_stat(node)
        return cls.enabled


class AtanhFrontExtractor(FrontExtractorOp):
    op = 'arctanh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Atanh.update_node_stat(node)
        return cls.enabled
