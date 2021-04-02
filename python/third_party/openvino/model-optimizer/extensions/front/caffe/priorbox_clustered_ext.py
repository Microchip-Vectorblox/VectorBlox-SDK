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
from extensions.ops.priorbox_clustered import PriorBoxClusteredOp
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp


class PriorBoxClusteredFrontExtractor(FrontExtractorOp):
    op = 'PriorBoxClustered'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.prior_box_param

        variance = param.variance
        if len(variance) == 0:
            variance = [0.1]

        update_attrs = {
            'width': list(param.width),
            'height': list(param.height),
            'flip': int(param.flip),
            'clip': int(param.clip),
            'variance': list(variance),
            'img_size': param.img_size,
            'img_h': param.img_h,
            'img_w': param.img_w,
            'step': param.step,
            'step_h': param.step_h,
            'step_w': param.step_w,
            'offset': param.offset,
        }

        mapping_rule = merge_attrs(param, update_attrs)

        mapping_rule.update(layout_attrs())

        # update the attributes of the node
        PriorBoxClusteredOp.update_node_stat(node, mapping_rule)
        return cls.enabled
