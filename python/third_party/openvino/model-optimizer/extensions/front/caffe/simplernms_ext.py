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
from extensions.ops.simplernms import SimplerNMSOp
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.extractor import FrontExtractorOp


class SimplerNMSFrontExtractor(FrontExtractorOp):
    op = 'SimplerNMS'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.simpler_nms_param
        update_attrs = {
            'cls_threshold': param.cls_threshold,
            'max_num_proposals': param.max_num_proposals,
            'iou_threshold': param.iou_threshold,
            'min_bbox_size': param.min_bbox_size,
            'feat_stride': param.feat_stride,
            'pre_nms_topn': param.pre_nms_topn,
            'post_nms_topn': param.post_nms_topn,
            'scale': param.scale,
        }

        mapping_rule = merge_attrs(param, update_attrs)

        # update the attributes of the node
        SimplerNMSOp.update_node_stat(node, mapping_rule)
        return cls.enabled
