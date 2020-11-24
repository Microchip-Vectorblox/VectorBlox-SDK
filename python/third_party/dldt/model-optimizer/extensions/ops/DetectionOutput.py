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

from mo.front.common.partial_infer.multi_box_detection import multi_box_detection_infer
from mo.graph.graph import Graph, Node
from mo.ops.op import Op


class DetectionOutput(Op):
    op = 'DetectionOutput'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': multi_box_detection_infer,
            'input_width': 1,
            'input_height': 1,
            'normalized': 1,
            'share_location': 1,
            'variance_encoded_in_target': 0,
            'type_infer': self.type_infer,
        }, attrs)

    def supported_attrs(self):
        return [
            'background_label_id',
            'clip_after_nms',
            'clip_before_nms',
            'code_type',
            'confidence_threshold',
            'decrease_label_id',
            'eta',
            'height',
            'height_scale',
            'input_height',
            'input_width',
            'interp_mode',
            'keep_top_k',
            'label_map_file',
            'name_size_file',
            'nms_threshold',
            'normalized',
            'num_classes',
            'num_test_image',
            'output_directory',
            'output_format',
            'output_name_prefix',
            'pad_mode',
            'pad_value',
            'prob',
            'resize_mode',
            'save_file',
            'share_location',
            'top_k',
            'variance_encoded_in_target',
            'visualize',
            'visualize_threshold',
            'width',
            'width_scale',
            'objectness_score',
        ]

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(np.float32)
