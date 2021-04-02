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

import unittest

from extensions.front.mxnet.multibox_detection_ext import MultiBoxDetectionOutputExtractor
from mo.utils.unittest.extractors import PB


class TestMultiBoxDetection_Parsing(unittest.TestCase):
    def test_multi_box_detection_check_attrs(self):
        params = {'attrs': {
            "force_suppress": "True",
            "nms_threshold": "0.4",
            "nms_topk": "400",
            "variances": "(0.1, 0.1, 0.2, 0.2)"
        }}

        node = PB({'symbol_dict': params})
        MultiBoxDetectionOutputExtractor.extract(node)

        exp_attrs = {
            'type': 'DetectionOutput',
            'num_classes': 21,
            'keep_top_k': 400,
            'variance_encoded_in_target': 0,
            'code_type': "caffe.PriorBoxParameter.CENTER_SIZE",
            'share_location': 1,
            'confidence_threshold': 0.01,
            'background_label_id': 0,
            'nms_threshold': 0.4,
            'top_k': 400,
            'decrease_label_id': 1,
            'clip_before_nms': 1,
            'normalized': 1,
        }

        for key in exp_attrs.keys():
            self.assertEqual(node[key], exp_attrs[key])

    def test_multi_box_detection_check_attrs_without_top_k(self):
        params = {'attrs': {
            "force_suppress": "True",
            "nms_threshold": "0.2",
            "threshold": "0.02",
            "variances": "(0.1, 0.1, 0.2, 0.2)"
        }}

        node = PB({'symbol_dict': params})
        MultiBoxDetectionOutputExtractor.extract(node)

        exp_attrs = {
            'type': 'DetectionOutput',
            'num_classes': 21,
            'keep_top_k': -1,
            'variance_encoded_in_target': 0,
            'code_type': "caffe.PriorBoxParameter.CENTER_SIZE",
            'share_location': 1,
            'confidence_threshold': 0.02,
            'background_label_id': 0,
            'nms_threshold': 0.2,
            'top_k': -1,
            'decrease_label_id': 1,
            'clip_before_nms': 1,
            'normalized': 1,
        }

        for key in exp_attrs.keys():
            self.assertEqual(node[key], exp_attrs[key])
