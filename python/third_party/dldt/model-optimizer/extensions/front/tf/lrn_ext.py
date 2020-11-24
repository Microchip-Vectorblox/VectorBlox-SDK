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
from mo.ops.lrn import AttributedLRN


class LRNExtractor(FrontExtractorOp):
    """
        TF and IE(CAFFE) parameters in LRN differs in several places :
            region (IE) : in TF there is no such parameter, they just use last dimension (feature dimension in case of NHWC)
            local-size (IE) : it's the size of 1D vector in Caffe. In TF they have 'depth_radius' that eq
            '(local-size * 2) + 1'
            alpha (IE) : in Caffe 'alpha' divides on local-size, so we should multiply alpha on local-size

        Caffe ref : http://caffe.berkeleyvision.org/tutorial/layers/lrn.html
        TF ref : https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
    """
    op = 'LRN'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.pb
        AttributedLRN.update_node_stat(node, {
            'alpha': pb.attr['alpha'].f * (2. * pb.attr['depth_radius'].i + 1.),
            'beta': pb.attr['beta'].f,
            'bias': pb.attr['bias'].f,
            'local_size': (2 * pb.attr['depth_radius'].i + 1),
        })
        return cls.enabled
