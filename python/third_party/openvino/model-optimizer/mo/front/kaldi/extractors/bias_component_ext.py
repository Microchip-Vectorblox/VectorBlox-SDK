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

from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import find_next_tag, read_placeholder, collect_until_token
from mo.front.kaldi.utils import read_binary_vector
from mo.ops.scale_shift import ScaleShiftOp


class FixedBiasComponentFrontExtractor(FrontExtractorOp):
    op = 'fixedbiascomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters
        collect_until_token(pb, b'<Bias>')
        biases = read_binary_vector(pb)
        find_next_tag(pb)
        read_placeholder(pb, 1)

        mapping_rule = {
            'layout': 'NCHW',
            'bias_term': True,
            'out-size': biases.shape[0],
        }
        embed_input(mapping_rule, 2, 'biases', biases)

        ScaleShiftOp.update_node_stat(node, mapping_rule)
        return cls.enabled
