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

from extensions.ops.RNN import RNN
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class RNNFrontExtractor(FrontExtractorOp):
    op = 'RNN'
    enabled = True

    @classmethod
    def extract(cls, node):
        direction = onnx_attr(node, 'direction', 's', b'forward').decode().lower()

        activation_alpha = onnx_attr(node, 'activation_alpha', 'floats',
                                     default=None, dst_type=lambda x: np.array(x, dtype=np.float32))
        activation_beta = onnx_attr(node, 'activation_beta', 'floats',
                                    default=None, dst_type=lambda x: np.array(x, dtype=np.float32))
        activations = onnx_attr(node, 'activations', 'strings',
                                default=['tanh', 'tanh'] if direction == 'bidirectional' else ['tanh'],
                                dst_type=lambda x: list(map(lambda s: s.decode(encoding="utf-8").lower(), list(x))))
        clip = onnx_attr(node, 'clip', 'f', default=None)

        # Since pytorch generates ONNX bidirectional RNN models with only one activation, duplicating activation
        if direction == 'bidirectional' and len(activations) == 1:
            activations.append(activations[0])

        attrs = {
            'batch_dim': 1,
            'sequence_dim': 0,
            'blobs_wrb': True,
            'has_num_directions': True,
            'num_layers': 1,
            'format': 'onnx',
            'multilayers': False,
            'gate_order': [0],

            # ONNX attrs
            'activation_alpha': activation_alpha,
            'activation_beta': activation_beta,
            'activations': activations,
            'clip': clip,
            'direction': direction,
            'hidden_size': np.array(onnx_attr(node, 'hidden_size', 'i'), dtype=np.int64),
        }

        RNN.update_node_stat(node, attrs)
        return cls.enabled
