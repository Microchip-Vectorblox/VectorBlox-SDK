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
from extensions.ops.identity import Identity, IdentityN
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_dtype_extractor
from mo.graph.graph import Node


class IdentityFrontExtractor(FrontExtractorOp):
    op = 'Identity'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Identity.update_node_stat(node, {
            'data_type': tf_dtype_extractor(node.pb.attr["T"].type),
        })
        return cls.enabled


class IdentityNFrontExtractor(FrontExtractorOp):
    op = 'IdentityN'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        dtypes = [tf_dtype_extractor(t) for t in node.pb.attr["T"].list.type]
        IdentityN.update_node_stat(node, {
            'data_types': dtypes,
            'in_ports_count': len(dtypes),
            'out_ports_count': len(dtypes),
        })
        return cls.enabled


class ReadVariableOpFrontExtractor(FrontExtractorOp):
    op = 'ReadVariableOp'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Identity.update_node_stat(node, {
            'data_type': tf_dtype_extractor(node.pb.attr["T"].type),
        })
        return cls.enabled


class StopGradientExtractor(FrontExtractorOp):
    op = 'StopGradient'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Identity.update_node_stat(node, {'op': 'StopGradient'})
        return cls.enabled
