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

from mo.graph.graph import Node


def node_pb_arg(pb_extractor: callable):
    return lambda node: pb_extractor(node.pb)


onnx_op_extractors = {}


def common_onnx_fields(node: Node):
    return {
        'kind': 'op',
        'name': node.id,
         # no reliable name for an onnx node, name can be empty, so we use that surrogate built as ID in the loader
        'op': node.op if node.has_valid('op') else node.pb.op_type,
    }


def onnx_op_extractor(node: Node, lowered_keys_map: dict):
    if not node.has_valid('pb'):
        return True, node.graph.node[node.id]

    result = common_onnx_fields(node)
    node.graph.node[node.id].update(result)
    supported = False
    op = result['op'].lower()
    if op in lowered_keys_map:
        op = lowered_keys_map[op]
        assert op in onnx_op_extractors
        attrs = onnx_op_extractors[op](node)
        if attrs:
            result.update(attrs)
            supported = True
    return supported, result
