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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class GatherTree(Op):
    op = 'GatherTree'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'type': __class__.op,
            'version': 'opset1',
            'infer': __class__.infer,
            'in_ports_count': 4,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return []

    @staticmethod
    def infer(node: Node):
        node.out_node().shape = node.in_node(0).shape
