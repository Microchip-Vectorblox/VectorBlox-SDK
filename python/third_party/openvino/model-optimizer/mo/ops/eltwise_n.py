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

from mo.graph.graph import Graph
from mo.ops.op import Op
from mo.utils.error import Error


class EltwiseN(Op):
    """
    The elementwise operation that has more than 2 inputs. This operation is replaced in a front phase with a number of
    simple elementwise operations with 2 inputs. Refer to EltwiseNFrontReplacer for a list of supported operations.
    """
    op = 'EltwiseN'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,  # type is None because this operation should not appear in IR
            'infer': None,
            'out_ports_count': 1,
        }, attrs)
        if 'operation' not in self.attrs:
            raise Error('"operation" attribute is not set for operation "{}".'.format(self.op))


class EltwiseNMul(EltwiseN):
    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {'operation': 'mul'})


class EltwiseNMin(EltwiseN):
    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {'operation': 'min'})


class EltwiseNMax(EltwiseN):
    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {'operation': 'max'})


class EltwiseNAdd(EltwiseN):
    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {'operation': 'sum'})
