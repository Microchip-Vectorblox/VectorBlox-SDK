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

from extensions.ops.elementwise import Minimum, Maximum
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, rename_node
from mo.ops.clamp import AttributedClamp


class ClampNormalizer(BackReplacementPattern):
    """
    Replaces Clamp with `min` and `max` as inputs with AttributedClamp with `min` and `max` as attributes.
    """
    enabled = True
    force_clean_up = True

    def pattern(self):
        return dict(
            nodes=[('clamp', dict(op='Clamp'))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        clamp = match['clamp']
        name = clamp.soft_get('name', clamp.id)

        min_value = max_value = None
        port_1_exist = clamp.has_port('in', 1) and not clamp.in_port(1).disconnected()
        port_2_exist = clamp.has_port('in', 2) and not clamp.in_port(2).disconnected()
        if port_1_exist and clamp.in_port(1).get_source().node.soft_get('type') == 'Const':
            min_value = clamp.in_port(1).data.get_value()
        if port_2_exist and clamp.in_port(2).get_source().node.soft_get('type') == 'Const':
            max_value = clamp.in_port(2).data.get_value()

        rename_node(clamp, name + '/TBR')
        if min_value is None or max_value is None:
            max_node = min_node = None
            if port_1_exist:
                max_node = Maximum(graph, {}).create_node()
                clamp.in_port(0).get_connection().set_destination(max_node.in_port(0))
                clamp.in_port(1).get_connection().set_destination(max_node.in_port(1))
                clamp.out_port(0).get_connection().set_source(max_node.out_port(0))
            if port_2_exist:
                min_node = Minimum(graph, {}).create_node()
                if max_node is not None:
                    max_node.out_port(0).get_connection().set_source(min_node.out_port(0))
                    max_node.out_port(0).connect(min_node.in_port(0))
                else:
                    clamp.in_port(0).get_connection().set_destination(min_node.in_port(0))
                    clamp.out_port(0).get_connection().set_source(min_node.out_port(0))
                clamp.in_port(2).get_connection().set_destination(min_node.in_port(1))
            assert min_node is not None or max_node is not None, 'Clamp node should have either min or max input used'
            rename_node(min_node if min_node is not None else max_node, name)
        else:
            a_clamp = AttributedClamp(graph, {'name': name, 'min': min_value, 'max': max_value}).create_node()
            rename_node(a_clamp, name)
            clamp.in_port(0).get_connection().set_destination(a_clamp.in_port(0))
            clamp.out_port(0).get_connection().set_source(a_clamp.out_port(0))
