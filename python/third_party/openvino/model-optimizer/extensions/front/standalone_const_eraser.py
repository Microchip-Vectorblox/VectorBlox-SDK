"""
 Copyright (C) 2018-2021 Intel Corporation

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

import logging as log

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph


class StandaloneConstEraser(FrontReplacementSubgraph):
    enabled = True
    # TODO: remove this transformation once all plugins support constant value network.
    # Now it avoids to be run recursively since Const->Result sub-graph can be encountered in a body graph of Loop node
    run_not_recursively = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[('const', dict(kind='op', op='Const')),
                   ('output', dict(kind='op', op='Result'))
                   ],
            edges=[('const', 'output')]
        )

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        if not len(match['const'].in_edges()) and len(match['const'].out_edges()) == 1:
            graph.erase_node(match['const'])
            graph.erase_node(match['output'])
            log.info("Standalone Const node \"{}\" was removed from the graph".format(match['const'].id))
