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

import networkx as nx

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class CheckForCycle(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
        return [TensorIteratorMerge]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        is_acyclic = nx.is_directed_acyclic_graph(graph)
        if not is_acyclic:
            raise Error('Graph contains a cycle. Can not proceed. ' + refer_to_faq_msg(97))
