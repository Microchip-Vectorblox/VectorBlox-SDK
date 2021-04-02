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
import logging as log

import numpy as np

from extensions.middle.FuseReshapesSequence import FuseReshapesSequence
from extensions.middle.pass_separator import PostMiddleStart
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class RemoveRedundantReshapes(MiddleReplacementPattern):
    """
    Finds Reshape layers that does nothing and removes them.
    """
    enabled = True
    force_clean_up = True
    run_not_recursively = True  # non-unified data nodes view in TI body (no Const ops, bare data node)

    def run_after(self):
        return [FuseReshapesSequence]

    def run_before(self):
        return [PostMiddleStart]

    def find_and_replace_pattern(self, graph: Graph):
        for reshape_node in graph.get_op_nodes(type='Reshape'):
            in_ports = [port for port in reshape_node.in_ports().values() if not port.disconnected()]
            assert len(in_ports) == 2, "`Reshape` node must have 2 inputs"
            previous_dim_op = reshape_node.in_port(1).get_source().node.op
            if previous_dim_op != 'Const':
                continue
            dim = reshape_node.in_port(1).get_connection().data.get_value()

            in_shape = reshape_node.in_port(0).data.get_shape()

            if np.array_equal(dim, in_shape) and len(reshape_node.out_nodes()):
                log.debug("Useless reshape with dim {} was deleted: {}".format(str(dim), reshape_node.name))
                reshape_node.out_port(0).get_connection().set_source(reshape_node.in_port(0).get_source())
                graph.remove_nodes_from([reshape_node.out_node(0).id, reshape_node.id])
