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

import numpy as np

from extensions.ops.elementwise import Add, Mul
from mo.front.common.layout import get_features_dim
from mo.front.extractor import split_node_in_port, get_node_id_with_ports
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class AddMeanScaleValues(MiddleReplacementPattern):
    enabled = True
    run_not_recursively = True

    def run_after(self):
        return []

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    @staticmethod
    def insert_pre_processing(graph: Graph, input_node: Node, node_mean_scale_values: np.array,
                              preprocessing_name: str):
        assert preprocessing_name in ['scale', 'mean']
        if node_mean_scale_values.get(preprocessing_name) is None:
            return
        user_value = node_mean_scale_values[preprocessing_name]
        value = 1 / user_value if preprocessing_name == 'scale' else user_value * (-1)
        optimize_value = int(preprocessing_name == 'scale')
        op = Mul if preprocessing_name == 'scale' else Add

        if all([x == optimize_value for x in value]):
            return
        assert input_node.has_valid('shape')
        features_dim_idx = get_features_dim(graph.graph['layout'], len(input_node.shape))
        assert value.size == input_node.shape[features_dim_idx] or value.size == 1

        shape = np.ones(len(input_node.shape), dtype=np.int64)
        shape[features_dim_idx] = value.size
        value = value.reshape(shape)

        name = input_node.soft_get('name', input_node.id) + '/' + preprocessing_name
        preprocessing = create_op_with_const_inputs(graph, op=op, port_value_dict={1: value}, op_attrs={'name': name})

        for dst in input_node.out_port(0).get_destinations():
            if dst.node.soft_get('type') != 'ShapeOf':
                # After the insertion of additional operations model optimizer
                # should keep the link to the input layer. Parameter node in framework
                # should map to parameter node in IR.
                # For this reason 'fw_tensor_debug_info' should be kept in data node.
                dst.get_connection().set_source(preprocessing.out_port(0), "source")

        input_node.out_port(0).connect(preprocessing.in_port(0))

    @staticmethod
    def apply_scale(graph: Graph, input_node: Node, node_mean_scale_values: dict):
        AddMeanScaleValues.insert_pre_processing(graph, input_node, node_mean_scale_values, preprocessing_name='scale')

    @staticmethod
    def apply_mean_value(graph: Graph, input_node: Node, node_mean_scale_values: dict):
        AddMeanScaleValues.insert_pre_processing(graph, input_node, node_mean_scale_values, preprocessing_name='mean')

    def find_and_replace_pattern(self, graph: Graph):
        values = graph.graph['cmd_params'].mean_scale_values
        input_nodes = graph.get_op_nodes(op='Parameter')

        if not isinstance(values, dict):
            # The case when input names to apply mean/scales weren't specified
            if len(values) != len(input_nodes):
                raise Error('Numbers of inputs and mean/scale values do not match. ' + refer_to_faq_msg(61))

            data = np.copy(values)
            values = {}
            for idx, node in enumerate(input_nodes):
                values.update(
                    {
                        node.soft_get('name', node.id): {
                            'mean': data[idx][0],
                            'scale': data[idx][1]
                        }
                    }
                )

        for node_name, node_mean_scale_values in values.items():
            node_id = None
            try:
                node_id, direction, port = get_node_id_with_ports(graph, node_name, skip_if_no_port=False)
                assert direction != 'out', 'Only input port can be specified for mean/scale application'
            except Error as e:
                log.warning('node_name {} is not found in graph'.format(node_name))
            if Node(graph, node_id) not in input_nodes:
                # if the user cutted-off input of the network then input node name specified in the --scale_values
                # or --mean_values doesn't correspond to a real input node generated by Model Optimizer. But
                # the information about initial input node name is stored in Placeholder's attribute 'initial_node_name'
                new_node_id = None
                for placeholder in input_nodes:
                    try:
                        placeholder_port = int(placeholder.id.split("_")[-1])
                    except Exception as ex:
                        log.debug('Can not get the port number from the node {}'.format(placeholder.id))
                        log.debug('Port will be defined as None')
                        port = None
                    if placeholder.has('initial_node_name') and placeholder.initial_node_name == node_id and (
                            port is None or placeholder_port == port):
                        new_node_id = placeholder.id
                        break
                if new_node_id is None:
                    raise Error('Input with name {} wasn\'t found!'.format(node_name) +
                                refer_to_faq_msg(83))
                node_id = new_node_id

            input_node = Node(graph, node_id)
            AddMeanScaleValues.apply_scale(graph, input_node, node_mean_scale_values)
            AddMeanScaleValues.apply_mean_value(graph, input_node, node_mean_scale_values)
