"""
 Copyright (C) 2020-2021 Intel Corporation

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

try:
    import tensorflow.compat.v1 as tf_v1
    # disable eager execution of TensorFlow 2 environment immediately
    tf_v1.disable_eager_execution()
except ImportError:
    import tensorflow as tf_v1

try:
    import tensorflow.contrib  # pylint: disable=no-name-in-module,import-error
except:
    pass  # we try to import contrib for loading models that use contrib operations

import logging as log

from extensions.load.loader import Loader
from mo.front.common.register_custom_ops import check_for_duplicates
from mo.front.common.register_custom_ops import update_extractors_with_extensions
from mo.front.extractor import restore_edges, extract_node_attrs, remove_control_dependency_inputs
from mo.front.tf.extractor import get_tf_edges, tf_op_extractor, tf_op_extractors
from mo.front.tf.loader import load_tf_graph_def, protobuf2nx
from mo.graph.graph import Graph
from mo.utils import tensorboard_util
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class TFLoader(Loader):
    enabled = True
    run_not_recursively = True

    def load(self, graph: Graph):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_custom_layer_libraries:
            libraries = argv.tensorflow_custom_layer_libraries.split(',')
            for library in libraries:
                log.info('Loading library "{}" with custom operations'.format(library))
                tf_v1.load_op_library(library)

        graph_def, variables_values = load_tf_graph_def(graph_file_name=argv.input_model,
                                                        is_binary=not argv.input_model_is_text,
                                                        checkpoint=argv.input_checkpoint,
                                                        user_output_node_names_list=argv.output,
                                                        model_dir=argv.saved_model_dir,
                                                        meta_graph_file=argv.input_meta_graph,
                                                        saved_model_tags=argv.saved_model_tags)

        try:
            tf_v1.import_graph_def(graph_def, name='')
        except:
            log.warning("TensorFlow post-processing of loaded model was unsuccessful. "
                        "This is an optional step that Model Optimizer performs for any input model but it is not usually "
                        "required for all models."
                        "It likely means that the original model is ill-formed. "
                        "Model Optimizer will continue converting this model.")

        log.debug("Number of nodes in graph_def: {}".format(len(graph_def.node)))  # pylint: disable=no-member

        if argv.tensorboard_logdir:
            tensorboard_util.dump_for_tensorboard(graph_def, argv.tensorboard_logdir)

        update_extractors_with_extensions(tf_op_extractors)

        try:
            protobuf2nx(graph, graph_def)
        except Exception as e:
            raise Error(
                'Cannot pre-process TensorFlow graph after reading from model file "{}". ' \
                'File is corrupt or has unsupported format. Details: {}. ' +
                refer_to_faq_msg(44),
                argv.model_name,
                str(e)
            ) from e

        graph.__setattr__('name', argv.model_name)
        # 'layout' parameter change may cause an issue in EltwiseInputReshape replacer
        # and convert_nhwc_to_nchw(graph)
        graph.graph['layout'] = 'NCHW' if argv.disable_nhwc_to_nchw else 'NHWC'
        graph.graph['fw'] = 'tf'

        graph.graph['variables_values'] = variables_values
        del variables_values

        restore_edges(graph, get_tf_edges)
        remove_control_dependency_inputs(graph)

        graph.check_empty_graph('protobuf2nx. It may happen due to problems with loaded model')
        extract_node_attrs(graph, lambda node: tf_op_extractor(node, check_for_duplicates(tf_op_extractors)))
