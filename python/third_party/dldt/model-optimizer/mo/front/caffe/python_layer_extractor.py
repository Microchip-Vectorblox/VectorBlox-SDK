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
from mo.front.extractor import FrontExtractorOp, CaffePythonFrontExtractorOp


class PythonFrontExtractorOp(FrontExtractorOp):
    op = 'Python'
    enabled = True

    @classmethod
    def extract(cls, node):
        module = node.pb.python_param.module
        layer = node.pb.python_param.layer
        layer_type = '{}.{}'.format(module, layer)
        if layer_type and layer_type in CaffePythonFrontExtractorOp.registered_ops:
            if hasattr(CaffePythonFrontExtractorOp.registered_ops[layer_type], 'extract'):
                # CaffePythonFrontExtractorOp.registered_ops[layer_type] is object of FrontExtractorOp and has the
                # function extract
                return CaffePythonFrontExtractorOp.registered_ops[layer_type].extract(node)
            else:
                # User defined only Op for this layer and CaffePythonFrontExtractorOp.registered_ops[layer_type] is
                # special extractor for Op
                return CaffePythonFrontExtractorOp.registered_ops[layer_type](node)
