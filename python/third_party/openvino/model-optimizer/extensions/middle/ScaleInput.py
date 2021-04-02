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
import numpy as np

from extensions.middle.AddMeanScaleValues import AddMeanScaleValues
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class ScaleInput(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.AddMeanScaleValues import AddMeanScaleValues
        return [AddMeanScaleValues]

    def pattern(self):
        return dict(
            nodes=[
                ('placeholder', dict(kind='op', op='Parameter')),
                ('data', dict(kind='data'))],
            edges=[
                ('placeholder', 'data'),
            ],
        )

    def replace_pattern(self, graph: Graph, match: dict):
        scale = graph.graph['cmd_params'].scale
        if scale is None or scale == 1:
            return
        assert (len(match['placeholder'].out_nodes()))

        AddMeanScaleValues.apply_scale(graph, match['placeholder'], {'scale': np.array([scale])})
