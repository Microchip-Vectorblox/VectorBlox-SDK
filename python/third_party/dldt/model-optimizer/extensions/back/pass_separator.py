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
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class BackStart(BackReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.ApplyPermutations import ApplyPermutation
        return [ApplyPermutation]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        pass


class BackFinish(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        return []

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        pass
