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

from mo.utils import class_registration
from mo.utils.replacement_pattern import ReplacementPattern


class BackReplacementPattern(ReplacementPattern):
    registered_ops = {}
    registered_cls = []

    def run_after(self):
        from extensions.back.pass_separator import BackStart
        return [BackStart]

    def run_before(self):
        from extensions.back.pass_separator import BackFinish
        return [BackFinish]

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.BACK_REPLACER


ReplacementPattern.excluded_replacers.append(BackReplacementPattern)
