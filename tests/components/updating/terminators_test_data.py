# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dummy data needed for the terminator unit test"""

from typing import Dict, List, Tuple


def count_condition_terminator_data() -> List[Dict]:
    """Data for count condition terminator"""

    return [
        {"trainer_steps": 10},
        {"trainer_walltime": 10},
        {"evaluator_steps": 10},
        {"evaluator_episodes": 10},
        {"executor_episodes": 10},
        {"executor_steps": 10},
    ]


def count_condition_terminator_failure_cases() -> List[Tuple]:
    """Data for count condition terminator"""

    return [
        ({"trainer_step": 10}, Exception),
        ({"trainer_steps": 10, "executor_steps": 10}, Exception),
        ({"trainer_step": -10}, Exception),
    ]
