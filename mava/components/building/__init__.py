# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

"""Components for building systems"""

from mava.components.building.adders import (
    ParallelNStepTransitionAdderSignature,
    ParallelSequenceAdderSignature,
    ParallelNStepTransitionAdder,
    ParallelSequenceAdder,
)
from mava.components.building.rate_limiters import OffPolicyRateLimiter
from mava.components.building.replay_tables import OffPolicyReplayTables
from mava.components.building.dataset_iterators import DatasetIterator
from mava.components.building.executors import Executor
from mava.components.building.trainers import Trainer
from mava.components.building.variable_source import VariableSource