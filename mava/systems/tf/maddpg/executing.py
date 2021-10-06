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

"""MADDPG system executor implementation."""

from mava.systems.execution import OnlineSystemExecutor
from mava.components.tf import execution as tf_executing

# construct default executor components
#######################################
observer = tf_executing.OnlineObserver()
preprocess = tf_executing.Batch()
policy = tf_executing.DistributionPolicy()
action_selection = tf_executing.OnlineActionSampling()

# Executor components
executor_components = [
    observer,
    preprocess,
    policy,
    action_selection,
]

system_executor = OnlineSystemExecutor(components=executor_components)