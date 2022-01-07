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

from mava.components import building
from mava.components.tf import execution as tf_executing

"""MADDPG system executor implementation."""

##############
#   Executor
##############

# config = {
#     policy_networks: Dict[str, snt.Module],
#     agent_specs: Dict[str, EnvironmentSpec],
#     agent_net_keys: Dict[str, str],
#     network_sampling_setup: List,
#     net_keys_to_ids: Dict[str, int],
#     evaluator: bool = False,
#     adder: Optional[adders.ParallelAdder] = None,
#     counts: Optional[Dict[str, Any]] = None,
#     variable_client: Optional[tf2_variable_utils.VariableClient] = None,
#     interval: Optional[dict] = None,
# }

observer = tf_executing.OnlineObserver()
preprocess = tf_executing.Batch()
policy = tf_executing.DistributionPolicy()
action_selection = tf_executing.OnlineActionSampling()

executor_components = [
    observer,
    preprocess,
    policy,
    action_selection,
]

# Executor
executor = building.Executor(config=..., components=executor_components)