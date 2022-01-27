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
from typing import Any, Dict, List, Optional

import numpy as np
import sonnet as snt
import tensorflow_probability as tfp
from acme.specs import EnvironmentSpec

# Internal imports.
from acme.tf import variable_utils as tf2_variable_utils
from dm_env import specs

from mava import adders
from mava.systems.tf import executors

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray
tfd = tfp.distributions


class MADDPGFeedForwardExecutor(executors.FeedForwardExecutor):
    """A feed-forward executor for discrete actions.
    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        network_sampling_setup: List,
        net_keys_to_ids: Dict[str, int],
        evaluator: bool = False,
        adder: Optional[adders.ReverbParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        interval: Optional[dict] = None,
    ):

        """Initialise the system executor
        Args:
            policy_networks: policy networks for each agent in
                the system.
            agent_specs: agent observation and action
                space specifications.
            agent_net_keys: specifies what network each agent uses.
            network_sampling_setup: List of networks that are randomly
                sampled from by the executors at the start of an environment run.
            net_keys_to_ids: Specifies a mapping from network keys to their integer id.
            adder: adder which sends data
                to a replay buffer. Defaults to None.
            counts: Count values used to record excutor episode and steps.
            variable_client:
                client to copy weights from the trainer. Defaults to None.
            evaluator: whether the executor will be used for
                evaluation.
            interval: interval that evaluations are run at.
        """

        # Store these for later use.
        self._agent_specs = agent_specs
        self._network_sampling_setup = network_sampling_setup
        self._counts = counts
        self._network_int_keys_extras: Dict[str, np.ndarray] = {}
        self._net_keys_to_ids = net_keys_to_ids
        self._evaluator = evaluator
        self._interval = interval
        super().__init__(
            policy_networks=policy_networks,
            agent_net_keys=agent_net_keys,
            adder=adder,
            variable_client=variable_client,
        )
