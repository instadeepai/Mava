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

"""MAD4PG system executor implementation."""

from typing import Dict, Optional

import sonnet as snt
from acme.specs import EnvironmentSpec
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.systems.tf.maddpg.execution import (
    MADDPGFeedForwardExecutor,
    MADDPGRecurrentExecutor,
)


class MAD4PGFeedForwardExecutor(MADDPGFeedForwardExecutor):
    """A feed-forward executor for MAD4PG.
    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        shared_weights: bool = True,
    ):
        """Initialise the system executor

        Args:
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            agent_specs (Dict[str, EnvironmentSpec]): agent observation and action
                space specifications.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            shared_weights (bool, optional): whether agents should share weights or not.
                Defaults to True.
        """

        super().__init__(
            policy_networks=policy_networks,
            agent_specs=agent_specs,
            adder=adder,
            variable_client=variable_client,
            shared_weights=shared_weights,
        )


class MAD4PGRecurrentExecutor(MADDPGRecurrentExecutor):
    """A recurrent executor for MAD4PG.
    An executor based on a recurrent policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        shared_weights: bool = True,
    ):
        """Initialise the system executor

        Args:
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            agent_specs (Dict[str, EnvironmentSpec]): agent observation and action
                space specifications.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            shared_weights (bool, optional): whether agents should share weights or not.
                Defaults to True.
        """

        super().__init__(
            policy_networks=policy_networks,
            agent_specs=agent_specs,
            adder=adder,
            variable_client=variable_client,
            shared_weights=shared_weights,
        )
