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
from typing import Any, Dict, List, Optional, Tuple, Union

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme import types
from acme.specs import EnvironmentSpec

# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from dm_env import specs

from mava import adders
from mava.systems.tf import executors
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray
tfd = tfp.distributions

from mava.callbacks import Callback


class SystemExecutor(mava.Executor, Callback):
    """[summary]

    Args:
        mava ([type]): [description]
        Callback ([type]): [description]

    Returns:
        [type]: [description]
    """


class OnlineSystemExecutor(SystemExecutor):
    """A generic feed-forward executor.
    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):
        """Initialise the system executor

        Args:
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
        """

        self.on_execution_init_start(self)

        self._agent_net_keys = agent_net_keys
        self._policy_networks = policy_networks
        self._adder = adder
        self._variable_client = variable_client

        self.on_execution_init_end(self)

    @tf.function
    def _policy(
        self, agent: str, observation: types.NestedTensor
    ) -> types.NestedTensor:
        """Agent specific policy function

        Args:
            agent (str): agent id
            observation (types.NestedTensor): observation tensor received from the
                environment.

        Returns:
            types.NestedTensor: agent action
        """
        self._agent = agent
        self._observation = observation

        self.on_execution_policy_start(self)

        self.on_execution_policy_preprocess(self)

        self.on_execution_policy_compute(self)

        self.on_execution_policy_sample_action(self)

        self.on_execution_policy_end(self)

        return self.action_info

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> Union[types.NestedArray, Tuple[types.NestedArray, types.NestedArray]]:
        """select an action for a single agent in the system

        Args:
            agent (str): agent id.
            observation (types.NestedArray): observation tensor received from the
                environment.

        Returns:
            Union[types.NestedArray, Tuple[types.NestedArray, types.NestedArray]]:
                agent action.
        """
        self._agent = agent
        self._observation = observation

        self.on_execution_select_action_start(self)

        self.on_execution_select_action(self)

        self.on_execution_select_action_end(self)

        return self.action

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """record first observed timestep from the environment

        Args:
            timestep (dm_env.TimeStep): data emitted by an environment at first step of
                interaction.
            extras (Dict[str, types.NestedArray], optional): possible extra information
                to record during the first step. Defaults to {}.
        """
        self._timestep = timestep
        self._extras = extras

        self.on_execution_observe_first_start(self)

        self.on_execution_observe_first(self)

        self.on_execution_observe_first_end(self)

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """record observed timestep from the environment

        Args:
            actions (Dict[str, types.NestedArray]): system agents' actions.
            next_timestep (dm_env.TimeStep): data emitted by an environment during
                interaction.
            next_extras (Dict[str, types.NestedArray], optional): possible extra
                information to record during the transition. Defaults to {}.
        """
        self._actions = actions
        self._next_timestep = next_timestep
        self._next_extras = next_extras

        self.on_execution_observe_start(self)

        self.on_execution_observe(self)

        self.on_execution_observe_end(self)

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Union[
        Dict[str, types.NestedArray],
        Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]],
    ]:
        """select the actions for all agents in the system

        Args:
            observations (Dict[str, types.NestedArray]): agent observations from the
                environment.

        Returns:
            Union[ Dict[str, types.NestedArray], Tuple[Dict[str, types.NestedArray],
                Dict[str, types.NestedArray]], ]: actions for all agents in the system.
        """
        self._observation = observations

        self.on_execution_select_actions_start(self)

        self.on_execution_select_actions(self)

        self.on_execution_select_actions_end(self)

        return self.actions_info

    def update(self, wait: bool = False) -> None:
        """update executor variables

        Args
            wait (bool, optional): whether to stall the executor's request for new
                variables. Defaults to False.
        """
        self._wait = wait

        self.on_execution_update_start(self)

        self.on_execution_update(self)

        self.on_execution_update_end(self)