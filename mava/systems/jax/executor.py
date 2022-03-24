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

"""Jax system executor."""

from typing import Any, Dict, List, Optional, Tuple, Union

import dm_env

from mava import adders, types
from mava.callbacks import Callback, ExecutorHookMixin
from mava.core_jax import SystemExecutor
from mava.systems.jax import ParameterClient


class Executor(SystemExecutor, ExecutorHookMixin):
    """A generic executor."""

    def __init__(
        self,
        executor_id: str,
        networks: Dict[str, Any],
        adder: Optional[adders.ReverbParallelAdder] = None,
        parameter_client: Optional[ParameterClient] = None,
        components: List[Callback] = [],
    ):
        """_summary_

        Args:
            executor_id : _description_
            networks : _description_
            adder : _description_.
            parameter_client : _description_.
            components : _description_.
        """
        self.executor_id = executor_id
        self.networks = networks
        self.adder = adder
        self.parameter_client = parameter_client
        self.callbacks = components

        self.on_execution_init_start()

        self.on_execution_init()

        self.on_execution_init_end()

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record observed timestep from the environment.

        Args:
            actions : system agents' actions
            timestep : data emitted by an environment during interaction
            extras : possible extra information to record during the transition
        """
        self._actions = actions
        self._timestep = timestep
        self._extras = extras

        self.on_execution_observe_start()

        self.on_execution_observe()

        self.on_execution_observe_end()

    def select_action(
        self,
        agent: str,
        observation: types.NestedArray,
        state: types.NestedArray = None,
    ) -> types.NestedArray:
        """Agent specific policy function.

        Args:
            agent : agent id
            observation : observation tensor received from the environment
            state : recurrent state
        Returns:
            agent action
        """
        self._agent = agent
        self._observation = observation
        self._state = state

        self.on_execution_select_action_start()

        self.on_execution_select_action_preprocess()

        self.on_execution_select_action_compute()

        self.on_execution_select_action_sample()

        self.on_execution_select_action_end()

        return self.attr.action_info, self.attr.policy_info

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Union[
        Dict[str, types.NestedArray],
        Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]],
    ]:
        """Select the actions for all agents in the system.

        Args:
            observations : agent observations from the environment
        Returns:
            actions for all agents in the system.
        """
        self._observations = observations

        self.on_execution_select_actions_start()

        self.on_execution_select_actions()

        self.on_execution_select_actions_end()

        return self.attr.actions_info, self.attr.policies_info

    def update(self, wait: bool = False) -> None:
        """Update executor parameters.

        Args:
            wait : whether to stall the executor's request for new parameter
        """
        self._wait = wait

        self.on_execution_update_start()

        self.on_execution_update()

        self.on_execution_update_end()
