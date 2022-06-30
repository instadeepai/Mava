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

from types import SimpleNamespace
from typing import Dict, List, Tuple, Union

import dm_env
from acme.types import NestedArray

from mava.callbacks import Callback, ExecutorHookMixin
from mava.core_jax import SystemExecutor


class Executor(SystemExecutor, ExecutorHookMixin):
    """A generic executor."""

    def __init__(
        self,
        store: SimpleNamespace,
        components: List[Callback] = [],
    ):
        """_summary_

        Args:
            store : _description_.
            components : _description_.
        """
        self.store = store
        self.callbacks = components

        self.on_execution_init_start()

        self.on_execution_init()

        self.on_execution_init_end()

        self._evaluator = self.store.is_evaluator

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, NestedArray] = {},
    ) -> None:
        """Record observed timestep from the environment.

        Args:
            timestep : data emitted by an environment during interaction
            extras : possible extra information to record during the transition
        """
        self.store.timestep = timestep
        self.store.extras = extras

        self.on_execution_observe_first_start()

        self.on_execution_observe_first()

        self.on_execution_observe_first_end()

    def observe(
        self,
        actions: Dict[str, NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, NestedArray] = {},
    ) -> None:
        """_summary_

        Args:
            actions : _description_
            next_timestep : _description_
            next_extras : _description_.
        """
        self.store.actions = actions
        self.store.next_timestep = next_timestep
        self.store.next_extras = next_extras

        self.on_execution_observe_start()

        self.on_execution_observe()

        self.on_execution_observe_end()

    def select_action(
        self,
        agent: str,
        observation: NestedArray,
        state: NestedArray = None,
    ) -> NestedArray:
        """Agent specific policy function.

        Args:
            agent : agent id
            observation : observation tensor received from the environment
            state : recurrent state
        Returns:
            agent action
        """
        self.store.agent = agent
        self.store.observation = observation
        self.store.state = state

        self.on_execution_select_action_start()

        self.on_execution_select_action_preprocess()

        self.on_execution_select_action_compute()

        self.on_execution_select_action_sample()

        self.on_execution_select_action_end()

        return self.store.action_info, self.store.policy_info

    def select_actions(
        self, observations: Dict[str, NestedArray]
    ) -> Union[
        Dict[str, NestedArray],
        Tuple[Dict[str, NestedArray], Dict[str, NestedArray]],
    ]:
        """Select the actions for all agents in the system.

        Args:
            observations : agent observations from the environment
        Returns:
            actions for all agents in the system.
        """
        self.store.observations = observations

        self.on_execution_select_actions_start()

        self.on_execution_select_actions()

        self.on_execution_select_actions_end()

        return self.store.actions_info, self.store.policies_info

    def update(self, wait: bool = False) -> None:
        """Update executor parameters.

        Args:
            wait : whether to stall the executor's request for new parameter
        """
        self.store._wait = wait

        self.on_execution_update_start()

        self.on_execution_update()

        self.on_execution_update_end()
