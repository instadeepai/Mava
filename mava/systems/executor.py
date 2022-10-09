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

from mava import constants
from mava.callbacks import Callback, ExecutorHookMixin
from mava.core_jax import SystemExecutor
from mava.utils.jax_training_utils import normalize_observations


class Executor(SystemExecutor, ExecutorHookMixin):
    """Core system executor."""

    def __init__(
        self,
        store: SimpleNamespace,
        components: List[Callback] = [],
    ):
        """Initialise the executor.

        Call to the init hooks.
        Save whether or not this is an evaluator.

        Args:
            store : builder store.
            components : list of system components.
        """
        self.store = store
        self.callbacks = components

        self._evaluator = self.store.is_evaluator

        self.on_execution_init_start()

        self.on_execution_init()

        self.on_execution_init_end()

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, NestedArray] = {},
    ) -> None:
        """Record observed timestep from environment first episode step.

        Args:
            timestep : data emitted by an environment during interaction.
            extras : possible extra information to record during the transition.
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
        """Record observed timestep from the environment.

        Args:
            actions : actions taken by agents in previous step.
            next_timestep : data emitted by an environment during interaction.
            next_extras : possible extra information to record during the transition.
        """
        self.store.actions = actions
        self.store.next_timestep = next_timestep
        self.store.next_extras = next_extras

        self.on_execution_observe_start()

        self.on_execution_observe()

        self.on_execution_observe_end()

    # NB: Not currently used. TODO Deprecate in future.
    def select_action(
        self,
        agent: str,
        observation: NestedArray,
        state: NestedArray = None,
    ) -> NestedArray:
        """Agent specific policy function.

        Args:
            agent : agent id.
            observation : observation tensor received from the environment.
            state : recurrent state.

        Returns:
            Action and policy info for agent.
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
            observations : agent observations from the environment.

        Returns:
            Action and policy info for all agents in the system.
        """

        # Normalise the observations before selecting actions.
        if self.store.global_config.normalize_observations:
            observations_stats = self.store.obs_norm_params[
                constants.OBS_NORM_STATE_DICT_KEY
            ]
            for key in observations.keys():
                observations[key] = normalize_observations(
                    observations_stats[key], observations[key]
                )

        self.store.observations = observations

        self.on_execution_select_actions_start()

        self.on_execution_select_actions()

        self.on_execution_select_actions_end()

        return self.store.actions_info, self.store.policies_info

    def update(self, wait: bool = False) -> None:
        """Update executor parameters.

        Args:
            wait : whether to stall the executor's request for new parameter.

        Returns:
            None.
        """
        self.store._wait = wait

        self.on_execution_update_start()

        self.on_execution_update()

        self.on_execution_update_end()

    def force_update(self, wait: bool = False) -> None:
        """Force immediate update executor parameters.

        Args:
            wait : whether to stall the executor's request for new parameter.

        Returns:
            None.
        """
        self.store._wait = wait

        self.on_execution_force_update_start()

        self.on_execution_force_update()

        self.on_execution_force_update_end()
