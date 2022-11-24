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

"""Parameter client for system builders"""
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Type

import jax.numpy as jnp
import numpy as np
from acme.jax import networks as networks_lib

from mava.callbacks import Callback
from mava.components import Component
from mava.components.training.trainer import BaseTrainerInit
from mava.core_jax import SystemBuilder
from mava.systems import ParameterClient


class BaseParameterClient(Component):
    def _get_count_parameters(self) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Registers parameters to count and store.

        Counts trainer_steps, trainer_walltime, evaluator_steps,
        evaluator_episodes, executor_episodes, executor_steps.

        Returns:
            Tuple of count parameters and network parameters.
        """
        counts = {
            "trainer_steps": np.array(0, dtype=np.int32),
            "trainer_walltime": np.array(0, dtype=np.float32),
            "evaluator_steps": np.array(0, dtype=np.int32),
            "evaluator_episodes": np.array(0, dtype=np.int32),
            "executor_episodes": np.array(0, dtype=np.int32),
            "executor_steps": np.array(0, dtype=np.int32),
        }
        return list(counts), counts

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required for this Component to function.

        BaseTrainerInit required to set up builder.store.networks
        and builder.store.trainer_networks.

        Returns:
            List of required component classes.
        """
        return [BaseTrainerInit]


@dataclass
class ExecutorParameterClientConfig:
    executor_parameter_update_period: int = 200


class ExecutorParameterClient(BaseParameterClient):
    def __init__(
        self,
        config: ExecutorParameterClientConfig = ExecutorParameterClientConfig(),
    ) -> None:
        """Component creates a parameter client for the executor.

        Args:
            config: ExecutorParameterClientConfig.
        """

        self.config = config

    def on_building_executor_parameter_client(self, builder: SystemBuilder) -> None:
        """Create and store the executor parameter client.

        Gets network parameters from store and registers them for tracking.
        Also works for the evaluator.

        Args:
            builder: SystemBuilder.
        """

        if not builder.store.parameter_server_client:
            builder.store.executor_parameter_client = None
            return

        # Create policy parameters
        params: Dict[str, Any] = {}
        # Executor does not explicitly set variables i.e. it adds to count variables
        # and hence set_keys is empty
        set_keys: List[str] = []
        get_keys: List[str] = []

        policy_param_keys, policy_params = self._get_policy_params(builder)
        get_keys.extend(policy_param_keys)
        params.update(policy_params)

        checkpointing_keys, checkpointing_params = self._get_checkpointing_params(
            builder
        )
        get_keys.extend(checkpointing_keys)
        params.update(checkpointing_params)

        count_keys, count_params = self._get_count_parameters()
        get_keys.extend(count_keys)
        params.update(count_params)
        builder.store.executor_counts = count_params

        norm_keys, norm_params = self._get_norm_params(builder)
        get_keys.extend(norm_keys)
        params.update(norm_params)

        # Create parameter client
        parameter_client = ParameterClient(
            client=builder.store.parameter_server_client,
            parameters=params,
            get_keys=get_keys,
            set_keys=set_keys,
            update_period=self.config.executor_parameter_update_period,
        )

        # Make sure not to use a random policy after checkpoint restoration by
        # assigning parameters before running the environment loop.
        parameter_client.get_and_wait()

        builder.store.executor_parameter_client = parameter_client

    def _get_policy_params(
        self, builder: SystemBuilder
    ) -> Tuple[List[str], Dict[str, networks_lib.Params]]:
        """Get the keys and params from the policy in the store

        Args:
            builder: SystemBuilder

        Returns:
            A list of parameter keys and a dictionary of policy parameters
        """
        policy_keys = []
        params: Dict[str, jnp.ndarray] = {}
        for agent_net_key in builder.store.networks.keys():
            policy_param_key = f"policy_network-{agent_net_key}"

            policy_keys.append(policy_param_key)
            params[policy_param_key] = builder.store.networks[
                agent_net_key
            ].policy_params

        return policy_keys, params

    # TODO (sasha): This should go in it's own checkpointer component and be called in
    #     on_building_init_end
    def _get_checkpointing_params(
        self, builder: SystemBuilder
    ) -> Tuple[List[str], Dict[str, networks_lib.Params]]:
        """Creates the keys and parameters need for checkpointing the best networks

        Args:
            builder: SystemBuilder

        Returns:
            best checkpoint keys and parameters
        """
        if not (builder.store.is_evaluator and builder.store.checkpoint_best_perf):
            return [], {}

        # Create best performance network params in case of evaluator
        builder.store.best_checkpoint: Dict[str, Any] = {}  # type:ignore
        for metric in builder.store.checkpointing_metric:
            builder.store.best_checkpoint[metric] = {}
            builder.store.best_checkpoint[metric]["best_performance"] = None
            for agent_net_key in builder.store.networks.keys():
                # add policy network
                builder.store.best_checkpoint[metric][
                    f"policy_network-{agent_net_key}"
                ] = copy.deepcopy(builder.store.networks[agent_net_key].policy_params)
                # add policy opt state
                builder.store.best_checkpoint[metric][
                    f"policy_opt_state-{agent_net_key}"
                ] = copy.deepcopy(builder.store.policy_opt_states[agent_net_key])

        return ["best_checkpoint"], {"best_checkpoint": builder.store.best_checkpoint}

    def _get_norm_params(
        self, builder: SystemBuilder
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Creates the keys and parameters need for observation normalisation

        Args:
            builder: SystemBuilder

        Returns:
            observation normalisation keys and parameters
        """
        # I hate this check so much. Really need a way to check:
        #     builder.has_component(Component)
        #     https://github.com/instadeepai/Mava/issues/845
        if hasattr(builder.store, "norm_params"):
            return ["norm_params"], {"norm_params": builder.store.norm_params}

        return [], {}

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "executor_parameter_client"


class ActorCriticExecutorParameterClient(ExecutorParameterClient):
    def _get_policy_params(
        self, builder: SystemBuilder
    ) -> Tuple[List[str], Dict[str, networks_lib.Params]]:
        policy_keys, policy_params = super()._get_policy_params(builder)

        for agent_net_key in builder.store.networks.keys():
            critic_param_key = f"critic_network-{agent_net_key}"
            policy_params[critic_param_key] = builder.store.networks[
                agent_net_key
            ].critic_params
            policy_keys.append(critic_param_key)

        return policy_keys, policy_params

    # TODO (sasha): this should get its own component
    def _get_checkpointing_params(
        self, builder: SystemBuilder
    ) -> Tuple[List[str], Dict[str, networks_lib.Params]]:
        if not (builder.store.is_evaluator and builder.store.checkpoint_best_perf):
            return [], {}

        keys, _ = super()._get_checkpointing_params(builder)

        for metric in builder.store.checkpointing_metric:
            for agent_net_key in builder.store.networks.keys():
                builder.store.best_checkpoint[metric][
                    f"critic_network-{agent_net_key}"
                ] = copy.deepcopy(builder.store.networks[agent_net_key].critic_params)
                builder.store.best_checkpoint[metric][
                    f"critic_opt_state-{agent_net_key}"
                ] = copy.deepcopy(builder.store.critic_opt_states[agent_net_key])

        return keys, {"best_checkpoint": builder.store.best_checkpoint}


@dataclass
class TrainerParameterClientConfig:
    trainer_parameter_update_period: int = 5


class TrainerParameterClient(BaseParameterClient):
    def __init__(
        self,
        config: TrainerParameterClientConfig = TrainerParameterClientConfig(),
    ) -> None:
        """Component creates a parameter client for the trainer.

        Args:
            config: TrainerParameterClientConfig.
        """

        self.config = config

    def on_building_trainer_parameter_client(self, builder: SystemBuilder) -> None:
        """Create and store the trainer parameter client.

        Gets network parameters from store and registers them for tracking.

        Args:
            builder: SystemBuilder.
        """
        if not builder.store.parameter_server_client:
            builder.store.trainer_parameter_client = None
            return

        # Create parameter client
        params: Dict[str, Any] = {}
        set_keys: List[str] = []
        get_keys: List[str] = []
        trainer_networks = set(builder.store.trainer_networks[builder.store.trainer_id])

        policy_set_keys, policy_get_keys, policy_params = self._get_policy_params(
            builder, trainer_networks
        )
        set_keys.extend(policy_set_keys)
        get_keys.extend(policy_get_keys)
        params.update(policy_params)

        count_keys, count_params = self._get_count_parameters()
        get_keys.extend(count_keys)
        params.update(count_params)
        builder.store.trainer_counts = count_params

        # Create parameter client
        parameter_client = ParameterClient(
            client=builder.store.parameter_server_client,
            parameters=params,
            get_keys=get_keys,
            set_keys=set_keys,
            update_period=self.config.trainer_parameter_update_period,
        )

        # Get all the initial parameters
        parameter_client.get_all_and_wait()

        builder.store.trainer_parameter_client = parameter_client

    def _get_policy_params(
        self, builder: SystemBuilder, trainer_networks: Set[str]
    ) -> Tuple[List[str], List[str], Dict[str, networks_lib.Params]]:
        """Get the set and get keys and params from the policy in the store

        Args:
            builder: SystemBuilder
            trainer_networks: the networks belonging to this trainer

        Returns:
            list of parameter keys to set, list of parameter keys to get and parameters
        """
        # TODO (dries): Only add the networks this trainer is working with.
        # Not all of them.
        set_keys = []
        get_keys = []
        params = {}

        for net_key in builder.store.networks.keys():
            params[f"policy_network-{net_key}"] = builder.store.networks[
                net_key
            ].policy_params

            if net_key in trainer_networks:
                set_keys.append(f"policy_network-{net_key}")
            else:
                get_keys.append(f"policy_network-{net_key}")

            params[f"policy_opt_state-{net_key}"] = builder.store.policy_opt_states[
                net_key
            ]
            set_keys.append(f"policy_opt_state-{net_key}")

        return set_keys, get_keys, params

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "trainer_parameter_client"


class ActorCriticTrainerParameterClient(TrainerParameterClient):
    def _get_policy_params(
        self, builder: SystemBuilder, trainer_networks: Set[str]
    ) -> Tuple[List[str], List[str], Dict[str, networks_lib.Params]]:
        """Get the set and get keys and params from the policy in the store

        Args:
            builder: SystemBuilder
            trainer_networks: the networks belonging to this trainer

        Returns:
            list of parameter keys to set, list of parameter keys to get and parameters
        """
        set_keys, get_keys, params = super()._get_policy_params(
            builder, trainer_networks
        )

        for net_key in builder.store.networks.keys():
            params[f"critic_network-{net_key}"] = builder.store.networks[
                net_key
            ].critic_params

            if net_key in set(trainer_networks):
                set_keys.append(f"critic_network-{net_key}")
            else:
                get_keys.append(f"critic_network-{net_key}")

            params[f"critic_opt_state-{net_key}"] = builder.store.critic_opt_states[
                net_key
            ]
            set_keys.append(f"critic_opt_state-{net_key}")

        return set_keys, get_keys, params
