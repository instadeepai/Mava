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

"""Trainer components for gradient step calculations."""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type

import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
from jax import jit

import mava.components.building.adders  # To avoid circular imports
import mava.components.training.model_updating  # To avoid circular imports
from mava import constants
from mava.callbacks import Callback
from mava.components.training.base import DQNTrainingState
from mava.components.training.step import Step
from mava.core_jax import SystemTrainer
from mava.systems.idqn.components.training.loss import IDQNLoss


@dataclass
class IDQNStepConfig:
    target_update_period: int = 100


class IDQNStep(Step):
    def __init__(
        self,
        config: IDQNStepConfig = IDQNStepConfig(),
    ):
        """Component defines the MAPGWithTrustRegion SGD step.

        Args:
            config: MAPGWithTrustRegionStepConfig.
        """
        self.config = config

    # flake8: noqa: C901
    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """Define and store the SGD step function for MAPGWithTrustRegion.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        @jit
        def sgd_step(
            states: DQNTrainingState, sample: reverb.ReplaySample
        ) -> Tuple[DQNTrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step.

            Args:
                states: Training states (network params and optimiser states).
                sample: Reverb sample.

            Returns:
                Tuple[new state, metrics].
            """

            # Extract the data.
            data = sample.data

            observations, actions, rewards, next_observations, discounts, _ = (
                data.observations,  # type: ignore
                data.actions,  # type: ignore
                data.rewards,  # type: ignore
                data.next_observations,  # type: ignore
                data.discounts,  # type: ignore
                data.extras,  # type: ignore
            )

            target_policy_params = states.target_policy_params
            policy_params = states.policy_params
            policy_opt_states = states.policy_opt_states

            policy_gradients, grad_metrics = trainer.store.policy_grad_fn(
                policy_params,
                target_policy_params,
                observations,
                actions,
                rewards,
                next_observations,
                discounts,
            )

            metrics: Dict[str, jnp.ndarray] = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                # Update the networks and optimisers.
                # TODO (dries): Use one optimiser per network type here and not
                # just one.
                (
                    policy_updates,
                    policy_opt_states[agent_net_key][constants.OPT_STATE_DICT_KEY],  # type: ignore
                ) = trainer.store.policy_optimiser.update(
                    policy_gradients[agent_key],
                    policy_opt_states[agent_net_key][constants.OPT_STATE_DICT_KEY],  # type: ignore
                )
                policy_params[agent_net_key] = optax.apply_updates(
                    policy_params[agent_net_key], policy_updates
                )

                metrics[agent_key] = {
                    "norm_policy_grad": optax.global_norm(policy_gradients[agent_key]),
                    "norm_policy_updates": optax.global_norm(policy_updates),
                }

            # update target net
            target_policy_params = rlax.periodic_update(
                policy_params,
                target_policy_params,
                states.trainer_iteration,  # type: ignore
                self.config.target_update_period,
            )

            # Set the metrics
            metrics = jax.tree_util.tree_map(jnp.mean, {**metrics, **grad_metrics})

            new_states = DQNTrainingState(
                policy_params=policy_params,
                target_policy_params=target_policy_params,
                policy_opt_states=policy_opt_states,
                random_key=states.random_key,
                trainer_iteration=states.trainer_iteration,
            )
            return new_states, metrics

        def step(sample: reverb.ReplaySample) -> Tuple[Dict[str, jnp.ndarray]]:
            """Step over the reverb sample and update the parameters / optimiser states.

            Args:
                sample: Reverb sample.

            Returns:
                Metrics from SGD step.
            """

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            networks = trainer.store.networks
            policy_params = {
                net_key: networks[net_key].policy_params for net_key in networks.keys()
            }
            target_policy_params = {
                net_key: networks[net_key].target_policy_params
                for net_key in networks.keys()
            }

            policy_opt_states = trainer.store.policy_opt_states

            _, random_key = jax.random.split(trainer.store.base_key)

            steps = trainer.store.trainer_counts["trainer_steps"]
            states = DQNTrainingState(
                policy_params=policy_params,
                target_policy_params=target_policy_params,
                policy_opt_states=policy_opt_states,
                random_key=random_key,
                trainer_iteration=steps,
            )

            new_states, metrics = sgd_step(states, sample)

            # Set the new variables
            # TODO (dries): key is probably not being store correctly.
            # The variable client might lose reference to it when checkpointing.
            # We also need to add the optimiser and random_key to the variable
            # server.
            trainer.store.base_key = new_states.random_key

            # UPDATING THE PARAMETERS IN THE NETWORK IN THE STORE
            for net_key in policy_params.keys():
                # The for loop below is needed to not lose the param reference.
                net_params = trainer.store.networks[net_key].policy_params
                for param_key in net_params.keys():
                    net_params[param_key] = new_states.policy_params[net_key][param_key]

                target_net_params = trainer.store.networks[net_key].target_policy_params
                for param_key in target_net_params.keys():
                    target_net_params[param_key] = new_states.target_policy_params[
                        net_key
                    ][param_key]

                # Update the policy optimiser
                # The opt_states need to be wrapped in a dict so as not to lose
                # the reference.
                trainer.store.policy_opt_states[net_key][
                    constants.OPT_STATE_DICT_KEY
                ] = new_states.policy_opt_states[net_key][constants.OPT_STATE_DICT_KEY]

            return metrics

        trainer.store.step_fn = step

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        Returns:
            List of required component classes.
        """
        return Step.required_components() + [
            IDQNLoss,
            mava.components.building.adders.ParallelTransitionAdder,
        ]
