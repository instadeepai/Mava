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

"""MAPPO trainer implementation."""
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import trfl
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava

tfd = tfp.distributions


class MAPPOTrainer(mava.Trainer):
    """MAPPO trainer.
    This is the trainer component of a MAPPO system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        dataset: tf.data.Dataset,
        shared_weights: bool,
        critic_learning_rate: float = 1e-3,
        policy_learning_rate: float = 1e-3,
        discount: float = 0.99,
        lambda_gae: float = 1.0,
        entropy_cost: float = 0.0,
        baseline_cost: float = 1.0,
        clipping_epsilon: float = 0.2,
        max_abs_reward: Optional[float] = None,
        max_gradient_norm: Optional[float] = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = False,
        checkpoint_subpath: str = "Checkpoints",
    ):
        """Initializes the learner.
        Args:
            policy_networks: ...
            critic_networks: ...
            shared_weights: ...
            discount: discount to use for TD updates.
            dataset: dataset to learn from, whether fixed or from a replay buffer
                (see `acme.datasets.reverb.make_dataset` documentation).
            critic_learning_rate: ...
            policy_learning_rate: ...
            lambda_gae: ...
            clipping_espilon: ...
            entropy_cost: ...
            baseline_cost: ...
            max_abs_reward: ...
            max_gradient_norm: ...
            clipping: whether to clip gradients by global norm.
            counter: counter object used to keep track of steps.
            logger: logger object to be used by learner.
            checkpoint: boolean indicating whether to checkpoint the learner.
        """
        self._shared_weights = shared_weights

        # Store networks.
        self._policy_networks = policy_networks
        self._critic_networks = critic_networks
        self._is_recurrent = isinstance(list(policy_networks.values())[0], snt.DeepRNN)

        # Get optimizers
        self._policy_optimizer = snt.optimizers.Adam(learning_rate=policy_learning_rate)
        self._critic_optimizer = snt.optimizers.Adam(learning_rate=critic_learning_rate)

        # Other trainer parameters.
        self._discount = discount
        self._entropy_cost = entropy_cost
        self._baseline_cost = baseline_cost
        self._lambda_gae = lambda_gae
        self._clipping_epsilon = clipping_epsilon

        # Dataset iterator
        self._iterator = dataset

        # Set up reward/gradient clipping.
        if max_abs_reward is None:
            max_abs_reward = np.inf
        if max_gradient_norm is None:
            max_gradient_norm = 1e10  # A very large number. Infinity results in NaNs.

        self._max_abs_reward = tf.convert_to_tensor(max_abs_reward)
        self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("trainer")
        self._system_checkpointer: Dict[Any, Any] = {}

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    # @tf.function
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        # Get batch of data from replay
        inputs: reverb.ReplaySample = next(self._iterator)

        # Convert to sequence data
        data = tf2_utils.batch_to_sequence(inputs.data)

        # Unpack input data as follows:
        all_obs, all_acts, all_rews, all_discs, extras = (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.extras,
        )

        # Get logits and possibly core_states
        all_logits = extras["logits"]
        if self._is_recurrent:
            all_core_states = extras["core_states"]

        metrics: Dict[str, Dict[str, Any]] = {}
        # Do forward passes through the networks and calculate the losses
        for agent in all_obs.keys():

            obs, acts, rews, discs, behaviour_logits = (
                all_obs[agent],
                all_acts[agent],
                all_rews[agent],
                all_discs[agent],
                all_logits[agent],
            )

            # Get agent core_states
            if self._is_recurrent:
                core_states = all_core_states[agent]
                core_states = tree.map_structure(lambda s: s[0], core_states)

            # Chop off final timestep for bootstrapping value
            rews = rews[:-1]
            discs = discs[:-1]

            # Get agent network
            agent_key = agent.split("_")[0] if self._shared_weights else agent
            policy_network = self._policy_networks[agent_key]
            critic_network = self._critic_networks[agent_key]

            with tf.GradientTape(persistent=True) as tape:
                if self._is_recurrent:
                    # Unroll current policy over observations.
                    logits, _ = snt.static_unroll(
                        policy_network, obs.observation, core_states
                    )

                    # Unroll current value network over observations.
                    values, _ = snt.static_unroll(
                        critic_network, obs.observation, core_states
                    )
                    values = tf.squeeze(values)
                else:
                    # Reshape inputs
                    dims = obs.observation.shape[:2]
                    obs = snt.merge_leading_dims(obs.observation, num_dims=2)
                    logits = policy_network(obs)
                    values = critic_network(obs)

                    # reshape the outputs
                    logits = tf.reshape(logits, (*dims, -1), name="policy")
                    values = tf.reshape(values, dims, name="value")

                # Values along the sequence T
                bootstrap_value = values[-1]
                state_values = values[:-1]

                # Optionally clip rewards.
                rews = tf.clip_by_value(
                    rews,
                    tf.cast(-self._max_abs_reward, rews.dtype),
                    tf.cast(self._max_abs_reward, rews.dtype),
                )

                # Generalized Return Estimation
                td_loss, td_lambda_extra = trfl.td_lambda(
                    state_values=state_values,
                    rewards=rews,
                    pcontinues=discs,
                    bootstrap_value=bootstrap_value,
                    lambda_=self._lambda_gae,
                    name="CriticLoss",
                )

                # Do not use the loss provided by td_lambda as they sum the losses over
                # the sequence length rather than averaging them.
                critic_loss = tf.reduce_mean(
                    tf.square(td_lambda_extra.temporal_differences), name="CriticLoss"
                )

                # Compute importance sampling weights: current policy / behavior policy.
                pi_behaviour = tfd.Categorical(logits=behaviour_logits)
                pi_target = tfd.Categorical(logits=logits)
                log_rhos = pi_target.log_prob(acts) - pi_behaviour.log_prob(acts)
                importance_ratio = tf.exp(log_rhos)[:-1]
                clipped_importance_ratio = tf.clip_by_value(
                    importance_ratio,
                    1.0 - self._clipping_epsilon,
                    1.0 + self._clipping_epsilon,
                )

                # Generalized Advantage Estimation
                gae = tf.stop_gradient(td_lambda_extra.temporal_differences)
                mean, variance = tf.nn.moments(gae, axes=[0, 1], keepdims=True)
                normalized_gae = (gae - mean) / tf.sqrt(variance)

                policy_gradient_loss = -tf.minimum(
                    tf.multiply(importance_ratio, normalized_gae),
                    tf.multiply(clipped_importance_ratio, normalized_gae),
                )

                # Entropy regularization.
                # TODO verify that this entropy regularization
                # works as we expect.
                scale = 1.0 / tf.math.log(
                    tf.convert_to_tensor(logits.shape[-1], dtype=float)
                )
                entropy = trfl.policy_entropy_loss(pi_target)
                entropy_loss = self._entropy_cost * entropy.loss[:-1]
                policy_entropy = scale * tf.reduce_mean(entropy.extra.entropy)

                # Combine weighted sum of actor & entropy regularization.
                policy_loss = tf.reduce_mean(
                    policy_gradient_loss + entropy_loss, name="PolicyLoss"
                )

            # Compute gradients.
            critic_grads = tape.gradient(
                critic_loss, critic_network.trainable_variables
            )
            policy_grads = tape.gradient(
                policy_loss, policy_network.trainable_variables
            )
            del tape

            # Optionally apply clipping.
            critic_grads, critic_norm = tf.clip_by_global_norm(
                critic_grads, self._max_gradient_norm
            )
            policy_grads, policy_norm = tf.clip_by_global_norm(
                policy_grads, self._max_gradient_norm
            )

            # Apply gradients.
            self._critic_optimizer.apply(
                critic_grads, critic_network.trainable_variables
            )
            self._policy_optimizer.apply(
                policy_grads, policy_network.trainable_variables
            )

            metrics.update(
                {
                    agent: {
                        "policy_loss": policy_loss,
                        "critic_loss": critic_loss,
                        "entropy_loss": tf.reduce_mean(entropy_loss),
                        "policy_entropy": policy_entropy,
                        "policy_gradient_loss": tf.reduce_mean(policy_gradient_loss),
                        "advantage": tf.reduce_mean(gae),
                        "state_value": tf.reduce_mean(state_values),
                        "temporal_difference": tf.reduce_mean(
                            td_lambda_extra.temporal_differences
                        ),
                        "returns": tf.reduce_mean(td_lambda_extra.discounted_returns),
                        "policy_grad_norm": policy_norm,
                        "critic_grad_norm": critic_norm,
                    }
                }
            )

        return metrics

    def step(self) -> None:

        # Run the learning step.
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        if self._timestamp:
            elapsed_time = timestamp - self._timestamp
        else:
            elapsed_time = 0
        self._timestamp = timestamp  # type: ignore

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint the networks.
        if len(self._system_checkpointer.keys()) > 0:
            for agent_key in self.unique_net_keys:
                checkpointer = self._system_checkpointer[agent_key]
                checkpointer.save()

        self._logger.write(fetches)

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        variables: Dict[str, Dict[str, np.ndarray]] = {}
        variables["network"] = {}
        for agent_key, network in self._networks.items():
            variables["network"][agent_key] = tf2_utils.to_numpy(
                network.trainable_variables
            )
        return variables
