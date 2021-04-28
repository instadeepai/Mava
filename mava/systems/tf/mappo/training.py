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

# TODO (Siphelele): implement MAPPO trainer
# Helper resources
#   - single agent impala learner in acme:
#       https://github.com/deepmind/acme/blob/master/acme/agents/tf/impala/learning.py
#   - multi-agent ddpg trainer in mava: mava/systems/tf/maddpg/trainer.py

"""MAPPO trainer implementation."""
import time
from typing import Any, Dict, List, Optional, Sequence

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
    """MADDPG trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        shared_weights: bool,
        networks: Dict[str, snt.RNNCore],
        dataset: tf.data.Dataset,
        critic_learning_rate: float,
        policy_learning_rate: float,
        discount: float = 0.99,
        lambda_gae: float = 1.0,
        clipping_epsilon: float = 0.2,
        entropy_cost: float = 0.0,
        baseline_cost: float = 1.0,
        max_abs_reward: Optional[float] = None,
        max_gradient_norm: Optional[float] = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "Checkpoints",
    ):
        """Initializes the learner.
        Args:
          policy_network: the online (optimized) policy.
          critic_network: the online critic.
          target_policy_network: the target policy (which lags behind the online
            policy).
          target_critic_network: the target critic.
          discount: discount to use for TD updates.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          dataset: dataset to learn from, whether fixed or from a replay buffer
            (see `acme.datasets.reverb.make_dataset` documentation).
          observation_network: an optional online network to process observations
            before the policy and the critic.
          target_observation_network: the target observation network.
          policy_optimizer: the optimizer to be applied to the DPG (policy) loss.
          critic_optimizer: the optimizer to be applied to the critic loss.
          clipping: whether to clip gradients by global norm.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        self._agents = agents
        self._agent_types = agent_types
        self._shared_weights = shared_weights

        # Store networks and get optimizers.
        self._networks = networks
        self._policy_optimizer = snt.optimizers.Adam(learning_rate=policy_learning_rate)
        self._critic_optimizer = snt.optimizers.Adam(learning_rate=critic_learning_rate)

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("trainer")

        # Other learner parameters.
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

        # TODO Create checkpointer
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

        # Get core states and logits
        all_core_states = extras["core_states"]
        all_logits = extras["logits"]

        metrics: Dict[str, Dict[str, Any]] = {}
        # Do forward passes through the networks and calculate the losses
        for agent in self._agents:

            obs, acts, rews, discs, behaviour_logits, core_states = (
                all_obs[agent],
                all_acts[agent],
                all_rews[agent],
                all_discs[agent],
                all_logits[agent],
                all_core_states[agent],
            )

            # Chop off final timestep for bootstrapping value
            acts = acts[:-1]
            rews = rews[:-1]
            discs = discs[:-1]

            core_states = tree.map_structure(lambda s: s[0], core_states)

            # Get agent network
            agent_key = agent.split("_")[0] if self._shared_weights else agent
            network = self._networks[agent_key]

            with tf.GradientTape(persistent=True) as tape:
                # Unroll current policy over observations.
                (logits, values), _ = snt.static_unroll(
                    network, obs.observation, core_states
                )

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
                critic_loss = self._baseline_cost * tf.reduce_mean(
                    tf.square(td_lambda_extra.temporal_differences), name="CriticLoss"
                )

                # Compute importance sampling weights: current policy / behavior policy.
                pi_behaviour = tfd.Categorical(logits=behaviour_logits[:-1])
                pi_target = tfd.Categorical(logits=logits[:-1])
                log_rhos = pi_target.log_prob(acts) - pi_behaviour.log_prob(acts)
                importance_ratio = tf.exp(log_rhos)
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
                scale = 1.0 / tf.math.log(
                    tf.convert_to_tensor(logits.shape[-1], dtype=float)
                )
                entropy = trfl.policy_entropy_loss(pi_target)
                entropy_loss = self._entropy_cost * entropy.loss
                policy_entropy = scale * tf.reduce_mean(entropy.extra.entropy)

                # Combine weighted sum of actor & entropy regularization.
                policy_loss = tf.reduce_mean(
                    policy_gradient_loss + entropy_loss, name="PolicyLoss"
                )

            # Compute gradients.
            critic_grads = tape.gradient(critic_loss, network.trainable_variables)
            policy_grads = tape.gradient(policy_loss, network.trainable_variables)
            del tape

            # Optionally apply clipping.
            critic_grads, critic_norm = tf.clip_by_global_norm(
                critic_grads, self._max_gradient_norm
            )
            policy_grads, policy_norm = tf.clip_by_global_norm(
                policy_grads, self._max_gradient_norm
            )

            # Apply gradients.
            self._critic_optimizer.apply(critic_grads, network.trainable_variables)
            self._policy_optimizer.apply(policy_grads, network.trainable_variables)

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
