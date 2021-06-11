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
import copy
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import trfl
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava
from mava.systems.tf import savers as tf2_savers
from mava.utils import training_utils as train_utils

train_utils.set_growing_gpu_memory()

tfd = tfp.distributions


class MAPPOTrainer(mava.Trainer):
    """MAPPO trainer.
    This is the trainer component of a MAPPO system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[Any],
        agent_types: List[str],
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        dataset: tf.data.Dataset,
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        shared_weights: bool,
        discount: float = 0.99,
        lambda_gae: float = 1.0,
        entropy_cost: float = 0.0,
        baseline_cost: float = 1.0,
        clipping_epsilon: float = 0.2,
        max_gradient_norm: Optional[float] = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = False,
        checkpoint_subpath: str = "~/mava/",
    ):
        """Initializes the learner.
        Args:
            policy_networks: ...
            critic_networks: ...
            policy_optimizer: policy optimizer,
                This can be a single optimizer or an optimizer per agent key.
            critic_optimizer: critic optimizer,
            shared_weights: ...
            discount: discount to use for TD updates.
            dataset: dataset to learn from, whether fixed or from a replay buffer
                (see `acme.datasets.reverb.make_dataset` documentation).
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
        # Store agents.
        self._agents = agents
        self._agent_types = agent_types
        self._checkpoint = checkpoint

        # Store shared_weights.
        self._shared_weights = shared_weights

        # Store networks.
        self._observation_networks = observation_networks
        self._policy_networks = policy_networks
        self._critic_networks = critic_networks

        # Dictionary with network keys for each agent.
        self.agent_net_keys = {agent: agent for agent in self._agents}
        if self._shared_weights:
            self.agent_net_keys = {agent: agent.split("_")[0] for agent in self._agents}

        self.unique_net_keys = self._agent_types if shared_weights else self._agents

        # Create optimizers for different agent types.
        if not isinstance(policy_optimizer, dict):
            self._policy_optimizers: Dict[str, snt.Optimizer] = {}
            for agent in self.unique_net_keys:
                self._policy_optimizers[agent] = copy.deepcopy(policy_optimizer)
        else:
            self._policy_optimizers = policy_optimizer

        self._critic_optimizers: Dict[str, snt.Optimizer] = {}
        for agent in self.unique_net_keys:
            self._critic_optimizers[agent] = copy.deepcopy(critic_optimizer)

        # Expose the variables.
        policy_networks_to_expose = {}
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "critic": {},
            "policy": {},
        }
        for agent_key in self.unique_net_keys:
            policy_network_to_expose = snt.Sequential(
                [
                    self._observation_networks[agent_key],
                    self._policy_networks[agent_key],
                ]
            )
            policy_networks_to_expose[agent_key] = policy_network_to_expose
            self._system_network_variables["critic"][agent_key] = critic_networks[
                agent_key
            ].variables
            self._system_network_variables["policy"][
                agent_key
            ] = policy_network_to_expose.variables

        # Other trainer parameters.
        self._discount = discount
        self._entropy_cost = entropy_cost
        self._baseline_cost = baseline_cost
        self._lambda_gae = lambda_gae
        self._clipping_epsilon = clipping_epsilon

        # Dataset iterator
        self._iterator = dataset

        # Set up gradient clipping.
        if max_gradient_norm is not None:
            self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)
        else:  # A very large number. Infinity results in NaNs.
            self._max_gradient_norm = tf.convert_to_tensor(1e10)

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("trainer")

        # Create checkpointer
        self._system_checkpointer = {}
        if checkpoint:
            for agent_key in self.unique_net_keys:
                objects_to_save = {
                    "counter": self._counter,
                    "policy": self._policy_networks[agent_key],
                    "critic": self._critic_networks[agent_key],
                    "observation": self._observation_networks[agent_key],
                    "policy_optimizer": self._policy_optimizers,
                    "critic_optimizer": self._critic_optimizers,
                }

                subdir = os.path.join("trainer", agent_key)
                checkpointer = tf2_savers.Checkpointer(
                    time_delta_minutes=15,
                    directory=checkpoint_subpath,
                    objects_to_save=objects_to_save,
                    subdirectory=subdir,
                )
                self._system_checkpointer[agent_key] = checkpointer

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp: Optional[float] = None

    def _get_critic_feed(
        self,
        observations_trans: Dict[str, np.ndarray],
        agent: str,
    ) -> tf.Tensor:

        # Decentralised based
        observation_feed = observations_trans[agent]

        return observation_feed

    def _transform_observations(
        self, observation: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        observation_trans = {}
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]
            observation_trans[agent] = self._observation_networks[agent_key](
                observation[agent].observation
            )
        return observation_trans

    @tf.function
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        # Get data from replay.
        inputs = next(self._iterator)

        self._forward(inputs)

        self._backward()

        # Log losses per agent
        return train_utils.map_losses_per_agent_ac(
            self.critic_losses, self.policy_losses
        )

    # Forward pass that calculates loss.
    def _forward(self, inputs: Any) -> None:
        # Convert to sequence data
        data = tf2_utils.batch_to_sequence(inputs.data)

        # Unpack input data as follows:
        observations, actions, rewards, discounts, extras = (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.extras,
        )

        # transform observation using observation networks
        observations_trans = self._transform_observations(observations)

        # Get log_probs.
        log_probs = extras["log_probs"]

        # Store losses.
        policy_losses: Dict[str, Any] = {}
        critic_losses: Dict[str, Any] = {}

        with tf.GradientTape(persistent=True) as tape:
            for agent in self._agents:

                action, reward, discount, behaviour_log_prob = (
                    actions[agent],
                    rewards[agent],
                    discounts[agent],
                    log_probs[agent],
                )

                actor_observation = observations_trans[agent]
                critic_observation = self._get_critic_feed(observations_trans, agent)

                # Chop off final timestep for bootstrapping value
                reward = reward[:-1]
                discount = discount[:-1]

                # Get agent network
                agent_key = agent.split("_")[0] if self._shared_weights else agent
                policy_network = self._policy_networks[agent_key]
                critic_network = self._critic_networks[agent_key]

                # Reshape inputs.
                dims = actor_observation.shape[:2]
                actor_observation = snt.merge_leading_dims(
                    actor_observation, num_dims=2
                )
                critic_observation = snt.merge_leading_dims(
                    critic_observation, num_dims=2
                )
                policy = policy_network(actor_observation)
                values = critic_network(critic_observation)

                # Reshape the outputs.
                policy = tfd.BatchReshape(policy, batch_shape=dims, name="policy")
                values = tf.reshape(values, dims, name="value")

                # Values along the sequence T.
                bootstrap_value = values[-1]
                state_values = values[:-1]

                # Generalized Return Estimation
                td_loss, td_lambda_extra = trfl.td_lambda(
                    state_values=state_values,
                    rewards=reward,
                    pcontinues=discount,
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
                log_rhos = policy.log_prob(action) - behaviour_log_prob
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

                policy_gradient_loss = tf.reduce_mean(
                    -tf.minimum(
                        tf.multiply(importance_ratio, normalized_gae),
                        tf.multiply(clipped_importance_ratio, normalized_gae),
                    ),
                    name="PolicyGradientLoss",
                )

                # Entropy regularization. Only implemented for categorical dist.
                try:
                    policy_entropy = tf.reduce_mean(policy.entropy())
                except NotImplementedError:
                    policy_entropy = tf.convert_to_tensor(0.0)

                entropy_loss = -self._entropy_cost * policy_entropy

                # Combine weighted sum of actor & entropy regularization.
                policy_loss = policy_gradient_loss + entropy_loss

                policy_losses[agent] = policy_loss
                critic_losses[agent] = critic_loss

        self.policy_losses = policy_losses
        self.critic_losses = critic_losses
        self.tape = tape

    # Backward pass that calculates gradients and updates network.
    def _backward(self) -> None:
        # Calculate the gradients and update the networks
        policy_losses = self.policy_losses
        critic_losses = self.critic_losses
        tape = self.tape

        for agent in self._agents:
            # Get agent_key.
            agent_key = agent.split("_")[0] if self._shared_weights else agent

            # Get trainable variables.
            policy_variables = (
                self._observation_networks[agent_key].trainable_variables
                + self._policy_networks[agent_key].trainable_variables
            )
            critic_variables = self._critic_networks[agent_key].trainable_variables

            # Get gradients.
            policy_gradients = tape.gradient(policy_losses[agent], policy_variables)
            critic_gradients = tape.gradient(critic_losses[agent], critic_variables)

            # Optionally apply clipping.
            critic_grads, critic_norm = tf.clip_by_global_norm(
                critic_gradients, self._max_gradient_norm
            )
            policy_grads, policy_norm = tf.clip_by_global_norm(
                policy_gradients, self._max_gradient_norm
            )

            # Apply gradients.
            self._critic_optimizers[agent_key].apply(critic_grads, critic_variables)
            self._policy_optimizers[agent_key].apply(policy_grads, policy_variables)

        train_utils.safe_del(self, "tape")

    def step(self) -> None:

        # Run the learning step.
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint and attempt to write the logs.
        if self._checkpoint:
            train_utils.checkpoint_networks(self._system_checkpointer)

        if self._logger:
            self._logger.write(fetches)

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        variables: Dict[str, Dict[str, np.ndarray]] = {}
        for network_type in names:
            variables[network_type] = {
                agent: tf2_utils.to_numpy(
                    self._system_network_variables[network_type][agent]
                )
                for agent in self.unique_net_keys
            }
        return variables


class CentralisedMAPPOTrainer(MAPPOTrainer):
    def __init__(
        self,
        agents: List[Any],
        agent_types: List[str],
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        dataset: tf.data.Dataset,
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        shared_weights: bool,
        discount: float = 0.99,
        lambda_gae: float = 1.0,
        entropy_cost: float = 0.0,
        baseline_cost: float = 1.0,
        clipping_epsilon: float = 0.2,
        max_gradient_norm: Optional[float] = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = False,
        checkpoint_subpath: str = "Checkpoints",
    ):

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            observation_networks=observation_networks,
            dataset=dataset,
            shared_weights=shared_weights,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            discount=discount,
            lambda_gae=lambda_gae,
            entropy_cost=entropy_cost,
            baseline_cost=baseline_cost,
            clipping_epsilon=clipping_epsilon,
            max_gradient_norm=max_gradient_norm,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

    def _get_critic_feed(
        self,
        observations_trans: Dict[str, np.ndarray],
        agent: str,
    ) -> tf.Tensor:
        # Centralised based
        observation_feed = tf.stack([x for x in observations_trans.values()], 2)

        return observation_feed
