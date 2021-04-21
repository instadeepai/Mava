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

"""IPPO trainer implementation."""
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import trfl
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava


class BaseIPPOTrainer(mava.Trainer):
    """IPPO trainer.
    This is the trainer component of a IPPO system. IE it takes a dataset as input
    and implements update functionality for each agent to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        discount: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        policy_optimizer: snt.Optimizer = None,
        critic_optimizer: snt.Optimizer = None,
        clipping: bool = True,
        lambda_gae: float = 0.95,
        clipping_epsilon: float = 0.2,
        entropy_cost: float = 0.01,
        baseline_cost: float = 0.5,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "Checkpoints",
    ):
        # NOTE We still need to update this stub.
        """Initializes the learner.
        Args:
        policy_network: the online (optimized) policy.
        critic_network: the online critic.
        target_policy_network: the target policy (which lags behind the online
            policy).
        target_critic_network: the target critic.
        discount: discount to use for TD updates.
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

        # Store online and target networks.
        self._policy_networks = policy_networks
        self._critic_networks = critic_networks
        self._observation_networks = observation_networks

        # General trainer book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("trainer")

        # Other trainer parameters.
        self._discount = discount
        self._clipping = clipping
        self._baseline_cost = baseline_cost
        self._entropy_cost = entropy_cost
        self._clipping_epsilon = clipping_epsilon
        self._lambda_gae = lambda_gae
        self._num_steps = tf.Variable(0, dtype=tf.int32)

        # NOTE Do not wrap the dataset in iter() because it already has been.
        # See make_dataset_iterator() in SystemBuilder.
        self._iterator = dataset

        # Create optimizers if they aren't given.
        self._critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-3)
        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-3)

        # Dictionary with network keys for each agent.
        self.agent_net_keys = {agent: agent for agent in self._agents}
        if self._shared_weights:
            self.agent_net_keys = {agent: agent.split("_")[0] for agent in self._agents}
        self.unique_net_keys = self._agent_types if shared_weights else self._agents

        # Expose the variables.
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "critic": {},
            "policy": {},
        }
        for agent_key in self.unique_net_keys:
            self._system_network_variables["critic"][agent_key] = self._critic_networks[
                agent_key].variables
            self._system_network_variables["policy"][agent_key] = self._policy_networks[
                agent_key].variables

        # Create checkpointer
        # NOTE (Siphelele+Claude) we are not sure how checkpointing works.
        self._system_checkpointer = {}
        if checkpoint:
            for agent_key in self.unique_net_keys:
                objects_to_save = {
                    "counter": self._counter,
                    "policy": self._policy_networks[agent_key],
                    "critic": self._critic_networks[agent_key],
                    "observation": self._observation_networks[agent_key],
                    "policy_optimizer": self._policy_optimizer,
                    "critic_optimizer": self._critic_optimizer,
                }

                checkpointer_dir = os.path.join(checkpoint_subpath, agent_key)
                checkpointer = tf2_savers.Checkpointer(
                    objects_to_save=objects_to_save,
                    time_delta_minutes=1,
                    directory=checkpointer_dir,
                    enable_checkpointing=True,
                )
                self._system_checkpointer[agent_key] = checkpointer

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    @tf.function
    def _transform_observations(
        self, obs: Dict[str, np.ndarray], next_obs: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        o_tm1 = {}
        o_t = {}
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]
            o_tm1[agent] = self._observation_networks[agent_key](obs[agent].observation)
            o_t[agent] = self._observation_networks[agent_key](
                next_obs[agent].observation
            )
        return o_tm1, o_t

    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        # Get data from replay.
        inputs = next(self._iterator)

        # Unpack input data as follows:
        # o_tm1 = dictionary of observations one for each agent
        # a_tm1 = dictionary of actions taken from obs in o_tm1
        # e_tm1 [Optional] = extra data for timestep t-1
        # that the agents persist in replay.
        # r_t = dictionary of rewards or rewards sequences
        #   (if using N step transitions) ensuing from actions a_tm1
        # d_t = environment discount ensuing from actions a_tm1.
        #   This discount is applied to future rewards after r_t.
        # o_t = dictionary of next observations or next observation sequences
        # e_t [Optional] = extra data for timestep t that the agents persist in replay.
        o_tm1, a_tm1, e_tm1, r_t, d_t, o_t, e_t = inputs.data
        o_tm1_trans, o_t_trans = self._transform_observations(o_tm1, o_t)
        prev_logits = e_t['logits']
        logged_losses: Dict[str, Dict[str, Any]] = {}

        for agent in self._agents:

            # Cast the additional discount to match the environment discount dtype.
            discount = tf.cast(self._discount, dtype=d_t[agent].dtype)

            with tf.GradientTape(persistent=True) as tape:
                # Maybe transform the observation before feeding into policy and critic.
                # Transforming the observations this way at the start of the learning
                # step effectively means that the policy and critic share observation
                # network weights.

                logits = self._policy_networks[agent](o_tm1_trans[agent])
                values = self._critic_networks[agent](o_tm1_trans[agent])

                # critic loss
                bootstrap_value = values[-1]
                values = values[:-1]

                td_loss, td_lambda_extra = trfl.td_lambda(
                    state_values=values,
                    rewards=r_t[agent],
                    pcontinues=discount,
                    bootstrap_value=bootstrap_value,
                    lambda_=self._lambda_gae,
                    name="CriticLoss",
                )

                # Do not use the loss provided by td_lambda as they sum the losses over
                # the sequence length rather than averaging them.
                critic_loss = self.baseline_cost * tf.reduce_mean(
                    tf.square(td_lambda_extra.temporal_differences), name="CriticLoss"
                )

                # Compute importance weights
                behaviour_logits = prev_logits[agent]
                pi_behaviour = tfp.distributions.Categorical(
                    logits=behaviour_logits[:-1]
                )
                pi_target = tfp.distributions.Categorical(logits=logits[:-1])
                log_rhos = pi_target.log_prob(a_tm1[agent]) - pi_behaviour.log_prob(
                    a_tm1[agent]
                )
                importance_ratio = tf.exp(log_rhos)
                clipped_importance_ratio = tf.clip_by_value(
                    importance_ratio,
                    1.0 - self._clipping_epsilon,
                    1.0 + self._clipping_epsilon,
                )

                gae = tf.stop_gradient(td_lambda_extra.discounted_returns - values[0])
                mean, variance = tf.nn.moments(gae, axes=[0, 1], keepdims=True)
                normalized_gae = (gae - mean) / tf.sqrt(variance)

                policy_gradient_loss = -tf.minimum(
                    tf.multiply(importance_ratio, normalized_gae),
                    tf.multiply(clipped_importance_ratio, normalized_gae),
                )

                # Entropy regulariser.
                # scale = 1.0 / tf.math.log(
                #     tf.convert_to_tensor(logits.shape[-1], dtype=float)
                # )

                entropy = trfl.policy_entropy_loss(pi_target)
                entropy_loss = self._entropy_cost * entropy.loss
                # policy_entropy = scale * tf.reduce_mean(entropy.extra.entropy)

                # Combine weighted sum of actor & critic losses.
                policy_loss = tf.reduce_mean(
                    policy_gradient_loss + entropy_loss, name="PolicyLoss"
                )

            policy_variables = self._policy_networks[agent].trainable_variables
            critic_variables = self._critic_networks[agent].trainable_variables

            policy_gradients = tape.gradient(policy_loss, policy_variables)
            critic_gradients = tape.gradient(critic_loss, critic_variables)

            # Maybe clip gradients.
            if self._clipping:
                policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.0)[0]
                critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.0)[0]

            # Apply gradients.
            self._policy_optimizer.apply(policy_gradients, policy_variables)
            self._critic_optimizer.apply(critic_gradients, critic_variables)

            logged_losses.update(
                {
                    f"{agent}_critic_loss": critic_loss,
                    f"{agent}_policy_loss": policy_loss,
                }
            )

            del tape

        return logged_losses

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
        for network_type in names:
            variables[network_type] = {}
            for agent in self.unique_net_keys:
                variables[network_type][agent] = tf2_utils.to_numpy(
                    self._system_network_variables[network_type][agent]
                )
        return variables


class IPPOTrainer(BaseIPPOTrainer):
    """IPPO trainer.
    This is the trainer component of a IPPO system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        discount: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        lambda_gae: float,
        clipping_epsilon: float,
        entropy_cost: float,
        baseline_cost: float,
        shared_weights: bool = False,
        policy_optimizer: snt.Optimizer = None,
        critic_optimizer: snt.Optimizer = None,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
    ):
        """Initializes the learner.
        Args:
          policy_network: the online (optimized) policy.
          critic_network: the online critic.
          target_policy_network: the target policy (which lags behind the online
            policy).
          target_critic_network: the target critic.
          discount: discount to use for TD updates.
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

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            discount=discount,
            dataset=dataset,
            observation_networks=observation_networks,
            shared_weights=shared_weights,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            clipping=clipping,
            lambda_gae=lambda_gae,
            clipping_epsilon=clipping_epsilon,
            entropy_cost=entropy_cost,
            baseline_cost=baseline_cost,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
        )
