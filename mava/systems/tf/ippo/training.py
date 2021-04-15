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


class IPPOTrainer(mava.Trainer):
    """IPPO trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        # target_policy_networks: Dict[str, snt.Module],
        # target_critic_networks: Dict[str, snt.Module],
        discount: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        # target_observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        policy_optimizer: snt.Optimizer = None,
        critic_optimizer: snt.Optimizer = None,
        clipping: bool = True,
        lambda_gae: float = 0.95,
        clipping_epsilon: float = 0.2,
        entropy_cost: float = 0.01,
        baseline_cost: float = 0.5,
        num_steps: int = 5,
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

        # Store online and target networks.
        self._policy_networks = policy_networks
        self._critic_networks = critic_networks
        # self._target_policy_networks = target_policy_networks
        # self._target_critic_networks = target_critic_networks

        self._observation_networks = observation_networks
        # self._target_observation_networks = target_observation_networks

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("trainer")

        # Other learner parameters.
        self._discount = discount
        self._clipping = clipping
        self._baseline_cost = baseline_cost
        self._entropy_cost = entropy_cost
        self._clipping_epsilon = clipping_epsilon
        self._lambda_gae = lambda_gae
        # self._num_steps = num_steps
        self._num_steps = tf.Variable(5, dtype=tf.int32)

        # Necessary to track when to update target networks.
        # self._num_steps = tf.Variable(0, dtype=tf.int32)
        # self._target_update_period = target_update_period

        # Create an iterator to go through the dataset.
        # TODO(b/155086959): Fix type stubs and remove.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

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
                agent_key
            ].variables
            self._system_network_variables["policy"][agent_key] = self._policy_networks[
                agent_key
            ].variables

        # Create checkpointer
        self._system_checkpointer = {}
        if checkpoint:
            # TODO (dries): Address this new warning: WARNING:tensorflow:11 out
            #  of the last 11 calls to
            #  <function MultiDeviceSaver.save.<locals>.tf_function_save at
            #  0x7eff3c13dd30> triggered tf.function retracing. Tracing is
            #  expensive and the excessive number tracings could be due to (1)
            #  creating @tf.function repeatedly in a loop, (2) passing tensors
            #  with different shapes, (3) passing Python objects instead of tensors.
            for agent_key in self.unique_net_keys:
                objects_to_save = {
                    "counter": self._counter,
                    "policy": self._policy_networks[agent_key],
                    "critic": self._critic_networks[agent_key],
                    "observation": self._observation_networks[agent_key],
                    # "target_policy": self._target_policy_networks[agent_key],
                    # "target_critic": self._target_critic_networks[agent_key],
                    # "target_observation":
                    # self._target_observation_networks[agent_key],
                    "policy_optimizer": self._policy_optimizer,
                    "critic_optimizer": self._critic_optimizer,
                    "num_steps": self._num_steps,
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
            o_t[agent] = self._target_observation_networks[agent_key](
                next_obs[agent].observation
            )
            # This stop_gradient prevents gradients to propagate into the target
            # observation network. In addition, since the online policy network is
            # evaluated at o_t, this also means the policy loss does not influence
            # the observation network training.
            o_t[agent] = tree.map_structure(tf.stop_gradient, o_t[agent])

            # TODO (dries): Why is there a stop gradient here? The target
            #  will not be updated unless included into the
            #  policy_variables or critic_variables sets.
            #  One reason might be that it helps with preventing the observation
            #  network from being updated from the policy_loss.
            #  But why would we want that? Don't we want both the critic
            #  and policy to update the observation network?
            #  Or is it bad to have two optimisation processes optimising
            #  the same set of weights? But the
            #  StateBasedActorCritic will then not work as the critic
            #  is not dependent on the behavior networks.
        return o_tm1, o_t

    @tf.function
    def _get_critic_feed(
        self,
        o_tm1_trans: Dict[str, np.ndarray],
        o_t_trans: Dict[str, np.ndarray],
        a_tm1: Dict[str, np.ndarray],
        a_t: Dict[str, np.ndarray],
        e_tm1: Dict[str, np.ndarray],
        e_t: Dict[str, np.array],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        # Decentralised critic
        o_tm1_feed = o_tm1_trans[agent]
        o_t_feed = o_t_trans[agent]
        a_tm1_feed = a_tm1[agent]
        a_t_feed = a_t[agent]
        return o_tm1_feed, o_t_feed, a_tm1_feed, a_t_feed

    # TODO actor feed

    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        # # Update the target networks
        # self._update_target_networks()

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
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
        prev_logits = e_t["logits"]

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

                critic_loss, td_lambda_extra = trfl.td_lambda(
                    state_values=values,
                    rewards=r_t[agent],
                    pcontinues=discount,
                    bootstrap_value=bootstrap_value,
                    lambda_=self._lambda_gae,
                    name="CriticLoss",
                )

                critic_loss *= self.baseline_cost

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
                policy_gradient_loss = -tf.minimum(
                    tf.multiply(importance_ratio, gae),
                    tf.multiply(clipped_importance_ratio, gae),
                )

                # Entropy regulariser.
                entropy_loss = (
                    self._entropy_cost * trfl.policy_entropy_loss(pi_target).loss
                )

                # Combine weighted sum of actor & critic losses.
                policy_loss = tf.reduce_mean(policy_gradient_loss + entropy_loss)

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
