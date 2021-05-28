# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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


"""MASAC trainer implementation."""
import copy
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree
import trfl
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava
from mava.utils import training_utils as train_utils


class MASACBaseTrainer(mava.Trainer):
    """MASAC trainer.
    This is the trainer component of a MASAC system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_Q_1_networks: Dict[str, snt.Module],
        critic_Q_2_networks: Dict[str, snt.Module],
        critic_V_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_V_networks: Dict[str, snt.Module],
        discount: float,
        tau: float,
        temperature: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        policy_update_frequency: int,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        soft_target_update: bool = False,
        shared_weights: bool = False,
        policy_optimizer: snt.Optimizer = None,
        critic_V_optimizer: snt.Optimizer = None,
        critic_Q_1_optimizer: snt.Optimizer = None,
        critic_Q_2_optimizer: snt.Optimizer = None,
        max_gradient_norm: float = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "Checkpoints",
    ):
        """Initializes the learner.
        Args:
          policy_network: the online (optimized) policy.
          critic_V_network: the online critic for predicting state values.
          critic_Q_1_network: the online critic for predicting state-action values.
          critic_Q_2_network: the online critic for predicting state-action values.
          target_policy_network: the target policy (which lags behind the online
            policy).
          target_critic_V_network: the target critic for predicting state values.
          discount: discount to use for TD updates.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          soft_target_update: determines wether to use soft or hard target update
          tau: parameter for soft target update
          policy_update_frequncy: controls the frequency of updating the policy
          temperature: parameter for controlling SAC
          dataset: dataset to learn from, whether fixed or from a replay buffer
            (see `acme.datasets.reverb.make_dataset` documentation).
          observation_network: an optional online network to process observations
            before the policy and the critic.
          target_observation_network: the target observation network.
          policy_optimizer: the optimizers to be applied to the DPG (policy) loss.
          critic_V_optimizer: the optimizers to be applied to the critic_V loss.
          critic_Q_1_optimizer: the optimizers to be applied to the critic_Q_1 loss.
          critic_Q_2_optimizer: the optimizers to be applied to the critic_Q_2 loss.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        self._agents = agents
        self._agent_types = agent_types
        self._shared_weights = shared_weights
        self._checkpoint = checkpoint

        # Store online and target networks.
        self._policy_networks = policy_networks
        self._critic_V_networks = critic_V_networks
        self._critic_Q_1_networks = critic_Q_1_networks
        self._critic_Q_2_networks = critic_Q_2_networks
        self._target_policy_networks = target_policy_networks
        self._target_critic_V_networks = target_critic_V_networks

        # Ensure obs and target networks are sonnet modules
        self._observation_networks = {
            k: tf2_utils.to_sonnet_module(v) for k, v in observation_networks.items()
        }
        self._target_observation_networks = {
            k: tf2_utils.to_sonnet_module(v)
            for k, v in target_observation_networks.items()
        }

        # Temperature
        self._temperature = temperature

        # Policy update frequency
        self._policy_update_frequency = policy_update_frequency

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("trainer")

        # Other learner parameters.
        self._discount = discount

        # Set up gradient clipping.
        if max_gradient_norm is not None:
            self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)
        else:  # A very large number. Infinity results in NaNs.
            self._max_gradient_norm = tf.convert_to_tensor(1e10)

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_averaging = target_averaging
        self._target_update_period = target_update_period
        self._target_update_rate = target_update_rate
        self._soft_target_update = soft_target_update
        self._tau = tau

        # Create an iterator to go through the dataset.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        # Dictionary with network keys for each agent.
        self.agent_net_keys = {agent: agent for agent in self._agents}
        if self._shared_weights:
            self.agent_net_keys = {agent: agent.split("_")[0] for agent in self._agents}

        self.unique_net_keys = self._agent_types if shared_weights else self._agents

        # Create optimizers for different agent types.
        # TODO(Kale-ab): Allow this to be passed as a system param.
        self._policy_optimizers: snt.Optimizer = {}
        self._critic_V_optimizers: snt.Optimizer = {}
        self._critic_Q_1_optimizers: snt.Optimizer = {}
        self._critic_Q_2_optimizers: snt.Optimizer = {}
        for agent_key in self.unique_net_keys:
            self._policy_optimizers[agent_key] = copy.deepcopy(policy_optimizer)
            self._critic_V_optimizers[agent_key] = copy.deepcopy(critic_V_optimizer)
            self._critic_Q_1_optimizers[agent_key] = copy.deepcopy(critic_Q_1_optimizer)
            self._critic_Q_2_optimizers[agent_key] = copy.deepcopy(critic_Q_2_optimizer)

        # Expose the variables.
        policy_networks_to_expose = {}
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "critic_V": {},
            "critic_Q_1": {},
            "critic_Q_2": {},
            "policy": {},
        }
        for agent_key in self.unique_net_keys:
            policy_network_to_expose = snt.Sequential(
                [
                    self._target_observation_networks[agent_key],
                    self._target_policy_networks[agent_key],
                ]
            )
            policy_networks_to_expose[agent_key] = policy_network_to_expose
            # TODO (dries): Determine why acme has a critic
            #  in self._system_network_variables
            self._system_network_variables["critic_V"][
                agent_key
            ] = target_critic_V_networks[agent_key].variables
            self._system_network_variables["critic_Q_1"][
                agent_key
            ] = critic_Q_1_networks[agent_key].variables
            self._system_network_variables["critic_Q_2"][
                agent_key
            ] = critic_Q_2_networks[agent_key].variables
            self._system_network_variables["policy"][
                agent_key
            ] = policy_network_to_expose.variables

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
                    "critic_V": self._critic_V_networks[agent_key],
                    "critic_Q_1": self._critic_Q_1_networks[agent_key],
                    "critic_Q_2": self._critic_Q_2_networks[agent_key],
                    "observation": self._observation_networks[agent_key],
                    "target_policy": self._target_policy_networks[agent_key],
                    "target_critic_V": self._target_critic_V_networks[agent_key],
                    "target_observation": self._target_observation_networks[agent_key],
                    "policy_optimizer": self._policy_optimizers,
                    "critic_V_optimizer": self._critic_V_optimizers,
                    "critic_Q_1_optimizer": self._critic_Q_1_optimizers,
                    "critic__Q_2_optimizer": self._critic_Q_2_optimizers,
                    "num_steps": self._num_steps,
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

    @tf.function
    def _update_target_networks(self) -> None:
        for key in self.unique_net_keys:
            # Update target network.
            online_variables = (
                *self._observation_networks[key].variables,
                *self._critic_V_networks[key].variables,
                *self._policy_networks[key].variables,
            )
            target_variables = (
                *self._target_observation_networks[key].variables,
                *self._target_critic_V_networks[key].variables,
                *self._target_policy_networks[key].variables,
            )

            # Make online -> target network update ops.
            if self._soft_target_update:
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(
                        tf.math.add(
                            tf.math.multiply(self._tau, src),
                            tf.math.multiply(dest, 1.0 - self._tau),
                        )
                    )
            elif self._target_averaging:
                assert 0.0 < self._target_update_rate < 1.0
                tau = self._target_update_rate
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(dest * (1.0 - tau) + src * tau)
            else:
                if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                    for src, dest in zip(online_variables, target_variables):
                        dest.assign(src)
            self._num_steps.assign_add(1)

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

        # Centralised based
        o_tm1_feed = tf.stack([o_tm1_trans[agent] for agent in self._agents], 1)
        o_t_feed = tf.stack([o_t_trans[agent] for agent in self._agents], 1)
        a_tm1_feed = tf.stack([a_tm1[agent] for agent in self._agents], 1)
        a_t_feed = tf.stack([a_t[agent] for agent in self._agents], 1)

        return o_tm1_feed, o_t_feed, a_tm1_feed, a_t_feed

    def _target_policy_actions(self, next_obs: Dict[str, np.ndarray]) -> Any:
        actions = {}
        log_probs = {}
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]
            next_observation = next_obs[agent]
            actions[agent], log_probs[agent] = self._target_policy_networks[agent_key](
                next_observation
            )
            # print("agent action....", actions[agent])
        return actions, log_probs

    def _get_dpg_feed(
        self,
        a_t: Dict[str, np.ndarray],
        dpg_a_t: np.ndarray,
        log_probs: Dict[str, np.ndarray],
        dpg_log_probs: np.ndarray,
        agent: str,
    ) -> tf.Tensor:
        # Centralised and StateBased DPG
        # Note (dries): Copy has to be made because the input
        # variables cannot be changed.
        tree.map_structure(tf.stop_gradient, a_t)
        tree.map_structure(tf.stop_gradient, log_probs)
        dpg_a_t_feed = copy.copy(a_t)
        dpg_log_prob_feed = copy.copy(log_probs)
        dpg_a_t_feed[agent] = dpg_a_t
        dpg_log_prob_feed[agent] = dpg_log_probs

        dpg_a_t_feed = tf.squeeze(
            tf.stack([dpg_a_t_feed[agent] for agent in self._agents], 1)
        )

        dpg_log_prob_feed = tf.squeeze(
            tf.stack([dpg_log_prob_feed[agent] for agent in self._agents], 1)
        )

        return dpg_a_t_feed, dpg_log_prob_feed

    @tf.function
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        # Update the target networks
        self._update_target_networks()

        # Draw a batch of data from replay.
        sample: reverb.ReplaySample = next(self._iterator)

        self._forward(sample)

        self._backward()

        # Log losses per agent
        return train_utils.map_losses_per_agent_acq(
            self.policy_losses,
            self.critic_V_losses,
            self.critic_Q_1_losses,
            self.critic_Q_2_losses,
        )

    # Forward pass that calculates loss.
    def _forward(self, inputs: Any) -> None:
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

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            policy_losses = {}
            critic_V_losses = {}
            critic_Q_1_losses = {}
            critic_Q_2_losses = {}

            o_tm1_trans, o_t_trans = self._transform_observations(o_tm1, o_t)
            a_t, log_probs = self._target_policy_actions(o_t_trans)

            for agent in self._agents:
                agent_key = self.agent_net_keys[agent]

                # Cast the additional discount to match the environment discount dtype.
                discount = tf.cast(self._discount, dtype=d_t[agent].dtype)

                # Get critic feed
                (o_tm1_feed, o_t_feed, a_tm1_feed, a_t_feed,) = self._get_critic_feed(
                    o_tm1_trans=o_tm1_trans,
                    o_t_trans=o_t_trans,
                    a_tm1=a_tm1,
                    a_t=a_t,
                    e_tm1=e_tm1,
                    e_t=e_t,
                    agent=agent,
                )

                # Critic Q Learning
                q_1_pred = self._critic_Q_1_networks[agent_key](o_tm1_feed, a_tm1_feed)
                q_2_pred = self._critic_Q_2_networks[agent_key](o_tm1_feed, a_tm1_feed)
                v_target = self._target_critic_V_networks[agent_key](o_t_feed)

                # Squeeze into the shape expected by the td_learning implementation.
                q_1_pred = tf.squeeze(q_1_pred, axis=-1)  # [B]
                q_2_pred = tf.squeeze(q_2_pred, axis=-1)  # [B]
                v_target = tf.squeeze(v_target, axis=-1)

                q_1_loss = trfl.td_learning(
                    q_1_pred, r_t[agent], discount * d_t[agent], v_target
                ).loss
                q_2_loss = trfl.td_learning(
                    q_2_pred, r_t[agent], discount * d_t[agent], v_target
                ).loss
                critic_Q_1_losses[agent] = tf.reduce_mean(q_1_loss, axis=0)
                critic_Q_2_losses[agent] = tf.reduce_mean(q_2_loss, axis=0)

                # Actor and critic V Learning
                o_tm1_agent_feed = o_tm1_trans[agent]
                dpg_at, dpg_log_probs = self._policy_networks[agent_key](
                    o_tm1_agent_feed
                )

                dpg_a_t_feed, dpg_log_prob_feed = self._get_dpg_feed(
                    a_t, dpg_at, log_probs, dpg_log_probs, agent
                )

                v_pred = self._critic_V_networks[agent_key](o_tm1_feed)
                q_pred = tf.math.minimum(
                    self._critic_Q_1_networks[agent_key](o_tm1_feed, dpg_a_t_feed),
                    self._critic_Q_2_networks[agent_key](o_tm1_feed, dpg_a_t_feed),
                )
                v_targ = q_pred - tf.multiply(self._temperature, dpg_log_prob_feed)
                v_targ = tf.stop_gradient(v_targ)

                v_loss = tf.reduce_mean(tf.square(v_pred - v_targ), axis=0)
                critic_V_losses[agent] = v_loss[0]

                count = self._counter.get_counts().get("trainer_steps", 0)

                if count % self._policy_update_frequency == 0:
                    advantage = q_pred - v_pred
                    actor_loss = tf.reduce_mean(
                        tf.multiply(self._temperature, dpg_log_prob_feed) - advantage,
                        axis=0,
                    )[0]
                else:
                    actor_loss = tf.zeros((), tf.float32)

                policy_losses[agent] = actor_loss

        self.policy_losses = policy_losses
        self.critic_V_losses = critic_V_losses
        self.critic_Q_1_losses = critic_Q_1_losses
        self.critic_Q_2_losses = critic_Q_2_losses
        self.tape = tape

    # Backward pass that calculates gradients and updates network.
    def _backward(self) -> None:
        # Calculate the gradients and update the networks

        policy_losses = self.policy_losses
        critic_V_losses = self.critic_V_losses
        critic_Q_1_losses = self.critic_Q_1_losses
        critic_Q_2_losses = self.critic_Q_2_losses
        tape = self.tape

        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]

            # Get trainable variables.
            policy_variables = (
                self._observation_networks[agent_key].trainable_variables
                + self._policy_networks[agent_key].trainable_variables
            )
            critic_V_variables = self._critic_V_networks[agent_key].trainable_variables

            critic_Q_1_variables = self._critic_Q_1_networks[
                agent_key
            ].trainable_variables

            critic_Q_2_variables = self._critic_Q_2_networks[
                agent_key
            ].trainable_variables

            # Compute gradients.
            # Note: Warning "WARNING:tensorflow:Calling GradientTape.gradient
            #  on a persistent tape inside its context is significantly less efficient
            #  than calling it outside the context." caused by losses.dpg, which calls
            #  tape.gradient.
            policy_gradients = tape.gradient(policy_losses[agent], policy_variables)
            critic_V_gradients = tape.gradient(
                critic_V_losses[agent], critic_V_variables
            )
            critic_Q_1_gradients = tape.gradient(
                critic_Q_1_losses[agent], critic_Q_1_variables
            )
            critic_Q_2_gradients = tape.gradient(
                critic_Q_2_losses[agent], critic_Q_2_variables
            )

            # Maybe clip gradients.
            policy_gradients = tf.clip_by_global_norm(
                policy_gradients, self._max_gradient_norm
            )[0]
            critic_V_gradients = tf.clip_by_global_norm(
                critic_V_gradients, self._max_gradient_norm
            )[0]
            critic_Q_1_gradients = tf.clip_by_global_norm(
                critic_Q_1_gradients, self._max_gradient_norm
            )[0]
            critic_Q_2_gradients = tf.clip_by_global_norm(
                critic_Q_2_gradients, self._max_gradient_norm
            )[0]

            # Apply gradients.
            self._policy_optimizers[agent_key].apply(policy_gradients, policy_variables)
            self._critic_V_optimizers[agent_key].apply(
                critic_V_gradients, critic_V_variables
            )
            self._critic_Q_1_optimizers[agent_key].apply(
                critic_Q_1_gradients, critic_Q_1_variables
            )
            self._critic_Q_2_optimizers[agent_key].apply(
                critic_Q_2_gradients, critic_Q_2_variables
            )
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


class MASACCentralisedTrainer(MASACBaseTrainer):
    """MASAC trainer.
    This is the trainer component of a MASAC system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_Q_1_networks: Dict[str, snt.Module],
        critic_Q_2_networks: Dict[str, snt.Module],
        critic_V_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_V_networks: Dict[str, snt.Module],
        discount: float,
        tau: float,
        temperature: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        policy_update_frequency: int,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        soft_target_update: bool = False,
        shared_weights: bool = False,
        policy_optimizer: snt.Optimizer = None,
        critic_V_optimizer: snt.Optimizer = None,
        critic_Q_1_optimizer: snt.Optimizer = None,
        critic_Q_2_optimizer: snt.Optimizer = None,
        max_gradient_norm: float = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "Checkpoints",
    ):
        """Initializes the learner.
        Args:
          policy_network: the online (optimized) policy.
          critic_V_network: the online critic for predicting state values.
          critic_Q_1_network: the online critic for predicting state-action values.
          critic_Q_2_network: the online critic for predicting state-action values.
          target_policy_network: the target policy (which lags behind the online
            policy).
          target_critic_V_network: the target critic for predicting state values.
          discount: discount to use for TD updates.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          soft_target_update: determines wether to use soft or hard target update
          tau: parameter for soft target update
          policy_update_frequncy: controls the frequency of updating the policy
          temperature: parameter for controlling SAC
          dataset: dataset to learn from, whether fixed or from a replay buffer
            (see `acme.datasets.reverb.make_dataset` documentation).
          observation_network: an optional online network to process observations
            before the policy and the critic.
          target_observation_network: the target observation network.
          policy_optimizer: the optimizers to be applied to the DPG (policy) loss.
          critic_V_optimizer: the optimizers to be applied to the critic_V loss.
          critic_Q_1_optimizer: the optimizers to be applied to the critic_Q_1 loss.
          critic_Q_2_optimizer: the optimizers to be applied to the critic_Q_2 loss.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_Q_1_networks=critic_Q_1_networks,
            critic_Q_2_networks=critic_Q_2_networks,
            critic_V_networks=critic_V_networks,
            target_policy_networks=target_policy_networks,
            target_critic_V_networks=target_critic_V_networks,
            discount=discount,
            tau=tau,
            temperature=temperature,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            policy_update_frequency=policy_update_frequency,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            soft_target_update=soft_target_update,
            shared_weights=shared_weights,
            policy_optimizer=policy_optimizer,
            critic_V_optimizer=critic_V_optimizer,
            critic_Q_1_optimizer=critic_Q_1_optimizer,
            critic_Q_2_optimizer=critic_Q_2_optimizer,
            max_gradient_norm=max_gradient_norm,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )
