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


"""MADDPG trainer implementation."""

import copy
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree
import trfl
from acme.tf import losses
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava
from mava.systems.tf import savers as tf2_savers
from mava.utils import training_utils as train_utils


class BaseMADDPGTrainer(mava.Trainer):
    """MADDPG trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        policy_optimizer: snt.Optimizer,
        critic_optimizer: snt.Optimizer,
        discount: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
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
        self._checkpoint = checkpoint

        # Store online and target networks.
        self._policy_networks = policy_networks
        self._critic_networks = critic_networks
        self._target_policy_networks = target_policy_networks
        self._target_critic_networks = target_critic_networks

        # Ensure obs and target networks are sonnet modules
        self._observation_networks = {
            k: tf2_utils.to_sonnet_module(v) for k, v in observation_networks.items()
        }
        self._target_observation_networks = {
            k: tf2_utils.to_sonnet_module(v)
            for k, v in target_observation_networks.items()
        }

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("trainer")

        # Other learner parameters.
        self._discount = discount
        self._clipping = clipping

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_update_period = target_update_period

        # Create an iterator to go through the dataset.
        # TODO(b/155086959): Fix type stubs and remove.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        self._critic_optimizer = critic_optimizer
        self._policy_optimizer = policy_optimizer

        # Dictionary with network keys for each agent.
        self.agent_net_keys = {agent: agent for agent in self._agents}
        if self._shared_weights:
            self.agent_net_keys = {agent: agent.split("_")[0] for agent in self._agents}

        self.unique_net_keys = self._agent_types if shared_weights else self._agents

        # Expose the variables.
        policy_networks_to_expose = {}
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "critic": {},
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
            self._system_network_variables["critic"][
                agent_key
            ] = target_critic_networks[agent_key].variables
            self._system_network_variables["policy"][
                agent_key
            ] = policy_network_to_expose.variables

        # Create checkpointer
        self._system_checkpointer = {}
        if checkpoint:
            for agent_key in self.unique_net_keys:
                objects_to_save = {
                    "counter": self._counter,
                    "policy": self._policy_networks[agent_key],
                    "critic": self._critic_networks[agent_key],
                    "observation": self._observation_networks[agent_key],
                    "target_policy": self._target_policy_networks[agent_key],
                    "target_critic": self._target_critic_networks[agent_key],
                    "target_observation": self._target_observation_networks[agent_key],
                    "policy_optimizer": self._policy_optimizer,
                    "critic_optimizer": self._critic_optimizer,
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
        self._timestamp = None

    def _update_target_networks(self) -> None:
        for key in self.unique_net_keys:
            # Update target network.
            online_variables = (
                *self._observation_networks[key].variables,
                *self._critic_networks[key].variables,
                *self._policy_networks[key].variables,
            )
            target_variables = (
                *self._target_observation_networks[key].variables,
                *self._target_critic_networks[key].variables,
                *self._target_policy_networks[key].variables,
            )

            # Make online -> target network update ops.
            if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(src)
            self._num_steps.assign_add(1)

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

    def _get_dpg_feed(
        self,
        a_t: Dict[str, np.ndarray],
        dpg_a_t: np.ndarray,
        agent: str,
    ) -> tf.Tensor:
        # Decentralised DPG
        dpg_a_t_feed = dpg_a_t
        return dpg_a_t_feed

    def _target_policy_actions(self, next_obs: Dict[str, np.ndarray]) -> Any:
        actions = {}
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]
            next_observation = next_obs[agent]
            actions[agent] = self._target_policy_networks[agent_key](next_observation)
        return actions

    # NOTE (Arnu): the decorator below was causing this _step() function not
    # to be called by the step() function below. Removing it makes the code
    # work. The docs on tf.function says it is useful for speed improvements
    # but as far as I can see, we can go ahead without it. At least for now.
    @tf.function
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        # Update the target networks
        self._update_target_networks()

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)

        self._forward(inputs)

        self._backward()

        # Log losses per agent
        return train_utils.map_losses_per_agent_ac(
            self.critic_losses, self.policy_losses
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
            critic_losses = {}

            o_tm1_trans, o_t_trans = self._transform_observations(o_tm1, o_t)
            a_t = self._target_policy_actions(o_t_trans)

            for agent in self._agents:
                agent_key = self.agent_net_keys[agent]

                # Cast the additional discount to match the environment discount dtype.
                discount = tf.cast(self._discount, dtype=d_t[agent].dtype)

                # Get critic feed
                o_tm1_feed, o_t_feed, a_tm1_feed, a_t_feed = self._get_critic_feed(
                    o_tm1_trans=o_tm1_trans,
                    o_t_trans=o_t_trans,
                    a_tm1=a_tm1,
                    a_t=a_t,
                    e_tm1=e_tm1,
                    e_t=e_t,
                    agent=agent,
                )

                # Critic learning.
                q_tm1 = self._critic_networks[agent_key](o_tm1_feed, a_tm1_feed)
                q_t = self._target_critic_networks[agent_key](o_t_feed, a_t_feed)

                # Squeeze into the shape expected by the td_learning implementation.
                q_tm1 = tf.squeeze(q_tm1, axis=-1)  # [B]
                q_t = tf.squeeze(q_t, axis=-1)  # [B]

                # Critic loss.
                critic_loss = trfl.td_learning(
                    q_tm1, r_t[agent], discount * d_t[agent], q_t
                ).loss

                # Actor learning.
                o_t_agent_feed = o_t_trans[agent]
                dpg_a_t = self._policy_networks[agent_key](o_t_agent_feed)

                # Get dpg actions
                dpg_a_t_feed = self._get_dpg_feed(a_t, dpg_a_t, agent)

                # Get dpg Q values.
                dpg_q_t = self._critic_networks[agent_key](o_t_feed, dpg_a_t_feed)

                # Actor loss. If clipping is true use dqda clipping and clip the norm.
                dqda_clipping = 1.0 if self._clipping else None

                policy_loss = losses.dpg(
                    dpg_q_t,
                    dpg_a_t,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=self._clipping,
                )
                policy_loss = tf.reduce_mean(policy_loss, axis=0)
                policy_losses[agent] = policy_loss

                critic_loss = tf.reduce_mean(critic_loss, axis=0)
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
            agent_key = self.agent_net_keys[agent]

            # Get trainable variables.
            policy_variables = (
                self._observation_networks[agent_key].trainable_variables
                + self._policy_networks[agent_key].trainable_variables
            )
            critic_variables = (
                # In this agent, the critic loss trains the observation network.
                self._observation_networks[agent_key].trainable_variables
                + self._critic_networks[agent_key].trainable_variables
            )

            # Compute gradients.
            # Note: Warning "WARNING:tensorflow:Calling GradientTape.gradient
            #  on a persistent tape inside its context is significantly less efficient
            #  than calling it outside the context." caused by losses.dpg, which calls
            #  tape.gradient.
            policy_gradients = tape.gradient(policy_losses[agent], policy_variables)
            critic_gradients = tape.gradient(critic_losses[agent], critic_variables)

            # Maybe clip gradients.
            if self._clipping:
                policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.0)[0]
                critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.0)[0]

            # Apply gradients.
            self._policy_optimizer.apply(policy_gradients, policy_variables)
            self._critic_optimizer.apply(critic_gradients, critic_variables)
        train_utils.safe_del(self, "tape")

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

        # Checkpoint and attempt to write the logs.
        if self._checkpoint:
            train_utils.checkpoint_networks(self._system_checkpointer)

        if self._logger:
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


class DecentralisedMADDPGTrainer(BaseMADDPGTrainer):
    """MADDPG trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        discount: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        policy_optimizer: snt.Optimizer,
        critic_optimizer: snt.Optimizer,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
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

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_update_period=target_update_period,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            shared_weights=shared_weights,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            clipping=clipping,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )


class CentralisedMADDPGTrainer(BaseMADDPGTrainer):
    """MADDPG trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        discount: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        policy_optimizer: snt.Optimizer = None,
        critic_optimizer: snt.Optimizer = None,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
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

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_update_period=target_update_period,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            shared_weights=shared_weights,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            clipping=clipping,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

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
        o_tm1_feed = tf.stack([x for x in o_tm1_trans.values()], 1)
        o_t_feed = tf.stack([x for x in o_t_trans.values()], 1)
        a_tm1_feed = tf.stack([x for x in a_tm1.values()], 1)
        a_t_feed = tf.stack([x for x in a_t.values()], 1)
        return o_tm1_feed, o_t_feed, a_tm1_feed, a_t_feed

    def _get_dpg_feed(
        self,
        a_t: Dict[str, np.ndarray],
        dpg_a_t: np.ndarray,
        agent: str,
    ) -> tf.Tensor:
        # Centralised and StateBased DPG
        # Note (dries): Copy has to be made because the input
        # variables cannot be changed.
        dpg_a_t_feed = copy.copy(a_t)
        dpg_a_t_feed[agent] = dpg_a_t
        tree.map_structure(tf.stop_gradient, dpg_a_t_feed)
        return dpg_a_t_feed


class StateBasedMADDPGTrainer(BaseMADDPGTrainer):
    """MADDPG trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        discount: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        policy_optimizer: snt.Optimizer = None,
        critic_optimizer: snt.Optimizer = None,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
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

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_update_period=target_update_period,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            shared_weights=shared_weights,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            clipping=clipping,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

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
        # State based
        o_tm1_feed = e_tm1["env_state"]
        o_t_feed = e_t["env_state"]
        a_tm1_feed = tf.stack([x for x in a_tm1.values()], 1)
        a_t_feed = tf.stack([x for x in a_t.values()], 1)
        return o_tm1_feed, o_t_feed, a_tm1_feed, a_t_feed

    def _get_dpg_feed(
        self,
        a_t: Dict[str, np.ndarray],
        dpg_a_t: np.ndarray,
        agent: str,
    ) -> tf.Tensor:
        # Centralised and StateBased DPG
        # Note (dries): Copy has to be made because the input
        # variables cannot be changed.
        dpg_a_t_feed = copy.copy(a_t)
        dpg_a_t_feed[agent] = dpg_a_t
        tree.map_structure(tf.stop_gradient, dpg_a_t_feed)

        return dpg_a_t_feed


class BaseRecurrentMADDPGTrainer(mava.Trainer):
    """MADDPG trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        discount: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        policy_optimizer: snt.Optimizer,
        critic_optimizer: snt.Optimizer,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
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
        self._checkpoint = checkpoint

        # Dictionary with network keys for each agent.
        self.agent_net_keys = {agent: agent for agent in self._agents}
        if self._shared_weights:
            self.agent_net_keys = {agent: agent.split("_")[0] for agent in self._agents}

        self.unique_net_keys = self._agent_types if shared_weights else self._agents

        # Store online and target networks.
        self._policy_networks = policy_networks
        self._critic_networks = critic_networks
        self._target_policy_networks = target_policy_networks
        self._target_critic_networks = target_critic_networks

        self._observation_networks = observation_networks
        self._target_observation_networks = target_observation_networks

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("trainer")

        # Other learner parameters.
        self._discount = discount
        self._clipping = clipping

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_update_period = target_update_period

        # Create an iterator to go through the dataset.
        # TODO(b/155086959): Fix type stubs and remove.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        self._critic_optimizer = critic_optimizer
        self._policy_optimizer = policy_optimizer

        # Expose the variables.
        policy_networks_to_expose = {}
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "critic": {},
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
            self._system_network_variables["critic"][
                agent_key
            ] = target_critic_networks[agent_key].variables
            self._system_network_variables["policy"][
                agent_key
            ] = policy_network_to_expose.variables

        # Create checkpointer
        self._system_checkpointer = {}
        if checkpoint:
            for agent_key in self.unique_net_keys:
                objects_to_save = {
                    "counter": self._counter,
                    "policy": self._policy_networks[agent_key],
                    "critic": self._critic_networks[agent_key],
                    "observation": self._observation_networks[agent_key],
                    "target_policy": self._target_policy_networks[agent_key],
                    "target_critic": self._target_critic_networks[agent_key],
                    "target_observation": self._target_observation_networks[agent_key],
                    "policy_optimizer": self._policy_optimizer,
                    "critic_optimizer": self._critic_optimizer,
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
        self._timestamp = None

    def _update_target_networks(self) -> None:
        for key in self.unique_net_keys:
            # Update target network.
            online_variables = (
                *self._observation_networks[key].variables,
                *self._critic_networks[key].variables,
                *self._policy_networks[key].variables,
            )
            target_variables = (
                *self._target_observation_networks[key].variables,
                *self._target_critic_networks[key].variables,
                *self._target_policy_networks[key].variables,
            )

            # Make online -> target network update ops.
            if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(src)
            self._num_steps.assign_add(1)

    def _combine_dim(self, tensor: tf.Tensor) -> tf.Tensor:
        dims = tensor.shape[:2]
        return snt.merge_leading_dims(tensor, num_dims=2), dims

    def _extract_dim(self, tensor: tf.Tensor, dims: tf.Tensor) -> tf.Tensor:
        return tf.reshape(tensor, [dims[0], dims[1], -1])

    def _transform_observations(
        self, observations: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        # Note (dries): We are assuming that only the policy network
        # is recurrent and not the observation network.
        obs_trans = {}
        obs_target_trans = {}
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]

            reshaped_obs, dims = self._combine_dim(observations[agent].observation)

            obs_trans[agent] = self._extract_dim(
                self._observation_networks[agent_key](reshaped_obs), dims
            )

            obs_target_trans[agent] = self._extract_dim(
                self._target_observation_networks[agent_key](reshaped_obs),
                dims,
            )

            # This stop_gradient prevents gradients to propagate into the target
            # observation network. In addition, since the online policy network is
            # evaluated at o_t, this also means the policy loss does not influence
            # the observation network training.
            obs_target_trans[agent] = tree.map_structure(
                tf.stop_gradient, obs_target_trans[agent]
            )
        return obs_trans, obs_target_trans

    def _get_critic_feed(
        self,
        obs_trans: Dict[str, np.ndarray],
        target_obs_trans: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        target_actions: Dict[str, np.ndarray],
        extras: Dict[str, np.ndarray],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        # Decentralised critic
        obs_trans_feed = obs_trans[agent]
        target_obs_trans_feed = target_obs_trans[agent]
        action_feed = actions[agent]
        target_actions_feed = target_actions[agent]
        return obs_trans_feed, target_obs_trans_feed, action_feed, target_actions_feed

    def _get_dpg_feed(
        self,
        target_actions: Dict[str, np.ndarray],
        dpg_actions: np.ndarray,
        agent: str,
    ) -> tf.Tensor:
        # Decentralised DPG
        dpg_actions_feed = dpg_actions
        return dpg_actions_feed

    def _target_policy_actions(
        self,
        target_obs_trans: Dict[str, np.ndarray],
        target_core_state: Dict[str, np.ndarray],
    ) -> Any:
        actions = {}

        for agent in self._agents:
            time.time()
            agent_key = self.agent_net_keys[agent]
            target_trans_obs = target_obs_trans[agent]
            # TODO (dries): Why is there an extra tuple
            #  wrapping that needs to be removed?
            agent_core_state = target_core_state[agent][0]

            transposed_obs = tf2_utils.batch_to_sequence(target_trans_obs)

            outputs, _ = snt.static_unroll(
                self._target_policy_networks[agent_key],
                transposed_obs,
                agent_core_state,
            )
            actions[agent] = tf2_utils.batch_to_sequence(outputs)
        return actions

    # NOTE (Arnu): the decorator below was causing this _step() function not
    # to be called by the step() function below. Removing it makes the code
    # work. The docs on tf.function says it is useful for speed improvements
    # but as far as I can see, we can go ahead without it. At least for now.
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
        return train_utils.map_losses_per_agent_ac(
            self.critic_losses, self.policy_losses
        )

    # Forward pass that calculates loss.
    def _forward(self, inputs: Any) -> None:
        data = inputs.data

        # Note (dries): The unused variable is start_of_episodes.
        observations, actions, rewards, discounts, _, extras = (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.start_of_episode,
            data.extras,
        )

        # Get initial state for the LSTM from replay and
        # extract the first state in the sequence..
        core_state = tree.map_structure(lambda s: s[:, 0, :], extras["core_states"])
        target_core_state = tree.map_structure(tf.identity, core_state)

        # TODO (dries): Take out all the data_points that does not need
        #  to be processed here at the start. Therefore it does not have
        #  to be done later on and saves processing time.

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            policy_losses: Dict[str, tf.Tensor] = {}
            critic_losses: Dict[str, tf.Tensor] = {}

            # Note (dries): We are assuming that only the policy network
            # is recurrent and not the observation network.

            obs_trans, target_obs_trans = self._transform_observations(observations)

            target_actions = self._target_policy_actions(
                target_obs_trans, target_core_state
            )

            for agent in self._agents:
                agent_key = self.agent_net_keys[agent]

                # Cast the additional discount to match
                # the environment discount dtype.
                discount = tf.cast(self._discount, dtype=discounts[agent].dtype)
                # Get critic feed
                (
                    obs_trans_feed,
                    target_obs_trans_feed,
                    action_feed,
                    target_actions_feed,
                ) = self._get_critic_feed(
                    obs_trans=obs_trans,
                    target_obs_trans=target_obs_trans,
                    actions=actions,
                    target_actions=target_actions,
                    extras=extras,
                    agent=agent,
                )

                # Critic learning.
                # Remove the last sequence step for the normal network
                obs_comb, _ = self._combine_dim(obs_trans_feed[:, :-1])
                act_comb, _ = self._combine_dim(action_feed[:, :-1])
                q_values = self._critic_networks[agent_key](obs_comb, act_comb)

                # Remove first sequence step for the target
                obs_comb, _ = self._combine_dim(target_obs_trans_feed[:, 1:])
                act_comb, _ = self._combine_dim(target_actions_feed[:, 1:])
                target_q_values = self._target_critic_networks[agent_key](
                    obs_comb, act_comb
                )

                # Squeeze into the shape expected by the td_learning implementation.
                q_values = tf.squeeze(q_values, axis=-1)  # [B]
                target_q_values = tf.squeeze(target_q_values, axis=-1)  # [B]

                # Critic loss.
                # Compute the transformed n-step loss.
                # TODO (dries): Is discounts and rewards correct?
                #  Or should it be [:, 1:]?

                agent_rewards, _ = self._combine_dim(rewards[agent][:, :-1])
                agent_discounts, _ = self._combine_dim(discounts[agent][:, :-1])

                # Critic loss.
                # TODO (dries): Change the critic losses to n step return losses?
                critic_loss = trfl.td_learning(
                    q_values,
                    agent_rewards,
                    discount * agent_discounts,
                    target_q_values,
                ).loss

                # Actor learning.
                obs_agent_feed = target_obs_trans[agent]
                # TODO (dries): Why is there an extra tuple?
                agent_core_state = core_state[agent][0]
                transposed_obs = tf2_utils.batch_to_sequence(obs_agent_feed)
                outputs, updated_states = snt.static_unroll(
                    self._policy_networks[agent_key], transposed_obs, agent_core_state
                )

                dpg_actions = tf2_utils.batch_to_sequence(outputs)

                # Get dpg actions
                dpg_actions_feed = self._get_dpg_feed(
                    target_actions, dpg_actions, agent
                )

                # Get dpg Q values.
                obs_comb, _ = self._combine_dim(target_obs_trans_feed)
                act_comb, _ = self._combine_dim(dpg_actions_feed)
                dpg_q_values = self._critic_networks[agent_key](obs_comb, act_comb)

                # Actor loss. If clipping is true use dqda clipping and clip the norm.
                # dpg_q_values = tf.squeeze(dpg_q_values, axis=-1)  # [B]

                dqda_clipping = 1.0 if self._clipping else None

                policy_loss = losses.dpg(
                    dpg_q_values,
                    act_comb,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=self._clipping,
                )

                policy_loss = tf.reduce_mean(policy_loss, axis=0)
                policy_losses[agent] = policy_loss

                critic_loss = tf.reduce_mean(critic_loss, axis=0)
                critic_losses[agent] = critic_loss

        self.policy_losses: Dict[str, tf.Tensor] = policy_losses
        self.critic_losses: Dict[str, tf.Tensor] = critic_losses
        self.tape = tape

    # Backward pass that calculates gradients and updates network.
    def _backward(self) -> None:
        # Calculate the gradients and update the networks
        policy_losses: Dict[str, tf.Tensor] = self.policy_losses
        critic_losses: Dict[str, tf.Tensor] = self.critic_losses
        tape = self.tape

        # Calculate the gradients and update the networks
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]

            # Get trainable variables.
            policy_variables = (
                self._observation_networks[agent_key].trainable_variables
                + self._policy_networks[agent_key].trainable_variables
            )
            critic_variables = (
                # In this agent, the critic loss trains the observation network.
                self._observation_networks[agent_key].trainable_variables
                + self._critic_networks[agent_key].trainable_variables
            )

            # Compute gradients.
            # Note: Warning "WARNING:tensorflow:Calling GradientTape.gradient
            #  on a persistent tape inside its context is significantly less efficient
            #  than calling it outside the context." caused by losses.dpg, which calls
            #  tape.gradient.
            policy_gradients = tape.gradient(policy_losses[agent], policy_variables)
            critic_gradients = tape.gradient(critic_losses[agent], critic_variables)

            # Maybe clip gradients.
            if self._clipping:
                policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.0)[0]
                critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.0)[0]

            # Apply gradients.
            self._policy_optimizer.apply(policy_gradients, policy_variables)
            self._critic_optimizer.apply(critic_gradients, critic_variables)

        train_utils.safe_del(self, "tape")

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
        if self._checkpoint:
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


class DecentralisedRecurrentMADDPGTrainer(BaseRecurrentMADDPGTrainer):
    """MADDPG trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        discount: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        policy_optimizer: snt.Optimizer,
        critic_optimizer: snt.Optimizer,
        shared_weights: bool = False,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
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

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_update_period=target_update_period,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            shared_weights=shared_weights,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            clipping=clipping,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )


class CentralisedRecurrentMADDPGTrainer(BaseRecurrentMADDPGTrainer):
    """MADDPG trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        discount: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        policy_optimizer: snt.Optimizer = None,
        critic_optimizer: snt.Optimizer = None,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
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
        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_update_period=target_update_period,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            shared_weights=shared_weights,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            clipping=clipping,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

    def _get_critic_feed(
        self,
        obs_trans: Dict[str, np.ndarray],
        target_obs_trans: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        target_actions: Dict[str, np.ndarray],
        extras: Dict[str, np.ndarray],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        # Centralised based
        obs_trans_feed = tf.stack([x for x in obs_trans.values()], -1)
        target_obs_trans_feed = tf.stack([x for x in target_obs_trans.values()], -1)
        action_feed = tf.stack([x for x in actions.values()], -1)
        target_actions_feed = tf.stack([x for x in target_actions.values()], -1)
        return obs_trans_feed, target_obs_trans_feed, action_feed, target_actions_feed

    def _get_dpg_feed(
        self,
        actions: Dict[str, np.ndarray],
        dpg_actions: np.ndarray,
        agent: str,
    ) -> tf.Tensor:
        # Centralised and StateBased DPG
        # Note (dries): Copy has to be made because the input
        # variables cannot be changed.
        dpg_actions_feed = copy.copy(actions)
        dpg_actions_feed[agent] = dpg_actions
        dpg_actions_feed = tf.squeeze(
            tf.stack([x for x in dpg_actions_feed.values()], -1)
        )
        tree.map_structure(tf.stop_gradient, dpg_actions_feed)
        return dpg_actions_feed


class StateBasedRecurrentMADDPGTrainer(BaseRecurrentMADDPGTrainer):
    """MADDPG trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        discount: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        policy_optimizer: snt.Optimizer = None,
        critic_optimizer: snt.Optimizer = None,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
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
        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_update_period=target_update_period,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            shared_weights=shared_weights,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            clipping=clipping,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

    def _get_critic_feed(
        self,
        obs_trans: Dict[str, np.ndarray],
        target_obs_trans: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        target_actions: Dict[str, np.ndarray],
        extras: Dict[str, np.ndarray],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        # State based
        obs_trans_feed = extras["env_state"]
        target_obs_trans_feed = extras["env_state"]
        action_feed = tf.stack([x for x in actions.values()], -1)
        target_actions_feed = tf.stack([x for x in target_actions.values()], -1)
        return obs_trans_feed, target_obs_trans_feed, action_feed, target_actions_feed

    def _get_dpg_feed(
        self,
        actions: Dict[str, np.ndarray],
        dpg_actions: np.ndarray,
        agent: str,
    ) -> tf.Tensor:
        # Centralised and StateBased DPG
        # Note (dries): Copy has to be made because the input
        # variables cannot be changed.
        dpg_actions_feed = copy.copy(actions)
        dpg_actions_feed[agent] = dpg_actions
        dpg_actions_feed = tf.squeeze(
            tf.stack([x for x in dpg_actions_feed.values()], -1)
        )
        tree.map_structure(tf.stop_gradient, dpg_actions_feed)
        return dpg_actions_feed
