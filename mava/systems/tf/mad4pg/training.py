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


"""MAD4PG trainer implementation."""

from typing import Any, Dict, List

import sonnet as snt
import tensorflow as tf
from acme.tf import losses
from acme.utils import counting, loggers

from mava.systems.tf.maddpg.training import (
    BaseMADDPGTrainer,
    CentralisedMADDPGTrainer,
    DecentralisedMADDPGTrainer,
    StateBasedMADDPGTrainer,
)


class BaseMAD4PGTrainer(BaseMADDPGTrainer):
    """MAD4PG trainer.
    This is the trainer component of a MAD4PG system. IE it takes a dataset as input
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

                # Critic loss.
                critic_loss = losses.categorical(
                    q_tm1, r_t[agent], discount * d_t[agent], q_t
                )
                # Actor learning.
                o_t_agent_feed = o_t_trans[agent]
                dpg_a_t = self._policy_networks[agent_key](o_t_agent_feed)

                # Get dpg actions
                dpg_a_t_feed = self._get_dpg_feed(a_t, dpg_a_t, agent)

                # Get dpg Q values.
                dpg_z_t = self._critic_networks[agent_key](o_t_feed, dpg_a_t_feed)
                dpg_q_t = dpg_z_t.mean()

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


class DecentralisedMAD4PGTrainer(BaseMAD4PGTrainer, DecentralisedMADDPGTrainer):
    """MAD4PG trainer.
    This is the trainer component of a MAD4PG system. IE it takes a dataset as input
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


class CentralisedMAD4PGTrainer(BaseMAD4PGTrainer, CentralisedMADDPGTrainer):
    """MAD4PG trainer.
    This is the trainer component of a MAD4PG system. IE it takes a dataset as input
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


class StateBasedMAD4PGTrainer(BaseMAD4PGTrainer, StateBasedMADDPGTrainer):
    """MAD4PG trainer.
    This is the trainer component of a MAD4PG system. IE it takes a dataset as input
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
