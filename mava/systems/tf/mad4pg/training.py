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
"""MAD4PG system trainer implementation."""

from typing import Any, Callable, Dict, List, Optional, Union

import reverb
import sonnet as snt
import tensorflow as tf
import tree
from acme.tf import losses
from acme.tf import utils as tf2_utils
from acme.utils import loggers

from mava import types as mava_types
from mava.adders.reverb.base import Trajectory
from mava.components.tf.losses.sequence import recurrent_n_step_critic_loss
from mava.systems.tf.maddpg.training import (
    MADDPGBaseRecurrentTrainer,
    MADDPGBaseTrainer,
    MADDPGCentralisedRecurrentTrainer,
    MADDPGCentralisedTrainer,
    MADDPGDecentralisedRecurrentTrainer,
    MADDPGDecentralisedTrainer,
    MADDPGStateBasedRecurrentTrainer,
    MADDPGStateBasedSingleActionCriticRecurrentTrainer,
    MADDPGStateBasedTrainer,
)
from mava.systems.tf.variable_utils import VariableClient
from mava.utils import training_utils as train_utils

train_utils.set_growing_gpu_memory()


class MAD4PGBaseTrainer(MADDPGBaseTrainer):
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
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):
        """Initialise MAD4PG trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            policy_networks: policy networks for each agent in
                the system.
            critic_networks: critic network(s), shared or for
                each agent in the system.
            target_policy_networks: target policy networks.
            target_critic_networks: target critic networks.
            policy_optimizer: optimizer(s) for updating policy networks.
            critic_optimizer: optimizer for updating critic networks.
            discount: discount factor for TD updates.
            target_averaging: whether to use polyak averaging for target network
                updates.
            target_update_period: number of steps before target networks are
                updated.
            target_update_rate: update rate when using averaging.
            dataset: training dataset.
            observation_networks: network for feature
                extraction from raw observation.
            target_observation_networks: target observation
                network.
            variable_client: The client used to manage the variables.
            counts: step counter object.
            agent_net_keys: specifies what network each agent uses.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            logger: logger object for logging trainer
                statistics.
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            variable_client=variable_client,
            counts=counts,
            agent_net_keys=agent_net_keys,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        )

    # Forward pass that calculates loss.
    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass

        Args:
            inputs (Any): input data from the data table (transitions)
        """

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
        trans = mava_types.Transition(*inputs.data)
        o_tm1, o_t, a_tm1, r_t, d_t, e_tm1, e_t = (
            trans.observations,
            trans.next_observations,
            trans.actions,
            trans.rewards,
            trans.discounts,
            trans.extras,
            trans.next_extras,
        )

        # Do forward passes through the networks and calculate the losses
        self.policy_losses = {}
        self.critic_losses = {}
        with tf.GradientTape(persistent=True) as tape:
            o_tm1_trans, o_t_trans = self._transform_observations(o_tm1, o_t)
            a_t = self._target_policy_actions(o_t_trans)

            for agent in self._agents:
                agent_key = self._agent_net_keys[agent]

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

                # Cast the additional discount to match the environment discount dtype.
                discount = tf.cast(self._discount, dtype=d_t[agent].dtype)

                # Critic loss.
                critic_loss = losses.categorical(
                    q_tm1, r_t[agent], discount * d_t[agent], q_t
                )
                self.critic_losses[agent] = tf.reduce_mean(critic_loss)
                # Actor learning.
                o_t_agent_feed = o_t_trans[agent]
                dpg_a_t = self._policy_networks[agent_key](o_t_agent_feed)

                # Get dpg actions
                dpg_a_t_feed = self._get_dpg_feed(a_t, dpg_a_t, agent)

                # Get dpg Q values.
                dpg_z_t = self._critic_networks[agent_key](o_t_feed, dpg_a_t_feed)
                dpg_q_t = dpg_z_t.mean()

                # Actor loss. If clipping is true use dqda clipping and clip the norm.
                dqda_clipping = 1.0 if self._max_gradient_norm is not None else None
                clip_norm = True if self._max_gradient_norm is not None else False

                policy_loss = losses.dpg(
                    q_max=dpg_q_t,
                    a_max=dpg_a_t,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=clip_norm,
                )
                self.policy_losses[agent] = tf.reduce_mean(policy_loss)
        self.tape = tape


class MAD4PGDecentralisedTrainer(MAD4PGBaseTrainer, MADDPGDecentralisedTrainer):
    """MAD4PG trainer for a decentralised architecture."""

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: snt.Optimizer,
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):
        """Initialise decentralised MAD4PG trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            policy_networks: policy networks for each agent in
                the system.
            critic_networks: critic network(s), shared or for
                each agent in the system.
            target_policy_networks: target policy networks.
            target_critic_networks: target critic networks.
            policy_optimizer: optimizer(s) for updating policy networks.
            critic_optimizer: optimizer for updating critic networks.
            discount: discount factor for TD updates.
            target_averaging: whether to use polyak averaging for target network
                updates.
            target_update_period: number of steps before target networks are
                updated.
            target_update_rate: update rate when using averaging.
            dataset: training dataset.
            observation_networks: network for feature
                extraction from raw observation.
            target_observation_networks: target observation
                network.
            variable_client: The client used to manage the variables.
            counts: step counter object.
            agent_net_keys: specifies what network each agent uses.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            logger: logger object for logging trainer
                statistics.
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            agent_net_keys=agent_net_keys,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            variable_client=variable_client,
            counts=counts,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        )


class MAD4PGCentralisedTrainer(MAD4PGBaseTrainer, MADDPGCentralisedTrainer):
    """MAD4PG trainer for a centralised architecture."""

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: snt.Optimizer,
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):
        """Initialise centralised MAD4PG trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            policy_networks: policy networks for each agent in
                the system.
            critic_networks: critic network(s), shared or for
                each agent in the system.
            target_policy_networks: target policy networks.
            target_critic_networks: target critic networks.
            policy_optimizer: optimizer(s) for updating policy networks.
            critic_optimizer: optimizer for updating critic networks.
            discount: discount factor for TD updates.
            target_averaging: whether to use polyak averaging for target network
                updates.
            target_update_period: number of steps before target networks are
                updated.
            target_update_rate: update rate when using averaging.
            dataset: training dataset.
            observation_networks: network for feature
                extraction from raw observation.
            target_observation_networks: target observation
                network.
            variable_client: The client used to manage the variables.
            counts: step counter object.
            agent_net_keys: specifies what network each agent uses.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            logger: logger object for logging trainer
                statistics.
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            agent_net_keys=agent_net_keys,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            variable_client=variable_client,
            counts=counts,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        )


class MAD4PGStateBasedTrainer(MAD4PGBaseTrainer, MADDPGStateBasedTrainer):
    """MAD4PG trainer for a state-based architecture."""

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: snt.Optimizer,
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):
        """Initialise state-based MAD4PG trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            policy_networks: policy networks for each agent in
                the system.
            critic_networks: critic network(s), shared or for
                each agent in the system.
            target_policy_networks: target policy networks.
            target_critic_networks: target critic networks.
            policy_optimizer: optimizer(s) for updating policy networks.
            critic_optimizer: optimizer for updating critic networks.
            discount: discount factor for TD updates.
            target_averaging: whether to use polyak averaging for target network
                updates.
            target_update_period: number of steps before target networks are
                updated.
            target_update_rate: update rate when using averaging.
            dataset: training dataset.
            observation_networks: network for feature
                extraction from raw observation.
            target_observation_networks: target observation
                network.
            variable_client: The client used to manage the variables.
            counts: step counter object.
            agent_net_keys: specifies what network each agent uses.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            logger: logger object for logging trainer
                statistics.
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            agent_net_keys=agent_net_keys,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            variable_client=variable_client,
            counts=counts,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        )


class MAD4PGBaseRecurrentTrainer(MADDPGBaseRecurrentTrainer):
    """Recurrent MAD4PG trainer.

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
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
        bootstrap_n: int = 10,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):
        """Initialise Recurrent MAD4PG trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            policy_networks: policy networks for each agent in
                the system.
            critic_networks: critic network(s), shared or for
                each agent in the system.
            target_policy_networks: target policy networks.
            target_critic_networks: target critic networks.
            policy_optimizer: optimizer(s) for updating policy networks.
            critic_optimizer: optimizer for updating critic networks.
            discount: discount factor for TD updates.
            target_averaging: whether to use polyak averaging for target network
                updates.
            target_update_period: number of steps before target networks are
                updated.
            target_update_rate: update rate when using averaging.
            dataset: training dataset.
            observation_networks: network for feature
                extraction from raw observation.
            target_observation_networks: target observation
                network.
            bootstrap_n: number of timestepsto use for bootstrapping.
            variable_client: The client used to manage the variables.
            counts: step counter object.
            agent_net_keys: specifies what network each agent uses.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            logger: logger object for logging trainer
                statistics.
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            agent_net_keys=agent_net_keys,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            variable_client=variable_client,
            counts=counts,
            bootstrap_n=bootstrap_n,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        )

    # Forward pass that calculates loss.
    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass

        Args:
            inputs: input data from the data table (transitions)
        """

        # TODO: Update this forward function to work like MAD4PG
        data: Trajectory = inputs.data

        # Note (dries): The unused variable is start_of_episodes.
        observations, actions, rewards, end_of_episode, _, extras = (
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

        self.policy_losses: Dict[str, tf.Tensor] = {}
        self.critic_losses: Dict[str, tf.Tensor] = {}

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            # Note (dries): We are assuming that only the policy network
            # is recurrent and not the observation network.
            obs_trans, target_obs_trans = self._transform_observations(observations)

            target_actions = self._target_policy_actions(
                target_obs_trans, target_core_state
            )

            for agent in self._agents:
                agent_key = self._agent_net_keys[agent]

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
                obs_comb, dims = train_utils.combine_dim(obs_trans_feed)
                act_comb, _ = train_utils.combine_dim(action_feed)
                q_values = self._critic_networks[agent_key](obs_comb, act_comb)
                q_values.set_dimensions(dims)

                # Remove first sequence step for the target
                obs_comb, _ = train_utils.combine_dim(target_obs_trans_feed)
                act_comb, _ = train_utils.combine_dim(target_actions_feed)
                target_q_values = self._target_critic_networks[agent_key](
                    obs_comb, act_comb
                )
                target_q_values.set_dimensions(dims)

                # Cast the additional discount to match
                # the environment discount dtype.
                agent_discount = end_of_episode[agent]
                discount = tf.cast(self._discount, dtype=agent_discount.dtype)
                agent_end_of_episode = end_of_episode[agent]
                ones_mask = tf.ones(shape=(agent_end_of_episode.shape[0], 1))
                step_not_padded = tf.concat(
                    [ones_mask, agent_end_of_episode[:, :-1]], axis=1
                )

                # Critic loss.
                critic_loss = recurrent_n_step_critic_loss(
                    q_values=q_values,
                    target_q_values=target_q_values,
                    rewards=rewards[agent],
                    discounts=discount * agent_discount,
                    bootstrap_n=self._bootstrap_n,
                    loss_fn=losses.categorical,
                )
                self.critic_losses[agent] = tf.reduce_mean(critic_loss)

                # Actor learning.
                obs_agent_feed = target_obs_trans[agent]
                # TODO (dries): Why is there an extra tuple?
                agent_core_state = core_state[agent][0]
                transposed_obs = tf2_utils.batch_to_sequence(obs_agent_feed)
                outputs, updated_states = snt.static_unroll(
                    self._policy_networks[agent_key],
                    transposed_obs,
                    agent_core_state,
                )

                dpg_actions = tf2_utils.batch_to_sequence(outputs)

                # Note (dries): This is done to so that losses.dpg can verify
                # using gradient.tape that there is a
                # gradient relationship between dpg_q_values and dpg_actions_comb.
                dpg_actions_comb, dim = train_utils.combine_dim(dpg_actions)

                # Note (dries): This seemingly useless line is important!
                # Don't remove it. See above note.
                dpg_actions = train_utils.extract_dim(dpg_actions_comb, dim)

                # Get dpg actions
                dpg_actions_feed = self._get_dpg_feed(
                    target_actions, dpg_actions, agent
                )

                # Get dpg Q values.
                obs_comb, _ = train_utils.combine_dim(target_obs_trans_feed)
                act_comb, _ = train_utils.combine_dim(dpg_actions_feed)
                dpg_z_values = self._critic_networks[agent_key](obs_comb, act_comb)
                dpg_q_values = dpg_z_values.mean()

                # Actor loss. If clipping is true use dqda clipping and clip the norm.
                dqda_clipping = 1.0 if self._max_gradient_norm is not None else None
                clip_norm = True if self._max_gradient_norm is not None else False

                policy_loss = losses.dpg(
                    dpg_q_values,
                    dpg_actions_comb,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=clip_norm,
                )
                policy_mask = tf.reshape(step_not_padded, policy_loss.shape)
                policy_loss = policy_loss * policy_mask
                self.policy_losses[agent] = tf.reduce_sum(policy_loss) / tf.reduce_sum(
                    policy_mask
                )
        self.tape = tape


class MAD4PGDecentralisedRecurrentTrainer(
    MAD4PGBaseRecurrentTrainer, MADDPGDecentralisedRecurrentTrainer
):
    """Recurrent MAD4PG trainer for a decentralised architecture."""

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
        bootstrap_n: int = 10,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):
        """Initialise Recurrent MAD4PG trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            policy_networks: policy networks for each agent in
                the system.
            critic_networks: critic network(s), shared or for
                each agent in the system.
            target_policy_networks: target policy networks.
            target_critic_networks: target critic networks.
            policy_optimizer: optimizer(s) for updating policy networks.
            critic_optimizer: optimizer for updating critic networks.
            discount: discount factor for TD updates.
            target_averaging: whether to use polyak averaging for target network
                updates.
            target_update_period: number of steps before target networks are
                updated.
            target_update_rate: update rate when using averaging.
            dataset: training dataset.
            observation_networks: network for feature
                extraction from raw observation.
            target_observation_networks: target observation
                network.
            bootstrap_n: number of timestepsto use for bootstrapping.
            variable_client: The client used to manage the variables.
            counts: step counter object.
            agent_net_keys: specifies what network each agent uses.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            logger: logger object for logging trainer
                statistics.
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            agent_net_keys=agent_net_keys,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            variable_client=variable_client,
            counts=counts,
            bootstrap_n=bootstrap_n,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        )


class MAD4PGCentralisedRecurrentTrainer(
    MAD4PGBaseRecurrentTrainer, MADDPGCentralisedRecurrentTrainer
):
    """Recurrent MAD4PG trainer for a centralised architecture."""

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
        bootstrap_n: int = 10,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):
        """Initialise Recurrent MAD4PG trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            policy_networks: policy networks for each agent in
                the system.
            critic_networks: critic network(s), shared or for
                each agent in the system.
            target_policy_networks: target policy networks.
            target_critic_networks: target critic networks.
            policy_optimizer: optimizer(s) for updating policy networks.
            critic_optimizer: optimizer for updating critic networks.
            discount: discount factor for TD updates.
            target_averaging: whether to use polyak averaging for target network
                updates.
            target_update_period: number of steps before target networks are
                updated.
            target_update_rate: update rate when using averaging.
            dataset: training dataset.
            observation_networks: network for feature
                extraction from raw observation.
            target_observation_networks: target observation
                network.
            bootstrap_n: number of timestepsto use for bootstrapping.
            variable_client: The client used to manage the variables.
            counts: step counter object.
            agent_net_keys: specifies what network each agent uses.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            logger: logger object for logging trainer
                statistics.
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            agent_net_keys=agent_net_keys,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            variable_client=variable_client,
            counts=counts,
            bootstrap_n=bootstrap_n,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        )


class MAD4PGStateBasedRecurrentTrainer(
    MAD4PGBaseRecurrentTrainer, MADDPGStateBasedRecurrentTrainer
):
    """Recurrent MAD4PG trainer for a state-based architecture."""

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
        bootstrap_n: int = 10,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):
        """Initialise Recurrent MAD4PG trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            policy_networks: policy networks for each agent in
                the system.
            critic_networks: critic network(s), shared or for
                each agent in the system.
            target_policy_networks: target policy networks.
            target_critic_networks: target critic networks.
            policy_optimizer: optimizer(s) for updating policy networks.
            critic_optimizer: optimizer for updating critic networks.
            discount: discount factor for TD updates.
            target_averaging: whether to use polyak averaging for target network
                updates.
            target_update_period: number of steps before target networks are
                updated.
            target_update_rate: update rate when using averaging.
            dataset: training dataset.
            observation_networks: network for feature
                extraction from raw observation.
            target_observation_networks: target observation
                network.
            bootstrap_n: number of timestepsto use for bootstrapping.
            variable_client: The client used to manage the variables.
            counts: step counter object.
            agent_net_keys: specifies what network each agent uses.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            logger: logger object for logging trainer
                statistics.
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            agent_net_keys=agent_net_keys,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            variable_client=variable_client,
            counts=counts,
            bootstrap_n=bootstrap_n,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        )


class MAD4PGStateBasedSingleActionCriticRecurrentTrainer(
    MAD4PGBaseRecurrentTrainer, MADDPGStateBasedSingleActionCriticRecurrentTrainer
):
    """Recurrent MAD4PG trainer for a state-based architecture."""

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
        bootstrap_n: int = 10,
    ):
        """Initialise State-Based Recurrent MAD4PG trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            policy_networks: policy networks for each agent in
                the system.
            critic_networks: critic network(s), shared or for
                each agent in the system.
            target_policy_networks: target policy networks.
            target_critic_networks: target critic networks.
            policy_optimizer: optimizer(s) for updating policy networks.
            critic_optimizer: optimizer for updating critic networks.
            discount: discount factor for TD updates.
            target_averaging: whether to use polyak averaging for target network
                updates.
            target_update_period: number of steps before target networks are
                updated.
            target_update_rate: update rate when using averaging.
            dataset: training dataset.
            observation_networks: network for feature
                extraction from raw observation.
            target_observation_networks: target observation
                network.
            bootstrap_n: number of timestepsto use for bootstrapping.
            variable_client: The client used to manage the variables.
            counts: step counter object.
            agent_net_keys: specifies what network each agent uses.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            logger: logger object for logging trainer
                statistics.
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            agent_net_keys=agent_net_keys,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            variable_client=variable_client,
            counts=counts,
            bootstrap_n=bootstrap_n,
        )
