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
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree
import trfl
from acme.tf import losses
from acme.tf import utils as tf2_utils
from acme.utils import loggers

import mava
from mava import types as mava_types
from mava.adders.reverb.base import Trajectory
from mava.components.tf.losses.sequence import recurrent_n_step_critic_loss
from mava.systems.tf.variable_utils import VariableClient
from mava.utils import training_utils as train_utils
from mava.utils.sort_utils import sort_str_num

train_utils.set_growing_gpu_memory()


class MADDPGBaseTrainer(mava.Trainer):
    """MADDPG trainer.

    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
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
        """Initialise MADDPG trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            policy_networks: policy networks for each agent in
                the system.
            critic_networks: critic network(s), shared or for
                each agent in the system.
            target_policy_networks: target policy networks.
            target_critic_networks: target critic networks.
            policy_optimizer:
                optimizer(s) for updating policy networks.
            critic_optimizer:
                optimizer for updating critic networks.
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

        self._agents = agents
        self._agent_net_keys = agent_net_keys
        self._variable_client = variable_client
        self._learning_rate_scheduler_fn = learning_rate_scheduler_fn

        # Setup counts
        self._counts = counts

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

        # Create an iterator to go through the dataset.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        # Dictionary with unique network keys.
        self.unique_net_keys = sort_str_num(self._policy_networks.keys())

        # Get the agents which shoud be updated and ran
        self._trainer_agent_list = self._agents

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
            "critics": {},
            "policies": {},
        }
        for agent_key in self.unique_net_keys:
            policy_network_to_expose = snt.Sequential(
                [
                    self._target_observation_networks[agent_key],
                    self._target_policy_networks[agent_key],
                ]
            )
            policy_networks_to_expose[agent_key] = policy_network_to_expose
            self._system_network_variables["critics"][
                agent_key
            ] = target_critic_networks[agent_key].variables
            self._system_network_variables["policies"][
                agent_key
            ] = policy_network_to_expose.variables

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp: Optional[float] = None

    def _update_target_networks(self) -> None:
        """Update the target networks using either target averaging or

        by directy copying the weights of the online networks every few steps."""
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

            if self._target_averaging:
                assert 0.0 < self._target_update_rate < 1.0
                tau = self._target_update_rate
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(dest * (1.0 - tau) + src * tau)
            else:
                # Make online -> target network update ops.
                if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                    for src, dest in zip(online_variables, target_variables):
                        dest.assign(src)
        self._num_steps.assign_add(1)

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Depreciated method."""
        pass

    def _transform_observations(
        self, obs: Dict[str, mava_types.OLT], next_obs: Dict[str, mava_types.OLT]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Transform the observatations using the observation networks of each agent."

        Args:
            obs: observations at timestep t-1
            next_obs: observations at timestep t

        Returns:
            Transformed observatations
        """
        o_tm1 = {}
        o_t = {}
        for agent in self._agents:
            agent_key = self._agent_net_keys[agent]
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
        e_t: Dict[str, np.ndarray],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get data to feed to the agent critic network(s)

        Args:
            o_tm1_trans: transformed (e.g. using observation
                network) observation at timestep t-1
            o_t_trans: transformed observation at timestep t
            a_tm1: action at timestep t-1
            a_t: action at timestep t
            e_tm1: extras at timestep t-1
            e_t: extras at timestep t
            agent: agent id

        Returns:
            agent critic network feeds
        """

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
        """Get data to feed to the agent networks

        Args:
            a_t: action at timestep t
            dpg_a_t: predicted action at timestep t
            agent: agent id

        Returns:
            agent policy network feed
        """
        # Decentralised DPG
        dpg_a_t_feed = dpg_a_t
        return dpg_a_t_feed

    def _target_policy_actions(self, next_obs: Dict[str, np.ndarray]) -> Any:
        """Select actions using target policy networks

        Args:
            next_obs: next agent observations.

        Returns:
            agent target actions
        """
        actions = {}
        for agent in self._agents:
            agent_key = self._agent_net_keys[agent]
            next_observation = next_obs[agent]
            actions[agent] = self._target_policy_networks[agent_key](next_observation)
        return actions

    @tf.function
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Trainer forward and backward passes.

        Returns:
            losses
        """

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
    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass

        Args:
            inputs: input data from the data table (transitions)
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

        self.policy_losses = {}
        self.critic_losses = {}
        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:

            o_tm1_trans, o_t_trans = self._transform_observations(o_tm1, o_t)
            a_t = self._target_policy_actions(o_t_trans)
            for agent in self._trainer_agent_list:
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

                # Squeeze into the shape expected by the td_learning implementation.
                q_tm1 = tf.squeeze(q_tm1, axis=-1)  # [B]
                q_t = tf.squeeze(q_t, axis=-1)  # [B]

                # Cast the additional discount to match the environment discount dtype.
                discount = tf.cast(self._discount, dtype=d_t[agent].dtype)

                # Critic loss.
                critic_loss = trfl.td_learning(
                    q_tm1, r_t[agent], discount * d_t[agent], q_t
                ).loss
                self.critic_losses[agent] = tf.reduce_mean(critic_loss)

                # Actor learning.
                o_t_agent_feed = o_t_trans[agent]
                dpg_a_t = self._policy_networks[agent_key](o_t_agent_feed)

                # Get dpg actions
                dpg_a_t_feed = self._get_dpg_feed(a_t, dpg_a_t, agent)

                # Get dpg Q values.
                dpg_q_t = self._critic_networks[agent_key](o_t_feed, dpg_a_t_feed)

                # Actor loss. If clipping is true use dqda clipping and clip the norm.
                dqda_clipping = 1.0 if self._max_gradient_norm is not None else None
                clip_norm = self._max_gradient_norm is not None

                policy_loss = losses.dpg(
                    dpg_q_t,
                    dpg_a_t,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=clip_norm,
                )

                self.policy_losses[agent] = tf.reduce_mean(policy_loss)
        self.tape = tape

    # Backward pass that calculates gradients and updates network.
    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

        # Calculate the gradients and update the networks
        policy_losses = self.policy_losses
        critic_losses = self.critic_losses
        tape = self.tape
        for agent in self._trainer_agent_list:
            agent_key = self._agent_net_keys[agent]

            # Get trainable variables.
            # Note: policy does not update the shared obs network
            policy_variables = self._policy_networks[agent_key].trainable_variables
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
            policy_gradients = tf.clip_by_global_norm(
                policy_gradients, self._max_gradient_norm
            )[0]
            critic_gradients = tf.clip_by_global_norm(
                critic_gradients, self._max_gradient_norm
            )[0]

            # Apply gradients.
            self._policy_optimizers[agent_key].apply(policy_gradients, policy_variables)
            self._critic_optimizers[agent_key].apply(critic_gradients, critic_variables)
        train_utils.safe_del(self, "tape")

    def step(self) -> None:
        """Trainer step to update the parameters of the agents in the system"""

        # Run the learning step.
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        raise NotImplementedError("A trainer statistics wrapper should overwrite this.")

        # Update our counts and record it.
        # TODO (dries): Can this be simplified? Only one set and one get?
        self._variable_client.add_async(
            ["trainer_steps", "trainer_walltime"],
            {"trainer_steps": 1, "trainer_walltime": elapsed_time},
        )

        # Update the variable source and the trainer
        self._variable_client.set_and_get_async()

        # Checkpoint and attempt to write the logs.
        if self._checkpoint:
            train_utils.checkpoint_networks(self._system_checkpointer)

        if self._logger:
            self._logger.write(fetches)

    def after_trainer_step(self) -> None:
        """Optionally decay lr after every training step."""

        if self._learning_rate_scheduler_fn:
            self._decay_lr(self._num_steps)
            info: Dict[str, Dict[str, float]] = {}
            for agent in self._agents:
                info[agent] = {}
                info[agent]["policy_learning_rate"] = self._policy_optimizers[
                    self._agent_net_keys[agent]
                ].learning_rate
                info[agent]["critic_learning_rate"] = self._critic_optimizers[
                    self._agent_net_keys[agent]
                ].learning_rate
            if self._logger:
                self._logger.write(info)

    def _decay_lr(self, trainer_step: int) -> None:
        """Decay lr.

        Args:
            trainer_step : trainer step time t.
        """
        train_utils.decay_lr_actor_critic(
            self._learning_rate_scheduler_fn,
            self._policy_optimizers,
            self._critic_optimizers,
            trainer_step,
        )


class MADDPGDecentralisedTrainer(MADDPGBaseTrainer):
    """MADDPG trainer for a decentralised architecture."""

    def __init__(
        self,
        agents: List[str],
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
        """Initialise the decentralised MADDPG trainer."""
        super().__init__(
            agents=agents,
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


class MADDPGCentralisedTrainer(MADDPGBaseTrainer):
    """MADDPG trainer for a centralised architecture."""

    def __init__(
        self,
        agents: List[str],
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
        """Initialise the centralised MADDPG trainer."""

        super().__init__(
            agents=agents,
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

    def _get_critic_feed(
        self,
        o_tm1_trans: Dict[str, np.ndarray],
        o_t_trans: Dict[str, np.ndarray],
        a_tm1: Dict[str, np.ndarray],
        a_t: Dict[str, np.ndarray],
        e_tm1: Dict[str, np.ndarray],
        e_t: Dict[str, np.ndarray],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get critic feed."""

        # Centralised based
        o_tm1_feed = tf.stack([o_tm1_trans[agent] for agent in self._agents], 1)
        o_t_feed = tf.stack([o_t_trans[agent] for agent in self._agents], 1)
        a_tm1_feed = tf.stack([a_tm1[agent] for agent in self._agents], 1)
        a_t_feed = tf.stack([a_t[agent] for agent in self._agents], 1)

        return o_tm1_feed, o_t_feed, a_tm1_feed, a_t_feed

    def _get_dpg_feed(
        self,
        a_t: Dict[str, np.ndarray],
        dpg_a_t: np.ndarray,
        agent: str,
    ) -> tf.Tensor:
        """Get DPG feed."""

        # Centralised and StateBased DPG
        # Note (dries): Copy has to be made because the input
        # variables cannot be changed.
        tree.map_structure(tf.stop_gradient, a_t)
        dpg_a_t_feed = copy.copy(a_t)
        dpg_a_t_feed[agent] = dpg_a_t

        dpg_a_t_feed = tf.squeeze(
            tf.stack([dpg_a_t_feed[agent] for agent in self._agents], 1)
        )

        return dpg_a_t_feed


class MADDPGNetworkedTrainer(MADDPGBaseTrainer):
    """MADDPG trainer for a networked architecture."""

    def __init__(
        self,
        agents: List[str],
        connection_spec: Dict[str, List[str]],
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
        """Initialise the networked MADDPG trainer."""

        super().__init__(
            agents=agents,
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
        self._connection_spec = connection_spec

    def _get_critic_feed(
        self,
        o_tm1_trans: Dict[str, np.ndarray],
        o_t_trans: Dict[str, np.ndarray],
        a_tm1: Dict[str, np.ndarray],
        a_t: Dict[str, np.ndarray],
        e_tm1: Dict[str, np.ndarray],
        e_t: Dict[str, np.ndarray],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get critic feed."""

        # Networked based
        connections = self._connection_spec[agent]
        o_tm1_vals = []
        o_t_vals = []
        a_tm1_vals = []
        a_t_vals = []

        # The agent has to be in the connections for training to work
        assert agent in connections

        for connected_agent in connections:
            o_tm1_vals.append(o_tm1_trans[connected_agent])
            o_t_vals.append(o_t_trans[connected_agent])
            a_tm1_vals.append(a_tm1[connected_agent])
            a_t_vals.append(a_t[connected_agent])
        o_tm1_feed = tf.stack(o_tm1_vals, 1)
        o_t_feed = tf.stack(o_t_vals, 1)
        a_tm1_feed = tf.stack(a_tm1_vals, 1)
        a_t_feed = tf.stack(a_t_vals, 1)

        return o_tm1_feed, o_t_feed, a_tm1_feed, a_t_feed

    def _get_dpg_feed(
        self,
        a_t: Dict[str, np.ndarray],
        dpg_a_t: np.ndarray,
        agent: str,
    ) -> tf.Tensor:
        """Get DPG feed."""

        # Networked based
        tree.map_structure(tf.stop_gradient, a_t)
        dpg_a_t_feed = copy.copy(a_t)
        dpg_a_t_feed[agent] = dpg_a_t

        connections = self._connection_spec[agent]

        # The agent has to be in the connections for training to work
        assert agent in connections

        a_t_vals = []
        for connected_agent in connections:
            a_t_vals.append(dpg_a_t_feed[connected_agent])
        dpg_a_t_feed = tf.squeeze(tf.stack(a_t_vals, 1))
        return dpg_a_t_feed


class MADDPGStateBasedTrainer(MADDPGBaseTrainer):
    """MADDPG trainer for a state-based architecture."""

    def __init__(
        self,
        agents: List[str],
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
        """Initialise the decentralised MADDPG trainer."""
        super().__init__(
            agents=agents,
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

    def _get_critic_feed(
        self,
        o_tm1_trans: Dict[str, np.ndarray],
        o_t_trans: Dict[str, np.ndarray],
        a_tm1: Dict[str, np.ndarray],
        a_t: Dict[str, np.ndarray],
        e_tm1: Dict[str, np.ndarray],
        e_t: Dict[str, np.ndarray],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get critic feed."""

        # State based
        o_tm1_feed = e_tm1["s_t"]
        o_t_feed = e_t["s_t"]
        a_tm1_feed = tf.stack([a_tm1[agent] for agent in self._agents], 1)
        a_t_feed = tf.stack([a_t[agent] for agent in self._agents], 1)

        return o_tm1_feed, o_t_feed, a_tm1_feed, a_t_feed

    def _get_dpg_feed(
        self,
        a_t: Dict[str, np.ndarray],
        dpg_a_t: np.ndarray,
        agent: str,
    ) -> tf.Tensor:
        """Get DPG feed."""

        # Centralised and StateBased DPG
        # Note (dries): Copy has to be made because the input
        # variables cannot be changed.
        tree.map_structure(tf.stop_gradient, a_t)
        dpg_a_t_feed = copy.copy(a_t)
        dpg_a_t_feed[agent] = dpg_a_t

        dpg_a_t_feed = tf.squeeze(
            tf.stack([dpg_a_t_feed[agent] for agent in self._agents], 1)
        )

        return dpg_a_t_feed


class MADDPGBaseRecurrentTrainer(mava.Trainer):
    """Recurrent MADDPG trainer.

    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
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
        """Initialise Recurrent MADDPG trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            policy_networks: policy networks for each agent in
                the system.
            critic_networks: critic network(s), shared or for
                each agent in the system.
            target_policy_networks: target policy networks.
            target_critic_networks: target critic networks.
            policy_optimizer:
                optimizer(s) for updating policy networks.
            critic_optimizer:
                optimizer for updating critic networks.
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
        self._bootstrap_n = bootstrap_n

        self._agents = agents
        self._agent_net_keys = agent_net_keys
        self._variable_client = variable_client
        self._learning_rate_scheduler_fn = learning_rate_scheduler_fn

        # Setup counts
        self._counts = counts

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

        # Create an iterator to go through the dataset.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        # Dictionary with unique network keys.
        self.unique_net_keys = sort_str_num(self._policy_networks.keys())

        # Get the agents which shoud be updated and ran
        self._trainer_agent_list = self._agents

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
            "critics": {},
            "policies": {},
        }
        for agent_key in self.unique_net_keys:
            policy_network_to_expose = snt.Sequential(
                [
                    self._target_observation_networks[agent_key],
                    self._target_policy_networks[agent_key],
                ]
            )
            policy_networks_to_expose[agent_key] = policy_network_to_expose
            self._system_network_variables["critics"][
                agent_key
            ] = target_critic_networks[agent_key].variables
            self._system_network_variables["policies"][
                agent_key
            ] = policy_network_to_expose.variables

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp: Optional[float] = None

    def _update_target_networks(self) -> None:
        """Sync the target parameters with the latest online

        parameters for all networks"""

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

            if self._target_averaging:
                assert 0.0 < self._target_update_rate < 1.0
                tau = self._target_update_rate
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(dest * (1.0 - tau) + src * tau)
            else:
                # Make online -> target network update ops.
                if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                    for src, dest in zip(online_variables, target_variables):
                        dest.assign(src)
        self._num_steps.assign_add(1)

    def _transform_observations(
        self, observations: Dict[str, mava_types.OLT]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Apply the observation networks to the raw observations from the dataset

        Args:
            obs: raw agent observations
            next_obs: raw next observations
        Returns:
            transformed
                observations (features)
        """

        # Note (dries): We are assuming that only the policy network
        # is recurrent and not the observation network.
        obs_trans = {}
        obs_target_trans = {}
        for agent in self._agents:
            agent_key = self._agent_net_keys[agent]

            reshaped_obs, dims = train_utils.combine_dim(
                observations[agent].observation
            )

            obs_trans[agent] = train_utils.extract_dim(
                self._observation_networks[agent_key](reshaped_obs), dims
            )

            obs_target_trans[agent] = train_utils.extract_dim(
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
        """Get data to feed to the agent critic network(s)

        Args:
            o_tm1_trans: transformed (e.g. using observation
                network) observation at timestep t-1
            o_t_trans: transformed observation at timestep t
            a_tm1: action at timestep t-1
            a_t: action at timestep t
            e_tm1: extras at timestep t-1
            e_t: extras at timestep t
            agent: agent id

        Returns:
            agent critic network
                feeds
        """

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
        """Get data to feed to the agent networks

        Args:
            a_t: action at timestep t
            dpg_a_t: predicted action at timestep t
            agent: agent id

        Returns:
            tf.Tensor: agent policy network feed
        """

        # Decentralised DPG
        dpg_actions_feed = dpg_actions
        return dpg_actions_feed

    def _target_policy_actions(
        self,
        target_obs_trans: Dict[str, np.ndarray],
        target_core_state: Dict[str, np.ndarray],
    ) -> Any:
        """Select actions using target policy networks

        Args:
            target_obs_trans: agent transformed target
                observations.
            target_core_state: target recurrent network state

        Returns:
            Any: agent target actions
        """

        actions = {}

        for agent in self._agents:
            time.time()
            agent_key = self._agent_net_keys[agent]
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

    @tf.function
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Trainer forward and backward passes.'

        Returns:
            losses
        """
        # Update the target networks
        self._update_target_networks()

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs: reverb.ReplaySample = next(self._iterator)

        self._forward(inputs)

        self._backward()

        # Log losses per agent
        return train_utils.map_losses_per_agent_ac(
            self.critic_losses, self.policy_losses
        )

    # Forward pass that calculates loss.
    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass

        Args:
            inputs: input data from the data table (transitions)
        """

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
                flat_q_values = self._critic_networks[agent_key](obs_comb, act_comb)
                q_values = train_utils.extract_dim(flat_q_values, dims)[:, :, 0]

                # Remove first sequence step for the target
                obs_comb, _ = train_utils.combine_dim(target_obs_trans_feed)
                act_comb, _ = train_utils.combine_dim(target_actions_feed)
                flat_target_q_values = self._target_critic_networks[agent_key](
                    obs_comb, act_comb
                )
                target_q_values = train_utils.extract_dim(flat_target_q_values, dims)[
                    :, :, 0
                ]

                # Critic loss.
                # Compute the transformed n-step loss.

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
                    loss_fn=trfl.td_learning,
                )

                self.critic_losses[agent] = tf.reduce_mean(critic_loss)

                # Actor learning.
                obs_agent_feed = target_obs_trans[agent]
                # TODO (dries): Why is there an extra tuple?
                agent_core_state = core_state[agent][0]
                transposed_obs = tf2_utils.batch_to_sequence(obs_agent_feed)
                outputs, updated_states = snt.static_unroll(
                    self._policy_networks[agent_key], transposed_obs, agent_core_state
                )

                dpg_actions = tf2_utils.batch_to_sequence(outputs)

                # Note (dries): This is done to so that losses.dpg
                # can verify using gradient.tape that there is a
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

                dpg_q_values = tf.squeeze(
                    self._critic_networks[agent_key](obs_comb, act_comb)
                )

                # Actor loss. If clipping is true use dqda clipping and clip the norm.
                dqda_clipping = 1.0 if self._max_gradient_norm is not None else None
                clip_norm = True if self._max_gradient_norm is not None else False

                policy_loss = losses.dpg(
                    q_max=dpg_q_values,
                    a_max=dpg_actions_comb,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=clip_norm,
                )

                # Multiply by discounts to not train on padded data.
                policy_mask = tf.reshape(step_not_padded, policy_loss.shape)
                policy_loss = policy_loss * policy_mask
                self.policy_losses[agent] = tf.reduce_sum(policy_loss) / tf.reduce_sum(
                    policy_mask
                )
        self.tape = tape

    # Backward pass that calculates gradients and updates network.
    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

        # Calculate the gradients and update the networks
        policy_losses = self.policy_losses
        critic_losses = self.critic_losses
        tape = self.tape
        for agent in self._agents:
            agent_key = self._agent_net_keys[agent]

            # Get trainable variables.
            # Note: policy does not update the shared obs network
            policy_variables = self._policy_networks[agent_key].trainable_variables

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
            policy_gradients = tf.clip_by_global_norm(
                policy_gradients, self._max_gradient_norm
            )[0]
            critic_gradients = tf.clip_by_global_norm(
                critic_gradients, self._max_gradient_norm
            )[0]

            # Apply gradients.
            self._policy_optimizers[agent_key].apply(policy_gradients, policy_variables)
            self._critic_optimizers[agent_key].apply(critic_gradients, critic_variables)
        train_utils.safe_del(self, "tape")

    def step(self) -> None:
        """Trainer step to update the parameters of the agents in the system"""

        # Run the learning step.
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        raise NotImplementedError("A trainer statistics wrapper should overwrite this.")

        # Update our counts and record it.
        # TODO (dries): Can this be simplified? Only one set and one get?
        self._variable_client.add_async(
            ["trainer_steps", "trainer_walltime"],
            {"trainer_steps": 1, "trainer_walltime": elapsed_time},
        )

        # Update the variable source and the trainer
        self._variable_client.set_and_get_async()

        # Checkpoint and attempt to write the logs.
        if self._checkpoint:
            train_utils.checkpoint_networks(self._system_checkpointer)

        if self._logger:
            self._logger.write(fetches)

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Get network variables

        Args:
            names: network names
        Returns:
            network variables
        """

        variables: Dict[str, Dict[str, np.ndarray]] = {}
        for network_type in names:
            variables[network_type] = {
                agent: tf2_utils.to_numpy(
                    self._system_network_variables[network_type][agent]
                )
                for agent in self.unique_net_keys
            }
        return variables

    def after_trainer_step(self) -> None:
        """Optionally decay lr after every training step."""
        if self._learning_rate_scheduler_fn:
            self._decay_lr(self._num_steps)
            info: Dict[str, Dict[str, float]] = {}
            for agent in self._agents:
                info[agent] = {}
                info[agent]["policy_learning_rate"] = self._policy_optimizers[
                    self._agent_net_keys[agent]
                ].learning_rate
                info[agent]["critic_learning_rate"] = self._critic_optimizers[
                    self._agent_net_keys[agent]
                ].learning_rate
            if self._logger:
                self._logger.write(info)

    def _decay_lr(self, trainer_step: int) -> None:
        """Decay lr.

        Args:
            trainer_step : trainer step time t.
        """
        train_utils.decay_lr_actor_critic(
            self._learning_rate_scheduler_fn,
            self._policy_optimizers,
            self._critic_optimizers,
            trainer_step,
        )


class MADDPGDecentralisedRecurrentTrainer(MADDPGBaseRecurrentTrainer):
    """Recurrent MADDPG trainer for a decentralised architecture.

    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
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

        super().__init__(
            agents=agents,
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


class MADDPGCentralisedRecurrentTrainer(MADDPGBaseRecurrentTrainer):
    """Recurrent MADDPG trainer for a centralised architecture.

    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
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

        super().__init__(
            agents=agents,
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
        obs_trans_feed = tf.stack([obs_trans[agent] for agent in self._agents], -1)
        target_obs_trans_feed = tf.stack(
            [target_obs_trans[agent] for agent in self._agents], -1
        )
        actions_feed = tf.stack([actions[agent] for agent in self._agents], -1)
        target_actions_feed = tf.stack(
            [target_actions[agent] for agent in self._agents], -1
        )

        return obs_trans_feed, target_obs_trans_feed, actions_feed, target_actions_feed

    def _get_dpg_feed(
        self,
        actions: Dict[str, np.ndarray],
        dpg_actions: np.ndarray,
        agent: str,
    ) -> tf.Tensor:

        # Centralised and StateBased DPG
        # Note (dries): Copy has to be made because the input
        # variables cannot be changed.
        tree.map_structure(tf.stop_gradient, actions)
        dpg_actions_feed = copy.copy(actions)
        dpg_actions_feed[agent] = dpg_actions
        dpg_actions_feed = tf.squeeze(
            tf.stack([dpg_actions_feed[agent] for agent in self._agents], -1)
        )
        return dpg_actions_feed


class MADDPGStateBasedRecurrentTrainer(MADDPGBaseRecurrentTrainer):
    """Recurrent MADDPG trainer for a state-based architecture.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
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

        super().__init__(
            agents=agents,
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
        obs_trans_feed = extras["s_t"]
        target_obs_trans_feed = extras["s_t"]
        if type(obs_trans_feed) == dict:  # type: ignore
            obs_trans_feed = obs_trans_feed[agent]
            target_obs_trans_feed = target_obs_trans_feed[agent]

        actions_feed = tf.stack([actions[agent] for agent in self._agents], -1)
        target_actions_feed = tf.stack(
            [target_actions[agent] for agent in self._agents], -1
        )

        return obs_trans_feed, target_obs_trans_feed, actions_feed, target_actions_feed

    def _get_dpg_feed(
        self,
        actions: Dict[str, np.ndarray],
        dpg_actions: np.ndarray,
        agent: str,
    ) -> tf.Tensor:

        # Centralised and StateBased DPG
        # Note (dries): Copy has to be made because the input
        # variables cannot be changed.
        tree.map_structure(tf.stop_gradient, actions)
        dpg_actions_feed = copy.copy(actions)
        dpg_actions_feed[agent] = dpg_actions
        dpg_actions_feed = tf.squeeze(
            tf.stack([dpg_actions_feed[agent] for agent in self._agents], -1)
        )
        return dpg_actions_feed


class MADDPGStateBasedSingleActionCriticRecurrentTrainer(MADDPGBaseRecurrentTrainer):
    """Recurrent MADDPG trainer for a state-based architecture.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
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

        super().__init__(
            agents=agents,
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
        obs_trans_feed = extras["s_t"][agent]
        target_obs_trans_feed = extras["s_t"][agent]
        actions_feed = actions[agent]
        target_actions_feed = target_actions[agent]
        return obs_trans_feed, target_obs_trans_feed, actions_feed, target_actions_feed

    def _get_dpg_feed(
        self,
        target_actions: Dict[str, np.ndarray],
        dpg_actions: np.ndarray,
        agent: str,
    ) -> tf.Tensor:
        # Decentralised DPG
        dpg_actions_feed = dpg_actions
        return dpg_actions_feed
