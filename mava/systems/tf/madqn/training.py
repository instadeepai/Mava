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


"""MADQN trainer implementation."""
import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree
import trfl
from acme.tf import utils as tf2_utils
from acme.utils import loggers

import mava
from mava import types as mava_types
from mava.systems.tf.variable_utils import VariableClient
from mava.utils import training_utils as train_utils
from mava.utils.sort_utils import sort_str_num

train_utils.set_growing_gpu_memory()


class MADQNTrainer(mava.Trainer):
    """MADQN trainer.

    This is the trainer component of a MADQN system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        value_networks: Dict[str, snt.Module],
        target_value_networks: Dict[str, snt.Module],
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
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
        """Initialise MADQN trainer.

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            value_networks: value networks for each agents in
                the system.
            target_value_networks: target value networks.
            optimizer: optimizer(s) for updating policy networks.
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
        self._agent_types = agent_types
        self._agent_net_keys = agent_net_keys
        self._variable_client = variable_client
        self._learning_rate_scheduler_fn = learning_rate_scheduler_fn

        # Setup counts
        self._counts = counts

        # Store online and target networks.
        self._value_networks = value_networks
        self._target_value_networks = target_value_networks

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
        self.unique_net_keys = sort_str_num(self._value_networks.keys())

        # Get the agents which shoud be updated and ran
        self._trainer_agent_list = self._agents

        # Create optimizers for different agent types.
        if not isinstance(optimizer, dict):
            self._optimizers: Dict[str, snt.Optimizer] = {}
            for agent in self.unique_net_keys:
                self._optimizers[agent] = copy.deepcopy(optimizer)
        else:
            self._optimizers = optimizer

        # Expose the variables.
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "observations": {},
            "values": {},
        }
        for agent_key in self.unique_net_keys:
            self._system_network_variables["observations"][
                agent_key
            ] = self._target_observation_networks[agent_key].variables
            self._system_network_variables["values"][agent_key] = self._value_networks[
                agent_key
            ].variables

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp: Optional[float] = None

    def _update_target_networks(self) -> None:
        """Update the target networks.

        Using either target averaging or
        by directy copying the weights of the online networks every few steps.
        """

        for key in self.unique_net_keys:
            # Update target network.
            online_variables = (
                *self._observation_networks[key].variables,
                *self._value_networks[key].variables,
            )
            target_variables = (
                *self._target_observation_networks[key].variables,
                *self._target_value_networks[key].variables,
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
        """Depricated"""

        pass

    def _transform_observations(
        self, obs: Dict[str, mava_types.OLT], next_obs: Dict[str, mava_types.OLT]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Transform the observations using the observation networks of each agent.

        We assume the observation network is non-recurrent.

        Args:
            obs: observations at timestep t-1
            next_obs: observations at timestep t

        Returns:
            Transformed observations
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

    @tf.function
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Trainer step.

        Returns:
            losses
        """

        # Draw a batch of data from replay.
        sample: reverb.ReplaySample = next(self._iterator)

        # Compute loss
        self._forward(sample)

        # Compute and apply gradients
        self._backward()

        # Update the target networks
        self._update_target_networks()

        # Log losses per agent
        return train_utils.map_losses_per_agent_value(self.value_losses)

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass.

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
        o_tm1, o_t, a_tm1, r_t, d_t, _, _ = (
            trans.observations,
            trans.next_observations,
            trans.actions,
            trans.rewards,
            trans.discounts,
            trans.extras,
            trans.next_extras,
        )

        self.value_losses = {}
        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:

            o_tm1_trans, o_t_trans = self._transform_observations(o_tm1, o_t)
            for agent in self._trainer_agent_list:
                agent_key = self._agent_net_keys[agent]

                # Double Q-learning
                q_tm1 = self._value_networks[agent_key](o_tm1_trans[agent])
                q_t_value = self._target_value_networks[agent_key](o_t_trans[agent])
                q_t_selector = self._value_networks[agent_key](o_t_trans[agent])

                # Legal action masking
                q_t_selector = tf.where(
                    tf.cast(o_t[agent].legal_actions, "bool"), q_t_selector, -999999999
                )

                # pcont
                discount = tf.cast(self._discount, dtype=d_t[agent].dtype)

                # Value loss.
                value_loss, _ = trfl.double_qlearning(
                    q_tm1,
                    a_tm1[agent],
                    r_t[agent],
                    discount * d_t[agent],
                    q_t_value,
                    q_t_selector,
                )

                self.value_losses[agent] = tf.reduce_mean(value_loss, axis=0)

        self.tape = tape

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

        # Calculate the gradients and update the networks
        value_losses = self.value_losses
        tape = self.tape
        for agent in self._trainer_agent_list:
            agent_key = self._agent_net_keys[agent]

            # Get trainable variables.
            variables = (
                self._observation_networks[agent_key].trainable_variables
                + self._value_networks[agent_key].trainable_variables
            )

            # Compute gradients.
            # Note: Warning "WARNING:tensorflow:Calling GradientTape.gradient
            #  on a persistent tape inside its context is significantly less efficient
            #  than calling it outside the context." caused by losses.dpg, which calls
            #  tape.gradient.
            gradients = tape.gradient(value_losses[agent], variables)

            # Maybe clip gradients.
            gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

            # Apply gradients.
            self._optimizers[agent_key].apply(gradients, variables)

        train_utils.safe_del(self, "tape")

    def step(self) -> None:
        """Trainer step to update the parameters of the agents in the system"""

        raise NotImplementedError("A trainer statistics wrapper should overwrite this.")

    def after_trainer_step(self) -> None:
        """Optionally decay lr after every training step."""
        if self._learning_rate_scheduler_fn:
            self._decay_lr(self._num_steps)
            info: Dict[str, Dict[str, float]] = {}
            for agent in self._agents:
                info[agent] = {}
                info[agent]["learning_rate"] = self._optimizers[
                    self._agent_net_keys[agent]
                ].learning_rate
            if self._logger:
                self._logger.write(info)

    def _decay_lr(self, trainer_step: int) -> None:
        """Decay lr.

        Args:
            trainer_step : trainer step time t.
        """
        train_utils.decay_lr(
            self._learning_rate_scheduler_fn,  # type: ignore
            self._optimizers,
            trainer_step,
        )


class MADQNRecurrentTrainer(mava.Trainer):
    """Recurrent MADQN trainer.

    This is the trainer component of a recurrent MADQN system. IE it takes a dataset
    as input and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        value_networks: Dict[str, snt.Module],
        target_value_networks: Dict[str, snt.Module],
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
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
        """Initialise Recurrent MADQN trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            value_networks: value networks for each agent in
                the system.
            target_value_networks: target value networks.
            optimizer: optimizer(s) for updating value networks.
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
        self._agent_type = agent_types
        self._agent_net_keys = agent_net_keys
        self._variable_client = variable_client
        self._learning_rate_scheduler_fn = learning_rate_scheduler_fn

        # Setup counts
        self._counts = counts

        # Store online and target networks.
        self._value_networks = value_networks
        self._target_value_networks = target_value_networks

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
        self.unique_net_keys = sort_str_num(self._value_networks.keys())

        # Get the agents which shoud be updated and ran
        self._trainer_agent_list = self._agents

        # Create optimizers for different agent types.
        if not isinstance(optimizer, dict):
            self._optimizers: Dict[str, snt.Optimizer] = {}
            for agent in self.unique_net_keys:
                self._optimizers[agent] = copy.deepcopy(optimizer)
        else:
            self._optimizers = optimizer

        # Expose the variables.
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "observations": {},
            "values": {},
        }
        for agent_key in self.unique_net_keys:
            self._system_network_variables["observations"][
                agent_key
            ] = self._target_observation_networks[agent_key].variables
            self._system_network_variables["values"][
                agent_key
            ] = self._target_value_networks[agent_key].variables

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp: Optional[float] = None

    def step(self) -> None:
        """Trainer step to update the parameters of the agents in the system"""

        raise NotImplementedError("A trainer statistics wrapper should overwrite this.")

    def _transform_observations(
        self, observations: Dict[str, mava_types.OLT]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Apply the observation networks to the raw observations from the dataset

        We assume that the observation network is non-recurrent.

        Args:
            observations: raw agent observations

        Returns:
            obs_trans: transformed agent observation
            obs_target_trans: transformed target network observations
        """

        # NOTE We are assuming that only the value network
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

            # This stop_gradient prevents gradients to propagate into
            # the target observation network.
            obs_target_trans[agent] = tree.map_structure(
                tf.stop_gradient, obs_target_trans[agent]
            )
        return obs_trans, obs_target_trans

    def _update_target_networks(self) -> None:
        """Update the target networks.

        Using either target averaging or
        by directy copying the weights of the online networks every few steps.
        """
        for key in self.unique_net_keys:
            # Update target network.
            online_variables = (
                *self._observation_networks[key].variables,
                *self._value_networks[key].variables,
            )
            target_variables = (
                *self._target_observation_networks[key].variables,
                *self._target_value_networks[key].variables,
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
        """Depricated"""
        pass

    @tf.function
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Trainer step.

        Returns:
            losses
        """

        # Draw a batch of data from replay.
        sample: reverb.ReplaySample = next(self._iterator)

        # Compute loss
        self._forward(sample)

        # Compute and apply gradients
        self._backward()

        # Update the target networks
        self._update_target_networks()

        # Log losses per agent
        return train_utils.map_losses_per_agent_value(self.value_losses)

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass.

        Args:
            inputs: input data from the data table (transitions)
        """
        # Convert to time major
        data = tree.map_structure(
            lambda v: tf.expand_dims(v, axis=0) if len(v.shape) <= 1 else v, inputs.data
        )
        data = tf2_utils.batch_to_sequence(data)

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
        # extract the first state in the sequence.
        core_state = tree.map_structure(lambda s: s[0, :, :], extras["core_states"])
        target_core_state = tree.map_structure(
            lambda s: s[0, :, :], extras["core_states"]
        )

        # TODO (dries): Take out all the data_points that does not need
        #  to be processed here at the start. Therefore it does not have
        #  to be done later on and saves processing time.

        self.value_losses: Dict[str, tf.Tensor] = {}

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            # Note (dries): We are assuming that only the policy network
            # is recurrent and not the observation network.
            obs_trans, target_obs_trans = self._transform_observations(observations)

            for agent in self._trainer_agent_list:
                agent_key = self._agent_net_keys[agent]

                # Double Q-learning
                q, _ = snt.static_unroll(
                    self._value_networks[agent_key],
                    obs_trans[agent],
                    core_state[agent][0],
                )
                q_tm1 = q[:-1]  # Chop off last timestep
                q_t_selector = q[1:]  # Chop off first timestep
                q_t_value, _ = snt.static_unroll(
                    self._target_value_networks[agent_key],
                    target_obs_trans[agent],
                    target_core_state[agent][0],
                )
                q_t_value = q_t_value[1:]  # Chop off first timestep

                # Legal action masking
                q_t_selector = tf.where(
                    tf.cast(observations[agent].legal_actions[1:], "bool"),
                    q_t_selector,
                    -999999999,
                )

                # Flatten out time and batch dim
                q_tm1, _ = train_utils.combine_dim(q_tm1)
                q_t_selector, _ = train_utils.combine_dim(q_t_selector)
                q_t_value, _ = train_utils.combine_dim(q_t_value)
                a_tm1, _ = train_utils.combine_dim(
                    actions[agent][:-1]  # Chop off last timestep
                )
                r_t, _ = train_utils.combine_dim(
                    rewards[agent][:-1]  # Chop off last timestep
                )
                d_t, _ = train_utils.combine_dim(
                    discounts[agent][:-1]  # Chop off last timestep
                )

                # Cast the additional discount to match
                # the environment discount dtype.
                discount = tf.cast(self._discount, dtype=discounts[agent].dtype)

                # Value loss
                value_loss, _ = trfl.double_qlearning(
                    q_tm1, a_tm1, r_t, discount * d_t, q_t_value, q_t_selector
                )

                # Zero-padding mask
                zero_padding_mask, _ = train_utils.combine_dim(
                    tf.cast(extras["zero_padding_mask"], dtype=value_loss.dtype)[:-1]
                )
                masked_loss = value_loss * zero_padding_mask
                self.value_losses[agent] = tf.reduce_sum(masked_loss) / tf.reduce_sum(
                    zero_padding_mask
                )

        self.tape = tape

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

        # Calculate the gradients and update the networks
        value_losses = self.value_losses
        tape = self.tape
        for agent in self._trainer_agent_list:
            agent_key = self._agent_net_keys[agent]

            # Get trainable variables.
            variables = (
                self._observation_networks[agent_key].trainable_variables
                + self._value_networks[agent_key].trainable_variables
            )

            # Compute gradients.
            gradients = tape.gradient(value_losses[agent], variables)

            # Maybe clip gradients.
            gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

            # Apply gradients.
            self._optimizers[agent_key].apply(gradients, variables)

        train_utils.safe_del(self, "tape")

    def after_trainer_step(self) -> None:
        """Optionally decay lr after every training step."""
        if self._learning_rate_scheduler_fn:
            self._decay_lr(self._num_steps)
            info: Dict[str, Dict[str, float]] = {}
            for agent in self._agents:
                info[agent] = {}
                info[agent]["learning_rate"] = self._optimizers[
                    self._agent_net_keys[agent]
                ].learning_rate
            if self._logger:
                self._logger.write(info)

    def _decay_lr(self, trainer_step: int) -> None:
        """Decay lr.

        Args:
            trainer_step : trainer step time t.
        """
        train_utils.decay_lr(
            self._learning_rate_scheduler_fn,  # type: ignore
            self._optimizers,
            trainer_step,
        )
