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


"""Value Decomposition trainer implementation."""

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
from mava.systems.tf.madqn.execution import MADQNFeedForwardExecutor
from mava.systems.tf.madqn.training import MADQNRecurrentTrainer
from mava.systems.tf.variable_utils import VariableClient
from mava.utils import training_utils as train_utils
from mava.utils.sort_utils import sort_str_num

train_utils.set_growing_gpu_memory()


class ValueDecompositionRecurrentTrainer(MADQNRecurrentTrainer):
    """MADQN trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
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
        """Initialise MADDPG trainer
        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
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

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            value_networks=value_networks,
            target_value_networks=target_value_networks,
            optimizer=optimizer,
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

        self._mixer = None
        self._target_mixer = None
        self._mixer_optimizer = None

    def setup_mixer(self, mixer: snt.Module, mixer_optimizer: snt.Module):
        self._mixer = mixer
        self._target_mixer = copy.deepcopy(mixer)
        self._mixer_optimizer = mixer_optimizer

    def _update_target_networks(self) -> None:
        """Update the target networks using either target averaging or
        by directy copying the weights of the online networks every few steps."""

        online_variables = []
        target_variables = []
        for key in self.unique_net_keys:
            # Update target network.
            online_variables += list((
                *self._observation_networks[key].variables,
                *self._value_networks[key].variables,
            ))
            target_variables += list((
                *self._target_observation_networks[key].variables,
                *self._target_value_networks[key].variables,
            ))
        # Add mixer variables
        online_variables += list((
                *self._mixer.variables,
        ))
        target_variables += list((
                *self._target_mixer.variables,
        ))

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

    # Forward pass that calculates loss.
    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass
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

        # Global environment state
        if "s_t" in extras:
            global_env_state = extras["s_t"]
        else:
            global_env_state = None

        # Get initial state for the LSTM from replay and
        # extract the first state in the sequence.
        core_state = tree.map_structure(lambda s: s[0, :, :], extras["core_states"])
        target_core_state = tree.map_structure(lambda s: s[0, :, :], extras["core_states"])

        # TODO (dries): Take out all the data_points that does not need
        #  to be processed here at the start. Therefore it does not have
        #  to be done later on and saves processing time.
        # NOTE (Claude) or do zeropadding mask

        self.value_losses: Dict[str, tf.Tensor] = {}
        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            # Note (dries): We are assuming that only the policy network
            # is recurrent and not the observation network.
            obs_trans, target_obs_trans = self._transform_observations(observations)

            # Lists for stacking tensors later
            chosen_action_q_value_all_agents = []
            max_action_q_value_all_agents = []
            reward_all_agents = []
            env_discount_all_agents = []
            for agent in self._agents:
                agent_key = self._agent_net_keys[agent]

                # Double Q-learning
                q_tm1_values, _ = snt.static_unroll(
                    self._value_networks[agent_key], obs_trans[agent], core_state[agent][0]
                )
                # Q-value of the action taken by agent
                chosen_action_q_value = trfl.batched_index(
                    q_tm1_values, actions[agent]
                )


                # Q-value of the next state
                q_t_selector =  tf.where(
                    tf.cast(observations[agent].legal_actions, 'bool'), 
                    q_tm1_values, -999999999
                )
                q_t_values, _ = snt.static_unroll(
                    self._target_value_networks[agent_key], 
                    target_obs_trans[agent], 
                    target_core_state[agent][0]
                )
                max_action = tf.argmax(q_t_selector, axis=-1)
                max_action_q_value = trfl.batched_index(
                    q_t_values, 
                    max_action
                )


                # Append agent values to lists
                chosen_action_q_value_all_agents.append(chosen_action_q_value)
                max_action_q_value_all_agents.append(max_action_q_value)
                reward_all_agents.append(rewards[agent])
                env_discount_all_agents.append(discounts[agent])

            # Stack list of tensors into tensor with trailing agent dim
            chosen_action_q_value_all_agents = tf.stack(
                chosen_action_q_value_all_agents, axis=-1
            ) # shape=(T,B, Num_Agents)
            max_action_q_value_all_agents = tf.stack(
                max_action_q_value_all_agents, axis=-1
            ) # shape=(T,B, Num_Agents)
            reward_all_agents = tf.stack(reward_all_agents, axis=-1)
            env_discount_all_agents = tf.stack(env_discount_all_agents, axis=-1)

            # Mixing
            chosen_action_q_value_all_agents = self._mixer(
                chosen_action_q_value_all_agents,
                states=global_env_state,
            )
            max_action_q_value_all_agents = self._target_mixer(
                max_action_q_value_all_agents, 
                states=global_env_state
            )   
            # NOTE Team reward is just the mean over agents indevidual rewards
            reward_all_agents = tf.reduce_mean(
                reward_all_agents, axis=-1, keepdims=True
            )
            # NOTE We assume all agents have the same env discount since
            # it is a team game
            env_discount_all_agents = tf.reduce_mean(
                env_discount_all_agents, axis=-1, keepdims=True
            )

            # Cast the additional discount to match
            # the environment discount dtype.
            discount = tf.cast(self._discount, dtype=discounts[agent].dtype)
            pcont = discount * env_discount_all_agents

            # Bellman target
            target = tf.stop_gradient(
                reward_all_agents[:-1] + pcont[:-1] * max_action_q_value_all_agents[1:]
            )

            # Temporal difference error and loss.
            td_error = target - chosen_action_q_value_all_agents[:-1]

            # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
            value_loss = 0.5 * tf.square(td_error)
            value_loss = tf.reduce_mean(value_loss)

            # TODO zero padding mask

            self.value_losses = {agent: value_loss for agent in self._agents}
            self.mixer_loss = value_loss

        self.tape = tape

    # Backward pass that calculates gradients and updates network.
    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

        # Calculate the gradients and update the networks
        value_losses = self.value_losses
        mixer_loss = self.mixer_loss
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
            gradients = tf.clip_by_global_norm(
                gradients, self._max_gradient_norm
            )[0]

            # Apply gradients.
            self._optimizers[agent_key].apply(gradients, variables)

        # TODO (Claude) what happens when there are multiple trainers @Dries
        # Mixer
        mixer_variables = self._mixer.trainable_variables

        gradients = tape.gradient(mixer_loss, mixer_variables)

        # Clip gradients.
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

        # Apply gradients.
        if mixer_variables:
            tf.print("OPTIMIZING MIXER")
            self._mixer_optimizer.apply(gradients, mixer_variables)

        train_utils.safe_del(self, "tape")