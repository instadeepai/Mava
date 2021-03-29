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

import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import sonnet as snt
import tensorflow as tf
import tree
import trfl
from acme.tf import losses
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava

# NOTE (Arnu): in TF2 this should be the default
# but for some reason it is not when I run it.
# tf.config.run_functions_eagerly(True)


class MADDPGTrainer(mava.Trainer):
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

        # Create optimizers if they aren't given.
        self._critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-4)
        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)

        agent_keys = self._agent_type if shared_weights else self._agents

        # Expose the variables.
        policy_networks_to_expose = {}
        self._system_network_variables = {}
        self._system_checkpointer = {}
        for agent_key in agent_keys:
            policy_network_to_expose = snt.Sequential(
                [
                    self._target_observation_networks[agent_key],
                    self._target_policy_networks[agent_key],
                ]
            )
            policy_networks_to_expose[agent_key] = policy_network_to_expose
            variables = {
                "critic": target_critic_networks[agent_key].variables,
                "policy": policy_network_to_expose.variables,
            }
            self._system_network_variables[agent_key] = variables
            checkpointer = tf2_savers.Checkpointer(
                time_delta_minutes=5,
                objects_to_save={
                    "counter": self._counter,
                    "policy": self._policy_networks[agent_key],
                    "critic": self._critic_networks[agent_key],
                    "target_policy": self._target_policy_networks[agent_key],
                    "target_critic": self._target_critic_networks[agent_key],
                    "policy_optimizer": self._policy_optimizer,
                    "critic_optimizer": self._critic_optimizer,
                    "num_steps": self._num_steps,
                },
                enable_checkpointing=checkpoint,
            )
            self._system_checkpointer[agent_key] = checkpointer

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    @tf.function
    def _update_target_networks(self) -> None:
        for key in self._keys:
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

    @tf.function
    def _transform_observations(
        self, state: Dict[str, np.ndarray], next_state: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        s_tm1 = {}
        s_t = {}
        for key in self._keys:
            s_tm1[key] = self._observation_networks[key](state[key].observation)
            s_t[key] = self._target_observation_networks[key](
                next_state[key].observation
            )

            # This stop_gradient prevents gradients to propagate into the target
            # observation network. In addition, since the online policy network is
            # evaluated at o_t, this also means the policy loss does not influence
            # the observation network training.
            s_t[key] = tree.map_structure(tf.stop_gradient, s_t[key])
        return s_tm1, s_t

    @tf.function
    def _policy_actions(self, next_state: Dict[str, np.ndarray]) -> Any:
        actions = {}
        for agent in self._agents:
            agent_key = agent.split("_")[0] if self._shared_weights else agent
            next_observation = next_state[agent]
            actions[agent] = self._target_policy_networks[agent_key](next_observation)
        return actions

    # NOTE (Arnu): the decorator below was causing this _step() function not
    # to be called by the step() function below. Removing it makes the code
    # work. The docs on tf.function says it is useful for speed improvements
    # but as far as I can see, we can go ahead without it. At least for now.
    # @tf.function
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        self._keys = self._agent_types if self._shared_weights else self._agents
        self._update_target_networks()

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)

        # Unpack input data as follows:
        # s_tm1 = dictionary of observations one for each agent
        #   (forming the system state)
        # a_tm1 = dictionary of actions taken from obs in s_tm1
        # r_t = dictionary of rewards or rewards sequences
        #   (if using N step transitions) ensuing from actions a_tm1
        # d_t = environment discount ensuing from actions a_tm1.
        #   This discount is applied to future rewards after r_t.
        # s_t = dictionary of next observations or next observation sequences
        # e_t [Optional] = extra data that the agents persist in replay.
        s_tm1, a_tm1, r_t, d_t, s_t, e_t = inputs.data

        logged_losses: Dict[str, Dict[str, Any]] = {}

        for agent in self._agents:
            agent_key = agent.split("_")[0] if self._shared_weights else agent

            # Cast the additional discount to match the environment discount dtype.
            discount = tf.cast(self._discount, dtype=d_t[agent_key].dtype)

            with tf.GradientTape(persistent=True) as tape:
                # Maybe transform the observation before feeding into policy and critic.
                # Transforming the observations this way at the start of the learning
                # step effectively means that the policy and critic share observation
                # network weights.
                s_tm1_trans, s_t_trans = self._transform_observations(s_tm1, s_t)
                a_t = self._policy_actions(s_t_trans)

                o_t_feed = s_t_trans[agent]

                # NOTE (Arnu): This is the centralised case where we concat
                # obs to form states and concat all agent actions.
                # s_tm1_feed = tf.concat([x.numpy() for x in s_tm1.values()], 1)
                # s_t_feed = tf.concat([x.numpy() for x in s_t.values()], 1)
                # a_tm1_feed = tf.concat([x.numpy() for x in a_tm1.values()], 1)
                # a_t_feed = tf.concat([x.numpy() for x in a_t.values()], 1)

                # Decentralised critic
                s_tm1_feed = s_tm1_trans[agent_key]
                s_t_feed = s_t_trans[agent_key]
                a_tm1_feed = a_tm1[agent_key]
                a_t_feed = a_t[agent_key]

                # Critic learning.
                q_tm1 = self._critic_networks[agent_key](s_tm1_feed, a_tm1_feed)
                q_t = self._target_critic_networks[agent_key](s_t_feed, a_t_feed)

                # Squeeze into the shape expected by the td_learning implementation.
                q_tm1 = tf.squeeze(q_tm1, axis=-1)  # [B]
                q_t = tf.squeeze(q_t, axis=-1)  # [B]

                # Critic loss.
                critic_loss = trfl.td_learning(
                    q_tm1, r_t[agent], discount * d_t[agent], q_t
                ).loss
                critic_loss = tf.reduce_mean(critic_loss, axis=0)

                # Actor learning.
                dpg_a_t = self._policy_networks[agent_key](o_t_feed)
                dpg_a_t_feed = dpg_a_t

                # NOTE (Arnu): Below is for centralised case
                # dpg_a_t_feed = a_t
                # dpg_a_t_feed[agent] = dpg_a_t
                # dpg_q_t = self._critic_networks[agent_key](s_t_feed, dpg_a_t_feed)

                dpg_q_t = self._critic_networks[agent_key](s_t_feed, dpg_a_t_feed)

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

            # Get trainable variables.
            policy_variables = self._policy_networks[agent_key].trainable_variables
            critic_variables = (
                # In this agent, the critic loss trains the observation network.
                self._observation_networks[agent_key].trainable_variables
                + self._critic_networks[agent_key].trainable_variables
            )

            # Compute gradients.
            policy_gradients = tape.gradient(policy_loss, policy_variables)
            critic_gradients = tape.gradient(critic_loss, critic_variables)

            # Delete the tape manually because of the persistent=True flag.
            del tape

            # Maybe clip gradients.
            if self._clipping:
                policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.0)[0]
                critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.0)[0]

            # Apply gradients.
            self._policy_optimizer.apply(policy_gradients, policy_variables)
            self._critic_optimizer.apply(critic_gradients, critic_variables)

            logged_losses[agent] = {
                "critic_loss": critic_loss,
                "policy_loss": policy_loss,
            }

        # Losses to track.
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

        # Checkpoint and attempt to write the logs.

        # NOTE (Arnu): ignoring checkpointing and logging for now
        # self._checkpointer.save()
        # self._logger.write(fetches)

    def get_variables(
        self, names: Dict[str, Sequence[str]]
    ) -> Dict[str, List[List[np.ndarray]]]:
        variables = {}
        for agent in self._agents:
            variables[agent] = [
                tf2_utils.to_numpy(self._system_network_variables[agent][name])
                for name in names[agent]
            ]
        return variables
