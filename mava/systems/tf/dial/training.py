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

# TODO (Kevin): implement DIAL trainer
# Helper resources
#   - single agent dqn learner in acme:
#           https://github.com/deepmind/acme/blob/master/acme/agents/tf/dqn/learning.py
#   - multi-agent ddpg trainer in mava: mava/systems/tf/maddpg/trainer.py
#   - https://github.com/deepmind/acme/agents/tf/r2d2/learning.py

"""DIAL trainer implementation."""
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava

# from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.systems.tf import savers as tf2_savers
from mava.utils import training_utils as train_utils


class DIALTrainer(mava.Trainer):
    """DIAL trainer.
    This is the trainer component of a DIAL system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        networks: Dict[str, snt.Module],
        discount: float,
        huber_loss_parameter: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        policy_optimizer: snt.Optimizer,
        shared_weights: bool = True,
        importance_sampling_exponent: float = None,
        replay_client: Optional[reverb.Client] = None,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        max_gradient_norm: Optional[float] = None,
    ):
        """Initializes the learner.
        Args:
          policy_network: the online (optimized) policy.
          target_policy_network: the target policy (which lags behind the online
            policy).
          discount: discount to use for TD updates.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          dataset: dataset to learn from, whether fixed or from a replay buffer
            (see `acme.datasets.reverb.make_dataset` documentation).
          observation_network: an optional online network to process observations
            before the policy
          target_observation_network: the target observation network.
          policy_optimizer: the optimizer to be applied to the (policy) loss.
          clipping: whether to clip gradients by global norm.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """
        self._agents = agents
        self._agent_types = agent_types
        self._shared_weights = shared_weights
        self._communication_module = networks["communication_module"]["all_agents"]

        # Store online and target networks.
        self._policy_networks = networks["policies"]
        self._target_policy_networks = networks["target_policies"]

        # self._observation_networks = observation_networks
        self._observation_networks = {
            k: tf2_utils.to_sonnet_module(v)
            for k, v in networks["observations"].items()
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

        # Create optimizers if they aren't given.
        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)

        # Dictionary with network keys for each agent.
        self.agent_net_keys = {agent: agent for agent in self._agents}
        if self._shared_weights:
            self.agent_net_keys = {agent: agent.split("_")[0] for agent in self._agents}

        self.unique_net_keys = self._agent_types if shared_weights else self._agents

        policy_networks_to_expose = {}
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "policy": {},
        }
        for agent_key in self.unique_net_keys:
            policy_network_to_expose = snt.Sequential(
                [
                    self._observation_networks[agent_key],
                    self._policy_networks[agent_key],
                ]
            )
            policy_networks_to_expose[agent_key] = policy_network_to_expose
            self._system_network_variables["policy"][
                agent_key
            ] = policy_network_to_expose.variables

        self._system_checkpointer = {}
        if checkpoint:
            for agent_key in self.unique_net_keys:
                objects_to_save = {
                    "counter": self._counter,
                    "policy": self._policy_networks[agent_key],
                    "observation": self._observation_networks[agent_key],
                    "target_policy": self._target_policy_networks[agent_key],
                    "policy_optimizer": self._policy_optimizer,
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

        self._timestamp = None

    @tf.function
    def _update_target_networks(self) -> None:
        for key in self.unique_net_keys:
            # Update target network.
            online_variables = (*self._policy_networks[key].variables,)
            target_variables = (*self._target_policy_networks[key].variables,)

            # Make online -> target network update ops.
            if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(src)
            self._num_steps.assign_add(1)

    @tf.function
    def _policy_actions_messages(
        self,
        target_obs_trans: Dict[str, np.ndarray],
        target_core_state: Dict[str, np.ndarray],
        target_core_message: Dict[str, np.ndarray],
    ) -> Any:
        actions = {}
        messages = {}

        for agent in self._agents:
            time.time()
            agent_key = self.agent_net_keys[agent]
            target_trans_obs = target_obs_trans[agent]
            # TODO (dries): Why is there an extra tuple
            #  wrapping that needs to be removed?
            agent_core_state = target_core_state[agent][0]
            agent_core_message = target_core_message[agent][0]

            transposed_obs = tf2_utils.batch_to_sequence(target_trans_obs)

            (output_actions, output_messages), _ = snt.static_unroll(
                self._target_policy_networks[agent_key],
                transposed_obs,
                agent_core_state,
                agent_core_message,
            )
            actions[agent] = tf2_utils.batch_to_sequence(output_actions)
            messages[agent] = tf2_utils.batch_to_sequence(output_messages)
        return actions, messages

    def _step(self) -> Dict[str, Dict[str, Any]]:
        # Update the target networks
        self._update_target_networks()

        inputs = next(self._iterator)

        self._forward(inputs)

        self._backward()

        # Log losses per agent
        return self.policy_losses

    # Forward pass that calculates loss.
    def _forward(self, inputs: Any) -> None:
        data = tree.map_structure(
            lambda v: tf.expand_dims(v, axis=0) if len(v.shape) <= 1 else v, inputs.data
        )
        data = tf2_utils.batch_to_sequence(data)

        observations, actions, rewards, discounts, _, extra = data

        core_states = extra["core_states"]

        self.bs = actions[self._agents[0]].shape[1]  # Batch Size
        T = actions[self._agents[0]].shape[0]  # Episode Length

        with tf.GradientTape(persistent=True) as tape:
            policy_losses = {agent_id: tf.zeros(self.bs) for agent_id in self._agents}

            # Unroll episode and store states, messages, action values
            # for policy and target policy
            policy = {}
            target_policy = {}

            policy["messages"] = {
                -1: {
                    agent_id: core_states[agent_id]["message"][0]
                    for agent_id in self._agents
                }
            }
            policy["states"] = {
                -1: {
                    agent_id: core_states[agent_id]["state"][0]
                    for agent_id in self._agents
                }
            }
            policy["actions"] = {}
            policy["channel"] = {}

            target_policy["messages"] = {
                -1: {
                    agent_id: core_states[agent_id]["message"][0]
                    for agent_id in self._agents
                }
            }
            target_policy["states"] = {
                -1: {
                    agent_id: core_states[agent_id]["state"][0]
                    for agent_id in self._agents
                }
            }
            target_policy["actions"] = {}
            target_policy["channel"] = {}

            # For all time-steps
            for t in range(0, T, 1):
                policy["channel"][t - 1] = self._communication_module.process_messages(
                    policy["messages"][t - 1]
                )[self._agents[0]]
                target_policy["channel"][
                    t - 1
                ] = self._communication_module.process_messages(
                    target_policy["messages"][t - 1]
                )[
                    self._agents[0]
                ]

                policy["messages"][t] = {}
                policy["states"][t] = {}
                policy["actions"][t] = {}

                target_policy["messages"][t] = {}
                target_policy["states"][t] = {}
                target_policy["actions"][t] = {}

                # For each agent
                for agent_id in self._agents:
                    agent_type = self.agent_net_keys[agent_id]
                    # Agent input at time t
                    obs_in = observations[agent_id].observation[t]

                    # Policy
                    state_in = policy["states"][t - 1][agent_id]
                    message_in = policy["channel"][t - 1]

                    (action, message), state = self._policy_networks[agent_type](
                        obs_in, state_in, message_in
                    )

                    # Target policy
                    target_state_in = policy["states"][t - 1][agent_id]
                    target_message_in = policy["channel"][t - 1]

                    (
                        target_action,
                        target_message,
                    ), target_state = self._target_policy_networks[agent_type](
                        obs_in, target_state_in, target_message_in
                    )

                    policy["messages"][t][agent_id] = message
                    policy["states"][t][agent_id] = state
                    policy["actions"][t][agent_id] = action

                    target_policy["messages"][t][agent_id] = target_message
                    target_policy["states"][t][agent_id] = target_state
                    target_policy["actions"][t][agent_id] = target_action

            # policy['channel'][t] = self._communication_module.process_messages(
            #     policy['messages'][t]
            # )[self._agents[0]]

            # target_policy['channel'][t] = self._communication_module.process_messages(
            #     target_policy['messages'][t]
            # )[self._agents[0]]

            # Backtrack episode and calculate loss
            # For t=T to 1, -1 do
            for t in range(T - 1, 0, -1):
                # For each agent a do
                for agent_id in self._agents:
                    # agent_type = self.agent_net_keys[agent_id]
                    # All at timestep t - 1
                    action = actions[agent_id][t - 1]
                    message = core_states[agent_id]["message"][t - 1]
                    reward = rewards[agent_id][t - 1]
                    terminal = t == T - 1

                    discount = tf.cast(
                        self._discount, dtype=discounts[agent_id][t, 0].dtype
                    )

                    # y_t_a = r_t
                    y_action = reward
                    y_message = reward

                    # y_t_a = r_t + discount * max_u Q(t)
                    if not terminal:
                        y_action += discount * tf.reduce_max(
                            target_policy["actions"][t][agent_id]
                        )
                        y_message += discount * tf.reduce_max(
                            target_policy["messages"][t][agent_id]
                        )

                    # d_Q_t_a = y_t_a - Q(t-1)
                    td_action = y_action - tf.gather(
                        policy["actions"][t - 1][agent_id], action, batch_dims=1
                    )

                    # d_theta = d_theta + d_Q_t_a ^ 2
                    policy_losses[agent_id] += td_action ** 2

                    # Communication grads
                    td_comm = y_message - tf.gather(
                        policy["messages"][t - 1][agent_id],
                        tf.argmax(message, axis=-1),
                        batch_dims=1,
                    )
                    policy_losses[agent_id] += td_comm ** 2

            # Average over batches
            for key in policy_losses.keys():
                policy_losses[key] = {
                    "policy_loss": tf.reduce_mean(policy_losses[key], axis=0)
                }

        self.policy_losses: Dict[str, Dict[str, tf.Tensor]] = policy_losses
        self.tape = tape

    def _backward(self) -> None:

        # TODO (dries): I still need to figure out the
        #  total_loss thing. Not per agent I see but per batch?

        # Calculate the gradients and update the networks
        policy_losses = self.policy_losses

        tape = self.tape
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]

            # Get trainable variables.
            # policy_variables = (
            #     self._observation_networks[agent_key].trainable_variables
            #     + self._policy_networks[agent_key].trainable_variables
            # )
            policy_variables = self._policy_networks[agent_key].trainable_variables

            # Compute gradients.
            # Note: Warning "WARNING:tensorflow:Calling GradientTape.gradient
            #  on a persistent tape inside its context is significantly less efficient
            #  than calling it outside the context." caused by losses.dpg, which calls
            #  tape.gradient.
            policy_gradients = tape.gradient(policy_losses[agent], policy_variables)

            # Maybe clip gradients.
            if self._clipping:
                policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.0)[0]

            # Apply gradients.
            self._policy_optimizer.apply(policy_gradients, policy_variables)
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
