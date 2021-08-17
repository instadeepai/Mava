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

"""MADQN system trainer implementation."""
import os
import copy
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree
import trfl
from acme.tf import losses
from acme.tf import utils as tf2_utils
from acme.tf.losses import transformed_n_step_loss
from acme.types import NestedArray
from acme.utils import counting, loggers

import mava
from mava import types as mava_types
from mava.adders import reverb as reverb_adders
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.components.tf.modules.stabilising.fingerprints import FingerPrintStabalisation
from mava.systems.tf import savers as tf2_savers
from mava.utils import training_utils as train_utils

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
        q_networks: Dict[str, snt.Module],
        target_q_networks: Dict[str, snt.Module],
        target_update_period: int,
        dataset: tf.data.Dataset,
        optimizer: Union[Dict[str, snt.Optimizer], snt.Optimizer],
        discount: float,
        agent_net_keys: Dict[str, str],
        exploration_scheduler: LinearExplorationScheduler,
        observation_networks: Optional[Dict[str, snt.Module]] = None,
        target_observation_networks: Optional[Dict[str, snt.Module]] = None,
        max_gradient_norm: float = None,
        importance_sampling_exponent: Optional[float] = None,
        replay_client: Optional[reverb.TFClient] = None,
        n_step: int = 1,
        huber_loss_parameter: float = 1.0,
        fingerprint_module: FingerPrintStabalisation = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
    ):
        """Initialise MADQN trainer

        Args:
            agents (List[str]): agent ids, e.g. "agent_0".
            agent_types (List[str]): agent types, e.g. "speaker" or "listener".
            q_networks (Dict[str, snt.Module]): q-value networks.
            target_q_networks (Dict[str, snt.Module]): target q-value networks.
            observation_networks(Dict[str, snt.Module]): observation networks.
            target_observation_networks(Dict[str, snt.Module]): target observation networks.
            target_update_period (int): number of steps before updating target networks.
            dataset (tf.data.Dataset): training dataset.
            optimizer (Union[snt.Optimizer, Dict[str, snt.Optimizer]]): type of
                optimizer for updating the parameters of the networks.
            discount (float): discount factor for TD updates.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            exploration_scheduler (LinearExplorationScheduler): function specifying a
                decaying scheduler for epsilon exploration.
            max_gradient_norm (float, optional): maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            fingerprint (bool, optional): whether to apply replay stabilisation using
                policy fingerprints. Defaults to False.
            n_step (int): For computing n-step returns.
            counter (counting.Counter, optional): step counter object. Defaults to None.
            huber_loss_parameter (float): parameter used for calculating the huber loss.
            importance_sampling_exponent (float, optional): exponent used for
                prioritized experience replay. None for no prioritized experience
                replay.
            max_priority_weight (float): weight used in prioritised experience
                replay. Only required if importance_sampling_exponent is not None.
            replay_client (reverb.TFClient, optional): The reverb replay client.
                Used to mutate priorities when using prioritised experience replay.
            replay_table_name (str): Name of the replay table on the reverb server.
            logger (loggers.Logger, optional): logger object for logging trainer
                statistics. Defaults to None.
            checkpoint (bool, optional): whether to checkpoint networks. Defaults to
                True.
            checkpoint_subpath (str, optional): subdirectory for storing checkpoints.
                Defaults to "~/mava/".
            communication_module (BaseCommunicationModule): module for communication
                between agents. Defaults to None.
        """

        self._agents = agents
        self._agent_types = agent_types
        self._agent_net_keys = agent_net_keys
        self._checkpoint = checkpoint

        # Store online and target q-networks.
        self._q_networks = q_networks
        self._target_q_networks = target_q_networks

        # TODO: remove this once observation networks are fully supported
        # in madqn system
        if observation_networks is None:
            self._observation_networks = {}
            self._target_observation_networks = {}
            for key in q_networks.keys():
                self._observation_networks[key] = tf2_utils.to_sonnet_module(
                    tf.identity
                )
                self._target_observation_networks[key] = tf2_utils.to_sonnet_module(
                    tf.identity
                )
        else:
            self._observation_networks = observation_networks
            self._target_observation_networks = target_observation_networks

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger

        # Other learner parameters.
        self._discount = discount
        self._n_step = n_step
        self._huber_loss_parameter = huber_loss_parameter

        # Set up gradient clipping.
        if max_gradient_norm is not None:
            self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)
        else:  # A very large number. Infinity results in NaNs.
            self._max_gradient_norm = tf.convert_to_tensor(1e10)

        self._fingerprint_module = fingerprint_module

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_update_period = target_update_period

        # Create an iterator to go through the dataset.
        self._iterator = iter(dataset)

        # Store the exploration scheduler
        self._exploration_scheduler = exploration_scheduler

        # Importance sampling hyper-parameters
        self._importance_sampling_exponent = importance_sampling_exponent

        # Replay client for updating priorities.
        self._replay_client = replay_client
        self._replay_table_name = replay_table_name

        # NOTE We make replay_client optional to make changes to MADQN trainer
        # compatible with the other systems that inherit from it (VDN, QMIX etc.)
        # TODO Include importance sampling in the other systems so that we can remove
        # this check.
        if self._importance_sampling_exponent is not None:
            assert isinstance(self._replay_client, reverb.Client)

        # Dictionary with network keys for each agent.
        self.unique_net_keys = set(self._agent_net_keys.values())

        # Create optimizers for different agent types.
        if not isinstance(optimizer, dict):
            self._optimizers: Dict[str, snt.Optimizer] = {}
            for agent in self.unique_net_keys:
                self._optimizers[agent] = copy.deepcopy(optimizer)
        else:
            self._optimizers = optimizer

        # Expose the variables.
        # TODO expose observation networks
        q_networks_to_expose = {}
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "q_network": {},
        }
        for agent_key in self.unique_net_keys:
            q_network_to_expose = self._target_q_networks[agent_key]

            q_networks_to_expose[agent_key] = q_network_to_expose

            self._system_network_variables["q_network"][
                agent_key
            ] = q_network_to_expose.variables

        # Checkpointer
        self._system_checkpointer = {}
        if checkpoint:
            for agent_key in self.unique_net_keys:

                subdir = os.path.join("trainer", agent_key)
                checkpointer = tf2_savers.Checkpointer(
                    directory=checkpoint_subpath,
                    time_delta_minutes=15,
                    objects_to_save={
                        "counter": self._counter,
                        "q_network": self._q_networks[agent_key],
                        "target_q_network": self._target_q_networks[agent_key],
                        "optimizer": self._optimizers,
                        "num_steps": self._num_steps,
                    },
                    enable_checkpointing=checkpoint,
                    subdirectory=subdir
                )

                self._system_checkpointer[agent_key] = checkpointer

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.

        self._timestamp: Optional[float] = None

    def get_epsilon(self) -> float:
        """Get the current value for the exploration parameter epsilon.

        Returns:
            float: epsilon parameter value
        """

        return self._exploration_scheduler.get_epsilon()

    def get_trainer_steps(self) -> float:
        """Get trainer step count

        Returns:
            float: number of trainer steps
        """

        return self._num_steps.numpy()

    def _decrement_epsilon(self) -> None:
        """Decay epsilon exploration value."""

        self._exploration_scheduler.decrement_epsilon()

    def _update_target_networks(self) -> None:
        """Sync target parameters with the latest online parameters."""

        for key in self.unique_net_keys:
            # TODO update observation networks

            # Get variables.
            online_variables = (*self._q_networks[key].variables,)

            target_variables = (*self._target_q_networks[key].variables,)

            # Make online -> target network update ops.
            if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(src)
        self._num_steps.assign_add(1)

    def _update_sample_priorities(self, keys: tf.Tensor, priorities: tf.Tensor) -> None:
        """Update sample priorities in replay table using importance weights.

        Args:
            keys (tf.Tensor): Keys of the replay samples.
            priorities (tf.Tensor): New priorities for replay samples.
        """
        # Maybe update the sample priorities in the replay buffer.
        if (
            self._importance_sampling_exponent is not None
            and self._replay_client is not None
        ):
            self._replay_client.mutate_priorities(
                table=self._replay_table_name,
                updates=dict(zip(keys.numpy(), priorities.numpy())),
            )

    def _get_feed(
        self,
        agent: str,
        o_tm1: Dict[str, np.ndarray],
        a_tm1: Dict[str, np.ndarray],
        r_t: Dict[str, np.ndarray],
        o_t: Dict[str, np.ndarray],
        d_t: Dict[str, np.ndarray],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get data to feed to the agent networks.

        Args:
            o_tm1 (Dict[str, np.ndarray]): observation at timestep t-1.
            a_tm1 (Dict[str, np.ndarray]): action at timestep t-1.
            r_t (Dict[str, np.ndarray]): reward at timestep t+n.
            o_t (Dict[str, np.ndarray]): observation at timestep t+n.
            d_t (Dict[str, np.ndarray]): environment discount at timestep t+n.
            agent (str): agent id

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: agent
            network feeds.
        """
        # Decentralised i.e. independent Q-learners
        o_tm1_feed = o_tm1[agent].observation
        a_tm1_feed = a_tm1[agent]
        r_t_feed = r_t[agent]
        o_t_feed = o_t[agent].observation
        d_t_feed = d_t[agent]

        return o_tm1_feed, a_tm1_feed, r_t_feed, o_t_feed, d_t_feed

    def _transform_observation(
        self, agent_key: str, o_tm1_feed: tf.Tensor, o_t_feed: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Transform the agents observation with its observation network.

        Args:
            agent_key (str): the key for the agent's observation network.
            o_tm1_feed (tf.Tensor): the agent's observation feed.
            o_t_feed (tf.Tensor): the agent's next observation feed.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Returns a tuple with the transformed
                observation and next observation.
        """
        o_tm1_trans = self._observation_networks[agent_key](o_tm1_feed)
        o_t_trans = self._target_observation_networks[agent_key](o_t_feed)

        return o_tm1_trans, o_t_trans

    def step(self) -> None:
        """Trainer step to update the parameters of the agents in the system."""

        # Run the learning step.
        fetches = self._step()

        # Add epsilon to fetches.
        fetches["epsilon"] = self.get_epsilon()

        # Decrement epsilon
        self._decrement_epsilon()

        # Maybe update priorities.
        # NOTE _update_sample_priorities must happen outside of
        # tf.function.
        if self._importance_sampling_exponent is not None:
            # Get the keys and priorities.
            keys = fetches.pop("keys")
            priorities = fetches.pop("priorities")
            # Update the sample priorities in reverb.
            self._update_sample_priorities(keys, priorities)

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

    @tf.function
    def _step(self) -> Dict:
        """Trainer forward and backward passes."""
        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)

        self._forward(inputs)

        self._backward()

        # Update the target networks
        self._update_target_networks()

        # Report loss & statistics for logging.
        fetches = {
            "loss": self._q_network_losses,
        }
        if self._importance_sampling_exponent is not None:
            fetches.update(
                {
                    "keys": self._keys,
                    "priorities": self._priorities,
                }
            )

        return fetches

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass

        Args:
            inputs (Any): input data from the data table (transitions)
        """

        # Get info about the samples from reverb.
        keys, probs = inputs.info[:2]

        # Initialize sample priorities at zero.
        # We will add priorities from each agent
        # to this and then average over all agents.
        priorities = tf.zeros_like(inputs.info.priority, dtype="float32")

        # Unpack input data as follows:
        # o_tm1 = dictionary of observations one for each agent
        # a_tm1 = dictionary of actions taken from obs in o_tm1
        # r_t = dictionary of rewards or rewards sequences
        #   (if using N step transitions) ensuing from actions a_tm1
        # d_t = environment discount ensuing from actions a_tm1.
        #   This discount is applied to future rewards after r_t.
        # o_t = dictionary of next observations or next observation sequences
        # e_t [Optional] = extra data that the agents persist in replay.
        trans = mava_types.Transition(*inputs.data)

        o_tm1, o_t, a_tm1, r_t, d_t, e_tm1, e_t = (
            trans.observation,
            trans.next_observation,
            trans.action,
            trans.reward,
            trans.discount,
            trans.extras,
            trans.next_extras,
        )

        with tf.GradientTape(persistent=True) as tape:
            q_network_losses: Dict[str, NestedArray] = {}

            for agent in self._agents:
                agent_key = self._agent_net_keys[agent]

                # Maybe transform the observation before feeding into policy and critic.
                # Transforming the observations this way at the start of the learning
                # step effectively means that the policy and critic share observation
                # network weights.

                o_tm1_feed, a_tm1_feed, r_t_feed, o_t_feed, d_t_feed = self._get_feed(
                    agent, o_tm1, a_tm1, r_t, o_t, d_t
                )

                # Cast the additional discount to match the environment discount dtype.
                discount = tf.cast(self._discount, dtype=d_t_feed.dtype)

                # Transform raw observation
                o_tm1_trans, o_t_trans = self._transform_observation(
                    agent_key, o_tm1_feed, o_t_feed
                )

                # Maybe apply fingerprints.
                if self._fingerprint_module is not None:
                    o_tm1_trans, o_t_trans = self._fingerprint_module.trainer_hook(
                        o_tm1_trans, o_t_trans, e_tm1, e_t
                    )

                # Compute q-values
                q_tm1 = self._q_networks[agent_key](o_tm1_trans)
                q_t_value = self._target_q_networks[agent_key](o_t_trans)
                q_t_selector = self._q_networks[agent_key](o_t_trans)

                # Compute the loss like in acme/dqn.
                _, extra = trfl.double_qlearning(
                    q_tm1,
                    a_tm1_feed,
                    r_t_feed,
                    discount * d_t_feed,
                    q_t_value,
                    q_t_selector,
                )

                # Compute the huber loss using td_error.
                loss = losses.huber(extra.td_error, self._huber_loss_parameter)

                # Maybe apply importance weights like in acme/dqn.
                if self._importance_sampling_exponent is not None:
                    importance_weights = 1.0 / probs
                    importance_weights **= self._importance_sampling_exponent
                    importance_weights /= tf.reduce_max(importance_weights)

                    # Reweight loss.
                    loss *= tf.cast(importance_weights, loss.dtype)  # [B]

                    # Compute priorities for reverb.
                    priorities += tf.abs(extra.td_error)

                loss = tf.reduce_mean(loss)  # []

                q_network_losses[agent] = {"q_value_loss": loss}

        # Store losses and tape
        self._q_network_losses = q_network_losses
        self.tape = tape

        # Store sample keys and priorities
        self._keys = keys
        # Averaged over agents.
        self._priorities = priorities / len(self._agents)

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""
        q_network_losses = self._q_network_losses
        tape = self.tape
        for agent in self._agents:
            agent_key = self._agent_net_keys[agent]

            # Get trainable variables
            q_network_variables = self._q_networks[agent_key].trainable_variables

            # Compute gradients
            gradients = tape.gradient(q_network_losses[agent], q_network_variables)

            # Clip gradients.
            gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

            # Apply gradients.
            self._optimizers[agent_key].apply(gradients, q_network_variables)

        train_utils.safe_del(self, "tape")

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Get network variables.

        Args:
            names (Sequence[str]): network names

        Returns:
            Dict[str, Dict[str, np.ndarray]]: network variables
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


class MADQNRecurrentTrainer(MADQNTrainer):
    """Recurrent MADQN trainer.

    This is the trainer component of a MADQN system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        q_networks: Dict[str, snt.Module],
        target_q_networks: Dict[str, snt.Module],
        target_update_period: int,
        dataset: tf.data.Dataset,
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        discount: float,
        agent_net_keys: Dict[str, str],
        exploration_scheduler: LinearExplorationScheduler,
        importance_sampling_exponent: Optional[float] = None,
        replay_client: Optional[reverb.TFClient] = None,
        max_priority_weight: float = 0.9,
        n_step: int = 1,
        max_gradient_norm: float = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        fingerprint: bool = False,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        communication_module: Optional[BaseCommunicationModule] = None,
        replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
    ):
        """Initialise recurrent MADQN trainer

        Args:
            agents (List[str]): agent ids, e.g. "agent_0".
            agent_types (List[str]): agent types, e.g. "speaker" or "listener".
            q_networks (Dict[str, snt.Module]): q-value networks.
            target_q_networks (Dict[str, snt.Module]): target q-value networks.
            target_update_period (int): number of steps before updating target networks.
            dataset (tf.data.Dataset): training dataset.
            optimizer (Union[snt.Optimizer, Dict[str, snt.Optimizer]]): type of
                optimizer for updating the parameters of the networks.
            discount (float): discount factor for TD updates.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            exploration_scheduler (LinearExplorationScheduler): function specifying a
                decaying scheduler for epsilon exploration.
            max_gradient_norm (float, optional): maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            counter (counting.Counter, optional): step counter object. Defaults to None.
            n_step (int): For computing N-step returns.
            importance_sampling_exponent (float, optional): exponent used for
                prioritized experience replay. None for no prioritized experience
                replay.
            max_priority_weight (float): weight used in prioritised experience
                replay. Only required if importance_sampling_exponent is not None.
            replay_client (reverb.TFClient, optional): The reverb replay client.
                Used to mutate priorities when using prioritised experience replay.
            replay_table_name (str): Name of the replay table on the reverb server.
            logger (loggers.Logger, optional): logger object for logging trainer
                statistics. Defaults to None.
            fingerprint (bool, optional): whether to apply replay stabilisation using
                policy fingerprints. Defaults to False.
            checkpoint (bool, optional): whether to checkpoint networks. Defaults to
                True.
            checkpoint_subpath (str, optional): subdirectory for storing checkpoints.
                Defaults to "~/mava/".
            communication_module (BaseCommunicationModule): module for communication
                between agents. Defaults to None.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            q_networks=q_networks,
            target_q_networks=target_q_networks,
            target_update_period=target_update_period,
            dataset=dataset,
            optimizer=optimizer,
            discount=discount,
            agent_net_keys=agent_net_keys,
            exploration_scheduler=exploration_scheduler,
            max_gradient_norm=max_gradient_norm,
            n_step=n_step,
            importance_sampling_exponent=importance_sampling_exponent,
            replay_client=replay_client,
            max_priority_weight=max_priority_weight,
            counter=counter,
            logger=logger,
            fingerprint=fingerprint,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
            replay_table_name=replay_table_name,
        )

    def _forward(self, inputs: Any) -> None:
        """Trainer forward pass

        Args:
            inputs (Any): input data from the data table (transitions)
        """
        # Get info about the samples from reverb.
        sample_info = inputs.info
        sample_keys = tf.transpose(inputs.info.key)
        sample_probs = tf.transpose(sample_info.probability)

        # Initialize sample priorities at zero.
        sample_priorities = np.zeros(len(inputs.info.key))

        data = tree.map_structure(
            lambda v: tf.expand_dims(v, axis=0) if len(v.shape) <= 1 else v, inputs.data
        )
        data = tf2_utils.batch_to_sequence(data)

        observations, actions, rewards, discounts, _, extras = data

        # core_states = extra["core_states"]
        core_states = tree.map_structure(
            lambda s: s[:, 0, :], inputs.data.extras["core_states"]
        )

        # Initial states.
        states = {agent: core_states[agent][0] for agent in self._agents}
        target_states = {agent: core_states[agent][0] for agent in self._agents}

        q_network_losses: Dict[str, NestedArray] = {}
        with tf.GradientTape(persistent=True) as tape:

            # Unroll over time dimension.
            T = list(observations.values())[0].observation.shape[0]  # time dimension
            q_stacks: Dict[str, list] = {agent: [] for agent in self._agents}
            q_target_stacks: Dict[str, list] = {agent: [] for agent in self._agents}
            for t in range(T):

                # Step each agent.
                for agent in self._agents:
                    agent_key = self._agent_net_keys[agent]

                    # Get agent observation at timestep t
                    observation = observations[agent].observation[t]

                    # Target Q-Network
                    q_target, new_target_state = self._target_q_networks[agent_key](
                        observation,
                        target_states[agent],
                    )

                    # Online Q-Network
                    q, new_state = self._q_networks[agent_key](
                        observation,
                        states[agent],
                    )

                    # Update states.
                    states[agent] = new_state
                    target_states[agent] = new_target_state

                    # Add qs to stack.
                    q_stacks[agent].append(q)
                    q_target_stacks[agent].append(q_target)

            # Loop through agents and compute losses.
            for agent in self._agents:
                agent_key = self._agent_net_keys[agent]

                # Get q by stacking
                q = tf.stack(q_stacks[agent], axis=0)
                q_target = tf.stack(q_target_stacks[agent], axis=0)

                # Argmax over online policy for double q_learning
                greedy_actions = tf.argmax(q, axis=-1)
                num_actions = q_target.shape[-1]
                target_policy_probs = tf.one_hot(
                    greedy_actions, depth=num_actions, dtype=q_target.dtype
                )

                # Cast discount type.
                discount = tf.cast(self._discount, dtype=discounts[agent][0].dtype)

                # See Acme/tf/losses/R2D2
                loss, loss_extras = transformed_n_step_loss(
                    q,
                    q_target,
                    actions[agent],
                    rewards[agent][:-1],
                    discount * discounts[agent][:-1],
                    target_policy_probs,
                    self._n_step,
                )

                # Maybe calculate importance weights and use them to scale the loss.
                if self._importance_sampling_exponent is not None:
                    importance_weights = 1.0 / sample_probs  # [B]
                    importance_weights **= self._importance_sampling_exponent
                    importance_weights /= tf.reduce_max(importance_weights)

                    # Reweight loss.
                    loss *= tf.cast(importance_weights, loss.dtype)  # [B]

                    # Update priorities.
                    errors = loss_extras.errors
                    abs_errors = tf.abs(errors)
                    mean_priority = tf.reduce_mean(abs_errors, axis=0)
                    max_priority = tf.reduce_max(abs_errors, axis=0)
                    sample_priorities += (
                        self._max_priority_weight * max_priority
                        + (1 - self._max_priority_weight) * mean_priority
                    )

                loss = tf.reduce_mean(loss)  # []

                q_network_losses[agent] = {"q_value_loss": loss}

        self._q_network_losses = q_network_losses
        self.tape = tape

        # Store sample keys and priorities
        self._sample_keys = sample_keys
        self._sample_priorities = sample_priorities / len(
            self._agents
        )  # averaged over agents.


class MADQNRecurrentCommTrainer(MADQNTrainer):
    """Recurrent MADQN trainer with communication.

    This is the trainer component of a MADQN system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        q_networks: Dict[str, snt.Module],
        target_q_networks: Dict[str, snt.Module],
        target_update_period: int,
        dataset: tf.data.Dataset,
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        discount: float,
        agent_net_keys: Dict[str, str],
        exploration_scheduler: LinearExplorationScheduler,
        communication_module: BaseCommunicationModule,
        max_gradient_norm: float = None,
        fingerprint: bool = False,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
    ):
        """Initialise recurrent MADQN trainer with communication

        Args:
            agents (List[str]): agent ids, e.g. "agent_0".
            agent_types (List[str]): agent types, e.g. "speaker" or "listener".
            q_networks (Dict[str, snt.Module]): q-value networks.
            target_q_networks (Dict[str, snt.Module]): target q-value networks.
            target_update_period (int): number of steps before updating target networks.
            dataset (tf.data.Dataset): training dataset.
            optimizer (Union[snt.Optimizer, Dict[str, snt.Optimizer]]): type of
                optimizer for updating the parameters of the networks.
            discount (float): discount factor for TD updates.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            exploration_scheduler (LinearExplorationScheduler): function specifying a
                decaying scheduler for epsilon exploration.
            communication_module (BaseCommunicationModule): module for communication
                between agents.
            max_gradient_norm (float, optional): maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            fingerprint (bool, optional): whether to apply replay stabilisation using
                policy fingerprints. Defaults to False.
            counter (counting.Counter, optional): step counter object. Defaults to None.
            logger (loggers.Logger, optional): logger object for logging trainer
                statistics. Defaults to None.
            checkpoint (bool, optional): whether to checkpoint networks. Defaults to
                True.
            checkpoint_subpath (str, optional): subdirectory for storing checkpoints.
                Defaults to "~/mava/".
            replay_table_name (str): name of the reverb replay table.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            q_networks=q_networks,
            target_q_networks=target_q_networks,
            target_update_period=target_update_period,
            dataset=dataset,
            optimizer=optimizer,
            discount=discount,
            agent_net_keys=agent_net_keys,
            exploration_scheduler=exploration_scheduler,
            max_gradient_norm=max_gradient_norm,
            fingerprint=fingerprint,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
            replay_table_name=replay_table_name,
        )

        self._communication_module = communication_module

    def _forward(self, inputs: Any) -> None:
        """Trainer forward pass

        Args:
            inputs (Any): input data from the data table (transitions)
        """

        data = tree.map_structure(
            lambda v: tf.expand_dims(v, axis=0) if len(v.shape) <= 1 else v, inputs.data
        )
        data = tf2_utils.batch_to_sequence(data)

        observations, actions, rewards, discounts, _, _ = (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.start_of_episode,
            data.extras,
        )

        # Using extra directly from inputs due to shape.
        core_state = tree.map_structure(
            lambda s: s[:, 0, :], inputs.data.extras["core_states"]
        )
        core_message = tree.map_structure(
            lambda s: s[:, 0, :], inputs.data.extras["core_messages"]
        )

        with tf.GradientTape(persistent=True) as tape:
            q_network_losses: Dict[str, NestedArray] = {
                agent: {"q_value_loss": tf.zeros(())} for agent in self._agents
            }

            T = actions[self._agents[0]].shape[0]

            state = {agent: core_state[agent][0] for agent in self._agents}
            target_state = {agent: core_state[agent][0] for agent in self._agents}

            message = {agent: core_message[agent][0] for agent in self._agents}
            target_message = {agent: core_message[agent][0] for agent in self._agents}

            # _target_q_networks must be 1 step ahead
            target_channel = self._communication_module.process_messages(target_message)
            for agent in self._agents:
                agent_key = self._agent_net_keys[agent]
                (q_targ, m), s = self._target_q_networks[agent_key](
                    observations[agent].observation[0],
                    target_state[agent],
                    target_channel[agent],
                )
                target_state[agent] = s
                target_message[agent] = m

            for t in range(1, T, 1):
                channel = self._communication_module.process_messages(message)
                target_channel = self._communication_module.process_messages(
                    target_message
                )

                for agent in self._agents:
                    agent_key = self._agent_net_keys[agent]

                    # Cast the additional discount
                    # to match the environment discount dtype.

                    discount = tf.cast(self._discount, dtype=discounts[agent][0].dtype)

                    (q_targ, m), s = self._target_q_networks[agent_key](
                        observations[agent].observation[t],
                        target_state[agent],
                        target_channel[agent],
                    )
                    target_state[agent] = s
                    target_message[agent] = m

                    (q, m), s = self._q_networks[agent_key](
                        observations[agent].observation[t - 1],
                        state[agent],
                        channel[agent],
                    )
                    state[agent] = s
                    message[agent] = m

                    loss, _ = trfl.qlearning(
                        q,
                        actions[agent][t - 1],
                        rewards[agent][t - 1],
                        discount * discounts[agent][t],
                        q_targ,
                    )

                    loss = tf.reduce_mean(loss)
                    q_network_losses[agent]["q_value_loss"] += loss

        self._q_network_losses = q_network_losses
        self.tape = tape
