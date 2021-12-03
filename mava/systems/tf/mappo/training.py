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

"""MAPPO system trainer implementation."""

import copy
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import trfl
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava
from mava.adders.reverb.base import Trajectory
from mava.systems.tf.variable_utils import VariableClient
from mava.types import OLT
from mava.utils import training_utils as train_utils
from mava.utils.sort_utils import sort_str_num

train_utils.set_growing_gpu_memory()

tfd = tfp.distributions


class MAPPOTrainer(mava.Trainer):
    """MAPPO trainer.
    This is the trainer component of a MAPPO system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[Any],
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        dataset: tf.data.Dataset,
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        discount: float = 0.999,
        lambda_gae: float = 1.0,  # Question (dries): What is this used for?
        entropy_cost: float = 0.01,
        baseline_cost: float = 0.5,
        clipping_epsilon: float = 0.2,
        max_gradient_norm: Optional[float] = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):
        """Initialise MAPPO trainer

        Args:
            agents (List[str]): agent ids, e.g. "agent_0".
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            critic_networks (Dict[str, snt.Module]): critic network(s), shared or for
                each agent in the system.
            dataset (tf.data.Dataset): training dataset.
            policy_optimizer (Union[snt.Optimizer, Dict[str, snt.Optimizer]]):
                optimizer(s) for updating policy networks.
            critic_optimizer (Union[snt.Optimizer, Dict[str, snt.Optimizer]]):
                optimizer for updating critic networks.
            variable_client: The client used to manage the variables.
            counts: step counter object.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            discount (float, optional): discount factor for TD updates. Defaults
                to 0.99.
            lambda_gae (float, optional): scalar determining the mix of bootstrapping
                vs further accumulation of multi-step returns at each timestep.
                Defaults to 1.0.
            entropy_cost (float, optional): contribution of entropy regularization to
                the total loss. Defaults to 0.0.
            baseline_cost (float, optional): contribution of the value loss to the
                total loss. Defaults to 1.0.
            clipping_epsilon (float, optional): Hyper-parameter for clipping in the
                policy objective. Defaults to 0.2.
            max_gradient_norm (float, optional): maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            counter (counting.Counter, optional): step counter object. Defaults to None.
            logger (loggers.Logger, optional): logger object for logging trainer
                statistics. Defaults to None.
            checkpoint (bool, optional): whether to checkpoint networks. Defaults to
                True.
            checkpoint_subpath (str, optional): subdirectory for storing checkpoints.
                Defaults to "~/mava/".
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
        """

        # Store agents.
        self._agents = agents

        # Store agent_net_keys.
        self._agent_net_keys = agent_net_keys

        # Setup the variable client
        self._variable_client = variable_client

        # Setup counts
        self._counts = counts

        # Setup learning rate scheduler_fn
        self._learning_rate_scheduler_fn = learning_rate_scheduler_fn

        # Store networks.
        self._observation_networks = observation_networks
        self._policy_networks = policy_networks
        self._critic_networks = critic_networks

        self.unique_net_keys = sort_str_num(policy_networks.keys())

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
            "critic": {},
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
            self._system_network_variables["critic"][agent_key] = critic_networks[
                agent_key
            ].variables
            self._system_network_variables["policy"][
                agent_key
            ] = policy_network_to_expose.variables

        # Other trainer parameters.
        self._discount = discount
        self._entropy_cost = entropy_cost
        self._baseline_cost = baseline_cost
        self._lambda_gae = lambda_gae
        self._clipping_epsilon = clipping_epsilon

        # Dataset iterator
        self._iterator = dataset

        # Set up gradient clipping.
        if max_gradient_norm is not None:
            self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)
        else:  # A very large number. Infinity results in NaNs.
            self._max_gradient_norm = tf.convert_to_tensor(1e10)

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("trainer")

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp: Optional[float] = None

    def _get_critic_feed(
        self,
        observations_trans: Dict[str, np.ndarray],
        extras: Dict[str, np.ndarray],
        agent: str,
    ) -> tf.Tensor:
        """Get critic feed.

        Args:
            observations_trans (Dict[str, np.ndarray]): transformed (e.g. using
                observation network) raw agent observation.
            extras: Extra information. E.g. the environment state can be included
            here.
            agent (str): agent id.

        Returns:
            tf.Tensor: agent critic network feed
        """

        # Decentralised based
        observation_feed = observations_trans[agent]

        return observation_feed

    def _transform_observations(
        self, observations: Dict[str, OLT]
    ) -> Dict[str, np.ndarray]:
        """apply the observation networks to the raw observations from the dataset

        Args:
            observations (Dict[str, np.ndarray]): raw agent observations

        Returns:
            Dict[str, np.ndarray]: transformed
                observations (features)
        """

        observation_trans = {}
        for agent in self._agents:
            agent_key = self._agent_net_keys[agent]

            reshaped_obs, dims = train_utils.combine_dim(
                observations[agent].observation
            )

            observation_trans[agent] = train_utils.extract_dim(
                self._observation_networks[agent_key](reshaped_obs), dims
            )
        return observation_trans

    # Warning (dries): Do not place a tf.function here. It breaks the itterator.
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Trainer forward and backward passes.

        Returns:
            Dict[str, Dict[str, Any]]: losses
        """

        # Get data from replay.
        inputs = next(self._iterator)

        # Do a forward and backwards pass using tf.function.
        return self.forward_backward(inputs)

    @tf.function
    def forward_backward(self, inputs: Any) -> Dict[str, Dict[str, Any]]:
        self._forward_pass(inputs)
        self._backward_pass()
        # Log losses per agent
        return train_utils.map_losses_per_agent_ac(
            self.critic_losses, self.policy_losses
        )

    # Forward pass that calculates loss.
    # This name is changed from _backward to make sure the trainer
    # wrappers are not overwriting the _step function.
    def _forward_pass(self, inputs: Any) -> None:
        """Trainer forward pass

        Args:
            inputs (Any): input data from the data table (transitions)
        """

        data: Trajectory = inputs.data

        # Unpack input data as follows:
        data = tf2_utils.batch_to_sequence(inputs.data)
        observations, actions, rewards, discounts, extras = (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.extras,
        )

        if "core_states" in extras:
            core_states = tree.map_structure(lambda s: s[0], extras["core_states"])

        # transform observation using observation networks
        observations_trans = self._transform_observations(observations)

        # Store losses.
        policy_losses: Dict[str, Any] = {}
        critic_losses: Dict[str, Any] = {}

        with tf.GradientTape(persistent=True) as tape:
            for agent in self._agents:
                action, reward, discount, behaviour_logits, actor_observation = (
                    actions[agent]["actions"],
                    rewards[agent],
                    discounts[agent],
                    actions[agent]["logits"],
                    observations_trans[agent],
                )

                critic_observation = self._get_critic_feed(
                    observations_trans, extras, agent
                )

                # Chop off final timestep for bootstrapping value
                action = action[:-1]
                reward = reward[:-1]
                discount = discount[:-1]

                # Get agent network
                agent_key = self._agent_net_keys[agent]
                policy_network = self._policy_networks[agent_key]
                critic_network = self._critic_networks[agent_key]

                if "core_states" in extras:
                    # Unroll current policy over actor_observation.
                    agent_core_state = core_states[agent][0]
                    # Question (dries): Maybe we can save training time by
                    # only unroll the LSTM part and doing a batch forward
                    # pass for the feedforward part of the policy?
                    logits, updated_states = snt.static_unroll(
                        policy_network,
                        actor_observation,
                        agent_core_state,
                    )
                else:
                    # Pass observations through the feedforward policy
                    actor_observation, dims = train_utils.combine_dim(actor_observation)
                    logits = train_utils.extract_dim(
                        policy_network(actor_observation), dims
                    )

                # Compute importance sampling weights: current policy / behavior policy.
                pi_behaviour = tfd.Categorical(logits=behaviour_logits[:-1])
                pi_target = tfd.Categorical(logits=logits[:-1])

                # Calculate critic values.
                dims = critic_observation.shape[:2]
                critic_observation = tf.reshape(
                    critic_observation, (dims[0] * dims[1], -1)
                )
                critic_output = critic_network(critic_observation)
                values = tf.reshape(critic_output, dims)
                # Values along the sequence T.
                bootstrap_value = values[-1]
                state_values = values[:-1]

                # Generalized Return Estimation
                td_loss, td_lambda_extra = trfl.td_lambda(
                    state_values=state_values,
                    rewards=reward,
                    pcontinues=discount,  # Question (dries): Why is self._discount
                    # not used. Why is discount not 0.0/1.0 but actually has
                    # discount values.
                    bootstrap_value=bootstrap_value,
                    lambda_=self._lambda_gae,
                    name="CriticLoss",
                )

                # Do not use the loss provided by td_lambda as they sum the losses over
                # the sequence length rather than averaging them.
                critic_loss = self._baseline_cost * tf.square(
                    td_lambda_extra.temporal_differences
                )

                # Compute importance sampling weights: current policy / behavior policy.
                log_rhos = pi_target.log_prob(action) - pi_behaviour.log_prob(action)
                importance_ratio = tf.exp(log_rhos)
                clipped_importance_ratio = tf.clip_by_value(
                    importance_ratio,
                    1.0 - self._clipping_epsilon,
                    1.0 + self._clipping_epsilon,
                )

                # Generalized Advantage Estimation
                gae = tf.stop_gradient(td_lambda_extra.temporal_differences)

                # Note (dries): Maybe add this in again? But this might be breaking
                # training. mean, variance = tf.nn.moments(gae, axes=[0, 1],
                # keepdims=True)
                # normalized_gae = (gae - mean) / tf.sqrt(variance)

                policy_gradient_loss = -tf.minimum(
                    tf.multiply(importance_ratio, gae),
                    tf.multiply(clipped_importance_ratio, gae),
                    name="PolicyGradientLoss",
                )

                # Entropy regulariser.
                entropy_loss = trfl.policy_entropy_loss(pi_target).loss

                # Combine weighted sum of actor & entropy regularization.
                policy_loss = policy_gradient_loss + entropy_loss

                # Multiply by discounts to not train on padded data.
                loss_mask = discount > 0.0
                # TODO (dries): Is multiplication maybe better here? As assignment
                # might not work with tf.function?
                policy_loss = policy_loss[loss_mask]
                critic_loss = critic_loss[loss_mask]
                policy_losses[agent] = tf.reduce_mean(policy_loss)
                critic_losses[agent] = tf.reduce_mean(critic_loss)

        self.policy_losses = policy_losses
        self.critic_losses = critic_losses
        self.tape = tape

    # Backward pass that calculates gradients and updates network.
    # This name is changed from _backward to make sure the trainer wrappers are not
    # overwriting the _step function.
    def _backward_pass(self) -> None:
        """Trainer backward pass updating network parameters"""

        # Calculate the gradients and update the networks
        policy_losses = self.policy_losses
        critic_losses = self.critic_losses
        tape = self.tape

        for agent in self._agents:
            # Get agent_key.
            agent_key = self._agent_net_keys[agent]

            # Get trainable variables.
            policy_variables = (
                self._observation_networks[agent_key].trainable_variables
                + self._policy_networks[agent_key].trainable_variables
            )
            critic_variables = self._critic_networks[agent_key].trainable_variables

            # Get gradients.
            policy_gradients = tape.gradient(policy_losses[agent], policy_variables)
            critic_gradients = tape.gradient(critic_losses[agent], critic_variables)

            # Optionally apply clipping.
            critic_grads, critic_norm = tf.clip_by_global_norm(
                critic_gradients, self._max_gradient_norm
            )
            policy_grads, policy_norm = tf.clip_by_global_norm(
                policy_gradients, self._max_gradient_norm
            )

            # Apply gradients.
            self._critic_optimizers[agent_key].apply(critic_grads, critic_variables)
            self._policy_optimizers[agent_key].apply(policy_grads, policy_variables)

        train_utils.safe_del(self, "tape")

    def step(self) -> None:
        """trainer step to update the parameters of the agents in the system"""

        # Run the learning step.
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        raise ValueError("This step should not be used. Use a trainer wrapper.")

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
        """get network variables

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

    def after_trainer_step(self) -> None:
        """Optionally decay lr after every training step."""
        if self._learning_rate_scheduler_fn:
            self._decay_lr(self._counter.get_counts().get("trainer_steps", 0))
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


class CentralisedMAPPOTrainer(MAPPOTrainer):
    """MAPPO trainer for a centralised architecture."""

    def __init__(
        self,
        agents: List[Any],
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        dataset: tf.data.Dataset,
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        discount: float = 0.999,
        lambda_gae: float = 1.0,
        entropy_cost: float = 0.01,
        baseline_cost: float = 0.5,
        clipping_epsilon: float = 0.2,
        max_gradient_norm: Optional[float] = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):

        super().__init__(
            agents=agents,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            observation_networks=observation_networks,
            dataset=dataset,
            agent_net_keys=agent_net_keys,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            variable_client=variable_client,
            counts=counts,
            discount=discount,
            lambda_gae=lambda_gae,
            entropy_cost=entropy_cost,
            baseline_cost=baseline_cost,
            clipping_epsilon=clipping_epsilon,
            max_gradient_norm=max_gradient_norm,
            counter=counter,
            logger=logger,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        )

    def _get_critic_feed(
        self,
        observations_trans: Dict[str, np.ndarray],
        extras: Dict[str, np.ndarray],
        agent: str,
    ) -> tf.Tensor:
        # Centralised based

        observation_feed = tf.stack(
            [observations_trans[agent] for agent in self._agents], 2
        )

        return observation_feed


class StateBasedMAPPOTrainer(MAPPOTrainer):
    """MAPPO trainer for a centralised architecture."""

    def __init__(
        self,
        agents: List[Any],
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        dataset: tf.data.Dataset,
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        counts: Dict[str, Any],
        variable_client: VariableClient,
        agent_net_keys: Dict[str, str],
        discount: float = 0.999,
        lambda_gae: float = 1.0,
        entropy_cost: float = 0.01,
        baseline_cost: float = 0.5,
        clipping_epsilon: float = 0.2,
        max_gradient_norm: Optional[float] = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):

        super().__init__(
            agents=agents,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            observation_networks=observation_networks,
            dataset=dataset,
            agent_net_keys=agent_net_keys,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            variable_client=variable_client,
            counts=counts,
            discount=discount,
            lambda_gae=lambda_gae,
            entropy_cost=entropy_cost,
            baseline_cost=baseline_cost,
            clipping_epsilon=clipping_epsilon,
            max_gradient_norm=max_gradient_norm,
            counter=counter,
            logger=logger,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        )

    def _get_critic_feed(
        self,
        observations_trans: Dict[str, np.ndarray],
        extras: Dict[str, np.ndarray],
        agent: str,
    ) -> tf.Tensor:
        # State based
        if type(extras["env_states"]) == dict:
            return extras["env_states"][agent]
        else:
            return extras["env_states"]
