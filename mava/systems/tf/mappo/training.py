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
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava
from mava.systems.tf.variable_utils import VariableClient
from mava.types import OLT, NestedArray
from mava.utils import training_utils as train_utils
from mava.utils.sort_utils import sort_str_num

tfd = tfp.distributions
train_utils.set_growing_gpu_memory()


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
        counts: Dict[str, Any],
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Optional[Union[snt.Optimizer, Dict[str, snt.Optimizer]]],
        variable_client: VariableClient,
        agent_net_keys: Dict[str, str],
        minibatch_size: Optional[int] = None,
        num_epochs: int = 10,
        discount: float = 0.999,
        lambda_gae: float = 0.95,
        entropy_cost: float = 0.01,
        baseline_cost: float = 0.5,
        clipping_epsilon: float = 0.2,
        max_gradient_norm: Optional[float] = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
        normalize_advantage: bool = False,
    ):
        """Initialise MAPPO trainer

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            observation_networks: observation networks
                for each agent in the system.
            policy_networks: policy networks for each
                agent in the system.
            critic_networks: critic network(s), shared
                or for each agent in the system.
            dataset (tf.data.Dataset): training dataset.
            policy_optimizer: optimizer
                for updating policy networks.
            critic_optimizer: optimizer
                for updating critic networks. This is not used if using
                single optim.
            agent_net_keys: specifies what network each agent uses.
                Defaults to {}.
            checkpoint_minute_interval: The number of minutes to wait between
                checkpoints.
            minibatch_size: size of minibatch that is sampled from
                the training batch. Minibatches are used for each gradient step.
            num_epochs: number of epochs for every training step.
                Recommendation as per https://arxiv.org/pdf/2103.01955.pdf, "15
                epochs for easy tasks,and 10 or 5 epochs for difficult tasks."
            discount: discount factor for TD updates. Defaults
                to 0.99.
            lambda_gae: scalar determining the mix of bootstrapping
                vs further accumulation of multi-step returns at each timestep.
                Defaults to 1.0.
            entropy_cost: contribution of entropy regularization to
                the total loss. Defaults to 0.0.
            baseline_cost: contribution of the value loss to the
                total loss. Defaults to 1.0.
            clipping_epsilon: Hyper-parameter for clipping in the
                policy objective. Defaults to 0.2.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            counter: step counter object. Defaults to None.
            logger: logger object for logging trainer
                statistics. Defaults to None.
            checkpoint: whether to checkpoint networks. Defaults to
                True.
            checkpoint_subpath: subdirectory for storing checkpoints.
                Defaults to "~/mava/".
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
            normalize_advantage: whether to normalize the advantage.
        """

        # Store agents.
        self._agents = agents

        # Store agent_net_keys.
        self._agent_net_keys = agent_net_keys

        # Setup the variable client
        self._variable_client = variable_client

        # Setup learning rate scheduler_fn
        self._learning_rate_scheduler_fn = learning_rate_scheduler_fn

        # Setup counts
        self._counts = counts

        # Store networks.
        self._observation_networks = observation_networks
        self._policy_networks = policy_networks
        self._critic_networks = critic_networks

        self.unique_net_keys = sort_str_num(policy_networks.keys())
        self._normalize_advantage = normalize_advantage

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
        self._minibatch_size = minibatch_size
        self._num_epochs = num_epochs
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
            observations_trans: transformed (e.g. using
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
        """Apply the observation networks to the raw observations from the dataset

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

    @tf.function
    def _minibatch_update(self, minibatch_data: Any) -> Dict:
        """Minibatch step.

        Args:
            minibatch_data : minibatch of data.

        Returns:
            loss per agent for minibatch.
        """
        return self.forward_backward(minibatch_data)

    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """PPO Trainer step

        Returns:
            Dict[str, Dict[str, Any]]: losses
        """

        losses: Dict[str, NestedArray] = {
            agent: {"critic_loss": tf.zeros(()), "policy_loss": tf.zeros(())}
            for agent in self._agents
        }
        # Get data from replay.
        inputs = next(self._iterator)
        batch_size = inputs.data.observations[self._agents[0]].observation.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices(inputs.data)
        for _ in range(self._num_epochs):
            # Split for possible minibatches
            dataset = dataset.shuffle(batch_size)
            minibatch_dataset = dataset.batch(self._minibatch_size)
            for minibatch_data in minibatch_dataset:
                loss = self._minibatch_update(minibatch_data)

                # Logging sum of losses
                for agent in self._agents:
                    losses[agent] = {
                        "critic_loss": losses[agent]["critic_loss"]
                        + loss[agent]["critic_loss"],
                        "policy_loss": losses[agent]["policy_loss"]
                        + loss[agent]["policy_loss"],
                    }

        # Log losses per agent
        return losses

    def forward_backward(self, inputs: Any) -> Dict[str, Dict[str, Any]]:
        """Do a single forward and backward pass

        Args:
            inputs: input data from the data table (transitions)

        Returns:
            Dict[str, Dict[str, Any]]: losses
        """
        self._forward_pass(inputs)
        self._backward_pass()
        # Log losses per agent
        return train_utils.map_losses_per_agent_ac(
            self.critic_losses, self.policy_losses
        )

    # Forward pass that calculates loss.
    def _forward_pass(self, inputs: Any) -> None:
        """Trainer forward pass

        Args:
            inputs: input data from the data table (transitions)
        """
        # Unpack input data as follows:
        data = tf2_utils.batch_to_sequence(inputs)
        observations, actions, rewards, discounts, extras = (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.extras,
        )

        if "core_states" in extras:
            core_states = tree.map_structure(lambda s: s[0], extras["core_states"])

        # Store losses.
        policy_losses: Dict[str, Any] = {}
        critic_losses: Dict[str, Any] = {}
        total_losses: Dict[str, Any] = {}

        with tf.GradientTape(persistent=True) as tape:
            # transform observation using observation networks
            observations_trans = self._transform_observations(observations)
            for agent in self._agents:
                action, reward, termination, behaviour_log_prob, actor_observation = (
                    actions[agent]["actions"],
                    rewards[agent],
                    discounts[agent],
                    actions[agent]["log_probs"],
                    observations_trans[agent],
                )

                loss_mask = tf.concat(
                    (tf.ones((1, termination.shape[1])), termination[:-1]), 0
                )
                critic_observation = self._get_critic_feed(
                    observations_trans, extras, agent
                )

                # Get agent network
                agent_key = self._agent_net_keys[agent]
                policy_network = self._policy_networks[agent_key]
                critic_network = self._critic_networks[agent_key]
                dims = actor_observation.shape[:2]

                # Do policy forward pass.
                policy_entropy = []
                if "core_states" in extras:
                    # Unroll current policy over actor_observation.
                    agent_core_state = core_states[agent][0]

                    # Manual perform unroll
                    action_prob = []
                    for t in range(len(actor_observation)):
                        outputs, agent_core_state = policy_network(
                            actor_observation[t], agent_core_state
                        )
                        action_prob.append(outputs.log_prob(action[t]))
                        policy_entropy.append(outputs.entropy())

                    action_prob = tf.stack(action_prob, axis=0)
                    policy_entropy = tf.stack(policy_entropy, axis=0)
                else:
                    # Reshape inputs.
                    actor_observation = snt.merge_leading_dims(
                        actor_observation, num_dims=2
                    )
                    policy = policy_network(actor_observation)

                    if isinstance(policy, tfp.distributions.Distribution):
                        # Tensorflow probability.
                        policy = tfd.BatchReshape(
                            policy, batch_shape=dims, name="policy"
                        )
                    else:
                        # Custom distribution function.
                        policy.batch_reshape(dims, name="policy")

                    action_prob = policy.log_prob(action)

                    policy_entropy = policy.entropy()

                critic_observation = snt.merge_leading_dims(
                    critic_observation, num_dims=2
                )
                value_pred = critic_network(critic_observation)

                # Compute importance sampling weights: current policy / behavior policy.
                value_pred = tf.reshape(value_pred, dims, name="value")

                # Exclude last step - it was used in bootstraping.
                bootstrap_value = value_pred[-1]
                value_pred = value_pred[:-1]
                reward = reward[:-1]

                pcontinues = termination * self._discount
                # Generalized Advantage Estimation
                advantages = train_utils.generalized_advantage_estimation(
                    values=value_pred,
                    final_value=bootstrap_value,
                    rewards=reward,
                    discounts=pcontinues[:-1],
                    td_lambda=self._lambda_gae,
                    time_major=True,
                )

                if self._normalize_advantage:
                    # Normalize at minibatch level
                    advantages = train_utils._normalize_advantages(
                        advantages, variance_epsilon=1e-8
                    )
                    raise NotImplementedError(
                        "Confirm that this is working."
                        + "It gave zeros when we tried it out."
                    )

                advantages = tf.stop_gradient(advantages)

                # td_lambda_returns
                returns = advantages + value_pred
                returns = tf.stop_gradient(returns)
                unclipped_critic_loss = tf.square(returns - value_pred)

                # TODO Clip values to reduce variablility
                # Need to keep track of old value estimates (either in replay or in
                # training state) and clip them.
                masked_critic_loss = unclipped_critic_loss * loss_mask[:-1]
                critic_loss = tf.reduce_sum(masked_critic_loss) / tf.reduce_sum(
                    loss_mask[:-1]
                )

                critic_loss = critic_loss * self._baseline_cost

                # Compute importance sampling weights: current policy / behavior policy.
                log_rhos = action_prob[:-1] - behaviour_log_prob[:-1]
                rhos = tf.exp(log_rhos)

                clipped_rhos = tf.clip_by_value(
                    rhos,
                    clip_value_min=1 - self._clipping_epsilon,
                    clip_value_max=1 + self._clipping_epsilon,
                )
                clipped_objective = -tf.minimum(
                    rhos * advantages, clipped_rhos * advantages
                )

                masked_policy_grad_loss = clipped_objective * loss_mask[:-1]
                policy_gradient_loss = tf.reduce_sum(
                    masked_policy_grad_loss
                ) / tf.reduce_sum(loss_mask[:-1])

                # Entropy regulariser.
                # Entropy regularization. Only implemented for categorical dist.
                # TODO (dries): Get this entropy term to work with univariate gaussian
                # distributions as well. The clipping needs to be fixed in that case.
                # (SAC paper, Appendix C)
                masked_entropy_loss = policy_entropy[:-1] * loss_mask[:-1]
                entropy_loss = -tf.reduce_sum(masked_entropy_loss) / tf.reduce_sum(
                    loss_mask[:-1]
                )
                entropy_loss = self._entropy_cost * entropy_loss

                # Combine weighted sum of actor & entropy regularization.
                policy_loss = policy_gradient_loss + entropy_loss

                policy_losses[agent] = policy_loss
                critic_losses[agent] = critic_loss
                total_losses[agent] = policy_loss + critic_loss

        self.policy_losses = policy_losses
        self.critic_losses = critic_losses
        self.total_losses = total_losses
        self.tape = tape

    # Backward pass that calculates gradients and updates network.
    def _backward_pass(self) -> None:
        """Trainer backward pass updating network parameters"""

        # Calculate the gradients and update the networks
        policy_losses = self.policy_losses
        critic_losses = self.critic_losses
        tape = self.tape

        for agent in self._agents:
            # Get agent_key.
            agent_key = self._agent_net_keys[agent]

            policy_variables = self._policy_networks[agent_key].trainable_variables
            # Only use critic vars to update the observation network
            # if we have two optims.
            critic_variables = (
                self._critic_networks[agent_key].trainable_variables
                + self._observation_networks[agent_key].trainable_variables
            )

            # Get gradients.
            critic_gradients = tape.gradient(critic_losses[agent], critic_variables)
            # Optionally apply clipping.
            critic_grads = tf.clip_by_global_norm(
                critic_gradients, self._max_gradient_norm
            )[0]
            # Apply gradients.
            self._critic_optimizers[agent_key].apply(critic_grads, critic_variables)

            # Get gradients.
            policy_gradients = tape.gradient(policy_losses[agent], policy_variables)

            # Optionally apply clipping.
            policy_grads = tf.clip_by_global_norm(
                policy_gradients, self._max_gradient_norm
            )[0]

            # Apply gradients.
            self._policy_optimizers[agent_key].apply(policy_grads, policy_variables)

        train_utils.safe_del(self, "tape")

    def step(self) -> None:
        """Trainer step to update the parameters of the agents in the system"""

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
        """Get network variables

        Args:
            names: network names

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
        counts: Dict[str, Any],
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Optional[Union[snt.Optimizer, Dict[str, snt.Optimizer]]],
        variable_client: VariableClient,
        agent_net_keys: Dict[str, str],
        minibatch_size: Optional[int] = None,
        num_epochs: int = 10,
        discount: float = 0.99,
        lambda_gae: float = 0.95,
        entropy_cost: float = 0.01,
        baseline_cost: float = 0.5,
        clipping_epsilon: float = 0.2,
        max_gradient_norm: Optional[float] = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
        normalize_advantage: bool = False,
    ):
        """Centralised MAPPO trainer.

        Args:
            agents :  agent ids, e.g. "agent_0".
            agent_types :  agent types, e.g. "speaker" or "listener".
            observation_networks : observation networks for each agent in the system.
            policy_networks : policy networks for each agent in the system.
            critic_networks : critic network(s), shared or for each agent in the system.
            dataset : training dataset.
            policy_optimizer : optimizer for updating policy networks.
            critic_optimizer : optimizer for updating critic networks. This is not
                necessary if using single optim.
            agent_net_keys : specifies what network each agent uses.
            checkpoint_minute_interval : The number of minutes to wait between
                checkpoints.
            minibatch_size : size of minibatch that is sampled from
                the training batch. Minibatches are used for each gradient step.
            num_epochs : number of epochs for every training step.
                Recommendation as per https://arxiv.org/pdf/2103.01955.pdf, "15
                epochs for easy tasks,and 10 or 5 epochs for difficult tasks."
            discount :  discount factor for TD updates.
            lambda_gae : scalar determining the mix of bootstrapping
                vs further accumulation of multi-step returns at each timestep.
            entropy_cost : contribution of entropy regularization to
                the total loss. Defaults to 0.0.
            baseline_cost : contribution of the value loss to the
                total loss. Defaults to 1.0.
            clipping_epsilon : Hyper-parameter for clipping in the
                policy objective. Defaults to 0.2.
            max_gradient_norm : maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            counter : step counter object. Defaults to None.
            logger : logger object for logging trainer
                statistics. Defaults to None.
            checkpoint : whether to checkpoint networks. Defaults to
                True.
            checkpoint_subpath : subdirectory for storing checkpoints.
                Defaults to "~/mava/".
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
            normalize_advantage: whether to normalize the advantage.
        """

        super().__init__(
            agents=agents,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            observation_networks=observation_networks,
            dataset=dataset,
            counts=counts,
            agent_net_keys=agent_net_keys,
            variable_client=variable_client,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            discount=discount,
            lambda_gae=lambda_gae,
            entropy_cost=entropy_cost,
            baseline_cost=baseline_cost,
            clipping_epsilon=clipping_epsilon,
            max_gradient_norm=max_gradient_norm,
            counter=counter,
            logger=logger,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
            normalize_advantage=normalize_advantage,
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
        counts: Dict[str, Any],
        policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        critic_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        variable_client: VariableClient,
        agent_net_keys: Dict[str, str],
        minibatch_size: Optional[int] = None,
        num_epochs: int = 10,
        discount: float = 0.999,
        lambda_gae: float = 0.95,
        entropy_cost: float = 0.01,
        baseline_cost: float = 0.5,
        clipping_epsilon: float = 0.2,
        max_gradient_norm: Optional[float] = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
        normalize_advantage: bool = False,
    ):

        super().__init__(
            agents=agents,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            observation_networks=observation_networks,
            dataset=dataset,
            counts=counts,
            agent_net_keys=agent_net_keys,
            variable_client=variable_client,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            discount=discount,
            lambda_gae=lambda_gae,
            entropy_cost=entropy_cost,
            baseline_cost=baseline_cost,
            clipping_epsilon=clipping_epsilon,
            max_gradient_norm=max_gradient_norm,
            counter=counter,
            logger=logger,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
            normalize_advantage=normalize_advantage,
        )

    def _get_critic_feed(
        self,
        observations_trans: Dict[str, np.ndarray],
        extras: Dict[str, np.ndarray],
        agent: str,
    ) -> tf.Tensor:
        # State based
        if type(extras["s_t"]) == dict:  # type: ignore
            return extras["s_t"][agent]
        else:
            return extras["s_t"]
