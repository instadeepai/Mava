from typing import Optional

import copy
from sonnet.src.base import Optimizer
from acme.tf.savers import Checkpointer

import tree
import trfl
import tensorflow as tf
import sonnet as snt
from acme.tf import utils as tf2_utils

class FeedForwardTrainer:
    def __init__(
        self,
        agents,
        qnetwork: snt.Module,
        dataset: tf.data.Dataset,
        optimizer: snt.Optimizer,
        tau: float,
        discount: float,
        logger = None,
        checkpoint_dir = None,
        mixer = None
    ):
        self._agents = agents
        self._qnetwork = qnetwork
        self._target_qnetwork = copy.deepcopy(qnetwork)
        self._iterator = iter(dataset)
        self._optimizer = optimizer
        self._tau = tau
        self._discount = discount
        self._logger = logger

        # Expose the variables.
        self._variables = {
            "qnetwork": self._target_qnetwork.variables
        }

        # Setup Checkpointer
        self._checkpointer = Checkpointer(
            objects_to_save={"qnetwork": self._qnetwork}, 
            enable_checkpointing=True, 
            directory=checkpoint_dir,
            subdirectory="qnetwork", 
            add_uid=False, 
            time_delta_minutes=10, 
        ) if checkpoint_dir else None

        # No Mixer for FeedForward Systems
        self._mixer = None

    def get_variables(self, names):
        return[tf2_utils.to_numpy(self._variables[name]) for name in names]

    def _soft_target_update(self):
        online_variables = (*self._qnetwork.variables,)
        target_variables = (*self._target_qnetwork.variables,)

        for src, dest in zip(online_variables, target_variables):
            dest.assign(self._tau * src + (1 - self._tau) * dest)

    def _forward(self, batch):
        data = batch.data

        observations, actions, rewards, next_observations, discounts, extras = (
            data.observations,
            data.actions,
            data.rewards,
            data.next_observations,
            data.discounts,
            data.extras
        )

        with tf.GradientTape() as tape:
            
            q_values_out = []
            target_q_values_out = []
            rewards_out = []
            discounts_out = []
            for agent in self._agents:
                q_values = self._qnetwork(observations[agent].observation)
                chosen_action_qvalues = trfl.batched_index(q_values, actions[agent])

                # Target Q-values
                target_q_values = self._target_qnetwork(next_observations[agent].observation)
                selector_q_values = self._qnetwork(next_observations[agent].observation)

                # Get legal actions at next timestep
                legal_actions = next_observations[agent].legal_actions
                legal_actions = tf.cast(legal_actions, dtype='bool')
                selector_q_values = tf.where(legal_actions, selector_q_values, -999999999)

                # Double Q-learning
                target_action = tf.argmax(selector_q_values, axis=-1)
                target_max_q_values = trfl.batched_index(target_q_values, target_action)

                # Append 
                q_values_out.append(chosen_action_qvalues)
                target_q_values_out.append(target_max_q_values)
                rewards_out.append(rewards[agent])
                discounts_out.append(discounts[agent])

            # Stack
            q_values_out = tf.stack(q_values_out, axis=-1)
            target_q_values_out = tf.stack(target_q_values_out, axis=-1)
            rewards_out = tf.stack(rewards_out, axis=-1)
            discounts_out = tf.stack(discounts_out, axis=-1)
            
            # Calculate 1-step Q-Learning targets
            targets = rewards_out + self._discount * discounts_out * target_q_values_out

            # TD error
            td_error = (q_values_out - tf.stop_gradient(targets))

            # Loss
            loss = tf.reduce_mean(td_error ** 2)

        self.loss = loss
        self.tape = tape


    def _backward(self):
        variables = self._qnetwork.trainable_variables

        if self._mixer is not None:
            variables += self._mixer.trainable_variables

        gradients = self.tape.gradient(self.loss, variables)
        self._optimizer.apply(gradients, variables)

        self._soft_target_update()

    @tf.function
    def _step(self):
        batch = next(self._iterator)

        self._forward(batch)

        self._backward()

        logs = {"loss": self.loss}

        return logs

    def step(self):

        logs = self._step()

        self._logger.write(logs)

        if self._checkpointer:
            self._checkpointer.save()

    def run(self):
        # Infinetly learn
        while True:
            self.step()


class RecurrentTrainer(FeedForwardTrainer):

    def __init__(
        self,
        agents,
        qnetwork: snt.Module,
        dataset: tf.data.Dataset,
        optimizer: snt.Optimizer,
        tau: float,
        discount: float,
        mixer: Optional[snt.Module],
        logger = None,
        checkpoint_dir = None
    ):
        super().__init__(
            agents=agents,
            qnetwork=qnetwork,
            dataset=dataset,
            optimizer=optimizer,
            tau=tau,
            discount=discount,
            mixer=mixer,
            logger=logger,
            checkpoint_dir=checkpoint_dir
        )

    def _forward(self, batch):
        data = tree.map_structure(
            lambda v: tf.expand_dims(v, axis=0) if len(v.shape) <= 1 else v, batch.data
        )
        data = tf2_utils.batch_to_sequence(data)

        observations, actions, rewards, discounts, _, extras = (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.start_of_episode,
            data.extras,
        )

        # Recurrent states
        core_state = tree.map_structure(
            lambda s: s[:, 0, :], batch.data.extras["core_states"]
        )

        with tf.GradientTape() as tape:
            
            q_values_out = []
            target_q_values_out = []
            rewards_out = []
            discounts_out = []
            for agent in self._agents:
                q_values, _ = snt.static_unroll(
                    self._qnetwork,
                    observations[agent].observation,
                    core_state[agent],
                )
                chosen_action_qvalues = trfl.batched_index(q_values, actions[agent])

                target_q_values, _ = snt.static_unroll(
                    self._target_qnetwork,
                    observations[agent].observation,
                    core_state[agent],
                )

                # Get legal actions
                legal_actions = observations[agent].legal_actions
                legal_actions = tf.cast(legal_actions, dtype='bool')

                # Double Q-learning
                selector_q_values = tf.where(legal_actions, tf.identity(q_values), -999999)
                target_action = tf.argmax(selector_q_values, axis=-1)
                target_max_qvalues = trfl.batched_index(target_q_values, target_action)

                # Append 
                q_values_out.append(chosen_action_qvalues)
                target_q_values_out.append(target_max_qvalues)
                rewards_out.append(rewards[agent])
                discounts_out.append(discounts[agent])

            # Stack
            q_values_out = tf.stack(q_values_out, axis=-1)
            target_q_values_out = tf.stack(target_q_values_out, axis=-1)
            rewards_out = tf.stack(rewards_out, axis=-1)
            discounts_out = tf.stack(discounts_out, axis=-1)

            # Discard the timesteps we don't need
            # Cut last timestep
            q_values_out = q_values_out[:-1]
            rewards_out = rewards_out[:-1]
            discounts_out = discounts_out[:-1]
            # Cut first timestep
            target_q_values_out = target_q_values_out[1:]

            # Mix
            if self._mixer is not None:
                rewards_out = tf.reduce_sum(rewards_out, axis=-1)
                discounts_out = tf.reduce_prod(discounts_out, axis=-1)

                q_values_out = self._mixer(q_values_out)
                target_q_values_out = self._mixer(target_q_values_out)

            # Calculate 1-step Q-Learning targets
            targets = rewards_out + self._discount * discounts_out * target_q_values_out

            # TD error
            td_error = (q_values_out - tf.stop_gradient(targets))

            # Normal L2 loss, take mean over actual data
            loss = tf.reduce_mean(td_error ** 2)

        self.loss = loss
        self.tape = tape

