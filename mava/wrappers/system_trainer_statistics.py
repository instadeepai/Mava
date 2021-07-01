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

"""Generic environment loop wrapper to track system statistics"""

import time
from typing import Any, Dict, List, Sequence

import numpy as np
import tensorflow as tf
from acme.utils import loggers

import mava
from mava.utils import training_utils as train_utils
from mava.utils.loggers import Logger
from mava.utils.wrapper_utils import RunningStatistics


class TrainerWrapperBase(mava.Trainer):
    """A base trainer stats class that wrappers a trainer and logs
    certain statistics.
    """

    def __init__(
        self,
        trainer: mava.Trainer,
    ) -> None:
        self._trainer = trainer

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        return self._trainer.get_variables(names)

    def _create_loggers(self, keys: List[str]) -> None:
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying trainer."""
        return getattr(self._trainer, name)


class TrainerStatisticsBase(TrainerWrapperBase):
    def __init__(
        self,
        trainer: mava.Trainer,
    ) -> None:
        super().__init__(trainer)
        self._require_loggers = True

    def step(self) -> None:
        # Run the learning step.
        fetches = self._step()

        if self._require_loggers:
            self._create_loggers(list(fetches.keys()))
            self._require_loggers = False

        # compute statistics
        self._compute_statistics(fetches)

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp: float = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        if self._system_checkpointer:
            train_utils.checkpoint_networks(self._system_checkpointer)

        if self._logger:
            self._logger.write(fetches)


class DetailedTrainerStatistics(TrainerStatisticsBase):
    """A trainer class that logs episode stats."""

    def __init__(
        self,
        trainer: mava.Trainer,
        metrics: List[str] = ["policy_loss"],
        summary_stats: List = ["mean", "max", "min", "var", "std"],
    ) -> None:
        super().__init__(trainer)

        self._metrics = metrics
        self._summary_stats = summary_stats

    def _create_loggers(self, keys: List[str]) -> None:

        # get system logger data
        trainer_label = self._logger._label
        base_dir = self._logger._directory
        (
            to_terminal,
            to_csv,
            to_tensorboard,
            time_delta,
            print_fn,
            time_stamp,
        ) = self._logger._logger_info

        self._network_running_statistics: Dict[str, Dict[str, float]] = {}
        self._networks_stats: Dict[str, Dict[str, RunningStatistics]] = {
            key: {} for key in keys
        }
        self._network_loggers: Dict[str, loggers.Logger] = {}

        # statistics dictionary
        for key in keys:
            network_label = trainer_label + "_" + key
            self._network_loggers[key] = Logger(
                label=network_label,
                directory=base_dir,
                to_terminal=to_terminal,
                to_csv=to_csv,
                to_tensorboard=to_tensorboard,
                time_delta=0,
                print_fn=print_fn,
                time_stamp=time_stamp,
            )
            for metric in self._metrics:
                self._networks_stats[key][metric] = RunningStatistics(f"{key}_{metric}")

    def _compute_statistics(self, data: Dict[str, Dict[str, float]]) -> None:
        for network, datum in data.items():
            for key, val in datum.items():
                network_running_statistics: Dict[str, float] = {}
                network_running_statistics[f"{network}_raw_{key}"] = val
                self._networks_stats[network][key].push(val)
                for stat in self._summary_stats:
                    network_running_statistics[
                        f"{network}_{stat}_{key}"
                    ] = self._networks_stats[network][key].__getattribute__(stat)()

                self._network_loggers[network].write(network_running_statistics)


class DetailedTrainerStatisticsWithEpsilon(DetailedTrainerStatistics):
    """Custom DetailedTrainerStatistics class for exposing get_epsilon()"""

    def __init__(
        self,
        trainer: mava.Trainer,
        metrics: List[str] = ["q_value_loss"],
        summary_stats: List = ["mean", "max", "min", "var", "std"],
    ) -> None:
        super().__init__(trainer, metrics, summary_stats)

    def get_epsilon(self) -> float:
        return self._trainer.get_epsilon()  # type: ignore

    def get_trainer_steps(self) -> float:
        return self._trainer.get_trainer_steps()  # type: ignore

    def step(self) -> None:
        # Run the learning step.
        fetches = self._step()

        if self._require_loggers:
            self._create_loggers(list(fetches.keys()))
            self._require_loggers = False

        # compute statistics
        self._compute_statistics(fetches)

        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        if self._system_checkpointer:
            train_utils.checkpoint_networks(self._system_checkpointer)

        fetches["epsilon"] = self.get_epsilon()
        self._trainer._decrement_epsilon()  # type: ignore

        if self._logger:
            self._logger.write(fetches)


# TODO(Kale-ab): Is there a better way to do this?
# Maybe using hooks or callbacks.
class NetworkStatisticsBase(TrainerWrapperBase):
    """
    A base class for logging network statistics.
        gradient_norms: List of norms (see tf.norm.ord) to apply to grads.
        weight_norms: List of norms (see tf.norm.ord) to apply to weights.
        log_interval: Log every [log_interval] learner steps.
    """

    def __init__(
        self,
        trainer: mava.Trainer,
        # Log only l2 norm by default.
        gradient_norms: List,
        weight_norms: List,
        log_interval: int,
        log_weights: bool,
        log_gradients: bool,
    ) -> None:
        super().__init__(trainer)

        self._network_loggers: Dict[str, loggers.Logger] = {}
        self.gradient_norms = gradient_norms
        self.weight_norms = weight_norms
        self._create_loggers(self._agents)
        self.log_interval = log_interval
        self.log_weights = log_weights
        self.log_gradients = log_gradients

        assert (
            self.log_weights or self.log_gradients
        ), "Nothing is selected to be logged."

    def _log_step(self) -> bool:
        return bool(
            self._counter
            and self._counter._counts
            and self._counter._counts.get("steps")
            and self._counter._counts.get("steps") % self.log_interval == 0
        )

    def _create_loggers(self, keys: List[str]) -> None:
        trainer_label = self._logger._label
        base_dir = self._logger._directory
        (
            to_terminal,
            to_csv,
            to_tensorboard,
            time_delta,
            print_fn,
            time_stamp,
        ) = self._logger._logger_info

        trainer_label = f"{trainer_label}_networks_stats"
        for key in keys:
            network_label = trainer_label + "_" + key
            self._network_loggers[key] = Logger(
                label=network_label,
                directory=base_dir,
                to_terminal=False,
                to_csv=False,
                to_tensorboard=to_tensorboard,
                print_fn=print_fn,
                time_stamp=time_stamp,
                time_delta=0,
            )

    # Function determines which weight layers to log.
    # We are usually only concerned with weights and grads of linear and conv layers.
    # TODO(Kale-ab) Can we be more robust.
    # Try getting layer type from policy or critic networks
    def _log_data(self, name: str) -> bool:
        # Log linear and conv weights and not bias units.
        return ("linear" in name.lower() or "conv" in name.lower()) and not (
            "b:" in name.lower()
        )

    def _apply_norms(self, value: tf.Tensor, norms_list: List) -> Dict:
        return {norm: tf.norm(value, ord=norm).numpy() for norm in norms_list}

    def _log_gradients(
        self, label: str, agent: str, variables_names: List, gradients: List
    ) -> None:
        assert len(variables_names) == len(
            gradients
        ), "Variable names and gradients do not match"
        grads_dict = {}
        # Log Grads per layer
        for index, grad in enumerate(gradients):
            variables_name = variables_names[index]
            if self._log_data(variables_name):
                grads_dict[f"{label}/grad/{variables_name}"] = grad
                grads_dict[f"{label}/gradnorm/{variables_name}"] = self._apply_norms(
                    grad, self.gradient_norms
                )

        # Log whole network grads
        all_grads_flat = tf.concat([tf.reshape(grad, -1) for grad in gradients], axis=0)
        grads_dict[f"{label}/grad/wholenetwork"] = all_grads_flat
        grads_dict[f"{label}/gradnorm/wholenetwork"] = self._apply_norms(
            all_grads_flat, self.gradient_norms
        )

        self._network_loggers[agent].write(grads_dict)

    def _log_weights(self, label: str, agent: str, weights: List) -> None:
        weights_dict = {}

        # Log Weights per layer
        for weight in weights:
            if self._log_data(weight.name):
                weights_dict[f"{label}/weight/{weight.name}"] = weight
                weights_dict[f"{label}/weightnorm/{weight.name}"] = self._apply_norms(
                    weight, self.weight_norms
                )

        # Log whole network weights
        all_weights_flat = tf.concat(
            [tf.reshape(weight, -1) for weight in weights], axis=0
        )
        weights_dict[f"{label}/weight/wholenetwork"] = all_weights_flat
        weights_dict[f"{label}/weightnorm/wholenetwork"] = self._apply_norms(
            all_weights_flat, self.weight_norms
        )

        self._network_loggers[agent].write(weights_dict)

    def step(self) -> None:
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp: float = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        if self._system_checkpointer:
            train_utils.checkpoint_networks(self._system_checkpointer)

        if self._logger:
            self._logger.write(fetches)


class NetworkStatistics(NetworkStatisticsBase):
    """
    A class for logging network statistics.
    This class assumes the trainer has the following:
        _forward: Forward pass. Stores a policy loss and tf.GradientTape.
        _backward: Updates network using policy loss and tf.GradientTape.
    """

    def __init__(
        self,
        trainer: mava.Trainer,
        # Log only l2 norm by default.
        gradient_norms: List = [2],
        weight_norms: List = [2],
        log_interval: int = 100,
        log_weights: bool = True,
        log_gradients: bool = True,
    ) -> None:
        super().__init__(
            trainer,
            gradient_norms,
            weight_norms,
            log_interval,
            log_weights,
            log_gradients,
        )

    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        # Update the target networks
        # Trying not to assume off policy.
        if hasattr(self, "_update_target_networks"):
            self._update_target_networks()

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)

        self._forward(inputs)

        self._backward()

        # Log losses per agent
        return self.policy_losses

    def _backward(self) -> None:
        policy_losses = self.policy_losses
        tape = self.tape
        log_current_timestep = self._log_step()

        # Calculate the gradients and update the networks
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]

            # Get trainable variables.
            policy_variables = (
                self._observation_networks[agent_key].trainable_variables
                + self._policy_networks[agent_key].trainable_variables
            )

            # Compute gradients.
            policy_gradients = tape.gradient(policy_losses[agent], policy_variables)

            # Maybe clip gradients.
            if self._clipping:
                policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.0)[0]

            # Apply gradients.
            self._policy_optimizers[agent_key].apply(policy_gradients, policy_variables)

            if log_current_timestep:
                if self.log_weights:
                    self._log_weights(
                        label="Policy", agent=agent, weights=policy_variables
                    )
                if self.log_gradients:
                    self._log_gradients(
                        label="Policy",
                        agent=agent,
                        variables_names=[vars.name for vars in policy_variables],
                        gradients=policy_gradients,
                    )
        train_utils.safe_del(self, "tape")


class NetworkStatisticsMixing(NetworkStatisticsBase):
    """
    A class for logging network statistics.
    This class assumes the trainer has the following:
        _forward: Forward pass. Stores a policy loss and tf.GradientTape.
        _backward: Updates network using policy loss and tf.GradientTape.
    """

    def __init__(
        self,
        trainer: mava.Trainer,
        # Log only l2 norm by default.
        gradient_norms: List = [2],
        weight_norms: List = [2],
        log_interval: int = 100,
        log_weights: bool = True,
        log_gradients: bool = True,
    ) -> None:
        super().__init__(
            trainer,
            gradient_norms,
            weight_norms,
            log_interval,
            log_weights,
            log_gradients,
        )

    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        # Update the target networks
        # Trying not to assume off policy.
        if hasattr(self, "_update_target_networks"):
            self._update_target_networks()

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)

        self._forward(inputs)

        self._backward()

        # Log losses per agent
        return {agent: {"q_value_loss": self.loss} for agent in self._agents}

    def _backward(self) -> None:
        log_current_timestep = self._log_step()
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]

            # Update agent networks
            variables = [*self._q_networks[agent_key].trainable_variables]
            gradients = self.tape.gradient(self.loss, variables)
            gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
            self._optimizers[agent_key].apply(gradients, variables)

            if log_current_timestep:
                if self.log_weights:
                    self._log_weights(label="Policy", agent=agent, weights=variables)
                if self.log_gradients:
                    self._log_gradients(
                        label="Policy",
                        agent=agent,
                        variables_names=[vars.name for vars in variables],
                        gradients=gradients,
                    )

        # Update mixing network
        variables = self.get_mixing_trainable_vars()
        gradients = self.tape.gradient(self.loss, variables)

        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
        self._optimizer.apply(gradients, variables)

        if log_current_timestep:
            if self.log_weights:
                self._log_weights(label="Mixing", agent=agent, weights=variables)
            if self.log_gradients:
                self._log_gradients(
                    label="Mixing",
                    agent=agent,
                    variables_names=[vars.name for vars in variables],
                    gradients=gradients,
                )

        train_utils.safe_del(self, "tape")


class NetworkStatisticsActorCritic(NetworkStatisticsBase):
    """
    A class for logging network statistics.
    This class assumes the trainer has the following:
        _forward: Forward pass. Stores a policy loss, critic loss and tf.GradientTape.
        _backward: Updates network using policy loss, critic loss and tf.GradientTape.
    """

    def __init__(
        self,
        trainer: mava.Trainer,
        # Log only l2 norm by default.
        gradient_norms: List = [2],
        weight_norms: List = [2],
        log_interval: int = 100,
        log_weights: bool = True,
        log_gradients: bool = True,
    ) -> None:
        super().__init__(
            trainer,
            gradient_norms,
            weight_norms,
            log_interval,
            log_weights,
            log_gradients,
        )

    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        # Update the target networks
        # Trying not to assume off policy.
        if hasattr(self, "_update_target_networks"):
            self._update_target_networks()

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)

        self._forward(inputs)

        self._backward()

        # Log losses per agent
        return train_utils.map_losses_per_agent_ac(
            self.critic_losses, self.policy_losses
        )

    def _backward(self) -> None:
        # Calculate the gradients and update the networks
        policy_losses = self.policy_losses
        critic_losses = self.critic_losses
        tape = self.tape
        log_current_timestep = self._log_step()

        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]

            # Get trainable variables.
            policy_variables = (
                self._observation_networks[agent_key].trainable_variables
                + self._policy_networks[agent_key].trainable_variables
            )
            critic_variables = (
                # In this agent, the critic loss trains the observation network.
                self._observation_networks[agent_key].trainable_variables
                + self._critic_networks[agent_key].trainable_variables
            )

            # Compute gradients.
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

            if log_current_timestep:
                if self.log_weights:
                    self._log_weights(
                        label="Policy", agent=agent, weights=policy_variables
                    )
                    self._log_weights(
                        label="Critic", agent=agent, weights=critic_variables
                    )

                if self.log_gradients:
                    self._log_gradients(
                        label="Policy",
                        agent=agent,
                        variables_names=[vars.name for vars in policy_variables],
                        gradients=policy_gradients,
                    )
                    self._log_gradients(
                        label="Critic",
                        agent=agent,
                        variables_names=[vars.name for vars in critic_variables],
                        gradients=critic_gradients,
                    )

        train_utils.safe_del(self, "tape")
