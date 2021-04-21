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

from acme.utils import loggers
import numpy as np
import mava
from mava.utils.loggers import Logger
from mava.utils.wrapper_utils import RunningStatistics
import tensorflow as tf


class TrainerWrapperBase(mava.Trainer):
    """A base trainer statistic class that wrappers a trainer and logs
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

    def _compute_statistics(self, data: Dict[str, Dict[str, float]]) -> None:
        raise NotImplementedError

    def __getattr__(self, attr: Any) -> Any:
        # Check if current class has attr first
        if hasattr(type(self), attr) is False:
            return self._trainer.__getattribute__(attr)
        return self.__getattribute__(attr)


class TrainerStatisticsBase(TrainerWrapperBase):
    """A base trainer statistic class that wrappers a trainer and logs
    certain statistics.
    """

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
        # NOTE (Arnu): getting type issues with the timestamp
        # not sure why. Look into a fix for this.
        timestamp = time.time()
        if self._timestamp:  # type: ignore
            elapsed_time = timestamp - self._timestamp  # type: ignore
        else:
            elapsed_time = 0
        self._timestamp = timestamp  # type: ignore

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint the networks.
        if len(self._system_checkpointer.keys()) > 0:
            for network_key in self.unique_net_keys:
                checkpointer = self._system_checkpointer[network_key]
                checkpointer.save()

        self._logger.write(fetches)


class DetailedTrainerStatistics(TrainerStatisticsBase):
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


class NetworkStatistics(TrainerWrapperBase):
    def __init__(
        self,
        trainer: mava.Trainer,
        # Log only l2 norm by default.
        gradient_norms: List = [2],
        weight_norms: List = [2],
    ) -> None:
        super().__init__(trainer)

        self._network_loggers: Dict[str, loggers.Logger] = {}
        self.gradient_norms = gradient_norms
        self.weight_norms = weight_norms
        self._create_loggers(self._agents)

    def _create_loggers(self, keys) -> None:
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
            )

    # We are usually concerned with weights and grads of linear and conv layers.
    # We have to currently use the name since it is a tf.var and we can't
    # check layer type once already a tf.var.
    # Log linear and conv weights and not bias units.
    def _should_log(self, name):
        if ("linear" in name.lower() or "conv" in name.lower()) and not (
            "b:" in name.lower()
        ):
            return True
        else:
            return False

    def _apply_norms(self, value, norms_list) -> Dict:
        return_data = {}
        for norm in norms_list:
            return_data[norm] = tf.norm(value, ord=norm).numpy()
        return return_data

    def _log_gradients(self, agent, variables_names, gradients):
        assert len(variables_names) == len(
            gradients
        ), "Variable names and gradients do not match"
        grads = {}
        for index, grad in enumerate(gradients):
            variables_name = variables_names[index]
            if self._should_log(variables_name):
                grads[f"{variables_name}_grad"] = grad
                grads[f"{variables_name}_grad_norm"] = self._apply_norms(
                    grad, self.gradient_norms
                )

        self._network_loggers[agent].write(grads)

    def _log_weights(self, agent, variables):
        weights = {}
        for weight in variables:
            if self._should_log(weight.name):
                weights[f"{weight.name}_weight"] = weight
                weights[f"{weight.name}_weight_norm"] = self._apply_norms(
                    weight, self.weight_norms
                )
        self._network_loggers[agent].write(weights)

    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        # Update the target networks
        self._update_target_networks()

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)

        policy_losses, critic_losses, tape = self._forward_pass(inputs)

        self._calc_gradients_update_network(policy_losses, critic_losses, tape)

    def _calc_gradients_update_network(self, policy_losses, critic_losses, tape):
        # Calculate the gradients and update the networks
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
            # TODO: Address warning. WARNING:tensorflow:Calling GradientTape.gradient
            #  on a persistent tape inside its context is significantly less efficient
            #  than calling it outside the context.
            # Caused by losses.dpg, which calls tape.gradient.
            policy_gradients = tape.gradient(policy_losses[agent], policy_variables)
            critic_gradients = tape.gradient(critic_losses[agent], critic_variables)

            # Maybe clip gradients.
            if self._clipping:
                policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.0)[0]
                critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.0)[0]

            # Apply gradients.
            self._policy_optimizer.apply(policy_gradients, policy_variables)
            self._critic_optimizer.apply(critic_gradients, critic_variables)

            self._log_weights(agent, policy_variables)
            self._log_gradients(
                agent,
                variables_names=[vars.name for vars in policy_variables],
                gradients=policy_gradients,
            )

            self._log_weights(agent, critic_variables)
            self._log_gradients(
                agent,
                variables_names=[vars.name for vars in critic_variables],
                gradients=critic_gradients,
            )

    def step(self) -> None:
        # Run the learning step.
        fetches = self._step()
