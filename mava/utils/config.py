# Copyright 2022 InstaDeep Ltd. All rights reserved.
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


import jax
from colorama import Fore, Style
from omegaconf import DictConfig


def base_sebulba_checks(config: DictConfig) -> None:
    """Checks that the given config does not have conflicting values."""
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation

    assert config.arch.num_envs % len(config.arch.learner_device_ids) == 0, (
        "Number of environments must be divisible by the number of learner."
        + "The output of each actor is equally split across the learners."
    )


def ppo_sebulba_checks(config: DictConfig) -> None:
    base_sebulba_checks(config)

    num_eval_samples = (
        int(config.arch.num_envs / len(config.arch.learner_device_ids))
        * config.system.rollout_length
    )
    assert num_eval_samples % config.system.num_minibatches == 0, (
        f"Number of training samples per evaluator ({num_eval_samples})"
        + f"must be divisible by num_minibatches ({config.system.num_minibatches})."
    )


def check_total_timesteps(config: DictConfig) -> DictConfig:
    """Check if total_timesteps is set, if not, set it based on the other parameters"""

    if config.arch.architecture_name == "anakin":
        n_devices = len(jax.devices())
        update_batch_size = config.system.update_batch_size
    else:
        n_devices = 1  # We only use a single device's output when updating.
        update_batch_size = 1

    if config.system.total_timesteps is None:
        config.system.num_updates = int(config.system.num_updates)
        config.system.total_timesteps = int(
            n_devices
            * config.system.num_updates
            * config.system.rollout_length
            * update_batch_size
            * config.arch.num_envs
        )
    else:
        config.system.total_timesteps = int(config.system.total_timesteps)
        config.system.num_updates = int(
            config.system.total_timesteps
            // config.system.rollout_length
            // update_batch_size
            // config.arch.num_envs
            // n_devices
        )
        print(
            f"{Fore.RED}{Style.BRIGHT} Changing the number of updates "
            + f"to {config.system.num_updates}: If you want to train"
            + " for a specific number of updates, please set total_timesteps to None!"
            + f"{Style.RESET_ALL}"
        )
    return config
