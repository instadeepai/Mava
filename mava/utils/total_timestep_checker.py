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


def check_total_timesteps(config: DictConfig) -> DictConfig:
    """Check if total_timesteps is set, if not, set it based on the other parameters"""
    n_devices = len(jax.devices())

    if config.system.total_timesteps is None:
        config.system.num_updates = int(config.system.num_updates)
        config.system.total_timesteps = int(
            n_devices
            * config.system.num_updates
            * config.system.rollout_length
            * config.system.update_batch_size
            * config.arch.num_envs
        )
    else:
        config.system.total_timesteps = int(config.system.total_timesteps)
        config.system.num_updates = int(
            config.system.total_timesteps
            // config.system.rollout_length
            // config.system.update_batch_size
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


def check_total_ppo_timesteps(config: DictConfig) -> DictConfig:
    """Check if total_timesteps is set, if not, set it based on the other parameters"""
    n_devices = len(jax.devices())

    def calculate_total_timesteps() -> int:
        return int(
            n_devices
            * config.system.num_updates
            * config.system.rollout_length
            * config.system.update_batch_size
            * config.arch.num_envs
        )

    if config.system.total_timesteps is None:
        config.system.num_updates -= config.system.num_updates % config.arch.num_evaluation
        config.system.total_timesteps = calculate_total_timesteps()
        print(
            f"{Fore.RED}{Style.BRIGHT} The number of updates is not "
            "divisible by the number of evaluations. "
            f"Adjusting the number of updates to {config.system.num_updates}.{Style.RESET_ALL}."
        )
    else:
        config.system.total_timesteps = int(config.system.total_timesteps)

        config.system.num_updates = (
            config.system.total_timesteps
            // config.system.rollout_length
            // config.system.update_batch_size
            // config.arch.num_envs
            // n_devices
        )
        config.system.num_updates -= (
            config.system.num_updates % config.arch.num_evaluation
        )  # remove the extra updates

        new_total_timesteps = calculate_total_timesteps()
        if new_total_timesteps != config.system.total_timesteps:
            print(
                f"{Fore.RED}{Style.BRIGHT} The number of updates required to run"
                f" {config.system.total_timesteps} time steps is not "
                "divisible by the number of evaluations. Adjusting the number of"
                f" total timesteps to {new_total_timesteps}{Style.RESET_ALL}."
            )
            config.system.total_timesteps = new_total_timesteps

    return config
