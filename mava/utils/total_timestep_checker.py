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
    if config.arch.total_timesteps is None:
        config.arch.total_timesteps = (
            n_devices
            * config.arch.num_updates
            * config.arch.rollout_length
            * config.system.update_batch_size
            * config.arch.num_envs
        )
    else:
        config.arch.num_updates = (
            config.arch.total_timesteps
            // config.arch.rollout_length
            // config.system.update_batch_size
            // config.arch.num_envs
            // n_devices
        )
        print(
            f"{Fore.RED}{Style.BRIGHT} Changing the number of updates "
            + f"to {config.arch.num_updates}: If you want to train"
            + " for a specific number of updates, please set total_timesteps to None!"
            + f"{Style.RESET_ALL}"
        )
    return config
