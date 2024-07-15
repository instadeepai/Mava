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

from typing import Callable, Union

from omegaconf import DictConfig


def make_learning_rate_schedule(init_lr: float, config: DictConfig) -> Callable:
    """Makes a very simple linear learning rate scheduler.

    Args:
    ----
        init_lr: initial learning rate.
        config: system configuration.

    Note:
    ----
        We use a simple linear learning rate scheduler based on the suggestions from a blog on PPO
        implementation details which can be viewed at http://tinyurl.com/mr3chs4p
        This function can be extended to have more complex learning rate schedules by adding any
        relevant arguments to the system config and then parsing them accordingly here.

    """

    def linear_scedule(count: int) -> float:
        frac: float = (
            1.0
            - (count // (config.system.ppo_epochs * config.system.num_minibatches))
            / config.system.num_updates
        )
        return init_lr * frac

    return linear_scedule


def make_learning_rate(init_lr: float, config: DictConfig) -> Union[float, Callable]:
    """Retuns a constant learning rate or a learning rate schedule.

    Args:
    ----
        init_lr: initial learning rate.
        config: system configuration.

    Returns:
    -------
        A learning rate schedule or fixed learning rate.

    """
    if config.system.decay_learning_rates:
        return make_learning_rate_schedule(init_lr, config)
    else:
        return init_lr
