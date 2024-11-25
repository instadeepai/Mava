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

from typing import Callable, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from mava.systems.ppo.types import PPOTransition, RNNPPOTransition


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


def _calculate_gae(
    traj_batch: PPOTransition,
    last_val: chex.Array,
    last_done: chex.Array,
    recurrent: bool,
    config: DictConfig,
) -> Tuple[chex.Array, chex.Array]:
    def _get_advantages(
        carry: Tuple[chex.Array, chex.Array, chex.Array], transition: RNNPPOTransition
    ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], chex.Array]:
        gae, next_value, next_done = carry
        done, value, reward = transition.done, transition.value, transition.reward
        gamma = config.system.gamma
        if not recurrent:
            next_done = done
        delta = reward + gamma * next_value * (1 - next_done) - value
        gae = delta + gamma * config.system.gae_lambda * (1 - next_done) * gae
        return (gae, value, done), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val, last_done),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value
