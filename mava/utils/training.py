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

import distrax
import jax.numpy as jnp
from chex import Array, PRNGKey
from omegaconf import DictConfig


def make_learning_rate_schedule(init_lr: float, config: DictConfig) -> Callable:
    """Makes a very simple linear learning rate scheduler.

    Args:
        init_lr: initial learning rate.
        config: system configuration.

    Note:
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
        init_lr: initial learning rate.
        config: system configuration.

    Returns:
        A learning rate schedule or fixed learning rate.
    """
    if config.system.decay_learning_rates:
        return make_learning_rate_schedule(init_lr, config)
    else:
        return init_lr


def select_action_cont_ppo(
    actor_output: Tuple[Array, Array],
    key: PRNGKey,
    env_name: str,
) -> Tuple[Array, Array, Array]:
    """Select action for the continous action for systems given the actor output."""

    actor_mean, actor_log_std = actor_output
    actor_policy = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))

    raw_action, log_prob = actor_policy.sample_and_log_prob(seed=key)
    action, log_prob = transform_actions_log(env_name, raw_action, log_prob)

    return raw_action, action, log_prob


def get_logprob_entropy(
    mean: Array,
    log_std: Array,
    action: Array,
    env_name: str,
) -> Tuple[Array, Array]:
    """Get the log probability and entropy of a given mean, log_std and action."""
    actor_policy = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
    log_prob = actor_policy.log_prob(action)
    entropy = actor_policy.entropy().mean()
    _, log_prob = transform_actions_log(env_name, action, log_prob)
    return log_prob, entropy


def select_action_eval_cont_ff(
    actor_output: Tuple[Array, Array],
    key: PRNGKey,
    env_name: str,
) -> Union[Array, Tuple[Array, Array]]:
    """Select action for the continous action for the FF-systems given the actor output."""
    mean, log_std = actor_output
    actor_policy = distrax.MultivariateNormalDiag(mean, log_std)

    raw_action = actor_policy.sample(seed=key)
    return transform_actions_log(env_name, raw_action, None)


def select_action_eval_cont_rnn(
    actor_output: Tuple[Array, Array],
    key: PRNGKey,
    env_name: str,
) -> Array:
    """Select action for the continous action for the Rec-systems given the actor output."""
    mean, log_std = actor_output

    actor_policy = distrax.MultivariateNormalDiag(mean, log_std)
    raw_action = actor_policy.sample(seed=key)
    action = transform_actions_log(env_name, raw_action, None)

    return action


def transform_actions_log(
    env_name: str, raw_action: Array, log_prob: Array, n: int = 1
) -> Union[Array, Tuple[Array, Array]]:
    """Transform action and log_prob values"""

    # IF action in [-N, N] and N != 1 ELSE [-1, 1] and N == 1
    # n = 0.4 if env_name == "humanoid_9|8" else 1
    action = n * jnp.tanh(raw_action)
    if log_prob is None:
        return action  # during evaluation.

    # Note: jnp.log(derivative of action equation)
    log_prob -= jnp.sum(jnp.log(n * (1 - jnp.tanh(raw_action) ** 2) + 1e-6), axis=-1)

    return action, log_prob
