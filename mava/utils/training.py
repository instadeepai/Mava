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
import jax
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

    def linear_scheduler(count: int) -> float:
        frac: float = (
            1.0
            - (count // (config.system.ppo_epochs * config.system.num_minibatches))
            / config.system.num_updates
        )
        return init_lr * frac

    return linear_scheduler


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


def get_actor_policy(mean: Array, log_std: Array) -> distrax.MultivariateNormalDiag:
    """Gets the actor policy based on mean and log_std from the actor's network.

    Args:
        mean: The mean values from the actor's network.
        log_std: The log of the std values from the actor's network.

    Returns:
        distrax.MultivariateNormalDiag: The actor policy distribution.
    """
    return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))


def select_action_cont_ppo(
    mean: Array,
    log_std: Array,
    key: PRNGKey,
) -> Tuple[Array, Array, Array]:
    """Selects an action for the continuous action for systems given the actor output.

    Args:
        mean: The mean values.
        log_std: The log of std values.
        key: The PRNGKey for random sampling.
        config: The system config.

    Returns:
        Tuple[Array, Array, Array]: The unbound action, the clipped action, and the log prob.
    """
    actor_policy = get_actor_policy(mean, log_std)
    unbound_action, log_prob = actor_policy.sample_and_log_prob(seed=key)
    action, log_prob = transform_actions_log(unbound_action, log_prob)
    return unbound_action, action, log_prob


def get_logprob_entropy(
    mean: Array,
    log_std: Array,
    action: Array,
) -> Tuple[Array, Array]:
    """Gets the log probability and entropy of a given mean, log_std, and action.

    Args:
        mean: The mean values.
        log_std (Array): The log of std values.
        action (Array): The action values.
        config (DictConfig): The system config.

    Returns:
        Tuple[Array, Array]: The log probability and the entropy.
    """
    actor_policy = get_actor_policy(mean, log_std)
    log_prob = actor_policy.log_prob(action)
    entropy = actor_policy.entropy().mean()
    _, log_prob = transform_actions_log(action, log_prob)
    return log_prob, entropy


def select_action_eval_cont(
    mean: Array,
    log_std: Array,
    key: PRNGKey,
) -> Array:
    """Selects an action for the continuous action given the actor output.

    Args:
        mean: The mean values.
        log_std: The log of std values.
        key: The PRNGKey for random sampling.
        config: The system config.

    Returns:
        Array: Return the clipped action.
    """
    actor_policy = get_actor_policy(mean, log_std)
    unbound_action = actor_policy.sample(seed=key)
    return transform_actions_log(unbound_action, None)


def transform_actions_log(
    unbound_action: Array,
    log_prob: Array,
) -> Union[Array, Tuple[Array, Array]]:
    """Transforms action and log_prob values.

    Args:
        config: The system config.
        unbound_action: The unbounded action values.
        log_prob: The log probability values.

    Returns:
        Union[Array, Tuple[Array, Array]]: Either the action or the action and log prob.
    """
    action = jnp.tanh(unbound_action)
    if log_prob is None:
        return action  # during evaluation.
    # Note: jnp.log(derivative of action equation)
    # log_prob -= jnp.sum(jnp.log(n * (1 - jnp.tanh(unbound_action) ** 2) + 1e-6), axis=-1)
    log_prob -= jnp.sum(
        2 * (jnp.log(2) - unbound_action - jax.nn.softplus(-2 * unbound_action)), axis=-1
    )
    return action, log_prob
