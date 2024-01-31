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

from typing import Any, Tuple

import distrax
import jax.numpy as jnp
from chex import Array, PRNGKey


def get_logprob_entropy(
    actor_output: Tuple[Array, Array],
    traj_action: Array,
    env_name: str,
    network: str = "feedforward",
) -> Tuple[Array, Array]:
    """Get the log probability and entropy of a given actor output and traj_batch action."""
    assert network in {"feedforward", "recurrent"}, "Please specify the correct network!"

    actor_mean, actor_log_std = actor_output
    actor_policy = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))

    log_prob = actor_policy.log_prob(traj_action)
    entropy = actor_policy.entropy().mean()

    _, log_prob = transform_actions_log(env_name, traj_action, log_prob)
    return log_prob, entropy


def select_action_ppo(
    actor_output: Tuple[Array, Array], key: PRNGKey, env_name: str, eval: bool = False
) -> Any:
    """Select action for the given actor output."""

    actor_mean, actor_log_std = actor_output
    actor_policy = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))

    raw_action, log_prob = actor_policy.sample_and_log_prob(seed=key)
    action, log_prob = transform_actions_log(env_name, raw_action, log_prob)

    return action if eval else (raw_action, action, log_prob)


def transform_actions_log(env_name: str, raw_action: Array, log_prob: Array) -> Tuple[Array, Array]:
    """Transform action and log_prob values"""

    # IF action in [-N, N] and N != 1 ELSE [-1, 1] and N == 1
    n = 0.4 if env_name == "humanoid_9|8" else 1
    action = n * jnp.tanh(raw_action) if raw_action is not None else raw_action

    # Note: jnp.log(derivative of action equation)
    log_prob -= jnp.sum(jnp.log(n * (1 - jnp.tanh(raw_action) ** 2)) + 1e-6, axis=-1)

    return action, log_prob
