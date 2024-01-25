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

from typing import Any, Callable, Dict, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey


def ff_sample_actor_output(
    actor_policy, key: PRNGKey, traj_actions=None, get_log_prob=False, action_type="continuos"
) -> Array:
    def get_continuous_act() -> Any:
        actor_mean_logits, actor_log_std_logits = actor_policy
        actor_distribution = distrax.MultivariateNormalDiag(
            actor_mean_logits, jnp.exp(actor_log_std_logits)
        )
        # TODO: log_prob -= jnp.sum(jnp.log(0.5 * (1 - jnp.tanh(actions) ** 2) + 1e-6), axis=-1)
        if get_log_prob:
            log_prob = actor_distribution.log_prob(traj_actions)
            return actor_distribution, log_prob
        else:
            actions, log_prob = actor_distribution.sample_and_log_prob(seed=key)
            # actions = actions * n (if action in [-n, n]) -> Default: [-1, 1]
            return actions, log_prob

    def get_discrete_act() -> Any:
        if get_log_prob:
            log_prob = actor_policy.log_prob(traj_actions)
            return actor_policy, log_prob

        else:
            actions, log_prob = actor_policy.sample_and_log_prob(seed=key)
            return actions, log_prob

    output_fn = get_discrete_act if action_type == "discrete" else get_continuous_act
    return output_fn()


def select_actions_ppo_recurrent(
    apply_fn: Callable,
    params: nn.FrozenDict,
    hstate: Array,
    obs: Array,
    done: Array,
    key: PRNGKey,
) -> Array:
    actor_in = (obs, jnp.expand_dims(done, axis=(0)))
    hstate, mean, log_std = apply_fn(params, hstate, actor_in)
    policy = distrax.MultivariateNormalDiag(mean.squeeze(0), jnp.exp(log_std.squeeze(0)))

    actions, _ = policy.sample_and_log_prob(seed=key)
    actions = jnp.tanh(actions) * 0.5 + 0.5

    return hstate, actions
