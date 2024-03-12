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

import jax.numpy as jnp
from chex import Array


def get_joint_action(actions: Array) -> Array:
    """
    Get the joint action from the individual actions of the agents.

    Joint actions are simply the concatenation of all agents actions.
    Shapes are transformed from (B, A, Act) -> (B, A, A * Act).
    Note: this returns the same joint action tiled for each agent.

    Args:
        actions (B, A, Act): the individual actions of the agents.

    Returns: (B, A, A * Act): the joint action repeated for each agent.
    """
    batch_size, num_agents, act_size = actions.shape
    repeated_action = jnp.tile(actions[:, jnp.newaxis, ...], (1, num_agents, 1, 1))
    return jnp.reshape(repeated_action, (batch_size, num_agents, act_size * num_agents))


def get_updated_joint_actions(rb_actions: Array, policy_actions: Array) -> Array:
    """
    Get the updated joint actions by replacing the actions from the replay buffer with the new
    actions from the policy. Only update joint action i with the new action for agent i.

    The effect of this is that each agents central critic sees what all other agents did in the
    past, but it sees how its agents policy is currently acting.

    Method explaination:
    The `rb_actions` (B, A, Act) will be repeated such that you have two agent dims: (B, A, A, Ac).
    Then the diagonal of the repeated actions will be replaced with the new actions from the policy.
    This replacement means that joint_action[i] will have the new action for agent[i].
    Finally squeeze out the last dim to get (B, A, A * Act).

    Args:
        rb_actions (B, A, Act): the actions from the replay buffer.
        policy_actions (B, A, Act): the new actions from the policy.

    Returns: (B, A, A * Act): the updated joint actions.
    """

    batch_size, num_agents, act_size = rb_actions.shape

    # Repeat the actions from the replay buffer such that you have (B, A, A, Act).
    # This gives you num agents joint actions with the action dim kept separate.
    actions_repeated = jnp.tile(rb_actions[:, jnp.newaxis, ...], (1, num_agents, 1, 1))
    # Find the indices of the diagonal of an (A, A) matrix.
    inds = jnp.diag_indices(num_agents)
    # Replace along the diagonal with the new action from the policy.
    # This replacement means that joint_action[i] will have the new action for agent[i].
    spliced_actions = actions_repeated.at[:, inds[0], inds[1], :].set(policy_actions)
    # Reshape to (B, A, A * Act) so that we create the joint action dim from the extra agent dim.
    return spliced_actions.reshape(batch_size, num_agents, num_agents * act_size)
