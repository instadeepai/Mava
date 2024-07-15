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

# Shape legend:
# B: batch size
# A: num agents
# Act: action dim/num actions


def get_joint_action(actions: Array) -> Array:
    """Get the joint action from the individual actions of the agents.

    Joint actions are simply the concatenation of all agents actions.
    Shapes are transformed from (B, A, Act) -> (B, A, A * Act).
    Note: this returns the same joint action tiled for each agent.

    Args:
    ----
        actions (B, A, Act): the individual actions of the agents.

    Returns: (B, A, A * Act): the joint action repeated for each agent.

    """
    batch_size, num_agents, act_size = actions.shape
    repeated_action = jnp.tile(actions[:, jnp.newaxis, ...], (1, num_agents, 1, 1))
    return jnp.reshape(repeated_action, (batch_size, num_agents, act_size * num_agents))


def get_updated_joint_actions(rb_actions: Array, policy_actions: Array) -> Array:
    """Get the updated joint actions by replacing the actions from the replay buffer with the new
    actions from the policy. Only update joint action i with the new action for agent i.

    The effect of this is that each agent's central critic sees what all other agents did in the
    past, but it sees how its agent's policy is currently acting.

    Method explanation:
    The `rb_actions` (B, A, Act) will be repeated such that you have two agent dims: (B, A, A, Act).
    Then the diagonal of the repeated actions will be replaced with the new actions from the policy.
    This replacement means that joint_action[i] will have the new action for agent[i].
    Finally join the last two dimensions to get (B, A, A * Act).

    Example:
    -------
    Given an action dim of 1, batch size of 1 and 3 agents.
    All agents action may look like this: [0, 1, 2].
    It is then repeated num agent times:
    [
      [0, 1, 2],
      [0, 1, 2],
      [0, 1, 2]
    ]
    Now new/updated actions from the policies may look like this: [3, 4, 5].
    We want to replace action[i] in joint_action[i] so we replace along the diagonal:
    [
      [3, 1, 2],
      [0, 4, 2],
      [0, 1, 5]
    ]
    Seeing as our action dim is 1 there is no need to do the final reshape step,
    but given an action dim > 1 you would need to join the last two dims.

    Args:
    ----
        rb_actions (B, A, Act): the actions from the replay buffer.
        policy_actions (B, A, Act): the new actions from the policy.

    Returns: (B, A, A * Act): the updated joint actions.

    """
    batch_size, num_agents, act_size = rb_actions.shape

    # Repeat the actions from the replay buffer such that you have (B, A, A, Act).
    # This gives num agents joint actions with the action dim kept separate.
    actions_repeated = jnp.tile(rb_actions[:, jnp.newaxis, ...], (1, num_agents, 1, 1))
    # Find the indices of the diagonal of an (A, A) matrix.
    inds = jnp.diag_indices(num_agents)
    # Replace along the diagonal with the new action from the policy.
    # This replacement means that joint_action[i] will have the new action for agent[i].
    updated_joint_actions = actions_repeated.at[:, inds[0], inds[1], :].set(policy_actions)
    # Reshape to (B, A, A * Act) so that we create the joint action dim from the extra agent dim.
    return updated_joint_actions.reshape(batch_size, num_agents, num_agents * act_size)
