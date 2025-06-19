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

# from jumanji.environments.routing.tmaze.viewer import TmazeViewer
from functools import cached_property
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
from jumanji import Environment, specs
from jumanji.types import StepType, TimeStep, termination, transition

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class State:
    agent_positions: jax.Array  # (2, 2)
    agent_targets: jax.Array  # (2)  agents target choice: 0 or 1 to index into target positions
    target_positions: jax.Array  # (2, 2) location of the target at this index
    step_count: jax.typing.ArrayLike  # ()
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    agents_view: chex.Array  # (num_agents, grid_size, grid_size)
    action_mask: chex.Array  # (num_agents, 5)
    step_count: chex.Array  # ()


class TMaze(Environment):
    def __init__(self, length: int, width: int, time_limit: int | None = None) -> None:
        self.time_limit = time_limit or (length + width) * 2
        self.length = length
        self.width = width  # only the width of the one side of the T

        super().__init__()

        # Mava params
        self.num_agents = 2
        self.action_dim = 7

        self.left_target = jnp.array([self.length, -self.width])
        self.right_target = jnp.array([self.length, 1 + self.width])

        self.start_positions = jnp.array([[0, 0], [0, 1]])
        self.target_positions = jnp.stack([self.left_target, self.right_target], axis=0)

        # NOOP, UP, RIGHT, DOWN, LEFT, CHOOSE_0, CHOOSE_1
        self.moves = jnp.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [0, 0], [0, 0]])

        self.viewer = None  # TmazeViewer(length=length, width=width)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        key, position_key, target_key = jax.random.split(key, 3)

        a0_pos_idx = jax.random.randint(position_key, (), 0, 2)
        a0_pos = self.start_positions[a0_pos_idx]
        a1_pos = self.start_positions[1 - a0_pos_idx]

        a0_target_idx = jax.random.randint(target_key, (), 0, 2)
        target_idx = jnp.array([a0_target_idx, 1 - a0_target_idx])
        targets = self.target_positions[target_idx]

        positions = jnp.stack([a0_pos, a1_pos], axis=0)

        state = State(
            agent_positions=positions,
            agent_targets=-jnp.ones(2, dtype=jnp.int32),  # only choose on first step
            target_positions=targets,
            step_count=jnp.zeros((), jnp.int32),
            key=key,
        )
        single_agent_reset_mask = self.get_reset_action_mask()
        action_mask = jnp.tile(single_agent_reset_mask[jnp.newaxis, :], (self.num_agents, 1))
        obs = Observation(
            jnp.zeros_like(self.get_obs(state, -jnp.ones((2,), jnp.int32))),
            action_mask,
            jnp.zeros((2,), jnp.int32),
        )

        ts = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.zeros((2,), dtype=jnp.float32),
            discount=jnp.ones((), dtype=jnp.float32),
            observation=obs,
            extras={"env_metrics": {}},
        )
        return state, ts

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        return jax.lax.cond(state.step_count == 0, self.choose_step, self.move_step, state, action)

    def choose_step(self, state: State, action: jax.Array) -> Tuple[State, TimeStep[Observation]]:
        # assert jnp.all(action > 5), "On the first step your action must be 6 or 7 (choose)"
        agent_targets = (action == 6).astype(int)
        new_state = state.replace(agent_targets=agent_targets, step_count=state.step_count + 1)  # type: ignore

        action_mask = jax.vmap(self.get_step_action_mask, (None, 0))(
            new_state, new_state.agent_positions
        )
        step_count = jnp.full((2,), new_state.step_count, dtype=jnp.int32)
        obs = Observation(self.get_obs(new_state, action), action_mask, step_count)
        ts = transition(jnp.zeros(2, jnp.float32), obs)
        ts.extras = {"env_metrics": {}}

        return new_state, ts

    def move_step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        # assert jnp.all(action <= 5)
        moves = self.moves[action]
        new_positions = moves + state.agent_positions

        not_colliding = jnp.any(new_positions[0] != new_positions[1])
        valid_next_pos = jax.vmap(self.empty_position, (None, 0))(state, new_positions)
        valid_move = (valid_next_pos & not_colliding) | (action == 0)  # NOOP always valid
        new_positions = jnp.where(valid_move[:, jnp.newaxis], new_positions, state.agent_positions)

        new_state = state.replace(agent_positions=new_positions, step_count=state.step_count + 1)  # type: ignore
        action_mask = jax.vmap(self.get_step_action_mask, (None, 0))(
            new_state, new_state.agent_positions
        )

        done_horizon = new_state.step_count >= self.time_limit
        done_targets = jnp.all(new_positions == state.target_positions[state.agent_targets])
        reward = jnp.ones(2, dtype=jnp.float32) * done_targets

        step_count = jnp.full((2,), new_state.step_count, dtype=jnp.int32)
        obs = Observation(self.get_obs(new_state, action), action_mask, step_count)
        ts = jax.lax.cond(done_horizon | done_targets, termination, transition, reward, obs)
        ts.extras = {"env_metrics": {}}

        # jax.debug.print(
        #     "Ag pos: {a} | targs: {t} | ordered targs: {o}",
        #     a=new_positions,
        #     t=state.target_positions,
        #     o=state.target_positions[state.agent_targets],
        # )

        return new_state, ts

    def get_obs(self, state: State, action: jax.Array) -> jax.Array:
        a0_obs = self.get_agent_obs(state, state.agent_positions[0], state.agent_positions[1])
        a1_obs = self.get_agent_obs(state, state.agent_positions[1], state.agent_positions[0])

        target_obs = jax.lax.cond(
            state.step_count == 0,
            lambda: -jnp.ones((2,), dtype=a0_obs.dtype),  # first target positions are unkown
            lambda: jnp.repeat((state.target_positions[0, 1] < 0), 2).astype(a0_obs.dtype),
        )

        obs = jnp.stack([a0_obs, a1_obs], axis=0)
        act_obs = (action[:, jnp.newaxis] == 6) + jnp.all(action > 5)
        obs = jnp.concatenate([obs, target_obs[:, jnp.newaxis], act_obs], axis=-1)
        return obs

    def get_agent_obs(
        self, state: State, agent_position: jax.Array, team_position: jax.Array
    ) -> jax.Array:
        surrounding_cell_values = self.surrounding_points(agent_position)
        return jax.vmap(self.get_cell_value, (None, None, 0))(
            agent_position,
            team_position,
            surrounding_cell_values,
        )

    def get_cell_value(
        self, my_pos: jax.Array, team_pos: jax.Array, cell_pos: jax.Array
    ) -> jax.Array:
        in_bounds = self.is_cell_in_bounds(cell_pos)
        # -1 if out of bounds
        # 1 if agent 1
        # 2 if agent 2
        # 0 if empty cell
        return (
            (-1 * ~in_bounds)
            + (1 * jnp.all(cell_pos == my_pos))
            + (2 * jnp.all(cell_pos == team_pos))
        )

    def is_cell_in_bounds(self, cell_pos: jax.Array) -> bool:
        x, y = cell_pos
        is_on_vertical = (x >= 0) & (x <= self.length)
        is_on_horizontal = x == self.length

        return (is_on_vertical & ((y == 0) | (y == 1))) | (
            is_on_horizontal & (y >= -self.width) & (y <= 1 + self.width)
        )

    def empty_position(self, state: State, new_position: jax.Array) -> jax.Array:
        return (
            self.is_cell_in_bounds(new_position)
            # not on top of an agent
            & jnp.any(new_position != state.agent_positions, axis=1).all()
        )

    def surrounding_points(self, cell_pos: jax.Array) -> jax.Array:
        surrounding_vecs = jnp.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, -1],
                [-1, 0],
                [-1, -1],
                [1, -1],
                [-1, 1],
            ]
        )
        return cell_pos + surrounding_vecs

    def get_reset_action_mask(self) -> jax.Array:
        """
        Generates the action mask for a single agent at the reset step (step 0).
        Only CHOOSE_0 (action 5) and CHOOSE_1 (action 6) are allowed.
        """
        # Total 7 actions: 0-4 are movement/NOOP, 5-6 are CHOOSE actions.
        mask = jnp.zeros(7, dtype=bool)
        # Allow action 5 (CHOOSE_0) and action 6 (CHOOSE_1)
        mask = mask.at[5:7].set(True)
        return mask

    def get_step_action_mask(self, state: State, my_pos: jax.Array) -> jax.Array:
        """
        Generates the action mask for a single agent for steps > 0.
        Movement actions (0-4) are allowed based on validity, NOOP is always True.
        CHOOSE actions (5-6) are never allowed after the first step.
        """
        # Calculate mask for movement actions (0-4: NOOP, UP, RIGHT, DOWN, LEFT)
        # self.moves[:5] corresponds to these actions.
        possible_movement_pos = my_pos + self.moves[:5]
        movement_mask_parts = jax.vmap(self.empty_position, (None, 0))(state, possible_movement_pos)

        # Ensure NOOP (action 0) is always valid among the movement actions
        movement_mask_final = movement_mask_parts.at[0].set(True)

        # Actions 5 and 6 (CHOOSE actions) are always False after the initial step
        choice_actions_mask = jnp.array([False, False], dtype=bool)

        return jnp.concatenate([movement_mask_final, choice_actions_mask])

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        agents_view = specs.BoundedArray(
            shape=(2, 11), dtype=jnp.int32, name="grid", minimum=-1, maximum=2
        )
        action_mask = specs.BoundedArray(
            shape=(2, 7), dtype=bool, minimum=False, maximum=True, name="action_mask"
        )
        step_count = specs.BoundedArray(
            shape=(2,),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.time_limit,
            name="step_count",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

    @cached_property
    def action_spec(self) -> specs.MultiDiscreteArray:
        return specs.MultiDiscreteArray(
            num_values=jnp.array([7] * 2), dtype=jnp.int32, name="action"
        )

    def render(self, state: State) -> Any:
        return self.viewer.render(state)  # type: ignore

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:  # type: ignore
        return self.viewer.animate(states, interval, save_path)  # type: ignore
