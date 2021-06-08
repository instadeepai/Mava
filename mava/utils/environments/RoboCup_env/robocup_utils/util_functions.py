# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

# type: ignore
import time
from typing import Dict, NamedTuple

import dm_env
import gym
import numpy as np
from acme import specs, types


def rad_rot_to_xy(rad_rot):
    return np.cos(rad_rot), np.sin(rad_rot)


def deg_rot_to_xy(deg_rot):
    return np.cos(deg_rot * np.pi / 180), np.sin(deg_rot * np.pi / 180)


def should_wait(wait_list):
    should_wait = False
    for entity in wait_list:
        if not entity.wm.new_data:
            should_wait = True
            break
    return should_wait


def wait_for_next_observations(obs_to_wait_for):
    """
    Wait for next observation before agents should think again.
    """
    # Wait for new data
    waiting = should_wait(obs_to_wait_for)

    # start = time.time()
    while waiting:
        waiting = should_wait(obs_to_wait_for)
        time.sleep(0.0001)
    # end = time.time()
    # print("Wait time: ", end-start)

    # Set new data false
    for entity in obs_to_wait_for:
        entity.wm.new_data = False


# Dummy class to get the observation in the correct format
class OLT(NamedTuple):
    """Container for (observation, legal_actions, terminal) tuples."""

    observation: types.Nest
    legal_actions: types.Nest
    terminal: types.Nest


class SpecWrapper(dm_env.Environment):
    """Spec wrapper for 2D RoboCup environment."""

    # Note: we don't inherit from base.EnvironmentWrapper because that class
    # assumes that the wrapped environment is a dm_env.Environment.

    def __init__(self, num_players: int):
        self._reset_next_step = True

        self.scaling = 200.0

        # Chose action
        act_min = [0.0] * 7  # 6 + No action
        act_max = [1.0] * 7  # 6 + No action

        # Action continuous component
        # All directions are in x, y format
        act_min.extend(
            [
                -100 / self.scaling,
                -1,
                -1,  # dash (power, direction)
                0,
                -1,
                -1,  # kick (power, direction)
                0,
                0,  # change_view (width, quality)
                -1,
                -1,
                0,  # tackle (direction, foul)
                -1,
                -1,  # turn (direction)
                -1,
                -1,
            ]
        )  # turn_neck(direction)

        act_max.extend([100 / self.scaling, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        assert len(act_min) == len(act_max)
        action_spec = specs.BoundedArray(
            shape=(len(act_min),),
            dtype="float32",
            name="action",
            minimum=act_min,
            maximum=act_max,
        )

        self.action_size = action_spec.shape[0]

        # obs_dict = {"time_left": 0, "side": 1, "sense_self": 2,
        #  "coords": (3, 5), "body_dir": (5, 7),
        #             "head_dir": (7, 9), "width": (9, 12),
        # "quality": 13, "stamina": 14, "effort": 15,
        #             "speed_amount": 16, "speed_direction": (17, 19),
        # "neck_direction": (19, 21),
        #             "see_ball": 21, "ball_dist": 22,
        # "ball_dir": (23, 25), "ball_dist_change": 25,
        #             "ball_dir_change": 26, "ball_speed": 27,
        # "last_action": (28, 28 + self.action_size),
        #             }

        # TODO: Check if all bounds are correct
        obs_min = [
            0.0,  # time_left
            0.0,  # side
            0.0,  # sense_self
            -100 / self.scaling,
            -50 / self.scaling,  # coords
            -1,
            -1,  # body_dir
            -1,
            -1,  # head_dir
            0,
            0,
            0,  # width
            0,  # quality
            0,  # stamina
            0,  # effort
            0,  # speed_amount
            -1,
            -1,  # speed_direction
            -1,
            -1,  # neck_direction
            0,  # see_ball
            0,  # ball_dist
            -1,
            -1,  # ball_dir
            -100 / self.scaling,  # ball_dist_change
            -180 / self.scaling,  # ball_dir_change
            0,  # ball_speed
        ]

        obs_max = [
            1.0,  # time_left
            1.0,  # side
            1.0,  # sense_self
            100 / self.scaling,
            50 / self.scaling,  # coords
            1,
            1,  # body_dir
            1,
            1,  # head_dir
            1,
            1,
            1,  # width
            1,  # quality
            1,  # stamina
            1,  # effort
            100 / self.scaling,  # speed_amount
            1,
            1,  # speed_direction
            1,
            1,  # neck_direction
            1,  # see_ball
            100 / self.scaling,  # ball_dist
            1,
            1,  # ball_dir
            100 / self.scaling,  # ball_dist_change
            180 / self.scaling,  # ball_dir_change
            100 / self.scaling,  # ball_speed
        ]

        # Last action
        obs_min.extend(action_spec.minimum)
        obs_max.extend(action_spec.maximum)

        # [see_player, is_on_team, player_distance,
        # player_direction] for num_agents-1
        self.num_agents = num_players

        # TODO: Add this in again.
        # for i in range(21):
        #     # [see_player, is_on_team, player_distance,
        # player_direction (x, y format)]
        #     obs_min.extend([0, 0, -200 / self.scaling, -1, -1])
        #     obs_max.extend([1, 1, +200 / self.scaling, 1, 1])

        assert len(obs_min) == len(obs_max)
        self.obs_size = len(obs_min)

        self.agents = ["player_" + str(r) for r in range(num_players)]

        self._observation_specs = {}
        self._action_specs = {}

        obs_spec = specs.BoundedArray(
            shape=(self.obs_size,),
            dtype="float32",
            name="observation",
            minimum=obs_min,
            maximum=obs_max,
        )

        # Time_left, ball coords, ball delta_coords
        state_min = [0, -100 / self.scaling, -100 / self.scaling, -10, -10]
        state_max = [1, 100 / self.scaling, 100 / self.scaling, 10, 10]

        # First player is the critic player
        # Players sides,  coords, delta_coords, body_angle (x, y format),
        # head_angle (x, y format)
        for i in range(num_players):
            state_min.extend(
                [
                    0.0,
                    -100 / self.scaling,
                    -100 / self.scaling,
                    -10,
                    -10,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
            state_max.extend(
                [1.0, +100 / self.scaling, +100 / self.scaling, +10, +10, 1, 1, 1, 1]
            )

        # Add all observations to state info
        for i in range(num_players):
            state_min.extend(obs_min)
            state_max.extend(obs_max)

        assert len(state_min) == len(state_max)
        self._state_spec = specs.BoundedArray(
            shape=(len(state_min),),
            dtype="float32",
            name="state",
            minimum=state_min,
            maximum=state_max,
        )

        self._discount = dict(zip(self.agents, [np.float32(1.0)] * len(self.agents)))

        # TODO: Delete this
        # self.previous_act = {"player_0": None}

        for agent in self.agents:
            # TODO: Why is the action spec in two places?
            self._observation_specs[agent] = OLT(
                observation=obs_spec,
                legal_actions=action_spec,
                terminal=specs.Array((1,), np.float32),
            )

            self._action_specs[agent] = action_spec
        # Obs: BoundedArray(shape=(4,), dtype=dtype('float32'), name='observation',
        # minimum=[-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38],
        #  maximum=[4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38])
        # Actions: DiscreteArray(shape=(), dtype=int64, name=action, minimum=0,
        #  maximum=1, num_values=2)

    def reset(self):
        pass

    def step(self):
        pass

    def observation_spec(self) -> types.NestedSpec:
        return self._observation_specs

    def action_spec(self) -> types.NestedSpec:
        return self._action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        reward_specs = {}
        for agent in self.agents:
            reward_specs[agent] = specs.Array((), np.float32)
        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        discount_specs = {}
        for agent in self.agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        return {"s_t": self._state_spec}

    def _proc_robocup_obs(
        self, observations: Dict, done: bool, nn_actions: Dict = None
    ) -> Dict:
        # TODO: Try to automatically normalise by min max boundries
        processed_obs_dict = {}
        for obs_i, agent_key in enumerate(self.agents):
            env_agent_obs = observations[agent_key]

            if nn_actions:
                last_action = nn_actions[agent_key]
            else:
                last_action = None

            proc_agent_obs = self.proc_agent_env_obs(env_agent_obs, last_action)

            observation = OLT(
                observation=proc_agent_obs,
                legal_actions=np.ones(self.action_size, dtype=np.float32),
                terminal=np.asarray([done], dtype=np.float32),
            )
            processed_obs_dict[agent_key] = observation
        return processed_obs_dict

    def proc_agent_env_obs(self, env_agent_obs, last_action):  # noqa: C901
        # All angles is in x,y format
        proc_agent_obs = np.zeros(self.obs_size, dtype=np.float32)
        obs_dict = {
            "time_left": 0,
            "side": 1,
            "sense_self": 2,
            "coords": (3, 5),
            "body_dir": (5, 7),
            "head_dir": (7, 9),
            "width": (9, 12),
            "quality": 12,
            "stamina": 13,
            "effort": 14,
            "speed_amount": 15,
            "speed_direction": (16, 18),
            "neck_direction": (18, 20),
            "see_ball": 20,
            "ball_dist": 21,
            "ball_dir": (22, 24),
            "ball_dist_change": 24,
            "ball_dir_change": 25,
            "ball_speed": 26,
            "last_action": (27, 27 + self.action_size),
        }

        # TODO: Fix this when there is other players
        assert self.obs_size == 27 + self.action_size

        # Time left obs
        if "game_step" in env_agent_obs and "game_length" in env_agent_obs:
            # Time left
            proc_agent_obs[obs_dict["time_left"]] = (
                1 - env_agent_obs["game_step"] / env_agent_obs["game_length"]
            )

        # Team side obs
        side_to_id = {"l": 0.0, "r": 1.0}
        proc_agent_obs[1] = side_to_id[env_agent_obs["side"]]

        if (
            "estimated_abs_coords" in env_agent_obs
            and env_agent_obs["estimated_abs_coords"][0] is not None
        ):
            # see_own_stats
            proc_agent_obs[obs_dict["sense_self"]] = 1.0

            # coords
            coords = env_agent_obs["estimated_abs_coords"]
            s, e = obs_dict["coords"]
            proc_agent_obs[s] = float(coords[0]) / self.scaling
            proc_agent_obs[e - 1] = float(coords[1]) / self.scaling

            # body_angle
            s, e = obs_dict["body_dir"]
            proc_agent_obs[s:e] = deg_rot_to_xy(env_agent_obs["estimated_abs_body_dir"])

            # head_angle
            s, e = obs_dict["head_dir"]
            proc_agent_obs[s:e] = deg_rot_to_xy(env_agent_obs["estimated_abs_neck_dir"])

            # view_width
            w_to_int = {"narrow": 0, "normal": 1, "wide": 2}
            onehot = np.zeros(3)
            onehot[w_to_int[env_agent_obs["view_width"]]] = 1
            s, e = obs_dict["width"]
            proc_agent_obs[s:e] = onehot

            # view_quality
            q_to_int = {"high": 0, "low": 1}
            proc_agent_obs[obs_dict["quality"]] = q_to_int[
                env_agent_obs["view_quality"]
            ]

            # stamina
            proc_agent_obs[obs_dict["stamina"]] = env_agent_obs["stamina"] / 8000

            # effort
            proc_agent_obs[obs_dict["effort"]] = env_agent_obs["effort"]

            # speed_amount
            proc_agent_obs[obs_dict["speed_amount"]] = (
                env_agent_obs["speed_amount"] / self.scaling
            )

            # speed_dir
            s, e = obs_dict["speed_direction"]
            proc_agent_obs[s:e] = deg_rot_to_xy(env_agent_obs["speed_direction"])

            # Relative neck dir
            s, e = obs_dict["neck_direction"]
            proc_agent_obs[s:e] = deg_rot_to_xy(env_agent_obs["neck_direction"])

        # See_ball, ball_distance, ball_direction, dist_change, dir_change, speed
        if "ball" in env_agent_obs and env_agent_obs["ball"] is not None:
            # Has ball flag
            proc_agent_obs[obs_dict["see_ball"]] = 1.0

            if env_agent_obs["ball"].distance is not None:
                proc_agent_obs[obs_dict["ball_dist"]] = (
                    env_agent_obs["ball"].distance / self.scaling
                )

            if env_agent_obs["ball"].direction is not None:
                s, e = obs_dict["ball_dir"]
                proc_agent_obs[s:e] = deg_rot_to_xy(env_agent_obs["ball"].direction)

            if env_agent_obs["ball"].dist_change is not None:
                proc_agent_obs[obs_dict["ball_dist_change"]] = (
                    env_agent_obs["ball"].dist_change / self.scaling
                )

            if env_agent_obs["ball"].dir_change is not None:
                proc_agent_obs[obs_dict["ball_dir_change"]] = (
                    env_agent_obs["ball"].dir_change / self.scaling
                )

            if env_agent_obs["ball"].speed is not None:
                proc_agent_obs[obs_dict["ball_speed"]] = (
                    env_agent_obs["ball"].speed / self.scaling
                )

        # Last player actions
        if last_action is not None:
            s, e = obs_dict["last_action"]
            proc_agent_obs[s:e] = last_action

        # [see_player, is_on_team, distance, direction, dist_change,
        # dir_change, speed, body_direction, neck_direction]
        if "players" in env_agent_obs and len(env_agent_obs["players"]) > 0:
            raise NotImplementedError(
                "Obs for more than one player not implemented yet."
            )
            # ["opponent", player.distance, player.direction (x, y format)]
            num_see_players = len(env_agent_obs["players"])

            # Place the players on the closest available spot
            players = env_agent_obs["players"]

            # TODO: Get better sorting algorithm. This one is good,
            # but also allows for player degrees not to
            #  always be assending. This is because a player can be
            # selected for a slot and then the player with a
            #  bigger positive degree might be selected for a lower
            # slot. Therefore this might happend:
            #  player[i].degree > player[i+1].degree.

            player_ids = list(range(num_see_players))
            spots = [[d_i, (d_i / 20) * 180 - 90] for d_i in range(21)]
            player_spots = [-1] * num_see_players
            for _ in range(num_see_players):
                best_p_i = None
                beset_spot = None
                best_s_count = None
                best_p_count = None
                beset_dist = np.inf

                # All remaining players select the closest available spot
                for p_count, p_i in enumerate(player_ids):
                    player_type, player_dist, player_dir = players[p_i]

                    # Go through all the spots and get the closest one to the player
                    for s_count, spot in enumerate(spots):
                        s_i, s_dir = spot

                        dist_to_spot = abs(s_dir - player_dir)

                        if dist_to_spot < beset_dist:
                            beset_dist = dist_to_spot
                            best_p_i = p_i
                            beset_spot = s_i
                            best_p_count = p_count
                            best_s_count = s_count

                # Give the closest player its spot.
                player_ids.pop(best_p_count)
                spots.pop(best_s_count)
                player_spots[best_p_i] = beset_spot

            assert len(player_ids) == 0
            # Calculate all the available slots to put player in
            slots = [
                [11 + self.action_size + s_i * 5, 11 + self.action_size + s_i * 5 + 5]
                for s_i in player_spots
            ]
            # slots = [i, i + 5] for i in
            # slots = [all_slots[p_spot] for p_spot in player_spots]

            assert len(slots) == num_see_players
            player_type_dict = {"opponent": 0.0, "team": 1.0}

            for p_i, player in enumerate(players):
                player_type, player_dist, player_dir = player

                # pop a random element from the slots list
                start_i, end_i = slots[p_i]

                # see_player, is_on_team, player_distance,
                # player_direction (x, y format)
                dir_x, dir_y = deg_rot_to_xy(player_dir)

                proc_agent_obs[start_i:end_i] = [
                    1.0,
                    player_type_dict[player_type],
                    player_dist / self.scaling,
                    dir_x,
                    dir_y,
                ]

        # if not env_agent_obs["obs_updated"]:
        #     proc_agent_obs[2] = 0.0

        return proc_agent_obs

    def _proc_robocup_state(self, state: Dict, proc_obs: Dict) -> np.array:
        state_dict = {
            "time_left": 0,
            "ball_coords": (1, 3),
            "ball_delta": (3, 5),
            "player_offset": 5,
            "p_obs_size": 9,
            "p_side": 0,
            "p_coords": (1, 3),
            "p_delta": (3, 5),
            "p_b_dir": (5, 7),
            "p_n_dir": (7, 9),
        }
        offset = state_dict["player_offset"]

        proc_agent_state = np.zeros(
            offset + self.num_agents * (9 + self.obs_size), dtype=np.float32
        )

        # Time left
        proc_agent_state[state_dict["time_left"]] = (
            1 - state["game_step"] / state["game_length"]
        )

        # Ball:
        ball = state["ball"]
        s, e = state_dict["ball_coords"]
        proc_agent_state[s:e] = [
            float(ball["coords"][0]) / self.scaling,
            float(ball["coords"][1]) / self.scaling,
        ]
        s, e = state_dict["ball_delta"]
        proc_agent_state[s:e] = [
            float(ball["delta_coords"][0]),
            float(ball["delta_coords"][1]),
        ]  # TODO: Should delta coords have a normaliser?

        # TODO: Add check to see if players are in the correct order
        # TODO: Include onehot indicating which player should be focussed on.
        # Also use dictionary of states. Players should not be reordered as this
        # messes with the action order provided to the Q value critic.
        players = state["players"]
        p_size = state_dict["p_obs_size"]
        if players:
            for i in range(self.num_agents):
                if len(players) > i:
                    player = players[i]
                    # 'side': 0, 'coords': (52.6498, 0.54963),
                    # 'delta_coords': (0.000227909, 0.00371977), 'body_angle': 163,
                    # 'neck_angle': 0}
                    proc_agent_state[
                        i * p_size + offset + state_dict["p_side"]
                    ] = player["side"]
                    s, e = state_dict["p_coords"]
                    proc_agent_state[
                        i * p_size + offset + s : i * p_size + offset + e
                    ] = [
                        player["coords"][0] / self.scaling,
                        player["coords"][1] / self.scaling,
                    ]
                    s, e = state_dict["p_delta"]
                    proc_agent_state[
                        i * p_size + offset + s : i * p_size + offset + e
                    ] = player[
                        "delta_coords"
                    ]  # TODO: Should delta coords have a normaliser?
                    s, e = state_dict["p_b_dir"]
                    proc_agent_state[
                        i * p_size + offset + s : i * p_size + offset + e
                    ] = deg_rot_to_xy(player["body_angle"])
                    s, e = state_dict["p_n_dir"]
                    proc_agent_state[
                        i * p_size + offset + s : i * p_size + offset + e
                    ] = deg_rot_to_xy(player["neck_angle"])

                else:
                    break

        # Add agent observations
        start_i = offset + self.num_agents * p_size
        for agent_i, agent_key in enumerate(self.agents):
            obs = proc_obs[agent_key].observation
            proc_agent_state[
                start_i
                + agent_i * self.obs_size : start_i
                + (agent_i + 1) * self.obs_size
            ] = obs

        return proc_agent_state

    def _proc_robocup_actions(self, actions: Dict) -> Dict:
        # dash (speed), turn (direction), kick (power, direction)
        processed_action_dict = {}

        for agent_key in actions.keys():
            action = actions[agent_key]
            assert len(action) == self.action_size
            processed_action_dict[agent_key] = self.proc_agent_action(action)

        return processed_action_dict

    def proc_agent_action(self, action):
        # TODO: Add catch, tackle, move, pointto, attentionto and say commands as well.
        int_to_command = [
            "dash",
            "kick",
            "change_view",
            "tackle",
            "turn",
            "turn_neck",
            "none",
        ]

        # Remove change_view and turn_neck action
        # TODONE: Remove this
        # action[2] = 0
        # action[5] = 0

        command = int_to_command[np.argmax(action[0 : len(int_to_command)])]

        # Do the command
        assert len(int_to_command) == 7

        act_dict = {
            "dash_pow": 7,
            "dash_dir": (8, 10),
            "kick_pow": 10,
            "kick_dir": (11, 13),
            "width": 13,
            "quality": 14,
            "tackle_dir": (15, 17),
            "tackle_foul": 17,
            "turn": (18, 20),
            "neck": (20, 22),
        }

        if command == "dash":
            power = action[act_dict["dash_pow"]] * self.scaling
            s, e = act_dict["dash_dir"]
            dir_x, dir_y = action[s:e]
            dir = np.arctan2(dir_y, dir_x) * 180 / np.pi
            robocup_action = "(dash " + str(power) + " " + str(dir) + ")"
        elif command == "kick":
            power = action[act_dict["kick_pow"]] * self.scaling
            s, e = act_dict["kick_dir"]
            dir_x, dir_y = action[s:e]
            dir = np.arctan2(dir_y, dir_x) * 180 / np.pi
            robocup_action = "(kick " + str(power) + " " + str(dir) + ")"
        elif command == "change_view":
            w_to_text = ["narrow", "normal", "wide"]
            width = w_to_text[int(action[act_dict["width"]] * 2.99)]
            q_to_text = ["high", "low"]
            quality = q_to_text[int(action[act_dict["quality"]] * 1.99)]
            robocup_action = "(change_view " + width + " " + quality + ")"
        elif command == "tackle":
            s, e = act_dict["tackle_dir"]
            dir_x, dir_y = action[s:e]
            dir = np.arctan2(dir_y, dir_x) * 180 / np.pi
            f_to_text = ["true", "false"]
            foul = f_to_text[int(action[act_dict["tackle_foul"]] * 1.99)]
            robocup_action = "(tackle " + str(dir) + " " + foul + ")"
        elif command == "turn":
            s, e = act_dict["turn"]
            dir_x, dir_y = action[s:e]
            dir = np.arctan2(dir_y, dir_x) * 180 / np.pi
            robocup_action = "(turn " + str(dir) + ")"
        elif command == "turn_neck":
            s, e = act_dict["neck"]
            x, y = action[s:e]
            turn_neck_dir = np.arctan2(y, x) * 180 / np.pi
            robocup_action = "(turn_neck " + str(turn_neck_dir) + ")"
        elif command == "none":
            robocup_action = "(done)"
        else:
            raise NotImplementedError("Command not implemented: ", command)
        return robocup_action

    @property
    def possible_agents(self) -> gym.Env:
        """Returns the number of possible agents."""
        return self.agents
