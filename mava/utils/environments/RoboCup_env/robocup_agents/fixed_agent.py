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

import numpy as np

from mava.utils.environments.RoboCup_env.robocup_utils.util_functions import (
    SpecWrapper,
    deg_rot_to_xy,
)


class NaiveBot(object):
    def __init__(self, agent_type, num_players):
        # Convert action and observation specs.

        spec_wrapper = SpecWrapper(num_players)
        self.scaling = spec_wrapper.scaling
        self.num_players = num_players
        assert self.scaling == 200.0
        self.kicked = False

    def reset_brain(self):
        self.kicked = False

    def get_action(self, observation):
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
        }
        s, e = obs_dict["coords"]
        p_loc = observation[s:e] * self.scaling

        s, e = obs_dict["body_dir"]
        p_deg = np.arctan2(observation[e - 1], observation[s]) * 180 / np.pi

        # see_ball, ball_distance, ball_direction (x, y format),
        see_ball = observation[obs_dict["see_ball"]]
        b_dist = observation[obs_dict["ball_dist"]] * self.scaling
        s, e = obs_dict["ball_dir"]
        b_deg = np.arctan2(observation[e - 1], observation[s]) * 180 / np.pi
        # Compute the policy, conditioned on the observation.
        # dash (speed), turn (direction in x,y format),
        # kick (direction in x,y format, power)
        command = {
            "do_dash": 0,
            "do_kick": 1,
            "do_change_view": 2,
            "do_tackle": 3,
            "do_turn": 4,
            "do_turn_neck": 5,
            "do_none": 6,
            "dash_power": 7,
            "dash_dir_x": 8,
            "dash_dir_y": 9,
            "turn_dir_x": 18,
            "turn_dir_y": 19,
            "kick_power": 10,
            "kick_dir_x": 11,
            "kick_dir_y": 12,
        }
        action = np.zeros(22)
        if see_ball < 0.5:
            # Search for the ball
            action[command["do_turn"]] = 1
            x, y = deg_rot_to_xy(20)
            action[command["turn_dir_x"]] = x
            action[command["turn_dir_y"]] = y
        elif b_dist < 0.5:
            if not self.kicked:
                # Kick the ball
                self.kicked = True

                goal_locs = np.array([52.5, 0])

                x_diff, y_diff = goal_locs - p_loc

                kick_abs_dir = -np.arctan2(y_diff, x_diff) * 180 / np.pi

                kick_dir = p_deg - 90 + kick_abs_dir

                action[command["do_kick"]] = 1
                action[command["kick_power"]] = 50 / self.scaling

                x, y = deg_rot_to_xy(kick_dir)
                action[command["kick_dir_x"]] = x
                action[command["kick_dir_y"]] = y
            else:
                action[command["do_none"]] = 1
                # print("Waiting..")
        elif np.abs(b_deg) > 10:
            # Turn towards the ball
            action[command["do_turn"]] = 1
            x, y = deg_rot_to_xy(b_deg / 3)
            action[command["turn_dir_x"]] = x
            action[command["turn_dir_y"]] = y
        else:
            # Dash towards the ball
            action[command["do_dash"]] = 1
            speed = 100

            if b_dist < 5:
                speed = 20

            action[command["dash_power"]] = speed / self.scaling
            x, y = deg_rot_to_xy(0)
            action[command["dash_dir_x"]] = x
            action[command["dash_dir_y"]] = y

        if b_dist > 5.0:
            self.kicked = False

        return action
