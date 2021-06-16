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

import math

import numpy as np

from mava.utils.environments.RoboCup_env.robocup_utils.game_object import Flag
from mava.utils.environments.RoboCup_env.robocup_utils.util_functions import (
    rad_rot_to_xy,
)

true_flag_coords = Flag.FLAG_COORDS


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def rel_to_abs_coords(obj):
    obj_x = obj.distance * math.sin(-obj.direction * math.pi / 180)
    obj_y = -obj.distance * math.cos(-obj.direction * math.pi / 180)
    return obj_x, obj_y


class WorldModel:
    """
    Holds and updates the model of the world as known from current and past
    data.
    """

    # constants for team sides
    SIDE_L = "l"
    SIDE_R = "r"

    class PlayModes:
        """
        Acts as a static class containing variables for all valid play modes.
        The string values correspond to what the referee calls the game modes.
        """

        BEFORE_KICK_OFF = "before_kick_off"
        PLAY_ON = "play_on"
        TIME_OVER = "time_over"
        KICK_OFF_L = "kick_off_l"
        KICK_OFF_R = "kick_off_r"
        KICK_IN_L = "kick_in_l"
        KICK_IN_R = "kick_in_r"
        FREE_KICK_L = "free_kick_l"
        FREE_KICK_R = "free_kick_r"
        CORNER_KICK_L = "corner_kick_l"
        CORNER_KICK_R = "corner_kick_r"
        GOAL_KICK_L = "goal_kick_l"
        GOAL_KICK_R = "goal_kick_r"
        DROP_BALL = "drop_ball"
        OFFSIDE_L = "offside_l"
        OFFSIDE_R = "offside_r"

        def __init__(self):
            raise NotImplementedError(
                "Don't instantiate a PlayModes class,"
                " access it statically through WorldModel instead."
            )

    class RefereeMessages:
        """
        Static class containing possible non-mode messages sent by a referee.
        """

        # these are referee messages, not play modes
        FOUL_L = "foul_l"
        FOUL_R = "foul_r"
        GOALIE_CATCH_BALL_L = "goalie_catch_ball_l"
        GOALIE_CATCH_BALL_R = "goalie_catch_ball_r"
        TIME_UP_WITHOUT_A_TEAM = "time_up_without_a_team"
        TIME_UP = "time_up"
        HALF_TIME = "half_time"
        TIME_EXTENDED = "time_extended"

        # these are special, as they are always followed by '_' and an int of
        # the number of goals scored by that side so far.  these won't match
        # anything specifically, but goals WILL start with these.
        GOAL_L = "goal_l_"
        GOAL_R = "goal_r_"

        def __init__(self):
            raise NotImplementedError(
                "Don't instantiate a RefereeMessages class,"
                " access it statically through WorldModel instead."
            )

    def __init__(self, action_handler):
        """
        Create the world model with default values and an ActionHandler class it
        can use to complete requested actions.
        """
        # we use the action handler to complete complex commands
        self.ah = action_handler

        # these variables store all objects for any particular game step
        self.ball = None
        self.flags = []
        self.goals = []
        self.players = []
        self.lines = []

        # the default position of this player, its home position
        self.home_point = (None, None)

        # scores for each side
        self.score_l = 0
        self.score_r = 0

        # the name of the agent's team
        self.teamname = None

        # handle player information, like uniform number and side
        self.side = None
        self.uniform_number = None

        # stores the most recent message heard
        self.last_message = None

        # the mode the game is currently in (default to not playing yet)
        self.play_mode = WorldModel.PlayModes.BEFORE_KICK_OFF

        # Obs updated
        # self.obs_updated = False

        # body state
        self.view_width = None
        self.view_quality = None
        self.stamina = None
        self.effort = None
        self.speed_amount = None
        self.speed_direction = None
        self.neck_direction = None
        self.new_data = False

        # counts of actions taken so far
        self.kick_count = None
        self.dash_count = None
        self.turn_count = None
        self.say_count = None
        self.turn_neck_count = None
        self.catch_count = None
        self.move_count = None
        self.change_view_count = None

        # apparent absolute player coordinates and neck/body directions
        self.abs_coords = (None, None)
        self.abs_body_dir = None
        self.abs_neck_dir = None

        # create a new server parameter object for holding all server params
        self.server_parameters = ServerParameters()

    def __calculate_abs_info(self):
        if self.flags is not None:
            rel_coords = []
            true_coords = []
            for flag in self.flags:
                if (
                    flag is not None
                    and flag.direction is not None
                    and flag.distance is not None
                    and flag.flag_id is not None
                ):
                    obs_coords = rel_to_abs_coords(flag)
                    rel_coords.append(obs_coords)
                    true_coords.append(true_flag_coords[flag.flag_id])

            for goal in self.goals:
                if (
                    goal is not None
                    and goal.direction is not None
                    and goal.distance is not None
                    and goal.goal_id is not None
                ):
                    obs_coords = rel_to_abs_coords(goal)
                    rel_coords.append(obs_coords)
                    true_coords.append(true_flag_coords[goal.goal_id])

            if len(true_coords) > 1 and len(rel_coords) > 1:
                # Get means
                rel_mean = np.mean(rel_coords, axis=0)
                true_mean = np.mean(true_coords, axis=0)

                mean_off = rel_mean - true_mean

                # Get rotation
                rel_de_mean = np.array(rel_coords) - rel_mean
                true_de_mean = np.array(true_coords) - true_mean

                if len(true_de_mean) > 1:
                    ang_offs = np.arctan2(
                        true_de_mean[:, 1], true_de_mean[:, 0]
                    ) - np.arctan2(rel_de_mean[:, 1], rel_de_mean[:, 0])
                    x, y = rad_rot_to_xy(ang_offs)

                    x_mean = np.mean(x)
                    y_mean = np.mean(y)

                    ang_offs = np.arctan2(y_mean, x_mean)

                    true_agent_loc = rotate(rel_mean, (0, 0), ang_offs) - mean_off
                    self.abs_coords = true_agent_loc
                    self.abs_body_dir = (ang_offs / math.pi) * 180

                    # TODO: Is this correct?
                    self.abs_neck_dir = self.abs_body_dir + self.neck_direction

    def process_new_info(self, ball, flags, goals, players, lines):
        """
        Update any internal variables based on the currently available
        information.  This also calculates information not available directly
        from server-reported messages, such as player coordinates.
        """

        # update basic information
        self.ball = ball
        self.flags = flags
        self.goals = goals
        self.players = players
        self.lines = lines

        self.__calculate_abs_info()

    def is_playon(self):
        """
        Tells us whether it's play time
        """
        return (
            self.play_mode == WorldModel.PlayModes.PLAY_ON
            or self.play_mode == WorldModel.PlayModes.KICK_OFF_L
            or self.play_mode == WorldModel.PlayModes.KICK_OFF_R
            or self.play_mode == WorldModel.PlayModes.KICK_IN_L
            or self.play_mode == WorldModel.PlayModes.KICK_IN_R
            or self.play_mode == WorldModel.PlayModes.FREE_KICK_L
            or self.play_mode == WorldModel.PlayModes.FREE_KICK_R
            or self.play_mode == WorldModel.PlayModes.CORNER_KICK_L
            or self.play_mode == WorldModel.PlayModes.CORNER_KICK_R
            or self.play_mode == WorldModel.PlayModes.GOAL_KICK_L
            or self.play_mode == WorldModel.PlayModes.GOAL_KICK_R
            or self.play_mode == WorldModel.PlayModes.DROP_BALL
            or self.play_mode == WorldModel.PlayModes.OFFSIDE_L
            or self.play_mode == WorldModel.PlayModes.OFFSIDE_R
        )

    def is_before_kick_off(self):
        """
        Tells us whether the game is in a pre-kickoff state.
        """

        return self.play_mode == WorldModel.PlayModes.BEFORE_KICK_OFF

    def is_kick_off_us(self):
        """
        Tells us whether it's our turn to kick off.
        """

        ko_left = WorldModel.PlayModes.KICK_OFF_L
        ko_right = WorldModel.PlayModes.KICK_OFF_R

        # return whether we're on the side that's kicking off
        return (
            self.side == WorldModel.SIDE_L
            and self.play_mode == ko_left
            or self.side == WorldModel.SIDE_R
            and self.play_mode == ko_right
        )

    def is_dead_ball_them(self):
        """
        Returns whether the ball is in the other team's posession and it's a
        free kick, corner kick, or kick in.
        """

        # shorthand for verbose constants
        kil = WorldModel.PlayModes.KICK_IN_L
        kir = WorldModel.PlayModes.KICK_IN_R
        fkl = WorldModel.PlayModes.FREE_KICK_L
        fkr = WorldModel.PlayModes.FREE_KICK_R
        ckl = WorldModel.PlayModes.CORNER_KICK_L
        ckr = WorldModel.PlayModes.CORNER_KICK_R

        # shorthand for whether left team or right team is free to act
        pm = self.play_mode
        free_left = pm == kil or pm == fkl or pm == ckl
        free_right = pm == kir or pm == fkr or pm == ckr

        # return whether the opposing side is in a dead ball situation
        if self.side == WorldModel.SIDE_L:
            return free_right
        else:
            return free_left

    def is_ball_kickable(self):
        """
        Tells us whether the ball is in reach of the current player.
        """

        # ball must be visible, not behind us, and within the kickable margin
        return (
            self.ball is not None
            and self.ball.distance is not None
            and self.ball.distance <= self.server_parameters.kickable_margin
        )

    def get_ball_speed_max(self):
        """
        Returns the maximum speed the ball can be kicked at.
        """

        return self.server_parameters.ball_speed_max

    def get_stamina(self):
        """
        Returns the agent's current stamina amount.
        """

        return self.stamina

    def get_stamina_max(self):
        """
        Returns the maximum amount of stamina a player can have.
        """

        return self.server_parameters.stamina_max

    def turn_body_to_object(self, obj):
        """
        Turns the player's body to face a particular object.
        """

        self.ah.turn(obj.direction)


class ServerParameters:
    """
    A storage container for all the settings of the soccer server.
    """

    def __init__(self):
        """
        Initialize default parameters for a server.
        """

        self.audio_cut_dist = 50
        self.auto_mode = 0
        self.back_passes = 1
        self.ball_accel_max = 2.7
        self.ball_decay = 0.94
        self.ball_rand = 0.05
        self.ball_size = 0.085
        self.ball_speed_max = 2.7
        self.ball_stuck_area = 3
        self.ball_weight = 0.2
        self.catch_ban_cycle = 5
        self.catch_probability = 1
        self.catchable_area_l = 2
        self.catchable_area_w = 1
        self.ckick_margin = 1
        self.clang_advice_win = 1
        self.clang_define_win = 1
        self.clang_del_win = 1
        self.clang_info_win = 1
        self.clang_mess_delay = 50
        self.clang_mess_per_cycle = 1
        self.clang_meta_win = 1
        self.clang_rule_win = 1
        self.clang_win_size = 300
        self.coach = 0
        self.coach_port = 6001
        self.coach_w_referee = 0
        self.connect_wait = 300
        self.control_radius = 2
        self.dash_power_rate = 0.006
        self.drop_ball_time = 200
        self.effort_dec = 0.005
        self.effort_dec_thr = 0.3
        self.effort_inc = 0.01
        self.effort_inc_thr = 0.6
        self.effort_init = 1
        self.effort_min = 0.6
        self.forbid_kick_off_offside = 1
        self.free_kick_faults = 1
        self.freeform_send_period = 20
        self.freeform_wait_period = 600
        self.fullstate_l = 0
        self.fullstate_r = 0
        self.game_log_compression = 0
        self.game_log_dated = 1
        self.game_log_dir = "./"
        self.game_log_fixed = 0
        self.game_log_fixed_name = "rcssserver"
        self.game_log_version = 3
        self.game_logging = 1
        self.game_over_wait = 100
        self.goal_width = 14.02
        self.goalie_max_moves = 2
        self.half_time = 300
        self.hear_decay = 1
        self.hear_inc = 1
        self.hear_max = 1
        self.inertia_moment = 5
        self.keepaway = 0
        self.keepaway_length = 20
        self.keepaway_log_dated = 1
        self.keepaway_log_dir = "./"
        self.keepaway_log_fixed = 0
        self.keepaway_log_fixed_name = "rcssserver"
        self.keepaway_logging = 1
        self.keepaway_start = -1
        self.keepaway_width = 20
        self.kick_off_wait = 100
        self.kick_power_rate = 0.027
        self.kick_rand = 0
        self.kick_rand_factor_l = 1
        self.kick_rand_factor_r = 1
        self.kickable_margin = 0.7
        self.landmark_file = "~/.rcssserver-landmark.xml"
        self.log_date_format = "%Y%m%d%H%M-"
        self.log_times = 0
        self.max_goal_kicks = 3
        self.maxmoment = 180
        self.maxneckang = 90
        self.maxneckmoment = 180
        self.maxpower = 100
        self.minmoment = -180
        self.minneckang = -90
        self.minneckmoment = -180
        self.minpower = -100
        self.nr_extra_halfs = 2
        self.nr_normal_halfs = 2
        self.offside_active_area_size = 2.5
        self.offside_kick_margin = 9.15
        self.olcoach_port = 6002
        self.old_coach_hear = 0
        self.pen_allow_mult_kicks = 1
        self.pen_before_setup_wait = 30
        self.pen_coach_moves_players = 1
        self.pen_dist_x = 42.5
        self.pen_max_extra_kicks = 10
        self.pen_max_goalie_dist_x = 14
        self.pen_nr_kicks = 5
        self.pen_random_winner = 0
        self.pen_ready_wait = 50
        self.pen_setup_wait = 100
        self.pen_taken_wait = 200
        self.penalty_shoot_outs = 1
        self.player_accel_max = 1
        self.player_decay = 0.4
        self.player_rand = 0.1
        self.player_size = 0.3
        self.player_speed_max = 1.2
        self.player_weight = 60
        self.point_to_ban = 5
        self.point_to_duration = 20
        self.port = 6000
        self.prand_factor_l = 1
        self.prand_factor_r = 1
        self.profile = 0
        self.proper_goal_kicks = 0
        self.quantize_step = 0.1
        self.quantize_step_l = 0.01
        self.record_messages = 0
        self.recover_dec = 0.002
        self.recover_dec_thr = 0.3
        self.recover_init = 1
        self.recover_min = 0.5
        self.recv_step = 10
        self.say_coach_cnt_max = 128
        self.say_coach_msg_size = 128
        self.say_msg_size = 10
        self.send_comms = 0
        self.send_step = 150
        self.send_vi_step = 100
        self.sense_body_step = 100
        self.simulator_step = 100
        self.slow_down_factor = 1
        self.slowness_on_top_for_left_team = 1
        self.slowness_on_top_for_right_team = 1
        self.stamina_inc_max = 45
        self.stamina_max = 4000
        self.start_goal_l = 0
        self.start_goal_r = 0
        self.stopped_ball_vel = 0.01
        self.synch_micro_sleep = 1
        self.synch_mode = 0
        self.synch_offset = 60
        self.tackle_back_dist = 0.5
        self.tackle_cycles = 10
        self.tackle_dist = 2
        self.tackle_exponent = 6
        self.tackle_power_rate = 0.027
        self.tackle_width = 1
        self.team_actuator_noise = 0
        self.text_log_compression = 0
        self.text_log_dated = 1
        self.text_log_dir = "./"
        self.text_log_fixed = 0
        self.text_log_fixed_name = "rcssserver"
        self.text_logging = 1
        self.use_offside = 1
        self.verbose = 0
        self.visible_angle = 90
        self.visible_distance = 3
        self.wind_ang = 0
        self.wind_dir = 0
        self.wind_force = 0
        self.wind_none = 0
        self.wind_rand = 0
        self.wind_random = 0
