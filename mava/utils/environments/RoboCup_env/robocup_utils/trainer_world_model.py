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
class WorldModel:
    """
    Holds and updates the model of the world as known from current and past
    data.
    """

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
        self.goals = []
        self.players = []

        # scores for each side
        self.score_l = 0
        self.score_r = 0

        # the name of the agent's team
        self.teamnames = [None, None]

        # stores the most recent message heard
        self.last_message = None

        self.new_global_data = False

        # the mode the game is currently in (default to not playing yet)
        self.play_mode = WorldModel.PlayModes.BEFORE_KICK_OFF

        # create a new server parameter object for holding all server params
        self.server_parameters = ServerParameters()

    def process_new_info(self, ball, goals, players):
        """
        Update any internal variables based on the currently available
        information.  This also calculates information not available directly
        from server-reported messages, such as player coordinates.
        """

        # update basic information
        # Fix state
        # TODO: Fix state offset problems. Why is the state values different?

        # Correct ball y coords
        ball["coords"] = (float(ball["coords"][0]), -float(ball["coords"][1]))
        ball["delta_coords"] = (
            float(ball["delta_coords"][0]),
            -float(ball["delta_coords"][1]),
        )
        for i in range(len(players)):
            # Correct y coords
            players[i]["coords"] = (
                float(players[i]["coords"][0]),
                -float(players[i]["coords"][1]),
            )
            players[i]["delta_coords"] = (
                float(players[i]["delta_coords"][0]),
                -float(players[i]["delta_coords"][1]),
            )
            # Correct body angle
            players[i]["body_angle"] = -float(players[i]["body_angle"]) + 90

            # Correct neck angle
            players[i]["neck_angle"] = -float(players[i]["neck_angle"]) + 90

            # Get side
            team_to_id = {"Team_A": 0, "Team_B": 1}
            players[i]["side"] = team_to_id[players[i]["teamname"]]

        self.ball = ball
        self.goals = goals
        self.players = players

    def get_state(self):
        return {"ball": self.ball, "players": self.players}

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
        print("is_before_kick_off!")
        return self.play_mode == WorldModel.PlayModes.BEFORE_KICK_OFF

    def is_kick_off_us(self):
        """
        Tells us whether it's our turn to kick off.
        """
        print("is_kick_off_us!")
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
