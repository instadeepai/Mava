# !/usr/bin/env python3
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

import random
import threading
import time

import numpy as np

from mava.utils.environments.RoboCup_env.robocup_utils import (
    handler,
    sock,
    sp_exceptions,
)
from mava.utils.environments.RoboCup_env.robocup_utils.game_object import Flag
from mava.utils.environments.RoboCup_env.robocup_utils.trainer_world_model import (
    WorldModel,
)

max_x_rand = Flag.out_x
max_y_rand = Flag.out_y

# Define some colours
green = (0, 255, 50)
light_blue = (0, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
yellow = (255, 255, 0)
white = (255, 255, 255)

# R G B
field_col = (0, 200, 50)
team_player = blue
opp_player = light_blue
own_goal = (255, 140, 0)
opp_goal = (128, 0, 128)


class Trainer:
    def __init__(self):
        # whether we're connected to a server yet or not
        self.__connected = False

        # set all variables and important objects to appropriate values for
        # pre-connect state.

        # the socket used to communicate with the server
        self.__sock = None

        # models and the message handler for parsing and storing information
        self.wm = None
        self.msg_handler = None

        # parse thread and control variable
        self.__parsing = False
        self.__msg_thread = None

        self.__thinking = False  # think thread and control variable
        self.__think_thread = None

        # whether we should run the think method
        self.__should_think_on_data = False

        # whether we should send commands
        self.__send_commands = True

        # adding goal post markers
        self.enemy_goal_pos = None
        self.own_goal_pos = None

    def connect(self, host, port, version=11):
        """
        Gives us a connection to the server as one player on a team.  This
        immediately connects the agent to the server and starts receiving and
        parsing the information it sends.
        """

        # if already connected, raise an error since user may have wanted to
        # connect again to a different server.
        if self.__connected:
            msg = "Cannot connect while already connected, disconnect first."
            raise sp_exceptions.AgentConnectionStateError(msg)

        # the pipe through which all of our communication takes place
        self.__sock = sock.Socket(host, port)

        # our models of the world and our body
        self.wm = WorldModel(handler.ActionHandler(self.__sock))

        # handles all messages received from the server
        self.msg_handler = handler.MessageHandler(self.wm)

        # set up our threaded message receiving system
        self.__parsing = True  # tell thread that we're currently running
        self.__msg_thread = threading.Thread(
            target=self.__message_loop, name="message_loop"
        )
        self.__msg_thread.daemon = True  # dies when parent thread dies

        # start processing received messages. this will catch the initial server
        # response and all subsequent communication.
        self.__msg_thread.start()

        # send the init message and allow the message handler to handle further
        # responses.
        # init_address = self.__sock.address
        # teamname = "John"
        init_msg = "(init (version %d))"
        self.__sock.send(init_msg % (version))

        # # wait until the socket receives a response from the server and gets its
        # # assigned port.
        # while self.__sock.address == init_address:
        #     time.sleep(0.0001)

        # create our thinking thread.  this will perform the actions necessary
        # to play a game of robo-soccer.
        self.__thinking = False
        # self.__think_thread = threading.Thread(target=self.__think_loop,
        #         name="think_loop")
        # self.__think_thread.daemon = True

        # set connected state.  done last to prevent state inconsistency if
        # something goes wrong beforehand.
        self.__connected = True
        time.sleep(1)
        self.__sock.send("(eye on)")
        self.__sock.send("(ear on)")

    def run(self):
        """
        Kicks off the thread that does the agent's thinking, allowing it to play
        during the game.  Throws an exception if called while the agent is
        already playing.
        """

        # ensure we're connected before doing anything
        if not self.__connected:
            msg = "Must be connected to a server to begin play."
            raise sp_exceptions.AgentConnectionStateError(msg)

        # throw exception if called while thread is already running
        if self.__thinking:
            raise sp_exceptions.AgentAlreadyPlayingError("Agent is already playing.")

        # run the method that sets up the agent's persistant variables
        self.setup_environment()

        # tell the thread that it should be running, then start it
        self.__thinking = True
        self.__should_think_on_data = True
        self.__think_loop()

    # def disconnect(self):
    #     """
    #     Tell the loop threads to stop and signal the server that we're
    #     disconnecting, then join the loop threads and destroy all our inner
    #     methods.

    #     Since the message loop thread can conceiveably block indefinitely while
    #     waiting for the server to respond, we only allow it (and the think loop
    #     for good measure) a short time to finish before simply giving up.

    #     Once an agent has been disconnected, it is 'dead' and cannot be used
    #     again.  All of its methods get replaced by a method that raises an
    #     exception every time it is called.
    #     """

    #     # don't do anything if not connected
    #     if not self.__connected:
    #         return

    #     # tell the loops to terminate
    #     self.__parsing = False
    #     self.__thinking = False

    #     # tell the server that we're quitting
    #     self.__sock.send("(bye)")

    #     # tell our threads to join, but only wait breifly for them to do so.
    #     # don't join them if they haven't been started (this can happen if
    #     # disconnect is called very quickly after connect).
    #     if self.__msg_thread.is_alive():
    #         self.__msg_thread.join(0.01)

    #     # if self.__think_thread.is_alive():
    #     #     self.__think_thread.join(0.01)

    #     # reset all standard variables in this object.  self.__connected gets
    #     # reset here, along with all other non-user defined internal variables.
    #     Agent.__init__(self)

    def __message_loop(self):
        """
        Handles messages received from the server.

        This SHOULD NOT be called externally, since it's used as a threaded loop
        internally by this object.  Calling it externally is a BAD THING!
        """

        # loop until we're told to stop
        while self.__parsing:
            # receive message data from the server and pass it along to the
            # world model as-is.  the world model parses it and stores it within
            # itself for perusal at our leisure.
            raw_msg = self.__sock.recv()

            msg_type = self.msg_handler.handle_message(raw_msg)

            # if b'goal_l' in raw_msg or b'goal_r' in raw_msg:
            # print("Trainer message: '" , raw_msg,
            # ". Scores: ", self.wm.score_l, self.wm.score_r)
            # exit()

            # if msg_type is not None:
            # print(type(raw_msg))

            # we send commands all at once every cycle, ie. whenever a
            # 'sense_body' command is received
            if msg_type == handler.ActionHandler.CommandType.SENSE_BODY:
                self.__send_commands = True

            # flag new data as needing the think loop's attention
            self.__should_think_on_data = True

    # def __think_loop(self):
    #     """
    #     Performs world model analysis and sends appropriate commands to the
    #     server to allow the agent to participate in the current game.
    #
    #     Like the message loop, this SHOULD NOT be called externally.  Use the
    #     play method to start play, and the disconnect method to end it.
    #     """
    #
    #     while self.__thinking:
    #         # tell the ActionHandler to send its enqueued messages if it is time
    #         if self.__send_commands:
    #             self.__send_commands = False
    #             self.wm.ah.send_commands()
    #
    #         # only think if new data has arrived
    #         if self.__should_think_on_data:
    #             # flag that data has been processed.  this shouldn't be a race
    #             # condition, since the only change would be to make it True
    #             # before changing it to False again, and we're already going to
    #             # process data, so it doesn't make any difference.
    #             self.__should_think_on_data = False
    #
    #             self.think()
    #         else:
    #             # prevent from burning up all the cpu time while waiting for data
    #             time.sleep(0.0001)

    def setup_environment(self):
        """
        Called before the think loop starts, this allows the user to store any
        variables/objects they'll want access to across subsequent calls to the
        think method.
        """

        self.in_kick_off_formation = False

    def send_done(self):
        self.__sock.send("(done)")

    def reset_game(
        self,
        players_per_team,
        team_names,
        game_setting,
        game_diff=None,
        reset_stamina=True,
    ):
        self.__sock.send("(change_mode before_kick_off)")
        if reset_stamina:
            self.__sock.send("(recover)")
        # github.com/rcsoccersim/rcssserver/blob/master/src/coach.cpp

        # game_easy = 1 - game_diff
        if game_setting == "domain_randomisation":
            # print("game_diff: ", game_diff)
            assert game_diff is not None

            # Move the ball

            if players_per_team[0] > 0 and players_per_team[1] > 0:
                rand_side = random.randint(0, 1) * 2 - 1
            elif players_per_team[0] > 0:
                rand_side = 1
            else:
                rand_side = 0
            s_x = 52
            # TODONE: Change this back to the harder setting!
            x = str(
                np.random.randint(
                    s_x - (s_x + max_x_rand) * game_diff,
                    s_x - (s_x - max_x_rand) * game_diff + 1,
                )
                * rand_side
            )
            # str(np.random.randint(s_x - (s_x + max_x_rand)*game_diff,
            # s_x - (s_x - max_x_rand)*game_diff + 1)*rand_side)
            # str(s_x*rand_side)
            y = str(
                np.random.randint(-max_y_rand * game_diff, 1 + max_y_rand * game_diff)
            )
            # str(np.random.randint(-max_y_rand*game_diff, 1 + max_y_rand*game_diff))
            # str(0)  #
            self.__sock.send("(move (ball) " + x + " " + y + ")")
            for t_i in range(len(players_per_team)):
                for p_i in range(1, players_per_team[t_i] + 1):
                    s_x = 51
                    x = str(
                        np.random.randint(
                            s_x - (s_x + max_x_rand) * game_diff,
                            s_x - (s_x - max_x_rand) * game_diff + 1,
                        )
                        * (1 - t_i * 2)
                    )
                    y = str(
                        np.random.randint(
                            -max_y_rand * game_diff, 1 + max_y_rand * game_diff
                        )
                    )
                    ang = str(np.random.randint(-180 * game_diff, 1 + 180 * game_diff))

                    command = (
                        "(move (player "
                        + team_names[t_i]
                        + " "
                        + str(p_i)
                        + ") "
                        + str(x)
                        + " "
                        + str(y)
                        + " "
                        + str(ang)
                        + " 0 0)"
                    )
                    self.__sock.send(command)
        elif game_setting == "reward_shaping":
            # Move the ball
            x = str(np.random.randint(-max_x_rand, max_x_rand + 1))
            y = str(np.random.randint(-max_y_rand, max_y_rand + 1))
            self.__sock.send("(move (ball) " + x + " " + y + ")")

            for t, team_name in enumerate(team_names):
                omt = 1 - t
                for p_i in range(1, players_per_team[0] + 1):
                    x = str(np.random.randint(-51 * omt, 51 * t + 1))
                    y = str(np.random.randint(-max_y_rand, max_y_rand + 1))
                    ang = str(np.random.randint(-180, 180 + 1))

                    command = (
                        "(move (player "
                        + team_name
                        + " "
                        + str(p_i)
                        + ") "
                        + str(x)
                        + " "
                        + str(y)
                        + " "
                        + str(ang)
                        + " 0 0)"
                    )
                    self.__sock.send(command)

        else:
            raise NotImplementedError(
                "Game setting not implemented in trainer: ", game_setting
            )
        self.__sock.send("(change_mode play_on)")
        self.__sock.send("(start)")
        self.__sock.send("(done)")

    def get_state_dict(self):
        return self.wm.get_state()
