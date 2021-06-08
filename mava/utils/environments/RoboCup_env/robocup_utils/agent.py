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


import threading
import time

import numpy as np

from mava.utils.environments.RoboCup_env.robocup_utils import (
    handler,
    sock,
    sp_exceptions,
)
from mava.utils.environments.RoboCup_env.robocup_utils.player_world_model import (
    WorldModel,
)
from mava.utils.environments.RoboCup_env.robocup_utils.util_functions import (
    SpecWrapper,
    wait_for_next_observations,
)

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


class Agent:
    def __init__(self, teamname, team_id, agent_id, num_players, agent_controller=None):
        # whether we're connected to a server yet or not
        self.__connected = False

        self.teamname = teamname
        self.agent_id = agent_id
        self.team_id = team_id
        self.agent_controller = agent_controller

        if self.agent_controller is not None:
            self.specWrapper = SpecWrapper(num_players=num_players)

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

        # whether we should send commands
        self.__send_commands = False

        # adding goal post markers
        self.enemy_goal_pos = None
        self.own_goal_pos = None

    def connect(self, host, port, version=11):
        """
        Gives us a connection to the server as one player on a team.  This
        immediately connects the agent to the server and starts receiving and
        parsing the information it sends.
        """
        self.start_comm_threads(host, port, self.teamname, version)

    def start_comm_threads(self, host, port, teamname, version):
        # if already connected, raise an error since user may have wanted to
        # connect again to a different server.
        if self.__connected:
            msg = "Cannot connect while already connected, disconnect first."
            raise sp_exceptions.AgentConnectionStateError(msg)

        # the pipe through which all of our communication takes place
        # p rint("Start comms: ", port)
        self.__sock = sock.Socket(host, port)

        # our models of the world and our body
        self.wm = WorldModel(handler.ActionHandler(self.__sock))

        # set the team name of the world model to the given name
        self.wm.teamname = teamname

        # handles all messages received from the server
        self.msg_handler = handler.MessageHandler(self.wm)

        # set up our threaded message receiving system
        self.__parsing = True  # tell thread that we're currently running

        init_address = self.__sock.address
        # send the init message and allow the message handler to handle further
        # responses.
        init_msg = "(init %s (version %d))"
        self.__sock.send(init_msg % (teamname, version))

        self.__message_tread = threading.Thread(
            target=self.__message_loop, name="message_loop"
        )
        self.__message_tread.daemon = True
        self.__message_tread.start()
        # wait until the socket receives a response from the server and gets its
        # assigned port.
        while self.__sock.address == init_address:
            time.sleep(0.0001)

        # create our thinking thread.  this will perform the actions necessary
        # to play a game of robo-soccer.
        self.__thinking = True

        # set connected state.  done last to prevent state inconsistency if
        # something goes wrong beforehand.
        self.__connected = True

        self.last_action = None

        # Move to random starting position
        x = np.random.randint(-52, 53)
        y = np.random.randint(-34, 35)

        self.__sock.send("(move " + str(x) + " " + str(y) + ")")
        self.__sock.send("(done)")

        if self.agent_controller is not None:
            self.__think_thread = threading.Thread(
                target=self.__think_loop, name="think_loop"
            )
            self.__think_thread.daemon = True
            self.__think_thread.start()

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
            if msg_type is not None:
                # we send commands all at once every cycle, ie. whenever a
                # 'sense_body' command is received
                if msg_type == handler.ActionHandler.CommandType.SENSE_BODY:
                    self.__send_commands = True

    def __think_loop(self):
        """
        Performs world model analysis and sends appropriate commands to the
        server to allow the agent to participate in the current game.

        Like the message loop, this SHOULD NOT be called externally.  Use the
        play method to start play, and the disconnect method to end it.
        """

        while self.__thinking:
            # Get and process the latest observation
            observation = self.specWrapper.proc_agent_env_obs(
                env_agent_obs=self.get_latest_observation(),
                last_action=self.last_action,
            )

            # Get the agent's action and covert it to a RoboCup environment command.

            self.last_action = self.agent_controller.get_action(observation)
            action = self.specWrapper.proc_agent_action(self.last_action)

            # Send the action
            self.__sock.send(action)
            self.__sock.send("(done)")

            # Wait for a new timestep indicator
            wait_for_next_observations([self])

    def get_latest_observation(self):
        # Generate new observation
        obs_dict = {}
        obs_dict["players"] = self.wm.players

        # Ball
        obs_dict["ball"] = self.wm.ball

        # Team side
        obs_dict["side"] = self.wm.side

        # Score
        obs_dict["scores"] = [self.wm.score_l, self.wm.score_r]

        obs_dict["view_width"] = self.wm.view_width
        obs_dict["view_quality"] = self.wm.view_quality
        obs_dict["stamina"] = self.wm.stamina
        obs_dict["effort"] = self.wm.effort
        obs_dict["speed_amount"] = self.wm.speed_amount

        # TODO: Is these two measurements relative? I think they are.
        obs_dict["speed_direction"] = self.wm.speed_direction
        obs_dict["neck_direction"] = self.wm.neck_direction

        # self.wm.obs_updated = False
        obs_dict["estimated_abs_coords"] = self.wm.abs_coords
        obs_dict["estimated_abs_body_dir"] = self.wm.abs_body_dir
        obs_dict["estimated_abs_neck_dir"] = self.wm.abs_neck_dir

        # Fix observations so that it is side almost agnostic. A agent can
        # be trained on one side and still execute what it has learned on
        # the other side of the field.
        if obs_dict["side"] == "r":
            if obs_dict["estimated_abs_coords"][0] is not None:
                obs_dict["estimated_abs_coords"] = [
                    -obs_dict["estimated_abs_coords"][0],
                    -obs_dict["estimated_abs_coords"][1],
                ]
                obs_dict["estimated_abs_neck_dir"] += 180
                obs_dict["estimated_abs_body_dir"] += 180

        else:
            assert obs_dict["side"] == "l"

        return obs_dict

    def do_action(self, action):
        # tell the ActionHandler to send its enqueued messages if it is time
        # only think if new data has arrived
        if self.agent_controller:
            action = self.agent_controller.get_action(
                self.get_latest_observation(), None
            )
            print("When is this actually used?")
            exit()

        self.__sock.send(action)
        self.__sock.send("(done)")
