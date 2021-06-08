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
class GameObject:
    """
    Root class for all percievable objects in the world model.
    """

    def __init__(self, distance, direction):
        """
        All objects have a distance and direction to the player, at a minimum.
        """

        self.distance = distance
        self.direction = direction


class Line(GameObject):
    """
    Represents a line on the soccer field.
    """

    def __init__(self, distance, direction, line_id):
        self.line_id = line_id

        GameObject.__init__(self, distance, direction)


class Goal(GameObject):
    """
    Represents a goal object on the field.
    """

    def __init__(self, distance, direction, goal_id):
        self.goal_id = goal_id

        GameObject.__init__(self, distance, direction)


class Flag(GameObject):
    """
    A flag on the field.  Can be used by the agent to determine its position.
    """

    # a dictionary mapping all flag_ids to their on-field (x, y) coordinates
    # TODO: Add real values form code here and do not use estimates.
    # TODO: these are educated guesses based on Figure 4.2 in the documentation.
    #       where would one find the actual coordinates, besides in the server
    #       code?
    # Location:
    # https://github.com/rcsoccersim/rcssserver/blob/master/src/landmarkreader.cpp
    # https://rcsoccersim.github.io/manual/soccerserver.html Fig 4.2

    pitch_x = 52.5
    pitch_y = 34.0
    outside_add = 5
    goal_y = 7.01
    pen_x = 36.0
    pen_y = 20.16

    out_x = pitch_x + outside_add
    out_y = pitch_y + outside_add

    FLAG_COORDS = {
        # perimiter flags
        # center flag
        "c": (0, 0),
        "ct": (0, pitch_y),
        "cb": (0, -pitch_y),
        # field boundary flags (on boundary lines)
        "r": (pitch_x, 0),
        "rt": (pitch_x, pitch_y),
        "rb": (pitch_x, -pitch_y),
        "l": (-pitch_x, 0),
        "lt": (-pitch_x, pitch_y),
        "lb": (-pitch_x, -pitch_y),
        # goal flags ('t' and 'b' flags can change based on server parameter
        # 'goal_width', but we leave their coords as the default values.
        # TODO: make goal flag coords dynamic based on server_params
        "grb": (pitch_x, -goal_y),
        "gr": (pitch_x, 0),
        "grt": (pitch_x, goal_y),
        "glb": (-pitch_x, -goal_y),
        "gl": (-pitch_x, 0),
        "glt": (-pitch_x, goal_y),
        # penalty flags
        "prb": (pen_x, -pen_y),
        "prc": (pen_x, 0),
        "prt": (pen_x, pen_y),
        "plb": (-pen_x, -pen_y),
        "plc": (-pen_x, 0),
        "plt": (-pen_x, pen_y),
        "t0": (0, out_y),
        "tr10": (10, out_y),
        "tr20": (20, out_y),
        "tr30": (30, out_y),
        "tr40": (40, out_y),
        "tr50": (50, out_y),
        "tl10": (-10, out_y),
        "tl20": (-20, out_y),
        "tl30": (-30, out_y),
        "tl40": (-40, out_y),
        "tl50": (-50, out_y),
        "b0": (0, -out_y),
        "br10": (10, -out_y),
        "br20": (20, -out_y),
        "br30": (30, -out_y),
        "br40": (40, -out_y),
        "br50": (50, -out_y),
        "bl10": (-10, -out_y),
        "bl20": (-20, -out_y),
        "bl30": (-30, -out_y),
        "bl40": (-40, -out_y),
        "bl50": (-50, -out_y),
        "r0": (out_x, 0),
        "rt10": (out_x, 10),
        "rt20": (out_x, 20),
        "rt30": (out_x, 30),
        "rb10": (out_x, -10),
        "rb20": (out_x, -20),
        "rb30": (out_x, -30),
        "l0": (-out_x, 0),
        "lt10": (-out_x, 10),
        "lt20": (-out_x, 20),
        "lt30": (-out_x, 30),
        "lb10": (-out_x, -10),
        "lb20": (-out_x, -20),
        "lb30": (-out_x, -30),
    }

    def __init__(self, distance, direction, flag_id):
        """
        Adds a flag id for this field object.  Every flag has a unique id.
        """

        self.flag_id = flag_id

        GameObject.__init__(self, distance, direction)


class MobileObject(GameObject):
    """
    Represents objects that can move.
    """

    def __init__(self, distance, direction, dist_change, dir_change, speed):
        """
        Adds variables for distance and direction deltas.
        """

        self.dist_change = dist_change
        self.dir_change = dir_change
        self.speed = speed

        GameObject.__init__(self, distance, direction)


class Ball(MobileObject):
    """
    A spcial instance of a mobile object representing the soccer ball.
    """

    def __init__(self, distance, direction, dist_change, dir_change, speed):
        MobileObject.__init__(self, distance, direction, dist_change, dir_change, speed)


class Player(MobileObject):
    """
    Represents a friendly or enemy player in the game.
    """

    def __init__(
        self,
        distance,
        direction,
        dist_change,
        dir_change,
        speed,
        team,
        side,
        uniform_number,
        body_direction,
        neck_direction,
    ):
        """
        Adds player-specific information to a mobile object.
        """

        self.team = team
        self.side = side
        self.uniform_number = uniform_number
        self.body_direction = body_direction
        self.neck_direction = neck_direction

        MobileObject.__init__(self, distance, direction, dist_change, dir_change, speed)
