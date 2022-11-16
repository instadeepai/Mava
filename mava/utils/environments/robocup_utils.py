# python3
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

"""Robocup environment factory."""

from typing import Dict, List, Tuple

import dm_env
import numpy as np

from mava.utils.environments.RoboCup_env.RoboCup2D_env import RoboCup2D
from mava.wrappers.robocup import RoboCupWrapper


def make_environment(
    game_name: str = "domain_randomisation",
    evaluation: bool = False,
) -> Tuple[dm_env.Environment, Dict[str, str]]:
    """Wraps the Robocup environment with some basic preprocessing.

    Args:
        game_name: str, the name of the Robocup game setting.
        evaluation: bool, to change the behaviour during evaluation.

    Returns:
        A Robocup environment with some standard preprocessing.
    """

    # Create environment
    if game_name == "domain_randomisation":
        players_per_team = [1, 0]
    elif game_name == "reward_shaping":
        players_per_team = [1, 0]
    # elif game_name == "fixed_opponent":
    #     players_per_team = [2, 2]
    else:
        raise NotImplementedError("Game type not implemented: ", game_name)

    # TODO: Change this to better assign ports
    rand_port = np.random.randint(6000, 60000)
    robocup_env = RoboCup2D(
        game_setting=game_name,
        team_names=["Team_A", "Team_B"],
        players_per_team=players_per_team,
        render_game=False,
        include_wait=False,
        game_length=1000,
        port=rand_port,
    )

    def _make_robocup_task_name(game_setting: str, players_per_team: List[int]) -> str:
        """A simple helper function to create a robocup task name.

        The task name will be a string created as:
        `<game_setting>_<num_players>_players`. For example in a
        reward_shaping game with 22 players in total the task name
        will be `reward_shaping_22_players`.
        """

        game_setting = str(game_setting)
        num_players = str(players_per_team[0] + players_per_team[1])

        task_name = f"{game_setting}_{num_players}_players"

        return task_name

    environment_task_name = {
        "environment_name": "robocup",
        "task_name": _make_robocup_task_name(
            game_setting=game_name, players_per_team=players_per_team
        ),
    }

    return RoboCupWrapper(robocup_env), environment_task_name
