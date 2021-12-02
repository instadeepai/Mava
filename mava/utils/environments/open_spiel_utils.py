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

try:
    from open_spiel.python import rl_environment  # type: ignore

    _has_open_spiel = True
except ModuleNotFoundError:
    _has_open_spiel = False
    pass


def load_open_spiel_env(game_name: str) -> "rl_environment.Environment":
    """Loads an open spiel environment given a game name Also, the possible agents in the
    environment are set"""
    if _has_open_spiel:
        env = rl_environment.Environment(game_name)
        env.agents = [f"player_{i}" for i in range(env.num_players)]
        env.possible_agents = env.agents[:]
    else:
        raise Exception("Openspiel is not installed.")

    return env
