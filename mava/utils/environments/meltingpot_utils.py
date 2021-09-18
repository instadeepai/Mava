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

from typing import Any

try:
    from meltingpot.python import scenario, substrate  # type: ignore
    from meltingpot.python.scenario import Scenario  # type: ignore
    from meltingpot.python.substrate import Substrate  # type: ignore
    from ml_collections import config_dict  # type: ignore
except ModuleNotFoundError:
    Scenario = Any
    Substrate = Any


def load_substrate(substrate_name: str) -> Substrate:
    """Loads a substrate from the available substrates

    Args:
        substrate_name (str): substrate name

    Returns:
        Substrate: A multi-agent environment
    """
    config = substrate.get_config(substrate_name)
    env_config = config_dict.ConfigDict(config)

    return substrate.build(env_config)


def load_scenario(scenario_name: str) -> Scenario:
    """Loads a substrate from the available substrates

    Args:
        scenerio_name (str): scenario name

    Returns:
        Scenario: A multi-agent environment with background bots
    """
    config = scenario.get_config(scenario_name)
    env_config = config_dict.ConfigDict(config)

    return scenario.build(env_config)
