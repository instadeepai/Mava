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


from typing import Any, Callable, Dict

# Define the ranges of valid attributes
LBF_ATTRIBUTES_RANGES: Dict[str, Any] = {
    "grid_size": range(5, 20),  # Grid size, the height and width are equal.
    "fov": range(2, 20),  # Field of view of agent.
    "num_agents": range(2, 20),
    "num_food": range(1, 10),
    "force_coop": [True, False],  # Force cooperation between agents.
}

RWARE_ATTRIBUTES_RANGES: Dict[str, Any] = {
    "num_agents": range(1, 20),
    "difficulties": {"-easy": 2, "": 1, "-hard": 0.5},
    "size": {
        "tiny": (1, 3),
        "small": (2, 3),
        "medium": (2, 5),
        "large": (3, 5),
    },
}


def get_task_config(env_name: str, scenario: str) -> Any:
    """
    Retrieves the attributes for a given environment and scenario.

    Args:
        env_name (str): The name of the environment.
        scenario (str): The scenario to retrieve attributes for.

    Returns:
        dict: A dictionary containing the attributes of the scenario.
    """
    if env_name not in ENV_FUNCTIONS:
        raise ValueError(f"Unknown environment name: {env_name}")

    # Extract and return scenario attributes
    env_fn = ENV_FUNCTIONS[env_name]
    return env_fn(scenario)


def get_lbf_config(scenario: str) -> Dict[str, Any]:
    """
    Checks if a scenario is valid and extracts its attributes for LevelBasedForaging.

    Args:
        scenario (str): The scenario to retrieve attributes for.

    Returns:
        dict: A dictionary containing the attributes of the scenario.
    """
    # Extract attributes based on valid ranges. E.g. "2s-10x10-3p-3f-coop"
    grid_size = int(scenario.split("x")[0].split("-")[-1])
    fov = int(scenario.split("s")[0]) if "s" in scenario else grid_size
    num_agents = int(scenario.split("p")[0].split("-")[-1])
    num_food = int(scenario.split("f")[0][-1])
    force_coop = "coop" in scenario.split("-")

    scenario_attributes = {
        "grid_size": grid_size,
        "fov": fov,
        "num_agents": num_agents,
        "num_food": num_food,
        "force_coop": force_coop,
    }

    # Validate attributes against the predefined ranges
    valid_attributes = all(
        LBF_ATTRIBUTES_RANGES[key].__contains__(value) for key, value in scenario_attributes.items()
    )

    if not valid_attributes:
        raise ValueError(f"Invalid attributes for scenario: {scenario}")

    # Get and return the scenario attributes
    scenario_attributes["max_agent_level"] = 2
    return scenario_attributes


def get_rware_config(scenario: str) -> Dict[str, Any]:
    """
    Checks if a scenario is valid and extracts its attributes for RobotWarehouse.

    Args:
        scenario (str): The scenario to retrieve attributes for.

    Returns:
        dict: A dictionary containing the attributes of the scenario.
    """
    # Extract attributes based on valid ranges. E.g. "tiny-4ag-easy"
    scenario_list = scenario.split("-")
    size = scenario_list[0]
    num_agents = int(scenario_list[1].split("ag")[0])
    difficulty = scenario_list[2] if len(scenario_list) == 3 else ""

    # Validate attributes against the predefined ranges
    attributes = {"num_agents": num_agents, "difficulties": difficulty, "size": size}
    valid_attributes = all(
        RWARE_ATTRIBUTES_RANGES[key].__contains__(value) for key, value in attributes.items()
    )

    if not valid_attributes:
        raise ValueError(f"Invalid attributes for scenario: {scenario}")

    return {
        "column_height": 8,
        "shelf_rows": RWARE_ATTRIBUTES_RANGES["size"][size][0],
        "shelf_columns": RWARE_ATTRIBUTES_RANGES["size"][size][1],
        "num_agents": num_agents,
        "sensor_range": 1,
        "request_queue_size": int(num_agents * RWARE_ATTRIBUTES_RANGES["difficulties"][difficulty]),
    }


ENV_FUNCTIONS: Dict[str, Callable] = {
    "LevelBasedForaging-v0": get_lbf_config,
    "RobotWarehouse-v0": get_rware_config,
}
