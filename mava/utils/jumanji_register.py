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
import json
import os
from typing import Any, Dict

# File path to store the generated jumanji's scenarios
_SCENARIOS_FILE_PATH = "../Mava/mava/configs/env/scenario/jumanji_scenarios.json"

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


class JumanjiScenarioManager:
    def __init__(self, env_name: str, task_name: str) -> None:
        self.env_name: str = env_name
        self.task_name: str = task_name
        self.file_path: str = _SCENARIOS_FILE_PATH
        self.scenarios: Dict[str, Dict[str, Any]] = self._load_scenarios()

        if self.env_name not in ENV_FUNCTIONS:
            raise ValueError(f"Unknown environment name: {env_name}")

    def _load_scenarios(self) -> Any:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        try:
            with open(self.file_path, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_scenarios(self) -> None:
        with open(self.file_path, "w") as file:
            json.dump(self.scenarios, file, indent=4)

    def register_environment(self) -> Any:
        """
        Registers a new environment and task in the scenarios JSON file.

        Args:
            env_name (str): The name of the environment to register.
            task_name (str): The name of the task to register.

        Returns:
            dict: A dictionary containing the attributes of the registered task.
        """

        if self.env_name not in self.scenarios:
            self.scenarios[self.env_name] = ENV_FUNCTIONS[self.env_name]["register"]()

        attributes = self.check_scenario()
        self.scenarios[self.env_name][self.task_name] = attributes
        self._save_scenarios()

        return attributes

    def check_scenario(self) -> Any:
        """Checks if a task is present in the JSON file and extracts its attributes.
        If the task is not found, it will be saved and its attributes extracted.
        """
        if self.task_name in self.scenarios[self.env_name]:
            # Check and extract the task's attributes
            attributes = self.scenarios[self.env_name][self.task_name]
            return attributes

        attributes = ENV_FUNCTIONS[self.env_name]["checker"](self.task_name)

        return attributes


def lbf_register_jumanji() -> Dict[str, Dict[str, Any]]:
    "Registers the LevelBasedForaging-v0 environment with various task configurations."

    # Define a limited ranges of values.
    grid_sizes = range(8, 16)
    num_agents = range(2, 5)
    num_food_items = range(2, 5)
    if_force_coop = [True, False]
    if_partial_observation = [True, False]

    # Generate scenarios by combining different values of the attributes
    # Construct the task name based on the attributes and use it as a key
    scenarios = {
        f"{'' if not fov else '2s-'}{grid_size}x{grid_size}-{n_agents}p-{n_food}f"
        + f"{'-coop' if force_coop else ''}": {
            # Individual scenario attributes
            "grid_size": grid_size,
            "fov": grid_size if not fov else 2,
            "num_agents": n_agents,
            "num_food": n_food,
            "force_coop": force_coop,
        }
        # Nested loops iterating over attribute values
        for grid_size in grid_sizes
        for n_agents in num_agents
        for n_food in num_food_items
        for force_coop in if_force_coop
        for fov in if_partial_observation
    }

    return scenarios


def lbf_check_task(task_name: str) -> Dict[str, Any]:
    "Checks if a task is valid and extracts its attributes for LBF."

    # Extract attributes based on valid ranges. E.g. "8x8-2p-2f-coop"
    grid_size = int(task_name.split("x")[0][-1])
    fov = int(task_name.split("s")[0][-1]) if "s" in task_name else grid_size
    num_agents = int(task_name.split("p")[0][-1])
    num_food = int(task_name.split("f")[0][-1])
    force_coop = "coop" in task_name.split("-")

    task_attributes = {
        "grid_size": grid_size,
        "fov": fov,
        "num_agents": num_agents,
        "num_food": num_food,
        "force_coop": force_coop,
    }

    # Validate attributes against the predefined ranges
    valid_attributes = all(
        LBF_ATTRIBUTES_RANGES[key].__contains__(value) for key, value in task_attributes.items()
    )

    if valid_attributes:
        # Add the task to scenarios with the extracted attributes
        return task_attributes
    else:
        raise ValueError(f"Invalid attributes for task: {task_name}")


def rware_register_jumanji() -> Dict[str, Dict[str, Any]]:
    "Registers the RobotWarehouse-v0 environment with various task configurations."

    # Define a limited ranges of values.
    num_agents = range(2, 5)
    sizes = {"tiny": (1, 3), "small": (2, 3)}
    difficulties = {"-easy": 2, "": 1}

    # Generate scenarios and construct the task name based on the attributes
    scenarios = {
        f"{size}-{n_agents}ag{difficulty}": {
            "column_height": 8,
            "shelf_rows": sizes[size][0],
            "shelf_columns": sizes[size][1],
            "num_agents": n_agents,
            "sensor_range": 1,
            "request_queue_size": int(n_agents * difficulties[difficulty]),
        }
        # Nested loops iterating over attribute values
        for size in sizes
        for n_agents in num_agents
        for difficulty in difficulties
    }

    return scenarios


def rware_check_task(task_name: str) -> Dict[str, Any]:
    "Checks if a task is valid and extracts its attributes for LBF."

    # Extract attributes based on valid ranges. E.g. "tiny-4ag-easy"
    task_name_list = task_name.split("-")
    size = task_name_list[0]
    num_agents = int(task_name_list[1].split("ag")[0])
    difficulty = task_name_list[2] if len(task_name_list) == 3 else ""

    # Validate attributes against the predefined ranges
    attributes = {"num_agents": num_agents, "difficulties": difficulty, "size": size}
    valid_attributes = all(
        RWARE_ATTRIBUTES_RANGES[key].__contains__(value) for key, value in attributes.items()
    )

    if valid_attributes:
        # Add the task to scenarios with the extracted attributes
        task_attributes = {
            "column_height": 8,
            "shelf_rows": RWARE_ATTRIBUTES_RANGES["size"][size][0],
            "shelf_columns": RWARE_ATTRIBUTES_RANGES["size"][size][1],
            "num_agents": num_agents,
            "sensor_range": 1,
            "request_queue_size": int(
                num_agents * RWARE_ATTRIBUTES_RANGES["difficulties"][difficulty]
            ),
        }
        return task_attributes
    else:
        raise ValueError(f"Invalid attributes for task: {task_name}")


ENV_FUNCTIONS: Dict[str, Dict[str, Any]] = {
    "LevelBasedForaging-v0": {"register": lbf_register_jumanji, "checker": lbf_check_task},
    "RobotWarehouse-v0": {"register": rware_register_jumanji, "checker": rware_check_task},
}
