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

# Mapping of environments to their respective action heads.
action_head_per_env = {
    "Cleaner": "mava.networks.heads.DiscreteActionHead",
    "MaConnector": "mava.networks.heads.DiscreteActionHead",
    "LevelBasedForaging": "mava.networks.heads.DiscreteActionHead",
    "Matrax": "mava.networks.heads.DiscreteActionHead",
    "RobotWarehouse": "mava.networks.heads.DiscreteActionHead",
    "Smax": "mava.networks.heads.DiscreteActionHead",
    "Gigastep": "mava.networks.heads.DiscreteActionHead",
    "MaBrax": "mava.networks.heads.ContinuousActionHead",
}


def get_action_head(env_name: str) -> dict:
    """Returns the appropriate action head config based on the environment name."""
    action_head = action_head_per_env.get(env_name)

    if action_head is None:
        raise ValueError(f"Environment {env_name} is not recognized.")

    return {"_target_": action_head}
