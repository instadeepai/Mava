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

import dm_env

from mava.specs import make_ma_environment_spec


class AgentGroup:
    def __init__(self, environment: dm_env.Environment) -> None:
        self._environment = environment

    def get_agent_groups(self) -> list:
        group_names = []
        for agent in self._environment.possible_agents:
            group_name = agent.split("_")[0]
            group_names.append(group_name)
        return list(set(group_names))

    def get_agent_group_specs(self) -> dict:
        groups = self.get_agent_groups()
        env_spec = make_ma_environment_spec(self._environment)
        group_specs = {}
        for group in groups:
            group_specs[group] = env_spec[f"{group}_0"]
        return group_specs
