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

"""Utils for the tests"""

from enum import Enum


class EnvSource(str, Enum):
    """Environment enumeration"""

    PettingZoo = "pettingzoo"
    RLLibMultiEnv = "rllibmultienv"
    Flatland = "flatland"


class MockedEnvironments(str, Enum):
    """Environment actions' type mocks"""

    Mocked_Dicrete = "discrete_mock"
    Mocked_Continous = "continous_mock"


class EnvSpec:
    """Environment spec"""

    def __init__(
        self,
        env_name: str,
        env_source: EnvSource = EnvSource.PettingZoo,
    ):
        """Init"""
        self.env_name = env_name
        self.env_source = env_source
