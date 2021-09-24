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

from typing import Any, Union

try:
    from meltingpot.python import scenario, substrate  # type: ignore
    from meltingpot.python.scenario import AVAILABLE_SCENARIOS, Scenario  # type: ignore
    from meltingpot.python.substrate import (  # type: ignore
        AVAILABLE_SUBSTRATES,
        Substrate,
    )
    from ml_collections import config_dict  # type: ignore
except ModuleNotFoundError:
    Scenario = Any
    Substrate = Any


class EnvironmentFactory:
    def __init__(
        self, substrate: str = None, scenario: str = None, mode: str = "substrate"
    ):
        """Initializes the env factory object

        Args:
            substrate (str, optional): what substrate to use. Defaults to None.
            scenario (str, optional): what scenario to use. Defaults to None.
            mode (str, optional): what mode to use either 'scenario' or 'substrate'.
            Defaults to "substrate".
        """
        assert mode in [
            "substrate",
            "scenario",
        ], f"mode can either be 'substrate' or 'scenario' not {mode}"
        if mode == "substrate":
            substrates = [*AVAILABLE_SUBSTRATES, "all"]
            assert (
                substrate in substrates
            ), f"substrate cannot be f{substrate}, use any of {substrates}"

            if substrate == "all":
                self.available_substrates = AVAILABLE_SUBSTRATES
            else:
                self.available_substrates = [substrate]
            self._env_fn = self._substrate

        if mode == "scenario":
            scenarios = [*[k for k in AVAILABLE_SCENARIOS], "all"]
            assert (
                scenario in scenarios
            ), f"substrate cannot be f{substrate}, use any of {scenarios}"
            if scenario == "all":
                self.available_scenarios = AVAILABLE_SCENARIOS
            else:
                self.available_scenarios = [scenario]
            self._env_fn = self._scenario

    def _substrate(self) -> Substrate:
        """Return a substrate as an environment

        Returns:
            [Substrate]: A substrate or None
        """
        if self.available_substrates:
            name = self.available_substrates.pop()
            env = load_substrate(name)
            return env
        else:
            return None

    def _scenario(self) -> Scenario:
        """Return a scenario as an environment

        Returns:
            [Scenario]: A scenario or None
        """
        if self.available_scenarios:
            name = self.available_scenarios.pop()
            env = load_scenario(name)
            return env
        else:
            return None

    def __call__(self, evaluation: bool = False) -> Union[Substrate, Scenario]:
        """Creates an environment

        Returns:
            [type]: The created environment
        """
        env = self._env_fn()  # type: ignore
        return env


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
