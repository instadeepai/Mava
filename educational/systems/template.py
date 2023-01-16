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
from dataclasses import dataclass
import logging
from typing import Any
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("system", "test agent", "What agent is running.")
flags.DEFINE_string(
    "base_dir", "~/mava", "Base dir to store experiment data e.g. checkpoints."
)


@dataclass
class InitConfig:
    seed: int = 42


@dataclass
class EnvironmentConfig:
    name: str = "random"
    seed: int = 42


@dataclass
class AgentConfig:
    name: str = "random"
    seed: int = 42


def init(config=InitConfig()):
    """Init system.

    This would handle thing to be done upon system once in the beginning of a run, e.g. set random seeds.

    Args:
        config : init config.
    """

    return


def make_environment(config=EnvironmentConfig()) -> Any:
    """Init and return environment or wrapper.

    Args:
        config : env config.

    Returns:
        env or wrapper.
    """

    class Environment:
        pass

    logging.info(config)
    env = Environment()
    return env


def make_agents(config=AgentConfig()) -> Any:
    """Inits and returns agents/networks.

    Args:
        config : system config.

    Returns:
        agents.
    """

    class System:
        pass

    logging.info(config)
    system = System()
    return system


def main(_):

    init()
    env = make_environment()
    system = make_agents()
    logging.info(f"Running {FLAGS.system}")

    # Run agents on env
    del env, system


if __name__ == "__main__":
    app.run(main)
