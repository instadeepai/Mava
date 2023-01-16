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
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("system", "random agent", "What agent is running.")


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
    """
    Init system.

    This would handle thing to be done upon system once in the beginning, e.g. set random seeds or get config.
    """
    return


def make_environment(config=EnvironmentConfig()):
    logging.info(config)
    return


def make_agents(config=AgentConfig()):
    logging.info(config)
    return


def main(_):
    init()
    make_environment()
    make_agents()
    logging.info(f"Running {FLAGS.system}")


if __name__ == "__main__":
    app.run(main)
