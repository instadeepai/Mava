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
import logging
from dataclasses import dataclass
from typing import Any, Tuple

from absl import app, flags

from mava import specs as mava_specs
from mava.utils.environments import debugging_utils

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
    env_name: str = "simple_spread"
    seed: int = 42
    type: str = "debug"
    action_space: str = "discrete"


@dataclass
class SystemConfig:
    name: str = "random"
    seed: int = 42


def init(config: InitConfig = InitConfig()) -> InitConfig:
    """Init system.

    This would handle thing to be done upon system once in the beginning of a run,
    e.g. set random seeds.

    Args:
        config : init config.

    Returns:
        config.
    """

    return config


def make_environment(
    config: EnvironmentConfig = EnvironmentConfig(),
) -> Tuple[Any, EnvironmentConfig]:
    """Init and return environment or wrapper.

    Args:
        config : env config.

    Returns:
        (env, config).
    """

    if config.type == "debug":
        env, _ = debugging_utils.make_environment(
            env_name=config.env_name, action_space=config.action_space
        )
    return env, config


def make_system(
    config: SystemConfig = SystemConfig(),
    environment_spec: mava_specs.MAEnvironmentSpec = None,
) -> Tuple[Any, SystemConfig]:
    """Inits and returns system/networks.

    Args:
        config : system config.
        environment_spec: spec for multi-agent env.

    Returns:
        system.
    """

    del environment_spec

    class System:
        pass

    logging.info(config)
    system = System()
    return system, config


def main(_: Any) -> None:
    """Template for educational system implementations.

    Args:
        _ : unused param - for absl.
    """

    init_config = init()
    env, env_config = make_environment()
    env_spec = mava_specs.MAEnvironmentSpec(env)
    system, system_config = make_system(environment_spec=env_spec)
    logging.info(f"Running {FLAGS.system}")

    # Run system on env
    del env, system, env_config, system_config, init_config


if __name__ == "__main__":
    app.run(main)
