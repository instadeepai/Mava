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
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from absl import app, flags

from mava import specs as mava_specs
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

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
    max_total_steps: int = 50000
    num_episodes: int = 1000


@chex.dataclass(frozen=True, mappable_dataclass=False)
class RandomSystemState:
    rng: jnp.ndarray


def init(config: InitConfig = InitConfig()) -> InitConfig:
    """Init system.

    This would handle thing to be done upon system once in the beginning of a run,
    e.g. set random seeds.

    Args:
        config : init config.
    """
    return config


def make_environment(
    config: EnvironmentConfig = EnvironmentConfig(),
) -> Tuple[Any, EnvironmentConfig]:
    """Init and return environment or wrapper.

    Args:
        config : env config.

    Returns:
        env or wrapper.
    """
    if config.type == "debug":
        env, _ = debugging_utils.make_environment(
            env_name=config.env_name, action_space=config.action_space
        )
    return env, config


def make_system(
    environment_spec: mava_specs.MAEnvironmentSpec,
    config: SystemConfig = SystemConfig(),
) -> Tuple[Any, SystemConfig]:
    """Inits and returns system/networks.

    Args:
        config : system config.

    Returns:
        system.
    """
    init_rng = jax.random.PRNGKey(config.seed)
    agent_specs = environment_spec.get_agent_environment_specs()

    @dataclass
    class RandomExecutor:
        def init(state) -> None:
            pass

        @jax.jit
        def select_actions(
            observations: chex.Array, state: chex.dataclass
        ) -> Tuple[Dict, chex.dataclass]:
            # Not used in random agent
            del observations
            key = state.rng
            key, subkey = jax.random.split(key)

            actions = {}
            for net_key, spec in agent_specs.items():
                action_spec = spec.actions
                actions[net_key] = jax.random.randint(
                    subkey,
                    action_spec.shape,
                    action_spec.minimum,
                    action_spec.maximum + 1,
                    action_spec.dtype,
                )
            return (actions, RandomSystemState(rng=key))

    class RandomSystem:
        def __init__(self, rng_key, executor) -> None:
            self._executor = executor
            self._state = RandomSystemState(rng=rng_key)
            pass

        def select_actions(self, observation) -> Dict:
            action, self._state = self._executor.select_actions(
                observation, self._state
            )
            return action

        def observe_first(self, observations) -> None:
            del observations
            pass

        def observe(self, actions, observations) -> None:
            del actions, observations
            pass

        def update(self) -> None:
            pass

    logging.info(config)
    system = RandomSystem(init_rng, RandomExecutor)
    return system, config


def main(_: Any) -> None:
    """Template for educational system implementations.

    Args:
        _ : unused param - for absl.
    """

    _ = init()
    env, _ = make_environment()
    env_spec = mava_specs.MAEnvironmentSpec(env)
    system, system_config = make_system(env_spec)
    num_episodes = system_config.num_episodes
    max_total_steps = system_config.max_total_steps

    result = {
        "episode_length": 0,
        "episode_return": 0,
        "steps_per_second": 0,
        "episodes": 0,
        "total_step_count": 0,
    }

    logger = logger_utils.make_logger(
        directory=FLAGS.base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=str(datetime.now()),
        time_delta=0,
        label="random_agent",
    )

    for episode in range(num_episodes):
        # Reset episode
        # TODO Remove _ once mava has been updated.
        timestep, _ = env.reset()
        start_time = time.time()
        episode_steps = 0

        episode_returns: Dict[str, float] = {}
        for agent, spec in env.reward_spec().items():
            episode_returns.update({agent: jnp.zeros(spec.shape, spec.dtype)})

        # Add first element to replay buffer
        system.observe_first(timestep.observation)
        while not timestep.last():

            action = system.select_actions(timestep.observation)
            timestep = env.step(action)

            # Add to replay buffer
            system.observe(action, timestep)

            # Update network parameters
            system.update()

            episode_steps += 1

            episode_returns = jax.tree_map(
                lambda x, y: x + y, episode_returns, timestep.reward
            )

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - start_time)
        result = {
            "episode_length": episode_steps,
            "episode_return": episode_returns,
            "mean_episode_return": jnp.mean(jnp.array(list(episode_returns.values()))),
            "steps_per_second": steps_per_second,
            "episodes": episode,
            "total_step_count": episode_steps + result["total_step_count"],
        }
        logger.write(result)

        if result["total_step_count"] >= max_total_steps:
            logging.info(f"Reached max step count: {max_total_steps}")
            break


if __name__ == "__main__":
    app.run(main)
