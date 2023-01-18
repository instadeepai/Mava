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
from typing import Any, Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from absl import app, flags

from mava import specs as mava_specs
from mava.utils.environments import debugging_utils, smac_utils
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("system", "test agent", "What agent is running.")
flags.DEFINE_string(
    "base_dir", "~/mava", "Base dir to store experiment data e.g. checkpoints."
)
flags.DEFINE_string(
    "env_name",
    "simple_spread",
    "Env name e.g. sinple_spread, 3m or 8m.",
)
flags.DEFINE_string(
    "env_type",
    "debug",
    "Env type e.g. debug or smac.",
)
flags.DEFINE_string(
    "env_action_type",
    "discrete",
    "Discrete or continous. Only applies for debug env,\
    other env have their predefined types.",
)
flags.DEFINE_integer(
    "max_total_steps",
    100000,
    "Max total steps (across all episodes).",
)

flags.DEFINE_integer(
    "num_episodes",
    None,
    "Max number of episodes.",
)


@dataclass
class InitConfig:
    seed: int = 42


@dataclass
class EnvironmentConfig:
    env_name: str
    type: str
    action_space: str
    seed: int = 42


@dataclass
class SystemConfig:
    max_total_steps: int
    num_episodes: int
    name: str = "random"
    seed: int = 42


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
    config: EnvironmentConfig,
) -> Tuple[Any, EnvironmentConfig]:
    """Init and return environment or wrapper.

    Args:
        config : env config.

    Returns:
        env or wrapper.
    """
    if config.type == "debug":
        env, _ = debugging_utils.make_environment(
            env_name=config.env_name,
            action_space=config.action_space,
            random_seed=config.seed,
        )
    elif config.type == "smac":
        env, _ = smac_utils.make_environment(
            map_name=config.env_name, random_seed=config.seed
        )
    return env, config


def make_system(
    environment_spec: mava_specs.MAEnvironmentSpec,
    config: SystemConfig,
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
        """Pure jittable executor/actor."""

        def init(state) -> None:
            """Init executor.

            Args:
                state : state that holds variables.
            """
            pass

        @jax.jit
        def select_actions(
            observations: chex.Array, state: chex.dataclass
        ) -> Tuple[Dict, chex.dataclass]:
            """Select actions using obs.

            Args:
                observations : observations from env.
                state : state for variables.

            Returns:
                actions per agent in a dict.
            """
            key = state.rng
            key, subkey = jax.random.split(key)

            actions = {}
            for net_key, spec in agent_specs.items():
                # Select random discrete action.
                # If we don't have legal actions, we could have used
                # jax.random.randint.
                mask = observations[net_key].legal_actions
                logits = jax.random.uniform(subkey, mask.shape)
                logits = jnp.where(
                    mask.astype(bool),
                    logits,
                    jnp.finfo(logits.dtype).min,
                )
                actions[net_key] = logits.argmax(axis=-1)
            return (actions, RandomSystemState(rng=key))  # type: ignore

    class RandomSystem:
        """Multi-Agent System (Group of agents)."""

        def __init__(self, rng_key: jnp.array, executor: Any) -> None:
            """Init system.

            Args:
                rng_key : rng jax key.
                executor : executor/actor.
            """
            self._executor = executor
            self._state = RandomSystemState(rng=rng_key)  # type: ignore

        def select_actions(self, observation: Dict) -> Dict:
            """Select actions.

            Args:
                observation : observation for current timestep.

            Returns:
                actions per agent in a dict.
            """
            action, self._state = self._executor.select_actions(
                observation, self._state
            )
            return action

        def observe_first(self, observations: Dict) -> None:
            """Add first element to replay buffer/queue.

            Args:
                observations : observation from first timestep.
            """
            del observations

        def observe(self, actions: Dict, observations: Dict) -> None:
            """Add element to replay buffer/queue.

            Args:
                actions : action taken.
                observations : observation.
            """
            del actions, observations

        def update(self) -> None:
            """Update networks parameters/do some learning."""
            pass

    logging.info(config)
    system = RandomSystem(init_rng, RandomExecutor)
    return system, config


def main(_: Any) -> None:
    """Template for educational system implementations.

    Args:
        _ : unused param - for absl.
    """

    # Init env and system.
    _ = init()
    env, _ = make_environment(
        EnvironmentConfig(
            env_name=FLAGS.env_name,
            type=FLAGS.env_type,
            action_space=FLAGS.env_action_type,
        )
    )
    env_spec = mava_specs.MAEnvironmentSpec(env)
    system, system_config = make_system(
        env_spec,
        SystemConfig(
            max_total_steps=FLAGS.max_total_steps, num_episodes=FLAGS.num_episodes
        ),
    )

    # Init variables.
    num_episodes = system_config.num_episodes
    max_total_steps = system_config.max_total_steps

    # Init results dict.
    result: Dict[str, Union[float, int]] = {
        "episode_length": 0,
        "episode_return": 0,
        "steps_per_second": 0,
        "episodes": 0,
        "total_step_count": 0,
    }

    # Create logger.
    # Log every time_delta seconds
    time_delta = 5
    logger = logger_utils.make_logger(
        directory=FLAGS.base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=str(datetime.now()),
        time_delta=time_delta,
        label="random_agent",
    )

    episode_count: int = 0
    step_count: int = 0

    def should_terminate(episode_count: Optional[int], step_count: int) -> bool:
        """Func which checks if we should stop running.

        Args:
            episode_count : max episodes.
            step_count : max steps.

        Returns:
            bool indicating if we should terminate.
        """
        should_stop = (num_episodes is not None and episode_count >= num_episodes) or (  # type: ignore # noqa: E501
            max_total_steps is not None and step_count >= max_total_steps
        )

        if should_stop:
            logging.info(
                f"Reached max step count: {max_total_steps} or max episode: {num_episodes} , \
                Current steps: {step_count}, Current episode: {episode_count}"
            )

        return should_stop

    # Episode loop.
    while not should_terminate(episode_count, step_count):
        # Reset env.
        # TODO Remove `_` once env wrappers are more consistent.
        timestep, _ = env.reset()
        start_time = time.time()
        episode_steps = 0

        # Returns dict.
        episode_returns: Dict[str, float] = {}
        for agent, spec in env.reward_spec().items():
            episode_returns.update({agent: jnp.zeros(spec.shape, spec.dtype)})

        # Add first element to replay buffer
        system.observe_first(timestep.observation)

        # Run single episode.
        while not timestep.last():

            action = system.select_actions(timestep.observation)
            timestep = env.step(action)

            # Check for extras
            # TODO Remove once return_state_info param get added to make env.
            if type(timestep) == tuple:
                timestep, _ = timestep
            else:
                _ = {}

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
            "mean_episode_return": jnp.mean(jnp.array(list(episode_returns.values()))),
            "steps_per_second": steps_per_second,
            "episodes": episode_count,
            "total_step_count": episode_steps + result["total_step_count"],
        }
        result.update({"episode_returns": episode_returns})  # type: ignore

        episode_count += 1
        step_count = int(result["total_step_count"])
        logger.write(result)


if __name__ == "__main__":
    app.run(main)
