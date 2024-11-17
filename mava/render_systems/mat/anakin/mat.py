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

import copy
from typing import Any, Dict, Tuple

import chex
import hydra
import jax
import jax.numpy as jnp
from colorama import Fore, Style
from jax import tree
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.networks.mat_network import MultiAgentTransformer
from mava.systems.mat.types import LearnerState
from mava.types import (
    LearnerFn,
    MarlEnv,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.network_utils import get_action_head
from mava.utils.total_timestep_checker import check_total_timesteps


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[LearnerState], Any, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get number of agents.
    config.system.num_agents = env.num_agents

    # PRNG keys.
    _, actor_net_key = keys

    # Initialise observation: Obs for all agents.
    init_x = env.observation_spec().generate_value()
    init_x = tree.map(lambda x: x[None, ...], init_x)

    _, action_space_type = get_action_head(env)

    if action_space_type == "discrete":
        init_action = jnp.zeros((1, config.system.num_agents), dtype=jnp.int32)
    elif action_space_type == "continuous":
        init_action = jnp.zeros((1, config.system.num_agents, env.action_dim), dtype=jnp.float32)
    else:
        raise ValueError("Invalid action space type")

    # Define network and optimiser.
    actor_network = MultiAgentTransformer(
        action_dim=env.action_dim,
        n_agent=config.system.num_agents,
        net_config=config.network,
        action_space_type=action_space_type,
    )

    # Initialise actor params and optimiser state.
    # `PRNGKey(0)` is just a dummy key we pass through the network since it needs a key for
    # computing the network entropy at train time.
    params = actor_network.init(actor_net_key, init_x, init_action, jax.random.PRNGKey(0))

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    return params, actor_network


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config, render=True)

    # PRNG keys.
    key, actor_net_key = jax.random.split(jax.random.PRNGKey(config.system.seed))

    # Setup learner.
    params, actor_network = learner_setup(env, (key, actor_net_key), config)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    assert (
        config.arch.num_envs % config.system.num_minibatches == 0
    ), "Number of envs must be divisibile by number of minibatches."

    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    reset_fn = jax.jit(eval_env.reset)
    step_fn = jax.jit(eval_env.step)
    states = []
    for _ in range(3):
        key, reset_key = jax.random.split(key)
        state, timestep = reset_fn(reset_key)
        states.append(state)
        while not timestep.last():
            key, action_key = jax.random.split(key)
            observation = jax.tree_util.tree_map(lambda x: x[None], timestep.observation)

            action, _, _ = actor_network.apply(  # type: ignore
                params,
                observation,
                action_key,
                method="get_actions",
            )

            state, timestep = step_fn(state, action.squeeze(axis=0))
            states.append(state)
        # Freeze the terminal frame to pause the GIF.
        for _ in range(3):
            states.append(state)

    eval_env.unwrapped.animate(states, interval=80, save_path="mat_lbf.gif")


@hydra.main(
    config_path="../../../configs/default",
    config_name="mat.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    cfg.logger.system_name = "mat"

    eval_performance = run_experiment(cfg)
    jax.block_until_ready(eval_performance)
    print(f"{Fore.CYAN}{Style.BRIGHT}MAT experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
