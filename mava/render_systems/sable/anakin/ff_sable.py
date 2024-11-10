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
from functools import partial
from typing import Callable, Dict, Tuple

import chex
import hydra
import jax
import jax.numpy as jnp
from colorama import Fore, Style
from jax import tree
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.networks import SableNetwork
from mava.networks.utils.sable import get_init_hidden_state
from mava.systems.sable.types import FFLearnerState as LearnerState
from mava.types import LearnerFn, MarlEnv
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.network_utils import get_action_head
from mava.utils.total_timestep_checker import check_total_timesteps


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[LearnerState], Callable, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get number of agents.
    config.system.num_agents = env.num_agents

    # PRNG keys.
    key, net_key = keys

    # Get number of agents and actions.
    action_dim = int(env.action_spec().num_values[0])
    n_agents = env.action_spec().shape[0]
    config.system.num_agents = n_agents
    config.system.num_actions = action_dim

    # Setting the chunksize - many agent problems require chunking agents
    # Create a dummy decay factor for FF Sable
    config.network.memory_config.decay_scaling_factor = 1.0
    if config.network.memory_config.agents_chunk_size:
        config.network.memory_config.chunk_size = config.network.memory_config.agents_chunk_size
        err = "Number of agents should be divisible by chunk size"
        assert n_agents % config.network.memory_config.chunk_size == 0, err
    else:
        config.network.memory_config.chunk_size = n_agents

    # Set positional encoding to False, since ff-sable does not use temporal dependencies.
    config.network.memory_config.timestep_positional_encoding = False

    _, action_space_type = get_action_head(env)

    # Define network.
    sable_network = SableNetwork(
        n_agents=n_agents,
        n_agents_per_chunk=config.network.memory_config.chunk_size,
        action_dim=action_dim,
        net_config=config.network.net_config,
        memory_config=config.network.memory_config,
        action_space_type=action_space_type,
    )

    # Get mock inputs to initialise network.
    init_obs = env.observation_spec().generate_value()
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)  # Add batch dim
    init_hs = get_init_hidden_state(config.network.net_config, config.arch.num_envs)
    init_hs = tree.map(lambda x: x[0, jnp.newaxis], init_hs)

    # Initialise params and optimiser state.
    params = sable_network.init(
        net_key,
        init_obs,
        init_hs,
        net_key,
        method="get_actions",
    )

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

    return params, sable_network


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config, render=True)

    # PRNG keys.
    key, net_key = jax.random.split(jax.random.PRNGKey(config.system.seed))

    # Setup learner.
    params, sable_network = learner_setup(env, (key, net_key), config)
    eval_apply_fn = partial(sable_network.apply, method="get_actions")
    eval_hs = get_init_hidden_state(config.network.net_config, 1)
    sable_execution_fn = partial(eval_apply_fn, hstates=eval_hs)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

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

            action, _, _, _ = sable_execution_fn(  # type: ignore
                params,
                observation=observation,
                key=action_key,
            )

            state, timestep = step_fn(state, action.squeeze(axis=0))
            states.append(state)
        # Freeze the terminal frame to pause the GIF.
        for _ in range(3):
            states.append(state)

    eval_env.unwrapped.animate(states, interval=80, save_path="ff_sable_rware.gif")


@hydra.main(
    config_path="../../../configs/default",
    config_name="ff_sable.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    cfg.logger.system_name = "ff_sable"

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}FF Sable experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
