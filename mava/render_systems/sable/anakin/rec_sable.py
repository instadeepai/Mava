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
from mava.systems.sable.types import (
    HiddenStates,
)
from mava.systems.sable.types import RecLearnerState as LearnerState
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
    _, net_key = keys

    # Get number of agents and actions.
    action_dim = int(env.action_spec().num_values[0])
    n_agents = env.action_spec().shape[0]
    config.system.num_agents = n_agents
    config.system.num_actions = action_dim

    # Setting the chunksize - smaller chunks save memory at the cost of speed
    if config.network.memory_config.timestep_chunk_size:
        config.network.memory_config.chunk_size = (
            config.network.memory_config.timestep_chunk_size * n_agents
        )
    else:
        config.network.memory_config.chunk_size = config.system.rollout_length * n_agents

    _, action_space_type = get_action_head(env)

    # Define network.
    sable_network = SableNetwork(
        n_agents=n_agents,
        n_agents_per_chunk=n_agents,
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
        restored_params, _ = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=True, THiddenState=HiddenStates
        )
        # Update the params and hidden states
        params = restored_params

    # Pack apply and update functions.
    apply_fns = (
        partial(sable_network.apply, method="get_actions"),  # Execution function
        sable_network.apply,  # Training function
    )

    return params, apply_fns[0]


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config, render=True)

    # PRNG keys.
    key, net_key = jax.random.split(jax.random.PRNGKey(config.system.seed))

    # Setup learner.
    params, sable_execution_fn = learner_setup(env, (key, net_key), config)

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
        hidden_state = get_init_hidden_state(config.network.net_config, 1)
        while not timestep.last():
            key, action_key = jax.random.split(key)
            observation = jax.tree_util.tree_map(lambda x: x[None], timestep.observation)

            action, _, _, hidden_state = sable_execution_fn(  # type: ignore
                params,
                observation,
                hidden_state,
                action_key,
            )

            hidden_state = HiddenStates(
                encoder=hidden_state.encoder,
                decoder_self_retn=jnp.zeros_like(hidden_state.decoder_self_retn),
                decoder_cross_retn=jnp.zeros_like(hidden_state.decoder_cross_retn),
            )

            state, timestep = step_fn(state, action.squeeze(axis=0))
            states.append(state)
        # Freeze the terminal frame to pause the GIF.
        for _ in range(3):
            states.append(state)

    eval_env.unwrapped.animate(states, interval=80, save_path="rec_sable_lbf_reset.gif")


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_sable.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    cfg.logger.system_name = "rec_sable"

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Rec Sable experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
