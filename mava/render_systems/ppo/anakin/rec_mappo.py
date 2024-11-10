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
from typing import Dict, Tuple

import chex
import hydra
import jax
import jax.numpy as jnp
from colorama import Fore, Style
from jax import tree
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.networks import RecurrentActor as Actor
from mava.networks import RecurrentValueNet as Critic
from mava.networks import ScannedRNN
from mava.systems.ppo.types import (
    HiddenStates,
    Params,
    RNNLearnerState,
)
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
) -> Tuple[LearnerFn[RNNLearnerState], Actor, RNNLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get number of agents.
    num_agents = env.num_agents
    config.system.num_agents = num_agents

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    action_head, _ = get_action_head(env)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    critic_pre_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_post_torso = hydra.utils.instantiate(config.network.critic_network.post_torso)

    actor_network = Actor(
        pre_torso=actor_pre_torso,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
        hidden_state_dim=config.network.hidden_state_dim,
    )
    critic_network = Critic(
        pre_torso=critic_pre_torso,
        post_torso=critic_post_torso,
        hidden_state_dim=config.network.hidden_state_dim,
        centralised_critic=True,
    )

    # Initialise observation with obs of all agents.
    init_obs = env.observation_spec().generate_value()
    init_obs = tree.map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)
    init_done = jnp.zeros((1, config.arch.num_envs, num_agents), dtype=bool)
    init_obs_done = (init_obs, init_done)

    # Initialise hidden state.
    init_policy_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )
    init_critic_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )

    # initialise params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_policy_hstate, init_obs_done)
    critic_params = critic_network.init(critic_net_key, init_critic_hstate, init_obs_done)

    # Pack params and initial states.
    params = Params(actor_params, critic_params)
    hstates = HiddenStates(init_policy_hstate, init_critic_hstate)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, restored_hstates = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=True, THiddenState=HiddenStates
        )
        # Update the params and hstates
        params = restored_params
        hstates = restored_hstates if restored_hstates else hstates

    return params, hstates, actor_network


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config=config, add_global_state=True, render=True)

    num_agents = env.num_agents

    # PRNG keys.
    key, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=3
    )

    # Setup learner.
    params, hstates, actor_network = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

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
        policy_hs = jnp.zeros_like(hstates.policy_hidden_state[0][None])
        states.append(state)
        while not timestep.last():
            key, action_key = jax.random.split(key)
            observation = jax.tree_util.tree_map(lambda x: x[None, None], timestep.observation)
            done = timestep.last()[None, None, None].repeat(num_agents, axis=-1)
            policy_hs, pi = actor_network.apply(params.actor_params, policy_hs, (observation, done))
            action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=action_key)
            state, timestep = step_fn(state, action.squeeze(axis=(0, 1)))
            states.append(state)
        # Freeze the terminal frame to pause the GIF.
        for _ in range(3):
            states.append(state)

    eval_env.unwrapped.animate(states, interval=80, save_path="rec_mappo_rware.gif")


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_mappo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    cfg.logger.system_name = "rec_mappo"

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Recurrent MAPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
