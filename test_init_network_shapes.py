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

import chex
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from mava.networks import (
    DiscreteActionHead,
    FeedForwardActor,
    FeedForwardCritic,
    MLPTorso,
    RecurrentActor,
    RecurrentCritic,
    ScannedRNN,
)
from mava.new_networks import NewRecurrentActor, NewRecurrentCritic
from mava.new_networks import ScannedRNN as NewScannedRNN
from mava.types import ObservationGlobalState
from mava.utils import make_env as environments

config = DictConfig(
    {
        "env": {
            "scenario": {
                "name": "RobotWarehouse-v0",
                "task_name": "tiny-2ag",
                "task_config": {
                    "column_height": 8,
                    "shelf_rows": 1,
                    "shelf_columns": 3,
                    "num_agents": 2,
                    "sensor_range": 1,
                    "request_queue_size": 2,
                },
            },
            "eval_metric": "episode_return",
            "env_name": "RobotWarehouse",
            "kwargs": {"time_limit": 500},
        },
        "system": {
            "add_agent_id": True,
        },
        "arch": {
            "num_envs": 8,
        },
    }
)


base_key = jax.random.PRNGKey(0)
keys = jax.random.split(base_key, 3)
env, eval_env = environments.make(config)

num_actions = int(env.action_spec().num_values[0])
num_agents = env.action_spec().shape[0]

# PRNG keys.
key, actor_net_key, critic_net_key = keys

# Define network and optimiser.
actor_torso = MLPTorso(layer_sizes=[128, 128], activation="relu", use_layer_norm=False)
actor_action_head = DiscreteActionHead(action_dim=num_actions)
critic_torso = MLPTorso(layer_sizes=[128, 128], activation="relu", use_layer_norm=False)
actor_rec_torso = MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False)
actor_rec_torso = MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False)
critic_rec_torso = MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False)

################
# TEST FF IPPO
################

actor_network = FeedForwardActor(
    torso=MLPTorso(layer_sizes=[128, 128], activation="relu", use_layer_norm=False),
    action_head=DiscreteActionHead(action_dim=num_actions),
)
critic_network = FeedForwardCritic(
    torso=MLPTorso(layer_sizes=[128, 128], activation="relu", use_layer_norm=False)
)

# Initialise observation: Select only obs for a single agent.
init_x = env.observation_spec().generate_value()
init_x = jax.tree_util.tree_map(lambda x: x[0], init_x)
init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

actor_params_single = actor_network.init(actor_net_key, init_x)
critic_params_single = critic_network.init(critic_net_key, init_x)

# Initialise observation with batch of agents.
init_x_batch = env.observation_spec().generate_value()
init_x_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_x_batch)

actor_network = FeedForwardActor(
    torso=MLPTorso(layer_sizes=[128, 128], activation="relu", use_layer_norm=False),
    action_head=DiscreteActionHead(action_dim=num_actions),
)
critic_network = FeedForwardCritic(
    torso=MLPTorso(layer_sizes=[128, 128], activation="relu", use_layer_norm=False)
)

actor_params_batch = actor_network.init(actor_net_key, init_x_batch)
critic_params_batch = critic_network.init(critic_net_key, init_x_batch)

chex.assert_trees_all_equal_structs(actor_params_single, actor_params_batch)
chex.assert_trees_all_equal_shapes_and_dtypes(actor_params_single, actor_params_batch)

chex.assert_trees_all_equal_structs(critic_params_single, critic_params_batch)
chex.assert_trees_all_equal_shapes_and_dtypes(critic_params_single, critic_params_batch)

################
# TEST REC IPPO
################

actor_network = RecurrentActor(
    pre_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    post_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    action_head=DiscreteActionHead(action_dim=num_actions),
)
critic_network = RecurrentCritic(
    pre_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    post_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
)

init_obs = env.observation_spec().generate_value()
init_obs = jax.tree_util.tree_map(lambda x: x[0], init_obs)
init_obs = jax.tree_util.tree_map(
    lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
    init_obs,
)
init_obs = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
init_done = jnp.zeros((1, config.arch.num_envs), dtype=bool)
init_x = (init_obs, init_done)

hidden_size = 128
init_policy_hstate = ScannedRNN.initialize_carry((config.arch.num_envs), hidden_size)
init_critic_hstate = ScannedRNN.initialize_carry((config.arch.num_envs), hidden_size)

actor_params_single = actor_network.init(actor_net_key, init_policy_hstate, init_x)
critic_params_single = critic_network.init(critic_net_key, init_critic_hstate, init_x)

init_obs = env.observation_spec().generate_value()
init_obs = jax.tree_util.tree_map(
    lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
    init_obs,
)
init_obs = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
init_done = jnp.zeros((1, config.arch.num_envs, num_agents), dtype=bool)
init_x_batch = (init_obs, init_done)

hidden_size = 128
init_policy_hstate = NewScannedRNN.initialize_carry((config.arch.num_envs, num_agents), hidden_size)
init_critic_hstate = NewScannedRNN.initialize_carry((config.arch.num_envs, num_agents), hidden_size)

actor_network = NewRecurrentActor(
    pre_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    post_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    action_head=DiscreteActionHead(action_dim=num_actions),
)
critic_network = NewRecurrentCritic(
    pre_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    post_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
)

actor_params_batch = actor_network.init(actor_net_key, init_policy_hstate, init_x_batch)
critic_params_batch = critic_network.init(critic_net_key, init_critic_hstate, init_x_batch)

chex.assert_trees_all_equal_structs(actor_params_single, actor_params_batch)
chex.assert_trees_all_equal_shapes_and_dtypes(actor_params_single, actor_params_batch)

chex.assert_trees_all_equal_structs(critic_params_single, critic_params_batch)
chex.assert_trees_all_equal_shapes_and_dtypes(critic_params_single, critic_params_batch)


################
# TEST FF MAPPO
################

env, eval_env = environments.make(config, add_global_state=True)

actor_network = FeedForwardActor(
    torso=MLPTorso(layer_sizes=[128, 128], activation="relu", use_layer_norm=False),
    action_head=DiscreteActionHead(action_dim=num_actions),
)
critic_network = FeedForwardCritic(
    torso=MLPTorso(layer_sizes=[128, 128], activation="relu", use_layer_norm=False),
    centralised_critic=True,
)

obs = env.observation_spec().generate_value()
# Select only obs for a single agent.
init_x = ObservationGlobalState(
    agents_view=obs.agents_view[0],
    action_mask=obs.action_mask[0],
    global_state=obs.global_state[0],
    step_count=obs.step_count[0],
)
init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

actor_params_single = actor_network.init(actor_net_key, init_x)
critic_params_single = critic_network.init(critic_net_key, init_x)

# Initialise observation with batch of agents.
obs = env.observation_spec().generate_value()
init_x_batch = jax.tree_util.tree_map(lambda x: x[None, ...], obs)

actor_params_batch = actor_network.init(actor_net_key, init_x_batch)
critic_params_batch = critic_network.init(critic_net_key, init_x_batch)

chex.assert_trees_all_equal_structs(actor_params_single, actor_params_batch)
chex.assert_trees_all_equal_shapes_and_dtypes(actor_params_single, actor_params_batch)

chex.assert_trees_all_equal_structs(critic_params_single, critic_params_batch)
chex.assert_trees_all_equal_shapes_and_dtypes(critic_params_single, critic_params_batch)

################
# TEST REC MAPPO
################

actor_network = RecurrentActor(
    pre_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    post_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    action_head=DiscreteActionHead(action_dim=num_actions),
)
critic_network = RecurrentCritic(
    pre_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    post_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    centralised_critic=True,
)

# Initialise observation: Select only obs for a single agent.
init_obs = env.observation_spec().generate_value()

# init_obs = jax.tree_util.tree_map(lambda x: x[0], init_obs)
init_obs = jax.tree_util.tree_map(
    lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
    init_obs,
)
init_obs = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)

# Select only a single agent
init_done = jnp.zeros((1, config.arch.num_envs), dtype=bool)
init_obs_single = ObservationGlobalState(
    agents_view=init_obs.agents_view[:, :, 0, :],
    action_mask=init_obs.action_mask[:, :, 0, :],
    global_state=init_obs.global_state[:, :, 0, :],
    step_count=init_obs.step_count[:, 0],
)
init_single = (init_obs_single, init_done)

hidden_size = 128
init_policy_hstate = ScannedRNN.initialize_carry((config.arch.num_envs), hidden_size)
init_critic_hstate = ScannedRNN.initialize_carry((config.arch.num_envs), hidden_size)

actor_params_single = actor_network.init(actor_net_key, init_policy_hstate, init_single)
critic_params_single = critic_network.init(critic_net_key, init_critic_hstate, init_single)

actor_network = NewRecurrentActor(
    pre_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    post_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    action_head=DiscreteActionHead(action_dim=num_actions),
)
critic_network = NewRecurrentCritic(
    pre_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    post_torso=MLPTorso(layer_sizes=[128], activation="relu", use_layer_norm=False),
    centralised_critic=True,
)


init_obs = env.observation_spec().generate_value()
init_obs = jax.tree_util.tree_map(
    lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
    init_obs,
)
init_obs = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)

# Select only a single agent
init_done = jnp.zeros((1, config.arch.num_envs, num_agents), dtype=bool)
init_batch = (init_obs, init_done)

hidden_size = 128
init_policy_hstate = NewScannedRNN.initialize_carry((config.arch.num_envs, num_agents), hidden_size)
init_critic_hstate = NewScannedRNN.initialize_carry((config.arch.num_envs, num_agents), hidden_size)

actor_params_batch = actor_network.init(actor_net_key, init_policy_hstate, init_batch)
critic_params_batch = critic_network.init(critic_net_key, init_critic_hstate, init_batch)

chex.assert_trees_all_equal_structs(actor_params_single, actor_params_batch)
chex.assert_trees_all_equal_shapes_and_dtypes(actor_params_single, actor_params_batch)

chex.assert_trees_all_equal_structs(critic_params_single, critic_params_batch)
chex.assert_trees_all_equal_shapes_and_dtypes(critic_params_single, critic_params_batch)
