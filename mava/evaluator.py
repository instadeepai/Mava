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

from typing import Any, Callable, Dict, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment

from mava.types import EvalState, ExperimentOutput, RNNEvalState


def get_ff_evaluator_fn(
    env: Environment, apply_fn: Callable, config: dict, eval_multiplier: int = 1
) -> Callable:
    """Get the evaluator function for feedforward networks.

    Args:
        env (Environment): An evironment isntance for evaluation.
        apply_fn (callable): Network forward pass method.
        config (dict): Experiment configuration.
        eval_multiplier (int): A scalar that will increase the number of evaluation
            episodes by a fixed factor. The reason for the increase is to enable the
            computation of the `absolute metric` which is a metric computed and the end
            of training by rolling out the policy which obtained the greatest evaluation
            performance during training for 10 times more episodes than were used at a
            single evaluation step.
    """

    def eval_one_episode(params: FrozenDict, init_eval_state: EvalState) -> Dict:
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def _env_step(eval_state: EvalState) -> EvalState:
            """Step the environment."""
            # PRNG keys.
            rng, env_state, last_timestep, step_count_, return_ = eval_state

            # Select action.
            rng, _rng = jax.random.split(rng)
            pi = apply_fn(params, last_timestep.observation)

            if config["evaluation_greedy"]:
                action = pi.mode()
            else:
                action = pi.sample(seed=_rng)

            # Step environment.
            env_state, timestep = env.step(env_state, action)

            # Log episode metrics.
            return_ += timestep.reward
            step_count_ += 1
            eval_state = EvalState(rng, env_state, timestep, step_count_, return_)
            return eval_state

        def not_done(carry: Tuple) -> bool:
            """Check if the episode is done."""
            timestep = carry[2]
            is_not_done: bool = ~timestep.last()
            return is_not_done

        eval_state = EvalState(
            init_eval_state.key,
            init_eval_state.env_state,
            init_eval_state.timestep,
            0,
            0.0,
        )

        final_state = jax.lax.while_loop(
            not_done,
            _env_step,
            eval_state,
        )

        step_count_, return_ = final_state.step_count_, final_state.return_
        eval_metrics = {
            "episode_return": return_,
            "episode_length": step_count_,
        }
        return eval_metrics

    def evaluator_fn(trained_params: FrozenDict, rng: chex.PRNGKey) -> ExperimentOutput:
        """Evaluator function."""

        # Initialise environment states and timesteps.
        n_devices = len(jax.devices())

        eval_batch = (config["num_eval_episodes"] // n_devices) * eval_multiplier

        rng, *env_rngs = jax.random.split(rng, eval_batch + 1)
        env_states, timesteps = jax.vmap(env.reset)(
            jnp.stack(env_rngs),
        )
        # Split rngs for each core.
        rng, *step_rngs = jax.random.split(rng, eval_batch + 1)
        # Add dimension to pmap over.
        reshape_step_rngs = lambda x: x.reshape(eval_batch, -1)
        step_rngs = reshape_step_rngs(jnp.stack(step_rngs))

        eval_state = EvalState(step_rngs, env_states, timesteps)
        eval_metrics = jax.vmap(eval_one_episode, in_axes=(None, 0), axis_name="eval_batch")(
            trained_params, eval_state
        )

        return ExperimentOutput(
            episodes_info=eval_metrics,
        )

    return evaluator_fn


def get_rnn_evaluator_fn(
    env: Environment,
    apply_fn: Callable,
    config: dict,
    scanned_rnn: nn.Module,
    eval_multiplier: int = 1,
) -> Callable:
    """Get the evaluator function for recurrent networks."""

    def eval_one_episode(params: FrozenDict, init_eval_state: RNNEvalState) -> Dict:
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def _env_step(eval_state: RNNEvalState) -> RNNEvalState:
            """Step the environment."""
            (
                rng,
                env_state,
                last_timestep,
                last_done,
                hstate,
                step_count_,
                return_,
            ) = eval_state

            # PRNG keys.
            rng, policy_rng = jax.random.split(rng)

            # Add a batch dimension and env dimension to the observation.
            batched_observation = jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, jnp.newaxis, :], last_timestep.observation
            )
            ac_in = (
                batched_observation,
                last_done[jnp.newaxis, jnp.newaxis, :][..., 0],
            )

            # Run the network.
            hstate, pi = apply_fn(params, hstate, ac_in)

            if config["evaluation_greedy"]:
                action = pi.mode()
            else:
                action = pi.sample(seed=policy_rng)

            # Step environment.
            env_state, timestep = env.step(env_state, action[-1].squeeze(0))

            # Log episode metrics.
            return_ += timestep.reward
            step_count_ += 1
            eval_state = RNNEvalState(
                rng,
                env_state,
                timestep,
                jnp.repeat(timestep.last(), config["num_agents"]),
                hstate,
                step_count_,
                return_,
            )
            return eval_state

        def not_done(carry: Tuple) -> bool:
            """Check if the episode is done."""
            timestep = carry[2]
            is_not_done: bool = ~timestep.last()
            return is_not_done

        eval_state = RNNEvalState(
            init_eval_state.key,
            init_eval_state.env_state,
            init_eval_state.timestep,
            init_eval_state.dones,
            init_eval_state.hstate,
            0,
            0.0,
        )

        final_state = jax.lax.while_loop(
            not_done,
            _env_step,
            eval_state,
        )

        step_count_, return_ = final_state.step_count_, final_state.return_
        eval_metrics = {
            "episode_return": return_,
            "episode_length": step_count_,
        }
        return eval_metrics

    def evaluator_fn(trained_params: FrozenDict, rng: chex.PRNGKey) -> ExperimentOutput:
        """Evaluator function."""

        # Initialise environment states and timesteps.
        n_devices = len(jax.devices())

        eval_batch = config["num_eval_episodes"] // n_devices * eval_multiplier

        rng, *env_rngs = jax.random.split(rng, eval_batch + 1)
        env_states, timesteps = jax.vmap(env.reset)(
            jnp.stack(env_rngs),
        )
        # Split rngs for each core.
        rng, *step_rngs = jax.random.split(rng, eval_batch + 1)
        # Add dimension to pmap over.
        reshape_step_rngs = lambda x: x.reshape(eval_batch, -1)
        step_rngs = reshape_step_rngs(jnp.stack(step_rngs))

        # Initialise hidden state.
        init_hstate = scanned_rnn.initialize_carry(eval_batch, 128)
        init_hstate = jnp.expand_dims(init_hstate, axis=1)
        init_hstate = jnp.expand_dims(init_hstate, axis=2)
        init_hstate = jnp.tile(init_hstate, (1, config["num_agents"], 1))

        # Initialise dones.
        dones = jnp.zeros(
            (
                eval_batch,
                config["num_agents"],
            ),
            dtype=bool,
        )

        eval_state = RNNEvalState(
            key=step_rngs,
            env_state=env_states,
            timestep=timesteps,
            dones=dones,
            hstate=init_hstate,
        )

        eval_metrics = jax.vmap(eval_one_episode, in_axes=(None, 0), axis_name="eval_batch")(
            trained_params, eval_state
        )

        return ExperimentOutput(
            episodes_info=eval_metrics,
        )

    return evaluator_fn


def evaluator_setup(
    eval_env: Environment,
    rng_e: chex.PRNGKey,
    network: Any,
    params: FrozenDict,
    config: Dict,
    use_recurrent_net: bool = False,
    scanned_rnn: nn.Module = None,
) -> Tuple[Callable, Callable, Tuple]:
    """Initialise evaluator_fn."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Vmap it over number of agents and create evaluator_fn.
    if use_recurrent_net:
        vmapped_eval_network_apply_fn = jax.vmap(
            network.apply, in_axes=(None, 1, (2, None)), out_axes=(1, 2)
        )
        evaluator = get_rnn_evaluator_fn(
            eval_env,
            vmapped_eval_network_apply_fn,
            config,
            scanned_rnn,
        )
        absolute_metric_evaluator = get_rnn_evaluator_fn(
            eval_env,
            vmapped_eval_network_apply_fn,
            config,
            scanned_rnn,
            10,
        )
    else:
        vmapped_eval_network_apply_fn = jax.vmap(
            network.apply,
            in_axes=(None, 0),
        )
        evaluator = get_ff_evaluator_fn(
            eval_env,
            vmapped_eval_network_apply_fn,
            config,
        )
        absolute_metric_evaluator = get_ff_evaluator_fn(
            eval_env,
            vmapped_eval_network_apply_fn,
            config,
            10,
        )

    evaluator = jax.pmap(evaluator, axis_name="device")
    absolute_metric_evaluator = jax.pmap(absolute_metric_evaluator, axis_name="device")

    # Broadcast trained params to cores and split rngs for each core.
    trained_params = jax.tree_util.tree_map(lambda x: x[:, 0, ...], params)
    rng_e, *eval_rngs = jax.random.split(rng_e, n_devices + 1)
    eval_rngs = jnp.stack(eval_rngs).reshape(n_devices, -1)

    return evaluator, absolute_metric_evaluator, (trained_params, eval_rngs)
