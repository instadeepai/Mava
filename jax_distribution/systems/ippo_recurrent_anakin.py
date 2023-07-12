import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import functools
import jumanji
# TODO: We can maybe use a hydra instance initialization here.
# Instead of having to import all the environments's functions.
from jumanji.environments.routing.multi_cvrp.generator import UniformRandomGenerator
from jax_distribution.systems.ippo_feedforward_anakin import TimeIt, LogWrapper

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell()(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size
        )


def process_observation(obs, env_name):
    # TODO: We probably need an preprocessing function here
    # If else is maybe not the best
    if "MultiCVRP" in env_name:
        # We should provide more observation details to the network
        obs = obs.vehicles.coordinates
    elif "RobotWarehouse" in env_name:
        obs = obs.agents_view
    else:
        raise NotImplementedError("This environment is not supported")
    return obs

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def get_runner_fn(env, network, config):
    # TRAIN LOOP
    def _update_step(runner_state, unused):
        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            rng, _rng = jax.random.split(rng)

            # SELECT ACTION
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # STEP ENV
            env_state, next_timestep = env.step(env_state, action)
            obsv = process_observation(next_timestep.observation, config["ENV_NAME"])
            
            done, reward, ep_returns, ep_lengths, ep_done = jax.tree_map(
                lambda x: jnp.repeat(x, config["NUM_AGENTS"]).reshape(-1),
                [next_timestep.last(),
                 next_timestep.reward,
                 env_state.returned_episode_returns,
                 env_state.returned_episode_lengths,
                 next_timestep.last()],
            )

            info = {
                    "returned_episode_returns": ep_returns,
                    "returned_episode_lengths": ep_lengths,
                    "returned_episode": ep_done,
                }

            transition = Transition(
                done, action, value, reward, log_prob, last_obs, 
                info,
            )
            runner_state = (train_state, env_state, obsv, done, hstate, rng)
            return runner_state, transition

        initial_hstate = runner_state[-2]
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
        )

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, last_done, hstate, rng = runner_state
        ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
        _, _, last_val = network.apply(train_state.params, hstate, ac_in)
        last_val = last_val.squeeze(0)
        last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                    # RERUN NETWORK
                    _, pi, value = network.apply(
                        params, init_hstate[:, 0], (traj_batch.obs, traj_batch.done)
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, targets
                )

                # AVERAGE ACROSS DEVICES AND BATCHES
                total_loss = jax.lax.pmean(total_loss, axis_name="core")
                grads = jax.lax.pmean(grads, axis_name="batch")
                grads = jax.lax.pmean(grads, axis_name="core")

                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state

            batch = (init_hstate, traj_batch, advantages, targets)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], config["NUM_MINIBATCHES"], -1]
                        + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                batch,
            )

            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, total_loss

        init_hstate = initial_hstate[:, None, :]  # TBH
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        train_state = update_state[0]
        metric = traj_batch.info
        rng = update_state[-1]

        runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
        return runner_state, metric

    def runner_fn(runner_state):
        """Vectorise and repeat the update."""
        batched_update_fn = jax.vmap(
            _update_step, axis_name="batch"
        )  # vectorize across batch.

        runner_state, metric = jax.lax.scan(
        batched_update_fn, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return runner_fn

def run_experiment(env_name, config):

    # INIT ENV
    # TODO: We need a more general way of doing this
    # If else statements do not scale well
    if "MultiCVRP" in env_name:
        config["NUM_AGENTS"] = 3
        generator = UniformRandomGenerator(
            num_vehicles=config["NUM_AGENTS"], num_customers=6
        )
        env = jumanji.make(env_name, generator=generator)
        num_actions = int(env.action_spec().maximum)

    elif "RobotWarehouse" in env_name:
        env = jumanji.make(env_name)
        num_actions = int(env.action_spec().num_values[0])
        config["NUM_AGENTS"] = env.num_agents

    env = LogWrapper(env)
    cores_count = len(jax.devices())
    # Number of updates
    timesteps_per_iteration = (
        cores_count * config["ROLLOUT_LENGTH"] * config["BATCH_SIZE"]
    )
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // timesteps_per_iteration
    )

   # INIT NETWORK AND OPTIMIZER
    network = ActorCriticRNN(num_actions, config=config)
    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng = jax.random.split(rng)
    init_x = (
        process_observation(env.observation_spec().generate_value(), env_name),
        jnp.zeros((1)),
    )
    init_hstate = ScannedRNN.initialize_carry(config["NUM_AGENTS"], 128)
    
    # Support for multiple agents
    init_net_x, init_net_hstate = jax.tree_util.tree_map(
        lambda x: x[None, ...],
        [init_x, init_hstate],
    )
    network_params = network.init(_rng, init_net_hstate, init_net_x)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    
    # INIT RUNNER
    # Get the runner function
    runner = get_runner_fn(
        env,
        network,
        config,
    )
    runner = jax.pmap(runner, axis_name="core")  # replicate over multiple cores.

    # BROADCAST TRAIN STATE
    def broadcast(x):
        if np.isscalar(x):  # x is an int or float
            x = jnp.array(x)  # convert it to a JAX array
        return jnp.broadcast_to(x, (cores_count, config["BATCH_SIZE"]) + x.shape)

    train_state, init_hstate = jax.tree_map(broadcast, [train_state, init_hstate])  # broadcast to cores and batch.

    # RESHAPE ENV STATES, TIMESTEPS and RNGS
    reshape = lambda x: x.reshape((cores_count, config["BATCH_SIZE"]) + x.shape[1:])
    rng, *reset_rngs = jax.random.split(rng, 1 + cores_count * config["BATCH_SIZE"])
    env_states, env_timesteps = jax.vmap(env.reset)(jnp.stack(reset_rngs))  # init envs.
    observation = process_observation(env_timesteps.observation, env_name)
    rng, *step_rngs = jax.random.split(rng, 1 + cores_count * config["BATCH_SIZE"])
    
    env_states, observation, step_rngs = jax.tree_util.tree_map(
        reshape,
        [env_states, observation, jnp.array(step_rngs)],
    )  # add dimension to pmap over.
    
    dones = jnp.zeros((cores_count, config["BATCH_SIZE"], config["NUM_AGENTS"]), dtype=bool)
    runner_state = (
        train_state,
        env_states,
        observation,
        dones,
        init_hstate,
        step_rngs,
    )

    # Time the compilation and execution of the runner function.
    with TimeIt(tag="COMPILATION"):
        out = runner(runner_state)  # compiles
        jax.block_until_ready(out)

    with TimeIt(tag="EXECUTION", frames=config["TOTAL_TIMESTEPS"]):
        out = runner(runner_state)
        jax.block_until_ready(out)
    val = out["metric"]["returned_episode_returns"].mean()
    print(f"Mean Episode Return: {val}")

if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "BATCH_SIZE": 4,
        "ROLLOUT_LENGTH": 128,
        "TOTAL_TIMESTEPS": 204800,
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "RobotWarehouse-v0",  # [RobotWarehouse-v0, MultiCVRP-v0]
        "SEED": 1234,
    }

    run_experiment(config["ENV_NAME"], config)

 