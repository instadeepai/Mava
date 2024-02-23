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
import time
from typing import Any, Callable, Dict, NamedTuple, Sequence, Tuple
import functools

import chex
import distrax
import flashbax as fbx
import flax.linen as nn
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from chex import PRNGKey
from colorama import Fore, Style
from flashbax.buffers.flat_buffer import TrajectoryBuffer
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax.core.scope import FrozenVariableDict
from jax import Array
from jax.typing import ArrayLike
from jumanji.env import Environment, State
from omegaconf import DictConfig, OmegaConf
from typing_extensions import TypeAlias

from mava.evaluator_idqn import evaluator_setup
from mava.types import Observation
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax import unreplicate_batch_dim, unreplicate_learner_state
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers import episode_metrics
from mava.networks import MLPTorso
from mava.types import RNNObservation
from flax.linen.initializers import orthogonal


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        new_rnn_state, y = nn.GRUCell(features=ins.shape[-1])(rnn_state, ins)

        new_rnn_state = jnp.where( # TODO: (Claude) is this reset not a timestep too early? 
            resets[:, :, jnp.newaxis],
            self.initialize_carry((ins.shape[0], ins.shape[1]), ins.shape[-1]),
            new_rnn_state,
        )
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: Sequence[int], hidden_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (*batch_size, hidden_size))
    
class ActionSelectionParams(NamedTuple):
    nw_params: FrozenVariableDict
    hidden_state: chex.Array
    terms: chex.Array
    t: int
    key: chex.PRNGKey

class InteractionParams(NamedTuple):
    select_params: ActionSelectionParams
    env_state: State
    obs: Observation
    buffer_state: TrajectoryBufferState


class QNetParams(NamedTuple):
    online: FrozenVariableDict
    target: FrozenVariableDict

class RNNQLearnerState(NamedTuple):
    """State of the `Learner` for recurrent architectures for Q-learning."""
    # purely interaction vars
    hidden_state: chex.Array
    terms: chex.Array
    t: int
    env_state: State
    obs: Observation

    # shared vars
    buffer_state: TrajectoryBufferState
    params: QNetParams
    key: chex.PRNGKey

    # purely training vars
    opt_state: optax.OptState

Metrics = Dict[str, Array]

class Optimisers(NamedTuple):
    q: optax.GradientTransformation

class Transition(NamedTuple):
    obs: Array
    action: Array
    reward: Array
    term: Array

BufferState: TypeAlias = TrajectoryBufferState[Transition]

class RecQNetwork(nn.Module):
    num_actions:int
    pre_torso: MLPTorso = MLPTorso((256,), nn.relu, False)
    post_torso: MLPTorso = MLPTorso((256,), nn.relu, False)

    @nn.compact
    def __call__(self, hidden_state: chex.Array, observation_term: RNNObservation,) -> Array:
        # unpack consumed parameters
        obs, term = observation_term

        embedding = self.pre_torso(obs.agents_view)

        # pack consumed parameters
        rnn_input = (embedding, term)
        hidden_state, embedding = ScannedRNN()(hidden_state, rnn_input)

        embedding = self.post_torso(embedding)

        q_values = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01))(embedding)

        return hidden_state, q_values



class Networks(NamedTuple):
    q: RecQNetwork

class NetworkInput(NamedTuple):
    hidden_state: chex.Array
    obs: Observation
    terms: bool

def init(
    cfg: DictConfig,
) -> Tuple[
    Tuple[Environment, Environment],
    Networks,
    Optimisers,
    TrajectoryBuffer,
    RNNQLearnerState,
    Array,
    MavaLogger,
    chex.PRNGKey,
]:
    
    hidden_size = 256
    logger = MavaLogger(cfg)

    # init key, get devices available
    key = jax.random.PRNGKey(cfg.system.seed)
    devices = jax.devices()
    n_devices = len(devices)

    def replicate(x: Any) -> Any:
        """First replicate the update batch dim then put on devices."""
        x = jax.tree_map(lambda y: jnp.broadcast_to(y, (cfg.system.update_batch_size, *y.shape)), x)
        return jax.device_put_replicated(x, devices)

    # make envs
    env, eval_env = environments.make(cfg)

    # actions, agents, keysplits
    num_actions = int(env.action_spec().num_values[0])
    num_agents = env.action_spec().shape[0]

    key, q_key = jax.random.split(key, 2)


    # ALL INITS TO MAKE NETWORK -> need TBAx
    # initialise observation for the sizes
    init_obs = env.observation_spec().generate_value() # A,x
    init_obs_batched = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], cfg.system.n_envs, axis=0),
        init_obs,
    ) # n_envs, A ,x
    init_obs_batched = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], init_obs_batched) # add time dim (1,n_envs,A,x)
    init_terms = jnp.zeros((1, cfg.system.n_envs,num_agents), dtype=bool) # (1,n_envs,A)

    init_x = (init_obs_batched, init_terms)
    # Initialise hidden states.
    init_hstate = ScannedRNN.initialize_carry((cfg.system.n_envs, num_agents), hidden_size) #(1,n_envs, A,x)

    # Making recurrent Q network
    q = RecQNetwork(num_actions)
    q_params = q.init(q_key, init_hstate, init_x) 

    # Pack params
    params = QNetParams(q_params, q_params)

    # OPTIMISER
    q_opt = optax.adam(cfg.system.q_lr)
    q_opt_state = q_opt.init(params.online)

    # Distribute params and opt states across all devices
    params = replicate(params)
    q_opt_state = replicate(q_opt_state)

    h_state = replicate(init_hstate)

    # BUFFER CREATION INITS
    init_acts = env.action_spec().generate_value() #A,
    #init_acts_batched = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], init_acts)
    init_obs = env.observation_spec().generate_value() # A,x
    #init_obs_batched = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], init_obs)

    # Create replay buffer
    init_transition = Transition(
        obs=Observation(*init_obs),
        action=init_acts,
        reward=jnp.zeros((num_agents,), dtype=float),
        term=jnp.zeros((num_agents,), dtype=bool)
    )

    # default use rollout length of trajectories
    if cfg.system.recurrent_chunk_size:
        rec_chunk_size = cfg.system.recurrent_chunk_size
    else: 
        rec_chunk_size = cfg.system.rollout_length

    rb = fbx.make_trajectory_buffer(
        sample_sequence_length = rec_chunk_size,
        period = rec_chunk_size,
        add_batch_size = n_devices, # is this accurate?
        sample_batch_size = cfg.system.batch_size,
        max_length_time_axis=cfg.system.buffer_size,
        min_length_time_axis=cfg.system.buffer_min_size,
    )
    buffer_state = replicate(rb.init(init_transition))

    nns = Networks(q) 

    # Reset env.
    n_keys = cfg.system.n_envs * n_devices * cfg.system.update_batch_size
    key_shape = (n_devices, cfg.system.update_batch_size, cfg.system.n_envs, -1)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_keys)
    reset_keys = jnp.reshape(reset_keys, key_shape)

    # Keys passed to learner
    first_keys = jax.random.split(key, (n_devices * cfg.system.update_batch_size))
    first_keys = first_keys.reshape((n_devices, cfg.system.update_batch_size, -1))

    env_state, first_timestep = jax.pmap(  # devices
        jax.vmap(  # update_batch_size
            jax.vmap(env.reset),  # n_envs
            axis_name="batch",
        ),
        axis_name="device",
    )(reset_keys)

    first_obs = first_timestep.observation
    first_terms = ~(first_timestep.discount).astype(bool)#[:,0,:,0] # SAME AS BELOW #BTAx

    t = jnp.zeros((n_devices, cfg.system.update_batch_size), dtype=int)

    # Initial learner state.
    learner_state = RNNQLearnerState(
        h_state, first_terms, t, env_state, first_obs, buffer_state, params, first_keys, q_opt_state
    )
    return (env, eval_env), nns, Optimisers(q_opt), rb, learner_state, logger, key 


def make_update_fns(
    cfg: DictConfig,
    env: Environment,
    nns: Networks,
    opts: Optimisers,
    rb: TrajectoryBuffer,
    # target_entropy: chex.Array,
) -> Callable[[RNNQLearnerState], Tuple[RNNQLearnerState, Tuple[Metrics,Metrics]]]:
    
    q = nns.q 
    q_opt = opts.q

    # used everywhere a new hidden state needs to be initialised
    hidden_size = 256 #standardise and get from nw

    #___________________________________________________________________________________________________
    # INTERACTION FUNCTIONS
    #
    # - 1. select_step_store (overarching)
    #    - 1.1 make_transition
    #    - 1.2 (select_eps_greedy_action)
    #        - 1.2.1 get_epsilon
    #        - 1.2.2 greedy
    #        - 1.2.3 select_random_action_batch
    #            - 1.2.3.1 select_single_random_action

    # INTERACT LEVEL 4 (1.2.3.1)
    # Using probability values allows for non-ragged masking - must be done per row because of "choice".
    def select_single_random_action(key, action_options, p_values):
        """Select one action per row with probabilities used for masking."""
        action = jax.random.choice(key,action_options,p=p_values)
        return action
    
    # INTERACT LEVEL 3 (1.2.3)
    def select_random_action_batch(
            selection_key:chex.PRNGKey,obs:Observation
            ) -> Tuple[chex.Array, chex.Array]:
        """Select actions randomly with masking."""
        action_mask = obs.action_mask

        batch_size, n_agents, n_actions = action_mask.shape

        # get one key per agent
        keys = jax.random.split(selection_key,batch_size*n_agents)
        # base of action indexes
        action_options = jnp.arange(n_actions)
        # get num avail actions to generate probabilities for choosing
        num_avail_per_agent = jnp.sum(action_mask,axis=-1)[...,jnp.newaxis]
        # get probabilities (uniform with "masking")
        p_vals = action_mask.astype("int32")/num_avail_per_agent
        # flatten so that we only need one vmap over batch, agent dim
        p_vals = jnp.reshape(p_vals,(batch_size*n_agents,n_actions))
        # vmap random selection
        actions = jax.vmap(select_single_random_action, (0,None,0))(keys,action_options,p_vals)
        # unflatten to mtch expected action block size
        actions = jnp.reshape(actions,(batch_size,n_agents))#,1))

        return actions
    
    # INTERACT LEVEL 3 (1.2.2)
    def greedy(
            hidden_state, nw_params, obs, terms
            ) -> Tuple[chex.Array,chex.Array]:
        """Uses online q values to greedily select the best action."""

        obs = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], obs)
        terms = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], terms)
        # get q values
        new_hidden_state, q_values = q.apply(nw_params, hidden_state, (obs, terms)) #TODO: Make nw compatible

        # right now q_values is 1,1,n_envs,A,x
        q_values = q_values[0]

        # make unavailable actions quite negative
        q_vals_masked = jnp.where(obs.action_mask,q_values,jnp.zeros(q_values.shape)-1000) #TODO: action masking in nw
        # greedy argmax over inner dimension
        action = jnp.argmax(q_vals_masked, axis=-1)
        action = action[0] #remove redundant first dim

        return (action, new_hidden_state)
    
    
    # INTERACT LEVEL 3 (1.2.1)
    def get_epsilon(t:int):
        """Calculate epsilon for exploration rate using config hyperparameters."""
        eps = jax.numpy.maximum(
            cfg.system.eps_min,
            1-(t/cfg.system.eps_decay)*(1-cfg.system.eps_min)
            )
        return eps

    def get_should_explore(key,eps):
        return jax.random.uniform(key)<eps

    # INTERACT LEVEL 2 (1.2)
    def select_eps_greedy_action(
            select_params:ActionSelectionParams,obs: Observation
            ) -> Tuple[ActionSelectionParams,chex.Array]:
        """Select action to take in eps-greedy way. Batch and agent dims are included."""

        # unpacking and create keys
        nw_params, hidden_state, terms, t, key = select_params
        new_key, explore_key, selection_key = jax.random.split(key,3)

        # get exploration rate
        epsilon = get_epsilon(t)

        greedy_action, new_hidden_state = greedy(hidden_state, nw_params, obs, terms)
        random_action = select_random_action_batch(selection_key,obs)

        action = jax.lax.select(
            get_should_explore(explore_key,epsilon),
            random_action,
            greedy_action
        )

        # repack new selection params
        new_select_params = ActionSelectionParams(nw_params, new_hidden_state, terms, t, new_key)

        return new_select_params, action
    
    # INTERACT LEVEL 2->1
    # decide that yes, this is how we select actions
    select_action = select_eps_greedy_action

    # INTERACT LEVEL 2 (1.1)
    def make_transition(
            timestep,obs:Observation, action
            ) -> Transition:
        
        # preprocessing timestep data
        rewards = timestep.reward
        terms = ~(timestep.discount).astype(bool) #inverts discounts with ~
        transition = Transition(obs, action, rewards, terms)

        # TODO refactor code so this is unnecessary - add batch size should be what?
        transition = jax.tree_util.tree_map(lambda x: x[jnp.newaxis,:, ...], transition)

        return Transition(*transition)

    # INTERACT LEVEL 1 (1.)
    def select_step_store(
            interact_state:InteractionParams,_:Any
            ) -> Tuple[InteractionParams,Dict]:
        """Selects an action, steps global env, stores timesteps in global rb and repacks the parameters for the next step."""
        
        # light unpacking
        selection_params, env_state, buffer_state, obs = interact_state

        # select the actions to take
        new_selection_params, action = select_action(selection_params,obs) # should update hidden state, key

        # step env with selected actions
        new_env_state, timestep = jax.vmap(env.step)(env_state, action)

        # make and store transition
        transition = make_transition(timestep,obs,action)
        new_buffer_state = rb.add(buffer_state, transition)

        # repack and update interact state's dones
        new_obs = timestep.observation # NB step!!
        new_selection_params = ActionSelectionParams(new_selection_params.nw_params,new_selection_params.hidden_state,~(timestep.discount).astype(bool),new_selection_params.t,new_selection_params.key)

        new_interact_state = InteractionParams(new_selection_params, new_env_state, new_buffer_state, new_obs)

        return new_interact_state, timestep.extras["episode_metrics"]
    
    #___________________________________________________________________________________________________
    # TRAIN FUNCTIONS
    #
    # - 2. train (overarching)
    #    - 2.1 update_q
    #        - 1.2.1 q_loss_fn
    #        - 1.2.2 scan_apply
    # 

    # TRAIN LEVEL 3 (2.1.1)
    def q_loss_fn(
            q_online_params: FrozenVariableDict, obs: Array,terms: Array, action: Array, target: Array
            ) -> Tuple[Array, Metrics]:
        """The portion of the calculation to grad, namely online apply and mse with target."""
        q_online = scan_apply(q_online_params,obs,terms)
        q_online = jnp.take(q_online, action[...,jnp.newaxis])
        q_online = jnp.sum(q_online, -2, keepdims=True)
        q_loss = jnp.mean((q_online - target) ** 2)

        loss_info = {
            "q_loss": q_loss,
            "mean_q": jnp.mean(q_online),
        }

        return q_loss, loss_info
    
    # flashbax gives B,T, we need T,B for RNN automatic batching
    def BT_to_TB(arr:chex.Array):
        arr: Dict[str, chex.Array] = jax.tree_map(
              lambda x: jax.numpy.swapaxes(x,0,1), arr
          )
        return arr

    # TRAIN LEVEL 3 and 4 (2.1.2, 2.1.1.1)
    def scan_apply(params, obs:Observation, terms):
        """Applies RNN to a batch of trajectories by scanning over batch dim."""
        hidden_state = ScannedRNN.initialize_carry((cfg.system.batch_size, obs.agents_view.shape[2]), hidden_size)
        obs = BT_to_TB(obs)
        terms = BT_to_TB(terms)
        obs_terms = (obs,terms)
        # swap time and batch dims

        new_hidden_state, next_q_vals_online = q.apply(params,hidden_state,obs_terms) #TODO: DOES THIS SCAN??
        #new_hidden_state, next_q_vals_online = jax.lax.scan(q.apply(params=params))(hidden_state,obs_terms)
        return BT_to_TB(next_q_vals_online) # rename to switch_leading_dims

    # TRAIN LEVEL 2 (2.1)
    def update_q(
        params: QNetParams, opt_states: optax.OptState, data: Transition
    ) -> Tuple[QNetParams, optax.OptState, Metrics]:
        """Update the Q parameters."""

        # # Make all data blocks the same num dims and prep for RNN

        # get the data associated with obs
        data_first: Dict[str, chex.Array] = jax.tree_map(
              lambda x: x[:, :-1, ...], data
          )
        # get the data associated with next_obs
        data_next: Dict[str, chex.Array] = jax.tree_map(
              lambda x: x[:,1:, ...], data
          )
        
        # TODO make rewards and terms with extra dim
        #next_term = data_next.term[...,jnp.newaxis]
        first_reward = data_first.reward[...,jnp.newaxis] # grab agent zeros assumes same for all agents
        first_term = data_first.term[...,jnp.newaxis] # grab agent zeros
        #first_term = data_first.term[...,jnp.newaxis]


        # scan over each sample
        next_q_vals_online = scan_apply(params.online, data_next.obs, data_next.term)
        next_q_vals_online = jnp.where(data_next.obs.action_mask, next_q_vals_online, jnp.zeros(next_q_vals_online.shape)-9999999) #TODO: action masking in nw

        next_q_vals_target = scan_apply(params.target, data_next.obs, data_next.term)

        # double q-value selection
        next_action = jnp.argmax(next_q_vals_online, axis=-1)
        next_q_val = jnp.take(next_q_vals_target, next_action[...,jnp.newaxis])
        next_q_val = jnp.sum(next_q_val, -2, keepdims=True)


        target_q_val = (first_reward + (1.0 - first_term) * cfg.system.gamma * next_q_val)

        # Update Q function.
        q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            params.online, 
            data_first.obs, #is scanned over
            data_first.term, # is scanned over
            data_first.action, # should only be final value in each traj, B,A (adds axis in fn)
            target_q_val # B,A -> add dim?
            )

        # Mean over the device and batch dimension.
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="device")
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="batch")
        q_updates, new_q_opt_state = q_opt.update(q_grads, opt_states)
        new_online_q_params = optax.apply_updates(params.online, q_updates)

        # Target network polyak update.
        new_target_q_params = optax.incremental_update(
            new_online_q_params, params.target, cfg.system.tau
        )

        # Repack params and opt_states.
        new_params = QNetParams(new_online_q_params, new_target_q_params)

        return new_params, new_q_opt_state, q_loss_info

    # TRAIN LEVEL 1 (2.)
    def train(
        train_carry: Tuple[BufferState, QNetParams, optax.OptState, chex.PRNGKey], _: Any
    ) -> Tuple[Tuple[BufferState, QNetParams, optax.OptState, chex.PRNGKey], Metrics]:
        """Sample, train and repack."""

        # unpack and get keys
        buffer_state, params, opt_states, key = train_carry
        new_key, buff_key = jax.random.split(key, 2)

        # sample
        data = rb.sample(buffer_state, buff_key).experience

        # learn
        new_params, new_opt_states, q_loss_info = update_q(params, opt_states, data)

        return (buffer_state, new_params, new_opt_states, new_key), q_loss_info
    
    #___________________________________________________________________________________________________
    # INTERACT-TRAIN LOOP

    scanned_interact = lambda state: lax.scan(select_step_store, state, None, length=cfg.system.rollout_length)
    scanned_train = lambda state: lax.scan(train, state, None, length=cfg.system.epochs)

    #interact and train
    def update_step(carry: RNNQLearnerState, _: Any) -> Tuple[RNNQLearnerState, Tuple[Metrics, Metrics]]:
        """Interact, then learn. The _ at the end of a var means updated."""

        # unpack and get random keys
        hidden_state, terms, t, env_state, obs, buffer_state, params, key, opt_state = carry
        key, interact_key, train_key = jax.random.split(key, 3)

        # Select actions, step env and store transitions
        selection_params = ActionSelectionParams(params.online, hidden_state, terms, t, interact_key)
        interact_carry = InteractionParams(selection_params, env_state, buffer_state, obs)
        ((_, hidden_state_, terms_, _, _), env_state_, buffer_state_, obs_), metrics = scanned_interact(interact_carry)

        # Sample and learn
        train_carry = (buffer_state_, params, opt_state, train_key)
        (_, params_, opt_state_, _), losses = scanned_train(train_carry)

        t += cfg.system.n_envs * cfg.system.rollout_length
        return (
            RNNQLearnerState(hidden_state_, terms_, t, env_state_, obs_, buffer_state_, params_, key, opt_state_),
            (metrics, losses),
        )

    pmaped_steps = 1024  # todo: config option # NOTE (Claude) this is related to steps between logging.

    pmaped_updated_step = jax.pmap(
        jax.vmap(
            lambda state: lax.scan(update_step, state, None, length=pmaped_steps),
            axis_name="batch",
        ),
        axis_name="device",
        donate_argnums=0,
    )

    return pmaped_updated_step, greedy


def run_experiment(cfg: DictConfig) -> float:
    n_devices = len(jax.devices())

    pmaped_steps = 1024  # todo: config option
    steps_per_rollout = n_devices * cfg.system.n_envs * cfg.system.rollout_length * pmaped_steps
    max_episode_return = -jnp.inf

    (env, eval_env), nns, opts, rb, learner_state, logger, key = init(cfg)
    update, greedy = make_update_fns(cfg, env, nns, opts, rb)

    key, eval_key = jax.random.split(key)
    # todo: don't need to return trained_params or eval keys
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key=eval_key,
        network=nns, #NOTE (Louise) this part needs redoing with that replacement actor function
        params=learner_state.params.online,
        config=cfg,
        greedy=greedy
    )

    if cfg.logger.checkpointing.save_model:
        checkpointer = Checkpointer(
            metadata=cfg,  # Save all config as metadata in the checkpoint
            model_name=cfg.logger.system_name,
            **cfg.logger.checkpointing.save_args,  # Checkpoint args
        )

    start_time = time.time()


    def get_epsilon(t:int):
        """Calculate epsilon for exploration rate using config variables."""
        eps = jax.numpy.maximum(
            cfg.system.eps_min,
            1-(t/cfg.system.eps_decay)*(1-cfg.system.eps_min)
            )
        return eps

    # Main loop:
    # We want start to align with the final step of the first pmaped_learn,
    # where we've done explore_steps and 1 full learn step.
    start = steps_per_rollout
    for eval_idx, t in enumerate(range(start, int(cfg.system.total_timesteps), steps_per_rollout)):
        # Learn loop:
        learner_state, (metrics, losses) = update(learner_state)
        jax.block_until_ready(learner_state)

        # Log:
        # Multiply by learn steps here because anakin steps per second is learn + act steps
        # But we want to make sure we're counting env steps correctly so it's not included
        # in the loop counter.

        sps = t * cfg.system.epochs / (time.time() - start_time)
        final_metrics = episode_metrics.get_final_step_metrics(metrics)
        loss_metrics = losses #| {"log_alpha": learner_state.params.log_alpha}
        logger.log({"step": t, "steps_per_second": sps}, t, eval_idx, LogEvent.MISC)
        logger.log(final_metrics, t, eval_idx, LogEvent.ACT)
        logger.log(loss_metrics, t, eval_idx, LogEvent.TRAIN)
        logger.log({"epsilon":get_epsilon(t)}, t, eval_idx, LogEvent.MISC)

        # Evaluate:
        key, eval_key = jax.random.split(key)
        eval_keys = jax.random.split(eval_key, n_devices)
        # todo: bug likely here -> don't have batch vmap yet so shouldn't unreplicate_batch_dim
        eval_output = evaluator(unreplicate_batch_dim(learner_state.params.online), eval_keys)
        jax.block_until_ready(eval_output)

        # # Log:
        # episode_return = jnp.mean(eval_output.episode_metrics["episode_return"])
        # logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.EVAL)

        # # Save best actor params.
        # if cfg.arch.absolute_metric and max_episode_return <= episode_return:
        #     best_params = copy.deepcopy(unreplicate_batch_dim(learner_state.params.online))
        #     max_episode_return = episode_return

        # # Checkpoint:
        # if cfg.logger.checkpointing.save_model:
        #     # Save checkpoint of learner state
        #     unreplicated_learner_state = unreplicate_learner_state(learner_state)  # type: ignore
        #     checkpointer.save(
        #         timestep=t,
        #         unreplicated_learner_state=unreplicated_learner_state,
        #         episode_return=episode_return,
        #     )

    # # Measure absolute metric.
    # if cfg.arch.absolute_metric:
    #     eval_keys = jax.random.split(key, n_devices)

    #     eval_output = absolute_metric_evaluator(best_params, eval_keys)
    #     jax.block_until_ready(eval_output)

    #     logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.ABSOLUTE)

    # logger.stop()

    return float(max_episode_return)


@hydra.main(config_path="../configs", config_name="default_rec_idqn.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    final_return = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}IDQN experiment completed{Style.RESET_ALL}")

    return float(final_return)


if __name__ == "__main__":
    hydra_entry_point()
