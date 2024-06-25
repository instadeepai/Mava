
import copy
import time
from typing import Any, Dict, Tuple, List
import threading
import chex
import flax
import gym.vector
import gym.vector.async_vector_env
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import queue
from collections import deque
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from optax._src.base import OptState
from rich.pretty import pprint

#from mava.evaluator import make_eval_fns
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardValueNet as Critic
from mava.systems.anakin.ppo.types import LearnerState, OptStates, Params, PPOTransition #todo: change this
from mava.types import ActorApply, CriticApply, ExperimentOutput, LearnerFn, Observation
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.total_timestep_checker import anakin_check_total_timesteps
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics
from flax import linen as nn
import gym
import rware 
from mava.wrappers import GymRwareWrapper, GymRecordEpisodeMetrics,  _multiagent_worker_shared_memory
@hydra.main(config_path="../../../configs", config_name="default_ff_ippo_seb.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    

    OmegaConf.set_struct(cfg, False)
    def f():
        base = gym.make(cfg.env.scenario)
        base = GymRwareWrapper(base, cfg.env.use_individual_rewards, False, True)
        return GymRecordEpisodeMetrics(base)
    
    base =  gym.vector.AsyncVectorEnv(  # todo : give them more descriptive names
        [
            lambda: f()
            for _ in range(3)
        ],
         worker=_multiagent_worker_shared_memory
    )
    base.reset()
    n = 0
    done = False
    while not done:
        n+= 1
        agents_view, reward, terminated, truncated, info = base.step([[0,0,0], [0,0,0]])
        done = np.logical_or(terminated, truncated).all()
        metrics = jax.tree_map(lambda *x : jnp.asarray(x), *info["metrics"])
        print(n, done, terminated, np.logical_or(terminated, truncated).shape, metrics)
        done = True
    base.close()
    print(done)
    

    #print(b)
    #r = 1+1
    # Create a sample input
    #env = gym.make(cfg.env.scenario)
    #env.reset()
    #a = env.step(jnp.ones((4)))

hydra_entry_point()
