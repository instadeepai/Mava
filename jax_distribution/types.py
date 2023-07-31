from typing import Dict, NamedTuple

import chex
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from optax._src.base import OptState

from jax_distribution.wrappers.jumanji import LogEnvState


class PPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: Dict


class RunnerState(NamedTuple):
    """State of the `Runner`."""

    params: FrozenDict
    opt_state: OptState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class Output(NamedTuple):
    """Output of the `Learner`."""

    runner_state: RunnerState
    info: Dict[str, Dict[str, chex.Array]]
