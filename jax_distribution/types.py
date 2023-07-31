from typing import Dict, NamedTuple

import chex


class PPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: Dict
