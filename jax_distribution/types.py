from typing import NamedTuple

import chex


class Transition(NamedTuple):
    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: dict
