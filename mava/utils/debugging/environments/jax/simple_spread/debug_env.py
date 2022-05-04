from typing import Dict

import jax.lax as lax
import jax.numpy as jnp
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from gym import spaces

from mava.types import OLT
from mava.utils.debugging.environments.jax.simple_spread.core import Action, Agent
from mava.utils.debugging.environments.jax.simple_spread.debug_env_base import (
    MultiAgentJaxEnvBase,
)


class MAJaxDiscreteDebugEnv(MultiAgentJaxEnvBase):
    def _process_action(self, action: int, agent: Agent) -> Action:
        agent.action.u = jnp.zeros(self.dim_p)

        def on_movable(act: Action):
            sensitivity = lax.cond(
                jnp.isnan(agent.accel), lambda: 5.0, lambda: agent.accel
            )

            return Action(
                u=jnp.array(
                    lax.switch(
                        act - 1,
                        [
                            lambda x: [-1.0, 0.0],
                            lambda x: [1.0, 0.0],
                            lambda x: [0.0, -1.0],
                            lambda x: [0.0, 1.0],
                        ],
                        None,
                    )
                )
                * sensitivity
            )

        return lax.cond(agent.movable, on_movable, lambda _: agent.action, action)
