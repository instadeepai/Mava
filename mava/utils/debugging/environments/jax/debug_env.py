from mava.utils.debugging.environments.jax.core import Agent, Action
from mava.utils.debugging.environments.jax.debug_env_base import MultiAgentJaxEnvBase

import jax.numpy as jnp
import jax.lax as lax


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
