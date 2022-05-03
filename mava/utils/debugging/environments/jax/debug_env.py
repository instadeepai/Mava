from typing import Dict

import jax.lax as lax
import jax.numpy as jnp
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from gym import spaces

from mava.types import OLT
from mava.utils.debugging.environments.jax.core import Action, Agent
from mava.utils.debugging.environments.jax.debug_env_base import MultiAgentJaxEnvBase


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

    # Convert Debugging environment observation so it's dm_env compatible.
    # Also, the list of legal actions must be converted to a legal actions mask.
    def _convert_observations(
        self, observes: Dict[str, jnp.ndarray], dones: Dict[str, bool]
    ) -> Dict[str, OLT]:
        observations: Dict[str, OLT] = {}
        for agent, observation in observes.items():
            if isinstance(observation, dict) and "action_mask" in observation:
                legals = observation["action_mask"]
                observation = observation["observation"]
            else:
                # TODO Handle legal actions better for continuous envs,
                #  maybe have min and max for each action and clip the agents actions
                #  accordingly
                if isinstance(self.action_spaces[agent], spaces.Discrete):
                    legals = jnp.ones(
                        _convert_to_spec(self.action_spaces[agent]).num_values,
                        dtype=self.action_spaces[agent].dtype,
                    )
                else:
                    legals = jnp.ones(
                        _convert_to_spec(self.action_spaces[agent]).shape,
                        dtype=self.action_spaces[agent].dtype,
                    )

            observation = jnp.array(observation, dtype=jnp.float32)
            observations[agent] = OLT(
                observation=observation,
                legal_actions=legals,
                terminal=jnp.asarray([dones[agent]], dtype=jnp.float32),
            )

        return observations

    def observation_spec(self) -> Dict[str, OLT]:
        observation_specs = {}
        for agent in self.agent_ids:

            # Legals spec
            if isinstance(self.action_spaces[agent], spaces.Discrete):
                legals = jnp.ones(
                    _convert_to_spec(self.action_spaces[agent]).num_values,
                    dtype=self.action_spaces[agent].dtype,
                )
            else:
                legals = jnp.ones(
                    _convert_to_spec(self.action_spaces[agent]).shape,
                    dtype=self.action_spaces[agent].dtype,
                )

            observation_specs[agent] = OLT(
                observation=_convert_to_spec(self.observation_spaces[agent]),
                legal_actions=legals,
                terminal=specs.Array((1,), jnp.float32),
            )
        return observation_specs
