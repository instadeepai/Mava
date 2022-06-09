import numpy as np
import copy
from mava.specs import MAEnvironmentSpec


def get_schema(environment):
    environment_spec = MAEnvironmentSpec(environment)
    agent_specs = environment_spec.get_agent_specs()

    schema = {}
    for agent in environment_spec.get_agent_ids():
        spec = agent_specs[agent]

        schema[agent + "_observations"] = spec.observations.observation
        schema[agent + "_legal_actions"] = spec.observations.legal_actions
        schema[agent + "_actions"] = spec.actions
        schema[agent + "_rewards"] = spec.rewards
        schema[agent + "_discounts"] = spec.discounts

    ## Extras
    # Zero-padding mask
    schema["zero_padding_mask"] = np.array(1, dtype=np.float32)

    # Global env state
    extras_spec = environment_spec.get_extra_specs()
    if "s_t" in extras_spec:
        schema["env_state"] = extras_spec["s_t"]

    schema["episode_return"] = np.array(0, dtype="float32")

    return schema
