# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

"""Example running MADDPG on debug MPE environments."""
from mava import specs as mava_specs
from absl import flags

from mava.systems.tf import maddpg, mad4pg, madqn, mappo, qmix, vdn
from mava.utils.environments import debugging_utils
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "simple_spread",
    "Debugging environment name (str).",
)
flags.DEFINE_string(
    "action_space",
    "discrete",
    "Environment action space type (str).",
)

class TestNetworks():
    # Test that we can load a env module and that it contains agents,
    #   agents and possible_agents.
    def test_network_keys(self) -> None:
         # Environment.
        env = debugging_utils.make_environment(env_name=FLAGS.env_name, action_space=FLAGS.action_space,evaluation=False)
        environment_spec = mava_specs.MAEnvironmentSpec(env)


        system_fns = [maddpg, mad4pg, madqn, mappo, qmix, vdn]

        for system in system_fns:
            # Test shared weights setup.
            agent_net_keys = {
                "agent_0": "agent",
                "agent_1": "agent",
                "agent_2": "agent",
            }
            networks = system.make_default_networks(environment_spec=environment_spec, # type: ignore
                        agent_net_keys=agent_net_keys)
            
            for key in networks.keys():
                assert ["agent"] == list(networks[key].keys())

            # Test individual agent network setup.
            agent_net_keys = {
                "agent_0": "agent_0",
                "agent_1": "agent_1",
                "agent_2": "agent_2",
            }
            networks = system.make_default_networks(environment_spec=environment_spec, # type: ignore
                        agent_net_keys=agent_net_keys)
            for key in networks.keys():
                assert ["agent_0", "agent_1", "agent_2"] == list(networks[key].keys())

TestNetworks().test_network_keys()
