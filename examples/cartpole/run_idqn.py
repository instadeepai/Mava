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

# pip install ray[rllib]
from ray.rllib.env.multi_agent_env import make_multi_agent

# from mava.environment_loop import ParallelEnvironmentLoop
from mava.wrappers import RLLibMultiAgentEnvWrapper

ma_cartpole_cls = make_multi_agent("CartPole-v1")
ma_cartpole = ma_cartpole_cls({"num_agents": 2})
wrapped_env = RLLibMultiAgentEnvWrapper(ma_cartpole)
