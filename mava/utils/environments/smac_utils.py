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
from mava.wrappers.env_preprocess_wrappers import ConcatAgentIdToObservation, ConcatPrevActionToObservation
from smac.env import StarCraft2Env
from mava.wrappers import SMACWrapper

def make_environment(map_name="3m", concat_prev_actions=True, concat_agent_id=True, evaluation = False, random_seed=None):
    env = StarCraft2Env(map_name=map_name, seed=random_seed)
    
    env = SMACWrapper(env)

    if concat_prev_actions:
        env = ConcatPrevActionToObservation(env)
    
    if concat_agent_id:
        env = ConcatAgentIdToObservation(env)

    return env