# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

from ConfigSpace import Categorical, ConfigurationSpace

tuner_config_space = ConfigurationSpace(
    {
        "ppo_epochs": Categorical("ppo_epochs", [2, 4, 8]),
        "num_minibatches": Categorical("num_minibatches", [2, 4, 8]),
        "ent_coef": Categorical("ent_coef", [0.0, 1e-2, 1e-5]),
        "clip_eps": Categorical("clip_eps", [0.05, 0.1, 0.2]),
        "max_grad_norm": Categorical("max_grad_norm", [0.5, 5.0, 10.0]),
        "critic_lr": Categorical("critic_lr", [1e-4, 2.5e-4, 5e-4]),
        "actor_lr": Categorical("actor_lr", [1e-4, 2.5e-4, 5e-4]),
    }
)
