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

# type: ignore

import os
import pickle

from acme.tf import savers as tf2_savers


def save_policy_and_obs(architecture, agent_type, checkpoint_subpath, pickle_save_loc):
    policy_networks = architecture.create_actor_variables()
    before_sum = policy_networks["policies"][agent_type].variables[1].numpy().sum()
    objects_to_save = {
        "policy": policy_networks["policies"][agent_type],
        "observation": policy_networks["observations"][agent_type],
    }

    checkpointer_dir = os.path.join(checkpoint_subpath, agent_type)
    tf2_savers.Checkpointer(
        time_delta_minutes=1,
        add_uid=False,
        directory=checkpointer_dir,
        objects_to_save=objects_to_save,
        enable_checkpointing=True,
    )
    after_sum = policy_networks["policies"][agent_type].variables[1].numpy().sum()

    assert before_sum != after_sum

    # Save policy variables
    policy = policy_networks["policies"][agent_type].set.variables
    policy_file = open(os.path.join(pickle_save_loc, "policy.obj"), "wb")
    pickle.dump(policy, policy_file)
    policy_file.close()

    # So observation variables
    observation = policy_networks["observations"][agent_type].variables
    obs_file = open(os.path.join(pickle_save_loc, "observations.obj"), "wb")
    pickle.dump(observation, obs_file)
    obs_file.close()


def load_policy_and_obs(pickle_save_loc):
    # Load policy variables
    filehandler = open(os.path.join(pickle_save_loc, "policy.obj"), "rb")
    policy_net = pickle.load(filehandler)

    # Load observation variables
    filehandler = open(os.path.join(pickle_save_loc, "policy.obj"), "rb")
    obs_net = pickle.load(filehandler)

    return policy_net, obs_net
