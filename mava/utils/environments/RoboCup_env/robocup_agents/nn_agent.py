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
from typing import Any, Dict, Mapping, MutableSequence, Sequence, Union

import numpy as np
import sonnet as snt

# import logging
import tensorflow as tf
import tensorflow_probability as tfp
from acme import types
from acme.tf import networks
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils

from mava import specs as mava_specs
from mava.components.tf.architectures import StateBasedQValueCritic
from mava.components.tf.networks.mad4pg import DiscreteValuedHead
from mava.utils.environments.RoboCup_env.robocup_utils.util_functions import SpecWrapper

tfd = tfp.distributions


def set_gpu_affinity(gpus: Any) -> Any:
    devices = tf.config.list_physical_devices("GPU")

    # process the requested gpus
    if gpus is None or len(devices) == 0:
        gpus = {}
    elif isinstance(gpus, int):
        gpus = set(devices if gpus == -1 else [devices[gpus]])
    elif isinstance(gpus, MutableSequence):
        gpus = {devices[k] for k in gpus}
    else:
        raise ValueError(
            f"Can't parse requested gpus, got {gpus} of type {type(gpus)}."
        )

    # set the visibility
    # if the set is empty, the function will be called on an empty list
    # so that no `LogicalDevice` will be created on the physical devices.
    tf.config.set_visible_devices(list(gpus), "GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        [gpu.name.split(":")[-1] for gpu in gpus]
    )

    # set memory growth
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    return gpus


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        256,
        256,
        256,
    ),
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    shared_weights: bool = True,
    sigma: float = 0.3,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""
    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    if isinstance(policy_networks_layer_sizes, Sequence):
        policy_networks_layer_sizes = {
            key: policy_networks_layer_sizes for key in specs.keys()
        }
    if isinstance(critic_networks_layer_sizes, Sequence):
        critic_networks_layer_sizes = {
            key: critic_networks_layer_sizes for key in specs.keys()
        }

    observation_networks = {}
    policy_networks = {}
    critic_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = np.prod(specs[key].actions.shape, dtype=int)

        # Create the shared observation network; here simply a state-less operation.
        observation_network = tf2_utils.to_sonnet_module(tf2_utils.batch_concat)

        # Create the policy network.
        policy_network = snt.Sequential(
            [
                observation_network,
                networks.LayerNormMLP(
                    policy_networks_layer_sizes[key], activate_final=True
                ),
                networks.NearZeroInitializedLinear(num_dimensions),
                networks.TanhToSpec(specs[key].actions),
                networks.ClippedGaussian(sigma),
                networks.ClipToSpec(specs[key].actions),
            ]
        )

        # Create the critic network.
        critic_network = snt.Sequential(
            [
                # The multiplexer concatenates the observations/actions.
                networks.CriticMultiplexer(),
                networks.LayerNormMLP(
                    critic_networks_layer_sizes[key], activate_final=False
                ),
                snt.Linear(1),
            ]
        )
        observation_networks[key] = observation_network
        policy_networks[key] = policy_network
        critic_networks[key] = critic_network

    return {
        "policies": policy_networks,
        "critics": critic_networks,
        "observations": observation_networks,
    }


def make_recurrent_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        256,
        256,
        256,
    ),
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    shared_weights: bool = True,
    sigma: float = 0.3,
    vmin: float = -150.0,
    vmax: float = 150.0,
    num_atoms: int = 51,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""
    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    if isinstance(policy_networks_layer_sizes, Sequence):
        policy_networks_layer_sizes = {
            key: policy_networks_layer_sizes for key in specs.keys()
        }
    if isinstance(critic_networks_layer_sizes, Sequence):
        critic_networks_layer_sizes = {
            key: critic_networks_layer_sizes for key in specs.keys()
        }

    observation_networks = {}
    policy_networks = {}
    critic_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = np.prod(specs[key].actions.shape, dtype=int)

        # Create the observation network.
        observation_network = tf2_utils.to_sonnet_module(tf2_utils.batch_concat)

        # Create the policy network.
        policy_network = snt.DeepRNN(
            [
                observation_network,
                snt.Flatten(),
                snt.nets.MLP(policy_networks_layer_sizes[key]),
                snt.LSTM(25),
                snt.nets.MLP([128]),
                networks.NearZeroInitializedLinear(num_dimensions),
                networks.TanhToSpec(specs[key].actions),
                networks.ClippedGaussian(sigma),
                networks.ClipToSpec(specs[key].actions),
            ]
        )

        # Create the critic network.
        critic_network = snt.Sequential(
            [
                # The multiplexer concatenates the observations/actions.
                networks.CriticMultiplexer(),
                networks.LayerNormMLP(
                    critic_networks_layer_sizes[key], activate_final=False
                ),
                DiscreteValuedHead(vmin, vmax, num_atoms),
            ]
        )

        observation_networks[key] = observation_network
        policy_networks[key] = policy_network
        critic_networks[key] = critic_network

    return {
        "observations": observation_networks,
        "policies": policy_networks,
        "critics": critic_networks,
    }


class NNBot(object):
    def __init__(
        self, agent_type, num_players, checkpoint_subpath: str = "Checkpoints"
    ):
        # Convert action and observation specs.
        spec_wrapper = SpecWrapper(num_players)

        # setup the resources
        set_gpu_affinity(0)

        environment_spec = mava_specs.MAEnvironmentSpec(spec_wrapper)

        # Create networks
        shared_weights = True
        system_networks = make_networks(
            environment_spec=environment_spec, shared_weights=shared_weights
        )

        architecture = StateBasedQValueCritic(
            environment_spec=environment_spec,
            policy_networks=system_networks["policies"],
            critic_networks=system_networks["critics"],
            observation_networks=system_networks["observations"],
            shared_weights=shared_weights,
        )

        # Create the policy_networks
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

        self.policy = policy_networks["policies"][agent_type]
        self.num_players = num_players

    def reset_brain(self):
        pass

    def get_action(self, observation):
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # Compute the policy, conditioned on the observation.
        policy = self.policy(batched_observation)

        # Sample from the policy if it is stochastic.
        action = policy.sample() if isinstance(policy, tfd.Distribution) else policy
        return action.numpy()[0]
