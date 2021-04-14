import importlib
from typing import Any, Dict, Mapping, Sequence, Union

import dm_env
import sonnet as snt
import tensorflow as tf
import trfl
from absl import app, flags
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils

from mava import specs as mava_specs
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import idqn
from mava.wrappers.pettingzoo import PettingZooParallelEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 10000, "Number of training episodes to run for.")

flags.DEFINE_integer(
    "num_episodes_per_eval",
    100,
    "Number of training episodes to run between evaluation " "episodes.",
)


def make_environment(
    env_class: str = "mpe", env_name: str = "simple_v2", **kwargs: int
) -> dm_env.Environment:
    """Creates a MPE environment."""
    env_module = importlib.import_module(f"pettingzoo.{env_class}.{env_name}")
    env = env_module.parallel_env(**kwargs)  # type: ignore
    environment = PettingZooParallelEnvWrapper(env)
    return environment


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    q_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (256, 256),
    shared_weights: bool = False,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""

    specs = environment_spec.get_agent_specs()

    if isinstance(q_networks_layer_sizes, Sequence):
        q_networks_layer_sizes = {key: q_networks_layer_sizes for key in specs.keys()}

    observation_networks = {}
    q_networks = {}
    behavior_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = specs[key].actions.num_values

        # Create the shared observation network
        observation_network = tf2_utils.to_sonnet_module(tf.identity)

        # Create the policy network.
        q_network = snt.Sequential(
            [
                networks.LayerNormMLP(q_networks_layer_sizes[key], activate_final=True),
                networks.NearZeroInitializedLinear(num_dimensions),
            ]
        )

        epsilon = tf.Variable(
            0.1, trainable=False
        )  # fixed for now. not sure where to update it. maybe in the learner
        behavior_network = snt.Sequential(
            [
                q_network,
                lambda q: tf.cast(
                    trfl.epsilon_greedy(q, epsilon=epsilon).sample(), "int64"
                ),
            ]
        )

        observation_networks[key] = observation_network
        q_networks[key] = q_network
        behavior_networks[key] = behavior_network

    return {
        "q_networks": q_networks,
        "observations": observation_networks,
        "behaviors": behavior_networks,
    }


def main(_: Any) -> None:
    # Create an environment, grab the spec, and use it to create networks.
    environment = make_environment(max_cycles=25)
    environment_spec = mava_specs.MAEnvironmentSpec(environment)
    system_networks = make_networks(environment_spec)

    # Construct the agent.
    system = idqn.IDQN(
        environment_spec=environment_spec,
        q_networks=system_networks["q_networks"],
        observation_networks=system_networks[
            "observations"
        ],  # pytype: disable=wrong-arg-types
        behavior_networks=system_networks["behaviors"],
    )

    # Create the environment loop used for training.
    train_loop = ParallelEnvironmentLoop(environment, system, label="train_loop")

    # TODO fix the eval loop

    # Create the evaluation policy.
    # eval_policies = {
    #     key: snt.Sequential(
    #         [system_networks["q_networks"][key], lambda q: tf.math.argmax(q, axis=1)]
    #     )
    #     for key in environment_spec.get_agent_specs().keys()
    # }

    # # Create the evaluation actor and loop.
    # eval_actor = executors.FeedForwardExecutor(policy_networks=eval_policies)
    # eval_env = make_environment()
    # eval_loop = ParallelEnvironmentLoop(
    #     eval_env, eval_actor, label="eval_loop"
    # )

    for _ in range(FLAGS.num_episodes // FLAGS.num_episodes_per_eval):
        train_loop.run(num_episodes=FLAGS.num_episodes_per_eval)
        # eval_loop.run(num_episodes=1)


if __name__ == "__main__":
    app.run(main)
