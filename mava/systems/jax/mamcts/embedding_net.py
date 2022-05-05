from typing import Callable, Dict, Any, Sequence, Optional

import jax
import jax.numpy as jnp
import haiku as hk
from acme.jax import utils

from mava import specs
from acme.jax import networks as networks_lib


class ResidualBlock(hk.Module):
    """Residual block."""

    def __init__(self, num_channels, name=None):
        super().__init__(name=name)
        self._num_channels = num_channels

    def __call__(self, x):
        main_branch = hk.Sequential(
            [
                jax.nn.relu,
                hk.Conv2D(
                    self._num_channels,
                    kernel_shape=[3, 3],
                    stride=[1, 1],
                    padding="SAME",
                ),
                jax.nn.relu,
                hk.Conv2D(
                    self._num_channels,
                    kernel_shape=[3, 3],
                    stride=[1, 1],
                    padding="SAME",
                ),
            ]
        )
        return main_branch(x) + x


class AtariDeepTorso(hk.Module):
    """Deep torso for Atari, from the IMPALA paper."""

    def __init__(
        self,
        name=None,
        num_channels=[16, 32, 32],
        num_blocks=[2, 2, 2],
    ):
        super().__init__(name=name)
        self.num_channels = num_channels
        self.num_blocks = num_blocks

    def __call__(self, x):
        torso_out = x

        for i, (num_channels, num_blocks) in enumerate(
            zip(self.num_channels, self.num_blocks)
        ):
            conv = hk.Conv2D(
                num_channels, kernel_shape=[3, 3], stride=[1, 1], padding="SAME"
            )
            torso_out = conv(torso_out)
            # TODO: should this max pool be here or should it be at the end of all the blocks
            torso_out = hk.max_pool(
                torso_out,
                window_shape=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding="SAME",
            )
            for j in range(num_blocks):
                block = ResidualBlock(num_channels, name="residual_{}_{}".format(i, j))
                torso_out = block(torso_out)

        torso_out = jax.nn.relu(torso_out)
        torso_out = hk.Flatten()(torso_out)
        torso_out = hk.Linear(256)(torso_out)
        torso_out = jax.nn.relu(torso_out)
        return torso_out


class EmbeddingGridModel(hk.Module):
    def __init__(
        self,
        name=None,
        vocab_size=128,
        embedding_dim=8,
        num_channels=[16, 32, 32],
        num_blocks=[2, 2, 2],
    ):
        super().__init__(name=name)
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim

    def __call__(self, x):
        embed = hk.Embed(self.vocab_size, self.embed_dim)
        atari = AtariDeepTorso(
            num_channels=self.num_channels, num_blocks=self.num_blocks
        )

        x = embed(x)
        return atari(x)


def make_discrete_embedding_networks(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    network_wrapper_fn: Callable[
        [networks_lib.FeedForwardNetwork, Dict[str, jnp.ndarray]], Any
    ],
    policy_layer_sizes: Sequence[int],
    critic_layer_sizes: Sequence[int],
    vocab_size: int = 128,
    embedding_dim: Optional[int] = 8,
    num_channels: Sequence[int] = [16, 32, 32],
    num_blocks: Sequence[int] = [2, 2, 2],
):
    """TODO: Add description here."""

    num_actions = environment_spec.actions.num_values

    # TODO (dries): Investigate if one forward_fn function is slower
    # than having a policy_fn and critic_fn. Maybe jit solves
    # this issue. Having one function makes obs network calculations
    # easier.
    def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        # inputs = inputs.astype(jnp.int32)
        policy_value_network = hk.Sequential(
            [
                utils.batch_concat,
                EmbeddingGridModel(
                    "EmbeddingGridModel",
                    vocab_size,
                    embedding_dim,
                    num_channels,
                    num_blocks,
                ),
                networks_lib.CategoricalValueHead(num_values=num_actions),
            ]
        )
        return policy_value_network(inputs)

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    network_key, key = jax.random.split(key)
    params = forward_fn.init(network_key, dummy_obs)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return network_wrapper_fn(
        network=forward_fn,
        params=params,
    )
