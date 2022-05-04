import jax
import haiku as hk


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
                block = ResidualBlock(
                    nurelum_channels, name="residual_{}_{}".format(i, j)
                )
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
        embedding_dim=None,
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
        atari = AtariDeepTorso(num_channels=self.num_channels, num_blocks=self.num_blocks)

        x = embed(x)
        return atari(x)

