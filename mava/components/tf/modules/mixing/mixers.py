import sonnet as snt
import tensorflow as tf


@snt.allow_empty_variables
class BaseMixer(snt.Module):
    """Base mixing class.

    Base mixer should take in agent q-values and environment global state tensors.
    """

    def __init__(self) -> None:
        """Initialise base mixer."""
        super().__init__()

    def __call__(self, agent_qs: tf.Tensor, states: tf.Tensor) -> tf.Tensor:
        """Call method."""
        return agent_qs


class VDN(BaseMixer):
    """VDN mixing network."""

    def __init__(self) -> None:
        """Initialise VDN mixer."""
        super().__init__()

    def __call__(self, agent_qs: tf.Tensor, states: tf.Tensor) -> tf.Tensor:
        """Call method."""
        return tf.reduce_sum(agent_qs, axis=-1, keepdims=True)


class QMIX(BaseMixer):
    """QMIX mixing network."""

    def __init__(
        self, num_agents: int, embed_dim: int = 32, hypernet_embed: int = 64
    ) -> None:
        """Inialize QMIX mixing network

        Args:
            num_agents: Number of agents in the enviroment
            state_dim: Dimensions of the global environment state
            embed_dim: The dimension of the output of the first layer
                of the mixer.
            hypernet_embed: Number of units in the hyper network
        """

        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.hypernet_embed = hypernet_embed

        self.hyper_w_1 = snt.Sequential(
            [
                snt.Linear(self.hypernet_embed),
                tf.nn.relu,
                snt.Linear(self.embed_dim * self.num_agents),
            ]
        )

        self.hyper_w_final = snt.Sequential(
            [snt.Linear(self.hypernet_embed), tf.nn.relu, snt.Linear(self.embed_dim)]
        )

        # State dependent bias for hidden layer
        self.hyper_b_1 = snt.Linear(self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = snt.Sequential([snt.Linear(self.embed_dim), tf.nn.relu, snt.Linear(1)])

    def __call__(self, agent_qs: tf.Tensor, states: tf.Tensor) -> tf.Tensor:
        """Call method."""
        bs = agent_qs.shape[1]
        state_dim = states.shape[-1]

        agent_qs = tf.reshape(agent_qs, (-1, 1, self.num_agents))
        states = tf.reshape(states, (-1, state_dim))

        # First layer
        w1 = tf.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = tf.reshape(w1, (-1, self.num_agents, self.embed_dim))
        b1 = tf.reshape(b1, (-1, 1, self.embed_dim))
        hidden = tf.nn.elu(tf.matmul(agent_qs, w1) + b1)

        # Second layer
        w_final = tf.abs(self.hyper_w_final(states))
        w_final = tf.reshape(w_final, (-1, self.embed_dim, 1))

        # State-dependent bias
        v = tf.reshape(self.V(states), (-1, 1, 1))

        # Compute final output
        y = tf.matmul(hidden, w_final) + v

        # Reshape and return
        q_tot = tf.reshape(y, (-1, bs, 1))  # [T, B, 1]

        return q_tot
