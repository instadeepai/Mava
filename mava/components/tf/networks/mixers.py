from typing import Any

import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow import keras


class AdditiveMixingNetwork(snt.Module):
    """Multi-agent monotonic mixing architecture."""

    def __init__(self) -> None:
        """Initializes the mixer."""
        super(AdditiveMixingNetwork, self).__init__()

    def __call__(self, q_values: tf.Tensor) -> tf.Tensor:
        """Monotonic mixing logic."""
        return tf.math.reduce_sum(q_values, keepdims=True)


class QMixer(snt.Module):
    def __init__(self, n_agents: Any, state_shape: Any, mixing_embed_dim: Any) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = mixing_embed_dim
        self.state_dim = int(np.prod(state_shape))

        self.hyper_w_1 = snt.Linear(self.embed_dim * self.n_agents)
        self.hyper_w_final = snt.Linear(self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = snt.Linear(self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = snt.Sequential(
            [snt.Linear(self.embed_dim), tf.keras.layers.ReLU(), snt.Linear(1)]
        )

        self.linear_layer = snt.Sequential(
            [snt.Linear(self.n_agents)]  # , tf.keras.layers.ReLU()]
        )

    def __call__(self, agent_qs: Any, states: Any) -> tf.Tensor:
        """Forward pass for the mixer.
        Args:
            agent_qs: Tensor of shape [B, n_agents]
            states: Tensor of shape [B, state_dim]
        """
        bs = agent_qs.shape[0]
        states = tf.reshape(states, (-1, self.state_dim))
        print("states reshape: ", states.shape)
        agent_qs = tf.reshape(agent_qs, (-1, self.n_agents))
        print("agent_qs: ", agent_qs.shape)
        # First layer
        w1 = tf.math.abs(self.hyper_w_1(states))
        print("w1: ", w1.shape)
        b1 = self.hyper_b_1(states)
        print("b1: ", b1.shape)
        w1 = tf.reshape(w1, (-1, self.n_agents, self.embed_dim))
        print("w1: ", w1.shape)
        b1 = tf.reshape(b1, (-1, self.embed_dim))
        print("b1: ", b1.shape)
        hidden = tf.keras.activations.elu(tf.linalg.matmul(agent_qs, w1) + b1)
        print("hidden: ", hidden.shape)
        hidden = tf.reshape(hidden, (-1, self.embed_dim))
        # Second layer
        w_final = tf.math.abs(self.hyper_w_final(states))
        print("w_final: ", w_final.shape)
        w_final = tf.reshape(w_final, (-1, self.embed_dim))
        print("w_final: ", w_final.shape)
        # State-dependent bias
        v = tf.reshape(self.V(states), (-1, 1))
        print("v: ", v.shape)
        # Compute final output

        # print("hidden: ", hidden.shape)
        # print("w_final: ", w_final.shape)
        # print("v: ", v.shape)

        y = tf.linalg.matmul(hidden, w_final, transpose_b=True) + v
        # print("tf.linalg.matmul(hidden, w_final): ", tf.linalg.matmul(hidden, w_final).shape)
        print("y: ", y.shape)

        # Reshape and return
        q_tot = tf.reshape(y, (bs, -1))
        # print("q_tot: ", q_tot.shape)
        return q_tot

    # hidden: (256, 1, 32)
    # w_final: (256, 32, 1)
    # tf.linalg.matmul(hidden, w_final): (256, 1, 1)

    # v: (256, 1, 1)

    # y: (256, 1, 1)
    # q_tot: (256, 1, 1)

    # def __call__(self, agent_qs: Any, states: Any) -> tf.Tensor:
    #     """Forward pass for the mixer.
    #     Args:
    #         agent_qs: Tensor of shape [B, n_agents]
    #         states: Tensor of shape [B, state_dim]
    #     """
    #     bs = agent_qs.shape[0]
    #     states = tf.reshape(states, (-1, self.state_dim))
    #     # First layer
    #     # print("states: ", states.shape) # 1, 19
    #     w = self.linear_layer(states)
    #
    #     w = tf.math.abs(w)
    #
    #     # print("agent_qs: ", agent_qs.shape)
    #     # print("w: ", w.shape) # 1, 3
    #
    #     # out = agent_qs*w
    #
    #     # y: (256, 1, 1)
    #     # q_tot: (256, 1, 1)
    #     # hidden: (256, 1, 256)
    #     # w_final: (256, 256, 1)
    #     # v: (256, 1, 1)
    #     # tf.linalg.matmul(hidden, w_final): (256, 1, 1)
    #     # y: (256, 1, 1)
    #     # q_tot: (256, 1, 1)
    #
    #     y = tf.linalg.matmul(hidden, w_final) # + v
    #     q_tot = tf.reshape(y, (bs, -1, 1))
    #     return q_tot
