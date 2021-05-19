"""Distributions, for use in acme/networks/distributional.py."""
from typing import Tuple, Union

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme.tf.networks.distributions import (
    DiscreteValuedDistribution as acme_DiscreteValuedDistribution,
)
from tensorflow_probability import distributions as tfd


class DiscreteValuedHead(snt.Module):
    """Represents a parameterized discrete valued distribution.

    The returned distribution is essentially a `tfd.Categorical`, but one which
    knows its support and so can compute the mean value.
    """

    def __init__(
        self,
        vmin: Union[float, np.ndarray, tf.Tensor],
        vmax: Union[float, np.ndarray, tf.Tensor],
        num_atoms: int,
        w_init: snt.initializers.Initializer = None,
        b_init: snt.initializers.Initializer = None,
    ):
        """Initialization.

        If vmin and vmax have shape S, this will store the category values as a
        Tensor of shape (S*, num_atoms).

        Args:
          vmin: Minimum of the value range
          vmax: Maximum of the value range
          num_atoms: The atom values associated with each bin.
          w_init: Initialization for linear layer weights.
          b_init: Initialization for linear layer biases.
        """
        super().__init__(name="DiscreteValuedHead")
        vmin = tf.convert_to_tensor(vmin)
        vmax = tf.convert_to_tensor(vmax)
        self._values = tf.linspace(vmin, vmax, num_atoms, axis=-1)
        self._distributional_layer = snt.Linear(
            tf.size(self._values), w_init=w_init, b_init=b_init
        )

    def __call__(self, inputs: tf.Tensor) -> tfd.Distribution:
        logits = self._distributional_layer(inputs)
        logits = tf.reshape(
            logits,
            tf.concat(
                [tf.shape(logits)[:1], tf.shape(self._values)], axis=0  # batch size
            ),
        )
        values = tf.cast(self._values, logits.dtype)

        return DiscreteValuedDistribution(values=values, logits=logits)


@tfp.experimental.register_composite
class DiscreteValuedDistribution(acme_DiscreteValuedDistribution):
    """This is a generalization of a categorical distribution.

    The support for the DiscreteValued distribution can be any real valued range,
    whereas the categorical distribution has support [0, n_categories - 1] or
    [1, n_categories]. This generalization allows us to take the mean of the
    distribution over its support.
    """

    def __init__(
        self,
        values: tf.Tensor,
        logits: tf.Tensor = None,
        probs: tf.Tensor = None,
        name: str = "DiscreteValuedDistribution",
    ):
        """Initialization.

        Args:
          values: Values making up support of the distribution. Should have a shape
            compatible with logits.
          logits: An N-D Tensor, N >= 1, representing the log probabilities of a set
            of Categorical distributions. The first N - 1 dimensions index into a
            batch of independent distributions and the last dimension indexes into
            the classes.
          probs: An N-D Tensor, N >= 1, representing the probabilities of a set of
            Categorical distributions. The first N - 1 dimensions index into a batch
            of independent distributions and the last dimension represents a vector
            of probabilities for each class. Only one of logits or probs should be
            passed in.
          name: Name of the distribution object.
        """
        self.true_dimensions = None
        super().__init__(values=values, logits=logits, probs=probs, name=name)

    def set_dimensions(self, dims: Tuple) -> tfd.Categorical:
        assert len(dims) == 2
        self._true_dimensions = [dims[0], dims[1]]

    def cut_dimension(
        self, axis: int, start: int = 0, end: Union[int, None] = None
    ) -> None:
        if end is None:
            end = self._true_dimensions[axis]
        if axis == 1:
            reshape_dims = self._true_dimensions + [self._values.shape[0]]
            self._logits: tf.Tensor = tf.reshape(
                tf.reshape(self._logits, reshape_dims)[:, start:end],
                [-1, self._values.shape[0]],
            )
            if end is not None:
                self._true_dimensions[axis] = end - start
        else:
            raise NotImplementedError
