import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd 
import tensorflow_probability.substrates.jax.bijectors as tfb
from chex import Array
from acme import specs
import tensorflow as tf
from typing import Union
import dataclasses

def action_mask_categorical_policies(
    distribution: tfd.Categorical, mask: Array
) -> tfd.Categorical:
    """TODO Add description"""
    masked_logits = jnp.where(
        mask.astype(bool),
        distribution.logits,
        jnp.finfo(distribution.logits.dtype).min,
    )
    return tfd.Categorical(logits=masked_logits)

@dataclasses.dataclass
class TanhToSpec:
  """Squashes real-valued inputs to match a BoundedArraySpec."""
  spec: specs.BoundedArray

  def __call__(self, inputs: Union[tf.Tensor, tfd.Distribution]) -> Union[tf.Tensor, tfd.Distribution]:
    scale = self.spec.maximum - self.spec.minimum
    offset = self.spec.minimum
    inputs = tfb.Tanh()(inputs)
    inputs = tfb.ScaleMatvecDiag(0.5 * scale)(inputs)
    output = tfb.Shift(offset + 0.5 * scale)(inputs)
    return output
