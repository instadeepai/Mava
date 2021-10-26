import tensorflow as tf
import sonnet as snt

@snt.allow_empty_variables
class VDN(snt.Module):

    def __call__(self, x):
       return tf.reduce_sum(x, axis=-1)
