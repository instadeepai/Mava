from typing import List

import sonnet as snt
import tensorflow as tf


class Conv1DNetwork(snt.Module):
    """Simple 1D convolutional network setup."""

    def __init__(
        self,
        channels: List[int] = [32, 64, 64],
        kernel: List[int] = [8, 4, 3],
        stride: List[int] = [4, 2, 2],
        name: str = None,
    ):
        super(Conv1DNetwork, self).__init__(name=name)
        assert len(channels) == len(kernel) == len(stride)
        seq_list = []
        for i in range(len(channels)):
            seq_list.append(
                snt.Conv1D(channels[i], kernel_shape=kernel[i], stride=stride[i])
            )
            seq_list.append(tf.nn.relu)
        seq_list.append(snt.Flatten())
        self._network = snt.Sequential(seq_list)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._network(inputs)
