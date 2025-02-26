import jax.numpy as jnp
from typing_extensions import NamedTuple
from flax import linen as nn
from typing import List
import jax

batch_size = 4
data_1_shape = 3
data_2_shape = 7

class Data(NamedTuple):
    x: jnp.ndarray
    task_num: int 

data_1 = Data(
    x=jnp.ones((batch_size, data_1_shape)),
    task_num=1
)
data_2 = Data(
    x=jnp.ones((batch_size, data_2_shape)),
    task_num=2
)

class Model(nn.Module):
    embed_dim: int
    num_tasks: int

    def setup(self):
        # Create the submodules in one shot.
        self.embed_layers = {
            i: nn.Dense(features=self.embed_dim, name=f"embed_{i}")
            for i in range(1, self.num_tasks + 1)
        }
        self.shared_forward = nn.Dense(features=13)

    # Get training output over all the data.
    def __call__(self, data: List[Data]):
        outputs = []
        for d in data:
            task = int(d.task_num)
            x = self.embed_layers[task](d.x)
            x = self.shared_forward(x)
            outputs.append(x)
        # Make output of shape (batch_size, num_tasks, shared_output_dim)
        return jnp.stack(outputs, axis=1)
    
    # Do inference on only a single task
    def single_task_forward(self, data: Data):
        task = int(data.task_num)
        x = self.embed_layers[task](data.x)
        x = self.shared_forward(x)
        return x


model = Model(embed_dim=32, num_tasks=2)
params = model.init(jax.random.PRNGKey(0), [data_1, data_2])

shared_output = model.apply(params, [data_1, data_2])
print(shared_output.shape)

single_task_output = model.apply(params, data_1, method="single_task_forward")
print(single_task_output.shape)