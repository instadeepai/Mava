# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os


def configure_computation_environment() -> None:
    """Configure the computation environment for JAX.

    Note: Must be called before any JAX computation is run.
    """
    os.environ["JAX_USE_PJRT_C_API_ON_TPU"] = ""

    # Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
    # Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
    os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
    os.environ["TF_CUDNN DETERMINISTIC"] = "1"
