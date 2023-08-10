# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

"""Install script for setuptools."""

import datetime
import sys
from importlib import util as import_util

from setuptools import find_namespace_packages, setup

spec = import_util.spec_from_file_location("_metadata", "mava/_metadata.py")
_metadata = import_util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(_metadata)  # type: ignore

testing_formatting_requirements = [
    "pytest==7.2.0",
    "pre-commit",
    "mypy==0.981",
    "pytest-xdist",
    "flake8==3.8.2",
    "black==22.3.0",
    "pytest-cov",
    "interrogate",
    "pydocstyle",
    "types-six",
]

long_description = """Mava is a library for building multi-agent reinforcement
learning (MARL) systems. Mava builds off of Acme and in a similar way strives
to expose simple, efficient, and readable components, as well as examples that
serve both as reference implementations of popular algorithms and as strong
baselines, while still providing enough flexibility to do novel research.
For more information see
[github repository](https://github.com/instadeepai/mava)."""

# Get the version from metadata.
version = _metadata.__version__  # type: ignore

# If we're releasing a nightly/dev version append to the version string.
if "--nightly" in sys.argv:
    sys.argv.remove("--nightly")
    version += ".dev" + datetime.datetime.now().strftime("%Y%m%d")

setup(
    name="id-mava",
    version=version,
    description="A Python library for Multi-Agent Reinforcement Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="InstaDeep Ltd",
    license="Apache License, Version 2.0",
    keywords="multi-agent reinforcement-learning python machine learning",
    packages=find_namespace_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    install_requires=[
        "jax>=0.2.26",
        "jaxlib>=0.1.74",
        "distrax",
        "optax",
        "flax",
        "numpy",
        "sacred",
        "tensorboard_logger",
        "git+https://github.com/instadeepai/jumanji.git",
    ],
    extras_require={
        "testing_formatting": testing_formatting_requirements,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
