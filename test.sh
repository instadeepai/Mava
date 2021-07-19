#!/bin/bash
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


# Bash settings: fail on any error and display all commands being run.
set -e
set -x

# Python must be 3.6 or higher.
python --version

# Set up a virtual environment.
python -m venv mava_testing
source mava_testing/bin/activate

# Install dependencies.
pip install --upgrade pip setuptools
pip --version
pip install .
pip install .[tf]
pip install .[reverb]
pip install .[envs]
pip install .[launchpad]
pip install .[testing]

N_CPU=$(grep -c ^processor /proc/cpuinfo)

# Run static type-checking.
mypy mava

# Run all tests.
pytest -n "${N_CPU}" tests

# Clean-up.
deactivate
rm -rf mava_testing/
