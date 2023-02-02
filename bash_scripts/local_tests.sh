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

# Local tests running inside a docker container.
# We already have a venv in the container so no need to
# re-install jax,tf and reverb depedencies.
export DEBIAN_FRONTEND=noninteractive

# Bash settings: fail on any error and display all commands being run.
set -e
set -x

integration=$1

# Python must be 3.6 or higher.
python --version

# For smac
apt-get -y install git

# Install mava, envs and testing tools.
pip install .[envs,testing_formatting]

N_CPU=$(grep -c ^processor /proc/cpuinfo)
# Use only 75% of local CPU cores
N_CPU_INTEGRATION=`expr $N_CPU \* 3 / 4`

if [ "$integration" = "true" ]; then \
    # Run all tests
    pytest --cov --cov-report=xml -n "${N_CPU_INTEGRATION}" tests ;
else
    # Run all unit tests (non integration tests).
    pytest --cov --cov-report=xml --durations=10 -n "${N_CPU}" tests --ignore-glob="*/*system_test.py" ;
fi
