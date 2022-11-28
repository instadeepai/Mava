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

export DEBIAN_FRONTEND=noninteractive

# Bash settings: fail on any error and display all commands being run.
set -e
set -x

integration=$1

# Python must be 3.6 or higher.
python --version

# Set up a virtual environment.
virtualenv mava_testing
source mava_testing/bin/activate

# For smac
apt-get -y install git

# Install depedencies
pip install .[jax,envs,reverb,testing_formatting,record_episode]

N_CPU=$(grep -c ^processor /proc/cpuinfo)

if [ "$integration" = "true" ]; then \
    # Run all tests
    pytest --cov=./ --cov-report=xml -n "${N_CPU}" tests ;
else
    # Run all unit tests (non integration tests).
    pytest --cov=./ --cov-report=xml --durations=10 -n "${N_CPU}" tests --ignore-glob="*/*system_test.py" ;
fi

# Clean-up.
deactivate
rm -rf mava_testing/
