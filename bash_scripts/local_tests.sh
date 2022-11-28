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

if grep -sq 'docker\|lxc' /proc/1/cgroup; then
   echo "Running inside of docker."
   apt_cmd="apt-get"
else
    echo "Running locally."
    apt_cmd="sudo apt-get"
fi

# Update
$apt_cmd update

# Python must be 3.6 or higher.
python --version

# Install dependencies.
pip install --upgrade pip setuptools
pip --version

# Set up a virtual environment.
pip install virtualenv
virtualenv mava_testing
source mava_testing/bin/activate

# Fix module 'enum' has no attribute 'IntFlag' for py3.6
pip uninstall -y enum34

# For smac
$apt_cmd -y install git

# For box2d
$apt_cmd install swig -y

# Install depedencies
pip install .[jax,envs,reverb,testing_formatting,record_episode]

# For atari envs
$apt_cmd -y install unrar-free
pip install autorom
AutoROM -v

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

# Clean-up.
deactivate
rm -rf mava_testing/
