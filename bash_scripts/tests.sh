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

# Update
apt-get update

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
apt-get -y install git

# For box2d
apt-get install swig -y

# Install depedencies
pip install .[jax,envs,reverb,testing_formatting,record_episode]

# For atari envs
apt-get -y install unrar-free
pip install autorom
AutoROM -v

N_CPU=$(grep -c ^processor /proc/cpuinfo)

if [ "$integration" = "true" ]; then \
    # Run all tests
    pytest --cov --cov-report=xml -n "${N_CPU}" tests ;
else
    # Run all unit tests (non integration tests).
    pytest --cov --cov-report=xml --durations=10 -n "${N_CPU}" tests --ignore-glob="*/*system_test.py" ;
fi

# Code coverage
curl https://keybase.io/codecovsecurity/pgp_keys.asc | gpg --no-default-keyring --keyring trustedkeys.gpg --import # One-time step
curl -Os https://uploader.codecov.io/latest/linux/codecov
curl -Os https://uploader.codecov.io/latest/linux/codecov.SHA256SUM
curl -Os https://uploader.codecov.io/latest/linux/codecov.SHA256SUM.sig
gpgv codecov.SHA256SUM.sig codecov.SHA256SUM
shasum -a 256 -c codecov.SHA256SUM
chmod +x codecov
./codecov -t ${CODECOV_TOKEN}

# Clean-up.
deactivate
rm -rf mava_testing/
