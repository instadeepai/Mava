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

# Update
apt-get update

# Python must be 3.8 or higher.
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

# For box2d
apt-get install swig -y

pip install .[testing_formatting]
# Check code follows black formatting.
black --check .
# stop the build if there are Python syntax errors or undefined names
flake8 mava  --count --select=E9,F63,F7,F82 --ignore=C901 --show-source --statistics
# exit-zero treats all errors as warnings.
flake8 mava --count --exit-zero --statistics

# Check types.
mypy .

# Check docstring code coverage.
interrogate -c pyproject.toml mava
# Clean-up.
deactivate
rm -rf mava_testing/