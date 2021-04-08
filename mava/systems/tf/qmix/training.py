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

# TODO (StJohn): implement Qmix trainer
# Helper resources
#   - single agent dqn learner in acme:
#           https://github.com/deepmind/acme/blob/master/acme/agents/tf/dqn/learning.py
#   - multi-agent ddpg trainer in mava: mava/systems/tf/maddpg/trainer.py

"""Qmix trainer implementation."""

# Imports

class BaseQmixTrainer(mava.Trainer):
    """Qmix trainer.
    This is the trainer component of a MADDPG system. i.e. it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__():
        """Initializes the learner.
        Args:
        """
    
    
class DecentralisedQmixTrainer(BaseQmixTrainer):
    def __init__():
        """"""
        super().__init__()


class CentralisedQmixTrainer(BaseQmixTrainer):
    def __init__():
        """"""
        super().__init__()
      

class StateBasedQmixTrainer(BaseQmixTrainer):
    def __init__():
        """"""
        super().__init__()
      