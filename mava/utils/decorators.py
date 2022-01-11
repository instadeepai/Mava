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


def execution(func):
    def wrapper_for_execution(self, *args, **kwargs):
        if not self.evaluation:
            func(self, *args, **kwargs)
        else:
            pass

    return wrapper_for_execution


def evaluation(func):
    def wrapper_for_evaluation(self, *args, **kwargs):
        if self.evaluation:
            func(self, *args, **kwargs)
        else:
            pass

    return wrapper_for_evaluation