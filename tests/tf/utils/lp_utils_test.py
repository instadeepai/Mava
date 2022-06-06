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

from typing import Dict, List

from absl.testing import parameterized
from launchpad.nodes.python.local_multi_processing import PythonProcess

from mava.utils import lp_utils

test_data = [
    dict(
        testcase_name="cpu_only",
        program_nodes=["replay", "counter", "trainer", "evaluator", "executor"],
        nodes_on_gpu=[],
        expected_resourse_list={
            "replay": PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
            "counter": PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
            "trainer": PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
            "evaluator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
            "executor": PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
        },
    ),
    dict(
        testcase_name="trainer_only_on_gpu",
        program_nodes=["replay", "counter", "trainer", "evaluator", "executor"],
        nodes_on_gpu=["trainer"],
        expected_resourse_list={
            "replay": PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
            "counter": PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
            "trainer": [],
            "evaluator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
            "executor": PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
        },
    ),
    dict(
        testcase_name="gpu_only",
        program_nodes=["replay", "counter", "trainer", "evaluator", "executor"],
        nodes_on_gpu=["replay", "counter", "trainer", "evaluator", "executor"],
        expected_resourse_list={
            "replay": [],
            "counter": [],
            "trainer": [],
            "evaluator": [],
            "executor": [],
        },
    ),
]


class TestLPResourceUtils(parameterized.TestCase):
    @parameterized.named_parameters(*test_data)
    def test_resource_specification(
        self, program_nodes: List, nodes_on_gpu: List, expected_resourse_list: Dict
    ) -> None:
        """Test resource allocation works for lp.

        Args:
            program_nodes (List): lp program nodes.
            nodes_on_gpu (List): which nodes to have on gpu.
            expected_resourse_list (List): expected resource list.
        """
        resource_list = lp_utils.to_device(
            program_nodes=program_nodes, nodes_on_gpu=nodes_on_gpu
        )
        assert resource_list == expected_resourse_list
