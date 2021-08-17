import dm_env
import numpy as np

from mava.adders.reverb import base
from mava.utils.wrapper_utils import parameterized_restart, parameterized_termination
from tests.adders.adders_utils import make_sequence, make_trajectory

agents = {"agent_0", "agent_1", "agent_2"}
reward_step1 = {"agent_0": 0.0, "agent_1": 0.0, "agent_2": 1.0}
reward_step2 = {"agent_0": 1.0, "agent_1": 0.0, "agent_2": 0.0}
reward_step3 = {"agent_0": 0.0, "agent_1": 1.0, "agent_2": 0.0}
reward_step4 = {"agent_0": 1.0, "agent_1": 1.0, "agent_2": 1.0}
reward_step5 = {"agent_0": -1.0, "agent_1": -1.0, "agent_2": -1.0}
reward_step6 = {"agent_0": 0.5, "agent_1": -5.0, "agent_2": 1.0}
reward_step7 = {"agent_0": 1.0, "agent_1": 3.0, "agent_2": 1.0}

obs_first = {agent: np.array([0.0, 1.0]) for agent in agents}
obs_step1 = {agent: np.array([1.0, 2.0]) for agent in agents}
obs_step2 = {agent: np.array([2.0, 3.0]) for agent in agents}
obs_step3 = {agent: np.array([3.0, 4.0]) for agent in agents}
obs_step4 = {agent: np.array([4.0, 5.0]) for agent in agents}
obs_step5 = {agent: np.array([5.0, 6.0]) for agent in agents}
obs_step6 = {agent: np.array([6.0, 7.0]) for agent in agents}
obs_step7 = {agent: np.array([7.0, 8.0]) for agent in agents}

default_discount = {agent: 1.0 for agent in agents}
default_action = {agent: 0.0 for agent in agents}
env_restart = parameterized_restart(
    reward={agent: 0.0 for agent in agents},
    discount=default_discount,
    observation=obs_first,
)

final_step_discount = {agent: 0.0 for agent in agents}

# Long Episode
max_sequence_length = 50
observation_dims = 5
observations = np.random.random_sample((max_sequence_length, observation_dims))
first_longepisode, steps_longepisode = make_trajectory(observations, agents)

TEST_CASES = [
    dict(
        testcase_name="ShortEps",
        max_sequence_length=2,  # nsteps +1
        first=env_restart,
        steps=(
            (
                default_action,
                parameterized_termination(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode,  next_extras)
            [
                base.Trajectory(
                    obs_first,
                    default_action,
                    reward_step1,
                    final_step_discount,
                    True,
                    {},
                ),
                base.Trajectory(
                    obs_step1,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    {agent: 0.0 for agent in agents},
                    False,
                    {},
                ),
            ],
        ),
        agents=agents,
    ),
    dict(
        testcase_name="ShortEpsWithExtras",
        max_sequence_length=2,  # nsteps +1
        first=(env_restart, {"state": -1}),
        steps=(
            (
                default_action,
                parameterized_termination(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=final_step_discount,
                ),
                {"state": 0},
            ),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode,  next_extras)
            [
                base.Trajectory(
                    obs_first,
                    default_action,
                    reward_step1,
                    final_step_discount,
                    True,
                    {"state": -1},
                ),
                base.Trajectory(
                    obs_step1,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    {agent: 0.0 for agent in agents},
                    False,
                    {"state": 0},
                ),
            ],
        ),
        agents=agents,
    ),
    dict(
        testcase_name="MediumEps",
        max_sequence_length=5,  # nsteps +1
        first=env_restart,
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step4,
                    observation=obs_step4,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode,  next_extras)
            [
                base.Trajectory(
                    obs_first, default_action, reward_step1, default_discount, True, {}
                ),
                base.Trajectory(
                    obs_step1, default_action, reward_step2, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step3,
                    default_action,
                    reward_step4,
                    final_step_discount,
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_step4,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    final_step_discount,
                    False,
                    {},
                ),
            ],
        ),
        agents=agents,
    ),
    dict(
        testcase_name="LargeEps",
        max_sequence_length=50,  # nsteps +1
        first=first_longepisode,
        steps=steps_longepisode,
        expected_sequences=[make_sequence(observations)],
        agents=agents,
    ),
]
