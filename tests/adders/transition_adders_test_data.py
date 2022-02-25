import dm_env
import numpy as np

from mava import types
from mava.utils.wrapper_utils import parameterized_restart, parameterized_termination
from tests.adders.adders_utils import calc_nstep_return

agents = {"agent_0", "agent_1", "agent_2"}
reward_step1 = {"agent_0": 0.0, "agent_1": 0.0, "agent_2": 1.0}
reward_step2 = {"agent_0": 1.0, "agent_1": 0.0, "agent_2": 0.0}
reward_step3 = {"agent_0": 0.0, "agent_1": 1.0, "agent_2": 0.0}

obs_first = {agent: np.array([0.0, 1.0]) for agent in agents}
obs_step1 = {agent: np.array([1.0, 2.0]) for agent in agents}
obs_step2 = {agent: np.array([2.0, 3.0]) for agent in agents}
obs_step3 = {agent: np.array([3.0, 4.0]) for agent in agents}

default_discount = {agent: 1.0 for agent in agents}
default_action = {agent: 0.0 for agent in agents}
env_restart = parameterized_restart(
    reward={agent: 0.0 for agent in agents},
    discount=default_discount,
    observation=obs_first,
)

final_step_discount = {agent: 0.0 for agent in agents}


TEST_CASES = [
    dict(
        testcase_name="OneStep",
        n_step=1,
        discount=default_discount,
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
                parameterized_termination(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_transitions=(
            # (observation, action, reward, discount, next_observation, extras,
            #  next_extras)
            types.Transition(
                obs_first, default_action, reward_step1, default_discount, obs_step1
            ),
            types.Transition(
                obs_step1, default_action, reward_step2, default_discount, obs_step2
            ),
            types.Transition(
                obs_step2,
                default_action,
                reward_step3,
                final_step_discount,
                obs_step3,
            ),
        ),
        agents=agents,
    ),
    dict(
        testcase_name="OneStepDiffActionDiscrete",
        n_step=1,
        discount=default_discount,
        first=env_restart,
        steps=(
            (
                {"agent_0": 1, "agent_1": 2, "agent_2": 3},
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
            ),
            (
                {"agent_0": 2, "agent_1": 3, "agent_2": 4},
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=default_discount,
                ),
            ),
            (
                {"agent_0": 5, "agent_1": 6, "agent_2": 7},
                parameterized_termination(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_transitions=(
            # (observation, action, reward, discount, next_observation, extras,
            #  next_extras)
            types.Transition(
                obs_first,
                {"agent_0": 1, "agent_1": 2, "agent_2": 3},
                reward_step1,
                default_discount,
                obs_step1,
            ),
            types.Transition(
                obs_step1,
                {"agent_0": 2, "agent_1": 3, "agent_2": 4},
                reward_step2,
                default_discount,
                obs_step2,
            ),
            types.Transition(
                obs_step2,
                {"agent_0": 5, "agent_1": 6, "agent_2": 7},
                reward_step3,
                final_step_discount,
                obs_step3,
            ),
        ),
        agents=agents,
    ),
    dict(
        testcase_name="OneStepDiffActionCont",
        n_step=1,
        discount=default_discount,
        first=env_restart,
        steps=(
            (
                {"agent_0": 1.5, "agent_1": 2.5, "agent_2": 3.5},
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
            ),
            (
                {"agent_0": 2.5, "agent_1": 3.5, "agent_2": 4.5},
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=default_discount,
                ),
            ),
            (
                {"agent_0": 5.5, "agent_1": 6.5, "agent_2": 7.5},
                parameterized_termination(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_transitions=(
            # (observation, action, reward, discount, next_observation, extras,
            #  next_extras)
            types.Transition(
                obs_first,
                {"agent_0": 1.5, "agent_1": 2.5, "agent_2": 3.5},
                reward_step1,
                default_discount,
                obs_step1,
            ),
            types.Transition(
                obs_step1,
                {"agent_0": 2.5, "agent_1": 3.5, "agent_2": 4.5},
                reward_step2,
                default_discount,
                obs_step2,
            ),
            types.Transition(
                obs_step2,
                {"agent_0": 5.5, "agent_1": 6.5, "agent_2": 7.5},
                reward_step3,
                final_step_discount,
                obs_step3,
            ),
        ),
        agents=agents,
    ),
    dict(
        testcase_name="OneStepDiffActionContArray",
        n_step=1,
        discount=default_discount,
        first=env_restart,
        steps=(
            (
                {
                    "agent_0": np.array([1.5, 2.5]),
                    "agent_1": np.array([2.5, 3.5]),
                    "agent_2": np.array([3.5, 4.5]),
                },
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
            ),
            (
                {
                    "agent_0": np.array([2.5, 3.5]),
                    "agent_1": np.array([3.5, 4.5]),
                    "agent_2": np.array([4.5, 5.5]),
                },
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=default_discount,
                ),
            ),
            (
                {
                    "agent_0": np.array([5.5, 6.5]),
                    "agent_1": np.array([6.5, 7.5]),
                    "agent_2": np.array([7.5, 8.5]),
                },
                parameterized_termination(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_transitions=(
            # (observation, action, reward, discount, next_observation, extras,
            # next_extras)
            types.Transition(
                obs_first,
                {
                    "agent_0": np.array([1.5, 2.5]),
                    "agent_1": np.array([2.5, 3.5]),
                    "agent_2": np.array([3.5, 4.5]),
                },
                reward_step1,
                default_discount,
                obs_step1,
            ),
            types.Transition(
                obs_step1,
                {
                    "agent_0": np.array([2.5, 3.5]),
                    "agent_1": np.array([3.5, 4.5]),
                    "agent_2": np.array([4.5, 5.5]),
                },
                reward_step2,
                default_discount,
                obs_step2,
            ),
            types.Transition(
                obs_step2,
                {
                    "agent_0": np.array([5.5, 6.5]),
                    "agent_1": np.array([6.5, 7.5]),
                    "agent_2": np.array([7.5, 8.5]),
                },
                reward_step3,
                final_step_discount,
                obs_step3,
            ),
        ),
        agents=agents,
    ),
    dict(
        testcase_name="OneStepWithExtras",
        n_step=1,
        discount=default_discount,
        first=(env_restart, {"state": -1}),
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
                {"state": 0},
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=default_discount,
                ),
                {"state": 1},
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=final_step_discount,
                ),
                {"state": 2},
            ),
        ),
        expected_transitions=(
            # (observation, action, reward, discount, next_observation, extras,
            # next_extras)
            types.Transition(
                obs_first,
                default_action,
                reward_step1,
                default_discount,
                obs_step1,
                {"state": -1},
                {"state": 0},
            ),
            types.Transition(
                obs_step1,
                default_action,
                reward_step2,
                default_discount,
                obs_step2,
                {"state": 0},
                {"state": 1},
            ),
            types.Transition(
                obs_step2,
                default_action,
                reward_step3,
                final_step_discount,
                obs_step3,
                {"state": 1},
                {"state": 2},
            ),
        ),
        agents=agents,
    ),
    dict(
        testcase_name="TwoStep",
        n_step=2,
        discount=default_discount,
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
                parameterized_termination(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_transitions=(
            # (observation, action, reward, discount, next_observation, extras,
            #  next_extras)
            types.Transition(
                obs_first, default_action, reward_step1, default_discount, obs_step1
            ),
            types.Transition(
                obs_first,
                default_action,
                calc_nstep_return(
                    r_t=reward_step1,
                    discounts=[default_discount],
                    rewards=[reward_step2],
                ),
                default_discount,
                obs_step2,
            ),
            types.Transition(
                obs_step1,
                default_action,
                calc_nstep_return(
                    r_t=reward_step2,
                    discounts=[default_discount],
                    rewards=[reward_step3],
                ),
                final_step_discount,
                obs_step3,
            ),
            types.Transition(
                obs_step2,
                default_action,
                reward_step3,
                final_step_discount,
                obs_step3,
            ),
        ),
        agents=agents,
    ),
    dict(
        testcase_name="TwoStepDiffDiscounts",
        n_step=2,
        discount=default_discount,
        first=env_restart,
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount={agent: 0.5 for agent in agents},
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount={agent: 0.25 for agent in agents},
                ),
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_transitions=(
            # (observation, action, reward, discount, next_observation, extras,
            #  next_extras)
            types.Transition(
                obs_first,
                default_action,
                reward_step1,
                {agent: 0.5 for agent in agents},
                obs_step1,
            ),
            types.Transition(
                obs_first,
                default_action,
                calc_nstep_return(
                    r_t=reward_step1,
                    discounts=[
                        {agent: 0.5 for agent in agents},
                    ],
                    rewards=[reward_step2],
                ),
                {agent: 0.125 for agent in agents},
                obs_step2,
            ),
            types.Transition(
                obs_step1,
                default_action,
                calc_nstep_return(
                    r_t=reward_step2,
                    discounts=[{agent: 0.25 for agent in agents}],
                    rewards=[reward_step3],
                ),
                final_step_discount,
                obs_step3,
            ),
            types.Transition(
                obs_step2,
                default_action,
                reward_step3,
                final_step_discount,
                obs_step3,
            ),
        ),
        agents=agents,
    ),
    dict(
        testcase_name="TwoStepWithExtras",
        n_step=2,
        discount=default_discount,
        first=(env_restart, {"state": -1}),
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
                {"state": 0},
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=default_discount,
                ),
                {"state": 1},
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=final_step_discount,
                ),
                {"state": 2},
            ),
        ),
        expected_transitions=(
            # (observation, action, reward, discount, next_observation, extras,
            #  next_extras)
            types.Transition(
                obs_first,
                default_action,
                reward_step1,
                default_discount,
                obs_step1,
                {"state": -1},
                {"state": 0},
            ),
            types.Transition(
                obs_first,
                default_action,
                calc_nstep_return(
                    r_t=reward_step1,
                    discounts=[default_discount],
                    rewards=[reward_step2],
                ),
                default_discount,
                obs_step2,
                {"state": -1},
                {"state": 1},
            ),
            types.Transition(
                obs_step1,
                default_action,
                calc_nstep_return(
                    r_t=reward_step2,
                    discounts=[default_discount],
                    rewards=[reward_step3],
                ),
                final_step_discount,
                obs_step3,
                {"state": 0},
                {"state": 2},
            ),
            types.Transition(
                obs_step2,
                default_action,
                reward_step3,
                final_step_discount,
                obs_step3,
                {"state": 1},
                {"state": 2},
            ),
        ),
        agents=agents,
    ),
    dict(
        testcase_name="ThreeStepDiffDiscounts",
        n_step=3,
        discount=default_discount,
        first=env_restart,
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount={agent: 0.5 for agent in agents},
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount={agent: 0.25 for agent in agents},
                ),
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_transitions=(
            # (observation, action, reward, discount, next_observation, extras,
            #  next_extras)
            types.Transition(
                obs_first,
                default_action,
                reward_step1,
                {agent: 0.5 for agent in agents},
                obs_step1,
            ),
            types.Transition(
                obs_first,
                default_action,
                calc_nstep_return(
                    r_t=reward_step1,
                    discounts=[
                        {agent: 0.5 for agent in agents},
                    ],
                    rewards=[reward_step2],
                ),
                {agent: 0.125 for agent in agents},
                obs_step2,
            ),
            types.Transition(
                obs_first,
                default_action,
                calc_nstep_return(
                    r_t=reward_step1,
                    discounts=[
                        {agent: 0.5 for agent in agents},
                        {agent: 0.125 for agent in agents},
                    ],
                    rewards=[reward_step2, reward_step3],
                ),
                final_step_discount,
                obs_step3,
            ),
            types.Transition(
                obs_step1,
                default_action,
                calc_nstep_return(
                    r_t=reward_step2,
                    discounts=[{agent: 0.25 for agent in agents}],
                    rewards=[reward_step3],
                ),
                final_step_discount,
                obs_step3,
            ),
            types.Transition(
                obs_step2,
                default_action,
                reward_step3,
                final_step_discount,
                obs_step3,
            ),
        ),
        agents=agents,
    ),
]
