import pytest


@pytest.fixture
def fast_config():
    return {
        "system": {
            # common
            "num_updates": 3,
            "rollout_length": 1,
            "update_batch_size": 1,
            # ppo:
            "ppo_epochs": 1,
            # sac:
            "explore_steps": 2,
            "epochs": 1,  # also for iql
            "policy_update_delay": 1,
            "buffer_size": 64,  # also for iql
            "batch_size": 2,
            # iql
            "min_buffer_size": 8,
            "sample_batch_size": 4,
            "sample_sequence_length": 2,
        },
        "arch": {
            "num_envs": 2,
            "num_eval_episodes": 1,
            "num_evaluation": 2,
        },
    }
