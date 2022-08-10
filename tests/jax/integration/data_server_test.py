import functools
from datetime import datetime
from typing import Any

import jax.numpy as jnp
import launchpad as lp
import optax
import pytest
import reverb
from tensorflow.python.data.ops import dataset_ops

from mava.systems.jax import System, mappo
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils


@pytest.fixture
def test_system() -> System:
    """Mappo system"""
    return mappo.MAPPOSystem()


def get_dataset(data_server: reverb.client.Client) -> dataset_ops.DatasetV1Adapter:
    """Batches 2 sequences to get samples"""
    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=data_server._server_address,
        table="trainer",
        max_in_flight_samples_per_worker=10,
    )
    dataset = dataset.batch(2)
    return dataset


def test_data_server(test_system: System) -> None:
    """Test if the data server instantiates processes as expected."""

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return mappo.make_default_networks(  # type: ignore
            policy_layer_sizes=(32, 32),
            critic_layer_sizes=(64, 64),
            *args,
            **kwargs,
        )

    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = "~/mava"
    mava_id = str(datetime.now())
    checkpoint_subpath = f"{base_dir}/{mava_id}"
    # Log every [log_every] seconds.
    log_every = 1
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=mava_id,
        time_delta=log_every,
    )
    # Optimizer.
    optimizer = optax.chain(
        optax.clip_by_global_norm(40.0),
        optax.adam(1e-4),
    )

    # Build the test_system
    test_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=10,
        multi_process=False,
        run_evaluator=True,
        num_executors=1,
        use_next_extras=False,
        max_queue_size=5000,
        sample_batch_size=2,
        max_in_flight_samples_per_worker=4,
        num_workers_per_iterator=-1,
        rate_limiter_timeout_ms=-1,
        checkpoint=True,
        nodes_on_gpu=[],
        lp_launch_type=lp.LaunchType.LOCAL_MULTI_PROCESSING,
        sequence_length=4,
        period=4,
    )

    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system._builder.store.system_build

    table_trainer = data_server.server_info()["trainer"]
    assert table_trainer.name == "trainer"
    assert table_trainer.max_size == 5000  # max_queue_size
    assert table_trainer.max_times_sampled == 1

    rate_limiter = table_trainer.rate_limiter_info
    assert rate_limiter.samples_per_insert == 1
    assert rate_limiter.max_diff == 5000
    assert rate_limiter.min_size_to_sample == 1

    signature = table_trainer.signature
    assert sorted(list(signature.observations.keys())) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    for observation in signature.observations.values():
        assert observation.observation
        assert observation.legal_actions
        assert observation.terminal
    assert sorted(list(signature.actions.keys())) == ["agent_0", "agent_1", "agent_2"]
    assert sorted(list(signature.rewards.keys())) == ["agent_0", "agent_1", "agent_2"]
    assert sorted(list(signature.discounts.keys())) == ["agent_0", "agent_1", "agent_2"]
    assert sorted(list(signature.extras.keys())) == ["network_keys", "policy_info"]
    assert sorted(list(signature.extras["network_keys"].keys())) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    assert sorted(list(signature.extras["policy_info"].keys())) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]

    assert table_trainer.num_episodes == 0
    assert table_trainer.num_unique_samples == 0

    # Run episodes
    for _ in range(0, 5):
        executor.run_episode()

    table_trainer = data_server.server_info()["trainer"]
    assert table_trainer.num_episodes == 5  # 5 episodes
    assert table_trainer.num_unique_samples != 0

    # dataset added by the exevutor via the adders
    dataset = get_dataset(data_server)
    for sample in dataset.take(1):
        assert sorted(list(sample.data.observations.keys())) == [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
        for observation in sample.data.observations.values():
            assert jnp.size(observation.observation) != 0
            assert jnp.size(observation.legal_actions) != 0
            assert jnp.size(observation.terminal) != 0
        assert sorted(list(sample.data.actions.keys())) == [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
        for agent_action in sample.data.actions.values():
            assert jnp.size(agent_action) != 0
        assert sorted(list(sample.data.rewards.keys())) == [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
        for agent_reward in sample.data.rewards.values():
            assert jnp.size(agent_reward) != 0
        assert sorted(list(sample.data.discounts.keys())) == [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
        for agent_discount in sample.data.discounts.values():
            assert jnp.size(agent_discount) != 0
        assert sorted(list(sample.data.extras.keys())) == [
            "network_keys",
            "policy_info",
        ]
        assert sorted(list(sample.data.extras["network_keys"].keys())) == [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
        assert sorted(list(sample.data.extras["policy_info"].keys())) == [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
