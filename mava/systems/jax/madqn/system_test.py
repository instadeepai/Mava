import functools
from typing import Tuple

import acme
import optax
import pytest
import reverb
from tqdm import tqdm  # type: ignore

from mava.systems.jax import madqn
from mava.systems.jax.madqn import MADQNSystem
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils
from mava.utils.schedules.linear_epsilon_scheduler import LinearEpsilonScheduler


#########################################################################
# Full system integration test.
# TODO: The tests are around executor and not the system.
#########################################################################
@pytest.fixture
def test_full_system() -> Tuple:
    """Creates a full system."""
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    network_factory = madqn.make_default_networks

    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = "~/mava"
    mava_id = "12345"
    checkpoint_subpath = f"{base_dir}/{mava_id}"

    # Log every [log_every] seconds.
    log_every = 10
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
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # epsilon scheduler.
    epsilon_scheduler = LinearEpsilonScheduler(1.0, 0.0, 100)
    # Build the system
    system = MADQNSystem()
    system.build(
        epsilon_scheduler=epsilon_scheduler,
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=20,
        multi_process=False,
        run_evaluator=True,
        num_executors=1,
        # use_next_extras=False,
        sample_batch_size=2,
    )

    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = system._builder.store.system_build

    return data_server, parameter_server, executor, evaluator, trainer


@pytest.fixture
def test_full_system_not_evaluator() -> Tuple:
    """Creates a full system."""
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    network_factory = madqn.make_default_networks

    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = "~/mava"
    mava_id = "12345"
    checkpoint_subpath = f"{base_dir}/{mava_id}"

    # Log every [log_every] seconds.
    log_every = 10
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
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # epsilon scheduler.
    epsilon_scheduler = LinearEpsilonScheduler(1.0, 0.0, 100)
    # Build the system
    system = MADQNSystem()
    system.build(
        epsilon_scheduler=epsilon_scheduler,
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=20,
        multi_process=False,
        run_evaluator=False,
        num_executors=1,
        # use_next_extras=False,
        sample_batch_size=2,
    )

    (
        data_server,
        parameter_server,
        executor,
        trainer,
    ) = system._builder.store.system_build

    return data_server, parameter_server, executor, trainer


@pytest.fixture
def test_full_system_trainer() -> Tuple:
    """Creates a full system."""
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    network_factory = madqn.make_default_networks

    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = "~/mava"
    mava_id = "12345"
    checkpoint_subpath = f"{base_dir}/{mava_id}"

    # Log every [log_every] seconds.
    log_every = 10
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
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # epsilon scheduler.
    epsilon_scheduler = LinearEpsilonScheduler(1.0, 0.0, 100)
    # Build the system
    system = MADQNSystem()
    system.build(
        epsilon_scheduler=epsilon_scheduler,
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=20,
        multi_process=False,
        run_evaluator=False,
        num_executors=1,
        # use_next_extras=False,
        sample_batch_size=2,
        target_update_period=10,
    )

    (
        data_server,
        parameter_server,
        executor,
        trainer,
    ) = system._builder.store.system_build

    return data_server, parameter_server, executor, trainer


def test_except_trainer(test_full_system: Tuple) -> None:
    """Test if the parameter server instantiates processes as expected."""
    data_server, parameter_server, executor, evaluator, trainer = test_full_system
    assert isinstance(executor, acme.core.Worker)


def test_executor_counter_init(test_full_system: Tuple) -> None:
    """Test if the counter in the executor is initialized correctly."""
    _, _, executor, _, _ = test_full_system
    assert executor._executor.store.steps_count == 0


def test_executor_counter_update(test_full_system: Tuple) -> None:
    """Test if the counter in the executor is updated correctly."""
    _, _, executor, _, _ = test_full_system
    nr_steps_per_episode = 50
    executor.run_episode()
    assert executor._executor.store.steps_count == nr_steps_per_episode
    executor.run_episode()
    assert executor._executor.store.steps_count == 2 * nr_steps_per_episode


def test_epsilon_scheduler_start(test_full_system: Tuple) -> None:
    """Test if the epsilon scheduler starts correctly."""
    _, _, executor, _, _ = test_full_system
    assert executor._executor.store.epsilon_scheduler.epsilon == 1.0


def test_epsilon_scheduler_update(test_full_system: Tuple) -> None:
    """Test if the epsilon scheduler updates correctly."""
    _, _, executor, _, _ = test_full_system
    executor.run_episode()
    # after one episode which is 50 steps, the epsilon should be 50/100 of the intial
    # and final epsilon, i.e. 50./100 * (1.0 + 0.0) = 0.5, where 100 is the decay steps
    # of the scheduler.
    assert executor._executor.store.epsilon_scheduler.epsilon == 0.5

    executor.run_episode()
    # after two episodes which are 100 steps, the epsilon should be the final value
    assert executor._executor.store.epsilon_scheduler.epsilon == 0.0

    executor.run_episode()
    # after more than 100 steps, the epsilon should remain the final value
    assert executor._executor.store.epsilon_scheduler.epsilon == 0.0


def test_data_server(test_full_system_not_evaluator: Tuple) -> None:
    """Test if the data server instantiates correctly."""
    data_server, _, _, _ = test_full_system_not_evaluator
    assert isinstance(data_server, reverb.client.Client)

    assert "trainer" in data_server.server_info()
    # server_infor can be used to check the signature as well.


def test_different_data_servers(test_full_system_trainer: Tuple) -> None:
    """Testing if the data-server of store and executor/trainer are same as ds"""
    data_server, _, executor, trainer = test_full_system_trainer

    assert data_server == executor._executor.store.data_server_client
    assert data_server == trainer.store.data_server_client


def test_data_server_filling(test_full_system_not_evaluator: Tuple) -> None:
    """Test if the data server fills correctly."""
    data_server, _, executor, _ = test_full_system_not_evaluator

    episode_length = 50
    for episode_counter in range(2):
        executor.run_episode()
        total_nr_steps = data_server.server_info()["trainer"].current_size
        assert total_nr_steps == (episode_counter + 1) * episode_length


@pytest.fixture
def test_full_system_training() -> Tuple:
    """Creates a full system."""
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    network_factory = madqn.make_default_networks

    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = "~/mava"
    mava_id = "12345"
    checkpoint_subpath = f"{base_dir}/{mava_id}"

    # Log every [log_every] seconds.
    log_every = 2
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
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # epsilon scheduler.
    epsilon_scheduler = LinearEpsilonScheduler(1.0, 0.1, 10_000)
    # Build the system
    system = MADQNSystem()
    system.build(
        epsilon_scheduler=epsilon_scheduler,
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=10,
        multi_process=False,
        run_evaluator=False,
        num_executors=1,
        # use_next_extras=False,
        sample_batch_size=128,
        target_update_period=10,
    )

    (
        data_server,
        parameter_server,
        executor,
        trainer,
    ) = system._builder.store.system_build

    return data_server, parameter_server, executor, trainer


def test_trainer_learning(test_full_system_training: Tuple) -> None:
    """_summary_"""
    from jax.config import config

    config.update("jax_disable_jit", True)
    data_server, _, executor, trainer = test_full_system_training
    # run ten episodes
    for _ in tqdm(range(40)):
        executor.run_episode()
    for _ in range(100):
        for _ in tqdm(range(5)):
            executor.run_episode()
        trainer.step()
