import functools
from datetime import datetime
from typing import Any, List

import launchpad as lp
import optax
import pytest
from launchpad.launch.test_multi_threading import (
    address_builder as test_address_builder,
)

from mava.components.jax.building.distributor import Distributor, DistributorConfig
from mava.core_jax import SystemBuilder
from mava.systems.jax import mappo
from mava.systems.jax.launcher import Launcher, NodeType
from mava.systems.jax.system import System
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

#########################################################################
# Full system integration test.


class MockLauncher(Launcher):
    def __init__(
        self,
        multi_process: bool,
        nodes_on_gpu: List = [],
        sp_trainer_period: int = 10,
        sp_evaluator_period: int = 10,
        name: str = "System",
        terminal: str = "current_terminal",
    ) -> None:
        """Create mock launcher component in order to be able to \
            overwrite the launch method which enables the testing of \
                a full distributed Mava system."""
        super().__init__(
            multi_process,
            nodes_on_gpu,
            sp_trainer_period,
            sp_evaluator_period,
            name,
            terminal,
        )

    def launch(self) -> None:
        """Mock launch method for testing a distributed mava system."""
        if self._multi_process:
            local_resources = lp_utils.to_device(
                program_nodes=self._program.groups.keys(),
                nodes_on_gpu=[],
            )

            lp.launch(
                self._program,
                launch_type="test_mt",
                local_resources=local_resources,
            )

        else:
            episode = 1
            executor_steps = 0

            _ = self._node_dict["data_server"]
            _ = self._node_dict["parameter_server"]
            executor = self._node_dict["executor"]
            evaluator = self._node_dict["evaluator"]
            trainer = self._node_dict["trainer"]

            while True:
                executor_stats = executor.run_episode_and_log()

                if episode % self._sp_trainer_period == 0:
                    _ = trainer.step()  # logging done in trainer
                    print("Performed trainer step.")
                if episode % self._sp_evaluator_period == 0:
                    _ = evaluator.run_episode_and_log()
                    print("Performed evaluator run.")

                print(f"Episode {episode} completed.")
                episode += 1
                executor_steps += executor_stats["episode_length"]


class MockDistributor(Distributor):
    def __init__(self, config: DistributorConfig = DistributorConfig()):
        """Initialize mock distributor component."""
        super().__init__(config)

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """Create program nodes in order to run mappo smoke test

        Args:
            builder : mava builder object
        """
        builder.store.program = MockLauncher(
            multi_process=self.config.multi_process,
            nodes_on_gpu=self.config.nodes_on_gpu,
            name=self.config.distributor_name,
            terminal=self.config.terminal,
        )

        # tables node
        data_server = builder.store.program.add(
            builder.data_server,
            node_type=NodeType.reverb,
            name="data_server",
        )

        # variable server node
        parameter_server = builder.store.program.add(
            builder.parameter_server,
            node_type=lp.CourierNode,
            name="parameter_server",
        )

        # executor nodes
        for executor_id in range(self.config.num_executors):
            builder.store.program.add(
                builder.executor,
                [f"executor_{executor_id}", data_server, parameter_server],
                node_type=lp.CourierNode,
                name="executor",
            )

        if self.config.run_evaluator:
            # evaluator node
            builder.store.program.add(
                builder.executor,
                ["evaluator", data_server, parameter_server],
                node_type=lp.CourierNode,
                name="evaluator",
            )

        # trainer nodes
        for trainer_id in builder.store.trainer_networks.keys():
            builder.store.program.add(
                builder.trainer,
                [trainer_id, data_server, parameter_server],
                node_type=lp.CourierNode,
                name="trainer",
            )

        if not self.config.multi_process:
            builder.store.system_build = builder.store.program.get_nodes()


@pytest.fixture
def test_full_system() -> System:
    """Full mava system fixture for testing"""
    return mappo.MAPPOSystem()


def test_mappo(
    test_full_system: System,
) -> None:
    """Full integration test of mappo system."""

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

    test_full_system.update(MockDistributor)

    # Build the system
    test_full_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=100,
        multi_process=True,
        run_evaluator=False,
        num_executors=1,
        use_next_extras=False,
        sample_batch_size=5,
        checkpoint=True,
    )

    (trainer_node,) = test_full_system._builder.store.program._program._groups[
        "trainer"
    ]
    trainer_node.disable_run()
    test_address_builder.bind_addresses([trainer_node])

    test_full_system.launch()
    trainer_run = trainer_node.create_handle().dereference()

    for _ in range(2):
        trainer_run.step()
