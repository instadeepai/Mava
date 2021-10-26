from datetime import datetime
import numpy as np
import tensorflow as tf
import sonnet as snt

import reverb
from acme.tf import utils as tf2_utils
from acme import datasets
from acme.adders import reverb as adders

from mava import specs as mava_specs
from mava.adders import reverb as reverb_adders
from mava.components.tf.modules.exploration import LinearExplorationScheduler, ExponentialExplorationScheduler
from mava.utils.loggers import logger_utils

from training import FeedForwardTrainer, RecurrentTrainer
from execution import RecurrentExecutor, FeedForwardExecutor
from utils import create_rail_env
from episode_runner import EpisodeRunner
from mixers import VDN

ENV_PARAMS = {
        # Test_0
        "n_agents": 2,
        "x_dim": 30,
        "y_dim": 30,
        "n_cities": 2,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "malfunction_rate": 1 / 200,
        "seed": 0,
        "observation_tree_depth": 2,
        "observation_max_path_depth": 30
}

CONFIG = {
    "sequence_length": 10,
    "period": 5,
    "max_replay_size": 5_000,
    "min_replay_size": 100,
    "batch_size": 32,
    "prefetch_size": 4,
    "mixer": VDN(),
    "learn_every": 8,
    "evaluate_every": 1000,
    "discount": 0.99,
    "feedforward": False,
    "learning_rate": 1e-4,
    "epsilon_decay": 1e-5,
    "epsilon_min": 0.05,
    "tau": 5e-4,
    "checkpoint_dir": "./logs/checkpoint" + str(datetime.now())
}

if CONFIG["feedforward"]:
    qnetwork = snt.nets.MLP(
        (128, 128, 5)
    )
else:
    qnetwork = snt.DeepRNN(
        [
            snt.GRU(64),
            snt.Linear(5)
        ]
    )

optimizer = snt.optimizers.Adam(learning_rate=CONFIG["learning_rate"])

rail_env = create_rail_env(**ENV_PARAMS)

#### REVERB ####
environment_spec = mava_specs.MAEnvironmentSpec(
    rail_env  # type:ignore
)

extra_spec = {}

if CONFIG["feedforward"]:
    adder_sig = reverb_adders.ParallelNStepTransitionAdder.signature(
        environment_spec, extra_spec
    )
else:
    core_state_specs = {}
    for agent in rail_env.agents:
        core_state_specs[agent] = tf2_utils.squeeze_batch_dim(
                qnetwork.initial_state(1)
        )
    extra_spec["core_states"] = core_state_specs

    adder_sig = reverb_adders.ParallelSequenceAdder.signature(
        environment_spec, CONFIG["sequence_length"], extra_spec
    )

limiter = reverb.rate_limiters.MinSize(1)

sampler = reverb.selectors.Uniform()

replay_table = reverb.Table(
    name=adders.DEFAULT_PRIORITY_TABLE,
    sampler=sampler,
    remover=reverb.selectors.Fifo(),
    max_size=CONFIG["max_replay_size"],
    rate_limiter=limiter,
    signature=adder_sig,
)

server = reverb.Server(tables=[
    replay_table
    ],
)

client = reverb.Client(f'localhost:{server.port}')

if CONFIG["feedforward"]:
    adder = reverb_adders.ParallelNStepTransitionAdder(
        client=client,
        n_step=1,
        discount=CONFIG["discount"]
    )
else:
    adder = reverb_adders.ParallelSequenceAdder(
        priority_fns=None,
        client=client,
        sequence_length=CONFIG["sequence_length"],
        period=CONFIG["period"],
    )

dataset = datasets.make_reverb_dataset(
    table=adders.DEFAULT_PRIORITY_TABLE,
    server_address=f'localhost:{server.port}',
    batch_size=CONFIG["batch_size"],
    prefetch_size=CONFIG["prefetch_size"],
    sequence_length=None if CONFIG["feedforward"] else CONFIG["sequence_length"]
)

################

train_logger = logger_utils.make_logger(
    label="trainer",
    directory="./logs",
    to_terminal=True,
    to_tensorboard=True,
    time_delta=10,
)

if CONFIG["feedforward"]:
    trainer = FeedForwardTrainer(
       agents=rail_env.agents,
        qnetwork=qnetwork,
        dataset=dataset,
        optimizer=optimizer,
        tau=CONFIG["tau"],
        discount=0.99,
        logger=train_logger,
        checkpoint_dir=CONFIG["checkpoint_dir"]
    )
else:
    trainer = RecurrentTrainer(
        agents=rail_env.agents,
        qnetwork=qnetwork,
        dataset=dataset,
        optimizer=optimizer,
        tau=CONFIG["tau"],
        discount=0.99,
        mixer=CONFIG["mixer"],
        logger=train_logger,
        checkpoint_dir=CONFIG["checkpoint_dir"]
    )

epsilon_scheduler = ExponentialExplorationScheduler(
    epsilon_start=1.0, 
    epsilon_min=CONFIG["epsilon_min"], 
    epsilon_decay=CONFIG["epsilon_decay"]
)


if CONFIG["feedforward"]:
    executor = FeedForwardExecutor(
        qnetwork=qnetwork,
        epsilon_scheduler=epsilon_scheduler,
        adder=adder
    )
else:
    executor = RecurrentExecutor(
        qnetwork=qnetwork,
        epsilon_scheduler=epsilon_scheduler,
        adder=adder
    )

logger = logger_utils.make_logger(
    label="envloop",
    directory="./logs",
    to_terminal=True,
    to_tensorboard=True,
    time_delta=10,
)

#### RUN ####

runner = EpisodeRunner(
    rail_env, 
    executor, 
    trainer, 
    min_replay_size=CONFIG["min_replay_size"], 
    logger=logger, 
    learn_every=CONFIG["learn_every"],
    evaluate_every=CONFIG["evaluate_every"]
)

runner.run(max_steps=10_000_000)