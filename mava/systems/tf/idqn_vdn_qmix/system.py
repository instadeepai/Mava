from datetime import datetime

import launchpad as lp
import reverb
import sonnet as snt
from acme.tf import utils as tf2_utils
from acme.adders import reverb as adders
from acme import datasets
from acme.tf import variable_utils

from mava.adders import reverb as reverb_adders
from mava.components.tf.modules.exploration import ExponentialExplorationScheduler
from mava.utils.loggers import logger_utils
from mava import specs as mava_specs
from mava.utils import lp_utils


from mava.systems.tf.idqn_vdn_qmix.execution import RecurrentExecutor
from mava.systems.tf.idqn_vdn_qmix.training import RecurrentTrainer
from mava.systems.tf.idqn_vdn_qmix.executor_loop import ExecutorLoop
from mava.systems.tf.idqn_vdn_qmix.utils import create_rail_env
from mava.systems.tf.idqn_vdn_qmix.mixers import VDN

class System:

    def __init__(self, CONFIG, ENV_PARAMS):
        self.feedforward = CONFIG["feedforward"]
        self.discount = CONFIG["discount"]
        self.sequence_length = CONFIG["sequence_length"]
        self.period = CONFIG["period"]
        self.epsilon_min = CONFIG["epsilon_min"]
        self.epsilon_decay = CONFIG["epsilon_decay"]
        self.hidden_size = CONFIG["hidden_size"]
        self.max_replay_size = CONFIG["max_replay_size"]
        self.batch_size = CONFIG["batch_size"]
        self.prefetch_size = CONFIG["prefetch_size"]
        self.mixer = CONFIG["mixer"]
        self.tau = CONFIG["tau"]
        self.learning_rate = CONFIG["learning_rate"]
        self.executor_variable_update_period = CONFIG["executor_variable_update_period"]
        self.num_executors = CONFIG["num_executors"]

        self.ENV_PARAMS = ENV_PARAMS
        self.time_stamp = str(datetime.now())

        self.checkpoint_dir = CONFIG["checkpoint_dir"]
        if not self.checkpoint_dir:
            self.checkpoint_dir = f"./logs/{self.time_stamp}"

    def _init_network_variables(self, qnetwork, env):
        # Environment spec
        env_spec = mava_specs.MAEnvironmentSpec(env)

        # Obs spec
        obs_spec = list(env_spec.get_agent_specs().values())[0].observations.observation

        # Init network variables
        tf2_utils.create_variables(qnetwork, [obs_spec])

        return qnetwork

        

    def replay(self):
        # Environment spec
        env = create_rail_env(**self.ENV_PARAMS)
        env_spec = mava_specs.MAEnvironmentSpec(env)

        # Q-network
        qnetwork = snt.DeepRNN(
            [
                snt.GRU(self.hidden_size),
                snt.Linear(5)
            ]
        )

        # Adder signiture
        extra_spec, core_state_specs = {}, {}
        for agent in env.agents:
            core_state_specs[agent] = tf2_utils.squeeze_batch_dim(
                qnetwork.initial_state(1)
            )
        extra_spec["core_states"] = core_state_specs
        adder_sig = reverb_adders.ParallelSequenceAdder.signature(
            env_spec, self.sequence_length, extra_spec
        )

        # Rate limiter
        limiter = reverb.rate_limiters.MinSize(1)

        # Sampler
        sampler = reverb.selectors.Uniform()

        # Replay table
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=sampler,
            remover=reverb.selectors.Fifo(),
            max_size=self.max_replay_size,
            rate_limiter=limiter,
            signature=adder_sig,
        )

        return [replay_table]


    def executor(self, client, variable_source, id):
        # Environment
        env = create_rail_env(**self.ENV_PARAMS)

        # Adder
        adder = reverb_adders.ParallelSequenceAdder(
            priority_fns=None,
            client=client,
            sequence_length=self.sequence_length,
            period=self.period,
        )

        # Exploration
        epsilon_scheduler = ExponentialExplorationScheduler(
            epsilon_start=1.0, 
            epsilon_min=self.epsilon_min, 
            epsilon_decay=self.epsilon_decay
        )

        # Q-network
        qnetwork = snt.DeepRNN(
            [
                snt.GRU(self.hidden_size),
                snt.Linear(5)
            ]
        )

        # Initialize network variables
        qnetwork = self._init_network_variables(qnetwork, env)

        # Get variables
        variables = {"qnetwork": qnetwork.variables}

        # Variable client
        variable_client = variable_utils.VariableClient(
            client=variable_source,
            variables=variables,
            update_period=self.executor_variable_update_period,
        )

        # Make sure not to use a random policy after checkpoint restoration by
        # assigning variables before running the environment loop.
        variable_client.update_and_wait()

        # Executor
        executor = RecurrentExecutor(
            qnetwork=qnetwork,
            epsilon_scheduler=epsilon_scheduler,
            variable_client=variable_client,
            adder=adder,
        )

        # Logger
        logger = logger_utils.make_logger(
            label=f"executor_{id}",
            time_stamp=self.time_stamp,
            directory="./logs",
            to_terminal=True,
            to_tensorboard=True,
            time_delta=10,
        )

        # Executor-Environment Loop
        executor_loop = ExecutorLoop(
            executor,
            env,
            logger
        )

        return executor_loop

    def trainer(self, client):
        # Dataset
        dataset = datasets.make_reverb_dataset(
            table=adders.DEFAULT_PRIORITY_TABLE,
            server_address=client.server_address,
            batch_size=self.batch_size,
            prefetch_size=self.prefetch_size,
            sequence_length=self.sequence_length
        )

        # Environment
        env = create_rail_env(**self.ENV_PARAMS)

        # Q-network
        qnetwork = snt.DeepRNN(
            [
                snt.GRU(self.hidden_size),
                snt.Linear(5)
            ]
        )

        # Initialize network variables
        qnetwork = self._init_network_variables(qnetwork, env)

        # Optimizer
        optimizer = snt.optimizers.Adam(self.learning_rate)

        # Logger
        logger = logger_utils.make_logger(
                label="trainer",
                time_stamp=self.time_stamp,
                directory="./logs",
                to_terminal=True,
                to_tensorboard=True,
                time_delta=10,
            )

        trainer = RecurrentTrainer(
            agents=env.agents,
            qnetwork=qnetwork,
            dataset=dataset,
            optimizer=optimizer,
            tau=self.tau,
            discount=self.discount,
            mixer=self.mixer,
            logger=logger,
            checkpoint_dir=self.checkpoint_dir
        )

        return trainer

    def build(self, name = "system"):

        program = lp.Program(name=name)

        with program.group("replay"):
            replay = program.add_node(lp.ReverbNode(self.replay))

        with program.group("trainer"):
            trainer = program.add_node(lp.CourierNode(self.trainer, replay))

        with program.group("executor"):
            # Add executors which pull round-robin from our variable sources.
            for id in range(self.num_executors):
                program.add_node(
                    lp.CourierNode(
                        self.executor,
                        replay,
                        trainer,
                        id=id
                    )
                )

        return program

def main():
    
    ENV_PARAMS = {
        # Test_0
        "n_agents": 5,
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
        "batch_size": 32,
        "prefetch_size": 4,
        "mixer": VDN(),
        "discount": 0.99,
        "learning_rate": 1e-4,
        "epsilon_decay": 1e-5,
        "epsilon_min": 0.05,
        "tau": 5e-4,
        "checkpoint_dir": None,
        "executor_variable_update_period": 1000,
        "num_executors": 2,
        "hidden_size": 64
    }

    program = System(
        CONFIG,
        ENV_PARAMS
    ).build()

    # Ensure only trainer runs on gpu, while other processes run on cpu.
    local_resources = lp_utils.to_device(
        program_nodes=program.groups.keys(), nodes_on_gpu=["trainer"]
    )

    # Launch.
    lp.launch(
        program,
        lp.LaunchType.LOCAL_MULTI_PROCESSING,
        terminal="current_terminal",
        local_resources=local_resources,
    )

if __name__ == "__main__":
    main()