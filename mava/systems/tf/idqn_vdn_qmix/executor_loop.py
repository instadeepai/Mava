from flatland.envs.step_utils.states import TrainState

class ExecutorLoop:

    def __init__(
        self,
        executor,
        env,
        logger
    ):

        self._executor = executor
        self._env = env
        self._logger = logger

        self._t = 0

    def run(self):
        # Run infinite episodes
        while True:
            timestep, info = self._env.reset()

            # Make the first observation.
            self._executor.observe_first(timestep)

            # Book-keeping
            episode_return = 0
            episode_steps = 0

            # Run an episode.
            while not timestep.last():
                actions = self._executor.select_actions(timestep.observation)

                timestep, info = self._env.step(actions)

                # Have the agent observe the timestep and let the actor update itself.
                self._executor.observe(actions, next_timestep=timestep)

                # Book-keeping.
                episode_steps += 1
                self._t += 1

                # Update executor variables
                self._executor.update()

            # Compute episode return
            episode_return += sum([r for r in timestep.reward.values()])

            # Compute completion rate
            tasks_finished = sum([1 if info['state'][agent] == TrainState.DONE else 0 for agent in self._env.agents])
            completion = tasks_finished / self._env.get_num_agents()

            # Compute normalized score
            normalized_score = episode_return / (self._env._max_episode_steps * self._env.get_num_agents())

            # Logging
            logs = {
                "episode_length": episode_steps,
                "episode_return": episode_return,
                "score": normalized_score,
                "completion": completion,
                "epsilon": self._executor._epsilon_scheduler.get_epsilon(),
                "timesteps": self._t
            }
            self._logger.write(logs)