from flatland.envs.step_utils.states import TrainState

from utils import create_rail_env

class EpisodeRunner:

    def __init__(
        self,
        env,
        executor,
        trainer,
        min_replay_size,
        logger,
        learn_every = 64,
        evaluate_every = 100,
        evaluator_episodes = 10
    ):
        self.env = env
        self.executor = executor
        self.trainer = trainer
        self.logger = logger
        self.min_replay_size = min_replay_size
        self.learn_every = learn_every
        self.evaluate_every = evaluate_every
        self.evaluator_episodes = evaluator_episodes

        self.t = 0

    def ready_to_learn(self):
        return self.t >= self.min_replay_size

    def run_episode(self):
        
        timestep, info = self.env.reset()
        env_extras = {}

        # Make the first observation.
        self.executor.observe_first(timestep, extras=env_extras)

        episode_return = 0
        episode_steps = 0

        # Run an episode.
        while not timestep.last():
            actions = self.executor.select_actions(timestep.observation)

            timestep, info = self.env.step(actions)
            env_extras = {}

            # Have the agent observe the timestep and let the actor update itself.
            self.executor.observe(
                actions, next_timestep=timestep, next_extras=env_extras
            )

            if self.ready_to_learn() and self.t % self.learn_every == 0:
                self.trainer.step()

            # Book-keeping.
            episode_steps += 1
            self.t += 1

        episode_return += sum([r for r in timestep.reward.values()])

        tasks_finished = sum([1 if info['state'][agent] == TrainState.DONE else 0 for agent in self.env.agents])
        completion = tasks_finished / self.env.get_num_agents()
        normalized_score = episode_return / (self.env._max_episode_steps * self.env.get_num_agents())

        result = {
                "episode_length": episode_steps,
                "episode_return": episode_return,
                "score": normalized_score,
                "completion": completion,
                "epsilon": self.executor._epsilon_scheduler.get_epsilon()
            }

        return result

    def run_evaluator_episode(self):
        timestep, info = self.env.reset()

        # Make the first observation.
        self.executor.observe_first(timestep)

        episode_return = 0
        episode_steps = 0

        # Run an episode.
        while not timestep.last():
            actions = self.executor.select_actions(timestep.observation, evaluator=True)

            timestep, info = self.env.step(actions)

            self.env.render()

            # Have the agent observe the timestep and let the actor update itself.
            self.executor.observe(
                actions, next_timestep=timestep
            )

            # Book-keeping.
            episode_steps += 1

        episode_return += sum([r for r in timestep.reward.values()])

        tasks_finished = sum([1 if info['state'][agent] == TrainState.DONE else 0 for agent in self.env.agents])
        completion = tasks_finished / self.env.get_num_agents()
        normalized_score = episode_return / (self.env._max_episode_steps * self.env.get_num_agents())

        result = {
                "episode_length": episode_steps,
                "episode_return": episode_return,
                "score": normalized_score,
                "completion": completion,
            }

        return result

    def run(self, max_steps):
        episode_count, step_count = 0, 0
        while step_count < max_steps:
            # Evaluate
            # if episode_count % self.evaluate_every == 0:
            #     for i in range(self.evaluator_episodes):
            #         result = self.run_evaluator_episode()
            #         print(result)

            # Train
            result = self.run_episode()
            episode_count += 1
            step_count += result["episode_length"]
            result["episodes"] = episode_count
            # Log the given results.
            self.logger.write(result)

            
