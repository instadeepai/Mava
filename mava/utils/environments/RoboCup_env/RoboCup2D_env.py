import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from mava.utils.environments.RoboCup_env.robocup_utils.agent import (  # type: ignore # noqa: E501
    Agent as basic_agent,
)
from mava.utils.environments.RoboCup_env.robocup_utils.trainer import (  # type: ignore # noqa: E501
    Trainer,
)
from mava.utils.environments.RoboCup_env.robocup_utils.util_functions import (  # type: ignore # noqa: E501
    wait_for_next_observations,
)


# spawn an agent of team_name, with position
def spawn_agent(
    team_name: str, team_id: int, agent_id: int, num_players: int, port: int
) -> basic_agent:
    """
    Used to run an agent in a seperate physical process.
    """
    # return type of agent by position, construct
    a = basic_agent(
        teamname=team_name, team_id=team_id, agent_id=agent_id, num_players=num_players
    )
    a.connect("localhost", port)
    return a


def run_server_thread(game_setting: str, include_wait: bool, port: int) -> None:
    # ./usr/local/bin/rcssmonitor
    command = (
        "/usr/local/bin/rcssserver -server::coach_w_referee=on"
        " -server::half_time=99999999 -server::game_logging=false"
        " -server::game_log_dated=false"
        " -server::keepaway_log_dated=false"
        " -server::keepaway_logging=false"
        " -server::text_log_dated=false"
        " -server::port="
        + str(port)
        + " -server::coach_port="
        + str(port + 1)
        + " -server::olcoach_port="
        + str(port + 2)
    )

    if game_setting == "domain_randomisation" or game_setting == "fixed_opponent":
        command += " -server::drop_ball_time=60000"
        pass
    elif game_setting == "reward_shaping":
        pass
    else:
        raise NotImplementedError(
            "Should drop_ball_time be included for this game_setting?"
        )

    if include_wait:
        command += " -server::synch_mode=false"
    else:
        command += " -server::synch_mode=true"
    os.system(command)


def start_trainer(port: int) -> Trainer:
    trainer = Trainer()
    trainer.connect("localhost", port + 1)
    return trainer


def run_monitor_thread(port: int) -> None:
    # ./usr/local/bin/rcssmonitor
    os.system("/usr/local/bin/rcssmonitor --server-port=" + str(port))


def start_server(game_setting: str, include_wait: bool, port: int) -> None:
    # Wait for server to startup completely
    # print("mode: ", mode)
    at = mp.Process(target=run_server_thread, args=(game_setting, include_wait, port))
    at.daemon = True
    at.start()
    time.sleep(0.2)


def start_monitor(port: int) -> None:
    # Wait for monitor process to complete
    at = mp.Process(target=run_monitor_thread, args=(port,))
    at.daemon = True
    at.start()
    time.sleep(0.2)


def connect_agents(
    team_name: str,
    team_id: int,
    num_per_team: int,
    num_players: int,
    start_id: int,
    port: int,
) -> List:
    # spawn all agents as seperate processes for maximum processing efficiency
    agents = []
    for agent_id in range(start_id, start_id + num_per_team):
        agents.append(
            spawn_agent(
                team_name=team_name,
                team_id=team_id,
                agent_id=agent_id,
                num_players=num_players,
                port=port,
            )
        )
        time.sleep(0.01)

    print("Spawned %d agents." % len(agents))
    return agents


class RoboCup2D:
    def __init__(
        self,
        game_setting: str = "reward_shaping",
        include_wait: bool = False,
        team_names: List = ["Team1", "Team2"],
        players_per_team: List = [11, 11],
        render_game: bool = False,
        game_length: int = 6000,
        beta: float = 0.1,
        port: int = 6000,
    ):

        self.game_setting = game_setting
        if type(players_per_team) is not list:
            players_per_team = [players_per_team, players_per_team]

        self.players_per_team = players_per_team
        self.num_players = players_per_team[0] + players_per_team[1]

        assert (
            type(players_per_team) == list
            and 0 < players_per_team[0] <= 11
            and 0 <= players_per_team[1] <= 11
        )

        if game_setting == "domain_randomisation":
            self.game_diff = 0.0
            game_length = 100
        elif game_setting == "reward_shaping":
            game_length = 6000
        elif game_setting == "fixed_opponent":
            self.game_diff = 0.0
            game_length = 200
        else:
            raise NotImplementedError(
                "Game setting not implemented yet: ", game_setting
            )

        self.beta = 0.01

        # Start the server
        start_server(game_setting, include_wait, port)

        self.team_names = team_names

        # Connect trainer which is used to control the RoboCup server
        self.trainer = start_trainer(port)

        # Initial reset to move agent to correct position
        if game_setting == "single_agent_score":
            time.sleep(0.001)
            self.trainer.reset_game(
                players_per_team, team_names, game_setting, self.game_diff
            )

        self.game_length = game_length
        self.game_step = 0
        self.game_scores = [0, 0]

        self.end_step = 0

        self.tot_out_step = 0.0
        self.tot_in_step_no_wait = 0.0
        self.tot_server_wait = 0.0

        # Render the game on a monitor
        if render_game:
            start_monitor(port)

        self.id_dict = {"l": 0, "r": 1}

        # Connect teams
        self.agents = {}
        # self.agent_ids = []
        self.previous_scores = [0, 0]
        start_id = 0
        for t_i in range(2):
            if players_per_team[t_i] > 0:
                agents = connect_agents(
                    team_name=team_names[t_i],
                    team_id=t_i,
                    num_per_team=players_per_team[t_i],
                    num_players=self.num_players,
                    start_id=start_id,
                    port=port,
                )
                for agent in agents:
                    self.agents["player_" + str(agent.agent_id)] = agent
                start_id += players_per_team[t_i]

    def step(self, actions: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
        # Do agent update
        # start_step = time.time()
        self.game_step += 1

        for agent_key, agent in self.agents.items():
            agent.do_action(actions[agent_key])

        # Send done to show that trainer is done.
        if self.trainer:
            self.trainer.send_done()

        # Wait for the environment to step and provide the next observations
        wait_for_next_observations([self.trainer])

        # Check if done with game
        done = self.game_step > self.game_length

        # Calculate rewards
        rewards = {}
        score_add = [
            self.trainer.wm.score_l - self.previous_scores[0],
            self.trainer.wm.score_r - self.previous_scores[1],
        ]
        self.previous_scores[0] = self.trainer.wm.score_l
        self.previous_scores[1] = self.trainer.wm.score_r
        self.game_scores[0] += score_add[0]
        self.game_scores[1] += score_add[1]

        # Reset the game if a goal is scored
        if (
            self.game_setting == "reward_shaping"
            and score_add[0] != 0
            or score_add[1] != 0
        ):
            self.trainer.reset_game(
                self.players_per_team,
                self.team_names,
                self.game_setting,
                reset_stamina=False,
            )

        wm = self.trainer.wm
        for agent_key, agent in self.agents.items():
            agent.do_action(actions[agent_key])
            team_id = agent.team_id
            opponent_id = (team_id + 1) % 2

            # Calculate rewards
            player = wm.players[agent.agent_id]

            if self.game_setting == "reward_shaping":
                if player["teamname"] == "Team_A":
                    destination_coords = wm.goals[0]["coords"]
                elif player["teamname"] == "Team_B":
                    destination_coords = wm.goals[1]["coords"]
                else:
                    raise ValueError("Unknown team: ", player["teamname"])

                ball_goal_dist = wm.euclidean_distance(
                    wm.ball["coords"], destination_coords
                )
                next_x = float(wm.ball["coords"][0]) + float(wm.ball["delta_coords"][0])
                next_y = float(wm.ball["coords"][1]) + float(wm.ball["delta_coords"][1])
                ball_goal_delta_dist = wm.euclidean_distance(
                    (next_x, next_y), destination_coords
                )
                ball_towards_goal = ball_goal_dist - ball_goal_delta_dist

                next_x = float(player["coords"][0]) + float(player["delta_coords"][0])
                next_y = float(player["coords"][1]) + float(player["delta_coords"][1])

                scored = score_add[team_id] - score_add[opponent_id]

                # TODO: Change back
                # if scored > 0.1:
                #     if self.beta < 0.001:
                #         self.beta = 0.0
                #         print("Beta minimum reached.")
                #     else:
                #         self.beta *= 0.99
                #         print("Beta: ", self.beta)

                self.beta = 0.01
                reward = (
                    scored + ball_towards_goal * self.beta
                )  # + player_towards_ball * self.beta + scored
            elif self.game_setting == "domain_randomisation":
                reward = score_add[team_id] - score_add[opponent_id]

                # # TODONE: Remove this. Masking zero rewards!
                # if reward < 0:
                #     reward = 0

            else:
                raise NotImplementedError("Unknown game setting: ", self.game_setting)

            rewards[agent_key] = np.array(reward, np.float32)

        return self.__get_latest_obs(), rewards, self.__get_state(), done

    def reset(self) -> Tuple[Any, Any, Any]:
        self.game_step = 0
        if self.game_setting == "domain_randomisation":
            goal_diff = self.game_scores[0] - self.game_scores[1]
            if goal_diff != 0:
                if self.game_diff < 1.0:
                    self.game_diff += 0.01
                    # print("Game difficulty increased: ", self.game_diff)
                else:
                    self.game_diff = 1.0
                    print("Max game difficulty surpassed!")
            elif self.game_diff < 1.0:
                if self.game_diff > 0.0:
                    self.game_diff -= 0.01

                if self.game_diff < 0.0:
                    # Sometimes the game_diff can be slightly smaller than zero
                    # which causes problems.
                    self.game_diff = 0.0
                    # print("Game difficulty decreased: ", self.game_diff)

            self.game_length = int(100 + self.game_diff * 900)
            # print("Reset game length: ", self.game_length)
            # TODO: Update alpha
            # self.alpha = 1 / (max_dist * game_length * (1 + beta))

            self.trainer.reset_game(
                self.players_per_team,
                self.team_names,
                self.game_setting,
                self.game_diff,
            )
        elif self.game_setting == "reward_shaping":
            self.trainer.reset_game(
                self.players_per_team, self.team_names, self.game_setting
            )
        else:
            raise NotImplementedError("Unknown game setting: ", self.game_setting)

        self.game_scores = [0, 0]

        # print("Resetting.")
        # Switch teams
        # TODO: Add team switching for better training. Also maybe shuffle agents in a
        # single team. Therefore change the agent each policy is controlling at every
        # reset step. Make sure that the dictionary stuff does not break.

        # Only shuffle if teams are equal for now.
        # if self.players_per_team[0] == self.players_per_team[1]:
        #     team1 = self.agents[0:self.players_per_team[0]]
        #     team2 = self.agents[self.players_per_team[0]:]
        #
        #     random.shuffle(team1)
        #     random.shuffle(team2)
        #
        #     self.agents = team2
        #     self.agents.extend(team1)
        rewards = [None] * len(self.agents)

        # Wait for the environment to step and provide the next observations
        wait_for_next_observations([self.trainer])
        return self.__get_latest_obs(), rewards, self.__get_state()

    def __get_state(self) -> Dict[str, Any]:
        # Return latest observations to agents
        state_dict = self.trainer.get_state_dict()
        # TODO: Return per agent states. The states should be mirrored. So a critic and
        # policy can train on only one side but execute on both sides.
        # state_dict
        state_dict["game_step"] = self.game_step
        state_dict["game_length"] = self.game_length
        return state_dict

    def __get_latest_obs(self) -> Dict[str, Any]:
        # Return latest observations to agents
        obs = {}
        for agent_key, agent in self.agents.items():
            obs_dict = agent.get_latest_observation()
            if obs_dict:
                obs_dict["game_step"] = self.game_step
                obs_dict["game_length"] = self.game_length

            obs[agent_key] = obs_dict

        return obs
