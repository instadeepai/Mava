#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simulate a traffic junction environment.

Used to test if MARL communication is working.
Each agent can observe its last action, its route id, its location,
and a vision square around itself.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
    - Action Space & Observation Space are according to an agent
    - Rewards
         -0.05 at each time step till the time
         -10 for each crash
    - Episode ends when all cars reach destination / max steps
    - Obs. State:
"""

import math
from typing import Any, Dict, List, Set, Tuple

import gym
import numpy as np
from gym import spaces


class TrafficJunctionEnv(gym.Env):

    # Set during reset
    alive_mask: np.ndarray
    wait: np.ndarray
    cars_in_sys: int
    chosen_path: List[List[int]]
    route_id: List[int]
    car_ids: np.ndarray
    car_loc: np.ndarray
    car_last_act: np.ndarray
    car_route_loc: np.ndarray
    stat: Dict[str, Any]

    # Set during step
    is_completed: np.ndarray

    # Set during init constructing routes
    routes: List[List[List[Tuple[int, int]]]]

    #
    # START OF ENVIRONMENT INIT METHODS
    #

    def __init__(
        self,
        num_agents: int = 5,
        dim: int = 5,
        vision: int = 1,
        add_rate_min: float = 0.3,
        add_rate_max: float = 0.3,
        curr_start: int = 0,
        curr_end: int = 0,
        difficulty: str = "easy",
        vocab_type: str = "bool",
        comm_range: float = 3,
    ):
        """Create a Traffic Junction environment.

        Args:
            num_agents: Max number of cars in the environment at once.
            dim: Dimension of box (i.e length of road).
            vision: Vision of car.
            add_rate_min: Rate at which to add car (till curr. start).
            add_rate_max: Max rate at which to add car.
            curr_start: Start increasing add_rate after this many episodes.
            curr_end: After how many episodes to make the game hardest.
            difficulty: Difficulty level (easy | medium | hard).
            vocab_type: Type of location vector to use (bool | scalar).
            comm_range: Agent communication range.
        """
        self.__version__ = "0.0.1"

        # Grid cell classes
        self.OUTSIDE_CLASS = 0
        self.ROAD_CLASS = 1
        self.CAR_CLASS = 2

        # Rewards
        self.TIMESTEP_PENALTY = -0.01
        self.CRASH_PENALTY = -10

        # Environment arguments
        self.num_agents = num_agents
        self.dim = dim
        self.vision = vision
        self.add_rate_min = add_rate_min
        self.add_rate_max = add_rate_max
        self.curr_start = curr_start
        self.curr_end = curr_end
        self.difficulty = difficulty
        self.vocab_type = vocab_type
        self.comm_range = comm_range

        # For tracking during episodes
        self.episode_over = False
        self.has_failed = 0

        # Check that environment dimensions are licit
        self.dims = dims = [self.dim, self.dim]

        if difficulty in ["medium", "easy"]:
            assert (
                dims[0] % 2 == 0
            ), "Only even dimension supported for medium / easy Traffic Junction"
            assert dims[0] >= 4 + vision, "Min dim: 4 + vision"

        if difficulty == "hard":
            assert dims[0] >= 9, "Min dim: 9"
            assert dims[0] % 3 == 0, "Hard version works for multiple of 3. dim. only."

        # For computing add rate
        self.exact_rate = self.add_rate = self.add_rate_min
        self.epoch_last_update = 0

        # Define what an agent can do
        # (0: GAS, 1: BRAKE) i.e. (0: Move 1-step, 1: STAY)
        self.n_actions = 2
        self.action_space = spaces.Discrete(self.n_actions)

        # Make number of dims odd for easy case.
        if difficulty == "easy":
            self.dims = list(dims)
            for i in range(len(self.dims)):
                self.dims[i] += 1

        # Set up number of paths
        n_roads = {"easy": 2, "medium": 4, "hard": 8}

        def n_pick_r(n: int, r: int) -> int:
            """Compute nPr."""
            f = math.factorial
            return f(n) // f(n - r)

        self.n_paths = n_pick_r(n_roads[difficulty], 2)

        # Setting max vocab size for 1-hot encoding
        dim_sum = dims[0] + dims[1]
        base = {"easy": dim_sum, "medium": 2 * dim_sum, "hard": 4 * dim_sum}

        if self.vocab_type == "bool":
            self.OUTSIDE_CLASS += base[difficulty]
            self.CAR_CLASS += base[difficulty]

            # car_type + base + outside + 0-index
            self.vocab_size = 1 + base[difficulty] + 1 + 1

            # Observation for each agent will be 4-tuple of:
            # (r_i, last_act, vision * vision * vocab)
            self.observation_space = spaces.Tuple(
                (
                    spaces.Discrete(self.n_actions),
                    spaces.Discrete(self.n_paths),
                    spaces.MultiBinary(
                        (2 * vision + 1, 2 * vision + 1, self.vocab_size)
                    ),
                )
            )
        else:
            # r_i, (x,y), vocab = [road class + car]
            self.vocab_size = 1 + 1

            # Observation for each agent will be 4-tuple of:
            # (r_i, last_act, len(dims), vision * vision * vocab)
            self.observation_space = spaces.Tuple(
                (
                    spaces.Discrete(self.n_actions),
                    spaces.Discrete(self.n_paths),
                    spaces.MultiDiscrete(self.dims),
                    spaces.MultiBinary(
                        (2 * vision + 1, 2 * vision + 1, self.vocab_size)
                    ),
                )
            )
            # Actual observation will be of the shape:
            # 1 * num_agents * ((x,y) , (2v+1) * (2v+1) * vocab_size)

        # Create the grid
        self._set_grid()

        # Create the paths
        if difficulty == "easy":
            self._set_paths_easy()
        else:
            self._set_paths(difficulty)

    def _set_grid(self) -> None:
        """Create the environment base grid.

        Returns:
            None.
        """
        self.grid = np.full(
            self.dims[0] * self.dims[1], self.OUTSIDE_CLASS, dtype=int
        ).reshape(self.dims)
        width, height = self.dims

        # Mark the roads
        roads = self._get_road_blocks(width, height)
        for road in roads:
            self.grid[road] = self.ROAD_CLASS
        if self.vocab_type == "bool":
            self.route_grid = self.grid.copy()
            start = 0
            for road in roads:
                sz = int(np.prod(self.grid[road].shape))
                self.grid[road] = np.arange(start, start + sz).reshape(
                    self.grid[road].shape
                )
                start += sz

        # Padding for vision
        self.pad_grid = np.pad(
            self.grid, self.vision, "constant", constant_values=self.OUTSIDE_CLASS
        )

        self.empty_bool_base_grid = self._onehot_initialization(self.pad_grid)

    def _get_road_blocks(
        self,
        width: int,
        height: int,
    ) -> List[Any]:
        """Get grid ranges for each of the roads.

        Args:
            width: env width.
            height: env height.

        Returns:
            List of roads. Each road is a np.IndexExpression.
        """

        # assuming 1 is the lane width for each direction.
        road_blocks: Dict[str, List[Any]] = {
            "easy": [np.s_[height // 2, :], np.s_[:, width // 2]],
            "medium": [
                np.s_[height // 2 - 1 : height // 2 + 1, :],
                np.s_[:, width // 2 - 1 : width // 2 + 1],
            ],
            "hard": [
                np.s_[height // 3 - 2 : height // 3, :],
                np.s_[2 * height // 3 : 2 * height // 3 + 2, :],
                np.s_[:, width // 3 - 2 : width // 3],
                np.s_[:, 2 * height // 3 : 2 * height // 3 + 2],
            ],
        }

        return road_blocks[self.difficulty]

    def _onehot_initialization(self, a: np.ndarray) -> np.ndarray:
        """Create a one-hot version of the given array.

        Args:
            a: Array to convert to a one-hot encoding.

        Returns:
            One-hot encoding of input array.
        """
        if self.vocab_type == "bool":
            ncols = self.vocab_size
        else:
            ncols = (
                self.vocab_size + 1
            )  # 1 is for outside class which will be removed later.
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    @staticmethod
    def _all_idx(idx: np.ndarray, axis: int) -> Tuple:
        """Convert array of numbers to index locations.

        Args:
            idx: Array of numbers to use for index locations.
            axis: Axis of array to index on.

        Returns:
            Tuple of index locations.
        """
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def _set_paths_easy(self) -> None:
        """Set the environment paths for the easy version.

        Returns:
            None.
        """
        h, w = self.dims
        routes: Dict[str, List[np.ndarray]] = {"TOP": [], "LEFT": []}

        # 0 refers to UP to DOWN, type 0
        full = [(i, w // 2) for i in range(h)]
        routes["TOP"].append(np.array([*full]))

        # 1 refers to LEFT to RIGHT, type 0
        full = [(h // 2, i) for i in range(w)]
        routes["LEFT"].append(np.array([*full]))

        self.routes = list(routes.values())

    def _set_paths(self, difficulty: str) -> None:
        """Set the environment paths for the medium or hard version.

        Returns:
            None.
        """
        route_grid = self.route_grid if self.vocab_type == "bool" else self.grid
        self.routes = self._get_routes(route_grid, difficulty)

        # Convert/unroll routes which is a list of list of paths
        paths = []
        for r in self.routes:
            for p in r:
                paths.append(p)

        # Check number of paths
        assert len(paths) == self.n_paths

        # Test all paths
        assert self._unittest_path(paths)

    @staticmethod
    def _unittest_path(paths: List[List[Tuple[int, int]]]) -> bool:
        """Check if paths are valid.

        Args:
            paths: Environment roads, as lists of points.

        Returns:
            Whether the paths are all valid.
        """
        for i, p in enumerate(paths[:-1]):
            next_dif = p - np.row_stack([p[1:], p[-1]])
            next_dif = np.abs(next_dif[:-1])
            step_jump = np.sum(next_dif, axis=1)
            if np.any(step_jump != 1):
                return False
            if not np.all(step_jump == 1):
                return False
        return True

    def _get_routes(
        self, grid: np.ndarray, difficulty: str
    ) -> List[List[List[Tuple[int, int]]]]:
        """Get the environment routes for medium or hard env.

        Args:
            grid: Environment grid.
            difficulty: Medium or hard.

        Returns:
            List of paths, each expressed as a list of points.
        """
        assert difficulty == "medium" or difficulty == "hard"

        arrival_points, finish_points, road_dir, junction = self._get_add_mat(
            grid, difficulty
        )

        n_turn1 = 3  # 0 - straight, 1-right, 2-left
        n_turn2 = 1 if difficulty == "medium" else 3

        routes = []
        # routes for each arrival point
        for i in range(len(arrival_points)):
            paths = []
            # turn 1
            for turn_1 in range(n_turn1):
                # turn 2
                for turn_2 in range(n_turn2):
                    total_turns = 0
                    curr_turn = turn_1
                    path = []
                    visited = set()
                    current = arrival_points[i]
                    path.append(current)
                    start = current
                    turn_step = 0
                    # "start"
                    while not self._goal_reached(i, current, finish_points):
                        visited.add(current)
                        current, turn_prog, turn_completed = self._next_move(
                            current,
                            curr_turn,
                            turn_step,
                            start,
                            grid,
                            road_dir,
                            junction,
                            visited,
                        )
                        if curr_turn == 2 and turn_prog:
                            turn_step += 1
                        if turn_completed:
                            total_turns += 1
                            curr_turn = turn_2
                            turn_step = 0
                            start = current
                        # keep going straight till the exit if 2 turns made already.
                        if total_turns == 2:
                            curr_turn = 0
                        path.append(current)
                    paths.append(path)
                    # early stopping, if first turn leads to exit
                    if total_turns == 1:
                        break
            routes.append(paths)
        return routes

    def _get_add_mat(
        self, grid: np.ndarray, difficulty: str
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], np.ndarray, np.ndarray]:
        """Get a sequence of information needed to create roads.

        Args:
            grid: environment grid.
            difficulty: 'medium' or 'hard'

        Returns:
            Tuple[car arrival points, finish points, road directions, junctions]
        """
        h, w = self.dims

        road_dir = grid.copy()
        junction = np.zeros_like(grid)
        arrival_points, finish_points = [], []

        if difficulty == "medium":
            arrival_points = [
                (0, w // 2 - 1),  # TOP
                (h - 1, w // 2),  # BOTTOM
                (h // 2, 0),  # LEFT
                (h // 2 - 1, w - 1),
            ]  # RIGHT

            finish_points = [
                (0, w // 2),  # TOP
                (h - 1, w // 2 - 1),  # BOTTOM
                (h // 2 - 1, 0),  # LEFT
                (h // 2, w - 1),
            ]  # RIGHT

            # mark road direction
            road_dir[h // 2, :] = 2
            road_dir[h // 2 - 1, :] = 3
            road_dir[:, w // 2] = 4

            # mark the Junction
            junction[h // 2 - 1 : h // 2 + 1, w // 2 - 1 : w // 2 + 1] = 1

        elif difficulty == "hard":
            arrival_points = [
                (0, w // 3 - 2),  # TOP-left
                (0, 2 * w // 3),  # TOP-right
                (h // 3 - 1, 0),  # LEFT-top
                (2 * h // 3 + 1, 0),  # LEFT-bottom
                (h - 1, w // 3 - 1),  # BOTTOM-left
                (h - 1, 2 * w // 3 + 1),  # BOTTOM-right
                (h // 3 - 2, w - 1),  # RIGHT-top
                (2 * h // 3, w - 1),
            ]  # RIGHT-bottom

            finish_points = [
                (0, w // 3 - 1),  # TOP-left
                (0, 2 * w // 3 + 1),  # TOP-right
                (h // 3 - 2, 0),  # LEFT-top
                (2 * h // 3, 0),  # LEFT-bottom
                (h - 1, w // 3 - 2),  # BOTTOM-left
                (h - 1, 2 * w // 3),  # BOTTOM-right
                (h // 3 - 1, w - 1),  # RIGHT-top
                (2 * h // 3 + 1, w - 1),
            ]  # RIGHT-bottom

            # mark road direction
            road_dir[h // 3 - 1, :] = 2
            road_dir[2 * h // 3, :] = 3
            road_dir[2 * h // 3 + 1, :] = 4

            road_dir[:, w // 3 - 2] = 5
            road_dir[:, w // 3 - 1] = 6
            road_dir[:, 2 * w // 3] = 7
            road_dir[:, 2 * w // 3 + 1] = 8

            # mark the Junctions
            junction[h // 3 - 2 : h // 3, w // 3 - 2 : w // 3] = 1
            junction[2 * h // 3 : 2 * h // 3 + 2, w // 3 - 2 : w // 3] = 1

            junction[h // 3 - 2 : h // 3, 2 * w // 3 : 2 * w // 3 + 2] = 1
            junction[2 * h // 3 : 2 * h // 3 + 2, 2 * w // 3 : 2 * w // 3 + 2] = 1

        return arrival_points, finish_points, road_dir, junction

    @staticmethod
    def _goal_reached(
        place_i: int, curr: Tuple[int, int], finish_points: List[Tuple[int, int]]
    ) -> bool:
        """Check whether location is at the end of a road.

        Args:
            place_i: Arrival point.
            curr: Location.
            finish_points: List of finish points.

        Returns:
            Whether curr is at the end of a road.
        """
        return curr in finish_points[:place_i] + finish_points[place_i + 1 :]

    @staticmethod
    def _next_move(
        curr: Tuple[int, int],
        turn: int,
        turn_step: int,
        start: Tuple[int, int],
        grid: np.ndarray,
        road_dir: np.ndarray,
        junction: np.ndarray,
        visited: Set[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], bool, bool]:
        """Get the next location of a road cell.

        Args:
            curr: Current car location.
            turn: How many turns made so far.
            turn_step: Step number in current turn.
            start: Start location of current path.
            grid: Environment grid.
            road_dir: Road directions.
            junction: Environment junctions.
            visited: Points already visited.

        Returns:
            Tuple[New car loc, if currently in turn, if turn completed].
        """
        move = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        h, w = grid.shape
        turn_completed = False
        turn_prog = False
        neigh = []
        for m in move:
            # check lane while taking left turn
            n = (curr[0] + m[0], curr[1] + m[1])
            if (
                0 <= n[0] <= h - 1
                and 0 <= n[1] <= w - 1
                and grid[n]
                and n not in visited
            ):
                # On Junction, use turns
                if junction[n] == junction[curr] == 1:
                    if (turn == 0 or turn == 2) and (
                        (n[0] == start[0]) or (n[1] == start[1])
                    ):
                        # Straight on junction for either left or straight
                        neigh.append(n)
                        if turn == 2:
                            turn_prog = True

                    # left from junction
                    elif turn == 2 and turn_step == 1:
                        neigh.append(n)
                        turn_prog = True

                    else:
                        # End of path
                        pass

                # Completing left turn on junction
                elif (
                    junction[curr]
                    and not junction[n]
                    and turn == 2
                    and turn_step == 2
                    and (abs(start[0] - n[0]) == 2 or abs(start[1] - n[1]) == 2)
                ):
                    neigh.append(n)
                    turn_completed = True

                # junction seen, get onto it;
                elif junction[n] and not junction[curr]:
                    neigh.append(n)

                # right from junction
                elif turn == 1 and not junction[n] and junction[curr]:
                    neigh.append(n)
                    turn_completed = True

                # Straight from jucntion
                elif turn == 0 and junction[curr] and road_dir[n] == road_dir[start]:
                    neigh.append(n)
                    turn_completed = True

                # keep going no decision to make;
                elif road_dir[n] == road_dir[curr] and not junction[curr]:
                    neigh.append(n)

        if neigh:
            return neigh[0], turn_prog, turn_completed

        raise RuntimeError("next move should be of len 1. Reached ambiguous situation.")

    #
    # END OF ENVIRONMENT INIT METHODS
    #

    def reset(self, epoch: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Reset the state of the environment and returns an initial observation.

        Returns:
            Tuple[initial observation, communication graph].
        """
        self.episode_over = False
        self.has_failed = 0

        self.alive_mask = np.zeros(self.num_agents)
        self.wait = np.zeros(self.num_agents)
        self.cars_in_sys = 0

        # Chosen path for each car:
        self.chosen_path = [[] for _ in range(self.num_agents)]
        # when dead => no route, must be masked by trainer.
        self.route_id = [-1] * self.num_agents

        # self.cars = np.zeros(self.num_agents)
        # Current car to enter system
        # self.car_i = 0
        # Ids i.e. indexes
        self.car_ids = np.arange(self.CAR_CLASS, self.CAR_CLASS + self.num_agents)

        # Starting loc of car: a place where everything is outside class
        self.car_loc = np.zeros((self.num_agents, len(self.dims)), dtype=int)
        self.car_last_act = np.zeros(
            self.num_agents, dtype=int
        )  # last act GAS when awake

        self.car_route_loc = np.full(self.num_agents, -1)

        # stat - like success ratio
        self.stat = dict()

        # set add rate according to the curriculum
        epoch_range = self.curr_end - self.curr_start
        add_rate_range = self.add_rate_max - self.add_rate_min
        if (
            epoch is not None
            and epoch_range > 0
            and add_rate_range > 0
            and epoch > self.epoch_last_update
        ):
            self.curriculum(epoch)
            self.epoch_last_update = epoch

        # Observation will be num_agents * vision * vision ndarray
        obs = self._get_obs()
        return obs, self._get_env_graph()

    def curriculum(self, epoch: int) -> None:
        """Adjust add rate based on current epoch.

        Args:
            epoch: Epoch number (usually just episode number).

        Returns:
            None.
        """
        step_size = 0.01
        step = (self.add_rate_max - self.add_rate_min) / (
            self.curr_end - self.curr_start
        )

        if self.curr_start <= epoch < self.curr_end:
            self.exact_rate = self.exact_rate + step
            self.add_rate = step_size * (self.exact_rate // step_size)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """Step the cars in the environment.

        Args:
            action: Action given for each agent in np array.

        Returns:
            Tuple[obs, rewards, if episode over, extras dict]
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")

        # Expected shape: either num_agents or num_agents x 1
        action = np.array(action).squeeze()

        assert np.all(
            action <= self.n_actions
        ), "Actions should be in the range [0,naction)."

        assert (
            len(action) == self.num_agents
        ), "Action for each agent should be provided."

        # No one is completed before taking action
        self.is_completed = np.zeros(self.num_agents)

        for i, a in enumerate(action):
            self._take_action(i, a)

        self._add_cars()

        obs = self._get_obs()
        env_graph = self._get_env_graph()
        reward = self._get_reward()

        extras = {
            "car_loc": self.car_loc,
            "alive_mask": np.copy(self.alive_mask),
            "wait": self.wait,
            "cars_in_sys": self.cars_in_sys,
            "is_completed": np.copy(self.is_completed),
            "env_graph": env_graph,
        }

        self.stat["success"] = 1 - self.has_failed
        self.stat["add_rate"] = self.add_rate

        return obs, reward, self.episode_over, extras

    def _take_action(self, idx: int, act: int) -> None:
        """Take action for particular car.

        Args:
            idx: Car id.
            act: Car action.

        Returns:
            None.
        """
        # non-active car
        if self.alive_mask[idx] == 0:
            return

        # add wait time for active cars
        self.wait[idx] += 1

        # action BRAKE i.e STAY
        if act == 1:
            self.car_last_act[idx] = 1
            return

        # GAS or move
        if act == 0:
            self.car_route_loc[idx] += 1
            curr = self.car_route_loc[idx]

            # car/agent has reached end of its path
            if curr == len(self.chosen_path[idx]):
                self.cars_in_sys -= 1
                self.alive_mask[idx] = 0
                self.wait[idx] = 0

                # put it at dead loc
                self.car_loc[idx] = np.zeros(len(self.dims), dtype=int)
                self.is_completed[idx] = 1
                return

            elif curr > len(self.chosen_path[idx]):
                raise RuntimeError("Out of bound car path")

            curr = self.chosen_path[idx][curr]

            self.car_loc[idx] = curr

            # Change last act for color:
            self.car_last_act[idx] = 0

    def _add_cars(self) -> None:
        for r_i, routes in enumerate(self.routes):
            if self.cars_in_sys >= self.num_agents:
                return

            # Add car to system and set on path
            if np.random.uniform() <= self.add_rate:

                # chose dead car on random
                idx = self._choose_dead()
                # make it alive
                self.alive_mask[idx] = 1

                # choose path randomly & set it
                p_i = np.random.choice(len(routes))
                # make sure all self.routes have equal len/ same no. of routes
                self.route_id[idx] = p_i + r_i * len(routes)
                self.chosen_path[idx] = routes[p_i]

                # set its start loc
                self.car_route_loc[idx] = 0
                self.car_loc[idx] = routes[p_i][0]

                # increase count
                self.cars_in_sys += 1

    def _choose_dead(self) -> np.ndarray:
        """Return a random car which is currently not in the environment.

        Returns:
            ID of random car not currently in environment.
        """
        # all idx
        car_idx = np.arange(len(self.alive_mask))
        # random choice of idx from dead ones.
        return np.random.choice(car_idx[self.alive_mask == 0])

    def _get_obs(self) -> np.ndarray:
        """Get all car observations.

        Returns:
            Agent observations as a np array.
        """
        h, w = self.dims
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        # Mark cars' location in Bool grid
        for i, p in enumerate(self.car_loc):
            self.bool_base_grid[
                p[0] + self.vision, p[1] + self.vision, self.CAR_CLASS
            ] += 1

        # remove the outside class.
        if self.vocab_type == "scalar":
            self.bool_base_grid = self.bool_base_grid[:, :, 1:]

        obs = []
        for i, p in enumerate(self.car_loc):
            # most recent action
            act = self.car_last_act[i] / (self.n_actions - 1)

            # route id
            r_i = self.route_id[i] / (self.n_paths - 1)

            # loc
            p_norm = p / (h - 1, w - 1)

            # vision square
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            v_sq = self.bool_base_grid[slice_y, slice_x]

            # when dead, all obs are 0. But should be masked by trainer.
            if self.alive_mask[i] == 0:
                act = np.zeros_like(act)
                r_i = np.zeros_like(r_i)
                p_norm = np.zeros_like(p_norm)
                v_sq = np.zeros_like(v_sq)

            if self.vocab_type == "bool":
                o = tuple((act, r_i, v_sq))
            else:
                o = tuple((act, r_i, p_norm, v_sq))
            obs.append(o)

        obs = self._flatten_obs(obs)
        return obs

    def _flatten_obs(self, obs: List) -> np.ndarray:
        """Get flattened agent observations.

        Args:
            obs: Observations for each agent.

        Returns:
            Flatten agent observations as a np array.
        """
        _obs = []
        for agent in obs:  # list/tuple of observations.
            ag_obs = []
            for obs_kind in agent:
                ag_obs.append(np.array(obs_kind).flatten())
            _obs.append(np.concatenate(ag_obs))
        obs_flat = np.stack(_obs)

        obs_flat = obs_flat.reshape(-1, self.observation_dim)
        return obs_flat

    @property
    def observation_dim(self) -> int:
        """The total number of elements in an agent observation.

        Returns:
            Number of elements.
        """
        # tuple space
        if hasattr(self.observation_space, "spaces"):
            total_obs_dim = 0
            for space in self.observation_space.spaces:
                if hasattr(self.action_space, "shape"):
                    total_obs_dim += int(np.prod(space.shape))
                else:  # Discrete
                    total_obs_dim += 1
            return total_obs_dim
        else:
            return int(np.prod(self.observation_space.shape))

    # Get communication graph -> just use distance for now
    def _get_env_graph(self) -> np.ndarray:
        """Get communication graph based on agent distances.

        Returns:
            Adjacency matrix of graph.
        """
        adj = np.zeros(
            (self.num_agents, self.num_agents)
        )  # 1 if agents can communicate
        for i in range(self.num_agents):
            if not self.alive_mask[i]:
                continue
            for j in range(i + 1, self.num_agents):
                if not self.alive_mask[j]:
                    continue
                car_loc1 = self.car_loc[i]
                car_loc2 = self.car_loc[j]
                squared_distance = (car_loc1[0] - car_loc2[0]) ** 2 + (
                    car_loc1[1] - car_loc2[1]
                ) ** 2
                if squared_distance <= self.comm_range**2:
                    adj[i][j] = 1
                    adj[j][i] = 1

        return adj

    def _get_reward(self) -> np.ndarray:
        """Get reward for all agents.

        Returns:
            Reward for all agents as a np array with shape [num_agents]
        """
        reward = np.full(self.num_agents, self.TIMESTEP_PENALTY) * self.wait

        for i, l in enumerate(self.car_loc):
            if (
                len(np.where(np.all(self.car_loc[:i] == l, axis=1))[0])
                or len(np.where(np.all(self.car_loc[i + 1 :] == l, axis=1))[0])
            ) and l.any():
                reward[i] += self.CRASH_PENALTY
                self.has_failed = 1

        reward = self.alive_mask * reward
        return reward

    def reward_terminal(self) -> np.ndarray:
        """Reward given when environment is terminal.

        Returns:
            Zeros np array in shape of rewards.
        """
        return np.zeros_like(self._get_reward())
