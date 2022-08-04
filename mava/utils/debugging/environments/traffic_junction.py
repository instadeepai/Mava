#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a traffic junction environment.
Used to test if MARL communication is working.
Each agent can observe its last action, its route id, its location, and a vision square around itself.

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
import curses
import gym
import numpy as np
from gym import spaces


class TrafficJunctionEnv(gym.Env):

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
        difficulty: str = 'easy',
        vocab_type: str = 'bool',
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

        # Set during reset
        self.alive_mask = None
        self.wait = None
        self.cars_in_sys = None
        self.chosen_path = None
        self.route_id = None
        self.car_ids = None
        self.car_loc = None
        self.car_last_act = None
        self.car_route_loc = None
        self.stat = None

        # Check that environment dimensions are licit
        self.dims = dims = (self.dim, self.dim)

        if difficulty in ['medium', 'easy']:
            assert dims[0] % 2 == 0, 'Only even dimension supported for medium / easy Traffic Junction'
            assert dims[0] >= 4 + vision, 'Min dim: 4 + vision'

        if difficulty == 'hard':
            assert dims[0] >= 9, 'Min dim: 9'
            assert dims[0] % 3 == 0, 'Hard version works for multiple of 3. dim. only.'

        # For computing add rate
        self.exact_rate = self.add_rate = self.add_rate_min
        self.epoch_last_update = 0

        # Define what an agent can do
        # (0: GAS, 1: BRAKE) i.e. (0: Move 1-step, 1: STAY)
        self.n_actions = 2
        self.action_space = spaces.Discrete(self.n_actions)

        # Make number of dims odd for easy case.
        if difficulty == 'easy':
            self.dims = list(dims)
            for i in range(len(self.dims)):
                self.dims[i] += 1

        # Set up number of paths
        n_roads = {'easy': 2,
                   'medium': 4,
                   'hard': 8}

        def n_pick_r(n, r):
            """Compute nPr."""
            f = math.factorial
            return f(n) // f(n - r)

        self.n_paths = n_pick_r(n_roads[difficulty], 2)

        # Setting max vocab size for 1-hot encoding
        dim_sum = dims[0] + dims[1]
        base = {'easy': dim_sum,
                'medium': 2 * dim_sum,
                'hard': 4 * dim_sum}

        if self.vocab_type == 'bool':
            self.OUTSIDE_CLASS += base[difficulty]
            self.CAR_CLASS += base[difficulty]

            # car_type + base + outside + 0-index
            self.vocab_size = 1 + base[difficulty] + 1 + 1

            # Observation for each agent will be 4-tuple of (r_i, last_act, vision * vision * vocab)
            self.observation_space = spaces.Tuple((
                                     spaces.Discrete(self.n_actions),
                                     spaces.Discrete(self.n_paths),
                                     spaces.MultiBinary(
                                        (2 * vision + 1, 2 * vision + 1, self.vocab_size)
                                     )))
        else:
            # r_i, (x,y), vocab = [road class + car]
            self.vocab_size = 1 + 1

            # Observation for each agent will be 4-tuple of (r_i, last_act, len(dims), vision * vision * vocab)
            self.observation_space = spaces.Tuple((
                                     spaces.Discrete(self.n_actions),
                                     spaces.Discrete(self.n_paths),
                                     spaces.MultiDiscrete(self.dims),
                                     spaces.MultiBinary(
                                        (2 * vision + 1, 2 * vision + 1, self.vocab_size)
                                     )))
            # Actual observation will be of the shape 1 * num_agents * ((x,y) , (2v+1) * (2v+1) * vocab_size)

        # Create the grid
        self._set_grid()

        # Create the paths
        if difficulty == 'easy':
            self._set_paths_easy()
        else:
            self._set_paths(difficulty)

    def _set_grid(self):
        self.grid = np.full(self.dims[0] * self.dims[1], self.OUTSIDE_CLASS, dtype=int).reshape(self.dims)
        width, height = self.dims

        # Mark the roads
        roads = self._get_road_blocks(width, height)
        for road in roads:
            self.grid[road] = self.ROAD_CLASS
        if self.vocab_type == 'bool':
            self.route_grid = self.grid.copy()
            start = 0
            for road in roads:
                sz = int(np.prod(self.grid[road].shape))
                self.grid[road] = np.arange(start, start + sz).reshape(self.grid[road].shape)
                start += sz

        # Padding for vision
        self.pad_grid = np.pad(self.grid, self.vision, 'constant', constant_values=self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.pad_grid)

    def _get_road_blocks(self,
                         width,
                         height,
                         ):

        # assuming 1 is the lane width for each direction.
        road_blocks = {
            'easy': [np.s_[height // 2, :],
                     np.s_[:, width // 2]],

            'medium': [np.s_[height // 2 - 1: height // 2 + 1, :],
                       np.s_[:, width // 2 - 1: width // 2 + 1]],

            'hard': [np.s_[height // 3 - 2: height // 3, :],
                     np.s_[2 * height // 3: 2 * height // 3 + 2, :],

                     np.s_[:, width // 3 - 2: width // 3],
                     np.s_[:, 2 * height // 3: 2 * height // 3 + 2]],
        }

        return road_blocks[self.difficulty]

    def _onehot_initialization(self, a):
        if self.vocab_type == 'bool':
            ncols = self.vocab_size
        else:
            ncols = self.vocab_size + 1  # 1 is for outside class which will be removed later.
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    @staticmethod
    def _all_idx(idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def _set_paths_easy(self):
        h, w = self.dims
        self.routes = {
            'TOP': [],
            'LEFT': []
        }

        # 0 refers to UP to DOWN, type 0
        full = [(i, w//2) for i in range(h)]
        self.routes['TOP'].append(np.array([*full]))

        # 1 refers to LEFT to RIGHT, type 0
        full = [(h//2, i) for i in range(w)]
        self.routes['LEFT'].append(np.array([*full]))

        self.routes = list(self.routes.values())

    def _set_paths(self, difficulty):
        route_grid = self.route_grid if self.vocab_type == 'bool' else self.grid
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

    def _get_routes(self,
                    grid,
                    difficulty
                    ):
        grid.dtype = int

        assert difficulty == 'medium' or difficulty == 'hard'

        arrival_points, finish_points, road_dir, junction = self._get_add_mat(grid, difficulty)

        n_turn1 = 3  # 0 - straight, 1-right, 2-left
        n_turn2 = 1 if difficulty == 'medium' else 3

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
                            current, curr_turn, turn_step, start, grid, road_dir, junction, visited
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

    def _get_add_mat(self,
                     grid,
                     difficulty
                     ):
        h, w = self.dims

        road_dir = grid.copy()
        junction = np.zeros_like(grid)
        arrival_points, finish_points = None, None

        if difficulty == 'medium':
            arrival_points = [(0, w // 2 - 1),  # TOP
                              (h - 1, w // 2),  # BOTTOM
                              (h // 2, 0),  # LEFT
                              (h // 2 - 1, w - 1)]  # RIGHT

            finish_points = [(0, w // 2),  # TOP
                             (h - 1, w // 2 - 1),  # BOTTOM
                             (h // 2 - 1, 0),  # LEFT
                             (h // 2, w - 1)]  # RIGHT

            # mark road direction
            road_dir[h // 2, :] = 2
            road_dir[h // 2 - 1, :] = 3
            road_dir[:, w // 2] = 4

            # mark the Junction
            junction[h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1] = 1

        elif difficulty == 'hard':
            arrival_points = [(0, w // 3 - 2),  # TOP-left
                              (0, 2 * w // 3),  # TOP-right

                              (h // 3 - 1, 0),  # LEFT-top
                              (2 * h // 3 + 1, 0),  # LEFT-bottom

                              (h - 1, w // 3 - 1),  # BOTTOM-left
                              (h - 1, 2 * w // 3 + 1),  # BOTTOM-right

                              (h // 3 - 2, w - 1),  # RIGHT-top
                              (2 * h // 3, w - 1)]  # RIGHT-bottom

            finish_points = [(0, w // 3 - 1),  # TOP-left
                             (0, 2 * w // 3 + 1),  # TOP-right

                             (h // 3 - 2, 0),  # LEFT-top
                             (2 * h // 3, 0),  # LEFT-bottom

                             (h - 1, w // 3 - 2),  # BOTTOM-left
                             (h - 1, 2 * w // 3),  # BOTTOM-right

                             (h // 3 - 1, w - 1),  # RIGHT-top
                             (2 * h // 3 + 1, w - 1)]  # RIGHT-bottom

            # mark road direction
            road_dir[h // 3 - 1, :] = 2
            road_dir[2 * h // 3, :] = 3
            road_dir[2 * h // 3 + 1, :] = 4

            road_dir[:, w // 3 - 2] = 5
            road_dir[:, w // 3 - 1] = 6
            road_dir[:, 2 * w // 3] = 7
            road_dir[:, 2 * w // 3 + 1] = 8

            # mark the Junctions
            junction[h // 3 - 2:h // 3, w // 3 - 2:w // 3] = 1
            junction[2 * h // 3:2 * h // 3 + 2, w // 3 - 2:w // 3] = 1

            junction[h // 3 - 2:h // 3, 2 * w // 3:2 * w // 3 + 2] = 1
            junction[2 * h // 3:2 * h // 3 + 2, 2 * w // 3:2 * w // 3 + 2] = 1

        return arrival_points, finish_points, road_dir, junction

    @staticmethod
    def _goal_reached(place_i,
                      curr,
                      finish_points
                      ):
        return curr in finish_points[:place_i] + finish_points[place_i + 1:]

    @staticmethod
    def _next_move(curr,
                   turn,
                   turn_step,
                   start,
                   grid,
                   road_dir,
                   junction,
                   visited
                   ):
        move = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        h, w = grid.shape
        turn_completed = False
        turn_prog = False
        neigh = []
        for m in move:
            # check lane while taking left turn
            n = (curr[0] + m[0], curr[1] + m[1])
            if 0 <= n[0] <= h - 1 and 0 <= n[1] <= w - 1 and grid[n] and n not in visited:
                # On Junction, use turns
                if junction[n] == junction[curr] == 1:
                    if (turn == 0 or turn == 2) and ((n[0] == start[0]) or (n[1] == start[1])):
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
                elif junction[curr] and not junction[n] and turn == 2 and turn_step == 2 \
                        and (abs(start[0] - n[0]) == 2 or abs(start[1] - n[1]) == 2):
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
        if len(neigh) != 1:
            raise RuntimeError("next move should be of len 1. Reached ambiguous situation.")

    #
    # END OF ENVIRONMENT INIT METHODS
    #

    def reset(self, epoch=None):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.has_failed = 0

        self.alive_mask = np.zeros(self.num_agents)
        self.wait = np.zeros(self.num_agents)
        self.cars_in_sys = 0

        # Chosen path for each car:
        self.chosen_path = [0] * self.num_agents
        # when dead => no route, must be masked by trainer.
        self.route_id = [-1] * self.num_agents

        # self.cars = np.zeros(self.num_agents)
        # Current car to enter system
        # self.car_i = 0
        # Ids i.e. indexes
        self.car_ids = np.arange(self.CAR_CLASS, self.CAR_CLASS + self.num_agents)

        # Starting loc of car: a place where everything is outside class
        self.car_loc = np.zeros((self.num_agents, len(self.dims)),dtype=int)
        self.car_last_act = np.zeros(self.num_agents, dtype=int) # last act GAS when awake

        self.car_route_loc = np.full(self.num_agents, - 1)

        # stat - like success ratio
        self.stat = dict()

        # set add rate according to the curriculum
        epoch_range = (self.curr_end - self.curr_start)
        add_rate_range = (self.add_rate_max - self.add_rate_min)
        if epoch is not None and epoch_range > 0 and add_rate_range > 0 and epoch > self.epoch_last_update:
            self.curriculum(epoch)
            self.epoch_last_update = epoch

        # Observation will be num_agents * vision * vision ndarray
        obs = self._get_obs()
        return obs, self._get_env_graph()

    # Get communication graph -> just use distance for now
    def _get_env_graph(self):
        adj = np.zeros((1, self.num_agents, self.num_agents))  # 1 if agents can communicate
        for i in range(self.num_agents):
            if not self.alive_mask[i]:
                continue
            for j in range(i + 1, self.num_agents):
                if not self.alive_mask[j]:
                    continue
                car_loc1 = self.car_loc[i]
                car_loc2 = self.car_loc[j]
                squared_distance = (car_loc1[0] - car_loc2[0])**2 + (car_loc1[1] - car_loc2[1])**2
                if squared_distance <= self.comm_range**2:
                    adj[0][i][j] = 1
                    adj[0][j][i] = 1

        return adj

    def step(self, action):
        """
        The agents(car) take a step in the environment.

        Parameters
        ----------
        action : shape - either num_agents or num_agents x 1

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :
            reward (num_agents x 1) : PENALTY for each timestep when in sys & CRASH PENALTY on crashes.
            episode_over (bool) : Will be true when episode gets over.
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")

        # Expected shape: either num_agents or num_agents x 1
        action = np.array(action).squeeze()

        assert np.all(action <= self.n_actions), "Actions should be in the range [0,naction)."

        assert len(action) == self.num_agents, "Action for each agent should be provided."

        # No one is completed before taking action
        self.is_completed = np.zeros(self.num_agents)

        for i, a in enumerate(action):
            self._take_action(i, a)

        self._add_cars()

        obs = self._get_obs()
        env_graph = self._get_env_graph()
        reward = self._get_reward()

        extras = {'car_loc':self.car_loc,
                  'alive_mask': np.copy(self.alive_mask),
                  'wait': self.wait,
                  'cars_in_sys': self.cars_in_sys,
                  'is_completed': np.copy(self.is_completed),
                  'env_graph': env_graph}

        self.stat['success'] = 1 - self.has_failed
        self.stat['add_rate'] = self.add_rate

        return obs, reward, self.episode_over, extras

    def render(self, mode='human', close=False):

        grid = self.grid.copy().astype(object)
        # grid = np.zeros(self.dims[0]*self.dims[1], dtypeobject).reshape(self.dims)
        grid[grid != self.OUTSIDE_CLASS] = '_'
        grid[grid == self.OUTSIDE_CLASS] = ''
        self.stdscr.clear()
        for i, p in enumerate(self.car_loc):
            if self.car_last_act[i] == 0: # GAS
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]).replace('_','') + '<>'
                else:
                    grid[p[0]][p[1]] = '<>'
            else: # BRAKE
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]).replace('_','') + '<b>'
                else:
                    grid[p[0]][p[1]] = '<b>'

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if row_num == idx == 0:
                    continue
                if item != '_':
                    if '<>' in item and len(item) > 3: #CRASH, one car accelerates
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(2))
                    elif '<>' in item: #GAS
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    elif 'b' in item and len(item) > 3: #CRASH
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(2))
                    elif 'b' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(5))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2))
                else:
                    self.stdscr.addstr(row_num, idx * 4, '_'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()

    def seed(self):
        return

    def _get_obs(self):
        h, w = self.dims
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        # Mark cars' location in Bool grid
        for i, p in enumerate(self.car_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.CAR_CLASS] += 1


        # remove the outside class.
        if self.vocab_type == 'scalar':
            self.bool_base_grid = self.bool_base_grid[:,:,1:]


        obs = []
        for i, p in enumerate(self.car_loc):
            # most recent action
            act = self.car_last_act[i] / (self.n_actions - 1)

            # route id
            r_i = self.route_id[i] / (self.n_paths - 1)

            # loc
            p_norm = p / (h-1, w-1)

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

            if self.vocab_type == 'bool':
                o = tuple((act, r_i, v_sq))
            else:
                o = tuple((act, r_i, p_norm, v_sq))
            obs.append(o)

        obs = tuple(obs)
        return obs


    def _add_cars(self):
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

    def _unittest_path(self,paths):
        for i, p in enumerate(paths[:-1]):
            next_dif = p - np.row_stack([p[1:], p[-1]])
            next_dif = np.abs(next_dif[:-1])
            step_jump = np.sum(next_dif, axis =1)
            if np.any(step_jump != 1):
                return False
            if not np.all(step_jump == 1):
                return False
        return True


    def _take_action(self, idx, act):
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
        if act==0:
            prev = self.car_route_loc[idx]
            self.car_route_loc[idx] += 1
            curr = self.car_route_loc[idx]

            # car/agent has reached end of its path
            if curr == len(self.chosen_path[idx]):
                self.cars_in_sys -= 1
                self.alive_mask[idx] = 0
                self.wait[idx] = 0

                # put it at dead loc
                self.car_loc[idx] = np.zeros(len(self.dims),dtype=int)
                self.is_completed[idx] = 1
                return

            elif curr > len(self.chosen_path[idx]):
                raise RuntimeError("Out of bound car path")

            prev = self.chosen_path[idx][prev]
            curr = self.chosen_path[idx][curr]

            # assert abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) == 1 or curr_path = 0
            self.car_loc[idx] = curr

            # Change last act for color:
            self.car_last_act[idx] = 0



    def _get_reward(self):
        reward = np.full(self.num_agents, self.TIMESTEP_PENALTY) * self.wait

        for i, l in enumerate(self.car_loc):
            if (len(np.where(np.all(self.car_loc[:i] == l,axis=1))[0]) or \
               len(np.where(np.all(self.car_loc[i+1:] == l,axis=1))[0])) and l.any():
               reward[i] += self.CRASH_PENALTY
               self.has_failed = 1

        reward = self.alive_mask * reward
        return reward

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())

    def _choose_dead(self):
        # all idx
        car_idx = np.arange(len(self.alive_mask))
        # random choice of idx from dead ones.
        return np.random.choice(car_idx[self.alive_mask == 0])

    def curriculum(self, epoch):
        step_size = 0.01
        step = (self.add_rate_max - self.add_rate_min) / (self.curr_end - self.curr_start)

        if self.curr_start <= epoch < self.curr_end:
            self.exact_rate = self.exact_rate + step
            self.add_rate = step_size * (self.exact_rate // step_size)
