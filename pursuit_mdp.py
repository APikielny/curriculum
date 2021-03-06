#!/usr/bin/env python

# Python imports.
import sys
import math
from typing import Tuple, List, Set, Callable, Iterable
import itertools
import random
import copy

# Other imports.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State

PrimitiveAction = str


class PursuitAction:
    def __init__(self, action: Tuple[str, ...]):
        self.action = action

    @property
    def length(self):
        return len(self.action)

    def as_list(self):
        return list(self.action)


class PursuitState(State):
    def __init__(self, predator_locs: Iterable[Tuple[int, int]], prey_loc: Tuple[int, int]):
        self.predator_locs = set(predator_locs)
        self.prey_loc = prey_loc
        State.__init__(self, data=[self.predator_locs, self.prey_loc])

    def __hash__(self):
        return hash((tuple(self.predator_locs), self.prey_loc))

    def is_terminal(self):
        return self.prey_loc in self.predator_locs

    def __str__(self):
        return f"<pred locs: {self.predator_locs}, prey_loc: {self.prey_loc}>"


class PursuitMDP(MDP):
    PRIMITIVE_ACTIONS = ["up", "down", "left", "right", "stay"]
    PREY_RANDOMNESS = 1.0

    def __init__(self,
                 width: int = 5,
                 height: int = 5,
                 num_predators: int = 1,
                 wall_locs: List[Tuple[int, int]] = [],
                 init_predator_locs: List[Tuple[int, int]] = [(0, 0)],
                 init_prey_loc: Tuple[int, int] = (2, 2),
                 rand_init: bool = False,
                 default_reward: int = -1,
                 gamma=0.99,
                 x_limit: Tuple[int, int] = (-math.inf, math.inf),
                 y_limit: Tuple[int, int] = (-math.inf, math.inf)):
        """
        :param x_limit: range of x values, inclusive -- defaults to (0, width)
        :param y_limit: range of y values, inclusive -- defaults to (0, height)
        """
        self.init_predator_locs = set(init_predator_locs)
        self.init_state = PursuitState(self.init_predator_locs, init_prey_loc)
        self.actions = [PursuitAction(a) for a in list(itertools.product(self.PRIMITIVE_ACTIONS, repeat=num_predators))]
        MDP.__init__(self, self.actions, self._transition_func, self._reward_func,
                     init_state=self.init_state, gamma=gamma)

        self.width = width
        self.height = height
        self.num_predators = num_predators

        self.wall_locs = wall_locs
        self.init_prey_loc = init_prey_loc
        self.rand_init = rand_init
        self.default_reward = default_reward
        self.x_limit = max(0, x_limit[0]), min(width - 1, x_limit[1])
        self.y_limit = max(0, y_limit[0]), min(height - 1, y_limit[1])

    @property
    def states(self):
        x_lo, x_hi = self.x_limit[0], self.x_limit[1] + 1
        y_lo, y_hi = self.y_limit[0], self.y_limit[1] + 1
        locs = [(i, j) for i in range(x_lo, x_hi) for j in range(y_lo, y_hi)]
        pred_loc_sets = [set(pred_loc_list) for pred_loc_list in itertools.product(locs, repeat=self.num_predators)
                        if len(set(pred_loc_list)) == self.num_predators]  # no collisions
        states = [PursuitState(pred_loc_set, prey_loc)
                  for pred_loc_set in pred_loc_sets for prey_loc in locs
                  if prey_loc not in pred_loc_set]
        return states

    def _transition_func(self, state: PursuitState, action: PursuitAction) -> PursuitState:
        if state.is_terminal():
            return state

        new_prey_loc = self._apply_action(state.prey_loc, self.prey_policy(state))

        pred_locs_list = sorted(list(state.predator_locs))
        if not len(pred_locs_list) == self.num_predators:
            raise RuntimeError("Predators list length does not match number of predators")
        if not action.length == self.num_predators:
            raise RuntimeError("Action length does not match number of predators")
        new_pred_locs = set(
            self._apply_action(loc, primitive_action)
            for loc, primitive_action in zip(state.predator_locs, action.as_list()))
        if len(new_pred_locs) < self.num_predators:
            # collision, revert to old positions (no one moves)
            new_pred_locs = state.predator_locs

        return PursuitState(predator_locs=new_pred_locs, prey_loc=new_prey_loc)

    def _apply_action(self, loc: Tuple[int, int], action: PrimitiveAction) -> Tuple[int, int]:
        """Move in specified direction, as long as wall is not there, and clamp"""
        # clamp state to limits (TODO: not actually what we want to do about agents leaving a subgrid???)

        def in_range(limit, val):
            return limit[0] <= val <= limit[1]
        last_x, last_y = loc

        if action == "up":
            new_x, new_y = (last_x, last_y + 1)
        elif action == "down":
            new_x, new_y = (last_x, last_y - 1)
        elif action == "right":
            new_x, new_y = (last_x + 1, last_y)
        elif action == "left":
            new_x, new_y = (last_x - 1, last_y)
        else:
            new_x, new_y = (last_x, last_y)

        if in_range(self.x_limit, new_x) and in_range(self.y_limit, new_y) and (new_x, new_y) not in self.wall_locs:
            return new_x, new_y
        else:
            return last_x, last_y

    def _reward_func(self, state: PursuitState, action: PursuitAction, next_state: PursuitState) -> float:
        return 0 if state.is_terminal() or next_state.is_terminal() else self.default_reward

    def prey_policy(self, state: PursuitState) -> str:
        if random.random() < self.PREY_RANDOMNESS:
            return random.choice(self.PRIMITIVE_ACTIONS)
        else:
            # move away from predators
            def distance(p1: Tuple[int, int], p2: Tuple[int, int]):
                return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

            def min_dist_to_nearest_predator(action: PrimitiveAction):
                """Given an action, finds the minimum distance to nearest predator
                which will result if the prey takes that action"""
                new_prey_loc = self._apply_action(state.prey_loc, action)
                predator_distances = [distance(l, new_prey_loc) for l in state.predator_locs]
                return min(predator_distances)

            _, best_action = max(
                (min_dist_to_nearest_predator(a), a)
                for a in self.PRIMITIVE_ACTIONS)
            return best_action

    def subgrid(self, x_lim, y_lim):
        """Returns a sub-gridworld. Note that
        - width and height are no longer accurate
        - features are NO LONGER GUARANTEED TO BE ON THE GRID!
        - initial location is now random"""
        return PursuitMDP(
            width=self.width,
            height=self.height,
            wall_locs=self.wall_locs,
            rand_init=True,
            default_reward=self.default_reward,
            x_limit=x_lim,
            y_limit=y_lim)

    def visualize(self, agent, filename: str) -> None:
        def max_q_policy(state):
            return agent.get_max_q_action(state)
        visualize_pursuit(copy.deepcopy(self), max_q_policy, filename)

def draw_state(state, mdp, ax):
    board = np.zeros((mdp.height, mdp.width))
    def update(pos, val):
        x, y = pos
        c = x
        r = mdp.height - y - 1
        board[r, c] = val

    for wall_loc in mdp.wall_locs:
        update(wall_loc, 1)

    for predator_loc in state.predator_locs:
        update(predator_loc, 0.2)
    update(state.prey_loc, 0.7)
    ax.imshow(board, interpolation='nearest')
    ax.set(xticks=[], yticks=[])
    ax.axis('image')


def generator(mdp, policy):
    for _ in range(200):
        yield mdp.cur_state
        if mdp.cur_state.is_terminal():
            return
        a = policy(mdp.cur_state)
        r, next_state = mdp.execute_agent_action(a)


def visualize_pursuit(pursuit_mdp: PursuitMDP,
                      policy: Callable[[PursuitState], str],
                      filename: str = None):
    """Visualize an optimal policy in a PursuitMDP environment"""
    fig, ax = plt.subplots()

    # print stuff, because animations aren't working for me
    rollout = list(generator(pursuit_mdp, policy))
    for state in rollout:
        print(f"state {state}")

    # animation = FuncAnimation(fig, draw_state, frames=generator(pursuit_mdp, policy), fargs=(pursuit_mdp, ax),
    #                           interval=500, repeat=False, cache_frame_data=False)
    # if filename:
    #     animation.save(filename)
    # else:
    #     plt.show()
