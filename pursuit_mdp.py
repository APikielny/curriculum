#!/usr/bin/env python

# Python imports.
import sys
import math
from typing import Tuple, List, Set, Callable
import itertools
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State


def choose_from_set(s: Set):
    return list(s)[0]


class PursuitState(State):
    def __init__(self, predator_locs: Set[Tuple[int, int]], prey_loc: Tuple[int, int]):
        State.__init__(self, data=[predator_locs, prey_loc])
        self.predator_locs = predator_locs
        self.prey_loc = prey_loc

    def __hash__(self):
        return hash((tuple(self.predator_locs), self.prey_loc))

    def is_terminal(self):
        return self.prey_loc in self.predator_locs


class PursuitMDP(MDP):
    PRIMITIVE_ACTIONS = ["up", "down", "left", "right", "stay"]
    PREY_RANDOMNESS = 1.0

    def __init__(self,
                 width: int = 10,
                 height: int = 10,
                 num_predators: int = 1,
                 wall_locs: List[Tuple[int, int]] = [],
                 init_predator_locs: List[Tuple[int, int]] = [(5, 5)],
                 init_prey_loc: Tuple[int, int] = (3, 3),
                 rand_init: bool = False,
                 default_reward: int = -1,
                 gamma=0.99,
                 x_limit: Tuple[int, int] = (-math.inf, math.inf),
                 y_limit: Tuple[int, int] = (-math.inf, math.inf)):
        """
        :param x_limit: range of x values, defaults to [0, width - 1]
        :param y_limit: range of y values, defaults to [0, height - 1]
        """
        self.init_predator_locs = set(init_predator_locs)
        self.init_state = PursuitState(self.init_predator_locs, init_prey_loc)
        MDP.__init__(self, PursuitMDP.PRIMITIVE_ACTIONS, self._transition_func, self._reward_func,
                     init_state=self.init_state, gamma=gamma)

        self.width = width
        self.height = height
        self.num_predators = num_predators

        # self.actions = itertools.product(self.PRIMITIVE_ACTIONS, repeat=num_predators)
        self.actions = self.PRIMITIVE_ACTIONS
        self.wall_locs = wall_locs
        self.init_prey_loc = init_prey_loc
        self.rand_init = rand_init
        self.default_reward = default_reward

        assert x_limit[0] <= x_limit[1] and y_limit[0] <= y_limit[1]
        self.x_limit = max(0, x_limit[0]), min(width - 1, x_limit[1])
        self.y_limit = max(0, y_limit[0]), min(height - 1, y_limit[1])

    def _transition_func(self, state: PursuitState, action: str) -> PursuitState:
        if state.is_terminal():
            return state

        new_pred_loc = self._apply_action(choose_from_set(state.predator_locs), action)
        new_prey_loc = self._apply_action(state.prey_loc, self.prey_policy(state))

        return PursuitState(predator_locs={new_pred_loc}, prey_loc=new_prey_loc)

    def _apply_action(self, loc: Tuple[int, int], action: str) -> Tuple[int, int]:
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
        return last_x, last_y

    def _reward_func(self, state: PursuitState, action: str, next_state: PursuitState) -> float:
        return 0 if state.is_terminal() else self.default_reward

    def prey_policy(self, state: PursuitState) -> str:
        return "stay"  # always stay
        # if random.random() < self.PREY_RANDOMNESS:
        #     return random.choice(self.PRIMITIVE_ACTIONS)
        # else:
        #     # move away from predators
        #     def distance(p1: Tuple[int, int], p2: Tuple[int, int]):
        #         return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        #
        #     _, best_action = max(
        #         (distance(choose_from_set(state.predator_locs), self._apply_action(state.prey_loc, a)), a)
        #         for a in self.PRIMITIVE_ACTIONS)
        #     return best_action

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

def visualize(state, mdp, ax):
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

    # animation = FuncAnimation(fig, visualize, frames=generator(pursuit_mdp, policy), fargs=(pursuit_mdp, ax),
    #                           interval=500, repeat=False, cache_frame_data=False)
    # if filename:
    #     animation.save(filename)
    # else:
    #     plt.show()

if __name__ == "__main__":
    mdp = PursuitMDP(
        width=5,
        height=5,
        init_predator_locs=[(0, 0)],
        init_prey_loc=(0, 4))

    def policy(_):
        return random.choice(PursuitMDP.PRIMITIVE_ACTIONS)

    visualize_pursuit(mdp, policy)
