#!/usr/bin/env python

# Python imports.
import sys
import math
from typing import Tuple, List, Set, Optional
import itertools
import random

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

    def is_terminal(self):
        return self.prey_loc in self.predator_locs


class PursuitMDP(MDP):
    PRIMITIVE_ACTIONS = ["up", "down", "left", "right"]
    PREY_RANDOMNESS = 0.2

    def __init__(self,
                 width: int = 10,
                 height: int = 10,
                 num_predators: int = 1,
                 wall_locs: List[Tuple[int, int]] = [],
                 init_predator_locs: List[Tuple[int, int]] = [(10, 10)],
                 init_prey_loc: Tuple[int, int] = (1, 1),
                 rand_init: bool = False,
                 default_reward: int = -1,
                 gamma=0.99,
                 x_limit: Tuple[int, int] = (-math.inf, math.inf),
                 y_limit: Tuple[int, int] = (-math.inf, math.inf)):
        """
        :param x_limit: range of x values, defaults to (1, width + 1). inclusive lower, exclusive upper
        :param y_limit: range of y values, defaults to (1, height + 1). inclusive lower, exclusive upper
        """
        self.init_predator_locs = set(init_predator_locs)
        self.init_state = PursuitState(self.init_predator_locs, init_prey_loc)
        MDP.__init__(self, PursuitMDP.PRIMITIVE_ACTIONS, self._transition_func, self._reward_func,
                     init_state=self.init_state, gamma=gamma)

        self.width = width
        self.height = height
        self.num_predators = num_predators

        self.actions = itertools.product(self.PRIMITIVE_ACTIONS, repeat=num_predators)
        self.wall_locs = wall_locs
        self.init_prey_loc = init_prey_loc
        self.rand_init = rand_init
        self.default_reward = default_reward
        self.x_limit = max(1, x_limit[0]), min(width + 1, x_limit[1])
        self.y_limit = max(1, y_limit[0]), min(height + 1, y_limit[1])

    def _transition_func(self, state: PursuitState, action: str) -> PursuitState:
        if state.is_terminal():
            return state

        new_pred_loc = self._apply_action(state.predator_locs[0], action)
        new_prey_loc = self._apply_action(state.prey_loc, self.prey_policy(state))

        return PursuitState(predator_locs={new_pred_loc}, prey_loc=new_prey_loc)

    def _apply_action(self, loc: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Move in specified direction and clamp"""
        last_x, last_y = loc
        if action == "up" and last_y < self.height:
            new_x, new_y = (last_x, last_y + 1)
        elif action == "down" and last_y > 1:
            new_x, new_y = (last_x, last_y - 1)
        elif action == "right" and last_x < self.width:
            new_x, new_y = (last_x + 1, last_y)
        elif action == "left" and last_x > 1:
            new_x, new_y = (last_x - 1, last_y)
        else:
            new_x, new_y = (last_x, last_y)

        # clamp state to limits (TODO: not actually what we want to do about agents leaving a subgrid???)
        return (
            max(self.x_limit[0], min(self.x_limit[1] - 1, new_x)),
            max(self.y_limit[0], min(self.y_limit[1] - 1, new_y)))

    def _reward_func(self, state: PursuitState, action: str, next_state: PursuitState) -> float:
        if state.is_terminal():
            return 0
        else:
            return self.default_reward

    def prey_policy(self, state: PursuitState) -> str:
        if random.random() < self.PREY_RANDOMNESS:
            return random.choice(self.PRIMITIVE_ACTIONS)
        else:
            # move away from predators
            def distance(p1: Tuple[int, int], p2: Tuple[int, int]):
                return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

            _, best_action = max(
                (distance(choose_from_set(state.predator_locs), self._apply_action(state.prey_loc, a)), a)
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