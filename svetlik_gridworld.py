#!/usr/bin/env python

# Python imports.
import sys
import math
import random

# Other imports.
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from typing import Tuple, List

class SvetlikGridWorldMDP(GridWorldMDP):
    def __init__(self,
                 fire_locs: List[Tuple[int, int]] = [],
                 pit_locs: List[Tuple[int, int]] = [],
                 treasure_locs: List[Tuple[int, int]] = [(5, 3)],
                 width: int = 5,
                 height: int = 3,
                 init_loc: Tuple[int, int] = (1, 1),
                 rand_init: bool = False,
                 default_reward: int = -1,
                 next_to_fire_reward: int = -250,
                 fire_reward: int = -500,
                 pit_reward: int = -2500,
                 treasure_reward: int = 200,
                 x_limit: Tuple[int, int] = (-math.inf, math.inf),
                 y_limit: Tuple[int, int] = (-math.inf, math.inf)):
        """
        :param x_limit: range of x values, defaults to (1, width + 1). inclusive lower, exclusive upper
        :param y_limit: range of y values, defaults to (1, height + 1). inclusive lower, exclusive upper
        """
        goal_locs = pit_locs + treasure_locs

        # if rand_init:
        #     init_loc = random.choice(range(*x_limit)), random.choice(range(*y_limit))

        GridWorldMDP.__init__(self,
                              width,
                              height,
                              init_loc=init_loc,
                              rand_init=rand_init,
                              goal_locs=goal_locs)
        self.fire_locs = fire_locs
        self.pit_locs = pit_locs
        self.treasure_locs = treasure_locs
        self.default_reward = default_reward
        self.next_to_fire_reward = next_to_fire_reward
        self.fire_reward = fire_reward
        self.pit_reward = pit_reward
        self.treasure_reward = treasure_reward
        self.x_limit = int(max(1, x_limit[0])), int(min(width + 1, x_limit[1]))
        self.y_limit = int(max(1, y_limit[0])), int(min(height + 1, y_limit[1]))

        for x, y in goal_locs:
            self.get_state(x, y).set_terminal(True)

    def get_init_state(self, evaluation=True):
        if self.rand_init and not evaluation:
            self.cur_state = self.get_state(random.choice(range(*self.x_limit)), random.choice(range(*self.y_limit)))
            return self.cur_state
        else:
            return self.get_state(*self.init_loc)

    def subgrid(self, x_lim, y_lim):
        """Returns a sub-gridworld. Note that
        - width and height are no longer accurate
        - features are NO LONGER GUARANTEED TO BE ON THE GRID!
        - initial location is now random"""
        return SvetlikGridWorldMDP(
            fire_locs=self.fire_locs,
            pit_locs=self.pit_locs,
            treasure_locs=self.treasure_locs,
            width=self.width,
            height=self.height,
            rand_init=True,
            default_reward=self.default_reward,
            next_to_fire_reward=self.next_to_fire_reward,
            fire_reward=self.fire_reward,
            pit_reward=self.pit_reward,
            treasure_reward=self.treasure_reward,
            x_limit=x_lim,
            y_limit=y_lim)

    def get_state(self, x, y):
        return GridWorldState(x=x, y=y)

    def state_is_goal(self, state: GridWorldState) -> bool:
        return (state.x, state.y) in self.goal_locs

    def state_is_treasure(self, state: GridWorldState) -> bool:
        return (state.x, state.y) in self.treasure_locs

    def state_is_fire(self, state: GridWorldState) -> bool:
        return (state.x, state.y) in self.fire_locs

    def state_is_pit(self, state: GridWorldState) -> bool:
        return (state.x, state.y) in self.pit_locs

    def state_is_next_to_fire(self, state: GridWorldState) -> bool:
        x, y = state.x, state.y
        return ((x + 1, y) in self.fire_locs
               or (x - 1, y) in self.fire_locs
               or (x, y + 1) in self.fire_locs
               or (x, y - 1) in self.fire_locs)

    def _reward_func(self, state, action, next_state):
        '''
        Args:
            state (State)
            action (str)
            next_state (State)

        Returns
            (float)
        '''

        reward = self.default_reward

        if self.state_is_next_to_fire(next_state):
            reward += self.next_to_fire_reward
        if self.state_is_fire(next_state):
            reward += self.fire_reward
        elif self.state_is_pit(next_state):
            reward += self.pit_reward
        elif self.state_is_treasure(next_state):
            reward += self.treasure_reward

        return reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (ColoredGridWorldState)
            action (str)
        Returns:
            (ColoredGridWorldState)
        '''
        if state.is_terminal():
            return state

        if action == "up" and state.y < self.height:
            next_state = self.get_state(state.x, state.y + 1)
        elif action == "down" and state.y > 1:
            next_state = self.get_state(state.x, state.y - 1)
        elif action == "right" and state.x < self.width:
            next_state = self.get_state(state.x + 1, state.y)
        elif action == "left" and state.x > 1:
            next_state = self.get_state(state.x - 1, state.y)
        else:
            next_state = self.get_state(state.x, state.y)

        # clamp state to limits
        clamped_x = max(self.x_limit[0], min(self.x_limit[1] - 1, next_state.x))
        clamped_y = max(self.y_limit[0], min(self.y_limit[1] - 1, next_state.y))
        clamped_next_state = self.get_state(clamped_x, clamped_y)

        if self.state_is_goal(clamped_next_state):
            clamped_next_state.set_terminal(True)

        return clamped_next_state


def main(open_plot=True):
    # Set up MDP, Agents.
    mdp = SvetlikGridWorldMDP(pit_locs=[(2, 2), (4, 2)], width=10, height=10, treasure_locs=[(10, 10)])
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Run experiment and make plot.
    # run_agents_on_mdp([ql_agent], mdp, instances=10, episodes=500, steps=40, open_plot=open_plot, cumulative_plot=False)


if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")