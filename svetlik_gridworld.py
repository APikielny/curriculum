#!/usr/bin/env python

# Python imports.
from collections import defaultdict
import sys

# Other imports.
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
# from simple_rl.run_experiments import run_agents_on_mdp 

class SvetlikGridWorldMDP(GridWorldMDP):

    def __init__(self,
                fire_locs=[],
                pit_locs=[],
                treasure_locs=[(5, 3)],
                width=5,
                height=3,
                init_loc=(1, 1),
                empty_reward=-1,
                next_to_fire_reward=-250,
                fire_reward=-500,
                pit_reward=-2500,
                treasure_reward=200):
        '''
        Args:
            fire_locs (list of tuples)
            pit_locs (list of tuples)
            treasure_locs (list of tuples)
            width (int)
            height (int)
            init_loc (tuple)
        '''
        GridWorldMDP.__init__(self,
                              width,
                              height,
                              init_loc=init_loc,
                              goal_locs=pit_locs + treasure_locs)

        self.fire_locs = fire_locs
        self.pit_locs = pit_locs
        self.treasure_locs = treasure_locs
        self.init_state = self.get_state_with_type(init_loc[0], init_loc[1])
        self.empty_reward = empty_reward
        self.next_to_fire_reward = next_to_fire_reward
        self.fire_reward = fire_reward
        self.pit_reward = pit_reward
        self.treasure_reward = treasure_reward

    def get_state_with_type(self, x, y):
        next_to_fire = (x + 1, y) in self.fire_locs \
                        or (x - 1, y) in self.fire_locs \
                        or (x, y + 1) in self.fire_locs \
                        or (x, y - 1) in self.fire_locs
        if ((x, y) in self.fire_locs):
            return SvetlikGridWorldState(x, y, 'fire', next_to_fire)
        if ((x, y) in self.pit_locs):
            return SvetlikGridWorldState(x, y, 'pit', next_to_fire)
        if ((x, y) in self.treasure_locs):
            return SvetlikGridWorldState(x, y, 'treasure', next_to_fire)
        return SvetlikGridWorldState(x, y, 'empty', next_to_fire)

    def _reward_func(self, state, action, next_state):
        '''
        Args:
            state (State)
            action (str)
            next_state (State)

        Returns
            (float)
        '''

        reward = self.next_to_fire_reward if next_state.next_to_fire else 0

        if next_state.type == 'fire':
            reward += self.fire_reward
        elif next_state.type == 'pit':
            reward += self.pit_reward
        elif next_state.type == 'treasure':
            reward += self.treasure_reward
        else:
            reward += self.empty_reward

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
            next_state = self.get_state_with_type(state.x, state.y + 1)
        elif action == "down" and state.y > 1:
            next_state = self.get_state_with_type(state.x, state.y - 1)
        elif action == "right" and state.x < self.width:
            next_state = self.get_state_with_type(state.x + 1, state.y)
        elif action == "left" and state.x > 1:
            next_state = self.get_state_with_type(state.x - 1, state.y)
        else:
            next_state = self.get_state_with_type(state.x, state.y)

        if (next_state.x, next_state.y) in self.goal_locs:
            next_state.set_terminal(True)

        return next_state


class SvetlikGridWorldState(GridWorldState):
    ''' Class for Svetlik Grid World States '''

    def __init__(self, x, y, type='empty', next_to_fire=False):
        GridWorldState.__init__(self, x=x, y=y)
        self.type = type
        self.next_to_fire = next_to_fire


    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + "," + str(self.color) + ")"

    def __eq__(self, other):
        return isinstance(other, SvetlikGridWorldState) and self.x == other.x and self.y == other.y and self.type == other.type

    def __hash__(self):
        return hash((self.x, self.y, self.type))

def main(open_plot=True):
    

    # Setup MDP, Agents.
    mdp = SvetlikGridWorldMDP(pit_locs=[(2, 2), (4, 2)], width=10, height=10, treasure_locs=[(10, 10)])
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Run experiment and make plot.
    # run_agents_on_mdp([ql_agent], mdp, instances=10, episodes=500, steps=40, open_plot=open_plot, cumulative_plot=False)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")