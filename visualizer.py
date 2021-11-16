import numpy as np
from agent import QLearningAgent
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from svetlik_gridworld import SvetlikGridWorldMDP
from matplotlib import pyplot as plt

def show_gridworld_q_func(grid_height: int,
                          grid_width: int,
                          q_learner: QLearningAgent,
                          gw: SvetlikGridWorldMDP):
    grid = np.zeros((grid_height, grid_width))
    for i in range(grid_height):
        for j in range(grid_width):
            grid[i][j] = q_learner.get_max_q_value(GridWorldState(x=i+1, y=j+1))

    data = np.ma.array(grid.reshape((grid_height, grid_width)), mask=grid == 0)
    fig, ax = plt.subplots()
    pos = ax.imshow(data, cmap="Greens", origin="lower", vmin=0)
    fig.colorbar(pos, ax=ax)

    ax.set_xticks(np.arange(grid_width + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_height + 1) - 0.5, minor=True)
    ax.grid(which="minor")
    ax.tick_params(which="minor", size=0)

    for i in range(grid_height):
        for j in range(grid_width):
            state = gw.get_state(i+1,j+1)
            if gw.state_is_fire(state):
                plt.text(i, j, "F")
            elif gw.state_is_pit(state):
                plt.text(i, j, "P")
            elif gw.state_is_treasure(state):
                plt.text(i, j, "T")

            # plt.text(i, j, q_learner.get_max_q_action(GridWorldState(x=i+1, y=j+1)))

    plt.show()
