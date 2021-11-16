import numpy as np
from agent import QLearningAgent
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from matplotlib import pyplot as plt

def show_gridworld_q_func(grid_height: int,
                          grid_width: int,
                          q_learner: QLearningAgent):
    grid = np.zeros((grid_height, grid_width))
    for i in range(grid_height):
        for j in range(grid_width):
            grid[i][j] = q_learner.get_max_q_value(GridWorldState(x=i+1, y=j+1))

    data = np.ma.array(grid.reshape((grid_height, grid_width)), mask=grid == 0)
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="Greens", origin="lower", vmin=0)

    ax.set_xticks(np.arange(grid_width + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_height + 1) - 0.5, minor=True)
    ax.grid(which="minor")
    ax.tick_params(which="minor", size=0)

    for i in range(grid_height):
        for j in range(grid_width):
            plt.text(i, j, q_learner.get_max_q_action(GridWorldState(x=i+1, y=j+1)))

    plt.show()
