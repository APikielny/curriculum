import numpy as np
from agent import QLearningAgent
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from svetlik_gridworld import SvetlikGridWorldMDP
from matplotlib import pyplot as plt

def show_gridworld_q_func(grid_height: int,
                          grid_width: int,
                          q_learner: QLearningAgent,
                          gw: SvetlikGridWorldMDP):
    """Visualize q function + optimal policy in a SvetlikGridWorldMDP environment"""
    q_vals = np.zeros((grid_height, grid_width))
    for x_idx in range(grid_width):
        for y_idx in range(grid_height):
            state = GridWorldState(x=x_idx+1, y=y_idx+1)
            q_vals[x_idx][y_idx] = q_learner.get_max_q_value(state)

    fig, ax = plt.subplots()
    pos = ax.imshow(q_vals.transpose(), cmap="RdYlGn", origin="lower")
    fig.colorbar(pos, ax=ax)

    ax.set_xticks(np.arange(grid_width + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_height + 1) - 0.5, minor=True)
    ax.grid(which="minor")
    ax.tick_params(which="minor", size=0)

    for x_idx in range(grid_width):
        for y_idx in range(grid_height):
            state = gw.get_state(x_idx+1, y_idx+1)
            best_action = q_learner.get_max_q_action(state)

            # show state features
            if gw.state_is_fire(state):
                plt.text(x_idx+0.2, y_idx+0.2, f"F")
            elif gw.state_is_pit(state):
                plt.text(x_idx+0.2, y_idx+0.2, f"P")
            elif gw.state_is_treasure(state):
                plt.text(x_idx+0.2, y_idx+0.2, f"T")
            else:
                # show coordinates for empty states
                # coords = f"{x_idx+1}, {y_idx+1}"
                # plt.text(x_idx-0.2, y_idx+0.2, f"{coords}")
                pass

            # show optimal policy
            arrow_args = { "head_width": 0.1, "head_length": 0.1 }
            if best_action == "left":
                plt.arrow(x_idx+0.2, y_idx, -0.2, 0, **arrow_args)
            elif best_action == "right":
                plt.arrow(x_idx-0.2, y_idx, 0.2, 0, **arrow_args)
            elif best_action == "up":
                plt.arrow(x_idx, y_idx-0.2, 0, 0.2, **arrow_args)
            elif best_action == "down":
                plt.arrow(x_idx, y_idx+0.2, 0, -0.2, **arrow_args)

            # show q value text
            # plt.text(x_idx-0.3, y_idx, f"{q_vals[x_idx][y_idx]:.2f}")

    plt.show()
