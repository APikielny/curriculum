import numpy as np
from agent import QLearningAgent
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from svetlik_gridworld import SvetlikGridWorldMDP
from matplotlib import pyplot as plt


def show_gridworld_q_func(gw: SvetlikGridWorldMDP,
                          q_learner: QLearningAgent = None,
                          filename: str = None):
    """Visualize q function + optimal policy in a SvetlikGridWorldMDP environment"""
    q_vals = np.zeros((gw.height + 1, gw.width + 1))
    for x_idx in range(*gw.x_limit):
        for y_idx in range(*gw.y_limit):
            state = GridWorldState(x=x_idx, y=y_idx)
            if q_learner is not None:
                q_vals[x_idx][y_idx] = q_learner.get_max_q_value(state)


    # in_bounds_img = (q_vals[gw.y_limit[0]:gw.y_limit[1]]
    #                  .transpose()[gw.x_limit[0]:gw.x_limit[1]])
    fig, ax = plt.subplots()
    pos = ax.imshow(q_vals.transpose(), cmap="RdYlGn", origin="lower")
    fig.colorbar(pos, ax=ax)

    # draw bounds
    bounds = plt.Rectangle((gw.x_limit[0] - 0.5, gw.y_limit[0] - 0.5),
                              gw.x_limit[1] - gw.x_limit[0],
                              gw.y_limit[1] - gw.y_limit[0],
                              fc=(1, 1, 1, 0),
                              ec=(0, 0, 0))
    ax.add_patch(bounds)

    ax.set_xticks(np.arange(gw.width + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(gw.height + 1) - 0.5, minor=True)
    ax.grid(which="minor")
    ax.tick_params(which="minor", size=0)

    for x_idx in range(*gw.x_limit):
        for y_idx in range(*gw.y_limit):
            state = gw.get_state(x_idx, y_idx)
            x_coord = x_idx # x coord in the image
            y_coord = y_idx # y coord in the image

            # show state features
            if gw.state_is_fire(state):
                plt.text(x_coord + 0.2, y_coord + 0.2, "F")
            elif gw.state_is_pit(state):
                plt.text(x_coord + 0.2, y_coord + 0.2, "P")
            elif gw.state_is_treasure(state):
                plt.text(x_coord + 0.2, y_coord + 0.2, "T")
            else:
                # show coordinates for empty states
                # coords = f"{x_idx+1}, {y_idx+1}"
                # plt.text(x_idx-0.2, y_idx+0.2, f"{coords}")
                pass

            if q_learner is not None:
                # show optimal policy
                best_action = q_learner.get_max_q_action(state)
                arrow_args = { "head_width": 0.1, "head_length": 0.1 }
                if best_action == "left":
                    plt.arrow(x_coord + 0.2, y_coord, -0.2, 0, **arrow_args)
                elif best_action == "right":
                    plt.arrow(x_coord - 0.2, y_coord, 0.2, 0, **arrow_args)
                elif best_action == "up":
                    plt.arrow(x_coord, y_coord - 0.2, 0, 0.2, **arrow_args)
                elif best_action == "down":
                    plt.arrow(x_coord, y_coord + 0.2, 0, -0.2, **arrow_args)

            # show q value text
            # plt.text(x_idx-0.3, y_idx, f"{q_vals[x_idx][y_idx]:.2f}")

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
