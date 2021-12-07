
from experiments import run_agent_curriculum, clip_and_smooth
from environments import SvetlikGridWorldEnvironments
import matplotlib.pyplot as plt
from random import random
import math

def gap(reward_with_transfer_dict, reward_no_transfer_dict):
    """
    Calculate difference in training steps to learn threshold with and without transfer
    :param reward_with_transfer: A list of reward per step on target with transfer
    :param reward_no_transfer: A list of reward per step on target without transfer
    :return: Difference in training steps to threshold
    """

    for task in reward_with_transfer_dict:
        # print("transfer task", task)
        if task == "target_transfer":
            x, reward_with_transfer = zip(*reward_with_transfer_dict[task]['val_per_step'].items())
            # print('x', x, 'y', y)

    for task in reward_no_transfer_dict:
        # print("no transfer task", task)
        if task == "target_no_transfer":
            x, reward_no_transfer = zip(*reward_no_transfer_dict[task]['val_per_step'].items())
            # print('x', x, 'y', y)
    # print('done iterating')

    threshold = (max(reward_no_transfer) + min(reward_no_transfer)) / 2

    steps_with_transfer = 0
    for i in range(0, len(reward_with_transfer), 10):
        if sum(reward_with_transfer[i:i + 10]) / 10 >= threshold:
            break
        else:
            steps_with_transfer += 10

    steps_no_transfer = 0
    for i in range(0, len(reward_no_transfer), 10):
        if sum(reward_no_transfer[i:i + 10]) / 10 >= threshold:
            break
        else:
            steps_no_transfer += 10

    return steps_no_transfer - steps_with_transfer


# experiment to measure how size of source grid effects "gap" ie how far is transferred learning vs. no transfer
# target is always the same
def gap_by_src_grid_size():
    num_trials = 1
    target_mdp = SvetlikGridWorldEnvironments.target_1010()

    curriculum_no_transfer = {
        'target_no_transfer': {
            'task': target_mdp,
            'episodes': 800,
            'reward_threshold_termination': math.inf,
            'sources': []
        }
    }
    no_transfer_result = run_agent_curriculum(curriculum_no_transfer, num_trials=num_trials)

    grid_min = 1
    grid_max = 11
    dims = [dim for dim in range(grid_min, grid_max)]  # for plot
    gaps = []
    for dim in range(grid_min, grid_max):
        source_mdp = target_mdp.subgrid((dim, 11), (dim, 11))
        curriculum1 = {
            'source_transfer': {
                'task': source_mdp,
                'episodes': 250,
                'reward_threshold_termination': math.inf,
                'sources': []
            },
            'target_transfer': {
                'task': target_mdp,
                'episodes': 800,
                'reward_threshold_termination': math.inf,
                'sources': ['source_transfer']
            }
        }
        print("Getting results for source grid size: ", dim, 11)
        result = run_agent_curriculum(curriculum1, num_trials=num_trials)
        # get gap to no transfer curriculum

        gaps.append(gap(result, no_transfer_result))

    # plot list of gaps
    # x axis should be size of subgrid
    plt.plot(dims, gaps)
    plt.show()


def main_gridworld():
    num_trials = 1

    target_mdp = SvetlikGridWorldEnvironments.target_1010()
    source1_mdp = target_mdp.subgrid((6, 11), (6, 11))
    source2_mdp = target_mdp.subgrid((4, 11), (4, 11))

    curriculum1 = {
        'source1_transfer': {
            'task': source1_mdp,
            'episodes': 250,
            'reward_threshold_termination': math.inf,
            'sources': []
        },
        'source2_transfer': {
            'task': source2_mdp,
            'episodes': 250,
            'reward_threshold_termination': math.inf,
            'sources': ['source1_transfer']
        },
        'target_transfer': {
            'task': target_mdp,
            'episodes': 800,
            'reward_threshold_termination': math.inf,
            'sources': ['source2_transfer']
        }
    }

    curriculum2 = {
        'target_no_transfer': {
            'task': target_mdp,
            'episodes': 800,
            'reward_threshold_termination': math.inf,
            'sources': []
        }
    }

    results1 = run_agent_curriculum(curriculum1, num_trials=num_trials)
    results2 = run_agent_curriculum(curriculum2, num_trials=num_trials)

    _, ax = plt.subplots()

    for task in results1:
        x, y = zip(*results1[task]['val_per_step'].items())
        ax.plot(x, clip_and_smooth(y), color=(random(), random(), random()), label=task)

    for task in results2:
        x, y = zip(*results2[task]['val_per_step'].items())
        ax.plot(x, clip_and_smooth(y), color=(random(), random(), random()), label=task)

    ax.legend(loc="upper left")
    plt.savefig("figures/reward_gridworld.png")
    plt.show()



if __name__ == '__main__':
    # main_gridworld()
    main_gridworld()
