from experiments import run_agent_curriculum, clip_and_smooth
from pursuit_mdp import PursuitMDP
import matplotlib.pyplot as plt
from random import random
import math

def main_pursuit():
    num_trials = 1

    target_mdp = PursuitMDP()

    curriculum1 = {
        'target_no_transfer': {
            'task': target_mdp,
            'episodes': 5000,
            'reward_threshold_termination': math.inf,
            'sources': []
        }
    }

    results1 = run_agent_curriculum(curriculum1, num_trials=num_trials)

    _, ax = plt.subplots()

    for task in results1:
        x, y = zip(*results1[task]['val_per_step'].items())
        ax.plot(x, clip_and_smooth(y), color=(random(), random(), random()), label=task)

    ax.legend(loc="upper left")
    plt.savefig("figures/reward_pursuit.png")
    plt.show()


if __name__ == '__main__':
    main_pursuit()