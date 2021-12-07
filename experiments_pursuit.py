from experiments import run_agent_curriculum, clip_and_smooth
from pursuit_mdp import PursuitMDP, PursuitState, PursuitAction
import matplotlib.pyplot as plt
from random import random
import math
from typing import Tuple, Set


def main_pursuit():
    num_trials = 1

    source_mdp = PursuitMDP(
        num_predators=1,
        init_predator_locs=[(0, 1)],
        rand_init=True)

    target_mdp = PursuitMDP(
        num_predators=2,
        init_predator_locs=[(0, 1), (1, 0)],
        collision_avoidance=True)

    def state_action_mapping(target_state: PursuitState,
                             target_action: PursuitAction) -> Set[Tuple[PursuitState, PursuitAction]]:
        return set(zip(
            [PursuitState(predator_locs={loc}, prey_loc=target_state.prey_loc)
             for loc in target_state.predator_locs_in_order],
            [PursuitAction((a,)) for a in target_action.as_list()]))

    curriculum1 = {
        'source_transfer': {
            'task': source_mdp,
            'episodes': 1500,  # 5000
            'reward_threshold_termination': math.inf,
            'sources': []
        },
        'target_transfer': {
            'task': target_mdp,
            'episodes': 2000,  # 8000
            'reward_threshold_termination': math.inf,
            'sources': ['source_transfer']
        }
    }

    curriculum2 = {
        'target_no_transfer': {
            'task': target_mdp,
            'episodes': 5000,
            'reward_threshold_termination': math.inf,
            'sources': []
        }
    }

    results1 = run_agent_curriculum(curriculum1, num_trials=num_trials, state_action_mapping=state_action_mapping)
    results2 = run_agent_curriculum(curriculum2, num_trials=num_trials, state_action_mapping=state_action_mapping)

    _, ax = plt.subplots()

    for task in results1:
        x, y = zip(*results1[task]['val_per_step'].items())
        ax.plot(x, clip_and_smooth(y, window=200), color=(random(), random(), random()), label=task)

    for task in results2:
        x, y = zip(*results2[task]['val_per_step'].items())
        ax.plot(x, clip_and_smooth(y, window=200), color=(random(), random(), random()), label=task)

    ax.legend(loc="upper left")
    plt.savefig("figures/reward_pursuit.png")
    # plt.show()


if __name__ == '__main__':
    main_pursuit()
