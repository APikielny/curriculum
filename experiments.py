import sys
import time
import multiprocessing
from collections import defaultdict
from random import random
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import trange

from agent import QLearningAgent
from pursuit_mdp import PursuitMDP, visualize_pursuit
from svetlik_gridworld import SvetlikGridWorldMDP
from environments import SvetlikGridWorldEnvironments
from visualizer import show_gridworld_q_func

from typing import List

REWARD_SAMPLING_RATE = 5000  # number of steps between each reward sample


def get_policy_reward(policy, mdp, steps, num_trials=1):
    """
    Gets the value of rolling out a policy with epsilon = 0
    """
    gamma = mdp.get_gamma()
    total_reward = 0
    for _ in range(num_trials):
        state = mdp.get_init_state()

        for step in range(steps):
            if state.is_terminal():
                break
            action = policy(state)
            next_state = mdp.transition_func(state, action)
            reward = mdp.reward_func(state, action, next_state)
            total_reward += reward * gamma ** step
            state = next_state

    return total_reward / num_trials


def run_single_agent_on_mdp(
        q_function,
        mdp,
        episodes,
        steps,
        name,
        result_dict,
        reset_at_terminal=False,
        resample_at_terminal=False,
        reward_threshold_termination=math.inf,
        num_eval_trials=1):
    '''
    Summary:
        Main loop of a single MDP experiment.

    Returns:
        (tuple): (bool:reached terminal, int: num steps taken, list: cumulative discounted reward per episode)
    '''
    agent = QLearningAgent(mdp.get_actions(), q_function=q_function)

    if reset_at_terminal and resample_at_terminal:
        raise ValueError("(simple_rl) ExperimentError: Can't have reset_at_terminal and resample_at_terminal set to True.")

    value_per_episode = [0] * episodes
    gamma = mdp.get_gamma()

    total_step_counter = 0
    off_policy_val_per_step = {}

    # For each episode.
    for episode in trange(1, episodes + 1, desc=name):
        cumulative_episodic_reward = 0

        # Compute initial state/reward.
        state = mdp.get_init_state()
        reward = 0
        episode_start_time = time.perf_counter()

        for step in range(1, steps + 1):

            # step time
            step_start = time.perf_counter()

            # Compute the agent's policy.
            action = agent.act(state, reward)

            if state.is_terminal():
                break

            # Execute in MDP.
            reward, next_state = mdp.execute_agent_action(action)

            # Track value.
            value_per_episode[episode - 1] += reward * gamma ** step
            cumulative_episodic_reward += reward

            if next_state.is_terminal():
                if reset_at_terminal:
                    # Reset the MDP.
                    next_state = mdp.get_init_state()
                    mdp.reset()
                elif resample_at_terminal and step < steps:
                    mdp.reset()
                    return True, step, value_per_episode

            # Update pointer.
            state = next_state

            # update reward data
            total_step_counter += 1
            if total_step_counter % REWARD_SAMPLING_RATE == 0:
                rwd = get_policy_reward(agent.get_max_q_action, mdp, steps, num_eval_trials)
                off_policy_val_per_step[total_step_counter] = rwd
                # terminate training if agent is good enough
                if rwd > reward_threshold_termination:
                    print(f"rwd {rwd} above {reward_threshold_termination}")
                    break


        # A final update.
        action = agent.act(state, reward)

        # Reset the MDP, tell the agent the episode is over.
        mdp.reset()
        agent.end_of_episode()

    # Only print if our experiment isn't trivially short.
    if steps >= 2000:
        print("\tLast episode reward:", cumulative_episodic_reward)

    result_dict[name] = {
        'q_function': agent.q_func,
        'total_steps': total_step_counter,
        'val_per_step': off_policy_val_per_step,
    }

    if issubclass(type(mdp), SvetlikGridWorldMDP):
        show_gridworld_q_func(mdp, agent, filename=f'figures/vis_{name}.png')
    elif issubclass(type(mdp), PursuitMDP):
        visualize_pursuit(mdp, lambda s: agent.get_max_q_action(s), filename=f'figures/vis_{name}.mp4')

    return False, steps, value_per_episode, off_policy_val_per_step, total_step_counter


# TODO: this function breaks if one of the val_per_step_dicts has no keys. Don't have time to decipher it right now so I'm getting rid of it. -Jackson
# def average_val_per_step(val_per_step_dicts, total_steps, sampling_rate=50):
#     '''
#     Average the results of multiple trials
#     '''
#
#
#     val_per_step_array = np.zeros((len(val_per_step_dicts), total_steps))
#     for i in range(val_per_step_array.shape[0]):
#         keys = list(val_per_step_dicts[i].keys())
#         keys = [0] + keys + [val_per_step_array.shape[1]]
#         val_per_step_dicts[i][0] = val_per_step_dicts[i][keys[1]]
#         val_per_step_dicts[i][val_per_step_array.shape[1]] = val_per_step_dicts[i][keys[-2]]
#         for j in range(len(keys) - 1):
#             for k in range(keys[j], keys[j + 1]):
#                 frac = (k - keys[j]) / (keys[j + 1] - keys[j])
#                 val_per_step_array[i, k] = val_per_step_dicts[i][keys[j]] * (1 - frac) + val_per_step_dicts[i][keys[j + 1]] * frac
#
#     average_val_per_step_array = np.mean(val_per_step_array, axis=0)
#     average_val_per_step_dict = {}
#     for i in range(0, total_steps, sampling_rate):
#         sample = average_val_per_step_array[
#                  max(0, i - sampling_rate // 2):min(average_val_per_step_array.shape[0] - 1, i + sampling_rate // 2)]
#         average_val_per_step_dict[i] = np.mean(sample)
#     return average_val_per_step_dict


def get_total_source_steps(curriculum, averaged_results, task):
    '''
    Given a curriculum and the averaged results from that curriculum, return the number
    of steps used by all source tasks for a given task
    '''

    if len(curriculum[task]['sources']) == 0:
        return 0

    total = 0
    for source in curriculum[task]['sources']:
        total += get_total_source_steps(curriculum, averaged_results, source) + averaged_results[source]['total_steps']

    return total


def run_agent_curriculum(curriculum,
                         num_trials=1,
                         steps=200,
                         max_num_concurrent_processes=8,
                         num_eval_trials=1):
    '''
    Performs Q learning on a curriculum of MDPs. Curriculum format:
    {
        'task_name': {
            'sources': ['source1_name', 'source2_name']
            'task': mdp_object,
            'episodes': num_episodes,
            'reward_threshold_termination': termination_threshold,
        },
        ...
    }
    '''

    manager = multiprocessing.Manager()
    results = manager.dict()
    complete_jobs = []
    active_jobs = {}
    while len(complete_jobs) < len(curriculum) * num_trials:
        # check if any jobs have finished
        for job_name in [k for k in active_jobs]:
            active_jobs[job_name].join(timeout=0)
            if not active_jobs[job_name].is_alive():
                del active_jobs[job_name]
                complete_jobs.append(job_name)

        # see what new jobs we can start
        for task in curriculum:
            for i in range(num_trials):
                if f'{task}_{i}' in complete_jobs or f'{task}_{i}' in active_jobs:
                    continue
                ready_to_start = True
                source_task_q_functions = []
                source_task_mdps = []
                # check to see if the source task jobs have finished for this task
                for source_task in curriculum[task]['sources']:
                    source_task_job_name = f'{source_task}_{i}'
                    if source_task_job_name not in complete_jobs:
                        ready_to_start = False
                        break
                    # get list of source task q functions to combine as an initialization for the current task
                    source_task_q_functions.append(results[source_task_job_name]['q_function'])
                    source_task_mdps.append(curriculum[task]['task'])

                if ready_to_start and len(active_jobs) < max_num_concurrent_processes:
                    # combine q functions from the source tasks
                    q_function = QLearningAgent.combine_q_functions(
                        source_q_functions=source_task_q_functions,
                        source_mdps=source_task_mdps,
                        target_mdp=curriculum[task]['task'])

                    # start job for current task
                    p = multiprocessing.Process(target=run_single_agent_on_mdp, name=f'{task}_{i}', args=(
                        q_function,
                        curriculum[task]['task'],
                        curriculum[task]['episodes'],
                        steps,
                        f'{task}_{i}',
                        results,
                        False,
                        False,
                        curriculum[task]['reward_threshold_termination'],
                        num_eval_trials
                    ))
                    active_jobs[f'{task}_{i}'] = p
                    p.start()

        # wait a second on each loop so we're not wasting resources waiting for jobs to finish
        time.sleep(1)

    averaged_results = {}

    for task in curriculum:
        val_per_step_dicts = []
        total_steps = 0
        for i in range(num_trials):
            val_per_step_dicts.append(results[f'{task}_{i}']['val_per_step'])
            total_steps = max(total_steps, results[f'{task}_{i}']['total_steps'])
        averaged_results[task] = {
            'total_steps': total_steps,
            'val_per_step': val_per_step_dicts[0]  # average_val_per_step(val_per_step_dicts, total_steps)
        }

    for task in curriculum:
        offset = get_total_source_steps(curriculum, averaged_results, task)
        averaged_results[task]['val_per_step'] = {k + offset: v for k, v in averaged_results[task]['val_per_step'].items()}

    return averaged_results


def clip_and_smooth(y: List[float]) -> List[float]:
    y_clip = np.clip(y, -500, 500)
    y_smooth = y_clip.copy()
    for i in range(len(y)):
        smooth_min = max(0, i - 5)
        smooth_max = min(len(y) - 1, i + 5)
        if smooth_max - smooth_min > 0:
            y_smooth[i] = sum(y_clip[smooth_min:smooth_max]) / (smooth_max - smooth_min)
    return y_smooth


def gap(reward_with_transfer, reward_no_transfer):
    """
    Calculate difference in training steps to learn threshold with and without transfer
    :param reward_with_transfer: A list of reward per step on target with transfer
    :param reward_no_transfer: A list of reward per step on target without transfer
    :return: Difference in training steps to threshold
    """
    threshold = max(reward_no_transfer) / 2
    steps_with_transfer = 0
    for reward in reward_with_transfer:
        if reward >= threshold:
            break
        else:
            steps_with_transfer += 1
    steps_no_transfer = 0
    for reward in reward_no_transfer:
        if reward >= threshold:
            break
        else:
            steps_no_transfer += 1
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
        result = run_agent_curriculum(curriculum1, num_trials=num_trials)
        # get gap to no transfer curriculum

        gaps.append(gap(result, no_transfer_result))
        # add to list or something?

    # plot list of gaps
    # x axis should be size of subgrid


def main_curriculum():
    num_trials = 1

    target_mdp = SvetlikGridWorldEnvironments.target_1010()
    source1_mdp = target_mdp.subgrid((8, 11), (6, 11))
    source2_mdp = target_mdp.subgrid((6, 11), (8, 11))

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
            'sources': []
        },
        'target_transfer': {
            'task': target_mdp,
            'episodes': 800,
            'reward_threshold_termination': math.inf,
            'sources': ['source1_transfer', 'source2_transfer']
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
        x, y = zip(*results1[task]['val_per_step'].items()) if results1[task]['val_per_step'] else ([], [])
        ax.plot(x, clip_and_smooth(y), color=(random(), random(), random()), label=task)

    for task in results2:
        x, y = zip(*results2[task]['val_per_step'].items()) if results2[task]['val_per_step'] else ([], [])
        ax.plot(x, clip_and_smooth(y), color=(random(), random(), random()), label=task)

    ax.legend(loc="upper left")
    plt.savefig("figures/reward.png")
    plt.show()


def pursuit_curriculum():
    num_trials = 1
    num_eval_trials = 10

    target_mdp = PursuitMDP(
        width=3,
        height=3,
        init_predator_locs=[(0, 0)],
        init_prey_loc=(0, 2))

    curriculum = {
        'target_no_transfer': {
            'task': target_mdp,
            'episodes': 1000,
            'reward_threshold_termination': math.inf,
            'sources': []
        }
    }

    results = run_agent_curriculum(curriculum, num_trials=num_trials, num_eval_trials=num_eval_trials)

    _, ax = plt.subplots()

    for task in results:
        x, y = zip(*results[task]['val_per_step'].items())
        ax.plot(x, clip_and_smooth(y), color=(random(), random(), random()), label=task)

    ax.legend(loc="upper left")
    plt.savefig("figures/reward.png")
    plt.show()


if __name__ == '__main__':
    main_curriculum()
