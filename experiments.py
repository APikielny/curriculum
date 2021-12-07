import sys
import time
import multiprocessing
from collections import defaultdict
import numpy as np
import math
from tqdm import trange
from typing import List, Set, Callable

from agent import QLearningAgent
from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State
from svetlik_gridworld import SvetlikGridWorldMDP
from visualizer import show_gridworld_q_func

REWARD_SAMPLING_RATE = 100  # number of steps between each reward sample


def defaultdict_q():
    return defaultdict(float)


def combine_q_functions_mapping(source_q_functions: List[dict],
                                target_mdp: MDP,
                                state_mapping_function: Callable[[State], Set[State]]) -> dict:
    """Combines Q functions via a state-mapping function.
    Simplifying assumptions (for now):
    - just one source q function
    - instead of an action mapping, we just have the identity function"""
    assert len(source_q_functions) == 1
    source_q_function = source_q_functions[0]

    def defaultdict_q():
        return defaultdict(float)

    def f(q_vals: List[float]) -> float:
        return sum(q_vals) / len(q_vals)

    q_func = defaultdict(defaultdict_q)

    target_state_set = target_mdp.states

    for target_state in target_state_set:
        source_states = state_mapping_function(target_state)
        for a in target_mdp.get_actions():
            source_q_vals = [source_q_function[s][a] for s in source_states]
            q_func[target_state][a] = f(source_q_vals)

    return q_func


def combine_q_functions(source_q_functions: List[dict],
                        source_mdps: List[MDP],
                        target_mdp: MDP) -> dict:
    """Simplifying assumptions (for now):
    - just one source q function
    - instead of an action mapping, we just have the identity function"""

    q_func = defaultdict(defaultdict_q)

    if len(source_q_functions) == 1:
        q_func = source_q_functions[0]

    else:
        state_set = set()
        for source_q_function in source_q_functions:
            state_set |= set(source_q_function.keys())

        for state in state_set:
            for action in target_mdp.get_actions():
                count = 0
                for source_q_function in source_q_functions:
                    if source_q_function[state][action] != 0:
                        q_func[state][action] += source_q_function[state][action]
                        count += 1
                if count > 0:
                    q_func[state][action] /= count

    return q_func


def get_policy_reward(policy, mdp, steps):
    """
    Gets the value of rolling out a policy with epsilon = 0
    """
    state = mdp.get_init_state()
    gamma = mdp.get_gamma()
    total_reward = 0

    for step in range(steps):
        if state.is_terminal():
            break
        action = policy(state)
        next_state = mdp.transition_func(state, action)
        reward = mdp.reward_func(state, action, next_state)
        total_reward += reward * gamma ** step
        state = next_state

    return total_reward


def run_single_agent_on_mdp(
        q_function,
        mdp,
        episodes,
        steps,
        name,
        result_dict,
        reset_at_terminal=False,
        resample_at_terminal=False,
        reward_threshold_termination=math.inf):
    '''
    Summary:
        Main loop of a single MDP experiment.

    Returns:
        (tuple): (bool:reached terminal, int: num steps taken, list: cumulative discounted reward per episode)
    '''
    agent = QLearningAgent(mdp.get_actions(), q_function=q_function, decay_q_intialization_steps=0)

    if reset_at_terminal and resample_at_terminal:
        raise ValueError("(simple_rl) ExperimentError: Can't have reset_at_terminal and resample_at_terminal set to True.")

    value_per_episode = [0] * episodes
    gamma = mdp.get_gamma()

    total_step_counter = 0
    off_policy_val_per_step = {}

    # For each episode.
    for episode in trange(1, episodes + 1, desc=name):
        rwd = get_policy_reward(agent.get_max_q_action, mdp, steps)
        if rwd > reward_threshold_termination:
            print(f"rwd {rwd} above {reward_threshold_termination}")
            off_policy_val_per_step[total_step_counter] = rwd
            break

        cumulative_episodic_reward = 0

        # Compute initial state/reward.
        state = mdp.get_init_state(False)
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
                off_policy_val_per_step[total_step_counter] = (
                    get_policy_reward(agent.get_max_q_action, mdp, steps))

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

    return False, steps, value_per_episode, off_policy_val_per_step, total_step_counter

def average_val_per_step(val_per_step_dicts, total_steps, sampling_rate=50):
    '''
    Average the results of multiple trials
    '''

    val_per_step_array = np.zeros((len(val_per_step_dicts), total_steps))
    for i in range(val_per_step_array.shape[0]):
        keys = list(val_per_step_dicts[i].keys())
        keys.sort()
        keys = [0] + keys + [val_per_step_array.shape[1]]
        val_per_step_dicts[i][0] = val_per_step_dicts[i][keys[1]]
        val_per_step_dicts[i][val_per_step_array.shape[1]] = val_per_step_dicts[i][keys[-2]]
        for j in range(len(keys) - 1):
            for k in range(keys[j], keys[j + 1]):
                frac = (k - keys[j]) / (keys[j + 1] - keys[j])
                val_per_step_array[i, k] = val_per_step_dicts[i][keys[j]] * (1 - frac) + val_per_step_dicts[i][keys[j + 1]] * frac
    
    average_val_per_step_array = np.mean(val_per_step_array, axis=0)
    average_val_per_step_dict = {}
    for i in range(0, total_steps, sampling_rate):
        sample = average_val_per_step_array[max(0, i - sampling_rate // 2):min(average_val_per_step_array.shape[0] - 1, i + sampling_rate // 2)]
        average_val_per_step_dict[i] = np.mean(sample)
    return average_val_per_step_dict

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
                         max_num_concurrent_processes=8):
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
                    q_function = combine_q_functions(
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
                        curriculum[task]['reward_threshold_termination']
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
            'val_per_step': average_val_per_step(val_per_step_dicts, total_steps)
        }

    for task in curriculum:
        offset = get_total_source_steps(curriculum, averaged_results, task)
        averaged_results[task]['val_per_step'] = { k + offset: v for k, v in averaged_results[task]['val_per_step'].items()}

    return averaged_results


def run_agent_mdp(target_mdp,
                  total_episodes,
                  steps,
                  source_reward_dict,
                  target_reward_dict,
                  index,
                  source_mdp,
                  source_episodes,
                  source_reward_threshold_termination):
    ql_agent = QLearningAgent(actions=target_mdp.get_actions())
    print(f'running agent {index}')

    source_value_per_step = {}
    target_value_per_step = {}
    total_steps_source = 0

    # source learning
    if source_mdp is not None:
        res_src = run_single_agent_on_mdp(
            None,
            source_mdp,
            source_episodes,
            steps,
            reward_threshold_termination=source_reward_threshold_termination)
        show_gridworld_q_func(source_mdp, ql_agent, filename="figures/vis_source.png")
        source_value_per_step = res_src[3]  # off policy
        total_steps_source = res_src[4]

    # target learning
    res_targ = run_single_agent_on_mdp(
        None,
        target_mdp,
        total_episodes - source_episodes,
        steps)
    show_gridworld_q_func(target_mdp, ql_agent, filename="figures/vis_target.png")
    target_value_per_step = {k + total_steps_source: v for k, v in res_targ[3].items()}

    source_reward_dict[index] = source_value_per_step
    target_reward_dict[index] = target_value_per_step

def reward_by_episode(target_mdp,
                      total_episodes,
                      source_mdp=None,
                      source_episodes=0,
                      source_reward_threshold_termination=math.inf,
                      num_trials=1,
                      max_steps=200):
    """Runs Q learning, reports data on [expected reward of optimal policy] vs. episode"""
    start = time.time()

    jobs = []

    manager = multiprocessing.Manager()
    source_reward_dict = manager.dict()
    target_reward_dict = manager.dict()

    # source_reward_at_episode = np.zeros(source_episodes)
    # target_reward_at_episode = np.zeros(total_episodes - source_episodes)
    source_reward_at_step = defaultdict(float)
    target_reward_at_step = defaultdict(float)

    for i in range(num_trials):
        p = multiprocessing.Process(target=run_agent_mdp, name=f'agent{i}', args=(
            target_mdp,
            total_episodes,
            max_steps,
            source_reward_dict,
            target_reward_dict,
            i,
            source_mdp,
            source_episodes,
            source_reward_threshold_termination))
        jobs.append(p)
        p.start()

    for i in range(len(jobs)):
        jobs[i].join()
        if num_trials > 1:
            raise ValueError("oops, can't average anymore (tell Jackson to fix this)")
        else:
            source_reward_at_step = source_reward_dict[i]
            target_reward_at_step = target_reward_dict[i]

    print(time.time() - start)

    return source_reward_at_step, target_reward_at_step

def clip_and_smooth(reward_data):
    n_episodes = len(reward_data)
    x = np.array(list(range(n_episodes)))
    y = np.clip(reward_data, -500, 500)
    y_smooth = [0 for _ in y]
    for i in range(n_episodes):
        smooth_min = max(0, i - 5)
        smooth_max = min(n_episodes - 1, i + 5)
        y_smooth[i] = sum(y[smooth_min:smooth_max]) / (smooth_max - smooth_min)
    return y_smooth