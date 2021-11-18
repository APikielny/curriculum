import sys
import time
import multiprocessing

from agent import QLearningAgent
from svetlik_gridworld import SvetlikGridWorldMDP
from environments import SvetlikGridWorldEnvironments
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import make_interp_spline, BSpline
from visualizer import show_gridworld_q_func

def get_policy_reward(policy, mdp, steps):
    """
    Gets the value of rolling out a deterministic policy
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


def run_single_agent_on_mdp(agent, mdp, episodes, steps, experiment=None, verbose=False, track_disc_reward=False, reset_at_terminal=False, resample_at_terminal=False):
    '''
    Summary:
        Main loop of a single MDP experiment.

    Returns:
        (tuple): (bool:reached terminal, int: num steps taken, list: cumulative discounted reward per episode)
    '''
    if reset_at_terminal and resample_at_terminal:
        raise ValueError("(simple_rl) ExperimentError: Can't have reset_at_terminal and resample_at_terminal set to True.")

    value_per_episode = [0] * episodes
    gamma = mdp.get_gamma()

    off_policy_val_per_episode = [0] * episodes

    # For each episode.
    for episode in range(1, episodes + 1):

        cumulative_episodic_reward = 0

        if verbose:
            # Print episode numbers out nicely.
            sys.stdout.write("\tEpisode %s of %s" % (episode, episodes))
            sys.stdout.write("\b" * len("\tEpisode %s of %s" % (episode, episodes)))
            sys.stdout.flush()

        # Compute initial state/reward.
        state = mdp.get_init_state()
        reward = 0
        episode_start_time = time.perf_counter()

        for step in range(1, steps + 1):

            # step time
            step_start = time.perf_counter()

            # Compute the agent's policy.
            action = agent.act(state, reward)

            # Terminal check.
            if state.is_terminal():

                if verbose:
                    sys.stdout.write("x")

                # if episodes == 1 and not reset_at_terminal and experiment is not None and action != "terminate":
                #     # Self loop if we're not episodic or resetting and in a terminal state.
                #     experiment.add_experience(agent, state, action, 0, state, time_taken=time.perf_counter()-step_start)
                #     continue
                break

            # Execute in MDP.
            reward, next_state = mdp.execute_agent_action(action)

            # Track value.
            value_per_episode[episode - 1] += reward * gamma ** step
            cumulative_episodic_reward += reward

            # Record the experience.
            if experiment is not None:
                reward_to_track = mdp.get_gamma()**(step + 1 + episode*steps) * reward if track_disc_reward else reward
                reward_to_track = round(reward_to_track, 5)

                experiment.add_experience(agent, state, action, reward_to_track, next_state, time_taken=time.perf_counter() - step_start)

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

        # A final update.
        action = agent.act(state, reward)

        # Process experiment info at end of episode.
        if experiment is not None:
            experiment.end_of_episode(agent)

        # Reset the MDP, tell the agent the episode is over.
        mdp.reset()
        agent.end_of_episode()

        # separately, keep track of off-policy reward
        off_policy_val_per_episode[episode - 1] = get_policy_reward(agent.get_max_q_action, mdp, steps)

        if verbose:
            print("\n")

    # Process that learning instance's info at end of learning.
    if experiment is not None:
        experiment.end_of_instance(agent)

    # Only print if our experiment isn't trivially short.
    if steps >= 2000:
        print("\tLast episode reward:", cumulative_episodic_reward)

    return False, steps, value_per_episode, off_policy_val_per_episode

def run_agent_mdp(target_mdp,
                  total_episodes,
                  steps,
                  source_reward_dict,
                  target_reward_dict,
                  index,
                  source_mdp=None,
                  source_episodes=0):
    ql_agent = QLearningAgent(actions=target_mdp.get_actions())
    print(f'running agent {index}')

    # source learning
    source_value_per_episode = [-math.inf for i in range(source_episodes)]
    if source_mdp:
        res_src = run_single_agent_on_mdp(ql_agent, source_mdp, source_episodes, steps, None)
        show_gridworld_q_func(source_mdp, ql_agent, filename="figures/vis_source.png")
        source_value_per_episode = res_src[3]

    # target learning
    res_targ = run_single_agent_on_mdp(ql_agent, target_mdp, total_episodes - source_episodes, steps, None)
    show_gridworld_q_func(target_mdp, ql_agent, filename="figures/vis_target.png")
    target_value_per_episode = res_targ[3]  # off policy

    source_reward_dict[index] = source_value_per_episode
    target_reward_dict[index] = target_value_per_episode

def reward_by_episode(target_mdp,
                      total_episodes,
                      source_mdp=None,
                      source_episodes=0,
                      num_trials=1,
                      max_steps=200):
    """Runs Q learning, reports data on [expected reward of optimal policy] vs. episode"""
    start = time.time()

    jobs = []

    manager = multiprocessing.Manager()
    source_reward_dict = manager.dict()
    target_reward_dict = manager.dict()

    source_reward_at_episode = np.zeros(source_episodes)
    target_reward_at_episode = np.zeros(total_episodes - source_episodes)

    for i in range(num_trials):
        p = multiprocessing.Process(target=run_agent_mdp, name=f'agent{i}', args=(
            target_mdp,
            total_episodes,
            max_steps,
            source_reward_dict,
            target_reward_dict,
            i,
            source_mdp,
            source_episodes))
        jobs.append(p)
        p.start()

    for i in range(len(jobs)):
        jobs[i].join()
        source_reward_at_episode[:len(source_reward_dict[i])] += np.array(source_reward_dict[i]) / num_trials
        target_reward_at_episode[:len(target_reward_dict[i])] += np.array(target_reward_dict[i]) / num_trials

    print(time.time() - start)

    return source_reward_at_episode, target_reward_at_episode

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

def main():
    num_trials = 1
    source_episodes = 400
    total_episodes = 800

    source_mdp = SvetlikGridWorldEnvironments.source_1010()
    target_mdp = SvetlikGridWorldEnvironments.target_1010()

    _, reward_source_source = reward_by_episode(source_mdp, total_episodes, num_trials=num_trials)
    _, reward_target_target = reward_by_episode(target_mdp, total_episodes, num_trials=num_trials)
    reward_transfer_source, reward_transfer_target = reward_by_episode(
        target_mdp, total_episodes, num_trials=num_trials,
        source_mdp=source_mdp, source_episodes=source_episodes)

    x_source = range(source_episodes)
    x_target = range(source_episodes, total_episodes)
    x_total = range(total_episodes)
    y0 = clip_and_smooth(reward_source_source)
    y1 = clip_and_smooth(reward_target_target)
    y2 = clip_and_smooth(reward_transfer_target)
    y3 = clip_and_smooth(reward_transfer_source)

    fig, ax = plt.subplots()
    ax.plot(x_total, y0, color='green', linestyle='--', label="no transfer (source)")
    ax.plot(x_total, y1, color='red', label="no transfer (target)")
    ax.plot(x_target, y2, color='blue', label="transfer (target)")
    ax.plot(x_source, y3, color='blue', linestyle='--', label="transfer (source)")
    ax.legend()
    plt.savefig("figures/reward.png")
    plt.show()

if __name__ == '__main__':
    main()
