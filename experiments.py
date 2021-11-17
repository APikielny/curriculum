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

    expected_value_per_step = []

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

            expected_value_per_step.append(get_policy_reward(agent.get_max_q_action, mdp, steps))

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

        if verbose:
            print("\n")

    # Process that learning instance's info at end of learning.
    if experiment is not None:
        experiment.end_of_instance(agent)

    # Only print if our experiment isn't trivially short.
    if steps >= 2000:
        print("\tLast episode reward:", cumulative_episodic_reward)

    return False, steps, value_per_episode, expected_value_per_step

def run_agent_mp(target_mdp,
                 target_episodes,
                 steps,
                 return_dict,
                 index,
                 source_mdp=None,
                 source_episodes=0):
    ql_agent = QLearningAgent(actions=target_mdp.get_actions())
    print(f'running agent {index}')

    # source learning
    if source_mdp:
        run_single_agent_on_mdp(ql_agent, source_mdp, source_episodes, steps, None)
        show_gridworld_q_func(ql_agent, source_mdp, filename="figures/vis_source.png")
    source_value_per_episode = [-math.inf for i in range(source_episodes)]

    # target learning
    res = run_single_agent_on_mdp(ql_agent, target_mdp, target_episodes, steps, None)
    show_gridworld_q_func(ql_agent, target_mdp, filename="figures/vis_target.png")
    target_value_per_episode = res[2]

    return_dict[index] = source_value_per_episode + target_value_per_episode

def reward_by_episode(target_mdp,
                      target_episodes,
                      source_mdp=None,
                      source_episodes=0,
                      num_trials=1,
                      max_steps=200):
    """Runs Q learning, reports data on [expected reward of optimal policy] vs. episode"""
    start = time.time()

    jobs = []

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    reward_at_episode = np.zeros(source_episodes + target_episodes)

    for i in range(num_trials):
        p = multiprocessing.Process(target=run_agent_mp, name=f'agent{i}', args=(
            target_mdp,
            target_episodes,
            max_steps,
            return_dict,
            i,
            source_mdp,
            source_episodes))
        jobs.append(p)
        p.start()

    for i in range(len(jobs)):
        jobs[i].join()
        reward_at_episode[:len(return_dict[i])] += np.array(return_dict[i]) / num_trials

    print(time.time() - start)

    return reward_at_episode

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
    num_trials = 10
    source_episodes = 200
    target_episodes = 500

    source_mdp = SvetlikGridWorldEnvironments.source_77()
    target_mdp = SvetlikGridWorldEnvironments.target_77()

    reward_target = reward_by_episode(target_mdp, source_episodes + target_episodes, num_trials=num_trials)
    reward_transfer = reward_by_episode(target_mdp, target_episodes, num_trials=num_trials,
                                        source_mdp=source_mdp, source_episodes=source_episodes)

    x = range(source_episodes + target_episodes)
    y1 = clip_and_smooth(reward_target)
    y2 = clip_and_smooth(reward_transfer)

    fig, ax = plt.subplots()
    ax.plot(x, y1, color='red', label="no transfer")
    ax.plot(x, y2, color='blue', label="transfer")
    ax.legend()
    plt.savefig("figures/reward.png")
    plt.show()

if __name__ == '__main__':
    main()
