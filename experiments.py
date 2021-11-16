import sys
import time
import multiprocessing

from agent import QLearningAgent
from svetlik_gridworld import SvetlikGridWorldMDP
import matplotlib.pyplot as plt
import numpy as np

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

def run_agent_mp(episodes, steps, return_dict, index):
    mdp = SvetlikGridWorldMDP(pit_locs=[], fire_locs=[], width=5, height=5, treasure_locs=[(5, 5)]) # ~750 steps to converge
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    print(f'running agent {index}')
    res = run_single_agent_on_mdp(ql_agent, mdp, episodes, steps, None)
    # mdp = SvetlikGridWorldMDP(pit_locs=[(2, 2), (4, 2)], fire_locs=[(2, 4), (3, 4)], width=5, height=5, treasure_locs=[(5, 5)]) # ~900 steps to converge
    # res = run_single_agent_on_mdp(ql_agent, mdp, episodes * 5, steps, None)
    return_dict[index] = res[2] # value_per_episode

def main():

    start = time.time()

    jobs = []

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    num_agents = 4
    episodes = 300
    steps = 200
    reward_at_episode = np.zeros(episodes)

    # 1 agent: 19s
    # 5 agents: 27s
    # 10 agents: 44s

    for i in range(num_agents):
        p = multiprocessing.Process(target=run_agent_mp, name=f'agent{i}', args=(
            episodes,
            steps,
            return_dict,
            i))
        jobs.append(p)
        p.start()
    
    for i in range(len(jobs)):
        jobs[i].join()
        reward_at_episode[:len(return_dict[i])] += np.array(return_dict[i]) / num_agents

    print(time.time() - start)
    plt.plot(range(episodes), np.clip(reward_at_episode, -500, 500))
    plt.show()

if __name__ == '__main__':
    main()