from experiments import clip_and_smooth
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    for label, filename in [('target: max', '2trials_max_conjunctive.pkl'),
                            ('target: min', '2trials_min_conjunctive.pkl'),
                            ('target: average', '5trials_average_conjunctive.pkl'),
                            ('baseline', 'baseline.pkl')]:

        with open(filename, 'rb') as f:
            results = pickle.load(f)

        for task in results:
            if task == 'target_transfer' or label == 'baseline':
                rewards = [(x, y) for (x, y) in results[task]['val_per_step'].items() if x < 350000]
                print(len(rewards))
                x, y = zip(*rewards)
                plt.plot(x, clip_and_smooth(y, window=500), label=label)

            elif label == 'target: average':
                rewards = [(x, y) for (x, y) in results[task]['val_per_step'].items() if x < 350000]
                print(len(rewards))
                x, y = zip(*rewards)
                plt.plot(x, clip_and_smooth(y, window=500), label='source')

    plt.legend(loc='upper right')
    plt.xticks([0, 100000, 200000, 300000, 400000])
    plt.xlim([0, 350000])
    plt.xlabel("Number of training steps")
    plt.ylabel("Expected reward")
    plt.savefig("figures/all_reward_pursuit.png")
    plt.close('all')
