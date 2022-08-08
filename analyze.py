import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
from sweeper import Sweeper

results_folder = "results/20220206_09953/"

sweeper = Sweeper(results_folder + "test_config.json")

all_results = []
# Load all the parameter combinations for hyperparameter analysis
# note: all_res is a file with repeats merged together
for i_param in range(sweeper.total_combinations):
    with open(f"results/20220206_09953/all_res_{i_param}.pkl", 'rb') as f:
        results = pickle.load(f)
        #
        # for k, v in results.items():
        #     results[k] = np.stack(v)
        all_results.append(results)


for i_param in range(sweeper.total_combinations):
    result_reps = [np.mean(rep) for rep in all_results[i_param]['discounted_returns']]
    print(result_reps)
    all_results[i_param]['avg_return'] = np.mean(result_reps)
# sweeper.config_dict[]
# hyperparams = {"step_size": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def compute_returns(trajectory):
    # This only works if we can guarantee that all episodes end with a done flag (not timing out)
    # computes a list of returns per episode
    # ignores the final episode if it's not complete
    returns = []
    total_reward = 0
    for s, a, r, done in trajectory:
        total_reward += r
        if done:
            returns.append(total_reward)
            total_reward = 0
    return returns



# discounted_returns = n

plt.plot([x['avg_return'] for x in all_results])

good_idxs = [i for i in range(sweeper.total_combinations) if all_results[i]['avg_return'] > 0.25]

for i in good_idxs:
    print(sweeper.parse(i), all_results[i]['avg_return'])

for i in good_idxs:
    plt.plot(all_results[0]["discounted_returns"][i])
plt.show()



