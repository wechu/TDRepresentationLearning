import numpy as np
import Agents
from sweeper import Sweeper
from training_engine import ConfigDictConverter, test_policy_evaluation_error
import pickle

######## Try loading
# with open("test_save_0.pkl", 'rb') as file:
#     results = pickle.load(file)
#
# print(results['test_error'])
#
# quit()
########## Try checking test errors
i_run = 0
sweeper = Sweeper('config/config.json')

# do some training here
config_dict = sweeper.parse(i_run)
cfg = ConfigDictConverter(config_dict)

agent = Agents.OfflinePolicyEvalTDAgent(**cfg.config_dict)
# load
agent.load_model('test_save_0_model_199.pyt')

path = 'data_generation/cartpoledata/'
test_states = np.load(path + 'Cartpole_test_states_0.npy')
test_values = np.load(path + 'Cartpole_test_values_0.npy')


train_data = np.load(path+"Cartpole_traindata_0.npy", allow_pickle=True)
# print(type(train_data))

np.set_printoptions(precision=3, linewidth=300, suppress=True)
# print(test_states)
#
# print(np.max(test_states, axis=0))
# print(np.min(test_states, axis=0))
# test_states = np.array([offline_data.normalize_state_cartpole(state) for state in test_states])
# for i in range(len(test_states)):
#     features = agent.get_features(test_states[i], add_bias=True)
#     print(len(features),features)
#     print(agent.get_prediction(test_states[i]), test_values[i])


# print(test_policy_evaluation_error(agent, test_states, test_values))

## Check error for linear regression

####### Check
all_features = agent.get_features(test_states, add_bias=True)

# print(agent.net.parameters().__next__())
# for x in agent.net.parameters():
#     print(agent.net.parameters().__next__()[1])

param, squared_error, rank, _ = np.linalg.lstsq(all_features, test_values, rcond=None)

print(len(test_values), all_features.shape)
print(agent.get_prediction(test_states[0]), test_values[0])
# print(test_states[0])
# print(agent.get_features(test_states[0]))

# print("feats", all_features[0])
# print(param)
print("rank", rank)

print("test error", np.sqrt(np.mean( (np.dot(all_features,param) - test_values)**2)))

# all_features = agent.get_features(test_states, add_bias=True)
# print("feats", all_features[0])

#
# print('-----')
#
# sweeper = Sweeper('config/config.json')
#
# # do some training here
# config_dict = sweeper.parse(i_run)
# cfg = ConfigDictConverter(config_dict)
#
# agent = Agents.OfflinePolicyEvalTDAgent(**cfg.config_dict)
# # load
# agent.load_model('test_save_0_model_9999.pyt')
#
# path = 'data_generation/cartpoledata/'
# test_states = np.load(path + 'Cartpole_test_states_0.npy')
# test_values = np.load(path + 'Cartpole_test_values_0.npy')
#
# print(agent.get_features(test_states[0]))
# # print(agent.net.parameters().__next__())


### Check representation alignment as in Ehsan's paper


agent.load_model('test_save_0_model_9999.pyt')
all_features = agent.get_features(test_states, add_bias=True)

scaled_all_features = all_features# / np.max(all_features)

singular_vectors, singular_values, _ = np.linalg.svd(scaled_all_features)

print("singular values", singular_values)

def compute_alignment(singular_vectors, y, singular_values, threshold):
    ''' Computes the alignment for the specified threshold '''
    scaled_singular_values = singular_values / np.max(singular_values)
    threshold_idx = np.searchsorted(-scaled_singular_values, -threshold)  # gives first index where value exceeds threshold
    # print("threshold idx", threshold_idx)
    # print("singular values", singular_values)
    alignments = np.transpose(singular_vectors[:, 0:threshold_idx]).dot(y)
    return np.sum(alignments**2)

alignments = []
thresholds = np.linspace(0, 100, 100)

for x in thresholds:
    alignments.append(compute_alignment(singular_vectors, test_values/100, singular_values, x))

import matplotlib.pyplot as plt
plt.plot(thresholds, alignments)
plt.show()