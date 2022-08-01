####
# Contains functions to load offline datasets and preprocessing
#
####
import numpy as np

def load_train_data(path, env_name, repeat_index, preprocess=True):
    ''' Loads training data for the specified environment
    path: String that indicates the folder with data for all the environments
    env_name: Name of environment to run '''
    if env_name.lower() == 'cartpole':
        data = load_cartpole_train_data(path + f"cartpoledata/Cartpole_traindata_{repeat_index}.npy", preprocess)
    else:
        raise AssertionError("No matching env data")
    return data

def load_test_states(path, env_name, preprocess=True):
    if env_name.lower() == 'cartpole':
        states = load_cartpole_test_states(path + f"cartpoledata/Cartpole_test_states_0.npy", preprocess)
    else:
        raise AssertionError("No matching env data")
    return states

def load_test_values(path, env_name):
    if env_name.lower() == 'cartpole':
        values = load_cartpole_test_values(path + f"cartpoledata/Cartpole_test_values_0.npy")
    else:
        raise AssertionError("No matching env data")
    return values

## CartPole
def load_cartpole_train_data(file_path, preprocess=True):
    data = np.load(file_path, allow_pickle=True)
    if preprocess:
        for i in range(len(data)):
            data[i] = preprocess_cartpole(data[i])
    return data

def load_cartpole_test_states(file_path, preprocess=True):
    states = np.load(file_path, allow_pickle=True)
    if preprocess:
        for i in range(len(states)):
            states[i] = normalize_state_cartpole(states[i])
    return states

def load_cartpole_test_values(file_path):
    return np.load(file_path, allow_pickle=True)


def preprocess_cartpole(transition):
    ''' Preprocesses transitions of the form (state,action, reward, next_state,done)
    Normalize states for cartpole. Rescales the state, so it's roughly between [0,1]
    assumes state is a numpy array
    cart position, cart velocity, pole angle, pole angular velocity '''
    state, action, reward, next_state, done = transition

    return normalize_state_cartpole(state), action, reward, normalize_state_cartpole(next_state), done

def normalize_state_cartpole(state):
    new_state = np.copy(state)
    new_state[0] /= 2.5
    new_state[1] /= 2.5
    new_state[2] /= 0.2
    new_state[3] /= 0.5
    return new_state
