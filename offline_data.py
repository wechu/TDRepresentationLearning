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
        file_path = path + f"cartpoledata/cartpole_traindata_{repeat_index}.npy"
        preprocess_fn = preprocess_cartpole
    elif env_name.lower() == 'mountaincar':
        file_path = path + f"mountaincardata/mountaincar_traindata_{repeat_index}.npy"
        preprocess_fn = preprocess_mountaincar
    elif env_name.lower() == 'sparse_mountaincar':
        file_path = path + f"mountaincardata/mountaincar_traindata_{repeat_index}.npy"
        preprocess_fn = preprocess_sparse_mountaincar
    elif env_name.lower() == 'wall_gridworld':
        file_path = path + f"gridworlddata/wallgridworld_traindata_{repeat_index}.npy"
        preprocess_fn = None
    else:
        raise AssertionError("No matching env data")

    data = np.load(file_path, allow_pickle=True)
    if preprocess and preprocess_fn is not None:
        for i in range(len(data)):
            data[i] = preprocess_fn(data[i])
    return data

def load_test_states(path, env_name, preprocess=True):
    if env_name.lower() == 'cartpole':
        file_path = path + f"cartpoledata/cartpole_test_states_0.npy"
        preprocess_state_fn = normalize_state_cartpole
    elif env_name.lower() == 'mountaincar':
        file_path = path + f"mountaincardata/mountaincar_test_states_0.npy"
        preprocess_state_fn = normalize_state_mountaincar
    elif env_name.lower() == 'sparse_mountaincar':
        file_path = path + f"mountaincardata/sparse_mountaincar_test_states_0.npy"
        preprocess_state_fn = normalize_state_mountaincar
    elif env_name.lower() == 'wall_gridworld':
        file_path = path + f"gridworlddata/wallgridworld_test_states.npy"
        preprocess_state_fn = None
    else:
        raise AssertionError("No matching env data")

    states = np.load(file_path, allow_pickle=True)
    if preprocess and preprocess_state_fn is not None:
        for i in range(len(states)):
            states[i] = preprocess_state_fn(states[i])

    return states

def load_test_values(path, env_name):
    if env_name.lower() == 'cartpole':
        file_path = path + f"cartpoledata/cartpole_test_values_0.npy"
    elif env_name.lower() == 'mountaincar':
        file_path = path + f"mountaincardata/mountaincar_test_values_0.npy"
    elif env_name.lower() == 'sparse_mountaincar':
        file_path = path + f"mountaincardata/sparse_mountaincar_test_values_0.npy"
    elif env_name.lower() == 'wall_gridworld':
        file_path = path + f"gridworlddata/wallgridworld_test_values.npy"
    else:
        raise AssertionError("No matching env data")
    values = np.load(file_path, allow_pickle=True)
    return values

#### Classic control envs

## CartPole
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

## Mountain Car
def preprocess_mountaincar(transition):
    ''' Preprocesses transitions of the form (state,action, reward, next_state,done)
    Normalize states for mountaincar. Rescales the state, so it's roughly between [0,1]
    assumes state is a numpy array
    position, velocity '''
    state, action, reward, next_state, done = transition

    return normalize_state_mountaincar(state), action, reward, normalize_state_mountaincar(next_state), done

def preprocess_sparse_mountaincar(transition):
    ''' Same as preprocess_mountaincar.
    Additionally, it transforms the rewards so that they become sparse: 1 at the goal and 0 elsewhere '''

    state, action, reward, next_state, done = transition
    if done:
        reward = 1
    else:
        reward = 0

    return normalize_state_mountaincar(state), action, reward, normalize_state_mountaincar(next_state), done

def normalize_state_mountaincar(state):
    new_state = np.copy(state)
    new_state[0] /= 1.2
    new_state[1] /= 0.07
    return new_state


if __name__ == '__main__':
    ## do linear regression and save the weights
    env = "mountaincar"
    test_states = load_test_states("data_generation/", env)
    test_values = load_test_values("data_generation/", env)
    new_test_states = np.append(test_states, np.ones((test_states.shape[0], 1), dtype=test_states.dtype), axis=1)
    w, _, _, _ = np.linalg.lstsq(new_test_states, test_values)
    print(new_test_states[0:10])
    print(w)
    np.save(f"linreg_{env}.npy", w)
