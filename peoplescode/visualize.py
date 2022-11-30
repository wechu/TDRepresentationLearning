from MinAtar.examples.dqn import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle


path = "MinAtar/saves/20221014/"
game = "asterix"  # breakout, asterix, space_invaders, seaquest, freeway
load_path = path + game + "_checkpoint_{i}.pyt"

env = Environment(game)
in_channels = env.state_shape()[2]
num_actions = env.num_actions()
device = 'cpu'

all_idxs = np.arange(0, 2000) * 1000
all_idxs = all_idxs[all_idxs >= 5000]

test_states = torch.load(path + game + "_t1500k_test_states.pyt", map_location=torch.device('cpu'))


def get_test_features(test_states, agent_network):
    output, reps = agent_network.forward(test_states, output_rep=True)
    layer_1, layer_2 = reps
    layer_1.detach().numpy()
    return layer_1.detach().numpy(), layer_2.detach().numpy()


def plot_test_feature_matrix(iteration):
    ''' Plots the feature matrix for the given update iteration.
    Assumes we have them saved.'''

    policy_net = QNetwork(in_channels, num_actions).to(device)
    checkpoint = torch.load(load_path.format(i=iteration), map_location=torch.device('cpu'))
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    layer_1, layer_2 = get_test_features(test_states, policy_net)
    activations_1 = (layer_1 > 0).astype('float')
    activations_2 = (layer_2 > 0).astype('float')

    sort_1_idx = np.lexsort(activations_1)
    sort_2_idx = np.lexsort(activations_2)

    f, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(activations_1[:, sort_1_idx].T, aspect='auto', cmap=plt.cm.gray, interpolation='none')
    axes[1].imshow(activations_2[:, sort_2_idx].T, aspect='auto', cmap=plt.cm.gray, interpolation='none')

    axes[0].set_title(f'Layer 1')
    axes[1].set_title(f'Layer 2')

    f.suptitle(f'{game} (iter {iteration})')
    f.tight_layout()
    # plt.title(f"iteration {iteration}")

def get_all_raw_image_arrays(idxs, save_and_load=True):
    # returns a list of all the activations over time
    filename = path + game + "_activation_arrays.pkl"
    load_path = path + game + "_checkpoint_{i}.pyt"

    if save_and_load and os.path.isfile(filename):  # load if previously saved
        print('loading activation arrays from file')
        with open(filename, 'rb') as file:
            activation_arrays = pickle.load(file)
        return activation_arrays

    all_image_1_arrays = []
    all_image_2_arrays = []
    for iteration in idxs:
        if iteration % 100000 == 0:
            print(f'iter {iteration}')
        policy_net = QNetwork(in_channels, num_actions).to(device)
        checkpoint = torch.load(load_path.format(i=iteration), map_location=torch.device('cpu'))
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

        layer_1, layer_2 = get_test_features(test_states, policy_net)

        all_image_1_arrays.append(layer_1)
        all_image_2_arrays.append(layer_2)

    if save_and_load:
        print(f'saving to {filename}')
        with open(filename, 'wb') as file:
            pickle.dump([all_image_1_arrays, all_image_2_arrays], file, protocol=pickle.HIGHEST_PROTOCOL)

    return all_image_1_arrays, all_image_2_arrays

def preprocess_activation_arrays(activation_arrays):
    # input is the activation arrays for one run and one layer
    # Makes the activations binary: active or not active
    # Sorts the features lexicographically
    # Transposes the activation matrix
    binary = False
    sort = False

    # try using a fixed reference point to sort
    sorting_array = activation_arrays[10]  # todo change this

    new_arrays = []
    for activations in activation_arrays:

        binary_activations = (activations > 0).astype('float')
        if sort:
            sort_idx = np.lexsort(binary_activations)
        elif sorting_array is not None:
            sort_idx = np.lexsort((sorting_array>0).astype('float'))
        else: # no sort
            sort_idx = np.arange(binary_activations.shape[1])

        if binary:
            image = binary_activations[:, sort_idx].T
        else:
            image = activations[:, sort_idx].T

        new_arrays.append(image)

    return np.array(new_arrays)

#

def _find_idxs(x, y):
    # returns the indices of each element of y in x
    # assumes that each element of y is indeed in x
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], y)
    indices = xsorted[ypos]
    return indices

all_image_1_arrays, all_image_2_arrays = get_all_raw_image_arrays(all_idxs)


all_image_1_arrays = np.array(all_image_1_arrays)
all_image_2_arrays = np.array(all_image_2_arrays)

idxs = np.arange(0, 100) * 1000
idxs = idxs[idxs >= 5000]
#
# idxs = np.arange(0, 100) * 10000
# idxs = idxs[idxs >= 5000]

temp_idxs = _find_idxs(all_idxs, idxs)
image_1_arrays = all_image_1_arrays[temp_idxs]
image_2_arrays = all_image_2_arrays[temp_idxs]

image_1_arrays = preprocess_activation_arrays(image_1_arrays)
image_2_arrays = preprocess_activation_arrays(image_2_arrays)

from matplotlib import animation
# Make animation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

im1 = axes[0].imshow(image_1_arrays[0], interpolation='none', aspect='auto', cmap=plt.cm.gray, vmin=0, vmax=0.5)
im2 = axes[1].imshow(image_2_arrays[0], interpolation='none', aspect='auto', cmap=plt.cm.gray, vmin=0, vmax=0.5)
axes[0].set_title(f'Layer 1 (iter 0)')
axes[1].set_title(f'Layer 1 (iter 0)')
fig.suptitle(f'{game}')
fig.tight_layout()


def init():
    im1.set_data(image_1_arrays[0])
    im2.set_data(image_2_arrays[0])
    fig.suptitle(f'{game}')
    return im1, im2


def update(i):
    im1.set_data(image_1_arrays[i])
    im2.set_data(image_2_arrays[i])
    prop1 = np.round(np.mean(image_1_arrays[i]), 3)
    prop2 = np.round(np.mean(image_2_arrays[i]), 3)

    axes[0].set_title(f'Layer 1 (iter {idxs[i]}), {prop1}')
    axes[1].set_title(f'Layer 2 (iter {idxs[i]}), {prop2}')

    # fig.suptitle(f'{game}')
    return im1, im2


gif_path = 'figs/20221014/'
anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(image_1_arrays), interval=200, blit=False)
# print('saving')
anim.save(gif_path + game + '_activations.gif')
# print('done')

import pygifsicle
pygifsicle.optimize(gif_path + game + "_activations.gif", gif_path + game + "_activations.gif")

plt.show()

