######
# Generates data for policy evaluation in cartpole
# You can hardcode a behaviour policy
# Collects data transitions (state, action, reward, next_state, done)
#
######





import gym
import numpy as np
import time
import copy
from random import randint
import sys


def process_state(state):
    return np.array([state['x'], state['x_dot'], state['theta'], state['theta_dot']])

def simple_agent(state):
    #
    # action=1 go right, action=0 go left
    pos,vel,theta,ang_momentum = state
    e = np.random.uniform()

    # if (np.abs(pos) > 0.5 and e < 0.8) or e < 0.1: #
    #     action = np.random.choice([0, 1])
    if e < 0.1:
        if vel > 0:
            action = 1
        else:
            action = 0

    else:
        if ang_momentum > 0:
            action = 1
        else:
            action = 0

    # action = env.action_space.sample()
    return action

if len(sys.argv) > 1:
    exp_name = sys.argv[1]
else:
    exp_name = "test"

if len(sys.argv) > 2:
    exp_num = sys.argv[2]  # used for generating test cartpoledata in parallel
else:
    exp_num = 0
rand_seed = int(exp_num) + 3
np.random.seed(rand_seed)
# env = gym.make('CartPole-v0')
# actionset = [0,1]

# cartpoledata = []
# gamma_series = []
# reward = 0.0
# nb_frames = 4999
# num_runs = 20
# for run in range(num_runs):
#     state = env.reset()
#     for frame_no in range(nb_frames):
#         time.sleep(0.01)
#         if np.random.random() < 1:
#             action = simple_agent(state)  # Good Action
#         else:
#             action = actionset[np.random.randint(0, len(actionset))]  # Random Action
#         next_state, reward, done, info = env.step(action)
#         if done:
#             gamma = 0
#             env.reset()
#         else:
#             gamma = 1
#         one_action = (state, action, reward, next_state)
#         state = next_state  # Next State
#         cartpoledata.append(one_action)
#         gamma_series.append(gamma)
#         print("Run:{} Frame: {}".format(run,frame_no))

# np.save('Acrobot_Data', cartpoledata)
# np.save('gamma_series', gamma_series)





if exp_name in ('train'):
    env = gym.make('CartPole-v0')
    env.seed(rand_seed+100)
    actionset = [0,1]
    env.reset()

    ### Generate trajectories
    data_nv = []

    reward = 0.0
    nb_frames = 25001
    num_runs = 10
    for run in range(num_runs):
        prev_state = None

        print("--- Run", run, "---")
        state = env.reset()
        done = False
        one_run_data_nv = []

        action = None

        for frame_i in range(nb_frames):
            # time.sleep(0.1)
            if done:
                state = env.reset()
                # print(frame_no)


            # save a transition
            if prev_state is not None:
                if exp_name == 'train':
                    transition = (prev_state, action, reward, state, done)
                one_run_data_nv.append(transition)

            # update saved states
            prev_state = state

            # move forward one step
            action = simple_agent(state)
            state, reward, done, _ = env.step(action)

            # print("Run:{} Frame: {}".format(run,frame_no))
        np.save('cartpoledata/Cartpole_traindata_{}'.format(run), np.array(one_run_data_nv, dtype=object))



# state_series = []
# value_series = []
# reward = 0.0
# nb_frames = 1000
# nb_frames_sample = 100
#
# np.random.seed(1)
# game = gym.make('CartPole-v0')
# actionset = [0,1]
# start_time = time.perf_counter()
#
# for i in range(500):  # #of cartpoledata points
#     state = game.reset()
#     for frame_no in range(randint(200, 500)):  # randomly sampling cartpoledata
#         action = simple_agent(state)  # Good Action
#         state, reward, done, _ = game.step(action)
#         if done:
#             game.reset()
#         next_state = state # Non-Visual State
#
#     sample_value = 0
#     count = 0
#     values = []
#
#     saved_state = game.unwrapped.state
#     state = saved_state
#
#     while True:
#         # game.render()
#         # print(state)
#         # if np.random.random() < 0.5:
#         action = simple_agent(state)  # Good Action
#         # else:
#         #     action = actionset[np.random.randint(0, len(actionset))]  # Random Action
#
#         state, reward, done, _ = game.step(action)
#         sample_value += reward
#
#         if done:
#             game.reset()
#             state = saved_state  # reloads from the save point
#             game.unwrapped.state = saved_state
#             # print("reload", count, sample_value)#, p.getGameState())
#             count += 1
#             if count == 499:
#                 # print(p.getGameState())
#                 break
#             values.append(sample_value)
#             sample_value = 0
#
#     print(i, time.perf_counter() - start_time)
#     print(values)
#     print(np.mean(values), "+/-", np.std(values) / np.sqrt(500), 'std', np.std(values))
#     state_series.append(saved_state)
#     value_series.append(np.mean(values))
#
# np.save('states3.npy', state_series)
# np.save('values3.npy', value_series)
#
#

if exp_name == 'test':
    ### Generate true state values for datagen
    state_series_nv = []
    value_series = []
    std_error_series = []

    nb_frames_burn_in = 1000
    nb_states_sample = 300  # run the program 5 times to get 500 states
    nb_rollouts = 2000

    game = gym.make('CartPole-v0')
    game.seed(rand_seed+1)
    
    actionset = [0, 1]
    start_time = time.perf_counter()
    state = game.reset()
    reward = 0.0
    sample_count = 0

    for frame_i in range(nb_frames_burn_in):  # to get some randomness in starting state
        action = simple_agent(state)
        state, reward, done, _ = game.step(action)
        if done:
             state = game.reset()

    while(sample_count < nb_states_sample):  # #of states sampled
        for frame_i in range(randint(300, 500)):  # take a random number of steps before picking one
            action = simple_agent(state)
            state, reward, done, _ = game.step(action)
            if done:
                state = game.reset()

        sample_count += 1

        sample_value = 0
        count = 0
        values = []

        # save state
        saved_state = game.unwrapped.state
        state = saved_state


        while True:
            # time.sleep(0.02)

            if done:
                game.reset()
                state = saved_state  # reloads from the save point
                game.unwrapped.state = saved_state

                # print("reload", count, sample_value)#, p.getGameState())
                count += 1
                if count == nb_rollouts:
                    state = saved_state  # reload once more to continue sampling states
                    game.unwrapped.state = saved_state
                    break
                values.append(sample_value)
                sample_value = 0

            action = simple_agent(state)

            state, reward, done, _ = game.step(action)
            sample_value += reward

            # print(sample_value, end=" ")

        print("Count", sample_count, time.perf_counter() - start_time)
        # print(values)
        print(np.mean(values), "+/-", np.std(values), np.std(values)/ np.sqrt(nb_rollouts))
        sys.stdout.flush()
        state_series_nv.append(saved_state)
        value_series.append(np.mean(values))
        std_error_series.append(np.std(values) / np.sqrt(nb_rollouts))
        # value_series.append(value / 500)

    print("Saving values")
    # test_data = list(zip(state_series_nv, value_series, std_error_series))
    # np.save('cartpoledata/Cartpole_test_data_{}'.format(exp_num), test_data)
    #
    np.save('cartpoledata/Cartpole_test_states_{}'.format(exp_num), state_series_nv)
    np.save('cartpoledata/Cartpole_test_values_{}'.format(exp_num), value_series)
    np.save('cartpoledata/Cartpole_test_values_std_error_{}'.format(exp_num), std_error_series)


if exp_name == 'merge':
    # Merges testing cartpoledata together into one file
    assert exp_num != 0
    test_states_nv = []
    test_values = []
    test_values_std = []
    for i in range(1, exp_num+1):
        test_states_nv.append(np.load('Cartpole_test_states_' + str(i)))
        test_values.append(np.load('Cartpole_test_values_' + str(i)))
        test_values_std.append(np.load('Cartpole_test_values_std_error_' + str(i)))

    test_states_nv = np.concatenate(test_states_nv)
    test_values = np.concatenate(test_values)
    test_values_std = np.concatenate(test_values_std)

    np.save('Cartpole_test_states', test_states_nv)
    np.save('Cartpole_test_values', test_values)
    np.save('Cartpole_test_values_std_error', test_values_std)


