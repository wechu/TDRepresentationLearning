######
# Generates data for policy evaluation in mountaincar
# You can hardcode a behaviour policy
# Collects data transitions (state, action, reward, next_state, done)
# Can use sparse or dense rewards
# Note: Be careful about the timeout condition. If the episode terminates due to timeout, the agent doesn't know this
# so the env is partially observable. To avoid this issue, make sure the behaviour policy is good enough so the timeout
# is not reached.
######
SPARSE_REWARD = True # if true: 1 at goal, 0 elsewhere. If false, -1 at each timestep.
SHAPED_REWARD = False  # Todo add a shaped reward which varies based on distance to the goal
discount = 1.0
if SPARSE_REWARD:
    discount = 0.99

import gym
import numpy as np
import time
import copy
from random import randint
import sys

# def process_state(state):
#     return np.array([state['x'], state['x_dot']])

def simple_agent(state):
    # action=0 accel left, action=1 no accel, action=2 accel right
    # this policy takes about 160 steps per episode
    pos, vel = state
    e = np.random.uniform()

    # if (np.abs(pos) > 0.5 and e < 0.8) or e < 0.1: #
    #     action = np.random.choice([0, 1])
    if e < 0.1:
        action = np.random.randint(0,3)
    else:
        if vel < 0:
            action = 0
        elif vel > 0:
            action = 2
        else:
            action = 1

    return action

if len(sys.argv) > 1:
    exp_name = sys.argv[1]
else:
    exp_name = "test"

if len(sys.argv) > 2:
    exp_num = sys.argv[2]  # used for generating test mountaincardata in parallel
else:
    exp_num = 0

rand_seed = int(exp_num) + 3
np.random.seed(rand_seed)


def sparse_reward(done):
    if done:
        return 1
    return 0

if exp_name in ('train'):
    env = gym.make('MountainCar-v0')
    env.seed(rand_seed + 100)
    actionset = [0, 1, 2]
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
                print(frame_i)

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
            if SPARSE_REWARD:
                reward = sparse_reward(done)

            # print("Run:{} Frame: {}".format(run,frame_no))
        np.save('mountaincardata/mountaincar_traindata_{}'.format(run), np.array(one_run_data_nv, dtype=object))

if exp_name == 'test':
    ### Estimate true state values using the designated discount factor
    state_series_nv = []
    value_series = []
    std_error_series = []

    nb_frames_burn_in = 1000
    nb_states_sample = 500  # run the program 5 times to get 500 states
    nb_rollouts = 1000

    game = gym.make('MountainCar-v0')
    game.seed(rand_seed + 1)

    actionset = [0, 1]
    start_time = time.perf_counter()
    state = game.reset()
    reward = 0.0
    sample_count = 0

    for frame_i in range(nb_frames_burn_in):  # to get some randomness in starting state
        action = simple_agent(state)
        state, reward, done, _ = game.step(action)
        if SPARSE_REWARD:
            reward = sparse_reward(done)
        if done:
            state = game.reset()

    while (sample_count < nb_states_sample):  # #of states sampled
        for frame_i in range(randint(200, 400)):  # take a random number of steps before picking one
            action = simple_agent(state)
            state, reward, done, _ = game.step(action)
            if SPARSE_REWARD:
                reward = sparse_reward(done)
            if done:
                state = game.reset()

        sample_count += 1

        sample_value = 0
        count = 0
        values = []

        # save state
        saved_state = game.unwrapped.state
        state = saved_state

        num_steps = 0
        while True:
            # time.sleep(0.02)

            if done:
                game.reset()
                state = saved_state  # reloads from the save point
                game.unwrapped.state = saved_state

                count += 1
                if count == nb_rollouts:
                    state = saved_state  # reload once more to continue sampling states
                    game.unwrapped.state = saved_state
                    break
                values.append(sample_value)
                sample_value = 0
                num_steps = 0

            action = simple_agent(state)

            state, reward, done, _ = game.step(action)
            if SPARSE_REWARD:
                reward = sparse_reward(done)
            sample_value += discount**num_steps * reward
            num_steps += 1

            # print(sample_value, end=" ")

        print("Count", sample_count, time.perf_counter() - start_time)
        # print(values)
        print(np.mean(values), "+/-", np.std(values), np.std(values) / np.sqrt(nb_rollouts))
        sys.stdout.flush()
        state_series_nv.append(saved_state)
        value_series.append(np.mean(values))
        std_error_series.append(np.std(values) / np.sqrt(nb_rollouts))

    print("Saving values")
    # test_data = list(zip(state_series_nv, value_series, std_error_series))
    # np.save('mountaincardata/mountaincar_test_data_{}'.format(exp_num), test_data)

    name = "mountaincar"
    if SPARSE_REWARD:
        name = "sparse_mountaincar"
    np.save('mountaincardata/{}_test_states_{}'.format(name, exp_num), state_series_nv)
    np.save('mountaincardata/{}_test_values_{}'.format(name, exp_num), value_series)
    np.save('mountaincardata/{}_test_values_std_error_{}'.format(name, exp_num), std_error_series)


if exp_name == 'merge':
    # Merges testing mountaincardata together into one file
    assert exp_num != 0
    test_states_nv = []
    test_values = []
    test_values_std = []
    name = "mountaincar"
    if SPARSE_REWARD:
        name = "sparse_mountaincar"

    for i in range(1, exp_num + 1):
        test_states_nv.append(np.load(f'{name}_test_states_' + str(i)))
        test_values.append(np.load(f'{name}_test_values_' + str(i)))
        test_values_std.append(np.load(f'{name}_test_values_std_error_' + str(i)))

    test_states_nv = np.concatenate(test_states_nv)
    test_values = np.concatenate(test_values)
    test_values_std = np.concatenate(test_values_std)

    np.save(f'{name}_test_states', test_states_nv)
    np.save(f'{name}_test_values', test_values)
    np.save(f'{name}_test_values_std_error', test_values_std)


