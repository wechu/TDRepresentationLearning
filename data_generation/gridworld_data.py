######
# Generates data for policy evaluation in a gridworld
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
import Envs


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


if exp_name in ('train'):
    env = Envs.WallGridWorldEnv()
    # env.seed(rand_seed + 100)
    actionset = [0, 1]
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
                # print(frame_i)

            # save a transition
            if prev_state is not None:
                if exp_name == 'train':
                    transition = (prev_state, action, reward, state, done)
                one_run_data_nv.append(transition)

            # update saved states
            prev_state = state

            # move forward one step
            # action = env.eps_good_policy(state)
            action = env.cover_policy(state)

            state, reward, done, _ = env.step(action)

            # print("Run:{} Frame: {}".format(run,frame_no))
        np.save('gridworlddata/wallgridworld_traindata_{}'.format(run), np.array(one_run_data_nv, dtype=object))


if exp_name == 'test':
    ### Generate all true state values since it's a small gridworld
    env = Envs.WallGridWorldEnv()
    policy = env.generate_entire_cover_policy()
    q_values, state_values = env.policy_evaluation(policy)

    states = []
    values = []
    for i in range(env.gridsize[0]):
        for j in range(env.gridsize[1]):
            state = env.preprocess_state([i,j])
            states.append(state)
            values.append(state_values[i,j])

    print("Saving values")

    np.save('gridworlddata/wallgridworld_test_states', states)
    np.save('gridworlddata/wallgridworld_test_values', values)

