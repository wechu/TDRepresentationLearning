import numpy as np
from itertools import product
from collections import defaultdict


class RandomMDP:
    def __init__(self, num_states, num_actions, setting=None):
        # continuing discounted MDP
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = 0.85

        if setting is not None:
            self.set_MDP(setting)
        # we sample the transition probs from a dirichlet distribution
        dirichlet_alpha = np.ones(self.num_states) / self.num_states
        self.transition_probs = np.random.dirichlet(dirichlet_alpha, size=(self.num_states, self.num_actions))

        # reward function
        self.reward_fn = None
        self.resample_reward()

        self.state = 0
        self.reset()

    def reset(self):
        self.state = 0

    def resample_reward(self):
        self.reset()
        self.reward_fn = np.random.uniform(0, 1, self.num_states)


    def reward(self, state, action):
        return self.reward_fn[state]

    def transition(self, state, action):
        return self.transition_probs[state, action]

    def step(self, action, state=None):
        bool_state = state is None

        if bool_state:
            state = self.state
        next_state = np.random.choice(np.arange(self.num_states), p=self.transition_probs[state, action])

        if bool_state:
            self.state = next_state

        reward = self.reward_fn[state]
        # print("rs", reward, state)
        return next_state, reward, False  # done is always false

    def value_iteration(self, init_values=None):
        # returns the optimal values and optimal policy
        tolerance = 1e-6
        max_error = 99999
        iter = 0

        if init_values:
            q_values = init_values
        else:
            q_values = 1/(1-self.discount) /2 * np.ones((self.num_states, self.num_actions))  # initialize a guess to the values
            # q_values = np.zeros((self.num_states, self.num_actions))
        while max_error > tolerance:
            iter += 1
            copy_q_values = q_values.copy()
            for s,a in product(range(self.num_states), range(self.num_actions)):
                max_q_values = np.max(q_values, axis=1)

                q_values[s,a] = self.reward(s,a) + self.discount * np.sum(self.transition(s,a) * max_q_values)

            max_error = np.max(np.abs(copy_q_values - q_values))
        # print("iter", iter)
        return q_values, np.argmax(q_values, axis=1)

    def set_MDP(self, setting):
        # Set the transition matrix to be from some preset choices
        if setting == "death":
            # two states + a terminal state (death)
            # should the agent move to the other state or the terminal state
            # in the other state, the agent can stay or go to terminal state
            self.num_states = 3
            self.num_actions = 2
            self.transition_probs = np.array([[[1, 0, 0], [0, 1, 0]],   # start state
                                             [[0, 1, 0], [0, 0, 1]],   # second state
                                             [[0, 0, 1], [0, 0, 1]]])   # terminal state


        if setting == "power_example":
            self.num_states = 8
            self.num_actions = 3
            self.transition_probs = np.array([
                                     [[0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0]], # center state
                                     [[0,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0]],  # terminal state
                                     [[0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,1,0,0,0]],  # left state 1
                                     [[0,0,0,0,1,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,1,0,0,0]],  # left state 2
                                     [[0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,1,0,0,0]],  # left state 3
                                     [[0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1]],  # right state 1
                                     [[0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1]],  # right state 2
                                     [[0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1]]]) # right state 3
            # I just put three actions per state so that it fit the generic template
            # and not have variable number of actions per state
            # So some actions are just duplicates of others
        return


class GridWorldEnv:  # todo refactor this to be able to use walls
    def __init__(self, image_obs=True, random_goal=True, give_map=False, random_map=False, no_map=False,
                 use_memory=False, start_state=1, gridsize=None, **kwargs):
        '''
        image_obs: Gives the state as a stack of grids.
        random_goal: If True, randomizes the goal at each episode
        give_map: If True, the map (goal location) is given to the agent from the start
        no_map: If True, no map is available to the agent
        use_memory: If True, the observation includes an extra channel which tells the agent which states it has visited
        start_state: Int. Specifies the starting location at cell (start_state, start_state)
        '''
        self.name = 'gridworld'
        # the x-position goes from [0, gridsize[0]-1] and y-position goes from [0, gridsize[1]-1]
        self.num_actions = 5
        self.discount = 0.95

        if gridsize is None:
            gridsize = [6,6]
        self.gridsize = gridsize
        self.image_obs = image_obs
        self.give_map = give_map
        self.random_map = random_map
        self.no_map = no_map
        self.use_memory = use_memory

        if self.use_memory:
            self.state_size = [4, gridsize[0], gridsize[1]]
        else:
            self.state_size = [3, gridsize[0], gridsize[1]]
        print("state_size", self.state_size)
        self.random_goal = random_goal

        self.steps = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])  # up, down, left, right, stay

        self.start_state = np.array([start_state, start_state])
        self.map_state = np.array([0, 0])
        self.goal_states = [[np.array([self.gridsize[0]-1, self.gridsize[1]-1]), 1]]  # [goal, reward]

        if self.no_map:
            self.map_state = np.array([-1, -1])  # unreachable location

        self.pos = self.start_state.copy()
        self.timestep = 0
        self.picked_up_map = self.give_map

        if self.use_memory:
            self.visited_locations = np.zeros(self.gridsize, dtype='int8')

        self.reset()


    def reset(self):
        self.pos = self.start_state.copy()
        self.timestep = 0

        self.picked_up_map = self.give_map

        if self.random_goal:
            goal_state = np.array([0,0])
            while np.all(goal_state < np.array([self.gridsize[0]/2,self.gridsize[1]/2])):  # at least one of the two dimensions should be 3 or above
                goal_state = np.random.randint([0, 0], self.gridsize)

            self.goal_states = [[goal_state, 1]]

        if self.use_memory:
            self.visited_locations = np.zeros(self.gridsize, dtype='int8')

        if self.image_obs:
            return self.state_to_image()
        return self.pos.copy()

    def copy(self):
        copy_env = GridWorldEnv(self.gridsize)
        copy_env.pos = self.pos.copy()
        return copy_env

    def _check_valid_pos(self, state):
        return 0 <= state[0] < self.gridsize[0] and 0 <= state[1] < self.gridsize[1]

    def _check_goal(self, state):
        for goal_state, goal_reward in self.goal_states:
            if np.all(goal_state == state):
                return True, goal_reward
        return False, 0.0

    def step(self, action):
        self.timestep += 1
        query_state = self.pos
        reward, next_states, next_state_probs, done = self.transition_dist(query_state, action)

        # print(next_states)
        if len(next_states) > 1:
            idx = np.random.choice(np.arange(len(next_states)), p=next_state_probs)
            next_state = next_states[idx]
        else:
            next_state = next_states[0]

        self.pos = np.array(next_state)

        if self.random_goal:
            if np.all(self.map_state == self.pos):
                self.picked_up_map = True

        obs = np.array(next_state)
        if self.image_obs:
            obs = self.state_to_image()

        if self.use_memory:
            self.visited_locations[self.pos[0], self.pos[1]] = 1

        return obs, reward, done, None

    def transition_dist(self, state, action):
        # returns a list of reward, next_states, next_states_prob and done
        # each returned value is a list
        # next_states and next_states_prob are a list of possible next states and their probabilities (nonzero)
        # done is a boolean which is true if state is terminal
        # Note that in this env, done is only true when the agent takes an action _after_ having reached a terminal state
        state = np.array(state)
        next_states_and_prob = defaultdict(lambda: 0.0)  # keys are next states, values are probabilities

        # # done and reward only depend on the current state
        # done, reward = self._check_goal(state)

        # for movement
        # first, consider the case of not slipping
        temp_state = state + self.steps[action]
        if self._check_valid_pos(temp_state):
            next_state = temp_state
        else:
            next_state = state
        next_states_and_prob[tuple(next_state)] = 1.0

        # done and reward only depend on the next state
        # this is only correct if next_state is deterministic
        done, reward = self._check_goal(next_state)

        return reward, list(next_states_and_prob.keys()), list(next_states_and_prob.values()), done

    def state_to_image(self):
        # Retunrs a 3x6x6 stack of grids
        img = np.zeros(shape=(3, self.gridsize[0], self.gridsize[1]), dtype='int8')
        img[0, self.pos[0], self.pos[1]] = 1  # agent position
        img[1, self.map_state[0], self.map_state[1]] = 1  # map position
        if self.picked_up_map:
            goal_pos = self.goal_states[0][0]
            img[2, goal_pos[0], goal_pos[1]] = 1  # goal position

        if self.use_memory:
            img = np.append(img, self.visited_locations.reshape((1, self.gridsize[0], self.gridsize[1])), axis=0)
        return img







if __name__ == "__main__":
    # # env = RandomMDP(8,3)
    # # env.set_MDP(setting="power_example")
    # # print(env.reward_fn)
    # # for i in range(10):
    # #     state, reward, _ = env.step(1)
    # #     print(state, reward)
    # #
    # # q_values, opt_policy = env.value_iteration()
    # # state_values = np.max(q_values, axis=1)
    # # pass
    #
    #
    # # sample many reward functions for the same mdp
    # # compute the optimal values for each
    # # estimate Power
    #
    # num_samples = 1000
    # # initial state is 0, so we compute Power of the initial state
    #
    # all_values = []
    # for i_reward in range(num_samples):
    #     env.resample_reward()
    #     q_values, opt_policy = env.value_iteration()
    #     state_values = np.max(q_values, axis=1)
    #     all_values.append(state_values)
    #
    # all_values = np.array(all_values)
    # all_power = (1-env.discount)/env.discount * (all_values - env.reward_fn)
    # power = np.mean(all_power, axis=0)
    #
    # # print(np.array([env.transition_probs[i, opt_policy[i], :] for i in opt_policy]))
    # # print(env.transition_probs[np.arange(env.num_states), opt_policy])
    # print(f"{power.round(4)} +/- \n {(2*np.std(all_power, axis=0)/np.sqrt(num_samples)).round(4)}")

    import deepRL
    import torch
    env = RandomGoalMapGridWorldEnv()
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 1e-2
    LR = 1e-3
    UPDATE_EVERY = 1
    n_step = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


