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


class GridWorldEnv:
    def __init__(self, image_obs, reward_type='sparse', goal_states=None, walls=None,
                 gridsize=None, rescale_state=True, **kwargs):
        '''
        image_obs: Gives the state as a stack of grids.
        random_goal: If True, randomizes the goal at each episode
        reward_type: In 'sparse', 'dense', 'shaped' todo implement this (how to deal with discounting?)
        walls: List of walls as indicated by pairs of adjacent cells for which there is a wall in between.
        rescale_state: If True, divides the state position by the gridsize so it's between 0 and 1
        start_state: Int. Specifies the starting location at cell (start_state, start_state)
        '''
        self.name = 'gridworld'
        # the x-position goes from [0, gridsize[0]-1] and y-position goes from [0, gridsize[1]-1]
        self.num_actions = 5

        self.discount = 0.9
        self.rescale_state = rescale_state

        if gridsize is None:
            gridsize = [6,6]
        self.gridsize = np.array(gridsize)
        self.image_obs = image_obs

        self.steps = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])  # up, down, left, right, stay

        self.start_state = np.array([0,0])
        self.goal_states = goal_states
        # self.goal_states = [[np.array([self.gridsize[0]-1, self.gridsize[1]-1]), 1]]  # [goal, reward]

        self.walls = np.array(walls)  # list of pairs of locations. Walls are between each pair of locations. E.g. ((0,0), (1,0))
        # self.walls is indexed by (wall_index, cell_index (0 or 1), cell_coordinate (0 or 1) )

        self.pos = self.start_state.copy()
        self.timestep = 0

        self.reset()

    def reset(self):
        self.pos = self.start_state.copy()
        self.timestep = 0

        obs = self.pos.copy()

        if self.image_obs:
            obs = self.state_to_image()
        else:
            obs = np.array(self.pos.copy())
            if self.rescale_state:
                obs = obs / self.gridsize

        return obs

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

        if self.image_obs:
            obs = self.state_to_image()
        else:
            obs = np.array(next_state)
            if self.rescale_state:
                obs = obs / self.gridsize

        return obs, reward, done, None

    def _check_wall(self, state, next_state):
        ''' Returns true if there is a collision with a wall.
        Assumes state, next_state and wall are numpy arrays '''
        for wall in self.walls:
            if np.all(state == wall[0]) and np.all(next_state == wall[1]):
                return True
            elif np.all(state == wall[1]) and np.all(next_state == wall[0]):
                return True
        return False

    def transition_dist(self, state, action):
        # returns a list of reward, next_states, next_states_prob and done
        # each returned value is a list
        # next_states and next_states_prob are a list of possible next states and their probabilities (nonzero)
        # done is a boolean which is true if state is terminal
        # Note that in this env, done is only true when the agent takes an action _after_ having reached a terminal state
        state = np.array(state)
        next_states_and_prob = defaultdict(lambda: 0.0)  # keys are next states, values are probabilities

        # # done and reward only depend on the current state
        done, reward = self._check_goal(state)

        # for movement
        # first, consider the case of not slipping
        temp_state = state + self.steps[action]

        if self._check_valid_pos(temp_state) and not self._check_wall(state, temp_state):
            next_state = temp_state
        else:
            next_state = state
        next_states_and_prob[tuple(next_state)] = 1.0

        # check this toggle
        # done and reward only depend on the next state
        # this is only correct if next_state is deterministic
        # done, reward = self._check_goal(next_state)

        return reward, list(next_states_and_prob.keys()), list(next_states_and_prob.values()), done

    def state_to_image(self):
        # Returns a 6x6 grid
        img = np.zeros(shape=(self.gridsize[0], self.gridsize[1]), dtype='int8')
        img[self.pos[0], self.pos[1]] = 1  # agent position

        return img

    def preprocess_state(self, state):
        # rescales the state between 0 and 1
        return np.array(state) / self.gridsize

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

    def policy_evaluation(self, policy):
        ''' Returns a list of all state-actions and their associated values
        Q-values and V-values
        policy: this is a numpy array with shape (gridsize[0], gridsize[1], num_actions) containing the policy
            probabilities in each entry '''
        tolerance = 1e-6
        max_error = 99999
        iter = 0

        q_values = 0.5 * np.ones((self.gridsize[0], self.gridsize[1], self.num_actions))

        while max_error > tolerance:
            iter += 1
            copy_q_values = q_values.copy()
            q_value_targets = np.sum(q_values * policy, axis=2)

            for i,j,a in product(range(self.gridsize[0]), range(self.gridsize[1]), range(self.num_actions)):
                reward, next_states, next_states_prob, done = self.transition_dist(np.array([i,j]), a)
                updated_q = reward
                # if i == 5 and j == 0:
                #     print("here", reward, done)
                if not done:
                    for idx in range(len(next_states)):
                        updated_q += self.discount * next_states_prob[idx] * q_value_targets[next_states[idx]]  # uses tuple indexing
                q_values[i,j,a] = updated_q
                # print(updated_q)
            #
            max_error = np.max(np.abs(copy_q_values - q_values))
            print(iter, max_error)

        state_values = np.sum(q_values * policy, axis=2)

        return q_values, state_values


class WallGridWorldEnv(GridWorldEnv):
    def __init__(self, rescale_state=True   ):
        ''' This env is a 6x6 gridworld with a horizontal wall between indices y=2 and y=3 extending from the left side
        leaving one space open on the right side. s is the start cell and x is the goal cell.
        o denotes a traversable cell.
         s o o o o o
         o o o o o o
         o o o o o o
         - - - - -
         o o o o o o
         o o o o o o
         x o o o o o
        '''
        super().__init__(image_obs=False, goal_states=[[np.array([5, 0]), 1]],
                         walls=[((2,i),(3,i)) for i in range(5)], gridsize=(6,6), rescale_state=rescale_state)

    def eps_good_policy(self, state, eps=0.1, unscale=True):
        if np.random.rand() < eps:
            return np.random.randint(5)
        else:
            return self._good_policy(state, unscale)

    def _eps_good_policy_dist(self, state, eps=0.1, unscale=True):
        num_actions = 5
        dist = eps * np.ones(num_actions) / num_actions
        dist[self._good_policy(state, unscale=unscale)] = 1 - eps * (num_actions - 1) / num_actions
        return dist

    def generate_entire_policy(self, eps=0.1):
        ''' Generates a dictionary of the policy at every state. Used for policy evaluation '''
        policy = np.zeros((self.gridsize[0], self.gridsize[1], self.num_actions))

        for i in range(self.gridsize[0]):
            for j in range(self.gridsize[1]):
                policy[i,j] = np.array(self._eps_good_policy_dist(state=(i, j), eps=eps, unscale=False))
        return policy

    def _good_policy(self, state, unscale=True):
        # change so it covers the state space better
        if unscale:
            state = np.array(state)
            state *= self.gridsize
            state = np.rint(state).astype('int32')

        if state[0] <= 2:
            if state[1] == 5:
                return 1  # go down
            else:
                return 3  # go right
        elif state[0] >= 3:
            if state[1] == 0:
                return 1  # go down
            else:
                return 2  # go left

    def cover_policy(self, state, unscale=True):
        # choose actions
        if unscale:
            state = np.array(state)
            state *= self.gridsize
            state = np.rint(state).astype('int32')

        if state[0] <= 2:
            if state[1] == 5:
                action = 1  # go down
            elif state[1] == 0 and state[0] in (0, 1):
                action = 1 if np.random.rand() < 0.5 else 3
            else:
                action = 3  # go right
        elif state[0] >= 3:
            if state[1] == 0:
                action = 1  # go down
            elif state[1] == 5 and state[0] in (3,4):
                action = 1 if np.random.rand() < 0.5 else 2
            else:
                action = 2  # go left

        eps = 0.1
        if np.random.rand() < eps:
            action = np.random.randint(self.num_actions)
        return action

    def generate_entire_cover_policy(self):
        eps = 0.1
        policy = self.generate_entire_policy(eps)
        for i in [0,1]:
            probs = eps / self.num_actions * np.ones(self.num_actions)
            probs[1] = probs[3] = (1-eps)/2 + eps/5
            policy[i,0] = np.array(probs)

        for i in [3,4]:
            probs = eps / self.num_actions * np.ones(self.num_actions)
            probs[1] = probs[2] = (1-eps)/2 + eps/5
            policy[i,5] = np.array(probs)

        return policy






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
    env = WallGridWorldEnv()
    pol = env.generate_entire_cover_policy()

    q_values, state_values = env.policy_evaluation(pol)
    np.set_printoptions(precision=3)
    print(state_values)



    # print(pol[2,0])
    #
    # s, r, done, _ = env.step(1)
    # print(s)
    # print(env.eps_good_policy_dist(s))
    # s, r, done, _ = env.step(1)
    # print(s)
    # print(env.eps_good_policy_dist(s))
    # s, r, done, _ = env.step(1)
    # print(s)
    # print(env.eps_good_policy_dist(s))
    # print(env.transition_dist(np.array([3,3]), 0))
