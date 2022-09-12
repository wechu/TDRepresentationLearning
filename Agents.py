import abc

import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import offline_data

torch.set_default_dtype(torch.float32)


class BaseRLAgent(abc.ABC):
    @abc.abstractmethod
    def act(self, state):
        pass
    #
    # def reset(self):
    #     pass

    def episode_update(self, trajectory, num_steps):
        pass

    def online_update(self, prev_state, action, reward, new_state, done):
        pass

    def additional_metrics(self, trajectory, num_steps, logged_episode_values):
        return None


class GridWorldQLearning(BaseRLAgent):
    def __init__(self, env, step_size, exploration=None, **kwargs):
        self.env = env
        assert env.name == "gridworld"

        self.step_size = step_size
        self.exploration = exploration
        if exploration not in (None, "eps_greedy", "UCB"):
            raise AssertionError(exploration, "isn't supported")
        self.params = kwargs  # extra parameters needed for
        self.prev_state = None
        self.prev_action = None

        if env.name == "gridworld":
            # self.q_values = np.zeros((env.gridsize[0], env.gridsize[1], env.num_actions))
            self.q_values = np.random.uniform(0, 0.0001, (env.gridsize[0], env.gridsize[1], env.num_actions))
            # small random values to break action selection ties randomly
            if self.exploration == "UCB":
                self.counts = np.zeros((env.gridsize[0], env.gridsize[1], env.num_actions))
                # requires param["ucb_constant"] to be set

    def reset(self):
        self.prev_state = None
        self.prev_action  = None
        self.q_values = np.random.uniform(0, 0.0001, self.q_values.shape)

        pass

    def act(self, state):
        action = self.get_action(state)
        self.prev_state = state
        self.prev_action = action
        if self.exploration == "UCB":
            self.counts[state[0], state[1], action] += 1
        return action

    def act_and_update(self, reward, done, state):
        self.online_update(reward, done, state)  # we update first to not override prev_state and prev_action
        action = self.act(state)
        return action

    def get_action(self, state):
        if self.env.name == "gridworld":
            if self.exploration is None:
                action = np.argmax(self.q_values[state[0], state[1]])
            elif self.exploration == "eps_greedy":
                if np.random.rand() > self.params['epsilon']:
                    action = np.argmax(self.q_values[state[0], state[1]])
                else:
                    action = np.random.randint(0, self.env.num_actions)
            elif self.exploration == "UCB":
                action = np.argmax(self.q_values[state[0], state[1]] + self._ucb_bonus(state))

        return action

    def _ucb_bonus(self, state):
        eps = 0.0001
        return self.params['ucb_constant'] / (np.sqrt(self.counts[state[0], state[1]] + eps))

    def online_update(self, prev_state, action, reward, state, done):
        # performs a q-learning update

        if self.env.name == "gridworld":
            if done:
                bootstrap_target = 0
            else:
                bootstrap_target = np.max(self.q_values[state[0], state[1]])

            self.q_values[self.prev_state[0], self.prev_state[1], self.prev_action] += \
                self.step_size * (reward + self.env.discount * bootstrap_target -
                             self.q_values[self.prev_state[0], self.prev_state[1], self.prev_action])

        return


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class VanillaValueNet(nn.Module):
    ''' Assumes we take a vector as input (e.g. state-based observations) '''
    def __init__(self, state_size, action_size, layer_size, num_hidden_layers, seed):
        super().__init__()
        # self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.dense_in = nn.Linear(state_size, layer_size)
        self.dense_hidden_layers = nn.ModuleList([nn.Linear(layer_size, layer_size) for i in range(num_hidden_layers-1)])
        self.dense_out = nn.Linear(layer_size, action_size)
        weight_init(([self.dense_in, self.dense_out]))

        if len(self.dense_hidden_layers) > 0:
            weight_init(self.dense_hidden_layers)

    def forward(self, input):
        """
        Handles variable number of hidden layers
        Returns the output and also the last hidden layer (used for features)
        """
        x = torch.relu(self.dense_in(input))
        for layer in self.dense_hidden_layers:
            x = torch.relu(layer(x))
        out = self.dense_out(x)
        return out, x

class OfflinePolicyEvalTDAgent:
    def __init__(self, state_size, action_size, layer_size=64, num_hidden_layers=2,
                 step_size=0.003, discount=0.99, seed=None, output_state_values=True,
                 target_net_step_size=0.01, target_net_update_freq=-1, batch_size=32, reset_freq=-1, device='cpu', offline_mode=True, env=None,
                 *args, **kwargs):
        ''' This agent uses a TD update to learn a value function with an offline dataset
        state or state-action? '''
        self.output_state_values = output_state_values

        if output_state_values:
            action_size = 1

        # for state-value prediction, we set action_size=1
        self.net_params = {"state_size": state_size, "action_size": action_size, "layer_size": layer_size,
                           "num_hidden_layers": num_hidden_layers, "seed": seed}
        self.net = VanillaValueNet(**self.net_params)
        self.data = None
        self.gamma = discount
        self.step_size = step_size

        self.batch_size = batch_size
        self.target_net_step_size = target_net_step_size
        self.target_net_update_freq = target_net_update_freq  # if positive int, uses hard updates for the target net
        self.device = device

        self.update_counter = 0  # used to know when to reset
        self.reset_freq = reset_freq  # if -1, don't do resets

        self.target_net = copy.deepcopy(self.net)


        self.optimizer = optim.Adam(self.net.parameters(), lr=step_size)

    def offline_update(self):
        ''' One update on the offline data. E.g. one minibatch SGD update '''

        # sample a minibatch
        samples = random.sample(list(self.data), k=self.batch_size)
        # samples = np.random.choice(self.data, size=self.batch_size, replace=False)
        states = np.stack([e[0].copy() for e in samples if e is not None])
        actions = np.vstack([e[1] for e in samples if e is not None])
        rewards = np.vstack([e[2] for e in samples if e is not None])
        next_states = np.stack([e[3].copy() for e in samples if e is not None])
        dones = np.vstack([e[4] for e in samples if e is not None])

        # make torch tensors

        tf = lambda t: torch.tensor(t).float().to(self.device)
        tint = lambda t: torch.tensor(t).long().to(self.device)
        states = tf(states)
        actions = tint(actions)
        rewards = tf(rewards)
        next_states = tf(next_states)
        dones = tf(dones)

        # Get max predicted Q values (for next states) from target model
        value_targets = self.target_net(next_states)[0].detach()

        if not self.output_state_values:
            pass  # have to index the action in the target and current values

        # no need to unsqueeze rewards and dones since they already have shape [batch_size, 1]
        value_targets = rewards + self.gamma * value_targets * (1 - dones)

        current_values = self.net(states)[0]
        # print('target', value_targets.shape)
        # print('value', current_values.shape)

        # Compute loss
        loss = F.mse_loss(current_values, value_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        if self.target_net_update_freq > 0:
            if self.reset_freq >= 0 and not (self.update_counter+1) % self.reset_freq < int(self.reset_freq / 4):
                # todo hardcoded the ratio here. May want to make it a variable
                if (self.update_counter+1) % self.target_net_update_freq == 0:
                    self.soft_update(self.net, self.target_net, 1.0)
        else:
            self.soft_update(self.net, self.target_net, self.target_net_step_size)

        # reset
        if self.reset_freq >= 0:
            if self.update_counter % self.reset_freq == 0:
                # we reset the fast network to random init but leave the target network untouched
                self.net = VanillaValueNet(**self.net_params)
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.step_size)

                print("update counter", self.update_counter)

        self.update_counter += 1


        return loss.detach().cpu().numpy()

    def initialize_data(self, data):
        ''' Loads the offline dataset. Assumes that the data is in a numpy array of tuples.
        Each tuple is of the form: (state, action, reward, next_state, done)
        states are numpy arrays
        '''
        self.data = data
        # self.data = np.load(offline_data_path, allow_pickle=True)
        # if preprocess_fn is not None:
        #     for i in range(len(self.data)):
        #         self.data[i] = preprocess_fn(self.data[i])
        # print(self.data)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_prediction(self, state, action=None):
        if action is None:
            action = 0
        state = torch.tensor(state).float().to(self.device)
        action = torch.tensor(action).long().to(self.device)

        value = self.net(torch.atleast_2d(state))[0][action.unsqueeze(-1)]

        return value.squeeze().detach().cpu().numpy()

    def get_features(self, state, add_bias=False):
        state = torch.tensor(state).float().to(self.device)

        features = self.net(torch.atleast_2d(state))[1]
        if add_bias:
            features = torch.cat([features, torch.ones(size=(features.size(0), 1))], dim=1)
        return features.squeeze().detach().cpu().numpy()

    def load_model(self, save_file_path):
        ''' Loads the model from saved weights '''
        self.net.load_state_dict(torch.load(save_file_path))
        self.target_net = copy.deepcopy(self.net)

    def save_model(self, save_file_path):
        ''' Saves the model '''
        torch.save(self.net.state_dict(), save_file_path)


class OfflinePolicyEvalMCAgent:
    def __init__(self, state_size, action_size, layer_size=64, num_hidden_layers=2,
                 step_size=0.003, discount=0.99, seed=None, output_state_values=True,
                 batch_size=32, device='cpu', offline_mode=True, env=None,
                 *args, **kwargs):
        ''' This agent uses a Monte Carlo update to learn a value function with an offline dataset
        state or state-action?
        Note there's no target network. '''
        self.output_state_values = output_state_values

        if output_state_values:
            action_size = 1

        # for state-value prediction, we set action_size=1
        self.net = VanillaValueNet(state_size, action_size, layer_size, num_hidden_layers, seed)
        self.data = None
        self.mc_returns = None  # used to store the Monte Carlo returns associated to each state todo make work with (s,a)
        self.last_done_index = None  # keeps track of the end of the last completed episode
        self.gamma = discount

        self.batch_size = batch_size
        self.device = device

        self.optimizer = optim.Adam(self.net.parameters(), lr=step_size)

    def offline_update(self):
        ''' One update on the offline data. E.g. one minibatch SGD update
        We only sample from the transitions included in completed episodes. The last partial episode is not used. '''

        # sample a minibatch
        sample_idxs = random.sample(list(range(self.last_done_index)), k=self.batch_size)

        states = np.stack([self.data[e][0].copy() for e in sample_idxs if e is not None])
        actions = np.vstack([self.data[e][1] for e in sample_idxs if e is not None])
        mc_returns = np.vstack([self.mc_returns[e] for e in sample_idxs if e is not None])

        # samples = random.sample(list(self.data), k=self.batch_size)
        # samples = np.random.choice(self.data, size=self.batch_size, replace=False)
        # states = np.stack([e[0].copy() for e in samples if e is not None])
        # actions = np.vstack([e[1] for e in samples if e is not None])
        # rewards = np.vstack([e[2] for e in samples if e is not None])
        # next_states = np.stack([e[3].copy() for e in samples if e is not None])
        # dones = np.vstack([e[4] for e in samples if e is not None])

        # make torch tensors
        tf = lambda t: torch.tensor(t).float().to(self.device)
        tint = lambda t: torch.tensor(t).long().to(self.device)
        states = tf(states)
        actions = tint(actions)
        mc_returns = tf(mc_returns)
        # rewards = tf(rewards)
        # next_states = tf(next_states)
        # dones = tf(dones)

        current_values = self.net(states)[0]  # todo modify this for state-actions

        # Compute loss
        loss = F.mse_loss(current_values, mc_returns)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def initialize_data(self, data):
        ''' Loads the offline dataset. Assumes that the data is in a numpy array of tuples.
        Each tuple is of the form: (state, action, reward, next_state, done)
        states are numpy arrays
        Computes and saves the Monte-Carlo returns and stores them in self.mc_returns. These match the order of the
        transitions in self.data
        '''
        self.data = data

        # find the index of the last complete episode
        last_done_index = len(self.data)-1
        while self.data[last_done_index][4] is not True:
            last_done_index -= 1

        self.last_done_index = last_done_index  # we record this so we only sample from complete episodes

        # compute returns
        self.mc_returns = []
        total_reward = 0
        for i in reversed(range(last_done_index)):
            if self.data[i][4] is True:  # if done
                total_reward = 0
            total_reward = self.data[i][2] + self.gamma * total_reward
            self.mc_returns.append(total_reward)  # this list is reversed for now

        self.mc_returns = list(reversed(self.mc_returns))


        # self.data = np.load(offline_data_path, allow_pickle=True)
        # if preprocess_fn is not None:
        #     for i in range(len(self.data)):
        #         self.data[i] = preprocess_fn(self.data[i])
        # print(self.data)

    def get_prediction(self, state, action=None):
        if action is None:
            action = 0
        state = torch.tensor(state).float().to(self.device)
        action = torch.tensor(action).long().to(self.device)

        value = self.net(torch.atleast_2d(state))[0][action.unsqueeze(-1)]

        return value.squeeze().detach().cpu().numpy()

    def get_features(self, state, add_bias=False):
        state = torch.tensor(state).float().to(self.device)

        features = self.net(torch.atleast_2d(state))[1]

        if add_bias:
            features = torch.cat([features, torch.ones(size=(features.size(0), 1))], dim=1)
        return features.squeeze().detach().cpu().numpy()

    def load_model(self, save_file_path):
        ''' Loads the model from saved weights '''
        self.net.load_state_dict(torch.load(save_file_path))

    def save_model(self, save_file_path):
        ''' Saves the model '''
        torch.save(self.net.state_dict(), save_file_path)


#
# class OfflinePolicyEvalMCAgent():
#     def __init__(self, offline_mode=True):
#         ''' '''
#
#     def offline_update(self):
#         ''' One update on the offline data. E.g. one minibatch SGD update'''
#         pass
#
#     def initialize_data(self, offline_data):
#         ''' '''

# from collections import deque
# import random
#
# class ReplayBuffer(object):
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)
#
#     def push(self, state, action, reward, next_state, done):
#         state = np.expand_dims(state, 0)
#         next_state = np.expand_dims(next_state, 0)
#
#         self.buffer.append((state, action, reward, next_state, done))
#
#     def sample(self, batch_size):
#         state, action, reward, next_state, done = zip(
#             *random.sample(self.buffer, batch_size)
#         )
#         return np.concatenate(state), action, reward, np.concatenate(next_state), done
#
#     def __len__(self):
#         return len(self.buffer)


