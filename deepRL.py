import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import random
import math
from collections import deque, namedtuple
import time

import Agents
# import logger
# from memory_profiler import profile

torch.set_default_dtype(torch.float32)

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class QR_DQN(nn.Module):
    ''' Assumes we take an image-like observation of shape (channels, dim1, dim2) '''
    def __init__(self, state_size, action_size, layer_size, n_step, seed, N, layer_type="ff"):
        super(QR_DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.N = N

        # input_size = 36*3
        print(self.input_shape, layer_size)
        # self.head_1 = nn.Linear(input_size, layer_size)
        self.conv_1 = nn.Conv2d(in_channels=state_size[0], out_channels=16, kernel_size=5, padding='same')
        # self.conv_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding='same')
        conv_size = 36*16

        self.ff_1 = nn.Linear(conv_size, layer_size)
        # self.ff_1_2 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size * N)
        weight_init([self.conv_1, self.ff_1])


    def forward(self, input):
        """
        1 conv and 1 dense layer
        """
        x = torch.relu(self.conv_1(input))
        # x = torch.relu(self.conv_2(x))
        x = torch.flatten(x, start_dim=1)   # assumes we get a minibatch of images

        # x = torch.relu(self.head_1(x))
        x = torch.relu(self.ff_1(x))
        # x = torch.relu(self.ff_1_2(x))
        out = self.ff_2(x)

        return out.view(input.shape[0], self.N, self.action_size)

    def get_action(self, input):
        x = self.forward(input)
        return x.mean(dim=1)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # print("before:", state,action,reward,next_state, done)
        state = np.array(state).copy()
        next_state = np.array(next_state).copy()
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return()
            # print("after:",state,action,reward,next_state, done)
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def calc_multistep_return(self):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma ** idx * self.n_step_buffer[idx][2]

        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], Return, self.n_step_buffer[-1][3], \
               self.n_step_buffer[-1][4]

    def sample(self):
        """Sample a batch of experiences from memory.
        Returns list of numpy arrays (for states) and standard scalars for  """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.stack([e.state.copy() for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.stack([e.next_state.copy() for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])
        return states, actions, rewards, next_states, dones

    # def sample(self):
    #     """Randomly sample a batch of experiences from memory."""
    #
    #     # idxs = np.random.randint(len(self.memory), size=self.batch_size)
    #     # with torch.no_grad():
    #     experiences = random.sample(self.memory, k=self.batch_size)
    #
    #     # using copy() is important here or else we get a memory leak
    #     states = torch.from_numpy(np.stack([e.state.copy() for e in experiences if e is not None])).float().to(self.device)
    #     actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
    #     rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
    #     next_states = torch.from_numpy(np.stack([e.next_state.copy() for e in experiences if e is not None])).float().to(
    #         self.device)
    #     dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
    #         self.device)
    #
    #     return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class QRDQN_Agent(Agents.BaseRLAgent):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 Network,
                 layer_size,
                 n_step,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 EPSILON,
                 EPS_DECAY_STEPS,
                 MIN_MEM_SIZE,
                 device,
                 env,
                 *args, **kwargs):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            Network (str): dqn network type
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """

        seed = np.random.randint(100000)
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.EPSILON = EPSILON
        self.EPS_DECAY_STEPS = EPS_DECAY_STEPS
        self.Q_updates = 0
        self.n_step = n_step
        self.N = 64
        self.quantile_tau = torch.FloatTensor([i / self.N for i in range(1, self.N + 1)]).to(device)
        self.min_memory = MIN_MEM_SIZE  # no updates and pick random actions if replay buffer is small
        # self.min_memory = 1000

        self.action_step = 1
        self.last_action = None

        # Q-Network
        self.qnetwork_local = QR_DQN(state_size, action_size, layer_size, n_step, seed, self.N).to(device)
        # with torch.no_grad():
        self.qnetwork_target = QR_DQN(state_size, action_size, layer_size, n_step, seed, self.N).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA, n_step)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.total_t_step = 0  # for decaying epsilon

    def online_update(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.min_memory:
            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
            if self.t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > self.BATCH_SIZE:
                    experiences = self.memory.sample()
                    loss = self.learn(experiences)
                    self.Q_updates += 1
                    return loss
        return None


    def act(self, state):
        """Returns actions for given state as per current policy.
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state

        """
        self.total_t_step += 1

        if len(self.memory) < self.min_memory:
            action = random.choice(np.arange(self.action_size))
            return action

        eps = self._eps_decay_value()
        # print(self.total_t_step, eps)
        # if self.action_step == 4:

        state = np.array(state)

        self.qnetwork_local.eval()
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action_values = self.qnetwork_local.get_action(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:  # select greedy action if random number is higher than epsilon or noisy network is used!
            action = np.argmax(action_values.cpu().data.numpy())
            self.last_action = action
            return action
        else:
            action = random.choice(np.arange(self.action_size))
            self.last_action = action
            return action
            # self.action_step = 0
        # else:
        #     self.action_step += 1
        #     return self.last_action

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        tf = lambda t: torch.tensor(t).float().to(self.device)
        tint = lambda t: torch.tensor(t).long().to(self.device)
        states = tf(states)
        actions = tint(actions)
        rewards = tf(rewards)
        next_states = tf(next_states)
        dones = tf(dones)

        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            Q_targets_next = self.qnetwork_target(next_states).detach().cpu()  # .max(2)[0].unsqueeze(1) #(batch_size, 1, N)
            action_indx = torch.argmax(Q_targets_next.mean(dim=1), dim=1, keepdim=True)

            Q_targets_next = Q_targets_next.gather(2, action_indx.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)).transpose(1,
                                                                                                                              2)
            assert Q_targets_next.shape == (self.BATCH_SIZE, 1, self.N)
            # Compute Q targets for current states
            Q_targets = rewards.unsqueeze(-1) + (
                        self.GAMMA ** self.n_step * Q_targets_next.to(self.device) * (1 - dones.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1))
        # Compute loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(self.quantile_tau - (td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1)  # , keepdim=True if per weights get multipl
        loss = loss.mean()
        # Minimize the loss
        loss.backward()
        # clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

    def _eps_decay_value(self):
        if self.total_t_step >= self.EPS_DECAY_STEPS + self.min_memory:
            return self.EPSILON
        else:
            return 1 - (1-self.EPSILON) * self.total_t_step / (self.EPS_DECAY_STEPS + self.min_memory)


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    # assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss


