import torch 
import torch.nn as nn
import torch.optim as optim
import random 
from collections import namedtuple, deque 
import copy
import numpy as np
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# create the critic network
class Critic(nn.Module):
    def __init__(self, num_state, num_nodes, num_action):
        super(Critic,self).__init__()
        self.layer1 = nn.Linear(num_state,num_nodes)
        self.layer2 = nn.Linear(num_nodes,num_nodes)
        self.layer3 = nn.Linear(num_nodes,num_action)

    def forward(self, state, action):
        # mapping state to action values
        x = torch.cat([state, action], 1)
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return self.layer3(x)

# create the actor network
class Actor(nn.Module):
    def __init__(self, num_state, num_nodes, num_action):
        super(Actor,self).__init__()
        self.layer1 = nn.Linear(num_state,num_nodes)
        self.layer2 = nn.Linear(num_nodes,num_nodes)
        self.layer3 = nn.Linear(num_nodes,num_action)

    def forward(self, state):
        # mapping state to action values
        x = torch.tanh(self.layer1(state))
        x = torch.tanh(self.layer2(x))
        return self.layer3(x)

# Replay buffer for storing experience
class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, *args):
        # add a new memory
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # sample a random memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # return length of memory
        return len(self.memory)

#OUNoise 
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

# # https://github.com/openai/gym/blob/master/gym/core.py
# class NormalizedEnv(gym.ActionWrapper):
#     """ Wrap action """

#     def _action(self, action):
#         act_k = (self.action_space.high - self.action_space.low)/ 2.
#         act_b = (self.action_space.high + self.action_space.low)/ 2.
#         return act_k * action + act_b

#     def _reverse_action(self, action):
#         act_k_inv = 2./(self.action_space.high - self.action_space.low)
#         act_b = (self.action_space.high + self.action_space.low)/ 2.
#         return act_k_inv * (action - act_b)

# DDPG agent to interact with the environment
class Agent():
    def __init__(self, num_state, num_action, gamma, tau, num_nodes, critic_alpha, actor_alpha, batch_size):
        self.num_state = num_state
        self.num_action = num_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor_net = Actor(num_state, num_nodes, num_action).to(device)
        self.actor_net_target = Actor(num_state, num_nodes, num_action).to(device)
        self.critic_net = Critic(num_state + num_action, num_nodes, num_action).to(device)
        self.critic_net_target = Critic(num_state + num_action, num_nodes, num_action).to(device)

        # make copies of original networks to initialize target networks
        self.actor_net_target = copy.deepcopy(self.actor_net)
        self.critic_net_target = copy.deepcopy(self.critic_net)

        self.memory = ReplayBuffer(10000)
        self.critic_loss_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), actor_alpha)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), critic_alpha)

    def choose_action(self, state):
        # state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor_net.forward(state)
        return action.detach().numpy()[0,0]

        # if not torch.is_tensor(state): state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        # with torch.no_grad():
        #     action = self.actor_net(state).max(1)[1].view(1, 1)
        #     return action.type(torch.FloatTensor)

    def optimize(self):
        if len(self.memory) < self.batch_size: return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # values of states, actions, and rewards
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        # rewards = rewards.flatten()
        next_states = torch.cat(batch.next_state)

        Q_vals = self.critic_net.forward(states, actions)
        next_Q_vals = self.critic_net_target.forward(next_states, self.actor_net_target.forward(next_states).detach())
        Qp_vals = next_Q_vals * self.gamma + rewards
        critic_loss = self.critic_loss_criterion(Q_vals, Qp_vals)
        actor_loss = -self.critic_net.forward(states, self.actor_net.forward(states))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.sum().backward()
        # actor_loss.backward()
        self.actor_optimizer.step()

        for target_parameter, parameter in zip(self.actor_net_target.parameters(), self.actor_net.parameters()):
            target_parameter.data.copy_(parameter.data * self.tau + target_parameter.data * (1.0 - self.tau))

        for target_parameter, parameter in zip(self.critic_net_target.parameters(), self.critic_net.parameters()):
            target_parameter.data.copy_(parameter.data * self.tau + target_parameter.data * (1.0 - self.tau))

            

