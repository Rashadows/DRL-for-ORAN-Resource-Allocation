import random
import torch 
import numpy as np
from td3_256 import action_dim
from td3_256 import gamma
from td3_256 import target_policy_net
from td3_256 import target_value_net1
from td3_256 import value_net1

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action.squeeze(), reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class GaussianExploration(object):
    def __init__(self, max_sigma=1.0, min_sigma=1.0, decay_period=1000000):
        self.low  = 0
        self.high = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
    
    def get_action(self, action, t=0):
        sigma  = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        action = action + np.random.normal(size=len(action)) * sigma
        return np.clip(action, self.low, self.high)
    
class OrnsteinUhlenbeckNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, decay_period=1000000):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.decay_period = decay_period
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def get_action(self, action, t=0):
        sigma = self.sigma - (self.sigma) * min(1.0, t / self.decay_period)
        # Applying the Ornstein-Uhlenbeck process
        dx = self.theta * (self.mu - self.state) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state = self.state + dx
        action_with_noise = action + self.state
        return action_with_noise

def compute_td_errors(transitions):
    states = transitions.state
    actions = torch.cat(transitions.action, dim=0)
    rewards = torch.cat(transitions.reward, dim=0).squeeze(1) 
    next_states = transitions.next_state
    dones = torch.tensor(transitions.done, dtype=torch.float32) 
    with torch.no_grad():
        next_actions = target_policy_net(next_states)
        target_q_values = target_value_net1(next_states, next_actions)
        target_q_values = rewards + (1 - dones) * gamma * target_q_values
    
    current_q_values = value_net1(states, actions)
    td_errors = (target_q_values - current_q_values).abs()

    return td_errors.cpu().detach().numpy()
