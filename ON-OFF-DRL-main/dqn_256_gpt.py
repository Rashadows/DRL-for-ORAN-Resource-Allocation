'''
this is adapted from chatgpt
'''

# -*- coding: utf-8 -*-
import os
from datetime import datetime
from collections import deque
import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
from env import Env
from argparser import args
from time import sleep

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (torch.FloatTensor(np.array(state)).to(device),
                torch.LongTensor(action).to(device),
                torch.FloatTensor(reward).to(device),
                torch.FloatTensor(np.array(next_state)).to(device),
                torch.FloatTensor(done).to(device))

    def __len__(self):
        return len(self.buffer)

# DQN Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        return self.network(x)

# Hyperparameters
state_dim = args.n_servers * args.n_resources + args.n_resources + 1
action_dim = args.n_servers
gamma = 0.99
lr = 0.001
batch_size = 64
replay_buffer_capacity = 10000
target_update_freq = 1000
max_episodes = 500
max_timesteps = 225
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 10000
log_freq = 500

# Initialize environment, networks, optimizer, and replay buffer
env = Env()
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
replay_buffer = ReplayBuffer(replay_buffer_capacity)

# Training loop
global_steps = 0
epsilon = epsilon_start

for episode in range(max_episodes):
    state = env.reset()
    episode_reward = 0

    for t in range(max_timesteps):
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy_net(state_tensor).argmax(dim=1).item()
        else:
            action = random.randint(0, action_dim - 1)

        # Take action in environment
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        global_steps += 1

        # Update epsilon
        epsilon = max(epsilon_end, epsilon_start - global_steps / epsilon_decay)

        # Sample and train the policy network
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + gamma * max_next_q_values * (1 - dones)
            loss = nn.MSELoss()(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network
        if global_steps % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

print("Training Complete.")
