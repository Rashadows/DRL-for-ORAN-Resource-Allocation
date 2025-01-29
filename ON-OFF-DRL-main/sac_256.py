# -*- coding: utf-8 -*-
############################### Import libraries ###############################

import os
import copy
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

################################## Set device to CPU or CUDA ##################################

print("============================================================================================")

# Set device to CPU or CUDA
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")

NN_size = 256
################################## Define SAC Policy ##################################

class EpisodicReplayMemory(object):
    def __init__(self, capacity, max_episode_length):
        self.num_episodes = capacity // max_episode_length
        self.buffer = deque(maxlen=self.num_episodes)
        self.buffer.append([])
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        self.buffer[self.position].append((state, action, reward, next_state, done))
        if done:
            self.buffer.append([])
            self.position = min(self.position + 1, self.num_episodes - 1)

    def sample(self, batch_size):
        episodes = random.sample(self.buffer, batch_size)
        batch = []
        for episode in episodes:
            batch.extend(episode)
        state, action, reward, next_state, done = zip(*batch)
        return torch.stack(state), torch.stack(action), torch.stack(reward), torch.stack(next_state), torch.stack(done)

    def __len__(self):
        return len(self.buffer)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=NN_size):
        super(ActorCritic, self).__init__()

        # Policy Network (outputs log action probabilities)
        self.policy_net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.LogSoftmax(dim=1)
        )

        # Q-value Networks
        self.q_net1 = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

        self.q_net2 = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, state):
        log_pi = self.policy_net(state)
        pi = log_pi.exp()
        q1 = self.q_net1(state)
        q2 = self.q_net2(state)
        return log_pi, pi, q1, q2

def compute_sac_loss(model, target_q_net1, target_q_net2, batch, alpha, gamma):
    states, actions, rewards, next_states, dones = batch

    # Get current Q estimates
    log_pi, pi, q1, q2 = model(states)
    q1 = q1.gather(1, actions)
    q2 = q2.gather(1, actions)

    # Compute target Q-values
    with torch.no_grad():
        log_pi_next, pi_next, q1_next_target, q2_next_target = model(next_states)
        min_q_next_target = torch.min(
            target_q_net1(next_states), target_q_net2(next_states)
        )
        v_next = (pi_next * (min_q_next_target - alpha * log_pi_next)).sum(dim=1, keepdim=True)
        q_target = rewards + gamma * (1 - dones) * v_next

    # Compute Q-function loss
    q1_loss = nn.MSELoss()(q1, q_target)
    q2_loss = nn.MSELoss()(q2, q_target)

    # Compute policy loss
    min_q = torch.min(model.q_net1(states), model.q_net2(states))
    policy_loss = (pi * (alpha * log_pi - min_q)).sum(dim=1).mean()

    # Entropy regularization term
    entropy = -(log_pi * pi).sum(dim=1).mean()
    entropy_loss = -alpha * entropy

    return q1_loss + q2_loss + policy_loss + entropy_loss, q1_loss.item(), q2_loss.item(), policy_loss.item(), entropy.item()

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.detach_()
        target_param.copy_(target_param * (1.0 - tau) + param * tau)

def off_policy_update(model, target_q_net1, target_q_net2, optimizer_policy, optimizer_q1, optimizer_q2, replay_buffer, batch_size, alpha, gamma, tau):
    if len(replay_buffer) < batch_size:
        return

    batch = replay_buffer.sample(batch_size)
    loss, q1_loss, q2_loss, policy_loss, entropy = compute_sac_loss(
        model, target_q_net1, target_q_net2, batch, alpha, gamma
    )

    optimizer_q1.zero_grad()
    optimizer_q2.zero_grad()
    optimizer_policy.zero_grad()
    loss.backward()
    optimizer_q1.step()
    optimizer_q2.step()
    optimizer_policy.step()

    # Soft update of target networks
    soft_update(target_q_net1, model.q_net1, tau)
    soft_update(target_q_net2, model.q_net2, tau)

    return q1_loss, q2_loss, policy_loss, entropy

################################# End of Part I ################################

print("============================================================================================")

################################### Training SAC ###################################

####### Initialize environment hyperparameters and SAC hyperparameters ######

print("Setting training environment:")

max_ep_len = 225            # max timesteps in one episode
gamma = 0.99                # discount factor
lr_actor = 0.0003           # learning rate for actor network
lr_critic = 0.001           # learning rate for critic networks
random_seed = 0             # set random seed
max_training_timesteps = 100000   # break from training loop if timesteps > max_training_timesteps
print_freq = max_ep_len * 4       # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2         # logging avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 4  # save model frequency (in num timesteps)
capacity = 10000                  # capacity of replay buffer
batch_size = 128                  # batch size for updates
tau = 0.005                       # target network update rate
alpha = 0.2                       # entropy temperature

env = Env()

# State space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# Action space dimension
action_dim = args.n_servers

## Note: print/save frequencies should be > than max_ep_len

###################### Saving files ######################

#### Create directories for logging and saving models
log_dir = "SAC_files"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_dir_1 = os.path.join(log_dir, 'resource_allocation', 'stability')
os.makedirs(log_dir_1, exist_ok=True)

log_dir_2 = os.path.join(log_dir, 'resource_allocation', 'reward')
os.makedirs(log_dir_2, exist_ok=True)

#### Get number of saving files in directory
run_num1 = len(next(os.walk(log_dir_1))[2])
run_num2 = len(next(os.walk(log_dir_2))[2])

#### Create new saving file for each run
log_f_name = os.path.join(log_dir_1, f'SAC_resource_allocation_log_{run_num1}.csv')
log_f_name2 = os.path.join(log_dir_2, f'SAC_resource_allocation_log_{run_num2}.csv')

print(f"Current logging run number for resource_allocation: {run_num1}")
print(f"Logging at: {log_f_name}")

#####################################################

################### Checkpointing ###################

run_num_pretrained = 0  #### Change this to prevent overwriting weights in same env_name folder

directory = "SAC_preTrained"
os.makedirs(directory, exist_ok=True)

directory = os.path.join(directory, 'resource_allocation')
os.makedirs(directory, exist_ok=True)

checkpoint_path = os.path.join(directory, f"SAC{NN_size}_resource_allocation_{random_seed}_{run_num_pretrained}.pth")
print(f"Save checkpoint path: {checkpoint_path}")

#####################################################

############# Print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print(f"Max training timesteps: {max_training_timesteps}")
print(f"Max timesteps per episode: {max_ep_len}")

print(f"Model saving frequency: {save_model_freq} timesteps")
print(f"Log frequency: {log_freq} timesteps")
print(f"Printing average reward over episodes in last: {print_freq} timesteps")

print("--------------------------------------------------------------------------------------------")

print(f"State space dimension: {state_dim}")
print(f"Action space dimension: {action_dim}")

print("--------------------------------------------------------------------------------------------")

print(f"Discount factor (gamma): {gamma}")

print("--------------------------------------------------------------------------------------------")

print(f"Optimizer learning rate (actor): {lr_actor}")
print(f"Optimizer learning rate (critic): {lr_critic}")

print("--------------------------------------------------------------------------------------------")
print(f"Setting random seed to {random_seed}")
torch.manual_seed(random_seed)
np.random.seed(random_seed)

#####################################################

print("============================================================================================")

################# Training Procedure ################

# Initialize SAC agent
model = ActorCritic(state_dim, action_dim).to(device)
target_q_net1 = copy.deepcopy(model.q_net1).to(device)
target_q_net2 = copy.deepcopy(model.q_net2).to(device)
optimizer_policy = optim.Adam(model.policy_net.parameters(), lr=lr_actor)
optimizer_q1 = optim.Adam(model.q_net1.parameters(), lr=lr_critic)
optimizer_q2 = optim.Adam(model.q_net2.parameters(), lr=lr_critic)
replay_buffer = EpisodicReplayMemory(capacity, max_ep_len)

start_time = datetime.now().replace(microsecond=0)
print(f"Started training at (GMT): {start_time}")

print("============================================================================================")

# Logging files
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,reward\n')
log_f2 = open(log_f_name2, "w+")
log_f2.write('episode,timestep,reward\n')

# Printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

# Start training loop
while time_step <= max_training_timesteps:
    print("New training episode:")
    sleep(0.1)  # We sleep to read the reward in console
    state = env.reset()
    current_ep_reward = 0

    for t in range(1, max_ep_len + 1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        log_pi, pi, q1, q2 = model(state_tensor)
        action_prob = pi.squeeze(0).cpu().detach().numpy()
        action = np.random.choice(action_dim, p=action_prob)
        action_tensor = torch.LongTensor([[action]]).to(device)
        next_state, reward, done, info = env.step(action)
        time_step += 1
        current_ep_reward += reward
        print(f"The current total episodic reward at timestep {time_step} is: {current_ep_reward}")
        sleep(0.1)  # We sleep to read the reward in console

        reward_tensor = torch.FloatTensor([[reward]]).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        done_tensor = torch.FloatTensor([[done]]).to(device)

        replay_buffer.push(state_tensor.squeeze(0).detach(), action_tensor, reward_tensor, next_state_tensor.squeeze(0).detach(), done_tensor)

        # Off-policy update
        off_policy_update(
            model,
            target_q_net1,
            target_q_net2,
            optimizer_policy,
            optimizer_q1,
            optimizer_q2,
            replay_buffer,
            batch_size,
            alpha,
            gamma,
            tau
        )

        # Log in logging file
        if time_step % log_freq == 0:
            # Log average reward till last episode
            if log_running_episodes > 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write(f'{i_episode},{time_step},{log_avg_reward}\n')
                log_f.flush()
                log_f2.write(f'{i_episode},{time_step},{log_avg_reward}\n')
                log_f2.flush()
                print("Saving reward to csv file")
                sleep(0.1)  # We sleep to read the reward in console

                log_running_reward = 0
                log_running_episodes = 0

        # Printing average reward
        if time_step % print_freq == 0:
            # Print average reward till last episode
            if print_running_episodes > 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print(f"Episode: {i_episode} \t\t Timestep: {time_step} \t\t Average Reward: {print_avg_reward}")
                sleep(0.1)  # We sleep to read the reward in console

                print_running_reward = 0
                print_running_episodes = 0

        # Save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print(f"Saving model at: {checkpoint_path}")
            sleep(0.1)  # We sleep to read the reward in console
            torch.save(model.state_dict(), checkpoint_path)
            print("Model saved")
            print("--------------------------------------------------------------------------------------------")

        state = next_state

        # Break if the episode is over
        if done:
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1

log_f.close()
log_f2.close()

################################ End of Part II ################################

print("============================================================================================")