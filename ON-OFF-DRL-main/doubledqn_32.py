# -*- coding: utf-8 -*-
############################### Import libraries ###############################

import os
from datetime import datetime
from collections import deque
import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
from env import Env  # Ensure you have this file with the Env class defined
from argparser import args  # Ensure you have this file with argument parsing

################################## set device to cpu or cuda ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")

NN_size = 32
################################## Define Double DQN Policy ##################################

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # Store transitions in a circular buffer
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state.detach(), action.detach(), reward.detach(), next_state.detach(), done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=NN_size):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.LayerNorm(hidden_size),    # Replaced BatchNorm1d with LayerNorm
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x):
        return self.layers(x)

def compute_double_dqn_loss(batch, online_net, target_net, gamma, device):
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states).to(device)
    actions = torch.cat(actions).to(device)
    rewards = torch.cat(rewards).to(device)
    next_states = torch.cat(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    # Compute Q(s_t, a_t) using the online network
    q_values = online_net(states).gather(1, actions)

    # Compute Q values for next states using online network to select best action
    next_state_actions = online_net(next_states).max(1)[1].unsqueeze(1)

    # Compute target Q values using target network and the selected action
    next_q_values = target_net(next_states).gather(1, next_state_actions).detach()

    # Compute expected Q values
    expected_q_values = rewards + (gamma * next_q_values * (1 - dones))

    # Compute loss
    loss = nn.MSELoss()(q_values, expected_q_values)

    return loss

################################# End of Part I ################################

print("============================================================================================")

################################### Training Double DQN with Enhancements ###################################

####### initialize environment hyperparameters and Double DQN hyperparameters ######

print("setting training environment : ")

max_ep_len = 225            # max timesteps in one episode
gamma = 0.99                # discount factor
lr = 0.0001                 # initial learning rate
random_seed = 0             # set random seed
max_training_timesteps = 100000   # break from training loop if timesteps > max_training_timesteps
print_freq = max_ep_len * 4        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2          # saving avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 4    # save model frequency (in num timesteps)
capacity = 100000
batch_size = 128
target_update_freq = 1000     # update target network every ... timesteps
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 10000
tau = 0.005

# Learning Rate Scheduler parameters
step_size = 25000         # Decay LR every 25,000 steps
gamma_scheduler = 0.3    # Decay LR by a factor of 0.5

env = Env()

# state space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# action space dimension
action_dim = args.n_servers

## Note : print/save frequencies should be > than max_ep_len

###################### saving files ######################

#### saving files for multiple runs are NOT overwritten

log_dir = "Double_DQN_files"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_dir_1 = os.path.join(log_dir, 'resource_allocation', 'stability')
if not os.path.exists(log_dir_1):
    os.makedirs(log_dir_1)

log_dir_2 = os.path.join(log_dir, 'resource_allocation', 'reward')
if not os.path.exists(log_dir_2):
    os.makedirs(log_dir_2)


#### get number of saving files in directory
run_num = 0
current_num_files1 = next(os.walk(log_dir_1))[2]
run_num1 = len(current_num_files1)
current_num_files2 = next(os.walk(log_dir_2))[2]
run_num2 = len(current_num_files2)


#### create new saving file for each run
log_f_name = os.path.join(log_dir_1, f'Double_DQN_{NN_size}_resource_allocation_log_{run_num1}.csv')

print("current logging run number for " + 'resource_allocation' + " : ", run_num1)
print("logging at : " + log_f_name)

#####################################################

################### checkpointing ###################

run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "Double_DQN_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = os.path.join(directory, 'resource_allocation')
if not os.path.exists(directory):
    os.makedirs(directory)


checkpoint_path = os.path.join(directory, f"Double_DQN{NN_size}_resource_allocation_{random_seed}_{run_num_pretrained}.pth")
print("save checkpoint path : " + checkpoint_path)

#####################################################

############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)

print("--------------------------------------------------------------------------------------------")

print("discount factor (gamma) : ", gamma)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate : ", lr)

print("--------------------------------------------------------------------------------------------")
print("setting random seed to ", random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

#####################################################

print("============================================================================================")

################# training procedure ################
# initialize Double DQN agent

online_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(online_net.state_dict())

optimizer = optim.Adam(online_net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma_scheduler)  # Initialize LR Scheduler
replay_buffer = ReplayMemory(capacity)

start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")


# logging files
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,reward\n')

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

# Function to get current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# start training loop
while time_step <= max_training_timesteps:
    print("New training episode:")
    state = env.reset()
    current_ep_reward = 0

    for t in range(1, max_ep_len + 1):
        # select action with epsilon-greedy policy
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * time_step / epsilon_decay)
        if random.random() < epsilon:
            action = torch.tensor([[random.randrange(action_dim)]], dtype=torch.long).to(device)
        else:
            # Action selection with LayerNorm (no issues with batch size =1)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = online_net(state_tensor)
                action = q_values.max(1)[1].view(1, 1)

        next_state, reward, done, info = env.step(action.item())
        time_step += 1
        current_ep_reward += reward
        print("The current total episodic reward at timestep:", time_step, "is:", current_ep_reward)

        # Store transition in replay buffer
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        reward_tensor = torch.FloatTensor([[reward]]).to(device)
        done_tensor = torch.FloatTensor([[done]]).to(device)
        replay_buffer.push(state_tensor, action, reward_tensor, next_state_tensor, done)

        # Update the online network
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = compute_double_dqn_loss(batch, online_net, target_net, gamma, device)
            optimizer.zero_grad()
            loss.backward()

            # Apply Gradient Clipping
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # Step the scheduler after each optimizer.step()

            # # Optional: Print the current learning rate
            # current_lr = get_lr(optimizer)
            # print(f"Current Learning Rate: {current_lr}")

        # Soft update target network
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

        # log in logging file
        if time_step % log_freq == 0:
            # log average reward till last episode
            if log_running_episodes > 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()
                print("Saving reward to csv file")
                log_running_reward = 0
                log_running_episodes = 0

        # printing average reward
        if time_step % print_freq == 0:
            # print average reward till last episode
            if print_running_episodes > 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                print_running_reward = 0
                print_running_episodes = 0

        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            torch.save({
                'model_state_dict': online_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'time_step': time_step,
                'episode': i_episode
            }, checkpoint_path)
            print("model saved")
            print("--------------------------------------------------------------------------------------------")

        state = next_state

        # break if the episode is over
        if done:
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1

log_f.close()

################################ End of Part II ################################

print("============================================================================================")