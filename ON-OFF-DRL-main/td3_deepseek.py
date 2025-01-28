# -*- coding: utf-8 -*-
############################### Import libraries ###############################

import os
from datetime import datetime
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch.optim as optim
from env import Env
from argparser import args
from time import sleep

################################## set device to cpu or cuda ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")

################################## Define TD3 Policy ##################################

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.float32)
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_size)  # Input: state + one-hot action
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x, a):
        # Concatenate state and one-hot encoded action
        x = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TD3:
    def __init__(self, state_dim, action_dim, hidden_size=256, gamma=0.95, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=4):
        self.actor = Actor(state_dim, action_dim, hidden_size).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic1 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=1e-3)

        self.critic2 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, args.n_servers - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.actor(state)
        action = torch.multinomial(probs, 1).item()
        return action

    def update(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action).unsqueeze(1).to(device)  # Shape: (batch_size, 1)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        # Convert action to one-hot encoding
        action_one_hot = torch.zeros(action.size(0), args.n_servers).to(device)  # Shape: (batch_size, action_dim)
        action_one_hot.scatter_(1, action, 1)  # One-hot encode actions

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_action_probs = self.actor_target(next_state)
            next_action = torch.multinomial(next_action_probs, 1)  # Shape: (batch_size, 1)
            noise = torch.FloatTensor(np.random.normal(0, self.policy_noise, size=next_action.size())).to(device)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(0, args.n_servers - 1).long()  # Convert to long

            # Convert next_action to one-hot encoding
            next_action_one_hot = torch.zeros(next_action.size(0), args.n_servers).to(device)  # Shape: (batch_size, action_dim)
            next_action_one_hot.scatter_(1, next_action, 1)  # One-hot encode next actions

            # Compute the target Q value
            target_Q1 = self.critic1_target(next_state, next_action_one_hot)
            target_Q2 = self.critic2_target(next_state, next_action_one_hot)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1 = self.critic1(state, action_one_hot)
        current_Q2 = self.critic2(state, action_one_hot)

        # Compute critic loss
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        # Optimize the critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            action_probs = self.actor(state)
            action = torch.multinomial(action_probs, 1)  # Shape: (batch_size, 1)
            action_one_hot = torch.zeros(action.size(0), args.n_servers).to(device)  # Shape: (batch_size, action_dim)
            action_one_hot.scatter_(1, action, 1)  # One-hot encode actions
            actor_loss = -self.critic1(state, action_one_hot).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

################################### Training TD3 ###################################

####### initialize environment hyperparameters and TD3 hyperparameters ######

print("setting training environment : ")

max_ep_len = 225            # max timesteps in one episode
gamma = 0.95                # discount factor
random_seed = 0             # set random seed
max_training_timesteps = 100000   # break from training loop if timeteps > max_training_timesteps
print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # saving avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 4         # save model frequency (in num timesteps)
capacity = 100000
batch_size = 128

env = Env()

# state space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# action space dimension
action_dim = args.n_servers

## Note : print/save frequencies should be > than max_ep_len

###################### saving files ######################

#### saving files for multiple runs are NOT overwritten

log_dir = "TD3_DS_files"
if not os.path.exists(log_dir):
      os.makedirs(log_dir)

log_dir_1 = log_dir + '/' + 'resource_allocation' + '/' + 'stability' + '/'
if not os.path.exists(log_dir_1):
      os.makedirs(log_dir_1)
      
log_dir_2 = log_dir + '/' + 'resource_allocation' + '/' + 'reward' + '/'
if not os.path.exists(log_dir_2):
      os.makedirs(log_dir_2)

#### get number of saving files in directory
run_num = 0
current_num_files1 = next(os.walk(log_dir_1))[2]
run_num1 = len(current_num_files1)
current_num_files2 = next(os.walk(log_dir_2))[2]
run_num2 = len(current_num_files2)

#### create new saving file for each run 
log_f_name = log_dir_1 + '/TD3_' + 'resource_allocation' + "_log_" + str(run_num1) + ".csv"
log_f_name2 = log_dir_2 + '/TD3_' + 'resource_allocation' + "_log_" + str(run_num2) + ".csv"

print("current logging run number for " + 'resource_allocation' + " : ", run_num1)
print("logging at : " + log_f_name)

################### checkpointing ###################

run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "TD3_DS_preTrained"
if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + 'resource_allocation' + '/' 
if not os.path.exists(directory):
      os.makedirs(directory)

checkpoint_path = directory + "TD3_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

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

print("optimizer learning rate (actor) : ", 3e-4)
print("optimizer learning rate (critic) : ", 3e-4)

print("--------------------------------------------------------------------------------------------")
print("setting random seed to ", random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

#####################################################

print("============================================================================================")

################# training procedure ################
# initialize TD3 agent

td3 = TD3(state_dim, action_dim)
replay_buffer = ReplayBuffer(capacity)

start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")

# logging file
log_f = open(log_f_name,"w+")
log_f.write('episode,timestep,reward\n')
log_f2 = open(log_f_name2,"w+")
log_f2.write('episode,timestep,reward\n')

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

# start training loop
while time_step <= max_training_timesteps:
    print("New training episode:")
    sleep(0.1) # we sleep to read the reward in console
    state = env.reset()
    current_ep_reward = 0

    for t in range(1, max_ep_len+1):
        # select action with policy
        action = td3.select_action(state)
        next_state, reward, done, info = env.step(action)
        time_step +=1
        current_ep_reward += reward
        print("The current total episodic reward at timestep:", time_step, "is:", current_ep_reward)
        sleep(0.1) # we sleep to read the reward in console
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        # Update TD3
        if len(replay_buffer) > batch_size:
            td3.update(replay_buffer, batch_size=128)

        # log in logging file
        if time_step % log_freq == 0:
            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()
            log_f2.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f2.flush()
            print("Saving reward to csv file")
            sleep(0.1) # we sleep to read the reward in console
            log_running_reward = 0
            log_running_episodes = 0
            
        # printing average reward
        if time_step % print_freq == 0:
            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)
            
            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
            sleep(0.1) # we sleep to read the reward in console
            print_running_reward = 0
            print_running_episodes = 0
            
        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            sleep(0.1) # we sleep to read the reward in console
            torch.save(td3.actor.state_dict(), checkpoint_path)
            print("model saved")
            print("--------------------------------------------------------------------------------------------")
        
        # break; if the episode is over
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