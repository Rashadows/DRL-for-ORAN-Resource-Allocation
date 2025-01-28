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
from torch.optim.lr_scheduler import StepLR 
from env import Env
from argparser import args
from time import sleep

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

################################## Define DQN Policy ##################################

class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x):
        return self.network(x)

def compute_dqn_loss(batch, model, target_model, gamma=0.95):
    state, action, reward, next_state, done = zip(*batch)

    state = torch.FloatTensor(np.array(state)).to(device)
    next_state = torch.FloatTensor(np.array(next_state)).to(device)
    action = torch.LongTensor(np.array(action)).to(device)
    reward = torch.FloatTensor(np.array(reward)).to(device)
    done = torch.FloatTensor(np.array(done)).to(device)

    q_values = model(state)
    next_q_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def update_target_model(model, target_model):
    target_model.load_state_dict(model.state_dict())

################################# End of Part I ################################

print("============================================================================================")

################################### Training DQN with 256 neurons###################################

####### initialize environment hyperparameters and DQN hyperparameters ######

print("setting training environment : ")

max_ep_len = 225  # max timesteps in one episode
gamma = 0.95  # discount factor
lr = 0.0003  # learning rate
random_seed = 0  # set random seed
max_training_timesteps = 100000  # break from training loop if timeteps > max_training_timesteps
print_freq = max_ep_len * 4  # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2  # saving avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 4  # save model frequency (in num timesteps)
capacity = 50000
batch_size = 32

env = Env()

# state space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# action space dimension
action_dim = args.n_servers

## Note : print/save frequencies should be > than max_ep_len

###################### saving files ######################

#### saving files for multiple runs are NOT overwritten

log_dir = "DQN_DS_files"
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
log_f_name = log_dir_1 + '/DQN_' + 'resource_allocation' + "_log_" + str(run_num1) + ".csv"
log_f_name2 = log_dir_2 + '/DQN_' + 'resource_allocation' + "_log_" + str(run_num2) + ".csv"

print("current logging run number for " + 'resource_allocation' + " : ", run_num1)
print("logging at : " + log_f_name)

#####################################################

################### checkpointing ###################

run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

directory = "DQN_DS_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + 'resource_allocation' + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "DQN256_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
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
# initialize DQN agent

model = DQN(args.n_servers * args.n_resources + args.n_resources + 1, args.n_servers).to(device)
target_model = DQN(args.n_servers * args.n_resources + args.n_resources + 1, args.n_servers).to(device)
update_target_model(model, target_model)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)  # Initialize the scheduler
replay_buffer = ReplayMemory(capacity)

start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")

# logging file
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,reward\n')
log_f2 = open(log_f_name2, "w+")
log_f2.write('episode,timestep,reward\n')

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0
num_steps = 50
log_interval = 10

# start training loop
while time_step <= max_training_timesteps:
    print("New training episode:")
    sleep(0.1)  # we sleep to read the reward in console
    state = env.reset()
    current_ep_reward = 0

    for t in range(1, max_ep_len + 1):
        # select action with policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = model(state_tensor)
        action = q_values.argmax().item()
        next_state, reward, done, info = env.step(action)
        time_step += 1
        current_ep_reward += reward
        print("The current total episodic reward at timestep:", time_step, "is:", current_ep_reward)
        sleep(0.1)  # we sleep to read the reward in console
        replay_buffer.push(state, action, reward, next_state, done)

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
            sleep(0.1)  # we sleep to read the reward in console
            log_running_reward = 0
            log_running_episodes = 0

        # printing average reward
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
            sleep(0.1)  # we sleep to read the reward in console
            print_running_reward = 0
            print_running_episodes = 0

        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            sleep(0.1)  # we sleep to read the reward in console
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved")
            print("--------------------------------------------------------------------------------------------")
        state = next_state

        # break; if the episode is over
        if done:
            break

    if len(replay_buffer) > num_steps:
        batch = replay_buffer.sample(num_steps)
        loss = compute_dqn_loss(batch, model, target_model)
        scheduler.step()  # Update the learning rate scheduler

    if time_step % 1000 == 0:
        update_target_model(model, target_model)

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1
    time_step += num_steps

log_f.close()
log_f2.close()

################################ End of Part II ################################

print("============================================================================================")