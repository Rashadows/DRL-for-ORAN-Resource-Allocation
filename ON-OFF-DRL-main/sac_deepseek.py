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


################################## Define SAC Policy ##################################

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
    
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(QNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state, action):
        # Ensure action is a 2D tensor with shape (batch_size, 1)
        if action.dim() == 1:
            action = action.unsqueeze(1)
        # Concatenate state and action along the last dimension
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Update the compute_sac_loss function to handle input shapes correctly
def compute_sac_loss(policy, q1, q2, value, target_value, states, actions, rewards, next_states, dones, gamma=0.99, alpha=0.2):
    with torch.no_grad():
        # Get action probabilities for the next states
        next_state_action_probs = policy(next_states)
        # Sample actions from the action probabilities
        next_state_actions = torch.multinomial(next_state_action_probs, 1)
        # Compute log probabilities of the sampled actions
        next_state_log_probs = torch.log(next_state_action_probs.gather(1, next_state_actions) + 1e-10)
        
        # Compute target Q-values for the next states and actions
        q1_next_target = q1_target(next_states, next_state_actions)
        q2_next_target = q2_target(next_states, next_state_actions)
        q_next_target = torch.min(q1_next_target, q2_next_target)
        
        # Compute the target value for the current states
        target_value = rewards + gamma * (1 - dones) * (q_next_target - alpha * next_state_log_probs)
    
    # Compute Q-values for the current states and actions
    q1_value = q1(states, actions)
    q2_value = q2(states, actions)
    q1_loss = F.mse_loss(q1_value, target_value)
    q2_loss = F.mse_loss(q2_value, target_value)
    
    # Compute value loss
    value_loss = F.mse_loss(value(states), target_value)
    
    # Compute policy loss
    action_probs = policy(states)
    actions = torch.multinomial(action_probs, 1)
    log_probs = torch.log(action_probs.gather(1, actions) + 1e-10)
    
    q1_value = q1(states, actions)
    q2_value = q2(states, actions)
    q_value = torch.min(q1_value, q2_value)
    
    policy_loss = (alpha * log_probs - q_value).mean()
    
    return policy_loss, q1_loss, q2_loss, value_loss

# Update the update function to handle input shapes correctly
def update(batch_size):
    if batch_size > len(replay_buffer):
        return
    
    # Sample a batch from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
    
    # Debugging: Print shapes of tensors
    print(f"states shape: {states.shape}")
    print(f"actions shape: {actions.shape}")
    print(f"next_states shape: {next_states.shape}")
    
    # Compute SAC loss
    policy_loss, q1_loss, q2_loss, value_loss = compute_sac_loss(policy, q1, q2, value, target_value, states, actions, rewards, next_states, dones)
    
    # Update policy network
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()
    
    # Update Q1 network
    optimizer_q1.zero_grad()
    q1_loss.backward()
    optimizer_q1.step()
    
    # Update Q2 network
    optimizer_q2.zero_grad()
    q2_loss.backward()
    optimizer_q2.step()
    
    # Update value network
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()
    
    # Soft update target networks
    soft_update(q1_target, q1)
    soft_update(q2_target, q2)
    soft_update(target_value, value)

################################# End of Part I ################################

print("============================================================================================")


################################### Training SAC ###################################

####### initialize environment hyperparameters and SAC hyperparameters ######

print("setting training environment : ")

max_ep_len = 225            # max timesteps in one episode
gamma = 0.99                # discount factor
lr = 0.0003                 # learning rate
random_seed = 0             # set random seed
max_training_timesteps = 100000   # break from training loop if timeteps > max_training_timesteps
print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # saving avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 4         # save model frequency (in num timesteps)
capacity = 10000

## Note : print/save frequencies should be > than max_ep_len

###################### saving files ######################

#### saving files for multiple runs are NOT overwritten

log_dir = "SAC_DS_files"
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
log_f_name = log_dir_1 + '/SAC_' + 'resource_allocation' + "_log_" + str(run_num1) + ".csv"
log_f_name2 = log_dir_2 + '/SAC_' + 'resource_allocation' + "_log_" + str(run_num2) + ".csv"

print("current logging run number for " + 'resource_allocation' + " : ", run_num1)
print("logging at : " + log_f_name)

#####################################################

################### checkpointing ###################

run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "SAC_DS_preTrained"
if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + 'resource_allocation' + '/' 
if not os.path.exists(directory):
      os.makedirs(directory)


checkpoint_path = directory + "SAC256_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
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
# initialize SAC agent

value = ValueNetwork(state_dim).to(device)
target_value = ValueNetwork(state_dim).to(device)
policy = PolicyNetwork(state_dim, action_dim).to(device)

target_value.load_state_dict(value.state_dict())

optimizer_q1 = optim.Adam(q1.parameters(), lr=lr)
optimizer_q2 = optim.Adam(q2.parameters(), lr=lr)
optimizer_value = optim.Adam(value.parameters(), lr=lr)
optimizer_policy = optim.Adam(policy.parameters(), lr=lr)

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
num_steps    = 50
log_interval = 10

# start training loop
while time_step <= max_training_timesteps:
    print("New training episode:")
    sleep(0.1) # we sleep to read the reward in console
    state = env.reset()
    current_ep_reward = 0

    for t in range(1, max_ep_len+1):
        # select action with policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs = policy(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        next_state, reward, done, info = env.step(action)
        time_step +=1
        current_ep_reward += reward
        print("The current total episodic reward at timestep:", time_step, "is:", current_ep_reward)
        sleep(0.1) # we sleep to read the reward in console
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
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'q1_state_dict': q1.state_dict(),
                'q2_state_dict': q2.state_dict(),
                'value_state_dict': value.state_dict(),
                'target_value_state_dict': target_value.state_dict(),
                'optimizer_policy_state_dict': optimizer_policy.state_dict(),
                'optimizer_q1_state_dict': optimizer_q1.state_dict(),
                'optimizer_q2_state_dict': optimizer_q2.state_dict(),
                'optimizer_value_state_dict': optimizer_value.state_dict(),
            }, checkpoint_path)
            print("model saved")
            print("--------------------------------------------------------------------------------------------")
        state = next_state
        
        # break; if the episode is over
        if done:
            break

    update(batch_size=128)
    
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