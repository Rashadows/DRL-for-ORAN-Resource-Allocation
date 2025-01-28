# -*- coding: utf-8 -*-
############################### Import libraries ###############################

"""
Script to train TD3 with 256 hidden units
TD3 code adapted for discrete action space
"""

import os
from datetime import datetime
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


################################## Define TD3 Networks ##################################
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def push(self, data):
        # data is a tuple (state, action, next_state, reward, done)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = random.sample(range(len(self.storage)), batch_size)
        state, action, next_state, reward, done = [], [], [], [], []

        for i in ind:
            s, a, s_next, r, d = self.storage[i]
            state.append(np.asarray(s))
            action.append(np.asarray(a))
            next_state.append(np.asarray(s_next))
            reward.append(np.asarray(r))
            done.append(np.asarray(d))

        return (np.array(state),
                np.array(action),
                np.array(next_state),
                np.array(reward).reshape(-1,1),
                np.array(done).reshape(-1,1))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        action = self.actor(x)
        return action * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()

        # Q1 architecture
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Q2 architecture
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        q1 = self.critic1(xu)
        q2 = self.critic2(xu)
        return q1, q2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        q1 = self.critic1(xu)
        return q1

def td3_update(batch_size):
    if len(replay_buffer.storage) < batch_size:
        return

    state, action, next_state, reward, done = replay_buffer.sample(batch_size)
    state = torch.FloatTensor(state).to(device)
    action = torch.FloatTensor(action).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    with torch.no_grad():
        # Select action according to policy and add clipped noise
        noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
        next_action = (actor_target(next_state) + noise).clamp(-max_action, max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * gamma * target_Q).detach()

    # Get current Q estimates
    current_Q1, current_Q2 = critic(state, action)

    # Compute critic loss
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

    # Optimize the critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Delayed policy updates
    global policy_iteration_step
    if policy_iteration_step % policy_delay == 0:
        # Compute actor loss
        actor_loss = -critic.Q1(state, actor(state)).mean()

        # Optimize the actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    policy_iteration_step += 1

################################# End of Part I ################################

print("============================================================================================")


################################### Training TD3 with 256 neurons ###################################

####### initialize environment hyperparameters and TD3 hyperparameters ######

print("setting training environment : ")

max_ep_len = 225            # max timesteps in one episode
gamma = 0.99                # discount factor
lr_actor = 0.0003           # learning rate for actor network
lr_critic = 0.0003          # learning rate for critic network
random_seed = 0             # set random seed
max_training_timesteps = 100000   # break from training loop if timeteps > max_training_timesteps
print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # saving avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 4         # save model frequency (in num timesteps)
batch_size = 256
tau = 0.005              # Target network update rate
policy_noise = 0.2       # Noise added to target policy during critic update
noise_clip = 0.5         # Range to clip target policy noise
policy_delay = 2         # Delayed policy updates parameter
exploration_noise = 0.1  # Exploration noise during training

env = Env()

# state space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# action space dimension
action_dim = args.n_servers

# Max action value
max_action = float(action_dim - 1)

## Note : print/save frequencies should be > than max_ep_len

###################### saving files ######################

#### saving files for multiple runs are NOT overwritten

log_dir = "TD3_GPT_files"
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

#####################################################

################### checkpointing ###################

run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "TD3_GPT_preTrained"
if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + 'resource_allocation' + '/' 
if not os.path.exists(directory):
      os.makedirs(directory)


checkpoint_path = directory + "TD3_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
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

print("optimizer learning rate (actor) : ", lr_actor)
print("optimizer learning rate (critic) : ", lr_critic)

print("--------------------------------------------------------------------------------------------")
print("setting random seed to ", random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

#####################################################

print("============================================================================================")

################# training procedure ################

# Initialize policy and value networks
actor = Actor(state_dim, action_dim, max_action).to(device)
actor_target = Actor(state_dim, action_dim, max_action).to(device)
actor_target.load_state_dict(actor.state_dict())

critic = Critic(state_dim, action_dim).to(device)
critic_target = Critic(state_dim, action_dim).to(device)
critic_target.load_state_dict(critic.state_dict())

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

replay_buffer = ReplayBuffer()

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
policy_iteration_step = 0

# start training loop
while time_step <= max_training_timesteps:
    print("New training episode:")
    sleep(0.1) # we sleep to read the reward in console
    state = env.reset()
    current_ep_reward = 0

    for t in range(1, max_ep_len+1):
        # select action with policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor(state_tensor).detach().cpu().numpy()[0]

        # Add exploration noise
        action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-max_action, max_action)

        # Discretize action for the environment
        action_discrete = np.argmax(action)

        next_state, reward, done, info = env.step(action_discrete)
        time_step += 1
        current_ep_reward += reward
        print("The current total episodic reward at timestep:", time_step, "is:", current_ep_reward)
        sleep(0.1) # we sleep to read the reward in console

        # Store data in replay buffer
        replay_buffer.push((state, action, next_state, reward, float(done)))

        # TD3 update
        td3_update(batch_size)

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
            torch.save({'actor_state_dict': actor.state_dict(),
                        'critic_state_dict': critic.state_dict(),
                        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                        }, checkpoint_path)
            print("model saved")
            print("--------------------------------------------------------------------------------------------")
        state = next_state

        print_running_reward += reward
        log_running_reward += reward

        # break; if the episode is over
        if done:
            break

    print_running_episodes += 1
    log_running_episodes += 1

    i_episode += 1

log_f.close()
log_f2.close()