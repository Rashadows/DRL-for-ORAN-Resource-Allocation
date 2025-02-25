# -*- coding: utf-8 -*-
############################### Import libraries ###############################

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

################################## set device to cpu or cuda ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")

NN_size = 32  # You can change this to 32, 64, or 256 as needed

################################## Define TD3 Networks ##################################
class ReplayBuffer(object):
    def __init__(self, max_size):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def push(self, data):
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

        return (np.array(state, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(reward, dtype=np.float32).reshape(-1, 1),
                np.array(done, dtype=np.float32).reshape(-1, 1))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=NN_size):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        return self.actor(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=NN_size):
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
        return self.critic1(xu), self.critic2(xu)

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        return self.critic1(xu)

################################## Training Functions ##################################

def select_action(state, exploration_noise=0.5, temperature=1.0):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    logits = actor(state_tensor)
    
    # Enhanced exploration with higher noise
    noise = torch.randn_like(logits) * exploration_noise
    noisy_logits = logits + noise
    
    # Temperature-controlled action selection
    action_probs = F.softmax(noisy_logits / temperature, dim=1)
    action_dist = torch.distributions.Categorical(action_probs)
    return action_dist.sample().item()

def td3_update(batch_size):
    if len(replay_buffer.storage) < batch_size:
        return

    # Sample from replay buffer
    state, action, next_state, reward, done = replay_buffer.sample(batch_size)
    state = torch.FloatTensor(state).to(device)
    action = torch.FloatTensor(action).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    with torch.no_grad():
        # Target policy smoothing with noise
        next_action_logits = actor_target(next_state)
        noise = torch.randn_like(next_action_logits) * policy_noise
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        noisy_next_action_logits = next_action_logits + noise
        
        next_action_probs = F.softmax(noisy_next_action_logits, dim=1)
        action_dist = torch.distributions.Categorical(next_action_probs)
        next_action_discrete = action_dist.sample()

        # One-hot encoding for critic
        next_action_one_hot = torch.zeros(batch_size, action_dim).to(device)
        next_action_one_hot.scatter_(1, next_action_discrete.unsqueeze(1), 1.0)

        # Calculate target Q-values
        target_Q1, target_Q2 = critic_target(next_state, next_action_one_hot)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * gamma * target_Q).detach()

    # Critic update
    current_Q1, current_Q2 = critic(state, action)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Delayed policy updates with entropy regularization
    global policy_iteration_step
    if policy_iteration_step % policy_delay == 0:
        action_logits = actor(state)
        action_probs = F.softmax(action_logits, dim=1)
        
        # Calculate policy entropy
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1, keepdim=True)
        
        # Calculate expected Q-values
        all_actions = torch.eye(action_dim).to(device)
        state_expanded = state.unsqueeze(1).expand(-1, action_dim, -1)
        q_values = critic.Q1(state_expanded.reshape(-1, state_dim), 
                           all_actions.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, action_dim))
        q_values = q_values.view(batch_size, action_dim)
        
        # Entropy-regularized actor loss
        expected_Q = torch.sum(action_probs * q_values, dim=1, keepdim=True)
        actor_loss = -expected_Q.mean() - 0.1 * entropy.mean()
        
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    policy_iteration_step += 1

################################### Training TD3 ###################################

####### initialize environment hyperparameters and TD3 hyperparameters ######

print("setting training environment : ")

# Hyperparameters (Modified for increased variability)
capacity = 1e6
max_ep_len = 225
gamma = 0.95
lr_actor = 0.0001
lr_critic = 0.001
random_seed = 0
max_training_timesteps = 100000
tau = 0.01              # Increased target network update rate
policy_delay = 2        # More frequent policy updates
policy_noise = 0.4      # Increased target noise
noise_clip = 0.7        # Wider noise clipping
exploration_noise = 0.5 # Higher exploration noise
batch_size = 512  # increased batch size
policy_delay = 4  # Delayed policy updates parameter (adjusted)

print_freq = max_ep_len * 4  # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2  # saving avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 4  # save model frequency (in num timesteps)

env = Env()

# state space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# action space dimension
action_dim = args.n_servers

## Note : print/save frequencies should be > than max_ep_len

###################### saving files ######################

#### saving files for multiple runs are NOT overwritten

log_dir = "TD3_files"
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
log_f_name = log_dir_1 + '/TD3_' + str(NN_size) + '_resource_allocation' + "_log_" + str(run_num1) + ".csv"
log_f_name2 = log_dir_2 + '/TD3_' + str(NN_size) + '_resource_allocation' + "_log_" + str(run_num2) + ".csv"

print("current logging run number for " + 'resource_allocation' + " : ", run_num1)
print("logging at : " + log_f_name)

################### checkpointing ###################

run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

directory = "TD3_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + 'resource_allocation' + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "TD3{}_{}_{}_{}.pth".format(NN_size, 'resource_allocation', random_seed, run_num_pretrained)
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


# Initialize networks
actor = Actor(state_dim, action_dim).to(device)
actor_target = Actor(state_dim, action_dim).to(device)
actor_target.load_state_dict(actor.state_dict())
critic = Critic(state_dim, action_dim).to(device)
critic_target = Critic(state_dim, action_dim).to(device)
critic_target.load_state_dict(critic.state_dict())

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

replay_buffer = ReplayBuffer(capacity)

cycle_length = max_training_timesteps//4  # 4 full cycles over 100k steps

policy_iteration_step = 0

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
policy_iteration_step = 0

# start training loop
while time_step <= max_training_timesteps:
    print("New training episode:")
    state = env.reset()
    current_ep_reward = 0   

    for t in range(1, max_ep_len + 1):
        # Select action with exploration
        current_temperature = 1.0 + 0.5 * np.sin(2 * np.pi * time_step / cycle_length)
        action_discrete = select_action(state, exploration_noise, current_temperature)
        next_state, reward, done, info = env.step(action_discrete)
        current_ep_reward += reward
        print("The current total episodic reward at timestep:", time_step, "is:", current_ep_reward)
        time_step += 1

        # One-hot encode the discrete action for the Critic
        action_one_hot = np.zeros(action_dim, dtype=np.float32)
        action_one_hot[action_discrete] = 1.0
        replay_buffer.push((state, action_one_hot, next_state, reward, float(done)))

        # TD3 update
        td3_update(batch_size)

        # Logging and saving logic (unchanged)
        if time_step % log_freq == 0:
            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes if log_running_episodes > 0 else 0
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()
            log_f2.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f2.flush()
            print("Saving reward to csv file")
            log_running_reward = 0
            log_running_episodes = 0

        # printing average reward
        if time_step % print_freq == 0:
            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes if print_running_episodes > 0 else 0
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
            print_running_reward = 0
            print_running_episodes = 0

        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            }, checkpoint_path)
            print("model saved")
            print("--------------------------------------------------------------------------------------------")

        state = next_state

        if done:
            break

    print_running_reward += current_ep_reward
    log_running_reward += current_ep_reward

    print_running_episodes += 1
    log_running_episodes += 1

    i_episode += 1

log_f.close()
log_f2.close()

################################ End of Part II ################################

print("============================================================================================")