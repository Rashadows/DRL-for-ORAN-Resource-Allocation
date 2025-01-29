# -*- coding: utf-8 -*-
############################### Import Libraries ###############################


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

################################## Set Device to CPU or CUDA ##################################

print("============================================================================================")

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    torch.cuda.empty_cache()
    print("Device set to: " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to: cpu")

print("============================================================================================")

NN_size = 256

################################## Define SAC Components ##################################

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            torch.FloatTensor(state).to(device),
            torch.LongTensor([action]).to(device),
            torch.FloatTensor([reward]).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor([done]).to(device)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return state, action.squeeze(-1), reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(ActorCritic, self).__init__()

        # Policy Network (outputs log probabilities over actions)
        self.policy_net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.LogSoftmax(dim=-1)
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
        log_probs = self.policy_net(state)
        probs = log_probs.exp()
        q_values1 = self.q_net1(state)
        q_values2 = self.q_net2(state)
        return log_probs, probs, q_values1, q_values2

def compute_sac_loss(model, target_q_net1, target_q_net2, batch, alpha, gamma):
    states, actions, rewards, next_states, dones = batch

    # Get action probabilities and log probabilities
    log_probs, probs, q_values1, q_values2 = model(states)

    # Gather the Q-values for the actions taken
    actions = actions.unsqueeze(-1)
    q_value1 = q_values1.gather(1, actions)
    q_value2 = q_values2.gather(1, actions)

    with torch.no_grad():
        # Next state actions and Q values
        log_probs_next, probs_next, q1_next, q2_next = model(next_states)
        min_q_next = torch.min(q1_next, q2_next) - alpha * log_probs_next
        v_next = (probs_next * min_q_next).sum(dim=1, keepdim=True)
        q_target = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * v_next

    # Compute Q-function loss
    q_loss1 = nn.MSELoss()(q_value1, q_target)
    q_loss2 = nn.MSELoss()(q_value2, q_target)

    # Policy loss
    min_q = torch.min(q_values1, q_values2)  # [batch_size, num_actions]
    policy_loss = (probs * (alpha * log_probs - min_q)).sum(dim=1).mean()

    # Entropy (for logging)
    entropy = -(log_probs * probs).sum(dim=1).mean()

    total_loss = q_loss1 + q_loss2 + policy_loss

    return total_loss, q_loss1.item(), q_loss2.item(), policy_loss.item(), entropy.item()

def soft_update(target_net, source_net, tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def off_policy_update(model, target_q_net1, target_q_net2, optimizer, replay_buffer, batch_size, alpha, gamma, tau):
    if len(replay_buffer) < batch_size:
        return

    batch = replay_buffer.sample(batch_size)
    loss, q_loss1, q_loss2, policy_loss, entropy = compute_sac_loss(
        model, target_q_net1, target_q_net2, batch, alpha, gamma
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Soft update of target networks
    soft_update(target_q_net1.q_net1, model.q_net1, tau)
    soft_update(target_q_net2.q_net2, model.q_net2, tau)

    return q_loss1, q_loss2, policy_loss, entropy

################################# End of Part I ################################

print("============================================================================================")

################################### Training SAC ###################################

####### Initialize Environment and SAC Hyperparameters ######

print("Setting training environment:")

env = Env()

# State space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# Action space dimension
action_dim = args.n_servers

max_ep_len = 225            # Max timesteps in one episode
gamma = 0.99                # Discount factor
lr = 3e-4                   # Learning rate for all networks
lr_alpha = 3e-4             # Learning rate for alpha
random_seed = 0             # Set random seed
max_training_timesteps = 100000   # Max training timesteps
print_freq = max_ep_len * 4       # Print avg reward in the interval
log_freq = max_ep_len * 2         # Log avg reward in the interval
save_model_freq = max_ep_len * 4  # Save model frequency
capacity = 100000                 # Replay buffer capacity
batch_size = 256                  # Batch size for updates
initial_random_steps = 1000       # Initial steps with random policy
target_entropy = -action_dim      # Target entropy
tau = 0.005                       # Soft update parameter


## Note: print/save frequencies should be > than max_ep_len

###################### Saving Files ######################

#### Saving files for multiple runs are NOT overwritten

log_dir = "SAC_files"
os.makedirs(log_dir, exist_ok=True)

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

run_num_pretrained = 0      #### Change this to prevent overwriting weights in same env_name folder

directory = "SAC_preTrained"
os.makedirs(directory, exist_ok=True)

directory = os.path.join(directory, 'resource_allocation')
os.makedirs(directory, exist_ok=True)

checkpoint_path = os.path.join(directory, f"SAC{NN_size}_resource_allocation_{random_seed}_{run_num_pretrained}.pth")
print(f"Save checkpoint path: {checkpoint_path}")

#####################################################

############# Print All Hyperparameters #############

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

print(f"Optimizer learning rate: {lr}")

print("--------------------------------------------------------------------------------------------")
print(f"Setting random seed to {random_seed}")
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

#####################################################

print("============================================================================================")

################# Training Procedure ################

# Initialize SAC agent
model = ActorCritic(state_dim, action_dim).to(device)
target_q_net1 = ActorCritic(state_dim, action_dim).to(device)
target_q_net2 = ActorCritic(state_dim, action_dim).to(device)
target_q_net1.load_state_dict(model.state_dict())
target_q_net2.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=lr)

# Automatic entropy tuning
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optimizer = optim.Adam([log_alpha], lr=lr_alpha)

replay_buffer = ReplayMemory(capacity)

start_time = datetime.now().replace(microsecond=0)
print(f"Started training at (GMT): {start_time}")

print("============================================================================================")

# Logging files
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,reward\n')
log_f.flush()
log_f2 = open(log_f_name2, "w+")
log_f2.write('episode,timestep,reward\n')
log_f2.flush()

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
    state = env.reset()
    current_ep_reward = 0

    for t in range(1, max_ep_len + 1):
        if time_step < initial_random_steps:
            action = random.choice(range(action_dim))
            action_prob = np.ones(action_dim) / action_dim
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                log_probs, probs, _, _ = model(state_tensor)
            action_prob = probs.cpu().numpy().flatten()
            action = np.random.choice(action_dim, p=action_prob)

        next_state, reward, done, info = env.step(action)
        time_step += 1
        current_ep_reward += reward
        print(f"The current total episodic reward at timestep: {time_step} is: {current_ep_reward}")

        # Store transition in the replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state

        # Off-policy update
        if time_step >= initial_random_steps:
            batch = replay_buffer.sample(batch_size)
    
            # Compute alpha loss and update alpha
            state_batch = batch[0]
            log_pi, pi, _, _ = model(state_batch)
            alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()

            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            alpha = log_alpha.exp().detach()

            # Update SAC networks
            off_policy_update(model, target_q_net1, target_q_net2, optimizer, replay_buffer, batch_size, alpha, gamma, tau)

        # Log in logging file
        if time_step % log_freq == 0 and log_running_episodes > 0:
            # Log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write(f'{i_episode},{time_step},{log_avg_reward}\n')
            log_f.flush()
            log_f2.write(f'{i_episode},{time_step},{log_avg_reward}\n')
            log_f2.flush()
            print("Saving reward to csv file")
            log_running_reward = 0
            log_running_episodes = 0

        # Printing average reward
        if time_step % print_freq == 0 and print_running_episodes > 0:
            # Print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print(f"Episode: {i_episode} \t\t Timestep: {time_step} \t\t Average Reward: {print_avg_reward}")
            print_running_reward = 0
            print_running_episodes = 0

        # Save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print(f"Saving model at: {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            print("Model saved")
            print("--------------------------------------------------------------------------------------------")

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