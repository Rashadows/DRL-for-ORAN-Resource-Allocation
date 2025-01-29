import os
import math
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gc
from gym_env import ClusterOptimizationEnv
from argparser import args
from time import sleep
from datetime import datetime
# from filter_env import makeFilteredEnv

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

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

class EpsilonGreedyNoise:
    def __init__(self, action_dim, epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.995):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate
        self.action_dim = action_dim

    def get_action(self, actor, state):
        if np.random.rand() < self.epsilon:
            # Random action (exploration)
            return np.random.randint(self.action_dim)
        else:
            # Action with highest probability (exploitation)
            with torch.no_grad():
                action_probs = actor(state)
                return torch.argmax(action_probs, dim=-1).item()

    def decay(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.decay_rate)

def fanin_init(size, fan_in):
    return torch.Tensor(size).uniform_(-1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in))

class BatchNormLayer(nn.Module):
    def __init__(self, num_features, activation=None):
        super(BatchNormLayer, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.activation = activation

    def forward(self, x):
        if x.size(0) == 1:
            # If batch size is 1, use running statistics
            self.bn.track_running_stats = True
            x = self.bn(x)
        else:
            # If batch size is greater than 1, use batch statistics
            self.bn.track_running_stats = False
            x = self.bn(x)
        return self.activation(x) if self.activation else x
    
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layer1_size, layer2_size):
        super(ActorNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, layer1_size)
        self.bn1 = BatchNormLayer(layer1_size, activation=nn.ReLU())
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.bn2 = BatchNormLayer(layer2_size, activation=nn.ReLU())
        self.layer3 = nn.Linear(layer2_size, action_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.layer3(x)
        return torch.softmax(x, dim=-1)  # Softmax for discrete action probabilities

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.time_step = 0

        # Create Q network
        self.q_network = self.create_q_network(state_dim, action_dim)
        self.q_network_target = self.create_q_network(state_dim, action_dim)

        # Copy parameters to target network
        self.update_target()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE_CRT)

        # Loss function
        self.criterion = nn.MSELoss()

    def create_q_network(self, state_dim, action_dim):
        class QNetwork(nn.Module):
            def __init__(self):
                super(QNetwork, self).__init__()
                self.fc1 = nn.Linear(state_dim, LAYER1_SIZE)
                self.fc2 = nn.Linear(LAYER1_SIZE, LAYER2_SIZE)
                self.action_fc = nn.Linear(action_dim, LAYER2_SIZE)
                self.fc3 = nn.Linear(LAYER2_SIZE, 1)

                self.fc1.weight.data.uniform_(-1 / math.sqrt(state_dim), 1 / math.sqrt(state_dim))
                self.fc2.weight.data.uniform_(-1 / math.sqrt(LAYER1_SIZE), 1 / math.sqrt(LAYER1_SIZE))
                self.action_fc.weight.data.uniform_(-1 / math.sqrt(action_dim), 1 / math.sqrt(action_dim))
                self.fc3.weight.data.uniform_(-3e-3, 3e-3)

            def forward(self, state, action):
                x = torch.relu(self.fc1(state))
                print(f"x shape: {x.shape}, action : {action}")
                # action_one_hot = F.one_hot(action, num_classes=action_dim).float()
                # print(f"x shape: {x.shape}, action shape: {action.shape}")
                x = torch.relu(self.fc2(x) + self.action_fc(action))
                return self.fc3(x)

        return QNetwork()

    def update_target(self):
        self.q_network_target.load_state_dict(self.q_network.state_dict())

    def forward(self, state, action):
        return self.q_network(state, action)

    def train(self, y_batch, state_batch, action_batch):
        self.time_step += 1

        # Convert inputs to tensors
        y_batch = torch.tensor(y_batch, dtype=torch.float32)
        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.float32)

        # Forward pass
        q_values = self.q_network(state_batch, action_batch)

        # Compute loss
        loss = self.criterion(q_values, y_batch) + self.l2_weight_decay()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def gradients(self, state_batch, action_batch):
        state_batch = torch.tensor(state_batch, dtype=torch.float32, requires_grad=True)
        action_batch = torch.tensor(action_batch, dtype=torch.float32, requires_grad=True)

        q_values = self.q_network(state_batch, action_batch)
        q_values.backward(torch.ones_like(q_values))
        return action_batch.grad

    def target_q(self, state_batch, action_batch):
        with torch.no_grad():
            state_batch = torch.tensor(state_batch, dtype=torch.float32)
            action_batch = torch.tensor(action_batch, dtype=torch.float32)
            return self.q_network_target(state_batch, action_batch)

    def q_value(self, state_batch, action_batch):
        with torch.no_grad():
            state_batch = torch.tensor(state_batch, dtype=torch.float32)
            action_batch = torch.tensor(action_batch, dtype=torch.float32)
            return self.q_network(state_batch, action_batch)

    def l2_weight_decay(self):
        weight_decay = 0
        for param in self.q_network.parameters():
            weight_decay += torch.sum(param**2)
        return L2 * weight_decay


class DDPG:
    def __init__(self):
        self.actor_network = ActorNetwork(state_dim, action_dim, LAYER1_SIZE, LAYER2_SIZE)
        self.actor_target_network = ActorNetwork(state_dim, action_dim, LAYER1_SIZE, LAYER2_SIZE)
        self.noise = EpsilonGreedyNoise(action_dim)  # Initialize the noise attribute
        self.critic_network = CriticNetwork(state_dim, action_dim)

        # Initialize target networks
        self.actor_target_network = ActorNetwork(state_dim, action_dim, LAYER1_SIZE, LAYER2_SIZE)
        self.critic_target_network = CriticNetwork(state_dim, action_dim)

        # Copy weights to target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.q_network.load_state_dict(self.critic_network.q_network.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=LEARNING_RATE_ACT)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=LEARNING_RATE_CRT)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Exploration noise
        self.exploration_noise = EpsilonGreedyNoise(action_dim)

        # Loss function
        self.critic_loss_fn = nn.MSELoss()

    def train(self):
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = torch.tensor(np.array([data[0] for data in minibatch]), dtype=torch.float32)
        action_batch = torch.tensor(np.array([data[1] for data in minibatch]), dtype=torch.float32)
        reward_batch = torch.tensor(np.array([data[2] for data in minibatch]), dtype=torch.float32).unsqueeze(1)
        next_state_batch = torch.tensor(np.array([data[3] for data in minibatch]), dtype=torch.float32)
        done_batch = torch.tensor(np.array([data[4] for data in minibatch]), dtype=torch.float32).unsqueeze(1)

        # Calculate target Q values
        with torch.no_grad():
            next_action_batch = self.actor_target_network(next_state_batch)
            target_q_values = self.critic_target_network.q_network_target(next_state_batch, next_action_batch)
            y_batch = reward_batch + (1 - done_batch) * GAMMA * target_q_values

        # Update critic network
        q_values = self.critic_network(state_batch, action_batch)
        critic_loss = self.critic_loss_fn(q_values, y_batch)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor network
        predicted_actions = self.actor_network(state_batch)
        actor_loss = -self.critic_network(state_batch, predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_network(self.actor_target_network, self.actor_network)
        self.update_target_network(self.critic_target_network, self.critic_network)

    def noise_action(self, state):
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
        self.actor_network.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            action_probs = self.actor_network(state_tensor).squeeze(0).numpy()
        self.actor_network.train()  # Set the model back to training mode
        return np.argmax(action_probs)  # Select action with highest probability


    def action(self, state):
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
        self.actor_network.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Ensure no gradients are calculated
            action = self.actor_network(state_tensor).squeeze(0).numpy()
        self.actor_network.train()  # Set the model back to training mode
        return action

    def perceive(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.train()

        if done:
            self.exploration_noise.reset()

    def update_target_network(self, target_network, source_network):
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(TAU * source_param.data + (1 - TAU) * source_param.data)

################################# End of Part I ################################

print("============================================================================================")

################################### Training DDPG with 256 neurons ###################################
# Hyper Parameters
LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE_ACT = 1e-4
LEARNING_RATE_CRT = 1e-3
TAU = 0.001
L2 = 0.01

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 100 # supposed 10000
BATCH_SIZE = 64
GAMMA = 0.99
random_seed = 0      

ENV_NAME = 'InvertedPendulum-v4'
EPISODES = 100000
max_ep_len = 225
TEST = 10

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # saving avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 64         # save model frequency (in num timesteps)


# state space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# action space dimension
action_dim = args.n_servers

hidden_dim = 256


## Note : print/save frequencies should be > than max_ep_len

###################### saving files ######################

#### saving files for multiple runs are NOT overwritten

log_dir = "DDPG2_files"
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
log_f_name = log_dir_1 + '/DDPG2_' + 'resource_allocation' + "_log_" + str(run_num1) + ".csv"
log_f_name2 = log_dir_2 + '/DDPG2_' + 'resource_allocation' + "_log_" + str(run_num2) + ".csv"

print("current logging run number for " + 'resource_allocation' + " : ", run_num1)
print("logging at : " + log_f_name)

#####################################################

################### checkpointing ###################

run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "DDPG2_preTrained"
if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + 'resource_allocation' + '/' 
if not os.path.exists(directory):
      os.makedirs(directory)


checkpoint_path = directory + "DDPG2256_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

#####################################################


############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", EPISODES)
print("max timesteps per episode : ", max_ep_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)

print("--------------------------------------------------------------------------------------------")
 
print("discount factor (gamma) : ", GAMMA)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate (actor) : ", LEARNING_RATE_ACT)
print("optimizer learning rate (critic) : ", LEARNING_RATE_CRT)

print("--------------------------------------------------------------------------------------------")
print("setting random seed to ", random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

#####################################################

print("============================================================================================")

################# training procedure ################

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
num_steps = 50
log_interval = 10

gc.enable()
agent = DDPG()
env = ClusterOptimizationEnv()

# env = makeFilteredEnv(env)

while time_step <= EPISODES:
    print("New training episode:")
    state = env.reset()
    # state = env.filter_observation(state)
    total_reward = 0

    for step in range(max_ep_len):
        # print(f"state: {state}")
        action = agent.noise_action(state)
        # print(f"action: {action}")
        next_state, reward, done, _ = env.step(action)
        agent.perceive(state, action, reward, next_state, done)
        total_reward += reward
        time_step += 1
        print("The current total episodic reward at timestep:", time_step, "is:", total_reward)
        sleep(0.1) # we sleep to read the reward in console        

        # log in logging file
        if time_step % log_freq == 0:
            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward)) 
            log_f.flush() # To force all buffered output to a particular log file
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
            torch.save(ActorNetwork.state_dict(), checkpoint_path) # state_dicts() dictionary object, maps each layer to its parameter tensor
            torch.save(CriticNetwork.state_dict(), checkpoint_path) 
            print("model saved")
            print("--------------------------------------------------------------------------------------------")
        state = next_state
        if bool(done):
            break
        
    print_running_reward += total_reward
    print_running_episodes += 1

    log_running_reward += total_reward
    log_running_episodes += 1

    i_episode += 1
    time_step += num_steps

log_f.close()
log_f2.close()

################################ End of Part II ################################

print("============================================================================================")