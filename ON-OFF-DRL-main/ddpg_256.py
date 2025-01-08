'''
this is adapted from https://github.com/LxzGordon/Deep-Reinforcement-Learning-with-pytorch/tree/master/algorithm/policy%20gradient
'''

import os
from datetime import datetime
from time import sleep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import gym
from env import Env
from argparser import args

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
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def store(self, transition):
        """Store a transition in the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples to draw a batch of size {batch_size}. Current size: {len(self.buffer)}.")
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class OU_Noise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state
    

class Actor(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Actor,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim//2)
        self.fc3=nn.Linear(hidden_dim//2,act_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return torch.tanh(x)
    
class Critic(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim+act_dim,hidden_dim//2)
        self.fc3=nn.Linear(hidden_dim//2,1)
        
    def forward(self,state,action):
        x=self.fc1(state)
        x=F.relu(x)
        x=self.fc2(torch.cat((x,action),1))
        x=F.relu(x)
        x=self.fc3(x)
        return x

def onehot_action(prob):
    y=torch.zeros_like(prob).to(device)
    index=torch.argmax(prob,dim=1).unsqueeze(1)
    y=y.scatter(1,index,1)
    return y.to(torch.long)

def gumbel_softmax(prob,temperature=1.0,hard=False):
    # print(prob)
    logits=torch.log(prob)
    seed=torch.FloatTensor(logits.shape).uniform_().to(device)
    logits=logits-torch.log(-torch.log(seed+eps)+eps) # gumbel sampling
    y = torch.nn.functional.softmax(logits/temperature,dim=1)
    if hard==True:   #one hot but differenttiable
        y_onehot=onehot_action(y)
        y=(y_onehot-y).detach()+y
    return y


class DDPG():
    def __init__(self):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        self.memory = ReplayBuffer(memory_size)
        self.Aoptimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.Coptimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.ou_noise = OU_Noise(action_dim)  # Initialize OU noise for exploration

    def choose_action(self, state, exploration=True):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        logits = self.actor(state).squeeze(0)  # Get logits from the actor network
        
        if exploration:
            # Add OU noise to logits for exploration
            noise = torch.FloatTensor(self.ou_noise.sample()).to(device)
            logits = logits + noise

        # Apply softmax to get action probabilities
        action_probs = F.softmax(logits, dim=-1)

        # Sample an action from the softmax probabilities
        action = torch.multinomial(action_probs, num_samples=1).item()
        return action, action_probs.cpu().detach().numpy()

    def actor_learn(self, batch):
        b_s = torch.FloatTensor(np.stack(batch[:, 0])).to(device)
        
        differentiable_a = torch.nn.functional.gumbel_softmax(
            torch.log(torch.nn.functional.softmax(self.actor(b_s), dim=1)), hard=True
        )
        loss = -self.critic(b_s, differentiable_a).mean()
        
        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()

    def critic_learn(self, batch):
        b_s = torch.FloatTensor(np.stack(batch[:, 0])).to(device)
        b_r = torch.FloatTensor(np.stack(batch[:, 1])).to(device)
        b_a = torch.FloatTensor(np.stack(batch[:, 2])).to(device)
        b_a = torch.nn.functional.one_hot(b_a.long().squeeze(), num_classes=action_dim).float().to(device)  # One-hot encoding
        b_s_ = torch.FloatTensor(np.stack(batch[:, 3])).to(device)
        b_d = torch.FloatTensor(np.stack(batch[:, 4])).to(device)
        
        eval_q = self.critic(b_s, b_a)

        next_action = torch.nn.functional.softmax(self.target_actor(b_s_), dim=1)
        index = torch.argmax(next_action, dim=1).unsqueeze(1)
        next_action = torch.zeros_like(next_action).scatter_(1, index, 1).to(device)

        target_q = (1 - b_d) * gamma * self.target_critic(b_s_, next_action).detach() + b_r
        td_error = eval_q - target_q
        loss = td_error.pow(2).mean()

        self.Coptimizer.zero_grad()
        loss.backward()
        self.Coptimizer.step()

    def soft_update(self):
        for param,target_param in zip(self.actor.parameters(),self.target_actor.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
        for param,target_param in zip(self.critic.parameters(),self.target_critic.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)

################################# End of Part I ################################

print("============================================================================================")

################################### Training DDPG with 256 neurons###################################
####### initialize environment hyperparameters and DDPG hyperparameters ######
print("setting training environment : ")

memory_size = 10000
lr_actor = 0.001
lr_critic = 0.0003
max_training_timesteps = 100000
max_ep_len = 225
batch_size = 128
random_seed = 0      
tau = 0.1
gamma = 0.99
eps = 1e-10  # for gumbel sampling

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # saving avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 4         # save model frequency (in num timesteps)

env = Env()
# env = gym.make('CartPole-v1')
# env=env.unwrapped
# action_dim=env.action_space.n
# state_dim=env.observation_space.shape[0]


# state space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# action space dimension
action_dim = args.n_servers

hidden_dim = 256


## Note : print/save frequencies should be > than max_ep_len

###################### saving files ######################

#### saving files for multiple runs are NOT overwritten

log_dir = "DDPG_files"
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
log_f_name = log_dir_1 + '/DDPG_' + 'resource_allocation' + "_log_" + str(run_num1) + ".csv"
log_f_name2 = log_dir_2 + '/DDPG_' + 'resource_allocation' + "_log_" + str(run_num2) + ".csv"

print("current logging run number for " + 'resource_allocation' + " : ", run_num1)
print("logging at : " + log_f_name)

#####################################################

################### checkpointing ###################

run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "DDPG_preTrained"
if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + 'resource_allocation' + '/' 
if not os.path.exists(directory):
      os.makedirs(directory)


checkpoint_path = directory + "DDPG256_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
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

#####################################################

print("============================================================================================")

ddpg=DDPG() #initialize DDPG agent
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

highest=0
while time_step <= max_training_timesteps:
    print("New training episode:")
    sleep(0.1) # we sleep to read the reward in console
    state = env.reset()  
    current_ep_reward = 0
    
    for step in range(max_ep_len):
        action, _ = ddpg.choose_action(state, 0.1)  
        next_state, reward, done, _ = env.step(action)  
        
        current_ep_reward += reward
        time_step += 1
        print("The current total episodic reward at timestep:", time_step, "is:", current_ep_reward)
        sleep(0.1) # we sleep to read the reward in console
        
        transition = [state, reward, action, next_state, done]
        # print(state)
        # print(next_state)
        # print(transition)
        ddpg.memory.store(transition)

        if len(ddpg.memory) < batch_size:
            continue
        batch = ddpg.memory.sample(batch_size)
        batch = np.array([np.array(item, dtype=object) for item in batch])
        # print("batch", batch)
        ddpg.critic_learn(batch)
        ddpg.actor_learn(batch)
        ddpg.soft_update()

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
        
        if time_step % print_freq == 0:
            # print average reward till last time_step
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
            torch.save(ddpg.actor.state_dict(), checkpoint_path) 
            torch.save(ddpg.critic.state_dict(), checkpoint_path) 
            print("model saved")
            print("--------------------------------------------------------------------------------------------")

            state = next_state    
            if done:
                break     
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