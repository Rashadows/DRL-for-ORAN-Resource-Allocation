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

class replay_memory():
    def __init__(self,replay_memory_size):
        self.memory_size=replay_memory_size
        self.memory=np.array([])
        self.cur=0
        self.length= 0
#[s,a,r,s_,done] make sure all info are lists, i.e. [[[1,2],[3]],[1],[0],[[4,5],[6]],[True]]
    def store(self, trans):
        trans = [np.array(item) for item in trans]

        if len(trans) != 5:  # Ensure all five elements are present
            raise ValueError("Invalid transition: expected 5 elements, got {}".format(len(trans)))

        if self.length < self.memory_size:
            if self.length == 0:
                # Initialize memory with dtype=object
                self.memory = np.empty(self.memory_size, dtype=object)
        else:
            # Overwrite oldest data in circular buffer
            self.length = self.length % self.memory_size
        self.memory[self.length] = trans
        self.length += 1

    def sample(self, batch_size):
        if self.length < batch_size:
            return -1
        indices = np.random.choice(self.length, batch_size, replace=False)
        return np.array([self.memory[i] for i in indices], dtype=object)

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
    #print(prob)
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
        self.actor=Actor(state_dim,action_dim).to(device)
        self.target_actor=Actor(state_dim,action_dim).to(device)
        self.critic=Critic(state_dim,action_dim).to(device)
        self.target_critic=Critic(state_dim,action_dim).to(device)
        self.memory=replay_memory(memory_size)
        self.Aoptimizer=torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.Coptimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)
    
    def choose_action(self,state,eps):
        # state = np.array(state).tolist()
        prob=self.actor.forward(torch.FloatTensor(state).to(device))
        prob=torch.nn.functional.softmax(prob,0)
        #print(prob)
        if np.random.uniform()>eps:
            action=torch.argmax(prob,dim=0).tolist()
        else:
            action=np.random.randint(0,action_dim)
        return action
    
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
        # Unpack the batch into separate arrays for states, actions, rewards, next states, and dones
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


replay_buffer_size = 1000000
lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network
rewards     = []
max_ep_len = 225            # max timesteps in one episode, also try 100

# state space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# action space dimension
action_dim = args.n_servers

hidden_dim = 256

value_net1 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
value_net2 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

target_value_net1 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_value_net2 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

soft_update(value_net1, target_value_net1, soft_tau=1.0)
soft_update(value_net2, target_value_net2, soft_tau=1.0)
soft_update(policy_net, target_policy_net, soft_tau=1.0)

value_optimizer1 = optim.Adam(value_net1.parameters(), lr=lr_critic)
value_optimizer2 = optim.Adam(value_net2.parameters(), lr=lr_critic)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_actor)