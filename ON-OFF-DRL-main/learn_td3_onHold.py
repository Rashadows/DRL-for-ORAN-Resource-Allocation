# -*- coding: utf-8 -*-
############################### Import libraries ###############################

import os
import itertools
from datetime import datetime
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
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


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states1 = []
        self.states2 = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states1[:]
        del self.states2[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.states1[idxs],
                        obs2=self.states2[idxs],
                        act=self.actions[idxs],
                        rew=self.rewards[idxs],
                        done=self.is_terminals[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
        
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Softmax(dim=1)
        )
        
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )

    def act(self, state):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values1 = self.critic1(state)
        state_values2 = self.critic2(state)
        
        return action_logprobs, state_values1, state_values2, dist_entropy
    
class TD3:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic1.parameters(), 'lr': lr_critic}, 
                        {'params': self.policy.critic2.parameters(), 'lr': lr_critic}
                    ])
                
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.policy.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        q_params = itertools.chain(self.policy.critic1.parameters(), self.policy.critic2.parameters())

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
            
        self.buffer.states1.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()
    
    # setup function to compute TD3 Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.policy.critic1(o,a)
        q2 = self.policy.critic2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.policy.actor(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, 0, act_limit)

            # Target Q-values
            q1_pi_targ = self.policy.critic1(o2, a2)
            q2_pi_targ = self.policy.critic2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy())

        return loss_q, loss_info
    
    
    # Set up function for computing TD3 pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        q1_pi = self.policy.critic1(o, self.actor(o))
        return -q1_pi.mean()
    
    def update(self, data, timer):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()        
        

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.policy.parameters(), self.policy.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
################################# End of Part I ################################
print("============================================================================================")
random_seed = 0         # set random seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)


max_ep_len = 200                     # max timesteps in one episode
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for TD3
gamma = 0.99                # discount factor
lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network
random_seed = 0         # set random seed
max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps
print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 4 # save model frequency (in num timesteps)
action_std = None
### new for td3
polyak=0.995
batch_size=100
policy_delay=2
act_noise=0.1
target_noise=0.2
noise_clip=0.5
update_after = max_ep_len * 4      # update policy every n timesteps
update_every=50
start_steps=10000


env=Env()
 
# state space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# action space dimension
action_dim = args.n_servers

# Action limit for clamping: critically, assumes all dimensions share the same bound!
act_limit = args.n_servers

# initialize a PPO agent
td3_agent = TD3(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
# preTrained weights directory
random_seed = 0             #### set this to load a particular checkpoint trained on random seed
run_num_pretrained = 0      #### set this to load a particular checkpoint num
directory = "TD3_preTrained" + '/' + 'resource_allocation' + '/' 
checkpoint_path = directory + "TD3256_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
print("loading network from : " + checkpoint_path)
td3_agent.load(checkpoint_path)
state = env.reset()
print("--------------------------------------------------------------------------------------------")


class learn_td3(object):
    
    def __init__(self):
        self.name = 'TD3'

    def step(self, obs):
        state = obs
        done = False
        for t in range(1, max_ep_len+1):
            action = td3_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            if done:
                break
        # clear buffer    
        td3_agent.buffer.clear()

        print("============================================================================================")
        
        return action
