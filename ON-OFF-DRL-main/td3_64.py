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
        
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
            )

        
        # critic
        self.critic1 = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
        #critic2
        self.critic2 = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
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


################################### Training ###################################


################ initialize environment hyperparameters and TD3 hyperparameters ################

print("setting training environment : ")

max_ep_len = 200                     # max timesteps in one episode
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for TD3
gamma = 0.99                # discount factor
lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network
random_seed = 0         # set random seed
max_training_timesteps = int(1e5)   # break training loop if timesteps > max_training_timesteps
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

# # Action limit for clamping: critically, assumes all dimensions share the same bound!
act_limit = args.n_servers

## Note : print/log frequencies should be > than max_ep_len

###################### logging ######################

#### log files for multiple runs are NOT overwritten

log_dir = "TD3_files"
if not os.path.exists(log_dir):
      os.makedirs(log_dir)

log_dir_1 = log_dir + '/' + 'resource_allocation' + '/' + 'stability' + '/'
if not os.path.exists(log_dir_1):
      os.makedirs(log_dir_1)
log_dir_2 = log_dir + '/' + 'resource_allocation' + '/' + 'reward' + '/'
if not os.path.exists(log_dir_2):
      os.makedirs(log_dir_2)


#### get number of log files in log directory
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

directory = "TD3_preTrained"
if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + 'resource_allocation' + '/'
if not os.path.exists(directory):
      os.makedirs(directory)


checkpoint_path = directory + "TD364_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
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

print("TD3 update frequency : " + str(update_after) + " timesteps") 
print("TD3 K epochs : ", K_epochs)
print("TD3 epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)

print("--------------------------------------------------------------------------------------------")

print("polyak averaging factor : ", polyak)
print("batch size : ", batch_size)
print("policy delay: ", policy_delay)
print("--------------------------------------------------------------------------------------------")

print("Actor Network Noise : ", act_noise)
print("Target Networks Noise : ", target_noise)
print("Noise Clip : ", noise_clip)

print("--------------------------------------------------------------------------------------------")
print("setting random seed to ", random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

#####################################################

print("============================================================================================")

################# training procedure ################

# initialize a TD3 agent
td3_agent = TD3(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

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
rewards = []

# start training loop
while time_step <= max_training_timesteps:
    print("New training episode:")
    sleep(0.1) # we sleep to read the reward in console
    state = env.reset()
    current_ep_reward = 0

    for t in range(1, max_ep_len+1):
        
        # select action with policy
        if t > start_steps:
            action = td3_agent.select_action(state, act_noise)
        else:
            action = env.sample_action()

        state2, reward, done, _ = env.step(action)
        
        # saving reward and is_terminals
        td3_agent.buffer.rewards.append(reward)
        td3_agent.buffer.is_terminals.append(done)
        state = state2

        time_step +=1
        current_ep_reward += reward
        print("The current total episodic reward at timestep:", time_step, "is:", current_ep_reward)
        sleep(0.1) # we sleep to read the reward in console

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = td3_agent.buffer.sample_batch(batch_size)
                td3_agent.update(data=batch, timer=j)

        # log in logging file
        if time_step % log_freq == 0:

            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)
            print("Saving reward to csv file")
            sleep(0.1) # we sleep to read the reward in console

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()
            log_f2.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f2.flush()
            log_running_reward = 0
            log_running_episodes = 0
            
        # printing average reward
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)
            rewards.append(print_avg_reward)
            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
            sleep(0.1) # we sleep to read the reward in console
            print_running_reward = 0
            print_running_episodes = 0
            
        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            sleep(0.1) # we sleep to read the reward in console
            td3_agent.save(checkpoint_path)
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
print("============================================================================================")

################################ End of Part II ################################