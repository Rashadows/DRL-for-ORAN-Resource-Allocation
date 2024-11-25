import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from env import Env
import gym

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
        self.new=0
    def size(self):
        return self.memory.shape[0]
#[s,a,r,s_,done] make sure all info are lists, i.e. [[[1,2],[3]],[1],[0],[[4,5],[6]],[True]]
    def store(self, trans):
        trans = [np.array(item) for item in trans]

        if self.size() < self.memory_size:
            if self.new == 0:
                # Initialize memory with dtype=object
                self.memory = np.empty(self.memory_size, dtype=object)
                self.memory[0] = trans
                self.new = 1
            else:
                self.memory[self.size()] = trans
        else:
            # Overwrite oldest data in circular buffer
            self.memory[self.cur] = trans
            self.cur = (self.cur + 1) % self.memory_size
        printable = [self.memory[i] for i in range(self.cur)]
        print(printable)

    def sample(self, batch_size):
        if self.size() < batch_size:
            return -1
        indices = np.random.choice(self.size(), batch_size, replace=False)
        return np.array([self.memory[i] for i in indices], dtype=object)

    
class Actor(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Actor,self).__init__()
        self.fc1=nn.Linear(state_dim,64)
        self.fc2=nn.Linear(64,32)
        self.fc3=nn.Linear(32,act_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return torch.tanh(x)
    
class Critic(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(state_dim,64)
        self.fc2=nn.Linear(64+act_dim,32)
        self.fc3=nn.Linear(32,1)
        
    def forward(self,state,action):
        x=self.fc1(state)
        x=F.relu(x)
        x=self.fc2(torch.cat((x,action),1))
        x=F.relu(x)
        x=self.fc3(x)
        return x

def gumbel_sample(shape,eps=1e-10):
    seed=torch.FloatTensor(shape).uniform_().to(device)
    return -torch.log(-torch.log(seed+eps)+eps)

def gumbel_softmax_sample(logits,temperature=1.0):
    #print(logits)
    logits=logits+gumbel_sample(logits.shape,1e-10)
    #print(logits)
    return (torch.nn.functional.softmax(logits/temperature,dim=1))

def gumbel_softmax(prob,temperature=1.0,hard=False):
    #print(prob)
    logits=torch.log(prob)
    y=gumbel_softmax_sample(prob,temperature)
    if hard==True:   #one hot but differenttiable
        y_onehot=onehot_action(y)
        y=(y_onehot-y).detach()+y
    return y

def onehot_action(prob):
    y=torch.zeros_like(prob).to(device)
    index=torch.argmax(prob,dim=1).unsqueeze(1)
    y=y.scatter(1,index,1)
    return y.to(torch.long)


lr=0.001
tau=0.05
max_t=200
gamma=0.9
memory_size=2000
batchsize=32
env = gym.make('CartPole-v1')
env=env.unwrapped
n_action=env.action_space.n
n_state=env.observation_space.shape[0]


class DDPG():
    def __init__(self):
        self.actor=Actor(n_state,n_action).to(device)
        self.target_actor=Actor(n_state,n_action).to(device)
        self.critic=Critic(n_state,n_action).to(device)
        self.target_critic=Critic(n_state,n_action).to(device)
        self.memory=replay_memory(memory_size)
        self.Aoptimizer=torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.Coptimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)
    
    def choose_action(self,state,eps):
        # state = np.array(state[0])
        prob=self.actor.forward(torch.FloatTensor(state).to(device))
        prob=torch.nn.functional.softmax(prob,0)
        #print(prob)
        if np.random.uniform()>eps:
            action=torch.argmax(prob,dim=0).tolist()
        else:
            action=np.random.randint(0,n_action)
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
    
for j in range(10):
    ddpg=DDPG()
    highest=0
    for episode in range(300):
        s, _ = env.reset()  
        t = 0
        total_reward = 0
        while t < max_t:
            a = ddpg.choose_action(s, 0.1)  
            s_, r, terminated, truncated, _ = env.step(a)  
            total_reward += r
            done = terminated or truncated
            transition = [s, [r], [a], s_, [done]]
            print(s)
            print(s_)
            print(transition)
            ddpg.memory.store(transition)
            if done:
                break
            s = s_ 
            if ddpg.memory.size() < batchsize:
                continue
            batch = ddpg.memory.sample(batchsize)
            print("batch", batch)
            batch = np.array([np.array(item, dtype=object) for item in batch])
            print("batch", batch)
            ddpg.critic_learn(batch)
            ddpg.actor_learn(batch)
            ddpg.soft_update()
            t += 1

        if episode%10==0:
            total_reward=0.0
            for i in range(1):
                t_s=env.reset()
                t_r=0.0
                tr=0.0
                time=0
                while(time<300):
                    time+=1
                    t_a=ddpg.choose_action(t_s,0)
                    ts_,tr,tdone,_=env.step(t_a)
                    t_r+=tr
                    if tdone:
                        break
                    t_s=ts_
                total_reward+=t_r
                if total_reward>highest:
                    highest=total_reward
                print("episode:"+format(episode)+",test score:"+format(total_reward))
    if(highest>20):
        print(format(j+1)+"th round did it")