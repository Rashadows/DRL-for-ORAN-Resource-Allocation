from collections import deque
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def add(self, transition, error):
        # Unpack the transition
        state, action, reward, next_state, done = transition

        # Convert to NumPy arrays if needed
        if not isinstance(state, np.ndarray):
            state = state.cpu().numpy()
        
        if not isinstance(next_state, np.ndarray):
            next_state = next_state.cpu().numpy()

        # Recreate the transition with validated data
        transition = (state, action, reward, next_state, done)

        # Add to buffer
        priority = float((error + 1e-5) ** self.alpha)
        self.buffer.append(transition)
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        # Ensure all priorities are scalar before conversion
        if not all(isinstance(p, (int, float)) for p in self.priorities):
            # print("Non-scalar values detected in priorities. Cleaning...")
            self.priorities = deque(float(p[0]) if isinstance(p, (list, np.ndarray)) else float(p) for p in self.priorities)

        # Convert to NumPy array
        priorities = np.array(self.priorities)
        if priorities.ndim != 1:
            raise ValueError(f"Invalid priorities shape: {priorities.shape}. Expected a 1D array.")

        # Calculate probabilities for sampling
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # Extract components of the sampled transitions
        states, actions, rewards, next_states, dones = zip(*samples)

        # Convert states and next_states to tensors
        states = np.vstack(states)  # Stacks all states into a single NumPy array
        states = torch.FloatTensor(states).to(device)

        next_states = np.vstack(next_states)
        next_states = torch.FloatTensor(next_states).to(device)

        # Calculate importance-sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        # Create a named tuple for the batch
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        batch = Transition(states, actions, rewards, next_states, dones)

        return batch, weights, indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)


class NormalizedActions:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.low = 0
        self.high = action_dim - 1

    def _action(self, action_prob):
        action = self.low + (action_prob + 1.0) * 0.5 * (self.high - self.low)
        action = torch.clamp(action, min=self.low, max=self.high)
        return action

    def _reverse_action(self, action_tens):
        action = action_tens.argmax(dim=-1)
        action = 2 * (action - self.low) / (self.high - self.low) - 1
        action = torch.clamp(action, self.low, self.high)
        return action


class OrnsteinUhlenbeckNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, decay_period=1000000):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.decay_period = decay_period
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def get_action(self, action, t=0):
        sigma = self.sigma - (self.sigma) * min(1.0, t / self.decay_period)
        # Applying the Ornstein-Uhlenbeck process
        dx = self.theta * (self.mu - self.state) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state = self.state + dx
        action_with_noise = action + self.state
        return action_with_noise

def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(PolicyNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.tanh(self.tanh(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.tanh(x))
        action_probs = F.softmax(self.linear3(x), dim=-1)
        return action_probs
    
    def get_action(self, state):
        action = self.forward(state)
        return action.detach()

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, state, action):
        if action.dim() == 1:
            action = action.unsqueeze(1)

        if action.shape[1] == 1: 
            num_classes = self.linear1.in_features - state.shape[1]
            action = F.one_hot(action.long(), num_classes=num_classes).float()
            action = action.view(state.size(0), -1)  # Reshape to [batch_size, action_dim]
        
        x = torch.cat([state, action], dim=1)

        x = F.relu(self.linear1(x))
        x = F.tanh(self.tanh(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x.view(-1, 1)


def td3_update(step, batch_size, gamma=0.99, soft_tau=1e-2, noise_std=0.2, noise_clip=0.5, policy_update=2, target_update = 10, beta=0.4):

    batch, weights, indices = replay_buffer.sample(batch_size, beta)
    state, action, reward, next_state, done = batch

    # Validate and stack states
    if isinstance(state, list):
        state = np.stack(state)
    
    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device).view(-1, 1)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    weights = weights.to(device)

    # Calculate next actions with target policy
    next_action = target_policy_net(next_state)
    # noise = torch.normal(torch.zeros(next_action.size()), noise_std).to(device)
    # noise = torch.clamp(noise, -noise_clip, noise_clip)
    next_action = noise.get_action(next_action, step).float()
    next_action = torch.clamp(next_action, 0, 1)

    target_q_value1 = target_value_net1(next_state, next_action).view(-1, 1)
    target_q_value2 = target_value_net2(next_state, next_action).view(-1, 1)
    target_q_value = torch.min(target_q_value1, target_q_value2)
    expected_q_value = reward + (1.0 - done) * gamma * target_q_value

    q_value1 = value_net1(state, action).view(-1, 1)
    q_value2 = value_net2(state, action).view(-1, 1)

    value_loss1 = (weights * F.mse_loss(q_value1, expected_q_value.detach(), reduction='none').squeeze()).mean()
    value_loss2 = (weights * F.mse_loss(q_value2, expected_q_value.detach(), reduction='none').squeeze()).mean()

    value_optimizer1.zero_grad()
    value_loss1.backward()
    value_optimizer1.step()

    value_optimizer2.zero_grad()
    value_loss2.backward()
    value_optimizer2.step()

    # Update priorities
    errors = torch.abs(q_value1 - expected_q_value).detach().cpu().numpy()
    replay_buffer.update_priorities(indices, errors)

    if step % policy_update == 0:
        policy_loss = -value_net1(state, action).mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

    if step % target_update == 0:
        soft_update(value_net1, target_value_net1, soft_tau=soft_tau)
        soft_update(value_net2, target_value_net2, soft_tau=soft_tau)
        soft_update(policy_net, target_policy_net, soft_tau=soft_tau)
    
    return target_q_value, expected_q_value
    
def load(policy_net, value_net1, value_net2, checkpoint_path):
    policy_net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    value_net1.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    value_net2.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

################################# End of Part I ################################

print("============================================================================================")

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

normalized_actions = NormalizedActions(action_dim)

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

replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_size)

torch.manual_seed(0)
# preTrained weights directory
random_seed = 0             #### set this to load a particular checkpoint trained on random seed
run_num_pretrained = 0      #### set this to load a particular checkpoint num
directory = "TD3_preTrained" + '/' + 'resource_allocation' + '/' 
checkpoint_path = directory + "TD3256_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
print("loading network from : " + checkpoint_path)
# load(policy_net, value_net1, value_net2, checkpoint_path)


env = Env()
noise = OrnsteinUhlenbeckNoise(action_dim)

state = env.reset()

class learn_td3(object):
    
    def __init__(self):
        self.name = 'TD3'
        
    def step(self, obs):
        state = obs
        done = False
        total_reward = 0
        for step in range(1, max_ep_len+1):
            action_tens = policy_net.get_action(state)
            action_tens = noise.get_action(action_tens, step)
            action = action_tens.multinomial(1)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)            
            state = next_state
            total_reward += reward
            if done:
                break
        return action