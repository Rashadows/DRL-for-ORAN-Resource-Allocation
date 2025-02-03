import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


###########################################
# PPO Agent
###########################################

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.name = 'PPO'
        self.model = PPOActorCritic(state_dim, action_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def step(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, _ = self.model(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
        return action

###########################################
# ACER Agent
###########################################

class ACERActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ACERActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        policy = self.actor(state).clamp(max=1-1e-20)
        q_value = self.critic(state)
        value = (policy * q_value).sum(-1, keepdim=True)
        return policy, q_value, value
    
class AcerAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.name = 'ACER'
        self.model = ACERActorCritic(state_dim, action_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def step(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            policy, _, _ = self.model(state)
            action = policy.multinomial(1).item()
        return action

#####################################################
# Double DQN Agent
#####################################################

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.name = 'Double_DQN'
        self.model = QNetwork(state_dim, action_dim).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def step(self, obs):
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action = q_values.argmax(dim=1).item()
        return action

#####################################################
# TD3 Agent
#####################################################

class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(TD3Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        logits = self.actor(x)
        return logits

class TD3Agent:
    def __init__(self, state_dim, action_dim, model_path):
        self.name = 'TD3'
        self.actor = TD3Actor(state_dim, action_dim).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()
    
    def step(self, obs):
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.actor(state_tensor)
            action_probs = F.softmax(logits, dim=1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
        return action

#####################################################
# SAC Agent
#####################################################

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, state):
        log_probs = self.policy_net(state)
        probs = log_probs.exp()
        return log_probs, probs

class SACAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.name = 'SAC'
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)

        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.policy_net.eval()

    def step(self, obs):
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            _, probs = self.policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
        return action
