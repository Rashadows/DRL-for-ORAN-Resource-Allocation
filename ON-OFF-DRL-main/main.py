# main.py

from argparser import args
from env import Env
import os
import numpy as np
from tqdm import tqdm
from greedy import Greedy
from opt import Optimal
from models import AcerAgent, PPOAgent, DoubleDQNAgent, TD3Agent, SACAgent
from plot import power_plot, latency_plot

if __name__ == '__main__':

    env = Env()
    
    # Define state and action sizes
    state_dim = args.n_servers * args.n_resources + args.n_resources + 1
    action_dim = args.n_servers

    # Paths to pre-trained models
    acer_model_path = 'ACER_preTrained/resource_allocation/ACER256_resource_allocation_0_0.pth'
    ppo_model_path = 'PPO_preTrained/resource_allocation/PPO64_resource_allocation_0_0.pth'
    double_dqn_model_path = 'Double_DQN_preTrained/resource_allocation/Double_DQN64_resource_allocation_0_0.pth'
    td3_model_path = 'TD3_preTrained/resource_allocation/TD364_resource_allocation_0_0.pth'
    sac_model_path = 'SAC_preTrained/resource_allocation/SAC64_resource_allocation_0_0.pth' 

    models = [
        PPOAgent(state_dim, action_dim, ppo_model_path),
        AcerAgent(state_dim, action_dim, acer_model_path),
        DoubleDQNAgent(state_dim, action_dim, double_dqn_model_path),
        TD3Agent(state_dim, action_dim, td3_model_path),
        SACAgent(state_dim, action_dim, sac_model_path), 
        Greedy(act_size=action_dim, n_servers=args.n_servers),
        Optimal(act_size=action_dim, n_servers=args.n_servers, tasks=env.tasks),
    ]

    for m in models:
        obs = env.reset()
        latency_list = []
        power_list = []
        for _ in tqdm(range(2000)):
            try:
                action = int(m.step(obs))
            except TypeError:
                action = env.sample_action()
            obs, _, done, info = env.step(action)
            latency, power = info
            latency_list.append(latency)
            power_list.append(power)
            
            if done:
                obs = env.reset()
        
        np.savetxt(os.path.join('logs', f'{m.name}_Latency.txt'), latency_list)
        np.savetxt(os.path.join('logs', f'{m.name}_Power.txt'), power_list)
    
    power_plot()
    latency_plot()