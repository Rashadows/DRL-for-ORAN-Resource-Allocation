# -*- coding: utf-8 -*-
############################### Import libraries ###############################

import os
import pandas as pd
import matplotlib.pyplot as plt

# Smooth out rewards to get smooth or less smooth plot lines by adjusting window size
window_len_var = 5
min_window_len_var = 1
linewidth_var = 2.5
alpha_var = 1

def step_plot():
    fig_num = 0  # Change this to prevent overwriting figures in the same env_name folder

    # Create directory for saving figures
    figures_dir = "plots/resource_allocation/reward/"
    os.makedirs(figures_dir, exist_ok=True)

    fig_save_path = figures_dir + f'step_reward_fig_{fig_num}.pdf'

    algorithms = {
        "ACER": "green",
        "PPO": "red",
        "TD3": "blue",
        "DDPG": "purple",
        "DDPG2": "yellow"
    }

    ax = plt.gca()
    for algo, color in algorithms.items():
        log_dir = f"{algo}_files/resource_allocation/reward/"
        os.makedirs(log_dir, exist_ok=True)

        num_runs = len(next(os.walk(log_dir))[2])
        all_runs = []
        
        for run_num in range(num_runs):
            log_f_name = f"{log_dir}/{algo}_resource_allocation_log_{run_num}.csv"
            print(f"loading data from: {log_f_name}")
            data = pd.read_csv(log_f_name)
            all_runs.append(data)

        for i, run in enumerate(all_runs):
            # Smooth rewards
            run[f'reward_var_{i}'] = run['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()
            # Plot lines with explicit labels
            run.plot(
                kind='line', x='timestep', y=f'reward_var_{i}', ax=ax,
                color=color, linewidth=linewidth_var, alpha=alpha_var,
                label=f'{algo} (run {i+1})' if num_runs > 1 else algo,
                marker='o', markevery=10
            )
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    # Set custom legend
    ax.legend(fontsize=11, loc='lower right')

    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(fig_save_path, dpi=600, facecolor='w', edgecolor='b', transparent=True)
    print(f"Figure saved at: {fig_save_path}")
    plt.show()

def episode_plot():
    fig_num = 0  # Change this to prevent overwriting figures in the same env_name folder

    # Create directory for saving figures
    figures_dir = "plots/resource_allocation/reward/"
    os.makedirs(figures_dir, exist_ok=True)

    fig_save_path = figures_dir + f'episode_reward_fig_{fig_num}.pdf'

    algorithms = {
        "ACER": "green",
        "PPO": "red",
        "TD3": "blue",
        "DDPG": "purple", 
        "DDPG2": "yellow"
    }

    ax = plt.gca()
    for algo, color in algorithms.items():
        log_dir = f"{algo}_files/resource_allocation/reward/"
        os.makedirs(log_dir, exist_ok=True)

        num_runs = len(next(os.walk(log_dir))[2])
        all_runs = []
        
        for run_num in range(num_runs):
            log_f_name = f"{log_dir}/{algo}_resource_allocation_log_{run_num}.csv"
            print(f"loading data from: {log_f_name}")
            data = pd.read_csv(log_f_name)
            all_runs.append(data)

        for i, run in enumerate(all_runs):
            # Smooth rewards
            run[f'reward_var_{i}'] = run['reward'].rolling(window=1, win_type='triang', min_periods=min_window_len_var).mean()
            # Plot lines with explicit labels
            run.plot(
                kind='line', x='episode', y=f'reward_var_{i}', ax=ax,
                color=color, linewidth=linewidth_var, alpha=alpha_var,
                label=f'{algo} (run {i+1})' if num_runs > 1 else algo
            )
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Episodes", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    plt.xlim(0, 350)

    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(fig_save_path, dpi=600, facecolor='w', edgecolor='b', transparent=True)
    print(f"Figure saved at: {fig_save_path}")
    plt.show()

if __name__ == '__main__':
    step_plot()
    episode_plot()
