# -*- coding: utf-8 -*-
############################### Import libraries ###############################

import os
import numpy as np
import matplotlib.pyplot as plt

linewidth_smooth = 2.5
alpha_smooth = 1

fig_save_path1 = os.path.join('plots','power.pdf')
fig_save_path2 = os.path.join('plots','latency.pdf')

def power_plot():
    # List of algorithms
    algorithms = ['Optimal', 'Greedy', 'ACER', 'PPO', 'TD3', 'Double_DQN', 'SAC']
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'cyan']
    markers = ['+', 's', 'o', '^', '*', 'x', 'd']

    ax = plt.gca()

    for i, alg in enumerate(algorithms):
        power_file = os.path.join('logs', f'{alg}_Power.txt')
        if os.path.exists(power_file):
            power_data = np.loadtxt(power_file)
            ax.plot(
                power_data,
                color=colors[i],
                linewidth=linewidth_smooth,
                alpha=alpha_smooth,
                marker=markers[i],
                markevery=100
            )
        else:
            print(f"Warning: {power_file} not found. Skipping {alg}.")

    ax.set_xlabel('Number of network users', fontsize=11)
    ax.set_ylabel('Energy (kWh)', fontsize=11)
    ax.legend(algorithms, fontsize=11, loc='lower right')
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.1)
    plt.xlim(auto=True)  # Let matplotlib auto-scale
    plt.ylim(auto=True)
    
    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(fig_save_path1, dpi=600, facecolor='w', edgecolor='b',
                orientation='landscape', format=None, transparent=True, 
                bbox_inches=None, pad_inches=0.1, metadata=None)
    plt.show()

def latency_plot():
    # List of algorithms
    algorithms = ['Optimal', 'Greedy', 'ACER', 'PPO', 'TD3', 'Double_DQN', 'SAC']
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'cyan']
    markers = ['+', 's', 'o', '^', '*', 'x', 'd']

    ax = plt.gca()

    for i, alg in enumerate(algorithms):
        power_file = os.path.join('logs', f'{alg}_Power.txt')
        latency_file = os.path.join('logs', f'{alg}_Latency.txt')
        if os.path.exists(power_file) and os.path.exists(latency_file):
            power_data = np.loadtxt(power_file)
            latency_data = np.loadtxt(latency_file)
            ax.plot(
                latency_data,
                power_data,
                color=colors[i],
                linewidth=linewidth_smooth,
                alpha=alpha_smooth,
                marker=markers[i],
                markevery=100
            )
        else:
            if not os.path.exists(power_file):
                print(f"Warning: {power_file} not found. Skipping {alg}.")
            if not os.path.exists(latency_file):
                print(f"Warning: {latency_file} not found. Skipping {alg}.")

    ax.set_xlabel('Latency (sec)', fontsize=11)
    ax.set_ylabel('Energy (kWh)', fontsize=11)
    ax.legend(algorithms, fontsize=11, loc='upper left')
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.1)
    plt.xlim(auto=True)  # Let matplotlib auto-scale
    plt.ylim(auto=True)

    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(fig_save_path2, dpi=600, facecolor='w', edgecolor='b',
                orientation='landscape', format=None, transparent=True, 
                bbox_inches=None, pad_inches=0.1, metadata=None)
    plt.show()