# -*- coding: utf-8 -*-
############################### Import libraries ###############################

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

############################### Configuration ###############################

fig_num = 0  # Change this to prevent overwriting figures in the same folder

# Smoothing parameters
window_len = 5
min_window_len = 1
linewidth = 2.5
alpha = 1

# Common color scheme for all plots
common_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

# Algorithms configuration
algorithms = {
    'ACER': {
        'file_dir': 'ACER_files',
        'marker': 'o',
        'legend_prefix': 'ACER'
    },
    'PPO': {
        'file_dir': 'PPO_files',
        'marker': '^',
        'legend_prefix': 'PPO'
    },
    'Double_DQN': {
        'file_dir': 'Double_DQN_files',
        'marker': 's',
        'legend_prefix': 'DQN'
    },
    'TD3': {
        'file_dir': 'TD3_files',
        'marker': '*',
        'legend_prefix': 'TD3'
    },
    'SAC': {
        'file_dir': 'SAC_files',
        'marker': 'D',
        'legend_prefix': 'SAC'
    }
}

############################### Helper Function ###############################

def plot_algorithm(alg_name, config, fig_num, colors):
    log_dir_base = config['file_dir']
    marker = config['marker']
    legend_prefix = config['legend_prefix']

    # Construct log directory path
    log_dir = os.path.join(log_dir_base, 'resource_allocation', 'stability')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created directory: {log_dir}")

    # Find all CSV files matching the pattern
    try:
        files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"No log files found in {log_dir}. Skipping {alg_name}...")
        return

    if not files:
        print(f"No log files found in {log_dir}. Skipping {alg_name}...")
        return

    # Read all runs and extract nnSize
    all_runs = []
    nn_sizes = []
    for filename in files:
        # Check if the file belongs to the current algorithm
        pattern = rf"^{alg_name}_(\d+)_resource_allocation_log_\d+\.csv$"
        match = re.match(pattern, filename)
        if not match:
            continue  # Skip files that don't match the pattern

        nn_size = match.group(1)
        nn_sizes.append(nn_size)

        log_f_name = os.path.join(log_dir, filename)
        print(f"Loading data from: {log_f_name}")

        data = pd.read_csv(log_f_name)
        all_runs.append(data)
        print("--------------------------------------------------------------------------------------------")

    if not all_runs:
        print(f"No valid log files found for {alg_name}. Skipping...")
        return

    # Prepare figures directory
    figures_dir = os.path.join("plots", 'resource_allocation', 'stability')
    os.makedirs(figures_dir, exist_ok=True)

    fig_save_path = os.path.join(figures_dir, f'{alg_name}_resource_allocation_fig_{fig_num}.pdf')

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (run, nn_size) in enumerate(zip(all_runs, nn_sizes)):
        run['reward_smoothed'] = run['reward'].rolling(
            window=window_len, win_type='triang', min_periods=min_window_len
        ).mean()

        run_label = f"{legend_prefix}_{nn_size}"

        ax.plot(
            run['timestep'],
            run['reward_smoothed'],
            color=colors[i % len(colors)],
            linewidth=linewidth,
            alpha=alpha,
            marker=marker,
            markevery=10,
            label=run_label
        )

    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    fig.tight_layout()
    plt.savefig(
        fig_save_path, dpi=600, facecolor='w', edgecolor='b',
        orientation='landscape', transparent=True, bbox_inches='tight',
        pad_inches=0.1
    )
    print(f"Figure saved at: {fig_save_path}")
    plt.show()
    print("============================================================================================")

############################### Main Execution ###############################

if __name__ == '__main__':
    for alg, config in algorithms.items():
        plot_algorithm(alg, config, fig_num, common_colors)
        fig_num += 1