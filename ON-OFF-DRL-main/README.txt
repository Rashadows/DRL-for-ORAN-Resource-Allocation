ON_OFF_DRL
This work is inspired from the paper "On-Policy vs. Off-Policy Deep Reinforcement Learning for Resource Allocation in Open Radio Access Network" which provided ACER and PPO comparative analysis for O-RAN resource allocation environment.
Here, we discretize TD3 and SAC algorithms and adapt them for the O-RAN discrete-action environment to show they outperform the existing algorithms. Our work is summarized in the paper "Optimizing Resource Allocation in Open RANs: A DRL-Based Approach".

- It has been tested on Windows 10 and Python 3.11.4

- To run the code, mainly you need:
  * pip install torch
  * pip install gurobipy

- To avoid the long training time, you could go directly to the tests folder and run:
  * reward_plot.py to get the step and episode reward figures (first two figures)
  * stability_plot.py to get the ACER and PPO reward figures for different NN architectures
  * plot.py to get the energy and energy per latency figures (last two figures)

- To start training from scratch, you need to generate the reward files and the trained models weights by running the following:
  * {algorithm}_32.py
  * {algorithm}_64.py 
  * {algorithm}_256.py 
  * reward_plot.py and stability_plot.py
  with {algorithm} being one of {ACER, PPO, Double_DQN, TD3, SAC}

As a result you will create folders in the format ({algorithm}_files, {algorithm}_preTrained) which contain the reward files and trained models respectively.

- models.py load the trained models to test them in energy and latency performance
  * run main.py to do these tests and plot the energy and energy per latency figures.
- opt.py and greedy.py implement the optimal MIP solution and the greedy solution respectively.

----- REFERENCES -----
  * https://github.com/higgsfield/RL-Adventure-2
  * https://github.com/nikhilbarhate99/PPO-PyTorch
  * https://github.com/nessry/ON-OFF-DRL
  * https://github.com/gohsyi/cluster_optimization