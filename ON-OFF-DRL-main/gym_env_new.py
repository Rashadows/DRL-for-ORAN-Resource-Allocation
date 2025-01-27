# -*- coding: utf-8 -*-
"""
Updated Environment Code for DQN Training
Adapted to fix the issue with infinite observation space bounds and initialization errors
"""

############################### Import libraries ###############################

import os
import heapq
import pandas as pd
import numpy as np
import random
from collections import deque
from argparser import args
import gym
from gym import spaces

class ClusterOptimizationEnv(gym.Env):
    def __init__(self):
        super(ClusterOptimizationEnv, self).__init__()

        self.P_0 = args.P_0
        self.P_100 = args.P_100
        self.T_on = args.T_on
        self.T_off = args.T_off
        self.n_servers = args.n_servers
        self.w1 = args.w1
        self.w2 = args.w2
        self.w3 = args.w3

        # Data paths
        self.machine_meta_path = os.path.join('data', 'machine_meta.csv')
        self.machine_usage_path = os.path.join('data', 'machine_usage.csv')
        self.container_meta_path = os.path.join('data', 'container_meta.csv')
        self.container_usage_path = os.path.join('data', 'container_usage.csv')
        self.batch_task_path = os.path.join('data', 'batch_task.csv')
        self.batch_instance_path = os.path.join('data', 'batch_instance.csv')

        # Data columns
        self.machine_meta_cols = [
            'machine_id',  # uid of machine
            'time_stamp',  # time stamp, in second
            'failure_domain_1',  # one level of container failure domain
            'failure_domain_2',  # another level of container failure domain
            'cpu_num',  # number of cpu on a machine
            'mem_size',  # normalized memory size. [0, 100]
            'status',  # status of a machine
        ]
        self.machine_usage_cols = [
            'machine_id',  # uid of machine
            'time_stamp',  # time stamp, in second
            'cpu_util_percent',  # [0, 100]
            'mem_util_percent',  # [0, 100]
            'mem_gps',  # normalized memory bandwidth, [0, 100]
            'mkpi',  # cache miss per thousand instruction
            'net_in',  # normalized incoming network traffic, [0, 100]
            'net_out',  # normalized outgoing network traffic, [0, 100]
            'disk_io_percent',  # [0, 100], abnormal values are of -1 or 101
        ]
        self.container_meta_cols = [
            'container_id',  # uid of a container
            'machine_id',  # uid of container's host machine
            'time_stamp',
            'app_du',  # containers with same app_du belong to same application group
            'status',
            'cpu_request',  # 100 is 1 core
            'cpu_limit',  # 100 is 1 core
            'mem_size',  # normalized memory, [0, 100]
        ]
        self.container_usage_cols = [
            'container_id',  # uid of a container
            'machine_id',  # uid of container's host machine
            'time_stamp',
            'cpu_util_percent',
            'mem_util_percent',
            'cpi',
            'mem_gps',  # normalized memory bandwidth, [0, 100]
            'mpki',
            'net_in',  # normalized incoming network traffic, [0, 100]
            'net_out',  # normalized outgoing network traffic, [0, 100]
            'disk_io_percent'  # [0, 100], abnormal values are of -1 or 101
        ]
        self.batch_task_cols = [
            'task_name',  # task name. unique within a job
            'instance_num',  # number of instances
            'job_name',  # job name
            'task_type',  # task type
            'status',  # task status
            'start_time',  # start time of the task
            'end_time',  # end time of the task
            'plan_cpu',  # number of cpu needed by the task, 100 is 1 core
            'plan_mem'  # normalized memory size, [0, 100]
        ]
        self.batch_instance_cols = [
            'instance_name',  # instance name of the instance
            'task_name',  # task name. unique within a job
            'instance_num',  # number of instances
            'job_name',  # job name
            'task_type',  # task type
            'status',  # task status
            'start_time',  # start time of the task
            'end_time',  # end time of the task
            'machine_id',  # uid of host machine of the instance
            'seq_no',  # sequence number of this instance
            'total_seq_no',  # total sequence number of this instance
            'cpu_avg',  # average cpu used by the instance, 100 is 1 core
            'cpu_max',  # max cpu used by the instance, 100 is 1 core
            'mem_avg',  # average memory used by the instance (normalized)
            'mem_max',  # max memory used by the instance (normalized, [0, 100])
        ]

        # Initialize variables
        self.cur = 0
        self.latency = []
        self.cur_time = 0

        # Load data
        self.loadcsv()

        # Now that data is loaded, we can define the observation space with finite bounds
        self.define_observation_space()

        # Define the action space
        self.action_space = spaces.Discrete(self.n_servers)

    def loadcsv(self):
        # Read CSV into DataFrames
        self.machine_meta = pd.read_csv(self.machine_meta_path, header=None, names=self.machine_meta_cols)
        self.machine_meta = self.machine_meta[self.machine_meta['time_stamp'] == 0]
        self.machine_meta = self.machine_meta[['machine_id', 'cpu_num', 'mem_size']]
        self.machine_meta.reset_index(drop=True, inplace=True)

        self.batch_instance = pd.read_csv(self.batch_instance_path, header=None, names=self.batch_instance_cols)
        self.batch_instance = self.batch_instance[(self.batch_instance['cpu_avg'] != "") & (self.batch_instance['mem_avg'] != "")]
        self.batch_instance = self.batch_instance[['cpu_avg', 'mem_avg']]
        self.batch_instance.reset_index(drop=True, inplace=True)

        self.batch_task = pd.read_csv(self.batch_task_path, header=None, names=self.batch_task_cols)
        self.batch_task = self.batch_task[self.batch_task['status'] == 'Terminated']
        self.batch_task = self.batch_task[self.batch_task['plan_cpu'] <= 100]
        self.batch_task = self.batch_task.sort_values(by='start_time')
        self.batch_task.reset_index(drop=True, inplace=True)

        self.n_machines = self.n_servers
        self.n_tasks = min(2000, len(self.batch_task))
        self.tasks = [Task(
            self.batch_task.iloc[i]['task_name'],
            self.batch_task.iloc[i]['start_time'],
            self.batch_task.iloc[i]['end_time'],
            self.batch_task.iloc[i]['plan_cpu'],
            self.batch_task.iloc[i]['plan_mem'],
        )
        for i in range(self.n_tasks)]
     
    def define_observation_space(self):
        # Define the maximum values for each feature
        max_cpu = 100.0  # Maximum CPU idle
        max_mem = 100.0  # Maximum memory empty
        max_last_time = 121971.0  # Maximum last_time (from your data)

        # Observation: [cpu_idle for all machines] + [mem_empty for all machines] + [nxt_task.plan_cpu, nxt_task.plan_mem, nxt_task.last_time]
        num_machines = self.n_servers
        observation_dim = num_machines * 2 + 3  # cpu_idle and mem_empty for each machine, plus 3 task features

        # Define the low and high bounds for the observation space
        obs_low = np.zeros(observation_dim, dtype=np.float32)  # All features have a minimum value of 0
        obs_high = np.full(observation_dim, max(max_cpu, max_mem, max_last_time), dtype=np.float32)  # All features share the same maximum value

        # Correctly define the observation space with low and high bounds
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.cur = 0
        self.power_usage = []
        self.latency = []
        self.machines = [Machine(
                100, 100,
                self.machine_meta.iloc[i]['machine_id']
            )
            for i in range(self.n_machines)
        ]
        observation = self.get_states(self.tasks[self.cur])
        info = {}
        return observation, info

    def step(self, action):
        self.cur_time = self.batch_task.iloc[self.cur]['start_time']
        cur_task = self.tasks[self.cur]

        terminated = False
        self.cur += 1

        if self.cur >= self.n_tasks:
            self.latency = [t.start_time - t.arrive_time for t in self.tasks]
            for i in range(1, len(self.latency)):
                self.latency[i] = self.latency[i] + self.latency[i - 1]

            terminated = True
            self.cur = 0  # Reset for potential next episode

        # Ensure nxt_task is valid
        if self.cur < self.n_tasks:
            nxt_task = self.tasks[self.cur]
        else:
            nxt_task = Task("", 0, 0, 0, 0)  # Dummy task

        # Simulate to current time
        for m in self.machines:
            m.process(self.cur_time)
        self.power_usage.append(np.sum([m.power_usage for m in self.machines]))

        self.machines[action].add_task(cur_task)

        truncated = False  # Assuming no truncation
        info = {"latency": self.latency, "power_usage": self.power_usage}

        observation = self.get_states(nxt_task)
        reward = self.get_reward(nxt_task)

        return observation, reward, terminated, truncated, info


    def get_states(self, nxt_task):
        """
        Get the current state of the environment and normalize the features.
        """
        # Define the maximum values for normalization
        max_cpu = 100.0  # Maximum CPU idle
        max_mem = 100.0  # Maximum memory empty
        max_last_time = 121971.0  # Maximum last_time (from your data)

        # Get the raw state values
        cpu_idle = [m.cpu_idle for m in self.machines]
        mem_empty = [m.mem_empty for m in self.machines]
        task_features = [nxt_task.plan_cpu, nxt_task.plan_mem, nxt_task.last_time]

        # Normalize the state values
        cpu_idle_normalized = [x / max_cpu for x in cpu_idle]
        mem_empty_normalized = [x / max_mem for x in mem_empty]
        task_features_normalized = [
            nxt_task.plan_cpu / max_cpu,
            nxt_task.plan_mem / max_mem,
            nxt_task.last_time / max_last_time,
        ]

        # Combine the normalized features into a single state array
        states = cpu_idle_normalized + mem_empty_normalized + task_features_normalized
        return np.array(states, dtype=np.float32)

    def get_reward(self, nxt_task):
        """
        Calculate the reward based on power usage and latency.
        """
        reward = -self.w1 * self.calc_total_power() - self.w2 * self.calc_total_latency()
        return reward

    def calc_total_power(self):
        total_power = 0
        for m in self.machines:
            cpu_usage = m.cpu()
            total_power += self.P_0 + (self.P_100 - self.P_0) * (2 * cpu_usage - cpu_usage ** 1.4)
        return total_power

    def calc_total_latency(self):
        latency = [t.start_time - t.arrive_time for t in self.tasks]
        for i in range(1, len(latency)):
            latency[i] = latency[i] + latency[i - 1]
        return np.sum(latency)

    def render(self, mode="human"):
        """
        Render the environment. Not implemented.
        """
        pass

    def close(self):
        """
        Clean up resources.
        """
        pass

class Task(object):
    def __init__(self, name, start_time, end_time, plan_cpu, plan_mem):
        self.name = name
        self.arrive_time = start_time
        self.last_time = end_time - start_time
        self.plan_cpu = plan_cpu
        self.plan_mem = plan_mem
        self.start_time = self.arrive_time
        self.end_time = end_time

    def start(self, start_time):
        self.start_time = start_time
        self.end_time = start_time + self.last_time

    def __lt__(self, other):
        return self.end_time < other.end_time

class Machine():
    def __init__(self, cpu_num, mem_size, machine_id):
        self.machine_id = machine_id
        self.P_0 = args.P_0
        self.P_100 = args.P_100
        self.T_on = args.T_on
        self.T_off = args.T_off

        self.pending_queue = deque()
        self.running_queue = []

        self.cpu_num = cpu_num
        self.mem_size = mem_size
        self.cpu_idle = cpu_num
        self.mem_empty = mem_size

        self.cur_time = 0
        self.awake_time = 0
        self.intervals = deque(maxlen=35 + 1)
        self.state = 'waken'  # waken, active, sleeping
        self.w = 0.5
        self.power_usage = 0
        self.last_arrive_time = 0

    def cpu(self):
        return 1 - self.cpu_idle / self.cpu_num

    def add_task(self, task):
        self.pending_queue.append(task)
        if self.state == 'sleeping':
            self.try_to_wake_up(task)
        self.process_pending_queue()

    def process_running_queue(self, cur_time):
        if self.is_empty(self.running_queue):
            return False
        if self.running_queue[0].end_time > cur_time:
            return False

        if self.is_empty(self.pending_queue) or \
                not self.enough_resource(self.pending_queue[0]) or \
                self.running_queue[0].end_time <= self.pending_queue[0].arrive_time:

            task = heapq.heappop(self.running_queue)
            self.state = 'active'
            self.cpu_idle += task.plan_cpu
            self.mem_empty += task.plan_mem

            # update power usage
            self.power_usage += self.calc_power(task.end_time)
            self.cur_time = task.end_time

            return True

        return False

    def process_pending_queue(self):
        if self.is_empty(self.pending_queue):
            return False
        if not self.enough_resource(self.pending_queue[0]):
            return False

        if self.is_empty(self.running_queue) or \
                self.pending_queue[0].arrive_time < self.running_queue[0].end_time:

            task = self.pending_queue.popleft()
            task.start(self.cur_time)
            self.cpu_idle -= task.plan_cpu
            self.mem_empty -= task.plan_mem
            heapq.heappush(self.running_queue, task)

            return True

        return False

    def process(self, cur_time):
        if self.cur_time == 0:
            self.cur_time = cur_time
            return
        if self.awake_time > cur_time:
            self.cur_time = cur_time
            return
        if self.awake_time > self.cur_time:
            self.cur_time = self.awake_time
            self.state = 'waken'

        while self.process_pending_queue() or self.process_running_queue(cur_time):
            pass
        self.power_usage += self.calc_power(cur_time)
        self.cur_time = cur_time

    def enough_resource(self, task):
        return task.plan_cpu <= self.cpu_idle and task.plan_mem <= self.mem_empty

    def is_empty(self, queue):
        return len(queue) == 0

    def calc_power(self, cur_time):
        if self.state == 'sleeping':
            return 0
        else:
            cpu = self.cpu()
            return (self.P_0 + (self.P_100 - self.P_0) * (2 * cpu - cpu ** 1.4)) * (cur_time - self.cur_time)

    def try_to_wake_up(self, task):
        if self.awake_time > task.arrive_time + self.T_on:
            self.awake_time = task.arrive_time + self.T_on

