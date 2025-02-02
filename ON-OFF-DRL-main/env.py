# -*- coding: utf-8 -*-

"""
Code adapted from: https://github.com/gohsyi/cluster_optimization
"""

############################### Import libraries ###############################

import os
import heapq
import pandas as pd
import numpy as np
import random
from collections import deque
from argparser import args


class Env():
    def __init__(self):
        
        self.P_0 = args.P_0
        self.P_100 = args.P_100
        self.T_on = args.T_on
        self.T_off = args.T_off
        self.n_servers = args.n_servers
        self.w1 = args.w1
        self.w2 = args.w2
        self.w3 = args.w3

        #  data paths
        self.machine_meta_path = os.path.join('data', 'machine_meta.csv')
        self.machine_usage_path = os.path.join('data', 'machine_usage.csv')
        self.container_meta_path = os.path.join('data', 'container_meta.csv')
        self.container_usage_path = os.path.join('data', 'container_usage.csv')
        self.batch_task_path = os.path.join('data', 'batch_task.csv')
        self.batch_instance_path = os.path.join('data', 'batch_instance.csv')

        #  data columns
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
            'net_in',  # normarlized in coming network traffic, [0, 100]
            'net_out',  # normarlized out going network traffic, [0, 100]
            'disk_io_percent',  # [0, 100], abnormal values are of -1 or 101 |
        ]
        self.container_meta_cols = [
            'container_id',  # uid of a container
            'machine_id',  # uid of container's host machine  
            'time_stamp',  # 
            'app_du',  # containers with same app_du belong to same application group
            'status',  # 
            'cpu_request',  # 100 is 1 core 
            'cpu_limit',  # 100 is 1 core 
            'mem_size',  # normarlized memory, [0, 100]
        ]
        self.container_usage_cols = [
            'container_id',  # uid of a container
            'machine_id',  # uid of container's host machine  
            'time_stamp',  # 
            'cpu_util_percent',
            'mem_util_percent',
            'cpi',
            'mem_gps',  # normalized memory bandwidth, [0, 100]
            'mpki',
            'net_in',  # normarlized in coming network traffic, [0, 100]
            'net_out',  # normarlized out going network traffic, [0, 100]
            'disk_io_percent'  # [0, 100], abnormal values are of -1 or 101
        ]
        self.batch_task_cols = [
            'task_name',  # task name. unique within a job
            'instance_num',  # number of instances  
            'job_name',  # job name
            'task_type',  # task type
            'status',  # task status
            'start_time',  # start time of the task
            'end_time',  # end of time the task
            'plan_cpu',  # number of cpu needed by the task, 100 is 1 core
            'plan_mem'  # normalized memorty size, [0, 100]
        ]
        self.batch_instance_cols = [
            'instance_name',  # instance name of the instance
            'task_name',  # task name. unique within a job
            'instance_num',  # number of instances  
            'job_name',  # job name
            'task_type',  # task type
            'status',  # task status
            'start_time',  # start time of the task
            'end_time',  # end of time the task
            'machine_id',  # uid of host machine of the instance
            'seq_no'  # sequence number of this instance
            'total_seq_no',  # total sequence number of this instance
            'cpu_avg',  # average cpu used by the instance, 100 is 1 core
            'cpu_max',  # max cpu used by the instance, 100 is 1 core
            'mem_avg',  # average memory used by the instance (normalized)
            'mem_max',  # max memory used by the instance (normalized, [0, 100])
        ]

        self.cur = 0
        self.loadcsv()
        self.latency = []
        
        
    def loadcsv(self):

        #  read csv into DataFrames
        self.machine_meta = pd.read_csv(self.machine_meta_path, header=None, names=self.machine_meta_cols)
        self.machine_meta = self.machine_meta[self.machine_meta['time_stamp'] == 0]
        self.machine_meta = self.machine_meta[['machine_id', 'cpu_num', 'mem_size']]

        self.batch_instance = pd.read_csv(self.batch_instance_path, header=None, names=self.batch_instance_cols)
        self.batch_instance = self.batch_instance[(self.batch_instance['cpu_avg'] != "") & (self.batch_instance['mem_avg'] != "")]
        self.batch_instance = self.batch_instance[['task_name', 'cpu_avg','mem_avg']]

        self.batch_task = pd.read_csv(self.batch_task_path, header=None, names=self.batch_task_cols)
        self.batch_task = self.batch_task[self.batch_task['status'] == 'Terminated']
        self.batch_task = self.batch_task[self.batch_task['plan_cpu'] < 100]  # will stuck the pending queue
        self.batch_task = self.batch_task.sort_values(by='start_time')

        self.n_machines = self.n_servers
        self.n_tasks = 2000
        self.tasks = [ Task(
            self.batch_task.iloc[i]['task_name'],
            self.batch_task.iloc[i]['start_time'],
            self.batch_task.iloc[i]['end_time'],
            self.batch_task.iloc[i]['plan_cpu'],
            self.batch_task.iloc[i]['plan_mem'],
        ) for i in range(self.n_tasks) ]

    def reset(self):
        self.cur = 0
        self.power_usage = []
        self.latency = []
        self.machines = [ Machine(
            100, 100,
            self.machine_meta.iloc[i]['machine_id']
        ) for i in range(self.n_machines) ]

        return self.get_states(self.tasks[self.cur])
    
    def step(self, action):
        self.cur_time = self.batch_task.iloc[self.cur]['start_time']
        cur_task = self.tasks[self.cur]

        done = False

        # Assign the current task to the chosen machine
        self.machines[action].add_task(cur_task)

        # Process tasks
        for m in self.machines:
            m.process(self.cur_time)

        # Now that we've processed tasks, cur_task.start_time should be set
        if cur_task.start_time is not None:
            current_latency = cur_task.start_time - cur_task.arrive_time
        else:
            # Task hasn't started yet; latency is the time until now
            current_latency = self.cur_time - cur_task.arrive_time

        self.latency.append(current_latency)

        # Calculate power usage for current time step
        current_power_usage = np.sum([m.power_usage for m in self.machines])
        self.power_usage.append(current_power_usage)

        # Advance to the next task
        self.cur += 1
        if self.cur == self.n_tasks:
            done = True
            self.cur = 0  # Reset for next episode

        nxt_task = self.tasks[self.cur]

        # Return scalar latency and power
        return self.get_states(nxt_task), self.get_reward(cur_task), done, (current_latency, current_power_usage)

    def get_states(self, nxt_task):
        states = [m.cpu_idle for m in self.machines] + \
                 [m.mem_empty for m in self.machines] + \
                 [nxt_task.plan_cpu, nxt_task.plan_mem, nxt_task.last_time]
        
        return np.array(states)
    
    def get_reward(self, nxt_task):
        # Calculate the total power consumption at time t
        total_power = self.calc_total_power()  # Sum over all machines

        # Calculate P_min and P_max
        P_min = self.n_machines * self.P_0
        P_max = self.n_machines * self.P_100

        # Normalize the total power consumption
        normalized_power = (total_power - P_min) / (P_max - P_min)

        # # Calculate the normalized reward
        normalized_reward = 1 - normalized_power

        # Return the normalized reward
        return normalized_reward
    
    # def get_reward(self, nxt_task):
    #     """
    #     Calculate the reward based on power usage, latency, task completion, 
    #     and penalties for exceeding CPU/memory constraints.
    #     """
    #     penalty_factor_base = 0.15
    #     task_completion_reward = 10  # Reward for completing a task successfully
    #     unfinished_task_penalty = -5  # Penalty for tasks left incomplete
    #     reward = -self.w1 * self.calc_total_power() - self.w2 * self.calc_total_latency()

    #     # Check if the task's CPU and memory utilization exceeds the critical threshold
    #     critical_threshold = 0.9
    #     task_row = None
    #     if 'task_name' in self.batch_instance.columns and hasattr(nxt_task, 'name'):
    #         # Select the row using task_name if available
    #         task_row = self.batch_instance.loc[self.batch_instance['task_name'] == nxt_task.name]
    #     elif isinstance(nxt_task, int) and 0 <= nxt_task < len(self.batch_instance):
    #         # Fallback to index-based access if task_name is unavailable
    #         task_row = self.batch_instance.iloc[[nxt_task]]

    #     if task_row is not None and not task_row.empty:
    #         cpu_avg = task_row['cpu_avg'].iloc[0]
    #         mem_avg = task_row['mem_avg'].iloc[0]

    #         if cpu_avg > critical_threshold or mem_avg > critical_threshold:
    #             penalty_factor = penalty_factor_base * max(cpu_avg - critical_threshold, mem_avg - critical_threshold)
    #             reward -= self.w3 * penalty_factor  # Apply scaled penalty

    #     # Reward for task completion
    #     if hasattr(nxt_task, 'start_time') and hasattr(nxt_task, 'end_time'):
    #         if nxt_task.end_time <= self.cur_time:
    #             reward += task_completion_reward
    #         else:
    #             reward += unfinished_task_penalty

    #     return reward
       
    def calc_total_power(self):
        total_power = 0
        for m in self.machines:
            cpu = m.cpu()
            power_m = self.P_0 + (self.P_100 - self.P_0) * cpu
            total_power += power_m
        return total_power
    
    def calc_total_latency(self):
        latency_list = [t.start_time - t.arrive_time for t in self.tasks]
        cumulative_latency = np.cumsum(latency_list)
        total_latency = cumulative_latency[-1]
        return total_latency
    
    
    def sample_action(self):
        rand_machine = random.randint(0, self.n_machines-1)
        return rand_machine
    
class Task(object):
    def __init__(self, name, start_time, end_time, plan_cpu, plan_mem):
        self.name = name
        self.arrive_time = start_time
        self.last_time = end_time - start_time
        self.plan_cpu = plan_cpu
        self.plan_mem = plan_mem
        self.start_time = None

    def start(self, start_time):
        self.start_time = start_time
        self.end_time = start_time + self.last_time
    """
    def done(self, cur_time):
        return cur_time >= self.start_time + self.last_time
    """
    def __lt__(self, other):
        return self.start_time + self.last_time < other.start_time + other.last_time


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
        if self.state == 'sleeping':
            self.try_to_wake_up(task)
        # Allocate resources immediately (even if server is waking up)
        self.cpu_idle -= task.plan_cpu
        self.mem_empty -= task.plan_mem
        self.pending_queue.append(task)
        self.process_pending_queue()  # Start processing if possible

    def process_running_queue(self, cur_time):
        """
        Process running queue, return whether we should process running queue or not
        We should process running queue first if it's not empty and any of these conditions holds:
        1. Pending queue is empty
        2. The first task in pending queue cannot be executed for the lack of resources (cpu or memory)
        3. The first task in pending queue arrives after any task in the running queue finishes
        """

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
        """
        We should process pending queue first if it's not empty and
        the server has enough resources (cpu and memory) for the first task in the pending queue to run and
        any of these following conditions holds:
        1. Running queue is empty
        2. The first task in the pending queue arrives before all tasks in the running queue finishes
        """
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
        """
        Keep running simulation until current time.
        """
        if self.cur_time == 0:  # First time, no task has come before
            self.cur_time = cur_time
            return
        if self.awake_time > cur_time:  # Will not be awakened at cur_time
            self.cur_time = cur_time
            return
        if self.awake_time > self.cur_time:  # Jump to self.awake_time
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
            time_elapsed_hours = (cur_time - self.cur_time) / 3600
            cpu_util = 1 - (self.cpu_idle / 100)  # Use 0-1 scale
            power_kW = self.P_0 + (self.P_100 - self.P_0) * cpu_util  # Linear model
            return power_kW * time_elapsed_hours  # kWh

    def try_to_wake_up(self, task):
        if (self.awake_time > task.arrive_time + self.T_on):
            self.awake_time = task.arrive_time + self.T_on