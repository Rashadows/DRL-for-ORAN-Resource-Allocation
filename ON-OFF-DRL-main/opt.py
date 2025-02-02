import gurobipy as grb
from argparser import args
import numpy as np

class Optimal(object):
    def __init__(self, act_size, n_servers, tasks):
        self.name = 'Optimal'     
        self.n_servers = n_servers
        self.act_size = act_size
        self.tasks = tasks
        self.P_0 = args.P_0  # Ensure P_100 > P_0 (e.g., P_0=0.1, P_100=1.0)
        self.P_100 = args.P_100

    def step(self, obs):
        opt_model = grb.Model(name="MIP Model")
        opt_model.setParam('OutputFlag', 0)

        # Extract observations
        cpu_idle = {i: obs[i] for i in range(self.n_servers)}
        mem_empty = {i: obs[i + self.n_servers] for i in range(self.n_servers)}
        task_cpu = obs[2 * self.n_servers]
        task_mem = obs[2 * self.n_servers + 1]
        task_last_time = obs[2 * self.n_servers + 2]

        # Define binary decision variables
        x_vars = {i: opt_model.addVar(vtype=grb.GRB.BINARY, name=f"x_{i}") for i in range(self.n_servers)}
        opt_model.addConstr(grb.quicksum(x_vars[i] for i in x_vars) == 1, "assignment_constraint")

        # Capacity constraints
        for i in range(self.n_servers):
            opt_model.addConstr(x_vars[i] * task_cpu <= cpu_idle[i], f"cpu_capacity_{i}")
            opt_model.addConstr(x_vars[i] * task_mem <= mem_empty[i], f"mem_capacity_{i}")

        # Objective: Minimize energy (kWh) using linear power model from the paper
        objective = grb.quicksum(
            x_vars[i] * (
                (self.P_0 + (self.P_100 - self.P_0) * (1 - (cpu_idle[i] / 100)))  # Linear power (kW)
                * (task_last_time / 3600)  # Convert seconds to hours
            ) for i in range(self.n_servers)
        )
        opt_model.setObjective(objective, grb.GRB.MINIMIZE)

        # Solve and handle results
        try:
            opt_model.optimize()
            if opt_model.Status == grb.GRB.OPTIMAL:
                for i in x_vars:
                    if x_vars[i].X > 0.5:
                        return i
            else:
                # Fallback: Select server with highest CPU idle
                return max(range(self.n_servers), key=lambda x: cpu_idle[x])
        except Exception as e:
            print(f"Optimization failed: {e}")
            return np.random.choice(self.n_servers)