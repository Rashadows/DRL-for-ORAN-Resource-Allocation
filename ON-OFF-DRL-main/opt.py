import gurobipy as grb
from argparser import args
from time import sleep

class Optimal(object):
    def __init__(self, act_size, n_servers, tasks):
        self.name = 'optimal'     
        self.n_servers = n_servers
        self.act_size = act_size
        self.tasks = tasks
        self.P_0 = args.P_0
        self.P_100 = args.P_100

    def step(self, obs):
        opt_model = grb.Model(name="MIP Model")
        opt_model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        # Extract observations
        cpu_idle = {i: obs[i] for i in range(self.n_servers)}
        mem_empty = {i: obs[i + self.n_servers] for i in range(self.n_servers)}

        # Calculate CPU utilization
        cpu_utilization = {i: 1 - (cpu_idle[i] / 100) for i in range(self.n_servers)}

        # Task requirements
        task_cpu = obs[2 * self.n_servers]
        task_mem = obs[2 * self.n_servers + 1]
        task_last_time = obs[2 * self.n_servers + 2]

        # Define binary decision variables
        x_vars = {i: opt_model.addVar(vtype=grb.GRB.BINARY, name="x_{}".format(i))
                  for i in range(self.n_servers)}

        # Update model to integrate new variables
        opt_model.update()

        # Add assignment constraint
        opt_model.addConstr(
            lhs=grb.quicksum(x_vars[i] for i in range(self.n_servers)),
            sense=grb.GRB.EQUAL,
            rhs=1,
            name="assignment_constraint"
        )

        # Add capacity constraints
        for i in range(self.n_servers):
            # CPU capacity constraint
            opt_model.addConstr(
                lhs=x_vars[i] * task_cpu,
                sense=grb.GRB.LESS_EQUAL,
                rhs=cpu_idle[i],
                name="cpu_capacity_{}".format(i)
            )
            # Memory capacity constraint
            opt_model.addConstr(
                lhs=x_vars[i] * task_mem,
                sense=grb.GRB.LESS_EQUAL,
                rhs=mem_empty[i],
                name="mem_capacity_{}".format(i)
            )

        # Set objective function (including task_last_time)
        objective = grb.quicksum(
            x_vars[i] * (
                self.P_0 + (self.P_100 - self.P_0) * (
                    2 * cpu_utilization[i] - cpu_utilization[i] ** 1.4
                )
            ) * task_last_time  # Multiply power rate by task duration
            for i in range(self.n_servers)
        )

        print("Task Requirements:")
        print(f" - CPU: {task_cpu}")
        print(f" - Memory: {task_mem}")
        print("Server Resource Availability:")
        for i in range(self.n_servers):
            print(f"Server {i}:")
            print(f" - CPU Idle: {cpu_idle[i]}")
            print(f" - Memory Empty: {mem_empty[i]}")
        sleep(100)
        # Set model sense to minimization
        opt_model.ModelSense = grb.GRB.MINIMIZE
        opt_model.setObjective(objective)
        opt_model.update()  # Ensure all changes are updated

        # Optimize the model
        opt_model.optimize()

        # Check if the optimization was successful
        status = opt_model.Status
        if status == grb.GRB.OPTIMAL or status == grb.GRB.SUBOPTIMAL:
            # Extract the result
            for v in opt_model.getVars():
                if v.X > 0.5:
                    selected_server = int(v.varName.split('_')[1])
                    return selected_server
            # If no variable is greater than 0.5, default action
            print("No variable assignment found in optimal solution.")
            return None
        else:
            # Handle infeasible or unbounded models
            print(f"Optimization was unsuccessful. Status code: {status}")
            if status == grb.GRB.INFEASIBLE:
                print("Model is infeasible. No feasible solution exists.")
                print("Computing IIS to diagnose infeasibility...")
                opt_model.computeIIS() # Compute the Irreducible Inconsistent Subsystem (IIS)
                opt_model.write("model.ilp") # Write a file model.ilp that contains the IIS for further analysis
                for c in opt_model.getConstrs():
                    if c.IISConstr:
                        print(f"Infeasible constraint: {c.ConstrName}") # Output constraints that are causing the infeasibility
            elif status == grb.GRB.UNBOUNDED:
                print("Model is unbounded.")
            else:
                print("Optimization ended with status:", status)
            # Decide how to handle this situation, e.g., return None or a random action
            return None