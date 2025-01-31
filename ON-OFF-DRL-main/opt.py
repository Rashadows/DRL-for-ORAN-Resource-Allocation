import gurobipy as grb

class Optimal(object):
    def __init__(self, act_size, n_servers, tasks):
        self.name = 'optimal'       # set model name
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

        # Set model sense to minimization
        opt_model.ModelSense = grb.GRB.MINIMIZE
        opt_model.setObjective(objective)
        opt_model.optimize()

        # Extract the result
        for v in opt_model.getVars():
            if v.x > 0.5:
                selected_server = int(v.varName.split('_')[1])
                return selected_server

        # If no feasible solution found
        return None