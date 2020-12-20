# Implement the Job Processing problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the IP
def solve_ip(id, m=4, n=8):
    # m machines, n jobs

    gm = gp.Model("job_processing")
    I, t = get_data(id, m, n)
    # print(I.shape, t.shape) # (8, 4) (4, 8)

    I = I - 1

    x = gm.addVars(m, n, lb=0, vtype=GRB.CONTINUOUS) # starting time of jobni on machine m, it should be n=8 jobs
    y = []
    for ni_ in range(n):
        y.append(gm.addVars(m, n, lb=0, ub=1, vtype=GRB.INTEGER))
    end_time = gm.addVars(m, lb=0, vtype=GRB.CONTINUOUS)
    end_max = gm.addVar(lb=0, vtype=GRB.CONTINUOUS)

    gm.setObjective(end_max, GRB.MINIMIZE)

    gm.addConstrs(end_time[mi] == sum([(x[mi, n - 1] + t[mi, jobni]) * y[jobni][mi, n - 1] for jobni in range(n)]) for mi in range(m))
    gm.addConstr(end_max == max_([end_time[mi] for mi in range(m)]))

    gm.addConstrs(sum([sum([y[jobni][mi,ni] for ni in range(n)]) for mi in range(m)]) == 4 for jobni in range(n)) # total = 4 for jobni
    gm.addConstrs(sum([y[jobni][mi,ni] for ni in range(n)]) == 1 for mi in range(m) for jobni in range(n)) # every machine has 1 for jobni
    gm.addConstrs(sum([y[jobni][mi,ni] for jobni in range(n)]) == 1 for mi in range(m) for ni in range(n)) # every place happen once


    for mi in range(m):
        gm.addConstrs(sum([y[jobni][mi,ni] * (x[mi,ni] + t[mi,jobni]) for jobni in range(n)]) <= sum([y[jobni][mi,ni+1] * x[mi,ni+1] for jobni in range(n)]) for ni in range(n-1))

    for jobni in range(n):
        for stepi in range(m-1):
            ma_step_i = I[jobni,stepi]
            ma_step_i1 = I[jobni,stepi+1]
            gm.addConstr(sum([(x[ma_step_i, ni] + t[ma_step_i, jobni]) * y[jobni][ma_step_i, ni] for ni in range(n)]) <= sum([x[ma_step_i1, ni] * y[jobni][ma_step_i1, ni] for ni in range(n)]))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

def get_data(id, m, n):
    np.random.seed(id)
    I = np.tile(np.arange(1, m + 1), (n, 1))
    for j in range(len(I)):
        I[j] = np.random.permutation(I[j])
    t = 1 + 9 * np.random.rand(m, n)
    print(t, I)
    return(I, t)

solve_ip(id=id)

