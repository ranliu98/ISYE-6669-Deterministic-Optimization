# Implement the Subset Selection problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the IP
def solve_ip(id, m=700, n=400, time_limit=30):

    gm = gp.Model("subset")
    h, c = get_data(id, m, n)
    # print(h.shape,c.shape) # (700, 400) (400,)

    F = gm.addVars(n, lb=0, ub=1, vtype=GRB.INTEGER)
    # F_a = gm.addVars(m, n, lb=-GRB.INFINITY)
    F_max = gm.addVars(m, n, lb=0, ub=1, vtype=GRB.INTEGER)

    gm.setObjective(sum([F_max[mi,ni]*h[mi,ni] for mi in range(m) for ni in range(n)]) - sum([c[ni]*F[ni] for ni in range(n)]), GRB.MAXIMIZE)

    gm.addConstrs(F_max[mi, ni] <= F[ni] for ni in range(n) for mi in range(m))
    gm.addConstrs(sum([F_max[mi,ni] for ni in range(n)]) == 1 for mi in range(m))

    #gm.addConstrs(F_max[mi] == max_([F_a[mi,ni] for ni in range(n)]) for mi in range(m))

    # Solve the model
    gm.Params.TimeLimit = time_limit
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, m, n):
    np.random.seed(id)
    h = abs(np.random.normal(size=(m, n)))
    c = abs(np.random.normal(size=n))
    return(h, c)

solve_ip(id=id)