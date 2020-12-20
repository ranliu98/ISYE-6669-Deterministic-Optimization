# Implement the Supply Allocation problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the IP
def solve_ip(id, m=50, n=500):

    gm = gp.Model("supplies")

    b, w, u = get_data(id, m, n)
    # print(b.shape, w.shape, u.shape) # (50,) (500,) (500, 50)

    x = gm.addVars(n,m, lb=0, ub=1, vtype=GRB.INTEGER)
    gm.setObjective(quicksum([sum([x[i,j]* u[i,j] for i in range(n)]) for j in range(m)]), GRB.MAXIMIZE)

    gm.addConstrs(quicksum([x[i,j] for j in range(m)]) <= 1 for i in range(n))
    gm.addConstrs(quicksum([x[i,j] * w[i] for i in range(n)]) <= b[j] for j in range(m))
        
    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, m, n):
    np.random.seed(id)
    r = n // m;
    b = 3 * r / 4 + (r * np.random.rand(m) / 4)
    w = np.random.rand(n)
    u = np.random.rand(n, m)
    return(b, w, u)


solve_ip(id=id)