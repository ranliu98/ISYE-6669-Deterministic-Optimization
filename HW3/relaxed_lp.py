# Implement the Relaxed LP problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the IP
def solve_ip(id, m=100, n=10, k=40):

    gm = gp.Model("relaxed_lp")

    A, b, c, lam = get_data(id, m, n)
    # print(A.shape, b.shape, c.shape) # (100, 10) (100,) (10,)

    x = gm.addVars(n, lb=-GRB.INFINITY)
    #x = gm.addVars(n, lb=0.0)
    decide = gm.addVars(m, lb=0, ub=1, vtype=GRB.INTEGER)
    #M = gm.addVar(lb=1)

    gm.setObjective(sum([c[i]*x[i] for i in range(n)]), GRB.MINIMIZE)

    gm.addConstrs(sum([A[j,i]*x[i] for i in range(n)]) >= lam for j in range(m))
    gm.addConstrs(sum([A[j,i]*x[i] for i in range(n)]) + (b[j]-lam)*decide[j] >= b[j] for j in range(m))
    gm.addConstr(sum([decide[j] for j in range(m)]) <= (m-k))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, m, n):
    np.random.seed(id)
    A = np.random.rand(m, n)
    b = -5 * np.random.rand(m)
    c = 100 + np.random.rand(n)
    lam = b.min()
    return(A, b, c, lam)

solve_ip(id=id)
