# Implement the Non-Euclidean distance minimization problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the QCQP
def solve_qcqp(id, m=400, n=200):

    gm = gp.Model("non_euclidean")
    M, b = get_data(id, m, n)
    # print(M.shape, b.shape) # (400, 200) (400,)

    z = gm.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    y = gm.addVars(m, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    x = gm.addVars(m, lb=0.0)

    gm.setObjective(sum([x[mi]* x[mi] for mi in range(m)]), GRB.MINIMIZE)

    gm.addConstrs(y[mi] ==(sum([M[mi,ni] * z[ni] for ni in range(n)])-b[mi]) for mi in range(m))
    gm.addConstrs(x[mi] >= y[mi]*y[mi] for mi in range(m))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, m, n):
    np.random.seed(id)
    M = np.sqrt(m * n) * np.random.rand(m, n)
    b = np.sqrt(m) * np.random.rand(m)
    #print(M,b)
    return(M, b)

solve_qcqp(id=id)