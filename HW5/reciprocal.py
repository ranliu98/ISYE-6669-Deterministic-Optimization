
# Implement the Linear Reciprocal Maximization problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the SOCP
def solve_socp(id, m=400, n=50):

    gm = gp.Model("reciprocal")
    A, b = get_data(id, m, n)
    # print(A.shape, b.shape) # (400, 50) (400,)

    x = gm.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    y = gm.addVars(m, lb=0, ub=GRB.INFINITY)

    Ax = gm.addVars(m, lb=-GRB.INFINITY)
    z1 = gm.addVars(m, lb=-GRB.INFINITY)
    z2 = gm.addVars(m, lb=0)

    gm.setObjective(sum([y[mi] for mi in range(m)]), GRB.MINIMIZE)

    gm.addConstrs(Ax[mi] == sum([A[mi, ni] * x[ni] for ni in range(n)]) for mi in range(m))
    gm.addConstrs((Ax[mi] - b[mi]) >= 0 for mi in range(m)) # Ax-b > 0

    gm.addConstrs(z1[mi] == y[mi] - Ax[mi] + b[mi] for mi in range(m))
    gm.addConstrs(z2[mi] == y[mi] + Ax[mi] - b[mi] for mi in range(m))

    gm.addConstrs(4 + z1[mi]*z1[mi] <= z2[mi]*z2[mi] for mi in range(m))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, m, n):
    np.random.seed(id)
    bm =  1 + 9 * np.random.rand(m)
    Am_base = 2 * (np.random.rand(m, n) - 0.5)
    Am_row_norm = np.sum(np.sqrt(Am_base ** 2), 1)
    scaling_factor = bm / Am_row_norm
    Am = np.multiply(Am_base.transpose(), scaling_factor).transpose()
    A = -Am
    b = -bm
    return(A, b)

solve_socp(id=id)