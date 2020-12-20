# Implement the Transportation Network problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the IP
def solve_ip(id, n=30):

    gm = gp.Model("transportation")
    b, f, c = get_data(id, n)
    # print(b.shape, f.shape, c.shape)

    x = gm.addVars(n, n, lb=0, vtype=GRB.CONTINUOUS, name='x') # x[i,j] from i to j. x[j,i] from j to i.
    y = gm.addVars(n, n, lb=0, ub=1, vtype=GRB.INTEGER, name='y')
    M = np.sum(np.abs(b))

    gm.setObjective(sum([x[i,j]*c[i,j] + y[i,j]*f[i,j] for i in range(n) for j in range(n) if i != j]), GRB.MINIMIZE)

    gm.addConstrs(x[i,i] == 0 for i in range(n))
    gm.addConstrs(x[i,j] <= y[i,j]*M for i in range(n) for j in range(n))
    gm.addConstrs(b[i] + sum([x[i,j] for j in range(n)]) - sum([x[j,i] for j in range(n)]) >= 0 for i in range(n))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, n):
    np.random.seed(id)
    b_base = np.random.rand(n)
    b_pivot = 0.95 * b_base.mean() + 0.05 * b_base.min()
    b = 10 * (b_base - b_pivot)
    base_reg = abs(np.array([range(-i, n - i) for i in range(0, n)]))
    f = base_reg * (0.75 + 0.25 * np.random.rand(n, n)) * 100
    c = np.square(base_reg) * (0.75 + 0.25 * np.random.rand(n, n))
    print(b, b.shape, ' --- \n ---', f, f.shape, '---\n---', c, c.shape)
    return(b, f, c)

solve_ip(id=id)
