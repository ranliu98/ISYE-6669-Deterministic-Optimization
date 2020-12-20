# Implement the Radius Maximization problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the LP
def solve_lp(id, n=100, m=1000):


    gm = gp.Model("max_radius")
    A, b = get_data(id, n, m)
    #print(A.shape, b.shape) #(1000, 100) (1000,)

    y = gm.addVars(n, lb=-GRB.INFINITY)
    r = gm.addVar(lb=0.0)

    gm.setObjective(r, GRB.MAXIMIZE)
    gm.addConstrs(sum([A[mi,ni]*y[ni] for ni in range(n)]) - b[mi] + r <= 0 for mi in range(m))
    #gm.addConstrs(y_ax[ni] == sum([A[mi, ni] * y[mi] for mi in range(m)]) - b[ni] for ni in range(n))
    #gm.addConstr(min_numb == min_([y_ax[ni] for ni in range(n)]))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, n, m):
    np.random.seed(id)
    b =  1 + 9 * np.random.rand(m)
    A_base = 2 * (np.random.rand(m, n) - 0.5)
    A_row_norm = np.sum(np.sqrt(A_base ** 2), 1)
    scaling_factor = b / A_row_norm
    A = np.multiply(A_base.transpose(), scaling_factor).transpose()
    return(A, b)


solve_lp(id=id)