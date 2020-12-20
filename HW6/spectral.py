# Implement the Spectrally-bounded QP problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the QP
def solve_qp(id, m=400, n=200):

    gm = gp.Model("spectral")
    Q0, A, b, c, d, gamma = get_data(id, m, n)

    x = gm.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    gm.setObjective(0.5* sum([x[i]* Q0[i,j]* x[j] for i in range(n) for j in range(n)])
                    + 0.5 * sum([x[i]* gamma * x[i] for i in range(n)]) - sum([c[i]*x[i] for i in range(n)]) + d
                    , GRB.MINIMIZE)

    #gm.addConstrs(sup_value >= 0.5* sum([x[i]* Q0[i,j]* x[j] for i in range(n) for j in range(n)])
    #              + sum([x[i]* gamma * x[i] for i in range(n)]) - sum([c[i]*x[i] for i in range(n)]) + d)
    gm.addConstrs(sum([A[k,j]*x[j] for j in range(n)]) <= b[k] for k in range(m))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, m, n):
    np.random.seed(id)
    # Q0 logic
    Q0_base = np.random.rand(n, n)
    Q0_orth, _ = np.linalg.qr(Q0_base)
    Q0_eigs = 10 * np.random.rand(n)
    Q0 = (Q0_orth @ np.diag(Q0_eigs)) @ Q0_orth.T
    # Other elements
    b =  1 + 9 * np.random.rand(m)
    A_base = 2 * (np.random.rand(m, n) - 0.5)
    A_row_norm = np.sum(np.sqrt(A_base ** 2), 1)
    scaling_factor = b / A_row_norm
    A = np.multiply(A_base.transpose(), scaling_factor).transpose()
    c = 2 * (np.random.rand(n) - 0.5)
    d = 100 * np.random.rand()
    gamma = 10 * np.random.rand()
    return(Q0, A, b, c, d, gamma)

solve_qp(id=id)