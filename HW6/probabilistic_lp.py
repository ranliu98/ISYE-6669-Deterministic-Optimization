# Implement the Probabilistic LP problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the QP
def solve_qp(id, m=50, n=200):

    gm = gp.Model("probabilistic_lp")
    mu, Sigma, alpha, A, b = get_data(id, m, n)

    #x = gm.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    y = gm.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    s = gm.addVar(lb=0, ub=GRB.INFINITY)

    gm.setObjective(sum([y[i] * Sigma[i, j] * y[j] for i in range(n) for j in range(n)]), GRB.MINIMIZE)

    gm.addConstrs(sum([A[mi, ni] * y[ni] for ni in range(n)]) == b[mi] * s for mi in range(m))
    gm.addConstr(sum([mu[j] * y[j] for j in range(n)]) - alpha * s == 1)

    gm.addConstrs(-y[j] <= 0 for j in range(n))

    # Solve the model
    #gm.setParam("NonConvex", 2)
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, m, n):
    np.random.seed(id)
    mu = 5 + 5 * np.random.rand(n)
    S_base = np.random.rand(n, n)
    S_orth, _ = np.linalg.qr(S_base)
    S_eigs = 5 * n * np.random.rand(n) 
    Sigma = (S_orth @ np.diag(S_eigs)) @ S_orth.T
    A = np.random.rand(m, n)
    A[0, 0:n] = 1
    b = np.random.rand(m)
    b[0] = 10
    alpha = np.random.rand(1)[0]
    return(mu, Sigma, alpha, A, b)

solve_qp(id=id)