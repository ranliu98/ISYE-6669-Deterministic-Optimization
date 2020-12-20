# Implement the Markowitz Portfolio Optimization problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the QP
def solve_qp(id, n=500):

    gm = gp.Model("markowitz")
    mu, Sigma, B, r = get_data(id, n)

    x = gm.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    y = gm.addVars(n, lb=0, ub=1, vtype=GRB.INTEGER)
    M = 2*B

    gm.setObjective(sum([x[i] * x[j] * Sigma[i,j] for i in range(n) for j in range(n)]), GRB.MINIMIZE)

    gm.addConstr(sum([mu[i]*x[i] for i in range(n)]) >= r)
    gm.addConstr(sum([x[i]*(1-y[i]) for i in range(n)]) + sum([-x[i]*y[i] for i in range(n)]) <= B)
    gm.addConstrs(M*(1-y[i]) >= x[i] for i in range(n))
    gm.addConstrs(-M*y[i] <= x[i] for i in range(n))
    gm.addConstr(-sum([y[i]*x[i] for i in range(n)]) <= 0.2 * sum([(1-y[i])*x[i] for i in range(n)]))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, n):
    np.random.seed(id)
    mu = 0.8 * np.random.rand(n) - 0.2
    S_base = np.random.rand(n, n)
    S_orth, _ = np.linalg.qr(S_base)
    S_eigs = abs(mu) * (0.9 + 0.2 * np.random.rand(n)) / np.sqrt(n)
    Sigma = (S_orth @ np.diag(S_eigs)) @ S_orth.T
    B = n * 100
    r = 0.8 * mu.mean() * B
    return(mu, Sigma, B, r)

solve_qp(id=id)