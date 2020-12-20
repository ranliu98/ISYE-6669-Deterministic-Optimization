# Implement the Uncertain LP problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the LP
def solve_lp(id, n=100, m=1000):

    gm = gp.Model("uncertain_lp")

    A_bar, b, delta = get_data(id, n, m)

    x = gm.addVars(n, lb=-GRB.INFINITY)
    posi_x = gm.addVars(n, lb=0.0)
    neg_x = gm.addVars(n, lb=-GRB.INFINITY, ub=0.0)
    #uncer = gm.addVars(m, n, lb=-GRB.INFINITY)

    gm.setObjective(sum([50 * neg_x[ni] + 150 * posi_x[ni] for ni in range(n)]), GRB.MINIMIZE)

    gm.addConstrs(sum([A_bar[mi,ni] * x[ni] + delta[mi] * (posi_x[ni]-neg_x[ni]) for ni in range(n)]) <= b[mi] for mi in range(m))
    #gm.addConstrs(sum([A_bar[mi,ni] * x[ni] + uncer[mi,ni] * x[ni] for ni in range(n)]) <= b[mi] for mi in range(m))
    #gm.addConstrs(sum([uncer[mi,ni] * x[ni] for ni in range(n)]) <= delta[mi] * sum([posi_x[ni]-neg_x[ni] for ni in range(n)]) for mi in range(m))
    #gm.addConstrs()
    gm.addConstrs(x[ni] == neg_x[ni] + posi_x[ni] for ni in range(n))

    #max_value = gm.addVars(m, lb=-GRB.INFINITY)
    #min_value = gm.addVars(m, lb=-GRB.INFINITY)

    #gm.setObjective(
    #    sum([50 * x[ni] * (1/2 - x[ni]/(2* abs_(x[ni] + 1e-7))) + 150 * x[ni] * (1/2 + x[ni]/(2* abs_(x[ni] + 1e-7))) for ni in range(n)]), GRB.MINIMIZE
    #)
    #gm.addConstrs(max_value[mi] == max_(([uncer[mi,ni] for ni in range(n)])) for mi in range(m))
    #gm.addConstrs(min_value[mi] == min_(([uncer[mi, ni] for ni in range(n)])) for mi in range(m))
    #gm.addConstrs(max_value[mi] <= delta[mi] for mi in range(m))
    #gm.addConstrs(min_value[mi] >= -delta[mi] for mi in range(m))

    # gm.setParam("NonConvex", 2)
    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, n, m):
    np.random.seed(id)
    b =  100 + 2 * (np.random.rand(m) - 0.5)
    A_base = 2 * (np.random.rand(m, n) - 0.5)
    A_row_norm = np.sum(np.sqrt(A_base ** 2), 1)
    scaling_factor = b / A_row_norm
    A_bar = np.multiply(A_base.transpose(), scaling_factor).transpose()
    delta = np.random.rand(m)
    return(A_bar, b, delta)

solve_lp(id=id)