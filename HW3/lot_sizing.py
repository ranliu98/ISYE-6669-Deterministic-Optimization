# Implement the Lot Sizing problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id= 903515184

# Invoke Gurobi to solve the IP
def solve_ip(id, T=365):

    gm = gp.Model("lot_sizing")

    c, p, s, b, D = get_data(id, T)
    #print(c.shape, p.shape, s.shape, b.shape, D.shape, c)

    x_pd = gm.addVars(T, lb=0, vtype=GRB.INTEGER) # x
    x_st = gm.addVars(T, lb=0, vtype=GRB.INTEGER) # e
    x_de = gm.addVars(T, lb=0, ub=1, vtype=GRB.INTEGER) # f
    x_pe = gm.addVars(T, lb=0, vtype=GRB.INTEGER) # g
    #M = gm.addVar(lb=1)
    M = np.sum(D)

    gm.setObjective(sum([x_de[t] * c[t] + x_pd[t] * p[t] + x_st[t] * s[t] + x_pe[t] * b[t] for t in range(T)]), GRB.MINIMIZE)

    gm.addConstrs(x_pd[t] <= x_de[t] * M for t in range(T))

    gm.addConstr(x_st[0] == x_pd[0] - D[0] + x_pe[0])
    gm.addConstrs(x_st[t+1] == x_st[t] + x_pd[t+1] - D[t+1] + x_pe[t+1] for t in range(T-1))

    gm.addConstr(x_pe[0] >= D[0] - 0 - x_pd[0])
    gm.addConstrs(x_pe[t] >= D[t] - x_st[t-1] - x_pd[t] for t in range(1,T))
    gm.addConstr(x_pe[T-1] == 0)
        
    # Solve the model
    gm.update()
    gm.optimize()

    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, T):
    np.random.seed(id)
    c = 200 * np.random.rand(T);
    p = 10 + 40 * np.random.rand(T);
    s = 10 + 90 * np.random.rand(T);
    b = 200 * np.random.rand(T);
    D = np.random.randint(1, 200, T);
    return(c, p, s, b, D)

solve_ip(id=id)