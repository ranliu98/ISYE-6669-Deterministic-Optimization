# Implement the Product Manufacturing problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the LP
def solve_lp(id, T=12):

    # Model setup
    gm = gp.Model("manufacturing")

    # Create the data
    D, L, C, H = get_data(id, T)

    prod = gm.addVars(T)
    have = gm.addVars(T)

    gm.setObjective(sum([prod[i]*C[i] + have[i]*H[i] for i in range(T)]), GRB.MINIMIZE)

    gm.addConstr(have[0] == 100 + prod[0] - D[0])
    gm.addConstrs(have[i] == have[i-1] * 0.95 + prod[i] - D[i] for i in range(1,T))
    gm.addConstrs(prod[i] - prod[i - 1] <= L[i-1] for i in range(1,T))
    gm.addConstrs(prod[i] - prod[i - 1] >= -L[i-1] for i in range(1, T))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, T):
    np.random.seed(id)
    D = np.random.randint(100, 200, T)
    L = np.random.randint(5, 10, T - 1)
    C = np.random.rand(T)
    H = np.random.rand(T)
    return(D, L, C, H)

solve_lp(id=id)