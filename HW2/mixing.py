# Implement the Mixing problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the LP
def solve_lp(id, M=1000, N=100):

    gm = gp.Model("mixing")

    # Create the data
    c, a, f, u, d, m = get_data(id, M, N)

    plan = gm.addVars(M,N, lb=0.0)
    gm.setObjective(sum([(c[i] + f[i,j])*plan[i,j] for i in range(M) for j in range(N)]), GRB.MINIMIZE)

    gm.addConstrs(gp.quicksum([plan[i,j] * a[i] for i in range(M)]) <= u[j] * gp.quicksum([plan[i,j] for i in range(M)]) for j in range(N))
    gm.addConstrs(gp.quicksum([plan[i,j] for i in range(M)]) >= d[j] for j in range(N))
    gm.addConstrs(gp.quicksum([plan[i, j] for j in range(N)]) <= m[i] for i in range(M))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, M, N):
    np.random.seed(id)
    c =  np.random.rand(M)
    a = 0.05 + 0.2 * np.random.rand(M)
    u = 0.20 + 0.05 * np.random.rand(N)
    f = np.random.rand(M, N)
    d = np.random.randint(100, 200, N)
    m = np.random.randint(50, 150, M)
    return(c, a, f, u, d, m)


solve_lp(id=id)