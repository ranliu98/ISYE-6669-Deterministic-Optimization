# Implement the Supply and Demand problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the QP
def solve_qp(id, q=200, N=50, M=75):

    gm = gp.Model("supply_and_demand")
    A, b, C, d, E, F, g = get_data(id, q, N, M)

    print(A.shape, b.shape, C.shape, d.shape, E.shape, F.shape, g.shape) # (50, 50) (50,) (75, 75) (75,) (200, 50) (200, 75) (200,)

    x = gm.addVars(N, lb=0.0) # products sold
    y = gm.addVars(M, lb=0.0) # inputs purchased

    gm.setObjective((sum([x[ni]*b[ni] for ni in range(N)]) - sum([x[ni1]*A[ni1, ni2]*x[ni2] for ni1 in range(N) for ni2 in range(N)]))
                    - (sum([y[mi]*d[mi] for mi in range(M)]) + sum([y[mi1]*C[mi1,mi2]*y[mi2] for mi1 in range(M) for mi2 in range(M)]))
                    , GRB.MAXIMIZE)
    gm.addConstrs(sum([E[k,ni]*x[ni] for ni in range(N)]) + sum([F[k,mi]*y[mi] for mi in range(M)]) <= g[k] for k in range(q))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, q, N, M):
    np.random.seed(id)
    A_chol = 0.1 * np.random.rand(N, N)
    A = np.matmul(A_chol, A_chol.transpose())
    b = 10 + 90 * np.random.rand(N)
    C_chol = 0.1 * np.random.rand(M, M)
    C = np.matmul(C_chol, C_chol.transpose())
    d = 10 + 90 * np.random.rand(M)
    g =  1 + 9 * np.random.rand(q)
    EF_base = 2 * (np.random.rand(q, N + M) - 0.5)
    EF_row_norm = np.sum(np.sqrt(EF_base ** 2), 1)
    scaling_factor = g / EF_row_norm
    EF = np.multiply(EF_base.transpose(), scaling_factor).transpose()
    E = EF[0:q, 0:N]
    F = EF[0:q, N:(N+M+1)]
    return(A, b, C, d, E, F, g)

solve_qp(id=id)