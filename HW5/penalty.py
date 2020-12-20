# Implement the Quadratic Penalty problem
import gurobipy as gp
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt

id = 903515184

# Invoke Gurobi to solve the QP
def solve_qp(id, rho, p=300, q=250, n=200):

    gm = gp.Model("penalty")
    A, E, b, c, d = get_data(id, p, q, n)
    # print(A.shape, E.shape, b.shape, c.shape, d.shape) # (300, 200) (250, 200) (300,) (200,) (250,)

    x = gm.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    gm.setObjective(sum([c[ni]*x[ni] for ni in range(n)]) +
                    0.5*rho*sum([(sum([A[pi,ni]*x[ni] for ni in range(n)]) - b[pi]) * (sum([A[pi,ni]*x[ni] for ni in range(n)]) - b[pi]) for pi in range(p)])
                    , GRB.MINIMIZE)

    gm.addConstrs(sum([E[qi,ni]*x[ni] for ni in range(n)]) >= d[qi] for qi in range(q))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)


def solve_lp(id, rho, p=300, q=250, n=200):

    gm = gp.Model("penaltylp")
    A, E, b, c, d = get_data(id, p, q, n)
    # print(A.shape, E.shape, b.shape, c.shape, d.shape) # (300, 200) (250, 200) (300,) (200,) (250,)

    x = gm.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    gm.setObjective(sum([c[ni]*x[ni] for ni in range(n)]), GRB.MINIMIZE)

    gm.addConstrs(sum([A[pi,ni]*x[ni] for ni in range(n)]) == b[pi] for pi in range(p))
    gm.addConstrs(sum([E[qi,ni]*x[ni] for ni in range(n)]) >= d[qi] for qi in range(q))

    # Solve the model
    gm.update()
    gm.optimize()
    #print(gm.getVars())
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, p, q, n):
    np.random.seed(id)
    A = np.random.rand(p, n) / np.sqrt(p * n)
    x0 = np.random.rand(n)
    b = A @ x0
    c = np.random.rand(n) / np.sqrt(n)
    d = -(1 + 9 * np.random.rand(q))
    E_base = 2 * (np.random.rand(q, n) - 0.5)
    E_row_norm = np.sum(np.sqrt(E_base ** 2), 1)
    scaling_factor = d / E_row_norm
    E = np.multiply(E_base.transpose(), scaling_factor).transpose()
    return(A, E, b, c, d)


def plot(id, p=300, q=250, n=200):
    A, _, b, _, _ = get_data(id, p, q, n) # (300, 200)
    norm = []
    datanorm = []
    log2 = []

    for rho_n in range(16):
        qp_model = solve_qp(id=id, rho=2 ** rho_n)
        lp_model = solve_lp(id=id, rho=2 ** rho_n)
        z = lp_model.getAttr("x", lp_model.getVars())
        z_rho = qp_model.getAttr("x", lp_model.getVars())
        #print(len(z), len(z_rho))

        norm.append((sum([(z[i]-z_rho[i])**2 for i in range(n)]))**0.5)
        datanorm.append(sum([(sum([A[pi, ni] * z_rho[ni] for ni in range(n)]) - b[pi])**2 for pi in range(p)])**0.5)
        log2.append(np.log2(rho_n))

    plt.plot(log2, norm)
    plt.xlabel('the values of log2(ρ)')
    plt.ylabel('the norm of (z - z_rho)')
    plt.show()

    plt.plot(log2, datanorm)
    plt.xlabel('the values of log2(ρ)')
    plt.ylabel('the norm of (A * z_rho - b)')
    plt.show()


solve_qp(id=id, rho=2**10)
solve_lp(id=id, rho=2**10)

#plot(id=id)
