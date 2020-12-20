# Implement the TSP problem with an MTZ formulation
import gurobipy as gp
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt

id = 903515184

# Invoke Gurobi to solve the IP
def solve_ip(id, n=25):

    gm = gp.Model("tsp")
    d = get_data(id, n)
    # print(d.shape)


    x = gm.addVars(n,n, lb=0, ub=1, vtype=GRB.INTEGER)
    u = gm.addVars(n)

    gm.setObjective(sum([d[i,j]*x[i,j] for i in range(n) for j in range(n)]),GRB.MINIMIZE)
    gm.addConstrs(sum([x[i, j] for j in range(n)]) == 1 for i in range(n))
    gm.addConstrs(sum([x[j, i] for j in range(n)]) == 1 for i in range(n))
    gm.addConstrs(x[i,i] == 0 for i in range(n))

    gm.addConstr(u[0] == 0)
    gm.addConstrs(u[i] >= 1 for i in range(1,n))
    gm.addConstrs(u[i] <= (n-1) for i in range(1, n))
    gm.addConstrs(u[i] - u[j] + 1 <= (n-1)*(1-x[i,j]) for i in range(1,n) for j in range(1,n))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

def solve_lp(id, n=25):

    gm = gp.Model("tsp")
    d = get_data(id, n)
    # print(d.shape)


    x = gm.addVars(n,n, lb=0, ub=1)
    u = gm.addVars(n)

    gm.setObjective(sum([d[i,j]*x[i,j] for i in range(n) for j in range(n)]),GRB.MINIMIZE)
    gm.addConstrs(sum([x[i, j] for j in range(n)]) == 1 for i in range(n))
    gm.addConstrs(sum([x[j, i] for j in range(n)]) == 1 for i in range(n))
    gm.addConstrs(x[i,i] == 0 for i in range(n))

    gm.addConstr(u[0] == 0)
    gm.addConstrs(u[i] >= 1 for i in range(1,n))
    gm.addConstrs(u[i] <= (n-1) for i in range(1, n))
    gm.addConstrs(u[i] - u[j] + 1 <= (n-1)*(1-x[i,j]) for i in range(1,n) for j in range(1,n))

    # Solve the model
    gm.update()
    gm.optimize()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, n):
    np.random.seed(id)
    base = 100 * np.random.rand(n, 2)
    base_sqr_norm = np.square(base).sum(1)
    xx_mat = np.outer(base_sqr_norm, np.ones(n))
    xy_mat = np.matmul(base, base.T)
    d = np.sqrt(abs(xx_mat - 2 * xy_mat + xx_mat.T))
    return(d)


def plot():
    num_pivots = []
    for n in range(5, 31):
        num_pivots.append(solve_ip(id=id, n=n).getAttr('IterCount'))

    plt.plot(range(5, 31), num_pivots)
    plt.xlabel('number n')
    plt.ylabel('the number of iterations')
    plt.show()

def plot_2():
    ratio = []
    for n in range(5, 31):
        ratio.append((solve_ip(id=id, n=n).getAttr('ObjVal'))/(solve_lp(id=id, n=n).getAttr('ObjVal')))

    plt.plot(range(5, 31), ratio)
    plt.xlabel('number n')
    plt.ylabel('ObjVal ratio')
    plt.show()

#solve_ip(id=id)

plot_2()
