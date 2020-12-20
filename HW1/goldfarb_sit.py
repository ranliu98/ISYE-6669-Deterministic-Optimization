# Implement the Goldfarb-Sit problem
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import *

num_pivots = []

def solve_lp(n=10, epsilon=0.01, beta=2):

    delta = beta * (2 + epsilon)
    gm = gp.Model("Goldfarb-Sit")

    xs = gm.addVars(n, lb=-GRB.INFINITY)
    gm.setObjective(sum([xs[i] for i in range(n)]), GRB.MAXIMIZE)

    gm.addConstr(xs[0] >= 0)
    gm.addConstr(xs[0] <= 1)
    gm.addConstrs(xs[i] >= beta* xs[i-1] for i in range(1,n))
    gm.addConstrs(xs[i] <= delta ** i -beta*xs[i-1] for i in range(1,n))

    def dual():
        ys = gm.addVars(2*n+1, lb=0.0)
        gm.setObjective(sum([(delta**i)*ys[2*i] for i in range(n)]), GRB.MINIMIZE)

        if n == 1:
            gm.addConstr(ys[0]>=1)
        else:
            A = np.zeros((2*n-1,n))
            A[0,0] = 1
            for numb in range(1,n):
                A[2*numb-1,numb-1] = beta
                A[2 * numb - 1, numb] = -1
                A[2*numb, numb - 1] = beta
                A[2 * numb, numb] = 1
            #print(A)
            gm.addConstrs(sum([A[row,col]*ys[row] for row in range(2*n-1)]) >=1 for col in range(n))

    # Solve the model using:
    #   (i) primal Simplex
    #   (ii) no presolver
    #   (iii) the Steepest Edge pricing policy
    gm.setParam("method", 0)
    gm.setParam("Presolve", 0)
    gm.setParam("SimplexPricing", 1)
    #gm.setParam("LogFile", "Sifting Logging")
    gm.update()
    gm.optimize()
    gm.write('goldfarb-sit.lp')
    return(gm)


def loop_prob():
    for n in range(1,11):
        num_pivots.append(solve_lp(n=n).getAttr('IterCount'))

    plt.plot(range(1, 11), num_pivots)
    plt.xlabel('problem dimension')
    plt.ylabel('the number of Simplex pivots')
    plt.show()

loop_prob()