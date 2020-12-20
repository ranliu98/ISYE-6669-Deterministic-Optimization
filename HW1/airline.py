# Implement the Airline Ticket Allocation problem
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import *

id = 903515184

def solve_lp(id, n=3, m=100, T=200):

    gm = gp.Model("Airline")
    r, C, p, alpha, pi = get_data(id, n, m, T)
    #print(r.shape, C.shape, p.shape, alpha.shape, pi.shape, r, p)

    #alpha_sub1 = alpha - np.ones(alpha.shape)
    #factor_m = alpha_sub1.transpose()*p + (np.repeat(r.reshape(r.shape[0], 1), m, axis=1)).transpose()
    #factor_m = factor_m.transpose()

    #print(r, C, p, alpha, pi)

    tickets_buy = gm.addVars(n, lb=0.0)
    seats_pen = gm.addVars(n,m, lb=0.0)
    gm.setObjective(sum([sum([tickets_buy[i]*(r[i]-p[i]) + seats_pen[i,k]*p[i] for i in range(n)]) * pi[k] for k in range(m)]), GRB.MAXIMIZE)
    #gm.setObjective(sum([sum([factor_m[i][k] * tickets_buy[i] for i in range(n)]) * pi[k] for k in range(m)]), GRB.MAXIMIZE)
    gm.addConstr(sum([tickets_buy[i] for i in range(n)])<=T)
    gm.addConstrs(seats_pen[i,k]<= tickets_buy[i]*alpha[i][k] for i in range(n) for k in range(m))
    gm.addConstrs(seats_pen[i,k] <= C[i] for i in range(n) for k in range(m))
    gm.addConstrs(seats_pen[i,k]<=tickets_buy[i] for i in range(n) for k in range(m))

    def printSolution():
        if gm.status == GRB.OPTIMAL:
            buyx = gm.getAttr('x', tickets_buy)
            for numb in range(n):
                print('%s %g' % (numb, buyx[numb]))
        else:
            print('No solution')

    # Solve the model
    gm.update()
    gm.optimize()
    gm.write('airline.lp')
    #printSolution()
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, n, m, T):
    np.random.seed(id)
    r = -np.sort(-np.random.randint(100, 300, n))
    C = np.sort(np.random.randint(1, np.floor(3 * T / 4), n))
    p = -np.sort(-np.random.randint(200, 400, n))
    alpha = 0.9 + (np.random.random((n, m)) * 0.1)
    pi_scores = np.random.random(m)
    pi = pi_scores / pi_scores.sum()
    return(r, C, p, alpha, pi)


solve_lp(id=id)
