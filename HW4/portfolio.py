# Implement the Portfolio Construction problem
import gurobipy as gp
import numpy as np
from gurobipy import *

id = 903515184

# Invoke Gurobi to solve the IP
def solve_ip(id, n=2000, p=100, batch_size=100):

    gm = gp.Model("portfolio")
    B, f, g, r, s, q = get_data(id, n, p, batch_size)
    #print(B, f.shape, g.shape, r.shape, s.shape, q.shape) # 200000000 (2000,) (2000,) (2000,) (2000,) (2000,)

    s = s-1

    s_binary = []
    for sector in range(p):
        sector_binary = np.where(s == sector)[0].tolist()
        s_binary.append(sector_binary)
    quality_c = np.where(q == 'C')[0].tolist()


    x = gm.addVars(n, lb=0, vtype=GRB.INTEGER) # unit of 100
    x_de = gm.addVars(n, lb=0, ub=1, vtype=GRB.INTEGER)
    M = B / np.min(100*f)
    s_value = gm.addVars(p, lb=0)
    # s_value_max = gm.addVars(p, lb=0, ub=1, vtype=GRB.INTEGER)

    gm.setObjective(sum([100 * x[i] * r[i] for i in range(n)]), GRB.MAXIMIZE)

    gm.addConstrs(x[i] <= M*x_de[i] for i in range(n))
    gm.addConstr(sum([g[i]*x_de[i] + 100*x[i]*f[i] for i in range(n)]) <= B)

    for sector_id in range(p):
        gm.addConstr(s_value[sector_id] == sum([100*x[i]*f[i] for i in s_binary[sector_id]]))

    #gm.addConstr(sum([s_value_max[i] for i in range(p)]) == 1)
    #gm.addConstr(s_value_max == max_([s_value[j] for j in range(p)]))
    gm.addConstrs(s_value[j] * p <= sum([2 * 100 * f[i] * x[i] for i in range(n)]) for j in range(p))

    gm.addConstr(sum([100 * f[i] * x[i] for i in quality_c]) <= 0.1 * sum([100 * f[i] * x[i] for i in range(n)]))

    # Solve the model
    gm.update()
    gm.optimize()    
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, n, p, batch_size):
    np.random.seed(id)
    B = (n * batch_size * 1000);
    q = np.random.choice(['A', 'B', 'C'], n, p=[0.5, 0.3, 0.2]);
    # Aux arrays
    i_A = (q == 'A');
    i_B = (q == 'B');
    i_C = (q == 'C');
    sector_distr_base = np.random.rand(p);
    sector_distr = sector_distr_base / sector_distr_base.sum();
    # Other arrays
    f = 0.5 +  0.5 * np.random.rand(n) * (i_A + 0.8 * i_B + 0.7 * i_C);
    g = (batch_size / 2) * np.random.rand(n) * (i_A + i_B * 1.25 + i_C * 2.0);
    r = 0.1 * np.random.rand(n) * (i_A + i_B * 1.1 + i_C * 5.0);
    s = np.random.choice(range(1, p + 1), n, p=sector_distr);
    return(B, f, g, r, s, q)

solve_ip(id=id)