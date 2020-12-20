# Implement the Call Center problem
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import *

id = 903515184

def solve_lp(id):

    gm = gp.Model("Call_Center")

    c, D = get_data(id, 3, 7)

    #print(c,D)

    D_shifts = {
        'Mon': [1, 1, 1, 1, 1, 0, 0],
        'Tue': [0, 1, 1, 1, 1, 1, 0],
        'Wed': [0, 0, 1, 1, 1, 1, 1],
        'Thu': [1, 0, 0, 1, 1, 1, 1],
        'Fri': [1, 1, 0, 0, 1, 1, 1],
        'Sat': [1, 1, 1, 0, 0, 1, 1],
        'Sun': [1, 1, 1, 1, 0, 0, 1],
    }

    work_start, pay = gp.multidict({
        'Mon': c[0],
        'Tue': c[1],
        'Wed': c[2],
        'Thu': c[2],
        'Fri': c[2],
        'Sat': c[2],
        'Sun': c[1],
    })

    arrange = gm.addVars(work_start, vtype= GRB.INTEGER, name="arrangement")
    gm.setObjective(arrange.prod(pay), GRB.MINIMIZE)
    gm.addConstrs((gp.quicksum(D_shifts[work_start_day][day_num] * arrange[work_start_day] for work_start_day in work_start) >= day_need for day_num, day_need in enumerate(D)))

    #print([gp.quicksum(D_shifts[work_start_day][day_num] for work_start_day in work_start) >= day_need for day_num, day_need in enumerate(D)])
    #print("---")


    # Solve the model
    gm.update()
    gm.optimize()

    if gm.status == GRB.OPTIMAL:
        print(gm.objVal)
        arrangex = gm.getAttr('x', arrange)
        for f in work_start:
            if arrange[f].x > 0.0001:
                print('%s %g' % (f, arrangex[f]))

    gm.write('call_center.lp')
    return(gm)

# Generate the data (DO NOT MODIFY)
def get_data(id, i_max, t_max):
    np.random.seed(id)
    c = np.sort(10 * np.random.random(i_max) + 10)
    D = np.random.randint(10, 30, t_max)
    return(c, D)

solve_lp(id=id)
