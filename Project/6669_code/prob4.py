import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import text_generator, judge_integer, solution_count


import gurobipy as gp
from gurobipy import *

id = 903515184


def model_1(id, n=2, x_type="linear", quiet=True):
    gm = gp.Model("p4")
    if quiet:
        gm.setParam('OutputFlag', False)
    w = get_data(id=id, n=n)

    if x_type == "integer":
        x = gm.addVars(2 * n, 2 * n, lb=0, ub=1, vtype=GRB.INTEGER)
    elif x_type == "linear":
        x = gm.addVars(2 * n, 2 * n, lb=0, ub=1, vtype=GRB.CONTINUOUS)
    else:
        print("x_type should be integer or linear")
        raise NotImplementedError

    # for i in range(2*n):
    #    for j in range(2*n):
    #        if i >= j:
    #            gm.addConstr(x[i,j] == 0)

    gm.setObjective(sum([w[i, j] * x[i, j] for i in range(2 * n) for j in range(2 * n) if i < j]), GRB.MAXIMIZE)

    for i in range(2 * n):
        gm.addConstr(sum([x[h, i] for h in range(i)]) + sum([x[i, j] for j in range(i + 1, 2 * n)]) <= 1)


    #gm.setParam("NonConvex", 2)

    gm.update()
    gm.optimize()
    return (gm)

def model_1_with_real_data(file = "china.txt", x_type="linear", quiet=True):
    gm = gp.Model("p4")
    if quiet:
        gm.setParam('OutputFlag', False)
    w, _ = get_real_data(file = file)
    n = round(w.shape[0] / 2)
    print("n is {}".format(n))

    if x_type == "integer":
        x = gm.addVars(2 * n, 2 * n, lb=0, ub=1, vtype=GRB.INTEGER)
    elif x_type == "linear":
        x = gm.addVars(2 * n, 2 * n, lb=0, ub=1, vtype=GRB.CONTINUOUS)
    else:
        print("x_type should be integer or linear")
        raise NotImplementedError

    # for i in range(2*n):
    #    for j in range(2*n):
    #        if i >= j:
    #            gm.addConstr(x[i,j] == 0)

    gm.setObjective(sum([w[i, j] * x[i, j] for i in range(2 * n) for j in range(2 * n) if i < j]), GRB.MAXIMIZE)

    for i in range(2 * n):
        gm.addConstr(sum([x[h, i] for h in range(i)]) + sum([x[i, j] for j in range(i + 1, 2 * n)]) <= 1)
        # print(list(range(i)))
        # print(list(range(i+1,2*n)))
        # print("--")

    gm.update()
    gm.optimize()
    return (gm)


def model_2(id, n=2, x_type="linear"):
    gm = gp.Model("p4")
    #gm.setParam('OutputFlag', False)
    w = get_data(id=id, n=n)

    if x_type == "integer":
        x = gm.addVars(2 * n, 2 * n, lb=0, ub=1, vtype=GRB.INTEGER)
    elif x_type == "linear":
        x = gm.addVars(2 * n, 2 * n, lb=0, ub=1, vtype=GRB.CONTINUOUS)
    else:
        print("x_type should be integer or linear")
        raise NotImplementedError

    gm.setObjective(sum([w[i, j] * x[i, j] for i in range(2 * n) for j in range(2 * n) if i < j]), GRB.MAXIMIZE)

    for i in range(2 * n):
        gm.addConstr(sum([x[h, i] for h in range(i)]) + sum([x[i, j] for j in range(i + 1, 2 * n)]) <= 1)

    for i in range(2*n):
        for j in range(2*n):
            for h in range(2*n):
                if i < j and j < h:
                    gm.addConstr(x[i,j] + x[j,h] + x[i,h] <= 1)

    gm.setParam("PreDual", 0)
    gm.setParam("Presolve", 0)
    #gm.setParam("LazyConstraints", 1)

    gm.setParam("MIPFocus", 3)
    gm.update()
    gm.optimize()
    return (gm)

def model_2_with_real_data(file = "china.txt", x_type="linear"):
    gm = gp.Model("p4")
    #gm.setParam('OutputFlag', False)
    w, _ = get_real_data(file=file)
    n = round(w.shape[0] / 2)
    print("n is {}".format(n))

    if x_type == "integer":
        x = gm.addVars(2 * n, 2 * n, lb=0, ub=1, vtype=GRB.INTEGER)
    elif x_type == "linear":
        x = gm.addVars(2 * n, 2 * n, lb=0, ub=1, vtype=GRB.CONTINUOUS)
    else:
        print("x_type should be integer or linear")
        raise NotImplementedError

    # for i in range(2*n):
    #    for j in range(2*n):
    #        if i >= j:
    #            gm.addConstr(x[i,j] == 0)

    gm.setObjective(sum([w[i, j] * x[i, j] for i in range(2 * n) for j in range(2 * n) if i < j]), GRB.MAXIMIZE)

    for i in range(2 * n):
        gm.addConstr(sum([x[h, i] for h in range(i)]) + sum([x[i, j] for j in range(i + 1, 2 * n)]) <= 1)
        # print(list(range(i)))
        # print(list(range(i+1,2*n)))
        # print("--")

    for i in range(2*n):
        for j in range(2*n):
            for h in range(2*n):
                if i < j and j < h:
                    gm.addConstr(x[i,j] + x[j,h] + x[i,h] <= 1)

    gm.update()
    gm.optimize()
    return (gm)

def model_3(id, n=2, x_type="linear", quiet=True):
    gm = gp.Model("p4")
    if quiet:
        gm.setParam('OutputFlag', False)
    w = get_data(id=id, n=n)

    if x_type == "integer":
        x = gm.addVars(2 * n, 2 * n, lb=0, ub=1, vtype=GRB.INTEGER)
    elif x_type == "linear":
        x = gm.addVars(2 * n, 2 * n, lb=0, ub=1, vtype=GRB.CONTINUOUS)
    else:
        print("x_type should be integer or linear")
        raise NotImplementedError

    '''
    if x_type == "linear":

        for i in range(2*n):
            for j in range(2*n):
                if i < j:
                    if np.random.random_sample() < 0.5:
                        x[i,j].vtype = GRB.INTEGER                 
    '''

    # for i in range(2*n):
    #    for j in range(2*n):
    #        if i >= j:
    #            gm.addConstr(x[i,j] == 0)

    gm.setObjective(sum([w[i, j] * x[i, j] for i in range(2 * n) for j in range(2 * n) if i < j]), GRB.MAXIMIZE)

    for i in range(2 * n):
        gm.addConstr(sum([x[h, i] for h in range(i)]) + sum([x[i, j] for j in range(i + 1, 2 * n)]) <= 1)
        #gm.addConstr(sum([x[h, i] for h in range(i)]) + sum([x[i, j] for j in range(i + 1, 2 * n)]) >= 1)

    for i in range(2*n):
        for j in range(2*n):
            for h in range(2*n):
                if i < j and j < h:
                    gm.addConstr(x[i,j] + x[j,h] + x[i,h] <= 1)

                for k in range(2*n):
                    for t in range(2*n):
                        if i<j and j<h and h<k and k<t:
                            gm.addConstr(x[i, j] + x[j, h] + x[h, k] + x[k, t] + x[i,t] <= 2)

    #gm.addConstr(sum([x[i,j] for i in range(2*n) for j in range(2*n) if i<j]) >= n)

    gm.update()
    gm.optimize()
    return (gm)



def get_data(id, n):
    np.random.seed(id)
    w = np.random.rand(2 * n, 2 * n)

    # for unweighted
    for i in range(2 * n):
        for j in range(2 * n):
            if i >= j:
                w[i, j] = 0

    # print(w)
    # print("data w shape:", w.shape)
    return w

def get_real_data(file = "china.txt"):

    data_original = []
    with open(file, encoding="utf-8") as datafile:
        lines = datafile.readlines()
    for line in lines:
        line = line.strip().split(" ")
        if len(line) >= 2:
            if file == "china.txt":
                data_original.append(float(line[2]))
            elif file == "china_detailed.txt":
                data_original.append(float(line[1]))

    data = np.zeros((len(data_original), len(data_original)))
    for x in range(len(data_original)):
        for y in range(len(data_original)):
            if x>y:
                data[y,x] = np.abs(data_original[x] - data_original[y])

    return data, data_original

def dataset_characterise(type = "province", number = 0):

    if type == "province":
        unique, indices = np.unique(get_real_data(file = "china.txt")[number], return_counts=True)

    elif type == "city":
        unique, indices = np.unique(get_real_data(file="china_detailed.txt")[number], return_counts=True)

    elif type == "test":
        unique, indices = np.unique(get_data(id=id, n=206), return_counts=True)

    if number == 0:
        unique = unique[1:]
        indices = indices[1:]

    print(unique)
    print(indices)

    unique = [np.log10(x) for x in unique]

    plt.scatter(unique, indices, c='r', marker='x')
    plt.title("{}-level dataset".format(type), fontdict={'fontsize': 'x-large'})
    plt.xlabel("log10(x)")
    plt.ylabel("frequency")
    #plt.show()
    plt.savefig("{}-level_dataset_{}.png".format(type,number))


class solver(object):
    def __init__(self, trials_size=100, x_type="linear", model_type = "1"):
        self.trials_size = trials_size
        self.x_type = x_type
        self.model_type = model_type
        self.visulizer_storage = {"solution": [],
                                  "data_w": [],
                                  "id": []}

    def single_model(self, id, n, quiet=True):
        start_time = time.time()

        if self.model_type == "1":
            model = model_1(id=id, n=n, x_type=self.x_type, quiet=quiet)
        elif self.model_type == "2":
            model = model_2(id=id, n=n, x_type=self.x_type)
        elif self.model_type == "3":
            model = model_3(id=id, n=n, x_type=self.x_type, quiet=quiet)
        else:
            raise NotImplementedError

        solution = model.getAttr("x", model.getVars())
        obj = model.getAttr('ObjVal')

        end_time = time.time()
        return solution, obj, end_time-start_time

    def single_probability_time(self, size_2n=100, quiet=True, sln_counter = None):
        # this is just for one 2n size but large trial size

        int_result_cnt = 0
        l_time_total = 0
        non_integer_sln_list = []

        n = int(size_2n / 2)
        for trials_id in (range(self.trials_size)):

            l_solution, l_obj, l_time = self.single_model(id = id + trials_id, n=n)
            l_time_total = l_time_total + l_time

            if sln_counter is not None:
                sln_counter.check(l_solution)

            if judge_integer(l_solution):
                int_result_cnt = int_result_cnt + 1
                #if not quiet: print("integer sol", "obj difference", linear_obj - integer_obj)
            else:
                self.visulizer_storage["data_w"].append(get_data(id = id + trials_id, n=n))
                self.visulizer_storage["solution"].append(np.array(l_solution).reshape((2*n,2*n)))
                self.visulizer_storage["id"].append(trials_id)

        return int_result_cnt / self.trials_size, l_time_total

    def a_good_linear_example(self, trials_id = 1, size_2n=100):

        n = int(size_2n / 2)
        int_result_cnt = 0

        l_solution, l_obj, l_time = self.single_model(id=id + trials_id, n=n)

        while int_result_cnt == 0:
            if judge_integer(l_solution):
                int_result_cnt = int_result_cnt + 1
                self.visulizer_storage["data_w"].append(get_data(id=id + trials_id, n=n))
                self.visulizer_storage["solution"].append(np.array(l_solution).reshape((2 * n, 2 * n)))
                self.visulizer_storage["id"].append(trials_id)

    def cycle(self, size_list=[100, 200, 400, 800, 1600], count_solution = False):
        # for visualization, set size_list to just a list with one number
        for size_2n in size_list:

            if count_solution:
                sln_counter = solution_count()
            else:
                sln_counter = None

            prob, time = self.single_probability_time(size_2n=size_2n, sln_counter = sln_counter)
            print(prob, time)

            if count_solution:
                print("unique solution number for 2n={} is {}. Time spent {}s".format(size_2n, sln_counter.solution_number, time))


    def result_visulizer(self, size_list=[4], mode = "program"):
        assert len(size_list) == 1
        print(self.visulizer_storage["id"])

        for size_2n in size_list:
            n = int(size_2n / 2)

        if mode == "freq":
            try:
                solution_for_count = np.stack(self.visulizer_storage["solution"])
                print(solution_for_count.shape)
                unique, counts = np.unique(solution_for_count, return_counts=True)
                print(dict(zip(unique, counts)))
            except:
                ...

        else:
            for trial in range(len(self.visulizer_storage["id"])):
                if mode == "program":
                    l_solution, l_obj, l_time = self.single_model(id=id + self.visulizer_storage["id"][trial], n=n, quiet=False)
                elif mode == "draw":

                    fig = plt.figure(figsize=(n*9.25/2, n*3/2))

                    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                                     nrows_ncols=(1,3),
                                     axes_pad=0.15,
                                     share_all=True,
                                     cbar_location="right",
                                     cbar_mode="single",
                                     cbar_size="7%",
                                     cbar_pad=0.15)

                    # data result
                    data_m = self.visulizer_storage["data_w"][trial]
                    grid[0].imshow(data_m, vmin=0.0, vmax=1.0)
                    text_generator(data_m, grid[0])

                    # linear result
                    solution_m = self.visulizer_storage["solution"][trial]
                    grid[1].imshow(solution_m, vmin=0.0, vmax=1.0)
                    text_generator(solution_m, grid[1])

                    # integer result
                    integer_model = model_1(id=id+ self.visulizer_storage["id"][trial], n=n, x_type="integer")
                    i_solution = integer_model.getAttr("x", integer_model.getVars())
                    i_solution_m = np.array(i_solution).reshape((2*n,2*n))

                    im = grid[2].imshow(i_solution_m, vmin=0.0, vmax=1.0)
                    text_generator(i_solution_m, grid[2])

                    # general setup
                    for grid_i in grid:
                        grid_i.set_xticks(np.arange(data_m.shape[0]))
                        grid_i.set_yticks(np.arange(data_m.shape[0]))

                    grid[0].set_title('Data', fontdict = {'fontsize': 'x-large'})
                    grid[1].set_title('Linear', fontdict = {'fontsize': 'x-large'})
                    grid[2].set_title('Integer', fontdict = {'fontsize': 'x-large'})

                    grid[2].cax.colorbar(im)
                    grid[2].cax.toggle_label(True)
                    #fig.colorbar(im)

                    plt.show()
                    #plt.savefig("id{}_size{}_m{}case.png".format(trial,size_2n,self.model_type))


def solvers():
    # draw linear fail solver
    solver_k = solver(trials_size = 100, x_type="linear", model_type = "1")
    solver_k.cycle(size_list=[6])
    solver_k.result_visulizer(size_list=[6], mode = "draw")

    # check unique solution solver
    Trial_size = 10000
    solver_integer = solver(trials_size = Trial_size, x_type="integer", model_type = "1")
    print("Trial_size is {}".format(Trial_size))
    solver_integer.cycle(size_list=[2,4,6,8,10,12], count_solution = True)

    # draw linear success solver
    solver_k = solver(trials_size = 100, x_type="linear", model_type = "1")
    solver_k.a_good_linear_example(size_2n=4)
    solver_k.result_visulizer(size_list=[4], mode = "draw")

    # check solution freq
    solver_k = solver(trials_size = 100, x_type="linear", model_type = "1")
    solver_k.cycle(size_list=[50])
    solver_k.result_visulizer(size_list=[50], mode = "freq")

    # model 3 test and compare
    number_single = 10
    trials_size = 400
    solver_k = solver(trials_size=trials_size, x_type="linear", model_type="2")
    solver_k.cycle(size_list=[number_single])
    solver_k.result_visulizer(size_list=[number_single], mode="draw")
    solver_k = solver(trials_size=trials_size, x_type="linear", model_type="3")
    solver_k.cycle(size_list=[number_single])
    solver_k.result_visulizer(size_list=[number_single], mode="draw")

    # computation loop
    for number in [20,50,80,100,120,150,180,200,220,250,300,350,400]:
        print("--- number {} ---".format(number))
        solver_k = solver(trials_size = 100, x_type="linear", model_type = "3")
        solver_k.cycle(size_list=[number])
        solver_k.result_visulizer(size_list=[number], mode = "freq")

    # probability challenge
    for number_single in [12, 20]:  # this is 2n
        trials_size = 10000
        # solver_k = solver(trials_size=trials_size, x_type="integer", model_type="2")
        # solver_k.cycle(size_list=[number_single])
        # solver_k.result_visulizer(size_list=[number_single], mode="freq")
        print("----")
        solver_k = solver(trials_size=trials_size, x_type="linear", model_type="3")
        solver_k.cycle(size_list=[number_single])
        solver_k.result_visulizer(size_list=[number_single], mode="freq")



#model = model_1_with_real_data(file = "china_detailed.txt", x_type="linear", quiet=False)
#solution = model.getAttr("x", model.getVars())
#print(judge_integer(solution))

#model_2(id = id, n = 206, x_type="linear")

#model_2_with_real_data(file = "china_detailed.txt", x_type="integer")
#model_1_with_real_data(file = "china_detailed.txt", x_type="integer", quiet=False)

model_2(id = id, n = 50, x_type="integer")

# Explored 0 nodes (5259 simplex iterations) in 2.70 seconds
# Explored 0 nodes (851 simplex iterations) in 1.98 seconds, gm.setParam("PreDual", 0) yes below
# Explored 0 nodes (5259 simplex iterations) in 3.18 seconds gm.setParam("PreDual", 1)
# Explored 0 nodes (5259 simplex iterations) in 2.58 seconds gm.setParam("PreDual", 2)

# Explored 0 nodes (740 simplex iterations) in 0.73 seconds gm.setParam("Presolve", 0) yes below
# Explored 0 nodes (851 simplex iterations) in 1.61 seconds gm.setParam("Presolve", 1)
# Explored 0 nodes (766 simplex iterations) in 6.37 seconds gm.setParam("Presolve", 2)
