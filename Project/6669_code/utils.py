import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class solution_count(object):
    def __init__(self):
        self.solution_set = []
        self.solution_number = 0
    def check(self, new_solution):

        if self.solution_number == 0:
            self.solution_set.append(new_solution)
            self.solution_number = self.solution_number + 1
        else:
            flag = 0
            for solution in self.solution_set:
                if np.array_equal(solution, new_solution):
                    flag = flag + 1
            if flag == 0:
                self.solution_set.append(new_solution)
                self.solution_number = self.solution_number + 1


def judge_integer(numberlist):
    # if integer, return True. Else, return False
    residual = 0.0

    for number in numberlist:
        residual = residual + np.abs(number - round(number))
    if residual < 1e-5:
        return True
    else:
        return False


def text_generator(data, ax):
    for xi in range(data.shape[0]):
        for yi in range(data.shape[1]):
            if xi > yi:
                ax.text(xi, yi, "{:.2f}".format(np.abs(data[yi,xi])), size='x-large', color='white', ha='center', va='center')

