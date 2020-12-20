# Implement Newton's method
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

id = 903515184

# Invoke Newton's method to solve the system of equations
def newton(id, epsilon_1 = 1e-10, secant=False):

    profit = 0.0
    c, L, U, alpha, beta, a, b = get_data(id)

    def function_(p):
        f = (p/2)*(a+b) - ((p**2)/2)*(alpha + beta) + (c**2/(2*p))*(b-a-beta*p + alpha*p) - c*(b-beta*p)
        return f

    def function_d(p):
        p = p + 1e-7
        f = 0.5*(a+b) - p * (alpha+beta) + c * beta - (c**2/(2*p**2))*(b-a)
        return f

    def q2_function(p, alpha, beta, c, a, b):
        f = 2 * (p ** 3) * (beta + alpha) - 2 * beta * c * (p ** 2) - a * (p ** 2 + c ** 2) - b * (p ** 2 - c ** 2)
        return f

    def q2_function_d(p, alpha, beta, c, a, b):
        f = 6 * (p ** 2) * (beta + alpha) - 4 * beta * c * p - 2 * a * p - 2 * b * p
        return f

    #print("check", function_(5), f_p(a - alpha * 5, b - beta * 5, c, 5))

    def function_dd(p):
        p = p + 1e-7
        f = -(alpha+beta) + (c**2/p**3) * (b-a)
        return f

    y_k = U
    epsilon_2 = 1e-5
    f_y_store = []
    question2_k = U

    for k in range(100):
        d_k = function_dd(y_k)
        #print("d_k", d_k)
        if np.abs(d_k) < epsilon_2:
            print("Derivative failure")
            raise AssertionError
        y_k = y_k - function_d(y_k) / d_k

        question2_k = question2_k - q2_function(question2_k, alpha, beta, c, a, b) / q2_function_d(question2_k, alpha,beta, c, a, b)
        #print("question2_k", question2_k)

        f_y_store.append(function_d(y_k))
        #print(f_y_store)
        if k >= 2:
            if np.abs(f_y_store[k]) >= np.abs(f_y_store[k-1]) and np.abs(f_y_store[k]) >= np.abs(f_y_store[k-2]):
                print("algorithm is not converging")
                raise AssertionError

        if np.abs(f_y_store[k]) <= epsilon_1:
            print("success at {}".format(k))
            break



    print(y_k, function_(y_k))

    if y_k >= U:
        y_k = U
    elif y_k <= L:
        y_k = L

    print("p should be {}".format(y_k))
    profit = function_(y_k)
    # Return the profit
    return(profit)

def check(id):
    c, L, U, alpha, beta, a, b = get_data(id)

    def function_(p):
        f = (p/2)*(a+b) - (p**2/2)*(alpha + beta) + (c**2/(2*p))*(b-a-beta*p + alpha*p) - c*(b-beta*p)
        return f

    def function_d(p):
        p = p + 1e-7
        f = 0.5*(a+b) - p * (alpha+beta) + c * beta - (c**2/(2*p**2))*(b-a)
        return f

    def q2_function_d(p, alpha, beta, c, a, b):
        f = 6 * (p ** 2) * (beta + alpha) - 4 * beta * c * p - 2 * a * p - 2 * b * p
        return f

    def q2_function(p, alpha, beta, c, a, b):
        f = 2 * (p ** 3) * (beta + alpha) - 2 * beta * c * (p ** 2) - a * (p ** 2 + c ** 2) - b * (p ** 2 - c ** 2)
        return f

    x = np.arange(410,420,1).tolist()
    y = [q2_function(xi, alpha, beta, c, a, b) for xi in x]
    #y = [function_(xi) for xi in x]
    plt.plot(x,[0 for xi in x])
    plt.plot(x,y)
    plt.xlabel("number of p")
    plt.ylabel("function value")
    plt.show()


# Generate the data (DO NOT MODIFY)
def get_data(id):
    id_str = str(id)
    if len(id_str) != 9:
        raise IndexError('Input GTID is not 9 digits!')
    c = 10
    L = 10
    U = 60
    alpha = 0.5
    beta = 2.4
    a = 300 + 2 * int(id_str[5]) + int(id_str[6])
    b = 2100 - 3 * int(id_str[7]) - 5 * int(id_str[8])
    return(c, L, U, alpha, beta, a, b)

#check(id=id)
print(newton(id=id))