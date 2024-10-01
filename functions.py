from sympy import *
import numpy as np

def system_of_eq(y, p, fun = 'exp', init = (1,1, 0.00001), min_value = -52830496.02941906, max_value = 69121610.5847813 ):

    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')

    if fun == 'exp':
        f1 = x1 -x2*exp(-x3*max_value) -1
        f2 = x1 -x2*exp(-x3*min_value) - 0 
        f3 = x1 -x2*exp(-x3* y ) - p

    # print(nsolve((f1, f2, f3), (x1, x2, x3), init))
    params = nsolve((f1, f2, f3), (x1, x2, x3), init).values()

    #str to float 
    params = [float(i) for i in params]

    return params



def tanh_fun(x, param):
    return (np.tanh(param*x))