

from sympy import *

x1 = Symbol('x1')
x2 = Symbol('x2')
x3 = Symbol('x3')

'''f1 = x1 -x2*exp(x3*(-994.9))
f2 = x1 -x2*exp(x3*(32.935)) - 1
f3 = x1 -x2*exp(x3*(-299.7)) - .7'''

f1 = x1 -x2*exp(-x3*(-994.9))
f2 = x1 -x2*exp(-x3*(32.935)) - 1
f3 = x1 -x2*exp(-x3*(-299.7)) - .7

print(nsolve((f1, f2, f3), (x1, x2, x3), (-1, -2, 0.0006)))
