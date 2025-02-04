#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: taraldbishop
"""

import autograd.numpy as np 
from autograd import jacobian 
import scipy.optimize as sp
import numpy.linalg as lg
import matplotlib.pyplot as plt

def f(x,B):
    B0,B1 =B
    return B0 + B1*x


# b
T = np.array([270.4, 270.6, 272.3, 273.6, 274.1, 275.5, 276.6, 277.1]) #K
exP= np.array([1.502, 1.556, 1.776, 2.096, 2.281, 2.721, 3.001, 3.556]) # mPa
y = np.log(exP)

# c
A= np.column_stack((np.ones_like(T),T))
AT = np.transpose(A)
ATA = np.dot(AT,A)
ATy = np.dot(AT,y)

Beta = lg.solve(ATA, ATy)

#d
x= np.linspace(np.min(T),np.max(T),100)

sol_y_pts=f(x,Beta)
y_calc_trans = np.exp(sol_y_pts)

plt.figure(1)
plt.plot(x,sol_y_pts, label="model")
plt.plot(T, y, "o" ,label = "experimental")
plt.legend()
plt.xlabel("T [C]")
plt.ylabel("ln(P)")
plt.grid()

plt.figure(2)
plt.plot(x,y_calc_trans, label="model")
plt.plot(T, exP, "o" ,label = "experimental")
plt.legend()
plt.xlabel("T [C]")
plt.ylabel("P [mPA]")
plt.grid()

res= y-np.dot(A,Beta)
norm= lg.norm(res, ord=2)

print(f'2-norm of model 1: {norm}')




 