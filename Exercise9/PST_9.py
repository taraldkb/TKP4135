#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:59:24 2024

@author: taraldbishop
"""

import autograd.numpy as np
from autograd import jacobian, hessian
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy import linalg as lg




def f(X):
    x= X[0]
    y=X[1]
    return (x**2 +y-11)**2 + (x+y**2 -7)**2


# task a
x0 = np.array([-0.20,-1.1])

jac= jacobian(f)
hes = hessian(f)

tol =1e-10
maxit = 70

i= 1
x=x0





while i< maxit:
    J = jac(x)
    H = hes(x)
    xk= x - np.matmul(lg.inv(H),J)
    i+=1
    
    
    if (abs(xk[0]-x[0])<tol) and (abs(xk[1]-x[1])<tol) :
        x=xk
        break
    x=xk


eig,vec=lg.eig(hes(x))
print('task a')
print(f'converged to x = {x[0]} and y = {x[1]}')
print(f'eigenvalues of hessian is: {eig}')
print("this might be a local maximum (negative eigenvalues)\n\n")

#task b
x0 = np.array([4.0,-2.0])

jac= jacobian(f)
hes = hessian(f)

tol =1e-10
maxit = 70

i= 1
x=x0





while i< maxit:
    J = jac(x)
    H = hes(x)
    xk= x - np.matmul(lg.inv(H),J)
    i+=1
    
    
    if (abs(xk[0]-x[0])<tol) and (abs(xk[1]-x[1])<tol) :
        x=xk
        break
    x=xk


eig,vec=lg.eig(hes(x))
print(f'converged to x = {x[0]} and y = {x[1]}')
print('task b')
print(f'eigenvalues of hessian is: {eig}')
print("this might be a local minimum(positiv eigenvalues)")




        

    