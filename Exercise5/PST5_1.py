#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:33:33 2024

@author: taraldbishop
"""

#import numpy as np
from numpy import linalg as la
import autograd.numpy as np
from autograd import jacobian



#a)

A=np.array([[-1, -3],[-2, 1]])
eigenval, eigenmat = la.eig(A)

# print(f'eigenvalues= {eigenval}')
# print(f'eigenvectors{eigenmat}')

# b&c) 

xb = np.array([1.0,0.0])
xc = np.array([-1.0,0.0])
def f(x):
    x1 = x[0]
    x2 = x[1]
    return np.array([-0.5*x1**2 + x2**3 , -x2*x1])

jac = jacobian(f)


eigenvalb, eigenmatb = la.eig(jac(xb))
eigenvalc, eigenmatc = la.eig(jac(xc))

print(f'eigenvalues= {eigenvalb}')
print(f'eigenvectors={eigenmatb}')

print(f'eigenvalues c= {eigenvalc}')
print(f'eigenvectorsc={eigenmatc}')
