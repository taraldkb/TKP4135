# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 16:29:40 2022

@author: P. Maxwell
"""

import autograd.numpy as np
from autograd import jacobian
from autograd import grad
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spO



def f_test1(x):
    return 11*x[0]**4 + 7*x[1]**3 + np.sin(x[2] + 5*x[3]**2)


def gradf_test1(x):
    return np.array([ 44*x[0]**3, 21*x[1]**2, np.cos(x[2] + 5*x[3]**2), 10*x[3]*np.cos(x[2]+5*x[3]**2) ], dtype=np.float64)


# Create func
gradad_test1 = grad(f_test1)

# Create a random sample point as 1d array of length 4
x = np.random.random_sample((4,))

# Test the autodiff compared to symoblic
df_sym_test1 = gradf_test1(x)
df_ad_test1 = gradad_test1(x)

print("Symbolic evaluation of df_test: ", df_sym_test1)
print("Autodiff evaluation of df_test: ", df_ad_test1)
print("Norm of difference: ", np.linalg.norm(df_sym_test1 - df_ad_test1))




