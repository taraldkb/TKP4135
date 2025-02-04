#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 20:50:02 2024

@author: taraldbishop
"""

import autograd.numpy as np 
from autograd import jacobian






def f_test4(x):
    return np.array([3*x[0], x[0]**2 * np.sin((x[1]-x[2])**2), (x[0]-x[1])*(x[1]-x[2])], 
                    dtype=np.float64)


jac_test4 = jacobian(f_test4)

x= np.array([2.0,3.0,5.0])
jac=jac_test4(x)
print(jac)