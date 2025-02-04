#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:33:00 2024

@author: taraldbishop
"""

import autograd.numpy as np 
from autograd import jacobian 
import scipy.optimize as sp

x0 = 3.0

def f (x):
    return np.arctan(x)

jac_f = jacobian(f)

sol_sp = sp.root(f, x0, jac=jac_f, method= 'lm')

print(sol_sp.x)

