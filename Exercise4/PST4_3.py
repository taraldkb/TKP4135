#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 23:02:32 2024

@author: taraldbishop
"""

import autograd.numpy as np 
from autograd import jacobian 
import scipy.optimize as sp

# define variables

p = 5 #bar
T = 390 #kelvin
A = np.array([3.97786, 4.00139, 3.93002]) # [pentane, hexane, cyclohexane]
B = np.array([1064.840, 1170.875, 1182.774])
C = np.array([-41.136, -48.833, -52.532])
z = np.array([0.5, 0.3, 0.2])
F = 100 #mol/s

#define initial guess

x0= np.array([0.3, 0.3, 0.4, 0.5, 0.3 , 0.2, 60.0, 40.0 ]) #[xp, xh, xc, yp, yh, yc, L, V]
#calculations
psat = np.float64(10)**(A-B/T+C)  #bar
K= psat/p

# defining function
def f_res(inp):
    x = inp[0:3]
    y = inp[3:6]
    V = inp[7]
    L = inp[6]
    
    
    res_MB = F*z-V*y-L*x
    res_EQ = y-K*x
    res_sum = np.array([1-sum(x), 1-sum(y)])
    
    return np.concatenate((res_MB, res_EQ, res_sum))


jac_res= jacobian(f_res)

sol_sp = sp.root(f_res, x0, jac=jac_res, method= 'lm')
sol_hybrid = sp.root(f_res, x0, jac=jac_res, method= 'hybr')

print(sol_sp.x)
print(sol_hybrid.x)




    
    

