#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:23:47 2024

@author: taraldbishop
"""

import autograd.numpy as np 
import matplotlib.pyplot as plt
from autograd import jacobian 
from scipy import optimize
from scipy.integrate import solve_ivp

# t_points for the collocation method using 5 Radau points
five_rad = np.array([0.057104, 0.276843, 0.583590,0.860240,1.])

# 5 uniform points
five_uni = np.array([0.2, 0.4, 0.6, 0.8, 1.])

# t_points for the collocation method using 3 Radau points
three_rad = np.array([0.155051, 0.644949, 1.])

# t_points for the collocation method using 3 uniform points
three_uni = np.array([1/3., 2/3, 1.])

# define an easy and hard test ODE
def dxdt_easy(t,x): 
    return x**2-2*x+1
def dxdt_hard(t, x):
    return -10*x + 10*np.cos(t)

# initial condition for both problems
x0= np.array([-3])

#define self made function
def collocation_weights(t_points):
    A = [] #matrix to be inversed
    B= []
    for i in range(len(t_points)):
        for j in range(len(t_points)):
            a= []
            b= []
            a.append(t_points[i]**(j+1))
            b.append((j+1)*t_points**(j))
            
        A.append(a)
        B.append(b)
        
    A = np.array(A)
    B = np.array(B)
    
            
    return np.matmul(B,np.linalg.inv(A))


M = collocation_weights(three_rad)
print(M)