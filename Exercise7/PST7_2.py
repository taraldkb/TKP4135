#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:23:47 2024

@author: taraldbishop
"""

#import autograd.numpy as np 
import numpy as np
import matplotlib.pyplot as plt
#from autograd import jacobian 
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
        a= []
        b= []
        for j in range(len(t_points)):
           
            a.append(t_points[i]**(j+1))
            b.append((j+1)*t_points[i]**(j))
            
        A.append(a)
        B.append(b)
        
    A = np.array(A)
    B = np.array(B)
    
            
    return np.matmul(B,np.linalg.inv(A))





def ortho_collec(t_points, x0, f):
    Minv =np.linalg.inv(collocation_weights(t_points))
    t0 = 0
    def f_colloc(x_t):
        # make the matrices for the equation 0 = A- M^-1 x B
        A= []
        B= []
        for i in range(len(x_t)):
            A.append(x_t[i]-x0)
            B.append(f(t_points[i],x_t[i]))
            
        A=np.array(A)
        B=np.array(B)  
        return A-np.matmul(Minv,B)
    #jac= jacobian(f)
    
    sol=optimize.root(f_colloc, 0.5*np.ones(len(t_points))*x0, method= 'lm',
                      jac=f, tol= 1e-10, options={"xtol":1e-8})
    t_colloc = [t0]
    t_colloc.extend(t_points)
    t_colloc = np.array(t_colloc)
    
    return sol

    
solv= ortho_collec(three_uni, x0, dxdt_hard)  
        
        
    
    
    
    
    
    
    
    
    
    
    
    
