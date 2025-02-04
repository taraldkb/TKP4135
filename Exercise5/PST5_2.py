#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:52:58 2024

@author: taraldbishop
"""

import numpy as np
import matplotlib.pyplot as plt


def forwardeuler(f, h, x0, t0, tf):
    
    steps = int(np.ceil((tf - t0) / h))
    
    vt_k = np.empty(steps + 1)
    vx_k = np.empty((steps + 1, len(x0)))
    
    vt_k[0] = t0
    vx_k[0] = x0
    
    
    for k in range(steps):
        tk = t0 + h*k 
        xk = vx_k[k]
        
        vx_k[k+1] = xk + h * f(tk, xk)
        vt_k[k + 1] = tk + h
        
    vt_k[-1] = tf
    vx_k[-1] = xk + h * f(vt_k[-1], vx_k[-1])
    
    
    return (vt_k, vx_k)

        
def fn_gh(t,x):
    return (1-2*t)*x

def exact(t):
    return np.exp(1/4 - (1/2 - t)**2)


### define stating conditions

h = 0.05
t0 = 0.0
tf = 1.5
x0 = np.array([1.0])

t, x = forwardeuler(fn_gh, h, x0, t0, tf)
plt.plot(t,exact(t), label = "exact")            
plt.plot(t,x, "ro-", label= 'euler')   
plt.legend() 
plt.grid() 
plt.xlabel('x')
plt.ylabel('y')    
                     
    