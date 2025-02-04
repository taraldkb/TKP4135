#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:15:21 2024

@author: taraldbishop
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# def functions

def SIR(t, x, N,R0, gamma, beta ):
    S=x[0]
    I = x[1]
    R = x[2]
    return np.array([(-beta*I*S)/N,
                     ((beta*I*S)/N)- gamma*I,
                     gamma*I], dtype=np.float64 )

def do_solve(f, tspan, x0, param_args):
    sol = solve_ivp(f, tspan, x0,
                            args=param_args,
                            method='RK45', dense_output=True)
                            
    return sol

# def params and intitial Conditions

N = 1000000.0
x0 = np.array([N-10.0, 10.0, 0.0])

r0= 4.0
gamma = 1/5
beta = r0*gamma
tspan = [0,100]

param_args = (N, r0, gamma, beta)

t_pts = list(range(tspan[0], tspan[1]+1))



sol_SIR = do_solve(SIR, tspan, x0, param_args)

S_output = sol_SIR.sol(t_pts)[0,:]
I_output = sol_SIR.sol(t_pts)[1,:]
R_output = sol_SIR.sol(t_pts)[2,:]

fig=plt.figure(figsize = (15,10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(t_pts,S_output, label = "Susceptible")
ax.plot(t_pts,I_output, label = "Infected")
ax.plot(t_pts,R_output, label = "Recovered")
ax.legend()
ax.grid()
ax.set(xlabel="t [Days]", ylabel= "Individuals (Millions)", 
       title = "SIR model output")
