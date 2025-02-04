#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:49:22 2024

@author: taraldbishop
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# def functions

def SIR(t, x, N,R0, gamma, beta, alpha, sigma, omega):
    S = x[0]
    E = x[1]
    I = x[2]
    R = x[3]
    D = x[4]
    m = max(0,(alpha*N-D)/N)
    return np.array([((-beta*I*S)/N) + omega*R ,
                     ((beta*I*S)/N) - sigma*E,
                     sigma*E - gamma*I -m*I,
                     gamma*I - omega*R,
                     m*I], dtype=np.float64 )
                     

def do_solve(f, tspan, x0, param_args):
    sol = solve_ivp(f, tspan, x0,
                            args=param_args,
                            method='RK45', dense_output=True)
                            
    return sol

# def params and intitial Conditions

N = 1000000.0
x0 = np.array([N-10.0, 10.0, 0.0, 0.0, 0.0])

r0= 5.0
my = 0.4
maxhealth = 85000
gamma = 1/5
alpha = 0.015
sigma = 1/5
omega = 1/180
beta = r0*(gamma+alpha)
tspan = [0,400]

param_args = (N, r0, gamma, beta, alpha, sigma, omega)

t_pts = list(range(tspan[0], tspan[1]+1))



sol_SIR = do_solve(SIR, tspan, x0, param_args)

S_output = sol_SIR.sol(t_pts)[0,:]
E_output = sol_SIR.sol(t_pts)[1,:]
I_output = sol_SIR.sol(t_pts)[2,:]
R_output = sol_SIR.sol(t_pts)[3,:]
D_output = sol_SIR.sol(t_pts)[4,:]

fig=plt.figure(figsize = (15,10))
ax = fig.add_subplot(2, 1, 1)
ax.plot(t_pts,S_output, label = "Susceptible")
ax.plot(t_pts,E_output, label = "Exposed")
ax.plot(t_pts,I_output, label = "Infected")
ax.plot(t_pts,R_output, label = "Recovered")
ax.plot(t_pts,D_output, label = "Dead")
ax.legend()
ax.grid()
ax.set(xlabel="t [Days]", ylabel= "Individuals (Millions)", 
       title = "SEIRSD model output")

ax1 = fig.add_subplot(2, 1, 2)
ax1.plot(t_pts, my*I_output,"r" ,label = "Amount using healthcare")
ax1.axhline(maxhealth, label = "Healtcare capacity")
ax1.legend()
ax1.grid()


