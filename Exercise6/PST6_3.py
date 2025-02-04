#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:46:52 2024

@author: taraldbishop
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian
from scipy.integrate import solve_ivp



#define functions

def do_solveRK45(f, tspan, x0, abstol, reltol):
    sol = solve_ivp(f, tspan, x0,
                            method='RK45', dense_output=True,
                            rtol=reltol, atol=abstol)
    return sol

def do_solveRadu(f, tspan, x0, abstol, reltol):
    sol = solve_ivp(f, tspan, x0,
                            method='Radau', dense_output=True,
                            rtol=reltol, atol=abstol)
    return sol

def f_robertson(t, y):
    y1= y[0]
    y2= y[1]
    y3= y[2]
    return np.array([-0.04*y1 + y2*y3*(10**4),
                      (-0.04*y1) - y2*y3*(10**4) - (y2**2) * (3*10**7),
                      (y2**2) * (3*10**7)], dtype=np.float64 )



jac_robertson_autograd = jacobian(f_robertson,1)
def jac_robertson(t,x):
    return jac_robertson_autograd(t,x)

# definie params and starting conditions

# Initial point
x0 = np.array([1.0, 0.0, 0.0])

# Time interval to integrate over
tspan = [0, 15]

# Set tolerances
reltol = 1e-4
abstol = 1e-7
t_pts = np.linspace(tspan[0], tspan[1], 10000)

sol_RK = do_solveRK45(f_robertson, tspan, x0, abstol, reltol)
sol_Rad = do_solveRadu(jac_robertson, tspan, x0, abstol, reltol)

fig = plt.figure(figsize=(10,15))
ax1=fig.add_subplot(3,2,1)
ax1.plot(t_pts,sol_RK.sol(t_pts)[0,:])
ax1.set(xlabel= "$t$", ylabel= " $y_1$", title= "Robertson $y_1$ (explicit)",
        xscale= "log")

ax2=fig.add_subplot(3,2,3)
ax2.plot(t_pts,sol_RK.sol(t_pts)[1,:])
ax2.set(xlabel= "$t$", ylabel= " $y_2$", title= "Robertson $y_2$ (explicit)",
        xscale= "log")


ax3=fig.add_subplot(3,2,5)
ax3.plot(t_pts,sol_RK.sol(t_pts)[2,:])
ax3.set(xlabel= "$t$", ylabel= " $y_3$", title= "Robertson $y_3$ (explicit)",
        xscale= "log")


import matplotlib as mpl
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian
from scipy.integrate import solve_ivp



#define functions

def do_solveRK45(f, tspan, x0, abstol, reltol):
    sol = solve_ivp(f, tspan, x0,
                            method='RK45', dense_output=True,
                            rtol=reltol, atol=abstol)
    return sol

def do_solveRadu(f, tspan, x0, abstol, reltol):
    sol = solve_ivp(f, tspan, x0,
                            method='Radau', dense_output=True,
                            rtol=reltol, atol=abstol)
    return sol

def f_robertson(t, y):
    y1= y[0]
    y2= y[1]
    y3= y[2]
    return np.array([-0.04*y1 + y2*y3*(10**4),
                     (-0.04*y1) - y2*y3*(10**4) - (y2**2) * (3*10**7),
                     (y2**2) * (3*10**7)], dtype=np.float64 )

jac_robertson_autograd = jacobian(f_robertson,1)
def jac_robertson(t,x):
    return jac_robertson_autograd(t,x)


# definie params and starting conditions

# Initial point
x0 = np.array([1.0, 0.0, 0.0])

# Time interval to integrate over
tspan = [0, 15]

# Set tolerances
reltol = 1e-4
abstol = 1e-7
t_pts = np.linspace(tspan[0], tspan[1], 10000)

sol_RK = do_solveRK45(f_robertson, tspan, x0, abstol, reltol)
sol_Rad = do_solveRadu(jac_robertson, tspan, x0, abstol, reltol)

fig = plt.figure(figsize=(10,15))
ax1=fig.add_subplot(3,2,1)
ax1.plot(t_pts,sol_RK.sol(t_pts)[0,:])
ax1.set(xlabel= "$t$", ylabel= " $y_1$", title= "Robertson $y_1$ (explicit)",
        xscale= "log")

ax2=fig.add_subplot(3,2,3)
ax2.plot(t_pts,sol_RK.sol(t_pts)[1,:])
ax2.set(xlabel= "$t$", ylabel= " $y_2$", title= "Robertson $y_2$ (explicit)",
        xscale= "log")


ax3=fig.add_subplot(3,2,5)
ax3.plot(t_pts,sol_RK.sol(t_pts)[2,:])
ax3.set(xlabel= "$t$", ylabel= " $y_3$", title= "Robertson $y_3$ (explicit)",
        xscale= "log")




