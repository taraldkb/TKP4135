#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 10:16:03 2024

@author: taraldbishop
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ex_de__ivp_lorenz import do_solve

#Take function describing Lorenz equation
def f_lorenz(t, w, sigma, beta, rho):
    """Lorenz differential equations"""
    x = w[0]
    y = w[1]
    z = w[2]
    return np.array([ sigma*(y-x), x*(rho-z) - y, x*y - beta*z ], dtype=np.float64 )


# Take the paramater definitions

# Parameters in order sigma, beta, rho.
param_args = (10.0, 8/3, 28)

# Initial point
epsilon = 1e-5

# Time interval to integrate over
tspan = [0, 60]

# Set tolerances
reltol = 1e-12
abstol = 1e-14

#make x axes
t_pts = np.linspace(tspan[0], tspan[1], 10000)

#Loop params

output_x=[]
output_y=[]
output_z=[]
starting=[]
k = 3

for i in range(k):
    r =np.random.normal(loc = 0, scale = epsilon )
    x0 = np.array([5.0+r, 5.0, 5.0])
    starting.append(x0[0])
    
    sol_lorenz = do_solve(f_lorenz, tspan, x0, abstol, reltol, param_args)
    output_x.append(sol_lorenz.sol(t_pts)[0,:])
    output_y.append(sol_lorenz.sol(t_pts)[1,:])
    output_z.append(sol_lorenz.sol(t_pts)[2,:])

fig = plt.figure(figsize=(15, 30))

ax1= fig.add_subplot(3,1,1)
ax1.plot(t_pts, output_x[0], label ="x")
ax1.plot(t_pts, output_y[0], label ="y")
ax1.plot(t_pts, output_z[0], label ="z")  
ax1.legend()

ax2= fig.add_subplot(3,1,2)
ax2.plot(t_pts, output_x[1], label ="x")
ax2.plot(t_pts, output_y[1], label ="y")
ax2.plot(t_pts, output_z[1], label ="z") 
ax2.legend()

ax3= fig.add_subplot(3,1,3)
ax3.plot(t_pts, output_x[2], label ="x")
ax3.plot(t_pts, output_y[2], label ="y")
ax3.plot(t_pts, output_z[2], label ="z") 
ax3.legend()
    