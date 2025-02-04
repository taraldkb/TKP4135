#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:27:05 2024

@author: taraldbishop
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ex_de__ivp_lorenz import do_solve


#define functions
def f_brussel(t, y, A, B ):
    y1=y[0]
    y2= y[1]
    return np.array([A+y1**2 *y2 - (B+1)*y1, 
                     B*y1-y1**2 *y2], dtype=np.float64 )




# starting conditions and parms 

# Parameters in order A,B
param_args = (1,3 )

# Initial point
x0 = np.array([1.0, 3.08])

# Time interval to integrate over
tspan = [0, 60]

# Set tolerances
reltol = 1e-12
abstol = 1e-14
t_pts = np.linspace(tspan[0], tspan[1], 10000)

sol_brussel = do_solve(f_brussel, tspan, x0, abstol, reltol, param_args)
output_y1= sol_brussel.sol(t_pts)[0,:]
output_y2= sol_brussel.sol(t_pts)[1,:]

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(output_y1, output_y2)
ax.set(xlabel="$y_1$", ylabel="$y_2$", title="Brusselator Simple" )
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(3,4, f"A=1 \nB=3 \nt_f = {tspan[1]}", bbox=props)
ax.grid()