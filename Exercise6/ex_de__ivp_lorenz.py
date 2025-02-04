# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 12:23:21 2022

@author: P. Maxwell
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def f_lorenz(t, w, sigma, beta, rho):
    """Lorenz differential equations"""
    x = w[0]
    y = w[1]
    z = w[2]
    return np.array([ sigma*(y-x), x*(rho-z) - y, x*y - beta*z ], dtype=np.float64 )



# Parameters in order sigma, beta, rho.
param_args = (10.0, 8/3, 28)

# Initial point
x0 = np.array([5.0, 5.0, 5.0])

# Time interval to integrate over
tspan = [0, 60]

# Set tolerances
reltol = 1e-12
abstol = 1e-14




def do_solve(f, tspan, x0, abstol, reltol, param_args):
    """Use SciPy to solve system explicitly with 8th order RK"""
    sol = solve_ivp(f, tspan, x0,
                            args=param_args,
                            method='DOP853', dense_output=True,
                            rtol=reltol, atol=abstol)
    return sol



    

def plot_lorenz(sol_lorenz, t_pts):
    # Plot using a 2x2 subplot figure, with t-x, t-y, t-z, and 3d x-y-z.
    fig = plt.figure(figsize=(30, 30))
    ax1 = fig.add_subplot(221)
    ax1.plot(t_pts, sol_lorenz.sol(t_pts)[0,:], 'b-', linewidth=1.0)
    ax1.tick_params(axis='both', labelsize=22)
    ax1.set_xlabel(r'$t$', fontsize=28)    
    ax1.set_ylabel(r'$x$', fontsize=28)    
    
    ax2 = fig.add_subplot(222)
    ax2.plot(t_pts, sol_lorenz.sol(t_pts)[1,:], 'b-', linewidth=1.0)
    ax2.tick_params(axis='both', labelsize=22)
    ax2.set_xlabel(r'$t$', fontsize=28)    
    ax2.set_ylabel(r'$y$', fontsize=28)    
    
    ax3 = fig.add_subplot(223)
    ax3.plot(t_pts, sol_lorenz.sol(t_pts)[2,:], 'b-', linewidth=1.0)
    ax3.tick_params(axis='both', labelsize=22)
    ax3.set_xlabel(r'$t$', fontsize=28)    
    ax3.set_ylabel(r'$z$', fontsize=28)    
            
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot3D(sol_lorenz.sol(t_pts)[0,:], sol_lorenz.sol(t_pts)[1,:], sol_lorenz.sol(t_pts)[2,:], 'b-', linewidth=1.0)
    ax4.tick_params(axis='both', labelsize=22)





sol_lorenz = do_solve(f_lorenz, tspan, x0, abstol, reltol, param_args)

t_pts = np.linspace(tspan[0], tspan[1], 10000)
plot_lorenz(sol_lorenz, t_pts)



