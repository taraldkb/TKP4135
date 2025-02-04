#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:00:32 2024

@author: taraldbishop
"""
import matplotlib.pyplot as plt
import numpy as np
from PST3_part2_def import *




### function definitions  ###

def f(x):
    return x**3 + 5*x**2 -4*x -20 

def df(x):
    return 3*x**2 + 10*x-4


##### intitializations

x0a = 5.0
tol = 1e-10
x0b = 0.0

##### newtons

xfa, listofXa = Newton1d(f,df, x0a, tol)
xfb, listofXb = Newton1d(f,df, x0b, tol)

listofXa = np.array(listofXa)
flistXa =  f(listofXa)

listofXb = np.array(listofXb)
flistXb =  f(listofXb)

###  plotting
xa= np.arange(0,6,0.1)
xb= np.arange(-6,1,0.1)
ya = f(xa)
yb = f(xb)
 
plt.figure(1)
plt.plot(xa,ya, label= 'Function')
plt.plot(listofXa,flistXa, "--^", label = 'Newton iteration')
plt.plot(xfa,f(xfa),"x", label = 'Converged')
plt.grid()
plt.title("Newton for $x^3 + 5x^2-4x-20$,  $ x_0 = 5.0$")
plt.text(1.1,152, f'xf= {xfa}')
plt.legend()

plt.figure(2)
plt.plot(xb,yb, label= 'Function')
plt.plot(listofXb,flistXb, "--^", label = 'Newton iteration')
plt.plot(xfb,f(xfb),"x", label = 'Converged')
plt.plot(-2,0,"x", label='closest root')
plt.grid()
plt.title("Newton for $x^3 + 5x^2-4x-20$,  $ x_0 = 0.0$")
plt.text(-3,-20, f'xf= {xfb}')
plt.legend()




