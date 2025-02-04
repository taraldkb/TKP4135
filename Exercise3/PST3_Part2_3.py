#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:28:45 2024

@author: taraldbishop
"""
import matplotlib.pyplot as plt
import numpy as np
from PST3_part2_def import *


### function definitions  ###

def f(x):
    return np.arctan(x)

def df(x):
    return 1/(x**2 + 1)

##### intitializations

x0a = 1.3
tol = 1e-10

##### newtons

xfa, listofXa = Newton1d(f,df, x0a, tol)



listofXa = np.array(listofXa)
flistXa =  f(listofXa)


###  plotting
xa= np.arange(-1,5,0.1)
ya= f(xa)
plt.figure(1)
plt.plot(xa,ya, label= 'Function')
plt.plot(listofXa,flistXa, "--^", label = 'Newton iteration')
plt.plot(xfa,f(xfa),"x", label = 'Converged')
plt.grid()
plt.title(f"Newton for arctan(x)  $ x_0 =$ {x0a}")
plt.text(2,0.5, f'xf= {xfa}')
plt.legend()