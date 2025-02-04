#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 23:06:28 2024

@author: taraldbishop
"""

import numpy as np
import matplotlib.pyplot as plt
from PST5_2 import *
import autograd.numpy as np


#def starting conditions
t0 = np.float64(0.0)
tf = np.float64(60.0)
x0 = np.array([ 40 ], dtype=np.float64)
h = 0.05

# def konstants
tau = 50
k1 = 0.024
k2 = 1
k3 = 120
Ts = 50


#define function

def f(t,y):
    return (-y+k1*k3*(Ts-y)+k2*(30*2*np.sin(t/4)))/tau
    

t, x = forwardeuler(f, h, x0, t0, tf)

plt.plot(t,x, label='Outlet temp')
plt.plot(t, 30*2*np.sin(t/4),"--", label="inlet temp")
plt.xlabel("time [s]")
plt.ylabel("temp C")
plt.legend()
plt.grid()




