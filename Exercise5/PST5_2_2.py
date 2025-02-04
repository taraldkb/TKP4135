#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:28:59 2024

@author: taraldbishop
"""

import numpy as np
import matplotlib.pyplot as plt
from PST5_2 import *
import autograd.numpy as np


t0 = np.float64(0.0)
tf = np.float64(1.5)
x0 = np.array([ 1.0 ], dtype=np.float64)

Nh = 5
vh = 10**np.linspace(-1, -5, Nh)
err_h = np.zeros(Nh, dtype=np.float64)

for i in range(0, Nh):
    (vt_k, vx_k) = forwardeuler(fn_gh, vh[i], x0, t0, tf)
    exact_sol = np.exp(1/4 - (1/2 - vt_k)**2)
    err_h[i] = np.linalg.norm(exact_sol - vx_k[:,0], np.inf)
    
    
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.loglog(vh, err_h, 'bo-', linewidth=3)
ax.grid()
ax.set_title(r"Log-log plot of inf absolute error against stepsize $h$", fontsize=20)
ax.set_xlabel(r'$h$', fontsize=20)
ax.set_ylabel(r'Err', fontsize=20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
