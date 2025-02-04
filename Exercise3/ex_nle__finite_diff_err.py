# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:08:19 2022

@author: peterma
"""



import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



def forwarddiff(f, x, h):
    return (f(x+h) - f(x)) / h

def centraldiff(f, x, h):
    return (f(x+h)-f(x-h))/(2*h)




def f_test(x):
    return np.exp(np.sin(x))

def df_test(x):
    return np.cos(x) * np.exp(np.sin(x))



N = 50
x0 = 0.0
h = np.float64(10)**np.linspace(-15, -1, 50)


# Calculate `true' derivative
df_x0 = df_test(x0)


# Calculate finite difference estimates using varying h
df_fd_h_x0 = forwarddiff(f_test, x0, h)
df_cd_h_x0 = centraldiff(f_test, x0, h)

# Calculate error
err_fd = abs(df_fd_h_x0 - df_x0)
err_cd = abs(df_cd_h_x0 - df_x0)


print(f'minimum: {np.min(err_cd)}')

# Plot it
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.loglog(h, err_fd, 'b--',label='forward' ,linewidth=3)
ax.loglog(h, err_cd, 'g--',label='central' ,linewidth=3)
ax.grid()
ax.legend()

ax.set_title(r'Error in difference approximation of $f''$', fontsize=20)
ax.set_xlabel(r'$h$', fontsize=20)
ax.set_ylabel(r'err', fontsize=20)
ax.tick_params(axis='both', labelsize=16)


