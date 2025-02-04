#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:07:56 2024

@author: taraldbishop
"""

import autograd.numpy as np 
from autograd import jacobian 
import scipy.optimize as sp
import numpy.linalg as lg


#defining params
rho = 50.0
a1 = 5.9755*10**9
a2 = 2.5962 *10**12
a3 = 9.6283 *10**15
V = 0.03
T = 6.744
Fa = 13.357
Fb = 24.482193508



# defining functions

def f_sep(inp, Feff):
    
    #unpack varaibler
    Fp, Fpurge, Fr_a, Fr_b, Fr_c, Fr_e, Fr_p, Fr_g = inp
    Fa_eff, Fb_eff, Fc_eff, Fe_eff, Fp_eff, Fg = Feff
    
    
    eq23 = -Fp_eff + 0.1*Fe_eff+Fp
    eq24 = -Fpurge + 0.9*(Fa_eff +Fb_eff + Fc_eff + 1.1*Fe_eff)
    eq25 = -Fr_a + 0.1 * Fa_eff
    eq26 = -Fr_b + 0.1 * Fb_eff
    eq27 = -Fr_c + 0.1 * Fc_eff
    eq28 = -Fr_e + 0.1 * Fe_eff
    eq29 = -Fr_p + 0.1 * Fp_eff
    eq30 = -Fr_g + 0.1 * Fg
    
    return np.array([eq23, eq24, eq25, eq26, eq27, eq28, eq29, eq30
                     ], dtype= np.float64)


def f_reac(reac_array,F_r):
    
    #unpack varaibler
    k1_m, k2_m, k3_m, k1, k2, k3, T2, r1, r2, r3, Fa_eff, Fb_eff, Fc_eff, Fe_eff,Fp_eff, Fg, F_sum, xa, xb, xc, xe, xp, xg = reac_array
    Fr_a, Fr_b, Fr_c, Fr_e, Fr_p, Fr_g = F_r
    
    
    eq0 = k1_m +np.log(a1)-120*T2
    eq1 = k2_m +np.log(a2)-150*T2
    eq2 = k3_m +np.log(a3)-200*T2
    eq3 = T2*T-1.0
    eq4 = k1 - np.exp(k1_m)
    eq5 = k2 - np.exp(k2_m)
    eq6 = k3 - np.exp(k3_m)
    eq7 = -r1 + k1*xa*xb*V*rho 
    eq8 = -r2 + k2*xc*xb*V*rho
    eq9 = -r3 + k3*xp*xc*V*rho
    eq10 = -Fa_eff + Fa +Fr_a -r1
    eq11 = -Fb_eff + Fb + Fr_b - (r1+r2)
    eq12 = -Fc_eff + Fr_c + 2.0*r1 -2.0*r2-r3
    eq13 = -Fe_eff + Fr_e + 2*r2
    eq14 = -Fp_eff + 0.1*Fr_e + r2 - 0.5*r3
    eq15 = -Fg + 1.5*r3
    eq16 = Fa_eff + Fb_eff + Fc_eff + Fe_eff + Fp_eff + Fg - F_sum
    eq17 = -Fa_eff + F_sum*xa
    eq18 = -Fb_eff + F_sum*xb
    eq19 = -Fc_eff + F_sum*xc
    eq20 = -Fe_eff + F_sum*xe
    eq21 = -Fp_eff + F_sum*xp
    eq22 = -Fg + F_sum*xg
    
    return np.array([eq0, eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8,
                     eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16, 
                     eq17, eq18, eq19, eq20, eq21, eq22], dtype= np.float64)
    
# defining jacobians
jac_reac = jacobian(f_reac)
jac_sep = jacobian(f_sep)


# intitial guess
Fr0 = np.array([4.69, 14.54, 0.769, 14.40, 1.91, 0.318])# Fr_a , b, c, e, p, g

reac = np.array([1, 2, 3, 300,
300, 600, 0.5, 1, 1, 1, 1, 1,
 1, 10, 1, 4, 20, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2])

sep = np.array([3, 3, 4.69, 14.54, 0.769, 14.40, 1.91, 0.318])

# initialize solving params
error = 1
conve_tol =1e-2
count= 0
Fr=Fr0
while error>conve_tol:
    sol_reac =  sp.root(f_reac, reac, args = (Fr0), method = "lm", jac = jac_reac,
                        tol = 1e-10, options={"xtol":1e-8})
    sol_sep = sp.root(f_sep, sep, args =sol_reac.x[10:16] , method = "lm", jac = jac_sep,
                        tol = 1e-10, options={"xtol":1e-8})
    solfr=sol_sep.x[2:8]
    error = np.linalg.norm(Fr-sol_sep.x[2:8])
    Fr = sol_sep.x[2:8]
    count += 1
    
    











    
    
    
    
    
    