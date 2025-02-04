# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:56:12 2024

@author: nicolay
"""

import autograd.numpy as np
import scipy.optimize as sp
from autograd import jacobian

rho = 50
a1 = 5.9755E9
a2 = 2.5962E12
a3 = 9.6283E15
V = 0.03
T = 6.744
Fa = 13.357
Fb = 24.482193508

def reactor(vec, Fr):
    Fra, Frb, Frc, Fre, Frp, Frg = Fr
    
    K1, K2, K3, k1, k2, k3, T2, r1, r2, r3, Fa_eff, Fb_eff, Fc_eff, Fe_eff, Fp_eff, Fg, Fsum, xa, xb, xc, xe, xp, xg = vec
    
    
    eq0 = -K1 + np.log(a1) - 120*T2
    eq1 = -K2 + np.log(a2) - 150*T2
    eq2 = -K3 + np.log(a3) - 200*T2
    eq3 = T2*T-1
    eq4 = k1 - np.exp(K1)
    eq5 = k2 - np.exp(K1)
    eq6 = k3 - np.exp(K3)
    eq7 = -r1 + k1*xa*xb*V*rho
    eq8 = -r2 + k2*xc*xb*V*rho
    eq9 = -r3 + k3*xp*xc*V*rho
    eq10 = -Fa_eff + Fa + Fra-r1
    eq11 = -Fb_eff + Fb+Frb-(r1+r2)
    eq12 = -Fc_eff+ Frc + 2*r1 - 2*r2 - r3
    eq13 =-Fe_eff + Fre + 2*r2
    eq14 = -Fp_eff + 0.1*Fre + r2 - 0.5*r3
    eq15 = -Fg + 1.5*r3
    eq16 = Fa_eff + Fb_eff + Fc_eff + Fe_eff + Fp_eff + Fg - Fsum
    eq17 = -Fa_eff + Fsum*xa
    eq18 = -Fb_eff + Fsum*xb  
    eq19 = -Fc_eff + Fsum*xc  
    eq20 = -Fe_eff + Fsum*xe  
    eq21 = -Fp_eff + Fsum*xp  
    eq22 = -Fg + Fsum*xg
    
    return np.array([eq0, eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq18, eq19, eq20, eq21, eq22], dtype=np.float64)
  
def separator(x, F):
    Fp, F_purge, Fra, Frb, Frc, Fre, Frp, Frg = x[0:8]
    Fa_eff, Fb_eff, Fc_eff, Fe_eff, Fp_eff, Fg = F
    
    eq23 = -Fp_eff + 0.1*Fe_eff + Fp
    eq24 = -F_purge + 0.9*(Fa_eff + Fb_eff + Fc_eff + 1.1*Fe_eff)
    eq25 = -Fra + 0.1*Fa_eff
    eq26 = -Frb + 0.1*Fb_eff
    eq27 = -Frc + 0.1*Fc_eff
    eq28 = -Fre + 0.1*Fe_eff
    eq29 = -Frp + 0.1*Fp_eff
    eq30 = -Frg + 0.1*Fg
    
    return np.array([eq23, eq24, eq25, eq26, eq27, eq28, eq29, eq30], dtype = np.float64)

jac1 = jacobian(reactor)
jac2 = jacobian(separator)

'''
vec0 = np.array([1, 2, 3, 300,
300, 600, 0.5, 1, 1, 1, 1, 1,
 1, 10, 1, 4, 20, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2])
'''
vec0 = np.array([4.71733931e+00, 6.34307707e+00, 7.14749256e+00, 1.11870204e+02,
 1.11870204e+02, 1.27091522e+03, 1.48279953e-01, 1.69482616e+00,
 1.09486252e+00, 4.39142020e+00, 1.23769231e+01, 3.28795252e+01,
 7.99552750e+00, 1.29281247e+02, 1.16083046e+01, 6.58713030e+00,
 2.00728657e+02, 6.16599705e-02, 1.63800853e-01, 3.98325162e-02,
 6.44059739e-01, 5.78308286e-02, 3.28160930e-02])
Fr = np.array([4.69, 14.54, 0.769, 14.40, 1.91, 0.318])


sep0 = np.array([3, 3, 4.69, 14.54, 0.769, 14.40, 1.91, 0.318])

error = 1
tol = 1e-2
j = 1
while error > tol:
    sol1 = sp.root(reactor, vec0, args = (Fr), method = "lm", jac = jac1, tol = 1e-10, options={"xtol":1e-8})
    #print(sol1.x)
    sol2 = sp.root(separator, sep0, args = sol1.x[2:8], method = "lm", jac = jac2, tol = 1e-10, options={"xtol":1e-8})
    error = np.linalg.norm(Fr-sol2.x[2:8])
    Fr = sol2.x[2:8]
    j += 1
    
print(sol1.x)
print(j)
for i in range(len(Fr)):
    print(round(Fr[i], 4))
    
        
    
        

