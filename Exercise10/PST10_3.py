#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: taraldbishop
"""

import autograd.numpy as np 
from autograd import jacobian 
import scipy.optimize as sp
import numpy.linalg as lg
import matplotlib.pyplot as plt

ol1T = np.array([270.4, 270.6, 272.3, 273.6, 274.1, 275.5, 276.2, 276.4 , 276.5
                 , 276.6 , 276.7 , 276.8 , 277.1])
ol1exP = np.array([1.502, 1.556, 1.776, 2.096, 2.281, 2.721, 3.001, 3.54110097,
                   3.59305276 , 3.64629988 , 3.70087773 , 3.75682281 ,3.556])
ol1exlnP = np.log(ol1exP)


ol2T = np.array([270.4, 270.6, 272.3, 273.6, 274.1, 275.5, 276.2, 276.6,
                 277.1])
ol2exP = np.array([1.502, 1.556, 1.776, 2.096, 2.281, 2.721, 3.001, 5.24629988,
                   3.556])
ol2exlnP = np.log(ol2exP)



T_o= np.array([270.4, 270.6, 272.3, 273.6, 274.1, 275.5, 276.6, 277.1]) #K
exP_o= np.array([1.502, 1.556, 1.776, 2.096, 2.281, 2.721, 3.001, 3.556]) # mPa
y_o = np.log(exP_o)

def OLS(T,exP,y,title):
    
    def f(x,B):
        B0,B1,B2 =B
        return B0 + B1*x + B2*x**(-2)
    
    
    
    A= np.column_stack((np.ones_like(T),T,T**(-2)))
    AT = np.transpose(A)
    ATA = np.dot(AT,A)
    ATy = np.dot(AT,y)
    
    Beta = lg.solve(ATA, ATy)
    print(title +f' beta:{Beta}')
    
    
    x= np.linspace(np.min(T),np.max(T),100)
    
    sol_y_pts=f(x,Beta)
    y_calc_trans = np.exp(sol_y_pts)
    
    
    fig,(ax1,ax2)=plt.subplots(constrained_layout = True, ncols=2,
                               figsize=(10,4))
    
    ax1.plot(x,sol_y_pts, label="model")
    ax1.plot(T, y, "o" ,label = "experimental")
    ax1.legend()
    ax1.set_xlabel("T [C]")
    ax1.set_ylabel("ln(P)")
    ax1.grid()
    
    ax2.plot(x,y_calc_trans, label="model")
    ax2.plot(T, exP, "o" ,label = "experimental")
    ax2.legend()
    ax2.set_xlabel("T [C]")
    ax2.set_ylabel("P [mPA]")
    ax2.grid()
    fig.suptitle(title)
    

OLS(ol1T,ol1exP,ol1exlnP,"Outlier 1")
OLS(ol2T,ol2exP,ol2exlnP,"Outlier 2")
OLS(T_o, exP_o, y_o, "No outliers")