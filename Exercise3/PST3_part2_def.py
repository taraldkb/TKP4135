#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:46:49 2024

@author: taraldbishop

function defining
"""

import numpy as np

def Newton1d(f, df, x0, tol, maxit=50 ):
    xk=x0  # init
    k=0
    xlist=[x0] # to store all x for plotting 
    while k<maxit:
        k += 1  # iteration counter
        xk1 = xk- (f(xk)/df(xk)) #newtons
      
        
        if abs(xk1-xk)<tol: # check convergance
            break
        xk=xk1 #update values
        xlist.append(xk1)
        
        
    if (k>=maxit):
        print("Iterations reached limit") #warn if not converged
    
    return xk1, xlist



        
        
        
        