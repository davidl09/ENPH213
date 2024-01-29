#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:28:15 2024

@author: davidl09
"""

#%% 
#imports

import numpy as np
import math as m
from scipy import integrate
import mpmath as mp
#%%

#example from class

# Rectangle rule for integration
n = 100
a = 0
b = 1
h = ( b - a ) / n # -1 : the error was here, the h-value is (b-a)/n, not (b-a)/(n-1) 
integral = 0 # Initialize the integral variable
for i in range (0 , n ) :
    xi = a + h * i # Determine the xi value for the loop
    integral = integral + ( 2 / np.pi ** 0.5 ) * np.exp (-(xi ** 2)) * h
    
abs_err = lambda a_f, f : np.abs((a_f - f) / f)
    
print(f"Analytical answer: {m.erf(1)}")
    
print(f"Given Method: {integral}, absolute error: {abs_err(integral, m.erf(1))}")


#%%

# efficient vectorized subroutine

def integrate_sq(f, a, b, n):
    xi = np.linspace(a, b, n, True)
    dx = (b - a) / n
    return (f(xi[:-1]) * dx).sum()

#%% 

#test efficient routine defined above

d_dx_erf = lambda x : 2 / np.pi ** 0.5 * np.exp(-(x**2))

print(f"Integrate square: {integrate_sq(d_dx_erf, 0, 1, n)}, absolute error: {abs_err(integrate_sq(d_dx_erf, 0, 1, n), m.erf(1))}")

#Results match 

#%%

#integrate with trapezoidal approximation

def integrate_tr(f, a, b, n):
    xi = np.linspace(a, b, n, True)
    dx = xi[1] - xi[0]
    return (np.add(f(xi[:-1]), f(xi[1:])) * dx / 2).sum()

print(f"Trapezoidal method: {integrate_tr(d_dx_erf, 0, 1, n)}, absolute error: {abs_err(integrate_tr(d_dx_erf, 0, 1, n), m.erf(1))}")
    

def weights(n):
    return np.array([1 if i == 0 else 1 if i == n - 1 else 4 if i % 2 == 1 else 2 for i in range(n)])

#Simpson's method returns inaccurate results for even n because the last integration 'window' does not reach the upper bound b. 
#This means the summed area is in the range [a, b - h) and not [a, b).

make_odd = lambda x : x + 1 - x % 2
def integrate_sp(f, a, b, n, makeOdd : bool = True):
    #constants
    #make n odd
    if makeOdd:
        n = make_odd(n)
    dx = (b - a) / n
    
    coeffs = weights(n) * dx / 3
    
    xi = np.linspace(a, b, n, False)
    
    return (coeffs * f(xi)).sum()
    

print(f"Simpson's Method: {integrate_sp(d_dx_erf, 0, 1, n)}, absolute error: {abs_err(integrate_sp(d_dx_erf, 0, 1, n), m.erf(1))}")


#%%

#adaptive step size

def adaptive_step(f, simpson : bool = False, n_init : int = 3, a = 0, b = 1):
    function = integrate_sp if simpson else integrate_tr
    err = abs_err(function(f, a, b, n_init), m.erf(1))
    while err > 1e-10 and n_init < 1000000: #Simpson's converges too slowly, so I stop this function at a large n to avoid taking too long
        err = abs_err(function(f, a, b, n_init), m.erf(1))
        print(f"Error: {err}")
        n_init = n_init * 2 - 1
        print(f"N: {n_init}")
        
print("Trap Rule, Adaptive step: ")
adaptive_step(d_dx_erf, False)
print("Simpson's, Adaptive Step: ")
adaptive_step(d_dx_erf, True)

#%%

#Q2

def func(x, y) -> float:
    return mp.sqrt(x ** 2 + y) * mp.sin(x) * mp.cos(y)

def simp2d(f, a, b, c, d, n, m):
    n = make_odd(n)
    m = make_odd(m)
    
    dx = (b - a) / n
    dy = (d - c) / m
    
    X = weights(n) * dx / 3
    Y = weights(m) * dy / 3
    
    wMat = np.outer(X, Y)
    
    xyVals = np.meshgrid(np.linspace(a, b, n, False), np.linspace(c, d, m, False), indexing='ij')
    
    xVals = xyVals[0]
    yVals = xyVals[1]
    
    vfunc = np.vectorize(f)
    
    return (wMat * vfunc(xVals, yVals)).sum()

n = 1001
m_ = 1001
print(f"Simpson's 2D rule: {simp2d(func, 0, np.pi, 0, np.pi / 2, n, m_)}")


#%%

#mpmath method
f = lambda x, y : mp.sqrt(x ** 2 + y) * mp.sin(x) * mp.cos(y)
print(f"Mpmath Quad: {mp.quad(f, [0, np.pi], [0, np.pi/2])}")

#%%
#scipy method
f_ = lambda y, x : f(x, y)
ans = integrate.dblquad(f_, 0, np.pi, 0, np.pi / 2)
print(f"Scipy method: {ans[0]}, error: {ans[1]}")