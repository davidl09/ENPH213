#%%
#Question 1

import sympy as sp
import numpy as np
import math as m
import matplotlib.pyplot as plt

x = sp.symbols('x')

#define a func object representing exp(sin(2x)) 
func = sp.exp(sp.sin(2*x))



def nthDiff(f, n : int): #returns the nth order derivative of a function using recursion
    return f if n <= 0 else nthDiff(sp.diff(f), n - 1)

lambdaFuncs = []
#create list of lambda function for derivatives, also printing derivatives

for i in range(4): 
    tempFunc = nthDiff(func, i)
    print(f"Derivative order {i}: {tempFunc}")
    lambdaFuncs.append(sp.lambdify(x, tempFunc))


#%%
#Create array of evenly spaced float in range [0, 2pi)
numRange = np.linspace(0, 2 * m.pi, 200)
print(len(numRange),numRange[0],numRange [ len(numRange)-1 ] - 2 * m.pi)

#%%
#plot derivatives of func
for i, func in enumerate(lambdaFuncs):
    plt.plot(numRange, [func(x) for x in numRange], label=f'Function {i+1}')
    
#Set x-axis ticks to match the values in numRange
plt.xticks(np.linspace(0, 2 * np.pi, 5), ['0', 'π/2', 'π', '3π/2', '2π'])

plt.show()
    
#%%
firstOrderFD = lambda f, x, h : (f(x + h) - f(x)) / h
firstOrderCD = lambda f, x, h : (f(x + h/2) - f(x - h/2)) / h

plt.plot(numRange, [lambdaFuncs[1](x) for x in numRange], label="First Order Analytical Derivative")
plt.plot(numRange, [firstOrderFD(func, x, 0.15) for x in numRange], label="First Order FD Derivative")
plt.plot(numRange, [firstOrderCD(func, x, 0.15) for x in numRange], label="First Order CD Derivative")



