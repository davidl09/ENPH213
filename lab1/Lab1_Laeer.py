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
plt.title("nth Derivatives for n := [0..3]")
for i, f in enumerate(lambdaFuncs):
    plt.plot(numRange, [f(x) for x in numRange], label=f'Derivative Order {i}')
    
#Set x-axis ticks to match the values in numRange
plt.xticks(np.linspace(0, 2 * np.pi, 5), ['0', 'π/2', 'π', '3π/2', '2π'])
plt.legend()
plt.xlabel("x")

plt.show()
    
#%%
#Forward difference and central difference derivatives
firstOrderFD = lambda f, x, h : (f(x + h) - f(x)) / h
firstOrderCD = lambda f, x, h : (f(x + h/2) - f(x - h/2)) / h

plt.title("First Order FD, CD, and Analytical Derivatives")

plt.plot(numRange, [lambdaFuncs[1](x) for x in numRange], label="Analytical")
plt.plot(numRange, [firstOrderFD(lambdaFuncs[0], i, 0.15) for i in numRange], label="FD, h:=0.15")
plt.plot(numRange, [firstOrderFD(lambdaFuncs[0], i, 0.5) for i in numRange], label="FD, h:=0.5")
plt.plot(numRange, [firstOrderCD(lambdaFuncs[0], i, 0.15) for i in numRange], label="CD, h:=0.15")
plt.plot(numRange, [firstOrderCD(lambdaFuncs[0], i, 0.5) for i in numRange], label="CD, h:=0.5")
plt.xticks(np.linspace(0, 2 * np.pi, 5), ['0', 'π/2', 'π', '3π/2', '2π'])
plt.legend()
plt.xlabel("x")

plt.show()

#%%

#array of precomputed h values
hVals = np.array(10 ** np.linspace(-16, 0, 17, dtype=np.float64))
plt.title("Absolute Error of Numerical CD and FD Derivatives w.r.t. h")

#machine precisison of float64
eps_m = np.finfo(np.float64).eps

plt.loglog(hVals, [abs(firstOrderFD(lambdaFuncs[0], 1, h_) - lambdaFuncs[1](1)) for h_ in hVals], label="FD Error")
plt.loglog(hVals, [abs(firstOrderCD(lambdaFuncs[0], 1, h_) - lambdaFuncs[1](1)) for h_ in hVals], label="CD Error")
plt.loglog(hVals, [2 * abs(lambdaFuncs[0](1)) * eps_m / h_ for h_ in hVals], label=u"Machine Rounding Error \u03B5")
plt.legend()
plt.xlabel("h")
plt.ylabel("Absolute Error")

plt.show()

#%%
#Question 2

#BONUS - 2(b) - Recursive cd_n definition: nthNumericalDiff function
#uses recursion, returns lambda functions that 
def nthNumDiff(f, h, n):
    if n <= 0:
        return f
    elif n == 1:
        return lambda x : firstOrderCD(f, x, h)
    else: return lambda x : nthNumDiff(lambda y : firstOrderCD(f, y, h), h, n - 1)(x)
    
#Testing nthNumDiff function
testFunc = lambda x : x **2
range_ = np.linspace(-1, 1, 100)

print("Testing nth order numerical differentiation")

for i in range(4):
    plt.plot(range_, list(map(nthNumDiff(testFunc, 0.1, i), range_)), label=f"f^({i})")
plt.legend()
plt.title("Extra: Testing numerical diff. func. on y = x**2")
plt.show()

#%%
def legendre(n : int):
    return lambda x : (1 / ((2 ** n) * m.factorial(n))) * nthNumDiff(lambda y : (y ** 2 - 1) ** n, 0.01, n)(x)

import scipy.special as scisp

f1 = plt.figure()
plt.title("Legendre Polynomials 0-4")
for i in range(5):
    plt.plot(range_, legendre(i)(range_), label=f"Legendre {i}")

plt.legend()
plt.show()
    
    
f1 = plt.figure()
plt.title("Legendre Polynomials 0-4 (Scipy Impl.)")
for i in range(5):
    plt.plot(range_, scisp.legendre(i)(range_), label=f"Legendre {i}")

plt.legend()
plt.show()