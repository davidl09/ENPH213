#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 21:40:45 2024

@author: davidl09
"""

import numpy as np
import math as m

#%%
#Q1

A = np.linspace(21, 36, 16, True).reshape(-1, 4)
print("Matrix: ")
print(A)

def mat_U(array: np.array) -> np.array:
    result = np.zeros(array.shape)
    for i in range(array.shape[0]):
        result[i, i:] = array[i, i:]
    return result

print("Upper triangular: ")
print(mat_U(A))

def matrix_L(array: np.array) -> np.array:
    result = np.zeros(array.shape)
    for i in range(array.shape[0]):
        result[i, :i+1] = array[i, :i+1]
    return result

print("Lower triangular: ")
print(matrix_L(A))

def frobenius(array: np.array):
    return np.sqrt((array ** 2).sum())

print("Frobenius norm: ")
print(frobenius(A))

def infnorm(array: np.array):
    return np.max(np.abs(array).sum(axis=1))
 
print("infnorm: ")
print(infnorm(A))

#%%
#1b)
def mat_1_0_n(n : int):
    return -np.ones((n, n)) + np.diag(np.ones(n), 0) + matrix_L(np.ones((n, n)))
newA = mat_1_0_n(4)
print(newA)

#%%

print("Solving: ")
print(newA)
print("Without pert.: ")
print(np.linalg.solve(newA, np.tile([1, -1], 2))[:3])
print("With: ")
newA[-1, -1] -= 0.001
print(np.linalg.solve(newA, np.tile([1, -1], 2))[:3])
#%%
#now with 16x16 matrix

newA = mat_1_0_n(16)
print("Solving: ")
print(newA)
print("Without pert.: ")
print(np.linalg.solve(newA, np.tile([1, -1], 8))[:3])
newA[-1, -1] -= 0.001
print("With: ")
print(np.linalg.solve(newA, np.tile([1, -1], 8))[:3])


#%%
#Q2

def backsub1(U, bs):
    n = bs.size
    xs = np.zeros(n)
    
    xs[n-1] = bs[n-1] / U[n-1, n-1]
    for i in range(n-2, -1, -1):
        bb = 0
        for j in range(i + 1, n):
            bb += U[i, j] * xs[j]
        xs[i] = (bs[i] - bb)/U[i, i]
    return xs

def mysolve (f ,A , bs ):
    xs = f (A , bs )
    print ( ' my solution is : ' , xs [ 0 ] , xs [ 1 ] , xs [ 2 ] )
    
    
from timeit import default_timer

U = np.linspace(21, 21 + 5000**2 - 1, 5000**2, True).reshape(-1, 5000)
bs = U[0]

start = default_timer()
mysolve(backsub1, U, bs)
start = default_timer() - start
print(f'time1: {start}')
#%%

def backsub2(U, bs):
    n = bs.size
    xs = np.zeros(n, dtype=np.float64)
    for i in range(n-1, -1, -1):
        xs[i] = (bs[i] - np.dot(U[i, i + 1:], xs[i+1:])) / U [i, i]
        
    return xs

start = default_timer()
mysolve(backsub2, U, bs)
start = default_timer() - start
print(f'time1: {start}')

A = np.array([
    [2, 1, 1],
    [1, 1, -2],
    [1, 2, 1]], dtype=np.float64)

sol = np.array([8, -2, 2])


def gauss_elim(A, b):
    augmat = np.column_stack((A, b))
    n = len(b)
    
    for i in range(n):
        augmat[i, :] /= augmat[i, i]
        
        for j in range(i + 1, n):
            fact = augmat[j, i]
            augmat[j, :] -= fact * augmat[i, :]
            
    xs = backsub2(augmat[:, :-1], augmat[:, -1])
    return xs

print(gauss_elim(A, sol))

def gauss_elim_pivot(A, b):
    augmat = np.column_stack((A, b))
    n = len(b)
    
    for i in range(n):
        pivotRow = np.argmax(np.abs(augmat[i:, i])) + i
        augmat[[i, pivotRow]] = augmat[[pivotRow, i]]
        augmat[i, :] /= augmat[i, i]
        
        for j in range(i + 1, n):
            factor = augmat[j, i]
            augmat[j, :] -= factor * augmat[i, :]
            
    xs = augmat[:, -1]
    for i in range(n-2, -1, -1):
        xs[i] -= np.dot(augmat[i, i+1:n], xs[i+1:])
    return xs

A = np.array([
    [2, 1, 1],
    [2, 1, -4],
    [1, 2, 1]], dtype=np.float64)

print(gauss_elim_pivot(A, sol))
print("Solutions are correct")


        
        