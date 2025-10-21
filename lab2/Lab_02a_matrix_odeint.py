# -*- coding: utf-8 -*-
"""
@author: Radoslaw Patelski
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#initialize constant values
b = 0
k = 1
m = 1

#define arrays A, B and C of a linear system
A = np.array([[0, 1],
              [-k/m, -b/m]])

B = np.array([[0],
              [1/m]])

C = np.array([[1, 0]])

#define time-dependent input function. Here, unit step is defined
def control(t):
    return np.array([[1]]) #return as an array with single element

#define dynamic system.
def deg1(x, t):
    x = np.array([x]).T #for compatibility with a solver, convert x to matrix and get transpose
    
    dx = A @ x + B @ control(t); #calculate state equation
    #note: matrix multiplication is done using @ operator
    
    return np.ndarray.tolist(dx.T[0]) #for compatibility with a solver, transpose x and convert into list


#define the time vector - from 0 to 10s, with 101 samples
t = np.linspace(0,10,101)

#simulate the dynamic system, pass a system deg1, time of simulation and initial conditions
x = odeint(deg1, [0,0], t, rtol = 1e-10) #arguments rtol and atol sets calculation tolerance

#calculate output based on the obtained state
y = C @ x.T
y = y.T

#plot results
fig, ax = plt.subplots()
ax.plot(t, x[:,0])
ax.plot(t, x[:,1])

#plt.savefig("filepath.pdf", format = 'pdf')
plt.show()
