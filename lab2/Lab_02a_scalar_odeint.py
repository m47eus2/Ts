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
A = [[0, 1],
    [-k/m, -b/m]]

B = [0,
    1/m]

C = [1, 0]

#define time-dependent input function. Here, unit step is defined
def control(t):
    return 1

#define dynamic system.
def deg1(x, t):
    u = control(t)
    
    dx1 = x[1]
    dx2 = x[0] * A[1][0] + x[1] * A[1][1] + B[1] * u
       
    return [dx1, dx2]

#define the time vector - from 0 to 10s, with 101 samples
t = np.linspace(0,10,101)

#simulate the dynamic system, pass a system deg1, time of simulation and initial conditions
x = odeint(deg1, [0,0], t, rtol = 1e-10) #arguments rtol and atol sets calculation tolerance

#calculate output based on the obtained state
y = x[:,0]

#plot results
fig, ax = plt.subplots()
ax.plot(t, x[:,0])
ax.plot(t, x[:,1])

#plt.savefig("filepath.pdf", format = 'pdf')
plt.show()