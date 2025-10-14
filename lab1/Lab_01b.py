# -*- coding: utf-8 -*-
"""
@author: Radoslaw Patelski
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.integrate import solve_ivp

#define transfer function numerator and denumerator vectors
A = [[-3, 1.25, -0.75, -2.75],[-6,3,-3.5,-6],[0,-1,0,1],[-6,5,-4.5,-6]]
B = [[0.5],[1],[0],[1]]
C = [[2,0,0,0]]
D = 0

ss = signal.StateSpace(A,B,C,D)
num,den = signal.ss2tf(A,B,C,D)

tf = signal.TransferFunction(num,den)

res_ss = signal.step(ss)
res_tf = signal.step(tf)

fig, ax = plt.subplots()
ax.plot(res_ss[0], res_ss[1])
plt.title('State Space')

fig, ax = plt.subplots()
ax.plot(res_tf[0], res_tf[1])
plt.title('Transfer Function')

plt.show()