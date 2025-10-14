# -*- coding: utf-8 -*-
"""
@author: Radoslaw Patelski
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.integrate import solve_ivp

#define transfer function numerator and denumerator vectors
num = [1, 4]
den = [2, 0, 1]

A = [[0,1],[-0.5, 0]]
B = [[0.5],[2]]
C = [[1, 0]]
D = [[0]]

tf = signal.TransferFunction(num,den)
sstf = signal.tf2ss(num,den)
ss = signal.StateSpace(A,B,C,D)

t = np.linspace(0,5,50)
u = np.full((50, 1), 1)

tout_sstf, yout_sstf, xout_sstf = signal.lsim(sstf, u, t, X0=[1, 2])
tout_tf, yout_tf, xout_tf = signal.lsim(tf, u, t, X0=[1, 2])
tout_ss, yout_ss, xout_ss = signal.lsim(ss, u, t, X0=[1, 2])

fig, ax = plt.subplots()
ax.plot(tout_ss, xout_ss.T[0])
ax.plot(tout_ss, xout_ss.T[1])
ax.plot(tout_ss, yout_ss, 'r:')
ax.legend(['x1','x2','y'])
plt.title('State Space')

fig, ax = plt.subplots()
#plt.plot(tout_tf, xout_tf.T[0])
#plt.plot(tout_tf, xout_tf.T[1])
ax.plot(tout_tf, yout_tf, 'r:')
ax.legend(['x1','x2','y'])
plt.title('Transfer Function')

fig, ax = plt.subplots()
ax.plot(tout_sstf, xout_sstf.T[0])
ax.plot(tout_sstf, xout_sstf.T[1])
ax.plot(tout_sstf, yout_sstf, 'r:')
ax.legend(['x1','x2','y'])
plt.title('State Space from Transfer Function')

plt.show()