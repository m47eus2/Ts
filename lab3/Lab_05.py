# -*- coding: utf-8 -*-
"""
@author: Radoslaw Patelski
zmiany: Paweł Parulski
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sterowanie(t):
    return np.array([[np.sin(np.pi*t)]])

def simulate(A,B,C,Tend):
    dt = 0.001
    T = Tend
    N = int(T/dt)
    x = np.zeros(2)

    t_list = []
    x_list = []
    y_list = []
    u_list = []

    for k in range(N):
        x=x.T
        t = k * dt
        y = C @ x
        x_list.append([x[0], x[0]])

        u = sterowanie(t)

        # Obiekt
        dx = A @ x + B.flatten() * u
        x = x + dt * dx

        t_list.append(t)
        y_list.append(y)
        u_list.append(u)

    return [t_list, y_list, x_list]


#initialize constant values
b = 0.5
k = 1
m = 1

#define arrays A, B and C of a linear system
A = np.array([[0, 1],
              [-k/m, -b/m]])

B = np.array([[0],
              [1/m]])

C = np.array([[1, 0]])

ret = simulate(A,B,C,10)


print(ret[1])