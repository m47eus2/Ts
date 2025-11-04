# -*- coding: utf-8 -*-
"""
@author: Radoslaw Patelski
zmiany: Paweł Parulski
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


folder = ''


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

w = 10;

l1 = 2*w - b/m
l2 = w*w - 2*w*b/m + (b*b)/(m*m) - k/m

L = np.array([[l1], [l2]])

#define time-dependent input function. Here, unit step is defined
def u(t):
    return np.array([[np.sin(np.pi*t)]]) #return as an array with single element

#define dynamic system.
def deg1(t, q):
    x = np.array([q]).T[0:2];
    xe = np.array([q]).T[2:4];
    
    y = C@x
       
    dx = A @ x + B @ u(t); #calculate state equation
    dxe = A @ xe + B @ u(t) + L @ (y - C @ xe)

    #note: matrix multiplication is done using @ operator
    
    return np.ndarray.tolist(np.hstack((dx.T[0], dxe.T[0])))

#simulate the dynamic system, pass a system deg1, time of simulation and initial conditions
res = solve_ivp(deg1, [0,10], [2,0,0,0], rtol=1e-8) #arguments rtol and atol sets calculation tolerance


#plot results
# plt.figure()
# plt.grid()
# plt.plot(res.t, res.y[0])
# plt.plot(res.t, res.y[1])
# plt.legend(["x1","x2"])
# plt.title('Przebieg zmiennych stanu')

nazwa = 'cw_3_z3_zm_ivp' 
# plt.savefig(folder+nazwa+'.png', bbox_inches='tight', dpi=200)


# plt.figure()
# plt.grid()
# plt.plot(res.t, res.y[2])
# plt.plot(res.t, res.y[3])
# plt.legend(["x1","x2"])
# plt.title('Przebieg estymat stanu')

nazwa = 'cw_3_z3_est_ivp' 
# plt.savefig(folder+nazwa+'.png', bbox_inches='tight', dpi=200)



plt.figure()
plt.grid()
plt.plot(res.t, res.y[0])
plt.plot(res.t, res.y[2],'r--')
plt.legend(["$x_1$","$\hat{x}_1$"])
plt.title('Stan vs estymata')


plt.figure()
plt.grid()
plt.plot(res.t, res.y[1])
plt.plot(res.t, res.y[3],'r--')
plt.legend(["$x_2$","$\hat{x}_2$"])
plt.title('Stan vs estymata')

plt.show()