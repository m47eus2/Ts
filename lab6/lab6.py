import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.integrate import solve_ivp

m=1
k=1
b=0.5
Tp = 0.01

A = np.array([[0,1],[-k/m, -b/m]])
B = np.array([[0],[1/m]])
C = np.array([[1, 0]])
D=0

t_in = 
u = 

tout, yout, xd = signal.dlsim((A,B,C,D,Tp), u)

plt.figure()
plt.plot(tout, yout)
plt.show()