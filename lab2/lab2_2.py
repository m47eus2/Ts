import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

R1 = 0.2
R2 = 0.5
L = 0.1
C = 0.05

A = np.array([[-R1/L, -1/L],[1/C, -1/(R2*C)]])
B = np.array([[1/L], [0]])
C = np.array([[1, -1/R2]])

def control(t):
    return np.array([[2]])

def control2(t):
    return np.array([[2*np.sin(10*t)]])
    
def deg1(t,x):
    x = np.array([x]).T
    dx = A@x + B@control(t)
    return np.ndarray.tolist(dx.T[0])

def deg2(t,x):
    x = np.array([x]).T
    dx = A@x + B@control2(t)
    return np.ndarray.tolist(dx.T[0])

res1 = solve_ivp(deg1, [0,2], [0,0], rtol=1e-10)
res2 = solve_ivp(deg2, [0,2], [0,0], rtol=1e-10)

y1 = C@np.array(res1.y)
y1 = np.ndarray.tolist(y1[0].T)

y2 = C@np.array(res2.y)
y2 = np.ndarray.tolist(y2[0].T)

plt.plot(res1.t, y1, label="u = 2")
plt.plot(res2.t, y2, label="u = 2sin(10t)")
plt.legend()
plt.show()