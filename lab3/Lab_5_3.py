import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

b = 0.5
k = 1
m = 1

A = np.array([[0, 1],[-k/m, -b/m]])
B = np.array([[0],[1/m]])
C = np.array([[1, 0]])

Tp = 0.001
t = np.arange(0, 5+Tp, Tp)

def u(t):
    return np.array([[np.sin(np.pi*t)]])

def yd(t):
    return np.sin(4*np.pi*t)**2

def ud(t):
    return np.array([[np.sin(4*np.pi*t)**2]])

def deg1(t, x):
    x = np.array([x]).T

    dx = A @ x + B @ (u(t) + ud(t))

    return np.ndarray.tolist(dx.T[0])

res = solve_ivp(deg1, [0,5], [2,0], t_eval=t, rtol=1e-10, atol=1e-10)

x1 = res.y[0]

x2Est = np.zeros_like(x1)
for k in range(1, len(x1)):
    x2Est[k] = (x1[k] - x1[k-1])/ Tp

plt.figure()
plt.title("Estymacja numeryczna $x_2$ z zaburzeniami procesu")
plt.grid()
plt.plot(res.t, res.y[0], label="$x_1$")
plt.plot(res.t, res.y[1], label="$x_2$")
plt.plot(res.t, x2Est, "--", label="$\hat{x}_2$")
plt.legend()
plt.show()