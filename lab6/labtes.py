import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.integrate import solve_ivp

m = 1
k = 1
b = 0.5
Tp = 1

A = np.array([[0, 1], [-k/m, -b/m]])
B = np.array([[0], [1/m]])
C = np.array([[1, 0]])
D = 0

Ad = np.array([[1,Tp],[-k*Tp / m, 1-(b*Tp)/m]])
Bd = np.array([[0],[Tp/m]])
Cd = C
Dd = D

t_end = 20
t_in = np.arange(0, t_end, Tp)
u = np.ones(len(t_in))

tout, yout, xd = signal.dlsim((Ad, Bd, Cd, Dd, Tp), u, t=t_in)

def control(t):
    return np.array([[1]])

def deg1(t, x):
    x = np.array([x]).T
    
    dx = A @ x + B @ control(t)
    
    return np.ndarray.tolist(dx.T[0])


res = solve_ivp(deg1, [0,20], [0,0], rtol=1e-10)


plt.plot(res.t, res.y[0], label="Obiekt ciągły")
plt.plot(tout, yout, "--", label="Obiekt dyskretny")
plt.legend()
plt.grid()
plt.show()

