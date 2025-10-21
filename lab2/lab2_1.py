import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m=1
k=1
b = [0, 0.5, 2]

A1 = np.array([[0,1],[-k/m,-b[0]/m]])
A2 = np.array([[0,1],[-k/m,-b[1]/m]])
A3 = np.array([[0,1],[-k/m,-b[2]/m]])
B = np.array([[0],[1/m]])
C = np.array([[1,0]])

def control(t):
    #return np.array([[1]])
    return np.array([[np.sin(2*t)]])

def deg1(t,x):
    x = np.array([x]).T
    dx = A1@x + B@control(t)
    return np.ndarray.tolist(dx.T[0])

def deg2(t,x):
    x = np.array([x]).T
    dx = A2@x + B@control(t)
    return np.ndarray.tolist(dx.T[0])

def deg3(t,x):
    x = np.array([x]).T
    dx = A3@x + B@control(t)
    return np.ndarray.tolist(dx.T[0])

res1 = solve_ivp(deg1, [0,10], [0,0], rtol=1e-10)
res2 = solve_ivp(deg2, [0,10], [0,0], rtol=1e-10)
res3 = solve_ivp(deg3, [0,10], [0,0], rtol=1e-10)

y1 = C@np.array(res1.y)
y1 = np.ndarray.tolist(y1[0].T)

y2 = C@np.array(res2.y)
y2 = np.ndarray.tolist(y2[0].T)

y3 = C@np.array(res3.y)
y3 = np.ndarray.tolist(y3[0].T)

plt.figure()
plt.title("Odpowiedź czasowa na skok jednostkowy przy różnych wartościach tłumienia")
plt.plot(res1.t, y1, label="b = 0")
plt.plot(res2.t, y2, label="b = 0.5")
plt.plot(res3.t, y3, label="b = 2")
plt.legend()
plt.show()