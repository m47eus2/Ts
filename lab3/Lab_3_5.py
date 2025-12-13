import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

b = 0.5
k = 1
m = 1

A = np.array([[0, 1],[-k/m, -b/m]])
B = np.array([[0],[1/m]])
C = np.array([[1, 0]])

def u(t):
    return np.array([[np.sin(np.pi*t)]])

def deg1(t, q):
    x = np.array([q]).T[0:2];
    xe = np.array([q]).T[2:4];
    
    y = C@x
       
    dx = A @ x + B @ u(t)
    dxe = A @ xe + B @ u(t) + L @ (y - C @ xe)
    
    return np.ndarray.tolist(np.hstack((dx.T[0], dxe.T[0])))

def sim(title):
    res = solve_ivp(deg1, [0,5], [2,0,0,0], rtol=1e-10, atol=1e-10)

    plt.figure()
    plt.title(title)
    plt.grid()
    plt.plot(res.t, res.y[0], label="$x_1$")
    plt.plot(res.t, res.y[2], "--", label="$\hat{x}_1$")
    plt.plot(res.t, res.y[1], label="$x_2$")
    plt.plot(res.t, res.y[3], "--", label="$\hat{x}_2$")
    plt.legend()

w = -1;
l1 = 2*w - b/m
l2 = w*w - 2*w*b/m + (b*b)/(m*m) - k/m
L = np.array([[l1], [l2]])
sim("Zmienne stanu i ich estymaty (w0=-1)")

w = 1;
l1 = 2*w - b/m
l2 = w*w - 2*w*b/m + (b*b)/(m*m) - k/m
L = np.array([[l1], [l2]])
sim("Zmienne stanu i ich estymaty (w0=1)")

w = 5;
l1 = 2*w - b/m
l2 = w*w - 2*w*b/m + (b*b)/(m*m) - k/m
L = np.array([[l1], [l2]])
sim("Zmienne stanu i ich estymaty (w0=5)")

w = 10;
l1 = 2*w - b/m
l2 = w*w - 2*w*b/m + (b*b)/(m*m) - k/m
L = np.array([[l1], [l2]])
sim("Zmienne stanu i ich estymaty (w0=10)")

plt.show()