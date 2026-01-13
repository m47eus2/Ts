import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

r = 5
l = 0.1
c = 0.01
Vin = 8

A = np.array([[0,1], [-1/(c*l), -1/(c*r)]])
B = np.array([[0], [Vin/(c*l)]])
C = np.array([[1, 0]])

w = 5
l1 = 2*w - 1/(c*r)
l2 = w**2 - 2*w/(c*r) + 1/(c**2*r**2) - 1/(c*l)
L = np.array([[l1], [l2]])

def generator(t):
    vd = 3 + 2*np.sin(t)
    vdD = 2*np.cos(t)
    vdDD = -2*np.sin(t)
    return [vd, vdD, vdDD]

def deg1(t, X):
    x = X[0:2]
    xe = X[2:4]

    [vd, vdD, vdDD] = generator(t)
    xd = np.array([vd, vdD]).T

    e = xd - x
    ufb = K @ e
    uff = (r*vd + l*vdD + c*l*r*vdDD) / (r*Vin)
    u = ufb + uff

    y = C@x
    dxe = A @ xe + B[:, 0] * u + L @ (y - C @ xe)

    dx = A @ x + B[:, 0] * u

    return np.concatenate((dx, dxe))

wc = 1
k1 = (c*l*wc**2 - 1) / Vin
k2 = (c*l/Vin) * (2*wc - 1/(c*r))
K = np.array([k1, k2])
res = solve_ivp(deg1, [0,10], [2,0,0,0], rtol=1e-10, atol=1e-10)

zadana = np.zeros_like(res.t)
zadana = [generator(t)[0] for t in res.t]
zadanaD = np.zeros_like(res.t)
zadanaD = [generator(t)[1] for t in res.t]

plt.figure()
plt.title("Stan układu")
plt.grid()
plt.plot(res.t, res.y[0], label="$x_1$")
plt.plot(res.t, res.y[1], label="$x_2$")
plt.plot(res.t, zadana, "--", label="$x_{1D}$")
plt.plot(res.t, zadanaD, "--", label="$x_{2D}$")
plt.legend()


plt.figure()
plt.title("Stan układu")
plt.grid()
plt.plot(res.t, res.y[2], label="$\hat{x_1}$")
plt.plot(res.t, res.y[3], label="$\hat{x_2}$")
plt.plot(res.t, zadana, "--", label="$x_{1D}$")
plt.plot(res.t, zadanaD, "--", label="$x_{2D}$")
plt.legend()
plt.show()