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

R = np.hstack((B, A@B))
#print(np.linalg.matrix_rank(R))

vd = 5
xd = np.array([vd, 0]).T

def deg1(t, x):
    e = xd - x
    ufb = K @ e
    ud = vd / Vin
    u = ufb + ud

    dx = A @ x + B[:, 0] * u
    return dx

wc = -1
k1 = (c*l*wc**2 - 1) / Vin
k2 = (c*l/Vin) * (2*wc - 1/(c*r))
K = np.array([k1, k2])
res1 = solve_ivp(deg1, [0,8], [2,0], rtol=1e-10, atol=1e-10)

wc = 1
k1 = (c*l*wc**2 - 1) / Vin
k2 = (c*l/Vin) * (2*wc - 1/(c*r))
K = np.array([k1, k2])
res2 = solve_ivp(deg1, [0,8], [2,0], rtol=1e-10, atol=1e-10)

wc = 5
k1 = (c*l*wc**2 - 1) / Vin
k2 = (c*l/Vin) * (2*wc - 1/(c*r))
K = np.array([k1, k2])
res3 = solve_ivp(deg1, [0,8], [2,0], rtol=1e-10, atol=1e-10)

wc = 10
k1 = (c*l*wc**2 - 1) / Vin
k2 = (c*l/Vin) * (2*wc - 1/(c*r))
K = np.array([k1, k2])
res4 = solve_ivp(deg1, [0,8], [2,0], rtol=1e-10, atol=1e-10)

plt.figure()
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("x []")
plt.title("Odpowiedź układu ze sprzężeniem od stanu")
plt.plot(res1.t, res1.y[0], label="$x_1 (\omega_c = -1)$")
plt.plot(res1.t, res1.y[1], label="$x_2 (\omega_c = -1)$")
plt.legend()

plt.figure()
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("x []")
plt.title("Odpowiedź układu ze sprzężeniem od stanu")
plt.plot(res2.t, res2.y[0], label="$x_1 (\omega_c = 1)$")
plt.plot(res2.t, res2.y[1], label="$x_2 (\omega_c = 1)$")
plt.plot(res3.t, res3.y[0], label="$x_1 (\omega_c = 5)$")
plt.plot(res3.t, res3.y[1], label="$x_2 (\omega_c = 5)$")
plt.plot(res4.t, res4.y[0], label="$x_1 (\omega_c = 10)$")
plt.plot(res4.t, res4.y[1], label="$x_2 (\omega_c = 10)$")
plt.legend()
plt.show()