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

def generator(t):
    vd = 3 + 2*np.sin(t)
    vdD = 2*np.cos(t)
    vdDD = -2*np.sin(t)
    return [vd, vdD, vdDD]

def deg1(t, x):
    [vd, vdD, vdDD] = generator(t)
    xd = np.array([vd, vdD]).T

    e = xd - x
    ufb = K @ e
    uff = (r*vd + l*vdD + c*l*r*vdDD) / (r*Vin)
    u = ufb + uff

    dx = A @ x + B[:, 0] * u
    return dx

def makeGraph(title, sim):
    zadana = np.zeros_like(sim.t)
    zadana = [generator(t)[0] for t in sim.t]
    zadanaD = np.zeros_like(sim.t)
    zadanaD = [generator(t)[1] for t in sim.t]
    
    plt.figure()
    plt.title(title)
    plt.grid()
    plt.xlabel("t [s]")
    plt.ylabel("U [v]")
    plt.plot(sim.t, sim.y[0], label="$V_o$")
    #plt.plot(sim.t, sim.y[1], label="$x_2$")
    plt.plot(sim.t, zadana, "--", label="$V_d$")
    #plt.plot(sim.t, zadanaD, "--", label="$x_{2D}$")
    plt.legend()

wc = 1
k1 = (c*l*wc**2 - 1) / Vin
k2 = (c*l/Vin) * (2*wc - 1/(c*r))
K = np.array([k1, k2])
res = solve_ivp(deg1, [0,10], [2,0], rtol=1e-10, atol=1e-10)

makeGraph("Zadanie śledzenia trajektorii dla $\omega = 1$", res)

wc = 5
k1 = (c*l*wc**2 - 1) / Vin
k2 = (c*l/Vin) * (2*wc - 1/(c*r))
K = np.array([k1, k2])
res = solve_ivp(deg1, [0,10], [2,0], rtol=1e-10, atol=1e-10)

makeGraph("Zadanie śledzenia trajektorii dla $\omega = 5$", res)

wc = 10
k1 = (c*l*wc**2 - 1) / Vin
k2 = (c*l/Vin) * (2*wc - 1/(c*r))
K = np.array([k1, k2])
res = solve_ivp(deg1, [0,10], [2,0], rtol=1e-10, atol=1e-10)

makeGraph("Zadanie śledzenia trajektorii dla $\omega = 10$", res)

plt.show()