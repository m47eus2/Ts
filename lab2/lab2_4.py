import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class ivpSolver():
    def __init__(self, r,l,c):
        self.r = r
        self.l = l
        self.c = c

    def deg(self, t, x):
        dx1 = (-self.r/self.l)*x[0] - (1/self.l)*x[1] + (1/self.l)*self.control(t)
        dx2 = (1/self.c)*x[0] - (1/self.c)*((0.25*x[1]) / (5-x[1]))
        return [dx1, dx2]

    def plot(self, control, title, labels):
        self.control = control
        plt.figure()
        plt.title(title)
        res = solve_ivp(self.deg, [0,2], [0,0], rtol=1e-10, atol=1e-10)
        y = (0.25*res.y[1]) / (5-res.y[1])
        plt.plot(res.t, y, label=labels[0])
        plt.plot(res.t, res.y[0], label=labels[1])
        plt.plot(res.t, res.y[1], label=labels[2])
        plt.legend()

R = 0.2
L = 0.1
C = 0.05

def control1(t):
    return 2
def control2(t):
    return -2

m = ivpSolver(R,L,C)
m.plot(control1, "Odpowiedź czasowa oraz przebiegi stanu dla u = 2*1(t)", ["y","x1","x2"])
m.plot(control2, "Odpowiedź czasowa oraz przebiegi stanu dla u = -2*1(t)", ["y","x1","x2"])

plt.show()