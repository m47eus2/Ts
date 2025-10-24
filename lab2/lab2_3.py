import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class ivpSolver():
    def __init__(self, m, l, J, b):
        self.m = m
        self.l = l
        self.J = J
        self.b = b
        self.g = 9.81

    def deg(self, t, x):
        a1 = self.b/((self.m*self.l**2)+self.J)
        a2 = (self.m*self.g*self.l)/((self.m*self.l**2)+self.J)
        dx1 = x[1]
        dx2 = -a1*x[1] - a2*np.sin(x[0]) + self.control(t)
        return [dx1, dx2]

    def plot(self, control, x0, endt, title, labels):
        self.control = control
        plt.figure()
        plt.title(title)
        res = solve_ivp(self.deg, [0,endt], [x0,0], rtol=1e-10, atol=1e-10)
        y1 = np.sin(res.y[0])
        y2 = -np.cos(res.y[0])
        plt.plot(res.t, y1, label=labels[0])
        plt.plot(res.t, y2, label=labels[1])
        plt.legend()

m=1
l=0.5
J=0.05
b=[0,0.1,0.5]

def control(t):
    return 0
def control1(t):
    return 0.1*np.sin(2*np.pi*2*t)
def control2(t):
    return 0.1*np.sin(2*np.pi*0.65*t)
def control3(t):
    return 0.1*np.sin(2*np.pi*0.2*t)

m1 = ivpSolver(m,l,J,b[0])
m2 = ivpSolver(m,l,J,b[1])
m3 = ivpSolver(m,l,J,b[2])

m1.plot(control,np.pi/2, 20, "Odpowiedź czasowa dla b=0, u=0", ["x","y"])
m2.plot(control,np.pi/2, 20, "Odpowiedź czasowa dla b=0.1, u=0", ["x","y"])
m3.plot(control,np.pi/2, 20, "Odpowiedź czasowa dla b=0.5, u=0", ["x","y"])

m2.plot(control1, 0, 60, "Odpowiedź czasowa dla b=0.1, u=0.1sin(2*pi*2t)", ["x","y"])
m2.plot(control2, 0, 60, "Odpowiedź czasowa dla b=0.1, u=0.1sin(2*pi*0.65t)", ["x","y"])
m2.plot(control3, 0, 60, "Odpowiedź czasowa dla b=0.1, u=0.1sin(2*pi*0.2t)", ["x","y"])

plt.show()