import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class ivpSolver():
    def __init__(self, a,b,c):
        self.A = a
        self.B = b
        self.C = c
        
    def deg(self, t, x):
        x = np.array([x]).T
        dx = self.a @ x + self.b @ self.control(t)
        return np.ndarray.tolist(dx.T[0])    

    def plot(self, controls, endt, title, labels):
        plt.figure()
        plt.title(title)
        for j in range(len(controls)):
            self.control = controls[j]
            label = labels[j]
            for i in range(len(self.A)):
                self.a = self.A[i]
                self.b = self.B[i]
                self.c = self.C[i]
                res = solve_ivp(self.deg, [0,endt], [0,0], rtol=1e-10, atol=1e-10)
                y = self.c @ np.array(res.y)
                y = np.ndarray.tolist(y[0].T)
                plt.plot(res.t, y, label=label[i])
        plt.legend()

R1 = 0.2
R2 = 0.5
L = 0.1
C = 0.05

A = np.array([[-R1/L, -1/L],[1/C, -1/(R2*C)]])
B = np.array([[1/L], [0]])
C = np.array([[1, -1/R2]])

def step2(t):
    return np.array([[2]])

def sin102(t):
    return np.array([[2*np.sin(10*t)]])

m = ivpSolver([A],[B],[C])
m.plot([step2, sin102], 2, "Przebiegi czasowe wyjścia dla różnych wymuszeń", [["u = 2*1(t)"],["u = 2sin(10t)"]])

m1 = ivpSolver([A,A],[B,B],[np.array([[1,0]]), np.array([[0,1]])])
m1.plot([step2, sin102], 2, "Przebiegi stanu dla u = 2*1(t) i u=2sin(10t)", [["x1 dla u=2*1(t)","x2 dla u=2*1(t)"],["x1 dla u=2sin(10t)","x2 dla u=2sin(10t)"]])

plt.show()