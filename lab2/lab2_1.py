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

    def plot(self, contr, endt, title, labels):
        self.control = contr
        plt.figure()
        plt.grid()
        plt.title(title)

        for i in range(len(self.A)):
            self.a = self.A[i]
            self.b = self.B[i]
            self.c = self.C[i]
            res = solve_ivp(self.deg, [0,endt], [0,0], rtol=1e-10, atol=1e-10)
            y = self.c @ np.array(res.y)
            y = np.ndarray.tolist(y[0].T)
            plt.plot(res.t, y, label=labels[i])
        plt.legend()
        plt.savefig(f"{title}.pdf", dpi=300, bbox_inches='tight')


m=1
k=1
b = [0, 0.5, 2]

A1 = np.array([[0,1],[-k/m,-b[0]/m]])
A2 = np.array([[0,1],[-k/m,-b[1]/m]])
A3 = np.array([[0,1],[-k/m,-b[2]/m]])
B = np.array([[0],[1/m]])
C = np.array([[1,0]])

def step(t):
    return np.array([[1]])

def sin2t(t):
    return np.array([[np.sin(2*t)]])

m = ivpSolver([A1,A2,A3], [B,B,B], [C,C,C])
m.plot(step, 20, "Odpowiedź czasowa na u=1(t) przy różnych wartościach tłumienia", ["b = 0","b = 0.5","b = 2"])
m.plot(sin2t, 20, "Odpowiedź czasowa na u=sin(2t) przy różnych wartościach tłumienia", ["b = 0","b = 0.5","b = 2"])

plt.show()