import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class ivpSolver():
    def __init__(self,a,b,c,w):
        b = 0.5
        k = 1
        m = 1

        self.A = np.array([[0, 1],
              [-k/m, -b/m]])

        self.B = np.array([[0],
              [1/m]])

        self.C = np.array([[1, 0]])

        l1 = 2*w - b/m
        l2 = w*w - 2*w*b/m + (b*b)/(m*m) - k/m

        self.L = np.array([[l1], [l2]])

    def u(self, t):
        return np.array([[np.sin(np.pi*t)]])
    
    def deg1(self, t, q):
        x = np.array([q]).T[0:2];
        xe = np.array([q]).T[2:4];
        
        y = self.C@x
        
        dx = self.A @ x + self.B @ self.u(t); #calculate state equation
        dxe = self.A @ xe + self.B @ self.u(t) + self.L @ (y - self.C @ xe)

    #note: matrix multiplication is done using @ operator
    
        return np.ndarray.tolist(np.hstack((dx.T[0], dxe.T[0])))
    
    def plot(self):
        res = solve_ivp(self.deg1, [0,10], [2,0,0], rtol=1e-8)

        plt.figure()
        plt.grid()
        plt.plot(res.t, res.y[0],"$x_1$")
        plt.plot(res.t, res.y[2],"$\hat{x}_1$")
        plt.legend()

        plt.figure()
        plt.grid()
        plt.plot(res.t, res.y[1],"$x_2$")
        plt.plot(res.t, res.y[3],"$\hat{x}_2$")
        plt.legend()


b = 0.5
k = 1
m = 1

#define arrays A, B and C of a linear system
A = np.array([[0, 1],
              [-k/m, -b/m]])

B = np.array([[0],
              [1/m]])

C = np.array([[1, 0]])


m = ivpSolver(A,B,C,1)
m.plot()