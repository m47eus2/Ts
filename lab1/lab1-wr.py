import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def stepResponseWP(G, title, x0, xend):
    t = np.linspace(0,xend,500)
    u = np.ones_like(t)
    plt.figure()
    plt.grid(True)
    plt.title(title)
    plt.xlabel("t [s]")
    plt.ylabel("h(t)")

    for i in x0:
        tout, yout, xout = signal.lsim(G, U=u, T=t, X0=i)
        plt.plot(tout,yout, label=f"x0 = {i}")
    plt.legend()
    #plt.savefig(f"{title}.pdf", dpi=300, bbox_inches='tight')

A = [[-2.0]]
B = [[10]]
C = [[1]]
D = [[0]]
G1_SS = signal.StateSpace(A,B,C,D)

A = [[0,1],[-0.5, 0]]
B = [[0],[2]]
C = [[1,0]]
D = [[0]]
G2_SS = signal.StateSpace(A,B,C,D)

A = [[0,-12.0,0],[0,0,1],[1,-16,-7]]
B = [[6],[0],[-2]]
C = [[0,1,0]]
D = [[0]]
G3_SS = signal.StateSpace(A,B,C,D)

stepResponseWP(G1_SS, "Odpowiedź skokowa G1 dla niezerowych warunków początkowych", [[0],[1],[2]], 3.5)
stepResponseWP(G2_SS, "Odpowiedź skokowa G2 dla niezerowych warunków początkowych", [[0,0],[1,0],[0,2]], 7)
stepResponseWP(G3_SS, "Odpowiedź skokowa G3 dla niezerowych warunków początkowych", [[0,0,0],[0,1,0],[1,0,2]], 5)
plt.show()