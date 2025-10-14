import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def stepResponse(G, title):
    t,y = signal.step(G)
    plt.figure()
    plt.grid(True)
    plt.title(title)
    plt.xlabel("t [s]")
    plt.ylabel("h(t)")
    plt.plot(t,y)

#Definiowanie obiektów za pomocą transmitancji
G1 = signal.TransferFunction(10, [1, 2])
G2 = signal.TransferFunction(4,[2,0,1])
G3 = signal.TransferFunction([-2,6],np.polymul(np.polymul([1,2],[1,2]),[1,3]))

#Definiowanie obiektów za pomocą równań stanu
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

#Przekształcenie transmitancji do przestzeni stanu
A,B,C,D = signal.tf2ss(10, [1, 2])
G1_TF2SS = signal.StateSpace(A,B,C,D)

A,B,C,D = signal.tf2ss(4,[2,0,1])
G2_TF2SS = signal.StateSpace(A,B,C,D)

A,B,C,D = signal.tf2ss([-2,6],np.polymul(np.polymul([1,2],[1,2]),[1,3]))
G3_TF2SS = signal.StateSpace(A,B,C,D)

#Wyświetlenie odpowiedzi skokowych

# stepResponse(G1, "Odpowiedź skokowa G1")
# stepResponse(G1_SS, "Odpowiedź skokowa G1_SS")
# stepResponse(G1_TF2SS, "Odpowiedź skokowa G1_TF2SS")

# stepResponse(G2, "Odpowiedź skokowa G2")
# stepResponse(G2_SS, "Odpowiedź skokowa G2_SS")
# stepResponse(G2_TF2SS, "Odpowiedź skokowa G2_TF2SS")

stepResponse(G3, "Odpowiedź skokowa G3")
stepResponse(G3_SS, "Odpowiedź skokowa G3_SS")
stepResponse(G3_TF2SS, "Odpowiedź skokowa G3_TF2SS")

plt.show()