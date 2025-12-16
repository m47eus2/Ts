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

wc = -1

k1 = ((wc**2)*c*l-1) / Vin
k2 = ((c*l)/Vin)*(2*wc - 1/c*r)

K = np.array([k1, k2])

vd = 5
xd = np.array([vd, 0]).T

def control(t):
    return np.array([[1]]) #return as an array with single element


def deg1(t, x):
    #x = np.array([x]).T

    e = xd - x
    ufb = K @ e
    ud = vd / Vin

    u = ufb + ud
    
    dx = A @ x + B.flatten() @ u
    
    return np.ndarray.tolist(dx.T[0])

res = solve_ivp(deg1, [0,10], [2,0], rtol=1e-10) #arguments rtol and atol sets calculation tolerance

y = C @ np.array(res.y)
y = np.ndarray.tolist(y[0].T)

plt.figure()
plt.grid()
plt.title("Odpowiedź układu")
plt.plot(res.t, res.y[0], label="x1")
#plt.plot(res.t, res.y[1], label="x2")
plt.legend()
plt.show()