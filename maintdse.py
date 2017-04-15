import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Global variables
dx = 0.1
dt = 0.1 * dx
N = 100

x = np.linspace(0, N*dx, N)
psi0 = np.sin(x)
psi1 = np.zeros(N, dtype = complex)


# The initial, trivial way to go about the problem
def trivial(matrix, v):

    diag_ul = (-1 / dx**2) * (dt / 1j)

    diag = (2/dx**2 - v) * (dt / 1j) - (1 / dt)

    for i in range(N):
        matrix[i][i] = diag
        if i < 99:
            matrix[i][i+1] = diag_ul
        if i > 0:
            matrix[i][i-1] = diag_ul
        if i == 99:
            matrix[i][0] = diag_ul
        if i == 0:
            matrix[i][99] = diag_ul

    psi1 = np.linalg.solve(matrix, psi0)

    plt.plot(x, psi1)
    plt.show()

    print(matrix)



# The main function in the program
def main():
    matrix = np.zeros(shape = (N,N), dtype = complex)

    v = 0

    trivial(matrix, v)




main()
