import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Global variables
dx = 0.1
dt = 0.1 * dx
N = 100
x0 = [0, 0]           # Initial x values
t0 = [0, 1]           # Initial t values

def psi():
    return


# Solves the Schrodinger eq'n
def schrod():
    return
#    sc.integrate.odeint(psi, )


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
    

    print(matrix)



# The main function in the program
def main():
    matrix = np.zeros(shape = (N,N), dtype = complex)
    v = 0

    trivial(matrix, v)




main()
