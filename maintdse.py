import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

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

    psi0, psi1 = psi1, psi0



# The main function in the program
def main():
    matrix = np.zeros(shape = (N,N), dtype = complex)
    v = 0
    trivial(matrix, v)
    

    
plt.clf()
fig, ax = plt.subplots(1,1,figsize=(12,4))
real_part, = ax.plot(x,np.real(psi0))
imag_part, = ax.plot(x,np.imag(psi0))


def update(n):
    if(n>1):
        global psi0, psi1
        main()
        real_part.set_data(x,np.real(psi0))
        imag_part.set_data(x,np.imag(psi0))
        return fig
    
anim = animation.FuncAnimation(fig, update,frames=200, interval=50, blit=False)
video = anim.to_html5_video()
HTML(video)


