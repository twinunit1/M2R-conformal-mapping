import matplotlib.pyplot as plt
import numpy as np

def splt(h, n=10, xbound = [-10,10], ybound = [-10,10], m = 5000):
    x = np.linspace(xbound[0],xbound[1],m)
    y = np.linspace(xbound[0],ybound[1],m)
    xval, yval = np.meshgrid(x,y)
    z = xval .+ yval * 1j
    f = h(z)
    Re = f.real
    Im = f.imag
    plt.contour(xval, yval, Im, [2*n], colors='black')
    plt.contour(xval, yval, Re, [0:1:1], colors=['red','blue'])
    plt.axis(xbound+ybound)
 

