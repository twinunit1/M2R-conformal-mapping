import matplotlib.pyplot as plt
import numpy as np


def splt(h, n=16, xbound = [-2,2], ybound = [-2,2], m = 500):
    x = np.linspace(xbound[0],xbound[1],m)
    y = np.linspace(xbound[0],ybound[1],m)
    xval, yval = np.meshgrid(x,y)
    z = xval + yval * 1j
    f = h(z)
    Re = f.real
    Im = f.imag
    Im = Im.astype(float)
    Im = np.where(Re<=1, Im, np.nan)
    Im = np.where(Re>=0, Im, np.nan)

    plt.axis('scaled')
    plt.contour(xval, yval, Im, n-1, colors='black', linestyles='solid')
    plt.contour(xval, yval, Re, [0,1], colors=['blue', 'red'], )
    plt.axis(xbound+ybound)
    
def splt2(h, cmax=10, n=150, xbound = [-2,2], ybound = [-2,2], figsize=[10,10], m = 500):
    x = np.linspace(xbound[0],xbound[1],m)
    y = np.linspace(xbound[0],ybound[1],m)
    xval, yval = np.meshgrid(x,y)
    z = xval + yval * 1j
    f = h(z)
    Re = f.real
    Im = f.imag
    Im = Im.astype(float)
    Im = np.where(Re<=1, Im, np.nan)
    Im = np.where(Re>=0, Im, np.nan)

    plt.axis('scaled')
    plt.contour(xval, yval, Im, np.linspace(-cmax,cmax,n), colors='black', linestyles='solid')
    plt.contour(xval, yval, Re, [0,1], colors=['blue', 'red'], )
    plt.axis(xbound+ybound)
