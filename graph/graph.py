import matplotlib.pyplot as plt
import numpy as np


def splt(h, n = 16, m = 500, sing = True, a = 1, xbound = [-2,2], ybound = [-2,2], figsize = 10, ax = True):
    x = np.linspace(xbound[0], xbound[1], m)
    y = np.linspace(ybound[0], ybound[1], m)
    xval, yval = np.meshgrid(x, y)
    z = xval + yval * 1j
    f = h(z)
    Re = f.real
    Im = f.imag
    Im = Im.astype(float)
    
    Im = np.where(Re <= 1, Im, np.nan)
    Im = np.where(Re >= 0, Im, np.nan)
    
    if not sing:
        Im = np.where(Im < np.nanmax(Im) - 10 ** (-a), Im, np.nan)
        Im = np.where(Im > np.nanmin(Im) + 10 ** (-a), Im, np.nan)

    plt.figure(figsize = [figsize, figsize])
    plt.contour(xval, yval, Im, n-1, colors = 'black', linestyles = 'solid')
    plt.contour(xval, yval, Re, [0,1], colors=['blue', 'red'])
    plt.axis('scaled')
    plt.axis(xbound + ybound)
    if not ax:
        plt.axis('off')
        
def splt2(h, n=10, m = 500, xbound = [-2,2], ybound = [-2,2], figsize= 10, ax=True):
    x = np.linspace(xbound[0],xbound[1],m)
    y = np.linspace(ybound[0],ybound[1],m)
    xval, yval = np.meshgrid(x,y)
    z = xval + yval * 1j
    f = h(z)
    Re = f.real
    Im = f.imag
    Im = Im.astype(float)

    Im = np.where(Re<=1, Im, np.nan)
    Im = np.where(Re>=0, Im, np.nan)

    imax = np.nanmax(Im)
    imin = np.nanmin(Im)
    b = (imax-imin)/(2*n)
    
    Im = np.where(Im<imax-b/4, Im, np.nan)
    Im = np.where(Im>imin+b/4, Im, np.nan)

    plt.figure(figsize=[figsize,figsize])
    plt.contour(xval, yval, Im, np.linspace(imin+b,imax-b,n), colors='black', linestyles='solid')
    plt.contour(xval, yval, Re, [0,1], colors=['blue', 'red'])
    plt.axis('scaled')
    plt.axis(xbound+ybound)
    if not ax:
        plt.axis('off')
