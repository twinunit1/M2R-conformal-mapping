import matplotlib.pyplot as plt
import numpy as np


def hplot(h, n=30, m = 500, xbound = [-2,2], ybound = [-2,2], figsize= 10, ax=True, shift=False, theta=np.pi/2, s=1/np.log(p)):
    '''
        Generates plot of feild lines from h

        Parameters:
            h (function): h of desired graph
            n (int): Number of field lines
            *xbound (list): Boundary of x in form of [xmin, xmax]
            *ybound (list): Boundary of x in form of [xmin, xmax]
            *figsize (float): Desired size of plot
            *ax (bool): Hides axis if False
            *shift (bool): Shifts the branch cut of log if True
            *theta (float): Argument of new branch cut
            *s (float): Scaling factor of Log(z)
    
        Return 
            graph (plot): Plot of field lines
    '''

    x = np.linspace(xbound[0],xbound[1],m )
    y = np.linspace(ybound[0],ybound[1],m)
    xval, yval = np.meshgrid(x,y)
    z = xval + yval * 1j
    f = h(z)
    Re = f.real
    Im = f.imag
    Im = Im.astype(float)

    Im = np.where(Re<=1, Im, np.nan)
    Im = np.where(Re>=0, Im, np.nan)

    if shift:
        if -np.pi<theta<np.pi:
            if theta>0:
                Im = np.where(Im>=theta/s, Im-2*np.pi*np.abs(s), Im)
            else:
                Im = np.where(Im<=theta/s, Im+2*np.pi*np.abs(s), Im)
        else:
            raise Exception('invalid theta')

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

def wplot(w, n, p, m=0, xbound = [-2,2], ybound = [-2,2], figsize=10, kn=1000, km=1000, ax=True):
    '''
        Generates plot of field lines from transformation of annulus.

            Parameter:
                w (function): Transformation function
                n (int): Number of electric field lines
                p (float): p-value of annulus
                *m (int): Number of potential lines other than the two electrode lines
                *xbound (list): Boundary of x in form of [xmin, xmax]
                *ybound (list): Boundary of x in form of [xmin, xmax]
                *figsize (float): Desired size of the plot
                *kn (int): Number of samples used for field lines (for smoothness)
                *km (int): Number of samples used for potential lines (for smoothness)
                *ax (bool): Display axis if True
    
            Return:
                graph (plot): Plot of feild lines
    '''
    t = np.linspace(0,2*np.pi,n, endpoint=False)
    r = np.linspace(p,1,kn)
    plt.figure(figsize=[figsize,figsize])
    for i in t:
        a = r*np.cos(i)+r*np.sin(i)*1j
        k=w(a)
        plt.plot(k.real,k.imag, color='black')

    t2 = np.linspace(0,2*np.pi,km)
    r2 = np.linspace(p,1,m+2)
    color = ['red']+m*['black']+['blue']
    for i in range(m+2):
        a = r2[i]*np.cos(t2)+r2[i]*np.sin(t2)*1j
        k=w(a)
        plt.plot(k.real,k.imag, color=color[i])

    plt.axis('scaled')
    plt.axis(xbound+ybound)
    if not ax:
        plt.axis('off')
        
def ceffn(f1, f2, fd = lambda D : [D,D+1], fval = lambda D : [D, 1/2+0*D], mind=1, maxd=10000, m=1000, method='a'):
    '''
        Generates graph of field ceff over d using newt2 to solve f1(R,p)=fd1(d) and f2(R,p)=fd2(d) 

            Parameter: 
                f1 (function): First function 
            f2 (function): Second function 
            *fd (function): Function of d in form of [fd1, fd2] 
            *fval (function): Function of d for initial guess in form of [fval1, fval2] 
            *mind (float): Minimum d of graph 
            *maxd (float): Maximum d of graph 
            *m (int): Number of steps between mind and maxd 
            *method (string): if method = 'l' then list would be used instead of array 
        **note: fd1, fd2, fval1, fval2 must all contain variable d i.e. 0*d 

        Return: 
            graph (plot): Plot of ceff over d 
    '''
    if method == 'l':
        d = np.linspace(mind,maxd,m)
        pval = []
        for i in d:
            val = fval(i)
            F = fd(i)
            pval.append(-2*np.pi/np.log(newt2(f1, f2, F, val)[1]))
        plt.plot(d,pval)
    elif method == 'a':
        d = np.linspace(mind,maxd,m)
        val = np.array(fval(d)).T
        F = np.array(fd(d)).T
        plt.plot(d,-2*np.pi/np.log(newt2(f1, f2, F, val, method='a').T[1]))
    else:
        raise Exception('invalid method')
    
def ceffc(f1, f2, val, d1=1, F0=[0,1], n=2,m=2,mind=1, maxd=100, k=100):
    '''
        Generates graph of field ceff over d using cont2 to solve f1(R,p)=d1 and f2(R,p)=d 

        Parameter: 
            f1 (function): First function 
            f2 (function): Second function 
            val (list): Point of initial guess in form of [val1, val2] 
            *d1 (float): Desired output of f1 
            *F0 (list): Output of initial guess in form of [f1(val), f2(val)] 
            *n (int): Number of iterations to reach d1 
            *m (int): Number of iterations to in between the d values 
            *mind (float): Minimum d of graph 
            *maxd (float): Maximum d of graph 
            *k (int): Number of steps between mind and maxd 

        Return: 
            graph (plot): Plot of ceff over d 
    '''
    dval = np.linspace(mind,maxd,k)
    s = np.linspace(((n-1)*F0[0]+d1)/n,d1,n)
    x = val
    pval = []
    for i in s:
        F = [i, F0[1]]
        x = newt2(f1, f2, F, x)
    if mind==F0[1]:
        pval += [x[1]]
    else:
        t = np.linspace(((m-1)*F0[1]+mind)/m, mind, m)
        for j in t:
            F = [1, j]
            x = newt2(f1, f2, F, x)  
        pval += [x[1]]
    
    for i in range(k-1):
        t = np.linspace(((m-1)*dval[i]+dval[i+1])/m, dval[i+1], m)
        for j in t:
            F = [1, j]
            x = newt2(f1, f2, F, x)  
        pval += [x[1]]
    plt.plot(dval, -2*np.pi/np.log(np.array(pval)))
