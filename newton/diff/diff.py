import dual as d
import numpy as np
import matplotlib.pyplot as plt


def diff1(f1, val):
    x = f1(d.Dual(real=val, dual={'1': 1}))
    return x.dual['1']

def diff2(f1, f2, val):
    if len(val)==2:
        x = []
        for k in range(2):
            x += [d.Dual(real=val[k], dual={f'x{k}': 1})]
        return np.array([[f1(*x).dual[f'x{k}'] for k in range(2)],
                         [f2(*x).dual[f'x{k}'] for k in range(2)]])
    else:
        raise Exception('wrong dimension for val')
        
def newt1(f1, F, val, err=1e-10, n=50):
    #initialisation
    m = n
    df = diff1(f1, val)
    f = F-f1(val)
    delta = f/df
    nval = val
    #loop
    while np.any(abs(delta)>=err) and m>0:
        nval = nval + delta
        df = diff1(f1, nval)
        f = F-f1(nval)
        delta = f/df
        m -= 1
    if m>0:
        return nval + delta
    else: 
        raise Exception('max iteration reached')    

def newt2(f1, f2, F, val, err=1e-10, n=50, method='l'):
    m = n
    if method == 'l':
        #initialisation
        df = diff2(f1, f2, val)
        f = np.array([-f1(*val), -f2(*val)])+np.array(F)
        delta = np.linalg.solve(df, f)
        nval = np.array(val)
        #loop
        while np.any(abs(delta)>=err) and m>0:
            nval = nval + delta
            df = diff2(f1, f2, nval)
            f = np.array([-f1(*nval), -f2(*nval)])+np.array(F)
            delta = np.linalg.solve(df, f)
            m -= 1
        if m>0:
            return nval + delta
        else: 
            raise Exception('max iteration reached')
    elif method == 'a':
        #initialisation
        df = diff2(f1, f2, val, method = 'a')
        f = np.array([-f1(*val.T), -f2(*val.T)]).T + np.array(F)
        delta = np.linalg.solve(df, f)
        nval = np.array(val)
        #loop
        while np.any(abs(delta)>=err) and m>0:
            nval = nval + delta
            df = diff2(f1, f2, nval, method = 'a')
            f = np.array([-f1(*nval.T), -f2(*nval.T)]).T+np.array(F)
            delta = np.linalg.solve(df, f)
            m -= 1
        if m>0:
            return nval + delta
        else: 
            raise Exception('max iteration reached')
    else:
        raise Exeption('invalid method')
        
def ceff(f1, f2, fd = lambda D : [D,D+1], fval = lambda D : [D, 1/2+0*D], mind=1, maxd=10000, m=1000, axis=1, method='l'):
    if method == 'l':
        d = np.linspace(mind,maxd,m)
        pval = []
        for i in d:
            val = fval(i)
            F = fd(i)
            pval.append(-2*np.pi/np.log(newt2(f1, f2, F, val)[axis]))
        plt.plot(d,pval)
    elif method == 'a':
        d = np.linspace(mind,maxd,m)
        val = np.array(fval(d)).T
        F = np.array(fd(d)).T
        plt.plot(d,-2*np.pi/np.log(newt2(f1, f2, F, val, method='a').T[1]))
    else:
        raise Exception('invalid method')
