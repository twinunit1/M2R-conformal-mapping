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

def newt(f1, f2, F, val, err=1e-10, n=100):
    #initialisation
    m = n
    df = diff2(f1, f2, val)
    f = np.array([-f1(*val), -f2(*val)]) + np.array(F)
    delta = np.linalg.solve(df, f)
    nval = np.array(val)
    #loop
    while 1 in [abs(a) >= err for a in delta] and m > 0:
        nval = nval + delta
        df = diff2(f1, f2, nval)
        f = np.array([-f1(*nval), -f2(*nval)]) + np.array(F)
        delta = np.linalg.solve(df, f)
        m -= 1
    if m > 0:
        return nval + delta
    else: 
        raise Exception('max iteration reached')
        
def ceff(f1, f2, fd, fval, mind=1, maxd=1000, m=1000, axis=1):
    d = np.linspace(mind,maxd,m)
    pval = []
    for i in d:
        val = fval(i)
        F = fd(i)
        pval.append(-2*np.pi/np.log(newt(f1, f2, F, val)[axis]))
    plt.plot(d,pval)
