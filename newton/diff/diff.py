import dual as d
import numpy as np
import matplotlib.pyplot as plt


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
        
def ceff(f1, f2, maxd):
    D = np.arange(1,1000,1)
    pval = []

    for i in range(len(D)):
        val = [D[i],1/2]
        F = [D[i], D[i]+1]
        pval.append(-2*np.pi/np.log(newt(f1, f2, F, val)[1]))
    plt.plot(D,pval)
