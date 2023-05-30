import dual as d
import numpy as np

def diff2(f1, f2, val):
    x = []
    for k in range(len(val)):
        x += [d.Dual(real=val[k], dual={f'x{k}': 1})]
    return np.array([list(f1(*x).dual.values()), list(f2(*x).dual.values())])

def newt(f1, f2, F, err, val=[0,0], n=100):
    #initialisation
    m = n
    df = diff2(f1, f2, val)
    f = np.array([f1(*val), f2(*val)])-np.array(F)
    delta = np.linalg.solve(df, f)
    nval = np.array(val)
    #loop
    while 1 in [a>=err for a in delta] and m>0:
        nval = nval - delta
        df = diff2(f1, f2, nval)
        f = np.array([f1(*nval), f2(*nval)])-np.array(F)
        delta = np.linalg.solve(df, f)
        m -= 1
    if m>0:
        return nval - delta
    else: 
        return 'max iteration reached'