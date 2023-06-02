import dual as d
import numpy as np
import matplotlib.pyplot as plt


def diff1(f1, val):
    x = f1(Dual(real=val, dual={'1': 1}))
    return x.dual['1']


def diff2(f1, f2, val, method='l'):  #Jacobean matrix, set method='a' if val is an array
    if method == 'l':
        n = len(val)
        x = []
        for k in range(n):
            x += [Dual(real=val[k], dual={f'x{k}': 1})]
        return np.array([[f1(*x).dual[f'x{k}'] for k in range(n)],
                         [f2(*x).dual[f'x{k}'] for k in range(n)]])
    elif method == 'a':
        m, n = np.shape(val)
        x = []
        for k in range(n):
            x += [Dual(real=0, dual={f'x{k}': 1})]
        xval = val + np.tile(x,(m,1))
        return np.array([[[f1(*xval[i]).dual[f'x{k}'] for k in range(n)], [f2(*xval[i]).dual[f'x{k}'] for k in range(n)]] for i in range(m)])
    else:
        raise Exception('invalid method')


def newt1(f1, F, val, eps=1e-10, n=50): 
    '''
        One-dimension Newtons method for solving f1(x)=F. 

        Parameters: 
            f1 (function): Function of equation 
            F (float/array): Desired output(s) of f1 
            val (float/array): Point(s) of initial guess 
            *eps (float): Desired error bound 

        Return:  
            estimate (float/array): Estimate of solution(s) with the same type as val 
    '''
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

def newt2(f1, f2, F, val, eps=1e-10, n=100, method='l'):  
    '''
        Two-dimension Newtons method for solving f1(x)=F1 and f2(x)=F2. 

        Parameters: 
            f1 (function): First function 
            f2 (function): Second function 
            F (list/array): Desired output(s) of f1, f2 in form of [F1, F2]     
            val (list/array): Point(s) of initial guess in form of [val1, val2] 
            *eps (float): Desired error bound 
            *method (string): if method = 'a' then F and val must be arrays 

        Return:  
            estimate (array): Estimate of solution(s) 
    '''
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
        raise Exception('invalid method')
    

def cont(f1, f2, d2, val, d1=1, F0=[0,1], n=10, m=10): 
    '''
        Continuation method for solving f1(x)=d1 and f2(x)=d2. 

        Parameters: 
            f1 (function): First function 
            f2 (function): Second function 
            d2 (float): Desired output of f2 
            val (list): Point of initial guess in form of [val1, val2] 
            *d1 (float): Desired output of f1 
            *F0 (list): Output of initial guess in form of [f1(val), f2(val)] 
            *n (int): Number of iterations to reach d1 
            *m (int): Number of iterations to reach d2 

        Return:  
        estimate (float): Estimate of solution 
    '''
    s = np.linspace(((n-1)*F0[0]+d1)/n,d1,n)
    x = val
    for i in s:
        F = [i, F0[1]]
        x = newt2(f1, f2, F, x)

    t = np.linspace((F0[1]+d2)/m, d, m)
    for i in t:
        F = [d1, i]
        x = newt2(f1, f2, F, x)

    return x
