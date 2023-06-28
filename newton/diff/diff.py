import dual as d
import numpy as np
import matplotlib.pyplot as plt


def diff1(f1, val):
    '''
        Differentiates f1 at val. 
        
            Parameter: 
                f1 (function): Function to differentiate  
                val : Point(s) of evaluation 
            
            Return: 
                derivative (float/array): Derivative of f1 at val with the same type as val 
    '''
    x = f1(Dual(real=val, dual={'1': 1}))
    return x.dual['1']

def diff2(f1, f2, val, method='l'):  #Jacobean matrix, set method='a' if val is an array
    '''
        Determine the Jacobian matrix of f1 and f2. 

            Parameters: 
                f1 (function): First function 
                f2 (function): Second function 
                val : Point(s) of evaluation 
                *method (string): Set method = 'a' if val is an array 

            Return:  
                matrix(es) (array): Jacobian matrix(es) 
    '''
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
            F : Desired output(s) of f1 
            val : Point(s) of initial guess 
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
    while np.any(abs(delta)>=eps) and m>0:
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
            F : Desired output(s) of f1, f2 in form of [F1, F2]     
            val : Point(s) of initial guess in form of [val1, val2] 
            *eps (float): Desired error bound 
            *method (string): if method = 'a' then F and val must be arrays 

        Return:  
            estimate (array): Estimate of solution(s) 
    '''
    m = n
    if method == 'l':
        #initialisation
        df = diff2(f1, f2, val).real
        f = np.array([-f1(*val).real, -f2(*val).real])+np.array(F)
        delta = np.linalg.solve(df, f)
        nval = np.array(val)
        #loop
        while np.any(abs(delta)>=eps) and m>0:
            nval = nval + delta
            df = diff2(f1, f2, nval).real
            f = np.array([-f1(*nval).real, -f2(*nval).real])+np.array(F)
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
        while np.any(abs(delta)>=eps) and m>0:
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
        Continuation method for solving f1(x)=d1 and f2(x)=d2 using square path along d2 then d1 

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
    s = np.linspace(((m-1)*F0[1]+d2)/m,d2,m)
    x = val
    for i in s:
        F = [F0[0], i]
        x = newt2(f1, f2, F, x)
    t = np.linspace(((n-1)*F0[0]+d1)/n, d1, n)
    for i in t:
        F = [i, d2]
        x = newt2(f1, f2, F, x)
    return x

def cont2(f1, f2, d2, val, d1=1, n=100): 
    '''
        Continuation method for solving f1(x)=d1 and f2(x)=d2 using straight line path

        Parameters: 
            f1 (function): First function 
            f2 (function): Second function 
            d2 : Desired output of f2 
            val : Point of initial guess in form of [val1, val2] 
            *d1 : Desired output of f1 
            *F0 : Output of initial guess in form of [f1(val), f2(val)] 
            *n (int): Number of iterations

        Return:  
        estimate (float): Estimate of solution 
    '''
    F0 = [f1(*val),f2(*val)]
    s = np.linspace(((n-1)*F0[1]+d2)/n, d2, n)
    t = np.linspace(((n-1)*F0[0]+d1)/n, d1, n)
    x = val
    for i in range(n):
        F = [t[i], s[i]]
        x = newt2(f1, f2, F, x)
    return x
