import dual as d
import numpy as np


def C(p, eps=1e-10):  #C-value in wA
    if np.all(isinstance(p, Dual)):
        q = p.real
    else:
        q = p
    if np.all(np.abs(q)<1):
        n=2
        c = (1-p**2)**2
        while p**(2*n)>=eps:
            c *= (1-p**(2*n))**2
            n +=1
        return c
    else:
        raise Exception('invalid p')

def P(z,p, eps=1e-10):  #P-value in wA
    zval = z
    if np.all(isinstance(z, Dual)):
        Z = zval.real
    else:
        Z = zval
    if np.all(isinstance(p, Dual)):
        q = p.real
    else:
        q = p
    if np.any(Z==0):
        if type(Z,np.ndarray):
            raise Exception('z=0 unsupported for array')  
        elif q==0:
            return 1
        else:
            raise Exception('z=0 is invalid') 
    if np.all(np.abs(q)<1):
        n = 2
        a = (1-zval*p**2)*(1-(p**2)/zval)
        while np.any(np.abs((Z+1/Z)*q**(2*n)-q**(4*n))>=eps):
            a *= (1-zval*p**(2*n))*(1-(p**(2*n))/zval)
            n +=1
        return (1-z)*a
    else:
        raise Exception('invalid p')
    
def wA(z,a,p, eps=1e-10):  #prime function of annulus
    return -a*P(z/a,p,eps**2)/C(p, eps**2)
