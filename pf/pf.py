import dual as d
import numpy as np


def C(p, eps=1e-10):
    if np.all(isinstance(p, Dual)):
        q = p.real
    else:
        q = p
    if np.all(np.abs(q)<1) and np.all(np.abs(q)>0):
        n=2
        c = (1-p**2)**2
        while p**(2*n)>=eps:
            c *= (1-p**(2*n))**2
            n +=1
        return c * (1-p**(2*n))**2
    else:
        raise Exception('invalid p')

def P(z,p, eps=1e-10):
    if np.all(isinstance(z, Dual)):
        Z = z.real
    else:
        Z = z
    if np.all(isinstance(p, Dual)):
        q = p.real
    else:
        q = p
    if np.all(np.abs(q)<1) and np.all(np.abs(q)>0):
        n = 2
        a = (1-z*p**2)*(1-(p**2)/z)
        while np.any(np.abs((Z+1/Z)*q**(2*n)-q**(4*n))>=eps):
            a *= (1-z*p**(2*n))*(1-(p**(2*n))/z)
            n +=1
        return (1-z) * a * (1-z*p**(2*n))*(1-(p**(2*n))/z)
    else:
        raise Exception('invalid p')
    
def wA(z,a,p, eps=1e-10):
    return -a*P(z/a,p,eps**2)/C(p, eps**2)


def C2(p, eps=1e-10):
    if isinstance(p, Dual):
        q = p.real
    else:
        q = p
    if 0<abs(q)<1:
        n=2
        c = (1-p**2)**2
        while p**(2*n)>=eps:
            c *= (1-p**(2*n))**2
            n +=1
        return c*(1-p**(2*n))**2
    else:
        raise Exception('invalid p')

def P2(z,p, eps=1e-10):
    if isinstance(z, d.Dual):
        Z = z.real
    else:
        Z = z
    if isinstance(p, Dual):
        q = p.real
    else:
        q = p
    if 0<abs(q)<1:
        n=2
        a = (1-z*p**2)*(1-(p**2)/z)
        while np.abs((Z+1/Z)*p**(2*n)-p**(4*n))>=eps:
            a *= (1-z*p**(2*n))*(1-(p**(2*n))/z)
            n +=1
        return (1-z)*a*(1-z*p**(2*n))*(1-(p**(2*n))/z)
    else:
        raise Exception('invalid p')

def wA2(z,a,p, eps=1e-10):
    return -a*P2(z/a,p,eps**2)/C2(p, eps**2)
