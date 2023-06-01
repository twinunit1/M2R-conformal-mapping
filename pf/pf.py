import dual as d
import numpy as np


def C(p, eps=1e-20):
    if isinstance(p, Dual):
        q = p.real
    else:
        q = p
    if q!=0:
        n=2
        c = (1-p**2)**2
        while p**(2*n)>=eps:
            c *= (1-p**(2*n))**2
            n +=1
        return c
    else:
        raise Exception('invalid p')

def P(z,p, eps=1e-20):
    if isinstance(z, d.Dual):
        Z = z.real
    else:
        Z = z
    if isinstance(p, Dual):
        q = p.real
    else:
        q = p
    if q!=0:
        n=2
        a = (1-z*p**2)*(1-(p**2)/z)
        while np.abs((Z+1/Z)*p**(2*n)-p**(4*n))>=eps:
            a *= (1-z*p**(2*n))*(1-(p**(2*n))/z)
            n +=1
        return (1-z)*a
    else:
        raise Exception('invalid p')

def wA(z,a,p, eps=1e-20):
    return -a*P(z/a,p,eps**2)/C(p, eps**2)
