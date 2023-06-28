import dual as d
import numpy as np


def C(p, eps=1e-10, t=100):  #C-value in wA
    '''
        Approximates C in wA. 
        
            Parameter: 
                p : p-value (s) 
                *eps (float): Desired error bound 
                *t (int):  Max number of iteration

            Return: 
                estimate (float/dual/array): Estimate of C at p with the same type as p 
    '''
    m=t
    if np.all(isinstance(p, Dual)):
        q = p.real
    else:
        q = p
    if np.all(np.abs(q)<1):
        n=2
        c = (1-p**2)**2
        while p**(2*n)>=eps and m>0:
            c *= (1-p**(2*n))**2
            n += 1
            m -= 1
        if m>0: 
            return c*(1-p**(2*n))**2
        else:
            raise Exception('Max iteration for C reached')
    else:
        raise Exception('invalid p')

def P(z,p, eps=1e-10, t=100):  #P-value in wA
    '''
    Approximates P in wA. 

        Parameters: 
            z : z-value(s) 
            p : p-value(s) 
            *eps (float): Desired error bound 
            *t (int):  Max number of iteration

        Return:  
            estimate (float/dual/array): Estimate of P at z, p with the same type as z and p 
    '''
    m = t
    if np.all(isinstance(z, Dual)):
        Z = z.real
    else:
        Z = z
    if np.all(isinstance(p, Dual)):
        q = p.real
    else:
        q = p
    if np.any(Z==0):
        if np.all(isinstance(Z,np.ndarray)):
            raise Exception('z=0 unsupported for array')  
        elif q==0:
            return 1 - z
        else:
            raise Exception('z=0 is invalid') 
    if np.all(np.abs(q)<1):
        n = 2
        a = (1-z*p**2)*(1-(p**2)/z)
        while np.any(np.abs((Z+1/Z)*q**(2*n)-q**(4*n))>=eps) and m>0:
            a *= (1-z*p**(2*n))*(1-(p**(2*n))/z)
            n += 1
            m -= 1
        if m>0: 
            return (1-z)*a*(1-z*p**(2*n))*(1-(p**(2*n))/z)
        else:
            raise Exception('Max iteration for P reached')
    else:
        raise Exception('invalid p')
    
def wA(z,a,p, eps=1e-10, t=100):  #prime function of annulus
    '''
    Approximates wA. 

        Parameters: 
            z : z-value(s) 
            a : a-value(s) 
            p : p-value(s) 
            *eps (float): Desired error bound 
            *t (int):  Max number of iteration

        Return:  
            estimate (float/dual/array): Estimate of P at z, a, p with the same type as z, a and p 
    '''
    return -a*P(z/a, p, eps**2, t=t)/C(p, eps**2, t=t)
    
def K(z, a, p, eps=1e-10, t=1000): #partial derivative of wA w.r.t z
    '''
        Approximates K. 
        
            Parameter: 
                z : z-value(s) 
                a : a-value(s) 
                p : p-value (s) 
                *eps (float): Desired error bound 
                *t (int):  Max number of iteration

            Return: 
                estimate (float/dual/array): Estimate of K
    '''
    m = t
    if isinstance(z, Dual):
        Z = z.real
    else:
        Z = z
    if isinstance(p, Dual):
        q = p.real
    else:
        q = p
    if isinstance(a, Dual):
        A = a.real
    else:
        A = a
    if q > 1:
        raise Exception("Invalid p")
    b =  -(z / a) / (1 - z / a) + (p ** 2) * (-z / a + a / z) / (1 - (p ** 2) * (z / a + a / z - (p ** 2)))
    n = 2
    while np.abs((q ** (2 * n)) * (-Z / A + A / Z) / (
            1 - (q ** (2 * n)) * (Z / A + A / Z - (q ** (2 * n))))) >= eps and m > 0:
        b += (p ** (2 * n)) * (-z / a + a / z) / (1 - (p ** (2 * n)) * (z / a + a / z - (p ** (2 * n))))
        n += 1
        m -= 1
    if m > 0:
        return b + (p ** (2 * n)) * (-z / a + a / z) / (1 - (p ** (2 * n)) * (z / a + a / z - (p ** (2 * n))))
    else:
        raise Exception("Max iteration for K reached")

