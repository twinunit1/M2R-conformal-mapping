import dual as d
import numpy as np


def C(p, eps=1e-10):  #C-value in wA
    '''
        Approximates C in wA. 
        
            Parameter: 
                p : p-value (s) 
                *eps (float): Desired error bound 

            Return: 
                estimate (float/dual/array): Estimate of C at p with the same type as p 
    '''
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
        return c*(1-p**(2*n))**2
    else:
        raise Exception('invalid p')

def P(z,p, eps=1e-10):  #P-value in wA
    '''
    Approximates P in wA. 

        Parameters: 
            z : z-value(s) 
            p : p-value(s) 
            *eps (float): Desired error bound 

        Return:  
            estimate (float/dual/array): Estimate of P at z, p with the same type as z and p 
    '''
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
        while np.any(np.abs((Z+1/Z)*q**(2*n)-q**(4*n))>=eps):
            a *= (1-z*p**(2*n))*(1-(p**(2*n))/z)
            n +=1
        return (1-z)*a*(1-z*p**(2*n))*(1-(p**(2*n))/z)
    else:
        raise Exception('invalid p')
    
def wA(z,a,p, eps=1e-10):  #prime function of annulus
    '''
    Approximates wA. 

        Parameters: 
            z : z-value(s) 
            a : a-value(s) 
            p : p-value(s) 
            *eps (float): Desired error bound 

        Return:  
            estimate (float/dual/array): Estimate of P at z, a, p with the same type as z, a and p 
    '''
    return -a*P(z/a,p,eps**2)/C(p, eps**2)
