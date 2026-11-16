import numpy as np
from scipy.special import eval_genlaguerre, gammaln

def _sin_x(m, n):
    d = m - n
    return ((-1) ** ((d - 1) // 2)
        * np.exp(0.5 * (gammaln(n + 1) - gammaln(m + 1)) + d * np.log(1 / np.sqrt(2)) - 1**2 / 4)
        * eval_genlaguerre(n, d, 1 / 2))
    
def ft(detune, t):
    if detune == 0:
        return t**2/4
    else:
        return (np.sin(detune/2 * t)/detune)**2    
        
def population_first_order(ni, nf, omega, t):
    omega_fi = nf - ni
    detune = omega_fi - omega
    return _sin_x(nf, ni)**2 * ft(detune)

def population_second_order(ni, nf, omega, t, cutoff=10):
    detune = (nf - ni) - omega
    sum = np.sum([_sin_x(nf, k)**2 * _sin_x(nf, k)**2 / (k - ni)
                  for k in range(0,cutoff+1)])
    return sum * ft(detune)