import numpy as np
from scipy.special import eval_genlaguerre, gammaln

def _sin_x(m, n):
    d = m - n
    return ((-1) ** ((d - 1) // 2)
        * np.exp(0.5 * (gammaln(n + 1) - gammaln(m + 1)) + d * np.log(1 / np.sqrt(2)) - 1**2 / 4)
        * eval_genlaguerre(n, d, 1 / 2))
    
def _sinc(detune, t):
    # integral_0^t exp(i*detune*t') dt'
    if detune == 0:
        return t
    else:
        return (np.exp(1j*detune*t) - 1)/(1j*detune)

def population_first_order(ni, nf, omega, t, A, rwa=True):
    # H' = A sin(x) cos(omega t)
    omega_fi = nf - ni
    amp = _sinc(omega_fi - omega, t)
    if not rwa:
        amp = amp + _sinc(omega_fi + omega, t)
    return (A * _sin_x(nf, ni) / 2)**2 * np.abs(amp)**2

def population_second_order_rwa(ni, nf, omega, cutoff=10):
    detune = (nf - ni) - omega
    sum = np.sum([_sin_x(nf, k)**2 * _sin_x(nf, k)**2 / (k - ni)
                  for k in range(0,cutoff+1)])
    return sum * _sinc(detune)