from qutip import basis, Qobj, propagator, num, position
from scipy.special import eval_genlaguerre, gammaln
import numpy as np


class QutipHarmonicOscillator:
    
    def __init__(self, n_max, V_periodic = None, V_anharmonic = None):
        self.dim = n_max + 1
        self.hamiltonian_0 = num(self.dim) + 0.5
        self.perturbation = []
        self.args = {}
        if V_periodic is not None:
            # same (omega, A) order as HarmonicOscillator._periodic_perturbation
            self.args["omega"], self.args["A"], k = V_periodic
            self.perturbation += [[self._periodic_operator(k), "A * cos(omega * t)"]]
        if V_anharmonic is not None:
            self.hamiltonian_0 += V_anharmonic * self._anharmonic_operator() 
        
    def _periodic_operator(self, k = 1):
        if k <= 0:
            raise ValueError(f"k must be positive, but {k} < 0 is given.")
        N = self.dim
        m, n = np.indices((N, N))                       
        m, n = np.maximum(m, n), np.minimum(m, n)       
        d = m - n
        V = ((-1.0) ** ((d - 1) // 2)
            * np.exp(0.5 * (gammaln(n + 1) - gammaln(m + 1)) + d * np.log(k / np.sqrt(2)) - k**2 / 4)
            * eval_genlaguerre(n, d, k**2 / 2))
        V = np.where(d % 2 == 1, V, 0)                  
        return Qobj(V)
    
    def _anharmonic_operator(self): 
        return Qobj((position(self.dim + 2)**4).full()[:self.dim, :self.dim])
    
    def initial_state(self, level: int):
        return basis(self.dim, level)
    
    def time_evolution(self, tlist, **options):
        
        hamiltonian = [self.hamiltonian_0] + self.perturbation
        return propagator(hamiltonian, tlist, args = self.args, options=options)