from collections.abc import Sequence
import numpy as np
from scipy import fft
from scipy import special

class HarmonicOscillator:
    
    def __init__(self, 
                boundary : Sequence[float],
                n : int,
                x0 : float = 0,):
        self.x0 = x0
        self.xlo, self.xhi = boundary
        self.n = n

    def xi(self):
        # Position grid
        L = self.xhi - self.xlo
        dx = L/self.n
        return np.array([self.xlo + i * dx for i in range(self.n)])

    def pi(self):
        # Momentum grid, uses fftfreq to fix the disrepancy 
        # that half of momentum grid should be at negative side
        L = self.xhi - self.xlo
        dx = L/self.n
        # k = 2 pi f
        return 2 * np.pi * fft.fftfreq(self.n, d=dx)
    
    def _T(self, momentum = True):
        if momentum:
            # Kinetic energy p^2/2
            return np.diag(np.square(self.pi())/ 2)
        else:
            return 0
    
    def _V(self):
        # Harmonic potential x^2/2
        return np.diag(np.square(self.xi())/2)
    
    def _periodic_perturbation(self, omega, A):
        def func(t):
            return np.diag(
                A *np.sin(self.xi()) * np.cos(omega * t)
            )
        return func
    
    @staticmethod
    def _dft(N) -> np.array:
        return fft.fft(np.eye(N), axis=0)
    
    @staticmethod
    def _idft(N) -> np.array:
        return fft.ifft(np.eye(N), axis=0)
    
    def time_evolution(self, tlist, omega, A, psi0, t0=0):
        psi = np.array(psi0, dtype=complex, copy=True)
        tprev = t0

        # Store potential and kinetic energy as vectors
        V0 = np.diag(self._V())
        T = np.diag(self._T(momentum=True))
        sin_x = np.sin(self.xi())

        for t in tlist:
            dt = t - tprev
            t_mid = tprev + dt / 2

            V_mid = V0 + A * sin_x * np.cos(omega * t_mid)

            exp_V = np.exp(-1j * V_mid * dt / 2)
            exp_T = np.exp(-1j * T * dt)

            # Half potential step
            psi = exp_V * psi

            # Full kinetic step in momentum space
            psi = fft.fft(psi)
            psi = exp_T * psi
            psi = fft.ifft(psi)

            # Half potential step
            psi = exp_V * psi

            tprev = t

        return psi
        
    def eigenstates(self):
        # diagonalise the hamiltonian in position basis
        F = HarmonicOscillator._dft(self.n)
        F_inv = HarmonicOscillator._idft(self.n)

        # H = F^(-1)TF + V
        H = F_inv @ self._T() @ F + self._V()

        eigenvalues, eigenvectors = np.linalg.eigh(H)

        return eigenvalues, eigenvectors

    def anal_eigenstates(self, n):
        x = self.xi()

        Hn = special.hermite(n, monic = False)

        norm = np.sqrt(
            2**n * special.factorial(n) * np.sqrt(np.pi)
        )

        return Hn(x) * np.exp(-x**2 / 2) / norm