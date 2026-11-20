# Lives in a .py file (not a notebook cell) so worker processes can import it:
# Windows starts workers with "spawn", which can't see functions defined in a notebook.
import numpy as np

from harmonic import HarmonicOscillator


def simulate(A=0.8, omega=1.0, dt=0.02, n=128, L=12, tmax=20, alpha=0):
    ho = HarmonicOscillator(boundary=(-L, L), n=n, alpha=alpha)
    x = ho.xi()
    times = np.linspace(0, tmax, int(np.ceil(tmax / dt)) + 1)

    energies, basis = ho.eigenstates()

    states = np.empty((len(times), n), dtype=complex)
    states[0] = basis[:, 0]

    for i in range(1, len(times)):
        states[i] = ho.time_evolution(
            [times[i]],
            omega=omega,
            A=A,
            psi0=states[i-1],
            t0=times[i-1]
        )

    amplitudes = states @ basis.conj()
    E0 = np.abs(amplitudes)**2 @ energies

    return x, times, states, E0


def populations(A, omega, tmax, levels, n=128, L=12, alpha=0):
    """Return times and P(0 -> k)(t) for each k in levels, shape (time, len(levels))."""
    _, basis = HarmonicOscillator(boundary=(-L, L), n=n, alpha=alpha).eigenstates()
    _, t, states, _ = simulate(A=A, omega=omega, n=n, L=L, tmax=tmax, alpha=alpha)
    return t, np.abs(states @ basis[:, levels].conj())**2


def transition_prob(A, omega, tmax, n=128, L=12, alpha=0):
    """Return times and P(0 -> 1)(t)."""
    t, P = populations(A, omega, tmax, [1], n=n, L=L, alpha=alpha)
    return t, P[:, 0]
