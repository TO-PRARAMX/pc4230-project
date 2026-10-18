import numpy as np
import matplotlib.pyplot as plt

from setup import HarmonicOscillator

ho = HarmonicOscillator(boundary = (-10, 10),
                n=100, x0 = 0)

E, psi = ho.eigenstates()
print(E[:5])
#plt.plot(eigenstates[1])
#plt.plot(ho.anal_eigenstates(n=2))

x = ho.xi()
dx = x[1] - x[0]

plt.figure(figsize=(10, 6))


for n in range(4):
    # Numerical eigenstate
    psi_num = psi[:, n] / np.sqrt(dx)

    # Analytical eigenstate
    psi_anal = ho.anal_eigenstates(n)

    # Calculate overlap to determine phase
    overlap = np.vdot(psi_anal, psi_num) * dx

    # Remove arbitrary phase
    psi_num = psi_num * np.exp(-1j * np.angle(overlap))

    plt.plot(x, psi_num.real, label=f"Numerical n={n}")
    plt.plot(x, psi_anal, "--", label=f"Analytical n={n}")

plt.xlabel("Position x")
plt.ylabel(r"$\psi_n(x)$")
plt.title("Numerical vs Analytical Eigenstates")
plt.legend()
plt.grid()
plt.show()