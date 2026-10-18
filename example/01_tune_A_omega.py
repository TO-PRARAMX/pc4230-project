import argparse
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonic import HarmonicOscillator

def evolve(ho, psi0, times, omega, A):
    """Calculate psi(t) = U(t, 0) psi0; no new FFT propagation logic here.

    NOTE: Calling time_evolution([dt], ...) each frame would be INCORRECT:
    it resets the driven potential's clock to zero for each step.
    """
    states = np.empty((len(times), ho.n), dtype=complex)
    states[0] = psi0
    for frame in range(1, len(times)):
        U = ho.time_evolution(times[1 : frame + 1], omega=omega, A=A)
        states[frame] = U @ psi0
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--A", type=float, default=0.8, help="Drive amplitude")
    parser.add_argument("--omega", type=float, default=1.0, help="Drive frequency")
    parser.add_argument("--n", type=int, default=48, help="Number of spatial grid points")
    parser.add_argument("--frames", type=int, default=125, help="Number of animation frames")
    parser.add_argument("--tmax", type=float, default=15.0, help="Total simulated time")
    args = parser.parse_args()

    if args.n < 8 or args.frames < 2 or args.tmax <= 0:
        parser.error("n must be >= 8, frames >= 2, and tmax > 0")
    if not (-3 <= args.A <= 3 and 0 <= args.omega <= 4):
        parser.error("Please use -3 <= A <= 3 and 0 <= omega <= 4")

    ho = HarmonicOscillator(boundary=(-7.0, 7.0), n=args.n)
    x = ho.xi()
    dx = x[1] - x[0]
    times = np.linspace(0.0, args.tmax, args.frames)

    # Initial state: lowest eigenstate of the SAME discretized Hamiltonian.
    _, eigvecs = ho.eigenstates()
    psi0 = eigvecs[:, 0]

    A = args.A
    omega = args.omega
    states = evolve(ho, psi0, times, omega, A)
    density = np.abs(states) ** 2 / dx  # continuum-normalized probability density
    frame_index = 0
    playing = True
    dirty = False

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.subplots_adjust(left=0.11, right=0.89, top=0.88, bottom=0.35)
    ax2 = ax.twinx()

    (prob_line,) = ax.plot(x, density[0], lw=2.3, color="tab:blue", label=r"$|\psi(x,t)|^2$")
    (potential_line,) = ax2.plot(x, x**2 / 2 + A * np.sin(x), lw=1.6,
                                 color="tab:orange", alpha=0.85, label=r"$V(x,t)$")
    mean_line = ax.axvline(np.sum(x * np.abs(states[0]) ** 2),
                           ls="--", lw=1.4, color="tab:green", label=r"$\langle x \rangle$")
    ax.set(xlim=(x[0], x[-1]), xlabel="Position x", ylabel=r"Probability density $|\psi|^2$")
    ax2.set_ylabel("Potential energy V(x,t)")
    ax2.set_ylim(-4.5, 30.0)
    ax.set_ylim(0, max(0.85, 1.15 * np.max(density)))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    title = ax.set_title("")

    ax_A = fig.add_axes([0.18, 0.185, 0.68, 0.032])
    ax_omega = fig.add_axes([0.18, 0.132, 0.68, 0.032])
    slider_A = Slider(ax_A, "A", -3, 3, valinit=A, valstep=0.05)
    slider_omega = Slider(ax_omega, r"$\omega$", 0, 4, valinit=omega, valstep=0.05)

    ax_pause = fig.add_axes([0.20, 0.045, 0.16, 0.050])
    ax_restart = fig.add_axes([0.42, 0.045, 0.16, 0.050])
    ax_apply = fig.add_axes([0.64, 0.045, 0.20, 0.050])
    button_pause = Button(ax_pause, "Pause")
    button_restart = Button(ax_restart, "Restart")
    button_apply = Button(ax_apply, "Apply A, omega")
    status = fig.text(0.5, 0.245, "Adjust a slider and release to restart with new parameters.",
                      ha="center", fontsize=9, color="0.4")

    def show_state():
        t = times[frame_index]
        d = density[frame_index]
        prob_line.set_ydata(d)
        potential_line.set_ydata(x**2 / 2 + A * np.sin(x) * np.cos(omega * t))
        x_mean = np.sum(x * np.abs(states[frame_index]) ** 2)
        mean_line.set_xdata([x_mean, x_mean])
        title.set_text(f"Driven harmonic oscillator     t = {t:.2f}     A = {A:.2f}     omega = {omega:.2f}")
        return prob_line, potential_line, mean_line, title

    def animate(_):
        nonlocal frame_index
        if playing:
            frame_index = (frame_index + 1) % len(times)
        return show_state()

    animation = FuncAnimation(fig, animate, interval=55, blit=False,
                              cache_frame_data=False)

    def mark_dirty(_):
        nonlocal dirty
        dirty = True
        status.set_text("Release the slider or click 'Apply A, omega' to recalculate.")
        fig.canvas.draw_idle()

    slider_A.on_changed(mark_dirty)
    slider_omega.on_changed(mark_dirty)

    def apply_parameters(_=None):
        nonlocal A, omega, states, density, dirty, frame_index
        if not dirty:
            return

        was_playing = playing
        if was_playing:
            animation.event_source.stop()
        status.set_text("Recalculating using HarmonicOscillator.time_evolution() ...")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        new_A, new_omega = slider_A.val, slider_omega.val
        updated_states = evolve(ho, psi0, times, new_omega, new_A)
        A, omega = new_A, new_omega
        states = updated_states
        density = np.abs(states) ** 2 / dx
        frame_index = 0
        dirty = False
        ax.set_ylim(0, max(0.85, 1.15 * np.max(density)))
        show_state()
        status.set_text("Adjust a slider and release to restart with new parameters.")
        fig.canvas.draw_idle()
        if was_playing:
            animation.event_source.start()

    def apply_on_release(event):
        if dirty and event.inaxes in (ax_A, ax_omega):
            apply_parameters()

    def toggle_pause(_):
        nonlocal playing
        playing = not playing
        if playing:
            button_pause.label.set_text("Pause")
            animation.event_source.start()
        else:
            button_pause.label.set_text("Play")
            animation.event_source.stop()
        fig.canvas.draw_idle()

    def restart(_):
        nonlocal frame_index
        frame_index = 0
        show_state()
        fig.canvas.draw_idle()

    button_pause.on_clicked(toggle_pause)
    button_restart.on_clicked(restart)
    button_apply.on_clicked(apply_parameters)
    fig.canvas.mpl_connect("button_release_event", apply_on_release)

    show_state()
    plt.show()


if __name__ == "__main__":
    main()
