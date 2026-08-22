"""A specialized reference inertial-particle integrator, vectorized in NumPy.

This exists only as a throughput reference for the tracking benchmark. It
advances many particles together through a piecewise-linear one-dimensional
fluid-velocity field using a fixed-step explicit update of the same particle
equation of motion that lptlib integrates,

    dv/dt = -0.75 * Cd * rho_f / (rho_p * dp) * |v - u| * (v - u),

with the drag coefficient taken from lptlib's own Variables routine so the
physics matches. It performs no point-location search and stores the field as a
plain 1D array, so it is deliberately much faster per particle than a general
curvilinear tracker. It is not a competitor to lptlib, it is an upper bound that
shows the cost of lptlib's general machinery.
"""

import time

import numpy as np

import common


def track_particles_reference(x_nodes, u_nodes, n_particles, dp, rho_p, rho_f,
                              temperature, model="stokes", dt=2e-8, t_end=8e-5,
                              x0=2e-5):
    """Advance n_particles through u(x) with a fixed-step vectorized update.

    Returns (seconds, total_steps) where total_steps is n_particles * n_time.
    """
    mu = common.sutherland_viscosity(temperature)
    a_sound = np.sqrt(common.GAMMA * common.GAS_CONSTANT * temperature)

    # Vectorized drag: Stokes has a closed form; for other models fall back to a
    # per-value call of lptlib's drag routine (still correct, just slower).
    stokes = model == "stokes"

    n_time = int(np.ceil(t_end / dt))
    x = np.full(n_particles, x0, dtype=float)
    v = np.interp(x, x_nodes, u_nodes)  # start at local fluid velocity

    t0 = time.perf_counter()
    for _ in range(n_time):
        u = np.interp(x, x_nodes, u_nodes)
        slip = v - u
        smag = np.abs(slip)
        if stokes:
            # Cd*|slip| = 24 mu/(rho_f dp), so acc = -(18 mu/(rho_p dp^2)) slip
            acc = -(18.0 * mu / (rho_p * dp ** 2)) * slip
        else:
            re = rho_f * smag * dp / mu
            mach = smag / a_sound
            cd = np.array([common.drag_coefficient(float(r), float(m), model)
                           if r > 1e-9 else 0.0 for r, m in zip(re, mach)])
            acc = -0.75 * cd * rho_f / (rho_p * dp) * smag * slip
        v = v + acc * dt
        x = x + v * dt
    seconds = time.perf_counter() - t0
    return seconds, n_particles * n_time
