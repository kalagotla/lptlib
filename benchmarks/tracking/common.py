"""Shared helpers for the lptlib particle-tracking benchmarks.

These utilities build a controlled single-block PLOT3D field in which the
streamwise fluid velocity follows a uniform / smooth-ramp / uniform profile
while density and temperature stay uniform. A profile like this is ideal for
verification because:

- lptlib reconstructs the field with linear interpolation, and a piecewise
  linear nodal velocity is reproduced exactly, so no interpolation error is
  introduced into the comparison.
- density and temperature are uniform, so the viscosity and hence the Stokes
  relaxation time are constant, which gives a clean reference solution.
- the particle is seeded in the uniform upstream region at the local fluid
  velocity, so it advects with zero slip until it reaches the ramp, matching
  how lptlib is used on real shock cases.

The particle equation of motion integrated by lptlib is

    dv/dt = -0.75 * Cd * rho_f / (rho_p * dp) * |v - u| * (v - u),

and the reference integrator in this package integrates the identical equation
with SciPy using lptlib's own drag-coefficient routine, so the comparison
isolates the accuracy of lptlib's adaptive integrator rather than any
difference in the drag law.

author: benchmark generated for lptlib (Dilip Kalagotla)
"""

import struct
import sys
import types
from pathlib import Path

import numpy as np

# The lptlib.io package imports DataIO, which imports mpi4py at module load.
# mpi4py needs a working MPI runtime that is not always present (for example in
# a minimal CI container). The particle tracker itself does not use MPI, so a
# lightweight stub lets the tracking benchmarks import lptlib without a system
# MPI. On a normal install with mpi4py present this stub is simply ignored.
try:
    from mpi4py import MPI as _MPI  # noqa: F401
except Exception:
    _stub = types.ModuleType("mpi4py")
    _stub.MPI = types.SimpleNamespace(COMM_WORLD=None)
    sys.modules["mpi4py"] = _stub
    sys.modules["mpi4py.MPI"] = _stub.MPI

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

GAMMA = 1.4
GAS_CONSTANT = 287.052874


def sutherland_viscosity(temperature):
    """Sutherland air viscosity in kg/m/s, matching lptlib's integration path."""
    c1 = 1.716e-5 * (273.15 + 110.4) / 273.15 ** 1.5
    return c1 * temperature ** 1.5 / (temperature + 110.4)


def ramp_velocity_nodes(x_nodes, x1, x2, u1, u2):
    """Piecewise-linear streamwise velocity: u1 for x<=x1, linear to u2 by x2."""
    return np.where(
        x_nodes <= x1, u1,
        np.where(x_nodes >= x2, u2, u1 + (u2 - u1) * (x_nodes - x1) / (x2 - x1)),
    )


def write_ramp_field(grid_path, flow_path, ni=500, nj=2, nk=2, lx=5e-4,
                     x1=1e-4, x2=2e-4, u1=12.0, u2=4.0, rho=1.0, temperature=300.0):
    """Write a uniform / ramp / uniform PLOT3D grid and flow pair.

    Returns (x_nodes, u_nodes) describing the exact nodal velocity profile so a
    reference integrator can reproduce the identical field.
    """
    x_nodes = np.linspace(0.0, lx, ni)
    u_nodes = ramp_velocity_nodes(x_nodes, x1, x2, u1, u2)
    y = np.linspace(0.0, 6e-5, nj)
    z = np.linspace(0.0, 6e-5, nk)
    xx, yy, zz = np.meshgrid(x_nodes, y, z, indexing="ij")
    grd = np.stack([xx, yy, zz], axis=-1).astype("f4")

    u = np.interp(xx, x_nodes, u_nodes)
    p = rho * GAS_CONSTANT * temperature
    e = p / (GAMMA - 1.0) + 0.5 * rho * u ** 2
    q = np.zeros((ni, nj, nk, 5), dtype="f4")
    q[..., 0] = rho
    q[..., 1] = rho * u
    q[..., 4] = e

    with open(grid_path, "wb") as f:
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<iii", ni, nj, nk))
        f.write(grd.tobytes(order="F"))
    with open(flow_path, "wb") as f:
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<iii", ni, nj, nk))
        f.write(np.array([0.0, 0.0, 0.0, 0.0], dtype="f4").tobytes())
        f.write(q.tobytes(order="F"))
    return x_nodes, u_nodes


def drag_coefficient(re, mach, model):
    """Drag coefficient from lptlib's own Variables.compute_drag_coefficient.

    Using the library routine guarantees the reference integrator applies the
    same drag law as lptlib, so the verification measures integrator accuracy.
    """
    from lptlib.function.variables import Variables

    # Variables needs a flow-like object only to read gamma/gas_constant; the
    # drag routine itself takes re and mach directly.
    dummy = types.SimpleNamespace(q=np.zeros((1, 1, 1, 5, 1)))
    v = Variables(dummy, gamma=GAMMA, gas_constant=GAS_CONSTANT)
    return v.compute_drag_coefficient(_re=re, _mach=mach, _model=model)


def reference_relaxation(x_nodes, u_nodes, x0, dp, rho_p, rho_f, temperature,
                         model="stokes", t_end=8e-5, max_step=1e-9):
    """High-accuracy reference solution of the particle EOM through u(x).

    Integrates dx/dt = v, dv/dt = -0.75 Cd rho_f/(rho_p dp)|v-u|(v-u) with SciPy
    DOP853 at tight tolerance, using lptlib's own drag coefficient and a
    piecewise-linear u(x) identical to the PLOT3D field. Returns (x, v) samples.
    """
    from scipy.integrate import solve_ivp

    mu = sutherland_viscosity(temperature)
    a_sound = np.sqrt(GAMMA * GAS_CONSTANT * temperature)

    def uf(xx):
        return np.interp(xx, x_nodes, u_nodes)

    def rhs(_t, s):
        x, v = s
        u = uf(x)
        slip = v - u
        smag = abs(slip)
        if smag < 1e-30:
            return [v, 0.0]
        re = rho_f * smag * dp / mu
        mach = smag / a_sound
        cd = drag_coefficient(re, mach, model)
        acc = -0.75 * cd * rho_f / (rho_p * dp) * smag * slip
        return [v, acc]

    sol = solve_ivp(rhs, [0.0, t_end], [x0, uf(x0)], method="DOP853",
                    rtol=1e-11, atol=1e-14, dense_output=True, max_step=max_step)
    return sol.y[0], sol.y[1]


def track_lptlib(grid_path, flow_path, x0, dp, rho_p, drag_model="stokes",
                 time_step=1e-8, max_time_step=1e-6, adaptivity=0.001,
                 magnitude_adaptivity=0.001, y0=3e-5, z0=3e-5):
    """Track one inertial particle with lptlib's adaptive scheme.

    Returns (x, vp, n_steps) where x and vp are the streamwise position and
    particle velocity histories.
    """
    from lptlib import Streamlines

    sl = Streamlines(grid_path, flow_path, point=[x0, y0, z0], integration="pRK4",
                     diameter=dp, density=rho_p, time_step=time_step,
                     max_time_step=max_time_step, drag_model=drag_model,
                     adaptivity=adaptivity, magnitude_adaptivity=magnitude_adaptivity)
    sl.compute(method="adaptive-ppath")
    x = np.array(sl.streamline)[:, 0]
    vp = np.array(sl.svelocity)[:, 0]
    return x, vp, len(x)
