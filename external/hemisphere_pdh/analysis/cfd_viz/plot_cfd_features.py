"""
Generate presentation-quality CFD contour plots for the hemisphere flow.

Reads the OpenFOAM data directly (avoids Plot3D resolution limits) and
produces several figures highlighting key flow features:
  1. Mach number  — bow shock, lambda foot, wake
  2. Pressure     — shock structure, stagnation, expansion
  3. Temperature  — thermal features, shock heating
  4. Velocity magnitude — freestream, deceleration, wake deficit
  5. Density gradient (numerical schlieren) — shock visualization
  6. x-z plane (top view) Mach — 3D shock shape
  7. y-z cross-sections of Mach at downstream stations

Usage:
    python plot_cfd_features.py
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────
FOAM_DIR = Path(
    r"C:\Users\kalagotla\OneDrive - Florida State University"
    r"\Documents\Scripts\hemisphere_openfoam\hemisphere_SST_3D"
)
OUT_DIR = Path(__file__).resolve().parents[2] / "figures" / "cfd_viz"
R = 0.0508  # hemisphere radius (m)
GAMMA = 1.4
R_GAS = 287.058

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": False,
    "mathtext.fontset": "cm",
})


def read_foam():
    """Read latest OpenFOAM timestep."""
    reader = pv.OpenFOAMReader(str(FOAM_DIR / "case.foam"))
    reader.set_active_time_value(reader.time_values[-1])
    mesh = reader.read()
    return mesh['internalMesh']


def slice_and_grid(internal, origin, normal, bounds, res=500):
    """
    Slice the mesh, then resample onto a uniform structured grid using
    PyVista's VTK-based interpolation (handles non-uniform meshes well).
    Returns (x_2d, y_2d, dict_of_fields).
    """
    # Determine in-plane axes and build a planar StructuredGrid for sampling
    if normal == 'z':
        c1_idx, c2_idx = 0, 1  # x, y
        fixed_idx, fixed_val = 2, origin[2] if hasattr(origin, '__len__') else 0
    elif normal == 'y':
        c1_idx, c2_idx = 0, 2  # x, z
        fixed_idx, fixed_val = 1, origin[1] if hasattr(origin, '__len__') else 0
    elif normal == 'x':
        c1_idx, c2_idx = 2, 1  # z, y
        fixed_idx, fixed_val = 0, origin[0] if hasattr(origin, '__len__') else 0

    c1_lin = np.linspace(bounds[0], bounds[1], res)
    c2_lin = np.linspace(bounds[2], bounds[3], res)
    c1_2d, c2_2d = np.meshgrid(c1_lin, c2_lin)

    # Build 3D probe points on the slice plane
    probe_pts = np.zeros((c1_2d.size, 3))
    probe_pts[:, c1_idx] = c1_2d.ravel()
    probe_pts[:, c2_idx] = c2_2d.ravel()
    probe_pts[:, fixed_idx] = fixed_val
    probe_grid = pv.StructuredGrid()
    probe_grid.points = probe_pts
    probe_grid.dimensions = [res, res, 1]

    # Sample directly from the 3D volume (cells have thickness, so probe hits)
    sampled = probe_grid.sample(internal)

    # Valid-point mask from VTK (1 = hit a cell, 0 = outside the mesh)
    valid = np.array(sampled.point_data['vtkValidPointMask']).reshape(res, res).astype(bool)

    def _field_2d(name, component=None):
        arr = np.array(sampled.point_data[name])
        if component is not None:
            arr = arr[:, component]
        grid = arr.reshape(res, res).astype(float)
        grid[~valid] = np.nan
        return grid

    fields = {}
    fields['Umag'] = np.sqrt(
        np.nansum(np.stack([_field_2d('U', i)**2 for i in range(3)]), axis=0)
    )
    fields['Ux'] = _field_2d('U', 0)
    fields['Uy'] = _field_2d('U', 1)
    fields['Uz'] = _field_2d('U', 2)
    fields['rho'] = _field_2d('rho')
    fields['p'] = _field_2d('p')
    fields['T'] = _field_2d('T')

    # Mach
    a = np.sqrt(GAMMA * R_GAS * np.nan_to_num(fields['T'], nan=1.0))
    fields['Mach'] = fields['Umag'] / a

    # Density gradient magnitude (numerical schlieren)
    # Compute gradient on the original CFD mesh (accurate at cell level),
    # then sample the gradient magnitude onto the uniform grid.
    # This avoids mesh-imprint artifacts from differentiating interpolated data.
    grad_mesh = internal.compute_derivative(scalars='rho', gradient='grad_rho')
    sampled_grad = probe_grid.sample(grad_mesh)
    grad_arr = np.array(sampled_grad.point_data['grad_rho'])
    grad_valid = np.array(
        sampled_grad.point_data['vtkValidPointMask']
    ).reshape(res, res).astype(bool)
    drho_2d = np.linalg.norm(grad_arr, axis=1).reshape(res, res).astype(float)
    drho_2d[~grad_valid] = np.nan
    # Light smoothing to suppress remaining cell-boundary noise
    from scipy.ndimage import gaussian_filter
    drho_2d = gaussian_filter(np.nan_to_num(drho_2d, nan=0.0), sigma=1.5)
    drho_2d[~valid] = np.nan
    fields['drho'] = drho_2d

    return c1_2d, c2_2d, fields


def hemisphere_outline(ax, plane='xy'):
    """Draw hemisphere on the given axis."""
    theta = np.linspace(0, np.pi, 300)
    if plane == 'xy':
        ax.plot(R * np.cos(theta), R * np.sin(theta), 'k-', lw=1.5, zorder=10)
        ax.fill_between(R * np.cos(theta), 0, R * np.sin(theta),
                         color='white', edgecolor='k', lw=1.5, zorder=10)
    elif plane == 'xz':
        phi = np.linspace(0, 2 * np.pi, 300)
        ax.plot(R * np.cos(phi), R * np.sin(phi), 'k-', lw=1.5, zorder=10)
        ax.fill(R * np.cos(phi), R * np.sin(phi),
                 color='white', edgecolor='k', lw=1.5, zorder=10)


def save_contour(c1, c2, field, title, cmap, label, fname,
                 plane='xy', levels=80, norm=None,
                 xlim=None, ylim=None, xlabel=None, ylabel=None,
                 use_imshow=False):
    """Generic contour plot helper. Use use_imshow=True for schlieren."""
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

    # Mask NaN
    field_masked = np.ma.masked_invalid(field)

    if use_imshow:
        # imshow gives smooth, pixel-level rendering — ideal for schlieren
        vmin = np.nanquantile(field_masked.compressed(), 0.01)
        vmax = np.nanquantile(field_masked.compressed(), 0.99)
        if norm is None:
            norm = Normalize(vmin=vmin, vmax=vmax)
        extent = [c1.min(), c1.max(), c2.min(), c2.max()]
        im = ax.imshow(field_masked, cmap=cmap, norm=norm,
                       extent=extent, origin='lower', aspect='auto',
                       interpolation='bilinear', zorder=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    else:
        if norm is None:
            vmin = np.nanquantile(field_masked.compressed(), 0.01)
            vmax = np.nanquantile(field_masked.compressed(), 0.99)
            lev = np.linspace(vmin, vmax, levels)
        else:
            lev = levels
        cf = ax.contourf(c1, c2, field_masked, levels=lev, cmap=cmap,
                          norm=norm, extend="both", zorder=0)
        cb = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.02)

    cb.set_label(label, fontsize=11)

    hemisphere_outline(ax, plane=plane)
    ax.set_aspect("equal")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel or r"$x$ (m)")
    ax.set_ylabel(ylabel or r"$y$ (m)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(direction="in", top=True, right=True)

    fig.savefig(str(OUT_DIR / fname))
    print(f"  Saved {fname}")
    plt.close(fig)


def save_grid(internal, fname="cfd_grid_xy.png",
              xlim=(-0.25, 0.35), ylim=(0, 0.20)):
    """Plot the computational grid on the z=0 symmetry plane."""
    # Slice the 3D mesh at z=0 to get the 2D cell faces
    sliced = internal.slice(normal='z', origin=(0, 0, 0))

    # Extract edges of the sliced mesh
    edges = sliced.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=True, non_manifold_edges=True
    )

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

    # Draw each edge segment
    pts = np.array(edges.points)
    lines = edges.lines
    i = 0
    segments = []
    while i < len(lines):
        n_pts = lines[i]
        idx = lines[i + 1: i + 1 + n_pts]
        seg_pts = pts[idx]
        segments.append(seg_pts[:, :2])  # x, y only
        i += 1 + n_pts

    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, colors='k', linewidths=0.3, zorder=1)
    ax.add_collection(lc)

    hemisphere_outline(ax, plane='xy')
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title("Computational Grid — symmetry plane ($z{=}0$)",
                  fontsize=13, fontweight="bold")
    ax.tick_params(direction="in", top=True, right=True)

    fig.savefig(str(OUT_DIR / fname))
    print(f"  Saved {fname}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Reading OpenFOAM data...")
    internal = read_foam()

    # ── Grid visualization ──
    print("Generating grid plot...")
    xlim_side = (-0.25, 0.35)
    ylim_side = (0, 0.20)
    save_grid(internal, xlim=xlim_side, ylim=ylim_side)

    # ── z=0 symmetry plane (side view, x-y) ──
    print("Slicing z=0 plane...")
    x_s, y_s, f_s = slice_and_grid(
        internal, origin=(0, 0, 0), normal='z',
        bounds=[xlim_side[0], xlim_side[1], ylim_side[0], ylim_side[1]],
        res=1200)

    print("Generating side-view contours...")

    # 1. Mach number
    save_contour(x_s, y_s, f_s['Mach'],
                 title="Mach Number — symmetry plane ($z{=}0$)",
                 cmap="coolwarm", label="Mach",
                 fname="cfd_mach_xy.png",
                 xlim=xlim_side, ylim=ylim_side,
                 norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=2.5))

    # 2. Pressure
    save_contour(x_s, y_s, f_s['p'] / 1000,
                 title="Static Pressure — symmetry plane ($z{=}0$)",
                 cmap="inferno", label="$p$ (kPa)",
                 fname="cfd_pressure_xy.png",
                 xlim=xlim_side, ylim=ylim_side)

    # 3. Temperature
    save_contour(x_s, y_s, f_s['T'],
                 title="Temperature — symmetry plane ($z{=}0$)",
                 cmap="hot", label="$T$ (K)",
                 fname="cfd_temperature_xy.png",
                 xlim=xlim_side, ylim=ylim_side)

    # 4. Velocity magnitude
    save_contour(x_s, y_s, f_s['Umag'],
                 title="Velocity Magnitude — symmetry plane ($z{=}0$)",
                 cmap="viridis", label="$|U|$ (m/s)",
                 fname="cfd_velocity_xy.png",
                 xlim=xlim_side, ylim=ylim_side)

    # 5. Numerical schlieren (density gradient)
    drho = f_s['drho']
    # Log scale for schlieren
    drho_log = np.log10(np.clip(drho, 1e-3, None))
    save_contour(x_s, y_s, drho_log,
                 title="Numerical Schlieren ($\\log_{10}|\\nabla\\rho|$)"
                       " — symmetry plane ($z{=}0$)",
                 cmap="binary", label=r"$\log_{10}|\nabla\rho|$ (kg/m$^4$)",
                 fname="cfd_schlieren_xy.png",
                 xlim=xlim_side, ylim=ylim_side,
                 use_imshow=True)

    # 6. Density
    save_contour(x_s, y_s, f_s['rho'],
                 title="Density — symmetry plane ($z{=}0$)",
                 cmap="YlOrRd", label=r"$\rho$ (kg/m$^3$)",
                 fname="cfd_density_xy.png",
                 xlim=xlim_side, ylim=ylim_side)

    # ── y=0.03 plane (top view, x-z) ──
    print("Slicing y=0.03 plane...")
    xlim_top = (-0.20, 0.35)
    zlim_top = (-0.18, 0.18)
    x_t, z_t, f_t = slice_and_grid(
        internal, origin=(0, 0.03, 0), normal='y',
        bounds=[xlim_top[0], xlim_top[1], zlim_top[0], zlim_top[1]],
        res=1200)

    print("Generating top-view contours...")

    # 7. Mach — top view
    save_contour(x_t, z_t, f_t['Mach'],
                 title="Mach Number — horizontal plane ($y{=}0.03$ m)",
                 cmap="coolwarm", label="Mach",
                 fname="cfd_mach_xz.png", plane='xz',
                 xlim=xlim_top, ylim=zlim_top,
                 ylabel=r"$z$ (m)",
                 norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=2.5))

    # 8. Schlieren — top view
    drho_t = f_t['drho']
    drho_t_log = np.log10(np.clip(drho_t, 1e-3, None))
    save_contour(x_t, z_t, drho_t_log,
                 title="Numerical Schlieren — horizontal plane ($y{=}0.03$ m)",
                 cmap="binary",
                 label=r"$\log_{10}|\nabla\rho|$ (kg/m$^4$)",
                 fname="cfd_schlieren_xz.png", plane='xz',
                 xlim=xlim_top, ylim=zlim_top,
                 ylabel=r"$z$ (m)",
                 use_imshow=True)

    # ── y-z cross-sections at downstream stations ──
    print("Slicing downstream y-z planes...")
    x_stations = [0.0, 0.08, 0.15, 0.30]
    fig, axes = plt.subplots(1, len(x_stations),
                              figsize=(4 * len(x_stations), 4.5),
                              constrained_layout=True)
    ylim_yz = (0, 0.15)
    zlim_yz = (-0.15, 0.15)

    for i, xs in enumerate(x_stations):
        ax = axes[i]
        z_yz, y_yz, f_yz = slice_and_grid(
            internal, origin=(xs, 0, 0), normal='x',
            bounds=[zlim_yz[0], zlim_yz[1], ylim_yz[0], ylim_yz[1]],
            res=400)

        mach = np.ma.masked_invalid(f_yz['Mach'])
        lev = np.linspace(0, 2.5, 60)
        cf = ax.contourf(z_yz, y_yz, mach, levels=lev, cmap="coolwarm",
                          norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=2.5),
                          extend="both")

        # Hemisphere cross-section
        if abs(xs) < R:
            r_xs = np.sqrt(R**2 - xs**2)
            th = np.linspace(0, np.pi, 200)
            ax.fill(r_xs * np.sin(th), r_xs * np.cos(th),
                    color='white', edgecolor='k', lw=1.2, zorder=10)

        ax.set_aspect("equal")
        ax.set_xlim(zlim_yz)
        ax.set_ylim(ylim_yz)
        ax.set_xlabel(r"$z$ (m)", fontsize=10)
        if i == 0:
            ax.set_ylabel(r"$y$ (m)", fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"$x = {xs:.2f}$ m", fontsize=11, fontweight="bold")
        ax.tick_params(direction="in", top=True, right=True, labelsize=9)

    cb = fig.colorbar(cf, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Mach", fontsize=11)
    fig.suptitle("Mach Number — downstream $y$-$z$ cross-sections",
                  fontsize=13, fontweight="bold")
    fig.savefig(str(OUT_DIR / "cfd_mach_yz_sections.png"))
    print("  Saved cfd_mach_yz_sections.png")
    plt.close(fig)

    print("\nAll CFD figures saved to figures/")


if __name__ == "__main__":
    main()
