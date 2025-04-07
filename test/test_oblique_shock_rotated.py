import numpy as np
import matplotlib.pyplot as plt
from src.lptlib import ObliqueShock, ObliqueShockData, ObliqueShockAlignedData

# Import your classes from the module where they are defined.
# For example, if the file is named "oblique_shock.py", you might do:
# from oblique_shock import ObliqueShock, ObliqueShockData, ObliqueShockAlignedData
# Here, we assume they are already in the current namespace.

def main():
    # Simulation parameters
    mach = 1.2
    deflection_deg = 0.02
    inlet_temperature = 300.0  # Kelvin
    inlet_density = 1.0  # kg/m^3

    # 3D domain extents and grid resolution
    nx_max = 5000e-3
    ny_max = 5000e-3
    nz_max = 1e-3 # full 3D domain in z
    xpoints = 1000
    ypoints = 1000
    zpoints = 2

    # Compute oblique shock properties
    shock = ObliqueShock(mach=mach, deflection=deflection_deg)
    shock.compute()

    # -------------------------------
    # Original Data (Shock at x = 0)
    # -------------------------------
    original_data = ObliqueShockData(oblique_shock=shock)
    original_data.shock_strength = 'weak'
    original_data.nx_max = nx_max
    original_data.ny_max = ny_max
    original_data.nz_max = nz_max
    original_data.xpoints = xpoints
    original_data.ypoints = ypoints
    original_data.zpoints = zpoints
    original_data.inlet_temperature = inlet_temperature
    original_data.inlet_density = inlet_density
    original_data.create_grid()
    original_data.create_flow()

    # -------------------------------
    # Aligned Data (Shock Plane Oriented by β)
    # -------------------------------
    aligned_data = ObliqueShockAlignedData(oblique_shock=shock)
    aligned_data.shock_strength = 'strong'
    aligned_data.nx_max = nx_max
    aligned_data.ny_max = ny_max
    aligned_data.nz_max = nz_max
    aligned_data.xpoints = xpoints
    aligned_data.ypoints = ypoints
    aligned_data.zpoints = zpoints
    aligned_data.inlet_temperature = inlet_temperature
    aligned_data.inlet_density = inlet_density
    aligned_data.create_grid()
    aligned_data.create_flow()

    # For visualization, extract a 2D slice at mid z-index (z = nz_max/2)
    mid_z = zpoints // 2
    # Original data slice
    grd_orig = original_data.grid.grd[:, :, mid_z, :]  # shape (ni, nj, 3)
    x_orig = grd_orig[..., 0, 0]
    y_orig = grd_orig[..., 1, 0]
    density_orig = original_data.flow.q[:, :, mid_z, 0, 0]

    # Aligned data slice
    grd_aligned = aligned_data.grid.grd[:, :, mid_z, :]
    x_aligned = grd_aligned[..., 0, 0]
    y_aligned = grd_aligned[..., 1, 0]
    density_aligned = aligned_data.flow.q[:, :, mid_z, 0, 0]

    # Create plots for the two cases
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original data plot: Shock at x = 0
    pcm0 = axes[0].pcolormesh(x_orig, y_orig, density_orig, shading='auto', cmap='viridis')
    axes[0].axvline(x=0, color='red', linestyle='--', label='Shock (x = 0)')
    axes[0].set_title('Original Oblique Shock Data\n(3D Slice at z = mid)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    fig.colorbar(pcm0, ax=axes[0], label='Density')

    # Aligned data plot: Shock plane aligned with computed β
    # Compute shock plane line in the x-y plane (passing through y = ny_max/2 at x = 0)
    beta_deg = shock.shock_angle[0] if hasattr(shock.shock_angle, '__iter__') else shock.shock_angle
    beta = np.radians(beta_deg)
    shock_line_x = np.linspace(-nx_max, nx_max, 100)
    shock_line_y = ny_max / 2 + np.tan(beta) * shock_line_x

    pcm1 = axes[1].pcolormesh(x_aligned, y_aligned, density_aligned, shading='auto', cmap='viridis')
    axes[1].plot(shock_line_x, shock_line_y, color='red', linestyle='--', label='Shock Plane')
    axes[1].set_title('Aligned Oblique Shock Data\n(3D Slice at z = mid)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].set_xlim(-nx_max, nx_max)
    axes[1].set_ylim(0, ny_max)
    axes[1].set_aspect('equal')
    fig.colorbar(pcm1, ax=axes[1], label='Density')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()