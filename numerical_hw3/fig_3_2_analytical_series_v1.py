# ==============================================================================
# Michael Jones (Michael_Jones6@student.uml.edu)
# [PHYS 6570] Electromagnetic Theory I
# Figure 3.2 – Analytical Series for Figure 3.2
# This script will keep computing terms until a threshold value is reached.
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# Constants
# ==============================================================================

# Bounds for the X axes
X_BOUNDS = [-1, 1]

# Step size determines how many points will be used. This directly affects the
# size of the calculations field
H_STEP_SIZE = 0.01

# Value before series is stopped
THRESHOLD = 1e-6

# The file in which the computed field will be saved
OUT_FILENAME = 'fig_3_2_analytical_series_v1.npz'

# ==============================================================================
# Initialize the coordinates and empty field
# ==============================================================================


def _init_coords(bounds):
    # Add one to the total number of steps so that the end bound is exclusive
    # and does not contribute to the step size
    total_steps = int(np.diff(bounds)[0] / H_STEP_SIZE) + 1
    # Create evenly spaced coordinate steps between the bounds
    coords = np.linspace(*bounds, total_steps)
    return total_steps, coords


x_steps, x_coords = _init_coords(X_BOUNDS)

# The field that will be filled in
field = np.full(x_steps, 1).astype('float32')

# ==============================================================================
# Iterate for the infinite series
# ==============================================================================

for x_idx, x_val in enumerate(x_coords):
    if x_idx == 0:
        field[x_idx] = -1
        continue

    def _compute_legendre_int(n):
        # Legendre of degree n integrated between 0 and 1
        basis = np.polynomial.legendre.Legendre.basis(n,
                                                      domain=[0, 1],
                                                      window=[0, 1])
        return np.sum(basis.integ(1).coef)

    # Equation 3.25
    def _compute_A_l(n):
        return (2 * n + 1) * _compute_legendre_int(n)

    def _compute_legendre_at_x(n):
        basis = np.polynomial.legendre.Legendre.basis(n)
        return basis.linspace(1, [x_val, x_val])[1][0]

    n = 1
    pixel_value = 0
    pixel_completed = False
    while not pixel_completed:
        # Equation 3.23
        iter_val = _compute_A_l(n) * _compute_legendre_at_x(n)
        pixel_value += iter_val
        if abs(iter_val) < THRESHOLD or n > 100:
            break
        n += 2

    field[x_idx] = pixel_value

# ==============================================================================
# Plot out the final field using mpl
# ==============================================================================

plt.plot(x_coords, field)
plt.title('Analytical Infinite Series')
plt.show()

# ==============================================================================
# Save the computed field
# ==============================================================================

np.savez(OUT_FILENAME, x_coords=x_coords, field=field)
