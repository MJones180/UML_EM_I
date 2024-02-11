# ==============================================================================
# Michael Jones (Michael_Jones6@student.uml.edu)
# [PHYS 6570] Electromagnetic Theory I
# Figure 3.2 – Analytical Series for Figure 3.2
# This script will compute all x values up until n terms. All intermediary terms
# will be saved and written out. This code should be pretty fast since
# everything is vectorized, it will just be a bit more memory intensive.
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

# Number of terms before computations stop
N_TERMS = 100

# The file in which the computed field will be saved
OUT_FILENAME = 'fig_3_2_analytical_series_v2.npz'

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

# The field that will be filled in, each row corresponds to a different
# term value
field = np.full([N_TERMS, x_steps], 1).astype('float32')

# ==============================================================================
# Iterate for the infinite series
# ==============================================================================

Legendre = np.polynomial.legendre.Legendre

for n_idx in range(N_TERMS):
    # Legendre of degree n integrated between 0 and 1
    legendre_int = np.sum(
        Legendre.basis(
            n_idx,
            domain=[0, 1],
            window=[0, 1],
        ).integ(1).coef)
    # Equation 3.25
    A_n = (2 * n_idx + 1) * legendre_int
    # Equation 3.23
    x_vals = A_n * Legendre.basis(n_idx).linspace(x_steps, X_BOUNDS)[1]
    # Add on the previous terms
    if n_idx > 0:
        x_vals += field[n_idx - 1, :]
    field[n_idx, :] = x_vals

# Field was between 0 and 2
field -= 1

# ==============================================================================
# Plot out the final field using mpl
# ==============================================================================

plt.plot(x_coords, field[-1])
plt.title('Analytical Infinite Series')
plt.xlabel('X-Coordinates')
plt.ylabel('f(x)')
plt.show()

# ==============================================================================
# Save the computed field
# ==============================================================================

np.savez(OUT_FILENAME, x_coords=x_coords, field=field)
