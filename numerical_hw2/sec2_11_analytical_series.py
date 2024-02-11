# ==============================================================================
# Michael Jones (Michael_Jones6@student.uml.edu)
# [PHYS 6570] Electromagnetic Theory I
# Section 2.10 – Analytical Series Approach
# ==============================================================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ==============================================================================
# Constants
# ==============================================================================

# Bounds for the X and Y axes
X_BOUNDS = [0, 2]
Y_BOUNDS = [0, 1]

# Step size determines how many boxes will be used. This directly affects the
# size of the calculations field
H_STEP_SIZE = 0.02

# Angle of beta in degrees
BETA_ANGLE = 35

# The file in which the computed field will be saved
OUT_FILENAME = 'analytical211.npz'

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
y_steps, y_coords = _init_coords(Y_BOUNDS)

# The field that will be filled in
field = np.full((y_steps, x_steps), 0).astype('float32')

# ==============================================================================
# Use just the first term in the series
# ==============================================================================

for x_idx, x_val in enumerate(x_coords[1:]):
    for y_idx, y_val in enumerate(y_coords[1:]):
        rho = np.sqrt(x_val**2 + y_val**2)
        phi = np.arctan(y_val / x_val)
        beta = np.radians(BETA_ANGLE)
        pi_over_beta = np.pi / beta
        if phi > beta:
            pixel_value = 0
        else:
            pixel_value = rho**(pi_over_beta) * np.sin(phi * pi_over_beta)
        field[y_idx, x_idx] = pixel_value

# ==============================================================================
# Plot out the final field using mpl
# ==============================================================================

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
ax.plot_surface(x_mesh, y_mesh, field)
ax.set_box_aspect(aspect=(X_BOUNDS[1], Y_BOUNDS[1], 1))
plt.title('Analytical Infinite Series')
plt.show()

# ==============================================================================
# Save the computed field
# ==============================================================================

np.savez(OUT_FILENAME, x_coords=x_coords, y_coords=y_coords, field=field)
