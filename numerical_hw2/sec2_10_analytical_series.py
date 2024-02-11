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
X_BOUNDS = [0, 1]
Y_BOUNDS = [0, 5]

# Step size determines how many boxes will be used. This directly affects the
# size of the calculations field
H_STEP_SIZE = 0.02

# Value before series is stopped
THRESHOLD = 1e-10
# Use only the first term in the series
THRESHOLD = np.inf

# The file in which the computed field will be saved
OUT_FILENAME = 'analytical210.npz'

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
field = np.full((y_steps, x_steps), 1).astype('float32')

# ==============================================================================
# Iterate for the infinite series
# ==============================================================================

for x_idx, x_val in enumerate(x_coords):
    # Handle the case of y = 0 differently, this is the boundary condition
    field[0, x_idx] = np.sin(np.pi * x_val / X_BOUNDS[1])

    for y_idx, y_val in enumerate(y_coords[1:]):

        def _analytical_series_iter(n):
            shared = n * np.pi / X_BOUNDS[1]
            return np.exp(-shared * y_val) * np.sin(shared * x_val) / n

        n = 1
        pixel_value = 0
        pixel_completed = False
        while not pixel_completed:
            iter_val = _analytical_series_iter(n)
            pixel_value += iter_val
            if iter_val < THRESHOLD:
                break
            n += 2
        # Scaling like in the textbook skewes the matchup
        # pixel_value *= 4 / np.pi
        field[y_idx + 1, x_idx] = pixel_value

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
