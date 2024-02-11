# ==============================================================================
# Michael Jones (Michael_Jones6@student.uml.edu)
# [PHYS 6570] Electromagnetic Theory I
# Problem 1.21 – Relaxation Algorithm Approach
# ==============================================================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ==============================================================================
# Constants
# ==============================================================================

# Display the text version of the field
TEXT_FIELD = False

# Bounds for the X and Y axes
X_BOUNDS = [0, 1]
Y_BOUNDS = [0, 1]

# Step size determines how many boxes will be used. This directly affects the
# size of the calculations field
H_STEP_SIZE = 0.05

# The total iterations in the relaxation algorithm
ITERATIONS = 3000

# The second half of the equation used to average points in the relaxation
# algorithm is constant since the density across all points is constant:
#   (DENSITY * H_STEP_SIZE^2 / (4 * EPSILON_0))
# EPSILON_0 has units of F * m^-1 (farads per meter)
RELAXATION_CONSTANT = 1 * H_STEP_SIZE**2 / (4 * 8.8541878128e-12)

# The file in which the computed field will be saved
OUT_FILENAME = 'relaxation_field.npz'

# ==============================================================================
# Initialize the coordinates
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

# ==============================================================================
# Initialize the field and empty vectors
# ==============================================================================

# The field that calculations will be done within
field = np.full((x_steps, y_steps), 1)

# Empty vectors that will be used to perform iterations faster
empty_x_row = np.full(y_steps, 0)[None, :]
empty_y_col = np.full(x_steps, 0)[:, None]

# ==============================================================================
# Set the boundaries of the potential
# ==============================================================================

# The sides of the square are the boundaries, so that means all four sides
# must be set to zero potential. The sides are set to zero potential in the
# order of top, bottom, left, and right.
field[0] = 0
field[-1] = 0
field[0:, 0] = 0
field[0:, -1] = 0

# ==============================================================================
# Set the mask of all points inside of the boundary
# ==============================================================================

# This mask will allow us to determine which points should be updated after each
# iteration in the relaxation algorithm. Since this is a very simple example,
# the field only consists of boundary points and inside points. Due to this, we
# know that if a point is not on the boundary, it must be inside of the
# boundary. This is because of the fact that there cannot be any points outside
# of the boundary for this example. Using this simple logic, we can determine
# that any points with a value of `1` must be inside of the boundary (all
# boundary points were set to `0`).
inner_mask = field == 1

# ==============================================================================
# Prepare the field for iterations
# ==============================================================================

# Conver the field to floats so that they can actually be non-integer values
field = field.astype('float32')

# ==============================================================================
# Iterate for the relaxation algorithm
# ==============================================================================

if TEXT_FIELD:
    print('Initial Field:')
    print(field)

for i in range(ITERATIONS):
    # The four directions specify which side has an empty vector shifting the
    # values over. So, `top` means that the top row has an empty vector and the
    # other rows are all being shifted down one.
    top = np.concatenate((empty_x_row, field[:-1]))
    bottom = np.concatenate((field[1:], empty_x_row))
    left = np.concatenate((empty_y_col, field[:, :-1]), axis=1)
    right = np.concatenate((field[:, 1:], empty_y_col), axis=1)
    # Do for each of the points at the same time:
    #   V_i+1,j + V_i-1,j + V_i,j+1 + V_i,j-1
    summed = top + bottom + left + right
    # Each new point can be calculated as:
    #   0.25 * summed + RELAXATION_CONSTANT
    updated = 0.25 * summed + RELAXATION_CONSTANT
    field[inner_mask] = updated[inner_mask]

if TEXT_FIELD:
    print(f'After {ITERATIONS} iterations:')
    print(field)

# ==============================================================================
# Plot out the final field using mpl
# ==============================================================================

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
ax.plot_surface(x_mesh, y_mesh, field)
plt.title('Relaxation Algorithm')
plt.show()

# ==============================================================================
# Save the computed field
# ==============================================================================

np.savez(OUT_FILENAME, x_coords=x_coords, y_coords=y_coords, field=field)
