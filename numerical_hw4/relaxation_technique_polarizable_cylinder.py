# ==============================================================================
# Michael Jones (Michael_Jones6@student.uml.edu)
# [PHYS 6570] Electromagnetic Theory I
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
X_BOUNDS = [-1, 1]
Y_BOUNDS = [-1, 1]

# Radius of the circle centered at the middle of the field. This must not flow
# outside of the field, otherwise the program will crash (it does NOT perform
# checks to verify that the radius is acceptable).
RADIUS = 0.75

# Step size determines how many boxes will be used. This directly affects the
# size of the calculations field
H_STEP_SIZE = 0.005

# The total iterations in the relaxation algorithm
ITERATIONS = 5000

# Divide by epsilon
ADD_EPSILON_SCALING = False

# Zero charge density
NO_CHARGE_DENSITY = True

# The file in which the computed field will be saved
OUT_FILENAME = 'relaxation.npz'

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

# The field that calculations will be done within.
# This represents the electric potential.
field = np.full((y_steps, x_steps), np.inf)

# Empty vectors that will be used to perform iterations faster
empty_x_col = np.full(y_steps, 0)[:, None]
empty_y_row = np.full(x_steps, 0)[None, :]

# ==============================================================================
# Initialize the epsilon field
# ==============================================================================

# Epsilon will default to a value of 1
epsilon_field = np.full((y_steps, x_steps), 1)

# ==============================================================================
# Set the boundaries of the electric potential and epsilon
# ==============================================================================


def _find_nearest_idx(coords, val):
    return (np.abs(coords - val)).argmin()


mid_x_coord_idx = (x_steps - 1) // 2
mid_x_coord = x_coords[mid_x_coord_idx]
mid_y_coord = y_coords[(y_steps - 1) // 2]
# Solving for y: y = sqrt(a^2 - x^2)
for rel_x_idx, x_idx in enumerate(np.arange(x_steps)[mid_x_coord_idx:]):
    x_val = x_coords[x_idx]
    rel_x_val = np.abs(x_val - mid_x_coord)
    if rel_x_val > RADIUS:
        break
    y_val = np.sqrt(RADIUS**2 - rel_x_val**2)
    y_idx_below = _find_nearest_idx(y_coords, mid_y_coord - y_val)
    y_idx_above = _find_nearest_idx(y_coords, mid_y_coord + y_val)
    # Pixels to the right of the middle are given by `x_idx`
    # Pixels to the left of the middle are given by `x_idx - (rel_x_idx * 2)`
    l_x_idx = x_idx - (rel_x_idx * 2)
    # The inside of the cylinder
    field[y_idx_below:y_idx_above, x_idx] = 0
    field[y_idx_below:y_idx_above, l_x_idx] = 0
    # The boundaries of the cylinder
    field[y_idx_below, x_idx] = 10
    field[y_idx_above, x_idx] = 10
    field[y_idx_below, l_x_idx] = 10
    field[y_idx_above, l_x_idx] = 10
    # The epsilon inside the cylinder
    epsilon_field[y_idx_below:y_idx_above, x_idx] = 5
    epsilon_field[y_idx_below:y_idx_above, l_x_idx] = 5

# ==============================================================================
# Set the mask of all points inside of the boundary and make them finite
# ==============================================================================

# This mask will allow us to determine which points should be updated after each
# iteration in the relaxation algorithm. We can determine that any points with a
# value of `np.inf` must be the boundary.
inner_mask = field != np.inf
# Turn the boundary points back into finite values
field[field == np.inf] = 0

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

# The second half of the equation used to average points in the relaxation
# algorithm is constant since the density across all points is constant:
#   DENSITY * H_STEP_SIZE^2 / EPSILON_0 [Optional]
relaxation_constant = H_STEP_SIZE**2
if ADD_EPSILON_SCALING:
    # Epislon has units of F * m^-1 (farads per meter)
    relaxation_constant /= 8.8541878128e-12
# No charge density present
if NO_CHARGE_DENSITY:
    relaxation_constant = 0

for i in range(ITERATIONS):
    # The four directions specify which side has an empty vector shifting the
    # values over. So, `top` means that the top row has an empty vector and the
    # other rows are all being shifted down one.
    top = np.concatenate((empty_y_row, field[:-1]))
    bottom = np.concatenate((field[1:], empty_y_row))
    left = np.concatenate((empty_x_col, field[:, :-1]), axis=1)
    right = np.concatenate((field[:, 1:], empty_x_col), axis=1)

    # The same now for the epsilon field, except for epsilon is only a half
    # step which means the points needs to be averaged together
    def _find_avg_eps_field(shifted_tuple, axis=0):
        return (epsilon_field + np.concatenate(shifted_tuple, axis=axis)) / 2

    top_eps = _find_avg_eps_field((empty_y_row, epsilon_field[:-1]))
    bottom_eps = _find_avg_eps_field((epsilon_field[1:], empty_y_row))
    left_eps = _find_avg_eps_field((empty_x_col, epsilon_field[:, :-1]), 1)
    right_eps = _find_avg_eps_field((epsilon_field[:, 1:], empty_x_col), 1)

    # Do all the points at the same time:
    # (   V_i+1,j * eps_i+0.5,j
    #   + V_i-1,j * eps_i-0.5,j
    #   + V_i,j+1 * eps_i,j+0.5
    #   + V_i,j-1 * eps_i,j-0.5
    #   + relaxation_constant
    # ) / (   eps_i+0.5,j
    #       + eps_i-0.5,j
    #       + eps_i,j+0.5
    #       + eps_i,j-0.5 )
    summed = (top * top_eps + bottom * bottom_eps + left * left_eps +
              right * right_eps + relaxation_constant) / (
                  top_eps + bottom_eps + left_eps + right_eps)
    field[inner_mask] = summed[inner_mask]

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

ax.set_box_aspect(aspect=(X_BOUNDS[1], Y_BOUNDS[1], 1))
plt.title('Relaxation Algorithm')
plt.show()

# ==============================================================================
# Save the computed field
# ==============================================================================

np.savez(OUT_FILENAME, x_coords=x_coords, y_coords=y_coords, field=field)
