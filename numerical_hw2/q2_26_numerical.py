# ==============================================================================
# Michael Jones (Michael_Jones6@student.uml.edu)
# [PHYS 6570] Electromagnetic Theory I
# Question 2.26 – Relaxation Algorithm Approach
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
X_BOUNDS = [0, 2]
Y_BOUNDS = [0, 1]

# Step size determines how many boxes will be used. This directly affects the
# size of the calculations field
H_STEP_SIZE = 0.02

# The total iterations in the relaxation algorithm
ITERATIONS = 5000

# Divide by epsilon
ADD_EPSILON_SCALING = False

# Angle of beta in degrees
BETA_ANGLE = 35

# Radius of the circle
A_RAD = 0.5

# The file in which the computed field will be saved
OUT_FILENAME = 'numerical226.npz'

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
field = np.full((y_steps, x_steps), np.inf)

# Empty vectors that will be used to perform iterations faster
empty_x_col = np.full(y_steps, 0)[:, None]
empty_y_row = np.full(x_steps, 0)[None, :]

# ==============================================================================
# Set the boundaries of the potential
# ==============================================================================

# Set the bottom to be zero potential
field[0] = 0


def _find_nearest_idx(coords, val):
    return (np.abs(coords - val)).argmin()


# Set all pixels on or above the slope to be a boundary
slope = np.tan(np.radians(BETA_ANGLE))
for x_idx, x_val in enumerate(x_coords):
    y_val = slope * x_val
    # If the slope has continued outside of the Y bounds, end the line
    if y_val > Y_BOUNDS[1]:
        continue
    y_idx = _find_nearest_idx(y_coords, y_val)
    field[y_idx:, x_idx] = 0

# Circle about the origin: x^2 + y^2 = a^2
# Solving for y: y = sqrt(a^2 - x^2)
x_circle_max = _find_nearest_idx(x_coords, A_RAD) + 1
for x_idx, x_val in enumerate(x_coords[:x_circle_max]):
    y_val = np.sqrt(A_RAD**2 - x_val**2)
    y_idx = _find_nearest_idx(y_coords, y_val)
    field[:y_idx, x_idx] = 0

field[:, -1] = 5

# ==============================================================================
# Set the mask of all points inside of the boundary and make them finite
# ==============================================================================

# This mask will allow us to determine which points should be updated after each
# iteration in the relaxation algorithm. We can determine that any points with a
# value of `inf` must not be a boundary.
inner_mask = field == np.inf
# Turn the inner points back into finite values
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

# For some reason, adding in the H_STEP_SIZE**2 actually skewes the results
relaxation_constant = 0

for i in range(ITERATIONS):
    # The four directions specify which side has an empty vector shifting the
    # values over. So, `top` means that the top row has an empty vector and the
    # other rows are all being shifted down one.
    top = np.concatenate((empty_y_row, field[:-1]))
    bottom = np.concatenate((field[1:], empty_y_row))
    left = np.concatenate((empty_x_col, field[:, :-1]), axis=1)
    right = np.concatenate((field[:, 1:], empty_x_col), axis=1)
    # Do all the points at the same time:
    #   (V_i+1,j + V_i-1,j + V_i,j+1 + V_i,j-1 + relaxation_constant) / 4
    summed = (top + bottom + left + right + relaxation_constant) / 4
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

# Add a line that represents the other wall
slope_x = np.array([x for x in x_coords])
slope_y = np.array([x * slope for x in x_coords])
slope_mask = slope_y <= Y_BOUNDS[1]
slope_x = slope_x[slope_mask]
slope_y = slope_y[slope_mask]
ax.plot(slope_x, slope_y, zorder=5)

circle_x = np.array([x for x in x_coords[:x_circle_max]])
circle_y = np.array([np.sqrt(A_RAD**2 - x**2) for x in circle_x])
ax.plot(circle_x, circle_y, zorder=5)

ax.set_box_aspect(aspect=(X_BOUNDS[1], Y_BOUNDS[1], 1))
plt.title('Relaxation Algorithm')
plt.show()

# ==============================================================================
# Save the computed field
# ==============================================================================

np.savez(OUT_FILENAME, x_coords=x_coords, y_coords=y_coords, field=field)
