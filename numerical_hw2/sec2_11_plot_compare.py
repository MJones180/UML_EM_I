# ==============================================================================
# Michael Jones (Michael_Jones6@student.uml.edu)
# [PHYS 6570] Electromagnetic Theory I
# Section 2.10 – Compare Plots
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np

X_TO_PLOT = [0.5]

# The npz file containing the numerical approach
NUMERICAL_FILE = 'numerical211.npz'

# The npz file containing the analytical approach
ANALYTICAL_FILE = 'analytical211.npz'

# Load in the data
numerical_file = np.load(NUMERICAL_FILE)
numerical_field = numerical_file['field']
n_x_coords = numerical_file['x_coords']
n_y_coords = numerical_file['y_coords']

analytical_file = np.load(ANALYTICAL_FILE)
analytical_field = analytical_file['field']
a_x_coords = analytical_file['x_coords']
a_y_coords = analytical_file['y_coords']

# Ensure the coordinates line up
if any(n_x_coords != a_x_coords) or any(n_y_coords != a_y_coords):
    print('Coords must match between the numerical and analytical runs.')
    quit()

for x_val in X_TO_PLOT:
    idx = np.argmax(n_x_coords == x_val)
    plt.plot(n_y_coords, numerical_field[:, idx], label=f'{x_val} Numerical')
    plt.plot(
        a_y_coords,
        analytical_field[:, idx],
        label=f'{x_val} Analytical',
        linestyle='dashed',
    )

plt.legend()
plt.title(f'x = {X_TO_PLOT}')
plt.xlabel('y-coordinates')
plt.ylabel('potential')
plt.show()
