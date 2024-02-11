# ==============================================================================
# Michael Jones (Michael_Jones6@student.uml.edu)
# [PHYS 6570] Electromagnetic Theory I
# Section 2.10 – Compare Plots
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np

Y_TO_PLOT = [0, 0.5]

# The npz file containing the numerical approach
NUMERICAL_FILE = 'numerical210.npz'

# The npz file containing the analytical approach
ANALYTICAL_FILE = 'analytical210.npz'

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

for y_val in Y_TO_PLOT:
    idx = np.argmax(n_y_coords == y_val)
    plt.plot(n_x_coords, numerical_field[idx], label=f'{y_val} Numerical')
    plt.plot(
        a_x_coords,
        analytical_field[idx],
        label=f'{y_val} Analytical',
        linestyle='dashed',
    )

plt.legend()
plt.title(f'y = {Y_TO_PLOT}')
plt.xlabel('x-coordinates')
plt.ylabel('potential')
plt.show()
