# ==============================================================================
# Michael Jones (Michael_Jones6@student.uml.edu)
# [PHYS 6570] Electromagnetic Theory I
# Problem 1.21 Plots
# ==============================================================================
from math import cosh, pi, sin
import matplotlib.pyplot as plt
import numpy as np

# The y-values we want to calculate for
Y_CONSTANT = [0.25, 0.5]

# X and Y bounds to generate the mesh between
X_LOW = 0.0000
X_HIGH = 1
# The increment between the x-bounds
X_STEP = 0.05

# How small the value must be from the iteration before stopping
THRESHOLD = 1e-8

# Units of F * m^-1 (farads per meter)
# EPSILON_0 = 8.8541878128e-12
EPSILON_0 = 8.8541878128e-12

# Constant `A` for the variational solution
A_VARIATIONAL = 5 / (4 * EPSILON_0)

# Constant value that the summed series should be multiplied by
SERIES_CONSTANT = 4 / (pi**3 * EPSILON_0)

# The npz file containing the Relaxation Algorithm output.
# The `relaxation_1.21.py` script must be run first.
RELAXATION_FILE = 'relaxation_field.npz'

# Load in the relaxation data
relaxation_file = np.load(RELAXATION_FILE)
relaxation_data = {
    'x_coords': relaxation_file['x_coords'],
    'y_coords': relaxation_file['y_coords'],
    'field': relaxation_file['field'],
}

# All the x-coordinate locations
x_coords = list(np.arange(X_LOW, X_HIGH + X_STEP, X_STEP))
x_coord_count = len(x_coords)
# Keep track of the results for each x-value being calculated
f_vals_series = {y: [] for y in Y_CONSTANT}
f_vals_variational = {y: [] for y in Y_CONSTANT}
f_vals_relaxation = {y: [] for y in Y_CONSTANT}

# Ensure the x-coordinates line up
if any(relaxation_data['x_coords'] != x_coords):
    print('X coordinates do not match between this file and the relaxation.')
    quit()

for y in Y_CONSTANT:
    print('====================')
    print(f'y = {y}')
    # Loop through each x coordinate
    for idx, x in enumerate(x_coords):
        print(f'On x-coord {idx + 1}/{x_coord_count}')
        m = 0
        running_sum = 0

        # Calculate one m value of the series
        def calc_series(m):
            m2 = (2 * m) + 1
            first_chunk = sin(m2 * pi * x) / m2**3
            second_chunk = cosh(m2 * pi * (y - 0.5)) / cosh(m2 * pi / 2)
            return first_chunk * (1 - second_chunk)

        # Keep looping until a value smaller than the threshold is reached.
        # This is needed for the exact solution
        while True:
            value_at_m = calc_series(m)
            running_sum += value_at_m
            if value_at_m < THRESHOLD:
                print(f'\tm = {m}')
                break
            m += 1
        f_val_series = running_sum * SERIES_CONSTANT
        f_vals_series[y].append(f_val_series)

        f_val_variational = A_VARIATIONAL * x * (1 - x) * y * (1 - y)
        f_vals_variational[y].append(f_val_variational)

    idx = np.argmax(relaxation_data['y_coords'] == y)
    f_vals_relaxation[y] = relaxation_data['field'][:, idx]

for key, val in f_vals_series.items():
    plt.plot(x_coords, val, label=f'{key} Series')
for key, val in f_vals_variational.items():
    plt.plot(x_coords, val, label=f'{key} Variational', linestyle='dashed')
for key, val in f_vals_relaxation.items():
    plt.plot(x_coords, val, label=f'{key} Relaxation', linestyle='dotted')
plt.legend()
plt.title(f'y = {Y_CONSTANT}')
plt.xlabel('x-coordinates')
plt.ylabel('phi(x, y)')
plt.show()
