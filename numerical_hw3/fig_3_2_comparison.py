# ==============================================================================
# Michael Jones (Michael_Jones6@student.uml.edu)
# [PHYS 6570] Electromagnetic Theory I
# Section 2.10 – Compare Plots
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np

TERMS_TO_PLOT = [10, 50, 100]

# The npz file containing the analytical approach
ANALYTICAL_FILE = 'fig_3_2_analytical_series_v2.npz'

# Load in the data
analytical_file = np.load(ANALYTICAL_FILE)
analytical_field = analytical_file['field']
x_coords = analytical_file['x_coords']

# Step function (truth values)
y_truth = np.zeros(len(x_coords))
y_truth[x_coords < 0] = -1
y_truth[x_coords > 0] = 1

for n_term in TERMS_TO_PLOT:
    plt.plot(
        x_coords,
        analytical_field[n_term - 1] - y_truth,
        label=f'{n_term} Terms',
    )

plt.legend()
plt.title('Difference Between Step Function vs Analytical Series')
plt.xlabel('Difference')
plt.ylabel('x-coords')
plt.show()
