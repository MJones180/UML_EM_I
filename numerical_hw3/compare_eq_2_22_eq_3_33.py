# ==============================================================================
# Michael Jones (Michael_Jones6@student.uml.edu)
# [PHYS 6570] Electromagnetic Theory I
# Comparison of results from equations 2.22 and 3.33
# Potential outside of a sphere along the z-axis with boundary held at fixed V.
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# Constants
# ==============================================================================

# Step size determines how many points will be used
STEP_SIZE = 0.01

# Radius of the circle
RADIUS = 1
RADIUS_SQ = RADIUS**2

# Bounds on the values that are computed
BOUNDS = [RADIUS, 20]

# Number of terms used in the calculation for the infinite series in 3.33
N_TERMS = 100

# ==============================================================================
# Initialize the coordinates
# ==============================================================================

# Add one to the total number of steps so that the end bound is exclusive
# and does not contribute to the step size
steps = int(np.diff(BOUNDS)[0] / STEP_SIZE) + 1

# Create evenly spaced coordinate steps between the bounds
coords = np.linspace(*BOUNDS, steps)

# ==============================================================================
# Equation 2.22
# ==============================================================================

numerator = coords**2 - RADIUS_SQ
denominator = coords * (coords**2 + RADIUS_SQ)**0.5
field_2_22 = 1 - (numerator / denominator)

# ==============================================================================
# Equation 3.33 – Infinite series solution
# ==============================================================================

field_3_33 = np.full(steps, 0).astype('float32')
for term, n_idx in enumerate(range(1, N_TERMS * 2, 2)):
    rad_over_z = (RADIUS / coords)**(n_idx + 1)
    # Technically, we should be dividing by the LCD of each Legendre polynomial,
    # but I could not figure out how to easily obtain this.
    coef = ((2 * n_idx) + 1) / (2**n_idx)
    term_val = rad_over_z * coef
    if term % 2 == 1:
        term_val *= -1
    field_3_33 += term_val

# ==============================================================================
# Plot out the final comparison using mpl
# ==============================================================================

plt.plot(coords, field_2_22, label='2.22')
plt.plot(coords, field_3_33, label='3.33')
plt.title('2.22 vs 3.33')
plt.xlabel('Z-Coordinates')
plt.ylabel('Potential')
plt.legend()
plt.show()
