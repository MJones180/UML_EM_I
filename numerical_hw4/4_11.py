# ==============================================================================
# Michael Jones (Michael_Jones6@student.uml.edu)
# [PHYS 6570] Electromagnetic Theory I
# Question 4.11
# Test of the Clausius-Mossotti relationship
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np

# NOTE: eps_r = epsilon / epsilon_0

# ==============================================================================
# Air setup
# ==============================================================================

# Relative density of air given in AIP Handbook (3rd ed, рage 4-165)
# Pressures in atm and temps in K
AIP_PRESSURES = (1, 4, 7, 10, 40, 70, 100)
AIP_TEMPS_200 = [1.3681, 5.511, 9.713, 13.976, 60.13, 112.66, 168.40]
AIP_TEMPS_300 = [0.9102, 3.644, 6.383, 9.125, 36.72, 64.34, 91.61]

# Linearly interpolate to 292 K
air_density_at_292 = [
    np.interp(292, [200, 300], [
        AIP_TEMPS_200[idx],
        AIP_TEMPS_300[idx],
    ]) for idx, _ in enumerate(AIP_PRESSURES)
]

# The pressures that we are going to compare to
AIR_PRESSURES = (20, 40, 60, 80, 100)
AIR_EPS_R = (1.0108, 1.0218, 1.0333, 1.0439, 1.0548)
# Linearly interpolate to each pressure
AIR_RELATIVE_DENSITY = np.interp(AIR_PRESSURES, AIP_PRESSURES,
                                 air_density_at_292)
# Values at 292 K, rows contain (pressure [atm], eps_r, relative density)
# Also, I know what you are thinking "Why did you define this as a constant when
# it is not really a constant?" Well, it just felt right...
AIR_TABLE = list(zip(AIR_PRESSURES, AIR_EPS_R, AIR_RELATIVE_DENSITY))

# ==============================================================================
# Pentane setup
# ==============================================================================

# Values at 303 K, rows contain (pressure [atm], eps_r, relative density)
PENTANE_TABLE = [
    (1, 1.82, 0.613),
    (1e3, 1.96, 0.701),
    (4e3, 2.12, 0.796),
    (8e3, 2.24, 0.865),
    (12e3, 2.33, 0.907),
]

# ==============================================================================
# Obtain results from the Clausius-Mossotti relationship
# ==============================================================================


def clausius_mossotti(eps_r):
    return (eps_r - 1) / (eps_r + 2)


air_cm_values = [clausius_mossotti(arr[1]) for arr in AIR_TABLE]
pentane_cm_values = [clausius_mossotti(arr[1]) for arr in PENTANE_TABLE]

# ==============================================================================
# Obtain results from the Cruder relationship
# ==============================================================================


def cruder(eps_r):
    return eps_r - 1


air_cruder_values = [cruder(arr[1]) for arr in AIR_TABLE]
pentane_cruder_values = [cruder(arr[1]) for arr in PENTANE_TABLE]

# ==============================================================================
# Plot out the Clausius-Mossotti relationship
# ==============================================================================


def plot_comparison(table, calc_density_cm, calc_density_cruder, mat='Air'):
    eps_vals = [arr[1] for arr in table]
    truth_density = [arr[2] for arr in table]
    # Code adapted from
    #    https://matplotlib.org/3.4.3/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin2.spines.right.set_position(('axes', 1.2))
    p1, = ax.plot(eps_vals, truth_density, 'b-')
    p2, = twin1.plot(eps_vals, calc_density_cm, 'r-')
    p3, = twin2.plot(eps_vals, calc_density_cruder, 'g-')
    ax.set_xlabel('epsilon / epsilon_0')
    ax.set_ylabel('Truth Relative Density')
    twin1.set_ylabel('Clausius-Mossotti Relative Density')
    twin2.set_ylabel('Cruder Relative Density')
    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())
    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)
    plt.title(f'Proportionality Comparison\n{mat}')
    plt.show()


plot_comparison(AIR_TABLE, air_cm_values, air_cruder_values)
plot_comparison(PENTANE_TABLE, pentane_cm_values, pentane_cruder_values,
                'Pentane')
