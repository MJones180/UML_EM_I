from sympy import Abs, besselj, besselk, cos, exp, integrate, symbols
import numpy as np

a, k, z = symbols('a k z', positive=True)
print('Part A')
print(integrate(besselk(1, k * a) * cos(k * z) * k, (k, 0, np.inf)))

print('Part B')
print(integrate(besselj(1, k * a) * exp(-k * Abs(z)) * k, (k, 0, np.inf)))
