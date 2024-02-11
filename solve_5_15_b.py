from sympy import var, solve
# Define the symbolic variables
A, B, C, D = var('A B C D')
a, b, d, I, mu_r, two_pi = var('a b d I mu_r two_pi')
# The equations that need to be solved
eq_1 = -A * a + B * a + C / a + I * d / (two_pi * a)
eq_2 = B * b + C / b - D / b
eq_3 = A - B * mu_r + C * mu_r / a**2 + I * d / (two_pi * a**2)
eq_4 = B * mu_r - C * mu_r / b**2 + D / b**2
solve_vars = [A, B, C, D]
# Solve for A, B, C, D
solutions = solve([eq_1, eq_2, eq_3, eq_4], solve_vars)
for solved_var in solve_vars:
    print(f'{solved_var} = ', solutions[solved_var].factor())
