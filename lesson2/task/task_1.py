import math

# Решение квадратного уравнения ax^2 + bx + c = 0
def solve_quadratic_equation(a, b, c):

  d = b**2 -4*a*c
  if d < 0:
    return None
  elif d == 0:
    x = -b / (2*a)
    return x
  else:
    sqrt_d = math.sqrt(d)
    x1 = (-b + sqrt_d) / (2*a)
    x2 = (-b - sqrt_d) / (2*a)
    return [x1, x2]

# Уравнение 5: y^2 -12y + 20 = 0
solution = solve_quadratic_equation(1, -12, 20)
print('Корни уровнения 5:', solution)

# Уравнение 6: z^2 +17z +72 = 0
solution = solve_quadratic_equation(1, 17, 20)
print('Корни уравнения 6:', solution)

# Уравнение 7: x^2 - 7x - 44 = 0
solution = solve_quadratic_equation(1, -7, -44)
print('Корни уравнения 7:', solution)

# Уравнение 8: y^2 + 9y + 8 = 0
solution = solve_quadratic_equation(1, 9, 8)
print('Корни уравнения 8:', solution)

# Уравнение 9: b^2 - 2b - 63 = 0
solution = solve_quadratic_equation(1, -2, -63)
print('Корни уравнения 9:', solution)
