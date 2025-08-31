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
print('Корни уровнения 5: y^2 -12y + 20 = 0\n', solution)

# Уравнение 6: z^2 +17z +72 = 0
solution = solve_quadratic_equation(1, 17, 20)
print('Корни уравнения 6: z^2 +17z +72 = 0\n', solution)

# Уравнение 7: x^2 - 7x - 44 = 0
solution = solve_quadratic_equation(1, -7, -44)
print('Корни уравнения 7: x^2 - 7x - 44 = 0\n', solution)

# Уравнение 8: y^2 + 9y + 8 = 0
solution = solve_quadratic_equation(1, 9, 8)
print('Корни уравнения 8: y^2 + 9y + 8 = 0\n', solution)

# Уравнение 9: b^2 - 2b - 63 = 0
solution = solve_quadratic_equation(1, -2, -63)
print('Корни уравнения 9: b^2 - 2b - 63 = 0\n', solution)

# Решение уравнения (x^2 - 8)^2 + 4(x^2 - 8) - 5 = 0
# Делаем замену переменной u = x^2 - 8 
# Решаем уравнене u^2 + 4u - 5 = 0
solution = solve_quadratic_equation(1, 4, -5)

solution_2 = []
for u in solution:
  # u = x^2 - 8 -> x^2 = u + 8
  x_sqrt = u + 8
  
  # если x^2 < 0 -> нет корней
  if x_sqrt < 0:
    continue
  # при x^2 = 0 -> один нулевой корень 
  elif x_sqrt == 0:
    solution_2.append(0)
  else:
    x1 = math.sqrt(x_sqrt)
    solution_2.append(x1)
    solution_2.append(-x1)

print('Корни уравнения (x^2 - 8)^2 + 4(x^2 - 8) - 5 = 0\n', solution_2)