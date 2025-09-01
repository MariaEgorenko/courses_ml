import math

# раздлим уравнение на несколько частей
# 1: a = cos(e^x) + ln(1 + x)^2
# 2: b = sqrt(e^(cos(x)) + sin^2(pi*x)
# 3: c = sqrt(1/x) + cos(x^2)
# y = (a + b + c)^sin(x)
x = 1.79
a = math.cos(math.exp(x)) + (math.log(1 + x))**2
b = math.sqrt(math.exp(math.cos(x))) + math.sin(math.pi*x)**2
c = math.sqrt(1/x) + math.cos(x**2)

y = (a + b + c)**math.sin(x)

print('Значение функции =', y)