import random

# создание списка из 5-10 значений случайных чисел от 1 до 25 
lst = [random.randint(1, 25) for _ in range(random.randint(5, 10))]

print('Исходный список:', lst)

def useless(s):
    
    max_val = max(s)
    return max_val / len(s)

print('безопасное число:', useless(lst))
