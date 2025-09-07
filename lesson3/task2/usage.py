from random import randint, uniform
from typing import List, Union, Type

from vector import Vector
from matrix import Matrix

def generate_rand_list(
        length: int,
        start: int = 0,
        end: int = 20,
        value_type: Type[Union[int, float]] = int
    ) -> List[Union[int, float]]:
    """
    Возвращает список случайных чисел по заданным параметрам.

    :param length: Длина списка
    :param start: Начало диапазона значений (по умолчанию 0)
    :param end: Конец диапазона значений включительно (по умолчанию 20)
    :param type: тип значений возвращаемого списка – int или float
        (по умолчанию int)
    :raise ValueError: Если длина меньше 1 или start > end
    :raise TypeError: Если start, end и length не целые числа
        или type_value не int или float
    """
    if (
        not isinstance(length, int)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError(
            "Значение длины, начала диапазона или окончания диапазона "
            "должны быть целыми числами"
        )
    if start > end:
        raise ValueError(
            "Начало диапазона значений должно быть меньше или равно "
            "концу диапазна"
            )
    if value_type is int:
        return [randint(start, end) for _ in range(length)]
    elif value_type is float:
        return [uniform(start, end) for _ in range(length)]
    else:
        raise TypeError("Передаваемый тип должен быть int или float")

# Создание вектора
vec1 = Vector(generate_rand_list(10))
vec2 = Vector(generate_rand_list(10))
print(f"vec1 = {vec1}\nvec2 = {vec2}\n")

# Доступ к элементам вектора
print(f"vec1[3] = {vec1[3]}\n")

# Изменение элементов вектора
vec1[2] = 25
print(f"vec1[2] = 25\nvec1 = {vec1}\n")

# Получение длины вектора
print(f"Длина vec1 = {len(vec1)}\n")

# Удаление элеменита вектора по индексу
del vec1[7]
print(f"del vec1[7]\nvec1 = {vec1}\n")

# Добаление значения в конец вектора
vec1.append(4.5)
print(f"vec1.append(4.5)\nvec1 = {vec1}\n")

# Сложение векторов
vec3 = vec1 + vec2
print(f"vec3 = vec1 + vec2\nvec3 = {vec3}\n")

# Вычитание векторов
vec3 = vec1 - vec2
print(f"vec3 = vec1 - vec2\nvec3 = {vec3}\n")

# Умножение векотора на скаляр
vec3 = vec1 * 2.5
print(f"vec3 = vec1 * 2.5\nvec3 = {vec3}\n")

# Умножение векторов
print(f"vec1 * vec2 = {vec1 * vec2}\n")

# Получение размерности вектора
print(f"vec1.shape() = {vec1.shape()}\n")

# Сравнение векторов
vec3 = Vector(vec1.data)
print(f"vec1 = {vec1}\nvec2 = {vec2}\nvec3 = {vec3}")
print(f"vec1 == vec2: {vec1 == vec2}\nvec1 == vec3: {vec1 == vec3}\n")

# Добавление последовательности чисел к вектору
lst = generate_rand_list(3)
vec1.extend(lst)
print(f"vec1.extend(lst)\nvec1 = {vec1}\n")