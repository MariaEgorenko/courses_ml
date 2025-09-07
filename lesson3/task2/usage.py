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

# Использование вектора

print("---Создание вектора---")
vec1 = Vector(generate_rand_list(10))
vec2 = Vector(generate_rand_list(10))
print(f"vec1 = {vec1}\nvec2 = {vec2}\n")

print("---Доступ к элементам вектора---")
print(f"vec1[3] = {vec1[3]}\n")

print("---Изменение элементов вектора---")
vec1[2] = 25
print(f"vec1[2] = 25\nvec1 = {vec1}\n")

print("Получение длины вектора")
print(f"Длина len(vec1) = {len(vec1)}\n")

print("---Удаление элеменита вектора по индексу---")
del vec1[7]
print(f"del vec1[7]\nvec1 = {vec1}\n")

print("---Добаление значения в конец вектора---")
vec1.append(4.5)
print(f"vec1.append(4.5)\nvec1 = {vec1}\n")

print("---Сложение векторов---")
vec3 = vec1 + vec2
print(f"vec3 = vec1 + vec2\nvec3 = {vec3}\n")

print("---Вычитание векторов---")
vec3 = vec1 - vec2
print(f"vec3 = vec1 - vec2\nvec3 = {vec3}\n")

print("---Умножение векотора на скаляр---")
vec3 = vec1 * 2.5
print(f"vec3 = vec1 * 2.5\nvec3 = {vec3}\n")

print("---Умножение векторов---")
print(f"vec1 * vec2 = {vec1 * vec2}\n")

print("---Получение размерности вектора---")
print(f"vec1.shape() = {vec1.shape()}\n")

print("---Сравнение векторов---")
vec3 = Vector(vec1.data)
print(f"vec1 = {vec1}\nvec2 = {vec2}\nvec3 = {vec3}")
print(f"vec1 == vec2: {vec1 == vec2}\nvec1 == vec3: {vec1 == vec3}\n")

print("---Добавление последовательности чисел к вектору---")
lst = generate_rand_list(3)
vec1.extend(lst)
print(f"vec1.extend(lst)\nvec1 = {vec1}\n")

# Использование матрицы

print("---Создание матриц---")
mat1 = Matrix([generate_rand_list(5) for _ in range(5)])
mat2 = Matrix([generate_rand_list(5) for _ in range(5)])
print(f"mat1 =\n{mat1}\nmat2 =\n{mat2}\n")

print("---Доступ к элементам матрицы---")
print(f"mat1[2] = {mat1[2]}\nmat2[3][1] = {mat2[3][1]}\n")

print("---Изменение элементов матрицыц---")
mat1[1] = [1, 1, 1, 1, 1]
print(f"mat1[1] = [1, 1, 1, 1, 1]\nmat1 =\n{mat1}")
mat1[1][2] = 9.5
print(f"mat1[1][2] = 9.5\nmat1 =\n{mat1}\n")

print("---Сложение матриц---")
mat3 = mat1 + mat2
print(f"mat3 = mat1 + mat2\nmat3 =\n{mat3}\n")

print("---Вычитание матриц---")
mat3 = mat1 - mat2
print(f"mat3 = mat1 - mat2\nmat3 =\n{mat3}\n")

print("---Умножение матрицы на число---")
mat3 = mat1 * 3.5
print(f"mat3 = mat1 * 3.5\nmat3 =\n{mat3}")

print("---Умножение матрицы на вектор---")
vec = Vector(generate_rand_list(5))
mat3 = mat1 * vec
print(f"vec = {vec}\nmat3 = mat1 * vec\nmat3 =\n{mat3}\n")

print("---Перемножение матриц---")
mat3 = mat1 @ mat2
print(f"mat3 = mat1 @ mat2\nmat3 =\n{mat3}\n")

print("---Получение размерности матрицы---")
print(f"mat1.shape() = {mat1.shape()}\n")

print("---Транспонирование матрицы---")
mat3 = Matrix([generate_rand_list(3) for _ in range(4)])
mat4 = mat3.transpose()
print(f"mat3 =\n{mat3}\nmat4 = mat3.transpose()\nmat4 =\n{mat4}\n")

print("---Проверка на содержание элемента в матрице---")
num = mat1[3][3]
print(f"mat1 =\n{mat1}\nnum = {num}\nnum in mat1 = {num in mat1}")
num = -100
print(f"num = {num}\nnum in mat1 = {num in mat1}\n")

print("---Сравнение двух матриц---")
mat4 = mat1
print(f"mat1 =\n{mat1}\nmat2 =\n{mat2}\nmat4 =\n{mat4}")
print(f"mat1 == mat2: {mat1 == mat2}\nmat1 == mat4: {mat1 == mat4}\n")

print("---Проверка на квадратную матрицу---")
print(f"mat1.is_square(): {mat1.is_square()}")
print(f"mat3.is_square(): {mat3.is_square()}\n")

print("---нахождение следа матрицы---")
trace_mat1 = mat1.trace()
print(f"mat1.trace() = {trace_mat1}\n")

print("---Добавление строки в конец матрицы---")
lst = generate_rand_list(5)
print(f"mat1 =\n{mat1}\nlst = {lst}")
mat1.append(lst)
print(f"mat1.append(lst)\nmat1 =\n{mat1}\n")

print("---Добавление нескольких строк в конец матрицы---")
mat = Matrix([generate_rand_list(5) for _ in range(2)])
print(f"mat =\n{mat}")
mat2.extend(mat)
print(f"mat2.append(mat)\nmat2 =\n{mat2}\n")

print("---Удаление строки матрицы---")
del mat2[5]
print(f"del mat2[5]\nmat2 =\n{mat2}\n")

print("---Удаление столбца матрицы---")
mat2.del_coloumn(3)
print(f"mat2.del_coloumn(3)\nmat2 =\n{mat2}")