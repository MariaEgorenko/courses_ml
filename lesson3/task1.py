import random


class Matrix:
    """Класс для представления матрицы и выполнения базовых оперций (сложение, умножение на скаляр)."""

    def __init__(self, data: list[list[int]]) -> None:
        """
        Инициализация матрицы.

        :param data: Двумерный список, представляющий матрицу.
        """
        if not data:
            raise ValueError("Некоректные данные!")
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def print_matrix(self):
        """Выводит матрицу на экран."""
        for r in self.data:
            print(r)

    def __add__(self, other: "Matrix") -> "Matrix":
        """
        Сложение двух матриц одинакового размера.

        :param other: Другая матрица типа Matrix.
        :return: Новая матрица типа Matrix - результат сложения. 
        """
        new_data = []
        for i in range(self.rows): 
            new_row = []

            for j in range(self.cols): 
                new_row.append(self.data[i][j] + other.data[i][j])

            new_data.append(new_row)

        return Matrix(new_data)
    
    def __mul__(self, number: int) -> "Matrix":
        """
        Умножеение матрицы на число.

        :param number: Скаляр (число).
        """
        new_data = []
        for i in range(self.rows):
            new_row = []

            for j in range(self.cols):
                new_row.append(self.data[i][j] * number)

            new_data.append(new_row)

        return Matrix(new_data)


print("Матрица 1:")
mat1 = Matrix([[random.randint(0, 20) for _ in range(5)] for _ in range(5)])
mat1.print_matrix()

print()
print("Матрица 2:")
mat2 = Matrix([[random.randint(0, 20) for _ in range(5)] for _ in range(5)])
mat2.print_matrix()

num = 3
print(f"\nУмножение матрицы на {num}:")
mat3 = mat1 * num
mat3.print_matrix()

print("\nСложение матрицы 1  матрицей 2:")
mat4 = mat1 + mat2
mat4.print_matrix()
