from .base_array import BaseArray
from .vector import Vector
from typing import List, Union, Any, Tuple


class Matrix(BaseArray):
    """
    Класс для представления двумерной числовой структуры — матрицы.
    Поддерживает операции сложения, умножения (на скаляр, вектор и матрицу),
    транспонирования, доступа по индексам, итерации и сравнения.
    """

    def __init__(self, data: List[List[Union[int, float]]]) -> None:
        """
        Инициализирует матрицу из двумерного списка.

        :param data: Список списков чисел (int или float). Все строки должны иметь одинаковую длину.
        :raises TypeError: Если data не список, или элементы не списки, или содержат не числа.
        :raises ValueError: Если строки имеют разную длину.
        """
        if not isinstance(data, list):
            raise TypeError("Список должен состоять из списков")
        if len(data) == 0:
            raise ValueError("Матрица не может быть пустой")
        
        count_cols = len(data[0])
        for i, row in enumerate(data):
            if not isinstance(row, list):
                raise TypeError(f"Строка {i} должна быть списком")
            if len(row) != count_cols:
                raise ValueError(f"Строка {i} имеет длину {len(row)}, ожидалась {count_cols}")
            if not all(isinstance(x, (int, float)) for x in row):
                raise TypeError("Все элементы матрицы должны быть числами")
            
        super.__init__(data)

    def __len__(self) -> int:
        """
        Возвращает общее количество элементов в матрице.

        :return: Количество элементов (строки * столбцы)
        """
        return len(self._data) * len(self._data[0])
    
    def __getitem__(self, index: Union[int, Tuple[int, int]]) -> Union[List[Union[int, float]], Union[int, float]]:
        """
        Возвращает строку по индексу или элемент по координатам (i, j).

        :param index: Целое число (для получения строки) или кортеж (i, j) для получения элемента
        :return: Строка (список) или число
        :raises TypeError: Если индекс имеет неверный тип
        :raises IndexError: Если индекс выходит за границы
        """
        if isinstance(index, int):
            return self._data[index]
        elif isinstance(index, tuple) and len(index) == 2:
            i, j = index
            return self._data[i, j]
        else:
            raise TypeError("Индекс должен бытььь int или кортежем (i, j)")
        
    def __setitem__(self, index: Union[int, Tuple[int, int]], value: Any) -> None:
        """
        Устанавливает строку или элемент по индексу.

        :param index: Целое число (для строки) или кортеж (i, j) для элемента
        :param value: Для строки — список чисел; для элемента — число
        :raises TypeError: Если тип индекса или значения некорректен
        :raises ValueError: Если размер строки не совпадает
        :raises IndexError: Если индекс вне диапазона
        """
        if isinstance(index, int):
            if not isinstance(value, list):
                raise TypeError("Значение должно быть списком для присвоения строки")
            if len(value) != len(self._data[0]):
                raise ValueError(f"Длина строки должна быть {len(self._data[0])}")
            if not all(isinstance(x, (int, float)) for x in value):
                raise TypeError("Все элементы строки должны быть числа")
            self._data[index] = value
        elif isinstance(index, tuple) and len(index) == 2:
            i, j = index
            if not isinstance(value, (int, float)):
                raise TypeError("Элемент матрицы должен быть числом")
            self._data[i][j] = value
        else:
            raise TypeError("Индекс должен быть int или кортежем (i, j)")
        
    def __add__(self, other: "Matrix") -> "Matrix":
        """
        Складывает текущую матрицу с другой матрицей.

        :param other: Другая матрица того же размера
        :return: Новая матрица — результат сложения
        :raises TypeError: Если other не Matrix
        :raises ValueError: Если размеры не совпадают
        """
        if not isinstance(other, Matrix):
            raise TypeError("Можно складывать только с другой Matrix")
        if self.spape() != other.shape():
            raise ValueError("Матрицы должны быть одинакового размера")
        return Matrix([
            [a + b for a, b in zip(self_row, other_row)]
            for self_row, other_row in zip(self._data, other._data)
        ])
    
    def __mul__(self, other: Any) -> Union["Matrix", "Vector", float]:
        """
        Умножает матрицу на скаляр, вектор или другую матрицу.

        :param other: Число (скаляр), Vector (длина = кол-во столбцов) или Matrix (совместимый размер)
        :return: Matrix (при умножении на скаляр или матрицу) или Vector (при умножении на вектор)
        :raises TypeError: Если тип other не поддерживается
        :raises ValueError: Если размеры несовместимы для умножения
        """
        if isinstance(other, (int, float)):
            return Matrix([[x * other for x in row] for row in self._data])
        elif isinstance(other, Vector):
            if len(other) != len(self._data[0]):
                raise ValueError("Длина вектора должна совпадать с количеством столбцов матрицы")
            result_data = [
                sum(m_ij * v_j for m_ij, v_j in zip(row, other._data))
                for row in self._data
            ]
            return Vector(result_data)
        elif isinstance(other, Matrix):
            if len(self._data[0]) != len(other._data):
                raise ValueError("Количество столбцов первой матрицы должно равняться количеству строк второй")
            cols = other.transpose()._data
            result_data = [
                [sum(a * b for a, b in zip(row, col)) for col in cols]
                for row in self._data
            ]
            return Matrix(result_data)
        else:
            raise TypeError("Умножение поддерживается только на число, Vector или Matrix")