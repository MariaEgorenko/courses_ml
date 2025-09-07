from .base_array import BaseArray
from .vector import Vector
from typing import List, Union, Any, Tuple


class Matrix(BaseArray):
    """
    Класс для представления двумерной числовой структуры — матрицы.
    Поддерживает операции сложения, умножения (на скаляр,
    вектор и матрицу), транспонирования, доступа по индексам,
    итерации и сравнения.
    """

    def __init__(self, data: List[List[Union[int, float]]]) -> None:
        """
        Инициализирует матрицу из двумерного списка.

        :param data: Список списков чисел (int или float).
            Все строки должны иметь одинаковую длину.
        :raises TypeError: Если data не список, или элементы не списки,
            или содержат не числа.
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
                raise ValueError(
                    f"Строка {i} имеет длину {len(row)}, "
                    f"ожидалась {count_cols}"
                )
            if not all(isinstance(x, (int, float)) for x in row):
                raise TypeError("Все элементы матрицы должны быть числами")
            
        super.__init__(data)

    def __len__(self) -> int:
        """
        Возвращает общее количество элементов в матрице.

        :return: Количество элементов (строки * столбцы)
        """
        return len(self._data) * len(self._data[0])
    
    @property
    def data(self):
        """
        Возвращает копию данных матрицы.

        :return: "Элементы список матрицы
        """
        return self._data.copy()

    def __getitem__(
            self, index: Union[int, Tuple[int, int]]
        ) -> (Union[List[Union[int, float]], Union[int, float]]):
        """
        Возвращает строку по индексу или элемент по координатам (i, j).

        :param index: Целое число (для получения строки) или
            кортеж (i, j) для получения элемента
        :return: Строка (список) или число
        :raises TypeError: Если индекс имеет неверный тип или
            элементы кортежа не int
        :raises IndexError: Если индекс выходит за границы
        """
        if isinstance(index, int):
            return self._data[index]
        elif isinstance(index, tuple) and len(index) == 2:
            i, j = index
            if not isinstance(i, int) or not isinstance(j, int):
                raise TypeError("Индексы матрицы должны быть целыми числами")
            return self._data[i][j]
        else:
            raise TypeError("Индекс должен быть int или кортежем (i, j)")
        
    def __setitem__(
            self, index: Union[int, Tuple[int, int]], value: Any
        ) -> (None):
        """
        Устанавливает строку или элемент по индексу.

        :param index: Целое число (для строки) или кортеж (i, j)
            для элемента
        :param value: Для строки — список чисел; для элемента — число
        :raises TypeError: Если тип индекса или значения некорректен
        :raises ValueError: Если размер строки не совпадает
        :raises IndexError: Если индекс вне диапазона
        """
        if isinstance(index, int):
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    "Значение должно быть списком для присвоения строки"
                )
            if len(value) != len(self._data[0]):
                raise ValueError(
                    f"Длина строки должна быть {len(self._data[0])}"
                )
            if not all(isinstance(x, (int, float)) for x in value):
                raise TypeError("Все элементы строки должны быть числами")
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
    
    def __sub__(self, other: "Matrix") -> "Matrix":
        """
        Вычитает одну матрицу из другой.

        :param other: Другая матрица
        :return: Новая матрица – результат вычитания
        :raise TypeError: Если другой объект не матрица
        :raise ValueError: Если матрицы разного размера
        """
        if not isinstance(other, Matrix):
            raise TypeError("Вычитать можно только другую матрицу")
        if self.shape() != other.shape():
            raise ValueError("Матриы должны быть одног размера")
        return Matrix([
            [a - b for a, b in zip(self_row, other_row)]
            for self_row, other_row in zip(self._data, other._data)
        ])
    
    def __mul__(self, other: Any) -> Union["Matrix", "Vector"]:
        """
        Умножает матрицу на скаляр, вектор или другую матрицу.

        :param other: Число (скаляр), Vector (длина = кол-во столбцов)
            или Matrix (совместимый размер)
        :return: Matrix (при умножении на скаляр или матрицу) или
            Vector (при умножении на вектор)
        :raises TypeError: Если тип other не поддерживается
        :raises ValueError: Если размеры несовместимы для умножения
        """
        if isinstance(other, (int, float)):
            return Matrix([[x * other for x in row] for row in self._data])
        elif isinstance(other, (Vector, list, tuple)):
            if len(other) != len(self._data[0]):
                raise ValueError(
                    "Длина вектора должна совпадать с количеством "
                    "столбцов матрицы"
                )
            if isinstance(other, Vector):
                other_list = other.data
            else:
                if all(isinstance(x, (int, float)) for x in other):
                    raise TypeError("Элементы списка должны быть числами")
                other_list = list(other)
            result_data = [
                sum(m_ij * v_j for m_ij, v_j in zip(row, other_list))
                for row in self._data
            ]
            return Vector(result_data)
        elif isinstance(other, Matrix):
            if len(self._data[0]) != len(other._data):
                raise ValueError(
                    "Количество столбцов первой матрицы должно равняться "
                    "количеству строк второй"
                )
            cols = other.transpose()._data
            result_data = [
                [sum(a * b for a, b in zip(row, col)) for col in cols]
                for row in self._data
            ]
            return Matrix(result_data)
        else:
            raise TypeError(
                "Умножение поддерживается только на число, Vector или Matrix"
            )
    
    def __rmul__(self, other: Any) -> Union["Vector", "Matrix"]:
        """
        Обратное умножение (например: scalar * matrix или
        vector * matrix).

        :param other: Число или Vector (длина = кол-во строк)
        :return: Matrix (если other — скаляр) или Vector (если other —
            вектор-строка)
        :raises TypeError: Если тип other не поддерживается или
            элементы списка не числа
        :raises ValueError: Если размеры несовместимы для умножения
        """
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        elif isinstance(other, (Vector, list, tuple)):
            if len(self._data) != len(other):
                raise ValueError(
                    "Длинна вектора должна совпадать совпадать с количеством "
                    "строк матрицы"
                )
            other_data = []
            if isinstance(other, Vector):
                other_data = other.data
            else:
                if not all(isinstance(x, (int, float)) for x in other):
                    raise TypeError("Данные списка должны быть числами")
                other_data = list(other)
            result_data = [
                sum(v_j * m_ij for v_j, m_ij in zip(other_data, col))
                for col in self._data
            ]
            return Vector(result_data)
        elif isinstance(other, Matrix):
            if len(other._data[0]) != len(self.data):
                raise ValueError(
                    "Количество столбцов первой матрицы должно равняться "
                    "количеству строк второй")
            cols = self.transpose()._data
            result_data = [
                [sum(a * b for a, b in zip(row, col)) for col in cols]
                for row in other._data
            ]
            return Matrix(result_data)
        else:
            raise ValueError(
                "Обратное умноженние поддерживается только для числа "
                "или Vector"
            )
        
    def __matmul__(self, other: Any) -> Union["Vector", "Matrix"]:
        """
        Оператор @ — матричное умножение.

        :param other: Число (скаляр), Vector (длина = кол-во столбцов)
            или Matrix (совместимый размер)
        :return: Matrix (при умножении на скаляр или матрицу) или
            Vector (при умножении на вектор)
        :raises TypeError: Если тип other не поддерживается
        :raises ValueError: Если размеры несовместимы для умножения
        """
        return self.__mul__(other)
    
    def __rmatmul__(self, other: Any) -> Union["Vector", "Matrix"]:
        """
        Оператор @ – обратное матричное умножение.

        :param other: Число или Vector (длина = кол-во строк)
        :return: Matrix (если other — скаляр) или Vector (если other —
            вектор-строка)
        :raises TypeError: Если тип other не поддерживается или
            элементы списка не числа
        :raises ValueError: Если размеры несовместимы для умножения
        """
        return self.__rmatmul__(other)
    
    def shape(self):
        """
        Возвращает размерность матрицы

        :return: Кортеж (строки, столбцы)
        """
        return (len(self._data), len(self._data[0]))
    
    def transpose(self):
        """
        Возвращает транспонированную матрицу

        :return: Новая матрица, где строки и столбцы поменяны местами
        """
        return Matrix([list(col) for col in zip(*self._data)])
    
    def __str__(self) -> str:
        """
        Строковое представление матрицы.

        :return: Строка с построчным отображениеим матрицы
        """
        rows = [
            "[" + ", ".join(f"{x}" for x in row + "]") for row in self._data
        ]
        return "[" + "\n".join(rows) + "]"
    
    def __contains__(self, item: Any) -> bool:
        """
        Проверяет, содержится ли элемент в матрице.

        :param: item: Элемент для поиска
        :return: True, если элемент найден
        """
        return any(item in row for row in self._data)
    
    def __eq__(self, other: Any) -> bool:
        """
        Сравнивает две матрицы

        :param other: Другой объект для сравнения
        :return: True, если other — Matrix и данные совпадают
        """
        if not isinstance(other, Matrix):
            return False
        return self._data == other._data
    
    def is_squre(self) -> bool:
        """
        Проверяет, является ли матрица квадратной.

        :return: True, если матрица квадратная
        """
        rows, cols = self.shape()
        return rows == cols
    
    def trace(self) -> float:
        """
        Возвращает след матрицы (сумму диагональных элементов).

        :return: Сумма элементов главной диагонали
        :raise VallueError: Если матрица не квадратная
        """
        if not self.is_squre():
            raise ValueError("След определен только для квадратных матриц")
        return sum(self._data[i][i] for i in range(len(self)))

    def append(self, row: List[Union[int, float]]) -> None:
        """
        Добавляется строку (список числе) в конец матрицы.

        :param row: Добавляемая строка
        :raise: TypeError: Если передан не список чисел
        :raise: ValueError: Если размер передаваемой строки не равен
            размеру строки матрицы
        """
        if not isinstance(row, (list, tuple)):
            raise TypeError("Передаваемое значение должно быть списком")
        if len(self._data[0]) != len(row):
            raise ValueError(
                "Размер добавляемой строки должен соответствовать размеру "
                "строки матрицы"
            )
        if all(isinstance(x, (int, float)) for x in row):
            raise TypeError("Элементы строки должны быть числами")
        self._data.append(row)

    def extend(self, rows: list[list[Union[int, float]]]) -> None:
        """
        Добавление нескольких строк в конец матрицы.

        :param rows: список добавляемых строк
        :raise TypeError: Если передан не список списков чисел
        :raise ValueError: Если размер строк в списке не равны размеру
            строк матрицы
        """
        if (isinstance(rows, (list, tuple))
            and all(isinstance(x, (list, tuple)) for x in rows)):
            
            if not all(
                len(self._data[0]) == len(rows[i]) for i in range(len(rows))
                ):

                raise ValueError(
                    "Строки в списке должны быть одного размера со "
                    "строками матрицы"
                    )
            if not all(isinstance(x, (int, float)) for row in rows for x in row):
                raise TypeError("Значения в списке списка должны быть числами")
            self._data.extend(rows)
        else:
            raise TypeError("Должен передаваться список списков")

            