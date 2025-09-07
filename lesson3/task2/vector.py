from .base_array import BaseArray
from typing import List, Union, Any, Tuple


class Vector(BaseArray):
    """
    Класс для представления одномерной числовой структуры — вектора.
    Поддерживает операции сложения, умножения (на скаляр и скалярного
    произведения), индексации, итерации и сравнения.
    """

    def __init__(self, data: List[Union[int, float]]) -> None:
        """
        Инициализирует вектор.

        :param data: Список чисел (int или float)
        :raises TypeError: Если элементы не являются числами
        """
        if not isinstance(data, list):
            raise TypeError("Данные должны быть списком")
        if not all(isinstance(x, (int, float)) for x in data):
            raise TypeError("Все эдементы вектра должны быть числами")
        super.__init__(data)

    def __len__(self) -> int:
        """
        Возвращает длину вектора.

        :return: Количество элементов в векторе
        """
        return len(self._data)
    
    def __getitem__(self, index: int) -> Union[int, float]:
        """
        Возвращает элемент по индексу.

        :param index: Целочисленный индекс
        :return: Значение элемента
        :raises IndexError: Если индекс вне диапазона
        """
        return self._data[index]
    
    def __setitem__(self, index: int, value: Union[int, float]) -> None:
        """
        Устанавливает значение элемента по индексу.

        :param index: Целочисленный индекс
        :param value: Новое числовое значение
        :raises IndexError: Если индекс вне диапазона
        :raises TypeError: Если значение не число
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Значение должно быть числом типа int или float")
        self._data[index] = value

    def __add__(self, other: "Vector") -> "Vector":
        """
        Складывает текущий вектор с другим вектором.

        :param other: Другой объект Vector
        :return: Новый вектор — результат сложения
        :raises TypeError: Если other не Vector
        :raises ValueError: Если длины векторов не совпадают
        """
        if isinstance(other, Vector):
            raise TypeError("Можно складывать только с другим Vector")
        if len(self) != len(other):
            raise ValueError("Векторы должны быть одинаковой длины")
        return Vector([a + b for a, b in zip(self._data, other._data)])
    
    def __sub__(self, other: "Vector") -> "Vector":
        """
        Вычитает другой вектор из текущего.

        :param Vector: другой объект вектор
        :return: Новый вектор – результат вычитания
        :raise TypeError: если other не Vector
        :raise ValueError: если длины вектора не совпадают 
        """
        if not isinstance(other, Vector):
            raise TypeError("Можно вычитать только другой вектор")
        if len(self._data) != len(other._data):
            raise ValueError("Векторы должны быть одинаковой длины")
        return Vector([a - b for a, b in zip(self._data, other._data)])

    def __mul__(
            self,
            other: Union[List[Union[int, float]], "Vector", Union[int, float]]
        ) -> Union["Vector", float]:
        """
        Умножает вектор на скаляр или вычисляет скалярное
        произведение с другим вектором.

        :param other: Число (скаляр) или другой Vector
        :return: Новый Vector (при умножении на скаляр) или число
            (скалярное произведение)
        :raises ValueError: Если векторы разной длины при скалярном
            произведении (если other — Vector)
        :raises TypeError: Если тип other не поддерживается
        """
        if isinstance(other, (int, float)):
            return Vector([x * other for x in self._data])
        elif isinstance(other, (Vector, list, tuple)):
            if len(self) != len(other):
                raise ValueError(
                    "Для скалярного произведения векторы должны быть "
                    "одинаковой длины"
                )
            if isinstance(other, Vector):
                return sum(a * b for a, b in zip(self._data, other._data))
            return sum(a * b for a, b in zip(self._data, other))
        else:
            raise TypeError(
                "Умножение поддерживается только на число или другой вектор"
            )
        
    def __rmul__(self, other: Any) -> "Vector":
        """
        Обратное умножение (например: 5 * vector).

        :param other: Число (скаляр)
        :return: Новый вектор
        :raises ValueError: Если векторы разной длины при скалярном
            произведении (если other — Vector)
        :raises TypeError: Если тип other не поддерживается
            (например, строка)
        """
        return self.__mul__(other)
    
    def shape(self) -> Tuple[int]:
        """
        Возвращает размерность вектора.

        :return: Кортеж вида (n,), где n — длина вектора
        """
        return (len(self._data),)
    
    def __str__(self) -> str:
        """
        Строковое представление вектора.

        :return: Строка вида "[1.0, 2.0, 3.0]"
        """
        return f"[{', '.join(map(str, self._data))}]"
    
    def __eq__(self, other: Any) -> bool:
        """
        Сравнивает вектор с другим на объектом на равенство

        :param other: Другой вектор, список или кортеж
        :return: True, если объекты равны поэлементно
        """
        if isinstance(other, Vector):
            return self._data == other._data
        if isinstance(other, (list, tuple)):
            return self._data == list(other)
        return False
    
    def append(self, value: Union[int, float]) -> None:
        """
        Добавляет элемент а конец вектора.

        :param value: Число (int или float для добавления)
        :raise TypeError: Если значение не является числом
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Можно добавлять только числа типа int или float")
        self._data.append(value)

    def extend(self, value: Union["Vector", List[Union[int, float]]]) -> None:
        """
        Добавляет несколько элементов в конец вектора

        :param value: список чисел (int или float)
        :raise TypeError: Если передана не последовательность или ее
            хотя бы одно значение не является числом
        """
        if not isinstance(value, (Vector, list, tuple)):
            raise TypeError(
                "Передаваемое значение должно быть последовательностью"
            )
        if not all(isinstance(x, (int, float)) for x in value):
            raise TypeError("Элементы последовательности должны быть числа")
        self._data.extend(value)

    