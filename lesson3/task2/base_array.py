from abc import ABC, abstractmethod
from typing import Any, Tuple, Iterator, Union


class BaseArray(ABC):
    """
    Абстрактный базовый класс для числовых структур.
    Обеспечивает единый интерфейс для одномерных и двумерных массивов.
    """

    def __init__(self, data: Any) -> None:
        """
        Инициализация объекта.
        :param data: список чисел или список списков
        """
        self._data = data

    @abstractmethod
    def __len__(self) -> int:
        """
        Возвращает количество элемнтов в стурктуре.
        
        :return: Количество элементов
        """
        pass

    @abstractmethod
    def __getitem__(self, index: Any) -> Any:
        """
        Возвращает элемент по индексу.

        :param index: Индекс или кортеж индексов
        :return: Значение по указанному индексу
        """
        pass

    @abstractmethod
    def __setitem__(self, index: int, value: Any) -> None:
        """
        Устанавливает значение по индексу.

        :param index: Индекс или кортеж индексов
        :param value: Новое значение
        """
        pass

    @abstractmethod
    def __add__(self, other: "BaseArray") -> "BaseArray":
        """
        Складывает текущую структуру с другой.

        :param other: Другая числовая структура
        :return: Новый объект результата сложения
        :raises ValueError: Если операция невозможна
        """
        pass

    @abstractmethod
    def __mul__(self, other: Any) -> Union["BaseArray", float]:
        """
        Умножает структуру на скаляр или другую структуру.

        :param other: Скаляр (int/float) или другая структура
        :return: Результат умножения 
        :raises TypeError: Если операция не поддерживается
        """
        pass

    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """
        Возвращает размерность структуры.

        :return: Кортеж с размерами (например, (3,) или (2, 3))
        """
        pass

    def __str__(self) -> str:
        """
        Возвращает строковое представление объекта.

        :return: Строковое представление данных
        """
        return str(self._data)
    
    def __iter__(self) -> Iterator[Any]:
        """
        Возвращает итератор по элементам структуры.

        :return: Итератор
        """
        return iter(self._data)
    
    def __contains__(self, item: Any) -> bool:
        """
        Проверяет наличие элемента в структуре.

        :param item: Элемент для проверки
        :return: True, если элемент содержится, иначе False
        """
        return item in self._data
    
    def __eq__(self, other: Any) -> bool:
        """"
        Проверяет равенство двух объектов.

        :param other: Другой объект
        :return: True, если объекты равны
        """
        if not isinstance(other, type(self)):
            return False
        return self._data == other._data
