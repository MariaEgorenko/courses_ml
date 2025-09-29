import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Применяет симоиду к входному массиву

    :param np.ndarray x: Входной массив вещественных чисел
    :return nd.array: Выход после применения сигмоиды
        (значения в диапазоне (0, 1))
    :raise TypeError: Если x не является np.ndarray
    :raise ValueError: Если x содержит нечисловые значения
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f'Ожидалось np.ndarray, получено {type(x)}')
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError('Входящий массив должен содержать только числовые значения')
    
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Производная сигмоиды

    :param np.ndarray x: Вход сигмоиды (уже активированный)
    :return np.ndarray: производная
    :raise TypeError: Если x не является np.ndarray
    :raise ValueError: Если x содержит нечисловые значения
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f'Ожидалось np.ndarra, получено {type(x)}')
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError('Входящий массив должен содержать только числовые значения')
    
    return x * (1 - x)

def relu(x: np.ndarray) -> np.ndarray:
    """
    Примемняет ReLu к входному массиву.

    :param np.ndarray x: Входной массив вещественных чисел
    :return np.ndarray: Выход после применения ReLu
    :raise TypeError: Если x не является np.ndarray
    :raise ValueError: Если x содержит нечисловые значения
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f'Ожидалось np.ndarra, получено {type(x)}')
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError('Входящий массив должен содержать только числовые значения')
    
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Производная ReLU

    :param np.ndarray x: Входной массив вещественных чисел
    :return np.ndarray: Производная ReLU (массив из 0 и 1)
    :raise TypeError: Если x не является np.ndarray
    :raise ValueError: Если x содержит нечисловые значения
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f'Ожидалось np.ndarra, получено {type(x)}')
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError('Входящий массив должен содержать только числовые значения')
    
    return (x > 0).astype(float)

def tanh(x: np.ndarray) -> np.ndarray:
    """
    Применяет гиперболический тангенс к входному массиву

    :param np.ndarray x: Входной массив вещественных чисел
    :return np.ndarray: Выход после применения tanh
        (значения в диапазоне [-1, 1])
    :raise TypeError: Если x не является np.ndarray
    :raise ValueError: Если x содержит нечисловые значения
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f'Ожидалось np.ndarra, получено {type(x)}')
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError('Входящий массив должен содержать только числовые значения')
    
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    Применяет гиперболический тангенс к входному массиву

    :param np.ndarray x: Выход tanh (уже активированный)
    :return np.ndarray: Производная
    :raise TypeError: Если x не является np.ndarray
    :raise ValueError: Если x содержит нечисловые значения
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f'Ожидалось np.ndarra, получено {type(x)}')
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError('Входящий массив должен содержать только числовые значения')

    return 1 - x**2

def linear(x: np.ndarray) -> np.ndarray:
    """
    Линейная функция

    :param np.ndarray x: Входной массив вещественных чисел
    :return np.ndarray: Возвращает x без изменений
    :raise TypeError: Если x не является np.ndarray
    :raise ValueError: Если x содержит нечисловые значения
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f'Ожидалось np.ndarra, получено {type(x)}')
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError('Входящий массив должен содержать только числовые значения')
    
    return x

def lineara_derivative(x: np.ndarray) -> np.ndarray:
    """
    Производная линейной функции (единичная матрица)

    :param np.ndarray x: Входной массив вещественных чисел
    :return np.ndarray: Массив из единиц такой же формы, как x
    :raise TypeError: Если x не является np.ndarray
    :raise ValueError: Если x содержит нечисловые значения
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f'Ожидалось np.ndarra, получено {type(x)}')
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError('Входящий массив должен содержать только числовые значения')
    
    return np.ones_like(x)

# Словарь функций активации
activations = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'linear': (linear, lineara_derivative),
    None: (linear, lineara_derivative)
}