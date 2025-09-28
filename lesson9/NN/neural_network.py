import numpy as np
from activations import activations


class Layer:
    """
    Класс, представляющий один полносвязный слой нейронной сети
    """
    def __init__(self, output_size: int, activation: str | None = None) -> None:
        """
        Инициализирует слой c заданным количеством нейронов и
        функцией активации

        :param int output_size: Количество нейронов в этом слое
        :param str | None activation: Функция активации.
            Допустимые значения: 'sigmoid', 'relu', 'tanh', 'linear', None. 
            Если None, используется линейная (без активации). 
            По умолчанию None.
        :raise TypeError: Если output_size не int или activation не str | None
        :raise ValeuError: Если activation не входит в список допустимых
            значений
        """
        if not isinstance(output_size, int):
            raise TypeError(f"output_size должен быть int, передано: {type(output_size)}")
        if activation is not None and not isinstance(activation, str):
            raise TypeError(f"activation должен быть str или None, передано: " + 
                            type(activation))
        if activation is not None and activation not in activations:
            raise ValueError(
                f"activation '{activation}' не поддерживается. Допустимые значения: " +
                list(activations.keys())
            )
        
        self.output_size = output_size
        self.activation_name = activation

        self.activation_func, self.activation_deriv = activations[activation]

        self.weights = None  # (input_size, output_size)
        self.biases = None   # (1, output_size)

        self.input = None   # (batch_size, input_size)
        self.output = None  # (batch_size, output_size)
    
    def build(self, input_size: int) -> None:
        """
        Инициализирует веса и смещения слоя на основе размера входа

        :param int input_size: Размер входного вектора (количество 
            нейронов в предыдущем слое)
        :raise TypeError: Если input_size не int
        :raise ValueError: Если input_size <= 0
        """
        if not isinstance(input_size, int):
            raise TypeError(f"input_size должен быть int, передано: {type(input_size)}")
        if input_size <= 0:
            raise ValueError(f"input_size должен быть положительным числом, передано: " +
                             input_size)
        
        limit = np.sqrt(6.0 / (input_size + self.output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, self.output_size))
        self.biases = np.zeros((1, self.output_size))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Выполняет прямой проход через слой

        :param  np.ndarray input_data: Входные данные размером 
            (batch_size, input_size)
        :return np.ndarray: Выход слоя размером (batch_size, output_size)
        :raise TypeError: Если input_data не numpy.ndarray
        """
        if not isinstance(input_data, np.ndarray):
            raise TypeError(f"input_data должен быть numpy.ndarray, передано: " +
                            type(input_data))
        
        self.input = input_data
        z = np.dot(input_data, self.weights) + self.biases
        self.output = self.activation_func(z)
        return self.output
    
    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Выполняет обратный проход через слой (backpropagation)

        :param np.ndarray grad_output: Градиент от следующего слоя
            размером (batch_size, output_size)
        :param float learning_rate: Скорость обучения
        :return np.ndarray: Градиент, передаваемый в предыдущий слой
            размером (batch_size, input_size)
        :raise TypeError: Если grad_output не numpy.ndarray или
            learning_rate не число
        """
        if not isinstance(grad_output, np.ndarray):
            raise TypeError(
                f"grad_output должен быть numpy.ndarray, передано: {type(grad_output)}"
            )
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(
                f"learning_rate должен быть числом (int или float), передано: " +
                type(learning_rate)
            )
        
        # Производная функции активации
        grad_activation = self.activation_deriv(self.output)
        delta = grad_output * grad_activation

        # Градиент по весам и смещениям
        grad_weights = np.dot(self.input.T, delta) 
        grad_biases = np.sum(delta, axis=0, keepdims=True)  

        # Обновление весов и смещений
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        # Градиент, передаваемый в предыдущий слой
        grad_input = np.dot(delta, self.weights.T)  # (batch_size, input_size)
        return grad_input
    

class NeuralNetwork:
    """
    Класс, реализующий нейронную сеть с возможностью добавления слоёв
    и обучения
    """

    def __init__(self) -> None:
        """
        Инициализирует нейронную сеть. Создаёт пустой список слоёв
        """
        self.layers = []

    def add_layer(self, output_size: int, activation: str | None = None) -> None:
        """
        Добавляет слой в сеть

        :param int output_size: Количество нейронов в новом слое
        param str | None activation: Функция активации.
            Допустимые значения: 'sigmoid', 'relu', 'tanh', 'linear', None. 
            Если None, используется линейная (без активации). 
            По умолчанию None.
        :raise TypeError: Если output_size не int или activation
            не str или None
        :raise ValueError: Если activation не поддерживается
        """
        layer = Layer(output_size, activation)
        self.layers.append(layer)

    def build(self, input_size: int) -> None:
        """
        Инициализирует веса всех слоёв на основе размера входа

        Вызывается автоматически при обучении, чтобы инициализировать веса,
        зная размер входного слоя

        :param int input_size: Размер входного слоя (количество признаков в X)
        :raise TypeError: Если input_size не int
        :raise ValueError: Если input_size <= 0
        """
        if not isinstance(input_size, int):
            raise TypeError(f"input_size должен быть int, передано: {type(input_size)}")
        if input_size <= 0:
            raise ValueError(f"input_size должен быть положительным числом, передано: " +
                             input_size)
        
        prev_size = input_size
        for layer in self.layers:
            layer.build(prev_size)
            prev_size = layer.output_size

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Выполняет прямой проход через всю нейронную сеть

        :param np.ndarray X: Входные данные размером (batch_size,
            input_size)
        :return np.ndarray: Выход сети размером (batch_size,
            output_size последнего слоя).
        :raise TypeError: Если X не numpy.ndarray
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X должен быть numpy.ndarray, передано: {type(X)}")
        
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y: np.ndarray, output: np.ndarray, learning_rate: float = 0.01) -> None:
        """
        Выполняет обратный проход (backpropagation) через всю сеть

        :param np.ndarray y: Целевые значения размером (batch_size,
            output_size)
        :param np.ndarray output: Выход сети, полученный при прямом
            проходе
        :param float learning_rate: Скорость обучения. По умолчанию 0.01
        :raise TypeError: Если y, output или learning_rate имеют
            неверный тип
        """ 
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y должен быть numpy.ndarray, передано: {type(y)}")
        if not isinstance(output, np.ndarray):
            raise TypeError(f"output должен быть numpy.ndarray, передано: {type(output)}")
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(
                f"learning_rate должен быть числом (int или float), передано: " +
                type(learning_rate)
            )

        grad = output - y
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train(
            self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float = 0.01
        ) -> None:
        """
        Обучает нейронную сеть на обучающих данных

        :param np.ndarray X: Входные данные размером (batch_size,
            input_size)
        :param np.ndarray y: Целевые значения размером (batch_size,
            output_size)
        :param int epochs: Количество эпох обучения
        :param float learning_rate: Скорость обучения. По умолчанию 0.01
        :raise TypeError: Если X, y, epochs или learning_rate имеют неверный тип
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X должен быть numpy.ndarray, передано: {type(X)}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y должен быть numpy.ndarray, передано: {type(y)}")
        if not isinstance(epochs, int):
            raise TypeError(f"epochs должен быть int, передано: {type(epochs)}")
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(
                f"learning_rate должен быть числом (int или float), передано: " + 
                type(learning_rate)
            )
        
        self.build(X.shape[1])
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(y, output, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)
                print(f'Epoch {epoch}, Loss: {loss:.6f}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Выполняет предсказание на основе обученной сети

        :param np.ndarray X: Входные данные размером (batch_size,
            input_size)
        :return np.ndarray: Предсказания размером (batch_size,
            output_size последнего слоя)
        :raise TypeError: Если X не numpy.ndarray
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X должен быть numpy.ndarray, передано: {type(X)}")
        
        return self.forward(X)
    
    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Валидирует модель на валидационной выборке

        :param np.ndarray X_val: Входные данные валидационной выборки
        :param np.ndarray y_val: Целевые значения валидационной выборки
        :return float: MSE на валидационной выборке
        :raise TypeError: Если X_val или y_val не numpy.ndarray
        """
        if not isinstance(X_val, np.ndarray):
            raise TypeError(f"X_val должен быть numpy.ndarray, передано: {type(X_val)}")
        if not isinstance(y_val, np.ndarray):
            raise TypeError(f"y_val должен быть numpy.ndarray, передано: {type(y_val)}")
        
        predictions = self.predict(X_val)
        mse = np.mean((y_val - predictions) ** 2)
        return float(mse)