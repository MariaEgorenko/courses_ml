import numpy as np
from collections import Counter
from typing import Optional, Tuple, Any, List, Union

class DecisionTree:
    """
    Класс дерева решений для задач классификации и регрессии.

    Поддерживает как классификацию (по умолчанию), так и регрессию
    (при установке параметра regression=True). Использует критерий
    Джини для классификации и среднеквадратичную ошибку (MSE) для
    регрессии.
    
    Attr:
        max_depth (int): Максимальная глубина дерева. По умолчанию 10.
        min_samples_split (int): Минимальное количество образцов,
            необходимое для разделения узла. По умолчанию 2.
        min_samples_leaf (int): Минимальное количество образцов, которое
            должно быть в листе. По умолчанию 1.
        regression (bool): Флаг режима регрессии. Если True — решает
            задачу регрессии,
    """
    def __init__(
            self,
            max_depth: int = 10,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            regression: bool = False
        ) -> None:
        """
        Инициализирует дерево решений с заданными гиперпараметрами.

        Param:
            max_depth (int): Максимальная глубина дерева. Дерево
                перестанет расти, если достигнет этой глубины.
            min_samples_split (int): Минимальное число объектов в узле,
                чтобы его можно было разделить.
            min_samples_leaf (int): Минимальное число объектов, которое
                должно остаться в каждом дочернем узле после разбиения.
            regression (bool): Если True — дерево решает задачу регрессии,
                иначе — классификации.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.regression = regression
        self.root = None

    class Node:
        """
        Внутренний класс, представляющий узел дерева решений.

        Attr:
            feature (int or None): Индекс признака, по которому
                выполняется разбиение. None для листьев.
            threshold (float or None): Пороговое значение для
                разбиения. None для листьев.
            left (Node or None): Левый дочерний узел.
            right (Node or None): Правый дочерний узел.
            value (any or None): Значение прогноза в листе. None для
                внутренних узлов.
        """
        def __init__(
                self,
                feature: Optional[int] = None,
                threshold: Optional[int] = None,
                left: Optional['DecisionTree.Node'] = None,
                right: Optional['DecisionTree.Node'] = None,
                value: Optional[Any] = None
            ) -> None:
            """
            Инициализирует узел дерева.

            Param:
                feature (int or None): Индекс признака для разбиения.
                threshold (float or None): Порог разбиения.
                left (Node or None): Левый потомок.
                right (Node or None): Правый потомок.
                value (any or None): Прогнозируемое значение (только
                    для листьев).
            """
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(
            self, X: Union[List, np.ndarray], y: Union[List, np.ndarray]
        ) -> None:
        """
        Обучает дерево решений на обучающей выборке.

        Param:
            X (Union[List, np.ndarray]): Матрица признаков формы (n_samples, n_features).
            y (Union[List, np.ndarray]): Вектор целевой переменной формы (n_samples,).
        Returns:
            None
        """
        X = np.array(X)
        y = np.array(y)
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y)

    def _grow_tree(
            self, X: np.ndarray, y: np.ndarray, depth: int = 0
        ) -> 'DecisionTree.Node':
        """
        Рекурсивно строит дерево решений.

        Param:
            X (np.ndarray): Подвыборка признаков формы (n_samples, n_features).
            y (np.ndarray): Подвыборка целевых значений формы (n_samples,).
            depth (int): Текущая глубина узла.
        Returns:
            DecisionTree.Node: Построенный узел (внутренний или лист).
        """
        n_samples = len(y)
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return self.Node(value=self._calculate_leaf_value(y))

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return self.Node(value=self._calculate_leaf_value(y))

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            return self.Node(value=self._calculate_leaf_value(y))

        left_child = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return self.Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _best_split(
            self, X: np.ndarray, y: np.ndarray
        ) -> Tuple[Optional[int], Optional[float]]:
        """
        Находит наилучшее разбиение по всем признакам и порогам.

        Param:
            X (np.ndarray): Матрица признаков (n_samples, n_features).
            y (np.ndarray): Целевые значения (n_samples,).
        Returns:
            Tuple[Optional[int], Optional[float]]: (индекс_признака, порог). 
            Если разбиение невозможно — (None, None).
        """
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in range(self.n_features):
            values = np.sort(np.unique(X[:, feature]))
            if len(values) < 2:
                continue
            # Пороги между соседними значениями
            thresholds = (values[:-1] + values[1:]) / 2

            for th in thresholds:
                gain = self._information_gain(y, X[:, feature], th)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = th
        return best_feature, best_threshold

    def _impurity(self, y: np.ndarray) -> float:
        """
        Вычисляет неоднородность узла.

        Для классификации — индекс Джини.
        Для регрессии — сумма квадратов отклонений от среднего (RSS).

        Param:
            y (np.ndarray): Целевые значения (n_samples,).
        Returns:
            float: Значение неоднородности.
        """
        if len(y) == 0:
            return 0.0
        if self.regression:
            return np.sum((y - y.mean()) ** 2) 
        else:
            # Gini
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1 - np.sum(probs ** 2)

    def _information_gain(
            self, y: np.ndarray, X_column: np.ndarray, threshold: float
        )-> float:
        """
        Вычисляет прирост информации от разбиения.

        Param:
            y (np.ndarray): Целевые значения (n_samples,).
            X_column (np.ndarray): Значения одного признака (n_samples,).
            threshold (float): Порог разбиения.
        Returns:
            float: Прирост информации. Возвращает 0, если разбиение недопустимо.
        """
        parent_imp = self._impurity(y)
        left_mask = X_column <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0

        imp_left = self._impurity(y[left_mask])
        imp_right = self._impurity(y[right_mask])

        if self.regression:
            # Для регрессии: gain = уменьшение суммарной ошибки (RSS)
            return parent_imp - (imp_left + imp_right)
        else:
            # Для классификации: взвешенная неоднородность
            n = len(y)
            n_left, n_right = np.sum(left_mask), np.sum(right_mask)
            child_imp = (n_left / n) * imp_left + (n_right / n) * imp_right
            return parent_imp - child_imp

    def _gini(self, y: np.ndarray) -> float:
        """
        Вычисляет индекс Джини для заданного набора меток.

        Param:
            y (np.ndarray): Массив меток классов.
        Returns:
            float: Индекс Джини.
        """
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p ** 2)

    def _gini_gain(
            self, y: np.ndarray, X_col: np.ndarray, threshold: float
        ) -> float:
        """
        Вычисляет прирост по индексу Джини (устаревший метод, не используется напрямую).

        Param:
            y (np.ndarray): Целевые метки.
            X_col (np.ndarray): Значения одного признака.
            threshold (float): Порог разбиения.
        Returns:
            float: Прирост Джини.
        """
        parent_gini = self._gini(y)
        left_idxs, right_idxs = self._split(X_col, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        gini_left = self._gini(y[left_idxs])
        gini_right = self._gini(y[right_idxs])
        gain = parent_gini - (len(left_idxs)/n * gini_left + len(right_idxs)/n * gini_right)
        return gain

    def _mse(self, y: np.ndarray) -> float:
        """
        Вычисляет среднеквадратичную ошибку (MSE) как дисперсию целевой переменной.

        Param:
            y (np.ndarray): Целевые значения.
        Returns:
            float: MSE (дисперсия).
        """
        if len(y) == 0:
            return 0
        return np.var(y)  # то же, что и np.mean((y - np.mean(y))**2)

    def _mse_reduction(
            self, y: np.ndarray, X_col: np.ndarray, threshold: float
            ) -> float:
        """
        Вычисляет уменьшение MSE при разбиении (устаревший метод, не используется напрямую).

        Param:
            y (np.ndarray): Целевые значения.
            X_col (np.ndarray): Значения одного признака.
            threshold (float): Порог разбиения.
        Returns:
            float: Уменьшение MSE.
        """
        parent_mse = self._mse(y)
        left_idxs, right_idxs = self._split(X_col, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        mse_left = self._mse(y[left_idxs])
        mse_right = self._mse(y[right_idxs])
        reduction = parent_mse - (
            len(left_idxs)/n * mse_left + len(right_idxs)/n * mse_right
        )
        return reduction

    def _split(self, X_col: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Разделяет индексы по порогу.

        Param:
            X_col (np.ndarray): Значения одного признака (n_samples,).
            threshold (float): Порог разбиения.
        Returns:
            Tuple[np.ndarray, np.ndarray]: (left_idxs, right_idxs) — индексы левой и правой групп.
        """
        left_idxs = np.argwhere(X_col <= threshold).flatten()
        right_idxs = np.argwhere(X_col > threshold).flatten()
        return left_idxs, right_idxs

    def _calculate_leaf_value(self, y: np.ndarray) -> Union[float, int]:
        """
        Вычисляет значение листа.

        Для регрессии — среднее значение.
        Для классификации — мода (наиболее частый класс).

        Param:
            y (np.ndarray): Целевые значения (n_samples,).
        Returns:
            Union[float, int]: Прогнозируемое значение листа.
        """
        if self.regression:
            return np.mean(y)
        else:
            return Counter(y).most_common(1)[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Выполняет предсказание для новых данных.

        Param:
            X (np.ndarray): Матрица признаков формы (n_samples, n_features).
        Returns:
            np.ndarray: Массив предсказаний формы (n_samples,).
        """
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray) -> Union[float, int]:
        """
        Предсказывает значение для одного объекта.

        Param:
            x (np.ndarray): Вектор признаков одного объекта (n_features,).
        Returns:
            Union[float, int]: Прогнозируемое значение.
        """
        node = self.root
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value