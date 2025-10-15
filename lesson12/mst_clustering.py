import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.metrics.pairwise import pairwise_distances

from typing import Optional, Union, Any


class MSTClustering:
    """
    Кластеризация на основе минимального покрывающего дерева (Minimum Spanning Tree, MST).

    Attr:
        edge_threshold (Optional[float]): Порог длины ребра для отсечения.
            Если задан, рёбра длиннее этого значения удаляются.
        n_clusters (Optional[int]): Желаемое количество кластеров.
            Должно быть в диапазоне [1, n_samples].
        metric (str): Метрика расстояния, используемая для вычисления попарных расстояний.
        labels_ (np.ndarray): Метки кластеров для каждой точки после вызова fit.
            Форма: (n_samples,), dtype: int.
        n_components_ (int): Фактическое количество полученных кластеров (компонент связности).
    """
    def __init__(
            self,
            edge_threshold: Optional[float] = None,
            n_clusters: Optional[int] = None,
            metric: str = 'euclidean'
        ) -> None:
        """
        Инициализирует объект MSTClustering.

        Args:
            edge_threshold (Optional[float]): Порог длины ребра MST.
                Если задан, рёбра длиннее этого значения удаляются.
                Не может использоваться одновременно с `n_clusters` (приоритет у `edge_threshold`).
                По умолчанию: None.
            n_clusters (Optional[int]): Целевое количество кластеров.
                Должно удовлетворять: 1 ≤ n_clusters ≤ n_samples.
                По умолчанию: None.
            metric (str): Метрика расстояния для вычисления попарных расстояний.
                Поддерживаемые значения — любые, принимаемые
                `sklearn.metrics.pairwise_distances`.
                По умолчанию: 'euclidean'.
        """
        self.edge_threshold = edge_threshold
        self.n_clusters = n_clusters
        self.metric = metric

    def fit(self, X: Union[np.ndarray, list]) -> 'MSTClustering':
        """
        Выполняет кластеризацию на основе минимального покрывающего дерева.

        Args:
            X (Union[np.ndarray, list]): 
                Обучающие данные — массив точек в пространстве признаков.
                Ожидается двумерная структура формы `(n_samples, n_features)`.
            y (Any, optional): 
                Игнорируется. Присутствует для совместимости с API scikit-learn.
        Returns:
            MSTClustering: 
                Экземпляр класса с установленными атрибутами:
                - `labels_` (`np.ndarray[int]`, shape `(n_samples,)`);
                - `n_components_` (`int`).
        Raises:
            ValueError: 
                - если `X` не двумерный;
                - если `X` пуст;
                - если `n_clusters` не в диапазоне [1, n_samples].
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X должен быть двумерным массивом.")

        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("Пустой набор данных.")

        # Вычисляем матрицу расстояний
        dist_matrix = pairwise_distances(X, metric=self.metric)

        # Строим минимальное покрывающее дерево (MST)
        mst = minimum_spanning_tree(dist_matrix)

        # Преобразуем MST в COO-формат для удобства работы с рёбрами
        mst_coo = mst.tocoo()

        if self.edge_threshold is not None:
            # Удаляем рёбра длиннее порога
            mask = mst_coo.data <= self.edge_threshold
            filtered_data = mst_coo.data[mask]
            filtered_row = mst_coo.row[mask]
            filtered_col = mst_coo.col[mask]
        elif self.n_clusters is not None:
            if self.n_clusters < 1 or self.n_clusters > n_samples:
                raise ValueError("n_clusters должно быть в диапазоне [1, n_samples].")
            n_edges_to_keep = n_samples - self.n_clusters
            if n_edges_to_keep <= 0:
                # Каждая точка — отдельный кластер
                self.labels_ = np.arange(n_samples)
                return self

            # Сортируем рёбра по длине и оставляем самые короткие (n_samples - n_clusters)
            sorted_indices = np.argsort(mst_coo.data)
            keep_indices = sorted_indices[:n_edges_to_keep]
            filtered_data = mst_coo.data[keep_indices]
            filtered_row = mst_coo.row[keep_indices]
            filtered_col = mst_coo.col[keep_indices]
        else:
            # Ничего не удаляем — один кластер
            filtered_data = mst_coo.data
            filtered_row = mst_coo.row
            filtered_col = mst_coo.col

        # Собираем отфильтрованное дерево (лес)
        filtered_mst = csr_matrix(
            (filtered_data, (filtered_row, filtered_col)),
            shape=(n_samples, n_samples)
        )

        # Находим компоненты связности
        n_components, labels = connected_components(
            filtered_mst, directed=False, return_labels=True
        )

        self.labels_ = labels
        self.n_components_ = n_components
        return self

    def fit_predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Выполняет кластеризацию и возвращает метки кластеров.
        Эквивалентен вызову `fit(X)` с последующим возвратом `self.labels_`.

        Args:
            X (Union[np.ndarray, list]): 
                Обучающие данные — массив точек в пространстве признаков.
                Ожидается двумерная структура формы `(n_samples, n_features)`.
        Returns:
            np.ndarray: 
                Метки кластеров для каждой точки.
                Форма: `(n_samples,)`, dtype: `int64` (или `int32` в зависимости от платформы).
        """
        self.fit(X)
        return self.labels_