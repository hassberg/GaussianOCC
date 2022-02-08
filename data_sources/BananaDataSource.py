from typing import Tuple

import numpy as np
import tensorflow as tf
from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.pools.discrete_vector_pool import DiscreteVectorPool, Pool


def generate(**kwargs):
    # Banana-shaped dataset generation
    number = kwargs['class_member']
    number_n = kwargs['outlier']

    # parameters for banana-shaped dataset
    sizeBanana = 3
    varBanana = 1.2
    param_1 = 0.02
    param_2 = 0.02
    param_3 = 0.98
    param_4 = -0.8  # x-axsis shift
    # generate
    class_p = param_1 * np.pi + np.random.rand(number, 1) * param_3 * np.pi
    data_p = np.append(sizeBanana * np.sin(class_p), sizeBanana * np.cos(class_p), axis=1)
    data_p = data_p + np.random.rand(number, 2) * varBanana
    data_p[:, 0] = data_p[:, 0] - sizeBanana * 0.5
    label_p = np.ones((number, 1), dtype=np.int64)

    class_n = param_2 * np.pi - np.random.rand(number_n, 1) * param_3 * np.pi
    data_n = np.append(sizeBanana * np.sin(class_n), sizeBanana * np.cos(class_n), axis=1)
    data_n = data_n + np.random.rand(number_n, 2) * varBanana
    data_n = data_n + np.ones((number_n, 1)) * [sizeBanana * param_4, sizeBanana * param_4]
    data_n[:, 0] = data_n[:, 0] + sizeBanana * 0.5
    label_n = -np.ones((number_n, 1), dtype=np.int64)

    # banana-shaped dataset
    data = np.append(data_p, data_n, axis=0)
    label = np.append(label_p, label_n, axis=0)

    return data, label


# 2D data of
class BananaDataSource(DataSource):
    def __init__(self):
        points, labels = generate(class_member=100, outlier=20)
        self.data_points = tf.convert_to_tensor(points)
        self.point_shape = points.shape
        self.true_labels = tf.convert_to_tensor(labels)
        self.value_shape = labels.shape

    # TODO someone should somewhere mention that these are indices.. and that its just one
    def query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # TODO this is only for interpolation
        data_points = []
        data_labels = []
        data_points.append(self.data_points[actual_queries[0].numpy()])
        data_labels.append(self.true_labels[actual_queries[0].numpy()])
        return data_points, data_labels

    def possible_queries(self) -> Pool:
        return DiscreteVectorPool(2, self.data_points, self.retrievementStrategy)
