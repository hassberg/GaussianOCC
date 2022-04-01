from typing import Tuple

import tensorflow as tf
from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.data_retrievement.pools.discrete_vector_pool import DiscreteVectorPool, Pool


class ParametrizedDiscreteDataSource(DataSource):

    def __init__(self, data_points, values):
        self.data_points = tf.convert_to_tensor(data_points, dtype=tf.float64)
        self.point_shape = self.data_points.shape

        labels = tf.convert_to_tensor(values)
        self.true_labels = tf.reshape(labels, (len(labels), 1))
        self.value_shape = self.true_labels.shape

    def query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        data_points = []
        data_labels = []
        for i in range(len(actual_queries)):
            data_points.append(self.data_points[actual_queries[i].numpy()])
            data_labels.append(self.true_labels[actual_queries[i].numpy()])
        return data_points, data_labels

    def possible_queries(self) -> Pool:
        return DiscreteVectorPool(2, self.data_points, self.retrievement_strategy)
