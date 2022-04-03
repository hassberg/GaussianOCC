from typing import Tuple

import tensorflow as tf

from sklearn.neighbors import KNeighborsClassifier
from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool, Pool


class ParametrizedContinuousDataSource(DataSource):

    def __init__(self, data_points, values):
        print("init")
        self.data_points = tf.convert_to_tensor(data_points, dtype=tf.float64)
        self.point_shape = self.data_points.shape

        labels = tf.convert_to_tensor(values)
        self.true_labels = tf.reshape(labels, (len(labels), 1))
        self.value_shape = self.true_labels.shape
        self.classifier = KNeighborsClassifier()
        self.classifier.fit(self.data_points.numpy(), values)

    def query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        print("query")
        data_points = []
        data_labels = []
        for i in range(len(actual_queries)):
            data_points.append(actual_queries[i].numpy())
            data_labels.append(self.classifier.predict([actual_queries[i].numpy()]))
        return data_points, data_labels

    def possible_queries(self) -> Pool:
        ranges = []
        for i in range(self.point_shape[1]):
            ranges.append([(tf.reduce_min(self.data_points[:, i]).numpy(), tf.reduce_max(self.data_points[:, i]).numpy())])

        return ContinuousVectorPool(self.point_shape[1], ranges)
