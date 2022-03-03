import random
from typing import Tuple

import numpy as np
import tensorflow as tf
import pandas as pd
from active_learning_ts.data_retrievement.data_source import DataSource
import os

from active_learning_ts.pools.discrete_vector_pool import DiscreteVectorPool, Pool


class WaveformDatasource(DataSource):
    def __init__(self):
        outlier_fraction = 0.05
        data_frame = pd.read_csv(os.path.join(os.path.dirname(__file__), "waveform-5000_csv.csv")).sort_values(
            by="class")

        obj = data_frame.values
        _, counts = np.unique(obj[:, -1], return_counts=True)

        r1 = range(counts[1])
        r2 = random.sample(range(counts[0], counts[0] + counts[1]), int(0.5 * outlier_fraction * counts[0]))
        r3 = random.sample(range(counts[1] + counts[0], counts[2] + counts[1] + counts[0]),
                           int(0.5 * outlier_fraction * counts[0]))

        sample_set_indices = list(r1) + r2 + r3
        self.data_points = tf.convert_to_tensor(obj[sample_set_indices][:, :-1])
        self.point_shape = self.data_points.shape
        labels = obj[sample_set_indices][:, -1]
        x = []

        for i in range(len(labels)):
            if labels[i] == 0:
                x.append([1])
            else:
                x.append([-1])

        self.true_labels = tf.convert_to_tensor(x)
        self.value_shape = self.true_labels.shape

    def query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        data_points = []
        data_labels = []
        for i in range(len(actual_queries)):
            data_points.append(self.data_points[actual_queries[i].numpy()])
            data_labels.append(self.true_labels[actual_queries[i].numpy()])
        return data_points, data_labels

    def possible_queries(self) -> Pool:
        return DiscreteVectorPool(2, self.data_points, self.retrievementStrategy)
