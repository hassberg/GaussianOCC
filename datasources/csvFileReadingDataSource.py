import random
from typing import Tuple

import numpy as np
import tensorflow as tf
import pandas as pd
from active_learning_ts.data_retrievement.data_source import DataSource
import os

from active_learning_ts.data_retrievement.pools.discrete_vector_pool import DiscreteVectorPool, Pool

##
## reads csv file to use data as datasource
## data format: [attr,attr,..,class]
## class 1 for inlier, class -1 for outlier
## uses discrete Pool
##
class CsvFileReadingDataSource(DataSource):

    def __init__(self, fileName:str):
        data_frame = pd.read_csv(fileName, header=None)

        obj = data_frame.values
        self.data_points = tf.convert_to_tensor(obj[:, :-1], dtype=tf.float64)
        self.point_shape = self.data_points.shape

        labels = tf.convert_to_tensor(obj[:, -1])
        self.true_labels = tf.reshape(labels, (len(labels),1))
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
