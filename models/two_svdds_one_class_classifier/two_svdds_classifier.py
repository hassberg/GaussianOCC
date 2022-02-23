from typing import Tuple

from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
import tensorflow as tf
import numpy as np

from models.common_resource.BaseSVDD import BaseSVDD


def add_dummy_point(array: np.array):
    return tf.concat((array, array), axis=0)


class TwoSVDDSClassifierSurrogateModel(SurrogateModel):

    def __init__(self):
        self.svdd_pos = BaseSVDD(display='off')
        self.svdd_neg = BaseSVDD(display='off')
        self.available_points = None
        self.labels = None

    def post_init(self, data_retriever: DataRetriever):
        self.query_pool = data_retriever.get_query_pool()
        self.point_shape = data_retriever.point_shape
        self.value_shape = data_retriever.value_shape

        self.available_points = data_retriever.get_query_pool().get_all_elements().numpy()
        self.labels = tf.ones([len(self.available_points), 1], dtype=tf.double).numpy()
        self.svdd_pos.fit(self.available_points, self.labels)

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        # TODO
        uncertainty = self.svdd_pos.get_distance(points.numpy()) - self.svdd_pos.radius
        return tf.convert_to_tensor(uncertainty)

    def learn(self, points: tf.Tensor, feedback: tf.Tensor):
        # TODO
        indices = np.where(self.available_points == points)
        if len(indices[0] == len(points[0])):
            self.labels[indices[0][0]] = feedback.numpy()
        else:
            print("else.. append?")

        pos = []
        neg = []

        for i in range(len(self.available_points)):
            if self.labels[i][0] == 1:
                pos.append(tf.concat([self.available_points[i], self.labels[i]], axis=0))
            else:
                neg.append(tf.concat([self.available_points[i], self.labels[i]], axis=0))

        if len(pos) > 0:
            self.svdd_pos = BaseSVDD(display='off')
            pos = tf.convert_to_tensor(pos)
            if len(pos) == 1:
                pos = add_dummy_point(pos)

            self.svdd_pos.fit(pos[:, :-1].numpy(), tf.reshape(pos[:, -1], (len(pos), 1)).numpy())
        else:
            self.svdd_pos = None

        if len(neg) > 0:
            self.svdd_neg = BaseSVDD(display='off')
            neg = tf.convert_to_tensor(neg)
            if len(neg) == 1:
                neg = add_dummy_point(neg)

            self.svdd_neg.fit(neg[:, :-1].numpy(), tf.multiply(tf.reshape(neg[:, -1], (len(neg), 1)), -1).numpy())
        else:
            self.svdd_neg = None

    def query(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        # TODO
        prediction = self.svdd_pos.predict(points.numpy())
        return points, tf.convert_to_tensor(prediction)
