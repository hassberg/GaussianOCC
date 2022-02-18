from typing import Tuple

from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
import tensorflow as tf
import numpy as np

from models.svdd_neg.BaseSVDD import BaseSVDD


class SVDDNegSurrogateModel(SurrogateModel):

    def __init__(self):
        self.svdd_model = BaseSVDD(display='off')
        self.available_points = None
        self.labels = None

    def post_init(self, data_retriever: DataRetriever):
        self.query_pool = data_retriever.get_query_pool()
        self.point_shape = data_retriever.point_shape
        self.value_shape = data_retriever.value_shape

        self.available_points = data_retriever.get_query_pool().get_all_elements().numpy()
        self.labels = tf.ones([len(self.available_points), 1], dtype=tf.double).numpy()
        self.svdd_model.fit(self.available_points, self.labels)

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        uncertainty = self.svdd_model.get_distance(points.numpy()) - self.svdd_model.radius
        return tf.convert_to_tensor(uncertainty)

    def learn(self, points: tf.Tensor, feedback: tf.Tensor):
        indices = np.where(self.available_points == points)
        if len(indices[0] == len(points[0])):
            self.labels[indices[0][0]] = feedback.numpy()
        else:
            print("else.. append?")

        self.svdd_model = BaseSVDD(display='off')
        self.svdd_model.fit(self.available_points, self.labels)

    def query(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        prediction = self.svdd_model.predict(points.numpy())
        return points, tf.convert_to_tensor(prediction)
