from typing import Tuple

from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel
import tensorflow as tf

from models.common_resource.BaseSVDD import BaseSVDD


class SVDDNegSurrogateModel(SurrogateModel):

    def __init__(self):
        self.svdd_model = BaseSVDD(C=1, display='off')
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
        uncertainty = self.svdd_model.get_distance(points.numpy())- self.svdd_model.radius
        return tf.convert_to_tensor(uncertainty)

    def learn(self, points: tf.Tensor, feedback: tf.Tensor):
        # updates label of point
        for point, curr_feedback in zip(points, feedback):
            contained = False
            for i in range(len(self.available_points)):
                equal = True
                for j in range(len(point)):
                    if point[j] != self.available_points[i][j]:
                        equal = False
                        break

                if equal:
                    self.labels[i] = curr_feedback
                    contained = True
                    break

            # TODO if point is not in set of available_points contained

        self.svdd_model = BaseSVDD(C=1, display='off')
        self.svdd_model.fit(self.available_points, self.labels)

    def query(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        prediction = self.svdd_model.predict(points.numpy())
        return points, tf.convert_to_tensor(prediction)
