import tensorflow as tf
from typing import Tuple


class LearningCurveMetric:

    def calculate_curve_step(self, prediction: Tuple[tf.Tensor, tf.Tensor]) -> tf.float64:
        pass

    # todo for what?
    def evaluate_learning_curve(self):
        pass
