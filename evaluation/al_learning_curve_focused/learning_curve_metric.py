import tensorflow as tf
from typing import Tuple


class LearningCurveMetric:

    def post_init(self, ground_truth: tf.Tensor):
        pass

    def calculate_curve_step(self, prediction: Tuple[tf.Tensor, tf.Tensor]) -> tf.float64:
        pass

    def evaluate_learning_curve(self, learning_curve: tf.Tensor):
        pass
