import numpy

from evaluation.al_learning_curve_focused.learning_curve_metric import LearningCurveMetric
import tensorflow as tf
from typing import Tuple, List


def point_not_converged(mean, stddev):
    if mean >= 0:
        return mean - stddev >= 0
    else:
        return mean + stddev < 0


class CertaintyReachedMetric(LearningCurveMetric):
    def __init__(self):
        self.ground_truth = None

    def post_init(self, ground_truth=List[tf.Tensor]):
        self.ground_truth = tf.convert_to_tensor(ground_truth).numpy().flatten()

    def calculate_curve_step(self, prediction: Tuple[tf.Tensor, tf.Tensor]) -> tf.float64:
        certain = True
        for i in range(len(prediction)):
            certain = certain and point_not_converged(prediction[0][i], prediction[1][i])

            if not certain:
                return 0

        return 1

    def evaluate_learning_curve(self, learning_curve: tf.Tensor):
        return '[' + ", ".join(str(x.numpy()) for x in learning_curve) + ']'
