from evaluation.al_learning_curve_focused.learning_curve_metric import LearningCurveMetric
import tensorflow as tf
from typing import Tuple


class BasicActiveLearningCurveMetric(LearningCurveMetric):

    def calculate_curve_step(self, prediction: Tuple[tf.Tensor, tf.Tensor]) -> tf.float64:
        return 1
