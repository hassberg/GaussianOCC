import numpy

from evaluation.al_learning_curve_focused.learning_curve_metric import LearningCurveMetric
import tensorflow as tf
from typing import Tuple, List


def get_membership_prediction(class_prediction=tf.Tensor):
    return tf.map_fn((lambda t: 1 if (t >= 0) else -1), class_prediction.detach())


class BasicActiveLearningCurveMetric(LearningCurveMetric):
    def __init__(self):
        self.ground_truth = None

    def post_init(self, ground_truth=List[tf.Tensor]):
        self.ground_truth = tf.convert_to_tensor(ground_truth).numpy().flatten()

    def calculate_curve_step(self, prediction: Tuple[tf.Tensor, tf.Tensor]) -> tf.float64:
        class_prediction = get_membership_prediction(prediction[0])
        correctness = 0
        for i in range(len(class_prediction)):
            if class_prediction[i] == self.ground_truth[i]:
                correctness += 1

        return numpy.divide(correctness, len(self.ground_truth))

    def evaluate_learning_curve(self, learning_curve: tf.Tensor):
        return '[' + ", ".join(str(x.numpy()) for x in learning_curve) + ']'
