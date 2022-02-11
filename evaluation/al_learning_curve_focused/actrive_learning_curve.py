import tensorflow as tf


class ActiveLearningCurve:

    def __init__(self):
        self.learning_curve = []

    def after_k_steps(self, k: int):
        i = 0

        # required, if k is not in list
        while k < self.learning_curve[i][0] and i < len(self.learning_curve):
            i += 1

        return self.learning_curve[1][i]

    def add_step(self, k: int, rating: tf.float64):
        self.learning_curve.append((k, rating))
