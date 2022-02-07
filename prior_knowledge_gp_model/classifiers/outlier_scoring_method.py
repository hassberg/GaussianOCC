import tensorflow as tf


class OutlierScoringMethod:
    def __init__(self, data: tf.Tensor):
        self.available_data = data

    def calculate_scoring(self) -> tf.Tensor:
        pass
