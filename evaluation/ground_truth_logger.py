from active_learning_ts.logging.data_blackboard import Blackboard
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.experiments.blueprint import Blueprint
from sklearn.metrics import matthews_corrcoef
import torch

import tensorflow as tf

from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel


class GroundTruthLogger(EvaluationMetric):
    """
    counts the number of rounds. Evaluation is the number of rounds
    """

    def __init__(self, eval_data=None):
        self.ground_truth = tf.convert_to_tensor(eval_data[:, -1], dtype=tf.float64)
        self.end_experiment = None

    def post_init(self, blackboard: Blackboard, blueprint: Blueprint):
        pass

    def eval(self):
        pass

    def get_evaluation(self):
        return '[' + ",".join(str(x.numpy()) for x in self.ground_truth) + ']'
