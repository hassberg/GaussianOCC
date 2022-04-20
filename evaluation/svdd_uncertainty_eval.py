from active_learning_ts.logging.data_blackboard import Blackboard
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.experiments.blueprint import Blueprint
from sklearn.metrics import matthews_corrcoef
import torch

import tensorflow as tf

from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel


class SvddUncertainty(EvaluationMetric):
    """
    counts the number of rounds. Evaluation is the number of rounds
    """

    def __init__(self, eval_data=None):
        self.all_queries = tf.convert_to_tensor(eval_data[:, :-1], dtype=tf.float64)
        self.ground_truth = tf.convert_to_tensor(eval_data[:, -1], dtype=tf.float64)
        self.surrogate_model: SurrogateModel = None
        self.iteration_scoring = []
        self.end_experiment = None

    def post_init(self, blackboard: Blackboard, blueprint: Blueprint):
        self.surrogate_model = blueprint.surrogate_model

        # self.append_round_scoring()

    def eval(self):
        self.append_round_scoring()

    def get_evaluation(self):
        return '[' + ",".join(self.single_log(x) for x in self.iteration_scoring) + ']'

    def single_log(self, X):
        return '[' + ",".join(str(x[0].numpy()) for x in X) + ']'

    def append_round_scoring(self):
        self.iteration_scoring.append(
            self.surrogate_model.uncertainty(torch.as_tensor(self.all_queries.numpy())))
